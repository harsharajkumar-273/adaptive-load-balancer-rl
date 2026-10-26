"""
Real-server agent-overload experiment: the same question as research/agent_sim.py, but the
"server" is a real OpenAI-compatible LLM server (vLLM / SGLang) or a built-in mock.

    # real GPU server (start vLLM first, e.g. `vllm serve MODEL --max-num-seqs 8 --enable-prefix-caching`)
    python -m research.agent_realsys --backend vllm --url http://localhost:8000 --model MODEL --slots 8
    # no GPU: the mock server has `slots` concurrent steps and lognormal service times
    python -m research.agent_realsys --backend mock --slots 8 --quick

What runs
  * Agents: tasks arrive as a Poisson process. A task has L steps (L and tool times resampled from the
    real coding-agent sessions in research/data/agent_empirical.npz). Each step sends the growing prompt to
    the server, waits for the reply (client timeout, cancel on timeout, exponential backoff, at most
    `retries` retries, then the task is lost), then "runs a tool" (sleeps).
  * Gateway (policy layer between the agents and the server): none | shed:Q | cap:N | little.
      none    admit everything
      shed:Q  refuse a step when more than Q requests wait beyond the `slots` in service
      cap:N   admit a new task only while fewer than N tasks are admitted
      little  admit while N < C (s + tau + D) / s with s, tau measured online and D = 0.1 x timeout
  * Time is measured in units of the server's step time: after a calibration run measures the
    saturated throughput X (steps/s) and step time s0 = C / X, the client timeout is 15 s0, the backoff base
    is s0 and tool times are scaled by s0 / 2 s, which keeps the ratios used in the simulator (s=2 s,
    timeout 30 s, backoff 2 s).
  * Offered load is `load` x the task capacity X / E[L]; goodput is completed steps over X x window.

Results are appended to research/results/agent_realsys.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp
import numpy as np
from aiohttp import web

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results", "agent_realsys.csv")
VOCAB = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma omega tensor graph node edge cache "
         "token queue server agent tool result error file line patch test build commit branch merge").split()


# --------------------------------------------------------------------------------------------
# Mock backend (stands in for vLLM when there is no GPU)
# --------------------------------------------------------------------------------------------

def mock_app(slots: int, mean_sec: float, cv: float = 0.5, seed: int = 0) -> web.Application:
    rng = random.Random(seed)
    sem = asyncio.Semaphore(slots)
    sigma2 = math.log(1 + cv * cv)
    mu = math.log(mean_sec) - sigma2 / 2

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        async with sem:
            await asyncio.sleep(math.exp(rng.gauss(mu, math.sqrt(sigma2))))
        return web.json_response({"choices": [{"text": " ok", "finish_reason": "length"}],
                                  "usage": {"prompt_tokens": len(body.get("prompt", "")) // 4}})

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    return app


async def start_mock(slots: int, mean_sec: float, port: int) -> web.AppRunner:
    runner = web.AppRunner(mock_app(slots, mean_sec), handler_cancellation=True)   # client disconnect aborts, like vLLM
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


# --------------------------------------------------------------------------------------------
# Gateway: policy layer + forwarding
# --------------------------------------------------------------------------------------------

def parse_policy(spec: str):
    kind, _, arg = spec.partition(":")
    if kind not in ("none", "shed", "cap", "little"):
        raise ValueError(f"unknown policy {spec!r}")
    return kind, (float(arg) if arg else None)


class Gateway:
    def __init__(self, session: aiohttp.ClientSession, url: str, model: str, policy: str, slots: int,
                 s0: float, timeout: float, kv_tasks: int = 10 ** 9, max_tokens: int = 48, d_frac: float = 0.1):
        self.session, self.url, self.model = session, url.rstrip("/") + "/v1/completions", model
        self.kind, self.arg = parse_policy(policy)
        self.slots, self.timeout, self.kv_tasks, self.max_tokens, self.d_frac = slots, timeout, kv_tasks, max_tokens, d_frac
        self.inflight = 0
        self.admitted: set = set()
        self.ewma_s, self.ewma_tau = s0, 0.0
        self.last_done: Dict[int, float] = {}
        self.t0 = time.monotonic()

    # -- admission ------------------------------------------------------------------------
    def admit(self, task_id: int, first_attempt_of_step: bool) -> bool:
        now = time.monotonic()
        if task_id in self.last_done and first_attempt_of_step:
            gap = now - self.last_done.pop(task_id)
            self.ewma_tau += 0.05 * (gap - self.ewma_tau)
        known = task_id in self.admitted
        if self.kind == "shed":
            ok = (self.inflight - self.slots) < (self.arg or 16)
        elif self.kind == "cap":
            ok = known or len(self.admitted) < (self.arg or 64)
        elif self.kind == "little":
            limit = min(self.slots * (self.ewma_s + self.ewma_tau + self.d_frac * self.timeout) / self.ewma_s,
                        self.kv_tasks)
            ok = known or len(self.admitted) < limit
        else:
            ok = True
        if ok:
            self.admitted.add(task_id)
        return ok

    def finish(self, task_id: int) -> None:
        self.admitted.discard(task_id)
        self.last_done.pop(task_id, None)

    # -- forwarding -------------------------------------------------------------------------
    async def forward(self, task_id: int, prompt: str) -> float:
        """Sends one step; returns the latency. Raises asyncio.TimeoutError after `timeout` (request cancelled)."""
        started_free = self.inflight < self.slots
        t0 = time.monotonic()
        self.inflight += 1
        try:
            payload = {"model": self.model, "prompt": prompt, "max_tokens": self.max_tokens, "temperature": 0.0,
                       "ignore_eos": True}
            async def call():
                async with self.session.post(self.url, json=payload) as resp:
                    resp.raise_for_status()
                    await resp.read()
            await asyncio.wait_for(call(), self.timeout)      # on timeout the request is cancelled (server aborts)
        finally:
            self.inflight -= 1
        lat = time.monotonic() - t0
        if started_free:                        # no queueing at arrival: latency ~ service time
            self.ewma_s += 0.05 * (lat - self.ewma_s)
        self.last_done[task_id] = time.monotonic()
        return lat


# --------------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------------

@dataclass
class TaskResult:
    arrival: float
    steps: int
    outcome: str = "unfinished"        # complete | lost | rejected | unfinished
    done: int = 0
    finish: float = 0.0
    step_times: Optional[List[float]] = None   # completion time of every finished step, seconds since the run started


def empirical(max_steps: int, max_tool: float):
    z = np.load(os.path.join(HERE, "data", "agent_empirical.npz"))
    return (np.minimum(z["steps"], max_steps).astype(int).tolist(), np.minimum(z["tool"], max_tool).astype(float).tolist())


def make_prompt(rng: random.Random, history: List[str], base_words: int, step_words: int) -> str:
    if not history:
        history.append(" ".join(rng.choice(VOCAB) for _ in range(base_words)))
    else:
        history.append(" ".join(rng.choice(VOCAB) for _ in range(step_words)))
    return " ".join(history)


async def run_task(gw: Gateway, tid: int, L: int, tools: List[float], res: TaskResult, rng: random.Random,
                   backoff: float, retries: int, base_words: int, step_words: int) -> None:
    history: List[str] = []
    res.step_times = []
    step, tries = 0, 0
    prompt: Optional[str] = None
    while step < L:
        if not gw.admit(tid, first_attempt_of_step=(tries == 0 and step > 0)):
            tries += 1
            if tries > retries:
                res.outcome = "rejected" if step == 0 else "lost"
                gw.finish(tid)
                return
            await asyncio.sleep(backoff * (2 ** (tries - 1)) * rng.uniform(0.5, 1.5))
            continue
        if prompt is None:
            prompt = make_prompt(rng, history, base_words, step_words)
        try:
            await gw.forward(tid, prompt)
        except (asyncio.TimeoutError, aiohttp.ClientError):
            tries += 1
            if tries > retries:
                res.outcome = "lost" if step > 0 else "rejected"
                gw.finish(tid)
                return
            await asyncio.sleep(backoff * (2 ** (tries - 1)) * rng.uniform(0.5, 1.5))
            continue
        step += 1
        res.done = step
        res.step_times.append(time.monotonic() - gw.t0)
        tries = 0
        prompt = None
        if step < L:
            await asyncio.sleep(tools[step - 1])
    res.outcome, res.finish = "complete", time.monotonic() - gw.t0
    gw.finish(tid)


async def calibrate(session, url, model, slots, base_words, step_words, max_tokens, seconds=15.0, seed=0) -> float:
    """Saturated throughput (steps/s) with `2 x slots` closed-loop workers sending typical prompts."""
    rng = random.Random(seed)
    done = 0
    stop = time.monotonic() + seconds

    async def worker():
        nonlocal done
        while time.monotonic() < stop:
            hist: List[str] = []
            for _ in range(rng.randint(1, 6)):
                prompt = make_prompt(rng, hist, base_words, step_words)
                async with session.post(url.rstrip("/") + "/v1/completions", json={
                        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                        "ignore_eos": True}) as r:
                    await r.read()
                done += 1

    t0 = time.monotonic()
    await asyncio.gather(*[worker() for _ in range(2 * slots)])
    return done / (time.monotonic() - t0)


async def run_one(args, session, X: float, policy: str, load: float, rep: int, steps_pool, tool_pool) -> Dict:
    s0 = args.slots / X
    time_scale = s0 / 2.0
    timeout = 15.0 * s0
    rng = random.Random(args.seed * 1000 + rep * 17 + int(load * 100))
    mean_L = sum(steps_pool) / len(steps_pool)
    lam = load * X / mean_L
    gw = Gateway(session, args.url, args.model, policy, args.slots, s0, timeout, args.kv_tasks, args.max_tokens)
    results: List[TaskResult] = []
    tasks = []
    t_start = gw.t0 = time.monotonic()
    t_arr = 0.0
    tid = 0
    while True:
        t_arr += rng.expovariate(lam)
        if t_arr >= args.duration:
            break
        delay = t_start + t_arr - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        L = rng.choice(steps_pool)
        tools = [rng.choice(tool_pool) * time_scale for _ in range(L)]
        res = TaskResult(arrival=t_arr, steps=L)
        results.append(res)
        tasks.append(asyncio.create_task(run_task(gw, tid, L, tools, res, random.Random(rng.random()), s0, args.retries,
                                                  args.base_words, args.step_words)))
        tid += 1
    await asyncio.wait(tasks, timeout=args.drain_factor * timeout) if tasks else None
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    win = [r for r in results if r.arrival >= args.warmup]
    span = args.duration - args.warmup
    done = [r for r in win if r.outcome == "complete"]
    lost = [r for r in win if r.outcome == "lost"]
    finished_steps = sum(r.done for r in win)
    lat = sorted(r.finish - r.arrival for r in done)
    return {
        "policy": policy, "load": load, "rep": rep, "tasks": len(win), "capacity_steps_per_s": X, "step_sec": s0,
        # steps of tasks that eventually completed, finished inside the measurement window, over server capacity
        "goodput_norm": sum(1 for r in done for t in r.step_times if args.warmup <= t < args.duration) / span / X,
        "completed": len(done) / max(len(win), 1),
        "lost": len(lost) / max(len(win), 1),
        "rejected": sum(r.outcome == "rejected" for r in win) / max(len(win), 1),
        "unfinished": sum(r.outcome == "unfinished" for r in win) / max(len(win), 1),
        "wasted_steps": (sum(r.done for r in win) - sum(r.steps for r in done)) / max(finished_steps, 1),
        "p99_task_s": lat[min(len(lat) - 1, int(0.99 * len(lat)))] if lat else float("nan"),
        "mean_admitted_end": len(gw.admitted),
    }


async def main_async(args) -> List[Dict]:
    runner = None
    if args.backend == "mock":
        runner = await start_mock(args.slots, args.mock_step_sec, args.port)
        args.url = f"http://127.0.0.1:{args.port}"
        args.model = "mock"
    steps_pool, tool_pool = empirical(args.max_steps, args.max_tool)
    rows: List[Dict] = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as session:
        X = await calibrate(session, args.url, args.model, args.slots, args.base_words, args.step_words, args.max_tokens,
                            seconds=args.calibrate_sec)
        print(f"calibrated: saturated throughput {X:.2f} steps/s with {args.slots} slots "
              f"(step time {args.slots / X:.2f} s); task capacity {X / (sum(steps_pool) / len(steps_pool)):.3f} tasks/s", flush=True)
        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        fresh = not os.path.exists(RESULTS)
        for load in args.loads:
            for policy in args.policies:
                for rep in range(args.reps):
                    row = await run_one(args, session, X, policy, load, rep, steps_pool, tool_pool)
                    row["backend"] = args.backend
                    row["model"] = args.model
                    row["slots"] = args.slots
                    rows.append(row)
                    with open(RESULTS, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(row))
                        if fresh:
                            w.writeheader()
                            fresh = False
                        w.writerow(row)
                    print(f"load {load:>4} {policy:>9} rep {rep}: goodput {row['goodput_norm']:.2f} done {row['completed']:.2f} "
                          f"lost {row['lost']:.2f} rej {row['rejected']:.2f} wasted {row['wasted_steps']:.2f} "
                          f"p99 {row['p99_task_s']:.0f}s", flush=True)
                    await asyncio.sleep(2 * 15.0 * args.slots / X)      # let the server drain between runs
    if runner:
        await runner.cleanup()
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--slots", type=int, default=8, help="the server's concurrent steps (vLLM --max-num-seqs)")
    p.add_argument("--loads", type=float, nargs="+", default=[0.6, 0.9, 1.2, 1.6, 2.4, 4.0])
    p.add_argument("--policies", nargs="+", default=["none", "shed:16", "cap:48", "little"])
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--duration", type=float, default=240.0, help="seconds of task arrivals per run")
    p.add_argument("--warmup", type=float, default=40.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--max-tool", type=float, default=300.0)
    p.add_argument("--kv-tasks", type=int, default=10 ** 9, help="little: never admit more tasks than fit in KV memory")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--base-words", type=int, default=300)
    p.add_argument("--step-words", type=int, default=80)
    p.add_argument("--drain-factor", type=float, default=6.0, help="seconds to wait for stragglers, in client timeouts")
    p.add_argument("--calibrate-sec", type=float, default=15.0)
    p.add_argument("--mock-step-sec", type=float, default=0.25)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="short smoke run")
    args = p.parse_args(argv)
    if args.quick:
        args.loads, args.policies, args.reps = [0.9, 2.4], ["none", "little"], 1
        args.duration, args.warmup, args.calibrate_sec = 60.0, 10.0, 8.0
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
