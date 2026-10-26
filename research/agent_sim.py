"""
Agent-overload simulation: what happens when many autonomous agents, each running
a multi-step task with automatic retries, offer more work than a server can do?

Model (one GPU server, discrete-event)
  * C parallel slots (a continuous-batching engine abstracted to C concurrent
    steps), step service time lognormal with mean `step_sec`.
  * A task is a chain of L steps (L ~ 1 + Geometric, mean `mean_steps`). After each
    step the agent runs a tool for Exp(`tool_sec`) seconds, then submits the next
    step. The step's prompt contains the whole history, so the KV state of a task
    stays resident between steps unless evicted.
  * KV memory holds `kv_tasks` tasks. When full, the least recently used task that
    is not in service is evicted; its next step pays a rebuild cost of
    `rebuild * steps_done * step_sec` extra service time.
  * Clients time a step out after `timeout` seconds, cancel it if it is still
    queued (a request in service completes and is wasted), and retry with
    exponential backoff and jitter, at most `retries` times; then the task is lost
    together with all the work already done on it.
  * Tasks arrive as a Poisson process; `load` is task arrival rate over the
    no-failure capacity C / (step_sec * mean_steps). The `surge` scenario raises
    the rate to `surge_factor`x for a window and returns to base load.

Admission and scheduling policies (see `parse_policy`)
  none        accept everything, FIFO
  shed        per-request shedding: reject a step when the queue is longer than Q
  prog        accept everything, serve requests of tasks with more progress first
  little      Little's-law cap: admit while N < C (s + tau + D) / s with s (step time) and tau
              (tool time between a step finishing and the task's next step arriving) measured
              online and D a target queueing wait, and never more than the KV capacity. In a closed system of N tasks the fluid
              wait is N s / C - s - tau, so this holds the wait at D without knowing the workload.
  cap         admit a new task only while fewer than N tasks are admitted (no length info)
  task        task-level admission with a work budget: admit a *new* task only if the
              estimated remaining work of all admitted tasks plus the new one fits in
              H seconds of capacity; steps of admitted tasks are always accepted and
              served by progress. Knowledge levels:
                id  = tag   the server knows which task a step belongs to
                    = infer a step is recognised with probability p_id; otherwise it is
                            treated as a new task (and can be rejected mid-task)
                len = oracle | pred (noisy, sigma) | mean   remaining-steps estimate
Metrics are over tasks that arrive in the measurement window.
"""
from __future__ import annotations

import heapq
import math
import os
import random
from functools import lru_cache
from collections import OrderedDict, deque
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@lru_cache(maxsize=None)
def empirical(max_steps: int = 100, max_tool: float = 300.0):
    """Task lengths and tool latencies of real coding-agent sessions (TraceLab), clipped to a sane range."""
    import numpy as np
    z = np.load(os.path.join(os.path.dirname(__file__), "data", "agent_empirical.npz"))
    return (np.minimum(z["steps"], max_steps).astype(int).tolist(),
            np.minimum(z["tool"], max_tool).astype(float).tolist())


@dataclass(frozen=True)
class AgentScenario:
    load: float = 1.6                 # task arrival rate / no-failure task capacity
    slots: int = 16                   # C
    step_sec: float = 2.0
    step_cv: float = 0.5
    mean_steps: float = 6.0
    tool_sec: float = 3.0
    timeout: float = 30.0
    retries: int = 3
    backoff: float = 2.0
    kv_tasks: int = 64
    rebuild: float = 0.25             # extra service per completed step when KV was evicted
    duration: float = 600.0           # arrival window
    warmup: float = 100.0
    drain: float = 300.0
    trace: bool = False               # draw task lengths and tool times from the TraceLab empirical distributions
    surge_factor: float = 1.0         # > 1: rate multiplier during [surge_start, surge_end)
    surge_start: float = 150.0
    surge_end: float = 210.0
    recovery_start: float = 260.0     # recovery metrics: tasks arriving in [recovery_start, duration)

    def to_row(self) -> Dict:
        row = asdict(self)
        row["scenario"] = "surge" if self.surge_factor != 1.0 else "steady"
        return row

    @classmethod
    def from_trace(cls, **kw) -> "AgentScenario":
        """Scenario whose means (used for capacity and the `mean` length policy) match the empirical data."""
        steps, tool = empirical()
        base = dict(trace=True, mean_steps=sum(steps) / len(steps), tool_sec=sum(tool) / len(tool),
                    duration=1800.0, warmup=300.0, drain=900.0)
        base.update(kw)
        return cls(**base)

    @property
    def capacity_tasks_per_sec(self) -> float:
        return self.slots / (self.step_sec * self.mean_steps)


@dataclass(frozen=True)
class Policy:
    kind: str = "none"          # none | shed | prog | cap | little | task
    q: float = 32.0             # shed: queue limit
    h: float = 20.0             # task: work budget in seconds of capacity
    ident: str = "tag"          # task: tag | infer
    p_id: float = 1.0           # infer: probability a step is recognised
    length: str = "oracle"      # task: oracle | pred | mean
    sigma: float = 0.5          # pred: log-noise of the length estimate
    d: float = 7.5              # little: target queueing wait in seconds
    df: float = 0.0             # little: if > 0, target wait as a fraction of the client timeout (overrides d)
    kvm: float = 1.0            # little: admit up to kvm x KV capacity (tasks in tool calls can be evicted)
    n: float = 32.0             # cap: maximum number of admitted tasks
    fifo: bool = False          # cap/task: serve in arrival order instead of by progress
    hint: bool = False          # rejected clients wait the server's drain estimate instead of backing off blindly


def parse_policy(spec: str) -> Policy:
    kind, *rest = spec.split("|")
    kw: dict = {}
    for item in rest:
        k, v = item.split("=")
        kw[k] = v if k in ("ident", "length") else (v == "1" if k in ("hint", "fifo") else float(v))
    return Policy(kind=kind, **kw)


class _Task:
    __slots__ = ("id", "L", "est", "arrival", "done", "good_service", "tries", "state", "n_service", "finish")

    def __init__(self, tid, L, est, arrival):
        self.id, self.L, self.est, self.arrival = tid, L, est, arrival
        self.done, self.good_service, self.tries = 0, 0.0, 0
        self.state = "new"           # new | active | complete | lost | rejected
        self.n_service = 0           # requests of this task currently in service (incl. cancelled ones)
        self.finish = None


class _Req:
    __slots__ = ("task", "state", "svc", "t")

    def __init__(self, task, t):
        self.task, self.state, self.svc, self.t = task, "queued", 0.0, t   # queued | service | dead


def simulate_agents(sc: AgentScenario, policy: str, seed: int = 0, policy_kwargs=None) -> Dict:
    pol = parse_policy(policy)
    rng = random.Random(seed * 7919 + 13)
    C, s = sc.slots, sc.step_sec
    sigma2 = math.log(1 + sc.step_cv ** 2)
    ln_mu, ln_sd = math.log(s) - sigma2 / 2, math.sqrt(sigma2)
    horizon = sc.duration + sc.drain
    p_geo = 1.0 / sc.mean_steps
    by_progress = pol.kind in ("prog", "cap", "task") and not pol.fifo

    events: list = []
    counter = 0

    def push(t, kind, payload):
        nonlocal counter
        counter += 1
        heapq.heappush(events, (t, counter, kind, payload))

    fifo: deque = deque()
    prio: list = []
    qlen = 0                               # live queued requests
    busy = 0
    busy_time = 0.0
    resident: "OrderedDict[int, None]" = OrderedDict()
    admitted: Dict[int, _Task] = {}
    tasks = []
    waits = []
    n_samples = [0, 0.0]
    ewma_s, ewma_tau = s, 0.0         # online estimates used by `little` (priors: step time, no tool time)
    last_done: Dict[int, float] = {}

    emp_steps, emp_tool = empirical() if sc.trace else (None, None)
    base_rate = sc.load * sc.capacity_tasks_per_sec
    t, tid = 0.0, 0
    while True:
        rate = base_rate * (sc.surge_factor if sc.surge_start <= t < sc.surge_end else 1.0)
        t += rng.expovariate(rate)
        if t >= sc.duration:
            break
        if sc.trace:
            L = rng.choice(emp_steps)
        else:
            L = 1
            while rng.random() > p_geo:
                L += 1
        est = max(1.0, L * math.exp(rng.gauss(0.0, pol.sigma)))
        task = _Task(tid, L, est, t)
        tasks.append(task)
        push(t, "submit", task)
        tid += 1

    def remaining(task):
        if pol.length == "oracle":
            return max(1.0, task.L - task.done)
        if pol.length == "pred":
            return max(1.0, task.est - task.done)
        return max(1.0, sc.mean_steps - task.done)

    def total_estimate(task):
        return {"oracle": task.L, "pred": task.est}.get(pol.length, sc.mean_steps)

    def touch_kv(task) -> bool:
        had = task.id in resident
        resident[task.id] = None
        resident.move_to_end(task.id)
        while len(resident) > sc.kv_tasks:
            for victim in resident:
                vt = admitted.get(victim)
                if victim != task.id and not (vt is not None and vt.n_service > 0):
                    del resident[victim]
                    break
            else:
                break
        return had

    def finish(task, state, now):
        last_done.pop(task.id, None)
        task.state, task.finish = state, now
        admitted.pop(task.id, None)
        resident.pop(task.id, None)

    def dispatch(now):
        nonlocal busy, qlen, busy_time
        while busy < C:
            req = None
            if by_progress:
                while prio:
                    cand = heapq.heappop(prio)[2]
                    if cand.state == "queued":
                        req = cand
                        break
            else:
                while fifo:
                    cand = fifo.popleft()
                    if cand.state == "queued":
                        req = cand
                        break
            if req is None:
                return
            qlen -= 1
            task = req.task
            if now >= sc.warmup:
                waits.append(now - req.t)
            svc = math.exp(rng.gauss(ln_mu, ln_sd))
            if not touch_kv(task):
                svc += sc.rebuild * task.done * s
            req.state, req.svc = "service", svc
            task.n_service += 1
            busy += 1
            busy_time += svc
            push(now + svc, "done", req)

    def enqueue(task, now):
        nonlocal qlen, counter
        req = _Req(task, now)
        qlen += 1
        counter += 1
        if by_progress:
            heapq.heappush(prio, (-task.done, counter, req))
        else:
            fifo.append(req)
        push(now + sc.timeout, "timeout", req)
        dispatch(now)

    def retry_or_give_up(task, now, hint=None):
        task.tries += 1
        if task.tries > sc.retries:
            finish(task, "rejected" if task.done == 0 and task.state != "active" else "lost", now)
            return
        delay = max(sc.backoff * 0.5, hint) if (pol.hint and hint is not None) else \
            sc.backoff * (2 ** (task.tries - 1)) * rng.uniform(0.5, 1.5)
        push(now + delay, "submit", task)

    while events:
        now, _, kind, payload = heapq.heappop(events)
        if now > horizon:
            break
        if sc.warmup <= now < sc.duration:
            n_samples[0] += 1
            n_samples[1] += len(admitted)
        if kind == "submit":
            task = payload
            if task.state in ("complete", "lost", "rejected"):
                continue
            hint = None
            accept = True
            if pol.kind == "shed":
                accept = qlen < pol.q
            elif pol.kind in ("cap", "little"):
                recognised = task.id in admitted and not (pol.ident == "infer" and rng.random() > pol.p_id)
                limit = pol.n if pol.kind == "cap" else min(C * (ewma_s + ewma_tau + (pol.df * sc.timeout if pol.df > 0 else pol.d)) / ewma_s, pol.kvm * sc.kv_tasks)
                accept = recognised or len(admitted) < limit
                if task.id in last_done and task.tries == 0 and kind == "submit" and accept:
                    ewma_tau += 0.05 * ((now - last_done.pop(task.id)) - ewma_tau)
            elif pol.kind == "task":
                recognised = task.id in admitted and not (pol.ident == "infer" and rng.random() > pol.p_id)
                if not recognised:
                    w_sys = sum(remaining(x) for x in admitted.values()) * s
                    w_new = total_estimate(task) * s
                    accept = (w_sys + w_new) / C <= pol.h
                    hint = (w_sys + w_new) / C - pol.h
            if not accept:
                retry_or_give_up(task, now, hint)
                continue
            if task.id not in admitted:
                admitted[task.id] = task
                task.state = "active"
            enqueue(task, now)
        elif kind == "done":
            req = payload
            busy -= 1
            task = req.task
            task.n_service -= 1
            ewma_s += 0.02 * (req.svc - ewma_s)        # the server times every step it runs, even abandoned ones
            if req.state == "dead" or task.state != "active":
                pass                                     # timed out or task already lost: wasted work
            else:
                req.state = "dead"
                task.done += 1
                task.tries = 0
                task.good_service += req.svc
                last_done[task.id] = now
                if task.done >= task.L:
                    finish(task, "complete", now)
                else:
                    push(now + (rng.choice(emp_tool) if sc.trace else rng.expovariate(1.0 / sc.tool_sec)), "submit", task)
            dispatch(now)
        else:                                            # timeout
            req = payload
            if req.state == "dead":
                continue
            if req.state == "queued":
                qlen -= 1
            req.state = "dead"                           # in-service requests keep running, wasted
            task = req.task
            if task.state == "active":
                retry_or_give_up(task, now)

    row = _summarize(sc, policy, seed, tasks, busy_time, waits)
    row["mean_admitted"] = n_samples[1] / max(n_samples[0], 1)
    row["est_tau"], row["est_s"] = ewma_tau, ewma_s
    return row


def _frac(tasks, state):
    return sum(1 for x in tasks if x.state == state) / max(len(tasks), 1)


def _summarize(sc, policy, seed, tasks, busy_time, waits) -> Dict:
    window = [x for x in tasks if x.arrival >= sc.warmup]
    rec = [x for x in tasks if x.arrival >= sc.recovery_start]
    done = [x for x in window if x.state == "complete"]
    lat = sorted(x.finish - x.arrival for x in done)
    span = sc.duration - sc.warmup
    s_step = sc.step_sec
    long_ = [x for x in window if x.L > 2 * sc.mean_steps]
    good = sum(x.good_service for x in tasks if x.state == "complete")
    lost = [x for x in window if x.state == "lost"]
    row = sc.to_row()
    row.update({
        "policy": policy, "seed": seed, "tasks": len(window),
        "goodput_norm": sum(x.L for x in done) / span / (sc.slots / s_step),   # completed steps / step capacity
        "goodput_tasks": len(done) / span / sc.capacity_tasks_per_sec,
        "long_completed": (sum(1 for x in long_ if x.state == "complete") / len(long_)) if long_ else float("nan"),
        "completed": len(done) / max(len(window), 1),
        "lost": _frac(window, "lost"), "rejected": _frac(window, "rejected"),
        "unfinished": _frac(window, "active") + _frac(window, "new"),
        "recovery_completed": _frac(rec, "complete") if rec else float("nan"),
        "p50_task_s": lat[len(lat) // 2] if lat else float("nan"),
        "p99_task_s": lat[min(len(lat) - 1, int(0.99 * len(lat)))] if lat else float("nan"),
        "mean_wait_s": sum(waits) / len(waits) if waits else float("nan"),
        "p99_wait_s": sorted(waits)[int(0.99 * (len(waits) - 1))] if waits else float("nan"),
        "wasted": 1.0 - good / busy_time if busy_time > 0 else 0.0,
        "lost_steps": sum(x.done for x in lost) / max(len(lost), 1),
    })
    return row
