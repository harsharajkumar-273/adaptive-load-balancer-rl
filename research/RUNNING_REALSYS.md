# Running the real-system grid yourself

`research/realsys.py` runs 2,160 configurations. It takes about 40–45 hours on a
4-core machine. The runner can be stopped and restarted at any time: it appends
each finished run to `research/results/realsys.csv` and skips configurations
already there. This branch already contains the finished runs, so a restart
continues where it left off.

## 1. Setup (Linux or macOS, Python 3.10+)

```bash
git clone <this repo> && cd adaptive-load-balancer-rl
git checkout claude/ecstatic-cannon-ot4thw
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r research/requirements.txt

# Redis (Ubuntu: sudo apt install redis-server; macOS: brew install redis)
redis-server --port 6379 --save '' --daemonize yes
redis-cli ping                     # -> PONG
```

On Linux, `taskset` (util-linux) pins each lane to its own cores. Where
`taskset` is missing (macOS), use `--lanes 1` so that parallel runs don't share
CPUs.

## 2. Smoke test (~5 minutes)

```bash
python -m research.realsys --quick
```

It should print 4 lines with no `FAILED`. In `lin_ts` the production agent
herds, so its median latency in seconds is expected.

## 3. Full grid (resumable)

```bash
nohup python -m research.realsys --lanes 2 > realsys.log 2>&1 &
tail -f realsys.log                 # progress: [n/total] ... p50 / p99 / fano
```

- Use `--lanes 2` on 4 cores and `--lanes 4` on 8+ cores, which is about 2×
  faster. Each lane needs 2 cores.
- To stop, run `pkill -f "research.realsys --lanes"`. Finished runs are kept.
  Run the same command again to resume.
- Keep the machine awake, and don't run heavy jobs alongside it: CPU
  contention skews the latencies.
- Commit `research/results/realsys.csv` now and then as a checkpoint.

## 4. Analysis

```bash
python -m research.analyze
```

This writes `research/results/summary.md`, which includes the real-vs-simulator
rank agreement and the real-system tables, plus
`research/figures/fig10_real_vs_sim.png`. The paper's real-system section
(`paper/main.tex`, `\label{sec:eval-real}`) is then filled from these results.

## Troubleshooting

- `FAILED ... start-up took longer than 15s`: the machine is overloaded. Use
  fewer lanes.
- Ports 8100–8499 and 9100–9800 must be free. Lane *i* uses Redis database
  *i + 1*.
- Leftover processes after a crash: `pkill -f src/backend_node.py; pkill -f src.gateway_pool`.
