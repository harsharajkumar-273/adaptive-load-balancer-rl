# Do Learned Load Balancers Herd?

**Question.** Production routing tiers run many router replicas that decide
independently from the same, slightly stale view of backend state, and they
increasingly score backends with learned models. Classic queueing work shows
that shortest-queue routing on stale shared state *herds*: every router picks
the same "best" server. **Do learned routers herd too, why, and what prevents it?**

The paper draft is [`../paper/main.tex`](../paper/main.tex). Every number below
comes from [`results/summary.md`](results/summary.md) (median and interquartile
range over 10 seeds; 56,050 simulated runs). These are simulation results under
the stated assumptions. The real-system validation (`realsys.py`, instructions in
[`RUNNING_REALSYS.md`](RUNNING_REALSYS.md)) is a separate, long-running experiment
whose results are not yet part of the paper.

## Findings

Unless stated otherwise: 16 routers, queueing model, ρ = 0.9, P99 latency.

1. **Learned argmin routing herds worse than JSQ.** With synchronised 50 ms
   state, learned greedy reaches 1,055 ms against 836 ms for JSQ; at 200 ms it
   reaches 12.5 s against 3.1 s. Desynchronising refresh phases rescues JSQ
   (282 ms at 50 ms) but not learned greedy (6.1 s at 200 ms, 7.0 s with 64 routers).
2. **A burst bound explains it** (paper §3). A deterministic function of the
   shared snapshot sends a whole refresh window to one server (Prop. 1); a
   candidate set drawn independently of the snapshot caps a server's burst at
   κ·λ·Δ (Prop. 2; κ = d/N for uniform sampling). Measured share of a 200 ms
   window going to the busiest server: JSQ 0.79, SED 0.96 (stampede); weighted
   random 0.22 (cap 0.20), P2C 0.26 (cap 0.25), learned P2C 0.25, learned P3C
   0.32 (cap 0.375); learned greedy 0.43 and Thompson 0.46 sit in between
   because per-router models disagree.
3. **The learner is not the problem.** A quadratic queue term, the router's own
   latency history, and an optimistic (LinUCB-style) learner do not fix argmin
   herding and can make it much worse (greedy 1.1 s → 59 s with the quadratic
   term). Power-of-d is insensitive to the model: 141–153 ms at 50 ms staleness
   for every feature set.
4. **Control intervals are hidden staleness.** Recomputing weights every 150 ms
   (as `src/agent.py` does) gives 204 ms even with live state, against 72 ms per
   request, and 800 ms at a 500 ms interval.
5. **Remedy 1: learned power-of-d.** Sample d candidates, rank them with the
   learned model. Without probes, steady-state P99 stays within 108–282 ms across
   all 28 combinations of staleness, router count and refresh phase. With 3
   live probes it matches an oracle that knows server speeds (84 vs. 83 ms) and
   beats a Prequal-style prober under a gray failure (112 vs. 144 ms).
6. **Remedy 2: age-aware stochastic split** (our construction, inspired by
   stochastic coordination but not that algorithm). Equalises expected delay given
   the snapshot's age and the other routers' expected arrivals. With learned
   service rates it is the best router in every gray-failure cell (e.g. 213 vs.
   283 ms for P3C at 200 ms, unsynchronised) and in steady state with
   unsynchronised refresh with nominal rates (147 vs. 193 ms). It loses in steady
   state with synchronised refresh at 200 ms (296 vs. 238 ms for P2C).
7. **LLM serving with prefix caches** (`llm_sim.py`; paper §5). Plain prefix
   affinity herds onto the replica holding a hot prefix (26.9 s P99 vs. 252 ms
   with a load threshold T=4); learned greedy is best with fresh state but herds
   when stale (2.4 s at 200 ms). *Ring-d* (candidates = the d rendezvous-hash
   replicas of the prefix, ranked by a learned model with a cache-hit feature) keeps
   locality and balance with no threshold: over 108 workload regimes, with T
   chosen on held-out seeds, ring-2 has median regret 1.06×, worst 1.98×, whereas
   five different T values win in different regimes and the best single value
   (T=4) is up to 11.2× worse (others up to 133×). Under a replica gray failure
   ring-3 beats the best threshold at every staleness level.

8. **Real traces** (`trace_sim.py`, Mooncake FAST'25 traces in `research/data/`; paper §5).
   Replaying production arrivals, lengths and KV-block sharing (40 cells: 2 traces ×
   5 loads × 2 staleness × healthy/gray) confirms herding (JSQ and learned greedy
   collapse at high load: slowdown 884/805 vs 78 for affinity) and that no fixed
   threshold is safe, but the best threshold flips from T=4 (synthetic) to T≥16, and
   **ring-d is not the best router there**: median regret 1.13× (cost-aware 1.09×),
   worst 5.7× (3.0×) at saturation, versus 1.00× for plain affinity / large T.
   Reproduce: `python -m research.run trace; python -m research.run trace_cost`.

## Simulators

| | `sim.py` (queueing model) | `llm_sim.py` (LLM serving) |
|---|---|---|
| Servers | 8 FIFO single-server queues, rates 200…50 req/s; exponential or lognormal service | 8 prefill replicas, FIFO, LRU prefix cache (24 prefixes), prefill 8k tokens/s |
| Arrivals | Poisson (or Markov-modulated) at ρ × 1000 req/s, spread uniformly over K routers | Poisson, 64 prefix classes with Zipf popularity, unique suffix per request |
| Shared state | queue-length snapshot every Δ, synchronised or per-router phase | queue length **and cached-prefix set** every Δ |
| Feedback | each router sees the latency of its own requests at completion | same (TTFT) |
| Faults | fastest server drops to 20% speed at t = 12 s (ρ = 0.75) | one replica drops to 25% speed at t = 20 s |
| Herding metrics | Fano factor of arrivals; busiest server's share of a refresh window | same |

Both are exact event simulations (FIFO makes each departure time known at
arrival), validated against M/M/1 theory and the classic stale-JSQ result
(`tests/test_research_sim.py`, `tests/test_llm_sim.py`).

## Policies

- **Heuristics (snapshot):** weighted random, JSQ, SED, P2C; LLM: JSQ, P2C,
  static hashing, prefix affinity, affinity + load threshold T ∈ {1, 2, 4, 8, 16, 32}.
- **Learned (snapshot, `policies.py`, `llm_sim.py`):** greedy, Thompson, TS-epoch
  (mirrors `src/agent.py`), LinUCB, P2C/P3C, ring-2/ring-3; model variants
  `q`, `q²`, `q + own latency`; `+lc` = local correction.
- **Age-aware split:** `awf` (nominal rates), `awf_learned`.
- **Probing:** Prequal-style prober (re-implementation, tuned in its favour),
  JSQ(d), SED(d), learned power-of-d with live probes.

## Reproduce

```bash
pip install -r requirements.txt -r research/requirements.txt
python -m research.run all        # every simulator experiment, about 1.5 h on 4 cores
python -m research.analyze        # figures/ + results/summary.md
PYTHONPATH=. pytest -q
```

`research.run` also takes a single experiment name: `main`, `robust`, `frontier`,
`probing`, `epoch`, `load`, `tuning`, `models`, `llm` or `llm_regimes`
(`--quick` runs a small smoke-test grid).

## Limitations

- Single-server FIFO backends. The LLM model has no batching, no decode phase,
  and one load signal; its constants are plausible, not measured on a trace.
- Linear latency models (the variants tested do not change the conclusions).
- The Prequal baseline is a re-implementation from the paper's description.
- The age-aware split needs the number of routers and an estimate of the total
  arrival rate.
- Real-system validation is in progress and not yet part of the paper's results.
