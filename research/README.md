# Do Learned Load Balancers Herd?

**Question.** Production routing tiers run many gateway replicas that decide
independently from the same, slightly stale shared view of backend load, and
they increasingly score backends with learned latency models. Classic queueing
work shows that shortest-queue routing on stale shared state *herds*: every
dispatcher sees the same "best" server and they all send to it. **Do learned
routers herd too, and what cheap change prevents it?**

The paper draft is in [`../paper/main.tex`](../paper/main.tex). All numbers
below come from [`results/summary.md`](results/summary.md), which reports the
median over 10 seeds with the interquartile range, across 12,630 simulated runs.
These are simulation results under the stated assumptions. The real-system
validation (`realsys.py`) repeats the same grid on real processes: gateways,
backends, Redis and the production LinTS agent. It covers every policy family,
1–64 gateways, synchronised and unsynchronised refresh, and steady and gray
scenarios, at a 10× slower time scale. That run is in progress; its results
will appear in `results/realsys.csv` and in the summary's rank-agreement
section.

## Findings

Unless stated otherwise: 16 gateways, ρ = 0.9, P99 latency.

1. **Learned argmin routing herds worse than JSQ.** With synchronised 50 ms
   state, learned greedy reaches 1,055 ms against 836 ms for JSQ. At 200 ms it
   reaches 12.5 s against 3.1 s.
2. **Desynchronising refresh rescues the classics, not the learner at scale.**
   With independent refresh phases, JSQ improves from 836 to 282 ms at 50 ms,
   but learned greedy still reaches 6.1 s at 200 ms, and 7.0 s with 64
   gateways.
3. **Two herding mechanisms.**
   - *Within a gateway* (temporal): local correction fixes it. At K=1, greedy
     goes from 630 to 100 ms.
   - *Across gateways* (agreement): local correction fails, with greedy+lc at
     8.1 s at K=64. Per-request Thompson sampling helps because independently
     trained posteriors disagree: 763 ms at K=1, 201 ms at K=16.
4. **Control intervals are hidden staleness.** Recomputing weights every
   150 ms (as `src/agent.py` does) gives 204 ms even with live state, against
   72 ms per request. The cost grows with the interval: 137 ms at 10 ms and
   800 ms at 500 ms.
5. **Learned power-of-d bounds herding.** Sample d backends uniformly, then
   send to the one with the lowest learned prediction. No backend receives more
   than d/N of the requests, whatever the staleness or gateway count. Without
   probes, P2C over learned scores stays between 108 and 282 ms in all 28
   steady-state configurations.
6. **Probing: learned power-of-d vs. Prequal at equal cost.**
   - At 3 probes per request, learned P3C gets 84 ms against 127 ms steady, and
     112 against 144 ms under gray failure.
   - With zero probes, P2C over learned scores (151 ms) beats Prequal limited to
     1 probe per request (180 ms).
7. **What learning buys.** In steady state, learned P3C matches SED(3), which
   is told the true server speeds (84 vs. 83 ms). Under a gray failure the
   oracle's speed knowledge is stale (206 ms) and the learned model is not
   (112 ms). With heavy-tailed service (CV 4) the oracle leads (419 vs.
   483 ms): noisy service slows learning.

## Simulator (`sim.py`)

| Component | Model |
|---|---|
| Servers | 8 FIFO single-server queues, rates 200/200/150/150/100/100/50/50 req/s; exponential or lognormal service (`service_cv`) |
| Arrivals | Poisson at ρ × 1000, or a two-state Markov-modulated process (`burstiness`); each request goes to one of K gateways uniformly |
| Shared state | queue-length snapshot every Δ s, with synchronised or per-gateway (`desync`) refresh phases |
| Feedback | each gateway sees the latency of its own requests, at completion |
| Probing | live requests-in-flight plus per-RIF latency estimate; synchronous probes are charged 0.5 ms |
| Gray failure | fastest server drops to 20% speed at t = 12 s (run at ρ = 0.75 so the system stays stable) |

It is validated against M/M/1 theory and the classic stale-JSQ result
(`tests/test_research_sim.py`).

## Policies (`policies.py`)

- **Heuristics on the snapshot:** weighted random, JSQ, SED, P2C.
- **Learned** (per-gateway discounted Bayesian linear model, latency ≈ a + b·queue):
  - greedy
  - per-request Thompson sampling
  - `ts_epoch`: 150 ms weights, mirroring `src/agent.py`
  - `p2c_learned` / `p3c_learned`
- **Probing:**
  - `prequal`: re-implementation, tuned in its favour
  - `p2c_probe`, `p3c_probe`, `sed3_probe`: classic power-of-d over live probes
  - `p2c_learned_probe`, `p3c_learned_probe`
- **`+lc` suffix:** local correction.

## Limitations

- Single-server FIFO backends, a two-feature model, and one fleet shape.
- The Prequal baseline is a re-implementation from the paper's description.
- Our probing variants probe synchronously; with sub-millisecond services that
  round-trip matters more than it does here.
- The related-work comparison is not complete yet (see the TODOs in the paper).

## Reproduce

```bash
pip install -r requirements.txt -r research/requirements.txt
python -m research.run all        # ~35 min on 4 cores
python -m research.analyze        # figures/ + results/summary.md
redis-server --daemonize yes
python -m research.realsys        # real-system grid: 2,160 runs, ~40-45 h on 4 cores, resumable
PYTHONPATH=. pytest -q
```
