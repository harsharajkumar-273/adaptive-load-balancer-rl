# Trust gate for learned routing: does grading the model before acting on it keep regret near the fallback?

**Question.** A learned router must decide, per request, whether to act on its latency model or on a safe default.
A *trust gate* grades each prediction against the realised latency **before the model learns from it**, grades a simple
baseline the same way, and lets the model act only while it beats the baseline over a recent window. Here the
router's fallback is P2C (or JSQ) on the same snapshot; the cost is P99 latency per run; regret is paired by seed.

Implementation: `GatedGreedy` (`research/policies.py`), simulator hooks (`research/sim.py`: `keep=`, `model_step_frac`),
grid `python -m research.run gate` (1,440 runs), tables/figure `python -m research.analyze_trust`
(`results/summary_trust.md`, `figures/figT1_trust_gate_regret.png`), offline replay `python -m research.trust_gate collect|replay`
(`results/summary_trust_replay.md`), tests `tests/test_trust_gate.py`.

## Setup (as built)
- Model: the project's linear latency model; a quadratic variant (the "confidently wrong" model, it herds hardest: `greedy_q2`);
  an EWMA-feature variant (mediocre).
- Grade: a prediction is *ok* if `|prediction - latency| <= 0.5 * max(latency, 1)` in latency-scale units. Baseline: the
  chosen server's smoothed own latency (persistence). Grading continues while the fallback acts, so trust can return.
  The gate starts untrusted and opens only once the model has been ok more often than the baseline over `w=50` graded steps.
  Routing is single-step, so "cancel a plan on its first mismatch" does not apply.
- Competitors: ungated; hard gate to P2C or to JSQ; fixed-probability mixing (act on the model with probability p, else P2C).
- 16 routers, steady (rho 0.9) and gray failure, 50 and 200 ms staleness, synchronised and unsynchronised refresh, 10 seeds.

## Results (paired by seed, regret = (P99 - P99 of fallback) / P99 of fallback; fallback P2C; all 8 cells x 10 seeds)

| model | policy | mean | worst decile | runs worse | worse by >2x |
|:---|:---|:---:|:---:|:---:|:---:|
| linear | ungated | +6.93 | +20.59 | 65% | 50% |
| linear | hard gate to P2C | +2.01 | +4.16 | 24% | 12% |
| linear | hard gate to JSQ | +7.23 | +14.85 | 80% | 46% |
| linear | mix p=.25 / .5 / .75 | -0.18 / +0.08 / +0.60 | -0.10 / +0.06 / +1.92 | 1% / 12% / 31% | 0% / 5% / 16% |
| linear | learned P2C (no gate, for reference) | -0.43 | -0.29 | 2% | 0% |
| quadratic (confidently wrong) | ungated | +167.67 | +340.12 | 100% | 99% |
| quadratic | hard gate to P2C | +99.45 | +318.21 | 84% | 79% |
| quadratic | mix p=.5 | +4.04 | +19.43 | 45% | 26% |
| EWMA (mediocre) | ungated | +21.69 | +56.52 | 68% | 61% |
| EWMA | hard gate to P2C | +1.43 | +0.18 | 14% | 8% |

1. **The ungated model is confidently wrong in a way that matters:** the quadratic model costs +168x (mean) against P2C in every
   run; the linear model is worse than P2C in 65% of runs.
2. **The hard gate helps but does not make it safe.** It cuts the linear model's mean regret from +6.9x to +2.0x and fixes the 50 ms cells
   (regret about -0.2: better than the fallback). It fails at 200 ms synchronised state (gray: +14.9x mean, worst decile +39.9x;
   steady: +1.8x). Against the confidently wrong model it only roughly halves the loss (+99x, 84% of runs worse).
   Longer windows help at 200 ms (w=200: steady -0.12, gray +4.3) but do not remove the failure; w=10 is worse (gray 200 ms: +9.5).
3. **Fixed-probability mixing beat the hard gate here**, with a better mean and no worse tails for the linear model (p=.25: 1% of runs worse),
   but it is not safe either when the model is confidently wrong (p=.5 quadratic: +4.0x, worst decile +19x, 26% worse by >2x).
   This is the opposite ordering of the earlier two-simulator finding (hard gates near zero, mixing unsafe tails).
4. **The fallback matters as much as the gate.** Gating onto JSQ gives +7.2x vs P2C (80% of runs worse), no better than ungated, because JSQ
   herds on stale state too. Learned P2C with no gate at all is never meaningfully worse than P2C (2% of runs), so the project's structural fix
   (random candidate sets) beats gating without a gate.
5. **Why the gate fails (replay of the two grade lists, `summary_trust_replay.md`).** The model is within tolerance about as often as the
   persistence baseline (linear: 0.48 vs 0.48 at 50 ms, 0.43 vs 0.43 at 200 ms), so the gate flips often (it withdraws 49-67% of steps), and its signal
   is weak: of the worst 1% of latencies, the share that occurred while the gate had withdrawn trust is 1.2-1.5x the share of all steps withdrawn at 50 ms,
   but **0.58-0.98x at 200 ms**, i.e. the worst steps are *less* likely than average to be flagged. Outcomes arrive after the response time, which is comparable to
   the refresh window, so a burst is over before its grades arrive.

## What this does and does not show
- It is one domain (shared stale state, many routers) in one simulator. The earlier finding came from other simulators and may hold there.
- A plausible reading, **not tested**: here the harm is collective (every router that trusts the model acts on the same snapshot) and delayed, so a
  per-router accuracy gate cannot see it, whereas mixing randomises. A direct test would give the gate zero-delay grades or share grades across routers.
- Regret is a ratio of P99s; with a ~300 ms fallback a few huge ratios dominate means, so read worst decile and fraction-worse alongside the mean.
  Cells have 10 seeds, so a "worst decile" is one run per cell; the pooled numbers use 80.
- Tolerance 0.5, window and the persistence baseline were not tuned beyond the window sweep. The quadratic model is "confidently wrong" because its fit
  oscillates with the herding it causes, not because an error was injected.
- A bug found and fixed during the work: with `w < min_grades` the gate could never open and trivially matched the fallback (`w=10` rows looked perfect);
  `min_grades` is now capped by `w` (regression test `test_short_window_can_still_open`). `gate_off_greedy` reproduces plain greedy exactly
  (`test_ungated_mode_reproduces_plain_greedy_exactly`).
