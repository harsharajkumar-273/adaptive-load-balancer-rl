# Admit tasks, not requests: overload control for agentic LLM workloads

**Question.** When agents run multi-step tasks and retry automatically, can admitting work per *task*
instead of per *request* keep a server stable and productive as offered load grows far past capacity,
and what rule does so without tuning?

Draft: [`../paper/agents.tex`](../paper/agents.tex). Numbers: [`results/summary_agents.md`](results/summary_agents.md).
Simulator: [`agent_sim.py`](agent_sim.py) (one server, C slots, KV memory, tools, timeouts, retries).

## Findings (simulation, 10 seeds)
1. **Per-request control collapses.** At 2.4x overload goodput is 0.26 of capacity with no admission control and
   0.53 with per-request shedding; 87% / 38% of tasks are lost mid-way, 75% / 45% of server time is wasted.
2. **Task-level admission holds capacity.** A task cap, a work budget and the Little's-law cap keep goodput at
   1.00-1.03 from 1.2x to 4x overload, with no task lost mid-way and <1% of server time wasted.
3. **Closed-population formula.** With N admitted tasks, C slots, step time s, tool time tau, the queueing wait is
   max(0, N s/C - s - tau) (verified in simulation: 15 of 18 cells within 0.4 s). Capping N at
   C (s + tau + D)/s therefore holds the wait at D. Safety uses only task counts, not length predictions.
4. **Insensitive and nearly tuning-free.** D from 1 to 10 s gives 1.02-1.03; across 81 regimes the never-tuned
   cap has median 0.99 of the best policy and is within 0.9 of it in 85% of regimes (a per-regime-tuned task cap: 91%;
   one fixed task cap: 65%; per-request shedding: median 0.68).
5. **Length information trades fairness for task counts.** Exact lengths complete 2.3x more tasks at 4x load but
   0% of long tasks complete (vs. 24% for the length-free cap); work goodput is unchanged.
6. **Limits (reported, not hidden).** KV-bound regimes (KV capacity 16 tasks) are the cap's weak spot (worst 0.35 of
   the best); a surge recovery test only separates unprotected servers from protected ones; nothing here is measured
   on a real GPU or a real agent trace.

## Reproduce
```bash
PYTHONPATH=. python -m research.run agents          # overload sweep   (1,740 runs)
PYTHONPATH=. python -m research.run agents_surge
PYTHONPATH=. python -m research.run agents_retries
PYTHONPATH=. python -m research.run agents_regimes  # 81 regimes       (13,770 runs)
PYTHONPATH=. python -m research.analyze_agents      # figures/figA*.png + results/summary_agents.md
PYTHONPATH=. python -m pytest -q tests/test_agent_sim.py
```

## Prior work to position against
Breakwater (OSDI'20), DAGOR (SoCC'18), SEDA (SOSP'01), metastable failures (HotOS'21, OSDI'22); agent serving:
Continuum, TokenCake, SAGA, CacheWise (all arXiv). Delay-targeted concurrency limits are not new; the task unit,
the tool-time-aware closed-population formula and the KV bound are what is specific here. A proper literature
pass (and the Netflix concurrency-limits / Vegas lineage) is still needed before submission.
