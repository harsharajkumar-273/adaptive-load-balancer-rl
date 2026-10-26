# Admit tasks, not requests: overload control for agentic LLM workloads

**Question.** When agents run multi-step tasks and retry automatically, can admitting work per *task*
instead of per *request* keep a server stable and productive as offered load grows far past capacity,
and what rule does so without tuning?

Draft: [`../paper/agents.tex`](../paper/agents.tex). Numbers: [`results/summary_agents.md`](results/summary_agents.md).
Simulator: [`agent_sim.py`](agent_sim.py) (one server, C slots, KV memory, tools, timeouts, retries).

## Findings (simulation, 10 seeds)
Two workloads: *synthetic* (geometric task lengths, exponential tool times) and *trace-calibrated* (lengths and tool
times resampled from 32k turns / 743k tool calls of real Claude Code and Codex sessions, TraceLab, CC BY 4.0;
extraction script `data/make_agent_empirical.py`). Headline numbers below are trace-calibrated unless marked.

0. **Trace-calibrated, 2.4x overload:** no control 0.21 goodput (71% of tasks lost, 81% of server time wasted), per-request
   shedding 0.61 (21% lost, 41% wasted), task cap / Little's-law cap 0.94 (0 lost, ~1% wasted). Without tuning the
   Little's-law cap (D = 0.1 x timeout) is within 0.9 of the best policy in 67% of 18 trace regimes (median 0.96, worst 0.60).
   With a 16-task KV memory, per-request shedding beats every task-level policy (0.36 vs 0.25).

Original synthetic findings:
1. **Per-request control collapses.** At 2.4x overload goodput is 0.26 of capacity with no admission control and
   0.53 with per-request shedding; 87% / 38% of tasks are lost mid-way, 75% / 45% of server time is wasted.
2. **Task-level admission holds capacity.** A task cap, a work budget and the Little's-law cap keep goodput at
   1.00-1.03 from 1.2x to 4x overload, with no task lost mid-way and <1% of server time wasted.
3. **Closed-population formula.** With N admitted tasks, C slots, step time s, tool time tau, the queueing wait is
   max(0, N s/C - s - tau) (verified in simulation: 15 of 18 cells within 0.4 s). Capping N at
   C (s + tau + D)/s therefore holds the wait at D. Safety uses only task counts, not length predictions.
4. **Insensitive and nearly tuning-free.** D from 1 to 10 s (or 0.05-0.2 x timeout) gives 1.02-1.03; a fixed D = 5 s failed for a 15 s timeout on trace tasks, hence D = 0.1 x timeout; across 81 regimes the never-tuned
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
PYTHONPATH=. python -m research.run agents_trace          # trace-calibrated sweep
PYTHONPATH=. python -m research.run agents_trace_regimes
PYTHONPATH=. python -m research.analyze_agents      # figures/figA*.png + results/summary_agents.md
PYTHONPATH=. python -m pytest -q tests/test_agent_sim.py
```

## Prior work to position against
Session-based admission control (Cherkasova & Phaal, 2002) is the closest classic idea; also Breakwater (OSDI'20),
DAGOR (SoCC'18), SEDA (SOSP'01), Netflix concurrency-limits, metastable failures (HotOS'21, OSDI'22); agent serving:
Continuum, TokenCake, SAGA, CacheWise, SMetric, Talaria (arXiv). Delay-targeted limits and session admission are not new;
what is specific here is the tool-time-aware closed-population formula, measuring service on abandoned work, the KV bound,
the length-information analysis and the real-trace evaluation. Bibliography entries for arXiv items were written from search
results and need checking.
