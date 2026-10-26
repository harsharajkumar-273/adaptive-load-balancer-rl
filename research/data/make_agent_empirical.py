"""
Extracts the empirical agent-task length and tool-latency distributions from the public
TraceLab coding-agent trace (Claude Code / Codex sessions; https://github.com/uw-syfi/TraceLab,
dataset CC BY 4.0) into a small .npz used by research/agent_sim.py.

    curl -L -o research/data/tracelab/syfi_coding_trace.duckdb \
      https://github.com/uw-syfi/TraceLab/releases/latest/download/syfi_coding_trace.duckdb
    pip install duckdb && python research/data/make_agent_empirical.py
"""
import os

import duckdb
import numpy as np

HERE = os.path.dirname(__file__)
con = duckdb.connect(os.path.join(HERE, "tracelab", "syfi_coding_trace.duckdb"), read_only=True)
steps = con.execute("select count(*) n from rounds where turn_id is not null group by turn_id").fetchnumpy()["n"]
tool = con.execute("select tool_wall_latency_ms / 1000.0 t from tool_calls "
                   "where tool_wall_latency_ms is not null").fetchnumpy()["t"]
np.savez_compressed(os.path.join(HERE, "agent_empirical.npz"), steps=steps.astype(np.int32),
                    tool=tool.astype(np.float32))
print("tasks", len(steps), "mean steps", steps.mean(), "tool calls", len(tool), "mean tool s", tool.mean())
