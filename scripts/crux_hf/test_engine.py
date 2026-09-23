"""Batch-mode case table for the hf driver (#3952; cop 2026-09-23). Stdlib only, no torch/transformers:
`python3 scripts/crux_hf/test_engine.py`. The cases are shared with the vLLM driver (scripts/lib/crux_batch_cases.py).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import crux_batch_cases as bc  # noqa: E402
import engine  # noqa: E402

failed = bc.run(engine, "transformers serve") + bc.run_sse() + bc.run_proc()
total = bc.CASE_COUNT + bc.SSE_CASE_COUNT + bc.PROC_CASE_COUNT
print(f"{total - failed}/{total} cases")
sys.exit(1 if failed else 0)
