#!/usr/bin/env python3
"""Print the `results=` line fat_driver.emit_results_output writes for a mutants section that deferred
3 NOT_MEASURED mutants. Used by scripts/check_ci_gate_mutants_cuda_rule.sh to feed the gate rule the
driver's REAL emission rather than a hand-written JSON (#4621)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fat_driver  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    os.environ["GITHUB_OUTPUT"] = os.path.join(td, "o")
    fat_driver.emit_results_output({"mutants": {"result": "success", "continue_on_error": False,
                                                "outputs": {"not_measured": "3", "not_measured_sha": "aa11"}}})
    with open(os.environ["GITHUB_OUTPUT"]) as f:
        print(next(line for line in f if line.startswith("results="))[len("results="):].strip())
