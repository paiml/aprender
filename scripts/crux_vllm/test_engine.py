"""Case table for the vLLM engine's refusal text (#3952). Stdlib only: `python3 scripts/crux_vllm/test_engine.py`.

The refusal must carry the engine core's ROOT error, because the parent process only sees "Engine core
initialization failed. See root cause above." The fixture is the tail of the real gx10 failure (2026-09-23).
"""

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine  # noqa: E402

GENERIC = RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")


def log(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return Path(f.name)


CASES = [
    # (name, logs, must contain, must NOT contain)
    ("real gx10 root cause is quoted", [HERE / "fixtures" / "gx10-memory-profiling.log"],
     "engine core: AssertionError: Error in memory profiling", None),
    ("a clean log adds nothing", [log("INFO all good\n")], None, "engine core:"),
    ("the generic parent line is never its own root", [log(f"RuntimeError: {GENERIC}\n")], None, "engine core:"),
    ("a generic line logged AFTER the root does not hide it",
     [log(f"x AssertionError: the real cause\nRuntimeError: {GENERIC}\n")],
     "engine core: AssertionError: the real cause", None),
    ("a missing log adds nothing", [Path("/nonexistent/x.log")], None, "engine core:"),
    ("the first log with a root wins", [log("INFO\n"), log("x ValueError: No available memory for the cache blocks\n")],
     "engine core: ValueError: No available memory", None),
    ("the LAST error line is the root", [log("KeyError: early\nx AssertionError: late\n")],
     "engine core: AssertionError: late", "early"),
]

failed = 0
for name, logs, must, must_not in CASES:
    got = engine.refusal(GENERIC, *logs)
    ok = got.startswith(f"RuntimeError: {GENERIC}") and (must is None or must in got) and (
        must_not is None or must_not not in got)
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got: {got[:300]}"))
    failed += not ok
print(f"{len(CASES) - failed}/{len(CASES)} cases")
sys.exit(1 if failed else 0)
