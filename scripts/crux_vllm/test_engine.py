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
    ("real gx10 root cause is quoted", [HERE / "fixtures" / "gx10-memory-profiling.txt"],
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


def snapshot(files: dict, rewrite: str | None = None, plain: str | None = None, xet: bool = False) -> Path:
    """A fake HF snapshot: blobs/<content hash> + snapshots/rev/<name> -> blob. `rewrite` names a file whose
    blob is then overwritten through its symlink (the #3971 shape); `plain` a file that is not a link."""
    import hashlib
    root = Path(tempfile.mkdtemp())
    (root / "blobs").mkdir()
    snap = root / "snapshots" / "rev"
    snap.mkdir(parents=True)
    for fname, data in files.items():
        if fname.endswith(".safetensors"):
            name = hashlib.sha256(data).hexdigest()
        else:
            name = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if xet and fname.endswith(".safetensors"):
            # huggingface_hub 1.x: blobs/<sha256> is itself a link into blobs/<xx>/<content-defined hash>
            (root / "blobs" / "b6").mkdir(exist_ok=True)
            (root / "blobs" / "b6" / ("b6" + "0" * 62)).write_bytes(data)
            (root / "blobs" / name).symlink_to(Path("b6") / ("b6" + "0" * 62))
        else:
            (root / "blobs" / name).write_bytes(data)
        if fname == plain:
            (snap / fname).write_bytes(data)
        else:
            (snap / fname).symlink_to(Path("../../blobs") / name)
    if rewrite:
        (snap / rewrite).write_bytes(files[rewrite] + b"rewritten")
    return snap


def verify(snap: Path) -> str:
    try:
        engine.crux_hf_verify.verify_snapshot_dir(snap)
        return "PASS"
    except RuntimeError as e:
        return str(e)


engine.crux_hf_verify.STAMPS = Path(tempfile.mkdtemp()) / "stamps.json"
FILES = {"model.safetensors": b"\x00" * 4096, "config.json": b'{"model_type": "qwen2"}'}
VERIFY = [
    # (name, snapshot, must contain)
    ("an intact snapshot verifies (positive control)", snapshot(FILES), "PASS"),
    ("a safetensors blob rewritten through its link is refused by name",
     snapshot(FILES, rewrite="model.safetensors"), "cached model.safetensors is not the file its name promises"),
    ("the refusal names the content sha256 and #3971", snapshot(FILES, rewrite="model.safetensors"), "#3971"),
    ("a rewritten config.json (git blob sha1) is refused", snapshot(FILES, rewrite="config.json"),
     "content git-blob-sha1"),
    ("the hub-1.x two-hop layout (sha256 link -> xet blob) verifies (positive control)",
     snapshot(FILES, xet=True), "PASS"),
    ("a two-hop blob rewritten through its link is refused", snapshot(FILES, rewrite="model.safetensors", xet=True),
     "cached model.safetensors is not the file its name promises"),
    ("a snapshot file that is not a link cannot be verified", snapshot(FILES, plain="config.json"),
     "is not a link into blobs/"),
]
for name, snap, must in VERIFY:
    got = verify(snap)
    ok = must in got if must != "PASS" else got == "PASS"
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got: {got[:300]}"))
    failed += not ok

# A verified blob is stamped and not re-hashed; a rewrite changes size, so it IS re-hashed and refused.
snap = snapshot(FILES)
first, second = engine.crux_hf_verify.verify_snapshot_dir(snap), engine.crux_hf_verify.verify_snapshot_dir(snap)
(snap / "model.safetensors").write_bytes(FILES["model.safetensors"] + b"rewritten")
after = verify(snap)
ok = first == 2 and second == 0 and "not the file its name promises" in after
print(f"{'ok  ' if ok else 'FAIL'} a stamp skips re-hashing, and a later rewrite is still caught"
      + ("" if ok else f"\n     got: {first} {second} {after[:200]}"))
failed += not ok
CASES_TOTAL = len(CASES) + len(VERIFY) + 1

# ── batch mode (shared table) + the vLLM-only preflight fan-out ──────────────────────────────────────────
sys.path.insert(0, str(HERE.parent / "lib"))
import crux_batch_cases as bc  # noqa: E402


def _no_cpu(a):
    if a.backend != "gpu":
        raise RuntimeError("the pinned vLLM wheel is the CUDA build and has no CPU backend; a cpu-lane cell cannot "
                           "run on it")


engine.preflight = _no_cpu
failed += bc.run(engine, "vllm serve")
failed += bc.run_sse()
bc.LOADS.update(inproc=0, serve=0)
_w = bc.batch_env()
engine.load_inproc = bc.fake_inproc()
engine.run_batch(bc.model_args(backend="cpu"), [{"prompt_id": "p", "verb": "run", "messages": bc.msgs(_w, "m", ["q"]),
                                                  "thinking": "off", "max_tokens": 5}])
_r = bc.rows(_w)
_ok = bc.LOADS["inproc"] == 0 and "no CPU backend" in (_r[0]["refused"] or "")
print(f"{'ok  ' if _ok else 'FAIL'} [batch] a preflight refusal fans out and loads nothing")
failed += not _ok
CASES_TOTAL += bc.CASE_COUNT + bc.SSE_CASE_COUNT + 1

for name, logs, must, must_not in CASES:
    got = engine.refusal(GENERIC, *logs)
    ok = got.startswith(f"RuntimeError: {GENERIC}") and (must is None or must in got) and (
        must_not is None or must_not not in got)
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got: {got[:300]}"))
    failed += not ok
print(f"{CASES_TOTAL - failed}/{CASES_TOTAL} cases")
sys.exit(1 if failed else 0)
