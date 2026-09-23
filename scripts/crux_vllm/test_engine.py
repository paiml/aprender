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

# The REAL engine loaders, kept before the batch cases below swap in fakes (#4029 drives these, not a stub).
REAL_LOAD_INPROC, REAL_SERVE_SESSION = engine.load_inproc, engine.serve_session

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
failed += bc.run_proc()
bc.LOADS.update(inproc=0, serve=0)
_w = bc.batch_env()
engine.load_inproc = bc.fake_inproc()
engine.run_batch(bc.model_args(backend="cpu"), [{"prompt_id": "p", "verb": "run", "messages": bc.msgs(_w, "m", ["q"]),
                                                  "thinking": "off", "max_tokens": 5}])
_r = bc.rows(_w)
_ok = bc.LOADS["inproc"] == 0 and "no CPU backend" in (_r[0]["refused"] or "")
print(f"{'ok  ' if _ok else 'FAIL'} [batch] a preflight refusal fans out and loads nothing")
failed += not _ok
CASES_TOTAL += bc.CASE_COUNT + bc.SSE_CASE_COUNT + bc.PROC_CASE_COUNT + 1

# ── #4029: vLLM must be HANDED context + the largest budget of the batch's runnable items ─────────────────
# The REAL load_inproc and serve_session run; only what they call out to is faked, and the fake records what vLLM
# was given, then stops the load. An item refused before any engine (bad verb) must not size the engine.
import types  # noqa: E402


class _Stop(Exception):
    pass


_given = {}


def _fake_llm(**kw):
    _given["inproc"] = kw["max_model_len"]
    raise _Stop("fake LLM: stop after recording max_model_len")


def _preflight_refuses(_a):
    raise RuntimeError("preflight: refused for the #4029 case")


def _fake_popen(cmd, **_kw):
    _given["serve"] = int(cmd[cmd.index("--max-model-len") + 1])
    raise _Stop("fake Popen: stop after recording --max-model-len")


# Everything the case replaces is restored in a finally, so no later case runs against these fakes.
_saved_mods = {m: sys.modules.get(m) for m in ("vllm", "transformers")}
_saved_attrs = {k: getattr(engine, k) for k in ("verified_source", "subprocess", "preflight", "load_inproc",
                                                "serve_session")}
try:
    sys.modules["vllm"] = types.SimpleNamespace(LLM=_fake_llm, SamplingParams=None)
    sys.modules["transformers"] = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _p: None))
    engine.verified_source = lambda _repo, _rev: Path("/nonexistent-source")
    engine.subprocess = types.SimpleNamespace(Popen=_fake_popen, STDOUT=None)
    engine.preflight = lambda a: None
    engine.load_inproc, engine.serve_session = REAL_LOAD_INPROC, REAL_SERVE_SESSION
    # (label, path, context, items as (verb, max_tokens, thinking), expected max_model_len, how). The incident numbers
    # (context 4096, ON budget 4096) plus a context that differs from every budget, so 2*context or a hardcoded +4096
    # cannot pass; one past 8192 and one at odd sizes, so no clamp or rounding can. A bad-verb item and a serve item
    # refused for sharing the batch with run items must not size the engine. Sizing never depends on thinking.
    # how: "batch" (run_batch), "gen" (the single-cell verb), "preflight" (preflight refuses: rows still record the
    # size the engine would have had), "none" (every item refused: nothing loads, rows record the context alone).
    _cases = [
        ("inproc incident", "inproc", 4096,
         [("run", 5, "on"), ("run", 4096, "on"), ("run", 300, "on"), ("no-such-verb", 100_000, "on")], 8192, "batch"),
        ("serve incident", "serve", 4096, [("serve run", 5, "on"), ("serve run", 4096, "on"), ("serve run", 300, "on"),
                                           ("no-such-verb", 100_000, "on")], 8192, "batch"),
        ("inproc context != budget", "inproc", 2048, [("run", 5, "on"), ("run", 1024, "on"), ("chat", 900, "on")],
         3072, "batch"),
        ("serve context != budget", "serve", 2048, [("serve run", 900, "on"), ("serve stream", 1024, "on")], 3072,
         "batch"),
        ("mixed: refused serve item does not size the run engine", "inproc", 4096,
         [("run", 5, "on"), ("serve run", 100_000, "on")], 4101, "batch"),
        ("serve past 8192", "serve", 16384, [("serve run", 100, "on"), ("code", 4096, "on")], 20480, "batch"),
        ("inproc odd sizes", "inproc", 1001, [("chat", 7, "on"), ("run", 3, "on")], 1008, "batch"),
        ("thinking off sizes the same", "inproc", 4096, [("run", 5, "on"), ("run", 4096, "off")], 8192, "batch"),
        ("serve, thinking unset", "serve", 2048, [("serve run", 1500, "unset")], 3548, "batch"),
        ("gen: one cell, the incident", "inproc", 4096, [("run", 4096, "off")], 8192, "gen"),
        ("preflight refusal still records the size", None, 4096, [("run", 4096, "on"), ("run", 5, "on")], 8192,
         "preflight"),
        ("every item refused: rows record the context alone", None, 3000,
         [("no-such-verb", 4096, "on"), ("run", 0, "on")], 3000, "none"),
    ]
    for _label, _path, _ctx, _items, _want, _how in _cases:
        _given.clear()
        _w = bc.batch_env()
        engine.preflight = (lambda a: None) if _how != "preflight" else _preflight_refuses
        _batch = [{"prompt_id": f"p{i}", "verb": v, "messages": bc.msgs(_w, f"p{i}", ["q"]), "thinking": th,
                   "max_tokens": n} for i, (v, n, th) in enumerate(_items)]
        if _how == "gen":
            (_one,) = _batch
            engine.gen(bc.model_args(context=_ctx, **_one))
        else:
            engine.run_batch(bc.model_args(context=_ctx), _batch)
        _r = bc.rows(_w)
        _loaded = [r for r in _r if "fake" in (r.get("refused") or "")]
        _ok = (len(_r) == len(_items) and all(r.get("max_model_len") == _want for r in _r)
               and (_given.get(_path) == _want and _loaded if _path else not _given and not _loaded))
        print(f"{'ok  ' if _ok else 'FAIL'} [#4029 {_label}] vLLM is handed context + the largest budget it will run, "
              "and every row records it" + ("" if _ok else f"\n     vLLM got {_given}, want {_want}; rows "
                                             f"{[(r.get('max_model_len'), (r.get('refused') or '')[:50]) for r in _r]}"))
        failed += not _ok
        CASES_TOTAL += 1
finally:
    for _m, _v in _saved_mods.items():
        if _v is None:
            sys.modules.pop(_m, None)
        else:
            sys.modules[_m] = _v
    for _k, _v in _saved_attrs.items():
        setattr(engine, _k, _v)
_ok = (engine.subprocess is __import__("subprocess")
       and all(sys.modules.get(m) is v for m, v in _saved_mods.items())
       and all(getattr(engine, k) is v for k, v in _saved_attrs.items()))
print(f"{'ok  ' if _ok else 'FAIL'} [#4029] the fakes are gone once the case ends")
failed += not _ok
CASES_TOTAL += 1

for name, logs, must, must_not in CASES:
    got = engine.refusal(GENERIC, *logs)
    ok = got.startswith(f"RuntimeError: {GENERIC}") and (must is None or must in got) and (
        must_not is None or must_not not in got)
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got: {got[:300]}"))
    failed += not ok
print(f"{CASES_TOTAL - failed - bc.ENV_CASES}/{CASES_TOTAL} cases" + (f", {bc.ENV_CASES} not measurable here (ENV)" if bc.ENV_CASES else ""))
sys.exit(1 if failed else (2 if bc.ENV_CASES else 0))
