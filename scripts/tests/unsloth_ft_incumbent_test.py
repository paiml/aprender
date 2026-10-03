#!/usr/bin/env python3
"""CPU self-test of the incumbent side runner (unsloth_ft_incumbent.py).

No torch, no GPU, no unsloth: it tests the parts that decide what the verdict
sees (argument rules, planted flags, token count, receipt fields) by feeding the
runner's receipts to unsloth_ft_verdict.decide itself. Then every mutant below
(one wrong edit to the runner) must make at least one case fail.

  python3 scripts/tests/unsloth_ft_incumbent_test.py     # exit 0 = all pass, all mutants killed
"""
import contextlib
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "bench")
RUNNER = os.path.join(BENCH, "unsloth_ft_incumbent.py")
WRAPPER = os.path.join(BENCH, "unsloth_ft_incumbent_side.sh")
sys.path.insert(0, BENCH)
import unsloth_ft_verdict as V  # noqa: E402

FULL = dict(unsloth="2026.9.12", unsloth_zoo="2026.9.6", torch="2.8.0",
            transformers="5.5.0", fla="0.3.2", causal_conv1d="1.5.2")
GPU = dict(name="NVIDIA RTX 4090", uuid="GPU-0000", trace="[TRACE] device=cuda:0")
ARGV = ["--side", "incumbent", "--model", "Qwen3.5-4B", "--gpu", "0", "--run", "1",
        "--receipt", "/dev/null"]


def load(src):
    m = types.ModuleType("inc")
    m.__file__ = RUNNER
    exec(compile(src, RUNNER, "exec"), m.__dict__)
    return m


def measured(after=True, tokens=200 * 4 * 511, seconds=10.0):
    return dict(precision="bf16", trainable_params=23592960, label_tokens_timed=tokens,
                timed_seconds=seconds, timed_after_compile=after)


def receipt(m, extra=(), vers=None, **kw):
    args = m.parse(ARGV + list(extra))
    return m.build_receipt(args, "ab" * 32, GPU, measured(**kw), dict(vers or FULL))


def apr_twin(r, seconds):
    a = {k: v for k, v in r.items() if k != "versions"}
    a.update(side="apr", apr_version="0.72.0", apr_git_sha="deadbeef", timed_seconds=seconds)
    return a


def verdict(m, extra=(), vers=None, **kw):
    inc = receipt(m, extra, vers, **kw)
    return V.decide([apr_twin(inc, inc["timed_seconds"] / 0.9)] * 3, [inc] * 3)


def refuses(fn):
    try:
        fn()
    except SystemExit as e:
        return e.code != 0
    return False


def c_fields_complete(m):
    inc = receipt(m)
    need = V.COMMON_FIELDS + ("versions",)
    return all(k in inc and inc[k] is not None for k in need) and inc["side"] == "incumbent"


def c_pins_match_verdict(m):
    inc = receipt(m)
    return all(inc[k] == want for k, want in V.PINNED.items())


def c_precision_name(m):
    return (m.precision_name("torch.bfloat16") == "bf16" and m.precision_name("torch.float32") == "fp32"
            and m.precision_name("torch.float8_e4m3fn") == "torch.float8_e4m3fn")


def c_canonical_task(m):
    inc = receipt(m)
    want = dict(rank=16, alpha=32, seq_len=512, batch=4, grad_accum=1,
                warmup_steps=50, timed_steps=200)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return all(inc[k] == v for k, v in want.items()) and sorted(inc["targets"]) == sorted(targets)


def c_pass_ratio(m):
    code, word, _, ratio = verdict(m)
    return code == 0 and word == "PASS" and abs(ratio - 0.9) < 1e-9


def installed(m, blocked):
    """m.versions(blocked) as if every package were installed at 9.9."""
    real = m.importlib.metadata.version
    m.importlib.metadata.version = lambda dist: "9.9"
    try:
        return m.versions(blocked)
    finally:
        m.importlib.metadata.version = real


def c_planted_no_fla(m):
    args = m.parse(ARGV + ["--planted-no-fla"])
    blocked = ("fla", "causal_conv1d") if args.planted_no_fla else ()
    code, word, _, _ = verdict(m, ["--planted-no-fla"], installed(m, blocked))
    return code == 2 and word == "INCUMBENT_SLOW_PATH"


def c_versions_blank_blocked(m):
    v = installed(m, ("fla", "causal_conv1d"))
    return (v["fla"] == "" and v["causal_conv1d"] == "" and v["unsloth"] == "9.9"
            and set(v) == set(V.INCUMBENT_VERSIONS))


def c_planted_adamw8bit(m):
    code, word, reason, _ = verdict(m, ["--planted-incumbent-adamw8bit"])
    return code == 1 and "SAME-WORK" in reason and "optimizer" in reason


def c_compile_in_window(m):
    inc = receipt(m, after=False)
    code, _, reason, _ = verdict(m, after=False)
    return inc["timed_after_compile"] is False and code == 1 and "timed_after_compile" in reason


def c_one_planted(m):
    return refuses(lambda: m.parse(ARGV + ["--planted-no-fla", "--planted-incumbent-adamw8bit"]))


def c_side_is_incumbent(m):
    return refuses(lambda: m.parse(["--side", "apr"] + ARGV[2:]))


def c_label_tokens(m):
    return (m.label_tokens([[5, 6, 0, 0]], 0) == 1 and m.label_tokens([[0, 0]], 0) == 0
            and m.label_tokens([[1, 2, 3], [4, 0, 0]], 0) == 2)


def c_block_fla(m):
    saved = {k: sys.modules.get(k, "absent") for k in ("fla", "causal_conv1d")}
    try:
        sys.modules["fla"] = types.ModuleType("fla")  # importable until blocked
        m.block_fla()
        try:
            __import__("fla")
        except ImportError:
            return True
        return False
    finally:
        for k, v in saved.items():
            if v == "absent":
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def c_no_data_refuses(m):
    old = os.environ.pop("APR_FT_DATA", None)
    try:
        return m.main(ARGV) == 3 and "unsloth" not in sys.modules
    except Exception:  # a mutant that reaches train() fails here
        return False
    finally:
        if old is not None:
            os.environ["APR_FT_DATA"] = old


def c_sha256(m):
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"abc")
    try:
        return m.sha256_file(f.name) == \
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    finally:
        os.unlink(f.name)


CASES = [c_fields_complete, c_pins_match_verdict, c_canonical_task, c_pass_ratio,
         c_planted_no_fla, c_versions_blank_blocked, c_planted_adamw8bit, c_compile_in_window,
         c_one_planted, c_side_is_incumbent, c_label_tokens, c_block_fla, c_no_data_refuses,
         c_sha256, c_precision_name]

# (name, old, new): each is one wrong edit to the runner that some case must catch.
MUTANTS = [
    ("M12 fp32 named bf16", '"torch.float32": "fp32"', '"torch.float32": "bf16"'),
    ("M1 planted adamw8bit ignored", '"adamw_8bit" if', '"adamw_fp32" if'),
    ("M2 no one-token shift", "t != pad_id) - 1, 0)", "t != pad_id), 0)"),
    ("M3 blocked fla still versioned", '"" if key in blocked else ', ""),
    ("M4 down_proj dropped", ', "down_proj"]', "]"),
    ("M5 two planted flags allowed", 'ap.error("one planted falsifier per run")', "pass"),
    ("M6 grad checkpointing on", "grad_checkpointing=False, packing", "grad_checkpointing=True, packing"),
    ("M7 compile flag forced", 'timed_after_compile=measured["timed_after_compile"]',
     "timed_after_compile=True"),
    ("M8 no data check", "if not data or not os.path.isfile(data):", "if False:"),
    ("M9 block_fla no-op", "sys.modules[name] = None", "pass"),
    ("M10 side mislabelled", 'side="incumbent", model', 'side="apr", model'),
    ("M11 rank drift", "rank=16, alpha=32", "rank=8, alpha=32"),
]


def run_cases(m):
    with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
        return [c.__name__ for c in CASES if not c(m)]


def wrapper_refuses():
    env = dict(os.environ, UNSLOTH_INCUMBENT_PROJECT="/nonexistent/unsloth-incumbent")
    p = subprocess.run([WRAPPER] + ARGV, env=env, capture_output=True, text=True)
    return p.returncode == 3 and "REFUSED" in p.stderr


def mutant_survivors(src):
    survivors = []
    for name, old, new in MUTANTS:
        if src.count(old) != 1:
            survivors.append(name + " (edit does not apply)")
        elif not run_cases(load(src.replace(old, new))):
            survivors.append(name)
    return survivors


def main():
    with open(RUNNER, encoding="utf-8") as f:
        src = f.read()
    failed = run_cases(load(src))
    if not wrapper_refuses():
        failed.append("wrapper_refuses")
    survivors = mutant_survivors(src)
    for name in failed:
        print("FAIL case", name)
    for name in survivors:
        print("SURVIVED", name)
    total = len(CASES) + 1 + len(MUTANTS)
    print("%d checks, %d failed" % (total, len(failed) + len(survivors)))
    return 1 if failed or survivors else 0


if __name__ == "__main__":
    sys.exit(main())
