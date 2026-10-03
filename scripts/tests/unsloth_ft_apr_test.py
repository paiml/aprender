#!/usr/bin/env python3
"""CPU self-test of the apr side adapter (unsloth_ft_apr.py).

No apr build, no GPU: a stub `apr` (APR_BIN) writes the receipt a finished R15
would write, or only the train-run-receipt-v1 base, or fails. The adapter's
receipts go through unsloth_ft_verdict.decide itself. Every mutant below (one
wrong edit to the adapter) must make at least one case fail.

It also prints R15's receipt requirements: the keys a TRR-only receipt lacks.

  python3 scripts/tests/unsloth_ft_apr_test.py     # exit 0 = all pass, all mutants killed
"""
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "bench")
ADAPTER = os.path.join(BENCH, "unsloth_ft_apr.py")
SIDE = os.path.join(BENCH, "unsloth_ft_apr_side.sh")
sys.path.insert(0, BENCH)
import unsloth_ft_verdict as V  # noqa: E402

FULL = 200 * 4 * 1 * 511
TRR_BASE = dict(status="success", apr_version="0.72.0", apr_git_sha="0123abcd",
                data_sha256=["d" * 64], recipe_sha256="r" * 64, seed=42, base_sha256=["b" * 64],
                backend="cuda", device="cuda:0", started_at="t0", ended_at="t1", steps=250,
                final_loss=1.0, output_sha256="o" * 64)
TRR_EXTENSION = [
    "device.name", "device.uuid", "device.trace_line", "recipe.rank", "recipe.alpha",
    "recipe.targets", "recipe.optimizer", "recipe.grad_checkpointing", "recipe.packing",
    "recipe.seq_len", "recipe.batch", "recipe.grad_accum", "recipe.warmup_steps",
    "recipe.timed_steps", "trainable_params", "timed_window.label_tokens",
    "timed_window.seconds", "timed_window.after_compile",
]
ARGV = ["--side", "apr", "--model", "Qwen3.5-4B", "--gpu", "0", "--run", "1"]

# A stub apr: reads its own flags, writes the receipt STUB_APR_MODE asks for.
STUB_APR = r'''#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
val = lambda f: a[a.index(f) + 1]
mode = os.environ.get("STUB_APR_MODE", "full")
if mode == "fail":
    sys.exit(5)
r = dict(status="success", apr_version="0.72.0", apr_git_sha="0123abcd",
         data_sha256=["d" * 64], device=dict(kind="cuda:0"))
if mode == "full":
    r["device"] = dict(name="NVIDIA GeForce RTX 4090", uuid="GPU-4090-a",
                       trace_line="[TRACE] device=cuda:0")
    r["recipe"] = dict(rank=int(val("--rank")), alpha=int(val("--alpha")),
                       targets=val("--targets").split(","), optimizer=val("--optimizer"),
                       grad_checkpointing="--no-grad-checkpointing" not in a,
                       packing="--no-packing" not in a, seq_len=int(val("--max-seq-len")),
                       batch=int(val("--batch-size")), grad_accum=int(val("--grad-accum")),
                       warmup_steps=int(val("--warmup-steps")), timed_steps=int(val("--timed-steps")))
    n = len(r["recipe"]["targets"])
    r["trainable_params"] = 23592960 * n // 7
    r["timed_window"] = dict(label_tokens=200 * 4 * 511, seconds=10.0, after_compile=True)
with open(val("--receipt"), "w") as f:
    json.dump(r, f)
'''


def load(src):
    m = types.ModuleType("apr_side")
    m.__file__ = ADAPTER
    exec(compile(src, ADAPTER, "exec"), m.__dict__)
    return m


def full_trr(**recipe):
    rec = dict(rank=16, alpha=32, targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                           "up_proj", "down_proj"],
               optimizer="adamw_fp32", grad_checkpointing=False, packing=False, seq_len=512,
               batch=4, grad_accum=1, warmup_steps=50, timed_steps=200)
    rec.update(recipe)
    t = dict(TRR_BASE)
    t.update(device=dict(name="NVIDIA GeForce RTX 4090", uuid="GPU-4090-a",
                         trace_line="[TRACE] device=cuda:0"),
             recipe=rec, trainable_params=23592960,
             timed_window=dict(label_tokens=FULL, seconds=10.0, after_compile=True))
    return t


def incumbent_twin(apr, ratio=0.9):
    r = {k: v for k, v in apr.items() if k not in ("apr_version", "apr_git_sha")}
    r.update(side="incumbent", timed_seconds=apr["timed_seconds"] * ratio,
             versions=dict(unsloth="2026.9.12", unsloth_zoo="2026.9.6", torch="2.8.0",
                           transformers="5.5.0", fla="0.3.2", causal_conv1d="1.5.2"))
    return r


def judge(apr):
    return V.decide([apr] * 3, [incumbent_twin(apr)] * 3)


def c_full_passes(m):
    r, gaps = m.to_verdict(full_trr(), "Qwen3.5-4B")
    return not gaps and judge(r)[:2] == (0, "PASS") and r["data_sha256"] == "d" * 64


def c_trr_only_gaps(m):
    r, gaps = m.to_verdict(dict(TRR_BASE), "Qwen3.5-4B")
    return r is None and gaps == TRR_EXTENSION


def c_failed_status(m):
    t = full_trr()
    t["status"] = "failed"
    r, gaps = m.to_verdict(t, "M")
    return r is None and gaps and "status" in gaps[0]


def c_two_data_files(m):
    t = full_trr()
    t["data_sha256"] = ["d" * 64, "e" * 64]
    r, gaps = m.to_verdict(t, "M")
    return r is None and any("data_sha256" in g for g in gaps)


def c_null_is_gap(m):
    t = full_trr()
    t["timed_window"]["seconds"] = None
    r, gaps = m.to_verdict(t, "M")
    return r is None and gaps == ["timed_window.seconds"]


def c_no_translation(m):
    r, _ = m.to_verdict(full_trr(optimizer="adamw"), "Qwen3.5-4B")
    code, _, reason, _ = judge(r) if r else (0, "", "", None)
    return r is not None and r["optimizer"] == "adamw" and code == 1 and "optimizer" in reason


def c_argv(m):
    a = m.apr_argv("apr", m.parse(ARGV + ["--receipt", "R"]), "B", "D", "T")
    h = m.apr_argv("apr", m.parse(ARGV + ["--receipt", "R", "--planted-half-targets"]), "B", "D", "T")
    val = lambda argv, f: argv[argv.index(f) + 1]  # noqa: E731
    return (a[:3] == ["apr", "finetune", "B"] and val(a, "--targets").count(",") == 6
            and val(h, "--targets") == "q_proj,k_proj,v_proj,o_proj" and val(a, "--rank") == "16"
            and val(a, "--alpha") == "32" and val(a, "--receipt") == "T" and val(a, "--data") == "D"
            and val(a, "--optimizer") == "adamw_fp32" and "--no-grad-checkpointing" in a)


CASES = [c_full_passes, c_trr_only_gaps, c_failed_status, c_two_data_files, c_null_is_gap,
         c_no_translation, c_argv]


def e2e(script, tmp, mode, extra=(), data=True, base=True):
    """Run the adapter as a process against the stub apr. Returns (rc, receipt or None)."""
    stub = os.path.join(tmp, "apr")
    with open(stub, "w") as f:
        f.write(STUB_APR)
    os.chmod(stub, 0o755)
    dpath = os.path.join(tmp, "data.jsonl")
    with open(dpath, "w") as f:
        f.write('{"text": "x"}\n')
    bpath = os.path.join(tmp, "base.apr")
    with open(bpath, "w") as f:
        f.write("base")
    receipt = os.path.join(tmp, "apr-1.json")
    if os.path.exists(receipt):
        os.unlink(receipt)
    env = dict(os.environ, APR_BIN=stub, STUB_APR_MODE=mode, APR_FT_DATA=dpath if data else "",
               APR_FT_BASE=bpath if base else os.path.join(tmp, "absent.apr"))
    p = subprocess.run([sys.executable, script] + ARGV + ["--receipt", receipt] + list(extra),
                       env=env, capture_output=True, text=True)
    if not os.path.exists(receipt):
        return p.returncode, None
    with open(receipt) as f:
        return p.returncode, json.load(f)


def e2e_cases(script, tmp):
    """(name, ok) for each end-to-end case."""
    rc, r = e2e(script, tmp, "full")
    full_ok = rc == 0 and r is not None and judge(r)[:2] == (0, "PASS")
    rc_h, r_h = e2e(script, tmp, "full", ["--planted-half-targets"])
    half = (V.decide([r_h] * 3, [incumbent_twin(r)] * 3) if r_h and r
            else (0, "", "", None))
    rc_t, r_t = e2e(script, tmp, "trr")
    rc_f, r_f = e2e(script, tmp, "fail")
    rc_d, r_d = e2e(script, tmp, "full", data=False)
    rc_b, r_b = e2e(script, tmp, "full", base=False)
    return [
        ("e2e full -> PASS", full_ok),
        ("e2e planted half targets -> SAME-WORK FAIL", rc_h == 0 and half[0] == 1
         and "targets differ" in half[2]),
        ("e2e TRR-only -> exit 4, no receipt", rc_t == 4 and r_t is None),
        ("e2e apr fails -> its rc, no receipt", rc_f == 5 and r_f is None),
        ("e2e no data -> exit 3, no receipt", rc_d == 3 and r_d is None),
        ("e2e no base -> exit 3, no receipt", rc_b == 3 and r_b is None),
    ]


def failures(src, script, tmp):
    m = load(src)
    with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
        bad = [c.__name__ for c in CASES if not c(m)]
    return bad + [name for name, ok in e2e_cases(script, tmp) if not ok]


MUTANTS = [
    ("A1 optimizer translated", "            out[field] = value",
     '            out[field] = "adamw_fp32" if field == "optimizer" else value'),
    ("A2 status ignored", 'if trr.get("status") != "success":', "if False:"),
    ("A3 gaps ignored", "return (None, gaps) if gaps else (out, [])", "return (out, [])"),
    ("A4 several data files accepted", "len(shas) == 1", "len(shas) >= 1"),
    ("A5 planted flag ignored", "HALF_TARGETS if args.planted_half_targets else TARGETS", "TARGETS"),
    ("A6 null counted as present", "return cur is not None, cur", "return True, cur"),
    ("A7 receipt written despite gaps", "    if gaps:\n        print(\"GAP: apr receipt lacks",
     "    if False:\n        print(\"GAP: apr receipt lacks"),
    ("A8 apr exit code ignored", "if rc != 0:", "if False:"),
    ("A9 base check dropped", "if data is None or base is None or", "if data is None or"),
    ("A10 model name as FILE", '[apr, "finetune", base,', '[apr, "finetune", args.model,'),
]


def survivors(src, tmp):
    left = []
    for name, old, new in MUTANTS:
        if src.count(old) != 1:
            left.append(name + " (edit does not apply)")
            continue
        path = os.path.join(tmp, "mutant.py")
        mutated = src.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(mutated)
        if not failures(mutated, path, tmp):
            left.append(name)
    return left


def side_wrapper_runs(tmp):
    """The .sh wrapper reaches the adapter (no data -> the adapter's exit 3)."""
    env = dict(os.environ, APR_FT_DATA="")
    p = subprocess.run([SIDE] + ARGV + ["--receipt", os.path.join(tmp, "w.json")], env=env,
                       capture_output=True, text=True)
    return p.returncode == 3 and "APR_FT_DATA" in p.stderr


def main():
    with open(ADAPTER, encoding="utf-8") as f:
        src = f.read()
    with tempfile.TemporaryDirectory() as tmp:
        failed = failures(src, ADAPTER, tmp)
        if not side_wrapper_runs(tmp):
            failed.append("side_wrapper_runs")
        left = survivors(src, tmp)
    print("R15 receipt requirements (keys a TRR-only receipt lacks): " + ", ".join(TRR_EXTENSION))
    for name in failed:
        print("FAIL case", name)
    for name in left:
        print("SURVIVED", name)
    total = len(CASES) + 6 + 1 + len(MUTANTS)
    print("%d checks, %d failed" % (total, len(failed) + len(left)))
    return 1 if failed or left else 0


if __name__ == "__main__":
    sys.exit(main())
