#!/usr/bin/env python3
"""Self-test for scripts/bench/unsloth_ft_verdict.py (0.72 T2, row R5).

Part 1 is a case table: each receipt set and the verdict it must get. The
planted falsifiers of beat-unsloth-finetune-throughput-v1 1.1.0 are cases
here: SAMEWORK (half targets), INCUMBENT-FASTPATH (no fla) and SAMEOPT
(adamw_8bit, "unsloth" checkpointing).

Part 2 applies each named mutant to a temp copy of the verdict script and
requires at least one case to go wrong (mutant KILLED). A mutant whose patch
does not change the text is a failure of this test (PATCH_FAILED), never a
survivor and never a pass.

Prints "N checks, M failed" and exits 1 if M > 0. Runs on any CPU box.
"""
import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench" / "unsloth_ft_verdict.py"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SHA_KEY = "apr_" + "g" + "it_sha"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FULL = 200 * 4 * 1 * 511  # timed_steps * batch * grad_accum * (seq_len - 1)


def run(side, rate, seconds=None):
    """One receipt doing the full window at `rate` tokens per (arbitrary) unit."""
    tokens = FULL
    r = dict(
        side=side, gpu_name="NVIDIA GeForce RTX 4090", gpu_uuid="GPU-4090-a",
        device_trace_line="[TRACE] device=cuda:0 RTX 4090", model="Qwen/Qwen3.5-4B",
        data_sha256="d" * 64, rank=16, alpha=32, targets=list(TARGETS),
        trainable_params=23_592_960, optimizer="adamw_fp32",
        grad_checkpointing=False, packing=False, seq_len=512, batch=4,
        grad_accum=1, warmup_steps=50, timed_steps=200, timed_after_compile=True,
        label_tokens_timed=tokens, timed_seconds=FULL / rate if seconds is None else seconds,
    )
    if side == "apr":
        r["apr_version"] = "0.72.0-dev"
        r[SHA_KEY] = "0123456789"
    else:
        r["versions"] = dict(unsloth="2026.9.12", unsloth_zoo="2026.9.9", torch="2.9.1",
                             transformers="5.5.0", fla="0.4.1", causal_conv1d="1.5.2")
    return r


def sides(apr_tok=(900, 900, 900), inc_tok=(1000, 1000, 1000)):
    return [run("apr", t) for t in apr_tok], [run("incumbent", t) for t in inc_tok]


def edit(which, key, value, runs=None):
    """Case builder: set key on every run (or the given run indexes) of one side."""
    def build():
        apr, inc = sides()
        target = apr if which == "apr" else inc
        for i, r in enumerate(target):
            if runs is None or i in runs:
                if value is DELETE:
                    del r[key]
                else:
                    r[key] = copy.deepcopy(value)
        return apr, inc
    return build


DELETE = object()


def no_fla():
    apr, inc = sides()
    for r in inc:
        r["versions"]["fla"] = None
    return apr, inc


def half_targets():
    apr, inc = sides()
    for r in apr:
        r["targets"] = TARGETS[:4]
        r["trainable_params"] = r["trainable_params"] // 2
    return apr, inc


def params_off(frac):
    def build():
        apr, inc = sides()
        for r in apr:
            r["trainable_params"] = int(r["trainable_params"] * (1 + frac))
        return apr, inc
    return build


# (name, builder, exit code, verdict, substring in reason, ratio or None)
CASES = [
    ("pass at 0.9", lambda: sides(), 0, "PASS", "ratio 0.9000", 0.9),
    ("fail at 0.5", lambda: sides((500, 500, 500)), 1, "FAIL", "ratio 0.5000", 0.5),
    ("0.8 exactly passes", lambda: sides((800, 800, 800)), 0, "PASS", "", 0.8),
    ("median, not mean", lambda: sides((1000, 1000, 10000)), 0, "PASS", "", 1.0),
    ("2 apr runs", lambda: sides((900, 900)), 2, "NOT_MEASURED", "need 3", None),
    ("2 incumbent runs", lambda: sides(inc_tok=(1000, 1000)), 2, "NOT_MEASURED", "need 3", None),
    ("planted no fla", no_fla, 2, "INCUMBENT_SLOW_PATH", "fla absent", None),
    ("planted adamw_8bit", edit("incumbent", "optimizer", "adamw_8bit"), 1, "FAIL",
     "SAME-WORK FAIL: incumbent run 1: optimizer", None),
    ("planted unsloth checkpointing", edit("incumbent", "grad_checkpointing", "unsloth"), 1,
     "FAIL", "grad_checkpointing", None),
    ("packing on", edit("apr", "packing", True), 1, "FAIL", "packing", None),
    ("timed before compile", edit("incumbent", "timed_after_compile", False), 1, "FAIL",
     "timed_after_compile", None),
    ("planted half targets", half_targets, 1, "FAIL", "SAME-WORK FAIL: targets differ", None),
    ("params 2% apart", params_off(0.02), 1, "FAIL", "trainable_params", None),
    ("params 0.5% apart", params_off(0.005), 0, "PASS", "", 0.9),
    ("data sha differs", edit("incumbent", "data_sha256", "e" * 64), 1, "FAIL",
     "data_sha256 differs", None),
    ("one run on another GPU", edit("apr", "gpu_uuid", "GPU-4090-b", runs=(2,)), 2,
     "NOT_MEASURED", "span 2 GPUs", None),
    ("no binary sha", edit("apr", SHA_KEY, DELETE), 2, "NOT_MEASURED", SHA_KEY, None),
    ("no device trace line", edit("apr", "device_trace_line", DELETE), 2, "NOT_MEASURED",
     "device_trace_line", None),
    ("empty timed window", edit("incumbent", "timed_seconds", 0.0, runs=(1,)), 2,
     "NOT_MEASURED", "empty timed window", None),
    ("short rows padded", edit("incumbent", "label_tokens_timed", FULL - 800), 1, "FAIL",
     "label_tokens_timed", None),
    ("apr counts differently", edit("apr", "label_tokens_timed", 200 * 4 * 512, runs=(1,)), 1,
     "FAIL", "full window", None),
    ("side mislabelled", edit("apr", "side", "incumbent", runs=(0,)), 2, "NOT_MEASURED",
     "side=", None),
]

# (name, text to replace, replacement). Each must be KILLED by the case table.
MUTANTS = [
    ("threshold halved", "if ratio >= threshold:", "if ratio >= threshold * 0.5:"),
    ("median -> mean", "ratio = statistics.median(tok_s(r) for r in apr) / statistics.median(",
     "ratio = statistics.mean(tok_s(r) for r in apr) / statistics.mean("),
    ("fla check dropped", 'if not v.get("fla"):', "if False:"),
    ("pins dropped", "if r[k] != want:", "if False:"),
    ("min runs 1", "MIN_RUNS = 3", "MIN_RUNS = 1"),
    ("param tolerance 5%", "PARAM_TOLERANCE = 0.01", "PARAM_TOLERANCE = 0.05"),
    ("gpu check dropped", "if len(gpus) != 1:", "if False:"),
    ("full-window check dropped", 'if r["label_tokens_timed"] != full_window(r):', "if False:"),
    ("targets check dropped", 'if sorted(r["targets"]) != want:', "if False:"),
]


def case_errors(mod):
    errors = []
    for name, build, code, verdict, needle, ratio in CASES:
        apr, inc = build()
        got_code, got_verdict, reason, got_ratio = mod.decide(apr, inc)
        ok = (got_code, got_verdict) == (code, verdict) and needle in reason
        if ratio is None:
            ok = ok and got_ratio is None
        else:
            ok = ok and got_ratio is not None and abs(got_ratio - ratio) < 1e-9
        if not ok:
            errors.append("%s: got %s %s %r ratio=%r" % (name, got_code, got_verdict, reason, got_ratio))
    return errors


def write_receipts(d):
    apr, inc = sides()
    paths = dict(apr=[], incumbent=[])
    for side, runs in (("apr", apr), ("incumbent", inc)):
        for i, r in enumerate(runs):
            p = pathlib.Path(d) / ("%s-%d.json" % (side, i))
            p.write_text(json.dumps(r))
            paths[side].append(str(p))
    return ["--apr"] + paths["apr"] + ["--incumbent"] + paths["incumbent"]


def cli_check(label, args, want_code, want_verdict):
    p = subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        out = dict()
    errors = []
    if p.returncode != want_code or out.get("verdict") != want_verdict:
        errors.append("%s: rc=%d stdout=%r" % (label, p.returncode, p.stdout))
    if want_verdict == "NOT_MEASURED" and "ratio" in out:
        errors.append("%s: NOT_MEASURED carries a ratio" % label)
    return errors


def cli_errors():
    """The CLI prints one JSON line and exits with decide()'s code."""
    with tempfile.TemporaryDirectory() as d:
        both = write_receipts(d)
        table = (
            ("cli pass", both, 0, "PASS"),
            ("cli threshold 0.95", both + ["--threshold", "0.95"], 1, "FAIL"),
            ("cli unreadable", ["--apr", str(pathlib.Path(d) / "absent.json")], 2, "NOT_MEASURED"),
            ("cli no receipts", [], 2, "NOT_MEASURED"),
        )
        return [e for row in table for e in cli_check(*row)]


def mutant_error(d, i, text, name, old, new):
    """None if the mutant is killed, else what went wrong."""
    if text.count(old) != 1:
        return "mutant %s: PATCH_FAILED (pattern found %d times)" % (name, text.count(old))
    path = pathlib.Path(d) / ("mutant_%d.py" % i)
    path.write_text(text.replace(old, new))
    killed_by = case_errors(load_module(path, "verdict_mutant_%d" % i))
    if not killed_by:
        return "mutant %s: SURVIVED" % name
    print("ok    mutant %s: KILLED (%d cases)" % (name, len(killed_by)))
    return None


def main():
    real = load_module(SCRIPT, "verdict_real")
    errors = case_errors(real) + cli_errors()
    text = SCRIPT.read_text()
    with tempfile.TemporaryDirectory() as d:
        for i, (name, old, new) in enumerate(MUTANTS):
            err = mutant_error(d, i, text, name, old, new)
            if err:
                errors.append(err)
    for e in errors:
        print("FAIL  " + e)
    checks = len(CASES) + 4 + len(MUTANTS)
    print("%d checks, %d failed" % (checks, len(errors)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
