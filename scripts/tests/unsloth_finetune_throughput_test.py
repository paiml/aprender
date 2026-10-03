#!/usr/bin/env python3
"""Self-test for scripts/bench/unsloth_finetune_throughput.sh (0.72 T2, row R5).

The orchestrator is run end to end on a CPU box: the two side runners and the
GPU queue are stubs written into a temp dir, so nothing touches a GPU. The real
verdict script decides every case, so each case checks the orchestrator's own
job: ABBA order, every run under the GPU queue, the planted flag on the right
side, refusals, a failed run skipped (never counted), and the verdict's exit
code passed through.

Part 1 is the case table. Part 2 applies each named mutant to a temp copy of
the orchestrator and requires at least one case to go wrong (mutant KILLED). A
mutant whose patch does not change the text is a failure (PATCH_FAILED).

Prints "N checks, M failed" and exits 1 if M > 0.
"""
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench" / "unsloth_finetune_throughput.sh"
VERDICT = ROOT / "scripts" / "bench" / "unsloth_ft_verdict.py"

# A side runner stub. It records its argv, honours the planted flags the way a
# real runner would, fails the run named in STUB_FAIL ("apr:2") before writing a
# receipt, and the run named in STUB_FAIL_AFTER after writing one.
STUB_SIDE = r'''#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
def val(k):
    return a[a.index(k) + 1]
side, run, receipt = val("--side"), val("--run"), val("--receipt")
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(dict(argv=a)) + "\n")
if os.environ.get("STUB_FAIL") == side + ":" + run:
    sys.exit(7)
targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
r = dict(side=side, gpu_name="NVIDIA GeForce RTX 4090", gpu_uuid="GPU-4090-a",
         device_trace_line="[TRACE] device=cuda:0", model=val("--model"),
         data_sha256="d" * 64, rank=16, alpha=32, targets=targets,
         trainable_params=23592960, optimizer="adamw_fp32", grad_checkpointing=False,
         packing=False, seq_len=512, batch=4, grad_accum=1, warmup_steps=50,
         timed_steps=200, timed_after_compile=True, timed_seconds=10.0,
         label_tokens_timed=int(os.environ.get("STUB_TOK_" + side.upper(), "1000")))
if side == "apr":
    r["apr_version"] = "0.72.0-dev"
    r["apr_" + "g" + "it_sha"] = "0123456789"
    if "--planted-half-targets" in a:
        r["targets"] = targets[:3]
        r["trainable_params"] = 11796480
else:
    r["versions"] = dict(unsloth="2026.9.12", unsloth_zoo="2026.9.9", torch="2.9.1",
                         transformers="5.5.0", fla="0.4.1", causal_conv1d="1.5.2")
    if "--planted-no-fla" in a:
        r["versions"]["fla"] = ""
    if "--planted-incumbent-adamw8bit" in a:
        r["optimizer"] = "adamw_8bit"
with open(receipt, "w") as f:
    json.dump(r, f)
if os.environ.get("STUB_FAIL_AFTER") == side + ":" + run:
    sys.exit(9)
'''

# A GPU queue stub: records that it wrapped the command, then runs it.
STUB_GPUQ = r'''#!/usr/bin/env bash
printf 'GPUQ %s\n' "$*" >> "$GPUQ_LOG"
while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done
shift
exec "$@"
'''


def write_exec(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class Env:
    def __init__(self, tmp):
        self.tmp = tmp
        self.side = tmp / "side.py"
        self.gpuq = tmp / "gpu-q"
        write_exec(self.side, STUB_SIDE)
        write_exec(self.gpuq, STUB_GPUQ)
        self.stub_log = tmp / "side.log"
        self.gpuq_log = tmp / "gpuq.log"
        self.active = tmp / "train-active"


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def read_verdict(path):
    text = path.read_text().strip() if path.exists() else ""
    return json.loads(text) if text else None


def reset(e, out, train_active):
    for p in (e.stub_log, e.gpuq_log, e.active):
        p.unlink(missing_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    if train_active:
        e.active.write_text("x")


def case_env(e, extra_env):
    env = dict(os.environ)
    env.update(APR_FT_SIDE_CMD=str(e.side), INCUMBENT_FT_SIDE_CMD=str(e.side),
               GPUQ=str(e.gpuq), GPUQ_LOG=str(e.gpuq_log), STUB_LOG=str(e.stub_log),
               APR_TRAIN_ACTIVE_MARKER=str(e.active), VERDICT=str(VERDICT), RUN_TIMEOUT="60")
    env.update(extra_env or {})
    return env


def run_case(script, env_dir, args, extra_env=None, train_active=False):
    e = Env(env_dir)
    out = env_dir / "out"
    reset(e, out, train_active)
    argv = ["bash", str(script), "--model", "Qwen/Qwen3.5-4B", "--gpu", "0", "--out", str(out)] + args
    p = subprocess.run(argv, env=case_env(e, extra_env), capture_output=True, text=True, timeout=120)
    calls = [json.loads(l)["argv"] for l in lines(e.stub_log)]
    return dict(rc=p.returncode, out=p.stdout, err=p.stderr, calls=calls,
                gpuq=lines(e.gpuq_log), verdict=read_verdict(out / "verdict.json"))


def order(calls):
    def val(a, k):
        return a[a.index(k) + 1]
    return [(val(a, "--run"), val(a, "--side")) for a in calls]


ABBA = [("1", "apr"), ("1", "incumbent"), ("2", "incumbent"), ("2", "apr"),
        ("3", "apr"), ("3", "incumbent")]


def planted_on(calls, flag):
    return sorted(set(a[a.index("--side") + 1] for a in calls if flag in a))


CASES = [
    ("pass: 6 runs, ABBA, PASS exit 0", [], dict(STUB_TOK_APR="900"), False,
     lambda r: r["rc"] == 0 and r["verdict"]["verdict"] == "PASS" and order(r["calls"]) == ABBA),
    ("every run goes through the GPU queue", [], None, False,
     lambda r: len(r["gpuq"]) == 6 and all("--prio 5 -- timeout 60" in g for g in r["gpuq"])),
    ("slow apr: ratio 0.5, FAIL exit 1", [], dict(STUB_TOK_APR="500"), False,
     lambda r: r["rc"] == 1 and r["verdict"]["verdict"] == "FAIL" and r["verdict"]["ratio"] == 0.5),
    ("planted half targets: apr side only, SAME-WORK FAIL", ["--planted-half-targets"], None, False,
     lambda r: r["rc"] == 1 and "SAME-WORK" in r["verdict"]["reason"]
     and planted_on(r["calls"], "--planted-half-targets") == ["apr"]),
    ("planted no fla: incumbent only, INCUMBENT_SLOW_PATH exit 2", ["--planted-no-fla"], None, False,
     lambda r: r["rc"] == 2 and r["verdict"]["verdict"] == "INCUMBENT_SLOW_PATH"
     and planted_on(r["calls"], "--planted-no-fla") == ["incumbent"]),
    ("planted adamw8bit: incumbent only, SAME-WORK FAIL", ["--planted-incumbent-adamw8bit"], None, False,
     lambda r: r["rc"] == 1 and "SAME-WORK" in r["verdict"]["reason"]
     and planted_on(r["calls"], "--planted-incumbent-adamw8bit") == ["incumbent"]),
    ("no planted flag reaches any side unasked", [], None, False,
     lambda r: not any(x.startswith("--planted") for a in r["calls"] for x in a)
     and all(len(a) == 10 for a in r["calls"])),
    ("a failed apr run is skipped: NOT_MEASURED exit 2", [], dict(STUB_FAIL="apr:2"), False,
     lambda r: r["rc"] == 2 and r["verdict"]["verdict"] == "NOT_MEASURED" and len(r["calls"]) == 6
     and "apr: 2 runs" in r["verdict"]["reason"]),
    ("receipt written then rc!=0: not counted, NOT_MEASURED", [], dict(STUB_FAIL_AFTER="incumbent:3"), False,
     lambda r: r["rc"] == 2 and "incumbent: 2 runs" in r["verdict"]["reason"]),
    ("train active: refused exit 3, no run", [], None, True,
     lambda r: r["rc"] == 3 and r["calls"] == [] and r["gpuq"] == [] and "train-active" in r["err"]),
    ("empty GPUQ: refused exit 3, no unlocked run", [], dict(GPUQ=""), False,
     lambda r: r["rc"] == 3 and r["calls"] == []),
    ("missing side runner: refused exit 3", [], dict(INCUMBENT_FT_SIDE_CMD="/nonexistent/side"), False,
     lambda r: r["rc"] == 3 and r["calls"] == []),
    ("dry-run: prints 6 queued commands, runs nothing", ["--dry-run"], None, False,
     lambda r: r["rc"] == 0 and r["calls"] == [] and r["gpuq"] == []
     and r["out"].count("dry-run: ") == 6 and r["out"].count(" -- timeout ") == 6),
    ("dry-run while train active says it would refuse", ["--dry-run"], None, True,
     lambda r: r["rc"] == 0 and "would refuse" in r["out"] and r["calls"] == []),
    ("two planted flags: usage exit 64", ["--planted-no-fla", "--planted-half-targets"], None, False,
     lambda r: r["rc"] == 64 and r["calls"] == []),
    ("--runs 0: usage exit 64", ["--runs", "0"], None, False,
     lambda r: r["rc"] == 64 and r["calls"] == []),
    ("--runs 1: 2 runs, NOT_MEASURED (min 3)", ["--runs", "1"], None, False,
     lambda r: r["rc"] == 2 and len(r["calls"]) == 2 and r["verdict"]["verdict"] == "NOT_MEASURED"),
]

# Each mutant: (name, old text, new text). Each must turn at least one case wrong.
MUTANTS = [
    ("M1 no GPU queue", '"$gpuq" --prio "$prio" -- timeout', 'timeout'),
    ("M2 AAAA order", "if [ $((k % 2)) -eq 1 ]; then", "if true; then"),
    ("M3 train-active ignored", 'if [ -e "$train_active" ]; then\n  echo "$0: REFUSED',
     'if false; then\n  echo "$0: REFUSED'),
    ("M4 half-targets to the wrong side", "apr:--planted-half-targets)", "incumbent:--planted-half-targets)"),
    ("M5 failed run counted", 'if [ "$rc" -ne 0 ] || [ ! -s "$receipt" ]; then',
     'if [ ! -s "$receipt" ]; then'),
    ("M6 verdict code dropped", 'exit "${PIPESTATUS[0]}"', 'exit "${PIPESTATUS[1]}"'),
    ("M7 empty GPUQ allowed", '[ -n "$gpuq" ] || {', 'true || {'),
    ("M8 planted flag to both sides", 'case "$1:$planted" in', 'case "apr:$planted" in'),
    ("M9 two planted flags allowed", 'if [ -n "$planted" ]; then echo', 'if false; then echo'),
    ("M10 dry-run executes", 'if [ "$dry" -eq 1 ]; then\n  if', 'if false; then\n  if'),
]


def run_table(script, tmp):
    results = []
    for name, args, extra, active, ok in CASES:
        r = run_case(script, tmp, args, extra, active)
        try:
            good = bool(ok(r))
        except (KeyError, TypeError, ValueError):
            good = False
        results.append((name, good, r))
    return results


def case_failures(tmp):
    """Part 1: one count per case; prints each wrong case."""
    failed = 0
    for name, good, r in run_table(SCRIPT, tmp):
        if not good:
            failed += 1
            print("FAIL case: %s rc=%s verdict=%s err=%s" % (name, r["rc"], r["verdict"], r["err"][-300:]))
    return failed


def mutant_fails(tmp, text, name, old, new):
    """Part 2: True when the mutant is PATCH_FAILED or SURVIVED."""
    patched = text.replace(old, new, 1)
    if old not in text or patched == text:
        print("PATCH_FAILED %s" % name)
        return True
    mdir = tmp / "mutant"
    mdir.mkdir(exist_ok=True)
    mpath = mdir / "unsloth_finetune_throughput.sh"
    mpath.write_text(patched)
    wrong = [n for n, good, _ in run_table(mpath, tmp) if not good]
    print(("KILLED %s by %s" % (name, wrong[0])) if wrong else ("SURVIVED %s" % name))
    return not wrong


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        failed = case_failures(tmp)
        text = SCRIPT.read_text()
        failed += sum(mutant_fails(tmp, text, *m) for m in MUTANTS)
    checks = len(CASES) + len(MUTANTS)
    print("%d checks, %d failed" % (checks, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
