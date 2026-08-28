#!/usr/bin/env bash
# check_parity_receipt.sh — the parity validator must DISCRIMINATE before any
# release reads its verdict.
#
# The rule this repo keeps relearning: a validator that has only ever seen
# valid input is indistinguishable from `exit 0`. Every case below is a receipt
# shape that ALREADY HAPPENED here or is one edit away from happening, and the
# annotation on each fixture says which.
#
# The one that matters most is 04/05/06/07 — #2696. The published apr takes the
# CPU path even when handed --gpu, so measuring it against a CUDA llama.cpp
# yields 0.099x. That number is not a kernel defect and must not be reportable
# as one: 04 is the honest way to write it, and 05/06/07 are the three ways to
# write it dishonestly.
set -euo pipefail

VALIDATOR="scripts/lib/bench_receipt.py"
CASES="scripts/lib/parity_receipt_cases"
MIN_CASES=23

rc=0
printf -- '--- parity receipt validator --------------------------------------\n'

[ -f "$VALIDATOR" ] || { printf 'FAIL  %s is missing\n' "$VALIDATOR"; exit 2; }
[ -d "$CASES" ]     || { printf 'FAIL  %s is missing\n' "$CASES"; exit 2; }

n=0
for f in "$CASES"/*.json; do
    [ -e "$f" ] || break
    n=$((n + 1))
    expect=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['_expect']['result'])" "$f")
    why=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['_expect']['why'])" "$f")
    if python3 "$VALIDATOR" --parity "$f" >/dev/null 2>&1; then got=valid; else got=invalid; fi
    if [ "$expect" = "$got" ]; then
        printf 'ok    %-48s %-8s %s\n' "$(basename "$f" .json)" "$got" "$why"
    else
        printf 'FAIL  %-48s expected %s, got %s\n' "$(basename "$f" .json)" "$expect" "$got"
        rc=1
    fi
done

# VACUITY. A table that shrinks sweeps clean, and the count is asserted rather
# than trusted — the same reason check_bench_receipt.sh carries a floor.
if [ "$n" -lt "$MIN_CASES" ]; then
    printf 'FAIL  the case table has %s case(s); at least %s are required.\n' "$n" "$MIN_CASES"
    printf '      A shrinking table is a validator with less to discriminate.\n'
    rc=1
fi

# BOTH DIRECTIONS. A table of only-invalid cases is passed by a validator that
# rejects everything, which is as useless as one that accepts everything.
valid_n=$(grep -l '"result": "valid"' "$CASES"/*.json 2>/dev/null | wc -l | tr -d ' ')
invalid_n=$((n - valid_n))
if [ "$valid_n" -lt 5 ] || [ "$invalid_n" -lt 13 ]; then
    printf 'FAIL  the table is one-sided: %s valid / %s invalid. A validator that\n' "$valid_n" "$invalid_n"
    printf '      rejects everything passes an all-invalid table.\n'
    rc=1
fi

# ── THE MODULE PATH MUST PRINT WHAT THE CLI PATH PRINTS (#2736 residual) ──
#
# `validate_parity` is PURE: it resets the REPORT buffer, accumulates, and
# returns errors. Each CALLER flushes with the label it owns -- _validate_one,
# _mode_bench and _mode_parity all pass the receipt path.
#
# parity_block.py, the ONLY producer of a parity block in this tree and the
# THIRD caller of that pure validator, imported the module and never flushed.
# Every REPORT it computed -- the comparator-shortfall line #2736 exists to
# raise, and #2736's own requested/completed deferral notice -- was dropped on
# the floor. "A deferral nobody can see is indistinguishable from no deferral",
# reproduced one level up by the commit that wrote the sentence.
#
# EVERY case above drives the CLI, which flushes. They therefore prove nothing
# about the path that actually writes receipts, which is why this residual
# survived a 29-case table. These drive the MODULE path instead.
printf -- '\n--- the module path (what parity_block.py runs) ---------------------\n'

python3 - <<'PYMOD' || rc=1
import contextlib, io, json, os, subprocess, sys, tempfile
sys.path.insert(0, "scripts/lib")
import bench_receipt

bad = 0

def load(name):
    b = json.load(open("scripts/lib/parity_receipt_cases/%s.json" % name))
    b = b.get("parity", b)
    return {k: v for k, v in b.items() if k != "_expect"}

def with_join_key(block):
    """Fill the join key so the ONLY difference between the two cases is the
    completion shortfall. Without this both emit a join-key REPORT and the
    silent control cannot be silent -- the pair would not be minimal."""
    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("provenance"), dict):
                o["provenance"].update(host="h1", accelerator="cpu",
                                       model="m.gguf", quantization="Q4_K")
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(block)
    return block

def module_path(name):
    """Exactly the two calls parity_block.py makes, in order -- and the STDERR
    they produce. Asserting the buffer instead would pass against a
    `flush_reports` that writes nothing, which is the defect itself."""
    block = with_join_key(load(name))
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        errors = bench_receipt.validate_parity(block)
        bench_receipt.flush_reports(name)
    return errors, buf.getvalue()

# A MINIMAL PAIR. Both are VALID receipts; they differ only in whether the
# comparator lost requests. One must speak, the other must stay silent -- a
# channel that reports on everything is as useless as one that reports nothing.
errs, err_text = module_path("27_the_comparator_loses_requests_is_reported")
hits = err_text.count("comparator completed 7 of 10 requested")
if errs:
    print("FAIL  27 must stay VALID; got %d error(s)" % len(errs)); bad += 1
elif hits != 2:
    print("FAIL  27: the comparator-shortfall REPORT did not reach STDERR on")
    print("      the module path (bands 8 and 16 expected, got %d)." % hits)
    bad += 1
else:
    print("ok    27 comparator shortfall  on STDERR via the module path (2 bands)")

errs, err_text = module_path("26_every_request_completes")
if errs:
    print("FAIL  26 must stay VALID; got %d error(s)" % len(errs)); bad += 1
elif err_text.strip():
    print("FAIL  26: an honest run must be SILENT; stderr carried:")
    print("      %s" % err_text.strip().splitlines()[0][:90]); bad += 1
else:
    print("ok    26 every request completes  SILENT, as an honest run must be")

# THE PRODUCER ITSELF. The two checks above prove the plumbing carries a
# REPORT; this proves parity_block.py actually PULLS it. Remove
# `flush_reports(args.out)` from that file and this is the row that reds --
# the pair above stays green, because it calls the module directly.
work = tempfile.mkdtemp()
runs = {"runs": [{"decode_tok_per_sec": 100.0, "prefill_tok_per_sec": 200.0,
                  "ttft_p50_ms": 10.0, "tokens_per_sec": 100.0}] * 3}
for side in ("apr", "llama"):
    for suffix in ["cpu"] + ["cpu-c%d" % c for c in (1, 4, 8, 16)]:
        json.dump(runs, open(os.path.join(work, "%s-%s.json" % (side, suffix)), "w"))
open(os.path.join(work, "lanes.txt"), "w").write("cpu cpu cpu\n")
sha = "a" * 64
proc = subprocess.run(
    [sys.executable, "scripts/lib/parity_block.py", "--work", work,
     "--apr", "/bin/true", "--apr-sha", sha, "--llama", "/bin/true",
     "--llama-sha", sha, "--llama-build", "deadbeef", "--model", "/bin/true",
     "--install-source", "local-build", "--out", os.path.join(work, "o.json")],
    capture_output=True, text=True)
if proc.returncode != 0:
    print("FAIL  producer exited %d building a healthy block" % proc.returncode)
    bad += 1
elif "REPORT " not in proc.stderr:
    print("FAIL  parity_block.py emitted NO REPORT. It computes them and drops")
    print("      them: the defect #2736's own channel had. Expected the")
    print("      requested/completed deferral notice on stderr.")
    bad += 1
else:
    print("ok    parity_block.py FLUSHES its REPORTs to stderr (producer wired)")

sys.exit(1 if bad else 0)
PYMOD

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  %s cases (%s valid / %s invalid): the validator separates a\n' "$n" "$valid_n" "$invalid_n"
    printf '      real parity receipt from every fabricated shape in the table.\n'
else
    printf 'FAIL  see rows above (#2696).\n'
fi
exit "$rc"
