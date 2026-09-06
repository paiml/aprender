#!/usr/bin/env bash
# scripts/lib/parity_block_selftest.sh — fixture generator for
# check_parity_block_refusals.sh's --selftest (#2735, #2887, PMAT-972).
#
# SOURCED, never executed: this library is option-neutral (no `set -e` /
# `pipefail` here) so sourcing it can never mutate the caller's shell options.
# Every function fails by RETURN STATUS, per the sourced-library rule this
# repo already enforces for scripts/apr_bin.sh-style helpers.
#
# Builds the HISTORICAL layout scripts/lib/parity_block.py's docstring
# describes: one report per lane per band, `{apr,llama}-<class>-c<N>.json`,
# read when no `band-*.json` executor metadata file is present. Every band in
# scripts/perf-matrix.yaml's committed `ladder.declared` ([1, 4, 8, 16]) is
# written with a clean, PASSing measurement (subject 1.1x the comparator, both
# metrics), so the fixture is otherwise a complete, valid corpus and only the
# one corrupted band (c=8) can be the cause of a refusal.

# pbs_make_fixture <work_dir> <corrupt_side> <corrupt_mode>
#   corrupt_side: none | subject | comparator  -- which side of band c=8 is bad
#   corrupt_mode: zero  -- every run's rate is 0.0 (a non-empty list of zeros)
#                 empty -- the corrupted side's band c=8 file has NO runs
# Returns 0 on success, 1 if python3 failed to write the fixture.
pbs_make_fixture() {
    local work=$1 side=$2 mode=$3
    python3 - "$work" "$side" "$mode" <<'PY'
import json
import os
import sys

work, corrupt_side, corrupt_mode = sys.argv[1], sys.argv[2], sys.argv[3]
BANDS = (1, 4, 8, 16)


def make_run(tok, decode, prefill, concurrency):
    return {
        "tokens_per_sec": tok,
        "decode_tok_per_sec": decode,
        "prefill_tok_per_sec": prefill,
        "ttft_p50_ms": 50.0,
        "concurrency": concurrency,
        "completion_tokens_total": 500,
        "total_requests": 5,
        "successful": 5,
        "request_details": [{"latency_ms": 20.0}],
    }


def write_side(path, concurrency, tok, decode, prefill, corrupt):
    if corrupt == "empty":
        runs = []
    elif corrupt == "zero":
        runs = [make_run(0.0, 0.0, 0.0, concurrency) for _ in range(5)]
    else:
        runs = [make_run(tok, decode, prefill, concurrency) for _ in range(5)]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"runs": runs}, handle)


with open(os.path.join(work, "lanes.txt"), "w", encoding="utf-8") as handle:
    # THREE fields: `<name> <subject class> <comparator class>` -- the
    # historical lanes.txt spelling _lane_rows() reads for len(row) == 3.
    handle.write("cpu cpu cpu\n")

for c in BANDS:
    apr_corrupt = corrupt_mode if (corrupt_side == "subject" and c == 8) else "clean"
    llama_corrupt = corrupt_mode if (corrupt_side == "comparator" and c == 8) else "clean"
    write_side(os.path.join(work, "apr-cpu-c%d.json" % c), c, 110.0, 110.0, 200.0, apr_corrupt)
    write_side(os.path.join(work, "llama-cpu-c%d.json" % c), c, 100.0, 100.0, 180.0, llama_corrupt)
PY
}

# pbs_run <work_dir> <out_file>  -- invoke parity_block.py against the fixture
# with fixed, valid-looking CLI provenance. Prints nothing; caller redirects.
pbs_run() {
    local work=$1 out=$2
    python3 "$PBS_ROOT/scripts/lib/parity_block.py" \
        --work "$work" \
        --apr "$work/fixture-apr" --apr-sha "$(printf 'a%.0s' $(seq 1 64))" \
        --llama /usr/bin/llama-server --llama-sha "$(printf 'b%.0s' $(seq 1 64))" \
        --llama-build deadbeef00000000000000000000000000000000 \
        --model fixture-model.gguf --install-source local-build \
        --out "$out"
}
