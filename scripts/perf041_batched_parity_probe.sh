#!/usr/bin/env bash
# PP-26 batch-invariance witness (FALSIFY-CB-006 / #2753), executable.
#
# Starts a CUDA server, takes an m=1 reference, fires the declared ladder and
# checks that every slot inside an m=c batch produces the SAME TOKEN SEQUENCE
# as the m=1 reference, to the divergence point declared in
# scripts/perf-matrix.yaml (`witness.min_agree_tokens`).
#
#   exit 0  PASS         every band measured and agreeing
#   exit 1  DEFECT       a batched slot diverged before the declared point
#   exit 2  UNMEASURABLE the run could not decide (env / model / no batch)
#
# The 2 matters: a guard that names a code cause for a box it could not
# evaluate has fired three times in this repo in one day. Both 1 and 2 are RED
# to the caller — PP-26 makes an absent witness INVALID-CORRECTNESS — but they
# are DIFFERENT reds and the marker says which.
#
# CUDA_BATCH_WINDOW_MS is EXPORTED, not inherited. The scheduler's default
# window is 0 ms (cuda_batch_scheduler.rs), so batches form only by queue
# contention and a quiet box witnesses nothing. It is recorded in witness.json
# `env` so the reader knows which window produced the result.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

. scripts/apr_bin.sh || exit 1
BIN="$APR"

MODEL="${APR_MODELS:-"$HOME/models"}/${PERF041_MODEL:-qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
PORT="${PERF041_PORT:-8473}"
# A FIXED world-writable default path is both a symlink surface and a collision
# between two runs on one box — the class check_no_competing_harnesses.sh's
# header names for the probe this file replaces. CI always sets PERF041_OUT.
OUT="${PERF041_OUT:-"$(mktemp -d -t perf041-parity.XXXXXX)"}"
LOG="$OUT/server.log"
WITNESS="$OUT/witness.json"
MARKER="$OUT/marker.json"

export CUDA_BATCH_WINDOW_MS="${CUDA_BATCH_WINDOW_MS:-200}"

# This is the MEASUREMENT's clock, not a build stamp. PP-30 requires
# `started_utc` on every receipt and check_perf041_marker.sh gates the marker's
# age against the matrix's witness.max_age_days; a SOURCE_DATE_EPOCH-derived
# value would make every witness look as old as the commit and the freshness
# gate would be vacuous.
# (bashrs reports DET002 here; its disable-line directive is not honoured
# by bashrs 7.0.1 for this rule — probed, not assumed — so no inert
# suppression is left behind. The line sits inside the shell-lint ratchet.)
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOSTNAME_SHORT="${PERF041_HOST:-$(hostname -s 2>/dev/null || hostname)}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

# sm_121 vs sm_89 decides which defects are even in scope (§9 #1a is Blackwell
# only), so the marker carries the compute capability rather than the host name
# alone. `12.1` -> `121`, matching GpuProfile.cc. Empty when nvidia-smi is not
# there, which the marker records as null rather than as a guess.
PERF041_CC="${PERF041_CC:-$(nvidia-smi --query-gpu=compute_cap \
    --format=csv,noheader 2>/dev/null | head -1 | tr -cd '0-9')}"
export PERF041_CC

mkdir -p "$OUT"

# The marker is written on EVERY exit path, including the ones that abort
# before the probe runs. A missing marker must mean "the lane did not run at
# all", never "the lane ran and something went wrong before it could speak" —
# check_perf041_marker.sh treats absence as RED and would otherwise be unable
# to tell those apart.
write_marker() {
    # $1 exit code, $2 status, $3 reason (may be empty), $4 max_m (may be empty)
    python3 - "$MARKER" "$1" "$2" "$3" "$4" "$HOSTNAME_SHORT" "$COMMIT" \
        "$STARTED_UTC" "$BIN" "$WITNESS" <<'PY'
import hashlib
import json
import os
import sys

(marker, code, status, reason, max_m, host, commit, started, binary,
 witness_path) = sys.argv[1:11]


def sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


max_from_witness = None
try:
    with open(witness_path, encoding="utf-8") as handle:
        bands = json.load(handle).get("bands") or []
    formed = [b.get("m_formed") for b in bands
              if isinstance(b.get("m_formed"), int)]
    max_from_witness = max(formed) if formed else 0
except (OSError, ValueError):
    max_from_witness = None

record = {
    "host": host,
    "cc": os.environ.get("PERF041_CC") or None,
    "commit": commit,
    "sha256": sha256(binary),
    "exit": int(code),
    "max_m": max_from_witness if max_m == "" else int(max_m),
    "started_utc": started,
    "status": status,
}
if reason:
    record["reason"] = reason
with open(marker, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

fail_unmeasured() {
    printf 'perf041-parity: %s\n' "$1" >&2
    write_marker 2 UNMEASURED "$1" ""
    exit 2
}

if [ ! -f "$MODEL" ]; then
    fail_unmeasured "model not found: $MODEL"
fi

# `--gpu-layers all`, never the boolean `--gpu` (PP-15 / §9 #9). A quantity is
# what the receipt can report as `gpu_layers_resolved`; a boolean is not.
"$BIN" serve run "$MODEL" --gpu-layers all --port "$PORT" --context-length 4096 \
    > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

i=0
while [ "$i" -lt 300 ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' \
        "http://127.0.0.1:${PORT}/health" 2>/dev/null)
    if [ "$code" = "200" ]; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail_unmeasured "the server exited before becoming healthy (see $LOG)"
    fi
    i=$((i + 1))
    sleep 1
done
if [ "$i" -ge 300 ]; then
    fail_unmeasured "server never became healthy after 300s"
fi

printf 'perf041-parity: %s\n' "$("$BIN" --version 2>&1)"
printf 'perf041-parity: model %s\n' "$MODEL"
printf 'perf041-parity: CUDA_BATCH_WINDOW_MS=%s\n' "$CUDA_BATCH_WINDOW_MS"

# Status read directly, never through a pipe.
python3 scripts/perf041_batched_parity_probe.py \
    --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --server-log "$LOG" \
    --json "$WITNESS" \
    --host "$HOSTNAME_SHORT" \
    --commit "$COMMIT" \
    --binary "$BIN" \
    --model "$MODEL"
rc=$?

case "$rc" in
    0) write_marker 0 PASS "" "" ;;
    1) write_marker 1 DEFECT "a batched slot diverged from the m=1 reference before the declared point" "" ;;
    *) write_marker "$rc" UNMEASURED "the probe could not decide (see $WITNESS)" "" ;;
esac

printf 'perf041-parity: exit %d (0=PASS 1=code defect 2=unmeasurable)\n' "$rc"
printf 'perf041-parity: witness %s\n' "$WITNESS"
printf 'perf041-parity: marker  %s\n' "$MARKER"
exit "$rc"
