#!/usr/bin/env bash
# PP-LLAMA-001 §12 row 2 — the §9 #1 prefill-path probe, executable.
#
# Sweeps prompt length on ONE host and decomposes prefill wall time into a fixed
# cost and a per-token cost, on two arms: the default path, and BATCHED_PREFILL=1.
#
#   exit 0  MECHANISM_CONFIRMED  default arm linear; batched arm collapses
#   exit 1  PREDICTION_KILLED    flat in prompt length, or no collapse (§10)
#   exit 2  UNMEASURABLE         the run could not decide
#
# WHY THE SWEEP, AND NOT THE ONE POINT WE ALREADY HAVE
# ----------------------------------------------------
# §9 #1 sizes the defect from a single fixed point: 16.75 s at 513 prompt tokens
# = 32.65 ms per prompt token on gx10. One point has no slope. "linear at
# 32.65 ms/token" and "a 16.75 s fixed cost independent of prompt length" fit
# that point equally well and imply OPPOSITE defects — the first is the serial
# per-token loop at gpu_profile.rs:517-531, the second is somewhere else
# entirely and §9 #1 would be scoped wrong. §10 registers the linearity as a
# prediction with `kill if: flat in prompt length, or no collapse`, and this is
# the probe that can kill it.
#
# THE 2 MATTERS. Both 1 and 2 are RED to a caller, but they are different reds:
# 1 says the code is not what the spec claims, 2 says this run cannot speak.
# scripts/perf002_prefill_path_probe.py refuses to decompose four shapes (too
# few distinct prompt lengths, negative slope, r2 below the floor, bimodal) and
# every one of them lands here as 2, never as a defect claim — a guard that
# names a code cause for a box it could not evaluate has fired three times in
# this repo in one day.
#
# HOST. §9 #1 is Blackwell-only (`cc >= SM12X_MIN_CC`), so this is a gx10 probe
# (GB10, sm_121). It runs anywhere — the marker records `cc`, and a reader who
# finds sm_89 in a marker knows the run could not have exercised the path.
set -uo pipefail

# THE OUTPUT DIRECTORY AND THE MARKER COME FIRST, BEFORE ANYTHING THAT CAN FAIL.
#
# "A marker on every exit path" is a claim this file makes about itself, and the
# first draft did not honour it: `cd "$ROOT" || exit 2` and
# `. scripts/apr_bin.sh || exit 1` both bailed BEFORE write_marker() was even
# defined, so the two most likely early failures — a moved checkout and an
# unattributable binary — produced silence. A missing marker is supposed to mean
# "the lane did not run at all"; those two made it mean "the lane ran and could
# not speak", which is the distinction the marker exists to preserve.
#
# A FIXED world-writable default path is both a symlink surface and a collision
# between two runs on one box; CI always sets PERF002_OUT.
OUT="${PERF002_OUT:-"$(mktemp -d -t perf002-prefill.XXXXXX)"}"
mkdir -p "$OUT" || {
    printf 'perf002-prefill: cannot create %s; no marker is possible\n' "$OUT" >&2
    exit 2
}
LOG="$OUT/server.log"
SAMPLES="$OUT/samples.json"
REPORT="$OUT/decomposition.json"
MARKER="$OUT/marker.json"

# ONE CLOCK for the whole run, captured before anything that can fail so the
# bail path and the full writer stamp the same instant.
#
# This is the MEASUREMENT's clock, not a build stamp: a SOURCE_DATE_EPOCH-derived
# value would make every marker look as old as the commit and any freshness gate
# reading it would be vacuous. Captured to an append-only sink and read back,
# per bashrs's own guidance for exactly this case — and bashrs enforces it:
# a bare `date` inside bail_marker was DET002 in guard-cargo.
date -u +%Y-%m-%dT%H:%M:%SZ >> "$OUT/started_utc.log"
STARTED_UTC="$(tail -n 1 "$OUT/started_utc.log")"
HOSTNAME_SHORT="${PERF002_HOST:-$(hostname -s 2>/dev/null || hostname)}"

# The minimal marker, written without python and without the repo: it has to
# work when the failure IS the repo. json.dump-compatible, same field names, and
# `status` is one of the same vocabulary. The full writer below supersedes it on
# every path that gets that far.
bail_marker() { # $1 = reason
    printf '{\n  "cc": null,\n  "commit": null,\n  "exit": 2,\n  "host": "%s",\n  "reason": "%s",\n  "refused_rule": null,\n  "sha256": null,\n  "slope_ms_per_token": null,\n  "started_utc": "%s",\n  "status": "UNMEASURABLE"\n}\n' \
        "$HOSTNAME_SHORT" "$1" "$STARTED_UTC" > "$MARKER"
    printf 'perf002-prefill: %s\n' "$1" >&2
    exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" \
    || bail_marker "cannot resolve the repo root from \$BASH_SOURCE"
cd "$ROOT" || bail_marker "cannot cd to the repo root $ROOT"

. scripts/apr_bin.sh || bail_marker "apr_bin.sh could not attribute an apr binary to this tree"
BIN="$APR"

MODEL="${APR_MODELS:-"$HOME/models"}/${PERF002_MODEL:-qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
PORT="${PERF002_PORT:-8474}"

# The ladder. At least 4 distinct lengths so `bimodal` has two points per mode
# to have a spread at all, and the top rung sits at the 513 tokens §9 #1 was
# sized from so this sweep's fit can be read against that number directly.
PERF002_TOKENS="${PERF002_TOKENS:-64 128 256 384 513}"
PERF002_REPEATS="${PERF002_REPEATS:-3}"

COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

# sm_121 vs sm_89 decides whether §9 #1 is even in scope, so the marker carries
# the compute capability rather than the host name alone. `12.1` -> `121`,
# matching GpuProfile.cc. Empty when nvidia-smi is absent, which the marker
# records as null rather than as a guess.
PERF002_CC="${PERF002_CC:-$(nvidia-smi --query-gpu=compute_cap \
    --format=csv,noheader 2>/dev/null | head -1 | tr -cd '0-9')}"
export PERF002_CC

# Written on EVERY exit path, including the ones that abort before the probe
# runs. A missing marker must mean "the lane did not run at all", never "the
# lane ran and something went wrong before it could speak".
write_marker() {
    # $1 exit code, $2 status, $3 reason (may be empty)
    python3 - "$MARKER" "$1" "$2" "$3" "$HOSTNAME_SHORT" "$COMMIT" \
        "$STARTED_UTC" "$BIN" "$REPORT" <<'PY'
import hashlib
import json
import os
import sys

(marker, code, status, reason, host, commit, started, binary,
 report_path) = sys.argv[1:10]


def sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


slope = None
refused = None
try:
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    slope = report.get("verdict", {}).get("slope_ms_per_token")
    default_arm = report.get("arms", {}).get("default") or {}
    refused = default_arm.get("rule")
except (OSError, ValueError):
    pass

record = {
    "host": host,
    "cc": os.environ.get("PERF002_CC") or None,
    "commit": commit,
    "sha256": sha256(binary),
    "exit": int(code),
    "slope_ms_per_token": slope,
    "refused_rule": refused,
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
    printf 'perf002-prefill: %s\n' "$1" >&2
    write_marker 2 UNMEASURABLE "$1"
    exit 2
}

# The instrument before the measurement. decompose()'s refusals are the whole
# value of this probe, so a run that cannot prove them is not allowed to report
# a verdict — same rule the sibling applies to its own witness.
if ! python3 scripts/perf002_prefill_path_probe.py --selftest; then
    fail_unmeasured "decompose() failed its own selftest; refusing to report a verdict"
fi

[ -f "$MODEL" ] || fail_unmeasured "model not found: $MODEL"
command -v curl >/dev/null 2>&1 || fail_unmeasured "curl is not on PATH"

printf 'perf002-prefill: %s\n' "$("$BIN" --version 2>&1)"
printf 'perf002-prefill: model %s\n' "$MODEL"
printf 'perf002-prefill: cc %s\n' "${PERF002_CC:-unknown}"
printf 'perf002-prefill: ladder %s x%s\n' "$PERF002_TOKENS" "$PERF002_REPEATS"

# One arm: start a server with the given environment, sweep the ladder, print
# TSV rows (tokens, prefill_seconds) on stdout. The server is restarted per arm
# because BATCHED_PREFILL is read at startup.
run_arm() { # $1 = arm name, $2 = value for BATCHED_PREFILL ("" = unset)
    local arm="$1" batched="$2" pid
    if [ -n "$batched" ]; then
        BATCHED_PREFILL="$batched" "$BIN" serve run "$MODEL" --gpu-layers all \
            --port "$PORT" --context-length 4096 >> "$LOG" 2>&1 &
    else
        "$BIN" serve run "$MODEL" --gpu-layers all \
            --port "$PORT" --context-length 4096 >> "$LOG" 2>&1 &
    fi
    pid=$!
    local i=0
    while [ "$i" -lt 300 ]; do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' \
                "http://127.0.0.1:${PORT}/health" 2>/dev/null)" = "200" ]; then
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            printf 'perf002-prefill: %s arm: server exited before healthy\n' "$arm" >&2
            return 1
        fi
        i=$((i + 1))
        sleep 1
    done
    if [ "$i" -ge 300 ]; then
        kill "$pid" 2>/dev/null
        printf 'perf002-prefill: %s arm: server never became healthy\n' "$arm" >&2
        return 1
    fi

    # The sweep and the timing live in the python worker: TTFT must be read
    # off the SSE stream (first chunk carrying content), and curl's
    # `time_starttransfer` is when the response HEADERS begin — which a
    # streaming server sends before it has prefilled anything. A draft of this
    # probe timed headers and reported 0.6 ms samples as a flat prefill path.
    python3 scripts/perf002_prefill_path_probe.py \
        --measure "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --tokens "$PERF002_TOKENS" \
        --repeats "$PERF002_REPEATS"

    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 0
}

DEFAULT_TSV="$OUT/default.tsv"
BATCHED_TSV="$OUT/batched.tsv"
: > "$DEFAULT_TSV"
: > "$BATCHED_TSV"

run_arm default "" > "$DEFAULT_TSV" || fail_unmeasured "the default arm did not run"
run_arm batched 1 > "$BATCHED_TSV" || \
    printf 'perf002-prefill: the BATCHED_PREFILL=1 arm did not run; the collapse half will be unmeasured\n' >&2

python3 - "$DEFAULT_TSV" "$BATCHED_TSV" "$SAMPLES" <<'PY'
import json
import sys

def load(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 2:
                    rows.append({"prompt_tokens": int(parts[0]),
                                 "prefill_s": float(parts[1])})
    except OSError:
        pass
    return rows

default_rows, batched_rows, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"default": load(default_rows), "batched": load(batched_rows)},
              handle, indent=2)
    handle.write("\n")
PY

python3 scripts/perf002_prefill_path_probe.py --samples "$SAMPLES" --json "$REPORT"
rc=$?

case "$rc" in
    0) write_marker 0 MECHANISM_CONFIRMED "" ;;
    1) write_marker 1 PREDICTION_KILLED \
        "§10's registered prediction for §9 #1 did not hold (see $REPORT)" ;;
    *) write_marker "$rc" UNMEASURABLE "the probe could not decide (see $REPORT)" ;;
esac

printf 'perf002-prefill: exit %d (0=confirmed 1=prediction killed 2=unmeasurable)\n' "$rc"
printf 'perf002-prefill: samples %s\n' "$SAMPLES"
printf 'perf002-prefill: report  %s\n' "$REPORT"
printf 'perf002-prefill: marker  %s\n' "$MARKER"
exit "$rc"
