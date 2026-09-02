#!/usr/bin/env bash
#
# measure_bandwidth.sh — commit a MEASURED device bandwidth so the roofline
# ceiling stops being a vendor number (PP-23, PP-LLAMA-001 §2.4, §12 row 14).
#
# WHY THIS EXISTS
# ---------------
# §2.4 states the ceiling as `measured_device_bytes_per_sec / model_bytes` and
# then admits the two ceilings it quotes (~215 tok/s on RTX 4090, ~58 tok/s on
# GB10) are `[X]` — computed from a spec sheet, not from an observation of the
# board the number will be applied to. PP-12 refuses an untagged vendor GB/s as
# a published figure, and §2.4 forbids publishing any percentage of a ceiling
# until a `[V]` bandwidth is committed. This script is that commit.
#
# WHAT IT MEASURES, AND WHY THAT DEFINITION
# -----------------------------------------
# `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` over 1 GiB buffers, n >= 5 timed
# replicates after a DURATION-based warmup (scripts/lib/bandwidth_d2d.cu). One such
# copy reads BYTES from DRAM and writes BYTES back, so the reported rate is
# `2 * BYTES / elapsed` — the DRAM TRAFFIC, which is the quantity a vendor peak
# names and the quantity a decode step is bounded by, since decode streams the
# whole model out of DRAM once per token. Reporting `BYTES / elapsed` instead
# would halve the ceiling and make every measured decode rate look twice as
# close to the hardware limit as it is.
#
# WHY THE WARMUP IS A DURATION. With two untimed copies the board was still
# idling in P8 at a 405 MHz memory clock when the first replicate was timed, and
# an n=15 run on an RTX 4090 came back TRIMODAL: five at ~673 GB/s, five at ~825,
# five at ~946. Two n=9 runs minutes apart reported 941.7 and 939.6; the n=15 run
# reported 825.0. A median over a mixture describes neither mode, which is the
# bimodal-median trap. The warmup now copies until the device has been busy for
# --warmup-ms, and the record carries the memory clock read either side of the
# timed window so a reader can SEE that it was settled rather than trust that it
# was. The spread is reported for the same reason: a wide max/min is the
# signature of a clock still ramping, not of a noisy device.
#
# WHAT IT REFUSES
# ---------------
#   · no nvcc, or no CUDA device                       -> rc 2, nothing written
#   · the probe fails or emits fewer than 5 replicates -> rc 3, nothing written
# An UNMEASURED bandwidth must stay unmeasured. Writing a file with a
# plausible-looking number from a partial run is how an `[X]` figure acquires a
# `[V]` tag without acquiring an observation.
#
# Usage:
#   scripts/measure_bandwidth.sh [--host <matrix-host>] [--replicates N]
#                               [--warmup-ms MS] [--copies-per-replicate K]
#                               [--model <gguf>] [--out-dir <dir>]
#   scripts/measure_bandwidth.sh --selftest
#
# `--model` only adds a REPORT line deriving `bytes_per_sec / model_bytes`; the
# derived ceiling is NOT written to the bandwidth file, because the ceiling is a
# property of a (bandwidth, model) pair and belongs in the receipt that names
# both (PP-23), not in the instrument's own output.
#
# Seams (selftest only, never set in a real run):
#   MEASURE_BW_PRODUCER  an executable emitting the probe's key=value protocol
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

HOST="${PERF_HOST:-}"
REPLICATES=5
WARMUP_MS=3000
BURST=32
MODEL=""
OUT_DIR="evidence/bandwidth"
SELFTEST=0

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST="${2:-}"; shift 2 ;;
        --replicates) REPLICATES="${2:-}"; shift 2 ;;
        --warmup-ms) WARMUP_MS="${2:-}"; shift 2 ;;
        --copies-per-replicate) BURST="${2:-}"; shift 2 ;;
        --model) MODEL="${2:-}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --selftest) SELFTEST=1; shift ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# The host name is the one perf-matrix.yaml uses, and it is NOT guessed from
# `hostname` for an unknown box. A bandwidth filed under the wrong host is a
# ceiling applied to the wrong device.
resolve_host() {
    if [ -n "$HOST" ]; then printf '%s' "$HOST"; return 0; fi
    case "$(hostname 2>/dev/null)" in
        noah-Lambda-Vector|lambda*) printf 'lambda' ;;
        gx10*|*-gb10) printf 'gx10' ;;
        *) return 1 ;;
    esac
}

# The device memory clock, MHz, or empty. Recorded either side of the timed
# window as EVIDENCE that the board was settled, rather than as an assertion
# that it was: a reader can see the two values instead of trusting the warmup.
mem_clock_mhz() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' | head -1
}

# median/min/max of the replicate rates on stdin (one integer-ish per line).
# awk, not bash arithmetic: these are ~1e12 and bash integers are fine but the
# even-n median needs a division that must not truncate to an integer.
stats_of() {
    sort -n | awk '
        { v[NR] = $1 }
        END {
            if (NR == 0) { exit 1 }
            if (NR % 2 == 1) { med = v[(NR + 1) / 2] }
            else             { med = (v[NR / 2] + v[NR / 2 + 1]) / 2 }
            printf "%.0f %.0f %.0f %d\n", med, v[1], v[NR], NR
        }'
}

# Run the probe and print its raw key=value output, or return non-zero.
# MEASURE_BW_PRODUCER replaces compile+run so the selftest can drive every
# branch without a GPU; a real run never sets it.
run_probe() { # run_probe <replicates> <warmup-ms> <copies-per-replicate>
    local n="$1" wms="${2:-0}" burst="${3:-1}"
    if [ -n "${MEASURE_BW_PRODUCER:-}" ]; then
        "$MEASURE_BW_PRODUCER" "$n" "$wms" "$burst"
        return $?
    fi
    command -v nvcc >/dev/null 2>&1 || return 2
    local src="scripts/lib/bandwidth_d2d.cu"
    [ -f "$src" ] || return 2
    local bd
    bd=$(mktemp -d) || return 2
    if ! nvcc -O2 -o "$bd/bandwidth_d2d" "$src" > "$bd/nvcc.log" 2>&1; then
        printf 'FAIL  nvcc could not build %s:\n' "$src" >&2
        sed 's/^/      /' "$bd/nvcc.log" >&2
        rm -rf "${bd:?}"
        return 2
    fi
    "$bd/bandwidth_d2d" "$n" "$wms" "$burst"
    local prc=$?
    rm -rf "${bd:?}"
    return "$prc"
}

# ---------------------------------------------------------------------------
# measure <host> <replicates> <out-dir>
#   0 = written; 2 = no instrument (no nvcc / no device / probe refused);
#   3 = the probe ran but produced fewer than 5 replicates.
measure() {
    local host="$1" n="$2" out_dir="$3" wms="${4:-0}" burst="${5:-1}"
    local raw prc clk_before clk_after
    clk_before=$(mem_clock_mhz)
    raw=$(run_probe "$n" "$wms" "$burst"); prc=$?
    clk_after=$(mem_clock_mhz)
    if [ "$prc" -ne 0 ]; then
        printf 'REFUSE bandwidth is UNMEASURED on %s: the probe returned rc=%s.\n' "$host" "$prc" >&2
        printf '       Nothing is written. §2.4 forbids publishing a percentage of a\n' >&2
        printf '       ceiling until a [V] bandwidth exists (PP-23).\n' >&2
        return 2
    fi

    local reps rates count
    rates=$(printf '%s\n' "$raw" | sed -n 's/^replicate=//p')
    count=$(printf '%s\n' "$rates" | grep -c '[0-9]')
    if [ "$count" -lt 5 ]; then
        printf 'REFUSE the probe produced %s replicate(s); at least 5 are required.\n' "$count" >&2
        printf '       n < 5 bounds no variance (§4.3), so a min/max over it would\n' >&2
        printf '       look like a spread while measuring none. Nothing is written.\n' >&2
        return 3
    fi

    local st med lo hi nn
    st=$(printf '%s\n' "$rates" | stats_of) || return 3
    med=$(printf '%s' "$st" | cut -d' ' -f1)
    lo=$(printf '%s' "$st" | cut -d' ' -f2)
    hi=$(printf '%s' "$st" | cut -d' ' -f3)
    nn=$(printf '%s' "$st" | cut -d' ' -f4)

    local dev bytes traffic started wmr wcopies
    dev=$(printf '%s\n' "$raw" | sed -n 's/^device_name=//p' | head -1)
    bytes=$(printf '%s\n' "$raw" | sed -n 's/^bytes=//p' | head -1)
    traffic=$(printf '%s\n' "$raw" | sed -n 's/^traffic_bytes=//p' | head -1)
    wmr=$(printf '%s\n' "$raw" | sed -n 's/^warmup_ms=//p' | head -1)
    wcopies=$(printf '%s\n' "$raw" | sed -n 's/^warmup_copies=//p' | head -1)
    local bper
    bper=$(printf '%s\n' "$raw" | sed -n 's/^copies_per_replicate=//p' | head -1)
    # DET002 suppressed: started_utc is a PROVENANCE field (PP-30). A
    # measurement stamped with the commit date instead of the moment it ran
    # would be a timestamp that cannot say when the device was measured.
    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002
    [ -n "$dev" ] || dev="unknown"

    mkdir -p "$out_dir" || return 2
    reps=$(printf '%s\n' "$rates" | grep '[0-9]' | paste -sd, -)
    cat > "$out_dir/$host.json" <<JSON
{
  "host": "$host",
  "device_name": "$dev",
  "started_utc": "$started",
  "method": "cudaMemcpy D2D 1GiB",
  "traffic_model": "median_bytes_per_sec is DRAM TRAFFIC per second, not copy size per second, and one timed window is a BURST of copies_per_replicate copies rather than a single copy: each D2D copy of $bytes B both reads and writes, moving ${traffic:-0} B through the memory system, so the reported rate is copies_per_replicate * ${traffic:-0} B divided by the elapsed time of the whole window. Reading the window as one copy understates the rate by a factor of copies_per_replicate. This is the quantity a vendor peak names and the quantity per-sequence decode is bounded by.",
  "copy_bytes": ${bytes:-0},
  "copies_per_replicate": ${bper:-null},
  "warmup_ms": ${wmr:-null},
  "warmup_copies": ${wcopies:-null},
  "mem_clock_mhz_before_warmup": ${clk_before:-null},
  "mem_clock_mhz_after_run": ${clk_after:-null},
  "n": $nn,
  "median_bytes_per_sec": $med,
  "min": $lo,
  "max": $hi,
  "max_over_min": $(awk -v a="$hi" -v b="$lo" 'BEGIN{ if (b > 0) printf "%.4f", a/b; else printf "null" }'),
  "replicates": [$reps],
  "provenance": "[V]"
}
JSON
    printf 'ok    %s: median %s B/s (%.1f GB/s) over n=%s [min %s, max %s]\n' \
        "$host" "$med" "$(awk -v m="$med" 'BEGIN{printf "%.1f", m/1e9}')" "$nn" "$lo" "$hi"
    printf 'ok    wrote %s\n' "$out_dir/$host.json"
    printf 'REPORT spread max/min = %s; memory clock %s MHz before the warmup, %s MHz\n' \
        "$(awk -v a="$hi" -v b="$lo" 'BEGIN{ if (b > 0) printf "%.3f", a/b; else printf "n/a" }')" \
        "${clk_before:-unread}" "${clk_after:-unread}"
    printf '       after the last replicate. The BEFORE value is expected to be the idle\n'
    printf '       clock; the AFTER value is the settled one, and a spread well above 1\n'
    printf '       means the replicates did not all run at it. Raise --warmup-ms or\n'
    printf '       --copies-per-replicate and re-measure rather than taking a median\n'
    printf '       over a mixture.\n'
    BW_MEDIAN="$med"
    return 0
}

# ---------------------------------------------------------------------------
if [ "$SELFTEST" -eq 1 ]; then
    rc=0
    printf -- '--- measure_bandwidth: the instrument, against stub producers -------\n'
    td=$(mktemp -d) || exit 2
    trap 'rm -rf "${td:?}"' EXIT

    mkprod() { # mkprod <file> <body>
        printf '#!/bin/sh\n%s\n' "$2" > "$1"; chmod +x "$1"
    }
    # 5 replicates, deliberately out of order: the median must SORT, and a
    # median that just takes the middle INPUT would read 5 here instead of 3.
    mkprod "$td/five" 'echo device_name=StubGPU; echo bytes=1073741824; echo traffic_bytes=2147483648; for r in 5 1 3 2 4; do echo "replicate=${r}000000000000"; done'
    mkprod "$td/six"  'echo device_name=StubGPU; echo bytes=1073741824; echo traffic_bytes=2147483648; for r in 1 2 3 4 5 6; do echo "replicate=${r}00000000000"; done'
    mkprod "$td/three" 'echo device_name=StubGPU; echo bytes=1073741824; echo traffic_bytes=2147483648; for r in 1 2 3; do echo "replicate=${r}000000000000"; done'
    mkprod "$td/dead" 'echo "error=no CUDA device present" >&2; exit 2'

    row() { # row <name> <producer> <expected-rc> <expected-median-or-NONE>
        local name="$1" prod="$2" want_rc="$3" want_med="$4" got_rc got_med=NONE
        rm -f "$td/out/$name.json"
        BW_MEDIAN=""
        MEASURE_BW_PRODUCER="$prod" measure "$name" 5 "$td/out" 0 1 >/dev/null 2>&1
        got_rc=$?
        if [ -f "$td/out/$name.json" ]; then
            got_med=$(sed -n 's/.*"median_bytes_per_sec": \([0-9]*\).*/\1/p' "$td/out/$name.json")
        fi
        if [ "$got_rc" != "$want_rc" ]; then
            printf 'FAIL  %-38s rc=%s, expected rc=%s\n' "$name" "$got_rc" "$want_rc"; rc=1; return
        fi
        if [ "$got_med" != "$want_med" ]; then
            printf 'FAIL  %-38s median=%s, expected %s\n' "$name" "$got_med" "$want_med"; rc=1; return
        fi
        printf 'ok    %-38s rc=%s median=%s\n' "$name" "$got_rc" "$got_med"
    }

    row bw_median_odd         "$td/five"  0 3000000000000
    row bw_median_even        "$td/six"   0 350000000000
    row bw_refuses_short      "$td/three" 3 NONE
    row bw_refuses_no_device  "$td/dead"  2 NONE
    row bw_refuses_absent     "$td/nope"  2 NONE

    printf '\n'
    if [ "$rc" -eq 0 ]; then
        printf 'PASS  the instrument sorts before it takes a median, and refuses a\n'
        printf '      short run, a dead device and an absent probe WITHOUT writing.\n'
    else
        printf 'FAIL  see rows above (PP-23).\n'
    fi
    exit "$rc"
fi

# ---------------------------------------------------------------------------
host=$(resolve_host) || {
    printf 'FAIL  cannot tell which perf-matrix host this is (hostname %s).\n' "$(hostname 2>/dev/null)" >&2
    printf '      Pass --host <lambda|gx10|intel|mini> or set PERF_HOST. A bandwidth\n' >&2
    printf '      filed under the wrong host is a ceiling applied to another device.\n' >&2
    exit 2
}
case "$REPLICATES" in ''|*[!0-9]*) printf 'FAIL  --replicates must be a number\n' >&2; exit 2 ;; esac
case "$WARMUP_MS" in ''|*[!0-9]*) printf 'FAIL  --warmup-ms must be a number\n' >&2; exit 2 ;; esac
case "$BURST" in ''|*[!0-9]*) printf 'FAIL  --copies-per-replicate must be a number\n' >&2; exit 2 ;; esac

printf -- '--- device bandwidth (PP-23, §12 row 14) ----------------------------\n'
BW_MEDIAN=""
measure "$host" "$REPLICATES" "$OUT_DIR" "$WARMUP_MS" "$BURST"
mrc=$?
[ "$mrc" -eq 0 ] || exit "$mrc"

# The derived ceiling is a REPORT line, never a written field. It is
# per-sequence decode only (PP-23): a batched aggregate legitimately exceeds it.
if [ -n "$MODEL" ]; then
    if [ -f "$MODEL" ]; then
        mbytes=$(stat -c %s "$MODEL" 2>/dev/null || stat -f %z "$MODEL")
        printf 'REPORT model %s is %s bytes\n' "$MODEL" "$mbytes"
        printf 'REPORT derived ceiling = %s / %s = %s tok/s (PER-SEQUENCE DECODE ONLY;\n' \
            "$BW_MEDIAN" "$mbytes" \
            "$(awk -v b="$BW_MEDIAN" -v m="$mbytes" 'BEGIN{printf "%.1f", b/m}')"
        printf '       a batched AGGREGATE may exceed it without being suspect)\n'
    else
        printf 'REPORT --model %s does not exist; no ceiling derived\n' "$MODEL" >&2
    fi
fi
exit 0
