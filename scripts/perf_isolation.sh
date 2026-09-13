#!/usr/bin/env bash
#
# perf_isolation.sh — record who else was on the device, before and after each
# band (§5.4 of PP-LLAMA-001 v3.0, PP-19).
#
# WHY A RECORD AND NOT A CHECK. §5.4 says any foreign compute PID is fatal to
# the band. That verdict belongs to the gate, which reads the receipt; this is
# the instrument that makes the verdict possible at all. Today no lane script
# looks at the device while it measures, so a run sharing a 4090 with another
# agent's build produces a number that is indistinguishable from a clean one.
#
# THE ABSENT PROBE IS NOT A CLEAN DEVICE. On a host with no nvidia-smi the
# record says `"probe": "absent"` and `compute_pids: null` — NOT an empty list.
# An empty list means "asked, and nobody was there"; null means "not asked".
# Collapsing the two is how an unmeasured isolation reads as isolation, which is
# the same shape as the coverage floor that was armed while the measurement
# returned 0/0.
#
# Usage:
#   scripts/perf_isolation.sh before <out.json>
#   scripts/perf_isolation.sh after  <out.json>
#   scripts/perf_isolation.sh --selftest
#
# Env:
#   PERF_ISOLATION_OWN_PIDS  space-separated PIDs this run owns; every compute
#                            PID not in the list is reported as FOREIGN
#   PERF_HOST                the perf-matrix host name for the record
#   PERF_ISOLATION_SMI       (selftest only) an executable standing in for
#                            nvidia-smi, so the parser can be driven without a GPU
set -uo pipefail

smi_cmd() {
    if [ -n "${PERF_ISOLATION_SMI:-}" ]; then
        printf '%s' "$PERF_ISOLATION_SMI"
        return 0
    fi
    command -v nvidia-smi 2>/dev/null
}

# `pid, used_memory` rows for every compute process, one `pid:mib` per line.
compute_rows() { # compute_rows <smi>
    "$1" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' \
        | grep -E '^[0-9]+,[0-9]+$' \
        | tr ',' ':'
}

# Total device memory in use, MiB, or the empty string.
device_used_mib() { # device_used_mib <smi>
    "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' | head -1
}

# The record. Prints JSON on stdout.
isolation_record() { # isolation_record <when>
    local when="$1" smi rows used host own pid mib
    host="${PERF_HOST:-$(hostname 2>/dev/null)}"
    smi=$(smi_cmd)
    if [ -z "$smi" ]; then
        printf '{"host": "%s", "when": "%s", "probe": "absent", "compute_pids": null, "foreign_pids": null, "memory_used_mib": null}\n' \
            "$host" "$when"
        return 0
    fi
    rows=$(compute_rows "$smi")
    used=$(device_used_mib "$smi")
    own=" ${PERF_ISOLATION_OWN_PIDS:-} "
    local all="" foreign=""
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        pid=${line%%:*}
        mib=${line##*:}
        all="$all,{\"pid\": $pid, \"used_memory_mib\": $mib}"
        case "$own" in *" $pid "*) ;; *) foreign="$foreign,$pid" ;; esac
    done <<< "$rows"
    printf '{"host": "%s", "when": "%s", "probe": "nvidia-smi", "compute_pids": [%s], "foreign_pids": [%s], "memory_used_mib": %s}\n' \
        "$host" "$when" "${all#,}" "${foreign#,}" "${used:-null}"
}

if [ "${1:-}" = "--selftest" ]; then
    rc=0
    printf -- '--- perf_isolation: the device record --------------------------------\n'
    td=$(mktemp -d) || exit 2
    trap 'rm -rf "${td:?}"' EXIT

    mksmi() { printf '#!/bin/sh\ncase "$1" in\n  --query-compute-apps=*) %s ;;\n  --query-gpu=*) %s ;;\nesac\n' "$2" "$3" > "$1"; chmod +x "$1"; }
    mksmi "$td/busy"  'printf "1234, 4096\n5678, 512\n"' 'printf "4608\n"'
    mksmi "$td/clean" 'true'                             'printf "758\n"'

    row() { # row <name> <smi> <own-pids> <jq-filter> <expected>
        local name="$1" got
        got=$(PERF_ISOLATION_SMI="$2" PERF_ISOLATION_OWN_PIDS="$3" \
              PERF_HOST=testhost isolation_record before | jq -c "$4" 2>&1)
        if [ "$got" = "$5" ]; then
            printf 'ok    %-44s %s\n' "$name" "$got"
        else
            printf 'FAIL  %-44s got %s, expected %s\n' "$name" "$got" "$5"; rc=1
        fi
    }

    command -v jq >/dev/null 2>&1 || { printf 'FAIL  jq is required for the case table\n'; exit 1; }
    row 'foreign_pid_breach (a foreign compute PID)' "$td/busy"  '1234' '.foreign_pids'    '[5678]'
    row 'foreign_pid_ok (every PID is ours)'         "$td/busy"  '1234 5678' '.foreign_pids' '[]'
    row 'both PIDs are recorded either way'       "$td/busy"  '1234 5678' '[.compute_pids[].pid]' '[1234,5678]'
    row 'per-PID memory is carried'               "$td/busy"  '1234 5678' '[.compute_pids[].used_memory_mib]' '[4096,512]'
    row 'device memory is carried'                "$td/busy"  '1234' '.memory_used_mib' '4608'
    row 'an idle device is an EMPTY list'         "$td/clean" ''     '.compute_pids'   '[]'
    row 'an idle device still reports memory'     "$td/clean" ''     '.memory_used_mib' '758'
    # THE ABSENT PROBE IS NULL, NOT EMPTY. Same call, no smi at all.
    got=$(PATH=/nonexistent PERF_HOST=testhost isolation_record after | jq -c '[.probe, (.compute_pids|type)]' 2>&1)
    if [ "$got" = '["absent","null"]' ]; then
        printf 'ok    %-44s %s\n' 'an ABSENT probe is null, not an empty list' "$got"
    else
        printf 'FAIL  %-44s got %s, expected ["absent","null"]\n' 'an ABSENT probe is null, not an empty list' "$got"; rc=1
    fi
    got=$(PERF_ISOLATION_SMI="$td/busy" PERF_HOST=testhost isolation_record after | jq -r '.when')
    if [ "$got" = "after" ]; then
        printf 'ok    %-44s when=after\n' 'the phase is recorded'
    else
        printf 'FAIL  %-44s when=%s, expected after\n' 'the phase is recorded' "$got"; rc=1
    fi

    printf '\n'
    if [ "$rc" -eq 0 ]; then
        printf 'PASS  the record distinguishes foreign from own, and an unasked probe\n'
        printf '      from an idle device.\n'
    else
        printf 'FAIL  see rows above (PP-19, §5.4).\n'
    fi
    exit "$rc"
fi

WHEN="${1:-}"
OUT="${2:-}"
case "$WHEN" in
    before|after) ;;
    *) printf 'usage: %s {before|after} <out.json>\n       %s --selftest\n' "$0" "$0" >&2; exit 2 ;;
esac
[ -n "$OUT" ] || { printf 'usage: %s %s <out.json>\n' "$0" "$WHEN" >&2; exit 2; }
isolation_record "$WHEN" > "$OUT"
