#!/usr/bin/env bash
# check_loop_admission.sh - APR-OBS-001 §5.1 (OBS-10, #4497): a consumer may act
# on speed signals only when it has been ADMITTED.
#
# A consumer (PRM-001 REX-10 perf ratchet, ARB-SELF-001 arbiter, any autotuner)
# is admitted iff ALL hold:
#   P  the prerequisite rows are GREEN: OBS-01, OBS-03, OBS-04, OBS-05. A row
#      counts GREEN when its contract is in contracts/ - §3 ships each contract
#      in the same PR as its row, with its falsifiers planted RED->GREEN.
#   N  >= 14 consecutive nights of ADMISSIBLE apr-perf-ledger-v1 rows on the
#      consumer's host x backend, ending tonight or last night. Admissible =
#      every §2.1 identity field present and non-empty, and a gpu_proof on any
#      non-cpu row (R-2: a missing field is a missing row).
#   W  the consumer declares its write paths, and none of them reaches a ledger,
#      a RED rule, a baseline, an APR-OBS-001 contract, the consumer registry,
#      this gate or the CI surface that runs it (R-10: writer != subject).
#
# Registry: docs/obs/loop-admission-consumers.tsv, one consumer per line:
#   consumer  host  backend  acting(yes|no)  writes(comma globs|none)  spec
# `writes` globs are repo-relative; `fleet_perf_dir/...` names the ledger dir.
#
# Exit 1 when a consumer marked acting=yes is refused, or the registry is empty
# (R-2: an empty registry is not a pass). A refused acting=no consumer is
# reported, not RED: it is not consuming signals yet.
#
# Usage: check_loop_admission.sh [--root DIR] [--registry F] [--ledger-dir D]
#                                [--today YYYY-MM-DD] [--json OUT]
#        check_loop_admission.sh --self-test
# Exit 0 = no acting consumer refused. 1 = refused or empty. 2 = usage/env.

set -euo pipefail

MIN_NIGHTS=14
PREREQS="OBS-01:apr-lane-row-v1 OBS-03:apr-serve-timing-v1 OBS-04:apr-selfreport-agreement-v1 OBS-05:apr-perf-ledger-v1"
# Representative paths of everything a consumer must not be able to write.
PROTECTED=(
    "fleet_perf_dir/apr-perf-ledger.jsonl"
    "contracts/apr-lane-row-v1.yaml"
    "contracts/apr-nightly-liveness-v1.yaml"
    "contracts/apr-serve-timing-v1.yaml"
    "contracts/apr-selfreport-agreement-v1.yaml"
    "contracts/apr-perf-ledger-v1.yaml"
    "contracts/apr-perf-red-rules-v1.yaml"
    "contracts/apr-perf-baseline-v1.yaml"
    "contracts/apr-trace-v1.yaml"
    "contracts/apr-perf-pr-trace-v1.yaml"
    "contracts/apr-loop-admission-v1.yaml"
    "docs/specifications/APR-OBS-001-dogfood-observability.md"
    "docs/obs/loop-admission-consumers.tsv"
    "scripts/check_loop_admission.sh"
    "scripts/guard_tree.sh"
    ".github/workflows/ci.yml"
    "ci/sections.yml"
)

usage() { sed -n '26,30p' "$0" >&2; exit 2; }

# admissible_dates HOST BACKEND FILE... -> sorted unique UTC dates with an admissible row
admissible_dates() {
    local host=$1 backend=$2
    shift 2
    [ $# -gt 0 ] || return 0
    jq -rR --arg h "$host" --arg b "$backend" '
        fromjson? | select(type == "object")
        | select((.schema // "") | tostring | startswith("apr-perf-ledger-v1"))
        | select(.host == $h and .backend == $b)
        | select([.ts, .apr_version, .apr_tag, .crate_tarball_sha256, .binary_sha256,
                  .build_identity, .model_id, .model_sha256, .request_id]
                 | all(. != null and . != ""))
        | select(.backend == "cpu" or (.gpu_proof != null and .gpu_proof != ""))
        | .ts | tostring | .[0:10]' "$@" | sort -u
}

# consecutive_nights TODAY < dates -> length of the run ending TODAY or TODAY-1
consecutive_nights() {
    local today=$1 d n=0 have=" "
    while IFS= read -r d; do have="$have$d "; done
    d=$today
    case "$have" in *" $d "*) ;; *) d=$(date -u -d "$today - 1 day" +%F) ;; esac
    while case "$have" in *" $d "*) true ;; *) false ;; esac; do
        n=$((n + 1))
        d=$(date -u -d "$d - 1 day" +%F)
    done
    echo "$n"
}

# write_hit WRITES -> prints the first protected path a write glob reaches
write_hit() {
    local writes=$1 w p
    local -a globs
    IFS=',' read -r -a globs <<< "$writes"
    for w in "${globs[@]}"; do
        w="${w#"${w%%[![:space:]]*}"}"
        w="${w%"${w##*[![:space:]]}"}"
        [ -n "$w" ] || continue
        case "$w" in */) w="${w}*" ;; esac
        for p in "${PROTECTED[@]}"; do
            # shellcheck disable=SC2053 # $w is a glob on purpose
            if [[ $p == $w ]]; then printf '%s (via %s)' "$p" "$w"; return 0; fi
        done
    done
    return 1
}

# judge ROOT LEDGER_DIR TODAY CONSUMER HOST BACKEND WRITES -> "admitted|reason"
judge() {
    local root=$1 ledger=$2 today=$3 consumer=$4 host=$5 backend=$6 writes=$7
    local pr row id missing="" nights hit
    local -a files=()
    for pr in $PREREQS; do
        row=${pr%%:*} id=${pr#*:}
        [ -f "$root/contracts/$id.yaml" ] || missing="$missing $row($id)"
    done
    if [ -n "$missing" ]; then
        echo "false|prerequisite rows not GREEN:$missing"; return
    fi
    case "$writes" in
        ''|unassigned|unknown|tbd|TBD)
            echo "false|no declared write paths (declare them, or 'none')"; return ;;
    esac
    if [ "$writes" != "none" ] && hit=$(write_hit "$writes"); then
        echo "false|holds a write path to $hit"; return
    fi
    if [ -d "$ledger" ]; then
        while IFS= read -r -d '' f; do files+=("$f"); done \
            < <(find "$ledger" -type f -name '*.jsonl' -print0 | sort -z)
    fi
    nights=$(admissible_dates "$host" "$backend" "${files[@]}" | consecutive_nights "$today")
    if [ "$nights" -lt "$MIN_NIGHTS" ]; then
        echo "false|$nights consecutive admissible nights on $host/$backend, needs $MIN_NIGHTS"; return
    fi
    echo "true|all of §5.1 hold ($nights nights on $host/$backend)"
}

run() { # run ROOT REGISTRY LEDGER TODAY JSON_OUT
    local root=$1 reg=$2 ledger=$3 today=$4 out=$5
    local consumer host backend acting writes _spec verdict admitted reason n=0 rc=0
    local rows=""
    [ -f "$reg" ] || { echo "FAIL: no consumer registry: $reg" >&2; return 1; }
    local line
    while IFS= read -r line; do
        # Tab is IFS whitespace: `read` would merge an EMPTY column into the next
        # one. Split on a non-whitespace separator so an empty `writes` stays empty.
        line=${line//$'\t'/$'\x1f'}
        IFS=$'\x1f' read -r consumer host backend acting writes _spec <<< "$line"
        case "$consumer" in ''|'#'*|consumer) continue ;; esac
        n=$((n + 1))
        verdict=$(judge "$root" "$ledger" "$today" "$consumer" "$host" "$backend" "${writes:-}")
        admitted=${verdict%%|*} reason=${verdict#*|}
        if [ "$admitted" = true ]; then
            printf 'ADMITTED %s: %s\n' "$consumer" "$reason"
        elif [ "$acting" = yes ]; then
            printf 'REFUSED  %s (ACTING on speed signals): %s\n' "$consumer" "$reason"; rc=1
        else
            printf 'refused  %s (not acting): %s\n' "$consumer" "$reason"
        fi
        rows+=$(jq -cn --arg c "$consumer" --argjson a "$admitted" --arg r "$reason" \
            '{consumer: $c, admitted: $a, reason: $r}')$'\n'
    done < "$reg"
    if [ "$n" -eq 0 ]; then echo "FAIL: consumer registry is empty: $reg" >&2; return 1; fi
    if [ -n "$out" ]; then
        printf '%s' "$rows" | jq -s '{loop_admission: .}' > "$out"
    fi
    [ "$rc" -eq 0 ] && printf 'OK: %d consumer(s) judged; none acts on speed signals unadmitted\n' "$n"
    return "$rc"
}

self_test() {
    local td fail=0 self
    self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    td=$(mktemp -d "${TMPDIR:-/tmp}/loopadm-self.XXXXXX") || exit 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${td:?}'" EXIT
    mkdir -p "$td/root/contracts" "$td/ledger"
    local pr
    for pr in $PREREQS; do : > "$td/root/contracts/${pr#*:}.yaml"; done

    nightly() { # nightly FILE HOST BACKEND FIRST_DAY_AGO LAST_DAY_AGO [DROP_FIELD]
        local f=$1 h=$2 b=$3 i proof='"trace:kernel-launch"'
        [ "$b" = cpu ] && proof=null
        for i in $(seq "$5" "$4"); do
            jq -cn --arg ts "$(date -u -d "2026-10-31 - $i day" +%F)T03:00:00Z" \
                --arg h "$h" --arg b "$b" --argjson gp "$proof" --arg drop "${6:-}" '
                {schema: "apr-perf-ledger-v1", ts: $ts, host: $h, apr_version: "0.71.0",
                 apr_tag: "v0.71.0", crate_tarball_sha256: "aa", binary_sha256: "bb",
                 build_identity: {rustc: "1.90"}, model_id: "qwen", model_sha256: "cc",
                 backend: $b, gpu_proof: $gp, request_id: "0190"} | del(.[$drop])'
        done >> "$f"
    }
    # gx10/cuda: nights 0..13 ago (14). lambda/cpu: 1..13 ago (13, ends last night).
    nightly "$td/ledger/gx10.jsonl" gx10 cuda 13 0
    nightly "$td/ledger/lambda.jsonl" lambda cpu 13 1
    # yoga/cuda: 14 nights, one missing model_sha256 (night 5) -> run of 5.
    nightly "$td/ledger/yoga.jsonl" yoga cuda 13 6
    nightly "$td/ledger/yoga.jsonl" yoga cuda 5 5 model_sha256
    nightly "$td/ledger/yoga.jsonl" yoga cuda 4 0
    # intel/wgpu: 14 nights, one with gpu_proof removed (night 7) -> run of 7.
    nightly "$td/ledger/intel.jsonl" intel wgpu 13 8
    nightly "$td/ledger/intel.jsonl" intel wgpu 7 7 gpu_proof
    nightly "$td/ledger/intel.jsonl" intel wgpu 6 0
    printf 'not json\n{"schema":"apr-perf-ledger-v1"}\n' >> "$td/ledger/gx10.jsonl"

    row() { # row WANT_RC MUST_MATCH LABEL REGISTRY_LINES [ROOT]
        local out rc
        printf 'consumer\thost\tbackend\tacting\twrites\tspec\n%b' "$4" > "$td/reg.tsv"
        out=$(bash "$self" --root "${5:-$td/root}" --registry "$td/reg.tsv" --ledger-dir "$td/ledger" \
            --today 2026-10-31 --json "$td/out.json" 2>&1) && rc=0 || rc=$?
        if [ "$rc" -eq "$1" ] && grep -qE -- "$2" <<< "$out"; then printf 'ok   self-test: %s (exit %s)\n' "$3" "$rc"
        else printf 'FAIL self-test: %s: want exit %s + /%s/, got %s: %s\n' "$3" "$1" "$2" "$rc" "$out"; fail=1; fi
    }
    row 0 '^ADMITTED c: all of' "all of §5.1 hold -> admitted" 'c\tgx10\tcuda\tyes\tcrates/aprender-serve/**\tS\n'
    row 1 'REFUSED  c .*write path to contracts/apr-perf-red-rules-v1.yaml' \
        "planted write path to a RED rule -> refused (done-when 1)" \
        'c\tgx10\tcuda\tyes\tcrates/**,contracts/apr-perf-red-rules-v1.yaml\tS\n'
    row 1 'REFUSED  c .*13 consecutive admissible nights on lambda/cpu, needs 14' \
        "13 nights (before night 14) -> refused (done-when 2)" 'c\tlambda\tcpu\tyes\tnone\tS\n'
    row 1 'REFUSED  c .*5 consecutive .*yoga/cuda' "a row missing model_sha256 breaks the run" 'c\tyoga\tcuda\tyes\tnone\tS\n'
    row 1 'REFUSED  c .*7 consecutive .*intel/wgpu' "a GPU row without gpu_proof breaks the run" 'c\tintel\twgpu\tyes\tnone\tS\n'
    row 1 'REFUSED  c .*0 consecutive .*gx10/cpu' "another backend's rows do not count" 'c\tgx10\tcpu\tyes\tnone\tS\n'
    row 1 'REFUSED  c .*write path to fleet_perf_dir' "a write path to the ledger dir -> refused" 'c\tgx10\tcuda\tyes\tfleet_perf_dir/\tS\n'
    row 1 'REFUSED  c .*write path to scripts/check_loop_admission.sh' "a write path to this gate -> refused" 'c\tgx10\tcuda\tyes\tscripts/**\tS\n'
    row 1 'REFUSED  c .*write path to docs/obs/loop-admission-consumers.tsv' "a write path to the registry -> refused" 'c\tgx10\tcuda\tyes\tdocs/obs/*\tS\n'
    row 1 'REFUSED  c .*write path' "a write-everything glob -> refused" 'c\tgx10\tcuda\tyes\t**\tS\n'
    row 1 'REFUSED  c .*no declared write paths' "undeclared write paths -> refused" 'c\tgx10\tcuda\tyes\t\tS\n'
    row 1 'REFUSED  c .*no declared write paths' "a placeholder ('unassigned') is not a declaration" \
        'c\tgx10\tcuda\tyes\tunassigned\tS\n'
    mkdir -p "$td/bare/contracts"
    row 1 'REFUSED  c .*not GREEN: OBS-01.*OBS-05' "prerequisite rows absent -> refused" \
        'c\tgx10\tcuda\tyes\tnone\tS\n' "$td/bare"
    row 0 '^refused  c \(not acting\)' "a refused consumer that does not act is reported, exit 0" 'c\tlambda\tcpu\tno\tnone\tS\n'
    row 1 'registry is empty' "an empty registry is RED (R-2)" ''
    row 0 'ADMITTED' "the §9 loop_admission JSON is written" 'c\tgx10\tcuda\tyes\tnone\tS\n'
    if jq -e '.loop_admission == [{"consumer":"c","admitted":true,"reason":"all of §5.1 hold (14 nights on gx10/cuda)"}]' \
        "$td/out.json" >/dev/null; then echo "ok   self-test: §9 JSON shape {consumer, admitted, reason}"
    else echo "FAIL self-test: §9 JSON shape: $(cat "$td/out.json")"; fail=1; fi
    return "$fail"
}

ROOT="" REG="" LEDGER="${FLEET_PERF_DIR:-}" TODAY="" JSON=""
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) self_test; exit $? ;;
        --root) [ $# -ge 2 ] || usage; ROOT=$2; shift 2 ;;
        --registry) [ $# -ge 2 ] || usage; REG=$2; shift 2 ;;
        --ledger-dir) [ $# -ge 2 ] || usage; LEDGER=$2; shift 2 ;;
        --today) [ $# -ge 2 ] || usage; TODAY=$2; shift 2 ;;
        --json) [ $# -ge 2 ] || usage; JSON=$2; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) usage ;;
    esac
done
command -v jq >/dev/null || { echo "jq not found" >&2; exit 2; }
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
REG=${REG:-$ROOT/docs/obs/loop-admission-consumers.tsv}
TODAY=${TODAY:-$(date -u +%F)}
run "$ROOT" "$REG" "$LEDGER" "$TODAY" "$JSON"
