#!/usr/bin/env bash
# check_perf_pr_trace.sh - APR-OBS-001 OBS-11 (aprender#4498), contract
# apr-perf-pr-trace-v1.
#
# §5.2: "Every 0.71 performance PR carries a before/after apr-trace-v1 diff for
# the layers it claims to change, measured on the same host at the same
# identity. A perf PR without it does not merge."
#
# A PR is a PERF PR when its title is a conventional `perf` commit
# (`perf: …`, `perf(scope): …`, `perf!: …`) or it carries the `perf` label.
# A perf PR must add or change at least one evidence/perf-pr-trace/*.json, and
# every such file must pass `"$APR" perf-pr-trace-lint --base B --head H`.
# A PR that is not a perf PR is GREEN without looking further.
#
# Usage:
#   check_perf_pr_trace.sh --title T [--labels a,b] --base SHA --head SHA
#                          [--changed-files FILE]
#   check_perf_pr_trace.sh --self-test
#
# --changed-files names a file with one repo path per line; without it the list
# is `git diff --name-only --diff-filter=AM BASE...HEAD`.
#
# R-6: REPORT-ONLY until first green plus 7 green nights. A RED verdict prints
# and exits 0; PERF_PR_TRACE_ENFORCE=1 makes it exit 1. --self-test is always
# enforced: it tests the check, not a PR.
#
# $APR must name the pinned binary (`. scripts/apr_bin.sh`); the check refuses
# to run a perf PR's evidence through a bare `apr`.

set -euo pipefail

ST_DIR=""
EVIDENCE_RE='^evidence/perf-pr-trace/[^/]+\.json$'

is_perf_pr() {
    local title="$1" labels="$2"
    if printf '%s\n' "$title" | grep -Eq '^perf(\(|:|!)'; then
        return 0
    fi
    case ",${labels}," in
        *,perf,*) return 0 ;;
    esac
    return 1
}

# Prints GREEN/RED and the reasons; returns 0 for GREEN, 1 for RED.
verdict() {
    local title="$1" labels="$2" base="$3" head="$4" changed="$5"
    if ! is_perf_pr "$title" "$labels"; then
        echo "GREEN: not a perf PR"
        return 0
    fi
    local files
    files="$(grep -E "$EVIDENCE_RE" "$changed" || true)"
    if [ -z "$files" ]; then
        echo "RED: perf PR carries no evidence/perf-pr-trace/*.json (§5.2)"
        return 1
    fi
    if [ -z "${APR:-}" ] || [ ! -x "${APR}" ]; then
        echo "RED: \$APR is not an executable apr binary; source scripts/apr_bin.sh"
        return 1
    fi
    local f rc=0
    while IFS= read -r f; do
        if [ ! -f "$f" ]; then
            echo "RED: $f is listed as changed but is not in the tree"
            rc=1
            continue
        fi
        if ! "$APR" perf-pr-trace-lint "$f" --base "$base" --head "$head"; then
            echo "RED: $f rejected by perf-pr-trace-lint"
            rc=1
        fi
    done <<< "$files"
    if [ "$rc" -eq 0 ]; then
        echo "GREEN: perf PR evidence passes apr-perf-pr-trace-v1"
    fi
    return "$rc"
}

self_test() {
    local fails=0
    ST_DIR="$(mktemp -d)"
    local dir="$ST_DIR"
    trap 'rm -rf "${ST_DIR:?}"' EXIT INT TERM
    local base="aca6f2d7f6" head="d617620271"
    mkdir -p "$dir/evidence/perf-pr-trace"
    printf 'crates/aprender-serve/src/lib.rs\n' > "$dir/code-only.txt"
    printf 'crates/aprender-serve/src/lib.rs\nevidence/perf-pr-trace/pr.json\n' > "$dir/with-ev.txt"
    printf 'evidence/perf-pr-trace/nested/pr.json\nevidence/perf-pr-trace/pr.txt\n' > "$dir/wrong-ev.txt"

    row() {
        local want="$1" name="$2" title="$3" labels="$4" changed="$5" got
        if (cd "$dir" && verdict "$title" "$labels" "$base" "$head" "$changed") > "$dir/out" 2>&1; then
            got=GREEN
        else
            got=RED
        fi
        if [ "$got" = "$want" ]; then
            printf '  %-52s %-5s (expected %s)  OK\n' "$name" "$got" "$want"
        else
            printf '  %-52s %-5s (expected %s)  FAIL\n' "$name" "$got" "$want"
            sed 's/^/      /' "$dir/out"
            fails=$((fails + 1))
        fi
    }

    echo "check_perf_pr_trace.sh --self-test"
    row GREEN "not a perf PR (fix:)" "fix: faster thing" "" "$dir/code-only.txt"
    row GREEN "'performance' in a fix: title is not perf" "fix: performance of x" "" "$dir/code-only.txt"
    row GREEN "'perfect:' is not perf" "perfect: x" "" "$dir/code-only.txt"
    row RED "perf: title, no evidence" "perf: fuse gate+up" "" "$dir/code-only.txt"
    row RED "perf(scope): title, no evidence" "perf(serve): x" "" "$dir/code-only.txt"
    row RED "perf!: title, no evidence" "perf!: x" "" "$dir/code-only.txt"
    row RED "perf label, fix: title, no evidence" "fix: x" "bug,perf" "$dir/code-only.txt"
    row GREEN "perfect label is not perf" "fix: x" "perfect" "$dir/code-only.txt"
    row RED "evidence nested or not .json does not count" "perf: x" "" "$dir/wrong-ev.txt"
    row RED "evidence listed but not in the tree" "perf: x" "" "$dir/with-ev.txt"

    if [ -n "${APR:-}" ] && [ -x "${APR}" ]; then
        local ok='{"schema":"apr-perf-pr-trace-v1","pr":1,"claimed":["attention"],'
        local id='"identity":{"schema":"apr-trace-v1","ts":"2026-09-27T14:00:00Z","host":"lambda","apr_version":"0.71.0","apr_tag":"v0.71.0-rc.1","crate_tarball_sha256":"'
        local sha64="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        local rest='","build_identity":{"rustc_vv":"r","target_triple":"x86_64-unknown-linux-gnu","features":["inference"],"uname_a":"u","accelerator":"none","driver":"none"},"model_id":"m","model_sha256":"'
        local tail='","backend":"cpu","gpu_proof":null,"request_id":"01927f3e-0000-7000-8000-000000000000"}'
        local tr='"trace":{"level":"layer","operations":1,"total_time_us":200,"breakdown":[{"name":"attention","time_us":100}],"provenance":"'
        local side_b side_a
        side_b="\"before\":{\"commit\":\"$base\",$id$sha64\",\"binary_sha256\":\"$sha64$rest$sha64$tail,${tr}measured\"}}"
        side_a="\"after\":{\"commit\":\"$head\",$id$sha64\",\"binary_sha256\":\"$sha64$rest$sha64$tail,${tr}measured\"}}"
        printf '%s%s,%s}\n' "$ok" "$side_b" "$side_a" > "$dir/evidence/perf-pr-trace/pr.json"
        row GREEN "perf PR, valid evidence (via \$APR)" "perf: x" "" "$dir/with-ev.txt"
        side_a="\"after\":{\"commit\":\"$head\",$id$sha64\",\"binary_sha256\":\"$sha64$rest$sha64$tail,${tr}wall_clock_total\"}}"
        printf '%s%s,%s}\n' "$ok" "$side_b" "$side_a" > "$dir/evidence/perf-pr-trace/pr.json"
        row RED "perf PR, after timings wall_clock_total" "perf: x" "" "$dir/with-ev.txt"
        side_a="\"after\":{\"commit\":\"$head\",${id/lambda/yoga}$sha64\",\"binary_sha256\":\"$sha64$rest$sha64$tail,${tr}measured\"}}"
        printf '%s%s,%s}\n' "$ok" "$side_b" "$side_a" > "$dir/evidence/perf-pr-trace/pr.json"
        row RED "perf PR, before/after on different hosts" "perf: x" "" "$dir/with-ev.txt"
    else
        echo "  SKIPPED 3 lint rows: \$APR not set (source scripts/apr_bin.sh to run them)"
        printf '%s\n' '{}' > "$dir/evidence/perf-pr-trace/pr.json"
        row RED "perf PR, evidence present, no \$APR -> RED not GREEN" "perf: x" "" "$dir/with-ev.txt"
    fi

    if [ "$fails" -ne 0 ]; then
        echo "SELF-TEST FAIL: $fails row(s)"
        return 1
    fi
    echo "SELF-TEST PASS"
}

main() {
    if [ "${1:-}" = "--self-test" ]; then
        self_test
        return
    fi
    local title="" labels="" base="" head="" changed=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --title) title="${2:?--title needs a value}"; shift 2 ;;
            --labels) labels="${2-}"; shift 2 ;;
            --base) base="${2:?--base needs a value}"; shift 2 ;;
            --head) head="${2:?--head needs a value}"; shift 2 ;;
            --changed-files) changed="${2:?--changed-files needs a value}"; shift 2 ;;
            *) echo "unknown argument: $1" >&2; return 2 ;;
        esac
    done
    if [ -z "$title" ] || [ -z "$base" ] || [ -z "$head" ]; then
        echo "usage: $0 --title T [--labels a,b] --base SHA --head SHA [--changed-files FILE]" >&2
        return 2
    fi
    local tmp=""
    if [ -z "$changed" ]; then
        tmp="$(mktemp)"
        git diff --name-only --diff-filter=AM "$base...$head" > "$tmp"
        changed="$tmp"
    fi
    local rc=0
    verdict "$title" "$labels" "$base" "$head" "$changed" || rc=$?
    if [ -n "$tmp" ]; then
        rm -f "${tmp:?}"
    fi
    if [ "$rc" -ne 0 ] && [ "${PERF_PR_TRACE_ENFORCE:-0}" != "1" ]; then
        echo "(report-only, R-6: set PERF_PR_TRACE_ENFORCE=1 to fail)"
        return 0
    fi
    return "$rc"
}

main "$@"
