#!/usr/bin/env bash
# check_sov_advisories_off.sh -- PMAT-5792: the vendored sov.lint `cargo deny check advisories` and
# sov.security `cargo audit` stop PRs on advisories already on main. They are off only when BOTH halves hold:
#   1. ci/sections.yml sovereign-ci.with sets advisories_diff_only: true
#   2. the vendored body reads that input in both steps and skips the scan on it
# usage: check_sov_advisories_off.sh [SECTIONS] [VENDORED] | --self-test
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"

check() { # SECTIONS VENDORED -> 0 ok, 1 gap
    local sec=$1 ven=$2 bad=0
    grep -Eq '^    advisories_diff_only: true$' "$sec" || { echo "GAP: $sec does not set advisories_diff_only: true under sovereign-ci.with" >&2; bad=1; }
    [ "$(grep -c 'SOV_ADVISORIES_DIFF_ONLY: \${{ inputs.advisories_diff_only }}' "$ven")" -ge 2 ] \
        || { echo "GAP: $ven does not pass the input to both the deny and the audit step" >&2; bad=1; }
    [ "$(grep -c '"${SOV_ADVISORIES_DIFF_ONLY:-}" = true' "$ven")" -ge 2 ] \
        || { echo "GAP: $ven does not skip both scans on the input" >&2; bad=1; }
    return "$bad"
}

self_test() {
    local d bad=0 rc
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    case_() { # name want sections vendored
        rc=0; check "$3" "$4" >/dev/null 2>&1 || rc=$?
        [ "$rc" -eq "$2" ] || { echo "SELF-TEST FAIL: $1 wanted rc=$2 got $rc" >&2; bad=1; }
    }
    S="$ROOT/ci/sections.yml"; V="$ROOT/ci/vendor/sovereign-ci.yml"
    case_ 'real tree' 0 "$S" "$V"
    sed 's/^    advisories_diff_only: true$/    advisories_diff_only: false/' "$S" >"$d/s.yml"
    case_ 'MUTANT input off in sections' 1 "$d/s.yml" "$V"
    sed 's/"${SOV_ADVISORIES_DIFF_ONLY:-}" = true/"${SOV_ADVISORIES_DIFF_ONLY:-}" = never/' "$V" >"$d/v.yml"
    case_ 'MUTANT vendored body does not honour it' 1 "$S" "$d/v.yml"
    sed '0,/SOV_ADVISORIES_DIFF_ONLY: ${{ inputs.advisories_diff_only }}/{//d}' "$V" >"$d/v2.yml"
    case_ 'MUTANT one step loses the env' 1 "$S" "$d/v2.yml"
    [ "$bad" -eq 0 ] && echo "check_sov_advisories_off self-test: 4 cases ok"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") check "$ROOT/ci/sections.yml" "$ROOT/ci/vendor/sovereign-ci.yml" ;;
    *) check "$1" "$2" ;;
esac
