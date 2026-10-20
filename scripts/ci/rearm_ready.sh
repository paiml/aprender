#!/usr/bin/env bash
# rearm_ready.sh -- which ready PRs a gate fix may re-arm (contract advisory-gate-diff-v1, ADG-INV-005).
#
#   rearm_ready.sh select FILE     FILE = TSV "pr  approved_sha  current_sha"; prints PRs to arm
#   rearm_ready.sh --self-test     case table
# A PR whose head moved after its approval is NOT re-armed (a changed head needs a fresh review).
set -euo pipefail

select_() {
    awk -F'\t' 'NF >= 3 && $1 != "" && $2 ~ /^[0-9a-f]{40}$/ && $2 == $3 { print $1 }' "$1"
}

self_test() {
    local d bad=0 a b
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    a=$(printf 'a%.0s' {1..40}); b=$(printf 'b%.0s' {1..40})
    printf '1\t%s\t%s\n2\t%s\t%s\n3\t\t%s\n' "$a" "$a" "$a" "$b" "$a" >"$d/t.tsv"
    [ "$(select_ "$d/t.tsv" | tr '\n' ' ')" = "1 " ] || { echo "SELF-TEST FAIL: only PR 1 (head unchanged) may be re-armed" >&2; bad=1; }
    printf '' >"$d/e.tsv"
    [ -z "$(select_ "$d/e.tsv")" ] || { echo "SELF-TEST FAIL: empty input arms nothing" >&2; bad=1; }
    [ "$bad" -eq 0 ] && echo "rearm_ready self-test: 2 cases ok"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    select) select_ "$2" ;;
    *) echo "usage: $0 select FILE | --self-test" >&2; exit 2 ;;
esac
