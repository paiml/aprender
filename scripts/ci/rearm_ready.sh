#!/usr/bin/env bash
# rearm_ready.sh -- which ready PRs a gate fix may re-arm (contract advisory-gate-diff-v1, ADG-INV-005).
#
#   rearm_ready.sh select FILE     FILE = TSV "pr  approved_sha  current_sha"; prints PRs to arm
#   rearm_ready.sh run             list open un-armed PRs (one gh call), select, arm each with plain `gh pr merge`
#   rearm_ready.sh --self-test     case table, plus `run` against a stub gh (RED when select_ is not called)
# A PR whose head moved after its approval is NOT re-armed (a changed head needs a fresh review).
set -euo pipefail

select_() {
    awk -F'\t' 'NF >= 3 && $1 != "" && $2 ~ /^[0-9a-f]{40}$/ && $2 == $3 { print $1 }' "$1"
}

# stdin: `gh pr list --json number,headRefOid,autoMergeRequest,reviews`; stdout: TSV "pr  approved_sha  current_sha"
# approved_sha = the full sha named in the latest review whose body says APPROVE and not HOLD/CHANGES.
plan_() {
    jq -r '.[] | select(.autoMergeRequest == null) | . as $p
        | ([.reviews[] | select(.body | test("APPROVE") and (test("HOLD|CHANGES-NEEDED|REQUEST_CHANGES") | not))] | last // {body: ""}) as $r
        | [$p.number, ($r.body | capture("(?<s>[0-9a-f]{40})").s // ""), $p.headRefOid] | @tsv'
}

run_() {
    local t n
    t=$(mktemp)
    # shellcheck disable=SC2064
    trap "rm -f -- '${t:?}'" RETURN
    gh pr list --state open --limit 100 --json number,headRefOid,autoMergeRequest,reviews | plan_ >"$t"
    for n in $(select_ "$t"); do
        gh pr merge "$n" || echo "rearm: PR $n not armed (see above)" >&2
    done
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
    # run_ end to end: stub gh lists PR 1 (approved, head unchanged), 2 (head moved), 3 (already armed); only 1 is armed.
    mkdir -p "$d/bin"
    cat >"$d/bin/gh" <<STUB
#!/usr/bin/env bash
if [ "\$2" = list ]; then
  printf '[{"number":1,"headRefOid":"$a","autoMergeRequest":null,"reviews":[{"body":"seat APPROVE @$a"}]},{"number":2,"headRefOid":"$b","autoMergeRequest":null,"reviews":[{"body":"seat APPROVE @$a"}]},{"number":3,"headRefOid":"$a","autoMergeRequest":{"x":1},"reviews":[{"body":"APPROVE @$a"}]}]'
else
  echo "\$*" >>"$d/armed"
fi
STUB
    chmod +x "$d/bin/gh"
    PATH="$d/bin:$PATH" run_ >/dev/null 2>&1 || true
    [ "$(cat "$d/armed" 2>/dev/null)" = "pr merge 1" ] || { echo "SELF-TEST FAIL: run must arm exactly PR 1 via select_, armed: $(tr '\n' ',' <"$d/armed" 2>/dev/null)" >&2; bad=1; }
    [ "$bad" -eq 0 ] && echo "rearm_ready self-test: 3 cases ok"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    select) select_ "$2" ;;
    run) run_ ;;
    *) echo "usage: $0 select FILE | run | --self-test" >&2; exit 2 ;;
esac
