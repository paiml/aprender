#!/usr/bin/env bash
# rearm_ready.sh -- which ready PRs a gate fix may re-arm (contract advisory-gate-diff-v1, ADG-INV-005).
#
#   rearm_ready.sh select FILE     FILE = TSV "pr  approved_sha  current_sha"; prints PRs to arm
#   rearm_ready.sh run             list open un-armed PRs (one gh call), select, arm each with plain `gh pr merge`
#   rearm_ready.sh --self-test     case table, plus `run` against a stub gh (RED when select_ is not called)
# A PR whose head moved after its approval is NOT re-armed (a changed head needs a fresh review).
set -euo pipefail

select_() {
    awk -F'\t' 'NF >= 3 && $1 != "" && $2 ~ /^[0-9a-f]{40}$/ && $2 == $3 { print $1 "\t" $3 "\t" $4 }' "$1"
}

# stdin: `gh pr list --json id,number,headRefOid,autoMergeRequest,reviews`; stdout: TSV "pr  approved_sha  current_sha  node_id"
# approved_sha = the full sha named in the latest APPROVE review, unless a LATER review is negative
# (HOLD, CHANGES-NEEDED, REQUEST_CHANGES, state CHANGES_REQUESTED): a later negative review cancels the approval.
plan_() {
    jq -r '.[] | select(.autoMergeRequest == null) | . as $p
        | def neg: (.state == "CHANGES_REQUESTED") or (.body | test("HOLD|CHANGES-NEEDED|REQUEST_CHANGES"));
          def pos: ((neg | not) and (.body | test("APPROVE")));
          [.reviews | to_entries[]] as $r
        | ([$r[] | select(.value | pos) | .key] | last) as $ai
        | (if $ai == null or any($r[]; .key > $ai and (.value | neg)) then ""
           else ($r[] | select(.key == $ai) | .value.body | capture("(?<s>[0-9a-f]{40})").s // "") end) as $sha
        | [$p.number, $sha, $p.headRefOid, $p.id] | @tsv'
}

# Arms atomically against the reviewed head: enqueuePullRequest(expectedHeadOid) is refused by GitHub if the head moved
# after the select (the --match-head-commit guard). The queue owns the merge method, so none is passed. If the enqueue is
# refused for a non-head reason (checks still pending), fall back to the plain `gh pr merge <pr>` that #5794 mandates,
# but only after re-reading the head and finding it still equals the approved sha.
run_() {
    local t n sha id cur
    t=$(mktemp)
    # shellcheck disable=SC2064
    trap "rm -f -- '${t:?}'" RETURN
    gh pr list --state open --limit 100 --json id,number,headRefOid,autoMergeRequest,reviews | plan_ >"$t"
    while IFS=$'\t' read -r n sha id; do
        if gh api graphql -f query='mutation($id:ID!,$h:GitObjectID!){enqueuePullRequest(input:{pullRequestId:$id,expectedHeadOid:$h}){clientMutationId}}' \
            -f id="$id" -f h="$sha" >/dev/null 2>&1; then
            continue
        fi
        cur=$(gh pr view "$n" --json headRefOid --jq .headRefOid 2>/dev/null || true)
        if [ "$cur" = "$sha" ]; then
            gh pr merge "$n" || echo "rearm: PR $n not armed" >&2
        else
            echo "rearm: PR $n head moved since review; not armed" >&2
        fi
    done < <(select_ "$t")
}

self_test() {
    local d bad=0 a b
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    a=$(printf 'a%.0s' {1..40}); b=$(printf 'b%.0s' {1..40})
    printf '1\t%s\t%s\n2\t%s\t%s\n3\t\t%s\n' "$a" "$a" "$a" "$b" "$a" >"$d/t.tsv"
    [ "$(select_ "$d/t.tsv" | cut -f1 | tr '\n' ' ')" = "1 " ] || { echo "SELF-TEST FAIL: only PR 1 (head unchanged) may be re-armed" >&2; bad=1; }
    printf '' >"$d/e.tsv"
    [ -z "$(select_ "$d/e.tsv")" ] || { echo "SELF-TEST FAIL: empty input arms nothing" >&2; bad=1; }
    # run_ end to end with a stub gh. PRs: 1 approved+unchanged; 2 head moved; 3 already armed;
    # 4 APPROVE then a later CHANGES-NEEDED; 5 APPROVE then a later HOLD; 6 CHANGES-NEEDED then a fresh APPROVE.
    mkdir -p "$d/bin"
    cat >"$d/bin/gh" <<STUB
#!/usr/bin/env bash
if [ "\$1 \$2" = "pr list" ]; then
  printf '[{"id":"N1","number":1,"headRefOid":"$a","autoMergeRequest":null,"reviews":[{"body":"seat APPROVE @$a"}]},{"id":"N2","number":2,"headRefOid":"$b","autoMergeRequest":null,"reviews":[{"body":"seat APPROVE @$a"}]},{"id":"N3","number":3,"headRefOid":"$a","autoMergeRequest":{"x":1},"reviews":[{"body":"APPROVE @$a"}]},{"id":"N4","number":4,"headRefOid":"$a","autoMergeRequest":null,"reviews":[{"body":"seat APPROVE @$a"},{"body":"seat CHANGES-NEEDED @$a"}]},{"id":"N5","number":5,"headRefOid":"$a","autoMergeRequest":null,"reviews":[{"body":"seat APPROVE @$a"},{"body":"HOLD @$a"}]},{"id":"N6","number":6,"headRefOid":"$a","autoMergeRequest":null,"reviews":[{"body":"CHANGES-NEEDED @$a"},{"body":"seat APPROVE @$a"}]}]'
elif [ "\$1" = api ]; then
  echo "\$*" >>"$d/armed"
fi
STUB
    chmod +x "$d/bin/gh"
    PATH="$d/bin:$PATH" run_ >/dev/null 2>&1 || true
    got=$(grep 'expectedHeadOid:\$h' "$d/armed" 2>/dev/null | grep -o 'id=N[0-9]* -f h=[0-9a-f]*' | tr '\n' ',')
    [ "$got" = "id=N1 -f h=$a,id=N6 -f h=$a," ] || { echo "SELF-TEST FAIL: run must enqueue exactly PR 1 and 6, each with expectedHeadOid=its approved sha (a later HOLD/CHANGES-NEEDED cancels), got: $got" >&2; bad=1; }
    [ "$bad" -eq 0 ] && echo "rearm_ready self-test: 3 cases ok"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    select) select_ "$2" ;;
    run) run_ ;;
    *) echo "usage: $0 select FILE | run | --self-test" >&2; exit 2 ;;
esac
