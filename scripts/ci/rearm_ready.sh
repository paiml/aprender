#!/usr/bin/env bash
# rearm_ready.sh -- which ready PRs a gate fix may re-arm (contract advisory-gate-diff-v1, ADG-INV-005).
#
#   rearm_ready.sh select FILE     FILE = TSV "pr  approved_sha  current_sha"; prints PRs to arm
#   rearm_ready.sh run             list open un-armed PRs (one gh call), select, enqueue each at its reviewed head (enqueuePullRequest with expectedHeadOid; no plain `gh pr merge`)
#   rearm_ready.sh --self-test     case table, plus `run` against a stub gh (RED when select_ is not called)
# A PR whose head moved after its approval is NOT re-armed (a changed head needs a fresh review).
set -euo pipefail

select_() {
    awk -F'\t' 'NF >= 3 && $1 != "" && $2 ~ /^[0-9a-f]{40}$/ && $2 == $3 { print $1 "\t" $3 "\t" $4 }' "$1"
}

# stdin: `gh pr list --json id,number,author,headRefOid,autoMergeRequest,reviews`; stdout: TSV "pr  approved_sha  current_sha  node_id"
# approved_sha = commit.oid of the latest anchored APPROVE review by an allowlisted reviewer (REARM_REVIEWERS, never the PR author);
# ANY later review that is not such an approval (negative or a plain comment) cancels it.
plan_() {
    jq -r --arg ok "${REARM_REVIEWERS:-}" '($ok | split(" ") | map(select(. != ""))) as $allow
        | .[] | select(.autoMergeRequest == null) | . as $p
        | def neg: (.state == "CHANGES_REQUESTED") or (.body | test("HOLD|CHANGES-NEEDED|REQUEST_CHANGES"));
          def tok: (.body | test("(^|\\n|: |- )APPROVE([^A-Za-z]|$)"));
          def pos: (neg | not) and tok and (.author.login as $l | ($allow | index($l)) != null and $l != $p.author.login);
          [.reviews | to_entries[]] as $r
        | ([$r[] | select(.value | pos) | .key] | last) as $ai
        | (if $ai == null or any($r[]; .key > $ai and (.value | pos | not)) then ""
           else ($r[] | select(.key == $ai) | .value.commit.oid // "") end) as $sha
        | [$p.number, $sha, $p.headRefOid, $p.id] | @tsv'
}

# Arms atomically against the reviewed head: enqueuePullRequest(expectedHeadOid) is refused by GitHub if the head moved
# after the select. The queue owns the merge method, so none is passed. There is NO `gh pr merge` fallback: #5794 bans
# every flag on it (so no --match-head-commit), and a plain merge after a re-read of the head is a check-then-act race that
# could arm a head nobody reviewed. A refused enqueue (head moved, or checks still pending) is left for the next run.
run_() {
    local t n sha id
    t=$(mktemp)
    # shellcheck disable=SC2064
    trap "rm -f -- '${t:?}'" RETURN
    gh pr list --state open --limit 100 --json id,number,author,headRefOid,autoMergeRequest,reviews | plan_ >"$t"
    while IFS=$'\t' read -r n sha id; do
        if gh api graphql -f query='mutation($id:ID!,$h:GitObjectID!){enqueuePullRequest(input:{pullRequestId:$id,expectedHeadOid:$h}){clientMutationId}}' \
            -f id="$id" -f h="$sha" >/dev/null 2>&1; then
            continue
        fi
        echo "rearm: PR $n enqueue refused at the reviewed head; not armed" >&2
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
    # rv LOGIN BODY OID -> one review object; pr N HEAD AUTOMERGE REVIEWS -> one PR object
    rv() { printf '{"author":{"login":"%s"},"state":"COMMENTED","body":"%s","commit":{"oid":"%s"}}' "$1" "$2" "$3"; }
    pr() { printf '{"id":"N%s","number":%s,"author":{"login":"au"},"headRefOid":"%s","autoMergeRequest":%s,"reviews":[%s]}' "$1" "$1" "$2" "$3" "$4"; }
    {
        printf '['
        pr 1 "$a" null "$(rv s1 'seat 1: APPROVE' "$a")"; printf ,
        pr 2 "$b" null "$(rv s1 'seat 1: APPROVE' "$a")"; printf ,
        pr 3 "$a" '{"x":1}' "$(rv s1 'seat 1: APPROVE' "$a")"; printf ,
        pr 4 "$a" null "$(rv s1 'seat 1: APPROVE' "$a"),$(rv s2 'seat 2: CHANGES-NEEDED' "$a")"; printf ,
        pr 5 "$a" null "$(rv s1 'seat 1: APPROVE' "$a"),$(rv s2 'HOLD' "$a")"; printf ,
        pr 6 "$a" null "$(rv s2 'CHANGES-NEEDED' "$a"),$(rv s1 'seat 1: APPROVE' "$a")"; printf ,
        pr 7 "$a" null "$(rv s1 'seat 1: NOT APPROVED' "$a")"; printf ,
        pr 8 "$a" null "$(rv s1 'seat 1: APPROVE' "$a"),$(rv s2 'just a comment' "$a")"; printf ,
        pr 9 "$a" null "$(rv rando 'seat 1: APPROVE' "$a")"; printf ,
        pr 10 "$a" null "$(rv au 'seat 1: APPROVE' "$a")"; printf ,
        pr 11 "$a" null "$(rv s1 'seat 1: APPROVE' "$b")"
        printf ']'
    } >"$d/prs.json"
    cat >"$d/bin/gh" <<STUB
#!/usr/bin/env bash
if [ "\$1 \$2" = "pr list" ]; then cat "$d/prs.json"
elif [ "\$1" = api ]; then echo "\$*" >>"$d/armed"; [ -z "\${STUB_REFUSE:-}" ]
else echo "\$*" >>"$d/other"; exit 1
fi
STUB
    chmod +x "$d/bin/gh"
    REARM_REVIEWERS="s1 s2 au" PATH="$d/bin:$PATH" run_ >/dev/null 2>&1 || true
    got=$(grep 'expectedHeadOid:\$h' "$d/armed" 2>/dev/null | grep -o 'id=N[0-9]* -f h=[0-9a-f]*' | tr '\n' ',')
    # 1 ok; 2 head moved; 3 armed; 4/5 later negative; 6 fresh approve after negative -> arms; 7 "NOT APPROVED" is no approval;
    # 8 any later non-approval cancels; 9 reviewer off the allowlist; 10 author self-approval; 11 commit.oid (b) != head (a), so not armed.
    [ "$got" = "id=N1 -f h=$a,id=N6 -f h=$a," ] || { echo "SELF-TEST FAIL: run must enqueue exactly PR 1 and 6 with expectedHeadOid=commit.oid, got: $got" >&2; bad=1; }
    rm -f "${d:?}/armed" "${d:?}/other"
    REARM_REVIEWERS="s1 s2 au" STUB_REFUSE=1 PATH="$d/bin:$PATH" run_ >/dev/null 2>&1 || true
    if [ -e "$d/other" ]; then
        echo "SELF-TEST FAIL: a refused enqueue must not fall back to any other gh call (the head pin is the mutation only)" >&2; bad=1
    fi
    [ "$bad" -eq 0 ] && echo "rearm_ready self-test: 4 cases ok"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    select) select_ "$2" ;;
    run) run_ ;;
    *) echo "usage: $0 select FILE | run | --self-test" >&2; exit 2 ;;
esac
