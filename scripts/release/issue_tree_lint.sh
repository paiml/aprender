#!/usr/bin/env bash
# issue_tree_lint.sh — ISSUE-TREE-001: the nightly issue-graph lint. The cop runs it from a timer.
#
# The operator ruling is five rules over the OPEN issues of one repo:
#   R1 tree   every open issue has exactly 1 parent (GitHub allows at most one, so the check is "not 0"),
#             except a ROOT: an open issue labelled `epic` that has no parent. Depth <= 3 (root = 0).
#             0 orphans. An orphan is a non-root with no parent, a CLOSED parent, a parent in another repo,
#             or a parent chain that loops.
#   R2 size   open issues <= 100
#   R3 fanout open sub-issues per issue <= 5 (roots included)
#   R4 wip    in-progress issues per worker <= 2. "In progress" means a remote branch
#             `<worker>/<issue>-<slug>` (the fleet's branch convention; worker = 2-3 hex session id, so
#             type prefixes like fix/ feat/ docs/ fold/ perf/ are not workers) whose issue is open and whose
#             head commit is at most ACTIVE_DAYS (default 7) old. A worker is its session id.
#
# Usage:
#   issue_tree_lint.sh fetch <dir> [owner/repo]    # read-only: dir/issues.jsonl, dir/branches.jsonl, dir/now.txt
#   issue_tree_lint.sh check <dir>                 # verdict JSON on stdout
#   issue_tree_lint.sh file  <verdict.json> [owner/repo]  # RED -> opens ONE issue, or comments on the open one
#   issue_tree_lint.sh run   <dir> [--file] [owner/repo]  # fetch + check (+ file)
#   issue_tree_lint.sh self-test                   # the case table; every rule RED on its own fixture
#
# Exit codes of check/run: 0 GREEN, 10 RED, 20 NO-DATA (0 issues read), others = the script crashed.
# RED and NO-DATA are >= 10 so an alarm never looks like a shell or jq crash (exit 1/2/5).
#
# Env: MAX_OPEN=100 MAX_DEPTH=3 MAX_CHILDREN=5 MAX_WIP=2 ACTIVE_DAYS=7
#      ROOT_LABEL=epic   ISSUE_TITLE="ISSUE-TREE-001 RED: issue graph lint"
#      ISSUE_PARENT=<n>  the filed RED issue is linked under it, so the lint's own ticket is never an orphan
#      ISSUE_MILESTONE=backlog
set -euo pipefail

REPO_DEFAULT=paiml/aprender
MAX_OPEN=${MAX_OPEN:-100}
MAX_DEPTH=${MAX_DEPTH:-3}
MAX_CHILDREN=${MAX_CHILDREN:-5}
MAX_WIP=${MAX_WIP:-2}
ACTIVE_DAYS=${ACTIVE_DAYS:-7}
ROOT_LABEL=${ROOT_LABEL:-epic}
ISSUE_TITLE=${ISSUE_TITLE:-"ISSUE-TREE-001 RED: issue graph lint"}
ISSUE_MILESTONE=${ISSUE_MILESTONE:-backlog}

die() { printf 'issue_tree_lint: %s\n' "$1" >&2; exit 2; }

fetch() {
    local dir="$1" repo="${2:-$REPO_DEFAULT}"
    local owner="${repo%%/*}" name="${repo##*/}"
    mkdir -p "$dir"
    : > "$dir/issues.jsonl"
    local cursor="" page
    # shellcheck disable=SC2016
    local q='query($o:String!,$n:String!,$c:String){repository(owner:$o,name:$n){issues(states:OPEN,first:100,after:$c){
      pageInfo{hasNextPage endCursor}
      nodes{number title labels(first:30){nodes{name}}
            parent{number state repository{nameWithOwner}}}}}}'
    while :; do
        if [ -n "$cursor" ]; then
            page=$(gh api graphql -f query="$q" -F o="$owner" -F n="$name" -F c="$cursor")
        else
            page=$(gh api graphql -f query="$q" -F o="$owner" -F n="$name")
        fi
        printf '%s' "$page" | jq -c --arg repo "$repo" '.data.repository.issues.nodes[]
          | {n: .number, title, labels: [.labels.nodes[].name],
             parent: (if .parent then {n: .parent.number, state: .parent.state,
                                       same_repo: (.parent.repository.nameWithOwner == $repo)} else null end)}' \
          >> "$dir/issues.jsonl"
        [ "$(printf '%s' "$page" | jq -r '.data.repository.issues.pageInfo.hasNextPage')" = true ] || break
        cursor=$(printf '%s' "$page" | jq -r '.data.repository.issues.pageInfo.endCursor')
    done
    : > "$dir/branches.jsonl"
    cursor=""
    # shellcheck disable=SC2016
    local qb='query($o:String!,$n:String!,$c:String){repository(owner:$o,name:$n){refs(refPrefix:"refs/heads/",first:100,after:$c){
      pageInfo{hasNextPage endCursor}
      nodes{name target{... on Commit{committedDate}}}}}}'
    while :; do
        if [ -n "$cursor" ]; then
            page=$(gh api graphql -f query="$qb" -F o="$owner" -F n="$name" -F c="$cursor")
        else
            page=$(gh api graphql -f query="$qb" -F o="$owner" -F n="$name")
        fi
        printf '%s' "$page" | jq -c '.data.repository.refs.nodes[] | {name, date: .target.committedDate}' \
          >> "$dir/branches.jsonl"
        [ "$(printf '%s' "$page" | jq -r '.data.repository.refs.pageInfo.hasNextPage')" = true ] || break
        cursor=$(printf '%s' "$page" | jq -r '.data.repository.refs.pageInfo.endCursor')
    done
    date -u +%FT%TZ > "$dir/now.txt"
    printf '%s\n' "$repo" > "$dir/repo.txt"
    printf 'fetched %s issues, %s branches into %s\n' \
        "$(wc -l < "$dir/issues.jsonl")" "$(wc -l < "$dir/branches.jsonl")" "$dir" >&2
}

# Pure: reads dir/{issues,branches}.jsonl + dir/now.txt, prints the verdict. Exit 0/10/20.
check() {
    local dir="$1"
    [ -f "$dir/issues.jsonl" ] && [ -f "$dir/branches.jsonl" ] && [ -f "$dir/now.txt" ] \
        || die "check: $dir needs issues.jsonl, branches.jsonl, now.txt (run fetch)"
    local v
    v=$(jq -n -c \
        --slurpfile issues "$dir/issues.jsonl" --slurpfile branches "$dir/branches.jsonl" \
        --arg now "$(cat "$dir/now.txt")" --arg root_label "$ROOT_LABEL" \
        --argjson max_open "$MAX_OPEN" --argjson max_depth "$MAX_DEPTH" \
        --argjson max_children "$MAX_CHILDREN" --argjson max_wip "$MAX_WIP" \
        --argjson active_days "$ACTIVE_DAYS" '
      ($issues | map({key: (.n|tostring), value: .}) | from_entries) as $by
      | def is_root: (.labels | index($root_label)) != null and .parent == null;
        # the in-repo OPEN parent of an issue, or null
        def up: if .parent != null and .parent.same_repo and .parent.state == "OPEN"
                   and $by[.parent.n|tostring] != null then $by[.parent.n|tostring] else null end;
        # depth to a root; -1 = the chain ends at a non-root (orphan) or loops
        def depth: [ limit(65; recurse(up; . != null)) ] as $chain
                   | if ($chain | length) > 64 then -1
                     elif ($chain | last | is_root) then ($chain | length) - 1 else -1 end;
      ($issues | map(select(is_root | not) | . as $i
          | if .parent == null then {n, reason: "no-parent"}
            elif (.parent.same_repo | not) then {n, reason: "parent-in-other-repo", parent: .parent.n}
            elif .parent.state != "OPEN" then {n, reason: "parent-closed", parent: .parent.n}
            elif $by[.parent.n|tostring] == null then {n, reason: "parent-not-read", parent: .parent.n}
            elif ($i | depth) < 0 then {n, reason: "chain-reaches-no-root", parent: .parent.n}
            else empty end)) as $orphans
      | ($issues | map({n, depth: depth}) | map(select(.depth > $max_depth))) as $deep
      | ($issues | map(select(up != null) | .parent.n) | group_by(.)
          | map({n: .[0], children: length}) | map(select(.children > $max_children))) as $fat
      | ($now | fromdateiso8601) as $t
      | ($branches | map(select(.date != null)
          | (.name | capture("^(?<w>[0-9a-f]{2,3})/(?<n>[0-9]+)-")) as $m
          | select($m != null)
          | select(($t - (.date | fromdateiso8601)) <= $active_days * 86400)
          | select($by[$m.n] != null)
          | {worker: $m.w, n: ($m.n | tonumber), branch: .name})) as $wip_rows
      | ($wip_rows | group_by(.worker)
          | map({worker: .[0].worker, issues: (map(.n) | unique)})
          | map(select((.issues | length) > $max_wip))) as $busy
      | ($issues | length) as $open
      | {
          rules: {
            R1_orphans:   {ok: (($orphans | length) == 0), count: ($orphans | length)},
            R1_depth:     {ok: (($deep | length) == 0), count: ($deep | length), max: $max_depth},
            R2_open:      {ok: ($open <= $max_open), count: $open, max: $max_open},
            R3_fanout:    {ok: (($fat | length) == 0), count: ($fat | length), max: $max_children},
            R4_wip:       {ok: (($busy | length) == 0), count: ($busy | length), max: $max_wip}
          },
          roots: ($issues | map(select(is_root) | .n)),
          detail: {orphans: $orphans, deep: $deep, fanout: $fat, wip: $busy},
          wip_rows: ($wip_rows | length),
          now: $now,
          verdict: (if $open == 0 then "NO-DATA"
                    elif ([$orphans, $deep, $fat, $busy] | map(length) | add) == 0 and $open <= $max_open
                    then "GREEN" else "RED" end)
        }')
    printf '%s\n' "$v"
    case "$(printf '%s' "$v" | jq -r .verdict)" in
        GREEN) return 0 ;;
        RED) return 10 ;;
        *) return 20 ;;
    esac
}

summary() {
    jq -r '"ISSUE-TREE-001 \(.verdict): open \(.rules.R2_open.count)/\(.rules.R2_open.max), orphans \(.rules.R1_orphans.count), depth>\(.rules.R1_depth.max) \(.rules.R1_depth.count), fanout>\(.rules.R3_fanout.max) \(.rules.R3_fanout.count), wip>\(.rules.R4_wip.max) \(.rules.R4_wip.count) workers, roots \(.roots | length)"'
}

# RED -> exactly ONE open issue titled $ISSUE_TITLE: comment on it if it exists, else open it.
file_issue() {
    local verdict="$1" repo="${2:-$REPO_DEFAULT}"
    [ "$(jq -r .verdict "$verdict")" = RED ] || { printf 'not RED, nothing filed\n' >&2; return 0; }
    local body existing
    body=$(printf '%s\n\n```json\n%s\n```\n\nFiled by `scripts/release/issue_tree_lint.sh` (ISSUE-TREE-001).\n' \
        "$(summary < "$verdict")" "$(jq '{rules, detail}' "$verdict")")
    existing=$(gh issue list -R "$repo" --state open --search "\"$ISSUE_TITLE\" in:title" --json number,title \
        | jq -r --arg t "$ISSUE_TITLE" 'map(select(.title == $t)) | .[0].number // empty')
    if [ -n "$existing" ]; then
        gh issue comment "$existing" -R "$repo" --body "$body" >/dev/null
        printf 'commented on #%s\n' "$existing"
        return 0
    fi
    local url n
    url=$(gh issue create -R "$repo" --title "$ISSUE_TITLE" --milestone "$ISSUE_MILESTONE" --body "$body")
    n=${url##*/}
    if [ -n "${ISSUE_PARENT:-}" ]; then
        local id
        id=$(gh api "repos/$repo/issues/$n" --jq .id)
        gh api -X POST "repos/$repo/issues/$ISSUE_PARENT/sub_issues" -F sub_issue_id="$id" >/dev/null
    fi
    printf 'opened #%s\n' "$n"
}

run() {
    local dir="$1"; shift
    local do_file=0 repo="$REPO_DEFAULT"
    for a in "$@"; do
        case "$a" in
            --file) do_file=1 ;;
            */*) repo="$a" ;;
            *) die "run: unknown argument $a" ;;
        esac
    done
    fetch "$dir" "$repo"
    local rc=0
    check "$dir" > "$dir/verdict.json" || rc=$?
    summary < "$dir/verdict.json"
    if [ "$rc" -eq 10 ] && [ "$do_file" -eq 1 ]; then file_issue "$dir/verdict.json" "$repo"; fi
    return "$rc"
}

# ---------------------------------------------------------------- self-test (case table)

SELF_FAILS=0
expect() {
    local label="$1" want="$2" got="$3"
    if [ "$want" = "$got" ]; then
        printf 'ok    %s\n' "$label"
    else
        printf 'FAIL  %s: want %s got %s\n' "$label" "$want" "$got"
        SELF_FAILS=$((SELF_FAILS + 1))
    fi
}

# issue <n> <labels-json> <parent-n|-> [state=OPEN] [same_repo=true]
issue() {
    local p=null
    if [ "$3" != "-" ]; then p=$(printf '{"n":%s,"state":"%s","same_repo":%s}' "$3" "${4:-OPEN}" "${5:-true}"); fi
    printf '{"n":%s,"title":"t%s","labels":%s,"parent":%s}\n' "$1" "$1" "$2" "$p"
}

# A GREEN base: root 1 (epic) -> 2 -> 3 -> 4 (depth 3), and 1 -> 5.
base() {
    local d="$1"
    mkdir -p "$d"
    {
        issue 1 '["epic"]' -
        issue 2 '[]' 1
        issue 3 '[]' 2
        issue 4 '[]' 3
        issue 5 '[]' 1
    } > "$d/issues.jsonl"
    printf '{"name":"89/2-work","date":"2026-09-27T10:00:00Z"}\n{"name":"89/3-more","date":"2026-09-27T10:00:00Z"}\n' \
        > "$d/branches.jsonl"
    printf '2026-09-27T12:00:00Z\n' > "$d/now.txt"
}

verdict_of() { local rc=0; check "$1" > "$1/v.json" || rc=$?; printf '%s/%s' "$(jq -r .verdict "$1/v.json")" "$rc"; }
rule_of() { jq -r ".rules.$2.ok" "$1/v.json"; }

self_test() {
    local T
    T=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf '${T:?}'" EXIT

    base "$T/green"
    expect "GREEN base (depth 3, fanout 2, wip 2) exits 0" "GREEN/0" "$(verdict_of "$T/green")"

    base "$T/orphan"; issue 6 '[]' - >> "$T/orphan/issues.jsonl"
    expect "R1 an issue with no parent is RED (exit 10)" "RED/10" "$(verdict_of "$T/orphan")"
    expect "R1 names the orphan" "6 no-parent" "$(jq -r '.detail.orphans[0] | "\(.n) \(.reason)"' "$T/orphan/v.json")"

    base "$T/closedp"; issue 6 '[]' 99 CLOSED >> "$T/closedp/issues.jsonl"
    expect "R1 a closed parent is an orphan" "parent-closed" "$(verdict_of "$T/closedp" >/dev/null; jq -r '.detail.orphans[0].reason' "$T/closedp/v.json")"

    base "$T/foreign"; issue 6 '[]' 7 OPEN false >> "$T/foreign/issues.jsonl"
    expect "R1 a parent in another repo is an orphan" "parent-in-other-repo" "$(verdict_of "$T/foreign" >/dev/null; jq -r '.detail.orphans[0].reason' "$T/foreign/v.json")"

    base "$T/loop"; { issue 6 '[]' 7; issue 7 '[]' 6; } >> "$T/loop/issues.jsonl"
    expect "R1 a parent loop is RED, not a hang" "RED/10" "$(verdict_of "$T/loop")"
    expect "R1 both loop members named" "6,7" "$(jq -r '[.detail.orphans[].n] | sort | map(tostring) | join(",")' "$T/loop/v.json")"

    base "$T/nonroot"; issue 6 '[]' - >> "$T/nonroot/issues.jsonl"; issue 7 '[]' 6 >> "$T/nonroot/issues.jsonl"
    expect "R1 a chain to a label-less top is an orphan chain" "6,7" "$(verdict_of "$T/nonroot" >/dev/null; jq -r '[.detail.orphans[].n] | sort | map(tostring) | join(",")' "$T/nonroot/v.json")"

    base "$T/epicchild"; issue 6 '["epic"]' 1 >> "$T/epicchild/issues.jsonl"
    expect "an epic WITH a parent is a node, not a root (still GREEN)" "GREEN/0" "$(verdict_of "$T/epicchild")"

    base "$T/deep"; issue 6 '[]' 4 >> "$T/deep/issues.jsonl"
    expect "R1 depth 4 is RED" "RED/10" "$(verdict_of "$T/deep")"
    expect "R1 depth rule is the one that fired" "false" "$(rule_of "$T/deep" R1_depth)"

    base "$T/big"; for n in $(seq 6 101); do issue "$n" '[]' 5 CLOSED >> "$T/big/issues.jsonl"; done
    expect "R2 101 open issues is RED" "false" "$(verdict_of "$T/big" >/dev/null; rule_of "$T/big" R2_open)"
    base "$T/hundred"
    for n in $(seq 6 100); do issue "$n" '["epic"]' - >> "$T/hundred/issues.jsonl"; done
    expect "R2 exactly 100 open issues is GREEN" "GREEN/0" "$(verdict_of "$T/hundred")"

    base "$T/fan"; for n in 6 7 8 9; do issue "$n" '[]' 1 >> "$T/fan/issues.jsonl"; done
    expect "R3 six children is RED" "false" "$(verdict_of "$T/fan" >/dev/null; rule_of "$T/fan" R3_fanout)"
    base "$T/fan5"; for n in 6 7 8; do issue "$n" '[]' 1 >> "$T/fan5/issues.jsonl"; done
    expect "R3 exactly five children is GREEN" "GREEN/0" "$(verdict_of "$T/fan5")"

    base "$T/wip"; printf '{"name":"89/4-third","date":"2026-09-27T09:00:00Z"}\n' >> "$T/wip/branches.jsonl"
    expect "R4 three in-progress issues for one worker is RED" "false" "$(verdict_of "$T/wip" >/dev/null; rule_of "$T/wip" R4_wip)"
    base "$T/wipstale"; printf '{"name":"89/4-third","date":"2026-09-01T09:00:00Z"}\n' >> "$T/wipstale/branches.jsonl"
    expect "R4 a branch older than ACTIVE_DAYS is not in progress" "GREEN/0" "$(verdict_of "$T/wipstale")"
    base "$T/wipclosed"; printf '{"name":"89/77-closed","date":"2026-09-27T09:00:00Z"}\n' >> "$T/wipclosed/branches.jsonl"
    expect "R4 a branch whose issue is closed is not in progress" "GREEN/0" "$(verdict_of "$T/wipclosed")"
    base "$T/wiptwo"; printf '{"name":"89/2-again","date":"2026-09-27T09:00:00Z"}\n{"name":"release/0.71","date":"2026-09-27T09:00:00Z"}\n' >> "$T/wiptwo/branches.jsonl"
    expect "R4 two branches on one issue count once; non-convention branches ignored" "GREEN/0" "$(verdict_of "$T/wiptwo")"

    base "$T/wiptype"; printf '{"name":"fix/4-a","date":"2026-09-27T09:00:00Z"}\n{"name":"fix/5-b","date":"2026-09-27T09:00:00Z"}\n{"name":"feat/2-c","date":"2026-09-27T09:00:00Z"}\n' >> "$T/wiptype/branches.jsonl"
    expect "R4 type prefixes (fix/ feat/) are not workers" "GREEN/0" "$(verdict_of "$T/wiptype")"

    mkdir -p "$T/empty"; : > "$T/empty/issues.jsonl"; : > "$T/empty/branches.jsonl"; printf '2026-09-27T12:00:00Z\n' > "$T/empty/now.txt"
    expect "0 issues read is NO-DATA (exit 20), never GREEN" "NO-DATA/20" "$(verdict_of "$T/empty")"

    local rc=0
    (check "$T/missing") >/dev/null 2>&1 || rc=$?
    expect "a dir without fetch output crashes (exit 2), not GREEN" "2" "$rc"

    expect "summary line" "ISSUE-TREE-001 RED: open 6/100, orphans 1, depth>3 0, fanout>5 0, wip>2 0 workers, roots 1" \
        "$(summary < "$T/orphan/v.json")"

    if [ "$SELF_FAILS" -eq 0 ]; then printf 'PASS  issue_tree_lint self-test\n'; else printf 'FAIL  %s case(s)\n' "$SELF_FAILS"; return 1; fi
}

main() {
    local cmd="${1:-}"
    [ -n "$cmd" ] || die "usage: fetch|check|file|run|self-test (see header)"
    shift
    case "$cmd" in
        fetch) [ $# -ge 1 ] || die "fetch <dir> [owner/repo]"; fetch "$@" ;;
        check) [ $# -eq 1 ] || die "check <dir>"; check "$1" ;;
        file) [ $# -ge 1 ] || die "file <verdict.json> [owner/repo]"; file_issue "$@" ;;
        run) [ $# -ge 1 ] || die "run <dir> [--file] [owner/repo]"; run "$@" ;;
        self-test) self_test ;;
        *) die "unknown command $cmd" ;;
    esac
}

main "$@"
