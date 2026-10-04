#!/usr/bin/env bash
# issue_tree_lint.sh — ISSUE-TREE-001: the nightly issue-graph lint. The cop runs it from a timer.
#
# The operator ruling is five rules over the OPEN issues of one repo:
#   R1 tree   every open issue has exactly 1 parent (GitHub allows at most one, so the check is "not 0"),
#             except a ROOT: an open issue labelled `epic` that has no parent. Depth <= 3 levels:
#             epic (1) -> ticket (2) -> sub-ticket (3). A sub-sub-ticket (4) is RED: the ruling allows none.
#             0 orphans. An orphan is a non-root with no parent, a CLOSED parent, a parent in another repo,
#             or a parent chain that loops.
#   R2 size   open TICKETS <= 100. A ticket is a direct child of a root. Epics and sub-tickets are not
#             counted; the open-issue and root totals are printed, not capped (0 open issues = NO-DATA).
#   R3 fanout open children per non-root issue <= 5, i.e. sub-tickets per ticket. A root's tickets are
#             not capped here: R2 counts them.
#   R4 wip    in-progress TICKETS per worker <= 2. A branch on a sub-ticket counts as its ticket; a branch
#             on an epic (a root) is not a ticket and is not counted; a branch on any other issue counts
#             as that issue. "In progress" means a remote branch
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
# Env: MAX_TICKETS=100 MAX_DEPTH=3 MAX_CHILDREN=5 MAX_WIP=2 ACTIVE_DAYS=7
#      ROOT_LABEL=epic   ISSUE_TITLE="ISSUE-TREE-001 RED: issue graph lint"
#      ISSUE_PARENT=<n>  the filed RED issue is linked under it, so the lint's own ticket is never an orphan
#      ISSUE_MILESTONE=backlog
set -euo pipefail

REPO_DEFAULT=paiml/aprender
MAX_TICKETS=${MAX_TICKETS:-100}
MAX_DEPTH=${MAX_DEPTH:-3}
MAX_CHILDREN=${MAX_CHILDREN:-5}
MAX_WIP=${MAX_WIP:-2}
ACTIVE_DAYS=${ACTIVE_DAYS:-7}
ROOT_LABEL=${ROOT_LABEL:-epic}
ISSUE_TITLE=${ISSUE_TITLE:-"ISSUE-TREE-001 RED: issue graph lint"}
ISSUE_MILESTONE=${ISSUE_MILESTONE:-backlog}

die() { printf 'issue_tree_lint: %s\n' "$1" >&2; exit 2; }

# The cap counts tickets now: a caller still setting the old knob would lose its cap without a word.
if [ -n "${MAX_OPEN+set}" ]; then die "MAX_OPEN is gone: the cap is MAX_TICKETS and counts tickets only (ISSUE-TREE-001)"; fi

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
        --argjson max_tickets "$MAX_TICKETS" --argjson max_depth "$MAX_DEPTH" \
        --argjson max_children "$MAX_CHILDREN" --argjson max_wip "$MAX_WIP" \
        --argjson active_days "$ACTIVE_DAYS" '
      ($issues | map({key: (.n|tostring), value: .}) | from_entries) as $by
      | def is_root: (.labels | index($root_label)) != null and .parent == null;
        # the in-repo OPEN parent of an issue, or null
        def up: if .parent != null and .parent.same_repo and .parent.state == "OPEN"
                   and $by[.parent.n|tostring] != null then $by[.parent.n|tostring] else null end;
        # level under a root: the root is 1, its tickets 2, their sub-tickets 3;
        # -1 = the chain ends at a non-root (orphan) or loops
        def level: [ limit(65; recurse(up; . != null)) ] as $chain
                   | if ($chain | length) > 64 then -1
                     elif ($chain | last | is_root) then ($chain | length) else -1 end;
        # the ticket an issue belongs to: a sub-ticket (or deeper) rolls up to its level-2 ancestor
        def ticket_of: if level > 2 then (up | ticket_of) else . end;
      ($issues | map(select(is_root | not) | . as $i
          | if .parent == null then {n, reason: "no-parent"}
            elif (.parent.same_repo | not) then {n, reason: "parent-in-other-repo", parent: .parent.n}
            elif .parent.state != "OPEN" then {n, reason: "parent-closed", parent: .parent.n}
            elif $by[.parent.n|tostring] == null then {n, reason: "parent-not-read", parent: .parent.n}
            elif ($i | level) < 0 then {n, reason: "chain-reaches-no-root", parent: .parent.n}
            else empty end)) as $orphans
      | ($issues | map({n, level: level}) | map(select(.level > $max_depth))) as $deep
      | ($issues | map(select(level == 2)) | length) as $tickets
      | ($tickets <= $max_tickets) as $tickets_ok
      | ($issues | map(select(up != null and (up | is_root | not)) | .parent.n) | group_by(.)
          | map({n: .[0], children: length}) | map(select(.children > $max_children))) as $fat
      | ($now | fromdateiso8601) as $t
      | ($branches | map(select(.date != null)
          | (.name | capture("^(?<w>[0-9a-f]{2,3})/(?<n>[0-9]+)-")) as $m
          | select($m != null)
          | select(($t - (.date | fromdateiso8601)) <= $active_days * 86400)
          | select($by[$m.n] != null)
          | select($by[$m.n] | is_root | not)
          | {worker: $m.w, n: ($m.n | tonumber), ticket: ($by[$m.n] | ticket_of | .n), branch: .name})) as $wip_rows
      | ($wip_rows | group_by(.worker)
          | map({worker: .[0].worker, tickets: (map(.ticket) | unique)})
          | map(select((.tickets | length) > $max_wip))) as $busy
      | ($issues | length) as $open
      | {
          rules: {
            R1_orphans:   {ok: (($orphans | length) == 0), count: ($orphans | length)},
            R1_depth:     {ok: (($deep | length) == 0), count: ($deep | length), max: $max_depth},
            R2_tickets:   {ok: $tickets_ok, count: $tickets, max: $max_tickets},
            R3_fanout:    {ok: (($fat | length) == 0), count: ($fat | length), max: $max_children},
            R4_wip:       {ok: (($busy | length) == 0), count: ($busy | length), max: $max_wip}
          },
          open: $open,
          roots: ($issues | map(select(is_root) | .n)),
          detail: {orphans: $orphans, deep: $deep, fanout: $fat, wip: $busy},
          wip_rows: ($wip_rows | length),
          now: $now,
          verdict: (if $open == 0 then "NO-DATA"
                    elif ([$orphans, $deep, $fat, $busy] | map(length) | add) == 0 and $tickets_ok
                    then "GREEN" else "RED" end)
        }') || die "check: jq failed on $dir (a malformed fetch row or knob value)"
    printf '%s\n' "$v"
    case "$(printf '%s' "$v" | jq -r .verdict)" in
        GREEN) return 0 ;;
        RED) return 10 ;;
        *) return 20 ;;
    esac
}

summary() {
    jq -r '"ISSUE-TREE-001 \(.verdict): tickets \(.rules.R2_tickets.count)/\(.rules.R2_tickets.max), orphans \(.rules.R1_orphans.count), depth>\(.rules.R1_depth.max) \(.rules.R1_depth.count), fanout>\(.rules.R3_fanout.max) \(.rules.R3_fanout.count), wip>\(.rules.R4_wip.max) \(.rules.R4_wip.count) workers; not capped: open \(.open), roots \(.roots | length)"'
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

# branch <dir> <name> [date]: one remote-branch row (default date: 3 h before the fixtures' now)
branch() { printf '{"name":"%s","date":"%s"}\n' "$2" "${3:-2026-09-27T09:00:00Z}" >> "$1/branches.jsonl"; }
# branch_undated <dir> <name>: a row with a null date (fetch writes one when the ref's target is not a commit)
branch_undated() { jq -n -c --arg name "$2" '{name: $name, date: null}' >> "$1/branches.jsonl"; }
EPIC_LABELS='["epic"]'
# issue_range <dir> <first> <last> <labels-json> <parent-n>: issues first..last, one parent
issue_range() { local n first="$2" last="$3"; for ((n = first; n <= last; n++)); do issue "$n" "$4" "$5" >> "$1/issues.jsonl"; done; }
# ticket_subs <dir> <first> <last>: tickets first..last under root 1, five sub-tickets each, numbered from 200
ticket_subs() {
    local t first="$2" last="$3"
    for ((t = first; t <= last; t++)); do
        issue "$t" '[]' 1 >> "$1/issues.jsonl"
        issue_range "$1" "$((200 + (t - first) * 5))" "$((204 + (t - first) * 5))" '[]' "$t"
    done
}

# A GREEN base: root 1 (epic, level 1) -> tickets 2 and 5 (level 2); ticket 2 -> sub-tickets 3 and 4 (level 3).
# Worker 89 has branches on tickets 2 and 5 (wip 2).
base() {
    local d="$1"
    mkdir -p "$d"
    {
        issue 1 "$EPIC_LABELS" -
        issue 2 '[]' 1
        issue 3 '[]' 2
        issue 4 '[]' 2
        issue 5 '[]' 1
    } > "$d/issues.jsonl"
    : > "$d/branches.jsonl"
    branch "$d" 89/2-work 2026-09-27T10:00:00Z
    branch "$d" 89/5-more 2026-09-27T10:00:00Z
    printf '2026-09-27T12:00:00Z\n' > "$d/now.txt"
}

# A third ticket (6, under root 1): the R4 fixtures need three distinct tickets to be RED.
third_ticket() { issue 6 '[]' 1 >> "$1/issues.jsonl"; }
# The exit code of `check` run as a fresh process, so a crash (exit 2) cannot end the self-test.
check_rc() { local rc=0; bash "$0" check "$1" >/dev/null 2>&1 || rc=$?; printf '%s' "$rc"; }
# The exit code of a fresh `check` run with one env knob set: proves the knob reaches the check.
knob_rc() { local -x "${1:?}"; check_rc "$2"; }

verdict_of() { local rc=0; check "$1" > "$1/v.json" || rc=$?; printf '%s/%s' "$(jq -r .verdict "$1/v.json")" "$rc"; }
rule_of() { jq -r ".rules.$2.ok" "$1/v.json"; }

self_test() {
    local T
    T=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf '${T:?}'" EXIT

    base "$T/green"
    expect "GREEN base (3 levels, 2 tickets, a ticket with 2 sub-tickets, wip 2) exits 0" "GREEN/0" "$(verdict_of "$T/green")"
    expect "R2 counts tickets only (base: 1 epic, 2 tickets, 2 sub-tickets)" "2" "$(jq -r .rules.R2_tickets.count "$T/green/v.json")"
    expect "env MAX_TICKETS=1 reaches the check (2 tickets is RED)" "10" "$(knob_rc MAX_TICKETS=1 "$T/green")"
    expect "env MAX_DEPTH=2 reaches the check (a sub-ticket is RED)" "10" "$(knob_rc MAX_DEPTH=2 "$T/green")"
    expect "env MAX_CHILDREN=1 reaches the check (2 sub-tickets is RED)" "10" "$(knob_rc MAX_CHILDREN=1 "$T/green")"
    expect "env MAX_WIP=1 reaches the check (2 tickets in progress is RED)" "10" "$(knob_rc MAX_WIP=1 "$T/green")"
    expect "env ROOT_LABEL=zzz reaches the check (no root, so every issue is an orphan: RED)" "10" "$(knob_rc ROOT_LABEL=zzz "$T/green")"
    expect "a non-numeric MAX_TICKETS crashes (exit 2), never NO-DATA" "2" "$(knob_rc MAX_TICKETS=abc "$T/green")"
    expect "the old MAX_OPEN knob stops the run (exit 2), never ignored" "2" "$(knob_rc MAX_OPEN=1 "$T/green")"
    expect "an empty MAX_OPEN stops the run too (the knob is gone, not defaulted)" "2" "$(knob_rc MAX_OPEN= "$T/green")"

    base "$T/orphan"; issue 6 '[]' - >> "$T/orphan/issues.jsonl"
    expect "R1 an issue with no parent is RED (exit 10)" "RED/10" "$(verdict_of "$T/orphan")"
    expect "R1 names the orphan" "6 no-parent" "$(jq -r '.detail.orphans[0] | "\(.n) \(.reason)"' "$T/orphan/v.json")"
    expect "R1 orphans rule is the one that fired" "false" "$(rule_of "$T/orphan" R1_orphans)"

    base "$T/closedp"; issue 6 '[]' 99 CLOSED >> "$T/closedp/issues.jsonl"
    expect "R1 a closed parent is an orphan" "parent-closed" "$(verdict_of "$T/closedp" >/dev/null; jq -r '.detail.orphans[0].reason' "$T/closedp/v.json")"

    base "$T/foreign"; issue 6 '[]' 7 OPEN false >> "$T/foreign/issues.jsonl"
    expect "R1 a parent in another repo is an orphan" "parent-in-other-repo" "$(verdict_of "$T/foreign" >/dev/null; jq -r '.detail.orphans[0].reason' "$T/foreign/v.json")"

    base "$T/loop"; { issue 6 '[]' 7; issue 7 '[]' 6; } >> "$T/loop/issues.jsonl"
    expect "R1 a parent loop is RED, not a hang" "RED/10" "$(verdict_of "$T/loop")"
    expect "R1 both loop members named" "6,7" "$(jq -r '[.detail.orphans[].n] | sort | map(tostring) | join(",")' "$T/loop/v.json")"

    base "$T/nonroot"; issue 6 '[]' - >> "$T/nonroot/issues.jsonl"; issue 7 '[]' 6 >> "$T/nonroot/issues.jsonl"
    expect "R1 a chain to a label-less top is an orphan chain" "6,7" "$(verdict_of "$T/nonroot" >/dev/null; jq -r '[.detail.orphans[].n] | sort | map(tostring) | join(",")' "$T/nonroot/v.json")"

    base "$T/epicchild"; issue 6 "$EPIC_LABELS" 1 >> "$T/epicchild/issues.jsonl"
    expect "an epic WITH a parent is a node, not a root (still GREEN)" "GREEN/0" "$(verdict_of "$T/epicchild")"
    expect "an epic-labelled child of a root is a ticket and is counted" "3" "$(jq -r .rules.R2_tickets.count "$T/epicchild/v.json")"

    base "$T/deep"; issue 6 '[]' 4 >> "$T/deep/issues.jsonl"
    expect "R1 a sub-sub-ticket (level 4) is RED" "RED/10" "$(verdict_of "$T/deep")"
    expect "R1 depth rule is the one that fired" "false" "$(rule_of "$T/deep" R1_depth)"

    base "$T/big"; issue_range "$T/big" 6 104 '[]' 1
    expect "R2 101 tickets is RED" "false" "$(verdict_of "$T/big" >/dev/null; rule_of "$T/big" R2_tickets)"
    expect "R2 names the count" "101" "$(jq -r .rules.R2_tickets.count "$T/big/v.json")"
    expect "R2 alone turns the verdict RED (exit 10)" "RED/10" "$(verdict_of "$T/big")"
    base "$T/bigepic"; issue_range "$T/bigepic" 6 104 "$EPIC_LABELS" 1
    expect "R2 101 tickets labelled epic is RED (a label never exempts a ticket)" "RED/10" "$(verdict_of "$T/bigepic")"
    base "$T/hundred"; issue_range "$T/hundred" 6 103 '[]' 1
    expect "R2 exactly 100 tickets under one epic is GREEN" "GREEN/0" "$(verdict_of "$T/hundred")"
    expect "R2 ok at exactly 100" "true" "$(rule_of "$T/hundred" R2_tickets)"
    # 18 more tickets (6..23), five sub-tickets each (200..289): 20 tickets, 113 open issues
    base "$T/subs"; ticket_subs "$T/subs" 6 23
    expect "R2 113 open issues but only 20 tickets is GREEN (sub-tickets are not counted)" "GREEN/0" "$(verdict_of "$T/subs")"
    expect "R2 counts the 20 tickets; open is printed" "20 113" "$(jq -r '"\(.rules.R2_tickets.count) \(.open)"' "$T/subs/v.json")"

    base "$T/fan"; issue_range "$T/fan" 6 9 '[]' 2
    expect "R3 a ticket with six sub-tickets is RED" "false" "$(verdict_of "$T/fan" >/dev/null; rule_of "$T/fan" R3_fanout)"
    expect "R3 alone turns the verdict RED (exit 10)" "RED/10" "$(verdict_of "$T/fan")"
    base "$T/fan5"; issue_range "$T/fan5" 6 8 '[]' 2
    expect "R3 a ticket with exactly five sub-tickets is GREEN" "GREEN/0" "$(verdict_of "$T/fan5")"
    base "$T/epicfan"; issue_range "$T/epicfan" 6 9 '[]' 1
    expect "R3 an epic with six tickets is GREEN (R2 counts tickets, R3 does not cap a root)" "GREEN/0" "$(verdict_of "$T/epicfan")"
    base "$T/labelfan"; issue 6 "$EPIC_LABELS" 1 >> "$T/labelfan/issues.jsonl"
    issue_range "$T/labelfan" 7 12 '[]' 6
    expect "R3 an epic-labelled ticket (it has a parent) is capped like any ticket" "false" "$(verdict_of "$T/labelfan" >/dev/null; rule_of "$T/labelfan" R3_fanout)"

    base "$T/wip"; third_ticket "$T/wip"; branch "$T/wip" 89/6-third
    expect "R4 three in-progress tickets for one worker is RED" "false" "$(verdict_of "$T/wip" >/dev/null; rule_of "$T/wip" R4_wip)"
    expect "R4 alone turns the verdict RED (exit 10)" "RED/10" "$(verdict_of "$T/wip")"
    base "$T/wipsubs"; branch "$T/wipsubs" 89/3-a; branch "$T/wipsubs" 89/4-b
    expect "R4 branches on the sub-tickets of a ticket count as that ticket (4 branches, 2 tickets)" "GREEN/0" "$(verdict_of "$T/wipsubs")"
    base "$T/wipstale"; third_ticket "$T/wipstale"; branch "$T/wipstale" 89/6-third 2026-09-01T09:00:00Z
    expect "R4 a branch older than ACTIVE_DAYS is not in progress" "GREEN/0" "$(verdict_of "$T/wipstale")"
    expect "env ACTIVE_DAYS=100 reaches the check (the 26-day-old branch is in progress: RED)" "10" "$(knob_rc ACTIVE_DAYS=100 "$T/wipstale")"
    base "$T/wipedge"; third_ticket "$T/wipedge"; branch "$T/wipedge" 89/6-edge 2026-09-20T12:00:00Z
    expect "R4 a branch exactly ACTIVE_DAYS old is still in progress (RED)" "RED/10" "$(verdict_of "$T/wipedge")"
    base "$T/wipclosed"; branch "$T/wipclosed" 89/77-closed
    expect "R4 a branch whose issue is closed is not in progress" "GREEN/0" "$(verdict_of "$T/wipclosed")"
    base "$T/wiptwo"; branch "$T/wiptwo" 89/2-again; branch "$T/wiptwo" release/0.71
    expect "R4 two branches on one issue count once; non-convention branches ignored" "GREEN/0" "$(verdict_of "$T/wiptwo")"

    base "$T/wiptype"; third_ticket "$T/wiptype"
    branch "$T/wiptype" fix/2-a; branch "$T/wiptype" fix/5-b; branch "$T/wiptype" fix/6-c; branch "$T/wiptype" feat/2-d
    expect "R4 type prefixes (fix/ feat/) are not workers, even with three tickets" "GREEN/0" "$(verdict_of "$T/wiptype")"
    base "$T/wipnested"; third_ticket "$T/wipnested"
    branch "$T/wipnested" topic/ab/2-a; branch "$T/wipnested" topic/ab/5-b; branch "$T/wipnested" topic/ab/6-c
    expect "R4 a worker id must start the branch name (topic/ab/N- is not worker ab)" "GREEN/0" "$(verdict_of "$T/wipnested")"
    base "$T/wipepic"; branch "$T/wipepic" 89/1-epic-notes
    expect "R4 a branch on an epic is not an in-progress ticket (still 2)" "GREEN/0" "$(verdict_of "$T/wipepic")"
    base "$T/wipnull"; third_ticket "$T/wipnull"; branch_undated "$T/wipnull" 89/6-undated
    expect "R4 a branch with no commit date is skipped, not a crash" "GREEN/0" "$(verdict_of "$T/wipnull")"

    mkdir -p "$T/empty"; : > "$T/empty/issues.jsonl"; : > "$T/empty/branches.jsonl"; printf '2026-09-27T12:00:00Z\n' > "$T/empty/now.txt"
    expect "0 issues read is NO-DATA (exit 20), never GREEN" "NO-DATA/20" "$(verdict_of "$T/empty")"

    expect "a dir without fetch output crashes (exit 2), not GREEN" "2" "$(check_rc "$T/missing")"
    base "$T/baddate"; branch "$T/baddate" 89/2-x not-a-date
    expect "a jq runtime error crashes (exit 2), never NO-DATA (20)" "2" "$(check_rc "$T/baddate")"

    expect "summary line" "ISSUE-TREE-001 RED: tickets 2/100, orphans 1, depth>3 0, fanout>5 0, wip>2 0 workers; not capped: open 6, roots 1" \
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
