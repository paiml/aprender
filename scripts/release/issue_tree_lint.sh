#!/usr/bin/env bash
# issue_tree_lint.sh — ISSUE-TREE-001: the nightly issue-graph lint. The cop runs it from a timer.
#
# The operator ruling is five rules over the OPEN issues of one repo:
#   R1 tree   every open issue has exactly 1 parent (GitHub allows at most one, so the check is "not 0"),
#             except a ROOT: an open issue labelled `epic` that has no parent. Depth <= 3 levels:
#             epic (1) -> ticket (2) -> sub-ticket (3). A sub-sub-ticket (4) is RED: the ruling allows none.
#             0 orphans. An orphan is a non-root with no parent, a CLOSED parent, a parent in another repo,
#             or a parent chain that loops.
#   R2 size   open TICKETS <= 100. A ticket is a direct child of a root. Roots and sub-tickets are not
#             counted. An `epic`-labelled issue WITH a parent is not a root: under a root it is a ticket and
#             is counted. The open-issue and root totals are printed, not capped (0 open issues = NO-DATA).
#   R3 fanout open children per non-root issue <= 5, i.e. sub-tickets per ticket. A root's tickets are
#             not capped here: R2 counts them.
#   R4 wip    in-progress TICKETS per worker <= 2. A branch on a sub-ticket counts as its ticket; a branch
#             on an epic (a root) is not a ticket and is not counted; a branch on any other issue (an
#             orphan too) counts as that issue. "In progress" means a remote branch
#             `<worker>/<issue>-<slug>` (the fleet's branch convention; worker = 2-3 lowercase hex, the
#             session id, so type prefixes like fix/ feat/ docs/ fold/ perf/ are not workers) whose issue is
#             open and whose head commit is at most ACTIVE_DAYS (default 7) old. A worker is its session id.
#             R4 sees only branches that follow the convention, so its count is a floor on WIP, not all of
#             it; and an all-hex type prefix (add/ bad/ fed/) would read as a worker.
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
# Env: MAX_TICKETS=100 MAX_DEPTH=3 MAX_CHILDREN=5 MAX_WIP=2 ACTIVE_DAYS=7 (whole numbers; anything else exits 2)
#      ROOT_LABEL=epic   ISSUE_TITLE="ISSUE-TREE-001 RED: issue graph lint"
#      SOURCE_DATE_EPOCH=<whole seconds>  pins the now.txt a hand-run fetch writes (default: the clock); run refuses it
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
# The knobs reach jq through --argjson, where jq sorts a string, [] or {} above every number, so a cap set
# to one never fires; 1.5 and -1 are not counts either. Whole numbers only, spelled digit by digit: in a
# locale like en_US.UTF-8 the ranges [1-9] and [0-9] also match other scripts' digits, which jq rejects.
# The group keeps bashrs from reading "][" as an array subscript (SC2180).
for knob in MAX_TICKETS MAX_DEPTH MAX_CHILDREN MAX_WIP ACTIVE_DAYS; do
    [[ ${!knob} =~ ^(0|[123456789]([0123456789])*)$ ]] || die "$knob must be a whole number, got '${!knob}'"
done

# The snapshot's now: R4 counts a branch whose head is at most ACTIVE_DAYS older. DET002: SOURCE_DATE_EPOCH
# pins it when set (a fetch run by hand, e.g. to rebuild a past night's window); otherwise the clock, which a
# live snapshot records. Whole seconds as date +%s writes them, the knob rule: jq alone would also take 1.5,
# -1, 1e9 or 007 as a time. At most 253402300799, the last second of 9999: check cannot take a five-digit year.
# A time jq cannot write stops the fetch with exit 2, as any input error does. jq formats it because BSD date
# has no `-d @`.
snapshot_now() {
    [[ ${SOURCE_DATE_EPOCH:-0} =~ ^(0|[123456789]([0123456789])*)$ ]] || die "SOURCE_DATE_EPOCH must be a whole number of seconds, got '${SOURCE_DATE_EPOCH-}'"
    jq -nr --argjson t "${SOURCE_DATE_EPOCH:-$(date +%s)}" 'if $t > 253402300799 then error("past 9999") else $t | todate end' \
        || die "SOURCE_DATE_EPOCH must be at most 253402300799 (9999-12-31T23:59:59Z), got '${SOURCE_DATE_EPOCH-}'"
}

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
    # Through a variable, not a redirect: a bad SOURCE_DATE_EPOCH stops the fetch before now.txt exists, where a
    # redirect would leave an empty one for check to trip on.
    local now
    now=$(snapshot_now)
    printf '%s\n' "$now" > "$dir/now.txt"
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
        }') || die "check: jq failed on $dir (a malformed fetch row)"
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
    # run reads the live graph, so its now is the clock. A SOURCE_DATE_EPOCH left in the environment (build
    # tooling exports it) would make R4 count every branch since ACTIVE_DAYS before that date instead.
    [ -z "${SOURCE_DATE_EPOCH:+set}" ] || die "run: SOURCE_DATE_EPOCH is set; it pins a hand-run fetch only, and run reads the live graph"
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
# The stderr of a fresh `check` run with one env knob set, in locale $3 (a locale the host lacks falls back to C).
knob_err() { local -x LC_ALL="${3:?}" "${1:?}"; bash "$0" check "$2" 2>&1 >/dev/null || :; }
# yes when a fresh bash in locale $1 matches an Arabic-Indic digit with [0-9], as a knob check with ranges would.
range_takes_wide_digit() { local -x LC_ALL="${1:?}"; bash -c 'if [[ ١ =~ ^[0-9]$ ]]; then printf yes; else printf no; fi'; }
# A fresh `fetch` or `run` ($1) into dir $3 with env assignment $2, against the stub gh in $T/stub (never the
# network), SOURCE_DATE_EPOCH unset unless $2 sets it. Prints the exit code.
stub_rc() {
    local rc=0
    env -u SOURCE_DATE_EPOCH PATH="$T/stub:$PATH" "${2:?}" bash "$0" "${1:?}" "${3:?}" >/dev/null 2>&1 || rc=$?
    printf '%s' "$rc"
}
# The now of a snapshot with SOURCE_DATE_EPOCH set to $1 (exported, as knob_rc does).
snap_with() { local -x SOURCE_DATE_EPOCH="$1"; snapshot_now; }
# failed when that snapshot fails. A die in it ends the command substitution it runs in, so read the status of that.
snap_fails() { local now; if now=$(snap_with "$1" 2>/dev/null); then printf 'wrote %s' "$now"; else printf failed; fi; }

verdict_of() { local rc=0; check "$1" > "$1/v.json" || rc=$?; printf '%s/%s' "$(jq -r .verdict "$1/v.json")" "$rc"; }
rule_of() { jq -r ".rules.$2.ok" "$1/v.json"; }

self_test() {
    local T quoted
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
    expect "a non-numeric MAX_TICKETS stops the run (exit 2), never NO-DATA" "2" "$(knob_rc MAX_TICKETS=abc "$T/green")"
    quoted='MAX_TICKETS="100"'
    expect "a quoted MAX_TICKETS stops the run (exit 2): jq reads 2 <= a string as true" "2" "$(knob_rc "$quoted" "$T/green")"
    expect "an array MAX_DEPTH stops the run (exit 2): no level is > [] in jq" "2" "$(knob_rc 'MAX_DEPTH=[]' "$T/green")"
    expect "an object MAX_CHILDREN stops the run (exit 2): no count is > {} in jq" "2" "$(knob_rc 'MAX_CHILDREN={}' "$T/green")"
    expect "a fractional MAX_WIP stops the run (exit 2)" "2" "$(knob_rc MAX_WIP=1.5 "$T/green")"
    expect "a negative ACTIVE_DAYS stops the run (exit 2): it would empty R4" "2" "$(knob_rc ACTIVE_DAYS=-1 "$T/green")"
    expect "a zero-padded MAX_WIP stops the run (exit 2): 007 is not how a count is written" "2" "$(knob_rc MAX_WIP=007 "$T/green")"
    # The next case is a test only where [0-9] matches an Arabic-Indic digit. A host without en_US.UTF-8 falls back
    # to C, where it does not, and the case would pass with the ranges back; this one makes that host RED instead.
    local wide_locale=en_US.UTF-8
    expect "a fresh bash in $wide_locale matches an Arabic-Indic digit with [0-9], so the next case can fail" "yes" \
        "$(range_takes_wide_digit "$wide_locale")"
    expect "Arabic-Indic digits in MAX_TICKETS stop the run at the knob check, even in $wide_locale where [0-9] matches them" \
        "1" "$(knob_err 'MAX_TICKETS=١٠٠' "$T/green" "$wide_locale" | grep -c 'MAX_TICKETS must be a whole number')"
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
    base "$T/foreignroot"; issue 6 '[]' 1 OPEN false >> "$T/foreignroot/issues.jsonl"
    expect "R2 a parent #1 in another repo is not local root #1: an orphan, not a ticket" "2 parent-in-other-repo" "$(verdict_of "$T/foreignroot" >/dev/null; jq -r '"\(.rules.R2_tickets.count) \(.detail.orphans[0].reason)"' "$T/foreignroot/v.json")"

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
    base "$T/deeper"; { issue 6 '[]' 4; issue 7 '[]' 6; } >> "$T/deeper/issues.jsonl"
    expect "R1 names every issue below level 3 (levels 4 and 5)" "6,7" "$(verdict_of "$T/deeper" >/dev/null; jq -r '[.detail.deep[].n] | sort | map(tostring) | join(",")' "$T/deeper/v.json")"

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
    base "$T/wipworkers"; third_ticket "$T/wipworkers"
    branch "$T/wipworkers" ab/2-a; branch "$T/wipworkers" ab/6-b; branch "$T/wipworkers" ac/5-c; branch "$T/wipworkers" ac/6-d
    expect "R4 caps each worker, not the fleet (89, ab, ac on 2 tickets each; 3 tickets in all)" "GREEN/0" "$(verdict_of "$T/wipworkers")"
    base "$T/wipviasubs"; third_ticket "$T/wipviasubs"; { issue 7 '[]' 5; issue 8 '[]' 6; } >> "$T/wipviasubs/issues.jsonl"
    branch "$T/wipviasubs" ad/3-a; branch "$T/wipviasubs" ad/7-b; branch "$T/wipviasubs" ad/8-c
    expect "R4 three tickets reached only through sub-ticket branches is RED" "false" "$(verdict_of "$T/wipviasubs" >/dev/null; rule_of "$T/wipviasubs" R4_wip)"
    base "$T/wiporphan"; { issue 6 '[]' -; issue 7 '[]' -; issue 8 '[]' -; } >> "$T/wiporphan/issues.jsonl"
    branch "$T/wiporphan" ae/6-a; branch "$T/wiporphan" ae/7-b; branch "$T/wiporphan" ae/8-c
    expect "R4 a branch on an orphan counts as that issue (3 orphans, one worker: R4 RED as well as R1)" "false" "$(verdict_of "$T/wiporphan" >/dev/null; rule_of "$T/wiporphan" R4_wip)"

    base "$T/wiptype"; third_ticket "$T/wiptype"
    branch "$T/wiptype" fix/2-a; branch "$T/wiptype" fix/5-b; branch "$T/wiptype" fix/6-c; branch "$T/wiptype" feat/2-d
    expect "R4 type prefixes (fix/ feat/) are not workers, even with three tickets" "GREEN/0" "$(verdict_of "$T/wiptype")"
    base "$T/wipnested"; third_ticket "$T/wipnested"
    branch "$T/wipnested" topic/ab/2-a; branch "$T/wipnested" topic/ab/5-b; branch "$T/wipnested" topic/ab/6-c
    expect "R4 a worker id must start the branch name (topic/ab/N- is not worker ab)" "GREEN/0" "$(verdict_of "$T/wipnested")"
    base "$T/wipdecoy"; third_ticket "$T/wipdecoy"
    branch "$T/wipdecoy" a/2-x; branch "$T/wipdecoy" a/5-y; branch "$T/wipdecoy" a/6-z
    branch "$T/wipdecoy" abcd/2-x; branch "$T/wipdecoy" abcd/5-y; branch "$T/wipdecoy" abcd/6-z
    branch "$T/wipdecoy" AB/2-x; branch "$T/wipdecoy" AB/5-y; branch "$T/wipdecoy" AB/6-z
    branch "$T/wipdecoy" ab/2-x; branch "$T/wipdecoy" ab/5-y; branch "$T/wipdecoy" ab/6
    expect "R4 a worker is 2-3 lowercase hex and needs <issue>-: a/ abcd/ AB/ and a dash-less ab/6 are not counted" "GREEN/0" "$(verdict_of "$T/wipdecoy")"
    expect "R4 reads only the rows that follow the convention (89 x2, ab x2)" "4" "$(jq -r .wip_rows "$T/wipdecoy/v.json")"
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

    # The now of a snapshot. The stub gh answers every query with one empty, final page: fetch and run go offline.
    mkdir -p "$T/stub"
    cat > "$T/stub/gh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' '{"data":{"repository":{"issues":{"pageInfo":{"hasNextPage":false},"nodes":[]},"refs":{"pageInfo":{"hasNextPage":false},"nodes":[]}}}}'
STUB
    chmod +x "$T/stub/gh"
    expect "SOURCE_DATE_EPOCH pins the now of the snapshot" "2026-09-21T14:13:20Z" "$(snap_with 1790000000)"
    expect "an unset SOURCE_DATE_EPOCH is the clock: a now after 2026-10-04, the day this case was written" "true" \
        "$(unset SOURCE_DATE_EPOCH; snapshot_now | jq -R 'fromdateiso8601 >= 1791072000')"
    expect "an empty SOURCE_DATE_EPOCH is the clock too, as an unset one is" "true" \
        "$(snap_with '' | jq -R 'fromdateiso8601 >= 1791072000')"
    expect "a SOURCE_DATE_EPOCH that is not a number fails the snapshot: no now is written from it" "failed" "$(snap_fails x)"
    expect "a fractional SOURCE_DATE_EPOCH fails the snapshot too: jq alone would write a now from 1.5" "failed" "$(snap_fails 1.5)"
    expect "a negative SOURCE_DATE_EPOCH fails the snapshot" "failed" "$(snap_fails -1)"
    expect "an exponent SOURCE_DATE_EPOCH fails the snapshot" "failed" "$(snap_fails 1e9)"
    expect "a zero-padded SOURCE_DATE_EPOCH fails the snapshot: date +%s never pads, and jq would take 007 as 7" \
        "failed" "$(snap_fails 007)"
    expect "a whole number past the range of jq fails the snapshot, never falls back to the clock" "failed" \
        "$(snap_fails 99999999999999999999)"
    expect "the last second of 9999 is still a now" "9999-12-31T23:59:59Z" "$(snap_with 253402300799)"
    expect "a SOURCE_DATE_EPOCH past 9999-12-31 fails the snapshot: check cannot take a five-digit year" "failed" \
        "$(snap_fails 253402300800)"
    expect "fetch with SOURCE_DATE_EPOCH set runs (exit 0)" "0" "$(stub_rc fetch SOURCE_DATE_EPOCH=1790000000 "$T/pinned")"
    expect "fetch writes the pinned now to now.txt" "2026-09-21T14:13:20Z" "$(cat "$T/pinned/now.txt")"
    expect "fetch with a fractional SOURCE_DATE_EPOCH stops (exit 2)" "2" "$(stub_rc fetch SOURCE_DATE_EPOCH=1.5 "$T/junk")"
    expect "... and leaves no now.txt for check to read" "absent" \
        "$(if [ -e "$T/junk/now.txt" ]; then printf present; else printf absent; fi)"
    expect "fetch with a SOURCE_DATE_EPOCH jq cannot date stops with exit 2, not the exit code of jq" "2" \
        "$(stub_rc fetch SOURCE_DATE_EPOCH=99999999999999999999 "$T/junk-far")"
    expect "run refuses a set SOURCE_DATE_EPOCH (exit 2) before it fetches" "2" "$(stub_rc run SOURCE_DATE_EPOCH=0 "$T/run-pinned")"
    expect "... and wrote nothing" "absent" "$(if [ -e "$T/run-pinned" ]; then printf present; else printf absent; fi)"
    expect "run with no SOURCE_DATE_EPOCH fetches and checks: the empty graph of the stub is NO-DATA (exit 20)" "20" \
        "$(stub_rc run ACTIVE_DAYS=7 "$T/run-clock")"
    expect "run with an empty SOURCE_DATE_EPOCH goes on too: empty means unset, as in the snapshot" "20" \
        "$(stub_rc run SOURCE_DATE_EPOCH= "$T/run-empty")"

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
