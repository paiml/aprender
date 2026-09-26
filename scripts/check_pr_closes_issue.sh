#!/usr/bin/env bash
# check_pr_closes_issue.sh - a PR body must close every issue it cites, or say
# why not.
#
# APR-RELEASE-001 section 6, predicate R-2. The 0.67.0 reconcile pass found 48
# merged PRs since the previous tag, of which only 9 carried a closing
# reference - the rest cited an issue with "Refs #N" and left it open forever.
# This guard runs on the PR body itself, before merge, so a future train does
# not have to reconcile the debt after the fact.
#
# A body PASSES (exit 0) when:
#   * every "#N" it cites is inside a GitHub closing keyword
#     (Close(s)(d)|Fix(es)(ed)|Resolve(s)(d), case-insensitive, optionally
#     "owner/repo#N"), or
#   * a non-closing reference exists but the body also carries a
#     "keep-open: <reason>" line with a non-empty reason, or
#   * the body cites no issue at all (a docs-only PR is not required to).
#
# A body FAILS (exit 1) when it cites at least one issue with no closing
# keyword and no valid keep-open line. The failing references are printed.
#
# A body ALSO FAILS (exit 1), unconditionally and before anything else, when
# it contains a "no-close:" line or any other hyphen-prefixed closing keyword
# next to a "#N" (e.g. "wont-fix: #N", "skip-close: #N"). #3400, measured
# 2026-09-15/16: GitHub's closing-keyword parser matches "close: #3091" INSIDE
# "no-close: #3091" -- the hyphen is a word boundary to it, so the negating
# prefix is invisible. #3091 was closed twice this way despite carrying that
# exact line. This guard's own CLOSE_RE has the identical blind spot (proven
# by a repro fixture: `no-close: #3090` / `no-close: #3477` both parse as
# "PASS: every cited issue has a closing keyword" -- the guard agreeing with
# GitHub's mistake instead of catching it), which is how #3090 and #3477 were
# closed a third time on 2026-09-18 by PR #3484 despite two explicit
# "no-close:" lines naming them. `keep-open:` is the sanctioned marker: it
# contains no substring GitHub's parser (or this guard's own CLOSE_RE) reads
# as a closing keyword.
#
# Exit 2 is reserved for usage errors and vacuity: no body was given at all
# (missing --body file, or an empty body).
#
# --list-owed judges nothing. It prints, one "#N" per line, the references a
# keep-open line has to answer for (cited, no closing keyword, not proven a pull
# request or a closed issue), whether or not the body already has one, and exits
# 0 -- 2 on an empty body. A body generator uses it to name those refs (#3699).
#
# usage:
#   check_pr_closes_issue.sh --body FILE
#   check_pr_closes_issue.sh < body.txt
#   check_pr_closes_issue.sh --list-owed --body FILE
#   check_pr_closes_issue.sh --self-test

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

usage() {
    printf 'usage: %s [--list-owed | --require-close [--author LOGIN]] (--body FILE | < body-on-stdin) | --self-test\n' "$(basename "$0")" >&2
    exit 2
}

# Closing keywords GitHub recognises, case-insensitive, with an optional
# "owner/repo" prefix before the '#'. This is CLASS: closing reference.
CLOSE_RE='(close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)[[:space:]]*:?[[:space:]]*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[0-9]+'
REF_RE='#[0-9]+'

# prose_lines < body > one output line per input line: the line as written if
# GitHub renders it as prose, a blank line for a blank line, and a \x01
# placeholder (non-blank, matches nothing) for every line inside a block. It is a
# line-state pass, so an UNCLOSED block runs to the end of the body exactly as
# GitHub renders it. Blocks: ``` / ~~~ fences (closed by the opening character,
# at least as long; a backtick fence whose info string holds a backtick is not a
# fence), a <!-- comment left open on its line (to -->), and the HTML blocks
# that may span a blank line (<pre|script|style|textarea> to the closing tag,
# <? to ?>, <![CDATA[ to ]]>, <!X to >) and a $$ math block (any line opening
# with $$ and not closing it, to a line that is $$ alone). A BOM on the first line is
# dropped, as cmark-gfm drops it (agy round 7 @1ae7cb5fd). Nothing that ends at a blank line is
# tracked -- inline code spans, tags, link reference definitions, blockquote
# lazy continuations, the other HTML blocks: a line inside one always has a
# non-blank neighbour, so the paragraph rule in isolated_lines() excludes it by
# construction, and a blockquote or tag line cannot match at column 0 anyway. Tracking
# them line by line was a race against a Markdown parser that lost six agy
# quorum rounds (@afc250626 .. @b91da50e1).
prose_lines() {
    # CRLF and a bare CR are line endings in CommonMark: normalise both to LF
    # first, or `prose\r```` hides a fence from the pass (agy round 8)
    perl -0777 -pe 's/\r\n?/\n/g' | perl -ne '
        BEGIN { $end = "" }
        s/\n\z//;
        s/^\xEF\xBB\xBF// if $. == 1;
        if ($end ne "") { $end = "" if /$end/; print "\x01\n"; next }
        if (/^[ \t]*$/) { print "\n"; next }
        if (/^ {0,3}(`{3,})[^`]*$/ || /^ {0,3}(~{3,})/) {
            $end = "^ {0,3}" . quotemeta(substr($1, 0, 1)) . "{" . length($1) . ",}[ \t]*\$"; print "\x01\n"; next }
        if (/^ {0,3}\$\$(?!.*\$\$)/) { $end = "^ {0,3}\\\$\\\$[ \t]*\$"; print "\x01\n"; next }
        if (/^ {0,3}<(pre|script|style|textarea)(?:[\s>]|$)/i) {
            $end = "(?i)</(?:pre|script|style|textarea)>" unless /<\/(?:pre|script|style|textarea)>/i; print "\x01\n"; next }
        if (/^ {0,3}<\?/)         { $end = "\\?>" unless /\?>/;     print "\x01\n"; next }
        if (/^ {0,3}<!\[CDATA\[/) { $end = "\\]\\]>" unless /\]\]>/; print "\x01\n"; next }
        if (/^ {0,3}<![A-Za-z]/)   { $end = ">" unless /^ {0,3}<![A-Za-z][^>]*>/; print "\x01\n"; next }
        # after the block starts: `<script> <!--` is a type-1 block to </script>, not a comment
        # that ends at --> (agy round 12). An unanchored <!-- elsewhere errs to RED.
        if (/<!--(?!.*-->)/) { $end = "-->"; print "\x01\n"; next }
        print "$_\n";'
}
# isolated_lines < prose_lines output > the lines that are a PARAGRAPH OF THEIR
# OWN: a blank line (or the body's edge) before AND after. CommonMark parses
# inlines within one paragraph and a paragraph ends at a blank line, so no code
# span, tag, link reference definition or lazy continuation opened on another
# line can reach an isolated line. prose_lines prints every blank line as "",
# and awk reads the unset l[0] and l[NR+1] as "", so the body's edges count as
# blank. Rule 18 reads discharge lines only from here.
isolated_lines() {
    awk '{ l[NR] = $0 } END { for (i = 1; i <= NR; i++) if (l[i-1] == "" && l[i+1] == "") print l[i] }'
}
# The three discharge forms, each a whole line at column 0 (trailing period allowed):
#   Closes #N | Fixes owner/repo#N | Resolves: #N     (any GitHub closing keyword)
#   Refs #P row <id>                                  (rule 20 checklist row)
#   no-issue: <reason>
# Separators are spaces only and trailing blanks are space or tab: [[:space:]]
# also takes \v and \f, which Markdown does not treat as a separator (agy round 8).
CLOSE_LINE_RE='^(close[sd]?|fix(e[sd])?|resolve[sd]?):? +([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[0-9]+[ 	]*\.?[ 	]*$'
ROW_LINE_RE='^refs?:? +#[0-9]+ +row +[A-Za-z0-9._-]+[ 	]*$'

# THE #3400 LANDMINE, CLASS: a hyphen-prefixed closing keyword. Requires a
# word character immediately before the hyphen so it does not also match a
# bare "close: #N" (that is CLOSE_RE's job, and is fine -- it really does
# close). "no-close:", "wont-fix:", "skip-resolve:" all match; "Closes #123"
# does not (nothing precedes "Closes").
NOCLOSE_LANDMINE_RE='[A-Za-z]+-(close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)[[:space:]]*:[[:space:]]*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[0-9]+'

# A REFERENCE TO A PULL REQUEST IS NOT AN UN-CLOSED ISSUE.
#
# Measured on this repo the hour this was wired (2026-09-13): 23 of 31 open PR
# bodies failed, citing 62 distinct references between them -- and exactly 31 of
# those 62 were PULL REQUESTS, not issues. 20 of the 23 failures cited nothing
# but sibling PRs. A cross-reference to the PR that landed the thing you are
# building on is not a promise to close anything, and there is no closing
# keyword that would even mean something for one.
#
# That ratio is the argument. A gate whose findings are 20-to-3 spurious does
# not get obeyed, it gets silenced: everyone learns to paste a `no-close:` line
# on every PR, and the three bodies that really do leave an issue open --
# #3134/#532, #3001/#2873, #2720/#338, the exact debt §6 R-2 exists to stop --
# become indistinguishable from the twenty that never owed anything.
#
# RESOLUTION IS OPTIONAL AND FAILS CLOSED. No gh, no token, an API error, a
# number that does not exist: the answer is "unknown" and the ref is judged as
# an ISSUE, which is the behaviour before this change. The guard can only ever
# become MORE permissive when it has evidence, never on the absence of it.
#
# PR_CLOSES_REF_KIND_CMD is the injection seam: it receives a number and prints
# `pr` or `issue`. The self-test sets it to a table lookup, so the case rows
# below run hermetically -- no network, no token, and both branches provable.
ref_kind() { # ref_kind NUM -> pr | closed | issue | unknown
    if [ -n "${PR_CLOSES_REF_KIND_CMD:-}" ]; then
        $PR_CLOSES_REF_KIND_CMD "$1" 2>/dev/null || printf 'unknown'
        return 0
    fi
    command -v gh > /dev/null 2>&1 || { printf 'unknown'; return 0; }
    gh api "repos/${PR_CLOSES_REPO:-paiml/aprender}/issues/$1" \
        -q 'if .pull_request then "pr" elif .state == "closed" then "closed" else "issue" end' \
        2>/dev/null || printf 'unknown'
}

# issue_body NUM -> the issue's body on stdout, exit 1 when it cannot be read.
# Same seam as ref_kind: PR_CLOSES_ISSUE_BODY_CMD pins it in the case table.
issue_body() {
    if [ -n "${PR_CLOSES_ISSUE_BODY_CMD:-}" ]; then
        $PR_CLOSES_ISSUE_BODY_CMD "$1" 2>/dev/null
        return
    fi
    command -v gh > /dev/null 2>&1 || return 1
    gh api "repos/${PR_CLOSES_REPO:-paiml/aprender}/issues/$1" -q '.body // ""' 2>/dev/null
}

# AN ISSUE THAT IS ALREADY CLOSED CANNOT BE LEFT OPEN FOREVER.
#
# That sentence is this guard's entire purpose, so demanding a `no-close:`
# reason for a reference to a closed issue is the same false positive as
# demanding one for a sibling PR. Measured in the same pass: of the references
# still flagged after PR refs were dropped, #2706, #338 and #532 were all
# already closed. The reason a body would have had to write is "it is closed",
# which the API already knows.
#
# Someone else closing the issue still counts, and that is not a loophole: the
# outcome R-2 wants is a closed issue, not a particular author closing it. What
# it cannot do is pass on an OPEN issue -- that is the row below, and the
# mutation that lets it through turns the table red.
#
# drop_pull_request_refs NUMS -> the same list without the numbers PROVEN to be
# pull requests or already-closed issues. Anything unproven stays, which is what
# makes this fail closed.
drop_pull_request_refs() {
    _dpr_out=""
    for _dpr_n in $1; do
        [ -n "$_dpr_n" ] || continue
        case "$(ref_kind "$_dpr_n")" in
            pr|closed) continue ;;
        esac
        _dpr_out="$_dpr_out$_dpr_n
"
    done
    printf '%s' "$_dpr_out"
}

# classify_refs BODY -> sets all_nums (every cited number not proven a pull
# request or a closed issue, one per line) and non_closing (" #A #B", the ones
# of those with no closing keyword). non_closing is exactly what a keep-open
# line has to answer for.
#
# ONE CLASSIFIER, TWO READERS (#3699). check_body_text judges a body with it;
# --list-owed prints it so a body GENERATOR -- scripts/release/prepare_bump.sh,
# whose bump PR failed this guard on 0.68.2 (#3498) and 0.69.0 (#3698) -- names
# the refs from this function instead of a second copy of CLOSE_RE that could
# drift from the one that judges it.
classify_refs() {
    closing_nums="$(printf '%s\n' "$1" | grep -oiE "$CLOSE_RE" | grep -oE '[0-9]+$' | LC_ALL=C sort -u || true)"
    all_nums="$(printf '%s\n' "$1" | grep -oE "$REF_RE" | tr -d '#' | LC_ALL=C sort -u || true)"
    all_nums="$(drop_pull_request_refs "$all_nums")"

    closing_sp=" $(printf '%s' "$closing_nums" | tr '\n' ' ') "

    non_closing=""
    while IFS= read -r num; do
        [ -z "$num" ] && continue
        case "$closing_sp" in
            *" $num "*) ;;
            *) non_closing="$non_closing #$num" ;;
        esac
    done <<EOF_NUMS
$all_nums
EOF_NUMS
}

# check_body_text BODY
# Prints a one-line report to stdout. Returns 0 (pass), 1 (fail), or 2
# (vacuous: empty body).
check_body_text() {
    body="$1"
    stripped="$(printf '%s' "$body" | tr -d '[:space:]')"
    if [ -z "$stripped" ]; then
        printf 'VACUOUS: empty PR body, nothing to judge.\n' >&2
        return 2
    fi

    # #3400: check the landmine BEFORE anything else. A body carrying
    # "no-close: #N" is not neutral -- GitHub (and this script's own
    # CLOSE_RE, unmodified) reads it as "close: #N" and closes N on merge.
    # This must fail regardless of what else the body does correctly.
    landmine_hits="$(printf '%s\n' "$body" | grep -oiE "$NOCLOSE_LANDMINE_RE" || true)"
    if [ -n "$landmine_hits" ]; then
        printf 'FAIL: hyphen-prefixed closing keyword (#3400 landmine) -- GitHub parses this as a REAL closing reference despite the negating prefix. Use "keep-open: #N <reason>" instead:\n%s\n' "$landmine_hits"
        return 1
    fi

    classify_refs "$body"

    if [ -z "$(printf '%s' "$all_nums" | tr -d '[:space:]')" ]; then
        printf 'PASS: body cites no issue.\n'
        return 0
    fi

    if [ -z "$non_closing" ]; then
        printf 'PASS: every cited issue has a closing keyword.\n'
        return 0
    fi

    keep_open_reason="$(printf '%s\n' "$body" | grep -iE '^[[:space:]]*keep-open:' | sed -E 's/^[[:space:]]*[Kk][Ee][Ee][Pp]-[Oo][Pp][Ee][Nn]:[[:space:]]*//' | tr -d '[:space:]' || true)"

    if [ -n "$keep_open_reason" ]; then
        printf 'PASS: non-closing ref(s)%s allowed by a keep-open reason.\n' "$non_closing"
        return 0
    fi

    printf 'FAIL: non-closing ref(s) with no keep-open reason:%s\n' "$non_closing"
    return 1
}

# EVERY PR DISCHARGES AN ISSUE (--require-close; APR-EPIC-001 rule 18, #4455).
#
# R-2 above judges what a body CITES; a body that cites nothing passes it. That is
# the gap rule 18 closes: on 2026-09-25, 33 issues were done on main and still open,
# because a PR landed the fix and nothing closed the ticket, and ~77 issues/day were
# being filed against a queue with no exit. So a PR body must also carry one of:
#   * a closing keyword naming an OPEN ISSUE of this repository (`Closes #N`,
#     `Fixes paiml/aprender#N`); a pull request, a closed issue, or another
#     repository's issue does not discharge anything;
#   * a rule-20 row reference `Refs #P row <id>` to an open parent checklist issue
#     whose body carries an UNTICKED line `- [ ] <id>: ...` (the PR ticks it; only
#     the last line's PR says `Closes #P`). The row must exist and be open: a row
#     ref closes nothing on merge, so `Refs #1 row x` against any long-lived open
#     issue would otherwise discharge rule 18 with no effect at all;
#   * a line `no-issue: <reason>` with a non-empty reason. It carries no closing
#     keyword, so it cannot be the #3400 landmine.
# An unresolvable target is RED `close-target-unverified`: a guess is not a close.
#
# PR-TIME ONLY. After a merge, the issue it closed IS closed, so a merged body judged
# with this flag goes RED -- which is why check_reconcile.sh (merged PRs) does not
# pass it, and ci.yml's pull_request step does.
check_require_close() {
    body="$(printf '%s\n' "$1" | prose_lines | isolated_lines)"
    repo_lc="$(printf '%s' "${PR_CLOSES_REPO:-paiml/aprender}" | tr '[:upper:]' '[:lower:]')"
    targets="$(printf '%s\n' "$body" | grep -iE "$CLOSE_LINE_RE" | tr '[:upper:]' '[:lower:]' \
        | sed -nE "s@^[a-z]+:?[[:space:]]+(${repo_lc}|)#([0-9]+)[[:space:].]*\$@\\2@p" || true)"
    open=0; unverified=0; seen=" "
    # Row refs first, as "P:id" pairs: the parent must be an open issue AND list the
    # row unticked. Each pair is judged on its own; a bare parent number is not added
    # to the close targets, so an open parent alone never discharges.
    row_pairs="$(printf '%s\n' "$body" | grep -iE "$ROW_LINE_RE" \
        | sed -nE 's@^[^#]*#([0-9]+)[[:space:]]+[Rr][Oo][Ww][[:space:]]+([^[:space:]]+)[[:space:]]*$@\1:\2@p' || true)"
    for pair in $row_pairs; do
        p="${pair%%:*}"; id="${pair#*:}"; id="${id%.}"   # a sentence-ending period is prose, not the id
        case "$(ref_kind "$p")" in
            issue) ;;
            unknown) printf '  #%s row %s: parent state UNREADABLE\n' "$p" "$id"; unverified=$((unverified + 1)); continue ;;
            *) printf '  #%s row %s: parent is not an open issue (does not discharge)\n' "$p" "$id"; continue ;;
        esac
        if ! pbody="$(issue_body "$p")"; then
            printf '  #%s row %s: parent body UNREADABLE\n' "$p" "$id"; unverified=$((unverified + 1)); continue
        fi
        if printf '%s\n' "$pbody" | prose_lines | sed -nE 's/^[[:space:]]*[-*][[:space:]]+\[ \][[:space:]]+([^:[:space:]]+):.*/\1/p' | grep -qxF -- "$id"; then
            printf '  discharges #%s row %s (open checklist line)\n' "$p" "$id"; open=$((open + 1))
        else
            printf '  #%s row %s: no unticked `- [ ] %s:` line in #%s (does not discharge)\n' "$p" "$id" "$id" "$p"
        fi
    done
    for n in $targets; do
        case "$seen" in *" $n "*) continue ;; esac
        seen="$seen$n "
        case "$(ref_kind "$n")" in
            issue) printf '  discharges #%s (open issue)\n' "$n"; open=$((open + 1)) ;;
            unknown) printf '  #%s: state UNREADABLE\n' "$n"; unverified=$((unverified + 1)) ;;
            *) printf '  #%s: not an open issue (does not discharge)\n' "$n" ;;
        esac
    done
    if [ "$open" -gt 0 ]; then
        printf 'PASS: discharges %s open issue(s) (rule 18).\n' "$open"
        return 0
    fi
    # an ASCII visible character is required: U+00A0 and friends are not a reason
    no_issue_reason="$(printf '%s\n' "$body" | grep -iE '^no-issue:' | sed -E 's/^[Nn][Oo]-[Ii][Ss][Ss][Uu][Ee]:[[:space:]]*//' | LC_ALL=C tr -cd '!-~' || true)"
    if [ -n "$no_issue_reason" ]; then
        printf 'PASS: no-issue reason given (rule 18).\n'
        return 0
    fi
    if [ "$unverified" -gt 0 ]; then
        printf 'FAIL close-target-unverified: %s close target(s) could not be resolved; a guess is not a close.\n' "$unverified"
        return 1
    fi
    printf 'FAIL no-close: the body closes no open issue of %s and has no `no-issue: <reason>` line (APR-EPIC-001 rule 18). `Refs #N` does not close; use `Closes #N`, `Refs #P row <id>` for a rule-20 checklist row, or `no-issue: <reason>`, each at column 0 as a paragraph of its own (a blank line before and after it).\n' "$repo_lc"
    return 1
}

# self_test: a must-RED/must-GREEN case table, plus a vacuity row and a usage
# row. The table itself is the mutation proof: a neutered check that always
# passes fails the RED rows (refs-only, noclose-empty, bare-mention); a
# neutered check that always fails fails the GREEN rows.
self_test() {
    tmp="$(mktemp -d)" || return 2
    fails=0

    # THE WHOLE TABLE IS HERMETIC, and it has to be now. Once this guard began
    # resolving references, a row citing "#123" stopped being a fixture and
    # became a live API call against a real issue in this repository -- and
    # #123 and #1 really are closed here, so the two must-RED rows that shipped
    # with the guard started PASSING for a reason unrelated to what they test.
    # Every row is pinned to the stub instead: no network, no token,
    # deterministic, and both branches of every decision provable.
    #
    #   9001  a pull request        9002  an OPEN issue
    #   9004  a CLOSED issue        9003  unresolvable
    #
    # Every other number answers `issue`, which is what the legacy rows that
    # use them were always written to mean.
    cat > "${tmp}/kindstub.sh" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
  9001) printf 'pr' ;;
  9004) printf 'closed' ;;
  9003) exit 1 ;;
  *)    printf 'issue' ;;
esac
STUB
    chmod +x "${tmp}/kindstub.sh"
    # Parent bodies for rule-20 row refs: 9002 lists row A8 open and A7 ticked;
    # 9006 is an open issue whose body cannot be read; 9007 shows a row only in a fence; every other body is empty.
    cat > "${tmp}/bodystub.sh" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
  9002) printf '## Collapsed rows\n- [ ] A8: findings ledger (#4455)\n- [x] A7: done (#4454)\n- [ ] A8b.1: sub row\n' ;;
  9006) exit 1 ;;
  9007) printf '```\n- [ ] F1: an example row in a code block\n```\n' ;;
  *)    printf '' ;;
esac
STUB
    chmod +x "${tmp}/bodystub.sh"
    # DERIVED, NEVER QUOTED. The success line used to print a literal "9
    # case(s)" -- true when written, and still 9 after four rows were added.
    # A count that cannot move is not a count, it is a sentence about the past.
    cases=0

    write_case_closes="Closes #123"
    write_case_refs_only="Refs #123"
    write_case_refs_keepopen=$'refs #123\nkeep-open: tracked by the epic'
    write_case_keepopen_empty=$'keep-open:\nRefs #1'
    write_case_no_refs="Bumps dependency versions. No ticket involved."
    write_case_cross_repo="Fixes paiml/aprender#3062"
    write_case_bare_mention="#3062 (see design doc)"
    # #3400 landmine rows: must FAIL even though each also demonstrates why an
    # author reaches for this phrasing (it reads like a valid escape hatch).
    write_case_landmine_noclose="no-close: #123"
    write_case_landmine_wontfix=$'Closes #999\nwont-fix: #123 -- tracked separately'
    write_case_landmine_mixed_with_valid_close=$'Closes #456\nno-close: #123 stays open for the GDN device path'

    run_case() {
        name="$1"
        body="$2"
        want="$3"
        printf '%s' "$body" > "${tmp}/${name}.txt"
        got=0
        cases=$((cases + 1))
        PR_CLOSES_REF_KIND_CMD="${tmp}/kindstub.sh" \
            bash "$SELF_PATH" --body "${tmp}/${name}.txt" >/dev/null 2>&1 || got=$?
        if [ "$got" -ne "$want" ]; then
            printf 'FAIL case %s: expected exit %s, got %s\n' "$name" "$want" "$got" >&2
            fails=$((fails + 1))
        fi
    }

    run_case "closes" "$write_case_closes" 0
    run_case "refs-only" "$write_case_refs_only" 1
    run_case "refs-keepopen" "$write_case_refs_keepopen" 0
    run_case "keepopen-empty" "$write_case_keepopen_empty" 1
    run_case "no-refs" "$write_case_no_refs" 0
    run_case "cross-repo" "$write_case_cross_repo" 0
    run_case "bare-mention" "$write_case_bare_mention" 1
    # #3400: the landmine must fail even though it looks like a valid
    # non-closing reason to a human reader -- GitHub reads "close: #N" inside
    # it regardless of the negating prefix.
    run_case "landmine-noclose" "$write_case_landmine_noclose" 1
    run_case "landmine-wontfix" "$write_case_landmine_wontfix" 1
    run_case "landmine-mixed-with-valid-close" "$write_case_landmine_mixed_with_valid_close" 1

    # --- a PR reference is not an un-closed issue ---------------------------
    run_kind_case() { # run_kind_case NAME BODY WANT [CMD]
        name="$1"; body="$2"; want="$3"; cmd="${4:-${tmp}/kindstub.sh}"
        printf '%s' "$body" > "${tmp}/${name}.txt"
        got=0
        cases=$((cases + 1))
        PR_CLOSES_REF_KIND_CMD="$cmd" \
            bash "$SELF_PATH" --body "${tmp}/${name}.txt" >/dev/null 2>&1 || got=$?
        if [ "$got" -ne "$want" ]; then
            printf 'FAIL case %s: expected exit %s, got %s\n' "$name" "$want" "$got" >&2
            fails=$((fails + 1))
        fi
    }

    # A body whose only reference is a sibling PR owes nothing. This is the row
    # that matters: 20 of 23 real failures measured 2026-09-13 were exactly it.
    run_kind_case "pr-ref-only"    "builds on #9001"              0
    # An ISSUE reference still fails without a keyword or a reason -- the guard
    # is not weakened, only made precise.
    run_kind_case "issue-ref-only" "see #9002"                    1
    # Mixed: the PR ref is dropped, the issue ref still decides.
    run_kind_case "pr-plus-issue"  "builds on #9001, see #9002"   1
    # FAILS CLOSED. An unresolvable number is judged as an issue, so a missing
    # token or a dead API can only make this guard stricter, never laxer.
    run_kind_case "unresolvable"   "see #9003"                    1
    # An ALREADY-CLOSED issue owes no reason either -- the reason would be
    # "it is closed", which the API already knows.
    run_kind_case "closed-ref-only" "see #9004"                   0
    # ...but an OPEN issue beside a closed one still decides. This is the row
    # that stops "closed" from becoming a way through.
    run_kind_case "closed-plus-open" "see #9004 and #9002"        1

    # --- --require-close: every PR discharges an issue (rule 18, #4455) -----
    # Each RED row is a body R-2 alone PASSES, so a row going green under a
    # mutant proves the rule-18 predicate, not R-2, is what refused it.
    run_rc_case() { # run_rc_case NAME BODY WANT [MARKER the output must carry] [AUTHOR]
        name="$1"; body="$2"; want="$3"; marker="${4:-}"; author="${5:-}"
        printf '%s' "$body" > "${tmp}/${name}.txt"
        got=0
        cases=$((cases + 1))
        PR_CLOSES_REF_KIND_CMD="${tmp}/kindstub.sh" PR_CLOSES_ISSUE_BODY_CMD="${tmp}/bodystub.sh" \
            bash "$SELF_PATH" --require-close ${author:+--author "$author"} --body "${tmp}/${name}.txt" > "${tmp}/${name}.out" 2>&1 || got=$?
        if [ "$got" -ne "$want" ]; then
            printf 'FAIL case %s: expected exit %s, got %s\n' "$name" "$want" "$got" >&2
            fails=$((fails + 1))
        elif [ -n "$marker" ] && ! grep -q -- "$marker" "${tmp}/${name}.out"; then
            printf 'FAIL case %s: exit %s but no "%s" line\n' "$name" "$got" "$marker" >&2
            fails=$((fails + 1))
        fi
    }
    run_rc_case "rc-closes-open"      "Closes #9002"                                 0 "PASS: discharges 1"
    run_rc_case "rc-own-repo-prefix"  "Fixes paiml/aprender#9002"                    0 "PASS: discharges 1"
    run_rc_case "rc-closed-plus-open" $'Closes #9004\n\nFixes #9002'                   0 "PASS: discharges 1"
    run_rc_case "rc-row-ref"          $'Refs #9002 row A8\n\nkeep-open: parent checklist' 0 "PASS: discharges 1"
    run_rc_case "rc-no-issue"         $'Docs only.\n\nno-issue: typo in a comment'    0 "PASS: no-issue"
    # GitHub closes only on a whole keyword and a whole number, in prose: not mid-word,
    # not `#Na`, not inside a code fence, an inline code span or an HTML comment.
    run_rc_case "rc-close-mid-word"   "aclose #9002"                                 1 "FAIL no-close"
    run_rc_case "rc-close-suffix"     "Closes #9002a"                                1 "FAIL no-close"
    run_rc_case "rc-close-fenced"     $'```\n\nCloses #9002\n\n```'                     1 "FAIL no-close"
    run_rc_case "rc-close-tilde-fence" $'~~~\n\nCloses #9002\n\n~~~'                    1 "FAIL no-close"
    run_rc_case "rc-close-backticked" 'Write `Closes #9002` to close it.'            1 "FAIL no-close"
    run_rc_case "rc-close-html-comment" '<!-- Closes #9002 -->'                      1 "FAIL no-close"
    run_rc_case "rc-row-ref-fenced"   $'```\n\nRefs #9002 row A8\n\n```\n\nkeep-open: parent checklist'                1 "FAIL no-close"
    run_rc_case "rc-no-issue-fenced"  $'```\n\nno-issue: example line\n\n```'          1 "FAIL no-close"
    run_rc_case "rc-two-close-lines"  $'Closes #9004.\n\nCloses #9002.'              0 "PASS: discharges 1"
    run_rc_case "rc-close-in-sentence" "This closes #9002 today."                    1 "FAIL no-close"
    run_rc_case "rc-close-indented"   "    Closes #9002"                             1 "FAIL no-close"
    run_rc_case "rc-close-blockquote" "> Closes #9002"                               1 "FAIL no-close"
    run_rc_case "rc-close-double-tick" '``Closes #9002``'                            1 "FAIL no-close"
    run_rc_case "rc-close-unclosed-fence" $'```\n\nCloses #9002'                      1 "FAIL no-close"
    run_rc_case "rc-close-unclosed-comment" $'<!--\n\nCloses #9002'                   1 "FAIL no-close"
    run_rc_case "rc-close-after-comment" $'<!--\nnote\n-->\n\nCloses #9002'           0 "PASS: discharges 1"
    run_rc_case "rc-mixed-fences"     $'```\n~~~\n```\n\nCloses #9002'               0 "PASS: discharges 1"
    run_rc_case "rc-comment-opened-in-fence" $'```\n<!--\n```\n\nCloses #9002'       0 "PASS: discharges 1"
    run_rc_case "rc-crlf-fence-closes" $'```\r\nx\r\n```\r\n\r\nCloses #9002\r\n'      0 "PASS: discharges 1"
    run_rc_case "rc-close-after-inline-comment" $'<!-- note -->\n\nCloses #9002'     0 "PASS: discharges 1"
    # agy round 3 (@4fe1eb7b1): HTML blocks, same-line comments, lazy quotes, NBSP, fenced parent rows
    run_rc_case "rc-close-in-pre"      $'<pre>\n\nCloses #9002\n\n</pre>'                 1 "FAIL no-close"
    run_rc_case "rc-close-after-comment-same-line" $'<!-- -->Closes #9002'           1 "FAIL no-close"
    run_rc_case "rc-close-in-html-block" $'<div>\nCloses #9002'                      1 "FAIL no-close"
    run_rc_case "rc-close-in-pi"       $'<?x\n\nCloses #9002\n\n?>'                     1 "FAIL no-close"
    run_rc_case "rc-close-lazy-quote"  $'> quote\nCloses #9002'                       1 "FAIL no-close"
    run_rc_case "rc-fence-after-quote" $'> q\n```\n\nCloses #9002\n\n```'                 1 "FAIL no-close"
    run_rc_case "rc-no-issue-nbsp"     $'no-issue: \xc2\xa0'                          1 "FAIL no-close"
    run_rc_case "rc-row-ref-parent-fenced" $'Refs #9007 row F1\n\nkeep-open: parent' 1 "FAIL no-close"
    run_rc_case "rc-close-after-pre"   $'<pre>\nx\n</pre>\n\nCloses #9002'              0 "PASS: discharges 1"
    run_rc_case "rc-close-after-html-blank" $'<details>\n<summary>s</summary>\n\nCloses #9002' 0 "PASS: discharges 1"
    run_rc_case "rc-close-after-quote-blank" $'> quote\n\nCloses #9002'              0 "PASS: discharges 1"
    run_rc_case "rc-close-in-cdata"    $'<![CDATA[\n\nCloses #9002\n\n]]>'             1 "FAIL no-close"
    run_rc_case "rc-close-in-declaration" $'<!X\n\nCloses #9002\n\n>'                  1 "FAIL no-close"
    run_rc_case "rc-close-after-quote-then-fence" $'> q\n```\nx\n```\n\nCloses #9002'  0 "PASS: discharges 1"
    # agy round 4 (@6f9582631): inline constructs spanning lines
    run_rc_case "rc-close-in-multiline-code" $'`\nCloses #9002\n`'                  1 "FAIL no-close"
    run_rc_case "rc-close-in-multiline-tag" $'text <a href="\nCloses #9002\n">'     1 "FAIL no-close"
    run_rc_case "rc-close-after-linkref" $'[foo]:\nCloses #9002'                    1 "FAIL no-close"
    run_rc_case "rc-close-in-code-after-comment" $'x `<!-- -->\nCloses #9002\n`'    1 "FAIL no-close"
    run_rc_case "rc-close-after-balanced-code" $'Fix `a` and ``b`c``\n\nCloses #9002' 0 "PASS: discharges 1"
    run_rc_case "rc-close-after-closed-tag" $'See <b>this</b>\n\nCloses #9002'        0 "PASS: discharges 1"
    run_rc_case "rc-close-after-open-code-para" $'`open\n\nCloses #9002'            0 "PASS: discharges 1"
    # agy round 5 (@25b0cb80d): a > inside a quoted attribute does not close the tag
    run_rc_case "rc-close-in-tag-quoted-gt" $'Text <a title=">"\nCloses #9002\n>'   1 "FAIL no-close"
    run_rc_case "rc-close-in-tag-squoted-gt" $'Text <a title=\'>\'\nCloses #9002\n>' 1 "FAIL no-close"
    run_rc_case "rc-close-after-tag-quoted-gt" $'See <a title=">">x</a>\n\nCloses #9002' 0 "PASS: discharges 1"
    run_rc_case "rc-close-after-tag-quote-new-para" $'x <a title="\n\nx <b>\n\nCloses #9002' 0 "PASS: discharges 1"
    run_rc_case "rc-close-after-quote-char-prose" $'It\'s "quoted"\n\nCloses #9002'     0 "PASS: discharges 1"
    run_rc_case "rc-backtick-info-not-fence" $'``` a`b\n\nCloses #9002'               0 "PASS: discharges 1"
    # agy round 6 (@b91da50e1): inline tracking kept losing to the parser, so a
    # discharge must be a paragraph of its own; these bodies are refused by that
    run_rc_case "rc-close-in-tag-backtick-attr" $'x <a href=foo `>\nCloses #9002\n`' 1 "FAIL no-close"
    run_rc_case "rc-close-in-code-escaped-tick" $'a \\` `\nCloses #9002\n`'   1 "FAIL no-close"
    run_rc_case "rc-close-not-isolated-before" $'Closes #9002\ntrailing prose'     1 "FAIL no-close"
    run_rc_case "rc-close-in-comment-isolated" $'<!--\n\nCloses #9002\n\n-->'   1 "FAIL no-close"
    run_rc_case "rc-close-between-html-paras" $'<div>\n\nCloses #9002\n\n</div>' 0 "PASS: discharges 1"
    run_rc_case "rc-close-blank-ws-lines" $'Intro.\n \t\nCloses #9002\n  '     0 "PASS: discharges 1"
    run_rc_case "rc-close-in-math-block" $'$$\n\nCloses #9002\n\n$$'        1 "FAIL no-close"
    run_rc_case "rc-close-after-math-block" $'$$\nx\n$$\n\nCloses #9002'       0 "PASS: discharges 1"
    run_rc_case "rc-close-touching-fence" $'```\nx\n```\nCloses #9002'          1 "FAIL no-close"
    run_rc_case "rc-close-before-fence" $'Closes #9002\n```\nx\n```'          1 "FAIL no-close"
    run_rc_case "rc-close-in-long-fence" $'````\n```\n\nCloses #9002\n\n````'  1 "FAIL no-close"
    run_rc_case "rc-close-fence-info-not-closer" $'```\n```x\n\nCloses #9002\n\n```' 1 "FAIL no-close"
    run_rc_case "rc-close-after-oneline-pre" $'<pre>x</pre>\n\nCloses #9002'    0 "PASS: discharges 1"
    # agy round 7 (@1ae7cb5fd): a leading BOM hid a fence; $$ with text opens math
    run_rc_case "rc-close-after-bom-fence" $'\xef\xbb\xbf```\n\nCloses #9002\n\n```' 1 "FAIL no-close"
    run_rc_case "rc-close-bom-first-line" $'\xef\xbb\xbfCloses #9002'           0 "PASS: discharges 1"
    run_rc_case "rc-close-mid-body-bom" $'Intro.\n\n\xef\xbb\xbfCloses #9002'   1 "FAIL no-close"
    run_rc_case "rc-close-in-math-text-opener" $'$$ x\n\nCloses #9002\n\n$$'   1 "FAIL no-close"
    run_rc_case "rc-close-after-oneline-math" $'$$x$$\n\nCloses #9002'          0 "PASS: discharges 1"
    # agy round 8 (@cb1ab552c): bare CR is a line ending; $$ closes only alone; \v is no separator
    run_rc_case "rc-close-after-bare-cr-fence" $'prose\r```\n\nCloses #9002\n\n```' 1 "FAIL no-close"
    run_rc_case "rc-close-bare-cr-lines" $'Intro.\r\rCloses #9002\r'             0 "PASS: discharges 1"
    run_rc_case "rc-close-in-math-early-dollars" $'$$\n\na $$ b\n\nCloses #9002\n\n$$' 1 "FAIL no-close"
    run_rc_case "rc-close-vtab-separator" $'Closes\v#9002'                         1 "FAIL no-close"
    run_rc_case "rc-row-ref-vtab-separator" $'Refs\v#9002 row A8\n\nkeep-open: parent checklist' 1 "FAIL no-close"
    # agy round 9 (@71e3c0ce3): <textarea> spans blank lines until </textarea> (errs RED); a
    # PR body has no YAML front matter, so --- ... --- are thematic breaks and the line is prose
    run_rc_case "rc-close-in-textarea"    $'<textarea>\n\nCloses #9002\n\n</textarea>' 1 "FAIL no-close"
    run_rc_case "rc-close-touching-textarea" $'<textarea>\n</textarea>\nCloses #9002'  1 "FAIL no-close"
    run_rc_case "rc-close-after-textarea" $'<textarea>\n</textarea>\n\nCloses #9002'  0 "PASS: discharges 1"
    run_rc_case "rc-close-between-front-matter-rules" $'---\n\nCloses #9002\n\n---' 0 "PASS: discharges 1"
    # agy round 10 (@d6ac5c1fa), each measured against commonmark.js 0.31.2 as <p>Closes #N</p>:
    # a closing tag opens no block, and a column-0 line after a blank ends a list item and its fence
    run_rc_case "rc-close-between-script-closers" $'</script>\n\nCloses #9002\n\n</script>' 0 "PASS: discharges 1"
    run_rc_case "rc-close-after-list-item-fence" $'- item\n\n    ```\n\nCloses #9002\n\n    ```' 0 "PASS: discharges 1"
    run_rc_case "rc-close-after-list-marker-fence" $'* ```\n\nCloses #9002\n\n```' 0 "PASS: discharges 1"
    # agy round 11 (@a512c6ae2): <details> is a type-6 HTML block, which ends at the blank line;
    # commonmark.js 0.31.2 -t xml: html_block, paragraph, html_block
    run_rc_case "rc-close-between-details-tags" $'<details>\n\nCloses #9002\n\n</details>' 0 "PASS: discharges 1"
    # agy round 12 (@cbe415e07): a comment opened on a block's start line does not end the block
    run_rc_case "rc-close-in-script-with-comment" $'<script> <!--\n-->\n\nCloses #9002\n\n</script>' 1 "FAIL no-close"
    run_rc_case "rc-close-in-pre-with-comment"    $'<pre><!--\n-->\n\nCloses #9002\n\n</pre>'       1 "FAIL no-close"
    run_rc_case "rc-close-trailing-tab" $'Closes #9002\t'                          0 "PASS: discharges 1"
    run_rc_case "rc-no-issue-code-reason" $'no-issue: `docs/` only'                 0 "PASS: no-issue"
    run_rc_case "rc-no-issue-indented" $'  no-issue: docs'                          1 "FAIL no-close"
    run_rc_case "rc-close-after-fence" $'```\nexample\n```\n\nCloses #9002'           0 "PASS: discharges 1"
    # a row ref starts at a word: "Xrefs #N row X" is prose, not `Refs #N row X`
    run_rc_case "rc-row-ref-mid-word" $'Refs #9002\n\nXrefs #9002 row A8\n\nkeep-open: parent checklist' 1 "FAIL no-close"
    # FALSIFY-FLOW-012: a body with only `Refs #N` (R-2 satisfied by keep-open) and no trailer.
    run_rc_case "rc-flow-012-refs-only" $'Refs #9002\nkeep-open: tracked by the epic' 1 "FAIL no-close"
    run_rc_case "rc-no-refs"          "Bumps dependency versions."                   1 "FAIL no-close"
    run_rc_case "rc-no-issue-empty"   $'Docs only.\n\nno-issue:'                       1 "FAIL no-close"
    run_rc_case "rc-no-issue-midline" "this line mentions no-issue: inline"          1 "FAIL no-close"
    run_rc_case "rc-closes-closed"    "Closes #9004"                                 1 "FAIL no-close"
    run_rc_case "rc-closes-pr"        "Closes #9001"                                 1 "FAIL no-close"
    run_rc_case "rc-cross-repo"       "Fixes paiml/infra#9002"                       1 "FAIL no-close"
    run_rc_case "rc-row-ref-closed"   $'Refs #9004 row A8'                           1 "FAIL no-close"
    # The row must be a real, unticked line of the parent (Sonnet-5 review of #4455).
    run_rc_case "rc-row-ref-fabricated" $'Refs #9002 row x\n\nkeep-open: parent checklist' 1 "FAIL no-close"
    run_rc_case "rc-row-ref-ticked"   $'Refs #9002 row A7\n\nkeep-open: parent checklist' 1 "FAIL no-close"
    run_rc_case "rc-row-ref-prefix"   $'Refs #9002 row A\n\nkeep-open: parent checklist'  1 "FAIL no-close"
    run_rc_case "rc-row-ref-dotted"   $'Refs #9002 row A8b.1\n\nkeep-open: parent checklist' 0 "PASS: discharges 1"
    run_rc_case "rc-row-ref-period"   $'Refs #9002 row A8.\n\nkeep-open: parent checklist' 0 "PASS: discharges 1"
    run_rc_case "rc-row-ref-in-sentence" $'Ticks Refs #9002 row A8.\n\nkeep-open: parent checklist' 1 "FAIL no-close"
    run_rc_case "rc-row-ref-no-body"  $'Refs #9006 row A8\n\nkeep-open: parent checklist' 1 "FAIL close-target-unverified"
    run_rc_case "rc-unresolvable"     "Closes #9003"                                 1 "FAIL close-target-unverified"
    # R-2 still runs first under the flag: a discharge does not excuse an un-closed citation.
    run_rc_case "rc-r2-still-applies" $'Closes #9002\n\nsee #9005'                     1 "non-closing ref"
    # A GitHub App (login ending in `[bot]`: dependabot, renovate) cannot write a
    # trailer, so rule 18 exempts it. R-2 still applies, and a human login that
    # merely looks like a bot is not exempt.
    run_rc_case "rc-bot-exempt"       "Bumps serde from 1.0.1 to 1.0.2."             0 "exempt: bot author" "dependabot[bot]"
    run_rc_case "rc-bot-lookalike"    "Bumps serde from 1.0.1 to 1.0.2."             1 "FAIL no-close"      "dependabot"
    run_rc_case "rc-bot-r2-applies"   "Bumps serde; see #9005"                       1 "non-closing ref"    "dependabot[bot]"
    run_rc_case "rc-human-author"     "Closes #9002"                                 0 "PASS: discharges 1" "noahgift"

    # THE DEFAULT PATH, not the seam. Every row above pins the stub, which
    # proves the DECISION and says nothing about the resolver. This one takes
    # the stub away and shadows `gh` with one that exits 1 — the shape a runner
    # with no token, a rate limit, or no network has. ref_kind must answer
    # `unknown`, the ref must be judged as an ISSUE, and the body must still
    # FAIL. Without this row, a resolver that answered `pr` or `closed` on its
    # own error path would be invisible here, and that is the only way this
    # change could make the guard laxer than it was.
    #
    # `gh` is shadowed rather than removed from PATH: it lives in /usr/bin
    # beside the grep and sed this script needs, so an empty PATH tests nothing
    # but exit 127.
    mkdir -p "${tmp}/ghfail"
    printf '#!/usr/bin/env bash\nexit 1\n' > "${tmp}/ghfail/gh"
    chmod +x "${tmp}/ghfail/gh"
    printf '%s' "see #1" > "${tmp}/no-gh.txt"
    got=0
    cases=$((cases + 1))
    ( PATH="${tmp}/ghfail:$PATH"; export PATH
      unset PR_CLOSES_REF_KIND_CMD
      bash "$SELF_PATH" --body "${tmp}/no-gh.txt" ) >/dev/null 2>&1 || got=$?
    if [ "$got" -ne 1 ]; then
        printf 'FAIL case gh-error-fails-closed: expected exit 1, got %s\n' "$got" >&2
        fails=$((fails + 1))
    fi

    # --- --list-owed: what a keep-open line has to answer for (#3699) --------
    # Owed: the open issue (9002) and the unresolvable one (9003, fails closed,
    # exactly as the judge treats it). Not owed: the PR (9001), the closed issue
    # (9004), the ref under a closing keyword (9005). The keep-open line already
    # in the body changes nothing -- the list is what it must NAME, not whether
    # one exists. A classifier that listed every #N, or none, fails this row.
    printf '%s' $'builds on #9001, see #9002 and #9004, Closes #9005, refs #9003\nkeep-open: already here' > "${tmp}/owed.txt"
    cases=$((cases + 1))
    owed="$(PR_CLOSES_REF_KIND_CMD="${tmp}/kindstub.sh" \
        bash "$SELF_PATH" --list-owed --body "${tmp}/owed.txt" 2>/dev/null | tr '\n' ' ')" || owed="exit!=0"
    if [ "$owed" != "#9002 #9003 " ]; then
        printf 'FAIL case list-owed: expected "#9002 #9003 ", got "%s"\n' "$owed" >&2
        fails=$((fails + 1))
    fi

    : > "${tmp}/empty.txt"
    got=0
    cases=$((cases + 1))
    bash "$SELF_PATH" --body "${tmp}/empty.txt" >/dev/null 2>&1 || got=$?
    if [ "$got" -ne 2 ]; then
        printf 'FAIL case empty-body: expected exit 2, got %s\n' "$got" >&2
        fails=$((fails + 1))
    fi

    got=0
    cases=$((cases + 1))
    bash "$SELF_PATH" --list-owed --body "${tmp}/empty.txt" >/dev/null 2>&1 || got=$?
    if [ "$got" -ne 2 ]; then
        printf 'FAIL case list-owed-empty-body: expected exit 2, got %s\n' "$got" >&2
        fails=$((fails + 1))
    fi

    got=0
    cases=$((cases + 1))
    bash "$SELF_PATH" --body "${tmp}/does-not-exist.txt" >/dev/null 2>&1 || got=$?
    if [ "$got" -ne 2 ]; then
        printf 'FAIL case missing-file: expected exit 2, got %s\n' "$got" >&2
        fails=$((fails + 1))
    fi

    rm -rf "${tmp:?}"

    if [ "$fails" -ne 0 ]; then
        printf 'self-test FAILED: %s case(s).\n' "$fails" >&2
        return 1
    fi
    # VACUITY FLOOR, and only that. A table that ran NOTHING would otherwise
    # print "self-test OK: 0 case(s)" and exit 0 -- the shape this repo keeps
    # finding. 9 is what the table shipped with before this change.
    #
    # IT DOES NOT CATCH A DELETED ROW, and was measured not to: removing the
    # bare-mention row leaves 12, which is above the floor, so the table still
    # passes. Completeness is a ratchet (scripts/*_baseline.txt), not a floor,
    # and pretending otherwise here would be the kind of gate that reads like a
    # proof and is not one.
    if [ "$cases" -lt 9 ]; then
        printf 'self-test VACUOUS: %s case(s) ran, fewer than the 9 this table shipped with.\n' "$cases" >&2
        return 1
    fi
    printf 'self-test OK: %s case(s).\n' "$cases"
    return 0
}

main() {
    # No arguments (guard_tree.sh runs every cargo-free guard bare): the self-test IS the bare run.
    if [ $# -eq 0 ] || [ "${1:-}" = "--self-test" ]; then
        self_test
        exit $?
    fi

    body_file=""
    list_owed=0
    require_close=0
    author=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --require-close)
                require_close=1
                shift
                ;;
            --body)
                body_file="${2:-}"
                shift 2
                ;;
            --author)
                author="${2:-}"
                shift 2
                ;;
            --list-owed)
                list_owed=1
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                usage
                ;;
        esac
    done

    if [ -n "$body_file" ]; then
        if [ ! -f "$body_file" ]; then
            printf 'ERROR: body file not found: %s\n' "$body_file" >&2
            exit 2
        fi
        body="$(cat "$body_file")"
    else
        if [ -t 0 ]; then
            usage
        fi
        body="$(cat -)"
    fi

    if [ "$list_owed" -eq 1 ]; then
        if [ -z "$(printf '%s' "$body" | tr -d '[:space:]')" ]; then
            printf 'VACUOUS: empty PR body, nothing to list.\n' >&2
            exit 2
        fi
        classify_refs "$body"
        for ref in $non_closing; do
            printf '%s\n' "$ref"
        done
        exit 0
    fi

    rc=0
    out="$(check_body_text "$body")" || rc=$?
    printf '%s\n' "$out"
    if [ "$rc" -eq 0 ] && [ "$require_close" -eq 1 ]; then
        case "$author" in
            *'[bot]')
                printf 'PASS: rule 18 exempt: bot author %s cannot write a trailer\n' "$author"
                exit 0
                ;;
        esac
        out="$(check_require_close "$body")" || rc=$?
        printf '%s\n' "$out"
    fi
    exit "$rc"
}

main "$@"
