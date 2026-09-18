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
# usage:
#   check_pr_closes_issue.sh --body FILE
#   check_pr_closes_issue.sh < body.txt
#   check_pr_closes_issue.sh --self-test

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

usage() {
    printf 'usage: %s (--body FILE | < body-on-stdin) | --self-test\n' "$(basename "$0")" >&2
    exit 2
}

# Closing keywords GitHub recognises, case-insensitive, with an optional
# "owner/repo" prefix before the '#'. This is CLASS: closing reference.
CLOSE_RE='(close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)[[:space:]]*:?[[:space:]]*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[0-9]+'
REF_RE='#[0-9]+'

# THE #3400 LANDMINE, CLASS: a hyphen-prefixed closing keyword. Requires a
# word character immediately before the hyphen so it does not also match a
# bare "close: #N" (that is CLOSE_RE's job, and is fine -- it really does
# close). "no-close:", "wont-fix:", "skip-resolve:" all match; "Closes #123"
# does not (nothing precedes "Closes").
#
# The colon is OPTIONAL, matching CLOSE_RE's own `:?` exactly -- an
# independent review of the first cut of this fix (evidence/pr-review/3497)
# found that a colon-less body ("no-fix #123 tracked elsewhere") satisfied
# CLOSE_RE (which never required the colon) while failing to satisfy a
# landmine regex that mandated one, so it was judged a VALID closing
# reference instead of the #3400 landmine it actually is. Confirmed as a real
# gap, not a false alarm: `no-fix #123` reproduces the exact bug this guard
# exists to catch, just spelled without the colon.
NOCLOSE_LANDMINE_RE='[A-Za-z]+-(close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)[[:space:]]*:?[[:space:]]*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[0-9]+'

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

    closing_nums="$(printf '%s\n' "$body" | grep -oiE "$CLOSE_RE" | grep -oE '[0-9]+$' | LC_ALL=C sort -u || true)"
    all_nums="$(printf '%s\n' "$body" | grep -oE "$REF_RE" | tr -d '#' | LC_ALL=C sort -u || true)"
    all_nums="$(drop_pull_request_refs "$all_nums")"

    if [ -z "$(printf '%s' "$all_nums" | tr -d '[:space:]')" ]; then
        printf 'PASS: body cites no issue.\n'
        return 0
    fi

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
    # PR #3497's own independent review (evidence/pr-review/3497) found this row
    # missing: the colon is optional in CLOSE_RE, so it must be optional here too,
    # or "no-fix #123" (no colon) reads as a VALID closing reference instead of
    # the landmine it is.
    write_case_landmine_no_colon="no-fix #123 tracked elsewhere"

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
    run_case "landmine-no-colon" "$write_case_landmine_no_colon" 1

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
    while [ $# -gt 0 ]; do
        case "$1" in
            --body)
                body_file="${2:-}"
                shift 2
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

    rc=0
    out="$(check_body_text "$body")" || rc=$?
    printf '%s\n' "$out"
    exit "$rc"
}

main "$@"
