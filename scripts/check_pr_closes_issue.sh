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
#     "no-close: <reason>" line with a non-empty reason, or
#   * the body cites no issue at all (a docs-only PR is not required to).
#
# A body FAILS (exit 1) when it cites at least one issue with no closing
# keyword and no valid no-close line. The failing references are printed.
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

    closing_nums="$(printf '%s\n' "$body" | grep -oiE "$CLOSE_RE" | grep -oE '[0-9]+$' | LC_ALL=C sort -u || true)"
    all_nums="$(printf '%s\n' "$body" | grep -oE "$REF_RE" | tr -d '#' | LC_ALL=C sort -u || true)"

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

    no_close_reason="$(printf '%s\n' "$body" | grep -iE '^[[:space:]]*no-close:' | sed -E 's/^[[:space:]]*[Nn][Oo]-[Cc][Ll][Oo][Ss][Ee]:[[:space:]]*//' | tr -d '[:space:]' || true)"

    if [ -n "$no_close_reason" ]; then
        printf 'PASS: non-closing ref(s)%s allowed by a no-close reason.\n' "$non_closing"
        return 0
    fi

    printf 'FAIL: non-closing ref(s) with no no-close reason:%s\n' "$non_closing"
    return 1
}

# self_test: a must-RED/must-GREEN case table, plus a vacuity row and a usage
# row. The table itself is the mutation proof: a neutered check that always
# passes fails the RED rows (refs-only, noclose-empty, bare-mention); a
# neutered check that always fails fails the GREEN rows.
self_test() {
    tmp="$(mktemp -d)" || return 2
    fails=0

    write_case_closes="Closes #123"
    write_case_refs_only="Refs #123"
    write_case_refs_noclose=$'refs #123\nno-close: tracked by the epic'
    write_case_noclose_empty=$'no-close:\nRefs #1'
    write_case_no_refs="Bumps dependency versions. No ticket involved."
    write_case_cross_repo="Fixes paiml/aprender#3062"
    write_case_bare_mention="#3062 (see design doc)"

    run_case() {
        name="$1"
        body="$2"
        want="$3"
        printf '%s' "$body" > "${tmp}/${name}.txt"
        got=0
        bash "$SELF_PATH" --body "${tmp}/${name}.txt" >/dev/null 2>&1 || got=$?
        if [ "$got" -ne "$want" ]; then
            printf 'FAIL case %s: expected exit %s, got %s\n' "$name" "$want" "$got" >&2
            fails=$((fails + 1))
        fi
    }

    run_case "closes" "$write_case_closes" 0
    run_case "refs-only" "$write_case_refs_only" 1
    run_case "refs-noclose" "$write_case_refs_noclose" 0
    run_case "noclose-empty" "$write_case_noclose_empty" 1
    run_case "no-refs" "$write_case_no_refs" 0
    run_case "cross-repo" "$write_case_cross_repo" 0
    run_case "bare-mention" "$write_case_bare_mention" 1

    : > "${tmp}/empty.txt"
    got=0
    bash "$SELF_PATH" --body "${tmp}/empty.txt" >/dev/null 2>&1 || got=$?
    if [ "$got" -ne 2 ]; then
        printf 'FAIL case empty-body: expected exit 2, got %s\n' "$got" >&2
        fails=$((fails + 1))
    fi

    got=0
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
    printf 'self-test OK: 9 case(s).\n'
    return 0
}

main() {
    if [ "${1:-}" = "--self-test" ]; then
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
