#!/usr/bin/env bash
# check_pr_ont_delta.sh — APR-RELEASE-001 §11.1, WIRED.
#
# §11.1: "Every sweep PR body carries one line, in the form
# scripts/check_pr_closes_issue.sh already enforces for `Closes #N`:
# `ont-delta: <type|shape|reason|resolves> <id>` or `ont-delta: none <reason>`.
# Absent is a PR-body lint failure, not a review comment."
#
# It was a review comment, because no lint existed. §11 shipped as prose in
# #3268 and nothing read it — `grep -rl ont-delta scripts/ .github/ Makefile`
# returned nothing. A rule whose enforcement is "not a review comment" and whose
# mechanism IS a review comment is the anti-theater class this repo keeps
# finding; this file is that rule's caller.
#
# WHAT A SWEEP PR IS (§11.1). Two halves, and only one of them may be a list:
#
#   * the prose sinks the spec NAMES. These are quoted from §11.1 and change
#     only when §11.1 changes, so they are constants here and are printed on
#     every run so a drift between the two is visible rather than silent.
#   * "a known-red list anywhere" — DERIVED, never typed. A new baseline file
#     arriving unclassified is exactly how this class stays alive
#     (check_baseline_ratchets.sh reason 1), and a tracked-only universe is a
#     free pass for a baseline that is present but not yet added (reason 3).
#     The universe is the UNION of a working-tree `find` and the index.
#
# WHAT IT DOES NOT DO. It does not judge whether the delta is GOOD — that is
# §11.2's ratchet, which moves only through `make ont-ratchet`. It decides only
# whether the sweep closed with one, which is the thing §11.1 states and nothing
# measured.
#
#   bash scripts/check_pr_ont_delta.sh --self-test
#   bash scripts/check_pr_ont_delta.sh --body B.txt --changed FILES.txt
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    printf 'usage: %s (--body FILE --changed FILE) | --self-test\n' "$(basename "$0")" >&2
    exit 2
}

# The sinks §11.1 names, verbatim. Under docs/specifications/ the sink is the release spec ITSELF —
# the train writes its findings there — never a design spec (aprender#3535: three maintainer PRs,
# each touching one design document, were red for a delta that did not exist).
ONT_SWEEP_SINKS=(
    ".github/workflows/night.yml"
    "docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md"
    "contracts/apr-cli-commands-v1.yaml"
    "README.md"
    "CLAUDE.md"
)

# Known-red lists: union of the working tree and the index, never one alone.
known_red_lists() {
    {
        find "$REPO_ROOT/scripts" -maxdepth 1 -name '*baseline*.txt' -type f -printf 'scripts/%f\n' 2>/dev/null || true
        git -C "$REPO_ROOT" ls-files 'scripts/*baseline*.txt' 2>/dev/null || true
    } | LC_ALL=C sort -u
}

# True when any changed path is a sweep surface. Prints the reason it fired.
sweep_reason() { # sweep_reason CHANGED_FILE
    local changed="$1" f sink
    local -a reds
    mapfile -t reds < <(known_red_lists)
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        for sink in "${ONT_SWEEP_SINKS[@]}"; do
            case "$sink" in
                */) [ "${f##"$sink"}" != "$f" ] && { printf 'prose sink %s (§11.1)\n' "$f"; return 0; } ;;
                *)  [ "$f" = "$sink" ] && { printf 'prose sink %s (§11.1)\n' "$f"; return 0; } ;;
            esac
        done
        for sink in "${reds[@]}"; do
            [ "$f" = "$sink" ] && { printf 'known-red list %s (derived)\n' "$f"; return 0; }
        done
    done < "$changed"
    return 1
}

# ont-delta line vocabulary. Anything else is OUTSIDE it, which is a failure and
# not a pass: an unknown key is a setting that does not exist.
ONT_DELTA_KINDS="type shape reason resolves"

check_body_text() { # check_body_text BODY_FILE -> 0 ok, 1 bad; prints the verdict
    local body="$1" line kind rest k ok
    line="$(grep -iE '^[[:space:]]*ont-delta:' "$body" | head -1 || true)"
    if [ -z "$line" ]; then
        printf 'FAIL  no `ont-delta:` line. §11.1: a sweep PR closes with a delta,\n'
        printf '      not a paragraph. Use one of:\n'
        printf '        ont-delta: type|shape|reason|resolves <id>\n'
        printf '        ont-delta: none <why this sweep produced no delta>\n'
        return 1
    fi
    rest="${line#*:}"
    rest="${rest#"${rest%%[![:space:]]*}"}"
    kind="${rest%%[[:space:]]*}"
    rest="${rest#"$kind"}"
    rest="${rest#"${rest%%[![:space:]]*}"}"
    kind="$(printf '%s' "$kind" | tr '[:upper:]' '[:lower:]')"
    if [ "$kind" = "none" ]; then
        if [ -z "$rest" ]; then
            printf 'FAIL  `ont-delta: none` with no reason. "none" is a CLAIM about the\n'
            printf '      sweep; an empty one is the paragraph §11.1 refuses.\n'
            return 1
        fi
        printf 'ok    ont-delta: none — %s\n' "$rest"
        return 0
    fi
    ok=0
    for k in $ONT_DELTA_KINDS; do [ "$kind" = "$k" ] && ok=1; done
    if [ "$ok" -ne 1 ]; then
        printf 'FAIL  `ont-delta: %s` is outside the vocabulary {%s|none}.\n' "$kind" "${ONT_DELTA_KINDS// /|}"
        printf '      An unrecognised kind is a delta that does not exist (§11.1).\n'
        return 1
    fi
    if [ -z "$rest" ]; then
        printf 'FAIL  `ont-delta: %s` with no id. The delta must NAME the thing it added.\n' "$kind"
        return 1
    fi
    printf 'ok    ont-delta: %s %s\n' "$kind" "$rest"
    return 0
}

self_test() {
    # NOT a RETURN trap: bash destroys the function's locals before the trap
    # body runs, so `rm -rf "$tmp"` there dies on an unbound variable under
    # `set -u` -- and the self-test had already PASSED by then, so the exit
    # status flipped 0 -> 1 with fifteen green rows above it. Explicit cleanup.
    local tmp pass=0 fail=0 rc
    tmp="$(mktemp -d)"
    # SEC011: an unvalidated `rm -rf "$tmp"` is a delete-anything primitive if the
    # variable is ever empty or `/`. Prove the shape before the sweep, not after.
    case "$tmp" in
        /tmp/*|/var/tmp/*) : ;;
        *) printf 'NO-GO: mktemp -d gave an unexpected path: %s\n' "$tmp" >&2; return 2 ;;
    esac

    row() { # row NAME WANT BODY
        local name="$1" want="$2" body="$3"
        printf '%s' "$body" > "$tmp/b.txt"
        set +e; check_body_text "$tmp/b.txt" >/dev/null 2>&1; rc=$?; set -e
        if [ "$rc" -eq "$want" ]; then pass=$((pass+1)); printf '  ok    %-46s want=%s\n' "$name" "$want"
        else fail=$((fail+1)); printf '  FAIL  %-46s want=%s got=%s\n' "$name" "$want" "$rc"; fi
    }
    printf 'check_pr_ont_delta self-test\n'
    row "type with an id"                 0 'ont-delta: type ONT-entity-run'
    row "shape with an id"                0 'ont-delta: shape lockfile-pin-shape'
    row "reason with an id"               0 'ont-delta: reason ont6-unread-window'
    row "resolves with an id"             0 'ont-delta: resolves scripts/x.sh'
    row "none WITH a reason"              0 'ont-delta: none spec paragraph only'
    row "case-insensitive key"            0 'ONT-DELTA: type ONT-x'
    row "leading whitespace"              0 '   ont-delta: type ONT-x'
    row "body with other lines"           0 'Closes #1

ont-delta: type ONT-x

trailer'
    row "absent"                          1 'a body that says nothing'
    row "none with NO reason"             1 'ont-delta: none'
    row "none, trailing spaces only"      1 'ont-delta: none   '
    row "kind outside the vocabulary"     1 'ont-delta: entity ONT-x'
    row "type with NO id"                 1 'ont-delta: type'
    row "shape with NO id"                1 'ont-delta: shape  '
    row "empty body"                      1 ''
    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$tmp" ] && [ -d "$tmp" ] && rm -rf "$tmp"
    [ "$fail" -eq 0 ]
}

# The PREDICATE half had no self-test: every row above feeds check_body_text, so
# sweep_reason could have classified every PR as a sweep (or none) and the
# self-test would stay green. Measured 2026-09-19: three PRs by a maintainer
# (#3499 a guard-regex fix, #3420 and #3424 design specs) were RED on this
# guard, every one of them because `docs/specifications/**` is a prefix sink
# and §11.1 says so verbatim. A design spec that IS the finding's home is not
# "writing a finding into a prose sink"; whether the sentence or the row moves
# is a §11.1 amendment, and the row below is the falsifying case either way.
predicate_row() { # predicate_row NAME WANT CHANGED_PATH  (want: 0 = sweep, 1 = not a sweep)
    local name="$1" want="$2" path="$3" tmp rc
    tmp="$(mktemp -d)"
    case "$tmp" in /tmp/*|/var/tmp/*) : ;; *) printf 'NO-GO: mktemp -d gave %s\n' "$tmp" >&2; return 2 ;; esac
    printf '%s\n' "$path" > "$tmp/c.txt"
    set +e; sweep_reason "$tmp/c.txt" >/dev/null 2>&1; rc=$?; set -e
    [ -n "$tmp" ] && [ -d "$tmp" ] && rm -rf "$tmp"
    if [ "$rc" -eq "$want" ]; then printf '  ok    %-46s want=%s\n' "$name" "$want"; return 0
    else printf '  FAIL  %-46s want=%s got=%s\n' "$name" "$want" "$rc"; return 1; fi
}

predicate_self_test() {
    local pass=0 fail=0
    printf 'check_pr_ont_delta predicate self-test (sweep_reason)\n'
    prow() { if predicate_row "$@"; then pass=$((pass+1)); else fail=$((fail+1)); fi; }
    prow "crate source is not a sweep"            1 'crates/aprender-core/src/lib.rs'
    prow "a contract yaml is not a sweep"         1 'contracts/tensor-layout-v1.yaml'
    prow "a workflow other than night is not"     1 '.github/workflows/ci.yml'
    prow "night.yml is a prose sink"              0 '.github/workflows/night.yml'
    prow "README.md is a prose sink"              0 'README.md'
    prow "CLAUDE.md is a prose sink"              0 'CLAUDE.md'
    prow "apr-cli-commands registry is a sink"    0 'contracts/apr-cli-commands-v1.yaml'
    prow "a known-red baseline is DERIVED"        0 'scripts/cb200_baseline.txt'
    prow "the release spec itself is a sink"      0 'docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md'
    # THE FALSIFYING ROW. A design spec (PP-QUANT-001, #3420) carries no finding
    # from a sweep; under the §11.1 sentence as first written it WAS a sweep (want=0)
    # and the guard redded a maintainer's spec PR for a line about a delta that
    # did not exist. The amendment (aprender#3535) makes this row want=1; restoring the
    # directory-wide sink turns it RED again, which is the proof it discriminates.
    prow "a DESIGN spec is not a sweep (§11.1 amended)"   1 'docs/specifications/PP-QUANT-001-MASTER.md'
    printf 'predicate self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

main() {
    local body="" changed="" reason
    if [ $# -eq 0 ] || [ "${1:-}" = "--self-test" ]; then
        self_test && predicate_self_test; return $?
    fi
    while [ $# -gt 0 ]; do
        case "$1" in
            --body)    body="${2:-}"; shift 2 ;;
            --changed) changed="${2:-}"; shift 2 ;;
            *) usage ;;
        esac
    done
    [ -n "$body" ] && [ -n "$changed" ] || usage
    [ -f "$body" ] || { printf 'NO-GO: body file %s does not exist.\n' "$body" >&2; exit 2; }
    [ -f "$changed" ] || { printf 'NO-GO: changed-files file %s does not exist.\n' "$changed" >&2; exit 2; }

    # Vacuity floor. A PR that changed nothing is not a PR; judging it as
    # "not a sweep" would be a pass this guard never earned.
    if [ ! -s "$changed" ]; then
        printf 'NO-GO: the changed-file list is EMPTY, so "not a sweep PR" is a verdict\n' >&2
        printf 'this guard cannot have reached. Refusing rather than passing vacuously.\n' >&2
        exit 2
    fi

    printf '== ont-delta (APR-RELEASE-001 §11.1) ==\n'
    printf 'prose sinks named by §11.1: %s\n' "${ONT_SWEEP_SINKS[*]}"
    printf 'known-red lists derived:    %s\n' "$(known_red_lists | wc -l)"
    printf 'changed paths:              %s\n' "$(grep -cve '^[[:space:]]*$' "$changed" || true)"

    if ! reason="$(sweep_reason "$changed")"; then
        printf 'ok    not a sweep PR — no prose sink and no known-red list touched.\n'
        printf 'PASS\n'
        return 0
    fi
    printf 'sweep PR: %s\n' "$reason"
    if check_body_text "$body"; then printf 'PASS\n'; return 0; fi
    printf 'FAILED: §11.1 — a sweep that ends in prose and nothing else is not finished.\n'
    return 1
}

main "$@"
