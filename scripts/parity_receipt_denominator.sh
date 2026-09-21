#!/usr/bin/env bash
# parity_receipt_denominator.sh — PMAT-3577 / #3577.
#
# THE PREDICATE, not the extractor. `evidence/parity/EXPECTED_RECEIPTS` records how many logit-parity
# receipts the tree holds; this script recomputes that number from the tree itself, by a rule written
# INDEPENDENTLY of the Rust extractor. Two implementations of one question is the whole point: an
# extractor checked against a number the extractor produced proves nothing.
#
# The rule: a file under `evidence/parity/**` is a receipt iff it is JSON whose top-level `schema` is
# `apr-parity-receipt/v2`. Anything else is not counted, and an UNMIGRATED legacy record — no schema but a
# top-level `metrics[]` or `parity` — is REFUSED, because a record the extractor cannot see is a record no
# shape can refuse.
#
# The universe is `git ls-files`, never `find`: `.claude/worktrees/lane-*` holds full clones at other
# commits, and `find` returns them (aprender#3579).
#
#   bash scripts/parity_receipt_denominator.sh            # verify against EXPECTED_RECEIPTS
#   bash scripts/parity_receipt_denominator.sh --print    # print the measured count and exit 0
#   bash scripts/parity_receipt_denominator.sh --self-test
#
# Exit: 0 agree · 1 disagree (or an unmigrated record) · 2 usage / the file is missing / git cannot
# list the tree (ENV: e.g. "dubious ownership" on a bind-mounted docker checkout, #3669).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_FILE="evidence/parity/EXPECTED_RECEIPTS"
SCHEMA="apr-parity-receipt/v2"

count_and_check() {
    local root=$1 records=0 unmigrated=() listing
    # #3669: on the docker runners git refuses the bind-mounted tree ("detected dubious ownership"),
    # and this listing used to discard that refusal (`2>/dev/null` inside a process substitution,
    # whose exit status nothing reads), so the loop saw an empty universe and reported "the tree
    # holds 0". A git failure is an ENV exit 2 that names git, never a count. git's own stderr is kept.
    if ! listing=$(git -c safe.directory="$root" -C "$root" ls-files \
            'evidence/parity/*.json' 'evidence/parity/**/*.json'); then
        printf 'ENV   git ls-files failed in %s: the receipt universe cannot be read, so no count is reported\n' \
            "$root" >&2
        return 2
    fi
    local f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        case "$(classify "$root/$f")" in
            record) records=$((records + 1)) ;;
            legacy) unmigrated+=("$f") ;;
            *) : ;;
        esac
    done < <(printf '%s\n' "$listing" | sort -u)
    if [ "${#unmigrated[@]}" -gt 0 ]; then
        printf 'FAIL  %s unmigrated legacy record(s) - no schema, but a top-level metrics[]/parity:\n' \
            "${#unmigrated[@]}" >&2
        printf '        %s\n' "${unmigrated[@]}" >&2
        return 1
    fi
    printf '%s\n' "$records"
}

# record | legacy | other — one file, by its own content.
classify() {
    python3 - "$1" "$SCHEMA" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("other"); raise SystemExit(0)
if not isinstance(d, dict):
    print("other"); raise SystemExit(0)
if d.get("schema") == sys.argv[2]:
    print("record")
elif isinstance(d.get("metrics"), list) or "parity" in d:
    print("legacy")
else:
    print("other")
PY
}

expected_of() {
    grep -vE '^\s*(#|$)' "$1/$EXPECTED_FILE" | head -1 | tr -d '[:space:]'
}

# Remove a directory this script created, and NOTHING else. The validation is here, immediately above
# the `rm`, rather than at the call site: a guard the reader has to go and find is a guard that gets
# moved away from what it protects (bashrs SEC011).
discard_tempdir() {
    dir=$1
    [ -n "$dir" ] || return 0
    [ -d "$dir" ] || return 0
    case "$dir" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'refusing to remove %s: not a temp directory this script made\n' "$dir" >&2; return 0 ;;
    esac
    rm -rf -- "$dir"
}

self_test() {
    local rc=0 td
    td=$(mktemp -d)
    [ -n "$td" ] && [ -d "$td" ] && [ "${#td}" -gt 8 ] || { printf 'FAIL  mktemp -d gave %s\n' "${td:-<empty>}" >&2; return 2; }
    trap 'discard_tempdir "$td"' RETURN
    # hooksPath off: the self-test's throwaway repos must not run the developer's global pre-commit hook.
    (cd "$td" && git init -q . && git config user.email t@t && git config user.name t \
        && git config core.hooksPath /dev/null)
    mkdir -p "$td/evidence/parity/l0-1/lambda"
    printf '{"schema":"%s","host":"h"}\n' "$SCHEMA" > "$td/evidence/parity/l0-1/lambda/a.json"
    printf '{"seed":1}\n' > "$td/evidence/parity/props-x.json"
    printf '1\n' > "$td/$EXPECTED_FILE"
    (cd "$td" && git add -A && git commit -qm t)

    # ONE record, one unrelated document, denominator 1 -> agree.
    if out=$(count_and_check "$td" 2>&1) && [ "$out" = 1 ]; then
        printf 'ok    a record is counted and an unrelated document is not (measured %s)\n' "$out"
    else
        printf 'FAIL  expected 1, got %s\n' "$out"; rc=1
    fi

    # A legacy record appears -> REFUSED, never counted, never skipped.
    printf '{"model":"./m.gguf","parity":true,"metrics":[]}\n' > "$td/evidence/parity/l0-1/lambda/legacy.json"
    (cd "$td" && git add -A && git commit -qm legacy)
    if count_and_check "$td" >/dev/null 2>&1; then
        printf 'FAIL  an unmigrated legacy record did not refuse\n'; rc=1
    else
        printf 'ok    an unmigrated legacy record is refused by name\n'
    fi
    rm "$td/evidence/parity/l0-1/lambda/legacy.json"
    (cd "$td" && git add -A && git commit -qm rm)

    # A receipt added without bumping the denominator -> disagree. THE falsifier the row names.
    printf '{"schema":"%s","host":"h2"}\n' "$SCHEMA" > "$td/evidence/parity/l0-1/lambda/b.json"
    (cd "$td" && git add -A && git commit -qm add)
    if verify "$td" >/dev/null 2>&1; then
        printf 'FAIL  a receipt added without bumping the denominator passed\n'; rc=1
    else
        printf 'ok    a receipt added without bumping the denominator disagrees\n'
    fi

    # And bumping it makes them agree again — BOTH directions, or the control proves nothing.
    printf '2\n' > "$td/$EXPECTED_FILE"
    (cd "$td" && git add -A && git commit -qm bump)
    if verify "$td" >/dev/null 2>&1; then
        printf 'ok    bumping the denominator makes them agree\n'
    else
        printf 'FAIL  the denominator was bumped and they still disagree\n'; rc=1
    fi

    # #3669: git itself refuses, as it does on the docker runners (exit 128, "dubious ownership").
    # That is ENV (exit 2) naming git. It is never "the tree holds 0".
    mkdir -p "$td/nogit"
    printf '#!/bin/sh\necho "fatal: detected dubious ownership in repository" >&2\nexit 128\n' > "$td/nogit/git"
    chmod 755 "$td/nogit/git"
    local got=0 gout
    gout=$(PATH="$td/nogit:$PATH" verify "$td" 2>&1) || got=$?
    if [ "$got" -eq 2 ] && grep -q 'git' <<<"$gout" && ! grep -q 'holds 0' <<<"$gout"; then
        printf 'ok    git refusing the tree is ENV exit 2 naming git, never a count of 0\n'
    else
        printf 'FAIL  git refused and the result was exit %s: %s\n' "$got" "$gout"; rc=1
    fi
    return "$rc"
}

verify() {
    local root=$1 measured expected
    [ -f "$root/$EXPECTED_FILE" ] || { printf 'FAIL  %s is missing\n' "$EXPECTED_FILE" >&2; return 2; }
    measured=$(count_and_check "$root") || return $?
    expected=$(expected_of "$root")
    if [ "$measured" = "$expected" ]; then
        printf 'PASS  %s receipt(s) under evidence/parity/**, and %s says %s.\n' \
            "$measured" "$EXPECTED_FILE" "$expected"
        return 0
    fi
    printf 'FAIL  %s says %s; the tree holds %s.\n' "$EXPECTED_FILE" "$expected" "$measured" >&2
    printf '      Unknown{WrongCorpus}: update the denominator in the SAME commit as the receipt.\n' >&2
    return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    --print)     count_and_check "$ROOT" ;;
    "")          verify "$ROOT" ;;
    *)           printf 'usage: %s [--print|--self-test]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
