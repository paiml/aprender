#!/usr/bin/env bash
# check_roadmap_sorted.sh — new docs/roadmaps/roadmap.yaml entries land at a
# SORTED position, not blindly appended to the tail (BSE-09a, PMAT-1065).
#
# WHY THIS EXISTS
# ----------------
# BSE-09b measured the failure this guard exists to prevent: two PRs each
# appended a new top-level entry to the tail of roadmap.yaml, with
# `.gitattributes merge=union` in place to reconcile them. GitHub's merge
# engine does not honour `merge=union` (it is a local-only git driver), so
# the second PR to land reported `mergeable=false` — CONFLICTING — even
# though the two edits touched conceptually unrelated tickets. Two PRs that
# both insert at the SAME line (the tail) always conflict; two PRs that
# insert at DIFFERENT lines (their ticket's sorted position among its own
# id-prefix peers) usually do not.
#
# THE RULE, enforced against docs/roadmaps/roadmap.yaml's top-level (0-indent)
# `- id:` entries only (nested subtask ids are a different guard's job —
# see check_roadmap_ids_unique.sh):
#
#   1. SORT ORDER. An id of the form PREFIX-NUMBER (letters/digits prefix,
#      hyphen, all-digit numeric suffix — e.g. PMAT-9, PMAT-10, GH-279) must
#      appear in numerically ascending order relative to every OTHER id
#      sharing the same PREFIX, in file order (so PMAT-9 sorts before
#      PMAT-10 — numeric, not lexicographic, or PMAT-10 would sort first).
#      This is enforced PER PREFIX, not as one global cross-prefix order:
#      the roadmap accreted over 18+ months with PMAT-, GH-, BENCH- and
#      PERF- id families interleaved by CREATION time, and a global
#      alphabetical-prefix order would require interleaving every PMAT
#      entry with every GH entry — a change with no bearing on the merge
#      conflict this guard actually prevents. A ticket is inserted next to
#      its own id-prefix peers, in ascending numeric order among them.
#   2. NO DUPLICATE ID. pmat's `work validate` (>= 3.39.0) refuses a
#      duplicate id outright (pmat#1169); this guard catches it locally,
#      first, with a line number.
#   3. NO TRACKED *.bak / *.lock. `pmat work` writes
#      docs/roadmaps/roadmap.yaml.bak and .lock as scratch files; if either
#      is ever `git add`-ed, every future edit races a stale copy in review
#      diffs. Checked via `git ls-files`, never presence-on-disk.
#
# LEGACY, PRE-ID-SCHEME ENTRIES ARE EXEMPT FROM RULE 1, NOT INVISIBLE. Before
# aprender adopted the PMAT-NNNN id convention, `id:` held the ticket's full
# title verbatim (e.g. `- id: Add APR-SPEC to book and verify examples`, one
# even a multi-line folded scalar). ~100 such entries exist in the current
# roadmap. They do not match the PREFIX-NUMBER shape at all, so there is no
# well-defined numeric position for them to violate — they are left exactly
# where they are (rule 1 does not apply to them), while rules 2 and 3 still
# cover them like any other entry.
#
#   bash scripts/check_roadmap_sorted.sh [ROADMAP.yaml]   # 0 sorted/clean · 1 violation · 2 env
#   bash scripts/check_roadmap_sorted.sh --self-test       # case table, no repo file touched
#
# Refs: PMAT-1065, BSE-09a, BSE-09b (measured conflict), pmat#1169.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ROADMAP="$REPO_ROOT/docs/roadmaps/roadmap.yaml"

usage() {
    printf 'usage: %s [ROADMAP.yaml]  or  %s --self-test\n' "$PROG" "$PROG" >&2
    exit 2
}

# extract_ids FILE -> "LINENO<TAB>ID" for every 0-indent `- id:` line, quotes
# stripped. Deliberately line-anchored (^- id:) so nested/indented subtask
# ids (check_roadmap_ids_unique.sh's job) are never matched here. A
# multi-line folded scalar id only contributes its FIRST line — sufficient
# for both checks below, since (a) a duplicate is decided the same way on
# both sides of the compare and (b) rule 1 never applies to a folded id (it
# can never match the single-line PREFIX-NUMBER shape).
extract_ids() {
    awk '
    /^- id:/ {
        line = $0
        sub(/^- id:[ \t]*/, "", line)
        n = length(line)
        if (n >= 2 && substr(line, 1, 1) == "\"" && substr(line, n, 1) == "\"") {
            line = substr(line, 2, n - 2)
        } else if (n >= 2 && substr(line, 1, 1) == "\x27" && substr(line, n, 1) == "\x27") {
            line = substr(line, 2, n - 2)
        }
        print NR "\t" line
    }
    ' "$1"
}

# scan_file FILE -> stdout PASS/FAIL line(s); rc 0 clean, 1 violation, 2 env
scan_file() {
    local f=$1 dir tracked lineno id prefix numeral key n_ids=0
    if [ ! -f "$f" ]; then
        printf '%s: ENV - %s is missing (the box cannot answer)\n' "$PROG" "$f" >&2
        return 2
    fi
    dir="$(dirname -- "$f")"

    # --- TRACKED-BAK-CHECK-BEGIN ---
    if ! tracked=$(git -C "$dir" ls-files -- '*.bak' '*.lock' 2>&1); then
        printf 'ENV   git ls-files failed under %s: %s — the tracked-backup check cannot be decided, refusing to pass\n' "$dir" "$tracked"
        return 2
    fi
    if [ -n "$tracked" ]; then
        printf 'FAIL  tracked backup/lock file(s) under %s (must be gitignored, never committed via git add):\n' "$dir"
        printf '%s\n' "$tracked" | sed 's/^/      /'
        return 1
    fi
    # --- TRACKED-BAK-CHECK-END ---

    local -A seen_line=()
    local -A last_numeral=()

    while IFS=$'\t' read -r lineno id; do
        [ -n "$lineno" ] || continue
        n_ids=$((n_ids + 1))

        # --- DUP-CHECK-BEGIN ---
        if [ -n "${seen_line[$id]+x}" ]; then
            printf 'FAIL  duplicate id %s at line %s (first seen at line %s) in %s\n' \
                "$id" "$lineno" "${seen_line[$id]}" "$f"
            return 1
        fi
        seen_line[$id]=$lineno
        # --- DUP-CHECK-END ---

        # --- SORT-CHECK-BEGIN ---
        if [[ "$id" =~ ^([A-Za-z][A-Za-z0-9_]*)-([0-9]+)([^0-9A-Za-z_].*)?$ ]]; then
            prefix="${BASH_REMATCH[1]}"
            numeral=$((10#${BASH_REMATCH[2]}))
            key="${last_numeral[$prefix]+x}"
            if [ -n "$key" ] && [ "$numeral" -lt "${last_numeral[$prefix]}" ]; then
                printf 'FAIL  %s-%s at line %s sorts before %s-%s earlier in the file — prefix %s must be numerically ascending among its own peers\n' \
                    "$prefix" "${BASH_REMATCH[2]}" "$lineno" "$prefix" "${last_numeral[$prefix]}" "$prefix"
                return 1
            fi
            last_numeral[$prefix]=$numeral
        fi
        # --- SORT-CHECK-END ---
    done < <(extract_ids "$f")

    printf 'PASS  %s: %s top-level id(s), sorted within each id-prefix, no duplicates, no tracked .bak/.lock\n' "$f" "$n_ids"
    return 0
}

# ---------------------------------------------------------------------------
# --self-test: fixtures under mktemp -d, no file in this repo is touched.
# ---------------------------------------------------------------------------
self_test() {
    local td rc n=0 red=0
    td=$(mktemp -d "${TMPDIR:-/tmp}/rmsorted-selftest.XXXXXX") || return 2
    cleanup() {
        local victim=${td:-}
        case "$victim" in
            *rmsorted-selftest.*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
            *) return 0 ;;
        esac
    }
    trap cleanup RETURN

    row() { # row WANT_RC LABEL MUST_MATCH FILE
        local want=$1 label=$2 pat=$3 f=$4 rc2=0 out
        n=$((n + 1))
        out=$(scan_file "$f" 2>&1); rc2=$?
        if [ "$rc2" = "$want" ] && printf '%s' "$out" | grep -qE -- "$pat"; then
            printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc2" "$label"
        else
            printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc2" "$want" "$pat" "$label"
            printf '%s\n' "$out" | sed 's/^/        /'
            red=1
        fi
    }

    mkdir -p "$td/sorted" "$td/unsorted" "$td/dup" "$td/bak" "$td/.empty-template"
    export GIT_TERMINAL_PROMPT=0
    for d in sorted unsorted dup bak; do
        # --template=<empty dir> so a locally-configured init.templatedir's
        # hooks are never copied into a throwaway fixture repo.
        git -C "$td/$d" init -q --template="$td/.empty-template"
        git -C "$td/$d" config user.email test@example.com
        git -C "$td/$d" config user.name test
    done

    cat >"$td/sorted/roadmap.yaml" <<'YAML'
roadmap_version: '1.0'
roadmap:
- id: PMAT-9
  title: nine
- id: PMAT-10
  title: ten
- id: Legacy freeform title, no id scheme
  title: legacy
- id: GH-3
  title: gh three
YAML

    cat >"$td/unsorted/roadmap.yaml" <<'YAML'
roadmap:
- id: PMAT-945
  title: nine-four-five
- id: PMAT-745
  title: seven-four-five
YAML

    cat >"$td/dup/roadmap.yaml" <<'YAML'
roadmap:
- id: PMAT-1
  title: one
- id: PMAT-1
  title: one again
YAML

    cat >"$td/bak/roadmap.yaml" <<'YAML'
roadmap:
- id: PMAT-1
  title: one
YAML
    cp "$td/bak/roadmap.yaml" "$td/bak/roadmap.yaml.bak"
    git -C "$td/bak" add roadmap.yaml roadmap.yaml.bak
    # `git ls-files` reads the index, not HEAD — no commit needed (and this
    # keeps a locally-configured init.templatedir's hooks out of the loop).

    row 0 "sorted fixture (mixed PMAT/GH prefixes + a legacy freeform id)" \
        '^PASS  .*4 top-level id' "$td/sorted/roadmap.yaml"
    row 1 "one out-of-order id (PMAT-945 before PMAT-745)" \
        'PMAT-745.*sorts before PMAT-945|FAIL.*sorts before' "$td/unsorted/roadmap.yaml"
    row 1 "duplicate id (PMAT-1 twice)" \
        'FAIL  duplicate id PMAT-1' "$td/dup/roadmap.yaml"
    row 1 "a tracked roadmap.yaml.bak" \
        'FAIL  tracked backup/lock file' "$td/bak/roadmap.yaml"
    row 2 "a missing file is ENV (exit 2), never a pass" \
        'ENV' "$td/absent/roadmap.yaml"

    printf '%s/%s checks, %s failed\n' "$((n - red))" "$n" "$red"
    [ "$red" = 0 ]
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    --self-test|--selftest) self_test; exit $? ;;
    --help|-h) usage ;;
    --*) usage ;;
esac

scan_file "${1:-$DEFAULT_ROADMAP}"
exit $?
