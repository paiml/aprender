#!/usr/bin/env bash
# check_roadmap_fragment_parity.sh — a PR writes ONE roadmap fragment, named
# for its own ticket, and never touches the aggregate (PMAT-3296, #3296).
#
# THE RULE
#   docs/roadmaps/roadmap.yaml is GENERATED from docs/roadmaps/entries/ by
#   `make roadmap-aggregate`, post-merge on main. A pull request therefore:
#     * MUST NOT modify docs/roadmaps/roadmap.yaml, and
#     * every docs/roadmaps/entries/<ID>.yaml it adds or edits must have <ID>
#       equal to a `Pmat-Ticket:` trailer on one of its own commits.
#
# WHY IT IS OPT-IN ON entries/
#   This logic is destined for `pmat comply`, called from the SHARED
#   sovereign-ci.yml, which runs in every consumer repo. A repo that has not
#   migrated still writes roadmap.yaml legitimately, and failing it would be
#   the "upstream armed a gate, blocked every merge" shape. So: no
#   docs/roadmaps/entries/ directory => NOT-RUN, said out loud, never a
#   silent pass.
#
# NOT-RUN IS NOT A PASS. It prints what it did not measure, like roadmap-valid.
#
#   check_roadmap_fragment_parity.sh [--base REF] [--head REF]
#   check_roadmap_fragment_parity.sh --self-test
#
# Exit: 0 ok or NOT-RUN · 1 a violation · 2 usage/vacuity.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
ENTRIES_DIR="docs/roadmaps/entries"
AGGREGATE="docs/roadmaps/roadmap.yaml"

usage() { printf 'usage: %s [--base REF] [--head REF] | --self-test\n' "${0##*/}" >&2; exit 2; }

# check_parity <base> <head> -> 0 ok/NOT-RUN, 1 violation
check_parity() {
    local base="$1" head="$2" files trailers bad=0 n=0

    if [ ! -d "$ENTRIES_DIR" ]; then
        printf 'fragment-parity: NOT-RUN — no %s/ in this repo (not migrated)\n' "$ENTRIES_DIR"
        return 0
    fi

    files=$(git diff --name-only "$base" "$head" 2>/dev/null) || {
        printf 'fragment-parity: ENV — cannot diff %s..%s\n' "$base" "$head" >&2
        return 2
    }

    # 1. the aggregate is generated; a PR never writes it.
    if printf '%s\n' "$files" | grep -qx "$AGGREGATE"; then
        printf 'FAIL  %s is GENERATED — run `make roadmap-aggregate` on main, never edit it in a PR\n' "$AGGREGATE"
        bad=1
    fi

    # 2. every fragment touched must be named for one of this PR's own tickets.
    trailers=$(git log --format='%(trailers:key=Pmat-Ticket,valueonly)' "$base..$head" 2>/dev/null \
               | tr -d ' ' | grep -v '^$' | sort -u)
    while IFS= read -r f; do
        case "$f" in "$ENTRIES_DIR"/*.yaml) ;; *) continue ;; esac
        n=$((n + 1))
        local id="${f##*/}"; id="${id%.yaml}"
        if ! printf '%s\n' "$trailers" | grep -qx "$id"; then
            printf 'FAIL  %s has no matching Pmat-Ticket: %s trailer on this branch\n' "$f" "$id"
            bad=1
        fi
    done <<< "$files"

    [ "$bad" = 0 ] && printf 'ok    fragment-parity: %d fragment(s), every one named for a Pmat-Ticket on this branch\n' "$n"
    return "$bad"
}

# ------------------------------------------------------------------ case table

ROWS=0; REDS=0
row() { # row <want-rc> <name> <got-rc> <got-out> [<must-contain>]
    local want="$1" name="$2" got="$3" out="$4" needle="${5:-}"
    ROWS=$((ROWS + 1))
    if [ "$got" != "$want" ]; then
        printf 'FAIL  %-54s rc=%s want=%s\n' "$name" "$got" "$want"; REDS=$((REDS + 1)); return
    fi
    if [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out"; then
        printf 'FAIL  %-54s rc ok but did not say %s\n' "$name" "$needle"; REDS=$((REDS + 1)); return
    fi
    printf 'ok    %-54s\n' "$name"
}

TMPS=""
cleanup() { [ -n "$TMPS" ] && rm -rf $TMPS || true; }
trap cleanup EXIT

mk_repo() { # -> a throwaway repo on stdout
    local d; d=$(mktemp -d); TMPS="$TMPS $d"
    git -C "$d" init -q
    git -C "$d" config user.email t@t; git -C "$d" config user.name t
    # a hermetic fixture must not run the developer's global hooks
    git -C "$d" config core.hooksPath /dev/null
    mkdir -p "$d/docs/roadmaps"
    printf '%s\n' 'roadmap_version: 1' 'roadmap:' > "$d/docs/roadmaps/roadmap.yaml"
    git -C "$d" add -A; git -C "$d" commit -qm base
    printf '%s' "$d"
}

run_in() { # run_in <dir> <base> <head> -> prints output, sets RC
    local d="$1"; shift
    set +e; OUT=$(cd "$d" && bash "$SELF" --base "$1" --head "$2" 2>&1); RC=$?; set -e
}

self_test() {
    local d

    # 1. a repo that has not migrated is NOT-RUN, never a pass and never a fail.
    d=$(mk_repo)
    printf '%s\n' 'roadmap_version: 1' 'roadmap:' '- id: X' > "$d/docs/roadmaps/roadmap.yaml"
    git -C "$d" commit -qam "edit" 
    run_in "$d" HEAD~1 HEAD
    row 0 "no entries/ dir -> NOT-RUN (shared workflow, other repos)" "$RC" "$OUT" "NOT-RUN"

    # 2. a fragment named for a trailer on this branch
    d=$(mk_repo); mkdir -p "$d/docs/roadmaps/entries"
    printf '%s\n' '- id: PMAT-1' > "$d/docs/roadmaps/entries/PMAT-1.yaml"
    git -C "$d" add -A; git -C "$d" commit -qm "add" -m "Pmat-Ticket: PMAT-1"
    run_in "$d" HEAD~1 HEAD
    row 0 "fragment matching its Pmat-Ticket trailer" "$RC" "$OUT" "1 fragment"

    # 3. a fragment with no such trailer
    d=$(mk_repo); mkdir -p "$d/docs/roadmaps/entries"
    printf '%s\n' '- id: PMAT-2' > "$d/docs/roadmaps/entries/PMAT-2.yaml"
    git -C "$d" add -A; git -C "$d" commit -qm "add" -m "Pmat-Ticket: PMAT-9"
    run_in "$d" HEAD~1 HEAD
    row 1 "fragment whose id is NOT a trailer on this branch" "$RC" "$OUT" "no matching Pmat-Ticket"

    # 4. the aggregate is generated, so a PR never writes it
    d=$(mk_repo); mkdir -p "$d/docs/roadmaps/entries"
    printf '%s\n' '- id: PMAT-3' > "$d/docs/roadmaps/entries/PMAT-3.yaml"
    printf '%s\n' 'roadmap_version: 1' 'roadmap:' '- id: PMAT-3' > "$d/docs/roadmaps/roadmap.yaml"
    git -C "$d" add -A; git -C "$d" commit -qm "add" -m "Pmat-Ticket: PMAT-3"
    run_in "$d" HEAD~1 HEAD
    row 1 "PR edits the GENERATED aggregate -> refused" "$RC" "$OUT" "is GENERATED"

    # 5. a PR touching neither is fine, and says it measured zero
    d=$(mk_repo); mkdir -p "$d/docs/roadmaps/entries"; printf 'x\n' > "$d/unrelated.txt"
    git -C "$d" add -A; git -C "$d" commit -qm "unrelated" -m "Pmat-Ticket: PMAT-4"
    run_in "$d" HEAD~1 HEAD
    row 0 "a PR touching no fragment passes, reporting 0" "$RC" "$OUT" "0 fragment"

    # 6. MUTATION: drop the aggregate check; row 4 must turn RED.
    d=$(mk_repo); mkdir -p "$d/docs/roadmaps/entries"
    printf '%s\n' '- id: PMAT-5' > "$d/docs/roadmaps/entries/PMAT-5.yaml"
    printf '%s\n' 'roadmap_version: 1' 'roadmap:' '- id: PMAT-5' > "$d/docs/roadmaps/roadmap.yaml"
    git -C "$d" add -A; git -C "$d" commit -qm "add" -m "Pmat-Ticket: PMAT-5"
    # The mutation is applied by PYTHON, not sed: the line being replaced
    # contains a pipe, and every sed delimiter worth using appears in it.
    local mut; mut=$(mktemp); TMPS="$TMPS $mut"
    python3 - "$SELF" "$mut" <<'MUTPY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
needle = 'if printf \'%s\\n\' "$files" | grep -qx "$AGGREGATE"; then'
assert s.count(needle) == 1, "mutation target not found exactly once"
open(dst, "w").write(s.replace(needle, "if false; then"))
MUTPY
    set +e; OUT=$(cd "$d" && bash "$mut" --base HEAD~1 --head HEAD 2>&1); RC=$?; set -e
    row 0 "MUTATION: without the aggregate check the row 4 case PASSES" "$RC" "$OUT"
    rm -f "$mut"

    printf '\n%d row(s), %d red\n' "$ROWS" "$REDS"
    [ "$REDS" = 0 ]
}

BASE=""; HEAD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) self_test; exit $? ;;
        --base) BASE="${2:-}"; shift 2 ;;
        --head) HEAD="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[ -n "$BASE" ] || BASE="$(git merge-base origin/main HEAD 2>/dev/null)" || usage
[ -n "$HEAD" ] || HEAD=HEAD
check_parity "$BASE" "$HEAD"
