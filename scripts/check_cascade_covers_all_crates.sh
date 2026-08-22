#!/usr/bin/env bash
#
# check_cascade_covers_all_crates.sh — the release cascade must be able to SEE
# every crate it is supposed to ship, and must ship the facades AFTER the crates
# they resolve from the registry.
#
# WHY THIS EXISTS (aprender#2559)
# -------------------------------
# `scripts/cascade-publish.sh` hand-maintains TIERS[]. On c22fe88ef that table
# held exactly the 70 publishable crates of the ROOT workspace — MEASURED, an
# exact match with no drift in either direction:
#
#     $ comm -13 <TIERS[] names> <root publishable>   ->  (empty)
#     $ comm -23 <TIERS[] names> <root publishable>   ->  (empty)
#
# So every signal available said the table was complete. It was complete with
# respect to the wrong universe. `crates/facades/` is a SECOND workspace,
# `exclude`d from the root, holding three publishable crates carried forward
# from the APR-MONO rename — `provable-contracts` (10.8K downloads),
# `provable-contracts-macros` (46.6K) and `provable-contracts-cli`. None of the
# three appeared anywhere in the cascade, the drain, or the publish-safety
# scan, because all three built their universe from the root workspace and the
# root workspace does not contain them.
#
# Then the FINAL VERIFICATION loop iterated TIERS[] as well. A crate missing
# from the table was skipped by the publish loop AND by the loop that checks
# the publish loop, so its absence printed as
#
#     ✅ ALL crates at 0.63.0
#
# That is the shape this repository keeps paying for: the loop cannot iterate
# what it cannot see, so absence reads as success. It is not fixed by adding
# three names — it is fixed by deriving the universe from cargo across every
# workspace (scripts/lib/cascade_universe.py) and then FAILING when the
# hand-written ordering table and that universe disagree.
#
# ADDING A CRATE IS NOT THE FIX, EITHER
# -------------------------------------
# Names go stale in both directions. A crate deleted from the repo but left in
# TIERS[] never publishes and never can, so the cascade defers it every pass
# and the drain reports EXHAUSTED — a release that reads as broken while
# nothing is wrong. Both directions are checked.
#
# WHAT IS CHECKED
# ---------------
#   R1 COVERAGE   every publishable crate in every workspace appears in TIERS[]
#   R2 NO GHOSTS  every name in TIERS[] is a publishable crate that exists
#   R3 ORDER      a facade that resolves an upstream FROM THE REGISTRY sits in a
#                 strictly LATER tier than that upstream. Publishing a facade
#                 first yields a crate that cannot compile for anyone — inside
#                 this tree `upstream` resolves through its `path` and every
#                 build is green, which is exactly what hides it.
#
# R3 is the STATIC half of the ordering constraint. The DYNAMIC half lives in
# cascade-publish.sh's `facade_upstream_ready`, which refuses to upload a facade
# until the exact upstream version it requires is live on the sparse index. Two
# halves because the tier table can be right while a `--tier 14` invocation
# still runs first, and the registry check can be right while the table quietly
# rots. Neither subsumes the other.
#
#   bash scripts/check_cascade_covers_all_crates.sh              # check
#   bash scripts/check_cascade_covers_all_crates.sh --self-test  # case table
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIVERSE_PY="${REPO_ROOT}/scripts/lib/cascade_universe.py"

# Print "<tier>\t<crate>" for every crate named in a cascade script's TIERS[].
tier_rows() {
    grep -oE 'TIERS\[[0-9]+\]="[^"]*"' "$1" \
        | sed -E 's/^TIERS\[([0-9]+)\]="(.*)"$/\1 \2/' \
        | while read -r tier names; do
              for n in $names; do printf '%s\t%s\n' "$tier" "$n"; done
          done
}

# Print "<facade>\t<upstream>" for every facade manifest carrying an `upstream`
# dependency with a version requirement — i.e. every facade that resolves its
# upstream FROM THE REGISTRY when published. A facade with no such line (the
# lib-only signpost) imposes no ordering and is correctly absent here.
facade_edges() {
    local dir
    for dir in "$1"/crates/facades/*/; do
        [ -f "${dir}Cargo.toml" ] || continue
        local up
        up=$(sed -n 's/^upstream *=.*package *= *"\([^"]*\)".*/\1/p' "${dir}Cargo.toml" | head -1)
        [ -n "$up" ] || continue
        sed -n 's/^upstream *=.*version *= *"\([^"]*\)".*/\1/p' "${dir}Cargo.toml" | head -1 \
            | grep -q . || continue
        printf '%s\t%s\n' "$(basename "$dir")" "$up"
    done
}

# The three rules, over an already-materialised universe file and cascade file.
# Factored out so the self-test can drive them against fixtures rather than
# against the live repo — a case table that can only run on the real tree can
# only ever be green, which is the fail mode this whole file is about.
run_rules() {  # universe_names_file cascade_file repo_for_edges
    local uni="$1" casc="$2" root="$3" rc=0 rows missing ghosts
    rows="$(tier_rows "$casc")"

    # Vacuity: a cascade whose table did not parse would report zero ghosts and
    # every crate missing, or with an empty universe, nothing missing at all.
    if [ -z "$rows" ]; then
        printf 'FAIL  R0 no TIERS[] rows parsed from %s — the READER is broken\n' "$casc"
        return 1
    fi
    if [ ! -s "$uni" ]; then
        printf 'FAIL  R0 the crate universe is empty — the ENUMERATION is broken\n'
        return 1
    fi

    missing="$(comm -23 <(sort -u "$uni") <(printf '%s\n' "$rows" | cut -f2 | sort -u))"
    if [ -n "$missing" ]; then
        printf 'FAIL  R1 publishable crate(s) absent from TIERS[] — the cascade cannot\n'
        printf '      ship what it cannot iterate, and FINAL VERIFICATION iterates the\n'
        printf '      same table, so this prints as success:\n'
        printf '%s\n' "$missing" | sed 's/^/        /'
        rc=1
    else
        printf 'ok    R1 every publishable crate appears in TIERS[]\n'
    fi

    ghosts="$(comm -13 <(sort -u "$uni") <(printf '%s\n' "$rows" | cut -f2 | sort -u))"
    if [ -n "$ghosts" ]; then
        printf 'FAIL  R2 TIERS[] names crate(s) that are not publishable here. They\n'
        printf '      DEFER on every pass, so the drain reports EXHAUSTED and a healthy\n'
        printf '      release reads as broken:\n'
        printf '%s\n' "$ghosts" | sed 's/^/        /'
        rc=1
    else
        printf 'ok    R2 every name in TIERS[] is a publishable crate\n'
    fi

    local edges
    edges="$(facade_edges "$root")"
    if [ -z "$edges" ]; then
        printf 'ok    R3 no facade declares a registry-resolved upstream (nothing to order)\n'
        return "$rc"
    fi
    while IFS=$'\t' read -r facade upstream; do
        [ -n "$facade" ] || continue
        local ft ut
        ft="$(printf '%s\n' "$rows" | awk -F'\t' -v c="$facade"   '$2==c{print $1; exit}')"
        ut="$(printf '%s\n' "$rows" | awk -F'\t' -v c="$upstream" '$2==c{print $1; exit}')"
        if [ -z "$ft" ] || [ -z "$ut" ]; then
            printf 'FAIL  R3 %s -> %s: one of them is not tiered (facade=%s upstream=%s)\n' \
                "$facade" "$upstream" "${ft:-none}" "${ut:-none}"
            rc=1
        elif [ "$ft" -gt "$ut" ]; then
            printf 'ok    R3 %s (T%s) publishes after its upstream %s (T%s)\n' \
                "$facade" "$ft" "$upstream" "$ut"
        else
            printf 'FAIL  R3 %s is in T%s but its upstream %s is in T%s. A facade\n' \
                "$facade" "$ft" "$upstream" "$ut"
            printf '      resolves its upstream FROM THE REGISTRY; published first it is a\n'
            printf '      crate that cannot compile for anyone. Inside this tree it resolves\n'
            printf '      through `path` and looks fine, which is what hides it.\n'
            rc=1
        fi
    done <<< "$edges"
    return "$rc"
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    TD="$(mktemp -d)" || exit 1
    case "$TD" in /tmp/*|/var/folders/*) : ;; *) printf 'bad tmp\n'; exit 1 ;; esac
    trap 'rm -rf "${TD:?}"' EXIT

    # A miniature repo: two upstreams, one facade that pins one of them, one
    # signpost facade with no upstream at all.
    mkdir -p "$TD/repo/crates/facades/face" "$TD/repo/crates/facades/signpost"
    printf 'upstream = { path = "../../up", version = "1.2.3", package = "up" }\n' \
        > "$TD/repo/crates/facades/face/Cargo.toml"
    printf '[package]\nname = "signpost"\n' \
        > "$TD/repo/crates/facades/signpost/Cargo.toml"
    printf 'up\nother\nface\nsignpost\n' | sort -u > "$TD/uni_good"

    mkcasc() { printf '%s\n' "$@" > "$TD/casc"; }

    row() {  # name expect_rc universe needle
        local name="$1" want="$2" uni="$3" needle="$4" out rc
        out="$(run_rules "$TD/$uni" "$TD/casc" "$TD/repo" 2>&1)"; rc=$?
        if [ "$rc" != "$want" ]; then
            printf 'FAIL  %s: exit %s, expected %s\n%s\n' "$name" "$rc" "$want" "$out"
            fails=1; return
        fi
        if [ -n "$needle" ] && ! grep -q -- "$needle" <<< "$out"; then
            printf 'FAIL  %s: exit %s as expected but did not name %s\n%s\n' \
                "$name" "$rc" "$needle" "$out"
            fails=1; return
        fi
        printf 'ok    %s\n' "$name"
    }

    # Row 1 is the CONTROL. Without a passing case every row below is satisfied
    # by a checker that fails unconditionally.
    mkcasc 'TIERS[1]="up other"' 'TIERS[2]="face signpost"'
    row 'row 1 a complete, correctly ordered table passes' 0 uni_good ''

    # Row 2 is the DEFECT, reproduced: a publishable crate the table cannot see.
    # This is the mutation that must turn RED — on c22fe88ef it was the live
    # state of the repo and nothing anywhere reported it.
    mkcasc 'TIERS[1]="up other"' 'TIERS[2]="signpost"'
    row 'row 2 a publishable crate missing from TIERS[] is REJECTED' 1 uni_good 'FAIL  R1'

    # Row 3: the reverse drift. A name with nothing behind it defers forever.
    mkcasc 'TIERS[1]="up other ghostcrate"' 'TIERS[2]="face signpost"'
    row 'row 3 a TIERS[] name that is not publishable is REJECTED' 1 uni_good 'FAIL  R2'

    # Row 4: the ORDER constraint. Same names, same coverage, wrong sequence —
    # so this row cannot pass by accident on coverage alone.
    mkcasc 'TIERS[1]="face signpost"' 'TIERS[2]="up other"'
    row 'row 4 a facade tiered BEFORE its upstream is REJECTED' 1 uni_good 'FAIL  R3'

    # Row 5: equal tiers are also rejected. Within one tier the cascade walks a
    # shell word list, so "same tier" is an ordering by luck, not by rule — the
    # class of defect where a verdict is decided by something nobody declared.
    mkcasc 'TIERS[1]="up other face signpost"'
    row 'row 5 a facade in the SAME tier as its upstream is REJECTED' 1 uni_good 'FAIL  R3'

    # Row 6: vacuity in each direction. A checker that reads nothing must be RED,
    # never a clean pass over nothing.
    : > "$TD/uni_empty"
    mkcasc 'TIERS[1]="up other"' 'TIERS[2]="face signpost"'
    row 'row 6a an empty universe is REJECTED, not a clean pass' 1 uni_empty 'FAIL  R0'
    mkcasc 'nothing here parses as a tier'
    row 'row 6b an unparsable cascade table is REJECTED' 1 uni_good 'FAIL  R0'

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (7/7)\n'
    exit 0
fi

# ---------------------------------------------------------------------------
printf '=== the release cascade must see every publishable crate ===\n\n'

UNI="$(mktemp)"
trap 'rm -f "${UNI:?}"' EXIT
if ! python3 "$UNIVERSE_PY" --names "$REPO_ROOT" > "$UNI"; then
    printf 'FAIL  the crate universe could not be enumerated; nothing was checked.\n'
    exit 1
fi
printf '%s publishable crate(s) across all workspaces\n\n' "$(grep -c . "$UNI")"

if run_rules "$UNI" "${REPO_ROOT}/scripts/cascade-publish.sh" "$REPO_ROOT"; then
    printf '\nPASS  the cascade covers and correctly orders every publishable crate\n'
    exit 0
fi
printf '\nFAIL  see rows above\n'
exit 1
