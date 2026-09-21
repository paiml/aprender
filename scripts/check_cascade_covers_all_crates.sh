#!/usr/bin/env bash
#
# check_cascade_covers_all_crates.sh — the release cascade must be able to SEE
# every crate it is supposed to ship, and must publish every crate AFTER the
# crates it needs on the registry.
#
# WHY THIS EXISTS (aprender#2559)
# -------------------------------
# `scripts/cascade-publish.sh` used to hand-maintain TIERS[]. On c22fe88ef that
# table held exactly the 70 publishable crates of the ROOT workspace — MEASURED,
# an exact match with no drift in either direction. It was complete with respect
# to the wrong universe: `crates/facades/` is a SECOND workspace, `exclude`d from
# the root, holding three publishable crates (`provable-contracts`, `-macros`,
# `-cli`). None appeared anywhere in the cascade, and FINAL VERIFICATION iterated
# the same table, so their absence printed as "✅ ALL crates at 0.63.0". The loop
# cannot iterate what it cannot see, so absence read as success.
#
# AND THE TABLE WAS NOT AN ORDER (aprender#3462)
# ----------------------------------------------
# MEASURED at v0.68.1: 47 (crate -> non-dev workspace dep) pairs sat with the dep
# in a LATER tier (aprender-core in T2 needs aprender-compute in T6; apr-cli in
# T10 needs eight T13 crates); at 225b2a9ab, 44 non-dev pairs and 1 versioned
# dev-dep. TIERS[] only ever published because cascade-drain.sh re-ran it until
# the deferrals stopped; under a stop-on-first-non-zero rule it stops at crate 13.
# The table is gone. The order is DERIVED (scripts/lib/cascade_universe.py
# --order) and cascade-publish.sh prints the sequence it walks (--print-order).
#
# WHAT IS CHECKED -- the sequence cascade-publish.sh WALKS, not the derivation
# ---------------------------------------------------------------------------
#   R0 VACUITY    the universe, the edge set and the sequence are all non-empty
#   R1 COVERAGE   every publishable crate in every workspace is in the sequence
#   R2 NO GHOSTS  every name in the sequence is a publishable crate, exactly once
#   R3 ORDER      every must-precede edge (a normal/build dep, or a VERSIONED
#                 dev-dep: cargo keeps it in the published manifest and resolves
#                 it on the registry, PMAT-955) points EARLIER in the sequence
#   R4 FACADES    every facade whose manifest pins a registry-resolved `upstream`
#                 (parsed from the manifest HERE, independently of the metadata
#                 the order was derived from) has that edge in the edge set AND
#                 publishes after it. A facade published first is a crate that
#                 cannot compile for anyone; in-tree it resolves through `path`,
#                 which is what hides it. R4 is the second instrument: an edge
#                 set that went blind to it would still pass R3.
#
#   R5 ARGUMENTS  (self-test) cascade-publish.sh's MODE allowlist, EXTRACTED from the real
#                 script and run on its own -- never the script itself, which would upload if the
#                 allowlist regressed: an unknown argument and the removed `--tier` exit 2 before
#                 any upload; the three real modes fall through; --print-order prints ORDER lines.
#                 A mutant with the catch-all arm deleted must turn the unknown-argument row RED.
#
#   bash scripts/check_cascade_covers_all_crates.sh              # check
#   bash scripts/check_cascade_covers_all_crates.sh --self-test  # case table
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIVERSE_PY="${REPO_ROOT}/scripts/lib/cascade_universe.py"

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
        local ver
        ver=$(sed -n 's/^upstream *=.*version *= *"\([^"]*\)".*/\1/p' "${dir}Cargo.toml" | head -1)
        grep -q . <<< "$ver" || continue
        printf '%s\t%s\n' "$(basename "$dir")" "$up"
    done
}

# The rules, over materialised files. Factored out so the self-test can drive
# them against fixtures rather than against the live repo — a case table that
# can only run on the real tree can only ever be green.
run_rules() {  # universe_names_file edges_file sequence_file repo_for_facades
    local uni="$1" edges="$2" seq="$3" root="$4" rc=0 missing ghosts dups viol fedges
    if [ ! -s "$uni" ]; then
        printf 'FAIL  R0 the crate universe is empty — the ENUMERATION is broken\n'; return 1
    fi
    if [ ! -s "$edges" ]; then
        printf 'FAIL  R0 the dependency edge set is empty — the EDGE READER is broken, and an\n'
        printf '      empty graph orders anything\n'; return 1
    fi
    if [ ! -s "$seq" ]; then
        printf 'FAIL  R0 no publish sequence was read (cascade-publish.sh --print-order) — the\n'
        printf '      READER is broken\n'; return 1
    fi

    missing="$(comm -23 <(sort -u "$uni") <(sort -u "$seq"))"
    if [ -n "$missing" ]; then
        printf 'FAIL  R1 publishable crate(s) absent from the publish sequence — the cascade\n'
        printf '      cannot ship what it cannot iterate:\n'
        printf '%s\n' "$missing" | sed 's/^/        /'
        rc=1
    else
        printf 'ok    R1 every publishable crate is in the publish sequence (%s)\n' "$(grep -c . "$seq")"
    fi

    ghosts="$(comm -13 <(sort -u "$uni") <(sort -u "$seq"))"
    dups="$(sort "$seq" | uniq -d)"
    if [ -n "$ghosts" ] || [ -n "$dups" ]; then
        [ -z "$ghosts" ] || { printf 'FAIL  R2 the sequence names crate(s) that are not publishable here (they defer\n'
                              printf '      forever, and the drain reports EXHAUSTED):\n'
                              printf '%s\n' "$ghosts" | sed 's/^/        /'; }
        [ -z "$dups" ] || { printf 'FAIL  R2 the sequence names crate(s) more than once:\n'
                            printf '%s\n' "$dups" | sed 's/^/        /'; }
        rc=1
    else
        printf 'ok    R2 every name in the sequence is a publishable crate, exactly once\n'
    fi

    # R3: position of each crate; every edge's dependency must sit strictly earlier.
    viol="$(awk -F'\t' 'NR == FNR { pos[$0] = FNR; next }
                        ($1 in pos) && ($2 in pos) && pos[$2] >= pos[$1] {
                            printf "%s (#%d) needs %s (#%d)\n", $1, pos[$1], $2, pos[$2] }' "$seq" "$edges")"
    if [ -n "$viol" ]; then
        printf 'FAIL  R3 the publish sequence is NOT a dependency order — each crate below would\n'
        printf '      be uploaded before a crate it needs on the registry:\n'
        printf '%s\n' "$viol" | sed 's/^/        /'
        rc=1
    else
        printf 'ok    R3 every must-precede edge (%s) points earlier in the sequence\n' "$(grep -c . "$edges")"
    fi

    fedges="$(facade_edges "$root")"
    if [ -z "$fedges" ]; then
        printf 'ok    R4 no facade declares a registry-resolved upstream (nothing to order)\n'
        return "$rc"
    fi
    while IFS=$'\t' read -r facade upstream; do
        [ -n "$facade" ] || continue
        if ! grep -qxF -- "$(printf '%s\t%s' "$facade" "$upstream")" "$edges"; then
            printf 'FAIL  R4 %s pins %s from the registry (its manifest says so), but the edge set\n' "$facade" "$upstream"
            printf '      the order is derived from does not carry that edge — the derivation is blind to it\n'
            rc=1; continue
        fi
        local fp up
        fp="$(grep -nxF -- "$facade" "$seq" | head -n 1 | cut -d: -f1)"
        up="$(grep -nxF -- "$upstream" "$seq" | head -n 1 | cut -d: -f1)"
        if [ -z "$fp" ] || [ -z "$up" ]; then
            printf 'FAIL  R4 %s -> %s: one of them is not in the sequence\n' "$facade" "$upstream"; rc=1
        elif [ "$fp" -gt "$up" ]; then
            printf 'ok    R4 %s (#%s) publishes after its upstream %s (#%s)\n' "$facade" "$fp" "$upstream" "$up"
        else
            printf 'FAIL  R4 %s (#%s) publishes BEFORE its upstream %s (#%s): a facade resolves its\n' "$facade" "$fp" "$upstream" "$up"
            printf '      upstream FROM THE REGISTRY, so published first it cannot compile for anyone\n'
            rc=1
        fi
    done <<< "$fedges"
    return "$rc"
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    TD="$(mktemp -d)" || exit 1
    case "$TD" in /tmp/*|/var/folders/*) : ;; *) printf 'bad tmp\n'; exit 1 ;; esac
    trap 'rm -rf "${TD:?}"' EXIT

    # A miniature repo: base <- mid <- top (normal deps), base <- test (a versioned
    # dev-dep edge), one facade that pins `up` from the registry, one signpost.
    mkdir -p "$TD/repo/crates/facades/face" "$TD/repo/crates/facades/signpost"
    printf 'upstream = { path = "../../up", version = "1.2.3", package = "up" }\n' \
        > "$TD/repo/crates/facades/face/Cargo.toml"
    printf '[package]\nname = "signpost"\n' > "$TD/repo/crates/facades/signpost/Cargo.toml"
    printf '%s\n' base mid top test up face signpost | sort -u > "$TD/uni"
    printf 'mid\tbase\ntop\tmid\ntest\tbase\nface\tup\n' > "$TD/edges"

    seq_() { printf '%s\n' "$@" > "$TD/seq"; }
    row() {  # name expect_rc needle [universe] [edges]
        local name="$1" want="$2" needle="$3" uni="${4:-uni}" edg="${5:-edges}" out rc
        out="$(run_rules "$TD/$uni" "$TD/$edg" "$TD/seq" "$TD/repo" 2>&1)"; rc=$?
        if [ "$rc" != "$want" ]; then
            printf 'FAIL  %s: exit %s, expected %s\n%s\n' "$name" "$rc" "$want" "$out"; fails=1; return
        fi
        if [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out"; then
            printf 'FAIL  %s: exit %s as expected but did not say "%s"\n%s\n' "$name" "$rc" "$needle" "$out"; fails=1; return
        fi
        printf 'ok    %s\n' "$name"
    }

    # Row 1 is the CONTROL: without a passing case every row below is satisfied by a
    # checker that fails unconditionally.
    seq_ base up mid test top signpost face
    row 'row 1 a complete dependency order passes' 0 ''

    # Row 2, THE #3462 MUTATION: the same names with two DEPENDENT crates swapped.
    seq_ base up top mid test signpost face
    row 'row 2 two dependent crates swapped (top before mid) is REJECTED' 1 'top (#3) needs mid (#4)'

    # Row 3: a versioned dev-dep is an edge too (PMAT-955).
    seq_ test base up mid top signpost face
    row 'row 3 a crate before its VERSIONED dev-dep is REJECTED' 1 'test (#1) needs base (#2)'

    # Row 4: coverage, reproduced (aprender#2559's live state on c22fe88ef).
    seq_ base up mid test top signpost
    row 'row 4 a publishable crate missing from the sequence is REJECTED' 1 'FAIL  R1'

    # Row 5: a ghost defers forever.
    seq_ base up mid test top signpost face ghostcrate
    row 'row 5 a name that is not publishable is REJECTED' 1 'ghostcrate'

    # Row 6: a crate walked twice.
    seq_ base up mid test top signpost face base
    row 'row 6 a crate named twice is REJECTED' 1 'more than once'

    # Row 7: the facade before its registry-resolved upstream.
    seq_ base face up mid test top signpost
    row 'row 7 a facade before its upstream is REJECTED' 1 'face (#2) needs up (#3)'

    # Row 8: the second instrument. The edge set lost the facade's edge (the metadata reader
    # went blind to it), so R3 alone passes a wrong order; the manifest parse must say so.
    printf 'mid\tbase\ntop\tmid\ntest\tbase\n' > "$TD/edges_blind"
    seq_ base face up mid test top signpost
    row 'row 8 an edge set blind to a facade upstream is REJECTED (R4)' 1 'the derivation is blind to it' uni edges_blind

    # Row 9: vacuity in each direction -- a checker that reads nothing must be RED.
    seq_ base up mid test top signpost face
    : > "$TD/empty"
    row 'row 9a an empty universe is REJECTED' 1 'the crate universe is empty' empty
    row 'row 9b an empty edge set is REJECTED' 1 'the dependency edge set is empty' uni empty
    : > "$TD/seq"
    row 'row 9c an empty sequence is REJECTED' 1 'no publish sequence was read'

    # R5: the MODE allowlist, extracted from the REAL cascade-publish.sh (never run whole: a regressed
    # allowlist there would publish). The block runs with a fixture ORDER; "FALLTHROUGH" means the
    # script would go on to the mode's own code.
    CASC="${REPO_ROOT}/scripts/cascade-publish.sh"
    awk '/^MODE="\$\{1:-publish\}"$/ {f = 1} f {print} f && /^esac$/ {exit}' "$CASC" > "$TD/modes.sh"
    if ! grep -q '^esac$' "$TD/modes.sh"; then
        printf 'FAIL  R5 no MODE allowlist block could be extracted from %s\n' "$CASC"; fails=1
    else
        mode_row() {  # name block expect_rc needle args...
            local name="$1" blk="$2" want="$3" needle="$4" out rc; shift 4
            # shift BEFORE sourcing: the block reads $1 as the MODE argument
            out="$(bash -c 'b=$1; shift; ORDER=(alpha beta); . "$b"; echo FALLTHROUGH' _ "$blk" "$@" 2>&1)"; rc=$?
            if [ "$rc" != "$want" ] || ! grep -qF -- "$needle" <<< "$out"; then
                printf 'FAIL  %s: exit %s (want %s), output: %s\n' "$name" "$rc" "$want" "$(tr '\n' '|' <<< "$out")"; fails=1; return 1
            fi
            printf 'ok    %s\n' "$name"
        }
        mode_row 'row 10a an unknown argument exits 2 before any upload' "$TD/modes.sh" 2 'Nothing was published' --chek
        mode_row 'row 10b the removed --tier exits 2 naming #3462' "$TD/modes.sh" 2 'derived, not tiered' --tier
        mode_row 'row 10c no argument falls through to the publish run' "$TD/modes.sh" 0 'FALLTHROUGH'
        mode_row 'row 10d --check falls through' "$TD/modes.sh" 0 'FALLTHROUGH' --check
        mode_row 'row 10e --order-check falls through' "$TD/modes.sh" 0 'FALLTHROUGH' --order-check
        mode_row 'row 10f --print-order prints the order and stops' "$TD/modes.sh" 0 'ORDER beta' --print-order
        # the MUTANT: the catch-all arm deleted, as before #3462 -- the unknown argument must now fall
        # through to the publish run, and row 10a's check must see it
        grep -v '^  \*) echo "ERROR: unknown argument' "$TD/modes.sh" > "$TD/modes_mut.sh"
        if cmp -s "$TD/modes.sh" "$TD/modes_mut.sh"; then
            printf 'FAIL  row 11 the catch-all anchor is gone -- the mutant is identical\n'; fails=1
        elif ( mode_row 'mutant' "$TD/modes_mut.sh" 2 'Nothing was published' --chek > /dev/null ); then
            printf 'FAIL  row 11 the catch-all-deleted mutant PASSED row 10a -- the row does not discriminate\n'; fails=1
        else
            printf 'ok    row 11 mutant: without the catch-all an unknown argument falls through to publish (row 10a RED)\n'
        fi
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (18/18)\n'
    exit 0
fi

# ---------------------------------------------------------------------------
printf '=== the release cascade must see every publishable crate, in dependency order ===\n\n'

W="$(mktemp -d)" || exit 1
case "$W" in /tmp/*|/var/folders/*) : ;; *) printf 'bad tmp\n'; exit 1 ;; esac
trap 'rm -rf "${W:?}"' EXIT
if ! python3 "$UNIVERSE_PY" --names "$REPO_ROOT" > "$W/uni"; then
    printf 'FAIL  the crate universe could not be enumerated; nothing was checked.\n'; exit 1
fi
if ! python3 "$UNIVERSE_PY" --edges "$REPO_ROOT" > "$W/edges"; then
    printf 'FAIL  the dependency edges could not be read; nothing was checked.\n'; exit 1
fi
bash "${REPO_ROOT}/scripts/cascade-publish.sh" --print-order > "$W/print" 2>&1; prc=$?
sed -n 's/^ORDER //p' "$W/print" > "$W/seq"
if [ "$prc" -ne 0 ]; then
    printf 'FAIL  scripts/cascade-publish.sh --print-order exited %s — the cascade has no order to walk:\n' "$prc"
    tail -n 3 "$W/print" | sed 's/^/        /'; exit 1
fi
printf '%s publishable crate(s) across all workspaces, %s must-precede edge(s)\n\n' "$(grep -c . "$W/uni")" "$(grep -c . "$W/edges")"

if run_rules "$W/uni" "$W/edges" "$W/seq" "$REPO_ROOT"; then
    printf '\nPASS  the cascade walks every publishable crate, in dependency order\n'
    exit 0
fi
printf '\nFAIL  see rows above\n'
exit 1
