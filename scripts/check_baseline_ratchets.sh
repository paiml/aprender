#!/usr/bin/env bash
# check_baseline_ratchets.sh — every baseline in scripts/ is classified, and
# every one classified as shrink-only is compared against a ref this branch
# cannot rewrite.
#
# WHY THIS EXISTS
# ---------------
# Twelve guards in this repository described their baseline as "shrink-only",
# "monotonically decreasing", "may only fall", "the count is enforced, not
# asserted". A sweep that appended ONE entry to each — cloned from that file's
# own last real entry, so no scanner could tell it from a real one — found
# 12 of 12 green. On two of them the appended entry then laundered a fresh
# violation created in the same commit: guard rc=0, defect in the tree.
#
# The individual guards now enforce their own baseline (each sources
# scripts/lib_baseline_ratchet.sh). This file is the COVERAGE backstop, and it
# is not a second opinion:
#
#   1. THE UNIVERSE. A baseline that no guard ratchets is invisible to every
#      per-guard fix, and the way this class stays alive is a NEW baseline file
#      arriving unclassified. So the universe is derived at run time from the
#      working tree, unioned with the index, and an unclassified file is a HARD
#      FAILURE — never a silent skip.
#
#   2. THE FILES WITH NO SHELL OWNER. contract_duplicate_stem_baseline.txt is
#      read by Rust (`pv lint`, PV-DUP-001/002) and hardcoded_path_shipped_-
#      baseline.txt only by `check_hardcoded_paths.sh --full`, which no
#      workflow runs. Their shrink-only claim would be unenforced in CI if this
#      guard only checked the guards.
#
#   3. `git ls-files` ALONE IS A FREE PASS. A baseline present in the working
#      tree but not yet added is invisible to a tracked-only universe — the
#      shape that cost this repo four separate guards. The universe is the
#      UNION of a `find` over the working tree and the index.
#
# WHAT IT DOES NOT DO. It does not decide whether a finding is real; the owning
# guard does that. It decides only whether the recorded set grew.
#
#   bash scripts/check_baseline_ratchets.sh              # check
#   bash scripts/check_baseline_ratchets.sh --self-test  # case table
#
# Refs: paiml/aprender#2706 (APR-PERF-GATE-001), PERF-008, PERF-028.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1

# ---------------------------------------------------------------------------
# THE CLASSIFICATION. Every scripts/*.txt is either a shrink-only ratchet with
# a comparison kind, or is NOT one and says why. There is no third answer, and
# an unlisted file takes none of these branches — it fails.
#
#   set   — the entry LIST may only shrink. Subset, not count: counting passes
#           a swap, which is an append wearing the old total.
#   count — the file holds one integer, which may only fall.
#   keyed — lines are <path><TAB><count>; no key may rise, no key may appear.
#   none  — not a ratchet by its own stated contract; the reason is the value.
#   set-aperture
#         — `set`, plus the ONE admission a widened guard needs (PERF-049): an
#           added <path>:<line> whose line PREDATES the comparand — byte-identical
#           there, or MOVED and still present in the same file at no greater
#           count — in a diff that also changes the owning guard, named in the
#           value.
#           Everything else is refused as `set` refuses it, and every admitted
#           coordinate is printed. Without this the guard could never be
#           widened at all: `check_no_claim_literals.sh` could not see the
#           `2.93× Ollama` it was built for, and reading it reveals 18 claims
#           that were already in the tree. See lib_baseline_ratchet.sh.
classify() { # classify <basename> -> "<kind>[<TAB>reason]", rc 1 if unclassified
    case "$1" in
        assertion_exclusion_baseline.txt)        printf 'keyed\n' ;;
        claim_literal_baseline.txt)              printf 'set-aperture\tscripts/check_no_claim_literals.sh\n' ;;
        contract_duplicate_stem_baseline.txt)    printf 'set\n' ;;
        contract_test_binding_baseline.txt)      printf 'keyed\n' ;;
        fabricated_baseline_rust_sites.txt)      printf 'set\n' ;;
        hand_rolled_parsers_baseline.txt)        printf 'set\n' ;;
        hardcoded_path_shipped_baseline.txt)     printf 'count\n' ;;
        lockfile_registry_siblings_baseline.txt) printf 'set\n' ;;
        perf_claim_citation_baseline.txt)        printf 'set-aperture\tscripts/check_perf_claims_cite_receipts.sh\n' ;;
        roadmap_uncited_completion_baseline.txt) printf 'set\n' ;;
        shell_lint_baseline.txt)                 printf 'count\n' ;;
        test_fixture_path_baseline.txt)          printf 'count\n' ;;
        tracked_ignored_baseline.txt)            printf 'count\n' ;;
        unwired_guards_baseline.txt)             printf 'set\n' ;;
        # NOT a ratchet, and deliberately so. This file MODELS INTENT: its own
        # header says the declared set must match the OBSERVED set EXACTLY, an
        # entry whose duplicate no longer exists FAILS as stale, and adding a
        # line is a reviewed claim that two packages must ship one bin name.
        # Freezing it against main would forbid renaming a crate. Growth here
        # is a decision, not a leak — the distinction this guard exists to keep.
        duplicate_bin_names_allowlist.txt)
            printf 'none\tintent model, exact-match against the observed set (stale entries FAIL)\n' ;;
        *) return 1 ;;
    esac
}

# THE UNIVERSE, and where its edge is.
#
# `find` over the working tree UNION the index. Tracked-only is a free pass —
# an untracked file is invisible to `git ls-files`, and untracked is exactly
# how a new baseline arrives. That shape has cost this repository four separate
# guards, so it is not repeated here.
#
# The edge: baselines live at scripts/*.txt, depth 1. Deeper scripts/**/*.txt
# are case fixtures for guard self-tests (scripts/lib/facade_cases/*, etc.) and
# enumerating them as exemptions would be the same defect one level up. So the
# depth-1 set is taken whole, and a deeper file is pulled in ONLY if its name
# claims to be one of these — `baseline`, `allowlist` or `ledger` anywhere in
# it. Such a file arrives with its directory in the key, so classify() refuses
# it and the guard says so out loud rather than skipping it silently.
universe() {
    {
        find "$REPO_ROOT/scripts" -maxdepth 1 -type f -name '*.txt' -printf '%f\n' 2>/dev/null
        find "$REPO_ROOT/scripts" -mindepth 2 -type f -name '*.txt' 2>/dev/null \
            | sed "s|^${REPO_ROOT}/scripts/||"
        git -C "$REPO_ROOT" ls-files 'scripts/*.txt' 2>/dev/null | sed 's|^scripts/||'
    } | LC_ALL=C sort -u | grep -vE '^$' | while IFS= read -r rel; do
        case "$rel" in
            */*) printf '%s\n' "$rel" | grep -iE 'baseline|allowlist|ledger' ;;
            *)   printf '%s\n' "$rel" ;;
        esac
    done
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ] || [ "${1:-}" = "--selftest" ]; then
    TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${TD:?}"' EXIT
    bad=0
    rows=0

    say_row() { # say_row <label> <want-rc> <got-rc>
        rows=$((rows + 1))
        if [ "$2" != "$3" ]; then
            printf 'FAIL  %-46s want rc=%s got rc=%s\n' "$1" "$2" "$3"
            bad=1
        fi
    }

    # -- set: subset semantics, including the SWAP that a count would pass.
    printf '%b' 'a\nb\nc\n'        > "$TD/set_base"
    printf '%b' '# note\na\nb\nc\n' > "$TD/set_same"
    printf '%b' 'a\nb\nc\nd\n'     > "$TD/set_grow"
    printf '%b' 'a\nb\nd\n'        > "$TD/set_swap"
    printf '%b' 'a\nb\n'           > "$TD/set_shrink"
    printf '%b' ''                 > "$TD/set_empty"
    printf '%b' 'a\n'              > "$TD/set_one"
    _br_cmp_set "$TD/set_base" "$TD/set_same";    say_row 'set   unchanged (comment added)' 0 $?
    _br_cmp_set "$TD/set_base" "$TD/set_grow";    say_row 'set   one entry appended'        1 $?
    _br_cmp_set "$TD/set_base" "$TD/set_swap";    say_row 'set   swap, count unchanged'     1 $?
    _br_cmp_set "$TD/set_base" "$TD/set_shrink";  say_row 'set   one entry deleted'         0 $?
    _br_cmp_set "$TD/set_empty" "$TD/set_one";    say_row 'set   empty base, one entry'     1 $?
    _br_cmp_set "$TD/set_base" "$TD/set_empty";   say_row 'set   emptied entirely'          0 $?

    # -- count: one integer, may only fall.
    printf '876\n'            > "$TD/cnt_base"
    printf '# why\n876\n'     > "$TD/cnt_same"
    printf '877\n'            > "$TD/cnt_grow"
    printf '875\n'            > "$TD/cnt_shrink"
    printf 'not-a-number\n'   > "$TD/cnt_junk"
    _br_cmp_count "$TD/cnt_base" "$TD/cnt_same";   say_row 'count unchanged (comment added)' 0 $?
    _br_cmp_count "$TD/cnt_base" "$TD/cnt_grow";   say_row 'count raised by one'             1 $?
    _br_cmp_count "$TD/cnt_base" "$TD/cnt_shrink"; say_row 'count lowered by one'            0 $?
    _br_cmp_count "$TD/cnt_base" "$TD/cnt_junk";   say_row 'count unreadable in tree'        2 $?
    _br_cmp_count "$TD/cnt_junk" "$TD/cnt_base";   say_row 'count unreadable at comparand'   2 $?

    # -- keyed: per-key non-increase, no new keys. A DROPPED key is a shrink,
    #    and a raised key is refused even while another key falls further --
    #    the keyed form of the swap.
    printf 'a\t3\nb\t2\n'          > "$TD/kv_base"
    printf '# hdr\na\t3\nb\t2\n'   > "$TD/kv_same"
    printf 'a\t4\nb\t2\n'          > "$TD/kv_raise"
    printf 'a\t3\nb\t2\nc\t1\n'    > "$TD/kv_newkey"
    printf 'a\t2\nb\t2\n'          > "$TD/kv_lower"
    printf 'a\t3\n'                > "$TD/kv_drop"
    printf 'a\t4\nb\t0\n'          > "$TD/kv_swap"
    _br_cmp_keyed "$TD/kv_base" "$TD/kv_same";   say_row 'keyed unchanged (header added)' 0 $?
    _br_cmp_keyed "$TD/kv_base" "$TD/kv_raise";  say_row 'keyed one key raised'          1 $?
    _br_cmp_keyed "$TD/kv_base" "$TD/kv_newkey"; say_row 'keyed new key appears'         1 $?
    _br_cmp_keyed "$TD/kv_base" "$TD/kv_lower";  say_row 'keyed one key lowered'         0 $?
    _br_cmp_keyed "$TD/kv_base" "$TD/kv_drop";   say_row 'keyed one key dropped'         0 $?
    _br_cmp_keyed "$TD/kv_base" "$TD/kv_swap";   say_row 'keyed raise + deeper fall'     1 $?

    # -- THE COMPARAND RESOLVER, against a scratch repository. The branch that
    #    must NEVER be taken is "fall back to comparing this branch against
    #    itself", and it cannot be exercised from inside a checkout that
    #    already satisfies it. A missing comparand has to be provably LOUD.
    SR="$TD/scratch"
    P='scripts/probe_baseline.txt'
    if mkdir -p "$SR/scripts" \
       && git -C "$SR" init -q >/dev/null 2>&1 \
       && git -C "$SR" config user.email selftest@example.invalid \
       && git -C "$SR" config user.name selftest \
       && printf 'x\n' > "$SR/scripts/unrelated.txt" \
       && git -C "$SR" add -A \
       && git -C "$SR" -c commit.gpgsign=false commit -qm 'no baseline yet' >/dev/null 2>&1; then
        SR_NOBASE=$(git -C "$SR" rev-parse HEAD)
        printf '# header\na\nb\n' > "$SR/$P"
        if git -C "$SR" add -A \
           && git -C "$SR" -c commit.gpgsign=false commit -qm 'add baseline' >/dev/null 2>&1; then
            SR_BASE=$(git -C "$SR" rev-parse HEAD)

            res_row() { # res_row <label> <want-mode> <ref>
                local got
                got=$(baseline_ratchet_resolve "$SR" "$3" "$P")
                got=${got%%$'\t'*}
                rows=$((rows + 1))
                if [ "$got" != "$2" ]; then
                    printf 'FAIL  %-46s want %-12s got %s\n' "$1" "$2" "$got"
                    bad=1
                fi
            }
            res_row 'resolve ref does not exist'  UNRESOLVABLE 'refs/heads/no-such-branch-xyzzy'
            res_row 'resolve ref predates baseline' ABSENT     "$SR_NOBASE"
            res_row 'resolve ref carries baseline' MERGEBASE   "$SR_BASE"

            # BOOTSTRAP: the commit that INTRODUCES a baseline. Without this
            # verdict the first new baseline since this library landed is
            # blocked by the gate it arms -- neither protected ref can carry a
            # file that does not exist yet. It must be reachable ONLY for the
            # real protected ref AND only while the file is in the working
            # tree, or it becomes a way to disarm any ratchet by deleting its
            # baseline. Both edges are rows here.
            git -C "$SR" update-ref refs/remotes/origin/main "$SR_NOBASE" 2>/dev/null
            res_row 'resolve new baseline vs origin/main' BOOTSTRAP 'origin/main'
            mv "$SR/$P" "$SR/$P.hidden"
            res_row 'resolve absent from tree too stays ABSENT' ABSENT 'origin/main'
            mv "$SR/$P.hidden" "$SR/$P"
            # An OVERRIDDEN comparand keeps the loud branch: a stale or hand-picked
            # ref must never silently become a bootstrap.
            res_row 'resolve overridden ref never bootstraps' ABSENT "$SR_NOBASE"
            git -C "$SR" update-ref -d refs/remotes/origin/main 2>/dev/null

            # TIP is not decoration: CI checks out shallow, so the CI path IS
            # the tip path. Force it with a ref that shares NO history.
            if git -C "$SR" checkout -q --orphan unrelated >/dev/null 2>&1 \
               && git -C "$SR" rm -q -rf . >/dev/null 2>&1 \
               && mkdir -p "$SR/scripts" \
               && printf 'z\n' > "$SR/scripts/other.txt" \
               && git -C "$SR" add -A \
               && git -C "$SR" -c commit.gpgsign=false commit -qm 'orphan' >/dev/null 2>&1; then
                res_row 'resolve no common ancestor -> TIP' TIP "$SR_BASE"
                git -C "$SR" checkout -q "$SR_BASE" >/dev/null 2>&1
            else
                printf 'FAIL  resolve TIP fallback UNTESTED — the orphan branch could not\n'
                printf '      be built, and the CI path is the tip path. Not a skip.\n'
                bad=1
            fi

            # END TO END, through the public entry point, on a real repository:
            # append -> RED, delete -> GREEN, unchanged -> GREEN.
            e2e_row() { # e2e_row <label> <want-rc> <content>
                local got
                printf '%b' "$3" > "$SR/$P"
                ( BASELINE_RATCHET_BASE_REF="$SR_BASE" \
                  baseline_ratchet_check "$SR" "$P" set ) >/dev/null 2>&1
                got=$?
                rows=$((rows + 1))
                if [ "$got" != "$2" ]; then
                    printf 'FAIL  %-46s want rc=%s got rc=%s\n' "$1" "$2" "$got"
                    bad=1
                fi
            }
            e2e_row 'end-to-end unchanged'        0 '# header\na\nb\n'
            e2e_row 'end-to-end one entry appended' 1 '# header\na\nb\nc\n'
            e2e_row 'end-to-end swap at equal count' 1 '# header\na\nc\n'
            e2e_row 'end-to-end one entry deleted' 0 '# header\na\n'

            # -- set-aperture (PERF-049). The admission is narrow and every
            # branch of it must be shown to REFUSE, not just to admit. A rule
            # exercised only on its happy path is a rule nobody has tested.
            #
            # The scratch repo gets a "guard" and two source files. `pre.md`
            # exists at the comparand and is untouched here, so its line
            # PREDATES; `fresh.md` is written by the working tree only.
            printf 'the old claim, 2.93 times faster\n' > "$SR/pre.md"
            printf 'GUARD v1\n' > "$SR/guard.sh"
            git -C "$SR" add -A >/dev/null 2>&1
            git -C "$SR" -c commit.gpgsign=false commit -qm 'aperture base' >/dev/null 2>&1
            SR_APER=$(git -C "$SR" rev-parse HEAD)

            ap_row() { # ap_row <label> <want-rc> <baseline-content> <guard-content> <pre.md-content>
                local got
                printf '%b' "$3" > "$SR/$P"
                printf '%b' "$4" > "$SR/guard.sh"
                printf '%b' "$5" > "$SR/pre.md"
                printf 'written by this branch\n' > "$SR/fresh.md"
                ( BASELINE_RATCHET_BASE_REF="$SR_APER" \
                  baseline_ratchet_check "$SR" "$P" set-aperture guard.sh ) >/dev/null 2>&1
                got=$?
                rows=$((rows + 1))
                if [ "$got" != "$2" ]; then
                    printf 'FAIL  %-46s want rc=%s got rc=%s\n' "$1" "$2" "$got"
                    bad=1
                fi
            }
            AP_BASE='# header\npre.md:1\n'
            # The one thing it must ADMIT: a line that predates the comparand,
            # in a diff that moves the guard.
            ap_row 'aperture reveal admitted'            0 "$AP_BASE" 'GUARD v2\n' 'the old claim, 2.93 times faster\n'
            # (b) The aperture did not move -> nothing may be recorded. This is
            # what keeps "record it instead of fixing it" off ordinary PRs.
            ap_row 'aperture guard UNCHANGED refuses'    1 "$AP_BASE" 'GUARD v1\n' 'the old claim, 2.93 times faster\n'
            # (a) PERF-028's laundering shape, which is the whole point: the
            # entry and its matching violation in one commit.
            ap_row 'aperture line REWRITTEN refuses'     1 "$AP_BASE" 'GUARD v2\n' 'a claim this branch just wrote\n'
            # (a2), THE MOVE. PERF-019 inserted a 28-line subsection above two
            # already-baselined claims in docs/benchmarking-gate-spec.md, and
            # under coordinate-identity alone both became brand-new violations
            # with no legal remedy. The claim did not change; only its line did.
            ap_row 'aperture line MOVED down is admitted'  0 '# header\npre.md:2\n' 'GUARD v2\n' \
                'a line inserted above it\nthe old claim, 2.93 times faster\n'
            # ...and the hole (a2) could have opened, closed by counting: a
            # launderer COPIES a baselined claim to a second site in the same
            # file, where its text IS present at the comparand. The occurrence
            # count is what tells the two apart. If this row goes green, (a2)
            # buys a bookkeeping convenience at the price of the property
            # PERF-028 exists to enforce.
            ap_row 'aperture line DUPLICATED refuses'      1 '# header\npre.md:2\n' 'GUARD v2\n' \
                'the old claim, 2.93 times faster\nthe old claim, 2.93 times faster\n'
            # A MOVED line still needs the aperture to have moved: (a2) widens
            # (a), it does not bypass (b).
            ap_row 'aperture MOVED without a guard edit refuses' 1 '# header\npre.md:2\n' 'GUARD v1\n' \
                'a line inserted above it\nthe old claim, 2.93 times faster\n'
            ap_row 'aperture file absent at comparand'   1 '# header\nfresh.md:1\n' 'GUARD v2\n' 'the old claim, 2.93 times faster\n'
            ap_row 'aperture line past end of file'      1 '# header\npre.md:9\n'  'GUARD v2\n' 'the old claim, 2.93 times faster\n'
            ap_row 'aperture non-coordinate refuses'     1 '# header\nnot-a-coordinate\n' 'GUARD v2\n' 'the old claim, 2.93 times faster\n'
            ap_row 'aperture non-numeric line refuses'   1 '# header\npre.md:x\n' 'GUARD v2\n' 'the old claim, 2.93 times faster\n'
            # THIS ROW IS THE ONLY ONE THE NUMERIC CHECK CATCHES ALONE, and it
            # was found by mutating that check rather than by reading it:
            # deleting it left the table GREEN, because `pre.md:x` is refused
            # one branch later ("empty in BOTH copies") when sed errors. `$` is
            # not an error — it is sed's LAST-LINE address, so an entry reading
            # `pre.md:$` would compare the last line to itself and be ADMITTED
            # while naming no line at all.
            ap_row 'aperture sed address ($) refuses'    1 '# header\npre.md:$\n' 'GUARD v2\n' 'the old claim, 2.93 times faster\n'
            # Removal stays free, and an unchanged file stays green with no
            # guard edit at all -- otherwise the new kind would be strictly
            # WORSE than `set` on the ordinary path.
            ap_row 'aperture unchanged is green'         0 '# header\n' 'GUARD v1\n' 'the old claim, 2.93 times faster\n'
            # A missing owner argument must fail CLOSED. Called with no guard
            # path, "could not check" must never read as "no growth".
            printf '%b' "$AP_BASE" > "$SR/$P"
            printf 'GUARD v2\n' > "$SR/guard.sh"
            ( BASELINE_RATCHET_BASE_REF="$SR_APER" \
              baseline_ratchet_check "$SR" "$P" set-aperture ) >/dev/null 2>&1
            say_row 'aperture with NO owning guard refuses' 1 $?
            # And the admission must be LOUD. A silent one is this file's own
            # defect one level up.
            printf '%b' "$AP_BASE" > "$SR/$P"
            ap_out=$( BASELINE_RATCHET_BASE_REF="$SR_APER" \
                      baseline_ratchet_check "$SR" "$P" set-aperture guard.sh 2>&1 )
            rows=$((rows + 1))
            case "$ap_out" in
                *"APERTURE REVEAL"*"pre.md:1"*) : ;;
                *)  printf 'FAIL  an aperture reveal was admitted without NAMING it. A\n'
                    printf '      silent admission is the defect this file is about.\n'
                    bad=1 ;;
            esac
            git -C "$SR" checkout -q -- . >/dev/null 2>&1
            rm -f "$SR/fresh.md"
            printf '# header\na\nb\n' > "$SR/$P"

            # A REAL RED IS NOT A CRASH, and both exit 1. A comparator returns
            # 1 BY DESIGN on the growth path, so a caller running `set -e`
            # (check_no_claim_literals.sh does) can die AT the call instead of
            # reporting through it: identical status, zero verdict rows, and a
            # guard that looks armed while proving nothing. Assert the ROW.
            printf '%b' '# header\na\nb\nc\n' > "$SR/$P"
            e_out=$(BASELINE_RATCHET_BASE_REF="$SR_BASE" bash -euo pipefail -c \
                '. "$1"/scripts/lib_baseline_ratchet.sh || exit 9
                 baseline_ratchet_check "$2" "$3" set' \
                _ "$REPO_ROOT" "$SR" "$P" 2>&1)
            e_rc=$?
            say_row 'errexit caller still REPORTS the growth' 1 "$e_rc"
            rows=$((rows + 1))
            case "$e_out" in
                *GREW*) : ;;
                *)  printf 'FAIL  errexit caller produced rc=%s with NO verdict row. That is a\n' "$e_rc"
                    printf '      crash wearing a RED exit code, not enforcement.\n'
                    bad=1 ;;
            esac
            printf '' > "$SR/$P"

            # A DELETED baseline is not "no growth". Both directions must be loud.
            rm -f "$SR/$P"
            ( BASELINE_RATCHET_BASE_REF="$SR_BASE" \
              baseline_ratchet_check "$SR" "$P" set ) >/dev/null 2>&1
            say_row 'end-to-end baseline deleted from tree' 1 $?
            ( BASELINE_RATCHET_BASE_REF='refs/heads/no-such-branch-xyzzy' \
              baseline_ratchet_check "$REPO_ROOT" 'scripts/shell_lint_baseline.txt' count ) >/dev/null 2>&1
            say_row 'end-to-end unresolvable comparand' 1 $?
        else
            printf 'FAIL  scratch repo: could not commit, so UNRESOLVABLE/ABSENT/TIP are\n'
            printf '      UNTESTED. A selftest that silently skips its hardest case is the\n'
            printf '      defect this guard is about.\n'
            bad=1
        fi
    else
        printf 'FAIL  scratch repo: could not be built, so the branch that must never be\n'
        printf '      taken is UNTESTED. That is not a skip.\n'
        bad=1
    fi

    # -- the classification is TOTAL over the universe it will actually meet.
    unclassified=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        classify "$f" >/dev/null || unclassified=$((unclassified + 1))
    done <<< "$(universe)"
    rows=$((rows + 1))
    if [ "$unclassified" -ne 0 ]; then
        printf 'FAIL  %s baseline file(s) in this tree are unclassified\n' "$unclassified"
        bad=1
    fi
    rows=$((rows + 1))
    if classify 'a_brand_new_baseline.txt' >/dev/null 2>&1; then
        printf 'FAIL  an unknown baseline was CLASSIFIED. The table must be total over\n'
        printf '      what it lists and REFUSE everything else, or a new baseline\n'
        printf '      arrives unratcheted and this guard says nothing.\n'
        bad=1
    fi

    if [ "$bad" -ne 0 ]; then
        printf '\nSELF-TEST FAILED\n'
        exit 1
    fi
    printf 'PASS  case table only: %s rows (set, count, keyed, comparand resolver,\n' "$rows"
    printf '      end-to-end, classification totality). NO baseline in this tree was\n'
    printf '      compared — run with no arguments for that.\n'
    exit 0
fi

# ---------------------------------------------------------------------------
printf '=== every baseline in scripts/ is classified and ratcheted (check_baseline_ratchets.sh) ===\n'

rc=0
n_total=0
n_ratchet=0
n_none=0

while IFS= read -r f; do
    [ -n "$f" ] || continue
    n_total=$((n_total + 1))
    if ! entry=$(classify "$f"); then
        printf 'FAIL  %s is not classified.\n' "scripts/$f"
        printf '      Every baseline is either shrink-only (set / count / keyed) or is\n'
        printf '      NOT a ratchet and says why. An unclassified file is neither, and\n'
        printf '      "no rule" is how a baseline arrives that nothing ever compares.\n'
        printf '      Add it to classify() in %s.\n' "$(basename "$0")"
        rc=1
        continue
    fi
    kind=${entry%%$'\t'*}
    if [ "$kind" = none ]; then
        n_none=$((n_none + 1))
        printf 'ok    exempt   %-44s %s\n' "scripts/$f" "${entry#*$'\t'}"
        continue
    fi
    n_ratchet=$((n_ratchet + 1))
    baseline_ratchet_check "$REPO_ROOT" "scripts/$f" "$kind" "${entry#*$'\t'}" || rc=1
done <<< "$(universe)"

# Vacuity: a glob that matched nothing would compare nothing and look like a
# pass. That is the exact failure mode this guard is about.
if [ "$n_total" -lt 10 ]; then
    printf '\nFAIL (vacuity): only %s baseline file(s) found under scripts/, expected 10+.\n' "$n_total"
    printf 'The universe is broken, not the tree. Fix the scan rather than this number.\n'
    exit 1
fi

printf '\n%s baseline file(s): %s ratcheted, %s exempt with a stated reason\n' \
    "$n_total" "$n_ratchet" "$n_none"
if [ "$rc" -ne 0 ]; then
    printf 'FAIL  see rows above (#2706 PERF-028).\n'
else
    printf 'PASS  no shrink-only baseline grew against a ref this branch cannot rewrite.\n'
fi
exit "$rc"
