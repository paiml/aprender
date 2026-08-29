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
classify() { # classify <basename> -> "<kind>[<TAB>reason]", rc 1 if unclassified
    case "$1" in
        assertion_exclusion_baseline.txt)        printf 'keyed\n' ;;
        claim_literal_baseline.txt)              printf 'set\n' ;;
        contract_duplicate_stem_baseline.txt)    printf 'set\n' ;;
        contract_test_binding_baseline.txt)      printf 'keyed\n' ;;
        fabricated_baseline_rust_sites.txt)      printf 'set\n' ;;
        hand_rolled_parsers_baseline.txt)        printf 'set\n' ;;
        hardcoded_path_shipped_baseline.txt)     printf 'count\n' ;;
        lockfile_registry_siblings_baseline.txt) printf 'set\n' ;;
        perf_claim_citation_baseline.txt)        printf 'set\n' ;;
        shell_lint_baseline.txt)                 printf 'count\n' ;;
        test_fixture_path_baseline.txt)          printf 'count\n' ;;
        tracked_ignored_baseline.txt)            printf 'count\n' ;;
        unwired_guards_baseline.txt)             printf 'set\n' ;;
        # NOT a ratchet either, and for a sharper reason: none of the three
        # shrink-only kinds can express this file. `count` wants one integer.
        # `keyed` wants <path><TAB><count>; the second field here is prose.
        # `set` compares the WHOLE data line, so rewording a reason reads as a
        # delete plus an add and FAILS — it would price honest edits to the
        # reasons, which are the only thing making the debt reviewable. MEASURED
        # 2026-08-29: classifying it `set` fails on this very branch, because
        # origin/main carries no such file and the ratchet rightly refuses a
        # missing comparand.
        #
        # What actually enforces it is stronger than shrink-only and is wired
        # into ci.yml by this batch ("Every declared pin field is consumed"):
        # check_pin_keys_consumed.sh ratchets the ledger in BOTH directions --
        # a dead key that is NOT listed FAILS (debt may not grow), and a listed
        # key that BECOMES consumed FAILS until its line is deleted (the ledger
        # may not rot into a list of things that are secretly fine). Shrink-only
        # forbids the first and is blind to the second. Both directions were
        # mutation-verified RED before this line was written.
        pin_unconsumed_ledger.txt)
            printf 'none\tledger of llama_pin.toml keys; check_pin_keys_consumed.sh ratchets it BOTH ways (dead-and-unlisted FAILS, listed-but-consumed FAILS)\n' ;;
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
    baseline_ratchet_check "$REPO_ROOT" "scripts/$f" "$kind" || rc=1
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
