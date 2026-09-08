#!/usr/bin/env bash
# check_guards_are_wired.sh — every scripts/check_*.sh must be named by at least
# one GitHub workflow.
#
# WHY THIS EXISTS
# ---------------
# A guard that no workflow invokes is a file that looks like enforcement and is
# not reachable by any automated path. Four were found this way, by accident,
# while looking for something else (#2512):
#
#   check_contract_test_binding.sh   ci=0  makefile=2
#   check_wasm32_core_builds.sh      ci=0  makefile=1
#   check_book_examples_executable.sh ci=0 makefile=0   <- invoked by NOTHING
#   check_package_includes.sh        ci=0  makefile=0   <- invoked by NOTHING
#
# Makefile-only means `make tier3`, which is not run in CI. The bottom two were
# reachable from nothing at all.
#
# `check_package_includes.sh` is the sharp one: it is the CB-510 guard, written
# because a `models/` pattern matched `src/models/` and hid source from
# crates.io. Its own header instructs the reader to run it after any `.gitignore`
# or `Cargo.toml` exclude change. Its sibling `check_include_files.sh` IS wired.
# Nothing enforced the instruction.
#
# This is the meta-guard: without it, the next one to go dark is found the same
# way these were.
#
# A shrink-only baseline holds the exemptions that were already argued for.
# SHRINK-ONLY IS NOW LITERAL: the list is compared against origin/main, so an
# entry may only LEAVE it. Recording a NEW exemption is refused, not merely
# discouraged — for one commit this header said "can be recorded rather than
# argued about" while nothing compared the file to anything, and appending a
# line here returned rc=0. The remedy for an unwired guard is to wire it.
#
#   bash scripts/check_guards_are_wired.sh              # check
#   bash scripts/check_guards_are_wired.sh --self-test  # case table
#   bash scripts/check_guards_are_wired.sh --update     # re-baseline

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/unwired_guards_baseline.txt"

# dispatcher_wired ROOT -- guards a workflow wires by DISPATCHING them.
#
# WHY NAME-MATCHING STOPPED BEING THE WHOLE ANSWER
# -------------------------------------------------
# ci.yml used to carry ~53 `bash scripts/check_X.sh` steps, so "named by a
# workflow" and "run by a workflow" were the same sentence. BSE-01 replaced
# them with ONE step -- `bash scripts/guard_tree.sh --no-cargo` -- because
# GitHub Actions steps are fail-fast and each red run named exactly one guard.
# Those 53 guards are run by CI on every PR and are named by nothing, and this
# meta-guard read that as 29 newly dark guards (4 -> 33). A guard that reports
# enforcement as absence is worse than no guard.
#
# The remedy is NOT a list of the dispatched names here -- that is the same
# hand-maintained-second-list defect this file's own header argues against, and
# it would go stale the first time a guard is added. The subset is DERIVED by
# asking the dispatcher, in the mode the workflow actually invokes it in.
#
# --dry-run, NOT --list. `--list` prints the raw cargo-classified subset;
# --dry-run prints the DECISION, so the guards guard_tree.sh skips (release-
# time, or argument/env-wired) come back as `skipped:` rows and are NOT
# counted wired by the dispatcher. Reporting a skipped guard as wired would be
# this repo's signature defect -- claiming a result that was never measured --
# committed by the very guard that exists to catch it. An argument-wired guard
# is still named by its own workflow and is found by the scan below on that
# account; a release-time guard is deliberately unwired and stays reported.
dispatcher_wired() {
    local root="$1" lines modes mode flag out
    [ -f "$root/scripts/guard_tree.sh" ] || return 0
    lines=$(grep -rh --include='*.yml' --include='*.yaml' -- 'guard_tree.sh' \
                "$root"/.github/workflows/ 2>/dev/null | sed 's/#.*$//') || lines=''
    # Invocation, not mention -- same test as the scan below.
    lines=$(grep -E "(^|[[:space:];&|(])((ba)?sh[[:space:]]+|\\./)?[^[:space:]]*guard_tree\\.sh([[:space:]]|$|['\"])" \
                <<< "$lines") || lines=''
    [ -n "$lines" ] || return 0
    modes=$(
        while IFS= read -r l; do
            case "$l" in
                *--no-cargo*)   printf 'no-cargo\n' ;;
                *--cargo-only*) printf 'cargo-only\n' ;;
                *)              printf 'all\n' ;;
            esac
        done <<< "$lines" | sort -u
    )
    for mode in $modes; do
        case "$mode" in
            no-cargo)   flag="--no-cargo" ;;
            cargo-only) flag="--cargo-only" ;;
            *)          flag= ;;
        esac
        out=$( cd "$root" && bash scripts/guard_tree.sh --dry-run $flag 2>/dev/null ) || out=''
        printf '%s\n' "$out" | sed -n 's|^run: .*/||p; s|^run: \([^/]*\)$|\1|p'
    done | LC_ALL=C sort -u
}

# Guards named by no workflow, one per line, sorted.
unwired_in() {
    local root="$1" g base seen="" dispatched
    dispatched=" $(dispatcher_wired "$root" | tr '\n' ' ') "
    # THE UNIVERSE WAS BUILT FROM THE FILENAME, AND A GUARD HID BEHIND ITS OWN.
    #
    # This globbed scripts/check_*.sh only. scripts/perf_gate.sh — which
    # implements Arms A/B1/B2/C/D/E and cell-completeness for
    # APR-PERF-GATE-001, the epic the operator calls the most important gate in
    # the project — is invoked by NO workflow, NO Makefile target and NO
    # [package.metadata.dogfood] entry, and this meta-guard reported PASS the
    # whole time because the file is not called check_*.
    #
    # Proven by mutation rather than argued: copying that file BYTE-FOR-BYTE to
    # scripts/check_perf_gate.sh makes this guard fail immediately —
    # "unwired guards grew 4 -> 5 / NEW: check_perf_gate.sh". Same content, same
    # non-wiring; only the name differed.
    #
    # So the universe is DERIVED, not enumerated: a script that ships a
    # `--selftest` / `--self-test` mode is CLAIMING to be a guard, and that
    # claim is what makes it one. Hardcoding a second list of "guards that are
    # not called check_*" would be the same defect one level up — the shape
    # already fixed three times in this epic (cascade TIERS[], book.yml paths,
    # book example features).
    for g in "$root"/scripts/check_*.sh "$root"/scripts/*.sh; do
        [ -f "$g" ] || continue
        case "$(basename "$g")" in
            check_*) ;;   # always in scope
            *) grep -qE '^[[:space:]]*(--selftest|--self-test|"--selftest"|"--self-test")\)' "$g" 2>/dev/null || continue ;;
        esac
        case " $seen " in *" $(basename "$g") "*) continue ;; esac
        seen="$seen $(basename "$g")"
        base=$(basename "$g")
        # Wired by dispatch: a workflow runs guard_tree.sh in a mode whose RUN
        # set contains this guard. No name appears anywhere; the wiring is real.
        case "$dispatched" in
            *" $base "*) continue ;;
        esac
        # EXECUTION, not mention. `grep -rqF -- "$base"` matched the script's
        # NAME anywhere in the workflows tree -- including inside a `#` comment.
        # So a guard could be documented and never run, and this meta-guard,
        # whose entire purpose is to catch exactly that, reported it as wired.
        #
        # Require a non-comment line that INVOKES it: `bash scripts/x.sh`,
        # `sh scripts/x.sh`, `./scripts/x.sh`, or a bare `scripts/x.sh` as a
        # command.
        #
        # `sed 's/#.*$//'` strips from the FIRST `#`, not just whole-line
        # comments. My first version filtered only lines matching `^\s*#`, and
        # a TRAILING comment defeated it: `run: echo skipped # bash
        # scripts/check_msrv.sh` still matched and reported the guard as wired.
        # That is the same mention-vs-execution defect this function exists to
        # fix, one level down, and it was caught by mutation-testing the fix
        # rather than by reading it. Erring strict is correct here: a `#` inside
        # a quoted YAML string would make a wired guard look unwired, which is a
        # loud false alarm rather than a silent miss.
        local mentions
        mentions=$(grep -rh --include='*.yml' --include='*.yaml' -- "$base" \
                "$root"/.github/workflows/ 2>/dev/null \
             | sed 's/#.*$//') || mentions=''
        if ! grep -qE "(^|[[:space:];&|(])((ba)?sh[[:space:]]+|\\./)?[^[:space:]]*${base}([[:space:]]|$|['\"])" <<< "$mentions" ; then
            printf '%s\n' "$base"
        fi
    done | LC_ALL=C sort
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${TD:?}"' EXIT
    fails=0
    mkdir -p "$TD/scripts" "$TD/.github/workflows"
    : > "$TD/scripts/check_wired.sh"
    : > "$TD/scripts/check_dark.sh"
    printf 'jobs:\n  x:\n    steps:\n      - run: bash scripts/check_wired.sh\n' \
        > "$TD/.github/workflows/ci.yml"

    got=$(unwired_in "$TD" | tr '\n' ' ')
    if [ "$got" = "check_dark.sh " ]; then
        printf 'ok    row 1 the unwired guard is reported, the wired one is not\n'
    else
        printf 'FAIL  row 1 got [%s], expected [check_dark.sh ]\n' "$got"; fails=1
    fi

    # Row 2 is the control: wire it up and the report must go EMPTY. Without
    # this, row 1 passes even if the scan reported every guard it saw.
    printf '      - run: bash scripts/check_dark.sh\n' >> "$TD/.github/workflows/ci.yml"
    if [ -z "$(unwired_in "$TD")" ]; then
        printf 'ok    row 2 wiring it clears the report\n'
    else
        printf 'FAIL  row 2 still reports: %s\n' "$(unwired_in "$TD" | tr '\n' ' ')"; fails=1
    fi

    # ── Rows 3 and 4: WIRED BY DISPATCH, named by nothing ──────────────────
    #
    # The fixture workflow contains ONE line and no guard name anywhere:
    #     bash scripts/guard_tree.sh --no-cargo
    # That is literally what ci.yml does since BSE-01, and it is what took this
    # guard from 4 unwired to 33. Row 3 asserts the dispatched guard is not
    # reported; row 4 deletes the dispatcher line from the SAME fixture and
    # asserts it comes back — without row 4, row 3 passes just as well for a
    # scan that reports nothing at all.
    #
    # A real git repo, because guard_tree.sh derives its universe from
    # `git ls-files` and would otherwise answer nothing — and answering nothing
    # is indistinguishable here from answering "wired".
    TD2=$(mktemp -d) || exit 1
    trap 'rm -rf "${TD:?}" "${TD2:?}"' EXIT
    mkdir -p "$TD2/scripts" "$TD2/.github/workflows"
    cp "$REPO_ROOT/scripts/guard_tree.sh" "$TD2/scripts/guard_tree.sh"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$TD2/scripts/check_dispatched.sh"
    printf 'jobs:\n  gate:\n    steps:\n      - run: bash scripts/guard_tree.sh --no-cargo\n' \
        > "$TD2/.github/workflows/ci.yml"
    mkdir -p "$TD2/.empty-git-template"
    git -C "$TD2" init -q --template="$TD2/.empty-git-template"
    git -C "$TD2" config user.email test@example.invalid
    git -C "$TD2" config user.name "check_guards_are_wired self-test"
    git -C "$TD2" add -A
    git -C "$TD2" -c commit.gpgsign=false commit -q -m fixture

    if [ -z "$(unwired_in "$TD2")" ]; then
        printf 'ok    row 3 a guard dispatched by guard_tree.sh --no-cargo counts as wired\n'
    else
        printf 'FAIL  row 3 dispatched guard still reported unwired: [%s]\n' "$(unwired_in "$TD2" | tr '\n' ' ')"; fails=1
    fi

    printf 'jobs:\n  gate:\n    steps:\n      - run: echo no dispatcher here\n' \
        > "$TD2/.github/workflows/ci.yml"
    got4=$(unwired_in "$TD2" | tr '\n' ' ')
    if [ "$got4" = "check_dispatched.sh " ]; then
        printf 'ok    row 4 removing the dispatcher line brings the guard back as unwired\n'
    else
        printf 'FAIL  row 4 got [%s], expected [check_dispatched.sh ]\n' "$got4"; fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (4/4)\n'
    exit 0
fi

printf '=== every check_*.sh must be named by a workflow (check_guards_are_wired.sh) ===\n'

# COUNT THE UNIVERSE THAT WAS ACTUALLY SCANNED, not a proxy for it. This
# counted check_*.sh only, so after the universe was widened it printed
# "72 guard(s) scanned, 5 named by no workflow" over a 73-file universe — a
# self-inconsistent line, and the exact shape of a number that misleads whoever
# reads it next. Derived the same way unwired_in() derives its universe.
total=$(
  {
    find "$REPO_ROOT/scripts" -maxdepth 1 -name 'check_*.sh'
    for f in "$REPO_ROOT"/scripts/*.sh; do
      case "$(basename "$f")" in check_*) continue ;; esac
      grep -qE '^[[:space:]]*(--selftest|--self-test|"--selftest"|"--self-test")\)' "$f" 2>/dev/null \
        && printf '%s\n' "$f"
    done
  } | sort -u | wc -l | tr -d ' ')

# Vacuity: a glob that matched nothing would report zero unwired guards and look
# like a pass. That is the exact failure mode this guard is about.
if [ "$total" -lt 20 ]; then
    printf '\nFAIL (vacuity): only %s guard(s) found under scripts/, expected 20+.\n' "$total"
    printf 'The scan is broken, not the wiring. Fix it rather than this number.\n'
    exit 1
fi

FOUND=$(unwired_in "$REPO_ROOT")
count=$(printf '%s\n' "$FOUND" | grep -c . || true)

printf '%s guard(s) scanned, %s named by no workflow\n' "$total" "$count"

if [ "${1:-}" = "--update" ]; then
    printf '%s\n' "$FOUND" | grep . > "$BASELINE" || : > "$BASELINE"
    printf 'baseline set to %s\n' "$count"
    exit 0
fi

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything above compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that is not a ratchet. NEW (a finding with no entry) and
# STALE (an entry with no finding) are the only two properties a working tree
# can answer, and a commit that appends one line AND lands the matching
# violation satisfies both at once: not new, because it is baselined; not
# stale, because the finding is real.
#
# Measured, not argued: appending one entry cloned from this file's own last
# real entry returned rc=0 from this guard, under its own words:
#     "the list may only get shorter"
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "${REPO_ROOT}" scripts/unwired_guards_baseline.txt set || exit 1

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE"
    exit 1
fi
baseline_count=$(grep -cvE '^\s*(#|$)' "$BASELINE" || true)

if [ "$count" -gt "$baseline_count" ]; then
    printf '\nFAIL: unwired guards grew %s -> %s.\n' "$baseline_count" "$count"
    printf 'A guard was added or unwired. Name it in a workflow. The baseline is\n'
    printf 'SHRINK-ONLY against origin/main, so %s is not an\n' "$(basename "$BASELINE")"
    printf 'exemption list you can append to — an entry may only leave it.\n\n'
    comm -13 <(grep -vE '^\s*(#|$)' "$BASELINE" | LC_ALL=C sort) \
            <(printf '%s\n' "$FOUND" | grep .) | sed 's|^|  NEW: |'
    exit 1
fi

if [ "$count" -lt "$baseline_count" ]; then
    printf '\nImproved: %s -> %s. Run --update to record it.\n' "$baseline_count" "$count"
fi

printf 'PASS (ratcheted)\n'
exit 0
