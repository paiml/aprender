#!/usr/bin/env bash
# check_competitive_parity.sh — the competitive-parity RATCHET.
#
# WHY THIS EXISTS
# ---------------
# The operator made competitive parity a permanent hard requirement. Enforced as
# literally worded — "every entry point proven equal or better" — that is a
# FABRICATION ENGINE: a rule that admits only wins makes DELETING a losing
# comparison the cheapest compliant action. This repository has already done
# exactly that. `git log --all --diff-filter=D` over crates/*/tests/beat_* and
# contracts/beat-* returns ONE commit, d7e08043b (squashed edfa106d0, PR #2040,
# PMAT-733), and it deleted the only two LOSING rows in the history — the
# StandardScaler beat that had just measured apr 0.69x on the canonical Intel
# runner, and its MinMaxScaler sibling. 395 deletions, replaced by a comment.
#
# So this gate is INVERTED. It never checks that a verdict says BETTER. It
# checks that a FRESH, DATED verdict EXISTS, drawn from a closed vocabulary:
#
#     BETTER / PARITY / WORSE / NOT_COMPARABLE / UNMEASURED
#
# WORSE is a MEASUREMENT. Recording it raises __MEASURED__; deleting it lowers
# __MEASURED__, and __MEASURED__ may never fall. Under the naive rule, deleting
# the 0.69x row was the cheapest way to comply. Under this one it is the most
# expensive thing you can do.
#
# WHAT IT ENFORCES
# ----------------
#   1. `pv parity-ledger` passes  — freshness evaluated AT CHECK TIME, for every
#      verdict class. An expired BETTER row degrades to UNMEASURED and blocks.
#      This is the half the first design got backwards: it bounded only
#      UNMEASURED rows, and MEASURED is exactly where both withdrawn claims
#      lived (ollama 1.371x; StandardScaler).
#   2. __MEASURED__ >= the baseline. Monotone, shrink-never.
#   3. __NON_WINS__ >= the baseline. A ledger that is all wins is untested in
#      the direction that matters.
#   4. The scope file is bound to the LIVE enumeration from a SHA-PINNED `apr`
#      (`. scripts/apr_bin.sh`), so a scope entry naming a subcommand that no
#      longer exists is RED rather than quietly true.
#   5. Every ledger row's entry_point is IN scope, so a row cannot be scored
#      against a universe it is not part of.
#   6. `--update-baseline` REFUSES to write a lower __MEASURED__/__NON_WINS__,
#      and refuses to shrink the scope unless each dropped entry has actually
#      left the runtime enumeration. That is the PMAT-733 countermeasure: you
#      cannot raise the ratio by deleting the denominator either.
#
#   bash scripts/check_competitive_parity.sh                    # check
#   bash scripts/check_competitive_parity.sh --self-test        # case table
#   bash scripts/check_competitive_parity.sh --update-baseline  # ratchet up only
#
# NOTE ON `pv`: this shells the WORKSPACE `pv` via `cargo run -q -p
# aprender-contracts-cli`, never a `pv` on PATH. The PATH copy on the dev box is
# 0.49.0 and does not have `parity-ledger` at all; a gate that silently measured
# a different binary than the one under test is the stale-`apr` defect wearing a
# different hat.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="contracts/apr-competitive-parity-v1.yaml"
SCOPE="scripts/competitive_parity_scope.txt"
BASELINE="scripts/competitive_parity_baseline.txt"

# ---------------------------------------------------------------------------
# Pure decision functions. Everything the --self-test table exercises lives
# here, so the table probes the real code path rather than a paraphrase of it.
# ---------------------------------------------------------------------------

# Read one anchored `__KEY__=<int>` line out of a `pv parity-ledger` report.
#
# ANCHORED ON PURPOSE. `grep "__MEASURED__"` would also match a prose line such
# as `__MEASURED__ fell from 4 to 0`, and PMAT-CI-PASSGREP-001 is this repo's
# standing lesson that an unanchored count-grep matches the wrong count (a
# `grep "0 failed"` that also matched "10 failed"). `^__K__=<digits>$` matches
# the emitter's line and nothing else.
cp_extract() {
    local key="$1" text="$2"
    grep -E "^${key}=[0-9]+$" <<<"$text" | head -1 | cut -d= -f2
}

# Ratchet comparison: is `actual` acceptable against `floor`?
# Non-numeric input is a FAILURE, never a pass — a missing measurement must be
# red, not absent (the coverage-floor lesson: `|| true` disarmed the gate AND
# the measurement reported 0/0).
cp_meets_floor() {
    local actual="$1" floor="$2"
    case "$actual" in ''|*[!0-9]*) return 1 ;; esac
    case "$floor" in ''|*[!0-9]*) return 1 ;; esac
    [ "$actual" -ge "$floor" ]
}

# Is `entry` present in the live universe `universe` (one item per line)?
#
# A scope entry is one of:
#   apr <subcommand> [...]   -> the subcommand must be in `apr --help`
#   bin:<name>               -> the name must be a workspace [[bin]] target
#   lib:<path>::<Symbol>     -> a library surface no subcommand exposes; not
#                               runtime-enumerable, so it is accepted here and
#                               its REMOVAL is policed separately.
cp_entry_is_live() {
    local entry="$1" universe="$2" needle
    case "$entry" in
        "apr "*)
            needle=${entry#apr }
            needle=${needle%% *}
            grep -qxF -- "apr $needle" <<<"$universe"
            ;;
        bin:*)
            grep -qxF -- "$entry" <<<"$universe"
            ;;
        lib:*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# May `entry` be REMOVED from the scope file?
#
# Only when it has genuinely left the world: an `apr <sub>` entry only when the
# subcommand is gone from the live enumeration, a `bin:` entry only when the bin
# target is gone, a `lib:` entry only when its symbol no longer appears under
# crates/. Anything else is a shrinking denominator, which is PMAT-733 with the
# arithmetic done from the other end.
cp_removal_allowed() {
    local entry="$1" universe="$2" root="${3:-$REPO_ROOT}" sym
    case "$entry" in
        lib:*)
            sym=${entry##*::}          # trailing member, e.g. fit_transform
            sym=${entry%::"$sym"}      # ... strip it
            sym=${sym##*::}            # the type, e.g. StandardScaler
            [ -n "$sym" ] || return 0
            ! grep -rqlF -- "$sym" "$root/crates" 2>/dev/null
            ;;
        *)
            ! cp_entry_is_live "$entry" "$universe"
            ;;
    esac
}

# Non-comment, non-blank lines of a scope file.
cp_scope_entries() {
    grep -vE '^[[:space:]]*(#|$)' "$1" 2>/dev/null
}

# The scope KEY of a ledger row's entry_point.
#
# A row may qualify its entry point -- `apr run --gpu`, `apr run --gpu
# (concurrency=1 single-request decode)` -- because those are genuinely
# different comparison surfaces and collapsing them would let one mask the
# other. The SCOPE, though, is a list of entry points, so both keys back to
# `apr run`. `lib:` and `bin:` entries are already exact.
cp_scope_key() {
    local e="$1" sub
    case "$e" in
        "apr "*)
            sub=${e#apr }
            sub=${sub%% *}
            printf 'apr %s\n' "$sub"
            ;;
        *) printf '%s\n' "$e" ;;
    esac
}

# Is this guard actually WIRED into the blocking gate?
#
# THREE conditions, because two of them have failed in this repo already:
#   (a) some workflow INVOKES the script (a mention in a comment is not an
#       invocation — #2551 fixed exactly that in the meta-guard);
#   (b) `parity-ledger` is in `gate.needs`;
#   (c) the gate BODY tests `needs.parity-ledger.result` explicitly. This is the
#       one that is easy to miss: `gate` runs `if: always()` and reads each
#       result by name, so a job listed in `needs` and absent from the body
#       cannot fail the gate. A required check that blocks nothing is worse than
#       no check, because it is counted.
#
# Args let the --self-test table point this at a sandbox workflow file.
cp_ci_wiring_ok() {
    local ci="${1:-$REPO_ROOT/.github/workflows/ci.yml}" bad=0
    [ -f "$ci" ] || return 1
    # (a) invocation, with `#` comments stripped first so a mention cannot pass.
    sed 's/#.*$//' "$ci" \
        | grep -qE '(^|[[:space:];&|(])((ba)?sh[[:space:]]+|\./)?[^[:space:]]*check_competitive_parity\.sh([[:space:]]|$)' \
        || bad=1
    # (b) in gate.needs.
    grep -qE '^[[:space:]]*needs:.*\bparity-ledger\b' "$ci" || bad=1
    # (c) explicitly result-checked, in a CONDITIONAL.
    #
    # This started as `grep -qF 'needs.parity-ledger.result'` and the mutation
    # that replaces the `if` with a constant SURVIVED it: the gate body also
    # ECHOES the result in its diagnostic line, so the bare substring matched a
    # body that could no longer fail. Mention-vs-execution again, one level
    # down — caught by re-running the mutation, not by reading the pattern.
    # Require the comparison itself.
    grep -qE 'if[[:space:]]+\[[[:space:]]*"\$\{\{[[:space:]]*needs\.parity-ledger\.result[[:space:]]*\}\}"[[:space:]]*!=[[:space:]]*"success"' "$ci" \
        || bad=1
    [ "$bad" -eq 0 ]
}

# ---------------------------------------------------------------------------
# --self-test: must-pass / must-fail table.
#
# Verification Discipline #7: the guard's own predicates ship a case table, and
# it is RE-RUN rather than re-read. Five `apr`-invocation patterns in this repo
# were wrong and every one was caught by a table, none by review.
# ---------------------------------------------------------------------------
cp_self_test() {
    local fails=0 n=0
    ok()  { n=$((n+1)); printf '  ok    %s\n' "$1"; }
    bad() { n=$((n+1)); fails=$((fails+1)); printf '  FAIL  %s\n' "$1"; }

    printf 'case table: cp_extract\n'
    # MUST match
    [ "$(cp_extract __MEASURED__ '__MEASURED__=4')" = "4" ] \
        && ok 'plain line' || bad 'plain line'
    [ "$(cp_extract __MEASURED__ $'ROW x\n__ROWS__=5\n__MEASURED__=12\n__EXPIRED__=0')" = "12" ] \
        && ok 'line among others' || bad 'line among others'
    # MUST NOT match — these are the ways an unanchored grep goes wrong.
    [ -z "$(cp_extract __MEASURED__ '__MEASURED__ fell from 4 to 0')" ] \
        && ok 'prose mentioning the key is not a value' || bad 'prose mentioning the key is not a value'
    [ -z "$(cp_extract __MEASURED__ 'X__MEASURED__=99')" ] \
        && ok 'prefixed key does not match' || bad 'prefixed key does not match'
    [ -z "$(cp_extract __MEASURED__ '__MEASURED__=4 (was 9)')" ] \
        && ok 'trailing text does not match' || bad 'trailing text does not match'
    [ -z "$(cp_extract __MEASURED__ '__MEASURED__=')" ] \
        && ok 'empty value does not match' || bad 'empty value does not match'
    [ -z "$(cp_extract __MEASURED__ '__NON_WINS__=4')" ] \
        && ok 'a different key does not match' || bad 'a different key does not match'

    printf 'case table: cp_meets_floor\n'
    if cp_meets_floor 4 4; then ok 'equal meets the floor'; else bad 'equal meets the floor'; fi
    if cp_meets_floor 5 4; then ok 'above meets the floor'; else bad 'above meets the floor'; fi
    # MUST REFUSE. A missing or non-numeric measurement is the coverage-floor
    # lesson in miniature: the gate reported 0/0 for months and `|| true` kept
    # it green. A measurement that did not happen must be RED, not absent.
    if cp_meets_floor 3 4;      then bad 'below is REFUSED';               else ok 'below is REFUSED'; fi
    if cp_meets_floor '' 4;     then bad 'missing measurement is REFUSED'; else ok 'missing measurement is REFUSED'; fi
    if cp_meets_floor 'four' 4; then bad 'non-numeric is REFUSED';         else ok 'non-numeric is REFUSED'; fi
    if cp_meets_floor 4 '';     then bad 'missing floor is REFUSED';       else ok 'missing floor is REFUSED'; fi

    printf 'case table: cp_entry_is_live\n'
    local uni
    uni=$'apr run\napr serve\napr qa\nbin:pv\nbin:apr'
    cp_entry_is_live 'apr run' "$uni"          && ok 'live subcommand'        || bad 'live subcommand'
    cp_entry_is_live 'apr run --gpu' "$uni"    && ok 'flags are ignored'      || bad 'flags are ignored'
    cp_entry_is_live 'bin:pv' "$uni"           && ok 'live bin target'        || bad 'live bin target'
    cp_entry_is_live 'lib:aprender-core::X' "$uni" && ok 'lib surface accepted' || bad 'lib surface accepted'
    if cp_entry_is_live 'apr finetune' "$uni"; then bad 'absent subcommand is DEAD'; else ok 'absent subcommand is DEAD'; fi
    if cp_entry_is_live 'bin:alimentar' "$uni"; then bad 'absent bin is DEAD'; else ok 'absent bin is DEAD'; fi
    # Substring traps: `apr ru` must NOT match `apr run`.
    if cp_entry_is_live 'apr ru' "$uni"; then bad 'prefix is not a match'; else ok 'prefix is not a match'; fi
    if cp_entry_is_live 'nonsense' "$uni"; then bad 'unparseable entry is DEAD'; else ok 'unparseable entry is DEAD'; fi

    printf 'case table: cp_removal_allowed (the PMAT-733 countermeasure)\n'
    local sandbox
    sandbox=$(mktemp -d) || return 2
    mkdir -p "$sandbox/crates/x/src"
    printf 'pub struct StillHere;\n' > "$sandbox/crates/x/src/lib.rs"
    if cp_removal_allowed 'apr run' "$uni" "$sandbox"; then
        bad 'removing a LIVE subcommand is REFUSED'
    else
        ok 'removing a LIVE subcommand is REFUSED'
    fi
    cp_removal_allowed 'apr finetune' "$uni" "$sandbox" \
        && ok 'removing a GONE subcommand is allowed' || bad 'removing a GONE subcommand is allowed'
    if cp_removal_allowed 'lib:aprender-core::StillHere::fit' "$uni" "$sandbox"; then
        bad 'removing a lib surface whose symbol still exists is REFUSED'
    else
        ok 'removing a lib surface whose symbol still exists is REFUSED'
    fi
    cp_removal_allowed 'lib:aprender-core::LongGone::fit' "$uni" "$sandbox" \
        && ok 'removing a lib surface whose symbol is gone is allowed' \
        || bad 'removing a lib surface whose symbol is gone is allowed'
    rm -rf "${sandbox:?}"

    printf 'case table: cp_ci_wiring_ok (a gate not in gate.needs is decoration)\n'
    local wd f
    wd=$(mktemp -d) || return 2
    f="$wd/ci.yml"
    cp_wire_fixture() {
        local mode="${1:-full}"
        {
            case "$mode" in
                no-invoke)    ;;  # nothing at all
                comment-only) printf '        # run: bash scripts/check_competitive_parity.sh\n' ;;
                *)            printf '        run: bash scripts/check_competitive_parity.sh\n' ;;
            esac
            [ "$mode" = "no-needs" ] || printf '    needs: [ci, workspace-test, parity-ledger]\n'
            case "$mode" in
                no-result-check) ;;
                # The body only MENTIONS the result (as the gate's own
                # diagnostic echo does) while the conditional is gone. The
                # first version of cp_ci_wiring_ok passed this.
                echo-only) printf '            echo "parity-ledger failed: ${{ needs.parity-ledger.result }}"\n' ;;
                *) printf '          if [ "${{ needs.parity-ledger.result }}" != "success" ]; then\n' ;;
            esac
        } > "$f"
    }
    cp_wire_fixture full
    if cp_ci_wiring_ok "$f"; then ok 'fully wired passes'; else bad 'fully wired passes'; fi
    cp_wire_fixture no-invoke
    if cp_ci_wiring_ok "$f"; then bad 'no invocation is REFUSED'; else ok 'no invocation is REFUSED'; fi
    cp_wire_fixture comment-only
    if cp_ci_wiring_ok "$f"; then bad 'a COMMENTED invocation is REFUSED'; else ok 'a COMMENTED invocation is REFUSED'; fi
    cp_wire_fixture no-needs
    if cp_ci_wiring_ok "$f"; then bad 'absent from gate.needs is REFUSED'; else ok 'absent from gate.needs is REFUSED'; fi
    cp_wire_fixture no-result-check
    if cp_ci_wiring_ok "$f"; then bad 'in needs but not result-checked is REFUSED'; else ok 'in needs but not result-checked is REFUSED'; fi
    cp_wire_fixture echo-only
    if cp_ci_wiring_ok "$f"; then bad 'ECHOING the result is not CHECKING it'; else ok 'ECHOING the result is not CHECKING it'; fi
    rm -rf "${wd:?}"

    printf '\n%d case(s), %d failure(s)\n' "$n" "$fails"
    [ "$fails" -eq 0 ]
}

# ---------------------------------------------------------------------------
# Runtime enumeration of the universe. NEVER a hardcoded list.
# ---------------------------------------------------------------------------

# Every `apr <subcommand>` the SHA-pinned binary itself reports, plus every
# workspace [[bin]] target, as `apr <name>` / `bin:<name>` lines.
#
# `. scripts/apr_bin.sh || return 1` and nothing else: no bare `apr`, no
# absolute path. Four `apr` binaries once coexisted here and a bare `apr`
# resolved to a 26-day-old one, so a gate that enumerates from the wrong binary
# produces a confident answer about code that is not running.
cp_live_universe() {
    # shellcheck source=scripts/apr_bin.sh
    . "$REPO_ROOT/scripts/apr_bin.sh" || return 1
    # The command column is EXACTLY two spaces; clap wraps long descriptions to
    # the description column (~27 spaces). Matching `^[[:space:]]+` instead of
    # `^  ` picked up wrapped description text as commands and reported 120
    # entries against a real 111 — including the non-commands `apr apr` and
    # `apr aprender`, both of which came out of a wrapped sentence. A universe
    # built from the wrong side is this repo's standing guard defect.
    "$APR" --help 2>&1 \
        | awk '
            /^Commands:/ { inblock = 1; next }
            /^[A-Za-z]/   { inblock = 0 }
            inblock && /^  [a-z][a-z0-9-]*( |$)/ { print "apr " $1 }
        ' \
        | LC_ALL=C sort -u
    cargo metadata --no-deps --format-version 1 --manifest-path "$REPO_ROOT/Cargo.toml" 2>/dev/null \
        | jq -r '.packages[].targets[] | select(.kind[] == "bin") | "bin:" + .name' \
        | LC_ALL=C sort -u
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
cd "$REPO_ROOT" || exit 2

case "${1:-}" in
    --self-test)
        cp_self_test
        exit $?
        ;;
esac

MODE="${1:-check}"
if [ "$MODE" != "check" ] && [ "$MODE" != "--update-baseline" ]; then
    printf 'usage: %s [--self-test|--update-baseline]\n' "$0" >&2
    exit 2
fi

for f in "$LEDGER" "$SCOPE" "$BASELINE"; do
    [ -f "$f" ] || { printf '✗ missing %s\n' "$f" >&2; exit 2; }
done

fail=0

# -- 0. this guard must be WIRED into the blocking gate ---------------------
# Checked first, and by the guard itself, because "the check exists" and "the
# check blocks a merge" are different claims and only the second one matters.
if ! cp_ci_wiring_ok; then
    printf '✗ NOT WIRED: this guard is not reachable from the blocking gate.\n' >&2
    printf '    Required, all three: .github/workflows/ci.yml must INVOKE\n' >&2
    printf '    check_competitive_parity.sh (a mention in a comment does not\n' >&2
    printf '    count), `parity-ledger` must appear in gate.needs, and the gate\n' >&2
    printf '    BODY must test needs.parity-ledger.result — `gate` runs with\n' >&2
    printf '    if: always() and reads each result by name, so a job in `needs`\n' >&2
    printf '    that the body never tests cannot fail the gate.\n' >&2
    fail=1
fi

# -- 1. the ledger itself, evaluated AS OF TODAY ----------------------------
# rc is read from the COMMAND, not through a pipe. Reading `$?` after a pipe
# gives the LAST command's status; that exact defect shipped twice here
# (#2336 qwen-story-daily's `tee`, #2360 make publish's post-publish check).
PV_OUT=$(cargo run -q -p aprender-contracts-cli --bin pv -- \
             parity-ledger "$LEDGER" 2>&1)
PV_RC=$?
printf '%s\n' "$PV_OUT"
if [ "$PV_RC" -ne 0 ]; then
    printf '\n✗ the parity ledger did not evaluate clean (pv rc=%d).\n' "$PV_RC" >&2
    printf '  Staleness blocks; the verdict VALUE does not. Re-measure the expired\n' >&2
    printf '  row, or record it as UNMEASURED with a fresh bound and an owner.\n' >&2
    fail=1
fi

MEASURED=$(cp_extract __MEASURED__ "$PV_OUT")
NON_WINS=$(cp_extract __NON_WINS__ "$PV_OUT")
ROWS=$(cp_extract __ROWS__ "$PV_OUT")

# -- 2. the live universe ---------------------------------------------------
UNIVERSE=$(cp_live_universe)
UNIVERSE_RC=$?
if [ "$UNIVERSE_RC" -ne 0 ] || [ -z "$UNIVERSE" ]; then
    printf '✗ could not enumerate the live universe from a HEAD-built apr.\n' >&2
    printf '  Build it (cargo build --release --bin apr) and re-run. This gate\n' >&2
    printf '  refuses to fall back to a hardcoded list: an enumeration that is\n' >&2
    printf '  not RUNTIME is a claim about a binary nobody ran.\n' >&2
    exit 1
fi

SCOPE_ENTRIES=$(cp_scope_entries "$SCOPE")
IN_SCOPE=$(printf '%s\n' "$SCOPE_ENTRIES" | grep -c .)

# -- 3. every scope entry must still be live --------------------------------
while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    if ! cp_entry_is_live "$entry" "$UNIVERSE"; then
        printf '✗ STALE SCOPE: %s is in %s but not in the live enumeration.\n' \
               "$entry" "$SCOPE" >&2
        printf '    Either the entry point was renamed/removed (update the scope AND\n' >&2
        printf '    the ledger), or the scope was written against a different binary.\n' >&2
        fail=1
    fi
done <<<"$SCOPE_ENTRIES"

# -- 4. every ledger row must be in scope -----------------------------------
# Rows are read from `pv`'s own report, not by grepping YAML: a hand-rolled
# YAML parser is banned here (check_no_hand_rolled_parsers.sh) and would drift
# from the schema the moment a field moved.
ROW_ENTRIES=$(grep -E '^ROW ' <<<"$PV_OUT" | sed -E 's/.*valid_until=[^ ]+ //')
while IFS= read -r row; do
    [ -n "$row" ] || continue
    if ! grep -qxF -- "$(cp_scope_key "$row")" <<<"$SCOPE_ENTRIES"; then
        printf '✗ OUT-OF-SCOPE ROW: %s\n' "$row" >&2
        printf '    A row scored against a universe it is not part of inflates the\n' >&2
        printf '    ratio without measuring anything. Add it to %s.\n' "$SCOPE" >&2
        fail=1
    fi
done <<<"$ROW_ENTRIES"

# -- 5. the baseline --------------------------------------------------------
# shellcheck source=scripts/competitive_parity_baseline.txt
MEASURED_MIN=$(grep -E '^MEASURED_MIN=[0-9]+$' "$BASELINE" | head -1 | cut -d= -f2)
NON_WINS_MIN=$(grep -E '^NON_WINS_MIN=[0-9]+$' "$BASELINE" | head -1 | cut -d= -f2)
IN_SCOPE_MIN=$(grep -E '^IN_SCOPE_MIN=[0-9]+$' "$BASELINE" | head -1 | cut -d= -f2)

if [ "$MODE" = "--update-baseline" ]; then
    refuse=0
    cp_meets_floor "$MEASURED" "$MEASURED_MIN" || {
        printf '✗ REFUSING to lower MEASURED_MIN %s -> %s.\n' "$MEASURED_MIN" "$MEASURED" >&2
        printf '    Deleting a comparison to raise the ratio is the failure this gate\n' >&2
        printf '    exists to block (PMAT-733). Record the loss instead: WORSE and\n' >&2
        printf '    UNMEASURED are first-class verdicts, and WORSE still COUNTS.\n' >&2
        refuse=1
    }
    cp_meets_floor "$NON_WINS" "$NON_WINS_MIN" || {
        printf '✗ REFUSING to lower NON_WINS_MIN %s -> %s.\n' "$NON_WINS_MIN" "$NON_WINS" >&2
        printf '    A ledger trending toward all-wins is trending toward the state\n' >&2
        printf '    this mechanism was built to detect.\n' >&2
        refuse=1
    }
    if ! cp_meets_floor "$IN_SCOPE" "$IN_SCOPE_MIN"; then
        # A shrinking denominator is allowed ONLY when the dropped entries have
        # genuinely left the runtime enumeration.
        PREV_SCOPE=$(git show "HEAD:$SCOPE" 2>/dev/null)
        while IFS= read -r gone; do
            [ -n "$gone" ] || continue
            grep -qxF -- "$gone" <<<"$SCOPE_ENTRIES" && continue
            if cp_removal_allowed "$gone" "$UNIVERSE"; then
                printf '  scope shrink OK: %s has left the enumeration\n' "$gone"
            else
                printf '✗ REFUSING to drop %s from the scope: it is still live.\n' "$gone" >&2
                refuse=1
            fi
        done < <(grep -vE '^[[:space:]]*(#|$)' <<<"$PREV_SCOPE")
    fi
    [ "$refuse" -eq 0 ] || exit 1

    {
        printf '# Competitive-parity ratchet baseline. SHRINK-NEVER.\n'
        printf '# Written by scripts/check_competitive_parity.sh --update-baseline,\n'
        printf '# which refuses to lower any of these. Do not hand-edit downward:\n'
        printf '# the whole mechanism is that deleting a losing row costs you a point.\n'
        printf 'MEASURED_MIN=%s\n' "$MEASURED"
        printf 'NON_WINS_MIN=%s\n' "$NON_WINS"
        printf 'IN_SCOPE_MIN=%s\n' "$IN_SCOPE"
    } > "$BASELINE"
    printf '✓ baseline updated: MEASURED_MIN=%s NON_WINS_MIN=%s IN_SCOPE_MIN=%s\n' \
           "$MEASURED" "$NON_WINS" "$IN_SCOPE"
    exit 0
fi

cp_meets_floor "$MEASURED" "$MEASURED_MIN" || {
    printf '✗ __MEASURED__ fell: %s < baseline %s.\n' "${MEASURED:-<none>}" "${MEASURED_MIN:-<none>}" >&2
    printf '    Either a row was DELETED, or a row EXPIRED. Both are the same\n' >&2
    printf '    defect from the ledger'"'"'s point of view: a claim that no longer has\n' >&2
    printf '    a fresh dated measurement behind it.\n' >&2
    fail=1
}
cp_meets_floor "$NON_WINS" "$NON_WINS_MIN" || {
    printf '✗ __NON_WINS__ fell: %s < baseline %s.\n' "${NON_WINS:-<none>}" "${NON_WINS_MIN:-<none>}" >&2
    printf '    Losses may be FIXED, not deleted. Turn a WORSE into a PARITY by\n' >&2
    printf '    measuring again, not by removing the row.\n' >&2
    fail=1
}
cp_meets_floor "$IN_SCOPE" "$IN_SCOPE_MIN" || {
    printf '✗ __IN_SCOPE__ fell: %s < baseline %s.\n' "${IN_SCOPE:-<none>}" "${IN_SCOPE_MIN:-<none>}" >&2
    printf '    The denominator shrank. Use --update-baseline, which will only let\n' >&2
    printf '    it through if the dropped entry points have left the live binary.\n' >&2
    fail=1
}

printf '\n'
printf 'entry points in scope : %s (floor %s)\n' "$IN_SCOPE" "$IN_SCOPE_MIN"
printf 'ledger rows           : %s\n' "${ROWS:-<none>}"
printf 'measured (fresh)      : %s (floor %s)\n' "${MEASURED:-<none>}" "$MEASURED_MIN"
printf 'non-wins recorded     : %s (floor %s)\n' "${NON_WINS:-<none>}" "$NON_WINS_MIN"

if [ "$fail" -ne 0 ]; then
    printf '\n✗ competitive-parity ratchet FAILED\n' >&2
    exit 1
fi
printf '\n✓ competitive-parity ratchet OK\n'
exit 0
