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
# THE BASELINE IS A SET, NOT A COUNT
# ----------------------------------
# The first version of this ratchet enforced `__MEASURED__ >= 4` and never
# recorded WHICH entry points held a verdict. A count is payable in the wrong
# currency: DELETE the StandardScaler 0.69x row, ADD a cheaper fabricated one,
# and every total is unchanged while the only losing measurement in the history
# has left the tree again. That is PMAT-733 with the arithmetic balanced, and it
# is the move this file exists to block.
#
# So the baseline holds SETS keyed by entry_point:
#
#   ROW=<entry_point>           every row that must still EXIST. Shrink-never.
#   MEASURED_ROW=<entry_point>  every row whose verdict must still be MEASURED
#                               -- unless a downgrade is RECORDED for it. Also
#                               shrink-never, which is what makes a downgrade a
#                               DEBT rather than a one-off payment: the key
#                               stays, so the row must either come back to
#                               measured or keep its record. A baseline that
#                               absorbed the drop would let the next commit
#                               delete the `downgrades:` entry unnoticed.
#
# HONESTY MUST STAY AFFORDABLE
# ----------------------------
# The mirror-image failure is a floor with no give. `MEASURED_MIN=4` made
# DOWNGRADING mechanically forbidden, and the live case was already in the
# ledger: the `apr code` row's own note says it should be UNMEASURED -- its
# cited receipt, evidence/phase-5/arena-scores.json, does not exist in this
# repository -- yet correcting it would have breached the floor. A ratchet that
# punishes increasing honesty produces dishonest ledgers.
#
# The two properties are therefore SEPARATED. The set of rows that EXIST may
# never shrink. The set of rows that are MEASURED may shrink, but only against a
# `downgrades:` record in the ledger naming that row, with a reason from a
# CLOSED serde vocabulary (prose fails to PARSE), an owner, and a bounded
# recheck date. And PARITY-012 requires the downgraded row to still be PRESENT,
# so "delete the row and file paperwork" is not a route.
#
# THE KEY CHANNEL IS VERIFIED, NOT TRUSTED
# ----------------------------------------
# Set membership travels from `pv` to this script as text, so the channel is
# part of the mechanism. Under the first wire format (`__ROW__=<rest of line>`)
# an entry_point containing a NEWLINE printed several well-formed key lines from
# ONE row, so a fabricated row could satisfy a DELETED row's baseline key at
# constant totals -- the set ratchet defeated by exactly the move it was built
# to block. Three independent controls, because any one of them is a single edit
# from useless: PARITY-002 refuses the character at the SOURCE; every key is
# LENGTH-PREFIXED (`__ROW__=<bytes>:<key>`) and a line whose declared length
# does not match what follows is DROPPED; and the NUMBER of key lines is
# cross-checked against the emitter's own __ROWS__ / __MEASURED__, which an
# injection can only inflate.
#
# THE BASELINE FILE IS ITSELF RATCHETED
# -------------------------------------
# Every --update-baseline refusal below guards a code path nobody is obliged to
# take: the baseline is unbound plaintext read with `cat`, so hand-editing it
# moves the bar and skips all of them -- drop the ROW key AND the ledger row and
# the working copy agrees with itself, silently. So the same refusals now run in
# CHECK mode against a value the editor does not control: the file AS COMMITTED,
# read from the merge base with the upstream default branch (unioned with HEAD,
# so the check is neither vacuous on a new branch nor defeated by making the
# drop in its own commit).
#
# WHAT IT ENFORCES
# ----------------
#   1. `pv parity-ledger` passes  — freshness evaluated AT CHECK TIME, for every
#      verdict class. An expired BETTER row degrades to UNMEASURED and blocks.
#      This is the half the first design got backwards: it bounded only
#      UNMEASURED rows, and MEASURED is exactly where both withdrawn claims
#      lived (ollama 1.371x; StandardScaler). PARITY-011 additionally CAPS how
#      far ahead `valid_until` may be set, because check-time freshness is only
#      as strong as the dates it reads: rewriting every expiry to "2099-12-31"
#      satisfied the first design completely.
#   2. Every baseline ROW= key still exists. Set-keyed, so losing a SPECIFIC row
#      is RED at constant totals.
#   3. Every baseline MEASURED_ROW= key is still measured, OR carries a recorded
#      downgrade in the ledger's `downgrades:` block.
#   4. __NON_WINS__ >= the baseline. A ledger that is all wins is untested in
#      the direction that matters.
#   5. The scope file is bound to the LIVE enumeration from a SHA-PINNED `apr`
#      (`. scripts/apr_bin.sh`), so a scope entry naming a subcommand that no
#      longer exists is RED rather than quietly true.
#   6. Every ledger row's entry_point is IN scope, so a row cannot be scored
#      against a universe it is not part of.
#   7. `--update-baseline` REFUSES to drop a ROW key that is still live, refuses
#      to drop a MEASURED_ROW key with no recorded downgrade, refuses to lower
#      __NON_WINS__, and refuses to shrink the scope unless each dropped entry
#      has actually left the runtime enumeration. That is the PMAT-733
#      countermeasure from both ends: you cannot raise the ratio by deleting the
#      numerator or by shrinking the denominator.
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

# The universe is enumerated from a SHA-pinned `apr`, and `apr_bin_assert_fresh`
# FAILS OPEN outside a git checkout unless this is set -- it prints "not a git
# checkout, freshness not asserted" and returns 0, so any binary passes the
# guard whose entire job is refusing unproven binaries. Exported HERE rather
# than relied on from the CI job's `env:`, because the gate must be strict when
# a human runs it too, and a variable set only in one workflow is a property of
# that workflow rather than of this check.
export APR_BIN_STRICT=1

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

# Every VALUE of an anchored, LENGTH-PREFIXED `KEY=<bytes>:<value>` line, one
# per line, with the declared length VERIFIED.
#
# NOT `cut -d= -f2`. Entry points contain `=` -- the llama.cpp row is literally
# `apr run --gpu (concurrency=1 single-request decode)` -- and cutting on the
# delimiter truncates that key to `apr run --gpu (concurrency`. A set-membership
# test over truncated keys reports a row as PRESENT whenever a different row
# shares its prefix, which reintroduces the exact hole the set is closing.
#
# NOR is stripping the `^KEY=` prefix sufficient, which is what this used to do.
# The value travelled as the REST OF THE LINE, so a key containing a NEWLINE
# printed several well-formed key lines from ONE row: an `entry_point` written
#
#     apr qa
#     __ROW__=lib:aprender-core::StandardScaler::fit_transform
#
# satisfies the DELETED StandardScaler row's baseline key from a fabricated row,
# at constant totals -- the set ratchet defeated by precisely the move it was
# built to block. Three independent controls close that, because any one of them
# is a single edit from useless:
#
#   (a) PARITY-002/012 refuse a non-printable-ASCII character in a key at the
#       SOURCE, so the newline never reaches the channel;
#   (b) this function verifies the declared byte length against what follows,
#       so a line that is not the WHOLE key is DROPPED (never repaired -- a
#       dropped __ROW__ makes the baseline check fail, which is the safe
#       direction);
#   (c) the caller cross-checks the NUMBER of key lines against the emitter's
#       own count (`__ROWS__`, `__MEASURED__`), and an injection can only ADD
#       lines.
#
# Verified, not trusted: a length prefix nobody checks is a comment.
cp_keys() {
    local key="$1" text="$2" line rest declared value
    local LC_ALL=C   # ${#value} must count BYTES, matching the emitter's len()
    while IFS= read -r line; do
        rest=${line#"$key="}
        [ "$rest" != "$line" ] || continue
        declared=${rest%%:*}
        case "$declared" in ''|*[!0-9]*) continue ;; esac
        value=${rest#*:}
        [ "${#value}" = "$declared" ] || continue
        printf '%s\n' "$value"
    done < <(grep -E "^${key}=[0-9]+:" <<<"$text")
}

# Count of well-formed key lines for `key`, for the cross-check against the
# emitter's own total. `grep -c` on the RAW lines would count malformed ones
# too, which is the opposite of what this is for.
cp_key_count() {
    cp_keys "$1" "$2" | grep -c .
}

# Write a set out in the SAME `KEY=<bytes>:<value>` wire format `pv` emits, so
# the baseline and the report are parsed by one function and cannot drift.
#
# It also binds the baseline a little further: hand-editing a key now means
# getting its byte length right too, and a mismatched length makes cp_keys DROP
# the line -- which reads as a missing ROW, which is RED.
cp_emit_keys() {
    local key="$1" set="$2" line
    local LC_ALL=C
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        printf '%s=%s:%s\n' "$key" "${#line}" "$line"
    done <<<"$set"
}

# Lines present in `want` and absent from `have`. Both are newline-separated
# sets; blank lines are ignored. Exact whole-line matching (`grep -qxF`), never
# substring: `apr run --gpu` must NOT satisfy a requirement for
# `apr run --gpu (concurrency=1 single-request decode)`.
cp_set_minus() {
    local want="$1" have="$2" line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        grep -qxF -- "$line" <<<"$have" || printf '%s\n' "$line"
    done <<<"$want"
}

# Which baseline-MEASURED keys stopped being measured WITHOUT a recorded
# downgrade? One key per line; empty output means every drop is paid for.
#
# This is the whole of the honest-downgrade rule, in one place so the case table
# can probe it directly. (a) a key that dropped AND has a downgrade record is
# silent; (b) a key that dropped with NO record is named. A key that is still
# measured is silent whether or not a record exists -- PARITY-014 refuses that
# combination on the contract side, which is where it belongs.
cp_unjustified_drops() {
    local baseline_measured="$1" live_measured="$2" live_downgrades="$3" k
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        grep -qxF -- "$k" <<<"$live_measured" && continue
        grep -qxF -- "$k" <<<"$live_downgrades" && continue
        printf '%s\n' "$k"
    done <<<"$baseline_measured"
}

# The commit this working tree should be ratcheted AGAINST.
#
# WHY THE BASELINE FILE ALONE IS NOT A BASELINE. Every `--update-baseline`
# refusal in this script -- refusing to drop a live ROW, refusing an
# unjustified MEASURED_ROW drop, refusing to lower NON_WINS_MIN -- guards a code
# path nobody is obliged to take. The file is unbound plaintext read straight
# off disk with `cat`, so hand-editing it moves the bar and skips every refusal.
# A guard whose enforcement is optional is decoration.
#
# The fix is not a checksum stored beside it (which the same edit updates) but a
# value the editor does not control: the file AS COMMITTED. `git show` of the
# merge base with the upstream default branch gives the bar as it stood before
# this change set, so the refusals now run in CHECK mode, on every run, against
# git -- and the hand-edit is exactly as expensive as `--update-baseline`.
#
# Merge base, not HEAD: ratcheting against HEAD is defeated by making the drop
# in its own commit, after which HEAD agrees with the drop. Against the merge
# base, every commit on the branch is judged against main.
# Every commit the working-tree baseline is judged against: the merge base with
# the upstream default branch, AND HEAD.
#
# BOTH, unioned, because each covers the other's blind spot. The merge base
# alone is vacuous while the file is new on this branch (it did not exist
# there), and it is the merge base that survives the "make the drop in its own
# commit" evasion. HEAD alone is defeated by that evasion and is non-vacuous
# from the first commit. Requiring the keys of both is strictly stronger than
# either and never weaker: a key legitimately gone from the world still passes
# `cp_removal_allowed`.
cp_base_refs() {
    local r
    for r in origin/main origin/master main master; do
        git rev-parse --verify --quiet "$r" >/dev/null 2>&1 || continue
        git merge-base "$r" HEAD 2>/dev/null
        break
    done
    git rev-parse --verify --quiet HEAD 2>/dev/null
}

# The baseline file's content at `ref`, or empty when the file did not exist
# there yet.
#
# Sets rc=2 -- distinct from "absent" -- when git itself cannot answer, because
# "the comparison could not be made" must be RED and not silently empty. An
# empty base set makes every regression check vacuously true, which is the
# coverage-floor failure (`|| true` over a measurement that reported 0/0).
cp_baseline_at() {
    local path="$1" ref
    git rev-parse --git-dir >/dev/null 2>&1 || return 2
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        git cat-file -e "$ref:$path" 2>/dev/null || continue
        git show "$ref:$path" 2>/dev/null || return 2
    done < <(cp_base_refs)
    return 0
}

# Baseline keys that were dropped from the COMMITTED baseline and are not
# accounted for. One key per line; empty output means the edit is legitimate.
#
#   $1 prior keys   $2 current keys   $3 live universe
#   $4 excused keys (a live downgrade pays for a MEASURED_ROW drop; pass '' for
#      the ROW set, where nothing pays for a drop except the entry point
#      actually having left the binary)
#   $5 repo root, for the `lib:` symbol probe
#
# Extracted as a pure function so the --self-test table probes the real code
# path. The refusal it implements is the same one --update-baseline applies;
# the point of running it HERE is that the hand-edit route no longer skips it.
cp_unbound_baseline_drops() {
    local prior="$1" cur="$2" uni="$3" excused="${4:-}" root="${5:-$REPO_ROOT}" k
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        [ -n "$excused" ] && grep -qxF -- "$k" <<<"$excused" && continue
        cp_removal_allowed "$k" "$uni" "$root" && continue
        printf '%s\n' "$k"
    done < <(cp_set_minus "$prior" "$cur")
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

    printf 'case table: cp_keys (a SET; `=` inside a key survives; length VERIFIED)\n'
    local pvout
    pvout=$'ROW  WORSE -> WORSE fresh valid_until=2026-09-30 apr code\n__ROWS__=5\n__ROW__=13:apr run --gpu\n__ROW__=51:apr run --gpu (concurrency=1 single-request decode)\n__MEASURED_ROW__=13:apr run --gpu\n__DOWNGRADE__=8:apr code'
    [ "$(cp_keys __ROW__ "$pvout" | wc -l)" = "2" ] \
        && ok 'both __ROW__ values are read' || bad 'both __ROW__ values are read'
    # THE TRAP: `cut -d= -f2` truncates this to "apr run --gpu (concurrency".
    grep -qxF 'apr run --gpu (concurrency=1 single-request decode)' \
        <<<"$(cp_keys __ROW__ "$pvout")" \
        && ok 'a key CONTAINING = survives intact' || bad 'a key CONTAINING = survives intact'
    # ...and a key containing a COLON, which is what the length prefix is
    # delimited by. `lib:aprender-core::Lasso::fit` is four of them.
    [ "$(cp_keys __ROW__ '__ROW__=29:lib:aprender-core::Lasso::fit')" \
        = 'lib:aprender-core::Lasso::fit' ] \
        && ok 'a key CONTAINING : survives the length delimiter' \
        || bad 'a key CONTAINING : survives the length delimiter'
    [ "$(cp_keys __MEASURED_ROW__ "$pvout")" = "apr run --gpu" ] \
        && ok '__MEASURED_ROW__ is a different set from __ROW__' \
        || bad '__MEASURED_ROW__ is a different set from __ROW__'
    [ "$(cp_keys __DOWNGRADE__ "$pvout")" = "apr code" ] \
        && ok '__DOWNGRADE__ is read' || bad '__DOWNGRADE__ is read'
    # MUST NOT match: the human ROW line mentions the same entry point.
    [ -z "$(cp_keys __EXPIRED_ROW__ "$pvout")" ] \
        && ok 'an absent key yields the empty set' || bad 'an absent key yields the empty set'
    [ -z "$(cp_keys __ROWS__ "$pvout")" ] \
        && ok '__ROWS__ is a COUNT line, not a key line' || bad '__ROWS__ is a COUNT line, not a key line'
    grep -qxF 'apr code' <<<"$(cp_keys __ROW__ "$pvout")" \
        && bad 'the prose ROW line is not a __ROW__ key' \
        || ok 'the prose ROW line is not a __ROW__ key'
    # MUST BE DROPPED. The length is VERIFIED, not decorative -- a prefix that
    # nobody checks is a comment. Dropping (rather than repairing) is the safe
    # direction: a missing __ROW__ reads as a DELETED row, which is RED.
    [ -z "$(cp_keys __ROW__ '__ROW__=99:apr code')" ] \
        && ok 'a key whose declared length is too long is DROPPED' \
        || bad 'a key whose declared length is too long is DROPPED'
    [ -z "$(cp_keys __ROW__ '__ROW__=3:apr code')" ] \
        && ok 'a key whose declared length is too short is DROPPED' \
        || bad 'a key whose declared length is too short is DROPPED'
    [ -z "$(cp_keys __ROW__ '__ROW__=apr code')" ] \
        && ok 'an UNPREFIXED key (the old wire format) is DROPPED' \
        || bad 'an UNPREFIXED key (the old wire format) is DROPPED'

    printf 'case table: cp_keys == THE KEY-INJECTION MUTATION\n'
    # An entry_point containing a NEWLINE printed several well-formed key lines
    # from ONE row under the old `__KEY__=<rest of line>` format, so a
    # fabricated row could satisfy a DELETED row's baseline key at constant
    # totals -- the set ratchet defeated by the move it exists to block.
    local inject
    inject=$'__ROW__=apr qa\n__ROW__=lib:aprender-core::StandardScaler::fit_transform'
    [ "$(grep -c '^__ROW__=' <<<"$inject")" = "2" ] \
        && ok 'the injection DOES print two well-formed lines in the old format' \
        || bad 'the injection DOES print two well-formed lines in the old format'
    [ -z "$(cp_keys __ROW__ "$inject")" ] \
        && ok 'neither survives length verification' || bad 'neither survives length verification'
    # The residual: an injected line CAN carry a correct prefix. Length alone
    # does not stop that, which is why control (c) -- the count cross-check --
    # exists, and why PARITY-002 refuses the character at the source.
    local crafted
    crafted=$'__ROW__=6:apr qa\n__ROW__=48:lib:aprender-core::StandardScaler::fit_transform'
    [ "$(cp_key_count __ROW__ "$crafted")" = "2" ] \
        && ok 'a CRAFTED prefix survives length verification (so counts must cross-check)' \
        || bad 'a CRAFTED prefix survives length verification (so counts must cross-check)'
    [ "$(cp_key_count __ROW__ '__ROW__=6:apr qa')" = "1" ] \
        && ok 'the honest emission of that row is ONE key line, so the count DIFFERS' \
        || bad 'the honest emission of that row is ONE key line, so the count DIFFERS'

    printf 'case table: cp_emit_keys (baseline and report share ONE wire format)\n'
    [ "$(cp_emit_keys ROW 'apr code')" = 'ROW=8:apr code' ] \
        && ok 'emits the byte length' || bad 'emits the byte length'
    # ROUND TRIP: what the baseline writer emits is exactly what cp_keys reads.
    # If these two ever drift the ratchet compares nothing, silently.
    local rt
    rt=$'apr run --gpu\nlib:aprender-core::Lasso::fit\napr run --gpu (concurrency=1 single-request decode)'
    [ "$(cp_keys ROW "$(cp_emit_keys ROW "$rt")")" = "$rt" ] \
        && ok 'writer and reader round-trip exactly' || bad 'writer and reader round-trip exactly'
    [ -z "$(cp_emit_keys ROW '')" ] \
        && ok 'the empty set emits nothing' || bad 'the empty set emits nothing'

    printf 'case table: cp_set_minus\n'
    local a b
    a=$'x\ny\nz'
    b=$'x\nz'
    [ "$(cp_set_minus "$a" "$b")" = "y" ] \
        && ok 'names exactly the missing member' || bad 'names exactly the missing member'
    [ -z "$(cp_set_minus "$b" "$a")" ] \
        && ok 'a subset is silent' || bad 'a subset is silent'
    [ -z "$(cp_set_minus '' "$a")" ] \
        && ok 'the empty want-set is silent' || bad 'the empty want-set is silent'
    [ "$(cp_set_minus "$a" '' | wc -l)" = "3" ] \
        && ok 'an empty have-set loses everything' || bad 'an empty have-set loses everything'
    # SUBSTRING TRAP. `apr run --gpu` present must NOT satisfy the qualified key.
    [ "$(cp_set_minus 'apr run --gpu (concurrency=1 single-request decode)' 'apr run --gpu')" \
        = 'apr run --gpu (concurrency=1 single-request decode)' ] \
        && ok 'a PREFIX in have does not satisfy want' || bad 'a PREFIX in have does not satisfy want'
    [ -z "$(cp_set_minus 'apr run' $'apr run\napr run --gpu')" ] \
        && ok 'exact whole-line membership counts' || bad 'exact whole-line membership counts'

    printf 'case table: cp_set_minus == THE DEFECT-1 MUTATION\n'
    # Delete one row, add a DIFFERENT one. Counts identical (3 -> 3); the set
    # must still name the loss. This is d7e08043b in miniature.
    local before after
    before=$'apr run --gpu\nlib:aprender-core::StandardScaler::fit_transform\napr code'
    after=$'apr run --gpu\napr qa\napr code'
    [ "$(grep -c . <<<"$before")" = "$(grep -c . <<<"$after")" ] \
        && ok 'the mutation keeps the COUNT identical (so a count ratchet is blind)' \
        || bad 'the mutation keeps the COUNT identical (so a count ratchet is blind)'
    [ "$(cp_set_minus "$before" "$after")" = 'lib:aprender-core::StandardScaler::fit_transform' ] \
        && ok 'the SET names the deleted row anyway' || bad 'the SET names the deleted row anyway'

    printf 'case table: cp_unjustified_drops (the DEFECT-3 give)\n'
    local bm lm ld
    bm=$'apr run --gpu\napr code\nlib:aprender-core::Lasso::fit'
    # (a) `apr code` dropped WITH a record -> silent.
    lm=$'apr run --gpu\nlib:aprender-core::Lasso::fit'
    ld=$'apr code'
    [ -z "$(cp_unjustified_drops "$bm" "$lm" "$ld")" ] \
        && ok '(a) a downgrade WITH a recorded reason PASSES' \
        || bad '(a) a downgrade WITH a recorded reason PASSES'
    # (b) the same drop with NO record -> named.
    [ "$(cp_unjustified_drops "$bm" "$lm" '')" = 'apr code' ] \
        && ok '(b) the same downgrade with NO record FAILS' \
        || bad '(b) the same downgrade with NO record FAILS'
    # A record for a DIFFERENT row does not launder this one.
    [ "$(cp_unjustified_drops "$bm" "$lm" 'apr run --gpu')" = 'apr code' ] \
        && ok 'a record for another row does not launder this drop' \
        || bad 'a record for another row does not launder this drop'
    # Nothing dropped -> silent, records or not.
    [ -z "$(cp_unjustified_drops "$bm" "$bm" '')" ] \
        && ok 'no drop is silent' || bad 'no drop is silent'
    # TWO drops, ONE record -> the unpaid one is named.
    [ "$(cp_unjustified_drops "$bm" 'apr run --gpu' 'apr code')" = 'lib:aprender-core::Lasso::fit' ] \
        && ok 'one record does not pay for two drops' || bad 'one record does not pay for two drops'
    # Silently dropping the ROW ENTIRELY is caught by cp_set_minus, not here --
    # asserted so the division of labour is a test rather than a comment.
    [ -n "$(cp_set_minus "$bm" 'apr run --gpu')" ] \
        && ok 'a vanished row is caught by the ROW set, not by the drop rule' \
        || bad 'a vanished row is caught by the ROW set, not by the drop rule'

    printf 'case table: cp_unbound_baseline_drops (the baseline is itself ratcheted)\n'
    # Without this the whole mechanism is optional: the baseline is unbound
    # plaintext read with `cat`, so drop the key AND the row and the working
    # copy agrees with itself. Judged against the COMMITTED file, which the
    # editor does not control.
    local bsand buni prior cur
    bsand=$(mktemp -d) || return 2
    mkdir -p "$bsand/crates/x/src"
    printf 'pub struct StillHere;\n' > "$bsand/crates/x/src/lib.rs"
    buni=$'apr run\napr serve\napr qa\nbin:pv\nbin:apr'
    prior=$'apr run --gpu\napr serve\nlib:aprender-core::StillHere::fit'
    # (a) nothing dropped -> silent.
    [ -z "$(cp_unbound_baseline_drops "$prior" "$prior" "$buni" '' "$bsand")" ] \
        && ok 'an unedited baseline is silent' || bad 'an unedited baseline is silent'
    # (b) a key hand-deleted while the entry point is STILL LIVE -> named.
    cur=$'apr run --gpu\nlib:aprender-core::StillHere::fit'
    [ "$(cp_unbound_baseline_drops "$prior" "$cur" "$buni" '' "$bsand")" = 'apr serve' ] \
        && ok 'a hand-deleted key for a LIVE entry point is named' \
        || bad 'a hand-deleted key for a LIVE entry point is named'
    # (c) a key whose entry point has genuinely left the binary -> silent.
    [ -z "$(cp_unbound_baseline_drops 'apr finetune' '' "$buni" '' "$bsand")" ] \
        && ok 'a key whose subcommand is GONE may be dropped' \
        || bad 'a key whose subcommand is GONE may be dropped'
    # (d) a lib: key whose symbol still exists -> named.
    [ "$(cp_unbound_baseline_drops 'lib:aprender-core::StillHere::fit' '' "$buni" '' "$bsand")" \
        = 'lib:aprender-core::StillHere::fit' ] \
        && ok 'a lib: key whose symbol still exists is named' \
        || bad 'a lib: key whose symbol still exists is named'
    # (e) EXCUSED: a MEASURED_ROW drop paid for by a live downgrade -> silent.
    [ -z "$(cp_unbound_baseline_drops 'apr serve' '' "$buni" 'apr serve' "$bsand")" ] \
        && ok 'a MEASURED_ROW drop with a live downgrade is excused' \
        || bad 'a MEASURED_ROW drop with a live downgrade is excused'
    # ...and a downgrade for a DIFFERENT row excuses nothing.
    [ "$(cp_unbound_baseline_drops 'apr serve' '' "$buni" 'apr qa' "$bsand")" = 'apr serve' ] \
        && ok 'a downgrade for another row excuses nothing' \
        || bad 'a downgrade for another row excuses nothing'
    # (f) THE WHOLE POINT: the drop is invisible to the working-copy checks,
    #     because after a hand-edit the baseline and the ledger AGREE.
    [ -z "$(cp_set_minus "$cur" "$cur")" ] \
        && ok 'the hand-edited baseline agrees with the ledger (so 5a is silent)' \
        || bad 'the hand-edited baseline agrees with the ledger (so 5a is silent)'
    rm -rf "${bsand:?}"

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

# The SETS. These, not the counts, are what makes a specific deletion visible.
LIVE_ROWS=$(cp_keys __ROW__ "$PV_OUT")
LIVE_MEASURED=$(cp_keys __MEASURED_ROW__ "$PV_OUT")
LIVE_DOWNGRADES=$(cp_keys __DOWNGRADE__ "$PV_OUT")

# -- 1b. the sets must agree with the emitter's own counts ------------------
# Control (c) on the key channel. A key line can only ever be ADDED to the
# stream by an injection (a newline inside a key printing extra well-formed key
# lines), so an injected line that got its length prefix right still puts the
# set out of step with the count the emitter computed from the parsed ledger.
# Cheap, and independent of both the character rule and the length prefix.
for pair in "__ROW__:$ROWS" "__MEASURED_ROW__:$MEASURED"; do
    k=${pair%%:*}; want=${pair#*:}
    got=$(cp_key_count "$k" "$PV_OUT")
    if ! [ "$got" = "$want" ]; then
        printf '✗ KEY CHANNEL CORRUPT: %d well-formed %s line(s) against a declared %s.\n' \
               "$got" "$k" "${want:-<none>}" >&2
        printf '    The set and the count come from the same parsed ledger, so they can only\n' >&2
        printf '    disagree if the text channel between them was perturbed -- an entry_point\n' >&2
        printf '    containing a newline prints EXTRA key lines and can satisfy a deleted\n' >&2
        printf '    row s baseline key from a fabricated row. Refusing to judge the sets.\n' >&2
        fail=1
    fi
done

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
BASELINE_TEXT=$(cat "$BASELINE")
NON_WINS_MIN=$(grep -E '^NON_WINS_MIN=[0-9]+$' "$BASELINE" | head -1 | cut -d= -f2)
IN_SCOPE_MIN=$(grep -E '^IN_SCOPE_MIN=[0-9]+$' "$BASELINE" | head -1 | cut -d= -f2)
BASE_ROWS=$(cp_keys ROW "$BASELINE_TEXT")
BASE_MEASURED=$(cp_keys MEASURED_ROW "$BASELINE_TEXT")

# -- 5x. the BASELINE FILE is itself ratcheted, against git ------------------
# Without this, every refusal below is optional: the baseline is unbound
# plaintext, so hand-editing it moves the bar and takes no code path that could
# refuse. Judged against the merge base with the upstream default branch, so a
# drop cannot be normalised by putting it in its own commit. Runs in BOTH modes:
# `--update-baseline` regenerates the ROW set from what is LIVE and compares it
# against the file on disk, so a hand-edited file would launder a deletion
# through the tool as well.
PRIOR_BASELINE=$(cp_baseline_at "$BASELINE")
PRIOR_RC=$?
if [ "$PRIOR_RC" -ne 0 ]; then
    printf '✗ could not read %s from git (rc=%d).\n' "$BASELINE" "$PRIOR_RC" >&2
    printf '    The baseline file is unbound plaintext; the only thing that binds it is\n' >&2
    printf '    its committed history, so a comparison that could not be MADE must be red\n' >&2
    printf '    rather than absent. Run this inside the git checkout.\n' >&2
    exit 1
fi
PRIOR_ROWS=$(cp_keys ROW "$PRIOR_BASELINE")
PRIOR_MEASURED=$(cp_keys MEASURED_ROW "$PRIOR_BASELINE")
PRIOR_NON_WINS_MIN=$(grep -E '^NON_WINS_MIN=[0-9]+$' <<<"$PRIOR_BASELINE" | head -1 | cut -d= -f2)
PRIOR_IN_SCOPE_MIN=$(grep -E '^IN_SCOPE_MIN=[0-9]+$' <<<"$PRIOR_BASELINE" | head -1 | cut -d= -f2)

while IFS= read -r gone; do
    [ -n "$gone" ] || continue
    printf '✗ BASELINE HAND-EDITED: ROW=%s was in the committed baseline and is not in\n' "$gone" >&2
    printf '    the working copy, and the entry point is still live. Editing this file is\n' >&2
    printf '    not a way to lower the bar: the refusals in --update-baseline are the same\n' >&2
    printf '    refusals, and they now run here too, against git. Without this the whole\n' >&2
    printf '    ratchet is optional -- drop the key here, delete the row, and the working\n' >&2
    printf '    copy agrees with itself.\n' >&2
    fail=1
done < <(cp_unbound_baseline_drops "$PRIOR_ROWS" "$BASE_ROWS" "$UNIVERSE" '')

while IFS= read -r gone; do
    [ -n "$gone" ] || continue
    printf '✗ BASELINE HAND-EDITED: MEASURED_ROW=%s was in the committed baseline and is\n' "$gone" >&2
    printf '    not in the working copy, with no downgrade in date for it. MEASURED_ROW is\n' >&2
    printf '    shrink-never precisely so that a downgrade is a DEBT the key keeps carrying,\n' >&2
    printf '    rather than a payment the baseline absorbs and the next commit deletes.\n' >&2
    fail=1
done < <(cp_unbound_baseline_drops "$PRIOR_MEASURED" "$BASE_MEASURED" "$UNIVERSE" "$LIVE_DOWNGRADES")

if [ -n "$PRIOR_NON_WINS_MIN" ] && ! cp_meets_floor "$NON_WINS_MIN" "$PRIOR_NON_WINS_MIN"; then
    printf '✗ BASELINE HAND-EDITED: NON_WINS_MIN %s -> %s is a LOWERING.\n' \
           "$PRIOR_NON_WINS_MIN" "${NON_WINS_MIN:-<none>}" >&2
    fail=1
fi
if [ -n "$PRIOR_IN_SCOPE_MIN" ] && ! cp_meets_floor "$IN_SCOPE_MIN" "$PRIOR_IN_SCOPE_MIN"; then
    printf '✗ BASELINE HAND-EDITED: IN_SCOPE_MIN %s -> %s is a LOWERING.\n' \
           "$PRIOR_IN_SCOPE_MIN" "${IN_SCOPE_MIN:-<none>}" >&2
    fail=1
fi
[ "$fail" -eq 0 ] || [ "$MODE" != "--update-baseline" ] || exit 1

# A baseline with no ROW set at all would make checks 5a-5c vacuously true --
# the "measurement that did not happen" failure, which must be RED and not
# absent (the coverage floor reported 0/0 for months while `|| true` kept it
# green). Refuse before judging anything against it.
if [ -z "$BASE_ROWS" ]; then
    if [ "$MODE" = "--update-baseline" ]; then
        # Bootstrap: seeding the very first set. Loud, because emptying the ROW
        # lines and re-seeding is the one route that would launder a deletion
        # through this tool -- and it is a visible diff on a tracked file, which
        # is what makes the loudness sufficient rather than the only defence.
        printf '! BOOTSTRAP: %s currently records no ROW= keys, so nothing is being\n' "$BASELINE" >&2
        printf '  ratcheted against. Review the emitted set against the ledger.\n' >&2
    else
        printf '✗ %s records no ROW= keys.\n' "$BASELINE" >&2
        printf '    The ratchet is a SET keyed by entry point; with an empty set every\n' >&2
        printf '    membership check passes vacuously and deleting any row is free.\n' >&2
        printf '    A check that cannot fail is worse than no check, because it is\n' >&2
        printf '    counted. Regenerate with --update-baseline.\n' >&2
        exit 1
    fi
fi

if [ "$MODE" = "--update-baseline" ]; then
    refuse=0

    # A ROW may leave the baseline only when the ENTRY POINT itself has left the
    # live enumeration -- the same test the scope uses. Deleting a comparison to
    # raise the ratio is the failure this gate exists to block (PMAT-733).
    while IFS= read -r gone; do
        [ -n "$gone" ] || continue
        if cp_removal_allowed "$gone" "$UNIVERSE"; then
            printf '  row drop OK: %s has left the enumeration\n' "$gone"
        else
            printf '✗ REFUSING to drop the row for %s: the entry point is still live.\n' "$gone" >&2
            printf '    Record the loss instead. WORSE and UNMEASURED are first-class\n' >&2
            printf '    verdicts and BOTH keep the row; only DELETION is refused.\n' >&2
            refuse=1
        fi
    done < <(cp_set_minus "$BASE_ROWS" "$LIVE_ROWS")

    # A MEASURED_ROW may leave the measured set only against a RECORDED
    # downgrade. This is the give that keeps the honest correction possible --
    # without it, filing the `apr code` row as UNMEASURED (which its own note
    # says it should be) is mechanically forbidden, and a ratchet that punishes
    # increasing honesty produces dishonest ledgers.
    while IFS= read -r dropped; do
        [ -n "$dropped" ] || continue
        printf '✗ REFUSING to drop %s from the MEASURED set: no downgrade is recorded.\n' \
               "$dropped" >&2
        printf '    Add a `downgrades:` entry to %s naming it, with a reason from the\n' "$LEDGER" >&2
        printf '    closed vocabulary (RECEIPT_MISSING / HARNESS_DELETED / ...), an\n' >&2
        printf '    owner, and a bounded recheck_by. Prose is not a reason: the\n' >&2
        printf '    vocabulary is a serde enum, so an invented one fails to PARSE.\n' >&2
        refuse=1
    done < <(cp_unjustified_drops "$BASE_MEASURED" "$LIVE_MEASURED" "$LIVE_DOWNGRADES")

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

    # MEASURED_ROW is SHRINK-NEVER, and that is what makes a downgrade a DEBT
    # rather than a one-off payment. If the baseline dropped the key once a
    # downgrade was recorded, the paperwork could be filed, absorbed, and then
    # deleted in the next commit with nothing left to notice. Keeping the key
    # means the row must EITHER come back to measured OR keep its record, for as
    # long as the row exists. The only keys that leave are those whose ROW left
    # legitimately (the entry point is gone from the live enumeration).
    NEW_MEASURED=$(
        {
            printf '%s\n' "$BASE_MEASURED"
            printf '%s\n' "$LIVE_MEASURED"
        } | grep -v '^$' | LC_ALL=C sort -u
    )
    NEW_MEASURED=$(
        while IFS= read -r k; do
            [ -n "$k" ] || continue
            grep -qxF -- "$k" <<<"$LIVE_ROWS" && printf '%s\n' "$k"
        done <<<"$NEW_MEASURED"
    )

    {
        printf '# Competitive-parity ratchet baseline. A SET, not a count.\n'
        printf '#\n'
        printf '# Written by scripts/check_competitive_parity.sh --update-baseline.\n'
        printf '# Do not hand-edit: the mechanism is that losing a SPECIFIC row is\n'
        printf '# visible even when every total is unchanged. A count ratchet is\n'
        printf '# payable in the wrong currency -- delete the StandardScaler 0.69x row,\n'
        printf '# add a cheaper one, and __MEASURED__ never moves. That is PMAT-733.\n'
        printf '#\n'
        printf '#   ROW=             must still EXIST. Droppable only when the entry\n'
        printf '#                    point has left the live `apr` enumeration.\n'
        printf '#   MEASURED_ROW=    SHRINK-NEVER. Must still be MEASURED, or carry a\n'
        printf '#                    recorded `downgrades:` entry in the ledger. The key\n'
        printf '#                    STAYS while the row exists, so a downgrade is a debt\n'
        printf '#                    with a due date, not a one-off payment that the\n'
        printf '#                    baseline absorbs and the next commit can delete.\n'
        printf 'NON_WINS_MIN=%s\n' "$NON_WINS"
        printf 'IN_SCOPE_MIN=%s\n' "$IN_SCOPE"
        printf '\n'
        cp_emit_keys ROW "$LIVE_ROWS"
        printf '\n'
        cp_emit_keys MEASURED_ROW "$NEW_MEASURED"
    } > "$BASELINE"
    printf '✓ baseline updated: %s row(s), %s measured, NON_WINS_MIN=%s IN_SCOPE_MIN=%s\n' \
           "$ROWS" "$MEASURED" "$NON_WINS" "$IN_SCOPE"
    exit 0
fi

# -- 5a. every baseline ROW must still EXIST --------------------------------
# THE FIX FOR THE COUNT RATCHET. Keyed by entry point, so deleting the
# StandardScaler row and adding a different one -- identical __ROWS__,
# __MEASURED__ and __NON_WINS__ -- names the missing key instead of passing.
while IFS= read -r missing; do
    [ -n "$missing" ] || continue
    printf '✗ ROW DELETED: %s is in %s and no longer in the ledger.\n' "$missing" "$BASELINE" >&2
    printf '    Totals prove nothing here: a row can be deleted and paid for with a\n' >&2
    printf '    cheaper one at constant __MEASURED__. That is exactly what d7e08043b\n' >&2
    printf '    did (PMAT-733), and it removed the only two losing rows in the\n' >&2
    printf '    history. Record the loss -- WORSE and UNMEASURED both KEEP the row.\n' >&2
    printf '    If the entry point genuinely left the binary, use --update-baseline,\n' >&2
    printf '    which will only allow it once the enumeration agrees.\n' >&2
    fail=1
done < <(cp_set_minus "$BASE_ROWS" "$LIVE_ROWS")

# -- 5b. every baseline MEASURED_ROW is measured, or downgraded ON RECORD ----
while IFS= read -r dropped; do
    [ -n "$dropped" ] || continue
    printf '✗ UNJUSTIFIED DOWNGRADE: %s was MEASURED and is not, with no record.\n' "$dropped" >&2
    printf '    Downgrading is ALLOWED -- a floor with no give forbids the honest\n' >&2
    printf '    correction, and that produces dishonest ledgers. It is not allowed\n' >&2
    printf '    SILENTLY. Add a `downgrades:` entry to %s naming this row, with a\n' "$LEDGER" >&2
    printf '    reason from the closed vocabulary, an owner and a bounded recheck_by.\n' >&2
    printf '    (If instead the row simply EXPIRED, re-measure it: an expired row\n' >&2
    printf '    also fails `pv parity-ledger` above.)\n' >&2
    fail=1
done < <(cp_unjustified_drops "$BASE_MEASURED" "$LIVE_MEASURED" "$LIVE_DOWNGRADES")

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
printf 'ledger rows           : %s (baseline set: %s)\n' \
       "${ROWS:-<none>}" "$(grep -c . <<<"$BASE_ROWS")"
printf 'measured (fresh)      : %s (baseline set: %s)\n' \
       "${MEASURED:-<none>}" "$(grep -c . <<<"$BASE_MEASURED")"
printf 'downgrades on record  : %s\n' "$(grep -c . <<<"$LIVE_DOWNGRADES")"
printf 'non-wins recorded     : %s (floor %s)\n' "${NON_WINS:-<none>}" "$NON_WINS_MIN"

if [ "$fail" -ne 0 ]; then
    printf '\n✗ competitive-parity ratchet FAILED\n' >&2
    exit 1
fi
printf '\n✓ competitive-parity ratchet OK\n'
exit 0
