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

# The commits this working tree is ratcheted AGAINST — the COMPARAND.
#
# WHY THE BASELINE FILE ALONE IS NOT A BASELINE. Every `--update-baseline`
# refusal in this script -- refusing to drop a live ROW, refusing an
# unjustified MEASURED_ROW drop, refusing to lower NON_WINS_MIN -- guards a code
# path nobody is obliged to take. The file is unbound plaintext read straight
# off disk with `cat`, so hand-editing it moves the bar and skips every refusal.
# A guard whose enforcement is optional is decoration.
#
# The fix is not a checksum stored beside it (which the same edit updates) but a
# value the editor does not control: the file AS COMMITTED.
#
# THE COMPARAND WAS VACUOUS IN THE CONFIGURATION CI RUNS IN
# ---------------------------------------------------------
# This used to be {merge-base, HEAD}, unioned, and it was defeated by the very
# evasion it named. `scripts/competitive_parity_baseline.txt` does NOT exist on
# origin/main -- it is new on this branch -- so the merge-base half contributed
# NOTHING, and the union collapsed to HEAD alone. HEAD alone is defeated by
# making the drop in its own commit: delete the StandardScaler 0.69x row, COMMIT
# the lowered baseline, and HEAD agrees with the working copy, so the ratchet
# prints OK at rc=0. PMAT-733 was payable in one commit. Both reviewers found
# this independently.
#
# Worse, the ABSENCE was silent. `cp_baseline_at` skipped a ref where the file
# did not exist, and a skip is an ACCEPTANCE -- inside the function whose own
# doc comment says a comparison that could not be MADE must be red rather than
# empty. A file the author edits and commits in the same change cannot audit
# that change; that is not a bug in the comparison, it is the absence of one.
#
# THE COMPARAND IS NOW THE WHOLE BRANCH
# -------------------------------------
# The merge base with the upstream default branch, UNION every commit between
# it and HEAD. Three properties, none of which the old pair had together:
#
#   * NON-VACUOUS from the second commit that touches the file, whatever main
#     contains. The bootstrap window -- main has no ledger at all -- is exactly
#     the window in which nothing can have been deleted yet, because there is
#     no prior state to delete from. The first commit introducing a ledger is
#     not a deletion; the second commit removing a row from it is, and the
#     first commit is in this list when the second one runs.
#   * IMMUNE to the own-commit evasion. Committing the lowered baseline no
#     longer normalises it: the commit before it still carries the key, and
#     every commit on the branch is judged against the union.
#   * MONOTONE. Adding refs can only ADD keys to the prior set, and the prior
#     set is what must still be satisfied, so a longer branch is strictly
#     stricter -- never a way to weaken the bar.
#
# WHAT IT STILL DOES NOT COVER, stated rather than left for a reviewer: branch
# history is rewritable. `git commit --amend` on the commit that introduced the
# ledger, or a rebase that squashes it away, removes evidence from this list.
# That is bounded in time and not in the general case: the moment the ledger
# lands on `main`, the merge base carries it, `main` is protected, and no
# rewrite can reach it. Inside the bootstrap window the comparand is only as
# strong as the branch's own history -- which is why a bootstrap must be
# DECLARED, out loud, in the file itself (see BOOTSTRAP below).
cp_base_refs() {
    local r mb
    mb=""
    for r in origin/main origin/master main master; do
        git rev-parse --verify --quiet "$r" >/dev/null 2>&1 || continue
        mb=$(git merge-base "$r" HEAD 2>/dev/null) || mb=""
        break
    done
    [ -n "$mb" ] && printf '%s\n' "$mb"
    # Every commit on the branch, newest first, HEAD included. Capped because
    # this costs two git calls per ref; a branch longer than the cap keeps the
    # merge base plus its most recent 500 commits, which is strictly more than
    # the two refs this replaced.
    if [ -n "$mb" ]; then
        git rev-list --max-count=500 "$mb..HEAD" 2>/dev/null
    else
        # No upstream default branch in this checkout (a bare clone of a fork, a
        # sandbox). Fall back to HEAD's own history rather than to nothing: an
        # absent comparand is RED, and it must be red for the RIGHT reason.
        git rev-list --max-count=500 HEAD 2>/dev/null
    fi
}

# The baseline file's content at every base ref, concatenated.
#
# Three outcomes, and the distinction between the last two is the whole of
# FATAL A:
#
#   rc=0  at least one ref supplied content -- a real comparand.
#   rc=2  git itself could not answer (not a checkout, object unreadable).
#   rc=3  the file exists at NO base ref. NOT the same as "no keys": it means
#         there is nothing to compare against, so every regression check below
#         would pass vacuously. This used to be a silent skip, i.e. an
#         ACCEPTANCE, which is the coverage-floor failure exactly (`|| true`
#         over a measurement that reported 0/0 for months).
#
# A comparison git cannot answer must be RED, never empty.
# TRUNCATED WHERE IT MATTERS, not merely shallow. `--is-shallow-repository`
# alone is the wrong test and saying why is the point: the dev box's checkout
# reports `true` with a graft boundary 739 commits back, while the merge base
# this ratchet needs is four commits back and fully present. Refusing that run
# would be a gate that reds for a reason unrelated to the property it guards,
# which trains people to re-run it -- and a red that gets re-run away is how a
# REAL red gets re-run away too.
#
# The property that matters is narrower: can a merge base with an upstream
# default branch be COMPUTED? If it can, the union spans it and every commit
# since, which is the entire comparand, and truncation older than that is
# irrelevant. If it cannot -- `fetch-depth: 1`, which fetches ONE ref and leaves
# no `origin/main`, no local `main`, and a detached HEAD -- cp_base_refs falls
# back to HEAD's own history, the file is present there, rc=0 comes back, and
# the comparand has collapsed to exactly the HEAD-alone behaviour that FATAL A
# was. Green, for a reason nobody would ever see.
#
# .github/workflows/ci.yml sets `fetch-depth: 0` on this job today, which is why
# this cannot fire there -- and is precisely why it must exist. A guard whose
# strength depends on a setting in another file has to assert that setting, or
# the day someone trims the checkout to speed it up the ratchet stops ratcheting
# and nothing goes red.
cp_history_is_truncated() {
    [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ] || return 1
    local r
    for r in origin/main origin/master main master; do
        git rev-parse --verify --quiet "$r" >/dev/null 2>&1 || continue
        git merge-base "$r" HEAD >/dev/null 2>&1 && return 1
        return 0
    done
    return 0
}

cp_baseline_at() {
    local path="$1" ref got=0
    git rev-parse --git-dir >/dev/null 2>&1 || return 2
    # rc=2, not rc=3: the history was not MEASURED as absent, it was not
    # there to measure. Different claims; only one is a bug in the tree.
    cp_history_is_truncated && return 2
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        git cat-file -e "$ref:$path" 2>/dev/null || continue
        git show "$ref:$path" 2>/dev/null || return 2
        got=1
    done < <(cp_base_refs)
    [ "$got" -eq 1 ] || return 3
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

# ---------------------------------------------------------------------------
# THE VERDICT-TRANSITION RATCHET
#
# EVERY DIMENSION EXCEPT `rows` WAS STILL A COUNT
# -----------------------------------------------
# Round 1 turned ROW and MEASURED_ROW into SETS precisely because a count is
# payable in the wrong currency. The other dimensions were left as counts, and
# they became the levers. The cheapest one, disclosed by the implementer and
# re-proved here: relabel BOTH recorded WORSE rows as NOT_COMPARABLE.
#
#     before: __ROWS__=5 __MEASURED__=3 __NON_WINS__=5
#     after:  __ROWS__=5 __MEASURED__=3 __NON_WINS__=5      rc=0
#
# Every total identical. Every baseline ROW key still present. Every
# MEASURED_ROW key still measured -- NOT_COMPARABLE is a measurement. And the
# StandardScaler 0.69x loss and the Lasso ~19x loss have both become "the
# competitor has no counterpart", for free. What left the tree is the
# DIRECTION of the result, which is what PMAT-733 was actually about.
#
# THE FIX IS NOT TO GATE ON THE VALUE
# -----------------------------------
# This gate deliberately NEVER checks that a verdict says BETTER, and that
# inversion is load-bearing: a rule admitting only wins makes deleting a losing
# comparison the cheapest compliant action, which is the move this repository
# has already made once. Adding "and it may not stop being a loss" would
# rebuild the fabrication engine on the other side of the fence.
#
# So the gate is on the TRANSITION. The VALUE stays completely unconstrained --
# any verdict may become any other -- and CHANGING it stops being FREE. A
# declared verdict that differs from every verdict in the COMMITTED baseline
# must be accompanied by a record naming both ends, with a reason from a closed
# serde vocabulary, an owner and a bounded recheck date. That is the machinery
# the MEASURED -> UNMEASURED downgrade already uses, GENERALISED, rather than a
# second mechanism beside it: two mechanisms means two vocabularies, two expiry
# rules and two sets of paperwork, of which exactly one stays current.
#
# The DECLARED verdict, never the effective one: a row degrading to UNMEASURED
# on its own expiry date is the clock moving, not an author relabelling, and
# demanding paperwork for it would make the gate red on a day nobody touched
# the file. Expiry is already handled, loudly, by `pv parity-ledger`.
# ---------------------------------------------------------------------------

# A composite key value is `FIELD<TAB>...<TAB>entry_point`. TAB is the
# separator because PARITY-002 admits only printable ASCII (0x20..0x7E) in a
# ratchet key, so a TAB can never occur INSIDE an entry point -- while spaces
# and `=` both genuinely do (`apr run --gpu (concurrency=1 single-request
# decode)`), which is why neither of those could be the separator.
CP_TAB=$'\t'

# The entry_point half of a `VERDICT<TAB>entry_point` value.
cp_verdict_entry() { printf '%s\n' "${1#*"$CP_TAB"}"; }

# The verdict half.
cp_verdict_of() { printf '%s\n' "${1%%"$CP_TAB"*}"; }

# Every DISTINCT entry_point named by a `V<TAB>entry` set.
cp_verdict_entries() {
    local line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        cp_verdict_entry "$line"
    done <<<"$1" | LC_ALL=C sort -u
}

# Every verdict `$2` has ever been recorded as in the set `$1`, one per line.
cp_verdicts_for() {
    local set="$1" entry="$2" line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        [ "$(cp_verdict_entry "$line")" = "$entry" ] || continue
        cp_verdict_of "$line"
    done <<<"$set"
}

# Rows whose DECLARED verdict changed with no in-date record describing the
# change. One `entry_point: WAS{,WAS...} -> NOW` line per unrecorded
# transition; empty output means every change is owned.
#
#   $1 prior verdicts  `V<TAB>entry` lines from the COMMITTED baseline. A UNION
#                      over every base ref, so a verdict this row has ALREADY
#                      held is prior -- which makes reverting a relabel free,
#                      as it should be: undoing a change needs no permission.
#   $2 live verdicts   `V<TAB>entry` lines from `pv parity-ledger`
#   $3 transitions     `FROM<TAB>TO<TAB>entry` lines, IN-DATE records only
#
# A row with NO prior verdict at all is a NEW row, not a change, and is silent:
# adding comparisons must stay free or the ledger stops growing.
cp_unrecorded_transitions() {
    local prior="$1" live="$2" trans="$3" line entry now was paid
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        entry=$(cp_verdict_entry "$line")
        now=$(cp_verdict_of "$line")
        was=$(cp_verdicts_for "$prior" "$entry")
        # Never recorded before -> a NEW row, not a transition.
        [ -n "$was" ] || continue
        # Unchanged, or back to a verdict this row has held before -> silent.
        grep -qxF -- "$now" <<<"$was" && continue
        # A record pays for it ONLY if it names this exact move. The `from`
        # end is read from the COMMITTED baseline, which the working tree does
        # not control, so a record cannot excuse a transition it does not
        # describe.
        paid=0
        while IFS= read -r from; do
            [ -n "$from" ] || continue
            grep -qxF -- "$from$CP_TAB$now$CP_TAB$entry" <<<"$trans" && paid=1
        done <<<"$was"
        [ "$paid" -eq 1 ] && continue
        printf '%s: %s -> %s\n' "$entry" "$(tr '\n' '/' <<<"$was" | sed 's:/$::')" "$now"
    done <<<"$live"
}

# Prior VERDICT_ROW values missing from the working-copy baseline whose row is
# still live. Nothing excuses one: the verdict history of a live row is
# shrink-never, because dropping the record of what a row USED to say is what
# makes the NEXT relabelling free. (`--update-baseline` therefore UNIONS rather
# than overwrites, and drops only keys whose row has left the enumeration.)
cp_unbound_verdict_drops() {
    local prior="$1" cur="$2" uni="$3" root="${4:-$REPO_ROOT}" k
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        cp_removal_allowed "$(cp_verdict_entry "$k")" "$uni" "$root" && continue
        printf '%s\n' "$k"
    done < <(cp_set_minus "$prior" "$cur")
}

# THE HIGHEST `KEY=<int>` value in a prior-baseline text, or empty if there is
# none.
#
# The prior baseline is a CONCATENATION over base refs, so every read of it has
# to be a set or an EXTREMUM -- never "the first line", which is whichever ref
# happened to be emitted first. This was `head -1` and it was wrong the moment
# the comparand widened from one ref to a union: cp_base_refs emits newest
# first, so `head -1` read the value from the commit UNDER TEST and the ratchet
# compared the mutation against itself. Caught by RE-RUNNING mutation A after
# widening the scope, never by reading the line -- extending a guard's scope
# requires re-mutating in the new scope, because the old proof does not
# transfer.
cp_prior_floor() {
    grep -E "^$1=[0-9]+\$" <<<"$2" | cut -d= -f2 | LC_ALL=C sort -n | tail -1
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

    printf 'case table: cp_base_refs / cp_baseline_at (the COMPARAND itself)\n'
    # FATAL A. This pair had NO fixtures -- only their definitions and one call
    # site -- so the table proved the COMPARATOR and never the COMPARAND. A
    # control whose input is never tested is tested only where it cannot fail,
    # and this one could not fail: the baseline file does not exist on
    # origin/main, so the merge-base half found nothing, `cp_baseline_at`
    # treated "absent here" as a SKIP (an acceptance), and the union collapsed
    # to HEAD alone -- which is defeated by committing the lowered baseline.
    #
    # These cases run against a REAL throwaway git repo, because every property
    # at issue is a property of git history and a string fixture cannot have one.
    local gsand grc gout
    gsand=$(mktemp -d) || return 2
    (
        cd "$gsand" || exit 2
        git init -q .
        git config user.email 'selftest@example.invalid'
        git config user.name 'selftest'
        printf 'x\n' > other.txt
        git add -A && git commit -qm 'c1: no baseline here'
        # `origin/main` as it really is: carrying no baseline file at all.
        git update-ref refs/remotes/origin/main HEAD
    ) >/dev/null 2>&1

    # (a) THE CI CONFIGURATION: the path exists at no base ref -> rc=3.
    gout=$(cd "$gsand" && cp_baseline_at b.txt)
    grc=$?
    [ "$grc" = 3 ] && [ -z "$gout" ] \
        && ok 'a path with NO history is rc=3 (NO COMPARAND), not rc=0 empty' \
        || bad "a path with NO history is rc=3 (NO COMPARAND), not rc=0 empty (got rc=$grc)"

    # A working-copy-only file is the same state: uncommitted is not history.
    printf 'ROW=1:A\n' > "$gsand/b.txt"
    gout=$(cd "$gsand" && cp_baseline_at b.txt)
    grc=$?
    [ "$grc" = 3 ] \
        && ok 'an UNCOMMITTED baseline is still NO COMPARAND' \
        || bad "an UNCOMMITTED baseline is still NO COMPARAND (got rc=$grc)"

    # (b) once committed, there IS a comparand.
    (cd "$gsand" && git add -A && git commit -qm 'c2: seed ROW=A') >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_baseline_at b.txt)
    grc=$?
    [ "$grc" = 0 ] && grep -qxF 'ROW=1:A' <<<"$gout" \
        && ok 'a committed baseline is rc=0 and carries its keys' \
        || bad "a committed baseline is rc=0 and carries its keys (rc=$grc)"

    # (c) THE OWN-COMMIT EVASION, which is how FATAL A was proved end to end:
    #     drop the key and COMMIT the lowered file. Against HEAD alone the
    #     comparand now agrees with the drop and the ratchet prints OK. Against
    #     the branch UNION the earlier commit still holds the key.
    printf 'ROW=1:B\n' > "$gsand/b.txt"
    (cd "$gsand" && git add -A && git commit -qm 'c3: drop ROW=A, in its own commit') >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_baseline_at b.txt)
    grc=$?
    [ "$grc" = 0 ] && grep -qxF 'ROW=1:A' <<<"$gout" \
        && ok 'a key dropped in its OWN COMMIT is still in the prior set' \
        || bad 'a key dropped in its OWN COMMIT is still in the prior set'
    # ...and that is exactly what makes the drop visible.
    [ "$(cp_set_minus "$(cp_keys ROW "$gout")" 'B')" = 'A' ] \
        && ok 'the own-commit drop is NAMED by the set difference' \
        || bad 'the own-commit drop is NAMED by the set difference'

    # (d) the ref list is the merge base PLUS every commit on the branch, so it
    #     grows with the branch and can only ever ADD prior keys.
    gout=$(cd "$gsand" && cp_base_refs | grep -c .)
    [ "$gout" -ge 3 ] \
        && ok 'cp_base_refs lists the merge base and every branch commit' \
        || bad "cp_base_refs lists the merge base and every branch commit (got $gout)"

    # (f) a checkout whose history is truncated WHERE THE COMPARAND LIVES does
    #     not have the comparand, and must say so. Left alone it collapses
    #     silently to HEAD-alone -- FATAL A again, arriving through a
    #     `fetch-depth:` line in a different file.
    local shal
    shal=$(mktemp -d) || return 2
    git clone -q --depth 1 "file://$gsand" "$shal/c" >/dev/null 2>&1
    if [ -d "$shal/c/.git" ]; then
        # A shallow clone that STILL has an upstream ref can compute a merge
        # base, so the comparand is present and the run is fine. This is the dev
        # box: `--is-shallow-repository` says true with a graft boundary 739
        # commits back, and the merge base is four commits back.
        (cd "$shal/c" && cp_history_is_truncated) \
            && bad 'shallow WITH a computable merge base is not truncated' \
            || ok 'shallow WITH a computable merge base is not truncated'
        # `fetch-depth: 1` is the real hazard: one ref fetched, no origin/main,
        # so cp_base_refs falls back to HEAD alone and says nothing about it.
        (
            cd "$shal/c" || exit 2
            # Faithful to a PR checkout at `fetch-depth: 1`: HEAD is
            # DETACHED at the SHA, there is no local `main`/`master`, and the
            # single fetched ref leaves no `origin/main` either.
            git checkout -q --detach HEAD
            # `git show-ref` rather than `git for-each-ref`: bashrs parses
            # the latter's `for-` prefix as the `for` KEYWORD and reports
            # SC1035, which would add a sixth false-positive error to this
            # file's lint count and make the real five harder to see.
            refs=$(git show-ref | cut -d' ' -f2)
            printf '%s\n' "$refs" | while IFS= read -r rr; do
                case "$rr" in
                    refs/heads/*|refs/remotes/*) git update-ref -d "$rr" ;;
                esac
            done
            git remote remove origin
        ) >/dev/null 2>&1
        (cd "$shal/c" && cp_history_is_truncated) \
            && ok 'shallow with NO upstream ref is truncated' \
            || bad 'shallow with NO upstream ref is truncated'
        gout=$(cd "$shal/c" && cp_baseline_at b.txt)
        grc=$?
        [ "$grc" = 2 ] \
            && ok 'a truncated history is rc=2, never a quietly smaller comparand' \
            || bad "a truncated history is rc=2, never a quietly smaller comparand (got rc=$grc)"
    else
        bad 'shallow-clone fixture could not be built'
    fi
    (cd "$gsand" && cp_history_is_truncated) \
        && bad 'a FULL checkout is not reported as truncated' \
        || ok 'a FULL checkout is not reported as truncated'
    rm -rf "${shal:?}"
    # (e) git unable to answer is rc=2 -- distinct from both rc=0 and rc=3,
    #     because "could not be measured" and "measured as absent" are
    #     different claims and only one of them is a bug in the tree.
    local nogit
    nogit=$(mktemp -d) || return 2
    gout=$(cd "$nogit" && cp_baseline_at b.txt)
    grc=$?
    [ "$grc" = 2 ] \
        && ok 'outside a git checkout is rc=2, distinct from rc=3' \
        || bad "outside a git checkout is rc=2, distinct from rc=3 (got rc=$grc)"
    rm -rf "${nogit:?}"
    rm -rf "${gsand:?}"

    printf 'case table: cp_unrecorded_transitions (relabelling is not free)\n'
    # The lever that survived two rounds: every dimension except `rows` was a
    # COUNT, so relabelling both WORSE rows NOT_COMPARABLE held __ROWS__,
    # __MEASURED__ and __NON_WINS__ constant while two recorded losses left the
    # tree. Note that no case below asserts anything about the verdict's VALUE.
    local pv_ lv_ tr_ sc_
    sc_="lib:aprender-core::StandardScaler::fit_transform"
    pv_="WORSE${CP_TAB}${sc_}"$'\n'"PARITY${CP_TAB}apr run --gpu"
    # (a) nothing changed -> silent.
    lv_="$pv_"
    [ -z "$(cp_unrecorded_transitions "$pv_" "$lv_" '')" ] \
        && ok 'an unchanged ledger is silent' || bad 'an unchanged ledger is silent'
    # (b) THE MOVE: WORSE -> NOT_COMPARABLE with no record -> named.
    lv_="NOT_COMPARABLE${CP_TAB}${sc_}"$'\n'"PARITY${CP_TAB}apr run --gpu"
    [ "$(cp_unrecorded_transitions "$pv_" "$lv_" '')" = "${sc_}: WORSE -> NOT_COMPARABLE" ] \
        && ok 'WORSE -> NOT_COMPARABLE with no record is NAMED' \
        || bad 'WORSE -> NOT_COMPARABLE with no record is NAMED'
    # (c) the SAME move, with a record that names it exactly -> silent.
    tr_="WORSE${CP_TAB}NOT_COMPARABLE${CP_TAB}${sc_}"
    [ -z "$(cp_unrecorded_transitions "$pv_" "$lv_" "$tr_")" ] \
        && ok 'a record naming exactly this move pays for it' \
        || bad 'a record naming exactly this move pays for it'
    # (d) a record naming a DIFFERENT destination pays for nothing. This is why
    #     both ends are required (PARITY-019): one open end would launder every
    #     future relabelling of the row.
    tr_="WORSE${CP_TAB}BETTER${CP_TAB}${sc_}"
    [ -n "$(cp_unrecorded_transitions "$pv_" "$lv_" "$tr_")" ] \
        && ok 'a record for a DIFFERENT move pays for nothing' \
        || bad 'a record for a DIFFERENT move pays for nothing'
    # (e) a record for a different ROW pays for nothing.
    tr_="WORSE${CP_TAB}NOT_COMPARABLE${CP_TAB}apr run --gpu"
    [ -n "$(cp_unrecorded_transitions "$pv_" "$lv_" "$tr_")" ] \
        && ok 'a record for another row pays for nothing' \
        || bad 'a record for another row pays for nothing'
    # (f) the UPWARD direction is gated identically. The gate never asks whether
    #     a verdict IMPROVED -- fabricating a win is as expensive as erasing a
    #     loss, and this repo has a fabricated win in its history too.
    lv_="BETTER${CP_TAB}${sc_}"$'\n'"PARITY${CP_TAB}apr run --gpu"
    [ "$(cp_unrecorded_transitions "$pv_" "$lv_" '')" = "${sc_}: WORSE -> BETTER" ] \
        && ok 'WORSE -> BETTER with no record is NAMED too' \
        || bad 'WORSE -> BETTER with no record is NAMED too'
    # (g) a NEW row has no prior verdict, so it is an addition, not a change.
    #     Adding comparisons must stay free or the ledger stops growing.
    lv_="$pv_"$'\n'"WORSE${CP_TAB}apr qa"
    [ -z "$(cp_unrecorded_transitions "$pv_" "$lv_" '')" ] \
        && ok 'a brand-new row is an addition, not a transition' \
        || bad 'a brand-new row is an addition, not a transition'
    # (h) REVERTING is free: the prior set is a UNION over every base ref, so a
    #     verdict this row has already held is not a new claim.
    pv_="WORSE${CP_TAB}${sc_}"$'\n'"NOT_COMPARABLE${CP_TAB}${sc_}"
    lv_="WORSE${CP_TAB}${sc_}"
    [ -z "$(cp_unrecorded_transitions "$pv_" "$lv_" '')" ] \
        && ok 'returning to a verdict the row has held before is free' \
        || bad 'returning to a verdict the row has held before is free'

    printf 'case table: cp_unbound_verdict_drops (forgetting is the cheapest relabel)\n'
    # Cheaper than any relabelling, because it needs no record at all: delete
    # the memory of what the row used to say and tomorrow's new verdict is a
    # brand-new row's first verdict as far as case (g) above can tell.
    local vsand vuni
    vsand=$(mktemp -d) || return 2
    mkdir -p "$vsand/crates/x/src"
    printf 'pub struct StillHere;\n' > "$vsand/crates/x/src/lib.rs"
    vuni=$'apr run\napr serve\napr qa\nbin:pv\nbin:apr'
    pv_="WORSE${CP_TAB}apr serve"$'\n'"PARITY${CP_TAB}apr run --gpu"
    [ -z "$(cp_unbound_verdict_drops "$pv_" "$pv_" "$vuni" "$vsand")" ] \
        && ok 'an unedited verdict history is silent' \
        || bad 'an unedited verdict history is silent'
    [ "$(cp_unbound_verdict_drops "$pv_" "PARITY${CP_TAB}apr run --gpu" "$vuni" "$vsand")" \
        = "WORSE${CP_TAB}apr serve" ] \
        && ok 'a hand-deleted verdict for a LIVE row is NAMED' \
        || bad 'a hand-deleted verdict for a LIVE row is NAMED'
    [ -z "$(cp_unbound_verdict_drops "WORSE${CP_TAB}apr finetune" '' "$vuni" "$vsand")" ] \
        && ok 'a verdict whose row has LEFT the enumeration may be dropped' \
        || bad 'a verdict whose row has LEFT the enumeration may be dropped'
    rm -rf "${vsand:?}"

    printf 'case table: cp_prior_floor (a UNION is read by extremum, never by head -1)\n'
    # The prior baseline is a CONCATENATION over base refs. `head -1` reads the
    # NEWEST -- the commit under test -- so the ratchet compared the mutation
    # against itself and a NON_WINS_MIN 5 -> 4 lowering went unnamed.
    local u_
    u_=$'NON_WINS_MIN=4\nIN_SCOPE_MIN=41\nNON_WINS_MIN=5\nIN_SCOPE_MIN=41'
    [ "$(cp_prior_floor NON_WINS_MIN "$u_")" = 5 ] \
        && ok 'the HIGHEST committed floor wins, not the first read' \
        || bad 'the HIGHEST committed floor wins, not the first read'
    [ "$(cp_prior_floor IN_SCOPE_MIN "$u_")" = 41 ] \
        && ok 'an unchanged floor reads back unchanged' \
        || bad 'an unchanged floor reads back unchanged'
    # Numeric, not lexicographic: `sort` without -n puts 9 above 41.
    [ "$(cp_prior_floor IN_SCOPE_MIN $'IN_SCOPE_MIN=9\nIN_SCOPE_MIN=41')" = 41 ] \
        && ok 'the comparison is NUMERIC (41 > 9)' \
        || bad 'the comparison is NUMERIC (41 > 9)'
    # Absent -> empty, which the caller treats as "no prior floor to enforce"
    # rather than as zero.
    [ -z "$(cp_prior_floor NON_WINS_MIN 'IN_SCOPE_MIN=41')" ] \
        && ok 'an absent floor is empty, not 0' || bad 'an absent floor is empty, not 0'
    # Anchored: a prose line mentioning the key is not a value.
    [ -z "$(cp_prior_floor NON_WINS_MIN '# NON_WINS_MIN=5 is the floor')" ] \
        && ok 'a COMMENT mentioning the key is not a value' \
        || bad 'a COMMENT mentioning the key is not a value'

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
# `VERDICT<TAB>entry_point` for every row (the DECLARED verdict), and
# `FROM<TAB>TO<TAB>entry_point` for every IN-DATE transition record.
LIVE_VERDICTS=$(cp_keys __VERDICT_ROW__ "$PV_OUT")
LIVE_TRANSITIONS=$(cp_keys __TRANSITION__ "$PV_OUT")

# -- 1b. the sets must agree with the emitter's own counts ------------------
# Control (c) on the key channel. A key line can only ever be ADDED to the
# stream by an injection (a newline inside a key printing extra well-formed key
# lines), so an injected line that got its length prefix right still puts the
# set out of step with the count the emitter computed from the parsed ledger.
# Cheap, and independent of both the character rule and the length prefix.
for pair in "__ROW__:$ROWS" "__MEASURED_ROW__:$MEASURED" "__VERDICT_ROW__:$ROWS"; do
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
# Read from the MACHINE-READABLE channel (`$LIVE_ROWS`, the length-prefixed
# `__ROW__=` set), not from the human report.
#
# This used to be `grep -E '^ROW ' | sed -E 's/.*valid_until=[^ ]+ //'` -- a
# hand-rolled parser of a DISPLAY format, and spoofable in both directions. A
# `note:` or an `owner:` containing a newline followed by `ROW  ` injects a row
# into this check's input; and an entry point containing the literal text
# `valid_until=x ` truncates the key the `sed` produces, so the scope lookup
# runs against something the ledger never said. The anchored, length-prefixed
# channel exists precisely so no consumer has to do this, and the same
# `cp_keys` that parses the baseline parses it -- the two cannot drift.
while IFS= read -r row; do
    [ -n "$row" ] || continue
    if ! grep -qxF -- "$(cp_scope_key "$row")" <<<"$SCOPE_ENTRIES"; then
        printf '✗ OUT-OF-SCOPE ROW: %s\n' "$row" >&2
        printf '    A row scored against a universe it is not part of inflates the\n' >&2
        printf '    ratio without measuring anything. Add it to %s.\n' "$SCOPE" >&2
        fail=1
    fi
done <<<"$LIVE_ROWS"

# -- 5. the baseline --------------------------------------------------------
BASELINE_TEXT=$(cat "$BASELINE")
NON_WINS_MIN=$(grep -E '^NON_WINS_MIN=[0-9]+$' "$BASELINE" | head -1 | cut -d= -f2)
IN_SCOPE_MIN=$(grep -E '^IN_SCOPE_MIN=[0-9]+$' "$BASELINE" | head -1 | cut -d= -f2)
BASE_ROWS=$(cp_keys ROW "$BASELINE_TEXT")
BASE_MEASURED=$(cp_keys MEASURED_ROW "$BASELINE_TEXT")
BASE_VERDICTS=$(cp_keys VERDICT_ROW "$BASELINE_TEXT")

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
if [ "$PRIOR_RC" -eq 2 ]; then
    printf '✗ could not read %s from git (rc=%d).\n' "$BASELINE" "$PRIOR_RC" >&2
    printf '    The baseline file is unbound plaintext; the only thing that binds it is\n' >&2
    printf '    its committed history, so a comparison that could not be MADE must be red\n' >&2
    printf '    rather than absent. Run this inside the git checkout.\n' >&2
    if cp_history_is_truncated; then
        printf '    THIS CHECKOUT CANNOT REACH ITS COMPARAND: the history is shallow AND no\n' >&2
        printf '    upstream default branch is present, so no merge base can be computed and\n' >&2
        printf '    the ratchet would fall back to HEAD alone -- exactly the collapsed\n' >&2
        printf '    comparand that let a deleted row pass. Check out with full history\n' >&2
        printf '    (actions/checkout `fetch-depth: 0`), or `git fetch --unshallow`.\n' >&2
    fi
    exit 1
fi
if [ "$PRIOR_RC" -eq 3 ]; then
    # NO COMPARAND. The file exists at no base ref, so every regression check
    # below would pass vacuously -- which is FATAL A: `origin/main` does not
    # carry this file, both halves of the old {merge-base, HEAD} pair skipped,
    # and a skip is an ACCEPTANCE. Deleting the StandardScaler 0.69x row and
    # committing the lowered baseline printed OK at rc=0.
    #
    # Renaming the file is the same hole by another route: point $BASELINE at a
    # path with no history and the comparand vanishes silently. It does not
    # vanish silently any more.
    #
    # The one legitimate instance of this state is the very first commit that
    # introduces the file, and it must be DECLARED -- in the file, in a commit
    # that says so. The declaration is self-limiting rather than trusted: it
    # only has effect while the path has no history at all, and once the file
    # is on `main` the merge base carries it forever, so this branch is
    # unreachable no matter what the file says.
    BOOTSTRAP_DECL=$(grep -E '^BOOTSTRAP=..*$' "$BASELINE" | head -1 | cut -d= -f2-)
    if [ -z "$BOOTSTRAP_DECL" ]; then
        printf '✗ NO COMPARAND: %s exists at none of the base refs.\n' "$BASELINE" >&2
        printf '    merge-base(origin/main, HEAD) and every commit on this branch were\n' >&2
        printf '    searched and none of them has this path. Nothing is being ratcheted\n' >&2
        printf '    against, so every check below would pass VACUOUSLY -- a file the\n' >&2
        printf '    author edits and commits in the same change cannot audit that change.\n' >&2
        printf '    A comparison git cannot answer must be RED, never empty.\n' >&2
        printf '    If this really is the commit that introduces the file, declare it:\n' >&2
        printf '      BOOTSTRAP=<why, in one line>   as a line in %s\n' "$BASELINE" >&2
        printf '    and say so in the commit message. That declaration stops working the\n' >&2
        printf '    moment the file has any history, which is permanent once it lands.\n' >&2
        exit 1
    fi
    printf '! BOOTSTRAP: %s has no committed history at any base ref, so nothing is\n' "$BASELINE" >&2
    printf '  being ratcheted against on THIS run. Declared reason: %s\n' "$BOOTSTRAP_DECL" >&2
    printf '  The declaration is accepted exactly once per path-with-no-history. Review\n' >&2
    printf '  the emitted sets against the ledger by hand; from the next commit the\n' >&2
    printf '  branch itself is the comparand and this branch is unreachable.\n' >&2
    # In bootstrap the file cannot be a LOWERED copy of the ledger, because
    # there is nothing to lower it from -- but it can be a copy that omits rows
    # the ledger has. That much IS checkable without history, so it is checked:
    # the seeded ROW set must be exactly the live one.
    while IFS= read -r missing; do
        [ -n "$missing" ] || continue
        printf '✗ BOOTSTRAP INCOMPLETE: the ledger has row %s and the baseline does not.\n' \
               "$missing" >&2
        printf '    A bootstrap seeds the CURRENT state; it is not an opportunity to seed\n' >&2
        printf '    a smaller one. Regenerate with --update-baseline.\n' >&2
        fail=1
    done < <(cp_set_minus "$LIVE_ROWS" "$BASE_ROWS")
fi
# The prior baseline is a CONCATENATION over refs, so every read of it has to
# be a set or an extremum -- never "the first line", which is whichever ref
# happened to be emitted first.
PRIOR_ROWS=$(cp_keys ROW "$PRIOR_BASELINE" | LC_ALL=C sort -u)
PRIOR_MEASURED=$(cp_keys MEASURED_ROW "$PRIOR_BASELINE" | LC_ALL=C sort -u)
PRIOR_VERDICTS=$(cp_keys VERDICT_ROW "$PRIOR_BASELINE" | LC_ALL=C sort -u)
# THE HIGHEST floor any base ref committed, not the first one read.
#
# Caught by re-running mutation A after widening the comparand, which is the
# standing lesson here: extending a guard's SCOPE requires re-mutating in the
# new scope, because the old proof does not transfer. With the comparand a
# single ref, `head -1` was the only value there was. With the comparand a
# UNION, `head -1` is the NEWEST ref -- which is the one carrying the lowered
# floor, so the check read the mutation's own value as the bar and the
# NON_WINS_MIN 5 -> 4 lowering in mutation A went unnamed. A ratchet must read
# the extremum of its history, never a member of it.
PRIOR_NON_WINS_MIN=$(cp_prior_floor NON_WINS_MIN "$PRIOR_BASELINE")
PRIOR_IN_SCOPE_MIN=$(cp_prior_floor IN_SCOPE_MIN "$PRIOR_BASELINE")

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

# The verdict HISTORY of a live row is shrink-never, with nothing that pays for
# a drop. Forgetting what a row used to say is what makes the next relabelling
# free -- and it is a cheaper move than any of them, because it needs no record
# at all: drop `VERDICT_ROW=54:WORSE<TAB>lib:...StandardScaler...` from the
# baseline today and tomorrow's NOT_COMPARABLE is a brand-new row's first
# verdict as far as the transition check can tell.
while IFS= read -r gone; do
    [ -n "$gone" ] || continue
    printf '✗ BASELINE HAND-EDITED: VERDICT_ROW=%q was in the committed baseline and is\n' "$gone" >&2
    printf '    not in the working copy, and that row is still live. The verdict a row USED\n' >&2
    printf '    to declare is what makes changing it cost something; deleting the memory is\n' >&2
    printf '    the cheapest relabelling of all, because it needs no record whatever.\n' >&2
    fail=1
done < <(cp_unbound_verdict_drops "$PRIOR_VERDICTS" "$BASE_VERDICTS" "$UNIVERSE")

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
    # A VERDICT may be re-declared only against a record that names the move.
    # Same refusal as check 5c, run here so `--update-baseline` cannot launder
    # a relabelling into the baseline -- the tool must never be the cheap route
    # to a state the check would refuse.
    while IFS= read -r moved; do
        [ -n "$moved" ] || continue
        printf '✗ REFUSING to record the verdict change %s: no in-date record names it.\n' \
               "$moved" >&2
        printf '    The VALUE is unconstrained -- any verdict may become any other. The\n' >&2
        printf '    CHANGE is not free: add a `downgrades:` entry with from_verdict /\n' >&2
        printf '    to_verdict matching this move, a reason from the closed vocabulary, an\n' >&2
        printf '    owner and a bounded recheck_by.\n' >&2
        refuse=1
    done < <(cp_unrecorded_transitions "$PRIOR_VERDICTS" "$LIVE_VERDICTS" "$LIVE_TRANSITIONS")

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

    # VERDICT_ROW accumulates: the UNION of every verdict a live row has ever
    # been recorded as, never an overwrite. Overwriting would erase the memory
    # of the previous verdict, and an erased memory is a free relabelling next
    # commit -- the cheapest move of all, because it needs no record. Same
    # retirement rule as MEASURED_ROW: only keys whose row has left the live
    # enumeration are dropped.
    NEW_VERDICTS=$(
        {
            printf '%s\n' "$BASE_VERDICTS"
            printf '%s\n' "$LIVE_VERDICTS"
        } | grep -v '^$' | LC_ALL=C sort -u
    )
    NEW_VERDICTS=$(
        while IFS= read -r k; do
            [ -n "$k" ] || continue
            grep -qxF -- "$(cp_verdict_entry "$k")" <<<"$LIVE_ROWS" && printf '%s\n' "$k"
        done <<<"$NEW_VERDICTS"
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
        printf '#   VERDICT_ROW=     <VERDICT><TAB><entry_point>, ACCUMULATING. Every\n'
        printf '#                    verdict a live row has ever been recorded as. A\n'
        printf '#                    declared verdict outside this set for its row needs a\n'
        printf '#                    `downgrades:` record naming from_verdict/to_verdict.\n'
        printf '#                    The gate never checks the verdict VALUE -- a rule that\n'
        printf '#                    admits only wins makes deleting a losing comparison the\n'
        printf '#                    cheapest compliant action, which is PMAT-733 itself. It\n'
        printf '#                    checks that CHANGING one is not free.\n'
        # Carried through, never invented. A bootstrapper is told to run this
        # tool, so the tool must not delete the declaration it just demanded --
        # and it cannot ADD one, because it only copies a line the file already
        # had. The declaration is inert anyway once the path has history.
        [ -n "${BOOTSTRAP_DECL:-}" ] && printf 'BOOTSTRAP=%s\n' "$BOOTSTRAP_DECL"
        printf 'NON_WINS_MIN=%s\n' "$NON_WINS"
        printf 'IN_SCOPE_MIN=%s\n' "$IN_SCOPE"
        printf '\n'
        cp_emit_keys ROW "$LIVE_ROWS"
        printf '\n'
        cp_emit_keys MEASURED_ROW "$NEW_MEASURED"
        printf '\n'
        cp_emit_keys VERDICT_ROW "$NEW_VERDICTS"
    } > "$BASELINE"
    printf '✓ baseline updated: %s row(s), %s measured, %s verdict value(s), NON_WINS_MIN=%s IN_SCOPE_MIN=%s\n' \
           "$ROWS" "$MEASURED" "$(grep -c . <<<"$NEW_VERDICTS")" "$NON_WINS" "$IN_SCOPE"
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

# -- 5c. every DECLARED VERDICT CHANGE carries a record that names it --------
# The lever that survived rounds 1 and 2: every dimension except `rows` was
# still a count, so relabelling both WORSE rows to NOT_COMPARABLE left
# __ROWS__, __MEASURED__ and __NON_WINS__ bit-for-bit identical at rc=0 while
# the 0.69x and ~19x losses became "no counterpart exists".
#
# Note what is NOT checked: nothing here reads the verdict's VALUE. WORSE may
# become BETTER, BETTER may become WORSE, anything may become anything. Gating
# on the value would rebuild the fabrication engine -- a rule admitting only
# wins is why the StandardScaler row was deleted in the first place. What is
# gated is the CHANGE, against the same closed-vocabulary, owned, dated,
# expiring record the MEASURED -> UNMEASURED downgrade already uses.
while IFS= read -r moved; do
    [ -n "$moved" ] || continue
    printf '✗ UNRECORDED VERDICT CHANGE: %s\n' "$moved" >&2
    printf '    The row still exists and every total is unchanged - which is exactly why\n' >&2
    printf '    this needs its own check. Relabelling both WORSE rows NOT_COMPARABLE holds\n' >&2
    printf '    __ROWS__, __MEASURED__ and __NON_WINS__ constant while the DIRECTION of the\n' >&2
    printf '    result leaves the tree, and the direction is what PMAT-733 was about.\n' >&2
    printf '    The value is NOT constrained: any verdict may become any other. Changing it\n' >&2
    printf '    is not FREE. Add a `downgrades:` entry to %s naming this row with\n' "$LEDGER" >&2
    printf '    from_verdict / to_verdict matching this move, a reason from the closed\n' >&2
    printf '    vocabulary, an owner and a bounded recheck_by - then --update-baseline.\n' >&2
    fail=1
done < <(cp_unrecorded_transitions "$PRIOR_VERDICTS" "$LIVE_VERDICTS" "$LIVE_TRANSITIONS")

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
printf 'verdict history       : %s value(s) over %s row(s)\n' \
       "$(grep -c . <<<"$BASE_VERDICTS")" "$(cp_verdict_entries "$BASE_VERDICTS" | grep -c .)"
printf 'excuse budget         : %s spent of %s (in-date records vs measured rows)\n' \
       "$(cp_extract __EXCUSES__ "$PV_OUT")" "$(cp_extract __EXCUSE_BUDGET__ "$PV_OUT")"
printf 'non-wins recorded     : %s (floor %s)\n' "${NON_WINS:-<none>}" "$NON_WINS_MIN"

if [ "$fail" -ne 0 ]; then
    printf '\n✗ competitive-parity ratchet FAILED\n' >&2
    exit 1
fi
printf '\n✓ competitive-parity ratchet OK\n'
exit 0
