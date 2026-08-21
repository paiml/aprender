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
# THE RATCHET IS A SET, NOT A COUNT
# ---------------------------------
# The first version enforced `__MEASURED__ >= 4` and never recorded WHICH entry
# points held a verdict. A count is payable in the wrong currency: DELETE the
# StandardScaler 0.69x row, ADD a cheaper fabricated one, and every total is
# unchanged while the only losing measurement in the history has left the tree
# again. That is PMAT-733 with the arithmetic balanced.
#
# So the comparison is between SETS keyed by entry_point:
#
#   __ROW__                     every row that must still EXIST. Shrink-never.
#   __DECLARED_MEASURED_ROW__   every row whose verdict must still be MEASURED
#                               -- unless a downgrade is RECORDED for it. Also
#                               shrink-never, which is what makes a downgrade a
#                               DEBT rather than a one-off payment.
#   __VERDICT_ROW__             what each row DECLARED. Changing it needs a
#                               record naming the exact move.
#   __COVERAGE_STEP__           the coverage schedule. Also shrink-never.
#
# THE COMPARAND IS THE LEDGER ON PROTECTED `main`
# -----------------------------------------------
# Four rounds each bounded ONE author-writable quantity, and each time the lever
# moved one level up: dates -> row sets -> the baseline FILE -> the transition
# PERMISSION SET. The invariant behind all of them:
#
#     ANY STATE THE AUTHOR WRITES AND THE GATE READS CAN BE MOVED IN THE SAME
#     COMMIT.
#
# scripts/competitive_parity_baseline.txt is therefore GONE, along with
# `--update-baseline`. `main` is protected -- changing it takes a PR, a review
# and this gate -- so the ledger at the upstream default branch is the one prior
# state a commit cannot rewrite from inside itself, and every expected value is
# DERIVED from it by running the same `pv parity-ledger` over it. See
# `cp_comparand_rev` for the full argument, including why the comparand is main
# ALONE and not main unioned with the branch.
#
# HONESTY MUST STAY AFFORDABLE, IN BOTH DIRECTIONS
# ------------------------------------------------
# The mirror-image failure is a floor with no give, and this file has now had it
# twice. `MEASURED_MIN=4` made the honest `apr code` DOWNGRADE mechanically
# forbidden. Then `NON_WINS_MIN=5` over 5 rows made recording an honest WIN
# mechanically forbidden -- the fabrication engine arrived at from the other
# side, because the cheapest compliant action for a genuine improvement becomes
# not recording it.
#
# So: the set of rows that EXIST may never shrink; the set of MEASURED rows may,
# against a `downgrades:` record; and the NON-WIN COUNT floor is
# `non_wins(main) - upgrades recorded here`, so a win costs one owned, dated,
# expiring record and nothing more. Recording an additional loss never breaches
# anything -- there is no ceiling.
#
# THE KEY CHANNEL IS VERIFIED, NOT TRUSTED
# ----------------------------------------
# Set membership travels from `pv` to this script as text, so the channel is
# part of the mechanism. Under the first wire format (`__ROW__=<rest of line>`)
# an entry_point containing a NEWLINE printed several well-formed key lines from
# ONE row, so a fabricated row could satisfy a DELETED row's key at constant
# totals -- the set ratchet defeated by exactly the move it was built to block.
# Three independent controls, because any one of them is a single edit from
# useless: PARITY-002 refuses the character at the SOURCE; every key is
# LENGTH-PREFIXED (`__ROW__=<bytes>:<key>`) and a line whose declared length
# does not match what follows is DROPPED; and the NUMBER of key lines is
# cross-checked against the emitter's own counts, which an injection can only
# inflate. The prior sets travel over the SAME channel, parsed by the SAME
# function, so the two sides cannot drift.
#
# WHAT IT ENFORCES
# ----------------
#   1. `pv parity-ledger` passes  -- freshness evaluated AT CHECK TIME, for
#      every verdict class; the excuse budget; and the COVERAGE RATCHET
#      (PARITY-021..024), whose schedule, reasoning and dissent live in the
#      contract rather than in a sibling file.
#   2. Every row in the ledger at `main` still EXISTS here. Set-keyed, so losing
#      a SPECIFIC row is RED at constant totals.
#   3. Every row DECLARED measured at `main` is still measured here, OR carries
#      a recorded downgrade in this tree's `downgrades:` block.
#   4. Every DECLARED VERDICT that differs from `main`'s carries a record naming
#      the exact move. The VALUE is never checked -- a rule admitting only wins
#      is why the StandardScaler row was deleted in the first place.
#   5. __NON_WINS__ >= non_wins(main) - recorded upgrades.
#   6. The scope SET at `main` is still covered, unless an entry point has left
#      the live enumeration from a SHA-PINNED `apr` (`. scripts/apr_bin.sh`).
#   7. Every ledger row's entry_point is IN scope, so a row cannot be scored
#      against a universe it is not part of.
#   8. Coverage may not fall, and no coverage step may be deleted, lowered or
#      DEFERRED relative to the schedule on `main`.
#
#   bash scripts/check_competitive_parity.sh                    # check
#   bash scripts/check_competitive_parity.sh --self-test        # case table
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

# ---------------------------------------------------------------------------
# THE COMPARAND: THE LEDGER ON PROTECTED `main`
#
# THE ROOT CAUSE THIS REPLACES
# ----------------------------
# Four rounds each bounded one author-writable quantity, and each time the next
# round found the lever one level up: dates -> row sets -> the baseline FILE ->
# the transition PERMISSION SET. Read together they are one invariant:
#
#     ANY STATE THE AUTHOR WRITES AND THE GATE READS CAN BE MOVED IN THE SAME
#     COMMIT.
#
# scripts/competitive_parity_baseline.txt was that state. Every refusal written
# against it -- ROW shrink-never, MEASURED_ROW shrink-never, NON_WINS_MIN
# never-lower, the accumulating VERDICT_ROW history -- read a value the same
# commit could rewrite. Round 4 tried to bind it to git and the binding read
# {merge-base} UNION {every commit on the branch}, which INCLUDES HEAD, so the
# author still wrote the bar: one commit renaming the baseline, deleting the
# StandardScaler 0.69x row, dropping its keys and lowering NON_WINS_MIN 5 -> 4
# exits 0. And measured on 4813bd41e, one hand-written line
#
#     VERDICT_ROW=63:NOT_COMPARABLE<TAB>lib:...StandardScaler::fit_transform
#
# committed BESIDE the relabel that needed it made re-declaring that recorded
# loss NOT_COMPARABLE exit 0. Deleting only that line made the identical tree
# exit 1. The permission for the change was issued by the change.
#
# A SIBLING FILE CANNOT AUDIT THE COMMIT THAT EDITS IT. There is no version of
# it that can; the fix is not a better binding but a different comparand.
#
# WHAT IS COMPARED NOW
# --------------------
# `main` is protected: changing it requires a PR, a review and a passing gate.
# So the ledger as it exists at the upstream default branch is the one prior
# state an author cannot rewrite from inside their own commit. Every expected
# value is DERIVED from it by running the SAME `pv parity-ledger` over it that
# runs over HEAD's ledger -- row set, declared-measured set, verdict per row,
# non-win count, scope, coverage schedule. There is no baseline file at all, so
# there is nothing beside the thing under test to edit.
#
# Three defects collapse into this one change:
#
#   * FATAL A (the baseline was hand-editable and HEAD was its own comparand):
#     the file is GONE.
#   * the transition PERMISSION SET: `from` verdicts come from `main`, and the
#     only thing HEAD can write is a `downgrades:` record -- which is dated,
#     owned, closed-vocabulary, expiring, and budget-limited. A prior verdict
#     can no longer be invented.
#   * the FLOORS: NON_WINS and coverage are computed from `main`, not read from
#     a number the author typed.
#
# WHY MAIN ALONE, AND NOT MAIN UNIONED WITH THE BRANCH
# ----------------------------------------------------
# Deliberate, and it is the difference between this and round 4. Unioning the
# branch's own commits ADDS prior verdicts, and an added prior verdict is a
# PERMISSION: `cp_unrecorded_transitions` treats a verdict a row has held
# before as free, so a two-commit branch could grant itself the relabel it
# wanted in commit 1 and take it in commit 2. Widening the comparand toward the
# author widens the permission set. The comparand is the protected state and
# nothing else.
#
# The cost, stated: a row added and deleted entirely within one branch is free.
# That is correct -- nothing that ever reached the protected state was lost --
# and it is the same freedom as never having written it.
#
# WHAT THIS STILL DOES NOT COVER, stated rather than left for a reviewer: it is
# only as strong as the protection on `main`. An operator who can force-push
# the default branch can move the comparand. That is a repository setting, not
# a property of this file, and it is the *reason* main was chosen rather than
# an oversight in choosing it.

# The upstream default-branch ref, or nothing.
cp_upstream_ref() {
    local r
    for r in origin/main origin/master main master; do
        git rev-parse --verify --quiet "$r" >/dev/null 2>&1 || continue
        printf '%s\n' "$r"
        return 0
    done
    return 1
}

# The COMPARAND REVISION: the protected state this tree is judged against.
#
# Normally the upstream default branch. The exception is a run whose HEAD IS
# that branch -- a push-to-main or a post-merge run -- where comparing a tree
# against itself is vacuous in exactly the way this whole round exists to
# remove. There the comparand steps back one commit, which is the previous
# protected state: the same comparison the merge-queue run already made, so it
# can red for nothing the queue did not already see.
cp_comparand_rev() {
    local up rev head
    up=$(cp_upstream_ref) || return 1
    rev=$(git rev-parse --verify --quiet "$up^{commit}") || return 1
    [ -n "$rev" ] || return 1
    head=$(git rev-parse --verify --quiet 'HEAD^{commit}') || head=""
    if [ "$rev" = "$head" ]; then
        rev=$(git rev-parse --verify --quiet "$up^^{commit}") || return 1
        [ -n "$rev" ] || return 1
    fi
    printf '%s\n' "$rev"
}

# Is the COMPARAND out of reach in this checkout?
#
# RE-DERIVED IN THE NEW SCOPE, not inherited. Round 3 added
# `cp_history_is_truncated` for the comparand of the time -- a MERGE BASE -- and
# its property was "can a merge base be computed". The comparand has moved, so
# that proof does not transfer: the question now is whether an upstream
# default-branch ref EXISTS and whether its tree is READABLE. Round 4's lesson
# is exactly this one (widening the scope surfaced a NON_WINS_MIN floor bug the
# first mutation could not), so the rule is rewritten here and re-mutated here
# rather than re-read.
#
# At `actions/checkout` `fetch-depth: 1` there is no `origin/main`, no local
# `main`, and a detached HEAD: this returns TRUE and the run is RED. Without it
# the comparand would silently collapse to "no prior ledger", which reads as a
# BOOTSTRAP -- i.e. the strongest possible pass -- from a line in a different
# file. ci.yml sets `fetch-depth: 0` today, which is why this cannot fire there
# and precisely why it must exist.
#
# NARROW ON PURPOSE, for the reason round 4 gave: `--is-shallow-repository`
# alone would red this dev box, whose graft boundary is hundreds of commits back
# while the ref it needs is present. A gate that reds for a reason unrelated to
# its property trains people to re-run it, and a red that gets re-run away is
# how a real red gets re-run away too.
cp_comparand_unreachable() {
    git rev-parse --git-dir >/dev/null 2>&1 || return 0
    cp_upstream_ref >/dev/null 2>&1 || return 0
    local rev
    rev=$(cp_comparand_rev) || return 0
    git cat-file -e "$rev^{tree}" 2>/dev/null || return 0
    return 1
}

# Every competitive-parity contract present at `$1`, one repo-relative path per
# line.
#
# DISCOVERED BY KIND, NEVER BY PATH, and that is what makes the bootstrap
# unrenewable. Round 4 keyed the "no prior state" window on the PATH of a
# sibling file, so `git mv` manufactured a fresh window silently -- the
# bootstrap was renewable, and a renewable bootstrap is `registry: true`
# wearing another hat. Renaming the ledger cannot hide a contract of this kind
# from a grep over the whole `contracts/` tree at the protected ref.
cp_parity_contracts_at() {
    local rev="$1"
    git grep -I -l -E '^[[:space:]]*kind:[[:space:]]*competitive-parity[[:space:]]*$' \
        "$rev" -- 'contracts/*.yaml' 'contracts/*.yml' 2>/dev/null \
        | sed "s|^${rev}:||"
}

# The comparand REPORT: `pv parity-ledger` evaluated over the ledger as it
# exists at `$1:$2`. Prints the report; rc=2 when it could not be produced.
#
# THE RC OF `pv` IS DELIBERATELY NOT THE TEST. `main`'s ledger legitimately
# exits non-zero as it ages -- a row past `valid_until` blocks, which is the
# mechanism working -- and it still EMITS its machine-readable block before
# doing so. What must be true is that the block was emitted at all, so the test
# is the presence of the emitter's own anchored `__ROWS__=` line. Using the rc
# would make every expired row on `main` collapse the comparand to nothing,
# which reads as BOOTSTRAP: the strongest possible pass, triggered by the clock.
#
# The one failure that IS fatal is a comparand that will not PARSE. That is why
# every block added to this schema arrives as an `Option` plus a validator rule
# (see `ParityLedger::coverage`): a newly-required field expressed in the TYPE
# would make every older `main` ledger a parse error, and a parse error emits
# no sets at all.
cp_prior_report() {
    local rev="$1" path="$2" tmp out
    git rev-parse --git-dir >/dev/null 2>&1 || return 2
    tmp=$(mktemp -d) || return 2
    if ! git show "$rev:$path" > "$tmp/ledger.yaml" 2>/dev/null; then
        rm -rf "${tmp:?}"
        return 2
    fi
    out=$(cargo run -q -p aprender-contracts-cli --bin pv -- parity-ledger "$tmp/ledger.yaml" 2>&1)
    rm -rf "${tmp:?}"
    printf '%s\n' "$out"
    grep -qE '^__ROWS__=[0-9]+$' <<<"$out" || return 2
    return 0
}

# Keys in the prior set that are gone from the current set and are not
# accounted for. One key per line; empty output means the change is legitimate.
#
#   $1 prior keys   $2 current keys   $3 live universe
#   $4 excused keys (a live downgrade pays for a DECLARED_MEASURED_ROW drop;
#      pass '' for the ROW set, where nothing pays for a drop except the entry
#      point actually having left the binary)
#   $5 repo root, for the `lib:` symbol probe
cp_unbound_drops() {
    local prior="$1" cur="$2" uni="$3" excused="${4:-}" root="${5:-$REPO_ROOT}" k
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        [ -n "$excused" ] && grep -qxF -- "$k" <<<"$excused" && continue
        cp_removal_allowed "$k" "$uni" "$root" && continue
        printf '%s\n' "$k"
    done < <(cp_set_minus "$prior" "$cur")
}

# THE NON-WIN FLOOR, WITH GIVE.
#
# THE MIRROR OF THE FABRICATION ENGINE. `NON_WINS_MIN=5` over 5 rows is
# SATURATED: every row must be a non-win, so the gate MECHANICALLY FORBIDS
# recording an honest BETTER. That is the failure this contract exists to
# disarm, arrived at from the other side -- the cheapest compliant action for a
# genuine improvement becomes NOT RECORDING IT, and a ratchet that punishes
# honesty produces dishonest ledgers. The identical lesson was already paid for
# once here, when a constant `MEASURED_MIN=4` made the honest `apr code`
# downgrade mechanically impossible.
#
# So the floor is `non_wins(main) - upgrades recorded in THIS tree`, where an
# upgrade is an in-date `downgrades:` record whose `to_verdict` is BETTER. A
# win therefore costs exactly what every other verdict change costs -- an
# owned, dated, closed-vocabulary, expiring record naming the exact move -- and
# nothing more. Deleting a loss still fails the ROW set; relabelling one still
# fails the transition check; and the count no longer stands in the way of the
# truth.
#
# Both directions have give: recording a genuine WORSE only ever RAISES
# non-wins, and nothing here is a ceiling.
#
# Fails CLOSED on junk: a non-numeric prior yields no floor (the caller treats
# that as no comparand, which is RED on its own), and a non-numeric upgrade
# count is read as ZERO, which is the STRICTER reading.
cp_nonwin_floor() {
    local prior="$1" upgrades="${2:-0}"
    case "$prior" in ''|*[!0-9]*) return 1 ;; esac
    case "$upgrades" in ''|*[!0-9]*) upgrades=0 ;; esac
    if [ "$upgrades" -ge "$prior" ]; then
        printf '0\n'
    else
        printf '%s\n' "$((prior - upgrades))"
    fi
}

# Coverage steps on `main` that this tree has DELETED, LOWERED or DEFERRED.
# One offending prior step per line; empty output means the schedule only moved
# in the allowed direction.
#
# A prior step `BY<TAB>MIN` is satisfied when some current step promises at
# least `MIN` no later than `BY`. Raising a floor or pulling a date forward is
# free; the asymmetry is the ratchet. ISO dates compare correctly as strings,
# which is the only reason a shell can do this honestly.
#
# Fails CLOSED: a prior or current step whose covered_min is not a number
# satisfies nothing.
cp_coverage_step_regressions() {
    local prior="$1" cur="$2" line pby pmin cline cby cmin okstep
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        pby=${line%%"$CP_TAB"*}
        pmin=${line##*"$CP_TAB"}
        case "$pmin" in ''|*[!0-9]*) printf '%s\n' "$line"; continue ;; esac
        okstep=0
        while IFS= read -r cline; do
            [ -n "$cline" ] || continue
            cby=${cline%%"$CP_TAB"*}
            cmin=${cline##*"$CP_TAB"}
            case "$cmin" in ''|*[!0-9]*) continue ;; esac
            [ "$cmin" -ge "$pmin" ] || continue
            [ "$cby" \> "$pby" ] && continue
            okstep=1
        done <<<"$cur"
        [ "$okstep" -eq 1 ] || printf '%s\n' "$line"
    done <<<"$prior"
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

# THERE IS NO `cp_unbound_verdict_drops` ANY MORE, AND ITS ABSENCE IS THE POINT.
#
# It existed to stop a row's verdict HISTORY being forgotten, because the history
# lived in a baseline file that ACCUMULATED every verdict a row had ever held --
# and dropping a line from it was the cheapest relabel of all, needing no record.
#
# With the comparand derived from protected `main`, that history is exactly ONE
# verdict per row: what `main` declares. "Forgetting" it is no longer a distinct
# move -- a row either declares a different verdict (5c names the transition and
# demands a record) or has been deleted (5a names the row). Keeping the check
# turned every LEGITIMATE recorded transition into a failure, which is the
# saturation defect this round exists to remove, rebuilt one field over.
#
# FOUND BY MUTATION, NOT BY REVIEW: the relabel-WITH-a-valid-record case (which
# MUST pass) came back rc=1 naming `VERDICT FORGOTTEN`. The transition check was
# already silent, as designed; the leftover rule fired anyway.


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

    printf 'case table: cp_unbound_drops (a prior key may only leave with its entry point)\n'
    local bsand buni prior cur
    bsand=$(mktemp -d) || return 2
    mkdir -p "$bsand/crates/x/src"
    printf 'pub struct StillHere;\n' > "$bsand/crates/x/src/lib.rs"
    buni=$'apr run\napr serve\napr qa\nbin:pv\nbin:apr'
    prior=$'apr run --gpu\napr serve\nlib:aprender-core::StillHere::fit'
    # (a) nothing dropped -> silent.
    [ -z "$(cp_unbound_drops "$prior" "$prior" "$buni" '' "$bsand")" ] \
        && ok 'an unchanged set is silent' || bad 'an unchanged set is silent'
    # (b) a key dropped while the entry point is STILL LIVE -> named.
    cur=$'apr run --gpu\nlib:aprender-core::StillHere::fit'
    [ "$(cp_unbound_drops "$prior" "$cur" "$buni" '' "$bsand")" = 'apr serve' ] \
        && ok 'a dropped key for a LIVE entry point is named' \
        || bad 'a dropped key for a LIVE entry point is named'
    # (c) a key whose entry point has genuinely left the binary -> silent.
    [ -z "$(cp_unbound_drops 'apr finetune' '' "$buni" '' "$bsand")" ] \
        && ok 'a key whose subcommand is GONE may be dropped' \
        || bad 'a key whose subcommand is GONE may be dropped'
    # (d) a lib: key whose symbol still exists -> named.
    [ "$(cp_unbound_drops 'lib:aprender-core::StillHere::fit' '' "$buni" '' "$bsand")" \
        = 'lib:aprender-core::StillHere::fit' ] \
        && ok 'a lib: key whose symbol still exists is named' \
        || bad 'a lib: key whose symbol still exists is named'
    # (e) EXCUSED: a measured drop paid for by a live downgrade -> silent.
    [ -z "$(cp_unbound_drops 'apr serve' '' "$buni" 'apr serve' "$bsand")" ] \
        && ok 'a measured drop with a live downgrade is excused' \
        || bad 'a measured drop with a live downgrade is excused'
    # ...and a downgrade for a DIFFERENT row excuses nothing.
    [ "$(cp_unbound_drops 'apr serve' '' "$buni" 'apr qa' "$bsand")" = 'apr serve' ] \
        && ok 'a downgrade for another row excuses nothing' \
        || bad 'a downgrade for another row excuses nothing'
    rm -rf "${bsand:?}"

    printf 'case table: THE COMPARAND (cp_upstream_ref / cp_comparand_rev / cp_parity_contracts_at)\n'
    # ROUND 5. The comparand is the ledger on PROTECTED `main`, so every
    # property at issue is a property of git refs and blobs. A string fixture
    # cannot have one; these run against real throwaway repositories.
    #
    # Round 4's fixtures proved a comparand that no longer exists (a merge base
    # plus the branch), and the lesson it recorded is that extending or moving a
    # guard's SCOPE requires re-mutating in the new scope. So the whole block is
    # re-derived here rather than adapted.
    local gsand grc gout
    gsand=$(mktemp -d) || return 2
    (
        cd "$gsand" || exit 2
        git init -q .
        git config user.email 'selftest@example.invalid'
        git config user.name 'selftest'
        mkdir -p contracts
        printf 'x\n' > other.txt
        git add -A && git commit -qm 'c1: no parity contract anywhere'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q -b work
        printf 'y\n' > work.txt
        git add -A && git commit -qm 'c2: branch work'
    ) >/dev/null 2>&1

    # (a) the upstream ref is found, and the comparand is IT -- not HEAD.
    gout=$(cd "$gsand" && cp_upstream_ref)
    [ "$gout" = 'origin/main' ] \
        && ok 'cp_upstream_ref finds origin/main' || bad "cp_upstream_ref finds origin/main (got $gout)"
    gout=$(cd "$gsand" && cp_comparand_rev)
    grc=$(cd "$gsand" && git rev-parse refs/remotes/origin/main)
    [ "$gout" = "$grc" ] \
        && ok 'the comparand is the upstream ref, never HEAD' \
        || bad 'the comparand is the upstream ref, never HEAD'
    # ...and it is REACHABLE.
    (cd "$gsand" && cp_comparand_unreachable) \
        && bad 'a checkout with an upstream ref is reachable' \
        || ok 'a checkout with an upstream ref is reachable'

    # (b) BOOTSTRAP: no contract of this KIND at the comparand.
    gout=$(cd "$gsand" && cp_parity_contracts_at "$(cd "$gsand" && cp_comparand_rev)")
    [ -z "$gout" ] \
        && ok 'no parity contract at the comparand is the BOOTSTRAP state' \
        || bad 'no parity contract at the comparand is the BOOTSTRAP state'

    # (c) once a contract of this kind is ON the comparand, the bootstrap
    #     branch is unreachable -- and it is found by KIND, so a RENAME does
    #     not re-open it. This is the property that makes the escape
    #     unrenewable, and it is the one the previous design got wrong.
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        printf 'metadata:\n  kind: competitive-parity\n' > contracts/led-v1.yaml
        git add -A && git commit -qm 'c3: land a parity ledger on main'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_parity_contracts_at "$(cd "$gsand" && cp_comparand_rev)")
    [ "$gout" = 'contracts/led-v1.yaml' ] \
        && ok 'a landed parity contract is FOUND at the comparand (no bootstrap)' \
        || bad "a landed parity contract is FOUND at the comparand (got '$gout')"
    # THE RENAME. The working tree may call it anything; the comparand still has
    # one, so the bootstrap stays shut.
    (cd "$gsand" && git mv contracts/led-v1.yaml contracts/renamed-v2.yaml) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_parity_contracts_at "$(cd "$gsand" && cp_comparand_rev)")
    [ -n "$gout" ] \
        && ok 'RENAMING the ledger does NOT manufacture a second bootstrap window' \
        || bad 'RENAMING the ledger does NOT manufacture a second bootstrap window'
    (cd "$gsand" && git checkout -q -- . && git reset -q --hard HEAD) >/dev/null 2>&1
    # A file that merely MENTIONS the kind in prose is not a contract of it.
    [ -z "$(cp_parity_contracts_at 'refs/nope')" ] \
        && ok 'an unresolvable rev yields no contracts (never a false bootstrap block)' \
        || bad 'an unresolvable rev yields no contracts'

    # (d) HEAD IS the protected branch: comparing a tree with itself is
    #     vacuous, so the comparand steps back one commit.
    (cd "$gsand" && git checkout -q main 2>/dev/null || (cd "$gsand" && git checkout -q master)) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_comparand_rev)
    grc=$(cd "$gsand" && git rev-parse 'HEAD^')
    [ "$gout" = "$grc" ] \
        && ok 'HEAD == origin/main compares against the PREVIOUS protected state' \
        || bad 'HEAD == origin/main compares against the PREVIOUS protected state'
    (cd "$gsand" && git checkout -q work) >/dev/null 2>&1

    # (e) NO upstream ref at all -- an `actions/checkout` at `fetch-depth: 1`.
    #     RE-MUTATED IN THE NEW SCOPE: round 3's rule asked whether a MERGE BASE
    #     was computable, which is not this comparand's property at all.
    local shal
    shal=$(mktemp -d) || return 2
    # `--no-single-branch --depth 3`: SHALLOW, but the upstream ref and the
    # commit behind it are both present. That is the dev box exactly -- a graft
    # boundary hundreds of commits back while the ref the ratchet needs is right
    # here -- and it must NOT red.
    git clone -q --no-single-branch --depth 3 "file://$gsand" "$shal/c" >/dev/null 2>&1
    if [ -d "$shal/c/.git" ]; then
        # A shallow clone that still HAS an upstream ref is fine: the comparand
        # is a ref and a tree, both present. Refusing it would be a gate that
        # reds for a reason unrelated to its property.
        (cd "$shal/c" && cp_comparand_unreachable) \
            && bad 'shallow WITH an upstream ref is reachable' \
            || ok 'shallow WITH an upstream ref is reachable'
        (
            cd "$shal/c" || exit 2
            git checkout -q --detach HEAD
            refs=$(git show-ref | cut -d' ' -f2)
            printf '%s\n' "$refs" | while IFS= read -r rr; do
                case "$rr" in
                    refs/heads/*|refs/remotes/*) git update-ref -d "$rr" ;;
                esac
            done
            git remote remove origin
        ) >/dev/null 2>&1
        (cd "$shal/c" && cp_comparand_unreachable) \
            && ok 'fetch-depth:1 (no upstream ref, detached HEAD) is UNREACHABLE' \
            || bad 'fetch-depth:1 (no upstream ref, detached HEAD) is UNREACHABLE'
        # ...and it must NOT read as a bootstrap, which is the whole hazard: a
        # collapsed comparand looks exactly like "there is no prior ledger",
        # which is the strongest possible pass.
        gout=$(cd "$shal/c" && cp_comparand_rev 2>/dev/null)
        gout=$(cd "$shal/c" && cp_parity_contracts_at "$gout" 2>/dev/null)
        [ -z "$gout" ] \
            && ok 'a collapsed comparand WOULD read as bootstrap, which is why 5-0 exits first' \
            || bad 'a collapsed comparand WOULD read as bootstrap'
    else
        bad 'shallow-clone fixture could not be built'
    fi
    rm -rf "${shal:?}"

    # (f) outside a git checkout at all.
    local nogit
    nogit=$(mktemp -d) || return 2
    (cd "$nogit" && cp_comparand_unreachable) \
        && ok 'outside a git checkout is UNREACHABLE' \
        || bad 'outside a git checkout is UNREACHABLE'
    gout=$(cd "$nogit" && cp_prior_report HEAD x.yaml)
    grc=$?
    [ "$grc" = 2 ] \
        && ok 'cp_prior_report outside a checkout is rc=2, never rc=0 empty' \
        || bad "cp_prior_report outside a checkout is rc=2 (got rc=$grc)"
    rm -rf "${nogit:?}"

    # (g) A PRIOR REPORT THAT DID NOT EMIT ITS BLOCK IS rc=2, NOT AN EMPTY SET.
    #     A missing comparand that reads as "no prior keys" is the coverage-floor
    #     failure exactly: a measurement that reported 0/0 for months while
    #     `|| true` kept it green.
    gout=$(cd "$gsand" && cp_prior_report "$(cd "$gsand" && cp_comparand_rev)" contracts/led-v1.yaml)
    grc=$?
    [ "$grc" = 2 ] \
        && ok 'a stub ledger that emits no __ROWS__ block is rc=2 (UNEVALUABLE)' \
        || bad "a stub ledger that emits no __ROWS__ block is rc=2 (got rc=$grc)"
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

    printf 'case table: cp_nonwin_floor == AN HONEST WIN CAN BE RECORDED\n'
    # THE MIRROR OF THE FABRICATION ENGINE, and it was live in this repo:
    # NON_WINS_MIN=5 over 5 rows is SATURATED, so the gate MECHANICALLY FORBADE
    # turning a recorded loss into a recorded win. The cheapest compliant action
    # for a genuine improvement became NOT RECORDING IT -- the same failure this
    # file exists to disarm, reached from the other side.
    [ "$(cp_nonwin_floor 5 0)" = 5 ] \
        && ok 'with no upgrade recorded the floor is what main recorded' \
        || bad 'with no upgrade recorded the floor is what main recorded'
    # THE CASE THE OLD FLOOR COULD NOT PASS. main recorded 5 non-wins over 5
    # rows; this tree turns ONE of them into a BETTER and records the
    # transition. non_wins falls 5 -> 4, and the floor falls with it.
    [ "$(cp_nonwin_floor 5 1)" = 4 ] \
        && ok 'ONE recorded upgrade lowers the floor by exactly one' \
        || bad 'ONE recorded upgrade lowers the floor by exactly one'
    if cp_meets_floor 4 "$(cp_nonwin_floor 5 1)"; then
        ok 'an honest WIN (5 non-wins -> 4, one record) MEETS the floor'
    else
        bad 'an honest WIN (5 non-wins -> 4, one record) MEETS the floor'
    fi
    # ...and the SAME win with NO record does not.
    if cp_meets_floor 4 "$(cp_nonwin_floor 5 0)"; then
        bad 'the same win with NO record is REFUSED'
    else
        ok 'the same win with NO record is REFUSED'
    fi
    # The other direction has give too: recording a genuine WORSE only ever
    # RAISES the count, and nothing here is a ceiling.
    if cp_meets_floor 6 "$(cp_nonwin_floor 5 0)"; then
        ok 'recording an ADDITIONAL loss never breaches a floor'
    else
        bad 'recording an ADDITIONAL loss never breaches a floor'
    fi
    # Two records buy two; three do not buy four.
    [ "$(cp_nonwin_floor 5 2)" = 3 ] \
        && ok 'two recorded upgrades lower the floor by two' \
        || bad 'two recorded upgrades lower the floor by two'
    [ "$(cp_nonwin_floor 2 5)" = 0 ] \
        && ok 'the floor never goes negative' || bad 'the floor never goes negative'
    # FAILS CLOSED. A non-numeric upgrade count is read as ZERO -- the STRICTER
    # reading -- and a non-numeric prior yields NO floor, which the caller
    # treats as an unusable comparand rather than as zero.
    [ "$(cp_nonwin_floor 5 'lots')" = 5 ] \
        && ok 'a non-numeric upgrade count is read as ZERO (stricter)' \
        || bad 'a non-numeric upgrade count is read as ZERO (stricter)'
    [ -z "$(cp_nonwin_floor '' 0)" ] \
        && ok 'a missing prior yields NO floor, not a floor of zero' \
        || bad 'a missing prior yields NO floor, not a floor of zero'
    [ -z "$(cp_nonwin_floor 'five' 0)" ] \
        && ok 'a non-numeric prior yields NO floor' || bad 'a non-numeric prior yields NO floor'

    printf 'case table: cp_coverage_step_regressions (the schedule is itself ratcheted)\n'
    # The coverage floor lives in the contract, and the contract is a file the
    # author edits. Deriving the SCHEDULE from `main` is what stops the floor
    # from being renewed downward every six months by whoever renews it.
    local ps_ cs_
    ps_="2026-08-21${CP_TAB}4"$'\n'"2027-02-14${CP_TAB}8"
    # (a) unchanged -> silent.
    [ -z "$(cp_coverage_step_regressions "$ps_" "$ps_")" ] \
        && ok 'an unchanged schedule is silent' || bad 'an unchanged schedule is silent'
    # (b) DELETED future step -> named.
    cs_="2026-08-21${CP_TAB}4"
    [ "$(cp_coverage_step_regressions "$ps_" "$cs_")" = "2027-02-14${CP_TAB}8" ] \
        && ok 'deleting the future step is NAMED' || bad 'deleting the future step is NAMED'
    # (c) LOWERED -> named.
    cs_="2026-08-21${CP_TAB}4"$'\n'"2027-02-14${CP_TAB}5"
    [ -n "$(cp_coverage_step_regressions "$ps_" "$cs_")" ] \
        && ok 'lowering covered_min is NAMED' || bad 'lowering covered_min is NAMED'
    # (d) POSTPONED -> named. This is the quiet one: the number is untouched and
    #     only the date moved, so a floor comparison sees nothing.
    cs_="2026-08-21${CP_TAB}4"$'\n'"2027-12-31${CP_TAB}8"
    [ "$(cp_coverage_step_regressions "$ps_" "$cs_")" = "2027-02-14${CP_TAB}8" ] \
        && ok 'DEFERRING a step to a later date is NAMED' \
        || bad 'DEFERRING a step to a later date is NAMED'
    # (e) RAISING is free.
    cs_="2026-08-21${CP_TAB}4"$'\n'"2027-02-14${CP_TAB}12"
    [ -z "$(cp_coverage_step_regressions "$ps_" "$cs_")" ] \
        && ok 'raising covered_min is free' || bad 'raising covered_min is free'
    # (f) PULLING A DATE FORWARD is free.
    cs_="2026-08-21${CP_TAB}4"$'\n'"2026-12-01${CP_TAB}8"
    [ -z "$(cp_coverage_step_regressions "$ps_" "$cs_")" ] \
        && ok 'pulling a step FORWARD is free' || bad 'pulling a step FORWARD is free'
    # (g) ONE step may satisfy several prior ones when it dominates them both.
    cs_="2026-08-01${CP_TAB}9"
    [ -z "$(cp_coverage_step_regressions "$ps_" "$cs_")" ] \
        && ok 'a step that DOMINATES the prior schedule satisfies all of it' \
        || bad 'a step that DOMINATES the prior schedule satisfies all of it'
    # (h) an EMPTY current schedule satisfies nothing.
    [ "$(cp_coverage_step_regressions "$ps_" '' | grep -c .)" = 2 ] \
        && ok 'an empty schedule loses every prior step' || bad 'an empty schedule loses every prior step'
    # (i) FAIL CLOSED on junk in either side.
    [ -n "$(cp_coverage_step_regressions "2026-08-21${CP_TAB}four" "$ps_")" ] \
        && ok 'a non-numeric PRIOR step is named, never silently satisfied' \
        || bad 'a non-numeric PRIOR step is named, never silently satisfied'
    [ -n "$(cp_coverage_step_regressions "$ps_" "2026-08-21${CP_TAB}many")" ] \
        && ok 'a non-numeric CURRENT step satisfies nothing' \
        || bad 'a non-numeric CURRENT step satisfies nothing'
    # (j) the ISO date comparison is a STRING compare and must order correctly
    #     across a year and a month boundary.
    [ -n "$(cp_coverage_step_regressions "2026-09-01${CP_TAB}8" "2026-10-01${CP_TAB}8")" ] \
        && ok 'October is LATER than September (month boundary)' \
        || bad 'October is LATER than September (month boundary)'
    [ -z "$(cp_coverage_step_regressions "2027-01-01${CP_TAB}8" "2026-12-31${CP_TAB}8")" ] \
        && ok 'December 2026 is EARLIER than January 2027 (year boundary)' \
        || bad 'December 2026 is EARLIER than January 2027 (year boundary)'

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
if [ "$MODE" != "check" ]; then
    # `--update-baseline` is GONE with the baseline file. It existed to write a
    # value the gate would later read, which is precisely the state this round
    # removed: the comparand is the ledger on protected `main` and nothing in
    # this tree can be regenerated into a lower bar.
    printf 'usage: %s [--self-test]\n' "$0" >&2
    exit 2
fi

for f in "$LEDGER" "$SCOPE"; do
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
DECLARED_MEASURED=$(cp_extract __DECLARED_MEASURED__ "$PV_OUT")
COVERED=$(cp_extract __COVERED__ "$PV_OUT")
COVERAGE_FLOOR=$(cp_extract __COVERAGE_FLOOR__ "$PV_OUT")
COVERAGE_STEPS=$(cp_extract __COVERAGE_STEPS__ "$PV_OUT")
# In-date `downgrades:` records whose `to_verdict` is BETTER. This is the GIVE
# in the non-win floor -- see cp_nonwin_floor.
UPGRADES=$(cp_extract __UPGRADES__ "$PV_OUT")

# The SETS. These, not the counts, are what makes a specific deletion visible.
LIVE_ROWS=$(cp_keys __ROW__ "$PV_OUT")
LIVE_MEASURED=$(cp_keys __MEASURED_ROW__ "$PV_OUT")
LIVE_DOWNGRADES=$(cp_keys __DOWNGRADE__ "$PV_OUT")
# `VERDICT<TAB>entry_point` for every row (the DECLARED verdict), and
# `FROM<TAB>TO<TAB>entry_point` for every IN-DATE transition record.
LIVE_VERDICTS=$(cp_keys __VERDICT_ROW__ "$PV_OUT")
LIVE_TRANSITIONS=$(cp_keys __TRANSITION__ "$PV_OUT")
# `<by><TAB><covered_min>` for every declared coverage step.
LIVE_STEPS=$(cp_keys __COVERAGE_STEP__ "$PV_OUT")

# -- 1b. the sets must agree with the emitter's own counts ------------------
# Control (c) on the key channel. A key line can only ever be ADDED to the
# stream by an injection (a newline inside a key printing extra well-formed key
# lines), so an injected line that got its length prefix right still puts the
# set out of step with the count the emitter computed from the parsed ledger.
# Cheap, and independent of both the character rule and the length prefix.
for pair in "__ROW__:$ROWS" "__MEASURED_ROW__:$MEASURED" "__VERDICT_ROW__:$ROWS" \
            "__DECLARED_MEASURED_ROW__:$DECLARED_MEASURED" "__COVERAGE_STEP__:$COVERAGE_STEPS"; do
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

# -- 5. THE COMPARAND: the ledger on PROTECTED `main` -----------------------
#
# Everything below is DERIVED from a state the author cannot rewrite inside
# their own commit. There is no baseline file: the previous design's refusals
# all read a value the same commit could edit, and no binding fixes that -- a
# sibling file cannot audit the change that edits it.

# 5-0. The comparand must be REACHABLE. A comparison git cannot answer is RED,
#      never empty. Re-derived for THIS comparand rather than inherited from
#      the merge-base one; see cp_comparand_unreachable.
if cp_comparand_unreachable; then
    printf '✗ COMPARAND UNREACHABLE: no upstream default-branch ref in this checkout.\n' >&2
    printf '    This ratchet judges the working tree against the ledger on PROTECTED\n' >&2
    printf '    `main`, because that is the one prior state a commit cannot rewrite from\n' >&2
    printf '    inside itself. With no origin/main (or origin/master, main, master) there\n' >&2
    printf '    is nothing to judge against, and every check below would pass VACUOUSLY.\n' >&2
    printf '    At actions/checkout `fetch-depth: 1` exactly this happens: one ref is\n' >&2
    printf '    fetched, HEAD is detached, no origin/main exists -- and silence would read\n' >&2
    printf '    as a BOOTSTRAP, i.e. the strongest possible pass, produced by a line in a\n' >&2
    printf '    DIFFERENT file. Check out with `fetch-depth: 0`, or fetch the default\n' >&2
    printf '    branch: git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main\n' >&2
    exit 1
fi

COMPARAND=$(cp_comparand_rev)
UPSTREAM=$(cp_upstream_ref)
PRIOR_LEDGERS=$(cp_parity_contracts_at "$COMPARAND")
PRIOR_LEDGER_COUNT=$(grep -c . <<<"$PRIOR_LEDGERS")

BOOTSTRAP=0
PRIOR_OUT=""
PRIOR_LEDGER=""
if [ "$PRIOR_LEDGER_COUNT" -eq 0 ]; then
    # BOOTSTRAP -- the ONE legitimate instance of "no prior state", and it is
    # self-limiting by CONSTRUCTION rather than by a declaration anyone writes.
    #
    # It is reachable only while NO contract of kind `competitive-parity` exists
    # anywhere under contracts/ at the protected ref. The moment one lands there
    # it is permanent (main is protected), and because the search is by KIND and
    # not by PATH, `git mv` cannot manufacture a second window -- which is how
    # the previous design's path-keyed bootstrap was renewable. There is no
    # BOOTSTRAP= line to write, no flag to pass, and nothing an author can put
    # in the tree that re-enters this branch. The operator has ruled NO
    # EXCEPTIONS (aprender#2557); a renewable escape would be `registry: true`
    # wearing its fifth hat.
    BOOTSTRAP=1
    printf '!\n' >&2
    printf '! BOOTSTRAP: no competitive-parity contract exists at %s (%s).\n' \
           "$UPSTREAM" "$COMPARAND" >&2
    printf '!   Nothing is being ratcheted against on THIS run: the row set, the\n' >&2
    printf '!   declared-measured set, every verdict, the non-win floor, the scope set\n' >&2
    printf '!   and the coverage schedule all have NO prior value. Review the emitted\n' >&2
    printf '!   sets below against the ledger BY HAND -- this run proves only that the\n' >&2
    printf '!   ledger is internally valid and in scope.\n' >&2
    printf '!   This branch is reachable exactly once per repository. It is keyed on the\n' >&2
    printf '!   ABSENCE of any contract of this KIND at the protected ref, so renaming\n' >&2
    printf '!   the ledger does not re-enter it, and once one lands on `main` -- which\n' >&2
    printf '!   requires a PR, a review and this gate -- it is unreachable forever.\n' >&2
    printf '!\n' >&2
elif [ "$PRIOR_LEDGER_COUNT" -gt 1 ]; then
    printf '✗ AMBIGUOUS COMPARAND: %s carries %s competitive-parity contracts:\n' \
           "$UPSTREAM" "$PRIOR_LEDGER_COUNT" >&2
    printf '%s\n' "$PRIOR_LEDGERS" | sed 's/^/      /' >&2
    printf '    One kind, one ledger. Two of them means the gate has to CHOOSE which\n' >&2
    printf '    prior state to enforce, and "whichever it picked" is not a ratchet --\n' >&2
    printf '    adding a second, emptier ledger would be a way to lower every bar at\n' >&2
    printf '    once. Consolidate them on the default branch first.\n' >&2
    exit 1
else
    PRIOR_LEDGER="$PRIOR_LEDGERS"
    PRIOR_OUT=$(cp_prior_report "$COMPARAND" "$PRIOR_LEDGER")
    PRIOR_RC=$?
    if [ "$PRIOR_RC" -ne 0 ]; then
        printf '✗ COMPARAND UNEVALUABLE: %s:%s could not be evaluated by `pv parity-ledger`.\n' \
               "$COMPARAND" "$PRIOR_LEDGER" >&2
        printf '    The prior ledger EXISTS, so this is not a bootstrap; it simply did not\n' >&2
        printf '    emit its machine-readable block, which means it did not PARSE. A stale\n' >&2
        printf '    row or a failed validation is fine here -- the block is emitted before\n' >&2
        printf '    either blocks, and `main` legitimately ages -- so this is specifically a\n' >&2
        printf '    schema break: something in this tree made the PROTECTED ledger\n' >&2
        printf '    unreadable. New blocks must arrive as optional fields plus a validator\n' >&2
        printf '    rule (see ParityLedger::coverage), never as a required field in the\n' >&2
        printf '    TYPE, precisely so the comparand keeps parsing.\n' >&2
        printf '    Report:\n%s\n' "$PRIOR_OUT" >&2
        exit 1
    fi
fi

# The prior SETS and FLOORS, derived rather than read.
PRIOR_ROWS=$(cp_keys __ROW__ "$PRIOR_OUT" | LC_ALL=C sort -u)
PRIOR_MEASURED=$(cp_keys __DECLARED_MEASURED_ROW__ "$PRIOR_OUT" | LC_ALL=C sort -u)
PRIOR_VERDICTS=$(cp_keys __VERDICT_ROW__ "$PRIOR_OUT" | LC_ALL=C sort -u)
PRIOR_STEPS=$(cp_keys __COVERAGE_STEP__ "$PRIOR_OUT" | LC_ALL=C sort -u)
PRIOR_NON_WINS=$(cp_extract __NON_WINS__ "$PRIOR_OUT")
PRIOR_COVERED=$(cp_extract __COVERED__ "$PRIOR_OUT")

# The SCOPE at the comparand. The scope file is the DENOMINATOR, and shrinking
# a denominator is PMAT-733 done from the other end, so it is ratcheted as a
# SET rather than as the count it used to be -- a count floor cannot tell
# "dropped `apr serve`, added `apr tui`" from "changed nothing".
PRIOR_SCOPE=""
if [ "$BOOTSTRAP" -eq 0 ]; then
    PRIOR_SCOPE=$(git show "$COMPARAND:$SCOPE" 2>/dev/null | grep -vE '^[[:space:]]*(#|$)')
    if [ -z "$PRIOR_SCOPE" ]; then
        printf '✗ COMPARAND SCOPE MISSING: %s:%s is absent or empty while a parity ledger\n' \
               "$COMPARAND" "$SCOPE" >&2
        printf '    exists at the same ref. The scope file is the DENOMINATOR of every\n' >&2
        printf '    coverage claim, so losing its prior value silently would make shrinking\n' >&2
        printf '    it free -- and renaming the file is exactly how the previous design lost\n' >&2
        printf '    its comparand. Not a bootstrap: the ledger is there, so the scope must\n' >&2
        printf '    be too.\n' >&2
        exit 1
    fi
fi

# -- 5a. every prior ROW must still EXIST -----------------------------------
# THE FIX FOR THE COUNT RATCHET, now against a state the commit cannot edit.
# Keyed by entry point, so deleting the StandardScaler row and adding a
# different one -- identical __ROWS__, __MEASURED__ and __NON_WINS__ -- names
# the missing key instead of passing.
while IFS= read -r missing; do
    [ -n "$missing" ] || continue
    printf '✗ ROW DELETED: %s is in the ledger at %s and is not in this tree.\n' \
           "$missing" "$UPSTREAM" >&2
    printf '    Totals prove nothing here: a row can be deleted and paid for with a\n' >&2
    printf '    cheaper one at constant __MEASURED__. That is exactly what d7e08043b\n' >&2
    printf '    did (PMAT-733), and it removed the only two losing rows in the\n' >&2
    printf '    history. Record the loss -- WORSE and UNMEASURED both KEEP the row.\n' >&2
    printf '    A row may leave only when the ENTRY POINT has left the live binary.\n' >&2
    fail=1
done < <(cp_unbound_drops "$PRIOR_ROWS" "$LIVE_ROWS" "$UNIVERSE" '')

# -- 5b. every prior DECLARED-measured row is measured, or downgraded ON RECORD
# Prior side DECLARED, current side EFFECTIVE: see
# `ParityLedger::declared_measured_rows`. Reading the prior side from effective
# verdicts would let the bar fall on its own as `main`'s rows aged, on a day
# nobody touched either file.
while IFS= read -r dropped; do
    [ -n "$dropped" ] || continue
    printf '✗ UNJUSTIFIED DOWNGRADE: %s is MEASURED at %s and is not here, with no record.\n' \
           "$dropped" "$UPSTREAM" >&2
    printf '    Downgrading is ALLOWED -- a floor with no give forbids the honest\n' >&2
    printf '    correction, and that produces dishonest ledgers. It is not allowed\n' >&2
    printf '    SILENTLY. Add a `downgrades:` entry to %s naming this row, with a\n' "$LEDGER" >&2
    printf '    reason from the closed vocabulary, an owner and a bounded recheck_by.\n' >&2
    printf '    (If instead the row simply EXPIRED, re-measure it: an expired row\n' >&2
    printf '    also fails `pv parity-ledger` above.)\n' >&2
    fail=1
done < <(cp_unjustified_drops "$PRIOR_MEASURED" "$LIVE_MEASURED" "$LIVE_DOWNGRADES")

# -- 5c. every DECLARED VERDICT CHANGE carries a record that names it --------
# The lever that survived rounds 1 and 2: every dimension except `rows` was
# still a count, so relabelling both WORSE rows to NOT_COMPARABLE left
# __ROWS__, __MEASURED__ and __NON_WINS__ bit-for-bit identical at rc=0.
#
# ROUND 5 CLOSES THE OTHER HALF. The `from` end used to be read from a baseline
# file the same commit could append to, so the PERMISSION was self-issued:
# measured on 4813bd41e, adding one `VERDICT_ROW=63:NOT_COMPARABLE<TAB>lib:...`
# line beside the relabel made it pass, and removing only that line made the
# identical tree fail. The `from` end is now the verdict on PROTECTED `main`.
# The only thing this tree can write is a `downgrades:` record -- dated, owned,
# closed-vocabulary, expiring, and capped by the excuse budget.
#
# Note what is NOT checked: nothing here reads the verdict's VALUE. WORSE may
# become BETTER, BETTER may become WORSE, anything may become anything. Gating
# on the value would rebuild the fabrication engine -- a rule admitting only
# wins is why the StandardScaler row was deleted in the first place.
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
    printf '    vocabulary, an owner and a bounded recheck_by.\n' >&2
    fail=1
done < <(cp_unrecorded_transitions "$PRIOR_VERDICTS" "$LIVE_VERDICTS" "$LIVE_TRANSITIONS")


# -- 5d. the NON-WIN floor, WITH GIVE ---------------------------------------
# `NON_WINS_MIN=5` over 5 rows was SATURATED: the gate mechanically FORBADE
# recording an honest BETTER, which is the fabrication failure arrived at from
# the other side. The floor is now `non_wins(main) - upgrades recorded HERE`,
# so a genuine win costs one owned, dated, expiring transition record and
# nothing more, while deleting a loss still fails 5a and relabelling one still
# fails 5c. Recording a genuine WORSE only ever raises the count; there is no
# ceiling anywhere.
if [ "$BOOTSTRAP" -eq 0 ]; then
    NON_WINS_FLOOR=$(cp_nonwin_floor "$PRIOR_NON_WINS" "$UPGRADES")
    if [ -z "$NON_WINS_FLOOR" ]; then
        printf '✗ the comparand reported no __NON_WINS__ count (%s). A missing measurement\n' \
               "${PRIOR_NON_WINS:-<none>}" >&2
        printf '    is RED, never absent.\n' >&2
        fail=1
    elif ! cp_meets_floor "$NON_WINS" "$NON_WINS_FLOOR"; then
        printf '✗ __NON_WINS__ fell: %s < %s (%s at %s, minus %s recorded upgrade(s)).\n' \
               "${NON_WINS:-<none>}" "$NON_WINS_FLOOR" "$PRIOR_NON_WINS" "$UPSTREAM" \
               "${UPGRADES:-0}" >&2
        printf '    Losses may be FIXED, not deleted. Turning a WORSE into a BETTER is a\n' >&2
        printf '    first-class move and pays for itself: record the transition in\n' >&2
        printf '    `downgrades:` with from_verdict / to_verdict and the floor moves with\n' >&2
        printf '    it. What is refused is a non-win that vanishes with nothing recorded.\n' >&2
        fail=1
    fi
fi

# -- 5e. the SCOPE set is shrink-never --------------------------------------
if [ "$BOOTSTRAP" -eq 0 ]; then
    while IFS= read -r gone; do
        [ -n "$gone" ] || continue
        printf '✗ SCOPE SHRANK: %s is in %s at %s and is not in this tree, and the entry\n' \
               "$gone" "$SCOPE" "$UPSTREAM" >&2
        printf '    point is still live. The denominator may only shrink when an entry\n' >&2
        printf '    point has actually LEFT the binary; anything else raises every ratio\n' >&2
        printf '    in this file without measuring a thing.\n' >&2
        fail=1
    done < <(cp_unbound_drops "$PRIOR_SCOPE" "$SCOPE_ENTRIES" "$UNIVERSE" '')
fi

# -- 5f. the COVERAGE RATCHET ------------------------------------------------
# The decision, its reasoning and its dissent are recorded in the contract
# (`parity.coverage`); PARITY-021..024 enforce that they exist, that the
# schedule is a schedule, that it still owes a future step, and that the step
# which has come due is MET. What is enforced HERE is the half that needs a
# comparand: the schedule may not be walked back.
if [ "$BOOTSTRAP" -eq 0 ]; then
    if ! cp_meets_floor "$COVERED" "$PRIOR_COVERED"; then
        printf '✗ COVERAGE FELL: %s distinct in-scope entry point(s) carry a row, against\n' \
               "${COVERED:-<none>}" >&2
        printf '    %s at %s. Coverage is shrink-never for the same reason the row set is.\n' \
               "${PRIOR_COVERED:-<none>}" "$UPSTREAM" >&2
        fail=1
    fi
    while IFS= read -r step; do
        [ -n "$step" ] || continue
        printf '✗ COVERAGE STEP WALKED BACK: %q is in the schedule at %s and no step in\n' \
               "$step" "$UPSTREAM" >&2
        printf '    this tree promises at least that many covered entry points by at least\n' >&2
        printf '    that date. Raising a floor or pulling a date FORWARD is free; deleting,\n' >&2
        printf '    lowering or postponing one is not. The schedule is the floor, so it\n' >&2
        printf '    needs the same shrink-never treatment the rows get -- otherwise the\n' >&2
        printf '    ratchet is renewed every six months by whoever is renewing it.\n' >&2
        fail=1
    done < <(cp_coverage_step_regressions "$PRIOR_STEPS" "$LIVE_STEPS")
fi

printf '\n'
COMPARAND_NOTE=" -> $PRIOR_LEDGER"
if [ "$BOOTSTRAP" -eq 1 ]; then
    COMPARAND_NOTE=" -- BOOTSTRAP, no prior ledger"
fi
printf 'comparand             : %s (%s)%s\n' "$UPSTREAM" "$COMPARAND" "$COMPARAND_NOTE"
printf 'entry points in scope : %s (prior set: %s)\n' \
       "$IN_SCOPE" "$(grep -c . <<<"$PRIOR_SCOPE")"
printf 'ledger rows           : %s (prior set: %s)\n' \
       "${ROWS:-<none>}" "$(grep -c . <<<"$PRIOR_ROWS")"
printf 'measured (fresh)      : %s (prior declared-measured: %s)\n' \
       "${MEASURED:-<none>}" "$(grep -c . <<<"$PRIOR_MEASURED")"
printf 'downgrades on record  : %s (of which upgrades to BETTER: %s)\n' \
       "$(grep -c . <<<"$LIVE_DOWNGRADES")" "${UPGRADES:-0}"
printf 'verdict history       : %s prior value(s) over %s row(s)\n' \
       "$(grep -c . <<<"$PRIOR_VERDICTS")" "$(cp_verdict_entries "$PRIOR_VERDICTS" | grep -c .)"
printf 'excuse budget         : %s spent of %s (in-date records vs measured rows)\n' \
       "$(cp_extract __EXCUSES__ "$PV_OUT")" "$(cp_extract __EXCUSE_BUDGET__ "$PV_OUT")"
printf 'coverage              : %s of %s in scope (floor due today %s, prior %s)\n' \
       "${COVERED:-<none>}" "$IN_SCOPE" "${COVERAGE_FLOOR:-<none>}" "${PRIOR_COVERED:-<none>}"
printf 'non-wins recorded     : %s (floor %s)\n' \
       "${NON_WINS:-<none>}" "${NON_WINS_FLOOR:-<none, bootstrap>}"

if [ "$fail" -ne 0 ]; then
    printf '\n✗ competitive-parity ratchet FAILED\n' >&2
    exit 1
fi
if [ "$BOOTSTRAP" -eq 1 ]; then
    printf '\n✓ competitive-parity ratchet OK (BOOTSTRAP: nothing was ratcheted against)\n'
    exit 0
fi
printf '\n✓ competitive-parity ratchet OK\n'
exit 0
