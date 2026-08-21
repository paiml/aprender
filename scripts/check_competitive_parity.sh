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
#   6. The scope SET at `main` is still covered. An entry point may leave only
#      when it has left the live enumeration from a SHA-PINNED `apr`
#      (`. scripts/apr_bin.sh`) AND a `removals:` record names it: the branch
#      builds the binary that `--help` came from, so "it is gone" is an excuse
#      the branch would otherwise be issuing to itself.
#   7. Every ledger row's entry_point is IN scope, so a row cannot be scored
#      against a universe it is not part of.
#   8. Coverage may not fall, and no coverage step may be deleted, lowered or
#      DEFERRED relative to the schedule on `main`.
#   9. The SECOND JOINT: the scope file itself must meet the `scope_min` floor
#      that has come due, so the audited surface keeps widening. An absolute
#      count, never a ratio against the live universe -- that universe is
#      author-written, so a ratio against it is payable by deleting commands.
#  10. The comparand is DISCOVERED BY PARSING every contract at the protected
#      ref, never by matching text. A file there that will not parse is RED and
#      must be repaired in this tree; "unreadable" may never read as "no ledger
#      exists", which is the bootstrap.
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

# THE DISCOVERY REPORT for a whole contracts tree: `pv parity-ledger
# --discover`, which answers "which of these are competitive-parity ledgers?"
# by PARSING them.
#
# DISCOVERED BY THE PARSED KIND, NEVER BY A REGEX OVER THE TEXT.
#
# Round 4 keyed the bootstrap window on the PATH of a sibling file, so `git mv`
# manufactured a fresh window silently. Keying it on the KIND was the right
# correction and it was implemented as a `git grep` for
#
#     ^[[:space:]]*kind:[[:space:]]*competitive-parity[[:space:]]*$
#
# which is not the kind; it is one spelling of the kind. Every difference
# between the two is a SEMANTICALLY NULL EDIT that reopens the strongest pass
# in the system:
#
#   kind: "competitive-parity"                     quoted        -> MISSES
#   kind: competitive-parity   # the ledger        commented     -> MISSES
#   metadata: {kind: competitive-parity, ...}      flow style    -> MISSES
#   ...a prose paragraph in an UNRELATED contract quoting the line
#                                                  matches       -> INVENTS a
#                                                                   second one
#
# The first three make the ledger vanish from the protected ref, which IS the
# bootstrap: nothing is ratcheted against. The fourth manufactures an AMBIGUOUS
# COMPARAND out of a sentence. A better regular expression is not the fix; not
# reading the text is the fix. `--discover` reports `metadata.kind` after
# `serde` has read the document.
#
# Prints the raw report. rc=0 when every file parsed; rc=2 when the report
# could not be produced AT ALL. A report that WAS produced but names
# unparseable files also returns 2 -- and the caller reads the sets anyway,
# exactly as `cp_prior_report` reads `__ROWS__=` regardless of `pv`'s rc,
# because "the block was emitted" and "the command succeeded" are different
# claims and only the first one gates whether the sets can be judged.
# `--manifest-path`, so this is CWD-INDEPENDENT. The case table below runs
# every git property against throwaway repositories and `cd`s into them; a
# bare `cargo run` resolves the manifest from the CURRENT directory, so the
# probe would fail inside every fixture and the table would be measuring the
# absence of a Cargo.toml rather than the property it names.
cp_discover_dir() {
    local dir="$1" out rc
    out=$(cargo run -q --manifest-path "$REPO_ROOT/Cargo.toml" \
              -p aprender-contracts-cli --bin pv -- \
              parity-ledger --discover "$dir" 2>&1)
    rc=$?
    printf '%s\n' "$out"
    grep -qE '^__SCANNED__=[0-9]+$' <<<"$out" || return 2
    return "$rc"
}

# The discovery report for the contracts tree at REVISION `$1`.
#
# `git archive` into a temp dir rather than `git show` per file: the set of
# files is itself part of what is being discovered, so listing them from the
# tree and materialising them has to be one operation over the same object.
#
# A ref with NO contracts/ directory is not an error -- it is a repository that
# has none, which is a legitimate (and permanent-once-left) bootstrap. It is
# distinguished from "the archive failed" by asking git for the tree first.
cp_discover_at() {
    local rev="$1" tmp out rc
    git rev-parse --git-dir >/dev/null 2>&1 || return 2
    # The rev must RESOLVE. Without this, an unresolvable ref falls through the
    # ls-tree below (which prints nothing for it) into the legitimate
    # "repository has no contracts/ directory" branch and returns an EMPTY SET
    # AT rc=0 -- which the caller reads as BOOTSTRAP. A ref that does not exist
    # and a repository with no contracts are not the same fact.
    git rev-parse --verify --quiet "$rev^{commit}" >/dev/null 2>&1 || return 2
    if [ -z "$(git ls-tree -d --name-only "$rev" -- contracts 2>/dev/null)" ]; then
        printf '__SCANNED__=0\n__UNPARSEABLE__=0\n__PARITY_CONTRACTS__=0\n'
        return 0
    fi
    tmp=$(mktemp -d) || return 2
    if ! git archive --format=tar "$rev" -- contracts 2>/dev/null | tar -x -C "$tmp" 2>/dev/null; then
        rm -rf "${tmp:?}"
        return 2
    fi
    out=$(cp_discover_dir "$tmp/contracts")
    rc=$?
    rm -rf "${tmp:?}"
    printf '%s\n' "$out"
    return "$rc"
}

# Paths on the discovery channel are relative to the `contracts/` ROOT on both
# sides, and are NEVER rewritten in flight.
#
# The obvious alternative -- re-prefixing the comparand's paths with
# `contracts/` so every path this script prints is repo-relative -- was written
# first and is a defect: the keys are LENGTH-PREFIXED, so `sed` inserting ten
# bytes into the value makes the declared length disagree with what follows and
# `cp_keys` DROPS every line it verifies. The set would come back empty, which
# reads as "no parity contract at the protected ref", which is the BOOTSTRAP.
# A cosmetic edit to a verified channel is a silent maximal pass; the prefix is
# added at the point of USE instead.
CP_CONTRACTS_DIR="contracts"

# The competitive-parity contracts named by a discovery report, one
# contracts-root-relative path per line.
cp_discovered_parity() { cp_keys __PARITY_CONTRACT__ "$1"; }

# The files a discovery report could not read, one `path<TAB>error` per line.
cp_discovered_unparseable() { cp_keys __UNPARSEABLE_FILE__ "$1"; }

# Do a discovery report's SETS agree with its own COUNTS?
#
# The same control the ledger channel gets (`-- 1b`), for the same reason and
# with the same asymmetry: an injection can only ADD well-formed key lines, so
# a disagreement means the text channel between the parse and the consumer was
# perturbed. Silent here, because a discovery report where the sets and counts
# disagree is judged by the caller.
cp_discovery_consistent() {
    local out="$1"
    [ "$(cp_key_count __PARITY_CONTRACT__ "$out")" = "$(cp_extract __PARITY_CONTRACTS__ "$out")" ] || return 1
    [ "$(cp_key_count __UNPARSEABLE_FILE__ "$out")" = "$(cp_extract __UNPARSEABLE__ "$out")" ] || return 1
    return 0
}

# The paths (first TAB field) of a `path<TAB>error` set.
cp_unparseable_paths() {
    local line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        printf '%s\n' "${line%%"$CP_TAB"*}"
    done <<<"$1"
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
#      pass '' for the ROW set, where nothing pays for a drop except a
#      `removals:` record NAMING the key while the entry point is also gone)
#   $5 repo root, for the `lib:` symbol probe
#   $6 the `removals:` set from THIS ledger -- see `cp_removal_allowed` for why
#      "it is gone from the binary" stopped being sufficient on its own
cp_unbound_drops() {
    local prior="$1" cur="$2" uni="$3" excused="${4:-}" root="${5:-$REPO_ROOT}" \
          removals="${6:-}" k
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        [ -n "$excused" ] && grep -qxF -- "$k" <<<"$excused" && continue
        cp_removal_allowed "$k" "$uni" "$root" "$removals" && continue
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

# A literal newline, for joining two sets into one before a set operation.
CP_NL=$'\n'

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

# Has `entry` genuinely LEFT THE WORLD as this tree builds it?
#
# An `apr <sub>` entry when the subcommand is gone from the live enumeration, a
# `bin:` entry when the bin target is gone, a `lib:` entry when its symbol no
# longer appears under crates/.
#
# THIS IS AN OBSERVATION OF A WORLD THE AUTHOR WRITES, which is why it is no
# longer sufficient on its own -- see `cp_removal_allowed`.
cp_entry_is_gone() {
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

# May `entry` be REMOVED from the ledger's row set, or from the scope file?
#
# TWO CONDITIONS, AND THE SECOND ONE IS ROUND 5's FIX.
#
#   (1) the entry point has genuinely left the world (`cp_entry_is_gone`), and
#   (2) a `removals:` RECORD in this ledger names it EXACTLY.
#
# WHY (1) ALONE WAS A HOLE. The live universe is enumerated at RUNTIME from
# `apr --help` of a binary built FROM THE BRANCH. Runtime enumeration was
# itself a deliberate fix -- a hand-written list cannot notice a new command
# and cannot notice one that DISAPPEARS -- and it is kept. The defect was never
# the enumeration; it was that the runtime being enumerated is the one under
# test. The author writes the CLI, so the author wrote the excuse:
#
#     delete `apr qa` from the clap tree
#   + delete `apr qa` from scripts/competitive_parity_scope.txt
#   + delete the `apr qa` row from the ledger
#   = rc=0, nothing recorded anywhere, one comparison fewer.
#
# That is PMAT-733 executed one level down: instead of deleting the
# measurement, delete the thing measured. Condition (1) does not merely fail to
# stop it -- it is what BLESSES it, and it blesses hardest exactly when the
# entry point being deleted carried a losing verdict.
#
# WHY NOT ANCHOR THE UNIVERSE TO THE PROTECTED REF INSTEAD. That was weighed;
# both variants were, and the cost of each is stated rather than implied:
#
#   * BUILD `apr` AT THE PROTECTED REF and treat anything present there and
#     absent here as a removal. It answers the question directly and covers
#     entry points that were never in scope. It costs a SECOND full release
#     build of the workspace on every run of this gate, and it introduces a
#     failure mode with no relation to parity: if `main`'s tree does not build
#     in this environment the gate reds, and a gate that reds for a reason
#     unrelated to its property is one that gets re-run away -- which is how a
#     REAL red gets re-run away too.
#   * ENUMERATE BOTH AND DECLARE THE SET DIFFERENCE. Same build cost; it adds
#     visibility of ADDITIONS, which the coverage schedule now bounds directly
#     (`scope_min`) without needing a second binary at all.
#
#   * WHAT IS DONE HERE: the removal is anchored to the protected ref WITHOUT a
#     second build, because the sets a removal is spent against -- the ROW set
#     and the SCOPE set -- are ALREADY read from `main`. An entry point in
#     either of those at `main` and absent from the live enumeration here is
#     precisely a removal, and it now costs an owned, dated, closed-vocabulary
#     record naming the exact key. The audited surface is therefore bounded at
#     the protected ref on both ends: it may not shrink without a record
#     (here), and it may not fail to grow (`scope_min`).
#
#     THE COST, STATED: an entry point that was live at `main` and was never in
#     `main`'s scope can still be deleted with nothing recorded. It appears in
#     no quantity this gate computes -- not the row set, not the scope set, not
#     coverage -- so today that deletion changes no bar. Putting it in scope is
#     the (ratcheted, shrink-never) way to protect it, which is the same answer
#     the coverage schedule gives.
#
# Condition (2) does NOT make the record sufficient: a record for a LIVE entry
# point excuses nothing, and PARITY-026 refuses a record parked beside a row
# that still exists. Both halves must hold.
cp_removal_allowed() {
    local entry="$1" universe="$2" root="${3:-$REPO_ROOT}" removals="${4:-}"
    grep -qxF -- "$entry" <<<"$removals" || return 1
    cp_entry_is_gone "$entry" "$universe" "$root"
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
    # (c) ROUND 5, FATAL 2. A key whose entry point has genuinely left the
    #     binary is STILL NAMED unless a `removals:` record names it. The live
    #     universe comes from a binary built FROM THE BRANCH, so "it is gone"
    #     is an excuse the branch writes for itself; the record is the price.
    [ "$(cp_unbound_drops 'apr finetune' '' "$buni" '' "$bsand" '')" = 'apr finetune' ] \
        && ok 'a GONE subcommand with NO removal record is still NAMED (fatal 2)' \
        || bad 'a GONE subcommand with NO removal record is still NAMED (fatal 2)'
    # ...and WITH the record, it may be dropped. Removal stays possible; a rule
    # that forbids retiring a command is a rule that gets deleted.
    [ -z "$(cp_unbound_drops 'apr finetune' '' "$buni" '' "$bsand" 'apr finetune')" ] \
        && ok 'a GONE subcommand WITH a removal record may be dropped' \
        || bad 'a GONE subcommand WITH a removal record may be dropped'
    # ...and the record must name THIS key, not a neighbour.
    [ "$(cp_unbound_drops 'apr finetune' '' "$buni" '' "$bsand" 'apr distill')" = 'apr finetune' ] \
        && ok 'a removal record for another entry point pays for nothing' \
        || bad 'a removal record for another entry point pays for nothing'
    # ...and it must name it EXACTLY. `apr run` must not discharge the deletion
    # of `apr run --gpu (concurrency=1 ...)`: those are different comparison
    # surfaces, and one record erasing several rows is the count currency again.
    # A line continuation INSIDE a `$( )` is what bashrs SC1078 trips on, so
    # the long calls below bind a variable first. Same call, one line.
    local qual dropped
    qual='apr run --gpu (concurrency=1 single-request decode)'
    dropped=$(cp_unbound_drops "$qual" '' 'apr serve' '' "$bsand" 'apr run')
    [ "$dropped" = "$qual" ] \
        && ok 'a removal record naming the SCOPE KEY does not discharge a qualified row' \
        || bad 'a removal record naming the SCOPE KEY does not discharge a qualified row'
    # ...and a record for a LIVE entry point buys nothing: BOTH halves hold.
    [ "$(cp_unbound_drops 'apr serve' '' "$buni" '' "$bsand" 'apr serve')" = 'apr serve' ] \
        && ok 'a removal record for a LIVE entry point buys nothing' \
        || bad 'a removal record for a LIVE entry point buys nothing'
    # (d) a lib: key whose symbol still exists -> named, record or not.
    local libkey scaler
    libkey='lib:aprender-core::StillHere::fit'
    dropped=$(cp_unbound_drops "$libkey" '' "$buni" '' "$bsand" '')
    [ "$dropped" = "$libkey" ] \
        && ok 'a lib: key whose symbol still exists is named' \
        || bad 'a lib: key whose symbol still exists is named'
    dropped=$(cp_unbound_drops "$libkey" '' "$buni" '' "$bsand" "$libkey")
    [ "$dropped" = "$libkey" ] \
        && ok 'a removal record does NOT excuse a lib: symbol that still exists' \
        || bad 'a removal record does NOT excuse a lib: symbol that still exists'
    # ...and the StandardScaler shape exactly: delete the TYPE from crates/ and
    # the row becomes droppable ONLY with a record. This is PMAT-733 executed
    # one level down - instead of deleting the measurement, delete the thing
    # measured - and it is the case the old rule blessed silently.
    scaler='lib:aprender-core::StandardScaler::fit_transform'
    dropped=$(cp_unbound_drops "$scaler" '' "$buni" '' "$bsand" '')
    [ "$dropped" = "$scaler" ] \
        && ok 'deleting the TYPE does not by itself release its row (fatal 2, lib: half)' \
        || bad 'deleting the TYPE does not by itself release its row (fatal 2, lib: half)'
    dropped=$(cp_unbound_drops "$scaler" '' "$buni" '' "$bsand" "$scaler")
    [ -z "$dropped" ] \
        && ok 'a deleted TYPE plus a removal record releases its row' \
        || bad 'a deleted TYPE plus a removal record releases its row'
    # (e) EXCUSED: a measured drop paid for by a live downgrade -> silent.
    [ -z "$(cp_unbound_drops 'apr serve' '' "$buni" 'apr serve' "$bsand")" ] \
        && ok 'a measured drop with a live downgrade is excused' \
        || bad 'a measured drop with a live downgrade is excused'
    # ...and a downgrade for a DIFFERENT row excuses nothing.
    [ "$(cp_unbound_drops 'apr serve' '' "$buni" 'apr qa' "$bsand")" = 'apr serve' ] \
        && ok 'a downgrade for another row excuses nothing' \
        || bad 'a downgrade for another row excuses nothing'
    rm -rf "${bsand:?}"

    printf 'case table: THE COMPARAND (cp_upstream_ref / cp_comparand_rev / cp_discover_at)\n'
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
    gout=$(cd "$gsand" && cp_discovered_parity "$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")")
    [ -z "$gout" ] \
        && ok 'no parity contract at the comparand is the BOOTSTRAP state' \
        || bad 'no parity contract at the comparand is the BOOTSTRAP state'

    # (c) once a contract of this kind is ON the comparand, the bootstrap
    #     branch is unreachable -- and it is found by the PARSED KIND, so
    #     neither a RENAME nor a RE-SPELLING re-opens it. This is the property
    #     that makes the escape unrenewable, and both previous designs got a
    #     different half of it wrong.
    #
    #     The fixture is a MINIMAL VALID CONTRACT, not a two-line stub: round 5
    #     requires a file that declares this kind to parse as a whole contract,
    #     so a stub would fail for the right reason at the wrong moment and the
    #     case below would be measuring the stub.
    local cp_fixture
    cp_fixture='metadata:
  version: "1.0.0"
  description: "self-test ledger"
  kind: competitive-parity
parity:
  rows:
    - entry_point: "apr qa"
'
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        printf '%s' "$cp_fixture" > contracts/led-v1.yaml
        git add -A && git commit -qm 'c3: land a parity ledger on main'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_discovered_parity "$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")")
    [ "$gout" = 'led-v1.yaml' ] \
        && ok 'a landed parity contract is FOUND at the comparand (no bootstrap)' \
        || bad "a landed parity contract is FOUND at the comparand (got '$gout')"
    # THE RENAME. The working tree may call it anything; the comparand still has
    # one, so the bootstrap stays shut.
    (cd "$gsand" && git mv contracts/led-v1.yaml contracts/renamed-v2.yaml) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_discovered_parity "$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")")
    [ -n "$gout" ] \
        && ok 'RENAMING the ledger does NOT manufacture a second bootstrap window' \
        || bad 'RENAMING the ledger does NOT manufacture a second bootstrap window'
    (cd "$gsand" && git checkout -q -- . && git reset -q --hard HEAD) >/dev/null 2>&1

    # (c2) ROUND 5, FATAL 1: THE SPELLINGS.
    #
    # The window used to be decided by `git grep -E '^[[:space:]]*kind:
    # [[:space:]]*competitive-parity[[:space:]]*$'`. Each spelling below changes
    # NO meaning and makes that regex miss, which reopens the bootstrap -- the
    # strongest pass in the system, renewable by a reflow. Discovery parses, so
    # every one of them must still be FOUND. These four are the acceptance
    # mutations for fatal 1 and they are run, not read.
    local sp spname
    for sp in 'quoted' 'commented' 'flow' 'reordered'; do
        case "$sp" in
            quoted)    spname='kind: "competitive-parity"' ;;
            commented) spname='kind: competitive-parity  # the ledger' ;;
            flow)      spname='metadata: {kind: competitive-parity, ...}' ;;
            reordered) spname='kind: first key in metadata' ;;
        esac
        (
            cd "$gsand" || exit 2
            git checkout -q main 2>/dev/null || git checkout -q master
            case "$sp" in
                quoted)
                    printf 'metadata:\n  version: "1.0.0"\n  description: "d"\n  kind: "competitive-parity"\nparity:\n  rows:\n    - entry_point: "apr qa"\n' > contracts/led-v1.yaml ;;
                commented)
                    printf 'metadata:\n  version: "1.0.0"\n  description: "d"\n  kind: competitive-parity  # the ledger\nparity:\n  rows:\n    - entry_point: "apr qa"\n' > contracts/led-v1.yaml ;;
                flow)
                    printf 'metadata: {version: "1.0.0", description: "d", kind: competitive-parity}\nparity:\n  rows:\n    - entry_point: "apr qa"\n' > contracts/led-v1.yaml ;;
                reordered)
                    printf 'metadata:\n  kind: competitive-parity\n  version: "1.0.0"\n  description: "d"\nparity:\n  rows:\n    - entry_point: "apr qa"\n' > contracts/led-v1.yaml ;;
            esac
            git add -A && git commit -qm "c-$sp"
            git update-ref refs/remotes/origin/main HEAD
            git checkout -q work
        ) >/dev/null 2>&1
        gout=$(cd "$gsand" && cp_discovered_parity "$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")")
        [ "$gout" = 'led-v1.yaml' ] \
            && ok "SPELLING: $spname is still FOUND (no renewed bootstrap)" \
            || bad "SPELLING: $spname is still FOUND (got '$gout')"
    done

    # (c3) A PROSE MENTION IS NOT A CONTRACT OF THE KIND. The other direction
    #      of the same defect: the regex matched the line wherever it appeared,
    #      so a paragraph quoting it in an UNRELATED contract manufactured an
    #      AMBIGUOUS COMPARAND out of a sentence.
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        printf 'metadata:\n  version: "1.0.0"\n  description: |\n    Historical note. The ledger declares\n    kind: competitive-parity\n    and is judged against main.\n  kind: kernel\n' > contracts/prose-v1.yaml
        git add -A && git commit -qm 'c-prose'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_discovered_parity "$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")")
    [ "$gout" = 'led-v1.yaml' ] \
        && ok 'a PROSE mention of the kind is not a contract of it (no false ambiguity)' \
        || bad "a PROSE mention of the kind is not a contract of it (got '$gout')"
    # ...and the same text WOULD have matched the regex this replaced. Asserted
    # rather than claimed: a mutation nobody can demonstrate is a comment.
    (cd "$gsand" && git show "$(cd "$gsand" && cp_comparand_rev):contracts/prose-v1.yaml" \
        | grep -qE '^[[:space:]]*kind:[[:space:]]*competitive-parity[[:space:]]*$') \
        && ok 'the prose fixture DOES match the old text regex (the mutation engaged)' \
        || bad 'the prose fixture DOES match the old text regex (the mutation engaged)'

    # (c4) A FILE THAT WILL NOT PARSE IS RED, NEVER A BOOTSTRAP. Named on the
    #      unparseable channel, and the report's rc is non-zero.
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        printf 'metadata:\n  kind: [this is not\n' > contracts/broken-v1.yaml
        git add -A && git commit -qm 'c-broken'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")
    grc=$?
    [ "$grc" -ne 0 ] \
        && ok 'an unparseable file at the comparand makes discovery rc NON-ZERO' \
        || bad 'an unparseable file at the comparand makes discovery rc NON-ZERO'
    grep -q 'broken-v1.yaml' <<<"$(cp_discovered_unparseable "$gout")" \
        && ok 'the unparseable file is NAMED on its own channel' \
        || bad 'the unparseable file is NAMED on its own channel'
    # ...and the ledger beside it is STILL found, so the report is usable and
    # the caller decides. A discovery that threw everything away on one bad
    # file would itself read as a bootstrap.
    [ "$(cp_discovered_parity "$gout")" = 'led-v1.yaml' ] \
        && ok 'the parity contract beside it is still reported' \
        || bad 'the parity contract beside it is still reported'
    # ...and the sets agree with the counts the same parse produced.
    cp_discovery_consistent "$gout" \
        && ok 'the discovery sets agree with the discovery counts' \
        || bad 'the discovery sets agree with the discovery counts'
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        git rm -q contracts/broken-v1.yaml
        git commit -qm 'c-unbroken'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1

    # (c5) AN INVENTED KIND is an error, not a miss. `kind: competitive-parityy`
    #      declares SOMETHING; reading it as "not a parity ledger" would be one
    #      keystroke away from the bootstrap.
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        printf 'metadata:\n  version: "1.0.0"\n  description: "d"\n  kind: competitive-parityy\n' > contracts/typo-v1.yaml
        git add -A && git commit -qm 'c-typo'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")
    grc=$?
    [ "$grc" -ne 0 ] \
        && ok 'an INVENTED metadata.kind is an error, not a silent miss' \
        || bad 'an INVENTED metadata.kind is an error, not a silent miss'
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        git rm -q contracts/typo-v1.yaml
        git commit -qm 'c-untypo'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1

    # (c6) A FILE WITH NO `metadata:` AT ALL parses and is simply not a
    #      contract of this kind. 47 sidecars and ticket records under
    #      contracts/ are exactly this, and reding on them would be a gate that
    #      fails for a reason unrelated to its property - which is how a gate
    #      teaches people to re-run it.
    (
        cd "$gsand" || exit 2
        git checkout -q main 2>/dev/null || git checkout -q master
        printf 'bindings:\n  - name: x\n    target: y\n' > contracts/binding.yaml
        git add -A && git commit -qm 'c-sidecar'
        git update-ref refs/remotes/origin/main HEAD
        git checkout -q work
    ) >/dev/null 2>&1
    gout=$(cd "$gsand" && cp_discover_at "$(cd "$gsand" && cp_comparand_rev)")
    grc=$?
    [ "$grc" -eq 0 ] \
        && ok 'a YAML with no metadata: block is not an error (no unrelated red)' \
        || bad 'a YAML with no metadata: block is not an error (no unrelated red)'
    [ "$(cp_discovered_parity "$gout")" = 'led-v1.yaml' ] \
        && ok 'and it is not counted as a parity contract' \
        || bad 'and it is not counted as a parity contract'

    # A rev that does not resolve: no report at all, rc=2 -- NEVER an empty set
    # that reads as a bootstrap.
    cp_discover_at 'refs/nope' >/dev/null 2>&1
    [ "$?" -eq 2 ] \
        && ok 'an unresolvable rev is rc=2, never a silent empty set' \
        || bad 'an unresolvable rev is rc=2, never a silent empty set'

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
        # which is the strongest possible pass. Round 5 adds a second layer --
        # discovery on an unresolvable rev is rc=2, not an empty set -- but 5-0
        # still exits FIRST, and this asserts the hazard is real rather than
        # asserting that a later check happens to cover it.
        gout=$(cd "$shal/c" && cp_comparand_rev 2>/dev/null)
        [ -z "$gout" ] \
            && ok 'a collapsed comparand yields NO rev, which is why 5-0 exits first' \
            || bad 'a collapsed comparand yields NO rev'
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

    printf 'case table: cp_entry_is_gone / cp_removal_allowed (the PMAT-733 countermeasure)\n'
    local sandbox
    sandbox=$(mktemp -d) || return 2
    mkdir -p "$sandbox/crates/x/src"
    printf 'pub struct StillHere;\n' > "$sandbox/crates/x/src/lib.rs"

    # The OBSERVATION half, unchanged in meaning: has it left the world?
    if cp_entry_is_gone 'apr run' "$uni" "$sandbox"; then
        bad 'a LIVE subcommand is not GONE'
    else
        ok 'a LIVE subcommand is not GONE'
    fi
    cp_entry_is_gone 'apr finetune' "$uni" "$sandbox" \
        && ok 'a subcommand absent from the enumeration IS gone' \
        || bad 'a subcommand absent from the enumeration IS gone'
    if cp_entry_is_gone 'lib:aprender-core::StillHere::fit' "$uni" "$sandbox"; then
        bad 'a lib surface whose symbol still exists is not GONE'
    else
        ok 'a lib surface whose symbol still exists is not GONE'
    fi
    cp_entry_is_gone 'lib:aprender-core::LongGone::fit' "$uni" "$sandbox" \
        && ok 'a lib surface whose symbol is absent IS gone' \
        || bad 'a lib surface whose symbol is absent IS gone'

    # ROUND 5, FATAL 2: the observation is NO LONGER SUFFICIENT. The universe
    # is enumerated from a binary built FROM THE BRANCH, so "it is gone" is an
    # excuse the deleting commit writes for itself. Removal now costs a
    # `removals:` record naming the exact key, AND the entry point must still
    # genuinely be gone. Both halves, both directions.
    if cp_removal_allowed 'apr finetune' "$uni" "$sandbox" ''; then
        bad 'GONE with NO removal record is REFUSED (fatal 2)'
    else
        ok 'GONE with NO removal record is REFUSED (fatal 2)'
    fi
    cp_removal_allowed 'apr finetune' "$uni" "$sandbox" 'apr finetune' \
        && ok 'GONE with a matching removal record is ALLOWED' \
        || bad 'GONE with a matching removal record is ALLOWED'
    if cp_removal_allowed 'apr run' "$uni" "$sandbox" 'apr run'; then
        bad 'LIVE with a removal record is still REFUSED'
    else
        ok 'LIVE with a removal record is still REFUSED'
    fi
    if cp_removal_allowed 'apr finetune' "$uni" "$sandbox" 'apr distill'; then
        bad 'a removal record for a DIFFERENT key buys nothing'
    else
        ok 'a removal record for a DIFFERENT key buys nothing'
    fi
    # A PREFIX of the key is not the key: substring matching here would let one
    # record erase every qualified row that shares a stem.
    if cp_removal_allowed 'apr finetune' "$uni" "$sandbox" 'apr fine'; then
        bad 'a PREFIX of the key is not the key'
    else
        ok 'a PREFIX of the key is not the key'
    fi
    cp_removal_allowed 'lib:aprender-core::LongGone::fit' "$uni" "$sandbox" \
                       'lib:aprender-core::LongGone::fit' \
        && ok 'a gone lib surface WITH a record is allowed' \
        || bad 'a gone lib surface WITH a record is allowed'
    if cp_removal_allowed 'lib:aprender-core::LongGone::fit' "$uni" "$sandbox" ''; then
        bad 'a gone lib surface with NO record is REFUSED'
    else
        ok 'a gone lib surface with NO record is REFUSED'
    fi
    rm -rf "${sandbox:?}"

    printf 'case table: THE SCOPE FLOOR (second joint; cp_meets_floor over scope_min)\n'
    # `covered_min` bounds rows against scope; `scope_min` bounds scope against
    # the surface. The property that matters is that the floor has NO
    # DENOMINATOR: it is an absolute integer, so shrinking the live universe --
    # fatal 2 by another route -- cannot satisfy it.
    cp_meets_floor 41 41 && ok 'the scope floor is met exactly at the value it lands on' \
        || bad 'the scope floor is met exactly at the value it lands on'
    if cp_meets_floor 40 41; then
        bad 'one entry point short of the floor is REFUSED'
    else
        ok 'one entry point short of the floor is REFUSED'
    fi
    # THE MUTATION THAT MUST NOT HELP: halve the live universe and the floor is
    # unchanged, because the floor never mentions the universe. The same
    # comparison with a RATIO floor would have flipped from fail to pass.
    local uni_small
    uni_small=$(printf '%s\n' "$uni" | head -2)
    [ "$(grep -c . <<<"$uni_small")" -lt "$(grep -c . <<<"$uni")" ] \
        && ok 'the shrink-the-universe mutation ENGAGED (fewer live entry points)' \
        || bad 'the shrink-the-universe mutation ENGAGED'
    if cp_meets_floor 40 41; then
        bad 'and it does NOT satisfy the scope floor (no denominator to shrink)'
    else
        ok 'and it does NOT satisfy the scope floor (no denominator to shrink)'
    fi
    # The schedule may not be walked back, by the same rule the coverage
    # schedule uses -- reused rather than reimplemented, so the two cannot
    # drift apart.
    [ -z "$(cp_coverage_step_regressions $'2026-08-21\t41' $'2026-08-21\t41\n2027-02-14\t55')" ] \
        && ok 'adding a HIGHER scope step is free' || bad 'adding a HIGHER scope step is free'
    [ "$(cp_coverage_step_regressions $'2026-08-21\t41' $'2026-08-21\t40')" = $'2026-08-21\t41' ] \
        && ok 'LOWERING a scope step is named' || bad 'LOWERING a scope step is named'
    [ "$(cp_coverage_step_regressions $'2026-08-21\t41' $'2027-02-14\t41')" = $'2026-08-21\t41' ] \
        && ok 'POSTPONING a scope step is named' || bad 'POSTPONING a scope step is named'
    [ "$(cp_coverage_step_regressions $'2026-08-21\t41' '')" = $'2026-08-21\t41' ] \
        && ok 'DELETING the scope schedule is named' || bad 'DELETING the scope schedule is named'

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
# `<by><TAB><covered_min>` for every declared coverage step, and
# `<by><TAB><scope_min>` for the SECOND joint's schedule.
LIVE_STEPS=$(cp_keys __COVERAGE_STEP__ "$PV_OUT")
LIVE_SCOPE_STEPS=$(cp_keys __SCOPE_STEP__ "$PV_OUT")
# The `removals:` set: entry points this ledger records as having LEFT. Spent
# by `cp_removal_allowed`, and only ever alongside the entry point actually
# being absent from the live enumeration.
LIVE_REMOVALS=$(cp_keys __REMOVAL__ "$PV_OUT")
LIVE_REMOVAL_REPLACEMENTS=$(cp_keys __REMOVAL_REPLACEMENT__ "$PV_OUT")
SCOPE_FLOOR=$(cp_extract __SCOPE_FLOOR__ "$PV_OUT")

# -- 1b. the sets must agree with the emitter's own counts ------------------
# Control (c) on the key channel. A key line can only ever be ADDED to the
# stream by an injection (a newline inside a key printing extra well-formed key
# lines), so an injected line that got its length prefix right still puts the
# set out of step with the count the emitter computed from the parsed ledger.
# Cheap, and independent of both the character rule and the length prefix.
for pair in "__ROW__:$ROWS" "__MEASURED_ROW__:$MEASURED" "__VERDICT_ROW__:$ROWS" \
            "__DECLARED_MEASURED_ROW__:$DECLARED_MEASURED" "__COVERAGE_STEP__:$COVERAGE_STEPS" \
            "__SCOPE_STEP__:$(cp_extract __SCOPE_STEPS__ "$PV_OUT")" \
            "__REMOVAL__:$(cp_extract __REMOVALS__ "$PV_OUT")"; do
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

# -- 5-1. DISCOVERY, BY PARSING, ON BOTH SIDES ------------------------------
#
# The bootstrap window -- "no competitive-parity contract exists at the
# protected ref" -- is the strongest pass this gate can give, so the question
# is answered by the PARSED `metadata.kind` and never by a regex over the file
# text. See `cp_discover_dir` for the four semantically-null edits that made
# the previous, text-matching implementation renewable.
#
# BOTH SIDES are discovered. The comparand side answers "is there a prior
# ledger, and exactly one?". The HEAD side is what lets an unreadable file at
# the comparand be handled honestly instead of by a hard stop nobody can clear
# -- see 5-1c.
HEAD_DISC=$(cp_discover_dir "$CP_CONTRACTS_DIR")
HEAD_DISC_RC=$?
if ! grep -qE '^__SCANNED__=[0-9]+$' <<<"$HEAD_DISC"; then
    printf '✗ DISCOVERY FAILED in this tree: `pv parity-ledger --discover %s` produced no\n' \
           "$CP_CONTRACTS_DIR" >&2
    printf '    report at all. Refusing to judge: "cannot read the contracts tree" must never\n' >&2
    printf '    resolve to "there is no parity ledger", which is this gate s bootstrap branch.\n' >&2
    printf '%s\n' "$HEAD_DISC" >&2
    exit 1
fi
if ! cp_discovery_consistent "$HEAD_DISC"; then
    printf '✗ DISCOVERY CHANNEL CORRUPT (this tree): the reported sets do not match the\n' >&2
    printf '    counts the same parse produced. Refusing to judge.\n' >&2
    exit 1
fi
HEAD_UNPARSEABLE=$(cp_unparseable_paths "$(cp_discovered_unparseable "$HEAD_DISC")")
HEAD_PARITY=$(cp_discovered_parity "$HEAD_DISC")

# 5-1a. THIS TREE may not ship an unreadable contract. Unconditional: a file
#       under contracts/ that no parser can read is one that no rule in this
#       repository is enforcing, and if it were the ledger the whole ratchet
#       would be inert.
if [ -n "$HEAD_UNPARSEABLE" ]; then
    printf '✗ UNREADABLE CONTRACT(S) IN THIS TREE:\n' >&2
    printf '%s\n' "$HEAD_UNPARSEABLE" | sed 's|^|      contracts/|' >&2
    printf '    A contract nobody can parse is a contract nothing enforces. Fix the YAML.\n' >&2
    fail=1
fi

# 5-1b. THIS TREE may carry exactly one contract of this kind. Checked here as
#       well as at the comparand, because "two ledgers" has to be refused on
#       the way IN -- a second, emptier one landing on `main` is how every bar
#       gets lowered at once, and the comparand-side check would then be
#       refusing something already merged.
HEAD_PARITY_COUNT=$(grep -c . <<<"$HEAD_PARITY")
if [ "$HEAD_PARITY_COUNT" -ne 1 ]; then
    printf '✗ THIS TREE carries %s competitive-parity contract(s); exactly one is required:\n' \
           "$HEAD_PARITY_COUNT" >&2
    printf '%s\n' "$HEAD_PARITY" | sed 's|^|      contracts/|' >&2
    fail=1
elif [ "$CP_CONTRACTS_DIR/$HEAD_PARITY" != "$LEDGER" ]; then
    printf '✗ THE LEDGER MOVED: the one competitive-parity contract in this tree is\n' >&2
    printf '    %s/%s, and this guard evaluates %s. The two must be the same file, or\n' \
           "$CP_CONTRACTS_DIR" "$HEAD_PARITY" "$LEDGER" >&2
    printf '    the gate would be ratcheting a document nobody is reading.\n' >&2
    fail=1
fi

PRIOR_DISC=$(cp_discover_at "$COMPARAND")
if ! grep -qE '^__SCANNED__=[0-9]+$' <<<"$PRIOR_DISC"; then
    printf '✗ DISCOVERY FAILED at %s (%s): the contracts tree at the protected ref could\n' \
           "$UPSTREAM" "$COMPARAND" >&2
    printf '    not be materialised or scanned. This is RED, and specifically not a\n' >&2
    printf '    bootstrap: the question is whether a prior ledger EXISTS there, and an\n' >&2
    printf '    unreadable tree is not evidence that one does not.\n' >&2
    printf '%s\n' "$PRIOR_DISC" >&2
    exit 1
fi
if ! cp_discovery_consistent "$PRIOR_DISC"; then
    printf '✗ DISCOVERY CHANNEL CORRUPT at %s: the reported sets do not match the counts\n' \
           "$UPSTREAM" >&2
    printf '    the same parse produced. Refusing to judge.\n' >&2
    exit 1
fi
PRIOR_UNPARSEABLE=$(cp_unparseable_paths "$(cp_discovered_unparseable "$PRIOR_DISC")")
PRIOR_LEDGERS=$(cp_discovered_parity "$PRIOR_DISC")
PRIOR_LEDGER_COUNT=$(grep -c . <<<"$PRIOR_LEDGERS")

# 5-1c. AN UNREADABLE FILE AT THE PROTECTED REF.
#
# Discovery at the comparand is INCOMPLETE for any file it could not parse:
# that file might have been a ledger, and if it was, the set below is missing
# it -- which either manufactures a bootstrap or hands the ratchet the weaker
# of two ledgers. Both are the failure this round exists to close, so it cannot
# simply be reported and passed over.
#
# It also cannot be a hard stop. The comparand is PROTECTED: a tree under test
# has no way to repair a file on `main` except by landing the repair, and a
# required check that no commit can turn green is a check that gets waived --
# and a waived check is worse than an absent one, because it is counted.
#
# So the condition is the one thing a commit CAN do about it: repair it HERE,
# in this diff, where a reviewer reads the change. Every file unreadable at the
# comparand must be readable in this tree, and none of the repaired ones may be
# a competitive-parity contract other than the ledger -- if a repair reveals a
# SECOND ledger, that is the ambiguous comparand and it is refused below.
#
# THE RESIDUAL, STATED: a malformed file at `main` that was in truth a parity
# ledger, repaired in this tree into some OTHER kind, escapes -- discovery
# cannot read the original to contradict the repair. That requires a malformed
# contract already on `main` plus a semantic edit to it inside a reviewed diff.
# It is narrower than the alternative (a permanent red that gets waived) and it
# is written down rather than left for a later round to find.
if [ -n "$PRIOR_UNPARSEABLE" ]; then
    printf '!\n' >&2
    printf '! INCOMPLETE DISCOVERY at %s (%s): %s contract file(s) there do not parse.\n' \
           "$UPSTREAM" "$COMPARAND" "$(grep -c . <<<"$PRIOR_UNPARSEABLE")" >&2
    printf '%s\n' "$PRIOR_UNPARSEABLE" | sed 's|^|!     contracts/|' >&2
    printf '!   Any one of them COULD have been a ledger, so the prior set below is only as\n' >&2
    printf '!   complete as those files are readable. This run therefore requires each of\n' >&2
    printf '!   them to be REPAIRED in this tree, where the repair is in the diff.\n' >&2
    printf '!\n' >&2
    while IFS= read -r bad_file; do
        [ -n "$bad_file" ] || continue
        if grep -qxF -- "$bad_file" <<<"$HEAD_UNPARSEABLE"; then
            printf '✗ STILL UNREADABLE: contracts/%s does not parse at %s and does not parse\n' \
                   "$bad_file" "$UPSTREAM" >&2
            printf '    here either. Discovery at the protected ref cannot rule out that this\n' >&2
            printf '    file is a competitive-parity ledger, so the prior set is unproven and\n' >&2
            printf '    nothing below can be judged against it. Fix the YAML.\n' >&2
            fail=1
        elif [ ! -f "$CP_CONTRACTS_DIR/$bad_file" ]; then
            printf '✗ UNREADABLE AND DELETED: contracts/%s does not parse at %s and has been\n' \
                   "$bad_file" "$UPSTREAM" >&2
            printf '    removed in this tree rather than repaired. Deleting a file nobody could\n' >&2
            printf '    read is exactly how a ledger would be made to disappear from the\n' >&2
            printf '    protected ref without ever being discoverable. Repair it, then argue\n' >&2
            printf '    the deletion against a version that parses.\n' >&2
            fail=1
        fi
    done <<<"$PRIOR_UNPARSEABLE"
fi

BOOTSTRAP=0
PRIOR_OUT=""
PRIOR_LEDGER=""
if [ "$PRIOR_LEDGER_COUNT" -eq 0 ]; then
    # BOOTSTRAP -- the ONE legitimate instance of "no prior state", and it is
    # self-limiting by CONSTRUCTION rather than by a declaration anyone writes.
    #
    # It is reachable only while NO contract of kind `competitive-parity` exists
    # anywhere under contracts/ at the protected ref, where the KIND is the one
    # `serde` read out of the parsed document. The moment one lands there it is
    # permanent (main is protected). Three renewal routes are closed by
    # construction rather than by a rule anyone has to remember:
    #
    #   * BY PATH -- `git mv` renaming the ledger. Closed since round 4: the
    #     window is keyed on the KIND, and the search covers the whole tree.
    #   * BY SPELLING -- quoting the kind, adding a trailing comment, reflowing
    #     the mapping into flow style. Closed HERE: round 4 keyed on the kind
    #     but MATCHED IT WITH A REGEX OVER THE TEXT, and each of those edits
    #     changes no meaning and makes the regex miss, so the strongest pass in
    #     the system was one cosmetic edit away, permanently. Discovery parses.
    #   * BY UNREADABILITY -- making the file (or any file) fail to parse, so
    #     discovery finds nothing. Closed at 5-1c: an unparseable file at the
    #     comparand is never evidence of absence; it must be repaired here.
    #
    # There is no BOOTSTRAP= line to write, no flag to pass, and nothing an
    # author can put in the tree that re-enters this branch. The operator has
    # ruled NO EXCEPTIONS (aprender#2557); a renewable escape would be
    # `registry: true` wearing its fifth hat.
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
    printf '!   ABSENCE of any contract of this KIND at the protected ref -- read from the\n' >&2
    printf '!   PARSED metadata.kind of every YAML under contracts/ there, not from the\n' >&2
    printf '!   file text -- so neither renaming the ledger nor re-spelling its kind\n' >&2
    printf '!   re-enters it, and once one lands on `main` -- which requires a PR, a\n' >&2
    printf '!   review and this gate -- it is unreachable forever.\n' >&2
    printf '!   Discovery at %s read %s file(s) with %s unreadable.\n' \
           "$UPSTREAM" "$(cp_extract __SCANNED__ "$PRIOR_DISC")" \
           "$(cp_extract __UNPARSEABLE__ "$PRIOR_DISC")" >&2
    printf '!\n' >&2
elif [ "$PRIOR_LEDGER_COUNT" -gt 1 ]; then
    printf '✗ AMBIGUOUS COMPARAND: %s carries %s competitive-parity contracts:\n' \
           "$UPSTREAM" "$PRIOR_LEDGER_COUNT" >&2
    printf '%s\n' "$PRIOR_LEDGERS" | sed 's|^|      contracts/|' >&2
    printf '    One kind, one ledger. Two of them means the gate has to CHOOSE which\n' >&2
    printf '    prior state to enforce, and "whichever it picked" is not a ratchet --\n' >&2
    printf '    adding a second, emptier ledger would be a way to lower every bar at\n' >&2
    printf '    once. Consolidate them on the default branch first.\n' >&2
    exit 1
else
    # Discovery reports paths relative to the contracts ROOT, on both sides and
    # unrewritten -- the keys are length-verified, so a `sed` re-prefix in
    # flight would make every line fail verification and be DROPPED, which
    # reads as "no parity contract at the protected ref", i.e. the bootstrap.
    # The prefix is added here, at the point of use.
    PRIOR_LEDGER="$CP_CONTRACTS_DIR/$PRIOR_LEDGERS"
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
PRIOR_SCOPE_STEPS=$(cp_keys __SCOPE_STEP__ "$PRIOR_OUT" | LC_ALL=C sort -u)
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
    printf '    A row may leave only when the ENTRY POINT has left the live binary AND\n' >&2
    printf '    a `removals:` record in %s names this exact key. "Gone from the\n' "$LEDGER" >&2
    printf '    binary" alone stopped being enough in round 5: the binary is built from\n' >&2
    printf '    THIS BRANCH, so deleting the subcommand wrote its own excuse.\n' >&2
    fail=1
done < <(cp_unbound_drops "$PRIOR_ROWS" "$LIVE_ROWS" "$UNIVERSE" '' "$REPO_ROOT" "$LIVE_REMOVALS")

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
        printf '✗ SCOPE SHRANK: %s is in %s at %s and is not in this tree.\n' \
               "$gone" "$SCOPE" "$UPSTREAM" >&2
        printf '    The denominator may shrink only when the entry point has actually LEFT\n' >&2
        printf '    the binary AND a `removals:` record in %s names this exact key.\n' "$LEDGER" >&2
        printf '    Round 5: "it is not in `apr --help` any more" is not an excuse a branch\n' >&2
        printf '    may issue to itself -- the branch builds the binary that --help came\n' >&2
        printf '    from, so deleting the subcommand was the whole of the excuse.\n' >&2
        fail=1
    done < <(cp_unbound_drops "$PRIOR_SCOPE" "$SCOPE_ENTRIES" "$UNIVERSE" '' "$REPO_ROOT" "$LIVE_REMOVALS")
fi

# -- 5e2. a REMOVAL RECORD must be TRUE, and must not be pre-authorisation ---
#
# The record is one half of the price of a deletion; this is what stops the
# half from being paid in advance or in counterfeit. Three conditions, each of
# which is a way a record could otherwise buy something it does not describe:
#
#   (a) the entry point it names must actually be GONE. A record beside a live
#       entry point is a permission banked for later -- issued in one commit
#       and spent in another, which is exactly the self-issued-permission shape
#       round 5 removed from the verdict channel. (PARITY-026 refuses the
#       narrower case of a record beside a live ROW; this is the same rule
#       against the WORLD rather than against the ledger.)
#   (b) a RENAMED / MERGED_INTO record must point at a successor that is itself
#       LIVE, or "rename" is just "delete" with better manners.
#   (c) the successor must be IN SCOPE, or the capability has been moved out of
#       the audited surface -- which shrinks what is measured while looking
#       like continuity.
while IFS= read -r rem; do
    [ -n "$rem" ] || continue
    if ! cp_entry_is_gone "$rem" "$UNIVERSE" "$REPO_ROOT"; then
        printf '✗ REMOVAL RECORDED FOR A LIVE ENTRY POINT: %s\n' "$rem" >&2
        printf '    %s records this as removed and it is still in the live enumeration.\n' "$LEDGER" >&2
        printf '    A record parked beside something that still exists is a permission\n' >&2
        printf '    banked for a deletion nobody has made yet - issued in one commit and\n' >&2
        printf '    spent in another, which is the self-issued permission this ratchet\n' >&2
        printf '    exists to refuse. Delete the record, or delete the entry point.\n' >&2
        fail=1
    fi
done <<<"$LIVE_REMOVALS"

while IFS= read -r pair2; do
    [ -n "$pair2" ] || continue
    rem=${pair2%%"$CP_TAB"*}
    successor=${pair2#*"$CP_TAB"}
    if ! cp_entry_is_live "$successor" "$UNIVERSE"; then
        printf '✗ REMOVAL POINTS AT A DEAD SUCCESSOR: %s -> %s\n' "$rem" "$successor" >&2
        printf '    A RENAMED / MERGED_INTO record claims the capability still exists under\n' >&2
        printf '    another name. %s is not in the live enumeration, so it does not.\n' "$successor" >&2
        printf '    Record the removal as RETIRED instead - which is the honest verdict and\n' >&2
        printf '    costs exactly the same.\n' >&2
        fail=1
    elif ! grep -qxF -- "$(cp_scope_key "$successor")" <<<"$SCOPE_ENTRIES"; then
        printf '✗ REMOVAL MOVES A CAPABILITY OUT OF SCOPE: %s -> %s\n' "$rem" "$successor" >&2
        printf '    The successor is live but is not in %s, so the surface being\n' "$SCOPE" >&2
        printf '    compared just got smaller while the record made it look like\n' >&2
        printf '    continuity. Add %s to the scope file.\n' "$(cp_scope_key "$successor")" >&2
        fail=1
    fi
done <<<"$LIVE_REMOVAL_REPLACEMENTS"

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
    while IFS= read -r step; do
        [ -n "$step" ] || continue
        printf '✗ SCOPE STEP WALKED BACK: %q is in the schedule at %s and no step in this\n' \
               "$step" "$UPSTREAM" >&2
        printf '    tree promises at least that many entry points IN SCOPE by at least that\n' >&2
        printf '    date. Same asymmetry as the coverage schedule: raising a floor or pulling\n' >&2
        printf '    a date forward is free, deleting or postponing one is not.\n' >&2
        fail=1
    done < <(cp_coverage_step_regressions "$PRIOR_SCOPE_STEPS" "$LIVE_SCOPE_STEPS")
fi

# -- 5g. the SECOND JOINT: the audited SURFACE must keep widening ------------
#
# `covered_min` (PARITY-024, checked inside `pv`) bounds ROWS against SCOPE.
# Nothing bounded SCOPE against the world, and the measured shape was five rows
# over 41 scope entries over 111 live subcommands: bounding one joint leaves
# the whole claim payable by simply never widening the surface, with every gate
# green while the audited FRACTION of a growing CLI falls.
#
# The floor is an ABSOLUTE COUNT on the same dated schedule, for the reason the
# contract's `rationale:` gives at the other joint and which is sharper here:
# the obvious ratio is `in_scope / live_universe`, and the LIVE UNIVERSE is
# enumerated from a binary built FROM THIS BRANCH. A ratio against it is
# payable by DELETING subcommands -- fatal 2 by another route, and the single
# thing this floor must not be satisfiable by. An integer has no denominator.
#
# Checked OUTSIDE the bootstrap guard, unlike the walk-back rules: this one
# needs no comparand. The floor is in the ledger and the count is in the scope
# file, so it binds on the first run, including the bootstrap run -- which is
# the run on which "we will widen it later" is otherwise free.
if ! cp_meets_floor "$IN_SCOPE" "$SCOPE_FLOOR"; then
    printf '✗ SCOPE FLOOR NOT MET: %s entry point(s) in %s, against a floor of %s due\n' \
           "$IN_SCOPE" "$SCOPE" "${SCOPE_FLOOR:-<none>}" >&2
    printf '    today from parity.coverage.steps[].scope_min.\n' >&2
    printf '    This is the SECOND joint. covered_min bounds rows against scope; this\n' >&2
    printf '    bounds scope against the surface, so "competitive parity" cannot stay a\n' >&2
    printf '    claim about 4.5%% of the CLI forever. Widening costs nothing but naming\n' >&2
    printf '    the entry points: a scope line needs no row, and a row may be UNMEASURED\n' >&2
    printf '    with an owner and a bound. What is NOT accepted is lowering the step -\n' >&2
    printf '    the schedule is judged against %s - or deleting subcommands to make the\n' "$UPSTREAM" >&2
    printf '    surface smaller, which is refused by the removal record above.\n' >&2
    fail=1
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
printf 'scope vs surface      : %s in scope of %s live entry point(s) (scope floor due today %s)\n' \
       "$IN_SCOPE" "$(grep -c . <<<"$UNIVERSE")" "${SCOPE_FLOOR:-<none>}"
printf 'removals on record    : %s (this run spent them against %s dropped key(s))\n' \
       "$(grep -c . <<<"$LIVE_REMOVALS")" \
       "$(cp_set_minus "$PRIOR_ROWS$CP_NL$PRIOR_SCOPE" "$LIVE_ROWS$CP_NL$SCOPE_ENTRIES" | grep -c .)"
printf 'discovery             : %s file(s) scanned here / %s at %s, %s parity contract(s) prior\n' \
       "$(cp_extract __SCANNED__ "$HEAD_DISC")" "$(cp_extract __SCANNED__ "$PRIOR_DISC")" \
       "$UPSTREAM" "$PRIOR_LEDGER_COUNT"
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
