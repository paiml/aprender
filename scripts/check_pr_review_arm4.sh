#!/usr/bin/env bash
# check_pr_review_arm4.sh - Arm 4 of ci.yml's `pr-review-receipt` job: THIS PR's own
# receipt. PR-REVIEW-SKILL-002 v2 S8 (`receipt_presence` = 100%, "no ratchet") and S6.3
# ("a missing receipt is RED, not skipped").
#
# WHY THIS IS A SCRIPT AND NOT SIX LINES OF YAML
# ----------------------------------------------
# Arm 4 used to be an inline `run:` block whose first branch was:
#
#     if [ ! -f .github/pr-review.pub ]; then echo "NOT ARMED: ..."; exit 0; fi
#
# The key was absent, nothing owned shipping it, and so the step exited 0 on every run
# it had ever had. That is a gate that cannot fail, sitting INSIDE the job written to
# prevent gates that cannot fail - the epic's own most common defect class (spec S11),
# and it was invisible because inline YAML is reachable from no test. Arm 1, Arm 2 and
# Arm 3 all live in scripts and all have case tables; Arm 4 did not, and Arm 4 is the
# one that broke. So Arm 4 lives here, with `--self-test`, like its neighbours.
#
# WHAT IT CHECKS, IN ORDER
#
#   A1  .github/pr-review.pub EXISTS.  Absent => exit 1, never 0.
#       The key is committed (PRREV-013). Its disappearance is a defect in the tree,
#       not a reason to pass: with no public key the guard rejects every receipt
#       ("an unverifiable signature is not a verified one"), so a green Arm 4 in that
#       state would mean precisely nothing was verified.
#
#   A2  A RECEIPT FOR THIS PR EXISTS, and it reviews a commit this PR contains.
#       The receipt is selected by reading `predicate.head_sha` out of every
#       receipt under <root>/<pr>/ and keeping those that are $PR_HEAD_SHA or an
#       ANCESTOR of it; the newest such commit wins.
#
#       THE ANCESTOR RULE IS NOT A RELAXATION, IT IS THE ONLY SATISFIABLE ONE.
#       The previous armed branch looked for exactly `<root>/<pr>/$PR_HEAD_SHA`. No
#       pull request can ever satisfy that: committing the receipt CHANGES the tip, so
#       the directory named after the tip cannot contain the review of the tip. That is
#       a gate that cannot PASS, the dual of the `exit 0` above, and the two shipped in
#       the same step. A review necessarily reviews a commit that precedes the commit
#       recording it; requiring an ancestor is what that sentence means mechanically.
#
#       How far behind the tip the newest receipt sits is MEASURED and printed
#       (`commits_after_reviewed`) and does NOT gate. S8 is explicit that a threshold
#       is set from 30 samples, never invented; inventing a freshness bound here would
#       be the thing this spec spends its whole S8 forbidding.
#
#   A3  POSITIVE CONTROL, BEFORE THE VERDICT (spec S6.1's idiom, and dogfood.sh's).
#       The selected receipt is copied, the copy's signature is corrupted, and the
#       guard MUST reject the copy. If a receipt with a broken signature is accepted,
#       the ACCEPT in A4 is a count of files and not a verdict, and this step fails
#       saying so. Without A3 an Arm 4 wired to a stubbed-out guard reads green.
#
#   A4  The guard ACCEPTS the selected receipt UNDER THE REPOSITORY DEFAULT PUBLIC KEY.
#       PR_REVIEW_PUBKEY is deliberately NOT set here. The dogfood run that produced
#       this repository's first receipt passed only with PR_REVIEW_PUBKEY pointed at a
#       throwaway key; against the default the same receipt was
#       `REJECT [B1] public key .github/pr-review.pub is absent`. A default nobody can
#       satisfy is not a default.
#
# ENVIRONMENT
#   PR_NUMBER                (required) pull request number
#   PR_HEAD_SHA              (required) tip commit of the PR branch
#   PR_REVIEW_EVIDENCE_ROOT  receipt root (default: evidence/pr-review)
#   PR_REVIEW_GUARD          guard to invoke (default: scripts/check_pr_review_receipt.sh)
#
#   No value of any of them turns a check off. PR_REVIEW_GUARD pointed at a permissive
#   stub fails A3; pointed at a refuse-everything stub fails A4. Both polarities are
#   rows of --self-test.
#
# EXIT
#   0  the key is committed, a receipt for this PR exists, the guard's rejection
#      mechanism fired, and the guard accepted the receipt under the default key.
#   1  a defect in the tree: no key, no receipt, a receipt for a commit this PR does
#      not contain, a positive control that did not fire, or a rejected receipt.
#   2  the BOX cannot answer: no git, no jq, a shallow clone that cannot resolve
#      ancestry. An unmeasured gate is not a passing gate; a distinct code is so a
#      broken runner is never read as a broken tree.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

PUBKEY_REL='.github/pr-review.pub'
GUARD_REL='scripts/check_pr_review_receipt.sh'

# ---------------------------------------------------------------------------
# safe_rm_scratch <path> <required-substring> - a recursive delete, guarded.
# SEC011, and not decoration: the two ways `rm -rf -- "$x"` goes wrong are an
# EMPTY x and an x that is not ours. Both are checked before the expansion, and
# a path that fails either check is left alone rather than deleted "carefully".
# ---------------------------------------------------------------------------
safe_rm_scratch() {
    local victim=${1:-} must=${2:-}
    [ -n "$victim" ] || return 0
    [ -n "$must" ]   || return 0
    [ "$victim" != "/" ] || return 0
    case "$victim" in
      *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
      *) return 0 ;;
    esac
}

ST_ROOT=''
cleanup_self_test() { safe_rm_scratch "$ST_ROOT" 'arm4-selftest.'; }
trap cleanup_self_test EXIT

die_env() { echo "$PROG: ENV - $*" >&2; exit 2; }
fail()    { echo "$PROG: FAIL - $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# corrupt_signature_line <minisig> - flip the first character of the base64
# signature to something it is NOT, and PROVE the file changed. A mutation that
# matches nothing is the oldest way a probe reports green over an unchanged file.
# ---------------------------------------------------------------------------
corrupt_signature_line() {
    local f=$1 before after first repl
    before=$(sha256sum <"$f" | cut -d' ' -f1)
    first=$(sed -n '2s/^\(.\).*/\1/p' "$f")
    [ -n "$first" ] || return 1
    if [ "$first" = "Z" ]; then repl=Y; else repl=Z; fi
    sed -i "2s/^./$repl/" "$f"
    after=$(sha256sum <"$f" | cut -d' ' -f1)
    [ "$before" != "$after" ]
}

# ---------------------------------------------------------------------------
# receipt_head <dir> - predicate.head_sha of the receipt in <dir>, or nothing.
# ---------------------------------------------------------------------------
receipt_head() {
    [ -f "$1/receipt.intoto.jsonl" ] || return 0
    jq -r '.predicate.head_sha // empty' "$1/receipt.intoto.jsonl" 2>/dev/null
}

# ---------------------------------------------------------------------------
# arm4 <root> <pr> <head_sha> - the four checks. Every status is read from the
# command that produced it, never from the tail of a pipeline (#2336, #2360).
# ---------------------------------------------------------------------------
arm4() {
    local root=$1 pr=$2 head=$3
    local guard=${PR_REVIEW_GUARD:-$REPO_ROOT/$GUARD_REL}
    local pubkey="$REPO_ROOT/$PUBKEY_REL"

    # -- A1 ----------------------------------------------------------------
    if [ ! -f "$pubkey" ]; then
        echo "  A1  $PUBKEY_REL is ABSENT." >&2
        echo "      The guard defaults PR_REVIEW_PUBKEY to it and rejects every receipt" >&2
        echo "      while it is missing, so this step passing would mean nothing was" >&2
        echo "      verified. It is committed (PRREV-013); its absence is a defect." >&2
        return 1
    fi
    echo "  A1  $PUBKEY_REL present ($(sed -n '2p' "$pubkey" | cut -c1-16)...)"

    # -- A2 ----------------------------------------------------------------
    git -C "$REPO_ROOT" rev-parse --verify --quiet "${head}^{commit}" >/dev/null \
        || die_env "PR_HEAD_SHA $head does not resolve in $REPO_ROOT (shallow clone? fetch it before Arm 4)"

    local dir best_dir='' best_head='' best_depth='' d h depth
    if [ ! -d "$root/$pr" ]; then
        echo "  A2  no receipt directory at $root/$pr" >&2
        echo "      S6.3: a missing receipt is RED, not skipped. S8 fixes" >&2
        echo "      receipt_presence at 100% with no ratchet." >&2
        return 1
    fi
    for dir in "$root/$pr"/*/; do
        d=${dir%/}
        [ -d "$d" ] || continue
        h=$(receipt_head "$d")
        [ -n "$h" ] || continue
        git -C "$REPO_ROOT" merge-base --is-ancestor "$h" "$head" >/dev/null 2>&1 || continue
        depth=$(git -C "$REPO_ROOT" rev-list --count "$h".."$head" 2>/dev/null) || continue
        if [ -z "$best_depth" ] || [ "$depth" -lt "$best_depth" ]; then
            best_depth=$depth; best_dir=$d; best_head=$h
        fi
    done
    if [ -z "$best_dir" ]; then
        echo "  A2  $root/$pr holds no receipt whose predicate.head_sha is $head or an" >&2
        echo "      ancestor of it. A receipt naming a commit this PR does not contain" >&2
        echo "      is a review of something else." >&2
        return 1
    fi
    echo "  A2  receipt $best_dir reviews $best_head"
    echo "      commits_after_reviewed = $best_depth   (MEASURED, not gating - S8 sets"
    echo "      thresholds from 30 samples and never invents them)"

    # -- A3 ----------------------------------------------------------------
    local scratch rc
    scratch=$(mktemp -d "${TMPDIR:-/tmp}/arm4-positive-control.XXXXXX") || die_env "mktemp failed"
    cp "$best_dir"/* "$scratch/" 2>/dev/null || true
    if [ ! -f "$scratch/receipt.intoto.jsonl.minisig" ]; then
        safe_rm_scratch "$scratch" 'arm4-positive-control.'
        echo "  A3  $best_dir carries no receipt.intoto.jsonl.minisig to corrupt." >&2
        return 1
    fi
    # Flip one base64 character of the signature line. Same bytes everywhere else,
    # so the ONLY thing this can test is signature verification. The replacement is
    # chosen to DIFFER from what is there: `s/^./Z/` on a line already starting with
    # Z is a mutation that mutates nothing, which is how a probe reports a suite
    # "34 passed" against an unchanged file (scripts/mutate-guard.sh, note 1).
    corrupt_signature_line "$scratch/receipt.intoto.jsonl.minisig" \
        || { safe_rm_scratch "$scratch" 'arm4-positive-control.'
             echo "  A3  could not corrupt the signature line." >&2; return 1; }
    rc=0
    ( cd "$REPO_ROOT" && bash "$guard" "$scratch" ) >/dev/null 2>&1 || rc=$?
    safe_rm_scratch "$scratch" 'arm4-positive-control.' 
    if [ "$rc" -eq 0 ]; then
        echo "  A3  POSITIVE CONTROL DID NOT FIRE: the guard ACCEPTED a receipt whose" >&2
        echo "      signature had been corrupted. Whatever A4 reports below is a count" >&2
        echo "      of files, not a verdict (S6.1)." >&2
        return 1
    fi
    echo "  A3  positive control fired: a corrupted signature is rejected (rc=$rc)"

    # -- A4 ----------------------------------------------------------------
    # PR_REVIEW_PUBKEY deliberately unset: the point is the REPOSITORY DEFAULT.
    rc=0
    ( cd "$REPO_ROOT" && unset PR_REVIEW_PUBKEY && bash "$guard" "$best_dir" ) || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  A4  the guard REJECTED $best_dir under the repository default key (rc=$rc)." >&2
        return 1
    fi
    echo "  A4  ACCEPT under $PUBKEY_REL, with no PR_REVIEW_PUBKEY override"
    return 0
}

# ---------------------------------------------------------------------------
# --self-test: both polarities of every rule above, against a PURPOSE-BUILT
# repository, never against this one.
#
# The first draft drove the table off aprender's own history: the committed
# receipt's head_sha, and `git rev-parse HEAD` as the subject. Every row passed,
# and every row was time-bombed. `feat/prrev-012-final` merges by SQUASH, so
# f5fe147 stops existing and the honest row turns into `ENV - does not resolve`;
# and the moment this branch lands, f5fe147 becomes an ancestor of origin/main,
# so the not-an-ancestor row - which used origin/main as its subject - silently
# starts returning 0 and the check that proves Arm 4 discriminates would itself
# stop discriminating. A case table whose rows expire is worse than none: it
# reads green until the day it is wrong.
#
# So the table runs against tests/fixtures/pr-review/make-fixture-repo.sh, the
# same deterministic repo the 26 fixture rows are written against - fixed SHAs, a
# real non-degenerate fork, and its own `.github/pr-review.pub` (the committed
# TEST key), so "the repository's own default key" is exercised as a RULE rather
# than as this repository's particular key. The REAL receipt under the REAL
# default key is not tested here at all: that is the shipped invocation, two
# steps down in ci.yml, and it runs on every pull request.
# ---------------------------------------------------------------------------
self_test() {
    local st_fail=0
    ST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/arm4-selftest.XXXXXX") || die_env "mktemp failed"

    local fix="$REPO_ROOT/tests/fixtures/pr-review"
    [ -x "$fix/make-fixture-repo.sh" ] || die_env "$fix/make-fixture-repo.sh is missing"

    local repo="$ST_ROOT/repo"
    "$fix/make-fixture-repo.sh" "$repo" >/dev/null 2>&1 \
        || die_env "could not build the fixture repository"

    # Everything the guard resolves relative to itself or to the working
    # directory has to exist INSIDE the synthetic tree, or the run would reach
    # back into this repository and stop being hermetic.
    mkdir -p "$repo/.github" "$repo/scripts" "$repo/tests/fixtures/pr-review"
    cp "$fix/keys/pr-review-test.pub"            "$repo/.github/pr-review.pub"
    cp "$REPO_ROOT/scripts/check_pr_review_arm4.sh" \
       "$REPO_ROOT/scripts/check_pr_review_receipt.sh" "$repo/scripts/"
    cp -a "$REPO_ROOT/schemas"                   "$repo/schemas"
    cp -a "$fix/positive-control"                "$repo/tests/fixtures/pr-review/positive-control"

    # row-14 is the GREEN discrimination fixture: a complete receipt on a GPU PR,
    # all four consultations, findings present, signed with the test key.
    local rcpt="$fix/row-14-complete-gpu-review" head base
    head=$(receipt_head "$rcpt")
    [ -n "$head" ] || die_env "$rcpt carries no predicate.head_sha"
    git -C "$repo" rev-parse --verify --quiet "${head}^{commit}" >/dev/null \
        || die_env "row-14 reviews $head, which the fixture repo does not contain"

    mkdir -p "$repo/evidence/pr-review/999/$head"
    cp "$rcpt"/* "$repo/evidence/pr-review/999/$head/"

    # A DESCENDANT of the reviewed commit, so the ancestor rule is exercised at a
    # depth of one and not only at the degenerate depth of zero. This is the
    # shape every real PR has: the receipt reviews a commit, and the commit
    # RECORDING the receipt sits on top of it.
    local tip
    tip=$(git -C "$repo" commit-tree -p "$head" -m "R1 record the receipt" "$head^{tree}" \
          2>/dev/null) || die_env "could not create a descendant of $head"

    # A commit the reviewed one is NOT an ancestor of. C3 sits on main, on the
    # other side of the fork - a property of the fixture topology, fixed forever.
    local not_ancestor
    not_ancestor=$(git -C "$repo" rev-parse refs/remotes/origin/main) \
        || die_env "the fixture repo has no refs/remotes/origin/main"

    # A copy of the tree with the public key removed. THE ROW THIS FILE EXISTS FOR.
    local nokey="$ST_ROOT/no-key"
    cp -a "$repo" "$nokey"
    rm -f "$nokey/.github/pr-review.pub"

    # A copy whose receipt signature does not verify.
    local badsig="$ST_ROOT/badsig"
    cp -a "$repo" "$badsig"
    corrupt_signature_line "$badsig/evidence/pr-review/999/$head/receipt.intoto.jsonl.minisig" \
        || die_env "could not corrupt the fixture signature"

    printf '#!/usr/bin/env bash\nexit 0\n' > "$ST_ROOT/accept-everything.sh"
    printf '#!/usr/bin/env bash\nexit 1\n' > "$ST_ROOT/refuse-everything.sh"
    chmod +x "$ST_ROOT/accept-everything.sh" "$ST_ROOT/refuse-everything.sh"

    # row <id> <want-rc> <description> <tree> <pr> <subject-sha> [VAR=VAL ...]
    row() {
        local id=$1 want=$2 desc=$3 tree=$4 pr=$5 subject=$6; shift 6
        local got=0
        env PR_NUMBER="$pr" PR_HEAD_SHA="$subject" "$@" \
            bash "$tree/scripts/check_pr_review_arm4.sh" >/dev/null 2>&1 || got=$?
        if [ "$got" -eq "$want" ]; then
            printf 'ok    %-28s rc=%s  %s\n' "$id" "$got" "$desc"
        else
            printf 'FAIL  %-28s rc=%s (wanted %s)  %s\n' "$id" "$got" "$want" "$desc"
            st_fail=$((st_fail + 1))
        fi
    }

    echo "--- check_pr_review_arm4.sh --self-test (hermetic: $(basename "$repo")) ---"

    row receipt-reviews-the-tip   0 "receipt reviews the subject itself (depth 0)" \
        "$repo"   999 "$head"
    row receipt-reviews-ancestor  0 "receipt reviews an ANCESTOR of the subject (depth 1) — the only shape a PR can have" \
        "$repo"   999 "$tip"
    row no-receipt-for-this-pr    1 "no receipt at all (§6.3: RED, not skipped)" \
        "$repo"  1000 "$tip"
    row receipt-not-an-ancestor   1 "the only receipt reviews a commit the subject does not contain" \
        "$repo"   999 "$not_ancestor"
    row corrupt-signature         1 "receipt present, signature does not verify (A4)" \
        "$badsig" 999 "$tip"
    row guard-accepts-everything  1 "A3: a permissive guard must not read green" \
        "$repo"   999 "$tip"  PR_REVIEW_GUARD="$ST_ROOT/accept-everything.sh"
    row guard-refuses-everything  1 "A4: a refuse-everything guard must not read green either" \
        "$repo"   999 "$tip"  PR_REVIEW_GUARD="$ST_ROOT/refuse-everything.sh"
    row public-key-absent         1 "no .github/pr-review.pub — the branch that used to exit 0 forever" \
        "$nokey"  999 "$tip"

    if [ "$st_fail" -ne 0 ]; then
        echo "--- $st_fail row(s) did not produce the required verdict ---" >&2
        return 1
    fi
    echo "--- 8/8 rows, both polarities ---"
    return 0
}

# ---------------------------------------------------------------------------
for t in git jq sha256sum; do
    command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
done

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  -h|--help)   sed -n '2,70p' "$0"; exit 0 ;;
  '')          ;;
  *)           fail "unknown argument: $1" ;;
esac

: "${PR_NUMBER:?PR_NUMBER is required (the pull request number)}"
: "${PR_HEAD_SHA:?PR_HEAD_SHA is required (the tip of the PR branch)}"
ROOT=${PR_REVIEW_EVIDENCE_ROOT:-$REPO_ROOT/evidence/pr-review}

echo "$PROG: Arm 4 - this PR's own receipt (PR $PR_NUMBER, head $PR_HEAD_SHA)"
if arm4 "$ROOT" "$PR_NUMBER" "$PR_HEAD_SHA"; then
    echo "$PROG: PASS"
    exit 0
fi
echo "$PROG: FAIL - see the arm above." >&2
exit 1
