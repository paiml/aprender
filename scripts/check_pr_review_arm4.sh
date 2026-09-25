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
#   A2  A RECEIPT FOR THIS PR EXISTS, and it binds THE DIFF BEING MERGED (#4421).
#       The subject's diff - <subject>^1..<subject> on merge_group, otherwise
#       merge-base(origin/main, subject)..subject, evidence/pr-review/<pr>/ excluded,
#       pinned flags (scripts/lib/pr_review_patch_id.sh) - is hashed with
#       `git patch-id --verbatim`, and a receipt is selected only if its SIGNED
#       predicate.diff_patch_id (stamped by pr_review_sign_receipt.sh) equals it.
#
#       WHY NOT THE HEAD SHA. It used to select by predicate.head_sha being the
#       subject or an ANCESTOR of it. The merge queue lands a PR as a one-parent
#       SQUASH, so the reviewed head is never an ancestor of the queue sha: that rule
#       verified 0 of 6 queue merges (merge-queue evaluation, 2026-09-25; a tree
#       binding 4 of 6, the patch-id 6 of 6). On a branch event the ancestor rule and
#       the diff rule agree for every honest PR - committing the receipt changes the
#       tip, not the diff - so nothing that passed before is lost there.
#
#       IT FAILS CLOSED. A receipt with no diff_patch_id (signed before #4421) binds
#       nothing. A subject with an empty diff, or a queue commit with more than one
#       parent, has no patch-id and verifies nothing. A 1-byte change to the patch,
#       whitespace included (--verbatim, not --stable), is a different id: what merges
#       is not what was reviewed, and that is RED.
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
# shellcheck source=scripts/lib/pr_review_patch_id.sh
. "$REPO_ROOT/scripts/lib/pr_review_patch_id.sh" \
    || { echo "$PROG: ENV - cannot source scripts/lib/pr_review_patch_id.sh" >&2; exit 2; }
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

    # THE BINDING IS THE DIFF, NOT THE COMMIT (#4421). The merge queue lands a PR as a
    # one-parent SQUASH, so the reviewed head_sha is never an ancestor of the queue sha:
    # the ancestor rule verified 0 of 6 queue merges, the diff patch-id 6 of 6. The
    # subject's diff is:
    #   queue  (GITHUB_EVENT_NAME=merge_group)  <subject>^1 .. <subject>
    #   branch (every other event)              merge-base(origin/main, subject) .. subject
    # with evidence/pr-review/<pr>/ excluded, and its `git patch-id --verbatim` must
    # equal a signed receipt's predicate.diff_patch_id. FAILS CLOSED: a receipt with
    # no diff_patch_id binds nothing, and a subject whose patch-id cannot be computed
    # (empty diff, a queue commit with more than one parent) verifies nothing.
    local kind=${PR_REVIEW_SUBJECT_KIND:-} base np pid rc=0
    if [ -z "$kind" ]; then
        if [ "${GITHUB_EVENT_NAME:-}" = merge_group ]; then kind=queue; else kind=branch; fi
    fi
    case "$kind" in
      queue)
        np=$(git -C "$REPO_ROOT" rev-list --parents -n 1 "$head" | wc -w)
        if [ "$np" -ne 2 ]; then
            echo "  A2  queue subject $head has $((np - 1)) parents; the queue SQUASHES to one." >&2
            echo "      Its diff against a single base is undefined, so nothing can bind to it." >&2
            return 1
        fi
        base=$(git -C "$REPO_ROOT" rev-parse "$head^1") ;;
      branch)
        base=$(git -C "$REPO_ROOT" merge-base refs/remotes/origin/main "$head" 2>/dev/null) \
            || die_env "no merge-base between refs/remotes/origin/main and $head (fetch origin/main)" ;;
      *) die_env "PR_REVIEW_SUBJECT_KIND='$kind' is not queue|branch" ;;
    esac
    pid=$(prpid_compute "$REPO_ROOT" "$base" "$head" "$pr") || rc=$?
    if [ "$rc" -eq 2 ]; then
        die_env "cannot compute a patch-id on this box (no git patch-id --verbatim, no python3)"
    elif [ "$rc" -ne 0 ]; then
        echo "  A2  the subject diff $base..$head has no patch-id (empty, or not a diff)." >&2
        echo "      Nothing was changed, so no receipt can be for it. FAIL CLOSED (#4421)." >&2
        return 1
    fi
    echo "  A2  subject ($kind) diff $base..$head, patch-id $pid"

    local dir best_dir='' best_head='' d h rp legacy=0 other=0 legacy_dir='' legacy_dirs=()
    if [ ! -d "$root/$pr" ]; then
        echo "  A2  no receipt directory at $root/$pr" >&2
        echo "      S6.3: a missing receipt is RED, not skipped. S8 fixes" >&2
        echo "      receipt_presence at 100% with no ratchet." >&2
        return 1
    fi
    for dir in "$root/$pr"/*/; do
        d=${dir%/}
        [ -f "$d/receipt.intoto.jsonl" ] || continue
        rp=$(jq -r '.predicate.diff_patch_id // empty' "$d/receipt.intoto.jsonl" 2>/dev/null)
        if [ -z "$rp" ]; then legacy=$((legacy + 1)); legacy_dirs+=("$d"); continue; fi
        if [ "$rp" != "$pid" ]; then other=$((other + 1)); continue; fi
        h=$(receipt_head "$d")
        # Several receipts may bind the same diff (a re-review); prefer the one whose
        # head the subject contains, which is every branch-event case.
        if [ -z "$best_dir" ] || { [ -n "$h" ] && git -C "$REPO_ROOT" merge-base --is-ancestor "$h" "$head" >/dev/null 2>&1; }; then
            best_dir=$d; best_head=$h
        fi
    done
    # A legacy receipt is accepted ONLY where the pre-#4421 rule would have accepted it,
    # or on the queue commit that rule could never bind (the defect #4421 fixes):
    #   branch  its signed head_sha must be an ANCESTOR of the subject - the old rule,
    #           unweakened. A legacy receipt that fails it is RED, exempt or not.
    #   queue   no commit binding exists for a squash; the signature (A3/A4), the PR
    #           ceiling and the 24h expiry are the whole of the check, by ruling.
    #   Every legacy receipt is tried, not the last one listed: a PR re-reviewed before
    #   #4421 holds several, and an older head that no longer binds must not hide one that does.
    if [ -z "$best_dir" ] && [ "${#legacy_dirs[@]}" -gt 0 ] && legacy_exempt "$pr"; then
        for d in "${legacy_dirs[@]}"; do
            h=$(receipt_head "$d")
            if [ -n "$h" ] && git -C "$REPO_ROOT" merge-base --is-ancestor "$h" "$head" >/dev/null 2>&1; then
                legacy_dir=$d; break
            fi
        done
        [ -z "$legacy_dir" ] && [ "$kind" = queue ] && legacy_dir=${legacy_dirs[0]}
        h=$(receipt_head "${legacy_dir:-${legacy_dirs[0]}}")
        if [ -n "$legacy_dir" ]; then
            best_dir=$legacy_dir; best_head=$h
            echo "  A2  LEGACY EXEMPT - $legacy_dir has no diff_patch_id (signed before #4421)."
            echo "      Accepted because PR $pr < $PR_REVIEW_LEGACY_BELOW (open when FLOW-04 merged), now"
            echo "      $(legacy_now) < $PR_REVIEW_LEGACY_UNTIL (24h after it), and ($kind) its head binds as"
            echo "      before #4421. Its signature is still checked (A3/A4)."
        else
            echo "  A2  none of the ${#legacy_dirs[@]} legacy receipt(s) reviewed an ancestor of $head;" >&2
            echo "      the exemption waives the diff binding, never the pre-#4421 ancestor rule." >&2
        fi
    fi
    if [ -z "$best_dir" ]; then
        echo "  A2  $root/$pr holds no receipt whose predicate.diff_patch_id is $pid." >&2
        echo "      $other receipt(s) bind a DIFFERENT diff (the change moved after review," >&2
        echo "      or the queue resolved a conflict); $legacy carry no diff_patch_id at all" >&2
        echo "      (signed before #4421 - re-sign to bind). What merges is not what was reviewed." >&2
        return 1
    fi
    echo "  A2  receipt $best_dir binds this diff (reviewed head $best_head)"
    if git -C "$REPO_ROOT" merge-base --is-ancestor "$best_head" "$head" >/dev/null 2>&1; then
        echo "      commits_after_reviewed = $(git -C "$REPO_ROOT" rev-list --count "$best_head".."$head" 2>/dev/null)   (MEASURED, not gating)"
    else
        echo "      reviewed head is not an ancestor of the subject (a queue squash or a rebase): the diff binds, not the commit"
    fi

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
    mkdir -p "$repo/.github" "$repo/scripts/lib" "$repo/tests/fixtures/pr-review"
    cp "$fix/keys/pr-review-test.pub"            "$repo/.github/pr-review.pub"
    cp "$REPO_ROOT/scripts/check_pr_review_arm4.sh" \
       "$REPO_ROOT/scripts/check_pr_review_receipt.sh" "$repo/scripts/"
    cp "$REPO_ROOT/scripts/lib/pr_review_patch_id.sh" \
       "$REPO_ROOT/scripts/lib/git_patch_id.py" "$repo/scripts/lib/"
    cp -a "$REPO_ROOT/schemas"                   "$repo/schemas"
    cp -a "$fix/positive-control"                "$repo/tests/fixtures/pr-review/positive-control"

    # row-14 is the GREEN discrimination fixture: a complete receipt on a GPU PR,
    # all five consultations, findings present, signed with the test key.
    local rcpt="$fix/row-14-complete-gpu-review" head base
    head=$(receipt_head "$rcpt")
    [ -n "$head" ] || die_env "$rcpt carries no predicate.head_sha"
    git -C "$repo" rev-parse --verify --quiet "${head}^{commit}" >/dev/null \
        || die_env "row-14 reviews $head, which the fixture repo does not contain"

    # #4421: the receipt must carry the diff patch-id of base_sha..head_sha, signed.
    # row-14 predates it, so the copy is stamped here and RE-SIGNED with the committed
    # TEST-ONLY key (keys/README.md) - the same key that signed row-14 originally.
    # The unstamped original is kept for the legacy row.
    command -v minisign >/dev/null 2>&1 || die_env "minisign is not on PATH"
    local rbase pid ev="$repo/evidence/pr-review/999/$head"
    rbase=$(jq -r '.predicate.base_sha // empty' "$rcpt/receipt.intoto.jsonl")
    pid=$(prpid_compute "$repo" "$rbase" "$head" 999) \
        || die_env "cannot compute the fixture's patch-id for $rbase..$head"
    mkdir -p "$ev"
    cp "$rcpt"/* "$ev/"
    jq -c --arg p "$pid" --arg a "$PRPID_ALGO" \
        '.predicate.diff_patch_id = $p | .predicate.diff_patch_id_algo = $a' \
        "$rcpt/receipt.intoto.jsonl" > "$ev/receipt.intoto.jsonl" || die_env "could not stamp row-14"
    rm -f -- "${ev:?}/receipt.intoto.jsonl.minisig"
    minisign -S -s "$fix/keys/pr-review-test-TEST-ONLY.key" -m "$ev/receipt.intoto.jsonl" \
        -t "arm4 self-test, #4421 stamped" </dev/null >/dev/null 2>&1 \
        || die_env "could not re-sign the stamped fixture with the TEST-ONLY key"

    # A DESCENDANT of the reviewed commit that COMMITS the receipt, as every real PR
    # does - so the evidence/pr-review/<pr>/ exclusion is exercised, not assumed.
    local tip idx="$ST_ROOT/idx" tree
    GIT_INDEX_FILE=$idx git -C "$repo" read-tree "$head" || die_env "read-tree"
    GIT_INDEX_FILE=$idx git -C "$repo" add -f "evidence/pr-review/999" || die_env "add evidence"
    tree=$(GIT_INDEX_FILE=$idx git -C "$repo" write-tree) || die_env "write-tree"
    tip=$(git -C "$repo" commit-tree -p "$head" -m "R1 record the receipt" "$tree" \
          2>/dev/null) || die_env "could not create a descendant of $head"

    # THE MERGE-QUEUE SHAPE: a SQUASH of the PR onto a main that MOVED, one parent.
    # The reviewed head is not its ancestor - the ancestor rule's 0-of-6 case.
    local main squash
    main=$(git -C "$repo" rev-parse refs/remotes/origin/main) || die_env "no origin/main"
    [ "$main" != "$rbase" ] \
        || die_env "fixture origin/main has not moved past the reviewed base; the queue row would be degenerate"
    rm -f -- "${idx:?}"
    GIT_INDEX_FILE=$idx git -C "$repo" read-tree "$main" || die_env "read-tree main"
    git -C "$repo" diff --full-index --binary "$rbase" "$tip" \
        | GIT_INDEX_FILE=$idx git -C "$repo" apply --cached \
        || die_env "the fixture PR does not apply cleanly onto origin/main"
    tree=$(GIT_INDEX_FILE=$idx git -C "$repo" write-tree) || die_env "write-tree"
    squash=$(git -C "$repo" commit-tree -p "$main" -m "PR 999 (squash)" "$tree") || die_env "commit-tree"

    # The same squash with the PR's first changed file altered by exactly ONE BYTE,
    # and with ONE trailing space - the negative arm, and the reason for --verbatim.
    local f1 squash_1b squash_ws
    f1=$(git -C "$repo" diff --name-only "$rbase" "$head" | head -1)
    [ -n "$f1" ] || die_env "the fixture PR changes no file"
    mut_squash() { # <sed-expr> -> sha of a squash whose $f1 is edited by <sed-expr>
        local b
        b=$(git -C "$repo" show "$squash:$f1" | sed "$1" | git -C "$repo" hash-object -w --stdin) || return 1
        [ "$b" != "$(git -C "$repo" rev-parse "$squash:$f1")" ] || return 1
        rm -f -- "${idx:?}"
        GIT_INDEX_FILE=$idx git -C "$repo" read-tree "$squash" || return 1
        GIT_INDEX_FILE=$idx git -C "$repo" update-index --cacheinfo "100644,$b,$f1" || return 1
        git -C "$repo" commit-tree -p "$main" -m "PR 999 (mutated squash)" \
            "$(GIT_INDEX_FILE=$idx git -C "$repo" write-tree)"
    }
    squash_1b=$(mut_squash '1s/^./\x01/') || die_env "could not build the 1-byte squash"
    squash_ws=$(mut_squash '1s/$/ /')     || die_env "could not build the whitespace squash"
    [ "$(git -C "$repo" diff "$squash" "$squash_1b" | grep -c '^[-+][^-+]')" -eq 2 ] \
        || die_env "the 1-byte squash does not differ by exactly one line"
    local merge2
    merge2=$(git -C "$repo" commit-tree -p "$main" -p "$tip" -m "two-parent merge" "$tree") \
        || die_env "commit-tree"

    # A commit the reviewed one is NOT an ancestor of. C3 sits on main, on the
    # other side of the fork - a property of the fixture topology, fixed forever.
    local not_ancestor
    not_ancestor=$(git -C "$repo" rev-parse refs/remotes/origin/main) \
        || die_env "the fixture repo has no refs/remotes/origin/main"

    # A copy of the tree with the public key removed. THE ROW THIS FILE EXISTS FOR.
    local nokey="$ST_ROOT/no-key"
    cp -a "$repo" "$nokey"
    rm -f "$nokey/.github/pr-review.pub"

    # A copy whose receipt is the UNSTAMPED row-14: signed, valid, and pre-#4421.
    local legacy="$ST_ROOT/legacy"
    cp -a "$repo" "$legacy"
    cp "$rcpt"/* "$legacy/evidence/pr-review/999/$head/"

    # The legacy copy with its signature corrupted: the exemption waives the diff
    # binding, never the signature.
    local legacy_bad="$ST_ROOT/legacy-badsig"
    cp -a "$legacy" "$legacy_bad"
    corrupt_signature_line "$legacy_bad/evidence/pr-review/999/$head/receipt.intoto.jsonl.minisig" \
        || die_env "could not corrupt the legacy fixture signature"

    # The legacy copy re-reviewed twice before #4421: a SECOND legacy receipt, listed
    # after the good one, whose head is NOT an ancestor. The exemption must still bind
    # the good one - not only the last receipt the glob returns.
    local legacy_two="$ST_ROOT/legacy-two" ev2
    cp -a "$legacy" "$legacy_two"
    ev2="$legacy_two/evidence/pr-review/999/zz-$not_ancestor"
    # bashrs SEC010: $ev2 is under the self-test's own mktemp -d root; the rest is a sha.
    # bashrs disable-next-line=SEC010
    mkdir -p "$ev2"
    jq -c --arg h "$not_ancestor" '.predicate.head_sha = $h' "$rcpt/receipt.intoto.jsonl" \
        > "$ev2/receipt.intoto.jsonl" || die_env "could not write the second legacy receipt"
    minisign -S -s "$fix/keys/pr-review-test-TEST-ONLY.key" -m "$ev2/receipt.intoto.jsonl" \
        -t "arm4 self-test, second legacy" </dev/null >/dev/null 2>&1 \
        || die_env "could not sign the second legacy receipt"

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
        # PR_REVIEW_CUTOFF=0 so every row below exercises the ENFORCING path; without
        # it the fixture PR numbers grandfather out and six rows silently pass on the
        # cutoff branch instead of the branch they name. A per-row override still wins:
        # a later `env VAR=VAL` beats an earlier one.
        env PR_NUMBER="$pr" PR_HEAD_SHA="$subject" PR_REVIEW_CUTOFF=0 PR_REVIEW_LEGACY_BELOW=0 "$@" \
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
    row receipt-not-an-ancestor   1 "subject is origin/main itself: an EMPTY diff binds nothing" \
        "$repo"   999 "$not_ancestor"
    # #4421 - the merge queue, and the binding failing closed.
    row queue-squash-binds        0 "merge_group: one-parent squash onto a MOVED main, head not an ancestor (0/6 before #4421)" \
        "$repo"   999 "$squash"    GITHUB_EVENT_NAME=merge_group
    row queue-one-byte-changed    1 "merge_group: the squash differs from the reviewed diff by ONE BYTE" \
        "$repo"   999 "$squash_1b" GITHUB_EVENT_NAME=merge_group
    row queue-whitespace-changed  1 "merge_group: one trailing space - --stable would have passed this" \
        "$repo"   999 "$squash_ws" GITHUB_EVENT_NAME=merge_group
    row queue-two-parents         1 "merge_group: a two-parent queue commit has no single diff (fail closed)" \
        "$repo"   999 "$merge2"    GITHUB_EVENT_NAME=merge_group
    row branch-squash-as-branch   0 "the same squash judged as a branch event (merge-base = its parent)" \
        "$repo"   999 "$squash"
    row legacy-receipt-no-id      1 "a valid signed receipt with no diff_patch_id binds nothing" \
        "$legacy" 999 "$tip"
    # THE LEGACY EXEMPTION, each bound in both polarities over the SAME legacy tree
    # (row() pins PR_REVIEW_LEGACY_BELOW=0 so no other row can reach it by accident).
    row legacy-exempt-in-window   0 "legacy receipt: PR open at merge (999 < 1000) and inside the 24h" \
        "$legacy" 999 "$tip"  PR_REVIEW_LEGACY_BELOW=1000 PR_REVIEW_LEGACY_UNTIL=2000 PR_REVIEW_NOW=1999
    row legacy-expired-at-expiry  1 "legacy receipt: the SAME PR at the expiry second is RED" \
        "$legacy" 999 "$tip"  PR_REVIEW_LEGACY_BELOW=1000 PR_REVIEW_LEGACY_UNTIL=2000 PR_REVIEW_NOW=2000
    row legacy-pr-opened-after    1 "legacy receipt: a PR opened after the merge (999 >= 999) is RED in the window" \
        "$legacy" 999 "$tip"  PR_REVIEW_LEGACY_BELOW=999 PR_REVIEW_LEGACY_UNTIL=2000 PR_REVIEW_NOW=1999
    row legacy-exempt-not-ancestor 1 "legacy receipt in the window, branch event, head NOT an ancestor: the old rule still binds" \
        "$legacy" 999 "$squash"  PR_REVIEW_LEGACY_BELOW=1000 PR_REVIEW_LEGACY_UNTIL=2000 PR_REVIEW_NOW=1999
    row legacy-two-last-stale     0 "two legacy receipts, the LAST listed not an ancestor: the other still binds" \
        "$legacy_two" 999 "$tip"  PR_REVIEW_LEGACY_BELOW=1000 PR_REVIEW_LEGACY_UNTIL=2000 PR_REVIEW_NOW=1999
    row legacy-exempt-queue       0 "legacy receipt in the window on the queue squash (the commit #4421 exists for)" \
        "$legacy" 999 "$squash"  PR_REVIEW_LEGACY_BELOW=1000 PR_REVIEW_LEGACY_UNTIL=2000 PR_REVIEW_NOW=1999 GITHUB_EVENT_NAME=merge_group
    row legacy-exempt-bad-sig     1 "legacy receipt in the window with a corrupted signature is RED (A4 still runs)" \
        "$legacy_bad" 999 "$tip"  PR_REVIEW_LEGACY_BELOW=1000 PR_REVIEW_LEGACY_UNTIL=2000 PR_REVIEW_NOW=1999
    row corrupt-signature         1 "receipt present, signature does not verify (A4)" \
        "$badsig" 999 "$tip"
    row guard-accepts-everything  1 "A3: a permissive guard must not read green" \
        "$repo"   999 "$tip"  PR_REVIEW_GUARD="$ST_ROOT/accept-everything.sh"
    row guard-refuses-everything  1 "A4: a refuse-everything guard must not read green either" \
        "$repo"   999 "$tip"  PR_REVIEW_GUARD="$ST_ROOT/refuse-everything.sh"
    # THE CUTOFF, BOTH POLARITIES, over the SAME receiptless tree. A one-sided test
    # here would be satisfied by a cutoff that grandfathers everything - which is the
    # shape of every gate in this repository that turned out to be unable to fail.
    row cutoff-grandfathers-below 0 "a PR BELOW the cutoff is reported, not failed" \
        "$repo"  1000 "$tip"  PR_REVIEW_CUTOFF=2840
    row cutoff-enforces-at-and-above 1 "a PR AT the cutoff with no receipt is RED" \
        "$repo"  2840 "$tip"  PR_REVIEW_CUTOFF=2840

    row public-key-absent         1 "no .github/pr-review.pub — the branch that used to exit 0 forever" \
        "$nokey"  999 "$tip"

    if [ "$st_fail" -ne 0 ]; then
        echo "--- $st_fail row(s) did not produce the required verdict ---" >&2
        return 1
    fi
    echo "--- 23/23 rows, both polarities ---"
    return 0
}

# ---------------------------------------------------------------------------
# THE LEGACY EXEMPTION (FLOW-04; operator ruling relayed by the cop, 2026-09-25 16:14
# Madrid). A receipt signed before #4421 has no diff_patch_id, so A2 cannot bind it.
# PRs already open when this merged were reviewed under the old signer. For them - and
# only them, and only for 24 hours - such a receipt is accepted in place of a
# patch-id match. Both bounds are constants, so a reviewer reads them in the diff:
#   PR_REVIEW_LEGACY_BELOW   the first PR number NOT exempt. 4427 = this PR (#4426) + 1.
#                            PR numbers only increase, so every PR opened after #4426 is
#                            excluded. That is stricter than "open at merge": a PR opened
#                            in the minutes between this PR and its merge must re-sign.
#   PR_REVIEW_LEGACY_UNTIL   epoch at which the exemption ends: 2026-09-26T16:00:00Z,
#                            24h after the 16:00Z merge target. A later merge makes the
#                            window SHORTER, never longer. After it this code is dead,
#                            and removing it is a follow-up diff, not a precondition.
# The receipt's signature is still verified (A3/A4); only the diff binding is waived.
PR_REVIEW_LEGACY_BELOW=${PR_REVIEW_LEGACY_BELOW:-4427}
PR_REVIEW_LEGACY_UNTIL=${PR_REVIEW_LEGACY_UNTIL:-1790438400}
# bashrs DET002: the 24h expiry IS a clock read; PR_REVIEW_NOW pins it in the self-test.
legacy_now() { echo "${PR_REVIEW_NOW:-$(date -u +%s)}"; }  # bashrs disable-line=DET002
legacy_exempt() {
    [ "$1" -lt "$PR_REVIEW_LEGACY_BELOW" ] 2>/dev/null && [ "$(legacy_now)" -lt "$PR_REVIEW_LEGACY_UNTIL" ] 2>/dev/null
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

# ---------------------------------------------------------------------------
# THE GRANDFATHER CUTOFF. `receipt_presence` is fixed at 100% with no ratchet, and on
# 2026-09-01 the repository's actual rate was 1 receipt across 24 open pull requests.
# Turning that into a REQUIRED check in one step would block every open PR at once -
# not a ratchet, an outage - and this repository has already had a day where one armed
# gate blocked all nine open PRs.
#
# So the rule arms forward. A pull request numbered below the cutoff is GRANDFATHERED:
# reported, counted, and not failed. At or above it, a missing receipt is RED and the
# check is in `gate`'s `needs:`.
#
# THE CUTOFF IS A NUMBER AND NOT A DATE for the reason MECHANISM_PATHS is a list and
# not a pattern: PR numbers only increase, the comparison is total, and there is no
# timezone in it. Raising it is a diff a reviewer can read; lowering it below an open
# PR is the visible act of exempting that PR.
PR_REVIEW_CUTOFF=${PR_REVIEW_CUTOFF:-2840}

if [ "$PR_NUMBER" -lt "$PR_REVIEW_CUTOFF" ] 2>/dev/null; then
    echo "$PROG: GRANDFATHERED - PR $PR_NUMBER predates the receipt cutoff ($PR_REVIEW_CUTOFF)."
    echo "  The review is still owed and its absence is still recorded by the S13.11"
    echo "  shadow lane, which reports REFUSE [Q1] for exactly this shape. What is"
    echo "  suppressed here is only the BLOCK, so arming the rule does not stop 24"
    echo "  in-flight pull requests at once."
    exit 0
fi

echo "$PROG: Arm 4 - this PR's own receipt (PR $PR_NUMBER, head $PR_HEAD_SHA)"
if arm4 "$ROOT" "$PR_NUMBER" "$PR_HEAD_SHA"; then
    echo "$PROG: PASS"
    exit 0
fi
echo "$PROG: FAIL - see the arm above." >&2
exit 1
