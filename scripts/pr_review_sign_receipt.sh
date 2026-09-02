#!/usr/bin/env bash
# pr_review_sign_receipt.sh - the CI signer §4.3 names and nothing implemented.
#
# PR-REVIEW-SKILL-002 v2 §4.3: the signing key's secret half "is held by whoever runs
# the reviewer ... and a copy is escrowed in the repository secret
# PR_REVIEW_SIGNING_KEY_B64 (base64 of the minisign secret-key *file* - §4.3's
# `minisign -S -s` takes a path, so A CI SIGNER MATERIALISES IT BEFORE USE)."
#
# THAT SIGNER DID NOT EXIST. Measured 2026-09-01: `PR_REVIEW_SIGNING_KEY_B64` is a real
# repository secret (created 2026-08-31) and `grep -rn PR_REVIEW_SIGNING_KEY
# .github/workflows/` matches only COMMENTS. An escrowed key with no consumer is a key
# that cannot sign anything, which is why the repository holds exactly one receipt and
# every pull request records `REFUSE [Q1] - receipt is missing`.
#
# WHY THIS IS NEEDED AT ALL. The review is performed by an agent; the agent runs where
# a human runs it, and the secret half is deliberately NOT in the repository or on the
# reviewer's box. So the two halves of a receipt are produced in two places: the agent
# writes the CONTENT, and this materialises the key and attaches the SIGNATURE.
#
# WHAT THE SIGNATURE THEREFORE MEANS, SAID PLAINLY. It binds the receipt to the key, not
# to a claim that a review happened - §4.3 already records `attestation_level: L1-self`
# and §13.12 already says a signed receipt is a provenance claim rather than proof of
# honesty. This changes nothing about that; it makes the provenance claim producible.
#
# IT REFUSES RATHER THAN WEAKENS. Every failure below is exit 1 with a named cause. In
# particular it VERIFIES ITS OWN OUTPUT against the committed public key before
# returning: a signature that does not verify under `.github/pr-review.pub` is worse
# than no signature, because Arm 4 would then reject a receipt whose content was fine
# and the reviewer would go looking in the wrong place.
#
# WHAT THE CASE TABLE PROVES, AND WHAT IT DOES NOT. Eight rows; THREE branches are
# mutation-verified as independently load-bearing (the self-verify, the delete-on-bad-
# signature, and the unset-secret refusal). The base64-decode check and the
# missing-receipt check are NOT: mutated away - even with the downstream minisign
# failure also suppressed - every row still passes, because `minisign` writes no
# artifact in those cases and the "reported success but wrote no signature" branch
# catches them anyway. They are kept as DIAGNOSTICS, not enforcement: "did not
# base64-decode to a non-empty file" tells the reader where to look and
# "minisign -S failed (rc 1)" does not. Recording the distinction so nobody reads
# "8 rows" as "8 independently killable rules" - which is the exact overclaim §6.4
# exists to prevent.
#
# USAGE
#   pr_review_sign_receipt.sh <receipt-dir>
#   pr_review_sign_receipt.sh --self-test
#
# ENVIRONMENT
#   PR_REVIEW_SIGNING_KEY_B64  base64 of the minisign secret-key FILE (required)
#   PR_REVIEW_SIGNING_PASSWORD passphrase, or empty for a `-W` (unencrypted) key
#   PR_REVIEW_PUBKEY           public key to verify against (default .github/pr-review.pub)
#
# EXIT
#   0  receipt.intoto.jsonl.minisig exists and VERIFIES under the public key
#   1  it does not, and the reason is named
#   2  the box cannot answer (no minisign, no base64)

set -uo pipefail

PROG=${0##*/}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
PUBKEY=${PR_REVIEW_PUBKEY:-$REPO_ROOT/.github/pr-review.pub}

WORK=''
ST_ROOT=''
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
# The materialised secret key never outlives the process, on any exit path.
cleanup() { safe_rm_scratch "$WORK" 'prsign.'; safe_rm_scratch "$ST_ROOT" 'prsign-selftest.'; }
trap cleanup EXIT

die_env() { echo "$PROG: ENV - $*" >&2; exit 2; }
fail()    { echo "$PROG: FAIL - $*" >&2; exit 1; }

sign_receipt() {
    local dir=$1 rcpt sig keyfile rc=0
    rcpt="$dir/receipt.intoto.jsonl"
    sig="$rcpt.minisig"

    command -v minisign >/dev/null 2>&1 || die_env "minisign is not on PATH"
    command -v base64   >/dev/null 2>&1 || die_env "base64 is not on PATH"

    [ -d "$dir" ]   || fail "no such receipt directory: $dir"
    [ -f "$rcpt" ]  || fail "no receipt.intoto.jsonl in $dir; the signer attaches a signature, it does not invent the document"
    [ -f "$PUBKEY" ] || fail "public key $PUBKEY is absent; signing without being able to verify the result is how an unverifiable signature ships"

    [ -n "${PR_REVIEW_SIGNING_KEY_B64:-}" ] \
      || fail "PR_REVIEW_SIGNING_KEY_B64 is unset or empty. It is a repository secret; this runs where that secret is available, and NOT on a reviewer's box (§4.3: the secret half is not in this repository and never will be)"

    WORK=$(mktemp -d -t prsign.XXXXXX) || die_env "cannot create a temp dir"
    chmod 700 "$WORK"
    keyfile="$WORK/minisign.key"

    printf '%s' "$PR_REVIEW_SIGNING_KEY_B64" | base64 -d > "$keyfile" 2>/dev/null
    rc=$?
    [ "$rc" -eq 0 ] && [ -s "$keyfile" ] \
      || fail "PR_REVIEW_SIGNING_KEY_B64 did not base64-decode to a non-empty file; §4.3 says it is base64 of the minisign secret-key FILE, not of the key line"
    chmod 600 "$keyfile"

    # minisign reads the passphrase from stdin. An unencrypted (-W) key still consumes
    # an empty line, so both shapes take the same path and neither can hang a CI job.
    printf '%s\n' "${PR_REVIEW_SIGNING_PASSWORD:-}" \
      | minisign -S -s "$keyfile" -m "$rcpt" \
                 -t "PR-REVIEW-SKILL-002 v2 §4.3 receipt" \
                 -c "signed by the CI signer" >/dev/null 2>&1
    rc=${PIPESTATUS[1]}
    [ "$rc" -eq 0 ] || fail "minisign -S failed (rc $rc); the key may be passphrase-protected with PR_REVIEW_SIGNING_PASSWORD unset"

    [ -f "$sig" ] || fail "minisign reported success but wrote no $sig; an exit code is not an artifact"

    # VERIFY OUR OWN OUTPUT. A signature that does not verify under the COMMITTED public
    # key is worse than none: Arm 4 would reject a receipt whose content was fine.
    minisign -V -m "$rcpt" -p "$PUBKEY" >/dev/null 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        rm -f "$sig"
        fail "the signature this just produced does NOT verify under $PUBKEY. The escrowed secret and the committed public half are not a pair - rotate per §4.3 (minisign -G -W, replace the .pub, re-set the secret, re-sign) rather than shipping a signature nothing can check. The bad signature has been removed."
    fi

    echo "SIGNED  $rcpt"
    echo "        verified under $PUBKEY"
    return 0
}

# ---------------------------------------------------------------------------
# --self-test - a hermetic keypair, so every branch runs without the real secret.
# ---------------------------------------------------------------------------
self_test() {
    local fails=0 pass_n=0 td pub sec b64
    command -v minisign >/dev/null 2>&1 || die_env "minisign is not on PATH"
    td=$(mktemp -d -t prsign-selftest.XXXXXX) || die_env "mktemp"
    ST_ROOT=$td

    # A throwaway keypair, and a SECOND one to prove the verify branch can fail.
    printf '\n\n' | minisign -G -W -p "$td/a.pub" -s "$td/a.key" >/dev/null 2>&1
    printf '\n\n' | minisign -G -W -p "$td/b.pub" -s "$td/b.key" >/dev/null 2>&1
    [ -f "$td/a.key" ] && [ -f "$td/b.pub" ] || die_env "could not generate throwaway keys"

    mkdir -p "$td/r"
    printf '{"_type":"https://in-toto.io/Statement/v1"}\n' > "$td/r/receipt.intoto.jsonl"

    row() { # desc expect env-b64 pubkey dir
        local desc=$1 expect=$2 b=$3 pk=$4 d=$5 rc=0 out
        rm -f "$d/receipt.intoto.jsonl.minisig" 2>/dev/null
        out=$(PR_REVIEW_SIGNING_KEY_B64="$b" PR_REVIEW_PUBKEY="$pk" \
              bash "$HERE/$PROG" "$d" 2>&1)
        rc=$?
        if { [ "$expect" = PASS ] && [ "$rc" -eq 0 ]; } \
        || { [ "$expect" = FAIL ] && [ "$rc" -eq 1 ]; } \
        || { [ "$expect" = ENV  ] && [ "$rc" -eq 2 ]; }; then
            printf 'ok   %s\n' "$desc"; pass_n=$((pass_n + 1))
        else
            printf 'FAIL %s (expected %s, rc=%s)\n     %s\n' "$desc" "$expect" "$rc" "$out"
            fails=$((fails + 1))
        fi
    }

    b64=$(base64 -w0 < "$td/a.key" 2>/dev/null || base64 < "$td/a.key" | tr -d '\n')

    echo "== pr_review_sign_receipt.sh --self-test =="
    row 'a matching keypair signs and verifies'          PASS "$b64"        "$td/a.pub" "$td/r"
    row 'the signature must verify under the PUBLIC key' FAIL "$b64"        "$td/b.pub" "$td/r"
    row 'an unset secret is a named refusal, not a skip' FAIL ""            "$td/a.pub" "$td/r"
    row 'a non-base64 secret is refused'                 FAIL "!!not-b64!!" "$td/a.pub" "$td/r"
    row 'an absent public key is refused'                FAIL "$b64"  "$td/nope.pub"    "$td/r"
    mkdir -p "$td/empty"
    row 'a directory with no receipt is refused'         FAIL "$b64"        "$td/a.pub" "$td/empty"
    row 'a missing directory is refused'                 FAIL "$b64"        "$td/a.pub" "$td/absent"

    # The mismatch row must also LEAVE NO SIGNATURE behind: a bad artifact on disk is
    # what a later run would read as success.
    rm -f "$td/r/receipt.intoto.jsonl.minisig"
    PR_REVIEW_SIGNING_KEY_B64="$b64" PR_REVIEW_PUBKEY="$td/b.pub" \
      bash "$HERE/$PROG" "$td/r" >/dev/null 2>&1
    if [ ! -f "$td/r/receipt.intoto.jsonl.minisig" ]; then
        printf 'ok   %s\n' 'a signature that fails verification is DELETED, not left on disk'
        pass_n=$((pass_n + 1))
    else
        printf 'FAIL %s\n' 'a signature that fails verification is DELETED, not left on disk'
        fails=$((fails + 1))
    fi

    printf '\n%s passed, %s failed\n' "$pass_n" "$fails"
    [ "$fails" -eq 0 ] || { echo 'SELF-TEST FAILED'; exit 1; }
    exit 0
}

case "${1:-}" in
  --self-test) self_test ;;
  -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
  '')          fail "usage: $PROG <receipt-dir> | --self-test" ;;
  *)           sign_receipt "$1" ;;
esac
