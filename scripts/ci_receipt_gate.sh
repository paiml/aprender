#!/usr/bin/env bash
# ci_receipt_gate.sh verify BODY_FILE PUB_FILE PR_HEAD_SHA  (PMAT-5797, #5797)
#
# The PR gate may run check/fmt/clippy instead of the full suite only for a batch whose
# signed receipt verifies. Three checks decide it, in this order:
#   signature   the armored ssh-keygen signature over the canonical bytes
#               (`"$JQ" -S -c 'del(.sig)'` plus LF, namespace below) verifies against the
#               single pinned batcher public key in PUB_FILE
#   tree        the receipt's tree == HEAD^{tree}, the PR merge tree (run inside the
#               checked-out merge commit)
#   toolchain   the receipt's toolchain == scripts/receipt_toolchain_id.sh here
# BODY_FILE is the PR body; the receipt is its one fenced block whose info string is
# exactly the schema id. Prints `RECEIPT_OK=1` or `RECEIPT_OK=0 reason=<code>` as the
# first line and exits 0 either way: the caller falls through to the full gate on 0.
# Anything missing or malformed fails closed. Never on a tag ref or a push event: that
# belongs to the caller (the full gate owns those).
# Cases: scripts/check_receipt_gate.sh.
set -u
# Absolute tools and a fixed PATH: a PATH entry or a BASH_ENV function must never stand in for
# ssh-keygen or jq (#5798). Callers also run this under `env -i`.
PATH=/usr/local/bin:/usr/bin:/bin; export PATH
unset BASH_ENV ENV
SSHKEYGEN=/usr/bin/ssh-keygen
JQ=/usr/bin/jq
NS="flow009-receipt"
SIGNER_ID="flow009-batcher"
SCHEMA="flow009.receipt/1"
HERE=$(cd "$(dirname "$0")" && pwd)

fail() { printf 'RECEIPT_OK=0 reason=%s\n' "$1"; exit 0; }

canonical() { "$JQ" -S -c 'del(.sig)' "$1"; }

# The fenced block's contents on stdout; non-zero unless there is exactly one.
extract_block() {
  tr -d '\r' <"$1" | awk -v info="\`\`\`$SCHEMA" '
    state == 0 && $0 == info { state = 1; n++; next }
    state == 1 && $0 == "```" { state = 0; next }
    state == 1 { print }
    END { if (n != 1 || state != 0) exit 1 }
  '
}

cmd_verify() {
  [ $# -eq 3 ] || fail usage
  body=$1; pub=$2
  [ -x "$SSHKEYGEN" ] || fail no-ssh-keygen
  [ -x "$JQ" ] || fail no-jq
  [ -f "$body" ] || fail no-body
  [ -f "$pub" ] || fail no-pubkey
  work=$(mktemp -d) || fail no-tmpdir
  trap 'rm -rf "${work:?}"' EXIT
  # The key: first non-comment line. A comment-only file is the placeholder.
  keyline=$(grep -v -E '^[[:space:]]*(#|$)' "$pub" | head -n1 | cut -d' ' -f1,2)
  [ -n "$keyline" ] || fail placeholder-pubkey
  case "$keyline" in ssh-ed25519\ *) ;; *) fail bad-pubkey ;; esac
  printf '%s namespaces="%s" %s\n' "$SIGNER_ID" "$NS" "$keyline" >"$work/allowed"
  extract_block "$body" >"$work/receipt.json" || fail fenced-block
  "$JQ" -e --arg s "$SCHEMA" '
    type == "object" and .schema == $s
    and (.sig | type == "string") and (.tree | type == "string") and (.toolchain | type == "string")
  ' "$work/receipt.json" >/dev/null 2>&1 || fail shape
  # signature
  canonical "$work/receipt.json" >"$work/canon" || fail canon
  "$JQ" -r '.sig' "$work/receipt.json" >"$work/sig"
  "$SSHKEYGEN" -Y verify -f "$work/allowed" -I "$SIGNER_ID" -n "$NS" -s "$work/sig" \
    <"$work/canon" >/dev/null 2>&1 || fail bad-signature
  # verdict: the signed bytes must say the batch's suite passed. A validly signed receipt for a
  # red batch must not reduce the gate. gate_results is an object, `tests` is present, and every
  # value is "pass".
  "$JQ" -e '.gate_results | type == "object" and has("tests") and (to_entries | all(.value == "pass"))' \
    "$work/receipt.json" >/dev/null 2>&1 || fail not-pass
  # tree
  want_tree=$(git rev-parse 'HEAD^{tree}' 2>/dev/null)
  [ -n "$want_tree" ] || fail no-head-tree
  [ "$("$JQ" -r .tree "$work/receipt.json")" = "$want_tree" ] || fail tree-mismatch
  # toolchain
  want_tc=$(bash "$HERE/receipt_toolchain_id.sh" 2>/dev/null)
  [ -n "$want_tc" ] || fail no-toolchain-id
  [ "$("$JQ" -r .toolchain "$work/receipt.json")" = "$want_tc" ] || fail toolchain-mismatch
  printf 'RECEIPT_OK=1\n'
}

case "${1:-}" in
  verify) shift; cmd_verify "$@" ;;
  *) fail usage ;;
esac
