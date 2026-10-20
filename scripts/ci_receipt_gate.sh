#!/usr/bin/env bash
# ci_receipt_gate.sh verify BODY_FILE PUB_FILE PR_HEAD_SHA  (PMAT-5797, #5797)
#
# The PR gate may run check/fmt/clippy instead of the full suite only for a batch whose
# signed receipt verifies. Three checks decide it, in this order:
#   signature   the armored ssh-keygen signature over the canonical bytes
#               (`jq -S -c 'del(.sig)'` plus LF, namespace below) verifies against the
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
NS="flow009-receipt"
SIGNER_ID="flow009-batcher"
SCHEMA="flow009.receipt/1"
HERE=$(cd "$(dirname "$0")" && pwd)

fail() { printf 'RECEIPT_OK=0 reason=%s\n' "$1"; exit 0; }

canonical() { jq -S -c 'del(.sig)' "$1"; }

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
  command -v ssh-keygen >/dev/null 2>&1 || fail no-ssh-keygen
  command -v jq >/dev/null 2>&1 || fail no-jq
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
  jq -e --arg s "$SCHEMA" '
    type == "object" and .schema == $s
    and (.sig | type == "string") and (.tree | type == "string") and (.toolchain | type == "string")
  ' "$work/receipt.json" >/dev/null 2>&1 || fail shape
  # signature
  canonical "$work/receipt.json" >"$work/canon" || fail canon
  jq -r '.sig' "$work/receipt.json" >"$work/sig"
  ssh-keygen -Y verify -f "$work/allowed" -I "$SIGNER_ID" -n "$NS" -s "$work/sig" \
    <"$work/canon" >/dev/null 2>&1 || fail bad-signature
  # tree
  want_tree=$(git rev-parse 'HEAD^{tree}' 2>/dev/null)
  [ -n "$want_tree" ] || fail no-head-tree
  [ "$(jq -r .tree "$work/receipt.json")" = "$want_tree" ] || fail tree-mismatch
  # toolchain
  want_tc=$(bash "$HERE/receipt_toolchain_id.sh" 2>/dev/null)
  [ -n "$want_tc" ] || fail no-toolchain-id
  [ "$(jq -r .toolchain "$work/receipt.json")" = "$want_tc" ] || fail toolchain-mismatch
  printf 'RECEIPT_OK=1\n'
}

case "${1:-}" in
  verify) shift; cmd_verify "$@" ;;
  *) fail usage ;;
esac
