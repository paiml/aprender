#!/usr/bin/env bash
# receipt_ok_arg.sh BODY_FILE PUB_FILE PR_HEAD_SHA  (PMAT-5797, #5797)
# Prints ONE word for the tier step's `--receipt-ok`: 1 only when scripts/ci_receipt_gate.sh
# verified the receipt in the PR body, 0 in every other case (no receipt, a bad one, a
# missing file, a verifier that crashed). Always exits 0: 0 sends the PR to the full gate.
# Cases: scripts/check_receipt_gate.sh (rows W1-W10).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
first=$(bash "$HERE/../ci_receipt_gate.sh" verify "$@" 2>/dev/null | head -n1) || first=""
if [ "$first" = "RECEIPT_OK=1" ]; then
  printf '%s\n' 1
else
  printf '%s\n' 0
fi
