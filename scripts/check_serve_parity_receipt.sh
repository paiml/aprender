#!/usr/bin/env bash
# check_serve_parity_receipt.sh -- the #4218 serve-parity release gate, as the
# release decision surfaces call it.
#
#   check_serve_parity_receipt.sh <tag_sha> <asset_sha256>   exit 0 only on a PASS
#                                                          receipt bound to both
#   check_serve_parity_receipt.sh --selftest                 every verdict and release
#                                                          row, each flipped by a mutant
#
# The receipt is evidence/serve-parity/gx10/<tag_sha>.json, written by
# `serve_parity_gate.py verdict --write-receipt` on the gx10 lane. The tag sha is
# the TAG commit. On a release branch that is not a main commit, so the caller
# resolves it with `git rev-parse <tag>^{commit}` and never from main.
# There is no bypass flag. A release without a PASS receipt is not a release.
set -euo pipefail

GATE="scripts/lib/serve_parity_gate.py"
[ -f "$GATE" ] || { printf 'FAIL  %s is missing\n' "$GATE"; exit 2; }

if [ "${1:-}" = "--selftest" ]; then
    python3 "$GATE" selftest   # set -e carries its exit status out
    exit 0
fi
if [ "$#" -ne 2 ]; then
    printf 'usage: %s <tag_sha> <asset_sha256> | --selftest\n' "$0" >&2
    exit 2
fi
python3 "$GATE" check-release "$1" "$2"
