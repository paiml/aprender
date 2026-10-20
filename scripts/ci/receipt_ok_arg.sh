#!/usr/bin/env bash
# receipt_ok_arg.sh BODY_FILE BASE_REF PR_HEAD_SHA  (PMAT-5797, #5797)
# Prints ONE word for the tier step's `--receipt-ok`: 1 only when the receipt in the PR body
# verified, 0 in every other case (no receipt, a bad one, a missing file, a crashed verifier,
# a base that has no judge yet). Always exits 0: 0 sends the PR to the full gate.
#
# A PR cannot edit its own judge (the #5793 pattern): the verifier, the toolchain-id recipe
# and the pinned batcher public key are read from BASE_REF (refs/remotes/origin/main, pinned
# by the caller), never from the working tree. The working tree supplies only what is being
# judged: HEAD^{tree} and rust-toolchain.toml. The caller must itself run THIS script from
# BASE_REF too (`git show BASE_REF:scripts/ci/receipt_ok_arg.sh`).
# Cases: scripts/check_receipt_gate.sh (rows W1-W14).
set -u
body=${1:-}; base=${2:-}; sha=${3:-}
[ -n "$base" ] && [ -n "$sha" ] || { printf '0\n'; exit 0; }
j=$(mktemp -d) || { printf '0\n'; exit 0; }
trap 'rm -rf "${j:?}"' EXIT
git rev-parse --verify -q "$base^{commit}" >/dev/null 2>&1 || { printf '0\n'; exit 0; }
git archive "$base" scripts/ci_receipt_gate.sh scripts/receipt_toolchain_id.sh .flow009/batcher.pub 2>/dev/null \
  | tar -x -C "$j" 2>/dev/null || { printf '0\n'; exit 0; }
first=$(bash "$j/scripts/ci_receipt_gate.sh" verify "$body" "$j/.flow009/batcher.pub" "$sha" 2>/dev/null | head -n1) || first=""
if [ "$first" = "RECEIPT_OK=1" ]; then printf '1\n'; else printf '0\n'; fi
