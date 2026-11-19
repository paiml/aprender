#!/usr/bin/env bash
# check_spec_invariants_pv.sh - every invariant row in the SIM3D spec must name a
# contract that `pv validate` accepts and an equation present in that file (#7252).
# LAB check: nightly only, never on a PR, merge-queue or release path.
set -euo pipefail
spec="${1:-docs/specifications/sim3d-provable-simulation.md}"
[ -r "$spec" ] || { echo "FAIL: cannot read $spec"; exit 2; }
n=0; bad=0
while IFS='|' read -r _ inv eq contract _; do
  inv="${inv# }"; eq="$(echo "$eq" | tr -d ' ')"; contract="$(echo "$contract" | tr -d ' ')"
  n=$((n + 1))
  rc=0; pv validate "$contract" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAIL: ${inv%% *}: pv rejects '$contract' (rc=$rc)"; bad=$((bad + 1)); continue; fi
  if ! grep -qE "^  ${eq}:" "$contract"; then
    echo "FAIL: ${inv%% *}: equation '$eq' not in $contract"; bad=$((bad + 1))
  fi
done < <(grep -E '^\| INV-' "$spec")
[ "$n" -gt 0 ] || { echo "FAIL: no invariant rows in $spec"; exit 1; }
[ "$bad" -eq 0 ] || exit 1
echo "PASS: $n invariants map to pv-accepted contracts"
