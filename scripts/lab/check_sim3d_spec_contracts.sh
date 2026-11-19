#!/usr/bin/env bash
# scripts/lab/check_sim3d_spec_contracts.sh -- aprender#7252 (SIM3D). Every invariant row in the invariant map of
# docs/specifications/sim3d-provable-simulation.md must name a contract that exists, that `pv validate` accepts,
# and that contains the named obligation id. A planted fake contract path or id turns it RED.
# A LAB check: run on demand or at night, never inside a PR, merge-queue or release job (C324).
# Usage: scripts/lab/check_sim3d_spec_contracts.sh [spec-file] | --self-test   Exit: 0 green, 1 RED, 2 cannot measure
set -euo pipefail
SPEC_DEFAULT=docs/specifications/sim3d-provable-simulation.md

check() { # spec-file -> RED lines, "CHECKED n" last
  local f="$1" n=0 bad=0 inv path ob
  [ -s "$f" ] || { echo "cannot measure: no $f" >&2; return 2; }
  [ -n "${PV:-}" ] || { . scripts/pv_bin.sh || { echo "cannot measure: no pv" >&2; return 2; }; }
  while IFS='|' read -r _ inv _ path ob _; do
    inv=$(echo "$inv" | tr -d ' '); path=$(echo "$path" | tr -d ' `'); ob=$(echo "$ob" | tr -d ' `')
    n=$((n + 1))
    if [ ! -f "$path" ]; then echo "RED $inv: contract $path does not exist"; bad=1; continue; fi
    if ! "$PV" validate "$path" > /dev/null 2>&1; then echo "RED $inv: pv validate rejects $path"; bad=1; continue; fi
    if ! grep -qF -- "$ob" "$path"; then echo "RED $inv: $path has no obligation $ob"; bad=1; fi
  done < <(grep -E '^\| SIM3D-INV-[0-9]+ ' "$f")
  echo "CHECKED $n invariant row(s)"
  [ "$n" -gt 0 ] || { echo "cannot measure: 0 invariant rows" >&2; return 2; }
  return "$bad"
}
if [ "${1:-}" = --self-test ]; then
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT; fail=0; s="$t/s.md"
  cp "$SPEC_DEFAULT" "$s"
  check "$s" > /dev/null || { echo "FAIL the real map is RED"; fail=1; }
  cp "$SPEC_DEFAULT" "$s"; sed -i '0,/sim3d-spec-v1.yaml/s//sim3d-no-such-v1.yaml/' "$s"
  check "$s" > /dev/null && { echo "FAIL planted fake contract passes (mutant)"; fail=1; }
  cp "$SPEC_DEFAULT" "$s"; sed -i '0,/SIM-SPEC-INV-001/s//SIM-SPEC-INV-999/' "$s"
  check "$s" > /dev/null && { echo "FAIL planted fake obligation id passes (mutant)"; fail=1; }
  : > "$s"; check "$s" > /dev/null 2>&1 && { echo "FAIL empty spec passes"; fail=1; }
  [ "$fail" = 0 ] && echo "check_sim3d_spec_contracts: self-test PASS (4 rows)"; exit "$fail"
fi
check "${1:-$SPEC_DEFAULT}"
