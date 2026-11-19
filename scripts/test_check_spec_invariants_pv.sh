#!/usr/bin/env bash
# Case table for check_spec_invariants_pv.sh: real spec passes; planted defects fail.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
spec="$here/../docs/specifications/sim3d-provable-simulation.md"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/sim3d.XXXXXX")"; trap 'rm -rf "${tmp:?}"' EXIT
fail=0
expect() { # name want-rc file
  rc=0; bash "$here/check_spec_invariants_pv.sh" "$3" >/dev/null 2>&1 || rc=$?
  if { [ "$2" = 0 ] && [ "$rc" -eq 0 ]; } || { [ "$2" = nz ] && [ "$rc" -ne 0 ]; }; then echo "ok   $1"; else echo "FAIL $1 rc=$rc"; fail=1; fi
}
expect real-spec 0 "$spec"
sed 's#simulation-step-v1.yaml#fake-contract-v9.yaml#' "$spec" > "$tmp/fake_contract.md"; expect fake-contract-id nz "$tmp/fake_contract.md"
sed 's#| audit_trail |#| no_such_equation |#' "$spec" > "$tmp/fake_eq.md"; expect fake-equation nz "$tmp/fake_eq.md"
grep -v '^| INV-' "$spec" > "$tmp/none.md"; expect no-rows nz "$tmp/none.md"
exit "$fail"
