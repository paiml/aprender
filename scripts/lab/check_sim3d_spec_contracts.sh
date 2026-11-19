#!/usr/bin/env bash
# scripts/lab/check_sim3d_spec_contracts.sh -- aprender#7252 (SIM3D). Every invariant row in the invariant map of
# docs/specifications/sim3d-provable-simulation.md must name a contract that exists, that `pv validate` accepts,
# and that contains the named obligation id. A planted fake contract path or id turns it RED.
# A LAB check: run on demand or at night, never inside a PR, merge-queue or release job (C324). The case-table
# test crates/aprender-contracts-cli/tests/sim3d_spec.rs is the separate PR-path check (fragment 871).
# Usage: scripts/lab/check_sim3d_spec_contracts.sh [spec-file] | --self-test
# Exit: 0 green, 1 RED (a bad row, a skipped or duplicate row, or only malformed rows), 2 cannot measure
# (no spec, no pv, or a spec that names no SIM3D-INV id at all). RED outranks cannot-measure.
set -euo pipefail
SPEC_DEFAULT=docs/specifications/sim3d-provable-simulation.md

check() { # spec-file -> RED lines, "CHECKED n" last
  local f="$1" n=0 bad=0 inv path ob named ids dups missing
  [ -s "$f" ] || { echo "cannot measure: no $f" >&2; return 2; }
  [ -n "${PV:-}" ] || { . scripts/pv_bin.sh || { echo "cannot measure: no pv" >&2; return 2; }; }
  ids=""
  while IFS='|' read -r _ inv _ path ob _; do
    inv=$(echo "$inv" | tr -d ' '); path=$(echo "$path" | tr -d ' `'); ob=$(echo "$ob" | tr -d ' `')
    n=$((n + 1)); ids="$ids$inv"$'\n'
    if ! [[ "$ob" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then echo "RED $inv: obligation cell '$ob' is empty or not an id"; bad=1; continue; fi
    if [ ! -f "$path" ]; then echo "RED $inv: contract $path does not exist"; bad=1; continue; fi
    if ! "$PV" validate "$path" > /dev/null 2>&1; then echo "RED $inv: pv validate rejects $path"; bad=1; continue; fi
    if ! grep -qE -- "^[[:space:]]*-[[:space:]]+id:[[:space:]]*[\"']?${ob}[\"']?[[:space:]]*$" "$path"; then echo "RED $inv: $path has no obligation $ob"; bad=1; fi
  done < <(grep -E '^\| SIM3D-INV-[0-9]+ ' "$f" || true)
  named=$({ grep -oE 'SIM3D-INV-[0-9]+' "$f" || true; } | sort -u)
  echo "CHECKED $n invariant row(s) of $(printf '%s' "$named" | grep -c . || true) named"
  if [ "$n" -eq 0 ] && [ -z "$named" ]; then echo "cannot measure: 0 invariant rows" >&2; return 2; fi
  dups=$(printf '%s' "$ids" | sort | uniq -d | tr '\n' ' ')
  if [ -n "$dups" ]; then echo "RED duplicate invariant row(s): $dups"; bad=1; fi
  missing=$(comm -13 <(printf '%s' "$ids" | sort -u) <(printf '%s\n' "$named") | tr '\n' ' ')
  if [ -n "${missing// /}" ]; then echo "RED named but not checked (a malformed row was skipped): $missing"; bad=1; fi
  if [ "$n" -eq 0 ]; then echo "RED 0 well-formed invariant rows although the spec names SIM3D-INV ids"; bad=1; fi
  return "$bad"
}
if [ "${1:-}" = --self-test ]; then
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT; fail=0; s="$t/s.md"; cases=0
  # expect <rc> <output substring> <label>: runs check on $s; the exit code must be EXACTLY <rc>.
  expect() {
    local want="$1" sub="$2" label="$3" out rc=0
    cases=$((cases + 1))
    out=$(check "$s" 2>&1) || rc=$?
    if [ "$rc" -ne "$want" ]; then echo "FAIL $label: exit $rc, want $want"; fail=1
    elif [ -n "$sub" ] && ! printf '%s' "$out" | grep -qF -- "$sub"; then echo "FAIL $label: output lacks '$sub'"; fail=1; fi
  }
  cp "$SPEC_DEFAULT" "$s"; expect 0 "CHECKED" "the real map is green"
  cp "$SPEC_DEFAULT" "$s"; sed -i '/^| SIM3D-INV-004/s/sim3d-spec-v1/sim3d-no-such-v1/' "$s"
  expect 1 "does not exist" "planted fake contract (mutant)"
  cp "$SPEC_DEFAULT" "$s"; sed -i '/^| SIM3D-INV-004/s/SIM-SPEC-INV-001/SIM-SPEC-INV-999/' "$s"
  expect 1 "has no obligation" "planted fake obligation id (mutant)"
  cp "$SPEC_DEFAULT" "$s"; sed -i '/^| SIM3D-INV-004/s/SIM-SPEC-INV-001/SIM-SPEC-INV-00/' "$s"
  expect 1 "has no obligation" "truncated obligation id (mutant)"
  cp "$SPEC_DEFAULT" "$s"; sed -i '/^| SIM3D-INV-004/s/`SIM-SPEC-INV-001`/`kind`/' "$s"
  expect 1 "has no obligation" "non-id word as obligation (mutant)"
  cp "$SPEC_DEFAULT" "$s"; sed -i '/^| SIM3D-INV-004/s/`SIM-SPEC-INV-001`/` `/' "$s"
  expect 1 "empty or not an id" "empty obligation cell (mutant)"
  cp "$SPEC_DEFAULT" "$s"; sed -i '/^| SIM3D-INV-004/{s/^| /|/;s/SIM-SPEC-INV-001/FAKE-999/}' "$s"
  expect 1 "malformed row was skipped" "malformed row skipped (mutant)"
  cp "$SPEC_DEFAULT" "$s"; sed -i '/^| SIM3D-INV-004/{s/^| /|/;s/SIM-SPEC-INV-001/FAKE-999/}' "$s"
  grep '^| SIM3D-INV-001 ' "$s" > "$t/row"; cat "$t/row" >> "$s"
  expect 1 "malformed row was skipped" "malformed row masked by a duplicate row (mutant)"
  cp "$SPEC_DEFAULT" "$s"; grep '^| SIM3D-INV-001 ' "$s" > "$t/row"; cat "$t/row" >> "$s"
  expect 1 "duplicate invariant row" "duplicate row (mutant)"
  printf '# header only\nSIM3D-INV-001 is named but has no table row\n' > "$s"
  expect 1 "RED" "every row malformed is RED, not cannot-measure (mutant)"
  printf '# header only, no ids\n' > "$s"
  expect 2 "0 invariant rows" "non-empty spec with zero rows is loud cannot-measure"
  : > "$s"; expect 2 "cannot measure" "empty spec file"
  if [ "$fail" = 0 ]; then echo "check_sim3d_spec_contracts: self-test PASS ($cases cases)"; fi
  exit "$fail"
fi
check "${1:-$SPEC_DEFAULT}"
