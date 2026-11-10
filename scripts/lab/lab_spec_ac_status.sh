#!/usr/bin/env bash
# scripts/lab/lab_spec_ac_status.sh -- aprender#6631/#6632/#6633 (spec-sweep, ship-model-2-spec.md AC-SHIP2-006/009/010).
# Rule: an AC's status row (the "AC movement" table) must exist and must name the SAME check the AC definition table
# defines. Drift = a status ("DISCHARGED", "BLOCKED") attached to a different check than the one the AC enforces.
# A LAB check: run on demand or at night, never inside a PR, merge-queue or release job (C324).
# Usage: scripts/lab/lab_spec_ac_status.sh [spec-file] | --self-test      Exit: 0 green, 1 RED, 2 cannot measure
set -euo pipefail
SPEC_DEFAULT=docs/specifications/aprender-train/ship-model-2-spec.md
ACS="006 009 010"

check() { # spec-file -> RED lines, "CHECKED n" last
  local f="$1" ac def lab n=0 bad=0
  [ -s "$f" ] || { echo "cannot measure: no $f" >&2; return 2; }
  for ac in $ACS; do
    def=$(grep -m1 -E "^\| AC-SHIP2-$ac +\|" "$f" || true)
    lab=$(grep -m1 -E "^\| \*\*AC-SHIP2-$ac\*\* \(" "$f" | sed -E 's/^\| \*\*AC-SHIP2-[0-9]+\*\* \(([^)]*)\).*/\1/' || true)
    if [ -z "$def" ] || [ -z "$lab" ]; then echo "RED AC-SHIP2-$ac: definition or status row missing"; bad=1; continue; fi
    n=$((n + 1))
    if ! grep -qiF -- "$lab" <<<"$def"; then
      echo "RED AC-SHIP2-$ac: status row names '$lab' but the definition is a different check"; bad=1
    fi
  done
  echo "CHECKED $n AC row(s)"
  [ "$n" -gt 0 ] || { echo "cannot measure: 0 AC rows" >&2; return 2; }
  return "$bad"
}
if [ "${1:-}" = --self-test ]; then
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT; fail=0; s="$t/s.md"
  good() { printf '%s\n' '| AC-SHIP2-006  | `fixture-qa` all PASS | F |' '| AC-SHIP2-009  | `fixture-bench` fast | F |' '| AC-SHIP2-010  | llama-cli interop | F |' \
    '| **AC-SHIP2-006** (fixture-qa) | PROPOSED |' '| **AC-SHIP2-009** (fixture-bench) | OK |' '| **AC-SHIP2-010** (llama-cli) | OK |' > "$s"; }
  good; check "$s" > /dev/null || { echo "FAIL consistent spec is RED"; fail=1; }
  good; sed -i 's/(fixture-bench)/(llama-cli interop)/' "$s"; check "$s" > /dev/null && { echo "FAIL label drift passes (mutant)"; fail=1; }
  good; sed -i '/AC-SHIP2-010\*\*/d' "$s"; check "$s" > /dev/null && { echo "FAIL missing status row passes (mutant)"; fail=1; }
  good; sed -i '/^| AC-SHIP2-006/d' "$s"; check "$s" > /dev/null && { echo "FAIL missing definition passes (mutant)"; fail=1; }
  : > "$s"; check "$s" > /dev/null 2>&1 && { echo "FAIL empty spec passes"; fail=1; }
  [ "$fail" = 0 ] && echo "lab_spec_ac_status: self-test PASS (5 rows)"; exit "$fail"
fi
check "${1:-$SPEC_DEFAULT}"
