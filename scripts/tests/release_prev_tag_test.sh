#!/usr/bin/env bash
# Fixture: v0.75.3 sits on a side branch, so it is NOT an ancestor of the cut.
# Sort-only picks 0.75.3 for target 0.75.4; the ancestor rule must pick 0.75.0.
set -uo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
script="$here/scripts/release/prev_final_tag.sh"
d=$(mktemp -d); trap 'rm -rf "${d:?}"' EXIT
cd "$d" || exit 2
git init -q -b main . && git config user.email t@t && git config user.name t
git commit -q --allow-empty -m base && git tag v0.75.0
git checkout -q -b side && git commit -q --allow-empty -m side && git tag v0.75.3
git checkout -q main && git commit -q --allow-empty -m m2 && git tag v0.75.4
got=$(bash "$script" 0.75.4 HEAD 2>/dev/null); rc=$?
if [[ "$got" == "0.75.0" && "$rc" -eq 0 ]]; then
  echo "PASS: prev of 0.75.4 is 0.75.0 (non-ancestor 0.75.3 skipped)"; exit 0
fi
echo "FAIL: prev of 0.75.4 got '${got}' rc=${rc}, want 0.75.0 rc=0"; exit 1
