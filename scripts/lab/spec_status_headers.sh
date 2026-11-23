#!/usr/bin/env bash
# scripts/lab/spec_status_headers.sh -- LAB (C324 items 4-5), spec-sweep 2026-10-31 (S1: #6604 #6628 #6629 #6636 #6637 #6638 #6639 #6643).
# Rule enforced: each spec in the registry below carries exactly ONE Status header in its first 12 lines, and the state
# is a member of the closed vocabulary (PROPOSED|REVIEW|ACCEPTED|ACTIVE|IMPLEMENTED|SUPERSEDED|WITHDRAWN, any case).
# A missing, duplicated, free-text or out-of-vocabulary Status is RED. A registry path that does not exist is RED
# (the universe may not empty itself). This does NOT judge whether the stated state is TRUE of the code; flipping a
# state is a separate, evidenced change. Never in a PR, merge-queue or release job; run nightly or on demand.
# Usage: scripts/lab/spec_status_headers.sh [repo-root] | --self-test      Exit: 0 green, 1 RED, 2 usage/vacuous
set -euo pipefail
REGISTRY=(
  docs/specifications/06x-release-schedule.md
  docs/specifications/book-completeness-spec.md
  docs/specifications/book-execution-validation-harness-spec.md
  docs/specifications/aprender-train/model-1-qlora-retry-plan.md
  docs/specifications/aprender-gpu/blackwell-backend-fix-spec.md
  docs/specifications/aprender-train/llama-ideas-inclusion-spec.md
  docs/specifications/aprender-train/yaml-mode-train.md
  docs/specifications/aprender-compute/ml-tuner-bricks.md
)
VOCAB='PROPOSED|REVIEW|ACCEPTED|ACTIVE|IMPLEMENTED|SUPERSEDED|WITHDRAWN'
HDR='^(- )?\*\*(Plan )?Status:?\*\*:?'
# one spec file -> RED lines (empty when green)
scan() {
  local f=$1 n state
  n=$(head -n 12 "$f" | grep -c -E "$HDR" || true)
  if [ "$n" -ne 1 ]; then echo "RED $f: $n Status headers in first 12 lines (want exactly 1)"; return; fi
  state=$(head -n 12 "$f" | grep -E "$HDR" | sed -E "s/${HDR}[[:space:]]*//" | sed -E 's/^([A-Za-z]+).*/\1/' | tr '[:lower:]' '[:upper:]')
  if [[ ! "$state" =~ ^($VOCAB)$ ]]; then echo "RED $f: Status '$state' not in {$VOCAB}"; fi
}
run() {
  local root=$1 f out="" reds=0
  [ -d "$root" ] || { echo "usage: no such dir $root" >&2; return 2; }
  for f in "${REGISTRY[@]}"; do
    if [ ! -f "$root/$f" ]; then echo "RED $f: registered spec missing"; reds=$((reds + 1)); continue; fi
    out=$(scan "$root/$f"); if [ -n "$out" ]; then echo "$out"; reds=$((reds + 1)); fi
  done
  if [ "$reds" -eq 0 ]; then echo "GREEN: ${#REGISTRY[@]} specs, all Status headers valid"; return 0; fi
  return 1
}
selftest() {
  local t rc fail=0 f skip=""
  t=$(mktemp -d)
  # mk [skip-index]: write a good tree, optionally omitting one registry file
  mk() { for f in "${REGISTRY[@]}"; do
      [ "$f" = "$skip" ] && continue
      mkdir -p "$t/$(dirname "$f")"; printf '# T\n\n**Status**: PROPOSED (x)\n' > "$t/$f"
    done; }
  expect() { # want-rc label
    rc=0; run "$t" >/dev/null || rc=$?
    [ "$rc" -eq "$1" ] || { echo "FAIL $2 rc=$rc want $1"; fail=1; }
  }
  mk; expect 0 "good tree"
  mk; printf '# T\n\nno header\n' > "$t/${REGISTRY[0]}"; expect 1 "missing header"
  mk; printf '# T\n**Status**: PROPOSED\n**Status**: ACTIVE\n' > "$t/${REGISTRY[1]}"; expect 1 "duplicate"
  mk; printf '# T\n**Status**: maybe later\n' > "$t/${REGISTRY[2]}"; expect 1 "bad vocab"
  t=$(mktemp -d); skip=${REGISTRY[3]}; mk; expect 1 "missing file"; skip=""
  mk; printf '# T\n- **Status:** Proposed\n' > "$t/${REGISTRY[4]}"; expect 0 "list form"
  if [ "$fail" -eq 0 ]; then echo "self-test: 6 cases ok"; fi
  return "$fail"
}
case "${1:-}" in
  --self-test) selftest ;;
  *) run "${1:-.}" ;;
esac
