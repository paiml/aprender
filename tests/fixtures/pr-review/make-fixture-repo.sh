#!/usr/bin/env bash
# Build the deterministic git repository the pr-review fixtures are written against.
#
# WHY A SYNTHESIZED REPO AND NOT REAL aprender COMMITS
# ----------------------------------------------------
# Spec S6.3 row 10 requires the guard to compute `git merge-base origin/main <head>`
# and compare it to the receipt's `base_sha`. Row 1 requires it to see that the diff
# touches a CUDA path. Neither is exercisable against this repository's own history:
# for ANY commit X reachable from origin/main, `git merge-base origin/main X` is X
# itself, so base_sha would be forced to equal head_sha and every diff would be empty.
# The check would pass vacuously - the same shape as `pv lint <FILE>` returning PASS
# over zero contracts.
#
# So the fixtures are written against a purpose-built repo with a genuine fork:
#
#     C1 ---- C2 ---- C3          <- main, and refs/remotes/origin/main
#      \
#       +---- F1                  <- gpu-pr head   (adds src/cuda/kernel.cu)
#       \
#        +--- D1                  <- docs-pr head  (adds docs/note.md)
#
#   merge-base(origin/main, F1) = C1   (real, non-degenerate)
#   C1 is an ancestor of F1            -> a fresh index
#   C3 is NOT an ancestor of F1        -> a stale index (row 9)
#
# The SHAs are DETERMINISTIC (fixed author/committer identity and dates, no signing,
# no hooks), so the fixture receipts can carry them as literal values and be signed as
# static committed bytes. This script ASSERTS the SHAs it produces against
# expected-shas.txt: if a future git changes object hashing, or someone edits the
# topology, the harness fails loudly instead of silently validating a different repo.
#
# Usage: make-fixture-repo.sh <destination-dir>
set -euo pipefail

DEST=${1:?usage: make-fixture-repo.sh <destination-dir>}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# This expands into rm -rf, so refuse anything that is not a plausible destination.
if [ -z "$DEST" ] || [ "$DEST" = "/" ]; then
  echo "refusing to build a fixture repo at $DEST" >&2
  exit 1
fi
rm -rf -- "$DEST"
mkdir -p -- "$DEST"

# Hermetic: fixed identity and timestamps make the SHAs reproducible; hooksPath and
# gpgsign are pinned so a developer's global config cannot change the objects.
export GIT_AUTHOR_NAME="prrev fixture"
export GIT_AUTHOR_EMAIL="prrev@fixture.invalid"
export GIT_COMMITTER_NAME="prrev fixture"
export GIT_COMMITTER_EMAIL="prrev@fixture.invalid"
export GIT_AUTHOR_DATE="2026-01-01T00:00:00+0000"
export GIT_COMMITTER_DATE="2026-01-01T00:00:00+0000"

g() { git -C "$DEST" "$@"; }

g init -q -b main .
g config core.hooksPath /dev/null
g config commit.gpgsign false
g config core.autocrlf false

commit() {  # commit <message>
  g add -A
  g commit -q -m "$1"
}

# --- C1: the merge base -------------------------------------------------------
mkdir -p "$DEST/crates/aprender-core/src" "$DEST/docs"
printf 'baseline\n'            > "$DEST/README.md"
printf 'pub fn base() {}\n'    > "$DEST/crates/aprender-core/src/lib.rs"
commit "C1 baseline: the merge base every fixture PR forks from"

# --- C2, C3: main advances past the fork (this is the diff-scope pollution
#     that S2's merge-base boundary exists to keep out of the review) ----------
printf 'another agent landed this\n' > "$DEST/crates/aprender-core/src/other.rs"
commit "C2 an unrelated PR from another agent lands on main"

printf 'and another\n' > "$DEST/crates/aprender-core/src/third.rs"
commit "C3 a second unrelated PR lands on main"

g update-ref refs/remotes/origin/main refs/heads/main

# --- F1: the GPU pull request, forked from C1 ---------------------------------
g checkout -q -b gpu-pr "$(g rev-parse main~2)"
mkdir -p "$DEST/src/cuda"
cat > "$DEST/src/cuda/kernel.cu" <<'CU'
// fixture only - never compiled
__global__ void fixture_kernel(const float* in, float* out, int n) {}
CU
commit "F1 gpu: add a fused kernel and launch it on the compute stream"

# --- D1: the docs-only pull request, forked from the same base ----------------
g checkout -q -b docs-pr "$(g rev-parse main~2)"
printf 'A documentation-only change. No code, no surface, no device claim.\n' \
  > "$DEST/docs/note.md"
commit "D1 docs: record the fixture topology"

g checkout -q main

# --- prove the topology is the one the fixtures assume ------------------------
C1=$(g rev-parse main~2)
C3=$(g rev-parse main)
F1=$(g rev-parse gpu-pr)
D1=$(g rev-parse docs-pr)

[ "$(g merge-base refs/remotes/origin/main "$F1")" = "$C1" ] \
  || { echo "FIXTURE REPO BROKEN: merge-base(origin/main, F1) != C1" >&2; exit 1; }
[ "$(g merge-base refs/remotes/origin/main "$D1")" = "$C1" ] \
  || { echo "FIXTURE REPO BROKEN: merge-base(origin/main, D1) != C1" >&2; exit 1; }
g merge-base --is-ancestor "$C1" "$F1" \
  || { echo "FIXTURE REPO BROKEN: C1 is not an ancestor of F1" >&2; exit 1; }
if g merge-base --is-ancestor "$C3" "$F1"; then
  echo "FIXTURE REPO BROKEN: C3 must NOT be an ancestor of F1 (row 9 needs a stale index)" >&2
  exit 1
fi
g diff --name-only "$C1" "$F1" | grep -qx 'src/cuda/kernel.cu' \
  || { echo "FIXTURE REPO BROKEN: the gpu PR diff does not touch src/cuda/" >&2; exit 1; }

# --- assert the SHAs are the ones the committed receipts were written against --
ACTUAL=$(printf 'C1 %s\nC3 %s\nF1 %s\nD1 %s\n' "$C1" "$C3" "$F1" "$D1")
if [ "${PRREV_WRITE_EXPECTED_SHAS:-0}" = "1" ]; then
  printf '%s\n' "$ACTUAL" > "$HERE/expected-shas.txt"
  echo "wrote $HERE/expected-shas.txt" >&2
fi
EXPECTED_FILE="$HERE/expected-shas.txt"
if [ ! -f "$EXPECTED_FILE" ]; then
  echo "FIXTURE REPO BROKEN: $EXPECTED_FILE is missing; the SHAs of the fixtures are unanchored" >&2
  exit 1
fi
if ! diff -u "$EXPECTED_FILE" <(printf '%s\n' "$ACTUAL") >&2; then
  cat >&2 <<'DRIFT'
FIXTURE REPO BROKEN: the generated commit SHAs do not match expected-shas.txt.
The committed receipts under tests/fixtures/pr-review/row-*/ carry the expected SHAs
as literal values, so a drift here means the guard would be validating them against a
DIFFERENT repository than the one they describe - and rows 1, 9, 10 and 14 would be
testing nothing. Fix the topology, or regenerate deliberately with
PRREV_WRITE_EXPECTED_SHAS=1 and re-sign every affected fixture.
DRIFT
  exit 1
fi

echo "$DEST"
