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
#       +---- F1                  <- gpu-pr head    (adds src/cuda/kernel.cu)
#       \
#       +---- D1                  <- docs-pr head   (adds docs/note.md)
#       \
#       +---- G1                  <- claim-pr head  (adds book/src/tools/apr-cli.md,
#       \                            which PUBLISHES a competitor ratio)
#       +---- S1                  <- code-pr head   (adds a plain .rs file)
#       \
#        +--- P1                  <- printed-pr head (adds one .rs file carrying the SAME
#        \                           ratio twice: in a plain // comment, and inside a
#         \                          format! a user reads)
#         +-- E1                  <- examples-pr head (publishes the SAME ratio under
#                                    book/src/examples/, the 34.7% of the book B4 could
#                                    not see until F6)
#
# G1 AND S1 EXIST BECAUSE TWO BLOCKING RULES CANNOT BE EXERCISED WITHOUT THEM.
# B4's diff half needs a head that publishes `2.93x Ollama` on a surface a user reads;
# without it the rule can only be tested against a receipt, which is the circularity it
# was added to remove. S3.D's trigger needs a head touching Rust source; F1 adds a .cu
# and D1 adds markdown, so neither triggers mutation and `not-triggered` was true on
# every head the fixtures had.
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
# The '..' check also covers every mkdir this script issues below: they all
# derive from this same validated $DEST, so a caller-controlled '../' segment
# is refused once, here, rather than trusted at each of the nine call sites.
if [ -z "$DEST" ] || [ "$DEST" = "/" ]; then
  echo "refusing to build a fixture repo at $DEST" >&2
  exit 1
fi
case "$DEST" in
  *..*)
    echo "refusing a destination containing '..': $DEST" >&2
    exit 1
    ;;
esac
if [ -n "$DEST" ] && [ "$DEST" != "/" ]; then
  rm -rf -- "$DEST"
fi
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

# --- G1: the pull request that PUBLISHES a competitor ratio --------------------
# The B4 subject. `book/` is where a user reads, which is where the 2.93x Ollama claim
# was actually published from a harness that never ran Ollama.
g checkout -q -b claim-pr "$(g rev-parse main~2)"
mkdir -p "$DEST/book/src/tools"
cat > "$DEST/book/src/tools/apr-cli.md" <<'MD'
# apr bench

apr sustains 2.93× Ollama on 1.5B Q4_K decode.
MD
commit "G1 book: publish the decode ratio"

# --- S1: an ordinary Rust change ----------------------------------------------
# Deliberately carries NO clap attribute, NO route registration and NO ratio, so the
# only trigger it fires is S3.D's. A head that fired three triggers at once could not
# show which rule a fixture pins.
g checkout -q -b code-pr "$(g rev-parse main~2)"
cat > "$DEST/crates/aprender-core/src/fused.rs" <<'RS'
pub fn fused(a: f32, b: f32) -> f32 {
    a * b
}
RS
commit "S1 core: add a fused helper"

# --- P1: the same ratio, published and merely quoted --------------------------
# The discrimination case for B4's .rs scope. Measured over 300 commits of origin/main,
# every comparative claim this repository adds to a plain `//` comment is a claim it is
# WITHDRAWING - "// #2696: this printed \"Performance: 800+ tok/s (2.8x Ollama)\"". A gate
# that blocks those has no honest exit, because S3.C.1's remedy is a comparator log and
# there is no log for a number nobody measured. So the comment must not fire and the
# format! must, and one head carries both so a fixture can tell them apart.
g checkout -q -b printed-pr "$(g rev-parse main~2)"
cat > "$DEST/crates/aprender-core/src/banner.rs" <<'RS'
// The book published 2.93x Ollama from a harness that never ran Ollama. This line
// QUOTES the claim in order to name it; a plain comment is not a surface a user reads.
pub fn banner() -> String {
    format!("apr sustains 2.93x Ollama on 1.5B Q4_K decode")
}
RS
commit "P1 core: add the banner helper"

# --- E1: the ratio published where B4 could not see it ------------------------
# F6's fixture head. `match_shipped_surface` excluded */examples/* -- a Rust cargo-target
# rule -- and the one directory that removed from the book was book/src/examples/: 153 of
# 441 published pages, 34.7%, every one listed in book/src/SUMMARY.md. da069a25f published
# `851.8 tok/s = 2.93x Ollama` into exactly that directory, and B4 fired ZERO times on it.
#
# This head is G1's diff moved one directory over, so rows 25/26 differ from rows 16/17 in
# the PATH and nothing else. Without it the case table would go green on a guard that
# still cannot see a third of the book -- the guard-universe defect, seventh instance.
g checkout -q -b examples-pr "$(g rev-parse main~2)"
mkdir -p "$DEST/book/src/examples"
cat > "$DEST/book/src/examples/showcase-benchmark.md" <<'MD'
# Case Study: Showcase Benchmark

- **GGUF GPU**: 851.8 tok/s = **2.93x Ollama** (291 tok/s baseline)
MD
commit "E1 book: publish the showcase decode ratio under examples/"

# ===========================================================================
# S13 (autonomous merge on quorum) needs five heads the S6.3 table never needed.
# Every one of them is a REFUSAL PATH of scripts/pr_review_quorum_arm.sh, and a
# refusal path with no head that exercises it is a gate that cannot fire.
#
# They are added AFTER E1 and forked from the same C1, so not one byte of C1..E1
# changes: git addresses a commit by its content and its parents, and a new child
# perturbs neither. The twenty-six committed row-* receipts keep their signatures.
# ===========================================================================

# --- M1: the pull request that edits the merge MECHANISM ----------------------
# S13.8: a PR that changes the machinery deciding whether it may merge does not get
# to apply the new machinery to itself. Without this head, Q9's mechanism clause can
# only be tested against a receipt, which is the circularity S13.2 exists to remove.
g checkout -q -b mech-pr "$(g rev-parse main~2)"
mkdir -p "$DEST/scripts"
printf '#!/usr/bin/env bash\n# a guard the PR is editing\nexit 0\n' \
  > "$DEST/scripts/check_pr_review_receipt.sh"
commit "M1 guard: change the receipt guard itself"

# --- H1: a guard-shaped file that is NOT a mechanism path ---------------------
# S13.2 clause (4) owes a GUARD-scoped mutation run on a guard-touching diff (S7 B3).
# M1 cannot exercise it: Q9 refuses M1 before the mutation clause is ever reached, so
# a fixture built on M1 would report a kill the clause never earned. H1 touches
# scripts/check_no_claim_literals.sh, which is guard-shaped and is not on
# MECHANISM_PATHS -- one variable different, opposite clause.
g checkout -q -b guard-pr "$(g rev-parse main~2)"
mkdir -p "$DEST/scripts"
printf '#!/usr/bin/env bash\n# an ordinary guard, not part of the merge mechanism\nexit 0\n' \
  > "$DEST/scripts/check_no_claim_literals.sh"
commit "H1 guard: harden the claim-literal guard"

# --- T1, T2: the two shapes of tip drift, forked from F1 ----------------------
# S13.3.a constrains the CONTENT of the commits between the reviewed SHA and the tip,
# not their count: they may touch evidence/pr-review/<pr>/** and nothing else. The two
# heads differ in exactly one file and nothing else, so the rule is shown to
# DISCRIMINATE rather than merely to refuse.
g checkout -q -b tipclean-pr gpu-pr
mkdir -p "$DEST/evidence/pr-review/2783/notes"
printf 'the receipt of the reviewed commit, committed after it\n' \
  > "$DEST/evidence/pr-review/2783/notes/receipt-note.txt"
commit "T1 evidence: record the review of F1 (the commit that necessarily follows it)"

g checkout -q -b tipdirty-pr gpu-pr
printf 'pub fn slipped_in_after_the_review() {}\n' \
  > "$DEST/crates/aprender-core/src/late.rs"
commit "T2 core: a change nobody reviewed, riding in behind the receipt"

# --- K1: an origin/main carrying the autonomy KILL SWITCH ---------------------
# Not on refs/heads/main and not on refs/remotes/origin/main: pointing the remote ref
# at this head is what a test does, so the DEFAULT topology stays the one the
# twenty-six row receipts were signed against. Forked from main's tip, so
# merge-base(K1, F1) is still C1 and every existing receipt stays valid under it.
g checkout -q -b ksmain-pr main
mkdir -p "$DEST/.github"
printf 'autonomy disabled by the operator, 2026-08-31\n' \
  > "$DEST/.github/pr-review-autonomy.disabled"
commit "K1 ops: disable autonomous merge"

g checkout -q main

# --- prove the topology is the one the fixtures assume ------------------------
C1=$(g rev-parse main~2)
C3=$(g rev-parse main)
F1=$(g rev-parse gpu-pr)
D1=$(g rev-parse docs-pr)
G1=$(g rev-parse claim-pr)
S1=$(g rev-parse code-pr)
P1=$(g rev-parse printed-pr)
E1=$(g rev-parse examples-pr)
M1=$(g rev-parse mech-pr)
H1=$(g rev-parse guard-pr)
T1=$(g rev-parse tipclean-pr)
T2=$(g rev-parse tipdirty-pr)
K1=$(g rev-parse ksmain-pr)

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

# The two new heads are asserted the same way: a fixture head that stopped carrying
# its subject would leave the rules it pins passing over nothing.
g diff "$C1" "$G1" -- book | grep -q 'Ollama' \
  || { echo "FIXTURE REPO BROKEN: the claim PR diff publishes no competitor ratio" >&2; exit 1; }
g diff --name-only "$C1" "$S1" | grep -qx 'crates/aprender-core/src/fused.rs' \
  || { echo "FIXTURE REPO BROKEN: the code PR diff touches no Rust source" >&2; exit 1; }
g diff "$C1" "$P1" | grep -q 'format!("apr sustains' \
  || { echo "FIXTURE REPO BROKEN: the printed PR publishes no ratio" >&2; exit 1; }
g diff "$C1" "$P1" | grep -q '^+// The book published' \
  || { echo "FIXTURE REPO BROKEN: the printed PR carries no merely-quoted ratio, so nothing distinguishes the two" >&2; exit 1; }
g diff --name-only "$C1" "$E1" | grep -qx 'book/src/examples/showcase-benchmark.md' \
  || { echo "FIXTURE REPO BROKEN: the examples PR does not publish under book/src/examples/" >&2; exit 1; }
g diff "$C1" "$E1" -- book | grep -q 'Ollama' \
  || { echo "FIXTURE REPO BROKEN: the examples PR diff publishes no competitor ratio" >&2; exit 1; }
if g diff --name-only "$C1" "$D1" | grep -qE '\.rs$'; then
  echo "FIXTURE REPO BROKEN: the docs PR must touch no Rust source, or S3.D triggers on it" >&2
  exit 1
fi

# The five S13 heads are asserted the same way, and for the same reason: a head that
# stopped carrying its subject leaves the refusal it pins passing over nothing.
g diff --name-only "$C1" "$M1" | grep -qx 'scripts/check_pr_review_receipt.sh' \
  || { echo "FIXTURE REPO BROKEN: the mechanism PR touches no MECHANISM_PATHS entry" >&2; exit 1; }
g diff --name-only "$C1" "$H1" | grep -qx 'scripts/check_no_claim_literals.sh' \
  || { echo "FIXTURE REPO BROKEN: the guard PR touches no guard-shaped file" >&2; exit 1; }
if g diff --name-only "$C1" "$H1" | grep -qx 'scripts/check_pr_review_receipt.sh'; then
  echo "FIXTURE REPO BROKEN: the guard PR must NOT touch a mechanism path, or Q9 refuses it before the mutation clause is reached" >&2
  exit 1
fi
g merge-base --is-ancestor "$F1" "$T1" \
  || { echo "FIXTURE REPO BROKEN: T1 must be a descendant of F1 (S13.3.a needs a reviewed ancestor)" >&2; exit 1; }
g merge-base --is-ancestor "$F1" "$T2" \
  || { echo "FIXTURE REPO BROKEN: T2 must be a descendant of F1" >&2; exit 1; }
if g diff --name-only "$F1" "$T1" | grep -qvx 'evidence/pr-review/2783/notes/receipt-note.txt'; then
  echo "FIXTURE REPO BROKEN: the clean-tip PR touches something outside evidence/pr-review/2783/, so it cannot show the rule ADMITTING the honest path" >&2
  exit 1
fi
g diff --name-only "$F1" "$T2" | grep -qx 'crates/aprender-core/src/late.rs' \
  || { echo "FIXTURE REPO BROKEN: the dirty-tip PR carries no unreviewed source change" >&2; exit 1; }
g cat-file -e "$K1:.github/pr-review-autonomy.disabled" \
  || { echo "FIXTURE REPO BROKEN: the kill-switch head does not carry the kill switch" >&2; exit 1; }
if g cat-file -e "refs/remotes/origin/main:.github/pr-review-autonomy.disabled" 2>/dev/null; then
  echo "FIXTURE REPO BROKEN: the DEFAULT origin/main must NOT carry the kill switch, or every positive fixture refuses Q8" >&2
  exit 1
fi
[ "$(g merge-base "$K1" "$F1")" = "$C1" ] \
  || { echo "FIXTURE REPO BROKEN: merge-base(K1, F1) != C1, so pointing origin/main at K1 would invalidate every committed receipt" >&2; exit 1; }

# --- assert the SHAs are the ones the committed receipts were written against --
ACTUAL=$(printf 'C1 %s\nC3 %s\nF1 %s\nD1 %s\nG1 %s\nS1 %s\nP1 %s\nE1 %s\nM1 %s\nH1 %s\nT1 %s\nT2 %s\nK1 %s\n' \
  "$C1" "$C3" "$F1" "$D1" "$G1" "$S1" "$P1" "$E1" "$M1" "$H1" "$T1" "$T2" "$K1")
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
