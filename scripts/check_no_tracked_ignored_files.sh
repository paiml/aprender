#!/usr/bin/env bash
# check_no_tracked_ignored_files.sh — a file the repo declares ignored must not
# be tracked.
#
# WHY THIS EXISTS
# ---------------
# `.gitignore:62` says `**/.pmat-work/`. Git tracked 461 of those files anyway,
# totalling 315 MB — the analyser's per-ticket scratch (140 × ~5 MB `contract.json`),
# regenerable, read by nothing in the Makefile, the workflows, `scripts/`, any
# Rust source, or any analyser config. Classic add-before-ignore: the pattern was
# written after the files were staged, and `git rm --cached` was never run. Every
# clone paid the checkout cost.
#
# This is the same shape as every other defect this repo has been closing:
# .gitignore DECLARES, the index CONTRADICTS, and nothing compared them.
#
# RATCHET, NOT A CLEANUP
# ----------------------
# 320 ignored-but-tracked files remain and they are NOT all removable:
#
#   * `proptest-regressions/*.txt` are matched by `.gitignore:22` and MUST stay
#     tracked — each records a failing proptest seed so the regression is
#     re-tested forever. There the DECLARATION is wrong, not the tracking.
#   * `.pmat-metrics/`, `benchmark-results/`, and Lean run logs are scratch and
#     should go, but each needs its own check that nothing reads it.
#
# So this ratchets: the count may only fall. Deciding each remaining group is a
# separate change, and the ratchet stops new ones arriving meanwhile.
#
#   bash scripts/check_no_tracked_ignored_files.sh              # check
#   bash scripts/check_no_tracked_ignored_files.sh --self-test  # case table
#   bash scripts/check_no_tracked_ignored_files.sh --update     # re-baseline

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/tracked_ignored_baseline.txt"

# Files that git tracks AND the repo's own ignore rules match.
tracked_ignored() {
  git -C "$1" ls-files -i -c --exclude-standard
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
  TD="$(mktemp -d)"; [ -d "$TD" ] || { printf 'FAIL: no temp dir\n' >&2; exit 1; }
  trap 'rm -rf "${TD:?}"' EXIT
  fails=0

  git -C "$TD" init -q 2>/dev/null
  git -C "$TD" config user.email t@t
  git -C "$TD" config user.name t
  mkdir -p "$TD/scratch" "$TD/src"
  printf 'scratch/\n' > "$TD/.gitignore"
  printf 'keep\n' > "$TD/src/real.rs"
  printf 'junk\n' > "$TD/scratch/junk.json"

  # Row 1: force-add an ignored file, then it must be reported.
  git -C "$TD" add -f .gitignore src/real.rs scratch/junk.json 2>/dev/null
  git -C "$TD" commit -qm x 2>/dev/null
  got="$(tracked_ignored "$TD")"
  if [ "$got" = "scratch/junk.json" ]; then
    printf 'ok    row 1 tracked-but-ignored file reported\n'
  else
    printf 'FAIL  row 1 got [%s], expected [scratch/junk.json]\n' "$got"; fails=1
  fi

  # Row 2: untrack it and the report must go empty. This is the control that
  # proves row 1 was not reporting every tracked file.
  git -C "$TD" rm --cached -q scratch/junk.json 2>/dev/null
  git -C "$TD" commit -qm y 2>/dev/null
  if [ -z "$(tracked_ignored "$TD")" ]; then
    printf 'ok    row 2 untracking clears the report (src/real.rs not flagged)\n'
  else
    printf 'FAIL  row 2 still reports: %s\n' "$(tracked_ignored "$TD")"; fails=1
  fi

  [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
  printf '\nSELF-TEST PASSED (2/2)\n'
  exit 0
fi

printf '=== no tracked file may be declared ignored (check_no_tracked_ignored_files.sh) ===\n'

FOUND="$(tracked_ignored "$REPO_ROOT")"
count="$(printf '%s\n' "$FOUND" | grep -c . || true)"
total_tracked="$(git -C "${REPO_ROOT}" ls-files | grep -c . || true)"

# Vacuity: `ls-files` returning nothing would make any count trivially zero.
if [ "$total_tracked" -lt 1000 ]; then
  printf '\nFAIL (vacuity): only %s tracked file(s) seen; git is not reporting the repo.\n' "$total_tracked"
  exit 1
fi

if [ "${1:-}" = "--update" ]; then
  printf '%s\n' "$count" > "$BASELINE"
  printf 'baseline set to %s (of %s tracked files)\n' "$count" "$total_tracked"
  exit 0
fi

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything above compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that is not a ratchet. NEW (a finding with no entry) and
# STALE (an entry with no finding) are the only two properties a working tree
# can answer, and a commit that appends one line AND lands the matching
# violation satisfies both at once: not new, because it is baselined; not
# stale, because the finding is real.
#
# Measured, not argued: appending one entry cloned from this file's own last
# real entry returned rc=0 from this guard, under its own words:
#     "the count may only fall"
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "${REPO_ROOT}" scripts/tracked_ignored_baseline.txt count || exit 1

if [ ! -f "$BASELINE" ]; then
  printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE"
  exit 1
fi
baseline="$(tr -d '[:space:]' < "$BASELINE")"

printf '%s tracked-but-ignored file(s), baseline %s (of %s tracked)\n' \
  "$count" "$baseline" "$total_tracked"

if [ "$count" -gt "$baseline" ]; then
  printf '\nFAIL: tracked-but-ignored files grew %s -> %s.\n' "$baseline" "$count"
  printf 'A file the repo declares ignored is in the index. Either untrack it\n'
  printf '(git rm --cached) or fix the ignore rule that wrongly claims it.\n\n'
  printf '%s\n' "$FOUND" | head -40 | sed 's|^|  |'
  exit 1
fi

if [ "$count" -lt "$baseline" ]; then
  printf '\nImproved: %s -> %s. Run --update to record it.\n' "$baseline" "$count"
fi

printf 'PASS (ratcheted)\n'
exit 0
