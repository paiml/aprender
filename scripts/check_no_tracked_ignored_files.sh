#!/usr/bin/env bash
# check_no_tracked_ignored_files.sh — a file the repo declares ignored must not
# be tracked.
#
# WHY THIS EXISTS
# ---------------
# `.gitignore:62` says `**/.pmat-work/`. Git tracked 461 of those files anyway,
# totalling 315 MB — pmat per-ticket scratch (140 × ~5 MB `contract.json`),
# regenerable, read by nothing in the Makefile, the workflows, `scripts/`, any
# Rust source, or any pmat config. Classic add-before-ignore: the pattern was
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

# The subset a packaged crate compiles from: everything under src/ (Rust
# sources and the data they include_str!), the build script and its siblings,
# and any manifest, nested crates included. One of these that is tracked
# AND ignored ships from this checkout but vanishes from any tree rebuilt with
# `git add` (a `git archive` re-import, a vendored copy): the crate then fails
# its own package verify with E0583 (#5093, three crates at once). No baseline:
# the count of these is zero and must stay zero.
packaged_sources() {
  grep -E '^((crates/[^/]+/)?(src/.+|build[a-z_]*\.rs)|(.*/)?Cargo\.toml)$' || true
}

# The hard row. It runs before the vacuity check: it can only fail a run,
# never pass one, so an empty scan cannot slip through it.
hard_row() {
  local src
  src="$(printf '%s\n' "$1" | packaged_sources)"
  [ -n "$src" ] || return 0
  printf '\nFAIL: a packaged crate source is tracked AND ignored. A tree rebuilt\n'
  printf 'with `git add` drops it and the crate fails its package verify (E0583).\n'
  printf 'Delete the ignore rule that claims it:\n\n'
  printf '%s\n' "$src" | sed 's|^|  |'
  return 1
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

  # Row 3: the #5093 shape. A crate-local .gitignore claims a tracked source
  # file; the packaged-source filter must name it.
  mkdir -p "$TD/crates/c/src"
  printf 'mod g;\n' > "$TD/crates/c/src/lib.rs"
  printf '\n' > "$TD/crates/c/src/g.rs"
  printf 'src/g.rs\n' > "$TD/crates/c/.gitignore"
  git -C "$TD" add -f crates 2>/dev/null
  git -C "$TD" commit -qm z 2>/dev/null
  got="$(tracked_ignored "$TD" | packaged_sources)"
  if [ "$got" = "crates/c/src/g.rs" ]; then
    printf 'ok    row 3 tracked-but-ignored crate source reported as packaged\n'
  else
    printf 'FAIL  row 3 got [%s], expected [crates/c/src/g.rs]\n' "$got"; fails=1
  fi

  # Row 4: scratch that is tracked and ignored is NOT a packaged source, so
  # the hard row stays quiet on what the ratchet alone governs.
  mkdir -p "$TD/crates/c/.pmat-metrics"
  printf '{}\n' > "$TD/crates/c/.pmat-metrics/m.json"
  printf 'src/g.rs\n.pmat-metrics/\n' > "$TD/crates/c/.gitignore"
  git -C "$TD" rm --cached -q crates/c/src/g.rs 2>/dev/null
  git -C "$TD" add -f crates/c/.gitignore crates/c/.pmat-metrics/m.json 2>/dev/null
  git -C "$TD" commit -qm w 2>/dev/null
  got="$(tracked_ignored "$TD" | packaged_sources)"
  if [ -z "$got" ] && [ -n "$(tracked_ignored "$TD")" ]; then
    printf 'ok    row 4 tracked-ignored non-source not reported as packaged\n'
  else
    printf 'FAIL  row 4 got [%s], expected empty\n' "$got"; fails=1
  fi

  # Row 5: data a source includes (src/, not .rs) and a nested crate's
  # manifest are packaged too.
  got="$(printf 'crates/c/src/t/fixture.json\ncrates/c/sub/Cargo.toml\nCargo.toml\ncrates/c/benches/x.json\n' | packaged_sources)"
  want="$(printf 'crates/c/src/t/fixture.json\ncrates/c/sub/Cargo.toml\nCargo.toml')"
  if [ "$got" = "$want" ]; then
    printf 'ok    row 5 src/ data and nested manifests are packaged; benches/ data is not\n'
  else
    printf 'FAIL  row 5 got [%s]\n' "$got"; fails=1
  fi

  # Rows 6 and 7 run the MAIN path, not the helpers: a copy of this script
  # inside the temp repo, so REPO_ROOT is that repo. Row 6 is the #5093 shape
  # and must exit 1 AT the hard row (naming the file, never reaching vacuity, so
  # `hard_row ... || true` cannot pass on the temp repo's vacuity rc); row 7 is
  # the control without it and must not print the hard row (it still fails
  # later, on vacuity). A neutered hard row passes rows 1-5 and fails row 6.
  mkdir -p "$TD/scripts"
  SELF="${REPO_ROOT}/scripts/check_no_tracked_ignored_files.sh"
  cat -- "$SELF" > "$TD/scripts/check_no_tracked_ignored_files.sh" \
    || { printf 'FAIL: could not copy %s into the temp repo\n' "$SELF"; exit 1; }
  git -C "$TD" add -f crates/c/src/g.rs 2>/dev/null
  git -C "$TD" commit -qm v 2>/dev/null
  out="$(bash "$TD/scripts/check_no_tracked_ignored_files.sh" 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && grep -q '^  crates/c/src/g.rs$' <<<"$out" \
     && ! grep -q 'FAIL (vacuity)' <<<"$out"; then
    printf 'ok    row 6 main path exits 1 at the hard row, naming crates/c/src/g.rs\n'
  else
    printf 'FAIL  row 6 rc=%s, expected 1 at the hard row naming crates/c/src/g.rs, not at vacuity\n' "$rc"; fails=1
  fi
  git -C "$TD" rm --cached -q crates/c/src/g.rs 2>/dev/null
  git -C "$TD" commit -qm u 2>/dev/null
  out="$(bash "$TD/scripts/check_no_tracked_ignored_files.sh" 2>&1)"; rc=$?
  if ! grep -q 'packaged crate source' <<<"$out"; then
    printf 'ok    row 7 control: no packaged source, no hard-row FAIL (rc=%s, vacuity)\n' "$rc"
  else
    printf 'FAIL  row 7 hard row fired without a packaged source\n'; fails=1
  fi

  [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
  printf '\nSELF-TEST PASSED (7/7)\n'
  exit 0
fi

printf '=== no tracked file may be declared ignored (check_no_tracked_ignored_files.sh) ===\n'

FOUND="$(tracked_ignored "$REPO_ROOT")"
count="$(printf '%s\n' "$FOUND" | grep -c . || true)"
total_tracked="$(git -C "${REPO_ROOT}" ls-files | grep -c . || true)"

# Hard row, before the ratchet and outside it: a packaged source that is
# tracked AND ignored fails at any count (#5093). Self-test rows 6 and 7.
hard_row "$FOUND" || exit 1

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
baseline="$(grep -vE '^[[:space:]]*(#|$)' "$BASELINE" | tr -d '[:space:]')"

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
