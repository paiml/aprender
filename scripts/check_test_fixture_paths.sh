#!/usr/bin/env bash
#
# check_test_fixture_paths.sh - a test must not gate itself on a path outside
# the workspace, because such a gate silently disarms the test.
#
# WHY THIS EXISTS
# ---------------
# The monorepo consolidated 20 sibling repos in-tree. Tests written before the
# merge gate on the OLD sibling checkouts:
#
#   let has_q4k = file_exists("/home/noah/src/realizar/src/quantize.rs");
#   if !has_q4k { eprintln!("SKIP - realizar not found"); return; }
#
# realizar is now crates/aprender-serve. That path cannot exist again on any
# machine, so every test behind such a gate is permanently, silently green.
# Five such paths remain, across falsification_2x_ollama_tests.rs and
# falsification_correctness_tests.rs, and one test even runs a command with
# `.current_dir("/home/noah/src/realizar")`. None of them reads the file it
# probes -- the path is used only to decide whether to skip.
#
# The gates are also invisible in CI: workspace-test runs `--lib`, so these
# integration targets are never even compiled. A skip nobody sees, in a test
# nobody runs, asserting a claim somebody trusts.
#
# This is a RATCHET, not a cleanup. Repointing the existing gates makes dormant
# tests execute for the first time and will surface real failures; that belongs
# in its own change. What this guard does is stop the population growing, and
# make the remaining debt a number that can only go down.
#
#   bash scripts/check_test_fixture_paths.sh              # check
#   bash scripts/check_test_fixture_paths.sh --self-test  # case table
#   bash scripts/check_test_fixture_paths.sh --update     # re-baseline (shrink only)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/test_fixture_path_baseline.txt"

# An absolute path into a developer home or a sibling checkout. Workspace-relative
# paths, /tmp, and env-var-derived paths are all fine -- those cannot silently
# disarm a test on someone else's machine.
PATTERN='"(/home/[^"]*|/Users/[^"]*)"'

scan() {
  local dir="$1"
  # Only test code. src/ may legitimately mention such a path in a doc comment.
  find "$dir" -type f -name '*.rs' 2>/dev/null \
    | grep -E '/tests?/' \
    | while IFS= read -r f; do
        grep -HnoE "$PATTERN" "$f" 2>/dev/null
      done
}

count_hits() { scan "$1" | wc -l | tr -d ' '; }

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
  TD="$(mktemp -d)"; [ -d "$TD" ] || { printf 'FAIL: no temp dir\n' >&2; exit 1; }
  trap 'rm -rf "${TD:?}"' EXIT
  mkdir -p "$TD/crates/x/tests"

  # Fixtures live in scripts/lib/ rather than inline heredocs: bashrs parses an
  # embedded heredoc as shell, so Rust `let x = ...;` lines read as SC1068
  # "spaces around = in let assignments". Six phantom errors, same class as the
  # embedded-awk false positives that moved assertions_exclude.awk out of line.
  cp "${REPO_ROOT}/scripts/lib/fixture_path_selftest_bad.rs.txt"  "$TD/crates/x/tests/bad.rs"
  cp "${REPO_ROOT}/scripts/lib/fixture_path_selftest_good.rs.txt" "$TD/crates/x/tests/good.rs"

  bad="$(scan "$TD" | grep -c 'bad.rs' || true)"
  good="$(scan "$TD" | grep -c 'good.rs' || true)"
  fails=0
  if [ "$bad" -eq 3 ]; then printf 'ok    3/3 defect shapes flagged\n'
  else printf 'FAIL  flagged %s of 3 defect shapes - guard is blind\n' "$bad"; fails=1; fi
  if [ "$good" -eq 0 ]; then printf 'ok    0 false positives on legitimate paths\n'
  else printf 'FAIL  %s false positive(s) on legitimate paths\n' "$good"; fails=1; fi

  [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
  printf '\nSELF-TEST PASSED\n'; exit 0
fi

printf '=== tests must not gate on paths outside the workspace (check_test_fixture_paths.sh) ===\n'

hits="$(scan "${REPO_ROOT}/crates")"
count="$(printf '%s' "$hits" | grep -c . || true)"

# Vacuity: a scan that examined nothing must not report clean.
scanned="$(find "${REPO_ROOT}/crates" -type f -name '*.rs' 2>/dev/null | grep -cE '/tests?/' || true)"
if [ "$scanned" -lt 200 ]; then
  printf '\nFAIL (vacuity): scanned only %s test file(s). Fix the scan, not this number.\n' "$scanned"
  exit 1
fi

if [ "${1:-}" = "--update" ]; then
  printf '%s\n' "$count" > "$BASELINE"
  printf 'baseline set to %s (from %s test files)\n' "$count" "$scanned"
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
#     "--update # re-baseline (shrink only)"
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "${REPO_ROOT}" scripts/test_fixture_path_baseline.txt count || exit 1

if [ ! -f "$BASELINE" ]; then
  printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE"
  exit 1
fi
baseline="$(tr -d '[:space:]' < "$BASELINE")"

printf 'scanned %s test file(s); %s out-of-workspace path(s), baseline %s\n' \
  "$scanned" "$count" "$baseline"

if [ "$count" -gt "$baseline" ]; then
  printf '\nFAIL: out-of-workspace fixture paths grew %s -> %s.\n' "$baseline" "$count"
  printf 'A test gated on a path outside the workspace is green on every machine\n'
  printf 'that lacks it, which is every machine but one. Use a workspace-relative\n'
  printf 'path, an env var with an explicit failure when unset, or a real fixture.\n\n'
  printf '%s\n' "$hits" | sed 's|^|  |'
  exit 1
fi

if [ "$count" -lt "$baseline" ]; then
  printf '\nImproved: %s -> %s. Run --update to record it.\n' "$baseline" "$count"
fi

printf 'PASS\n'
exit 0
