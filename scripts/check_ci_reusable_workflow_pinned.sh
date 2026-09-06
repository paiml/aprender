#!/usr/bin/env bash
# check_ci_reusable_workflow_pinned.sh — the `uses:` reference to the
# paiml/.github sovereign-ci.yml reusable workflow in .github/workflows/ci.yml
# is pinned to a full 40-hex-char commit sha, never a branch or tag ref.
#
# WHY THIS EXISTS (PMAT-976, C0-2, #2891, epic #2873)
# ----------------------------------------------------
# `uses: paiml/.github/.github/workflows/sovereign-ci.yml@main` re-resolves on
# every push: whatever paiml/.github merges next runs on this repo's very next
# CI run, with no diff in THIS repository to review. A commit sha is
# immutable — the callee's content at pin time is exactly what re-runs until
# someone deliberately re-pins it, and re-pinning shows up as a reviewable
# one-line diff.
#
# WHAT THIS DOES NOT DO (read before wiring more onto it)
# ---------------------------------------------------------
# Pinning by sha does NOT make `pmat comply check`'s CB-2100 (Comply Gate
# Effect) reachability rule pass. Verified directly against pmat 3.39.0's
# source (services/gate_effect/resolve.rs::local_reusable_path): an external
# `uses:` reference — `owner/repo/...@<anything>`, branch, tag, or sha alike —
# is always `Resolution::Opaque`, never `Resolution::Job`. Only a workflow
# referenced as `./...` (local to this repository) is readable. Confirmed
# empirically too: pinning ci.yml's `uses:` to a real, content-identical sha
# left CB-2100's output byte-for-byte the same (still "whose steps this
# repository cannot read"). Closing CB-2100 needs a required, LOCAL job that
# actually invokes `pmat comply check`/`comply status` unsuppressed — see the
# comment above `uses:` in ci.yml and #2891 for why that isn't wired here yet
# (it would red every PR on six untracked pre-existing failures).
#
# Text-only: reads .github/workflows/ci.yml, builds nothing.
#
#   bash scripts/check_ci_reusable_workflow_pinned.sh
#   bash scripts/check_ci_reusable_workflow_pinned.sh --self-test
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW_REL=".github/workflows/ci.yml"
CALLEE_REPO="paiml/.github"
CALLEE_PATH=".github/workflows/sovereign-ci.yml"

# ref_is_pinned_sha <ref> -> 0 when ref is a full 40-char lowercase-hex commit
# sha, 1 otherwise (a branch name, a tag, a short sha, or garbage).
ref_is_pinned_sha() {
  ref="$1"
  [ "${#ref}" -eq 40 ] && case "$ref" in
    *[!0-9a-f]*) return 1 ;;
    *) return 0 ;;
  esac
  return 1
}

# find_uses_ref <file> -> prints the ref after `@` on the line whose `uses:`
# names $CALLEE_REPO/$CALLEE_PATH, or nothing (and a non-zero return) when no
# such line exists in the file.
find_uses_ref() {
  file="$1"
  [ -r "$file" ] || return 2
  line="$(grep -E "^\s*uses:\s*${CALLEE_REPO}/${CALLEE_PATH}@" "$file" | head -n1)" || true
  [ -n "$line" ] || return 1
  printf '%s\n' "${line##*@}" | tr -d ' \t\r'
}

# ---------------------------------------------------------------------------
# --self-test: both polarities of ref_is_pinned_sha, plus find_uses_ref
# against synthetic workflow files — including the exact mutation this guard
# exists to catch (re-pointing at @main).
# ---------------------------------------------------------------------------
self_test() {
  printf '=== case table: check_ci_reusable_workflow_pinned.sh ===\n'
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:?}"' RETURN

  fails=0
  assert() {  # assert <label> <expected_rc> <actual_rc>
    if [ "$2" -eq "$3" ]; then
      printf '  ok   %-46s rc=%s\n' "$1" "$3"
    else
      printf '  FAIL %-46s expected rc=%s got rc=%s\n' "$1" "$2" "$3"
      fails=$((fails + 1))
    fi
  }

  # ref_is_pinned_sha: MUST PASS (rc=0) — a real 40-char lowercase hex sha.
  ref_is_pinned_sha "4453399ee3794714800ff8db316ea7e1d3705a00"; assert 'ref: 40-char lowercase hex sha'   0 $?
  zeros="$(printf '0%.0s' $(seq 1 40))"
  ref_is_pinned_sha "$zeros"; assert 'ref: all-zero but valid 40-hex sha' 0 $?

  # ref_is_pinned_sha: MUST FAIL (rc=1) — the mutation this guard exists for,
  # plus every other shape a `uses:` ref can take.
  ref_is_pinned_sha "main";                                    assert 'ref: @main (the registered mutation)' 1 $?
  ref_is_pinned_sha "master";                                  assert 'ref: @master'                          1 $?
  ref_is_pinned_sha "v1.2.3";                                  assert 'ref: a tag'                             1 $?
  ref_is_pinned_sha "4453399";                                 assert 'ref: a short (abbreviated) sha'         1 $?
  ref_is_pinned_sha "4453399ee3794714800ff8db316ea7e1d3705a0G"; assert 'ref: 40 chars but non-hex tail'        1 $?
  ref_is_pinned_sha "";                                        assert 'ref: empty'                             1 $?

  # find_uses_ref: MUST FIND (rc=0), extracting exactly the ref after `@`.
  cat > "$tmp/pinned.yml" <<'EOF'
jobs:
  ci:
    uses: paiml/.github/.github/workflows/sovereign-ci.yml@4453399ee3794714800ff8db316ea7e1d3705a00
EOF
  got="$(find_uses_ref "$tmp/pinned.yml")"; rc=$?
  assert 'find_uses_ref: locates the pinned line' 0 "$rc"
  [ "$got" = "4453399ee3794714800ff8db316ea7e1d3705a00" ] && assert 'find_uses_ref: extracts the exact sha' 0 0 || assert 'find_uses_ref: extracts the exact sha' 0 1

  # find_uses_ref: MUST FIND but return the mutant ref (the check on top of it fails, not this parser).
  cat > "$tmp/unpinned.yml" <<'EOF'
jobs:
  ci:
    uses: paiml/.github/.github/workflows/sovereign-ci.yml@main
EOF
  got="$(find_uses_ref "$tmp/unpinned.yml")"; rc=$?
  assert 'find_uses_ref: locates the @main line too' 0 "$rc"
  [ "$got" = "main" ] && assert 'find_uses_ref: extracts "main" verbatim' 0 0 || assert 'find_uses_ref: extracts "main" verbatim' 0 1

  # find_uses_ref: MUST NOT FIND (rc=1) — no such uses: line, or an unrelated caller.
  cat > "$tmp/other-caller.yml" <<'EOF'
jobs:
  pr-gate:
    uses: paiml/.github/.github/workflows/pr-gate.yml@main
EOF
  find_uses_ref "$tmp/other-caller.yml" > /dev/null 2>&1; assert 'find_uses_ref: unrelated callee is not a match' 1 $?

  # find_uses_ref: MUST FAIL CLOSED (rc=2) — a missing file certifies nothing.
  find_uses_ref "$tmp/does-not-exist.yml" > /dev/null 2>&1; assert 'find_uses_ref: missing file is not clean'    2 $?

  if [ "$fails" -gt 0 ]; then
    printf '\nFAIL: %s case(s) failed. The guard does not do what it claims.\n' "$fails"
    return 1
  fi
  printf 'PASS: all cases behave as declared.\n'
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

printf '=== sovereign-ci.yml is pinned by commit sha in %s (check_ci_reusable_workflow_pinned.sh) ===\n' "$WORKFLOW_REL"

ref="$(find_uses_ref "$REPO_ROOT/$WORKFLOW_REL")"
rc=$?
if [ "$rc" -eq 2 ]; then
  printf 'FAIL: %s is unreadable -- a check that read nothing certifies nothing.\n' "$WORKFLOW_REL"
  exit 1
fi
if [ "$rc" -eq 1 ]; then
  printf 'FAIL: no `uses: %s/%s@...` line found in %s.\n' "$CALLEE_REPO" "$CALLEE_PATH" "$WORKFLOW_REL"
  exit 1
fi

if ! ref_is_pinned_sha "$ref"; then
  printf 'FAIL: %s references %s/%s@%s -- not a full 40-char commit sha.\n' "$WORKFLOW_REL" "$CALLEE_REPO" "$CALLEE_PATH" "$ref"
  printf 'A branch or tag ref re-resolves on every push; only a sha is immutable and reviewable.\n'
  exit 1
fi

printf 'PASS: %s references %s/%s@%s (a pinned commit sha).\n' "$WORKFLOW_REL" "$CALLEE_REPO" "$CALLEE_PATH" "$ref"
exit 0
