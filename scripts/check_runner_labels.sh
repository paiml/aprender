#!/usr/bin/env bash
# Poka-yoke: every inline self-hosted CI job MUST pin a discriminating runner
# label, so a job can never silently land on the wrong self-hosted runner.
#
# Root cause it guards (aprender#2269): GitHub auto-assigns self-hosted/Linux/X64
# as DEFAULT labels that cannot be removed, so a GPU dev runner (lambda-4090)
# matched the bare `[self-hosted, X64, Linux]` selector and the mutants job ran
# there — a box with no sovereign-ci registry — failing non-deterministically.
#
# Rule: any `runs-on:` that names `self-hosted` must ALSO name one of:
#   - clean-room  (the provisioned sovereign-ci pool: registry + cached image)
#   - a GPU label: cuda | gpu | rtx4090 | ada | blackwell | gb10
#   - a macOS label: apple-silicon | m4
# Reusable-workflow jobs (`uses:`) have no `runs-on` and are naturally exempt.
# GitHub-hosted jobs (ubuntu-latest, …) don't name self-hosted and are exempt.
set -euo pipefail
cd "$(dirname "$0")/.."

# A label DISCRIMINATES only if it names ONE box (or one provisioned pool). That is a
# property of the fleet, not of the word, so it changes when the fleet does.
#
# 2026-09-09: `gpu` and `cuda` were REMOVED from this list. They were admitted when gx10
# was the only GPU runner, so naming either did pick a single box. yoga-gpu is now
# rack-mounted and permanent and carries BOTH — measured:
#   gx10-blackwell  [self-hosted,Linux,ARM64,gpu,gx10,cuda,blackwell,gb10]
#   yoga-gpu        [self-hosted,Linux,X64,gpu,cuda,yoga,ada]
# so `[self-hosted, Linux, gpu, cuda]` reaches either one and this guard PASSED it. That
# is the #2269 collision again with a new pair of boxes: a selector that looks pinned,
# lands anywhere. Host and chip labels are what discriminate now.
#
# `perf-solo` is here because intel-clean-room-16 carries it INSTEAD of `clean-room`; a
# job pinning it would otherwise be failed by this guard for naming a real, single host.
DISCRIM='clean-room|perf-solo|gx10|yoga|rtx4090|ada|blackwell|gb10|apple-silicon|m4'

# --- case table (house rule: a guard regex ships one; five were wrong before it did) ---
# Re-run with: bash scripts/check_runner_labels.sh --self-test
self_test() {
  local rc=0
  # NOTE: written with `if`, not `&&`/`||`. Under `set -e` a must_fail whose grep does
  # not match returns non-zero from the `&&` chain and aborts the function mid-table —
  # the table then reports nothing and looks like it ran. Measured here, 2026-09-09.
  must_pass() { if grep -qE "$DISCRIM" <<< "$1"; then :; else echo "SELF-TEST FAIL (should pass): $1" >&2; rc=1; fi; }
  must_fail() { if grep -qE "$DISCRIM" <<< "$1"; then echo "SELF-TEST FAIL (should fail): $1" >&2; rc=1; fi; }
  # every selector in the tree today
  must_pass '[self-hosted, clean-room, intel]'
  must_pass '[self-hosted, gpu, gx10, cuda, blackwell]'
  must_pass '[self-hosted, gpu, Linux, ARM64, cuda, blackwell]'
  must_pass '[self-hosted, Linux, X64, clean-room]'
  must_pass '[self-hosted, X64, Linux, clean-room]'
  # a yoga-pinned CUDA unit-test job, the reason yoga is racked
  must_pass '[self-hosted, gpu, Linux, X64, cuda, yoga]'
  must_pass '[self-hosted, X64, Linux, perf-solo]'
  # THE REGRESSION THIS CHANGE EXISTS FOR: reaches gx10 AND yoga
  must_fail '[self-hosted, Linux, gpu, cuda]'
  must_fail '[self-hosted, gpu]'
  must_fail '[self-hosted, cuda]'
  # the original #2269 shape
  must_fail '[self-hosted, X64, Linux]'
  must_fail '[self-hosted]'
  [ "$rc" -eq 0 ] && echo "✓ check_runner_labels --self-test: 12 rows, both polarities"
  return "$rc"
}
[ "${1:-}" = "--self-test" ] && { self_test; exit $?; }
fail=0

while IFS=: read -r file line sel; do
  # Only inline self-hosted selectors.
  grep -q 'self-hosted' <<< "$sel" || continue
  if ! grep -qE "$DISCRIM" <<< "$sel" ; then
    echo "::error file=${file},line=${line}::self-hosted job lacks a discriminating runner label (need clean-room or a GPU/macOS label): ${sel# }"
    fail=1
  fi
done < <(grep -rHnE '^[[:space:]]*runs-on:.*self-hosted' .github/workflows/*.yml 2>/dev/null)
# -H: with exactly ONE matching workflow file grep omits the filename, so `IFS=:` reads the
# line number into $file and the annotation points at nothing. Harmless in this repo today
# (many workflows) and wrong the moment a fixture has one — which is how it was found.

# --- PASS 2: runs-on behind an expression (the hole PASS 1 cannot see) -------
#
# `runs-on: ${{ matrix.runner }}` carries no labels on its line, so PASS 1's
# `grep self-hosted` skips the job entirely and it is neither checked nor reported.
# Today nightly.yml:63 is the only one and every value in its matrix is
# GitHub-hosted -- but that is a fact about the current file, not a guarantee, and
# adding one self-hosted entry to that matrix would route a build to a GPU box with
# nothing failing. So: resolve the referenced matrix key, and check every value it
# can take. An expression we cannot resolve FAILS -- unresolvable is not proven safe.
while IFS=: read -r file line _rest; do
  [ -n "$file" ] || continue
  key=$(sed -n "${line}p" "$file" | grep -oE '\$\{\{[[:space:]]*matrix\.[A-Za-z0-9_-]+' | grep -oE '[A-Za-z0-9_-]+$' || true)
  if [ -z "$key" ]; then
    echo "::error file=${file},line=${line}::runs-on is an expression this guard cannot resolve; it may select a self-hosted GPU box and nothing would fail. Use a literal selector."
    fail=1; continue
  fi
  vals=$(grep -oE "^[[:space:]]*(- )?${key}:[[:space:]]*.*" "$file" | sed -E "s/^[[:space:]]*(- )?${key}:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//" | grep -v '^$' || true)
  if [ -z "$vals" ]; then
    echo "::error file=${file},line=${line}::runs-on references matrix.${key} but no ${key}: values were found; cannot prove it avoids the GPU runners."
    fail=1; continue
  fi
  while IFS= read -r v; do
    grep -q 'self-hosted' <<< "$v" || continue          # GitHub-hosted value: fine
    if ! grep -qE "$DISCRIM" <<< "$v"; then
      echo "::error file=${file},line=${line}::matrix.${key} value '${v}' is self-hosted without a discriminating label; it can land on gx10 or yoga."
      fail=1
    fi
  done <<< "$vals"
done < <(grep -rHnE '^[[:space:]]*runs-on:[[:space:]]*\$\{\{' .github/workflows/*.yml 2>/dev/null)

if [ "$fail" -ne 0 ]; then
  echo "FAIL: pin each self-hosted job to a provisioned pool (clean-room) or a GPU/macOS label." >&2
  echo "      Bare [self-hosted, X64, Linux] can land on ANY self-hosted runner (incl. GPU dev boxes)." >&2
  echo "      And [self-hosted, gpu, cuda] is NOT pinned: gx10 and yoga both carry those. Name the host." >&2
  exit 1
fi
echo "✓ check_runner_labels: every self-hosted job pins a discriminating runner label"
