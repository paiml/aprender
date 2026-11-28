#!/usr/bin/env bash
# release_prev_tag_test.sh — the nightly reconcile window picks the SAME previous tag as release.yml (aprender R2, 2026-11-28).
#
# Fixture repo: v0.1.0 on main (an ancestor of the target), v0.2.0 on a side branch (NOT an ancestor of main), v0.3.0 on main.
#   A sort-only pick names v0.2.0 and check_reconcile.sh refuses it (rc 2). The ancestor-filtered pick must name v0.1.0.
#   Cases:
#     1. scripts/release/prev_final_tag.sh 0.3.0 v0.3.0 prints 0.1.0 (never the non-ancestor 0.2.0)
#     2. no ancestor tag below the version: exit 2, never an empty name
#     3. NEGATIVE: check_reconcile.sh v0.3.0 --prev-tag v0.2.0 (non-ancestor) refuses with exit 2 and says so;
#        the gate is not lowered: a window that is not a real range never passes
set -uo pipefail

here="$(cd "$(dirname "$0")/../.." && pwd)"
helper="${here}/scripts/release/prev_final_tag.sh"
check="${here}/scripts/check_reconcile.sh"
tmp="$(mktemp -d)" || exit 2
trap 'rm -rf "${tmp:?}"' EXIT
fails=0
cases=0

pass() { cases=$((cases + 1)); }
fail() { cases=$((cases + 1)); printf 'FAIL %s\n' "$1" >&2; fails=$((fails + 1)); }

repo="${tmp}/repo"
mkdir -p "$repo" && cd "$repo" || exit 2
git init -q -b main
git config user.email "test@example.invalid"
git config user.name "release-prev-tag-test"
git commit -q --allow-empty -m "base"
git tag v0.1.0
git checkout -q -b side
git commit -q --allow-empty -m "side release line"
git tag v0.2.0
git checkout -q main
git commit -q --allow-empty -m "main work"
git tag v0.3.0

# 1. ancestor-filtered prev tag
got="$(bash "$helper" 0.3.0 v0.3.0 2>/dev/null)" || got="<rc=$?>"
if [ "$got" = "0.1.0" ]; then pass; else fail "prev-tag: want 0.1.0 (ancestor of v0.3.0), got '${got}'"; fi

# 2. nothing below 0.1.0 is an ancestor: exit 2 with no name printed
rc=0; out="$(bash "$helper" 0.1.0 v0.1.0 2>/dev/null)" || rc=$?
if [ "$rc" -eq 2 ] && [ -z "$out" ]; then pass; else fail "no-ancestor: want exit 2 and empty output, got rc=${rc} out='${out}'"; fi

# 3. negative control: the sort-only pick (v0.2.0, not an ancestor) is refused, never reconciled
rc=0; err="$(bash "$check" v0.3.0 --prev-tag v0.2.0 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$err" | grep -q "is not an ancestor"; then pass
else fail "negative: want non-zero refusal naming the ancestor, got rc=${rc} err='${err}'"; fi

printf 'release_prev_tag_test: %s case(s), %s failure(s)\n' "$cases" "$fails"
[ "$fails" -eq 0 ]
