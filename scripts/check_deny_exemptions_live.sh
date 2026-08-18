#!/usr/bin/env bash
# check_deny_exemptions_live.sh — every deny.toml advisory exemption must
# correspond to an advisory that actually fires.
#
# WHY THIS EXISTS
# ---------------
# An exemption is a standing decision to accept a known vulnerability. It should
# expire when the vulnerability leaves the graph -- otherwise the list grows
# permissions for nothing, and a reviewer reading deny.toml cannot tell which
# entries are load-bearing.
#
# 20 of 29 exemptions were dead: the advisory no longer fired at all, because the
# dependency had been upgraded or removed. `RUSTSEC-2026-0002` exempted
# "lru 0.12.5 ... fixed in 0.16 but ratatui pins 0.12" while the lockfile already
# resolved lru 0.16.4 -- the fixed version. The rationale described a world that
# had moved on.
#
# `cargo deny` ALREADY reports this, as `advisory-not-detected` warnings. Nobody
# acted on them because they are warnings and the command exits 0. This turns
# that existing signal into a gate.
#
# Deliberately NOT part of the advisory check itself: a newly-fixed upstream must
# not fail anyone's build. It fails only the guard, whose fix is deleting a line.
#
#   bash scripts/check_deny_exemptions_live.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

printf '=== every deny.toml exemption must be live (check_deny_exemptions_live.sh) ===\n'

LOG="$(mktemp)"
trap 'rm -f "${LOG:?}"' EXIT

# Redirect, never pipe: reading this through a pipe would report the exit status
# of the last stage instead of cargo-deny's.
cargo deny check advisories > "$LOG" 2>&1
rc=$?

declared="$(grep -c 'id = "RUSTSEC' deny.toml || true)"
# Two steps, no line-continuation inside the command substitution: bashrs
# mis-parses nested quotes across a continued `$( ... )` and reports SC1078 on
# valid bash.
dead_block="$(grep -A 3 'advisory-not-detected' "$LOG" || true)"
dead_ids="$(printf '%s\n' "$dead_block" | grep -oE 'RUSTSEC-[0-9]{4}-[0-9]+' | sort -u)"
dead="$(printf '%s\n' "$dead_ids" | grep -c . || true)"

# Vacuity: a run that parsed no exemptions cannot certify anything.
if [ "$declared" -lt 1 ]; then
  printf '\nFAIL (vacuity): no RUSTSEC exemptions parsed from deny.toml.\n'
  printf 'Either the file moved or the pattern broke. Fix the scan, not this check.\n'
  exit 1
fi

printf '%s exemption(s) declared, %s no longer fire\n' "$declared" "$dead"

if [ "$dead" -gt 0 ]; then
  printf '\nFAIL: these exemptions grant permission for an advisory that no longer\n'
  printf 'appears in the dependency graph. Delete them from deny.toml:\n\n'
  printf '%s\n' "$dead_ids" | sed 's|^|  |'
  printf '\nA dead exemption hides which of the remaining entries are load-bearing,\n'
  printf 'and silently re-permits the advisory if the dependency ever returns.\n'
  exit 1
fi

if [ "$rc" -ne 0 ]; then
  printf '\nFAIL: `cargo deny check advisories` exited %s.\n' "$rc"
  tail -20 "$LOG" | sed 's|^|  |'
  exit "$rc"
fi

printf 'PASS: all %s exemption(s) correspond to a live advisory.\n' "$declared"
exit 0
