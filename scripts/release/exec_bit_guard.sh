#!/usr/bin/env bash
# exec_bit_guard.sh -- fail when a workflow or script runs a tracked script DIRECTLY (command position,
# no interpreter) while git records it non-executable (mode != 100755).  PMAT-6729, aprender#6774.
#   bash scripts/release/exec_bit_guard.sh [ROOT]     # exit 0 = every direct call is executable
# Why: 100644 is the repo norm for scripts (callers say `bash x.sh`); one_command.sh called
# scripts/bump-version.sh bare and Release run 42234289410 died "Permission denied".
# Counts what it scanned and exits 2 on zero, so "0 violations over 0 calls" cannot pass.
# Command positions: `&&` `||` `;` `|` `$(` and a ${VAR:-default} value, in workflows and shell scripts;
# plus a bare line start / `run:` in workflows only (in a .sh a line starting `scripts/x.sh` is a case
# pattern or data row, not a call -- known blind spot: a bare line-start call inside a .sh).
set -uo pipefail
ROOT="${1:-$(git rev-parse --show-toplevel)}" || exit 2
cd "$ROOT" || exit 2
P='(\./)?scripts/[A-Za-z0-9_./-]+\.sh'
COMMON='&&[[:space:]]+|[|][|][[:space:]]+|;[[:space:]]*|[|][[:space:]]+|[$][(][[:space:]]*|:-'
CMD_WF="(^[[:space:]]*(-[[:space:]]+)?(run:[[:space:]]+)?|$COMMON)"
CMD_SH="($COMMON)"
files=$(git ls-files '.github/workflows/*.yml' '.github/workflows/*.yaml' 'scripts/*.sh' 'scripts/**/*.sh') || exit 2
calls=0; bad=0
while IFS= read -r f; do
  [ -f "$f" ] || continue
  case "$f" in .github/*) cmd=$CMD_WF ;; *) cmd=$CMD_SH ;; esac
  while IFS= read -r hit; do
    path=${hit#./}; calls=$((calls+1))
    mode=$(git ls-files -s -- "$path" | cut -d' ' -f1)
    [ -n "$mode" ] || continue
    [ "$mode" = 100755 ] || { echo "FAIL $f invokes $path directly but git mode is $mode (use 'bash $path' or chmod +x)"; bad=$((bad+1)); }
  done < <(grep -vE '^[[:space:]]*#' "$f" | grep -oE "${cmd}${P}" | grep -oE "$P" | sort -u)
done <<< "$files"
[ "$calls" -gt 0 ] || { echo "STOP scanned 0 direct calls: guard is blind"; exit 2; }
echo "exec_bit_guard: $calls direct call(s) scanned, $bad non-executable"
[ "$bad" = 0 ]
