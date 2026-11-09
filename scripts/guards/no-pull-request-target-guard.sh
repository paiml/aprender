#!/bin/sh
# no-pull-request-target-guard.sh -- fail if a workflow uses the pull_request_target trigger.
#
# pull_request_target runs with a write token and secrets in the BASE repo context, on
# PRs from forks: the "pwn request" hole. paiml accepts no fork contributions (NO-FORK-1),
# so every PR-triggered workflow uses `pull_request` and guards on a same-repo head.
#
# Usage: no-pull-request-target-guard.sh [WORKFLOW_DIR] [ALLOWLIST_FILE]
#   Allowlist lines: `<workflow file basename>  # reason`. An allowlisted file must not
#   contain a checkout step (the fork-pr-closer case: metadata only, never PR code).
# Exit: 0 clean, 1 violation, 2 usage or no workflows scanned (0 files is not a pass).
set -eu

dir="${1:-.github/workflows}"
allow="${2:-.github/pull-request-target-allowlist.txt}"

[ -d "$dir" ] || { echo "guard: no such directory: $dir" >&2; exit 2; }

scanned=0
bad=0
for f in "$dir"/*.yml "$dir"/*.yaml; do
  [ -f "$f" ] || continue
  scanned=$((scanned + 1))
  # strip WHOLE-LINE comments only: a " #" inside a quoted string must not hide a later match
  hits=$(grep -v "^[[:space:]]*#" "$f" | grep -c "pull_request_target" || true)
  # YAML double-quoted scalars decode \uXXXX and \xXX, so a name can be spelled without the literal text
  esc=$(grep -v "^[[:space:]]*#" "$f" | grep -cE '\\[ux][0-9A-Fa-f]' || true)
  if [ "$esc" -gt 0 ]; then
    echo "FAIL $f: \\u or \\x escape in a workflow file (can spell the trigger without its name)" >&2
    bad=$((bad + 1))
  fi
  [ "$hits" -gt 0 ] || continue
  base=$(basename "$f")
  if [ -f "$allow" ] && sed -e 's/[[:space:]]*#.*$//' -e 's/[[:space:]]*$//' "$allow" | grep -qx "$base"; then
    if grep -v '^[[:space:]]*#' "$f" | grep -q 'actions/checkout'; then
      echo "FAIL $f: allowlisted pull_request_target workflow contains actions/checkout" >&2
      bad=$((bad + 1))
    fi
    continue
  fi
  echo "FAIL $f: uses pull_request_target ($hits line(s)); use pull_request with a same-repo head guard" >&2
  bad=$((bad + 1))
done

echo "no-pull-request-target-guard: scanned $scanned workflow file(s), $bad violation(s)"
[ "$scanned" -gt 0 ] || exit 2
[ "$bad" -eq 0 ]
