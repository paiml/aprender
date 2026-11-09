#!/bin/sh
# no-pull-request-target-guard.sh -- fail if a workflow file triggers on pull_request_target.
#
# pull_request_target runs with a write token and secrets in the BASE repo context, on PRs from
# forks: the "pwn request" hole. paiml accepts no fork contributions (NO-FORK-1), so every
# PR-triggered workflow uses `pull_request` and guards on a same-repo head.
#
# The trigger is read with a real YAML parser (yq), never a regex: `.on` may be a string, a
# list or a map; anchors, aliases, multi-document files, BOMs, CRLF, quoted and line-folded
# scalars and \u \U \x escapes are all resolved by the parser, not matched against text.
# Every *.yml / *.yaml under the directory counts: hidden files, subdirectories, symlinks.
# A file that does not parse fails (a trigger that cannot be read cannot be cleared).
#
# Allowlist (.github/pull-request-target-allowlist.txt), one line per exception:
#     <path relative to the workflow dir>  sha256:<64 hex>   # reason
# The exception holds only while the file's sha256 matches, so ANY edit (a new job, a checkout,
# a local workflow_call) fails until a reviewer re-pins the hash.
#
# Usage: no-pull-request-target-guard.sh [WORKFLOW_DIR] [ALLOWLIST_FILE]
# Exit: 0 clean, 1 violation, 2 usage / no yq / no workflow scanned (0 files is not a pass).
set -eu

dir="${1:-.github/workflows}"
allow="${2:-.github/pull-request-target-allowlist.txt}"

[ -d "$dir" ] || { echo "guard: no such directory: $dir" >&2; exit 2; }
command -v yq >/dev/null 2>&1 || { echo "guard: yq not found; cannot read workflow triggers" >&2; exit 2; }

# Event names of one document: `.on` (or YAML 1.1's boolean `true` key) as string, list or map.
EXPR='explode(.) | select(kind == "map") | .[] | select((key | downcase) == "on" or key == "true") | [ (select(kind == "scalar") | .), (select(kind == "seq") | .[] | select(kind == "scalar") | .), (select(kind == "seq") | .[] | select(kind == "map") | keys | .[]), (select(kind == "map") | keys | .[]) ] | .[] | downcase'

list=$(mktemp)
[ -n "$list" ] && [ -f "$list" ] || exit 2
trap 'rm -f "${list:?}"' EXIT
find "$dir" \( -type f -o -type l \) \( -name '*.yml' -o -name '*.yaml' \) -print | sort > "$list"

scanned=0
bad=0
while IFS= read -r f; do
  scanned=$((scanned + 1))
  if [ ! -f "$f" ]; then
    echo "FAIL $f: dangling symlink or not a regular file" >&2
    bad=$((bad + 1))
    continue
  fi
  if ! events=$(yq e "$EXPR" "$f" 2>/dev/null); then
    echo "FAIL $f: does not parse as YAML, so its trigger cannot be read" >&2
    bad=$((bad + 1))
    continue
  fi
  printf '%s\n' "$events" | grep -qx 'pull_request_target' || continue
  rel=${f#"$dir"/}
  want=""
  if [ -f "$allow" ]; then
    want=$(sed -e 's/[[:space:]]*#.*$//' "$allow" | awk -v p="$rel" '$1 == p && $2 ~ /^sha256:[0-9a-f]+$/ { sub(/^sha256:/, "", $2); print $2; exit }')
  fi
  if [ -z "$want" ]; then
    echo "FAIL $f: triggers on pull_request_target and is not in the allowlist with a sha256 pin" >&2
    bad=$((bad + 1))
    continue
  fi
  if [ -L "$f" ]; then
    echo "FAIL $f: an allowlisted workflow must be a regular file, not a symlink" >&2
    bad=$((bad + 1))
    continue
  fi
  have=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then
    echo "FAIL $f: allowlisted pull_request_target workflow changed (sha256 $have, pinned $want); re-review, then re-pin" >&2
    bad=$((bad + 1))
  fi
done < "$list"

echo "no-pull-request-target-guard: scanned $scanned workflow file(s), $bad violation(s)"
[ "$scanned" -gt 0 ] || exit 2
[ "$bad" -eq 0 ]
