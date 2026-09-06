#!/usr/bin/env bash
# check_roadmap_ids_unique.sh — every `id:` in docs/roadmaps/roadmap.yaml is
# unique, NESTED records included (PMAT-1072, #3028).
#
# WHY THIS EXISTS
# ---------------
# pmat 3.39.0's `work validate` (PMAT-674, pmat PR #1196) refuses a roadmap
# whose id appears twice — every `id:` line, subtask records included. The
# fleet's PATH pmat moved 3.37.0 -> 3.39.0 by hand on 2026-09-06 and the
# upstream `roadmap-valid` step (sovereign-ci.yml, bare `pmat` from PATH)
# turned aprender main RED on 12 legacy nested subtask records — stale copies
# of children under four parents — that the in-repo pin (G-10a, 3.37.0) can
# never see. The rule is right and the data was wrong; but a rule that lives
# only in whichever pmat happens to be on a runner's PATH is not a guard the
# repo owns. This one reads the PARSED tree (an id inside a block-scalar
# body is text, not an id), names every duplicate with its path and line,
# and is wired in ci.yml's guard-runner-labels — case table first.
#
#   bash scripts/check_roadmap_ids_unique.sh [<roadmap.yaml>]   # 0 all unique · 1 duplicate(s)/unparsable · 2 usage or env
#   bash scripts/check_roadmap_ids_unique.sh --self-test        # 6 rows, both polarities
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_roadmap_ids_unique
DEFAULT_ROADMAP="$ROOT/docs/roadmaps/roadmap.yaml"

usage() {
    printf 'usage: %s [ROADMAP.yaml]  or  %s --self-test\n' "$PROG" "$PROG" >&2
    exit 2
}

# scan FILE: prints PASS/FAIL lines; exit 0 unique, 1 duplicates or unparsable, 2 env
scan() {
    local f=$1
    if [ ! -f "$f" ]; then
        printf '%s: ENV - %s is missing (the box cannot answer)\n' "$PROG" "$f" >&2
        return 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        printf '%s: ENV - python3 is missing\n' "$PROG" >&2
        return 2
    fi
    python3 - "$f" <<'PY'
import re
import sys

import yaml

path = sys.argv[1]
try:
    doc = yaml.safe_load(open(path, encoding="utf-8"))
except yaml.YAMLError as e:
    print(f"FAIL  {path} does not parse as YAML: {e}")
    sys.exit(1)

# raw-text line index (reporting only; the verdict comes from the parsed tree)
lines_of = {}
for n, line in enumerate(open(path, encoding="utf-8"), 1):
    m = re.match(r"^\s*-?\s*id:\s*['\"]?([^'\"\s]+)['\"]?\s*$", line)
    if m:
        lines_of.setdefault(m.group(1), []).append(n)

seen = {}
def walk(node, where):
    if isinstance(node, dict):
        rid = node.get("id")
        if isinstance(rid, (str, int)):
            seen.setdefault(str(rid), []).append(where)
        for k, v in node.items():
            walk(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{where}[{i}]")

walk(doc, "$")
dups = {k: v for k, v in seen.items() if len(v) > 1}
if dups:
    for rid, wheres in sorted(dups.items()):
        lns = lines_of.get(rid, [])
        spots = ", ".join(f"{w} (line {lns[i]})" if i < len(lns) else w for i, w in enumerate(wheres))
        print(f"FAIL  duplicate id {rid}: {spots}")
    print(f"FAIL  {len(dups)} duplicate id(s) in {path} — every id must be unique, nested records included (pmat >= 3.39.0 work validate refuses this file)")
    sys.exit(1)
print(f"PASS  {len(seen)} ids in {path}, all unique (nested records included)")
PY
}

# ---------------------------------------------------------------------------
# --self-test: fixtures under mktemp -d, both polarities
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/rmids-selftest.XXXXXX")
    # SEC011: validate the scratch path before rm -rf (non-empty, not /, carries the mktemp tag)
    cleanup() {
        local victim=${TD:-}
        case "$victim" in
            *rmids-selftest.*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
            *) return 0 ;;
        esac
    }
    trap cleanup EXIT
    n=0; red=0
    # row WANT_RC LABEL MUST_MATCH FILE
    row() {
        local want=$1 label=$2 pat=$3 f=$4 rc=0
        n=$((n + 1))
        bash "$0" "$f" >"$TD/out.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ] && grep -qE -- "$pat" "$TD/out.$n"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"; sed 's/^/        /' "$TD/out.$n"; red=1; fi
    }
    cat >"$TD/clean.yaml" <<'YAML'
roadmap:
- id: A-1
  title: first
  subtasks: []
- id: A-2
  title: second
YAML
    cat >"$TD/nested.yaml" <<'YAML'
roadmap:
- id: A-1
  title: first
  subtasks:
  - id: A-2
    title: A-2
    status: planned
- id: A-2
  title: second
YAML
    cat >"$TD/toplevel.yaml" <<'YAML'
roadmap:
- id: A-1
  title: first
- id: A-1
  title: first again
YAML
    cat >"$TD/scalar.yaml" <<'YAML'
roadmap:
- id: A-1
  title: first
  spec: |
    a block scalar that mentions
    - id: A-2
    is prose, not an id
- id: A-2
  title: second
YAML
    printf 'roadmap:\n- id: A-1\n  title: [unclosed\n' >"$TD/broken.yaml"
    row 0 "two unique ids, no nested records"                          'PASS  2 ids'                 "$TD/clean.yaml"
    row 1 "a nested subtask record duplicating a top-level id (the #3028 shape)" 'duplicate id A-2: .*subtasks\[0\].*line 5' "$TD/nested.yaml"
    row 1 "two top-level entries with the same id"                     'duplicate id A-1'            "$TD/toplevel.yaml"
    row 0 "an id inside a block scalar is prose, not an id"            'PASS  2 ids'                 "$TD/scalar.yaml"
    row 1 "an unparsable roadmap is a defect, never a pass"            'does not parse'              "$TD/broken.yaml"
    row 2 "a missing file is ENV (exit 2), never a pass"               'ENV'                         "$TD/absent.yaml"
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
fi

case "${1:-}" in
    --help|-h|--*) usage ;;
    '') scan "$DEFAULT_ROADMAP" ;;
    *) scan "$1" ;;
esac
