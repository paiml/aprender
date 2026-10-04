#!/usr/bin/env bash
# workflow_path_filters_rule3_parity.sh — rule 3 of check_workflow_path_filters.sh moved from the .py (PyYAML) into
# scripts/lib/workflow_runs_nightly.awk (T44, #4686). This runs the OLD check (both files read from --base) and the
# NEW one (this tree) over the SAME workflows: fixture shapes, every workflow of this tree, every workflow at
# --base. The rule-3 verdicts must agree on every workflow except the ones EXPECTED below, and those must move
# one way only: RED under the old rule, green under the new, because they chain from "Nightly pick".
#
#   bash scripts/tests/workflow_path_filters_rule3_parity.sh [--base REF]    (default origin/main)
# Exit: 0 parity holds; 1 RED; 2 not_measured (no python3 with yaml: both sides dump filters with it); 3 caller error.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE=origin/main
case "${1:-}" in
    --base) BASE="${2:-}"; [ -n "$BASE" ] || { echo "caller error: --base REF" >&2; exit 3; } ;;
    '') ;;
    *) echo "caller error: unknown argument '$1' (--base REF)" >&2; exit 3 ;;
esac
# The difference the move is for, and nothing else.
EXPECTED="fx-chain fx-chain_sq head-book head-book-contracts head-install-script"

if ! python3 -c 'import yaml' > /dev/null 2>&1; then
    echo "not_measured: no python3 with yaml here; both sides of the check need it to dump the filters"; exit 2
fi
git -C "$ROOT" rev-parse --verify -q "$BASE^{commit}" > /dev/null || { echo "caller error: no commit $BASE" >&2; exit 3; }

T="$(mktemp -d "${TMPDIR:-/tmp}/r3-parity.XXXXXX")" || { echo "caller error: no temp dir" >&2; exit 3; }
trap 'rm -rf -- "${T:?}"' EXIT
for s in old new; do mkdir -p "$T/$s/scripts/lib" "$T/$s/.github/workflows"; done
git -C "$ROOT" show "$BASE:scripts/check_workflow_path_filters.sh" > "$T/old/scripts/check_workflow_path_filters.sh" || exit 3
git -C "$ROOT" show "$BASE:scripts/lib/workflow_path_filters.py" > "$T/old/scripts/lib/workflow_path_filters.py" || exit 3
cp "$ROOT/scripts/check_workflow_path_filters.sh" "$T/new/scripts/" || exit 3
cp "$ROOT/scripts/lib/workflow_path_filters.py" "$ROOT/scripts/lib/workflow_runs_nightly.awk" "$T/new/scripts/lib/" || exit 3

W="$T/wf"; mkdir -p "$W"
P='  push:\n    paths: ["book/**"]\n  pull_request:\n    paths: ["book/**"]\n'
fx() { printf 'on:\n%b%b' "$P" "$2" > "$W/fx-$1.yml"; } # NAME TRIGGERS
fx sched "  schedule:\n    - cron: '30 22 * * *'\n"
fx none ""
fx empty "  schedule: []\n"
fx chain '  workflow_run:\n    workflows: ["Nightly pick"]\n    types: [completed]\n    branches: [main]\n'
fx chain_sq "  workflow_run:\n    workflows: ['Nightly pick']\n    types: [ completed ]\n"
fx chain_other '  workflow_run:\n    workflows: ["CI"]\n    types: [completed]\n'
fx chain_two '  workflow_run:\n    workflows: ["Nightly pick", "CI"]\n    types: [completed]\n'
fx chain_req '  workflow_run:\n    workflows: ["Nightly pick"]\n    types: [requested]\n'
fx chain_cmt '  # workflow_run:\n  #   workflows: ["Nightly pick"]\n  #   types: [completed]\n'
fx cron_cmt "  schedule:\n    # - cron: '30 22 * * *'\n"
fx cron_empty "  schedule:\n    - cron:\n"
fx cron_qempty "  schedule:\n    - cron: ''\n"
fx cron_outside_on "  workflow_dispatch:\njobs:\n  a:\n    schedule:\n      - cron: '1 1 * * *'\n"
fx flow_list "  schedule: [{cron: '1 1 * * *'}]\n"
fx flow_item "  schedule:\n    - {cron: \"1 1 * * *\"}\n"
fx flow_empty "  schedule: [{cron: ''}]\n"
for f in "$ROOT"/.github/workflows/*.yml; do cp "$f" "$W/head-$(basename -- "$f")"; done
git -C "$ROOT" ls-tree --name-only "$BASE" .github/workflows/ | grep -e '\.yml$' | while IFS= read -r p; do
    git -C "$ROOT" show "$BASE:$p" > "$W/base-$(basename -- "$p")"
done
for s in old new; do cp "$W"/*.yml "$T/$s/.github/workflows/"; done

# rule-3 RED set per side: the workflows whose FAIL line is rule 3's
r3() { # SIDE PATTERN
    bash "$T/$1/scripts/check_workflow_path_filters.sh" > "$T/$1.out" 2>&1
    grep -e "^FAIL .*$2" "$T/$1.out" | sed -e 's/^FAIL \([^:]*\):.*/\1/' | sed -e 's/\.yml$//' | sort -u
}
r3 old 'no `schedule:` trigger' > "$T/old.r3"
r3 new 'no nightly trigger' > "$T/new.r3"
n="$(find "$W" -name '*.yml' | wc -l)"
grep -q -e '^scanned ' "$T/old.out" && grep -q -e '^scanned ' "$T/new.out" || { echo "RED: a side did not finish its scan"; tail -n 5 "$T/old.out" "$T/new.out"; exit 1; }

rc=0
only_old="$(comm -23 "$T/old.r3" "$T/new.r3")"
only_new="$(comm -13 "$T/old.r3" "$T/new.r3")"
want="$(printf '%s\n' $EXPECTED | sort -u)"
printf 'rule 3 over %s workflows: old RED %s, new RED %s\n' "$n" "$(wc -l < "$T/old.r3")" "$(wc -l < "$T/new.r3")"
if [ -n "$only_new" ]; then printf 'RED  new rule 3 is RED where the old was not:\n%s\n' "$only_new"; rc=1; fi
if [ "$only_old" != "$want" ]; then
    printf 'RED  old-RED/new-green set differs from EXPECTED\n  got:  %s\n  want: %s\n' "$(printf "%s\n" "$only_old" | tr "\n" " ")" "$(printf "%s\n" "$want" | tr "\n" " ")"; rc=1
fi
[ "$rc" -eq 0 ] && printf 'PASS parity: the verdicts agree on every workflow but %s, each old RED -> new green (the chain)\n' "$(printf "%s\n" "$want" | tr "\n" " ")"
exit "$rc"
