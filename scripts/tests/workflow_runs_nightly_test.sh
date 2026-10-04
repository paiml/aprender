#!/usr/bin/env bash
# workflow_runs_nightly_test.sh — case table for scripts/lib/workflow_runs_nightly.awk, rule 3 of
# check_workflow_path_filters.sh (T44, #4686). No python: the awk is judged on its own, and each mutant of it must
# turn a row RED. The old-vs-new parity run is scripts/tests/workflow_path_filters_rule3_parity.sh.
#
#   bash scripts/tests/workflow_runs_nightly_test.sh [--mutants]
# Exit: 0 every row as specified (and, with --mutants, every mutant killed); 1 RED; 3 caller error.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
AWK_SRC="$ROOT/scripts/lib/workflow_runs_nightly.awk"
[ -f "$AWK_SRC" ] || { echo "caller error: no $AWK_SRC" >&2; exit 3; }
P='  push:\n    paths: ["book/**"]\n  pull_request:\n    paths: ["book/**"]\n'

# NAME WANT(0 nightly, 1 not) TRIGGERS
ROWS="sched	0	  schedule:\n    - cron: '30 22 * * *'\n
flow_list	0	  schedule: [{cron: '1 1 * * *'}]\n
flow_item	0	  schedule:\n    - {cron: \"1 1 * * *\"}\n
chain	0	  workflow_run:\n    workflows: [\"Nightly pick\"]\n    types: [completed]\n    branches: [main]\n
chain_sq	0	  workflow_run:\n    workflows: ['Nightly pick']\n    types: [ completed ]\n
chain_two_types	0	  workflow_run:\n    workflows: [\"Nightly pick\"]\n    types: [requested, completed]\n
none	1
empty	1	  schedule: []\n
cron_cmt	1	  schedule:\n    # - cron: '30 22 * * *'\n
cron_empty	1	  schedule:\n    - cron:\n
cron_qempty	1	  schedule:\n    - cron: ''\n
flow_empty	1	  schedule: [{cron: ''}]\n
cron_outside_on	1	  workflow_dispatch:\njobs:\n  a:\n    schedule:\n      - cron: '1 1 * * *'\n
chain_other	1	  workflow_run:\n    workflows: [\"CI\"]\n    types: [completed]\n
chain_two	1	  workflow_run:\n    workflows: [\"Nightly pick\", \"CI\"]\n    types: [completed]\n
chain_req	1	  workflow_run:\n    workflows: [\"Nightly pick\"]\n    types: [requested]\n
chain_no_types	1	  workflow_run:\n    workflows: [\"Nightly pick\"]\n
chain_cmt	1	  # workflow_run:\n  #   workflows: [\"Nightly pick\"]\n  #   types: [completed]\n
chain_under_other_key	1	  workflow_dispatch:\n    workflows: [\"Nightly pick\"]\n    types: [completed]\n
comment_col0_in_on	0	# a note at column 0 inside on:\n  schedule:\n    - cron: '1 1 * * *'\n
schedule_after_on	1	  workflow_dispatch:\njobs:\n  schedule:\n    - cron: '1 1 * * *'\n
chain_nested_deeper	1	  workflow_run:\n    types: [completed]\n    x:\n      workflows: [\"Nightly pick\"]\n"

table() { # AWK-FILE -> prints rows, returns 1 when any is wrong
    local awkf="$1" t name want trig rc bad=0 n=0
    t="$(mktemp -d "${TMPDIR:-/tmp}/wrn.XXXXXX")" || exit 3
    while IFS="$(printf '\t')" read -r name want trig; do
        [ -n "$name" ] || continue
        n=$((n + 1))
        printf 'on:\n%b%b' "$P" "$trig" > "$t/$name.yml"
        awk -f "$awkf" "$t/$name.yml" > /dev/null 2>&1; rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    %s\n' "$name"; else printf 'RED   %-24s rc=%s want %s\n' "$name" "$rc" "$want"; bad=1; fi
    done <<< "$ROWS"
    rm -rf -- "${t:?}"
    [ "$bad" -eq 0 ] && printf 'TABLE PASSED: %s rows\n' "$n"
    return "$bad"
}

MUTANTS='m01_comments_live	s/^\/\^\[\[:space:\]\]\*(#|\$)\/ { next }$/\/^[[:space:]]*$\/ { next }/
m02_any_workflow	s/list1(\$0) == "Nightly pick"/1/
m03_any_type	s/ && \$0 ~ \/\[\[,\[:space:\]\]completed\[\],\[:space:\]\]\/ { t = 1 }/ { t = 1 }/
m04_chain_alone	s/exit !(cron || (w && t))/exit !(cron || w)/
m05_on_never_ends	/^on && ind(\$0) == 0 { on = 0 }$/d
m06_empty_cron	s/\["\\047\]?\[^\[:space:\],}"\\047#\]/./
m07_no_flow_list	/^                if (sc && hascron(\$0)) cron = 1$/d
m08_no_flow_item	s/- \[{\]?\[\[:space:\]\]\*cron:/- cron:/
m09_any_indent	s/wr && ind(\$0) == 4 && \/^ +workflows:/wr \&\& \/workflows:/'

mutants() {
    local t name expr killed=0 total=0 errors=0
    t="$(mktemp -d "${TMPDIR:-/tmp}/wrn-mu.XXXXXX")" || exit 3
    while IFS="$(printf '\t')" read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        sed -e "$expr" "$AWK_SRC" > "$t/m.awk"
        if cmp -s "$AWK_SRC" "$t/m.awk"; then printf 'ERROR %-24s the patch did not apply\n' "$name"; errors=$((errors + 1)); continue; fi
        if table "$t/m.awk" > "$t/out" 2>&1; then printf 'SURVIVED %s\n' "$name"
        else killed=$((killed + 1)); printf 'killed   %-24s %s\n' "$name" "$(grep -c -e '^RED ' "$t/out")"; fi
    done <<< "$MUTANTS"
    rm -rf -- "${t:?}"
    printf 'MUTANTS killed=%s total=%s errors=%s\n' "$killed" "$total" "$errors"
    [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ]
}

case "${1:-}" in
    '') table "$AWK_SRC" ;;
    --mutants) table "$AWK_SRC" > /dev/null || { echo "RED: the table fails unmutated" >&2; exit 1; }; mutants ;;
    *) echo "caller error: unknown argument '$1' (--mutants)" >&2; exit 3 ;;
esac
