#!/usr/bin/env bash
# check_ci_failfast.sh -- a RED GATE section in a merge_group run must cancel its run (operator 2026-10-19, refs #5394, #5738).
#  * x86-main: the "Fail fast" step runs on `!cancelled() && merge_group`, asks scripts/ci/failfast_gate.sh and only then
#    runs `gh run cancel`; a non-gate red (vendored-schemas) must not cancel.
#  * workspace-test-shard: the "Fail fast" step is `failure() && merge_group` -> `gh run cancel`, and the job holds `actions: write`.
#  * failfast_gate.sh's GATE_SECTIONS equals the `X86:<section>` pairs of the gate job's loop in ci.yml.
#  * failfast_gate.sh, step_budget_check.sh --self-test and every ci/step-budgets.tsv row (3 fields).
# Comment lines never count. --self-test mutates the workflow and demands RED for each mutant.
#   bash scripts/check_ci_failfast.sh [workflow] | --self-test
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

# step_lines <wf> <job>: the non-comment lines of that job's "Fail fast" step (up to the next `- name:`)
step_lines() {
  awk -v job="$2" '
    $0 ~ "^  " job ":" { on = 1; next }
    on && /^  [a-z0-9-]+:/ { on = 0; st = 0 }
    on && /^      - name: Fail fast/ { st = 1; print; next }
    on && st && /^      - name:/ { st = 0 }
    on && st && $0 !~ /^[[:space:]]*#/ { print }' "$1"
}

check() {
  local wf=$1 rc=0 s
  s=$(step_lines "$wf" x86-main)
  printf '%s\n' "$s" | grep -qE '^        if: \$\{\{ !cancelled\(\) && github\.event_name == .merge_group. \}\}$' \
    || { echo "FAIL: x86-main Fail-fast step lacks the exact 'if: !cancelled() && merge_group'"; rc=1; }
  printf '%s\n' "$s" | grep -qE 'if bash scripts/ci/failfast_gate\.sh "\$RUNNER_TEMP/fat/results\.json"; then$' \
    || { echo "FAIL: x86-main Fail-fast step does not ask failfast_gate.sh (a non-gate red would cancel)"; rc=1; }
  printf '%s\n' "$s" | grep -qE '^ +gh run cancel "\$GITHUB_RUN_ID"' \
    || { echo "FAIL: x86-main Fail-fast step has no live gh run cancel"; rc=1; }
  s=$(step_lines "$wf" workspace-test-shard)
  printf '%s\n' "$s" | grep -qE '^        if: \$\{\{ failure\(\) && github\.event_name == .merge_group. \}\}$' \
    || { echo "FAIL: shard Fail-fast step lacks the exact 'if: failure() && merge_group'"; rc=1; }
  printf '%s\n' "$s" | grep -qE 'run: gh run cancel "\$GITHUB_RUN_ID"' \
    || { echo "FAIL: shard Fail-fast step has no live gh run cancel"; rc=1; }
  awk '/^  workspace-test-shard:/{on=1;next} on&&/^  [a-z0-9-]+:/{on=0} on&&/^      actions: write/{f=1} END{exit f?0:1}' "$wf" \
    || { echo "FAIL: workspace-test-shard lacks a live 'actions: write' (gh run cancel would 403)"; rc=1; }
  local gate list
  gate=$(awk '/^  gate:/{on=1} on&&/for pair in /{print; exit}' "$wf" | grep -oE 'X86:[a-z0-9.-]+' | sed 's/^X86://' | sort | tr '\n' ' ')
  list=$(sed -n 's/^GATE_SECTIONS="\(.*\)"$/\1/p' scripts/ci/failfast_gate.sh | tr ' ' '\n' | sort | tr '\n' ' ')
  { [ -n "$gate" ] && [ "$gate" = "$list" ]; } \
    || { echo "FAIL: failfast_gate.sh GATE_SECTIONS ($list) != the gate job's X86 sections ($gate)"; rc=1; }
  return $rc
}

case "${1:-}" in -h|--help) echo "usage: check_ci_failfast.sh [workflow] | --self-test"; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
  wf=.github/workflows/ci.yml
  check "$wf" || { echo "FAIL: the real workflow is not fail-fast"; exit 1; }
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  mutant() { # name sed-expr
    sed -E "$2" "$wf" > "$t/m.yml"
    if cmp -s "$wf" "$t/m.yml"; then echo "FAIL: mutant '$1' changed nothing (stale pattern)"; exit 1; fi
    check "$t/m.yml" > /dev/null && { echo "FAIL: mutant '$1' still passed"; exit 1; }
  }
  mutant "cancel removed" '/gh run cancel/d'
  mutant "cancel commented out" 's/^( +)gh run cancel/\1# gh run cancel/'
  mutant "x86 guard always false" "s/(if: \\$\\{\\{ !cancelled\\(\\) && github\\.event_name == 'merge_group') \\}\\}/\\1 \&\& false }}/"
  mutant "shard guard always false" "s/(if: \\$\\{\\{ failure\\(\\) && github\\.event_name == 'merge_group') \\}\\}/\\1 \&\& false }}/"
  mutant "gate helper bypassed" 's/failfast_gate\.sh/true/'
  mutant "actions write commented" 's/^( +)actions: write/\1# actions: write/'
  mutant "gate loop gains a section" 's/X86:sov\.gate/X86:sov.gate X86:vendored-schemas/'
  bash scripts/ci/failfast_gate.sh --self-test || exit 1
  bash scripts/ci/step_budget_check.sh --self-test || exit 1
  awk -F'\t' '!/^#/ && NF != 3 { bad = 1 } END { exit bad }' ci/step-budgets.tsv || { echo "FAIL: ci/step-budgets.tsv row without 3 fields"; exit 1; }
  echo "ok: check_ci_failfast case table (7 mutants RED)"; exit 0
fi
check "${1:-.github/workflows/ci.yml}" && echo "ok: GATE failures cancel the merge_group run"
