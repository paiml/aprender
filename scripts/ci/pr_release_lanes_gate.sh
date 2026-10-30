#!/usr/bin/env bash
# pr_release_lanes_gate.sh -- run release_lanes.sh's own gates on a PR that touches it (ruling N38).
#
# WHY: the guards that judge scripts/release/*.sh (guard-cargo m16 bashrs, m28 mutants) skip on
# pull_request events, so #6529 (SEC001) failed only in merge_group. pr-light calls this script;
# it is a plain blocking step (C324 rule 6: no continue-on-error, no advisory state).
#
# WHAT: diff the PR against the base ref, then
#   scripts/release/release_lanes.sh changed -> release_lanes.sh --self-test and --mutants
#   any scripts/release/*.sh changed         -> scripts/check_bashrs_gate.sh --root REPO (the release's bashrs verdict, whole repo)
#   neither                                  -> print "not triggered" and exit 0
#
# Usage: pr_release_lanes_gate.sh [BASE_REF]   (default origin/main; uses BASE...HEAD)
#        pr_release_lanes_gate.sh --plan       (reads changed paths on stdin, prints the plan)
#        pr_release_lanes_gate.sh --self-test
# Exit: 0 green or not triggered, 1 a gate is red, 2 environment (bashrs absent or vacuous, git failed).
set -euo pipefail

PROG=pr_release_lanes_gate.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# plan: changed paths on stdin -> lines "lanes" and/or "bashrs" (any scripts/release/*.sh: the gate is whole-repo)
plan() {
  local p lanes=0 sh=0
  while IFS= read -r p; do
    case "$p" in
      scripts/release/release_lanes.sh) lanes=1; sh=1 ;;
      scripts/release/*.sh) sh=1 ;;
    esac
  done
  [ "$lanes" = 1 ] && echo lanes
  [ "$sh" = 1 ] && echo bashrs
  return 0
}

self_test() {
  local bad=0 got
  expect() { # NAME EXPECTED INPUT
    got="$(printf '%b' "$3" | plan | tr '\n' ',')"
    if [ "$got" = "$2" ]; then echo "  ok    $1"; else echo "  BAD   $1: got '$got' want '$2'"; bad=$((bad + 1)); fi
  }
  expect triggers_on_release_lanes "lanes,bashrs," 'scripts/release/release_lanes.sh\n'
  expect gates_other_release_script "bashrs," 'scripts/release/promote_rc.sh\n'
  expect lanes_plus_sibling "lanes,bashrs," 'a.rs\nscripts/release/release_lanes.sh\nscripts/release/x.sh\n'
  expect silent_on_unrelated "" 'src/lib.rs\nscripts/check_bashrs_gate.sh\n.github/workflows/ci.yml\n'
  expect silent_on_non_shell_release_file "" 'scripts/release/ledger.py\nscripts/release/fleet-waivers.tsv\n'
  expect silent_on_nested_dir "" 'scripts/release/sub/dir.sh.txt\nxscripts/release/release_lanes.sh\n'
  expect silent_on_empty "" ''
  # wiring: pr-light must call this script (a gate nobody calls is theatre)
  local wf="$ROOT/.github/workflows/ci.yml" n
  n="$(awk '/^  pr-light:/{j=1;next} /^  [a-z0-9-]+:$/{j=0} j && /pr_release_lanes_gate\.sh/{c++} END{print c+0}' "$wf")"
  if [ "$n" -ge 1 ]; then echo "  ok    wired_into_pr_light"; else echo "  BAD   wired_into_pr_light: no call in pr-light"; bad=$((bad + 1)); fi
  if grep -A3 'pr_release_lanes_gate\.sh' "$wf" | grep -q 'continue-on-error'; then echo "  BAD   no_continue_on_error"; bad=$((bad + 1)); else echo "  ok    no_continue_on_error"; fi
  # the bashrs verdict is check_bashrs_gate.sh's, never bashrs's raw exit, and the pinned bashrs is asserted first
  local raw_re='(^|[;&|(] *)(command +)?(bashrs|"?[$][{]?BASHRS[}]?"?) +lint' c
  for c in 'bashrs lint x' 'x && bashrs lint x' 'command bashrs lint x' '"$BASHRS" lint x' '$BASHRS lint x' '${BASHRS} lint x'; do
    if printf '%s\n' "$c" | grep -Eq "$raw_re"; then echo "  ok    raw_re_matches[$c]"; else echo "  BAD   raw_re_matches[$c]"; bad=$((bad + 1)); fi
  done
  for c in 'echo bashrs lint x' 'bashrs_lint x' 'x=bashrs lintel' 'bash scripts/check_bashrs_gate.sh --root r'; do
    if printf '%s\n' "$c" | grep -Eq "$raw_re"; then echo "  BAD   raw_re_ignores[$c]"; bad=$((bad + 1)); else echo "  ok    raw_re_ignores[$c]"; fi
  done
  if grep -Ev '^[[:space:]]*#|for c in' "$HERE/pr_release_lanes_gate.sh" | grep -Eq "$raw_re"; then echo "  BAD   no_raw_bashrs_exit: a bashrs lint call is back"; bad=$((bad + 1)); else echo "  ok    no_raw_bashrs_exit"; fi
  if grep -Ev '^[[:space:]]*(#|echo|printf)' "$HERE/pr_release_lanes_gate.sh" | grep -q 'scripts/check_bashrs_gate[.]sh --root "[$]ROOT"'; then echo "  ok    calls_check_bashrs_gate"; else echo "  BAD   calls_check_bashrs_gate: not called"; bad=$((bad + 1)); fi
  n="$(awk '/^  pr-light:/{j=1;next} /^  [a-z0-9-]+:$/{j=0} j && /check_tool_versions\.sh/ && !t{t=NR} j && /pr_release_lanes_gate\.sh/ && !g{g=NR} END{print (t && g && t < g) ? 1 : 0}' "$wf")"
  if [ "$n" = 1 ]; then echo "  ok    pr_light_asserts_pinned_bashrs_before_gate"; else echo "  BAD   pr_light_asserts_pinned_bashrs_before_gate: no check_tool_versions.sh step before the gate"; bad=$((bad + 1)); fi
  echo "$PROG self-test: $bad broke"
  [ "$bad" = 0 ]
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --plan) plan; exit 0 ;;
esac

base="${1:-origin/main}"
cd "$ROOT"
changed="$(git diff --name-only "$base"...HEAD)" || { echo "$PROG: git diff $base...HEAD failed" >&2; exit 2; }
steps="$(printf '%s\n' "$changed" | plan)"
if [ -z "$steps" ]; then echo "$PROG: not triggered (no scripts/release/*.sh in the diff)"; exit 0; fi

rc=0
if printf '%s\n' "$steps" | grep -qx lanes; then
  bash scripts/release/release_lanes.sh --self-test || rc=1
  bash scripts/release/release_lanes.sh --mutants || rc=1
fi
if printf '%s\n' "$steps" | grep -qx bashrs; then
  # The release's own bashrs verdict (SEC/DET/IDEM gating, positive control, receipt count), never bashrs's raw exit
  # (0/1/2 are overloaded). Whole repo: the gate has no per-file mode. Its exit 2 (bashrs absent, silent control, receipt
  # mismatch) is an environment failure and stays 2: a gate that did not run is not a pass.
  grc=0
  bash scripts/check_bashrs_gate.sh --root "$ROOT" || grc=$?
  case "$grc" in
    0) ;;
    1) rc=1 ;;
    *) echo "$PROG: check_bashrs_gate.sh exited $grc (environment or vacuity, not a pass)" >&2; exit 2 ;;
  esac
fi
exit "$rc"
