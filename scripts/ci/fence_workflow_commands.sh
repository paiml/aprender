#!/usr/bin/env bash
# fence_workflow_commands.sh -- run a case table with GitHub's workflow-command
# processing stopped for its output, so a planted row cannot raise a live
# annotation on the job (#4936).
#
# WHY THIS EXISTS
# ---------------
# `fat_driver.py self-test` drives the real code paths with planted failures,
# and those paths print real `::error::` workflow commands: "section a:
# failure", "section a: cancelled" and "external job 'workspace-test': not
# completed within 12000s". The table passed 60/60, yet every x86-main and
# workspace-test shard job carried those three as error annotations about a
# minute in. x86-main runs no --external-job at all, so its 12000s line could
# only ever come from the table; when the job went red half an hour later on
# another section, the annotation was read as the cause (#4813, #4913).
#
# The fence is GitHub's own `::stop-commands::<token>` ... `::<token>::`. The
# token is drawn fresh per call and never exported, so the child cannot end
# the fence early. The child's stderr is merged into stdout so every line
# crosses one pipe in order and none lands after the resume line. The verdict
# is untouched: the child's exit status is the step's, and a failing table gets
# one live ::error:: naming it, after the fence.
#
# Only case tables are fenced. `fat_driver.py run` and `wait` stay unfenced:
# their deadline ::error:: is the real one, and the wiring check refuses a
# fenced one.
#
#   bash scripts/ci/fence_workflow_commands.sh -- python3 scripts/ci/fat_driver.py self-test
#   bash scripts/ci/fence_workflow_commands.sh            # wiring check of ci.yml
#   bash scripts/ci/fence_workflow_commands.sh --self-test
#   bash scripts/ci/fence_workflow_commands.sh --live <log>   # live annotation count
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKFLOW_REL=".github/workflows/ci.yml"
FENCE_REL="scripts/ci/fence_workflow_commands.sh"

# fenced <cmd...> -> the child's output between a stop line and its resume
# line. rc = the child's rc; 2 when no command is given or no token is drawn.
fenced() {
  if [ "$#" -eq 0 ]; then
    printf '::error::%s: no command to fence\n' "$FENCE_REL"
    return 2
  fi
  local tok rc
  tok="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
  if [ "${#tok}" -ne 32 ]; then
    printf '::error::%s: no stop-commands token drawn\n' "$FENCE_REL"
    return 2
  fi
  printf '::stop-commands::%s\n' "$tok"
  "$@" 2>&1
  rc=$?
  printf '::%s::\n' "$tok"
  if [ "$rc" -ne 0 ]; then
    printf '::error::case table failed (rc=%s): %s\n' "$rc" "$*"
  fi
  return "$rc"
}

# live_commands <log> -> how many annotation commands the runner would act on:
# ::error/::warning/::notice lines outside every stop..resume region.
live_commands() {
  awk '
    stop != "" { if ($0 == "::" stop "::") stop = ""; next }
    /^::stop-commands::/ { stop = substr($0, 18); next }
    /^::(error|warning|notice)( [^:]*)?::/ { n++ }
    END { print n + 0 }' "$1"
}

# check_wiring <ci.yml> -> one FAIL line per defect. rc 0 = every
# `fat_driver.py self-test` runs through the fence and no `fat_driver.py run`
# or `wait` does; 1 = a defect; 2 = unreadable, or no self-test step found.
check_wiring() {
  [ -r "$1" ] || return 2
  local bad=0 seen=0 line
  while IFS= read -r line; do
    seen=$((seen + 1))
    case "$line" in
      *"$FENCE_REL -- python3 scripts/ci/fat_driver.py self-test"*) ;;
      *)
        printf 'FAIL: %s runs the fat_driver case table unfenced -- its planted rows raise live ::error:: annotations: %s\n' \
          "$WORKFLOW_REL" "$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//')"
        bad=1 ;;
    esac
  done < <(grep -E 'fat_driver\.py self-test' "$1")
  while IFS= read -r line; do
    printf 'FAIL: %s fences a real fat_driver run -- its deadline ::error:: must stay live: %s\n' \
      "$WORKFLOW_REL" "$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//')"
    bad=1
  done < <(grep -E "$FENCE_REL.*fat_driver\\.py (run|wait)" "$1")
  [ "$seen" -gt 0 ] || return 2
  return "$bad"
}

self_test() {
  printf '=== case table: fence_workflow_commands.sh ===\n'
  local tmp fails=0 rc t1 t2 good
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:?}"' RETURN
  row() {  # row <label> <want> <got>
    if [ "$2" = "$3" ]; then
      printf '  ok   %-66s %s\n' "$1" "$3"
    else
      printf '  FAIL %-66s want=%s got=%s\n' "$1" "$2" "$3"
      fails=$((fails + 1))
    fi
  }
  # The three lines the fat_driver table printed live on #4813 and #4913.
  printf '%s\n' '#!/usr/bin/env bash' \
    "printf '%s\n' '::error::section a: failure' '::error::section a: cancelled'" \
    "printf '%s\n' \"::error::external job 'workspace-test': not completed within 12000s\"" \
    'printf "fat_driver self-test: 60/60 rows as expected\n"' \
    'exit "${1:-0}"' > "$tmp/table.sh"

  bash "$tmp/table.sh" > "$tmp/bare.out" 2>&1
  row 'control: the #4813 table unfenced raises 3 live annotations' 3 "$(live_commands "$tmp/bare.out")"

  fenced bash "$tmp/table.sh" > "$tmp/pass.out"; rc=$?
  row 'the #4813 table fenced: passes' 0 "$rc"
  row 'the #4813 table fenced: 0 live annotations' 0 "$(live_commands "$tmp/pass.out")"
  t1="$(sed -n '1s/^::stop-commands:://p' "$tmp/pass.out")"
  row 'the fence opens on line 1 with a 32-hex token' 1 "$(printf '%s\n' "$t1" | grep -cE '^[0-9a-f]{32}$')"
  row 'the fence closes on the last line with the same token' 1 "$([ "$(tail -n 1 "$tmp/pass.out")" = "::$t1::" ] && echo 1 || echo 0)"
  row 'the table output is kept, inside the fence' 1 "$(grep -c '^fat_driver self-test: 60/60 rows as expected$' "$tmp/pass.out")"

  fenced bash "$tmp/table.sh" 3 > "$tmp/fail.out"; rc=$?
  row 'a failing table still fails: its rc is the step rc' 3 "$rc"
  row 'a failing table: exactly 1 live annotation, after the fence' 1 "$(live_commands "$tmp/fail.out")"
  row 'a failing table: the live annotation names it' 1 \
    "$(tail -n 1 "$tmp/fail.out" | grep -cF "::error::case table failed (rc=3): bash $tmp/table.sh 3")"

  fenced sh -c 'printf "::error::from stderr\n" >&2' > "$tmp/stderr.out"
  row 'stderr is fenced too (merged into the one ordered pipe)' 0 "$(live_commands "$tmp/stderr.out")"

  fenced sh -c 'printf "::0123456789abcdef0123456789abcdef::\n::error::after a guessed resume\n"' > "$tmp/guess.out"
  row 'a resume line with another token does not end the fence' 0 "$(live_commands "$tmp/guess.out")"

  fenced env > "$tmp/env.out"
  t2="$(sed -n '1s/^::stop-commands:://p' "$tmp/env.out")"
  row 'the token is not in the child environment' 0 "$(sed '1d;$d' "$tmp/env.out" | grep -c "${t2:-no-token-drawn}")"
  row 'the token is fresh per call' 1 "$([ -n "$t1" ] && [ "$t1" != "$t2" ] && echo 1 || echo 0)"

  fenced > "$tmp/none.out"; rc=$?
  row 'no command to fence is refused' 2 "$rc"

  # Wiring: the self-test steps go through the fence, the real runs never do.
  good="          python3 scripts/ci/fat_driver.py run --sections x"
  printf '        run: bash %s -- python3 scripts/ci/fat_driver.py self-test\n%s\n' "$FENCE_REL" "$good" > "$tmp/ok.yml"
  check_wiring "$tmp/ok.yml" > /dev/null; row 'wiring: fenced self-test, unfenced run' 0 "$?"
  printf '        run: python3 scripts/ci/fat_driver.py self-test\n%s\n' "$good" > "$tmp/bare.yml"
  check_wiring "$tmp/bare.yml" > /dev/null; row 'wiring: an unfenced self-test is RED (the #4936 state)' 1 "$?"
  printf '        run: bash %s -- python3 scripts/ci/fat_driver.py self-test\n        run: bash %s -- python3 scripts/ci/fat_driver.py run --sections x\n' \
    "$FENCE_REL" "$FENCE_REL" > "$tmp/runfenced.yml"
  check_wiring "$tmp/runfenced.yml" > /dev/null; row 'wiring: a fenced real run is RED (its deadline must stay live)' 1 "$?"
  printf '%s\n' "$good" > "$tmp/noselftest.yml"
  check_wiring "$tmp/noselftest.yml" > /dev/null; row 'wiring: no self-test step found is not clean' 2 "$?"
  check_wiring "$tmp/absent.yml" > /dev/null; row 'wiring: an unreadable workflow is not clean' 2 "$?"

  if [ "$fails" -gt 0 ]; then
    printf '\nFAIL: %s case(s) failed. The fence does not do what it claims.\n' "$fails"
    return 1
  fi
  printf 'PASS: all cases behave as declared.\n'
  return 0
}

case "${1:-}" in
  --self-test)
    self_test
    exit $?
    ;;
  --)
    shift
    fenced "$@"
    exit $?
    ;;
  --live)
    [ -r "${2:-}" ] || { printf 'usage: %s --live <log>\n' "$FENCE_REL" >&2; exit 2; }
    live_commands "$2"
    exit 0
    ;;
  "")
    printf '=== every fat_driver case table in ci.yml is fenced, no real run is (fence_workflow_commands.sh) ===\n'
    check_wiring "$REPO_ROOT/$WORKFLOW_REL"
    rc=$?
    if [ "$rc" -eq 2 ]; then
      printf 'FAIL: %s is unreadable or runs no fat_driver self-test -- a check that read nothing certifies nothing.\n' "$WORKFLOW_REL"
      exit 1
    fi
    [ "$rc" -eq 0 ] || exit 1
    printf 'PASS: %s fat_driver self-test step(s), each fenced; no real run fenced.\n' \
      "$(grep -cE 'fat_driver\.py self-test' "$REPO_ROOT/$WORKFLOW_REL")"
    exit 0
    ;;
  *)
    printf 'usage: %s [--self-test | --live <log> | -- <cmd> [args...]]\n' "$FENCE_REL" >&2
    exit 2
    ;;
esac
