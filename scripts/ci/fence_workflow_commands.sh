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
  # The resume line must start a line. A child whose last line has no newline
  # would get it glued on, the fence would never close, and the failure
  # ::error:: below would be swallowed too. One blank line is the cost.
  printf '\n::%s::\n' "$tok"
  if [ "$rc" -ne 0 ]; then
    printf '::error::case table failed (rc=%s): %s\n' "$rc" "$*"
  fi
  return "$rc"
}

# live_commands <log> -> how many annotation commands the runner would act on,
# outside every stop..resume region. Each rule errs on the high side, so a 0
# here is a 0 on the runner and a 1 is at most 1:
#  - Lines split on \r as well as \n, as the runner splits them. The runner
#    acts on at most one command per line, so a line counts once.
#  - An annotation is ::error, ::warning or ::notice, or the legacy ##[error,
#    ##[warning or ##[notice, anywhere in a line and in any case. The runner
#    wants it first after any whitespace.
#  - A region starts only at the fence's own stop line: column 0, lower case,
#    a 32-hex token and nothing after it.
#  - A region ends at the first line holding ::<token> or ##[<token> in any
#    case. That line is never live: it is the resume, or the runner is still
#    stopped.
#  - Doubt: any other line naming stop-commands may stop the runner, and a
#    region that ends on a line other than ::<token>:: may not resume it.
#    After either, no region starts again, so every later annotation counts.
# Problem matchers are not workflow commands: the fence does not stop them,
# and this does not count them.
live_commands() {
  LC_ALL=C awk '
    function one(s,  lc, t) {
      lc = tolower(s)
      if (stop != "") {
        if (index(lc, "::" stop) == 0 && index(lc, "##[" stop) == 0) return
        if (lc != "::" stop "::") doubt = 1
        stop = ""; return
      }
      if (index(lc, "stop-commands")) {
        t = substr(s, 18)
        if (!doubt && substr(s, 1, 17) == "::stop-commands::" && length(t) == 32 && t !~ /[^0-9a-f]/) { stop = t; return }
        doubt = 1
      }
      if (lc ~ /(::|##[[])(error|warning|notice)/) n++
    }
    { k = split($0, part, "\r"); for (i = 1; i <= k; i++) one(part[i]) }
    END { print n + 0 }' "$1"
}

# check_wiring <ci.yml> -> one FAIL line per defect. rc 0 = every
# `fat_driver.py self-test` line is exactly the fenced command, alone after an
# optional `run:` (so nothing on the line masks its rc), and no
# `fat_driver.py run` or `wait` is fenced; 1 = a defect; 2 = unreadable, or no
# self-test step found.
check_wiring() {
  [ -r "$1" ] || return 2
  local bad=0 seen=0 line cmd
  while IFS= read -r line; do
    seen=$((seen + 1))
    cmd="$(printf '%s' "$line" | sed -E 's/^[[:space:]]*(-[[:space:]]+)?(run:[[:space:]]+)?//; s/[[:space:]]+$//')"
    case "$cmd" in
      "bash $FENCE_REL -- python3 scripts/ci/fat_driver.py self-test") ;;
      *)
        printf 'FAIL: %s runs the fat_driver case table other than as exactly `bash %s -- python3 scripts/ci/fat_driver.py self-test` -- unfenced, its planted rows raise live ::error:: annotations; with more on the line, its rc can be masked: %s\n' \
          "$WORKFLOW_REL" "$FENCE_REL" "$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//')"
        bad=1 ;;
    esac
  done < <(grep -E 'fat_driver\.py[[:space:]]+self-test' "$1")
  while IFS= read -r line; do
    printf 'FAIL: %s fences a real fat_driver run -- its deadline ::error:: must stay live: %s\n' \
      "$WORKFLOW_REL" "$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//')"
    bad=1
  done < <(grep -E "$FENCE_REL.*fat_driver\\.py[[:space:]]+(run|wait)" "$1")
  [ "$seen" -gt 0 ] || return 2
  return "$bad"
}

self_test() {
  printf '=== case table: fence_workflow_commands.sh ===\n'
  local tmp fails=0 rc t1 t2 t3 tk tk2 got i f good
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
  # The counter can only say "at most 1". The exact resume line just before the
  # last line is what makes the failure annotation live.
  t3="$(sed -n '1s/^::stop-commands:://p' "$tmp/fail.out")"
  row 'a failing table: the last line names it, after the exact resume line' '1 1' \
    "$(tail -n 2 "$tmp/fail.out" | head -n 1 | grep -cxF "::$t3::") $(tail -n 1 "$tmp/fail.out" | grep -cF "::error::case table failed (rc=3): bash $tmp/table.sh 3")"

  fenced sh -c 'printf "::error::planted\nno newline at the end"; exit 3' > "$tmp/partial.out"; rc=$?
  t2="$(sed -n '1s/^::stop-commands:://p' "$tmp/partial.out")"
  row 'a last line with no newline: the fence still closes on its own line' 1 "$(grep -cxF "::$t2::" "$tmp/partial.out")"
  row 'a last line with no newline: rc 3 and only the failure annotation is live' '3 1' "$rc $(live_commands "$tmp/partial.out")"

  # stdout and stderr are two pipes on the runner, so they go to two files here:
  # a stderr line that kept its own pipe would land in stderr.err, unfenced.
  fenced sh -c 'printf "::error::from stderr\n" >&2' > "$tmp/stderr.out" 2> "$tmp/stderr.err"
  row 'stderr is fenced too: none on its own pipe, 1 inside the fence, 0 live' '0 1 0' \
    "$(wc -l < "$tmp/stderr.err" | tr -d ' ') $(sed '1d;$d' "$tmp/stderr.out" | grep -cxF '::error::from stderr') $(live_commands "$tmp/stderr.out")"

  # The counter is the oracle for every "0 live" row above, so each form the
  # runner acts on must count, one file each.
  got=""; i=0
  for f in '  ::error::indented' '\f::error::after a form feed' '\0302\0240::notice::after a no-break space' \
    '::Error::mixed case' '::WARNING file=a.rs:1,line=2::upper case, a colon in a property' \
    '##[notice]the legacy form' 'progress 40%\r::error::after a carriage return' \
    '::error::one\r::warning::two\r::notice::three, all on one line'; do
    i=$((i + 1)); printf '%b\n' "$f" > "$tmp/form$i.log"; got="$got $(live_commands "$tmp/form$i.log")"
  done
  row 'the live counter counts each of 8 forms the runner acts on' ' 1 1 1 1 1 1 1 3' "$got"
  tk=0123456789abcdef0123456789abcdef
  printf '%b\n' "::stop-commands::$tk" 'inside' '  ::0123456789ABCDEF0123456789ABCDEF:: trailing' '::error::after it' > "$tmp/end1.log"
  printf '%b\n' "::stop-commands::$tk" "x\r::$tk::\r::error::after the resume" > "$tmp/end2.log"
  printf '%b\n' "::stop-commands::$tk" 'x ##[0123456789ABCDEF0123456789ABCDEF] y' '::error::after a legacy resume' > "$tmp/end3.log"
  row 'a region ends at the first line holding ::TOKEN or ##[TOKEN, in any case and place' '1 1 1' \
    "$(live_commands "$tmp/end1.log") $(live_commands "$tmp/end2.log") $(live_commands "$tmp/end3.log")"
  printf '%b\n' "  ::stop-commands::$tk" '::error::after an indented stop' "::$tk::" > "$tmp/start1.log"
  printf '%b\n' '::stop-commands::short' '::error::after a short token' '::short::' > "$tmp/start2.log"
  row 'a region starts only at the fence form of the stop line' '1 1' \
    "$(live_commands "$tmp/start1.log") $(live_commands "$tmp/start2.log")"
  # On the runner each of these is still stopped, or stopped again, where a
  # fence region would start: the stop line is ignored, and the old token
  # resumes it. The annotation after that is live.
  tk2=fedcba9876543210fedcba9876543210
  printf '%b\n' '  ::stop-commands::odd' "::stop-commands::$tk" '::odd::' '::error::live after an odd stop' > "$tmp/doubt1.log"
  printf '%b\n' "::stop-commands::$tk" "x ::$tk::" "::stop-commands::$tk2" "::$tk::" '::error::live after an odd end' > "$tmp/doubt2.log"
  row 'after a line that leaves the runner in doubt, no region starts again' '1 1' \
    "$(live_commands "$tmp/doubt1.log") $(live_commands "$tmp/doubt2.log")"

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
  got=""; i=0
  for f in ' || true' '; true' ' | tee x.log' ' # ok'; do
    i=$((i + 1)); printf '        run: bash %s -- python3 scripts/ci/fat_driver.py self-test%s\n' "$FENCE_REL" "$f" > "$tmp/mask$i.yml"
    check_wiring "$tmp/mask$i.yml" > /dev/null; got="$got $?"
  done
  printf '        - run: bash %s -- python3 scripts/ci/fat_driver.py  self-test\n' "$FENCE_REL" > "$tmp/mask5.yml"
  check_wiring "$tmp/mask5.yml" > /dev/null; got="$got $?"
  row 'wiring: more on the self-test line, or another spacing, is RED' ' 1 1 1 1 1' "$got"
  printf '          bash %s -- python3 scripts/ci/fat_driver.py self-test  \n' "$FENCE_REL" > "$tmp/block.yml"
  check_wiring "$tmp/block.yml" > /dev/null; row 'wiring: the fenced command alone in a run block is clean' 0 "$?"
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
