#!/usr/bin/env bash
# #7277 (slice of #7181): dogfood-nightly is a LAB lane red 7+ nights. Guard that (a) its workflow has no
# `schedule` trigger while stopped (block key, quoted key, or flow-style `on:`), (b) every step carries an
# positive-integer timeout-minutes, so a hung step ends with the runner's timeout line instead of a silent 88-minute
# exit 1, (c) every job carries an integer timeout-minutes and its steps' timeouts sum within it.
# Restoring the cron is the owner's act: re-add `schedule:` in the workflow AND delete check (a) here
# (the `sched` rule in the awk program), or this guard fails the PR.
# The guard fails closed: a missing or unreadable workflow, an awk failure, or zero steps inspected is a FAIL.
# Usage: check_dogfood_nightly_lab.sh [WORKFLOW]   --self-test runs the planted-defect case table.
set -uo pipefail
here=$(dirname -- "$0")
here=$(cd -- "$here" && pwd)
if [ -z "$here" ]; then exit 1; fi

# The awk program: exit 0 clean (prints one "inspected" line), exit 1 with one finding per line.
AWK_PROG=$(cat <<'AWK'
function strip(s,   i, n, c, q) { # drop a trailing comment (a # after whitespace, outside quotes) and trailing blanks
  q = ""; n = length(s)
  for (i = 1; i <= n; i++) {
    c = substr(s, i, 1)
    if (q == "") {
      if (c == "\042" || c == "\047") q = c
      else if (c == "#" && (i == 1 || substr(s, i - 1, 1) ~ /[ \t]/)) { s = substr(s, 1, i - 1); break }
    } else if (c == q) q = ""
  }
  sub(/[ \t]+$/, "", s)
  return s
}
function close_step() { if (step && !hastm) miss = miss " " name; step = 0 }
function close_job() {
  close_step()
  if (job != "") {
    if (jobsteps == 0) nostep = nostep " " job
    if (jtm == "") jmiss = jmiss " " job
    else if (sum + 0 > jtm + 0) over = over " " job "(" sum ">" jtm ")"
  }
  job = ""; jtm = ""; sum = 0; jobsteps = 0; insteps = 0; stepind = 0
}
function steptm(v) { # v is the text after "timeout-minutes:"; a positive integer only
  if (v ~ /^[ \t]*[1-9][0-9]*[ \t]*$/) { hastm = 1; sum += v + 0 }
}
{
  sub(/\r$/, "")
  line = strip($0)
  if (line ~ /^[ \t]*$/) next
  match(line, /^ */); ind = RLENGTH
  if (ind == 0 && line !~ /^-/) { # a top-level key
    close_job()
    injobs = (line ~ /^jobs[ \t]*:[ \t]*$/)
    inon = (line ~ /^[\042\047]?on[\042\047]?[ \t]*:/)
  }
  if (inon && line ~ /(^|[^A-Za-z0-9_-])[\042\047]?schedule[\042\047]?[ \t]*($|[:,\]}])/) sched = 1
  if (!injobs) next
  if (ind == 2 && line !~ /^ *-/) {
    if (line ~ /^  [^ \t][^:]*:[ \t]*$/) { close_job(); job = line; sub(/^ +/, "", job); sub(/[ \t]*:$/, "", job) }
    else { close_job(); odd = odd " " substr(line, 1, 30) }
    next
  }
  if (job == "") next
  if (ind == 4 && line !~ /^ *-/) { # a job-level key: ends any steps block
    close_step(); insteps = 0
    if (line ~ /^    timeout-minutes[ \t]*:/) {
      v = line; sub(/^    timeout-minutes[ \t]*:/, "", v)
      if (v ~ /^[ \t]*[1-9][0-9]*[ \t]*$/) jtm = v + 0
    } else if (line ~ /^    steps[ \t]*:[ \t]*$/) { insteps = 1; stepind = 0 }
    else if (line ~ /^    steps[ \t]*:/) flow = flow " " job
    next
  }
  if (!insteps) next
  if (line ~ /^ *-( |$)/ && (stepind == 0 || ind == stepind) && ind >= 4) {
    if (stepind == 0) stepind = ind
    close_step(); step = 1; hastm = 0; nsteps++; jobsteps++
    name = line; sub(/^ *-[ \t]*/, "", name); name = substr(name, 1, 30); gsub(/[ \t]/, "_", name)
    if (name == "") name = "step" nsteps
    rest = line; sub(/^ *-[ \t]*/, "", rest)
    if (rest ~ /^timeout-minutes[ \t]*:/) { sub(/^timeout-minutes[ \t]*:/, "", rest); steptm(rest) }
    next
  }
  if (step && stepind > 0 && ind == stepind + 2 && line ~ /^ *timeout-minutes[ \t]*:/) { v = line; sub(/^ *timeout-minutes[ \t]*:/, "", v); steptm(v) }
}
END {
  close_job()
  if (nsteps + 0 == 0) { print "no steps inspected (empty or unparseable workflow)"; bad = 1 }
  if (sched) { print "schedule trigger present while lane is STOPPED"; bad = 1 }
  if (miss != "") { print "steps without a positive integer timeout-minutes:" miss; bad = 1 }
  if (jmiss != "") { print "jobs without a positive integer timeout-minutes:" jmiss; bad = 1 }
  if (nostep != "") { print "jobs with zero steps inspected:" nostep; bad = 1 }
  if (flow != "") { print "jobs with a flow-style or inline steps value:" flow; bad = 1 }
  if (odd != "") { print "unsupported job-level shape (flow-style job?):" odd; bad = 1 }
  if (over != "") { print "step timeouts sum beyond the job timeout:" over; bad = 1 }
  if (!bad) print "inspected " nsteps " steps"
  exit bad
}
AWK
)

check() { # workflow -> prints findings, rc 0 clean / 1 found
  local wf=$1 out arc=0
  if [ ! -f "$wf" ] || [ ! -r "$wf" ]; then echo "workflow missing or unreadable: $wf"; return 1; fi
  out=$(awk "$AWK_PROG" "$wf") || arc=$?
  printf '%s\n' "$out"
  return "$arc"
}

run_main() { # workflow -> prints PASS or FAIL line, returns rc
  local wf=$1
  if check "$wf"; then
    echo "PASS dogfood-nightly stopped and every step has a timeout"
    return 0
  fi
  echo "FAIL dogfood-nightly LAB guard (#7277)"
  return 1
}

selftest() {
  local good fails=0 n=0 out rc
  t=$(mktemp -d) || exit 1 # global: the EXIT trap runs after this function returns
  if [ -z "$t" ] || [ ! -d "$t" ]; then exit 1; fi
  trap 'rm -rf -- "${t:?}"' EXIT
  good="$here/../.github/workflows/dogfood-nightly.yml"

  base() { # a minimal clean workflow
    cat <<'EOF'
name: x
on:
  workflow_dispatch:
jobs:
  night:
    timeout-minutes: 25
    steps:
      - uses: a/b@v1
        timeout-minutes: 10
      - name: two
        timeout-minutes: 10
        run: echo hi
EOF
  }
  # expect NAME FILE want(pass|fail): runs the script itself as a subprocess, so rc and the PASS line are end to end
  expect() {
    n=$((n + 1)); rc=0
    out=$(bash "$0" "$2" 2>&1) || rc=$?
    case $3 in
      pass) if [ "$rc" -ne 0 ] || ! printf '%s\n' "$out" | grep -q '^PASS '; then echo "FAIL case $n ($1): clean input rejected: $out"; fails=$((fails + 1)); fi ;;
      fail) if [ "$rc" -eq 0 ] || printf '%s\n' "$out" | grep -q '^PASS '; then echo "FAIL case $n ($1): defect not caught (rc=$rc)"; fails=$((fails + 1)); fi ;;
      *) echo "FAIL case $n ($1): bad expectation"; fails=$((fails + 1)) ;;
    esac
  }
  put() { base > "$t/$1.yml"; } # fresh base copy

  expect "real workflow" "$good" pass
  put base; expect "minimal base" "$t/base.yml" pass

  # missing, unreadable, empty, no steps
  expect "missing file" "$t/does-not-exist.yml" fail
  mkdir -p "$t/adir"; expect "directory not a file" "$t/adir" fail
  put unread; chmod 000 "$t/unread.yml"
  if [ -r "$t/unread.yml" ]; then echo "NOTE unreadable case not provable as this user (can read mode 000); directory case stands in"; else expect "unreadable file" "$t/unread.yml" fail; fi
  chmod 600 "$t/unread.yml"
  : > "$t/empty.yml"; expect "empty file" "$t/empty.yml" fail
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 5\n' > "$t/nosteps.yml"; expect "zero steps" "$t/nosteps.yml" fail

  # schedule: block, quoted, flow-style, list-style, scalar
  { sed 's/^  workflow_dispatch:/  schedule:/' "$t/base.yml" | awk '{print} /^  schedule:/{print "    - cron: \"5 20 * * *\""}'; } > "$t/s1.yml"; expect "schedule block" "$t/s1.yml" fail
  awk '/^  workflow_dispatch:/{print "  \"schedule\":"; print "    - cron: \"5 20 * * *\""} {print}' "$t/base.yml" > "$t/s2.yml"; expect "schedule double-quoted key" "$t/s2.yml" fail
  awk '/^  workflow_dispatch:/{print "  \047schedule\047:"; print "    - cron: \"5 20 * * *\""} {print}' "$t/base.yml" > "$t/s3.yml"; expect "schedule single-quoted key" "$t/s3.yml" fail
  awk '/^on:/{print "on: { schedule: [{ cron: \"5 20 * * *\" }], workflow_dispatch: {} }"; skip = 1; next} skip && /^  /{next} {skip = 0; print}' "$t/base.yml" > "$t/s4.yml"; expect "schedule flow-style mapping" "$t/s4.yml" fail
  awk '/^on:/{print "on: [schedule, workflow_dispatch]"; skip = 1; next} skip && /^  /{next} {skip = 0; print}' "$t/base.yml" > "$t/s5.yml"; expect "schedule flow list" "$t/s5.yml" fail
  awk '/^on:/{print "\"on\":"; next} {print}' "$t/base.yml" | awk '/^  workflow_dispatch:/{print "  schedule:"; print "    - cron: \"1 1 * * *\""} {print}' > "$t/s6.yml"; expect "quoted on key with schedule" "$t/s6.yml" fail
  awk '/^  workflow_dispatch:/{print "  # schedule: restored later"} {print}' "$t/base.yml" > "$t/s7.yml"; expect "schedule only in a comment" "$t/s7.yml" pass
  awk '/^  workflow_dispatch:/{print "  workflow_dispatch: # schedule is stopped"; next} {print}' "$t/base.yml" > "$t/s8.yml"; expect "schedule in a trailing comment" "$t/s8.yml" pass

  # timeout-minutes values
  grep -v '^        timeout-minutes: 10$' "$t/base.yml" > "$t/t1.yml"; expect "all step timeouts removed" "$t/t1.yml" fail
  awk '/^        timeout-minutes: 10$/ && !d {print "        timeout-minutes:"; d = 1; next} {print}' "$t/base.yml" > "$t/t2.yml"; expect "empty timeout value" "$t/t2.yml" fail
  awk '/^        timeout-minutes: 10$/ && !d {print "        timeout-minutes: abc"; d = 1; next} {print}' "$t/base.yml" > "$t/t3.yml"; expect "non-integer timeout" "$t/t3.yml" fail
  awk '/^        timeout-minutes: 10$/ && !d {print "        timeout-minutes: 1.5"; d = 1; next} {print}' "$t/base.yml" > "$t/t4.yml"; expect "float timeout" "$t/t4.yml" fail
  awk '/^        timeout-minutes: 10$/ && !d {print "        timeout-minutes: 10m"; d = 1; next} {print}' "$t/base.yml" > "$t/t5.yml"; expect "unit-suffixed timeout" "$t/t5.yml" fail
  awk '/^        timeout-minutes: 10$/ && !d {print "        timeout-minutes: 10   # ten"; d = 1; next} {print}' "$t/base.yml" > "$t/t6.yml"; expect "integer timeout with trailing comment" "$t/t6.yml" pass
  awk '/^        timeout-minutes: 10$/ && !d {d = 1; next} {print}' "$t/base.yml" > "$t/t7.yml"; expect "one step loses its timeout" "$t/t7.yml" fail

  # step shapes that do not start with uses:/name:
  { base; printf '      - run: echo hi\n'; } > "$t/u1.yml"; expect "step starting with run: and no timeout" "$t/u1.yml" fail
  { base; printf '      - id: z\n        run: echo hi\n'; } > "$t/u2.yml"; expect "step starting with id: and no timeout" "$t/u2.yml" fail
  { base; printf '      - id: z\n        run: echo hi\n        timeout-minutes: 1\n'; } > "$t/u3.yml"; expect "step starting with id: with timeout" "$t/u3.yml" pass
  { base; printf '      - timeout-minutes: 1\n        run: echo hi\n'; } > "$t/u4.yml"; expect "timeout on the first line of a step" "$t/u4.yml" pass

  # job keys: digits, uppercase, trailing comment, quoted
  { base; printf '  Job_2: # second\n    timeout-minutes: 10\n    steps:\n      - uses: a/b@v1\n'; } > "$t/j1.yml"; expect "uppercase/digit/commented job with an unguarded step" "$t/j1.yml" fail
  { base; printf '  Job_2: # second\n    timeout-minutes: 10\n    steps:\n      - uses: a/b@v1\n        timeout-minutes: 3\n'; } > "$t/j2.yml"; expect "uppercase/digit/commented job, guarded" "$t/j2.yml" pass
  { base; printf '  dogfood2:\n    timeout-minutes: 10\n    steps:\n      - uses: a/b@v1\n'; } > "$t/j3.yml"; expect "digit job after a guarded job, unguarded step" "$t/j3.yml" fail
  { base; printf '  "q-job":\n    timeout-minutes: 10\n    steps:\n      - uses: a/b@v1\n'; } > "$t/j4.yml"; expect "quoted job key, unguarded step" "$t/j4.yml" fail

  # job timeout and the sum
  grep -v '^    timeout-minutes: 25$' "$t/base.yml" > "$t/k1.yml"; expect "job without timeout" "$t/k1.yml" fail
  sed 's/^    timeout-minutes: 25$/    timeout-minutes: 19/' "$t/base.yml" > "$t/k2.yml"; expect "step sum 20 beyond job 19" "$t/k2.yml" fail
  sed 's/^    timeout-minutes: 25$/    timeout-minutes: abc/' "$t/base.yml" > "$t/k3.yml"; expect "non-integer job timeout" "$t/k3.yml" fail
  put k4; sed 's/^    timeout-minutes: 25$/    timeout-minutes: 20 # exact/' "$t/k4.yml" > "$t/k5.yml"; expect "step sum equals job timeout" "$t/k5.yml" pass

  # lane-2 FAIL at b55299d: zero timeouts, per-job zero steps, indentation, sibling blocks, `on :`, CRLF, quoted #
  sed 's/^        timeout-minutes: 10$/        timeout-minutes: 0/' "$t/base.yml" > "$t/z1.yml"; expect "step timeout 0" "$t/z1.yml" fail
  sed 's/^    timeout-minutes: 25$/    timeout-minutes: 0/' "$t/base.yml" > "$t/z2.yml"; expect "job timeout 0" "$t/z2.yml" fail
  sed 's/^        timeout-minutes: 10$/        timeout-minutes: 00/' "$t/base.yml" > "$t/z3.yml"; expect "step timeout 00" "$t/z3.yml" fail
  { base; printf '  dogfood:\n    timeout-minutes: 420\n    steps:\n    - uses: a/b@v1\n    - run: echo no-timeout\n'; } > "$t/i1.yml"; expect "second job, 4-space steps, untimed" "$t/i1.yml" fail
  { base; printf '  dogfood:\n    timeout-minutes: 420\n    steps:\n    - uses: a/b@v1\n      timeout-minutes: 5\n    - run: echo hi\n      timeout-minutes: 5\n'; } > "$t/i2.yml"; expect "second job, 4-space steps, timed" "$t/i2.yml" pass
  { base; printf '  dogfood:\n    timeout-minutes: 420\n    steps:\n            - run: echo deep\n'; } > "$t/i3.yml"; expect "second job, 12-space steps, untimed" "$t/i3.yml" fail
  { base; printf '  dogfood:\n    timeout-minutes: 420\n    runs-on: x\n'; } > "$t/i4.yml"; expect "second job with zero steps" "$t/i4.yml" fail
  { base; printf '  dogfood: { timeout-minutes: 420, steps: [ { run: echo hi } ] }\n'; } > "$t/i5.yml"; expect "flow-style job" "$t/i5.yml" fail
  { base; printf '  dogfood:\n    timeout-minutes: 420\n    steps: [ { run: echo hi } ]\n'; } > "$t/i6.yml"; expect "flow-style steps" "$t/i6.yml" fail
  awk '/^        timeout-minutes: 10$/ {c++; if (c == 2) next} {print}' "$t/base.yml" > "$t/y2.yml"; printf '    strategy:\n      matrix:\n        timeout-minutes: 1\n' >> "$t/y2.yml"; expect "sibling block timeout does not cover an untimed step" "$t/y2.yml" fail
  awk '/^on:/{print "on :"; next} {print}' "$t/base.yml" | awk '/^  workflow_dispatch:/{print "  schedule:"; print "    - cron: \"1 1 * * *\""} {print}' > "$t/o1.yml"; expect "on with space before colon plus schedule" "$t/o1.yml" fail
  awk '/^on:/{print "\"on\" :"; next} {print}' "$t/base.yml" | awk '/^  workflow_dispatch:/{print "  schedule:"; print "    - cron: \"1 1 * * *\""} {print}' > "$t/o2.yml"; expect "quoted on with space before colon plus schedule" "$t/o2.yml" fail
  awk '/^on:/{print "on: { x: \"a #b\", schedule: [{ cron: \"5 20 * * *\" }], workflow_dispatch: {} }"; skip = 1; next} skip && /^  /{next} {skip = 0; print}' "$t/base.yml" > "$t/o3.yml"; expect "quoted hash before schedule in flow on" "$t/o3.yml" fail
  sed 's/$/\r/' "$t/base.yml" > "$t/c1.yml"; expect "CRLF clean file" "$t/c1.yml" pass
  { sed 's/^  workflow_dispatch:/  schedule:/' "$t/base.yml" | awk '{print} /^  schedule:/{print "    - cron: \"5 20 * * *\""}'; } | sed 's/$/\r/' > "$t/c2.yml"; expect "CRLF schedule" "$t/c2.yml" fail
  printf 'on:\r\n  workflow_dispatch:\r\njobs:\r\n  night:\r\n    timeout-minutes: 5\r\n    steps:\r\n      - run: x\r\n        timeout-minutes: 0\r\n' > "$t/c3.yml"; expect "CRLF zero step timeout" "$t/c3.yml" fail

  # real-workflow mutants
  awk '/^  workflow_dispatch:/{print "  schedule:"; print "    - cron: \"5 20 * * *\""} {print}' "$good" > "$t/r1.yml"; expect "real workflow plus schedule" "$t/r1.yml" fail
  grep -v '^        timeout-minutes:' "$good" > "$t/r2.yml"; expect "real workflow without step timeouts" "$t/r2.yml" fail
  sed 's/^        timeout-minutes: 300$/        timeout-minutes: 500/' "$good" > "$t/r3.yml"; expect "real workflow, step sum beyond job" "$t/r3.yml" fail

  if [ "$fails" -ne 0 ]; then echo "FAIL dogfood-nightly LAB guard self-test: $fails of $n cases wrong"; return 1; fi
  echo "PASS dogfood-nightly LAB guard: $n cases (missing, unreadable, empty, zero steps, schedule forms, timeout shapes, job keys, sums) all as expected"
  return 0
}

if [ "${1:-}" = --self-test ]; then selftest; exit $?; fi
run_main "${1:-$here/../.github/workflows/dogfood-nightly.yml}"
exit $?
