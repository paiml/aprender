#!/usr/bin/env bash
# #7277 (slice of #7181): dogfood-nightly is a LAB lane red 7+ nights. Guard that (a) its workflow has no
# `schedule` trigger while stopped (block key, quoted key, or flow-style `on:`), (b) every step carries an
# positive-integer timeout-minutes, so a hung step ends with the runner's timeout line instead of a silent 88-minute
# exit 1, (c) every job carries an integer timeout-minutes and its steps' timeouts sum within it.
# Restoring the cron is the owner's act: re-add `schedule:` in the workflow AND delete check (a) here
# (the `sched` rule in the awk program), or this guard fails the PR.
# The guard fails closed: a missing or unreadable workflow, an awk failure, or zero steps inspected is a FAIL, and a
# workflow in a YAML form it does not judge (flow style, tags, anchors, '? ' keys, a BOM, escapes in quoted scalars,
# a block scalar in a job header, a job-key indent that differs from the first job's) is UNJUDGED, exit 2.
# Usage: check_dogfood_nightly_lab.sh [WORKFLOW]   --self-test runs the planted-defect case table.
set -uo pipefail
here=$(dirname -- "$0")
here=$(cd -- "$here" && pwd)
if [ -z "$here" ]; then exit 1; fi

# The awk program: exit 0 clean (prints one "inspected" line), exit 1 with one finding per line, exit 2 when the file
# uses a YAML form this script does not judge (printed as "UNJUDGED ..."): refuse rather than guess.
AWK_PROG=$(cat <<'AWK'
function adj(s, ci, lc,   k) { # the colon at ci follows a quoted key that closed at lc, with only blanks between (JSON-style "k" :"v")
  for (k = ci - 1; k >= 1 && substr(s, k, 1) ~ /[ \t]/; k--) ;
  return (k == lc)
}
function strip(s,   i, n, c, q, p, pp, j, L, dd, lc, k) { # drop a trailing comment (a # after whitespace, outside quotes) and trailing blanks
  # a quote opens a scalar only at a token start (line start or after blank , : [ ] { } -), never mid-word (it's);
  # noq is the line with its quoted spans removed; uq is set when a quote is still open at the end of the line
  q = ""; n = length(s); noq = ""; uq = 0; dd = fdepth + 0; lc = -9
  for (i = 1; i <= n; i++) {
    c = substr(s, i, 1); p = (i == 1) ? " " : substr(s, i - 1, 1); pp = (i <= 2) ? " " : substr(s, i - 2, 1)
    if (q == "") {
      # a hyphen never opens a quote (a-' is mid-word); a colon does only right after a closing quote ("k":"v")
      if ((c == "\042" || c == "\047") && (p ~ /[ \t,[\]{}]/ || (p == ":" && adj(s, i - 1, lc)))) {
        # inside a flow collection a quote opens a scalar only after : , [ { or a block dash; after a word it is text of a
        # plain scalar (a '), and which quote closes which is then ambiguous: refuse the line (midq) rather than guess
        L = " "; for (j = i - 1; j >= 1; j--) { L = substr(s, j, 1); if (L !~ /[ \t]/) break }
        if (dd > 0 && L !~ /[ \t:,[{]/) midq = midq + 1
        # block context: a quote after a word in KEY position (no ": " before it), or with flow brackets after it on the line, is
        # text of a plain scalar that this scan would read as a quote opener, and which brackets are real is then ambiguous
        if (dd == 0 && L !~ /[ \t:,[{]/ && substr(s, 1, i - 1) !~ /^[ \t]*(-[ \t]+)*$/ && (substr(s, 1, i - 1) !~ /:([ \t]|$)/ || substr(s, i + 1) ~ /[{}[\]]/)) midq = midq + 1
        q = c
      }
      else if ((c == "\042" || c == "\047") && p == ":" && dd > 0) midq = midq + 1 # an adjacent quote after a colon that did not follow a quoted key ([]:"v", a:"v"): refuse
      else if (c == "#" && p ~ /[ \t]/) { s = substr(s, 1, i - 1); break }
      else if (c == "{" && p == "$" && substr(s, i + 1, 1) == "{") { # a ${{ }} expression: its braces are not flow; inside a flow collection refuse
        if (dd > 0 || substr(s, 1, i - 1) !~ /^[ \t]*(-[ \t]+)*([A-Za-z0-9_.\/-]+|"[^"]*"|\047[^\047]*\047)[ \t]*:[ \t]/) midq = midq + 1 # only a block VALUE may hold an expression; in a key or in flow, refuse
        j = index(substr(s, i), "}}"); if (j == 0) { midq = midq + 1; noq = noq c; continue } # unclosed expression: refuse the line
        noq = noq substr(s, i, j + 1); i = i + j
      }
      else { noq = noq c; if (c ~ /[{[]/) dd++; else if (c ~ /[}\]]/) dd-- }
    } else if (q == "\042" && c == "\\") { esc = 1; i++ }
    else if (c == q) { if (q == "\047" && substr(s, i + 1, 1) == "\047") i++; else { q = ""; lc = i } }
  }
  if (q != "") uq = 1
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
BEGIN { bsind = -1 }
{
  sub(/\r$/, "")
  if (FNR == 1 && index($0, "\357\273\277") == 1) bom = 1
  if (bsind >= 0) { # inside a block scalar: its text is data, not structure
    if ($0 ~ /^[ \t]*$/) next
    match($0, /^ */)
    if (RLENGTH > bsind) next
    bsind = -1
  }
  esc = 0
  line = strip($0)
  if (esc) escn++
  if (uq) mlq++
  if (line ~ /^[ \t]*$/) next
  match(line, /^ */); ind = RLENGTH
  isdash = (line ~ /^ *-( |$)/); kcol = ind; bsr = line
  if (isdash) { bsr = line; sub(/^ *-[ \t]*/, "", bsr); kcol = length(line) - length(bsr) }
  if (line ~ /(:|-)[ \t]+[|>][-+0-9]*$/) bsind = (isdash && bsr ~ /^[|>]/) ? ind : kcol
  isbs = (bsind >= 0)
  if (line ~ /(^[ \t]*(-[ \t]+)*|:[ \t]+)[!&*]/) tag = tag + 1
  else if (noq ~ /[{[,][ \t]*[!&*]/ || noq ~ /:[ \t]*[!&*]/) tag = tag + 1 # tag, anchor or alias inside a flow collection
  if (line ~ /^[ \t]*(-[ \t]+)*\?([ \t]|$)/) qkey = qkey + 1
  t = noq
  if (fdepth > 0 && ind == 0) flowcont = flowcont + 1
  fdepth += gsub(/[{[]/, "", t); fdepth -= gsub(/[}\]]/, "", t)
  if (fdepth < 0) { negflow = negflow + 1; fdepth = 0 }
  if (fdepth > 0) flowml = flowml + 1
  if (ind == 0) { # a top-level key; anything else at column 0 is a form this script does not judge
    if (line !~ /^([A-Za-z0-9_.-]+|"[^"]*"|\047[^\047]*\047)[ \t]*:([ \t]|$)/) badtop = badtop " " substr(line, 1, 20)
    if (line ~ /^["\047]?jobs["\047]?[ \t]*:[ \t]*[^ \t]/) badtop = badtop " jobs-inline"
    close_job()
    injobs = (line ~ /^["\047]?jobs["\047]?[ \t]*:[ \t]*$/)
    inon = (line ~ /^[\042\047]?on[\042\047]?[ \t]*:/)
    jind = 0; jkind = 0
  }
  if (inon && line ~ /(^|[^A-Za-z0-9_-])[\042\047]?schedule[\042\047]?[ \t]*($|[:,\]}])/) sched = 1
  if (!injobs || ind == 0) next
  if (jind == 0 && !isdash) jind = ind
  if (!isdash && ind < jind) { close_job(); odd = odd " " substr(line, 1, 30); next }
  if (ind == jind && !isdash) { # a job key; the first job's indent is the file's job indent
    if (line ~ /^ *[^ \t][^:]*:[ \t]*$/) { close_job(); job = line; sub(/^ +/, "", job); sub(/[ \t]*:$/, "", job); jseen = 0 }
    else { close_job(); odd = odd " " substr(line, 1, 30) }
    next
  }
  if (job == "") next
  if (jkind == 0 && !isdash) jkind = ind # the first job's key indent is the file's job-key indent
  if (!jseen && !isdash) { jseen = 1; if (ind != jkind) odd = odd " " job "(key indent differs)" }
  if (!isdash && ind > jind && ind < jkind) { odd = odd " " job "(key indent differs)"; next }
  if (ind == jkind && !isdash) { # a job-level key: ends any steps block
    close_step(); insteps = 0
    if (isbs) blockhdr = blockhdr " " job
    if (line ~ /^ *timeout-minutes[ \t]*:/) {
      v = line; sub(/^ *timeout-minutes[ \t]*:/, "", v)
      if (v ~ /^[ \t]*[1-9][0-9]*[ \t]*$/) jtm = v + 0
    } else if (line ~ /^ *steps[ \t]*:[ \t]*$/) { insteps = 1; stepind = 0 }
    else if (line ~ /^ *steps[ \t]*:/) flow = flow " " job
    next
  }
  if (!insteps) next
  if (isdash && (stepind == 0 || ind == stepind) && ind >= jkind) {
    if (stepind == 0) stepind = ind
    close_step(); step = 1; hastm = 0; nsteps++; jobsteps++; childind = kcol
    name = line; sub(/^ *-[ \t]*/, "", name); name = substr(name, 1, 30); gsub(/[ \t]/, "_", name)
    if (name == "") name = "step" nsteps
    if (bsr !~ /^[A-Za-z0-9_-]+[ \t]*:([ \t]|$)/) badstep = badstep " " name
    rest = bsr
    if (rest ~ /^timeout-minutes[ \t]*:/) { sub(/^timeout-minutes[ \t]*:/, "", rest); steptm(rest) }
    next
  }
  # the step's own timeout is a DIRECT child key only: at the column of the step's first key, never nested deeper
  if (step && !isdash && ind == childind && line ~ /^ *timeout-minutes[ \t]*:/) { v = line; sub(/^ *timeout-minutes[ \t]*:/, "", v); steptm(v) }
}
END {
  close_job()
  unj = 0
  if (bom) { print "UNJUDGED: byte-order mark at the start of the file"; unj = 1 }
  if (badtop != "") { print "UNJUDGED: column-0 line that is not a plain or quoted top-level key (flow continuation, document marker, anchor):" badtop; unj = 1 }
  if (flowcont + 0 > 0) { print "UNJUDGED: multi-line flow collection continued at column 0"; unj = 1 }
  if (mlq + 0 > 0) { print "UNJUDGED: quoted scalar continued over several lines (" mlq " lines)"; unj = 1 }
  if (flowml + 0 > 0) { print "UNJUDGED: multi-line flow collection (" flowml " lines)"; unj = 1 }
  if (midq + 0 > 0) { print "UNJUDGED: quote after a plain-scalar word inside a flow collection (" midq " lines)"; unj = 1 }
  if (negflow + 0 > 0) { print "UNJUDGED: closing flow bracket with no opener (" negflow " lines)"; unj = 1 }
  if (tag + 0 > 0) { print "UNJUDGED: tag, anchor or alias at a key or value start (" tag " lines)"; unj = 1 }
  if (qkey + 0 > 0) { print "UNJUDGED: explicit '? ' key (" qkey " lines)"; unj = 1 }
  if (escn + 0 > 0) { print "UNJUDGED: backslash escape inside a double-quoted scalar (" escn " lines)"; unj = 1 }
  if (blockhdr != "") { print "UNJUDGED: block scalar in a job header:" blockhdr; unj = 1 }
  if (badstep != "") { print "UNJUDGED: step whose first line is not a plain key:" badstep; unj = 1 }
  if (odd != "") { print "UNJUDGED: unsupported job-level shape (flow-style job, or job-key indent differs from the first job's):" odd; unj = 1 }
  if (flow != "") { print "UNJUDGED: jobs with a flow-style or inline steps value:" flow; unj = 1 }
  bad = unj
  if (nsteps + 0 == 0) { print "no steps inspected (empty or unparseable workflow)"; bad = 1 }
  if (sched) { print "schedule trigger present while lane is STOPPED"; bad = 1 }
  if (miss != "") { print "steps without a positive integer timeout-minutes:" miss; bad = 1 }
  if (jmiss != "") { print "jobs without a positive integer timeout-minutes:" jmiss; bad = 1 }
  if (nostep != "") { print "jobs with zero steps inspected:" nostep; bad = 1 }
  if (over != "") { print "step timeouts sum beyond the job timeout:" over; bad = 1 }
  if (!bad) print "inspected " nsteps " steps"
  exit bad ? (unj ? 2 : 1) : 0
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
  local wf=$1 rc=0
  check "$wf" || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS dogfood-nightly stopped and every step has a timeout"
    return 0
  fi
  if [ "$rc" -eq 2 ]; then
    echo "UNJUDGED dogfood-nightly LAB guard (#7277): the workflow uses a YAML form this guard does not judge; rewrite it in plain block style"
    return 2
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
  expect_unj() { # NAME FILE: must exit 2 with an UNJUDGED line and never PASS
    n=$((n + 1)); rc=0
    out=$(bash "$0" "$2" 2>&1) || rc=$?
    if [ "$rc" -ne 2 ] || ! printf '%s\n' "$out" | grep -q '^UNJUDGED ' || printf '%s\n' "$out" | grep -q '^PASS '; then echo "FAIL case $n ($1): not refused as unjudged (rc=$rc)"; fails=$((fails + 1)); fi
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

  # lane-2 FAIL at ce07919: job-key indent, nested timeout keys, escapes, YAML 1.2 forms; the guard refuses them (exit 2)
  printf 'on:\n  workflow_dispatch:\njobs:\n    night:\n        timeout-minutes: 25\n        steps:\n            - uses: a/b@v1\n              timeout-minutes: 10\n            - run: x\n              timeout-minutes: 10\n' > "$t/w1.yml"; expect "wide indent, clean" "$t/w1.yml" pass
  printf 'on:\n  workflow_dispatch:\njobs:\n    night:\n        timeout-minutes: 25\n        steps:\n            - uses: a/b@v1\n            - run: x\n              timeout-minutes: 10\n' > "$t/w2.yml"; expect "wide indent, untimed step" "$t/w2.yml" fail
  { base; printf '   other:\n      timeout-minutes: 5\n      steps:\n        - run: x\n'; } > "$t/w3.yml"; expect_unj "second job with a different key indent" "$t/w3.yml"
  { base; printf '  other:\n      timeout-minutes: 5\n      steps:\n        - run: x\n'; } > "$t/w4.yml"; expect_unj "second job, key indent 6 vs first job 4" "$t/w4.yml"
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      -   uses: a/b@v1\n          with:\n            timeout-minutes: 10\n' > "$t/n1.yml"; expect "timeout nested under with: does not count" "$t/n1.yml" fail
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - uses: a/b@v1\n        env:\n          timeout-minutes: 10\n' > "$t/n2.yml"; expect "timeout nested under env: does not count" "$t/n2.yml" fail
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: |\n          echo hi\n          timeout-minutes: 10\n' > "$t/n3.yml"; expect "timeout text inside a block scalar does not count" "$t/n3.yml" fail
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      -   run: x\n          timeout-minutes: 10\n' > "$t/n4.yml"; expect "wide dash, direct timeout counts" "$t/n4.yml" pass
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: |\n          echo "a # b" != c\n        timeout-minutes: 10\n' > "$t/n5.yml"; expect "block scalar text with # and != is data" "$t/n5.yml" pass
  printf 'on: { x: "a\\"#b", schedule: [] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/e1.yml"; expect_unj "escaped quote in flow on" "$t/e1.yml"
  printf 'on:\n  "sched\\u0075le":\n    - cron: x\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/e2.yml"; expect_unj "unicode escape in a quoted key" "$t/e2.yml"
  { printf '\357\273\277'; base; } > "$t/b1.yml"; expect_unj "byte-order mark" "$t/b1.yml"
  printf 'on:\n  ? schedule\n  : [{cron: x}]\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/q1.yml"; expect_unj "explicit ? key" "$t/q1.yml"
  printf '? on\n: { schedule: [] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/q2.yml"; expect_unj "explicit ? top-level key" "$t/q2.yml"
  printf 'on: {\nschedule: [{cron: x}]\n}\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/f1.yml"; expect_unj "flow continuation at column 0" "$t/f1.yml"
  printf 'on:\n  workflow_dispatch:\njobs: { night: { timeout-minutes: 25, steps: [{run: x}] } }\n' > "$t/f2.yml"; expect_unj "flow-style jobs" "$t/f2.yml"
  printf 'on:\n  workflow_dispatch:\n---\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/f3.yml"; expect_unj "document marker" "$t/f3.yml"
  printf 'on:\n  !!map schedule: []\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/g1.yml"; expect_unj "tag on a key" "$t/g1.yml"
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: !!int 10\n' > "$t/g2.yml"; expect_unj "tag on a timeout value" "$t/g2.yml"
  printf 'on:\n  workflow_dispatch:\nx: &a [1]\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/g3.yml"; expect_unj "anchor" "$t/g3.yml"
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: |\n      25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/h1.yml"; expect_unj "block scalar in a job header" "$t/h1.yml"
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - "run": x\n        timeout-minutes: 10\n' > "$t/h2.yml"; expect_unj "step whose first line is a quoted key" "$t/h2.yml"
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - { run: x }\n' > "$t/h3.yml"; expect_unj "flow-style step" "$t/h3.yml"
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - uses: a/b@v1\n        timeout-minutes: 10\n        runs-on: [x, y]\n' > "$t/h4.yml"; expect "flow list as a plain value is not refused" "$t/h4.yml" pass

  # lane-2 FAIL at 126c2c2: mid-word quote, multi-line flow values, multi-line quoted scalars, quoted "jobs"; all refused (exit 2)
  printf "name: x\non: { workflow_dispatch: { inputs: { a: { description: it's }, b: { default: '}}}} #x' } } }, schedule: [{ cron: '5 20 * * *' }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n" > "$t/m1.yml"; expect "mid-word apostrophe truncating flow on" "$t/m1.yml" fail
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    steps:\n      - run: echo hi\n        env: {\n          timeout-minutes: 10\n        }\n    env: {\n      timeout-minutes: 25\n    }\n' > "$t/m2.yml"; expect_unj "multi-line flow env forging timeouts" "$t/m2.yml"
  printf 'name: "dogfood\n  x"\non:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: "echo hi\n        timeout-minutes: 10\n        "\n' > "$t/m3.yml"; expect_unj "multi-line quoted scalar forging a timeout" "$t/m3.yml"
  printf 'on:\n  workflow_dispatch:\n"jobs":\n  night:\n    steps:\n      - run: echo untimed\n' > "$t/m4.yml"; expect "quoted jobs key with an untimed step" "$t/m4.yml" fail
  printf 'on:\n  workflow_dispatch:\n"jobs":\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n' > "$t/m6.yml"; expect "quoted jobs key is judged, not skipped" "$t/m6.yml" pass
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo it'"'"'s fine\n        timeout-minutes: 10\n' > "$t/m5.yml"; expect "mid-word apostrophe in a plain scalar stays judged" "$t/m5.yml" pass

  # lane-2 FAIL at df2e3e1: hyphen or colon before a mid-word quote, unbalanced closers, tags inside flow collections; all refused (exit 2)
  printf "name: x\non: { workflow_dispatch: { inputs: { a: { default: a-' }, b: { default: '}}}} # ' } } }, schedule: [{ cron: '5 20 * * *' }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n" > "$t/d1.yml"; expect "mid-word hyphen before a quote is not a token start" "$t/d1.yml" fail
  printf "name: x\non: { workflow_dispatch: { inputs: { a: { default: a:\" }, b: { default: '}}}} # ' } } }, schedule: [{ cron: '5 20 * * *' }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n" > "$t/d2.yml"; expect "mid-word colon before a quote is not a token start" "$t/d2.yml" fail
  printf "name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    env: { a: x-', b: '} # ',\n    timeout-minutes: 25\n    }\n    steps:\n      - run: echo untimed\n        env: { a: x-', b: '} # ',\n        timeout-minutes: 10\n        }\n" > "$t/d3.yml"; expect_unj "multi-line flow env forging timeouts" "$t/d3.yml"
  printf 'on:\n  workflow_dispatch:\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n        }\n' > "$t/d4.yml"; expect_unj "closing bracket with no opener" "$t/d4.yml"
  printf 'name: x\nenv: { &s "schedule": "1" }\non: { workflow_dispatch: {}, *s : [{ cron: "5 20 * * *" }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n' > "$t/d5.yml"; expect_unj "anchor and alias inside flow collections" "$t/d5.yml"
  printf 'on:\n  workflow_dispatch: {}\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        env: {"a":!!str b}\n        timeout-minutes: 10\n' > "$t/d6.yml"; expect_unj "tag after a JSON-style colon in flow" "$t/d6.yml"

  # lane-1 FAIL at 4576b77: a space then a quote inside a plain scalar in a flow collection; refused (exit 2)
  printf "name: x\non: { workflow_dispatch: { inputs: { a: { default: a ' }, b: { default: '}}}} # ' } } }, schedule: [{ cron: '5 20 * * *' }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n" > "$t/p1.yml"; expect_unj "space-quote in a plain scalar in flow" "$t/p1.yml"
  printf "name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    env: { a: x ', b: '} # ',\n    timeout-minutes: 25\n    , c: x ', d: '{x' }\n    steps:\n      - run: echo untimed\n        env: { a: x ', b: '} # ',\n        timeout-minutes: 10\n        , c: x ', d: '{x' }\n" > "$t/p2.yml"; expect_unj "multi-line flow env with space-quote forging timeouts" "$t/p2.yml"
  printf "on:\n  workflow_dispatch: {}\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        env: { a: 'q', b: [ 'r', \"s\" ] }\n        timeout-minutes: 10\n" > "$t/p3.yml"; expect "quotes after colon, comma and bracket in flow stay judged" "$t/p3.yml" pass
  printf "on:\n  workflow_dispatch: {}\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo 'a b'\n        if: \${{ github.x == 'y' }}\n        timeout-minutes: 10\n" > "$t/p4.yml"; expect "quotes in block plain scalars stay judged" "$t/p4.yml" pass

  printf "on:\n  workflow_dispatch: {}\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        env: { a: \${{ x ' }}, b: '}} # ' }\n        timeout-minutes: 10\n" > "$t/p5.yml"; expect_unj "expression inside a flow collection" "$t/p5.yml"

  # lane FAIL at e14700c: a dash before space-quote in flow; a colon after a quote that did not close a scalar; refused or judged
  printf "name: x\non: { workflow_dispatch: { inputs: { a: { default: a- ' }, b: { default: '}}}} # ' } } }, schedule: [{ cron: '5 20 * * *' }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n" > "$t/r4.yml"; expect_unj "dash then space-quote in flow" "$t/r4.yml"
  printf "name: x\non: { workflow_dispatch: { inputs: { a: { default: x':' }, b: { default: '}}}} # ' } } }, schedule: [{ cron: '5 20 * * *' }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n" > "$t/r5.yml"; expect "colon after a mid-word quote is not a token start" "$t/r5.yml" fail
  printf 'on:\n  workflow_dispatch: {}\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        env: {"a":"b", "c":\x27d\x27}\n        timeout-minutes: 10\n' > "$t/r6.yml"; expect "JSON-style quoted pairs stay judged" "$t/r6.yml" pass

  printf 'name: x\non: { workflow_dispatch: { inputs: { a: { "default" :"}}}} # " } } }, schedule: [{ cron: "5 20 * * *" }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n' > "$t/r7.yml"; expect "blank before the colon of a JSON-style pair" "$t/r7.yml" fail
  printf "name: x\non: { workflow_dispatch: { inputs: { a: { 'default' :'}}}} # ' } } }, schedule: [{ cron: '5 20 * * *' }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n" > "$t/r8.yml"; expect "blank before the colon, single quotes" "$t/r8.yml" fail
  printf 'name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    env: { "a" :"} # ",\n    timeout-minutes: 25\n    , "b" :"{x" }\n    steps:\n      - run: echo untimed\n        env: { "a" :"} # ",\n        timeout-minutes: 10\n        , "b" :"{x" }\n' > "$t/r9.yml"; expect "blank before the colon, multi-line env forging timeouts" "$t/r9.yml" fail

  printf 'name: x\non: { workflow_dispatch: { inputs: { a: { []:"}}}} # " } } }, schedule: [{ cron: "5 20 * * *" }] }\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: echo ok\n        timeout-minutes: 10\n' > "$t/r10.yml"; expect "flow collection key then adjacent quote" "$t/r10.yml" fail
  printf 'name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    env: { []:"} # ",\n    timeout-minutes: 25\n    , []:"{x" }\n    steps:\n      - run: echo untimed\n        env: { []:"} # ",\n        timeout-minutes: 10\n        , []:"{x" }\n' > "$t/r11.yml"; expect "flow key then adjacent quote, multi-line env" "$t/r11.yml" fail

  printf 'name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    runs-on: ubuntu-latest\n    ${{ a } }: "\n    timeout-minutes: 25\n    " # "\n    steps:\n      - run: echo UNTIMED\n        ${{ a } }: "\n        timeout-minutes: 10\n        " # "\n' > "$t/r12.yml"; expect_unj "unclosed expression opener hiding a multi-line quote" "$t/r12.yml"

  printf "name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    a ' : { b: 'x',\n    timeout-minutes: 25\n    , d: x ' } # '\n    steps:\n      - run: echo untimed\n        a ' : { b: 'x',\n        timeout-minutes: 10\n        , d: x ' } # '\n" > "$t/r13.yml"; expect_unj "space-quote in a block plain key hiding flow brackets" "$t/r13.yml"
  printf 'name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    a " : { b: "x",\n    timeout-minutes: 25\n    , d: x " } # "\n    steps:\n      - run: echo untimed\n        a " : { b: "x",\n        timeout-minutes: 10\n        , d: x " } # "\n' > "$t/r14.yml"; expect_unj "same with double quotes" "$t/r14.yml"

  printf 'name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    runs-on: ubuntu-latest\n    env:\n      ${{ x : " # }}\n    timeout-minutes: 25\n    " # "\n    steps:\n      - run: echo UNTIMED\n        env:\n          ${{ x : " # }}\n        timeout-minutes: 10\n        " # "\n' > "$t/r15.yml"; expect_unj "expression in key position hiding a multi-line quote" "$t/r15.yml"

  printf "name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    runs-on: ubuntu-latest\n    a- ' : [ # '\n    timeout-minutes: 25\n    , a- ' ] # '\n    steps:\n      - run: echo untimed\n        a- ' : [ # '\n        timeout-minutes: 10\n        , a- ' ] # '\n" > "$t/r16.yml"; expect_unj "word-dash-space-quote in block context" "$t/r16.yml"
  printf "on:\n  workflow_dispatch: {}\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - 'run': x\n        timeout-minutes: 10\n" > "$t/r17.yml"; expect_unj "quoted key after a block dash is judged as a quoted first key" "$t/r17.yml"

  printf 'name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    runs-on: ubuntu-latest\n    a- ${{ x : " # }}\n    timeout-minutes: 25\n    " # "\n    steps:\n      - run: echo UNTIMED\n        a- ${{ x : " # }}\n        timeout-minutes: 10\n        " # "\n' > "$t/r18.yml"; expect_unj "expression after a dash inside a plain key" "$t/r18.yml"
  printf 'name: x\non:\n  workflow_dispatch:\njobs:\n  night:\n    runs-on: ubuntu-latest\n    steps:\n      - ${{ x : " # }}\n        timeout-minutes: 10\n        " # "\n' > "$t/r19.yml"; expect_unj "expression as a compact key after a block dash" "$t/r19.yml"
  printf 'name: x\non:\n  workflow_dispatch:\n    inputs:\n      a - ${{ : " # }}\nfake: 1\n      " # "\n  schedule:\n    - cron: "5 20 * * *"\njobs:\n  night:\n    timeout-minutes: 25\n    steps:\n      - run: x\n        timeout-minutes: 10\n' > "$t/r20.yml"; expect "expression in a plain key hiding schedule" "$t/r20.yml" fail

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
