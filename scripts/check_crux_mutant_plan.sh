#!/usr/bin/env bash
# check_crux_mutant_plan.sh: the case table for scripts/lib/crux_mutant_plan.py (#4098), the planner that decides
# which F6 mutants scripts/check_crux_inference_judge.sh runs. A sampling rule is a guard: each row below is a
# must-match or a must-refuse, and the last row proves the table itself can see a planner that never samples.
#
# Rows:
#   1. no CI event (a local run)            → all 24
#   2. schedule / workflow_dispatch        → all
#   3. pull_request, diff untouched        → exactly 6, deterministic for a head, a different slice for another head
#   4. pull_request, the judge touched     → all, the reason names the file
#   5. every WATCHED file touched alone    → all (none of them can be dropped from the list silently)
#   6. pull_request, diff unreadable       → the sample, its reason naming the unreadable diff
#   7. rotation: the slices of 24 consecutive offsets together cover every mutant
#   8. refusals (exit 2): a sample under the floor, an unknown CRUX_MUTANTS, an unknown event, no labels
#   9. the judge table refuses a thin run: CRUX_MIN_TABLE_ROWS above what it judged → a BROKE naming the floor
#  10. MUTANT: WATCHED emptied → row 4 sees a sample where it must see all
#  11. the REAL table in fabricated repos: a push and a merge_group head that touch the judge → all 24;
#      an untouched push samples with its diff READ (never 'could not be read'); the table still names itself
#  12. the sourced diff lib leaves its caller's PROG and REPO_ROOT alone
#  13. CRUX_MUTANTS_PLAN_ONLY is never a verdict: it says 0 ran and exits 3
#  14. a self-referential CRUX_MUTANT_DIFF_BASE is an unreadable diff, never 'untouched'
#  15. no runner setting makes a zero-mutant run pass: floor 0, CRUX_NO_MUTANTS, CRUX_MUTANTS=none,
#      CRUX_JUDGE_OVERRIDE (+ CRUX_NO_MUTANTS), CRUX_MIN_TABLE_ROWS=0
#
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_mutant_plan
PLAN="$ROOT/scripts/lib/crux_mutant_plan.py"
command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 is missing\n' "$PROG" >&2; exit 2; }
[ -f "$PLAN" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$PLAN" >&2; exit 2; }
TMP=$(mktemp -d) || exit 2
_rm_tmp() {
  case "${TMP:-}" in
    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
    *) : ;;
  esac
}
trap _rm_tmp EXIT

PASS=0
FAIL=0
ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }

LABELS=$(seq -f 'm%02g' 1 24 | paste -sd,)
HEAD_A=375b34522aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
HEAD_B=c460c63a7bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
: > "$TMP/none.txt"
printf 'scripts/lib/crux_inference_judge.py\nREADME.md\n' > "$TMP/judge.txt"

# pl <planner> <args...> -> "<mode> <n selected> <selected csv>|<reason>" or "REFUSED <rc>"
pl() {
  local p="$1"; shift
  local out rc
  out=$(python3 "$p" --labels "$LABELS" "$@" 2> /dev/null); rc=$?
  [ "$rc" -eq 0 ] || { echo "REFUSED $rc"; return; }
  python3 -c 'import json,sys; p=json.loads(sys.argv[1]); print("%s %d %s|%s" % (p["mode"], len(p["selected"]), ",".join(p["selected"]), p["reason"]))' "$out"
}

printf '%s: the F6 mutant planner (#4098)\n' "$PROG"

r=$(pl "$PLAN" --head "$HEAD_A")
case "$r" in "all 24 "*) ok "no CI event: all 24" ;; *) broke "no event: $r" ;; esac

a=$(pl "$PLAN" --event schedule --head "$HEAD_A"); b=$(pl "$PLAN" --event workflow_dispatch --head "$HEAD_A")
case "$a|$b" in "all 24 "*"|all 24 "*) ok "schedule and workflow_dispatch: all 24" ;; *) broke "full events: $a / $b" ;; esac

a=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A")
a2=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A")
b=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_B")
if [ "${a%%|*}" = "${a2%%|*}" ] && [ "${a%%|*}" != "${b%%|*}" ]; then
  case "$a|$b" in "sample 6 "*"|sample 6 "*) ok "pull_request, untouched: 6, the same for one head, another slice for another head" ;;
    *) broke "untouched sample: $a / $b" ;; esac
else
  broke "sample not deterministic per head or not rotating: $a / $a2 / $b"
fi

r=$(pl "$PLAN" --event pull_request --changed "$TMP/judge.txt" --head "$HEAD_A")
case "$r" in "all 24 "*"crux_inference_judge.py"*) ok "pull_request, the judge touched: all 24, the file named" ;; *) broke "judge touched: $r" ;; esac

miss=""
for f in $(python3 -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("p", sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(" ".join(m.WATCHED))' "$PLAN"); do
  printf '%s\n' "$f" > "$TMP/one.txt"
  r=$(pl "$PLAN" --event merge_group --changed "$TMP/one.txt" --head "$HEAD_A")
  case "$r" in "all 24 "*) ;; *) miss="$miss $f" ;; esac
done
n_watched=$(python3 -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("p", sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(len(m.WATCHED))' "$PLAN")
if [ -z "$miss" ] && [ "$n_watched" -ge 6 ]; then ok "each of the $n_watched watched files, touched alone, runs all 24"
else broke "watched files that did not force all: ${miss:-none} (watched: $n_watched)"; fi

r=$(pl "$PLAN" --event push --head "$HEAD_A")
case "$r" in "sample 6 "*"could not be read"*) ok "an unreadable diff is named, never read as untouched; the sample still runs" ;; *) broke "unreadable diff: $r" ;; esac

cover=$(for off in $(seq 0 23); do
  h=$(printf '%08x' "$off")aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$h" | cut -d' ' -f3 | cut -d'|' -f1 | tr ',' '\n'
done | sort -u | wc -l)
[ "$cover" -eq 24 ] && ok "24 consecutive offsets cover every mutant (none skipped for ever)" || broke "rotation covered $cover of 24"

r1=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A" --sample 3 --floor 6)
r2=$(pl "$PLAN" --mode bogus --head "$HEAD_A")
r3=$(pl "$PLAN" --event release --head "$HEAD_A")
r4=$(python3 "$PLAN" --labels "" > /dev/null 2>&1; echo "REFUSED $?")
if [ "$r1|$r2|$r3|$r4" = "REFUSED 2|REFUSED 2|REFUSED 2|REFUSED 2" ]; then
  ok "refused (exit 2): a sample under the floor, an unknown CRUX_MUTANTS, an unknown event, no labels"
else
  broke "refusals: floor $r1, mode $r2, event $r3, labels $r4"
fi

out=$(CRUX_MUTANTS=none CRUX_MIN_TABLE_ROWS=100000 timeout 600 bash "$ROOT/scripts/check_crux_inference_judge.sh" 2>&1); rc=$?
case "$rc:$out" in
  1:*"under the floor of 100000"*) ok "the judge table refuses a thin run: a floor above what it judged is a BROKE" ;;
  *) broke "table floor: rc $rc, $(printf '%s' "$out" | tail -2 | tr '\n' ' ')" ;;
esac

# Rows 11-13: the REAL table in fabricated repos, one per CI shape, so the diff the table READS is tested, not a
# hand-built file. GITHUB_BASE_REF exists only on pull_request; a push or merge group that touched the judge once
# sampled because its diff was never read (quorum round 1, lane 1, measured). CRUX_MUTANTS_PLAN_ONLY stops after
# the plan line.
# The fixture commits bypass hooks: a global pre-commit hook (pmat complexity) refused them, so no HEAD existed.
fab() { # fab <dir>: a repo holding the scripts and the golden file the table reads, one commit, origin/main on it
  local d="$1"
  mkdir -p "$d/crates/apr-cli/src/commands"
  cp -r "$ROOT/scripts" "$d/scripts"
  cp "$ROOT/crates/apr-cli/src/commands/golden_output.rs" "$d/crates/apr-cli/src/commands/"
  git -C "$d" init -q && git -C "$d" add -A && git -C "$d" -c user.email=t@t -c user.name=t -c core.hooksPath=/dev/null commit --no-verify -q -m base \
    && git -C "$d" update-ref refs/remotes/origin/main HEAD
}
touch_commit() { # touch_commit <dir> <path>: one commit that changes <path>
  printf '\n# touched\n' >> "$1/$2"
  git -C "$1" add -A && git -C "$1" -c user.email=t@t -c user.name=t -c core.hooksPath=/dev/null commit --no-verify -q -m "touch $2"
}
plan_line() { # plan_line <dir> <event>: the table's own plan line, and its summary line (the name it reports)
  ( cd "$1" && env -u GITHUB_BASE_REF -u CRUX_MUTANT_DIFF_BASE GITHUB_EVENT_NAME="$2" CRUX_MUTANTS_PLAN_ONLY=1 \
      timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 | grep -E 'F6 mutants:|ok, [0-9]+ broke$' | tr '\n' ' ' )
}
P1="$TMP/push-judge"; fab "$P1"; touch_commit "$P1" scripts/lib/crux_inference_judge.py
git -C "$P1" update-ref refs/remotes/origin/main HEAD          # a push to main: HEAD is the origin/main tip
l1=$(plan_line "$P1" push)
P2="$TMP/mg-judge"; fab "$P2"; touch_commit "$P2" scripts/lib/crux_inference_judge.py
l2=$(plan_line "$P2" merge_group)                              # the queue head: single parent == origin/main tip
P3="$TMP/push-readme"; fab "$P3"; touch_commit "$P3" scripts/guards_nightly_manifest.txt
git -C "$P3" update-ref refs/remotes/origin/main HEAD
l3=$(plan_line "$P3" push)
case "$l1|$l2|$l3" in
  *"all, 24 of 24"*"crux_inference_judge.py"*" check_crux_inference_judge: "*"|"*"all, 24 of 24"*"crux_inference_judge.py"*" check_crux_inference_judge: "*"|"*"sample, 6 of 24"*"touches none"*" check_crux_inference_judge: "*)
    ok "the REAL table reads its diff on push and merge_group: the judge touched → all 24; an untouched push samples, diff read; the table still names itself" ;;
  *) broke "table-read diff: push '$l1' / merge_group '$l2' / untouched push '$l3'" ;;
esac

# Row 13: CRUX_MUTANTS_PLAN_ONLY is never a verdict: the plan line says 0 ran and the table exits 3, even when a
# runner sets it on a full run (it once printed "all, 24 of 24" and exited 0 with no mutant run).
out=$( cd "$P1" && CRUX_MUTANTS=all CRUX_MUTANTS_PLAN_ONLY=1 timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 ); rc=$?
case "$rc:$out" in
  3:*"PLAN ONLY -- 0 run"*"no mutant ran, exit 3"*) ok "a plan-only run says 0 ran and exits 3, never a pass" ;;
  *) broke "plan-only run: rc $rc, $(printf '%s' "$out" | grep -E 'F6 mutants|exit 3|ok, ' | tr '\n' ' ')" ;;
esac

# Row 14: a self-referential CRUX_MUTANT_DIFF_BASE (HEAD) is an UNREADABLE diff, never "untouched" (it once diffed
# the tree against itself and read "the diff touches none of the mutants' files").
s14=$( . "$ROOT/scripts/lib/crux_mutant_plan.sh"; CRUX_MUTANT_DIFF_BASE=HEAD crux_mutant_changed "$P1" "$TMP/s14.txt"; echo $? )
s14b=$( . "$ROOT/scripts/lib/crux_mutant_plan.sh"; CRUX_MUTANT_DIFF_BASE=HEAD~1 crux_mutant_changed "$P1" "$TMP/s14b.txt"; echo "$?:$(grep -c crux_inference_judge "$TMP/s14b.txt")" )
[ "$s14|$s14b" = "1|0:1" ] && ok "CRUX_MUTANT_DIFF_BASE=HEAD is refused (unreadable); HEAD~1 reads the judge change" \
  || broke "diff-base override: HEAD rc $s14 (want 1), HEAD~1 $s14b (want 0:1)"

# Row 15: nothing a runner can set turns a zero-mutant run into a pass (quorum round 4, lane 2, each measured at
# exit 0): a floor of 0 is refused by the planner; a top-level CRUX_NO_MUTANTS=1 and CRUX_MUTANTS=none each exit 3
# naming why; CRUX_MIN_TABLE_ROWS=0 cannot lower the row floor.
r0=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A" --sample 0 --floor 0)
( cd "$P1" && CRUX_NO_MUTANTS=1 timeout 600 bash scripts/check_crux_inference_judge.sh > "$TMP/r15a.log" 2>&1 ); ra=$?
( cd "$P1" && CRUX_MUTANTS=none timeout 600 bash scripts/check_crux_inference_judge.sh > "$TMP/r15b.log" 2>&1 ); rb=$?
( cd "$P1" && CRUX_JUDGE_OVERRIDE="$P1/scripts/lib/crux_inference_judge.py" CRUX_NO_MUTANTS=1 timeout 600 bash scripts/check_crux_inference_judge.sh > "$TMP/r15c.log" 2>&1 ); rc_=$?
if [ "$r0" = "REFUSED 2" ] && [ "$ra" = 3 ] && grep -q 'CRUX_NO_MUTANTS is set: no mutant ran, exit 3' "$TMP/r15a.log" \
   && [ "$rb" = 3 ] && grep -q 'the plan selected no mutant (CRUX_MUTANTS=none): no mutant ran, exit 3' "$TMP/r15b.log" \
   && [ "$rc_" = 3 ] && grep -q 'CRUX_JUDGE_OVERRIDE is set' "$TMP/r15c.log"; then
  ok "no runner setting makes a zero-mutant run pass: floor 0 refused; CRUX_NO_MUTANTS, CRUX_MUTANTS=none and CRUX_JUDGE_OVERRIDE+CRUX_NO_MUTANTS exit 3 naming why"
else
  broke "zero-mutant escapes: floor0 '$r0', CRUX_NO_MUTANTS rc $ra, none rc $rb, override+no-mutants rc $rc_"
fi
out=$( cd "$P1" && CRUX_MUTANTS_PLAN_ONLY=1 CRUX_MIN_TABLE_ROWS=0 timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 | grep -c 'under the floor' )
out2=$( cd "$P1" && CRUX_MUTANTS_PLAN_ONLY=1 CRUX_MIN_TABLE_ROWS=100000 timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 | grep -c 'under the floor of 100000' )
[ "$out:$out2" = "0:1" ] && ok "CRUX_MIN_TABLE_ROWS raises the row floor but 0 cannot lower it below 120 (the table stays above it)" \
  || broke "row floor env: lowering '$out' (want 0 BROKE since the table is above 120), raising '$out2' (want 1)"

# Row 12: the sourced lib leaves its caller's globals alone (it once set PROG and REPO_ROOT in the caller's shell).
g=$( PROG=caller-prog; REPO_ROOT=caller-root; . "$ROOT/scripts/lib/crux_mutant_plan.sh" \
     && crux_mutant_changed "$P1" "$TMP/g.txt"; printf '%s|%s|%s' "$PROG" "$REPO_ROOT" "$(grep -c crux_inference_judge "$TMP/g.txt" 2>/dev/null)" )
[ "$g" = "caller-prog|caller-root|1" ] && ok "the sourced diff lib leaves the caller's PROG and REPO_ROOT alone, and still reads the diff" \
  || broke "caller globals after crux_mutant_changed: $g (want caller-prog|caller-root|1)"

python3 - "$PLAN" "$TMP/mutant-plan.py" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r"WATCHED = \((.*?)\n\)", s, re.S)
assert m, "WATCHED anchor moved: update this check with the planner"
open(sys.argv[2], "w").write(s[:m.start()] + "WATCHED = ()" + s[m.end():])
PY
r=$(pl "$TMP/mutant-plan.py" --event pull_request --changed "$TMP/judge.txt" --head "$HEAD_A")
case "$r" in "all 24 "*) broke "MUTANT (WATCHED emptied) not caught: $r" ;; *) ok "MUTANT (WATCHED emptied) caught: the judge-touched PR samples ($r)" ;; esac

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
