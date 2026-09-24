You are one of 2 independent reviewers. Judge whether this diff does what its ticket says, and nothing the ticket forbids. Try to REFUTE it: default to FAIL when a test asserts the opposite of the ticket, when a gate is weakened, when a receipt claim is not backed by the diff, or when the change does something the ticket does not ask for. Every finding needs file, line, claim and grounding (cited = you quote the diff; measured = you ran a command; asserted = neither). Return PASS only if you found nothing that refutes it.

## Ticket(s) PMAT-4098 — the diff is judged against ALL of them
### PMAT-4098
📊 Status for: PMAT-4098

   Title: check_crux_inference_judge.sh: CI-fast mode — sampled F6 mutants on PRs, the full set nightly
   Status: InProgress
   Priority: High
   Progress: 50%
   GitHub: #4098



## Diff (origin/chore/0.69.1-merge-back...HEAD)
```diff
diff --git a/.github/workflows/guards-nightly.yml b/.github/workflows/guards-nightly.yml
index 9ac9ed527..84bfc467f 100644
--- a/.github/workflows/guards-nightly.yml
+++ b/.github/workflows/guards-nightly.yml
@@ -170,6 +170,14 @@ jobs:
         run: bash scripts/check_hermetic_stdin_tests.sh
       - name: Publish safety (binary/oversize files in published packages)
         run: bash scripts/check_publish_safety.sh
+      # #4098: the CRUX judge table's F6 mutants each re-run the whole table (655 s in all on
+      # lambda). On a PR, merge group or push the table runs a rotating sample of them
+      # (scripts/lib/crux_mutant_plan.py; every mutant when the diff touches the judge), so
+      # the WHOLE set runs here, once a day, and fails the job like any guard.
+      - name: CRUX judge, every F6 mutant (#4098)
+        env:
+          CRUX_MUTANTS: all
+        run: bash scripts/check_crux_inference_judge.sh
       - name: Perf gate can still discriminate (APR-PERF-GATE-001, case table)
         run: bash scripts/perf_gate.sh --selftest
       - name: Book rust examples compile
diff --git a/docs/roadmaps/roadmap.yaml b/docs/roadmaps/roadmap.yaml
index ca3ad7c2c..a3f9f5fa7 100644
--- a/docs/roadmaps/roadmap.yaml
+++ b/docs/roadmaps/roadmap.yaml
@@ -21444,3 +21444,24 @@ roadmap:
   estimated_effort: null
   labels: []
   notes: null
+- id: PMAT-4098
+  github_issue: 4098
+  item_type: task
+  title: 'check_crux_inference_judge.sh: CI-fast mode — sampled F6 mutants on PRs, the full set nightly'
+  status: in_progress
+  priority: high
+  assigned_to: aprender-83
+  created: 2026-09-24T00:30:00Z
+  updated: 2026-09-24T01:30:00Z
+  spec: null
+  acceptance_criteria:
+  - 'On a PR, merge group or push the table runs a deterministic sample of the F6 mutants: every mutant when the diff touches a file they test, else a rotating slice keyed by the head sha'
+  - 'The PR mode refuses vacuously small runs: a floor on the mutants sampled and on the rows the table judged'
+  - 'The full set runs nightly (guards-nightly.yml) and fails the job loudly'
+  - 'Not in scope: raising guard-tree''s timeout'
+  phases: []
+  subtasks: []
+  estimated_effort: null
+  labels:
+  - kind:code
+  notes: null
diff --git a/scripts/check_crux_inference_judge.sh b/scripts/check_crux_inference_judge.sh
index 3068ade54..78eb473b2 100755
--- a/scripts/check_crux_inference_judge.sh
+++ b/scripts/check_crux_inference_judge.sh
@@ -1074,17 +1074,28 @@ d=$(b2 b2_floored_cut "Two and two make <answer>4</answer>" "$LL_CLOSED" "x$(pri
 expect "B2: a rendering floored to a char boundary (199 bytes) is NOT read as whole -- unknown, so RED by name" "$d" 1 $P RED
 
 # ── #3957 F6 MUTANTS. Each rule deleted in a copy of the judge; the WHOLE table must then break.
+# #4098: each mutant re-runs the whole table (~26 s; the full set ~630 of 655 s), so which mutants run is planned by
+# scripts/lib/crux_mutant_plan.py: every mutant locally, on schedule/dispatch and in guards-nightly.yml
+# (CRUX_MUTANTS=all); on a PR, merge group or push, every mutant when the diff touches a file they test, else a
+# rotating slice keyed by the head sha. A table that judged too few rows is refused first, sampled or not.
+# A run that runs ZERO mutants, or judges anything but the real judge, is never a verdict: it says why and exits 3
+# after its summary (a broken row still exits 1 first). This holds for the mutant loop's own recursion too — the loop
+# grades each mutant by its BROKE lines and never reads the child's exit code, so nothing is lost, and there is no
+# "am I the recursion?" test to spoof (quorum round 4, lane 2: CRUX_NO_MUTANTS / CRUX_MUTANTS=none / a floor of 0
+# each exited 0 with no mutant; round 5, lane 2: CRUX_JUDGE_OVERRIDE, the proxy used for "recursion", was settable
+# by anyone and made the same run pass silently).
+NOT_A_VERDICT=""
+[ -n "${CRUX_JUDGE_OVERRIDE:-}" ] && NOT_A_VERDICT="CRUX_JUDGE_OVERRIDE is set: the judge under test is $JUDGE, not the real one"
+[ -n "${CRUX_NO_MUTANTS:-}" ] && NOT_A_VERDICT="${NOT_A_VERDICT:+$NOT_A_VERDICT; }CRUX_NO_MUTANTS is set"
+# the row floor applies to every run; the env var may RAISE it, never lower it
+MIN_ROWS=120
+[ "${CRUX_MIN_TABLE_ROWS:-0}" -gt "$MIN_ROWS" ] 2>/dev/null && MIN_ROWS=$CRUX_MIN_TABLE_ROWS
+if [ "$PASS" -lt "$MIN_ROWS" ]; then
+  broke "the table judged only $PASS row(s) ok before the mutants, under the floor of $MIN_ROWS: a thin table proves nothing"
+fi
 if [ -z "${CRUX_NO_MUTANTS:-}" ]; then
-  # label|the row that MUST break (#3887: a kill for the wrong reason is no kill)|sed deleting the rule
-  while IFS='|' read -r label must expr; do
-    [ -n "$label" ] || continue
-    m="$TMP/mut-$label.py"; sed "$expr" "$JUDGE" > "$m"
-    if cmp -s "$JUDGE" "$m"; then broke "F6 mutant $label did not apply"; continue; fi
-    CRUX_JUDGE_OVERRIDE="$m" CRUX_NO_MUTANTS=1 bash "$ROOT/scripts/check_crux_inference_judge.sh" > "$TMP/mut-$label.log" 2>&1
-    nb=$(grep -c '^  BROKE' "$TMP/mut-$label.log")
-    if grep -q "^  BROKE.*$must" "$TMP/mut-$label.log"; then ok "F6 mutant $label killed by '$must' ($nb row(s) broke)"
-    else broke "F6 mutant $label SURVIVED: '$must' stayed ok ($nb other row(s) broke)"; fi
-  done <<'MUT'
+  mutant_list() {
+    cat <<'MUT'
 bad-control-ignored|the only control a token loop|s/^        if bad:$/        if False:/
 no-control-ok|is no control: RED|s/^    if not ctl:$/    if False:/
 split-ignored|is a SPLIT|s/^        elif len(set(vals.values())) > 1 or None in vals.values():$/        elif False:/
@@ -1110,7 +1121,48 @@ admission-mode-off|admitted only for thinking ON|s/^        if admitted_mode is
 admission-off|NOT admitted for this model is RED|s/^        elif admitted is not None and k\[5\] not in admitted.get(k\[0\], ()):$/        elif False:/
 certification-off|no certification receipt declines|s/^        certified = certification_ok(args.prompts, getattr(args, "certification", None))$/        certified = True/
 MUT
+  }
+  MUTANTS=$(mutant_list)
+  labels=$(printf '%s\n' "$MUTANTS" | cut -d'|' -f1 | paste -sd,)
+  changed_arg=()
+  # shellcheck source=scripts/lib/crux_mutant_plan.sh
+  if . "$ROOT/scripts/lib/crux_mutant_plan.sh" && crux_mutant_changed "$ROOT" "$TMP/changed.txt"; then
+    changed_arg=(--changed "$TMP/changed.txt")
+  fi
+  plan=$(python3 "$ROOT/scripts/lib/crux_mutant_plan.py" --labels "$labels" --event "${GITHUB_EVENT_NAME:-}" \
+    --mode "${CRUX_MUTANTS:-}" --head "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)" \
+    --sample "${CRUX_MUTANT_SAMPLE:-6}" --floor "${CRUX_MUTANT_FLOOR:-6}" "${changed_arg[@]}" 2> "$TMP/plan.err")
+  if [ -z "$plan" ]; then
+    broke "F6 mutant plan refused: $(cat "$TMP/plan.err")"
+    selected=""
+  else
+    printf '  F6 mutants: %s\n' "$(python3 -c 'import json,sys; p=json.loads(sys.argv[1]); print("%s, %d of %d -- %s" % (p["mode"], len(p["selected"]), p["total"], p["reason"]))' "$plan")"
+    selected=$(python3 -c 'import json,sys; print(" ".join(json.loads(sys.argv[1])["selected"]))' "$plan")
+  fi
+  # CRUX_MUTANTS_PLAN_ONLY=1: the table and its plan line, no mutant (check_crux_mutant_plan.sh drives the plan
+  # through this very script in fabricated push / merge_group repos). It is NEVER a verdict: it says so and exits 3
+  # below, so a runner that set it cannot pass a table that ran no mutant (quorum round 3, lane 2, measured).
+  if [ -n "${CRUX_MUTANTS_PLAN_ONLY:-}" ]; then
+    printf '  F6 mutants: PLAN ONLY -- 0 run; this is not a verdict\n'
+    selected=""
+    NOT_A_VERDICT="CRUX_MUTANTS_PLAN_ONLY is set"
+  fi
+  [ -n "$plan" ] && [ -z "$selected" ] && [ -z "$NOT_A_VERDICT" ] && NOT_A_VERDICT="the plan selected no mutant (${CRUX_MUTANTS:+CRUX_MUTANTS=$CRUX_MUTANTS})"
+  # label|the row that MUST break (#3887: a kill for the wrong reason is no kill)|sed deleting the rule
+  while IFS='|' read -r label must expr; do
+    [ -n "$label" ] || continue
+    case " $selected " in *" $label "*) ;; *) continue ;; esac
+    m="$TMP/mut-$label.py"; sed "$expr" "$JUDGE" > "$m"
+    if cmp -s "$JUDGE" "$m"; then broke "F6 mutant $label did not apply"; continue; fi
+    CRUX_JUDGE_OVERRIDE="$m" CRUX_NO_MUTANTS=1 bash "$ROOT/scripts/check_crux_inference_judge.sh" > "$TMP/mut-$label.log" 2>&1
+    nb=$(grep -c '^  BROKE' "$TMP/mut-$label.log")
+    if grep -q "^  BROKE.*$must" "$TMP/mut-$label.log"; then ok "F6 mutant $label killed by '$must' ($nb row(s) broke)"
+    else broke "F6 mutant $label SURVIVED: '$must' stayed ok ($nb other row(s) broke)"; fi
+  done <<< "$MUTANTS"
 fi
 
 printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
-[ "$FAIL" -eq 0 ]
+# a broken row is a FAIL whatever else is true; only a run with nothing broken can be "not a verdict"
+[ "$FAIL" -eq 0 ] || exit 1
+[ -z "$NOT_A_VERDICT" ] || { printf '%s: %s: no mutant ran, exit 3 (never a pass)\n' "$PROG" "$NOT_A_VERDICT"; exit 3; }
+exit 0
diff --git a/scripts/check_crux_mutant_plan.sh b/scripts/check_crux_mutant_plan.sh
new file mode 100755
index 000000000..7cf388093
--- /dev/null
+++ b/scripts/check_crux_mutant_plan.sh
@@ -0,0 +1,223 @@
+#!/usr/bin/env bash
+# check_crux_mutant_plan.sh: the case table for scripts/lib/crux_mutant_plan.py (#4098), the planner that decides
+# which F6 mutants scripts/check_crux_inference_judge.sh runs. A sampling rule is a guard: each row below is a
+# must-match or a must-refuse, and the last row proves the table itself can see a planner that never samples.
+#
+# Rows:
+#   1. no CI event (a local run)            → all 24
+#   2. schedule / workflow_dispatch        → all
+#   3. pull_request, diff untouched        → exactly 6, deterministic for a head, a different slice for another head
+#   4. pull_request, the judge touched     → all, the reason names the file
+#   5. every WATCHED file touched alone    → all (none of them can be dropped from the list silently)
+#   6. pull_request, diff unreadable       → the sample, its reason naming the unreadable diff
+#   7. rotation: the slices of 24 consecutive offsets together cover every mutant
+#   8. refusals (exit 2): a sample under the floor, an unknown CRUX_MUTANTS, an unknown event, no labels
+#   9. the judge table refuses a thin run: CRUX_MIN_TABLE_ROWS above what it judged → a BROKE naming the floor
+#  10. MUTANT: WATCHED emptied → row 4 sees a sample where it must see all
+#  11. the REAL table in fabricated repos: a push and a merge_group head that touch the judge → all 24;
+#      an untouched push samples with its diff READ (never 'could not be read'); the table still names itself
+#  12. the sourced diff lib leaves its caller's PROG and REPO_ROOT alone
+#  13. CRUX_MUTANTS_PLAN_ONLY is never a verdict: it says 0 ran and exits 3
+#  14. a self-referential CRUX_MUTANT_DIFF_BASE is an unreadable diff, never 'untouched'
+#  15. no runner setting makes a zero-mutant run pass: floor 0, CRUX_NO_MUTANTS, CRUX_MUTANTS=none,
+#      CRUX_JUDGE_OVERRIDE (+ CRUX_NO_MUTANTS), CRUX_MIN_TABLE_ROWS=0
+#  16. every floor 1-5 refused, 6 accepted; a MUTANT off-by-one bound (floor < 1) is seen
+#
+# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
+set -uo pipefail
+
+ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
+PROG=check_crux_mutant_plan
+PLAN="$ROOT/scripts/lib/crux_mutant_plan.py"
+command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 is missing\n' "$PROG" >&2; exit 2; }
+[ -f "$PLAN" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$PLAN" >&2; exit 2; }
+TMP=$(mktemp -d) || exit 2
+_rm_tmp() {
+  case "${TMP:-}" in
+    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
+    *) : ;;
+  esac
+}
+trap _rm_tmp EXIT
+
+PASS=0
+FAIL=0
+ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
+broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }
+
+LABELS=$(seq -f 'm%02g' 1 24 | paste -sd,)
+HEAD_A=375b34522aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
+HEAD_B=c460c63a7bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
+: > "$TMP/none.txt"
+printf 'scripts/lib/crux_inference_judge.py\nREADME.md\n' > "$TMP/judge.txt"
+
+# pl <planner> <args...> -> "<mode> <n selected> <selected csv>|<reason>" or "REFUSED <rc>"
+pl() {
+  local p="$1"; shift
+  local out rc
+  out=$(python3 "$p" --labels "$LABELS" "$@" 2> /dev/null); rc=$?
+  [ "$rc" -eq 0 ] || { echo "REFUSED $rc"; return; }
+  python3 -c 'import json,sys; p=json.loads(sys.argv[1]); print("%s %d %s|%s" % (p["mode"], len(p["selected"]), ",".join(p["selected"]), p["reason"]))' "$out"
+}
+
+printf '%s: the F6 mutant planner (#4098)\n' "$PROG"
+
+r=$(pl "$PLAN" --head "$HEAD_A")
+case "$r" in "all 24 "*) ok "no CI event: all 24" ;; *) broke "no event: $r" ;; esac
+
+a=$(pl "$PLAN" --event schedule --head "$HEAD_A"); b=$(pl "$PLAN" --event workflow_dispatch --head "$HEAD_A")
+case "$a|$b" in "all 24 "*"|all 24 "*) ok "schedule and workflow_dispatch: all 24" ;; *) broke "full events: $a / $b" ;; esac
+
+a=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A")
+a2=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A")
+b=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_B")
+if [ "${a%%|*}" = "${a2%%|*}" ] && [ "${a%%|*}" != "${b%%|*}" ]; then
+  case "$a|$b" in "sample 6 "*"|sample 6 "*) ok "pull_request, untouched: 6, the same for one head, another slice for another head" ;;
+    *) broke "untouched sample: $a / $b" ;; esac
+else
+  broke "sample not deterministic per head or not rotating: $a / $a2 / $b"
+fi
+
+r=$(pl "$PLAN" --event pull_request --changed "$TMP/judge.txt" --head "$HEAD_A")
+case "$r" in "all 24 "*"crux_inference_judge.py"*) ok "pull_request, the judge touched: all 24, the file named" ;; *) broke "judge touched: $r" ;; esac
+
+miss=""
+for f in $(python3 -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("p", sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(" ".join(m.WATCHED))' "$PLAN"); do
+  printf '%s\n' "$f" > "$TMP/one.txt"
+  r=$(pl "$PLAN" --event merge_group --changed "$TMP/one.txt" --head "$HEAD_A")
+  case "$r" in "all 24 "*) ;; *) miss="$miss $f" ;; esac
+done
+n_watched=$(python3 -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("p", sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(len(m.WATCHED))' "$PLAN")
+if [ -z "$miss" ] && [ "$n_watched" -ge 6 ]; then ok "each of the $n_watched watched files, touched alone, runs all 24"
+else broke "watched files that did not force all: ${miss:-none} (watched: $n_watched)"; fi
+
+r=$(pl "$PLAN" --event push --head "$HEAD_A")
+case "$r" in "sample 6 "*"could not be read"*) ok "an unreadable diff is named, never read as untouched; the sample still runs" ;; *) broke "unreadable diff: $r" ;; esac
+
+cover=$(for off in $(seq 0 23); do
+  h=$(printf '%08x' "$off")aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
+  pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$h" | cut -d' ' -f3 | cut -d'|' -f1 | tr ',' '\n'
+done | sort -u | wc -l)
+[ "$cover" -eq 24 ] && ok "24 consecutive offsets cover every mutant (none skipped for ever)" || broke "rotation covered $cover of 24"
+
+r1=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A" --sample 3 --floor 6)
+r2=$(pl "$PLAN" --mode bogus --head "$HEAD_A")
+r3=$(pl "$PLAN" --event release --head "$HEAD_A")
+r4=$(python3 "$PLAN" --labels "" > /dev/null 2>&1; echo "REFUSED $?")
+if [ "$r1|$r2|$r3|$r4" = "REFUSED 2|REFUSED 2|REFUSED 2|REFUSED 2" ]; then
+  ok "refused (exit 2): a sample under the floor, an unknown CRUX_MUTANTS, an unknown event, no labels"
+else
+  broke "refusals: floor $r1, mode $r2, event $r3, labels $r4"
+fi
+
+out=$(CRUX_MUTANTS=none CRUX_MIN_TABLE_ROWS=100000 timeout 600 bash "$ROOT/scripts/check_crux_inference_judge.sh" 2>&1); rc=$?
+case "$rc:$out" in
+  1:*"under the floor of 100000"*) ok "the judge table refuses a thin run: a floor above what it judged is a BROKE" ;;
+  *) broke "table floor: rc $rc, $(printf '%s' "$out" | tail -2 | tr '\n' ' ')" ;;
+esac
+
+# Rows 11-13: the REAL table in fabricated repos, one per CI shape, so the diff the table READS is tested, not a
+# hand-built file. GITHUB_BASE_REF exists only on pull_request; a push or merge group that touched the judge once
+# sampled because its diff was never read (quorum round 1, lane 1, measured). CRUX_MUTANTS_PLAN_ONLY stops after
+# the plan line.
+# The fixture commits bypass hooks: a global pre-commit hook (pmat complexity) refused them, so no HEAD existed.
+fab() { # fab <dir>: a repo holding the scripts and the golden file the table reads, one commit, origin/main on it
+  local d="$1"
+  mkdir -p "$d/crates/apr-cli/src/commands"
+  cp -r "$ROOT/scripts" "$d/scripts"
+  cp "$ROOT/crates/apr-cli/src/commands/golden_output.rs" "$d/crates/apr-cli/src/commands/"
+  git -C "$d" init -q && git -C "$d" add -A && git -C "$d" -c user.email=t@t -c user.name=t -c core.hooksPath=/dev/null commit --no-verify -q -m base \
+    && git -C "$d" update-ref refs/remotes/origin/main HEAD
+}
+touch_commit() { # touch_commit <dir> <path>: one commit that changes <path>
+  printf '\n# touched\n' >> "$1/$2"
+  git -C "$1" add -A && git -C "$1" -c user.email=t@t -c user.name=t -c core.hooksPath=/dev/null commit --no-verify -q -m "touch $2"
+}
+plan_line() { # plan_line <dir> <event>: the table's own plan line, and its summary line (the name it reports)
+  ( cd "$1" && env -u GITHUB_BASE_REF -u CRUX_MUTANT_DIFF_BASE GITHUB_EVENT_NAME="$2" CRUX_MUTANTS_PLAN_ONLY=1 \
+      timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 | grep -E 'F6 mutants:|ok, [0-9]+ broke$' | tr '\n' ' ' )
+}
+P1="$TMP/push-judge"; fab "$P1"; touch_commit "$P1" scripts/lib/crux_inference_judge.py
+git -C "$P1" update-ref refs/remotes/origin/main HEAD          # a push to main: HEAD is the origin/main tip
+l1=$(plan_line "$P1" push)
+P2="$TMP/mg-judge"; fab "$P2"; touch_commit "$P2" scripts/lib/crux_inference_judge.py
+l2=$(plan_line "$P2" merge_group)                              # the queue head: single parent == origin/main tip
+P3="$TMP/push-readme"; fab "$P3"; touch_commit "$P3" scripts/guards_nightly_manifest.txt
+git -C "$P3" update-ref refs/remotes/origin/main HEAD
+l3=$(plan_line "$P3" push)
+case "$l1|$l2|$l3" in
+  *"all, 24 of 24"*"crux_inference_judge.py"*" check_crux_inference_judge: "*"|"*"all, 24 of 24"*"crux_inference_judge.py"*" check_crux_inference_judge: "*"|"*"sample, 6 of 24"*"touches none"*" check_crux_inference_judge: "*)
+    ok "the REAL table reads its diff on push and merge_group: the judge touched → all 24; an untouched push samples, diff read; the table still names itself" ;;
+  *) broke "table-read diff: push '$l1' / merge_group '$l2' / untouched push '$l3'" ;;
+esac
+
+# Row 13: CRUX_MUTANTS_PLAN_ONLY is never a verdict: the plan line says 0 ran and the table exits 3, even when a
+# runner sets it on a full run (it once printed "all, 24 of 24" and exited 0 with no mutant run).
+out=$( cd "$P1" && CRUX_MUTANTS=all CRUX_MUTANTS_PLAN_ONLY=1 timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 ); rc=$?
+case "$rc:$out" in
+  3:*"PLAN ONLY -- 0 run"*"no mutant ran, exit 3"*) ok "a plan-only run says 0 ran and exits 3, never a pass" ;;
+  *) broke "plan-only run: rc $rc, $(printf '%s' "$out" | grep -E 'F6 mutants|exit 3|ok, ' | tr '\n' ' ')" ;;
+esac
+
+# Row 14: a self-referential CRUX_MUTANT_DIFF_BASE (HEAD) is an UNREADABLE diff, never "untouched" (it once diffed
+# the tree against itself and read "the diff touches none of the mutants' files").
+s14=$( . "$ROOT/scripts/lib/crux_mutant_plan.sh"; CRUX_MUTANT_DIFF_BASE=HEAD crux_mutant_changed "$P1" "$TMP/s14.txt"; echo $? )
+s14b=$( . "$ROOT/scripts/lib/crux_mutant_plan.sh"; CRUX_MUTANT_DIFF_BASE=HEAD~1 crux_mutant_changed "$P1" "$TMP/s14b.txt"; echo "$?:$(grep -c crux_inference_judge "$TMP/s14b.txt")" )
+[ "$s14|$s14b" = "1|0:1" ] && ok "CRUX_MUTANT_DIFF_BASE=HEAD is refused (unreadable); HEAD~1 reads the judge change" \
+  || broke "diff-base override: HEAD rc $s14 (want 1), HEAD~1 $s14b (want 0:1)"
+
+# Row 15: nothing a runner can set turns a zero-mutant run into a pass (quorum round 4, lane 2, each measured at
+# exit 0): a floor of 0 is refused by the planner; a top-level CRUX_NO_MUTANTS=1 and CRUX_MUTANTS=none each exit 3
+# naming why; CRUX_MIN_TABLE_ROWS=0 cannot lower the row floor.
+r0=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A" --sample 0 --floor 0)
+( cd "$P1" && CRUX_NO_MUTANTS=1 timeout 600 bash scripts/check_crux_inference_judge.sh > "$TMP/r15a.log" 2>&1 ); ra=$?
+( cd "$P1" && CRUX_MUTANTS=none timeout 600 bash scripts/check_crux_inference_judge.sh > "$TMP/r15b.log" 2>&1 ); rb=$?
+( cd "$P1" && CRUX_JUDGE_OVERRIDE="$P1/scripts/lib/crux_inference_judge.py" CRUX_NO_MUTANTS=1 timeout 600 bash scripts/check_crux_inference_judge.sh > "$TMP/r15c.log" 2>&1 ); rc_=$?
+if [ "$r0" = "REFUSED 2" ] && [ "$ra" = 3 ] && grep -q 'CRUX_NO_MUTANTS is set: no mutant ran, exit 3' "$TMP/r15a.log" \
+   && [ "$rb" = 3 ] && grep -q 'the plan selected no mutant (CRUX_MUTANTS=none): no mutant ran, exit 3' "$TMP/r15b.log" \
+   && [ "$rc_" = 3 ] && grep -q 'CRUX_JUDGE_OVERRIDE is set' "$TMP/r15c.log"; then
+  ok "no runner setting makes a zero-mutant run pass: floor 0 refused; CRUX_NO_MUTANTS, CRUX_MUTANTS=none and CRUX_JUDGE_OVERRIDE+CRUX_NO_MUTANTS exit 3 naming why"
+else
+  broke "zero-mutant escapes: floor0 '$r0', CRUX_NO_MUTANTS rc $ra, none rc $rb, override+no-mutants rc $rc_"
+fi
+out=$( cd "$P1" && CRUX_MUTANTS_PLAN_ONLY=1 CRUX_MIN_TABLE_ROWS=0 timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 | grep -c 'under the floor' )
+out2=$( cd "$P1" && CRUX_MUTANTS_PLAN_ONLY=1 CRUX_MIN_TABLE_ROWS=100000 timeout 600 bash scripts/check_crux_inference_judge.sh 2>&1 | grep -c 'under the floor of 100000' )
+[ "$out:$out2" = "0:1" ] && ok "CRUX_MIN_TABLE_ROWS raises the row floor but 0 cannot lower it below 120 (the table stays above it)" \
+  || broke "row floor env: lowering '$out' (want 0 BROKE since the table is above 120), raising '$out2' (want 1)"
+
+# Row 16b: EVERY floor under the minimum is refused, not only 0 — 0 is also caught by the empty-selection path, so a
+# regression of the bound to `floor < 1` stayed green while a real run executed 1 of 24 (round 6, lane 2, measured).
+# And a MUTANT of exactly that off-by-one must be seen.
+bad=""
+for f in 1 2 3 4 5; do
+  r=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A" --sample "$f" --floor "$f")
+  [ "$r" = "REFUSED 2" ] || bad="$bad $f"
+done
+ok6=$(pl "$PLAN" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A" --sample 6 --floor 6)
+sed 's/^    if floor < MIN_FLOOR:$/    if floor < 1:/' "$PLAN" > "$TMP/mutant-floor.py"
+if cmp -s "$PLAN" "$TMP/mutant-floor.py"; then mf="ANCHOR MOVED"; else
+  mf=$(pl "$TMP/mutant-floor.py" --event pull_request --changed "$TMP/none.txt" --head "$HEAD_A" --sample 1 --floor 1); fi
+case "$bad|$ok6|$mf" in
+  "|sample 6 "*"|sample 1 "*) ok "floors 1-5 are each refused, 6 is accepted; a MUTANT bound (floor < 1) is seen accepting a 1-mutant sample" ;;
+  *) broke "floor bound: refused-missing for [$bad ], floor 6 '$ok6', mutant '$mf'" ;;
+esac
+
+# Row 12: the sourced lib leaves its caller's globals alone (it once set PROG and REPO_ROOT in the caller's shell).
+g=$( PROG=caller-prog; REPO_ROOT=caller-root; . "$ROOT/scripts/lib/crux_mutant_plan.sh" \
+     && crux_mutant_changed "$P1" "$TMP/g.txt"; printf '%s|%s|%s' "$PROG" "$REPO_ROOT" "$(grep -c crux_inference_judge "$TMP/g.txt" 2>/dev/null)" )
+[ "$g" = "caller-prog|caller-root|1" ] && ok "the sourced diff lib leaves the caller's PROG and REPO_ROOT alone, and still reads the diff" \
+  || broke "caller globals after crux_mutant_changed: $g (want caller-prog|caller-root|1)"
+
+python3 - "$PLAN" "$TMP/mutant-plan.py" <<'PY'
+import re, sys
+s = open(sys.argv[1]).read()
+m = re.search(r"WATCHED = \((.*?)\n\)", s, re.S)
+assert m, "WATCHED anchor moved: update this check with the planner"
+open(sys.argv[2], "w").write(s[:m.start()] + "WATCHED = ()" + s[m.end():])
+PY
+r=$(pl "$TMP/mutant-plan.py" --event pull_request --changed "$TMP/judge.txt" --head "$HEAD_A")
+case "$r" in "all 24 "*) broke "MUTANT (WATCHED emptied) not caught: $r" ;; *) ok "MUTANT (WATCHED emptied) caught: the judge-touched PR samples ($r)" ;; esac
+
+printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
+[ "$FAIL" -eq 0 ] || exit 1
+exit 0
diff --git a/scripts/guards_nightly_manifest.txt b/scripts/guards_nightly_manifest.txt
index 8eaddc126..72a73af78 100644
--- a/scripts/guards_nightly_manifest.txt
+++ b/scripts/guards_nightly_manifest.txt
@@ -23,12 +23,14 @@
 #
 # FORMAT: `<seconds on run 34449608126><TAB><step name, verbatim>`. (The perf-gate
 # line, added by #3676, is seconds on main run 35563537942: it stays at PR time
-# PATH-SCOPED via scripts/check_perf_gate_selftest_scoped.sh and runs whole here.) The step
+# PATH-SCOPED via scripts/check_perf_gate_selftest_scoped.sh and runs whole here.) The CRUX judge line (#4098) is
+# seconds measured on lambda, 2026-09-23, on #4046's head: its PR run is a sampled subset, and the whole set runs here. The step
 # name is the join key: it must appear VERBATIM as a `- name:` in
 # .github/workflows/guards-nightly.yml (that workflow's first step asserts it,
 # and fails closed if a line here names a step nothing runs), and it is also
 # what keeps scripts/check_guards_are_wired.sh finding these guards wired --
 # that guard greps every file under .github/workflows/, not ci.yml alone.
+655	CRUX judge, every F6 mutant (#4098)
 586	Test-tier decision case table (BSE-17): quick/full/reuse, drift is ENV
 405	Perf gate can still discriminate (APR-PERF-GATE-001, case table)
 256	model-tests falsification suites (was gated by nothing, aprender#2522) -- SATD ceiling measured on the comparand (BSE-03)
diff --git a/scripts/lib/crux_mutant_plan.py b/scripts/lib/crux_mutant_plan.py
new file mode 100644
index 000000000..4ec6e02ea
--- /dev/null
+++ b/scripts/lib/crux_mutant_plan.py
@@ -0,0 +1,117 @@
+"""crux_mutant_plan.py: which F6 mutants scripts/check_crux_inference_judge.sh runs (#4098).
+
+Each F6 mutant deletes one rule from a copy of the judge and re-runs the WHOLE case table, so the full set costs
+~630 of the table's 655 s (lambda, 2026-09-23), and guard-tree runs every cargo-free guard under one 30-minute job
+limit (the 0.69.1 freeze head's guard-tree was cancelled at 30.7 min). So:
+
+  all      every mutant. The default wherever no CI event says otherwise (a local run), on `schedule` and
+           `workflow_dispatch`, and in guards-nightly.yml, which asks for it by name (CRUX_MUTANTS=all).
+  sample   on `pull_request`, `merge_group` and `push`: every mutant when the diff touches a file the mutants
+           test (the judge, its imports, this table, this planner) — a change to a rule re-proves every rule —
+           otherwise a ROTATING slice of `--sample` mutants, its offset taken from the head sha, so successive
+           heads cover different rules and none is skipped for ever.
+  none     CRUX_MUTANTS=none: the table's recursion into itself for each mutant, never a CI mode.
+
+A sample is refused below its floor rather than run vacuously: `--sample` under `--floor` (or a selection that came
+out smaller than the floor while more mutants exist) exits 2 naming why. CRUX_MUTANTS overrides the event; an
+unknown value exits 2. An unreadable diff never passes as "untouched": it is named in the reason, and the sample
+still runs (the nightly runs the rest).
+
+  crux_mutant_plan.py --labels a,b,c [--event E] [--mode M] [--changed FILE|-] [--head SHA] [--sample N] [--floor N]
+  prints one JSON object: {"mode", "reason", "selected": [...], "total"}
+"""
+import argparse
+import json
+import sys
+
+# A change to any of these can change what a mutant tests, so it re-runs every mutant.
+WATCHED = (
+    "scripts/lib/crux_inference_judge.py",
+    "scripts/lib/crux_oracles.py",
+    "scripts/lib/crux_serve_routes.py",
+    "scripts/lib/crux_prompt_certify.py",
+    "scripts/check_crux_inference_judge.sh",
+    "scripts/lib/crux_mutant_plan.py",
+    "scripts/lib/crux_mutant_plan.sh",
+    "scripts/lib/resolve_base.sh",
+)
+FULL_EVENTS = ("schedule", "workflow_dispatch")
+MIN_FLOOR = 6  # the floor may be RAISED by --floor, never lowered: 0 made a 0-of-24 sample a pass (round 4, lane 2)
+SAMPLED_EVENTS = ("pull_request", "merge_group", "push")
+
+
+def plan(labels, event, mode, changed, head, sample, floor):
+    total = len(labels)
+    if mode:
+        if mode not in ("all", "sample", "none"):
+            raise SystemExit("crux_mutant_plan: CRUX_MUTANTS=%r is not all, sample or none" % mode)
+        why = "CRUX_MUTANTS=%s" % mode
+    elif event in FULL_EVENTS or not event:
+        mode, why = "all", ("event %s runs every mutant" % event) if event else "no CI event (a local run): every mutant"
+    elif event in SAMPLED_EVENTS:
+        mode, why = "sample", "event %s" % event
+    else:
+        raise SystemExit("crux_mutant_plan: event %r is neither a full nor a sampled event; name it in this planner"
+                         % event)
+    if mode == "none":
+        return {"mode": "none", "reason": why, "selected": [], "total": total}
+    if mode == "all":
+        return {"mode": "all", "reason": why, "selected": list(labels), "total": total}
+    if floor < MIN_FLOOR:
+        raise SystemExit("crux_mutant_plan: a floor of %d is under the minimum of %d: the floor can be raised, never "
+                         "lowered" % (floor, MIN_FLOOR))
+    if sample < floor:
+        raise SystemExit("crux_mutant_plan: a sample of %d is under the floor of %d: a smaller sample would pass "
+                         "vacuously" % (sample, floor))
+    if changed is None:
+        why += "; the diff could not be read, so nothing counts as touched (the nightly runs the rest)"
+    else:
+        hit = sorted(set(changed) & set(WATCHED))
+        if hit:
+            return {"mode": "all", "reason": why + "; the diff touches %s: every mutant" % ", ".join(hit),
+                    "selected": list(labels), "total": total}
+        why += "; the diff touches none of the mutants' files"
+    if total <= sample:
+        return {"mode": "sample", "reason": why + "; %d mutants, all run" % total, "selected": list(labels),
+                "total": total}
+    try:
+        off = int((head or "")[:8], 16) % total
+    except ValueError:
+        raise SystemExit("crux_mutant_plan: head %r is not a sha: the rotation needs one" % head)
+    sel = [labels[(off + i) % total] for i in range(sample)]
+    if len(sel) < floor:
+        raise SystemExit("crux_mutant_plan: selected %d, under the floor of %d" % (len(sel), floor))
+    return {"mode": "sample", "reason": why + "; rotating slice from %d of %d (head %s)" % (off, total, head[:9]),
+            "selected": sel, "total": total}
+
+
+def main():
+    p = argparse.ArgumentParser()
+    p.add_argument("--labels", required=True)
+    p.add_argument("--event", default="")
+    p.add_argument("--mode", default="")
+    p.add_argument("--changed", help="a file of changed paths, one per line; absent = the diff was unreadable")
+    p.add_argument("--head", default="")
+    p.add_argument("--sample", type=int, default=6)
+    p.add_argument("--floor", type=int, default=6)
+    a = p.parse_args()
+    labels = [x for x in a.labels.split(",") if x]
+    if not labels:
+        raise SystemExit("crux_mutant_plan: no mutant labels: an empty set is never a pass")
+    changed = None
+    if a.changed:
+        try:
+            changed = [l.strip() for l in open(a.changed) if l.strip()]
+        except OSError:
+            changed = None
+    print(json.dumps(plan(labels, a.event, a.mode, changed, a.head, a.sample, a.floor)))
+
+
+if __name__ == "__main__":
+    try:
+        main()
+    except SystemExit as e:
+        if isinstance(e.code, str):
+            sys.stderr.write(e.code + "\n")
+            sys.exit(2)
+        raise
diff --git a/scripts/lib/crux_mutant_plan.sh b/scripts/lib/crux_mutant_plan.sh
new file mode 100644
index 000000000..a588db124
--- /dev/null
+++ b/scripts/lib/crux_mutant_plan.sh
@@ -0,0 +1,40 @@
+# shellcheck shell=bash
+# crux_mutant_plan.sh — sourced by scripts/check_crux_inference_judge.sh (#4098). Option-neutral: no `set` here; a
+# failure is the return status.
+#
+# crux_mutant_changed <repo root> <out file>: the paths HEAD changes against its base, one per line, into <out>.
+# Returns 1 when no base can be named — the caller then says the diff was unreadable, never "untouched".
+#
+# The base comes from scripts/lib/resolve_base.sh, the project's one resolver for every CI checkout shape: the
+# merge-base with origin/main; the merge commit's first parent on a depth-1 pull_request checkout; the queue
+# head's parent on a merge_group; the FIRST PARENT on a push to main. GITHUB_BASE_REF alone — what this table
+# used first — exists only on pull_request, so a merge group or a push that touched the judge sampled instead of
+# running every mutant (quorum round 1, lane 1, measured). CRUX_MUTANT_DIFF_BASE=<ref> overrides (<ref>...HEAD).
+crux_mutant_changed() {
+  local root=$1 out=$2
+  if [ -n "${CRUX_MUTANT_DIFF_BASE:-}" ]; then
+    # A base whose merge-base with HEAD is HEAD itself (HEAD, or anything ahead of it) diffs the tree against itself:
+    # empty, and read as "untouched". Refused like resolve_base refuses it (quorum round 3, lane 2, measured).
+    local mb head
+    mb=$(git -C "$root" merge-base "$CRUX_MUTANT_DIFF_BASE" HEAD 2> /dev/null) || return 1
+    head=$(git -C "$root" rev-parse HEAD 2> /dev/null) || return 1
+    [ "$mb" != "$head" ] || return 1
+    git -C "$root" diff --no-renames --name-only "$mb" HEAD > "$out" 2> /dev/null
+    return
+  fi
+  # In a SUBSHELL: resolve_base needs REPO_ROOT and PROG, and setting them here clobbered the sourcing table's own
+  # PROG, so every real run printed another guard's name in its summary (quorum round 2, lane 1, measured).
+  local base
+  base=$(
+    REPO_ROOT=$root
+    PROG=crux_mutant_plan
+    # shellcheck source=scripts/lib/resolve_base.sh
+    . "$root/scripts/lib/resolve_base.sh" || exit 1
+    BASE_REF=""
+    resolve_base HEAD 2> /dev/null || exit 1
+    printf '%s' "$BASE_REF"
+  ) || return 1
+  [ -n "$base" ] || return 1
+  # --no-renames: a rename lists BOTH paths, so a watched file moved away still counts as touched (#3664)
+  git -C "$root" diff --no-renames --name-only "$base" HEAD > "$out" 2> /dev/null
+}
```
