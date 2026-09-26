You are one of 1 independent reviewers. Judge whether this diff does what its ticket says, and nothing the ticket forbids. Try to REFUTE it: default to FAIL when a test asserts the opposite of the ticket, when a gate is weakened, when a receipt claim is not backed by the diff, or when the change does something the ticket does not ask for. Every finding needs file, line, claim and grounding (cited = you quote the diff; measured = you ran a command; asserted = neither). Return PASS only if you found nothing that refutes it.

## Ticket(s) PMAT-3459 — the diff is judged against ALL of them
### PMAT-3459
📊 Status for: PMAT-3459

   Title: release train: move the release autopilot's tag path into the repo; tag step calls check_milestone_cut.sh (#3459)
   Status: Planned
   Priority: Critical
   Progress: 0%
   GitHub: #3459



## Receipt

## Receipt PMAT-3459
---
status: complete
ticket: PMAT-3459
github_issue: 3459
part: 2 (the must-carry label universe); part 1 (the in-repo gated tag path) landed in 6a657fc61 / PR #3617
kind: code
model: claude-opus-5-5 (author)
---
# implementation receipt: PMAT-3459 part 2 (#3459)

## Scope: where part 2 comes from

The roadmap title names only part 1. The issue #3459 carries part 2 in its own body, under "Tag-path defect 3
(operator ruling 2026-09-20 ~14:00Z) — `check_milestone_cut.sh` counts PRs":

> **The rule it must implement instead: count ISSUES carrying the must-carry label. Never PRs.**

The same section measures why: S3 asserted "one open PR -> RED", so every open PR on the milestone blocked the tag
(0.69.0: `open_issues=9` from the milestone API, 3 from `gh issue list`; the six PRs made up the difference).

dd's MUST-row sweep (#4159, 2026-09-24T07:11Z comment on #3459) records part 1 as DONE and part 2 as "OPEN.
Claimed by aprender-f5".

The cop (aprender-cf) ruled on 2026-09-24, and the ruling bounds this diff. This is the cop's ruling, not an
operator quotation:
- (a) Create the `must-carry` label, described as "Blocks the release cut of its milestone (check_milestone_cut.sh)".
- (b) The narrowing is approved on one condition: nothing is silently left behind. At the cut, every open item
  that is not must-carry is MOVED, with a one-line comment. It goes to the next release if that release's epic
  lists it, otherwise to `backlog`.
- A milestone that is tagged while items are still open is RED. Add a self-test row: an unlabelled open issue left
  in a tagged milestone gives RED.
- The move lives in the autopilot, before the tag. The gate only verifies.

So the carry script and the autopilot's move-before-tag step are the approval condition for narrowing the
blocker set. They are not extra scope. Without the move, `--must-carry` alone would let unlabelled items silently
ride a tagged milestone.

## What the diff does

| file | change |
|---|---|
| `scripts/check_milestone_cut.sh` | Adds a `--must-carry` mode. It is RED only on an OPEN ISSUE labelled `must-carry`, never on a PR. Every other open item is printed `TO CARRY`, and the JSON gains `mode` and `to_carry`. The strict default is unchanged. New self-test rows S24–S29; S27 is the cop's row (an unlabelled open issue under strict gives RED). |
| `scripts/release/carry_milestone_items.sh` | New. For each open non-must-carry item, it moves the item to the next open semver milestone when that milestone's "EPIC: release train <next>" lists it, otherwise to `backlog`, each move with a `slipped_from:` comment. It refuses (rc 1, zero writes) while any must-carry issue is open. rc 2 means "cannot act": a failed read, no next milestone, or a partial write. |
| `scripts/release/autopilot.sh` `cut_tag()` | Runs the must-carry verdict, then the carry step, then the EXISTING strict call (the line and its case block are byte-identical), then `git tag`. The strict gate still sees every open item, so anything the carry step missed turns the tag RED. |
| `scripts/check_tag_step_gated.sh` | Asserts that order. Must-carry rc 1 or 2 must neither carry nor tag, and carry rc 2 must not tag. New mutants: M4 (must-carry verdict discarded) and M5 (carry call deleted). It also runs the carry script's own self-test. |

The strict gate is not weakened. S3 still asserts "one open PR -> RED" under strict, and strict remains the last
thing before `git tag`. S24 inverts S3 only in `--must-carry` mode, which is exactly the rule the issue asks for.

## Measured (HEAD of this branch)

```
bash scripts/check_milestone_cut.sh --self-test          -> 31/31 ok; a mutant ignoring the label is killed by S26
bash scripts/release/carry_milestone_items.sh --self-test -> SELF-TEST PASSED (7 rows, stub gh)
bash scripts/check_tag_step_gated.sh --self-test          -> SELF-TEST PASSED (M1..M5 RED, real subject GREEN)
bash scripts/check_bashrs_gate.sh                         -> PASS, 0 SEC/DET/IDEM errors
check_shell_lint_ratchet / no_pipe_into_grep_q / no_hand_rolled_parsers / guards_are_wired -> PASS
gh api repos/paiml/aprender/labels/must-carry             -> exists, color B60205
```

Not done here, per the cop's ruling: applying `must-carry` to the 0.70 scope. dd does that at the 0.70 scope-cut GO.

## Diff (origin/main...HEAD)
```diff
diff --git a/docs/audits/impl-PMAT-3459-receipt.md b/docs/audits/impl-PMAT-3459-receipt.md
new file mode 100644
index 000000000..975929056
--- /dev/null
+++ b/docs/audits/impl-PMAT-3459-receipt.md
@@ -0,0 +1,61 @@
+---
+status: complete
+ticket: PMAT-3459
+github_issue: 3459
+part: 2 (the must-carry label universe); part 1 (the in-repo gated tag path) landed in 6a657fc61 / PR #3617
+kind: code
+model: claude-opus-5-5 (author)
+---
+# implementation receipt: PMAT-3459 part 2 (#3459)
+
+## Scope: where part 2 comes from
+
+The roadmap title names only part 1. The issue #3459 carries part 2 in its own body, under "Tag-path defect 3
+(operator ruling 2026-09-20 ~14:00Z) — `check_milestone_cut.sh` counts PRs":
+
+> **The rule it must implement instead: count ISSUES carrying the must-carry label. Never PRs.**
+
+The same section measures why: S3 asserted "one open PR -> RED", so every open PR on the milestone blocked the tag
+(0.69.0: `open_issues=9` from the milestone API, 3 from `gh issue list`; the six PRs made up the difference).
+
+dd's MUST-row sweep (#4159, 2026-09-24T07:11Z comment on #3459) records part 1 as DONE and part 2 as "OPEN.
+Claimed by aprender-f5".
+
+The cop (aprender-cf) ruled on 2026-09-24, and the ruling bounds this diff. This is the cop's ruling, not an
+operator quotation:
+- (a) Create the `must-carry` label, described as "Blocks the release cut of its milestone (check_milestone_cut.sh)".
+- (b) The narrowing is approved on one condition: nothing is silently left behind. At the cut, every open item
+  that is not must-carry is MOVED, with a one-line comment. It goes to the next release if that release's epic
+  lists it, otherwise to `backlog`.
+- A milestone that is tagged while items are still open is RED. Add a self-test row: an unlabelled open issue left
+  in a tagged milestone gives RED.
+- The move lives in the autopilot, before the tag. The gate only verifies.
+
+So the carry script and the autopilot's move-before-tag step are the approval condition for narrowing the
+blocker set. They are not extra scope. Without the move, `--must-carry` alone would let unlabelled items silently
+ride a tagged milestone.
+
+## What the diff does
+
+| file | change |
+|---|---|
+| `scripts/check_milestone_cut.sh` | Adds a `--must-carry` mode. It is RED only on an OPEN ISSUE labelled `must-carry`, never on a PR. Every other open item is printed `TO CARRY`, and the JSON gains `mode` and `to_carry`. The strict default is unchanged. New self-test rows S24–S29; S27 is the cop's row (an unlabelled open issue under strict gives RED). |
+| `scripts/release/carry_milestone_items.sh` | New. For each open non-must-carry item, it moves the item to the next open semver milestone when that milestone's "EPIC: release train <next>" lists it, otherwise to `backlog`, each move with a `slipped_from:` comment. It refuses (rc 1, zero writes) while any must-carry issue is open. rc 2 means "cannot act": a failed read, no next milestone, or a partial write. |
+| `scripts/release/autopilot.sh` `cut_tag()` | Runs the must-carry verdict, then the carry step, then the EXISTING strict call (the line and its case block are byte-identical), then `git tag`. The strict gate still sees every open item, so anything the carry step missed turns the tag RED. |
+| `scripts/check_tag_step_gated.sh` | Asserts that order. Must-carry rc 1 or 2 must neither carry nor tag, and carry rc 2 must not tag. New mutants: M4 (must-carry verdict discarded) and M5 (carry call deleted). It also runs the carry script's own self-test. |
+
+The strict gate is not weakened. S3 still asserts "one open PR -> RED" under strict, and strict remains the last
+thing before `git tag`. S24 inverts S3 only in `--must-carry` mode, which is exactly the rule the issue asks for.
+
+## Measured (HEAD of this branch)
+
+```
+bash scripts/check_milestone_cut.sh --self-test          -> 31/31 ok; a mutant ignoring the label is killed by S26
+bash scripts/release/carry_milestone_items.sh --self-test -> SELF-TEST PASSED (7 rows, stub gh)
+bash scripts/check_tag_step_gated.sh --self-test          -> SELF-TEST PASSED (M1..M5 RED, real subject GREEN)
+bash scripts/check_bashrs_gate.sh                         -> PASS, 0 SEC/DET/IDEM errors
+check_shell_lint_ratchet / no_pipe_into_grep_q / no_hand_rolled_parsers / guards_are_wired -> PASS
+gh api repos/paiml/aprender/labels/must-carry             -> exists, color B60205
+```
+
+Not done here, per the cop's ruling: applying `must-carry` to the 0.70 scope. dd does that at the 0.70 scope-cut GO.
diff --git a/scripts/check_milestone_cut.sh b/scripts/check_milestone_cut.sh
index e3a5c7dc1..a99a91cd2 100644
--- a/scripts/check_milestone_cut.sh
+++ b/scripts/check_milestone_cut.sh
@@ -26,6 +26,15 @@
 # immediately before `git tag`, after the bump PR has merged. Never while the
 # bump PR is open: it sits in the milestone and reads RED.
 #
+# TWO MODES (#3459 part 2, cop ruling 2026-09-24):
+#   --must-carry  the BLOCKING set: open ISSUES labelled `must-carry`. Pull requests and
+#                 unlabelled issues do not block; each is listed as TO CARRY, because the
+#                 release autopilot MOVES it (scripts/release/carry_milestone_items.sh) before
+#                 the tag. Nothing is silently left behind: see strict.
+#   (default)     STRICT: the milestone holds nothing open but its release epic. cut_tag() runs
+#                 it AFTER the carry, so an item the carry missed, or an item nobody carried,
+#                 is RED at the tag. A tagged milestone with an open item is never clean.
+#
 # Exit 0 = zero open items in the milestone.
 # Exit 1 = at least one open item; each is named with its remedy.
 # Exit 2 = cannot judge, never a silent pass: gh or python3 missing, gh
@@ -42,11 +51,11 @@ set -euo pipefail
 SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
 
 usage() {
-    printf 'usage: %s <milestone-title> [--repo O/R] [--json OUT] | --self-test\n' "$(basename "$0")" >&2
+    printf 'usage: %s <milestone-title> [--repo O/R] [--json OUT] [--must-carry] | --self-test\n' "$(basename "$0")" >&2
     exit 2
 }
 
-# judge_from_dir DIR TITLE [JSON_OUT]
+# judge_from_dir DIR TITLE [JSON_OUT] [MODE]   MODE = strict (default) | must-carry
 # DIR holds milestones.jsonl and items.jsonl, one JSON object per line: the
 # shape `gh api --paginate --jq '.[]'` writes, so a read of more than one page
 # is never a concatenation of arrays. Prints the verdict and returns 0, 1 or 2.
@@ -55,6 +64,10 @@ judge_from_dir() {
 import json, sys
 
 d, title, json_out = sys.argv[1], sys.argv[2], sys.argv[3]
+mode = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "strict"
+if mode not in ("strict", "must-carry"):
+    print("ENV: unknown mode %s" % mode, file=sys.stderr)
+    sys.exit(2)
 
 def env(msg):
     print("ENV: " + msg, file=sys.stderr)
@@ -124,18 +137,34 @@ if len(admitted) > 1:
     env("%d open items claim to be the release epic of %s: %s" % (len(admitted), title, " ".join("#%s" % r["number"] for r in admitted)))
 rows = [r for r in all_rows if not is_release_epic(r)]
 
+# --must-carry: only open ISSUES labelled must-carry block the cut. Everything else is
+# listed TO CARRY: the autopilot moves it before the tag, and the STRICT run after the
+# move is what proves nothing was left behind (#3459 part 2).
+to_carry = []
+if mode == "must-carry":
+    to_carry = [r for r in rows if not (r["kind"] == "issue" and "must-carry" in r["labels"])]
+    rows = [r for r in rows if r["kind"] == "issue" and "must-carry" in r["labels"]]
+
 verdict = "RED" if rows else "PASS"
 if json_out:
     with open(json_out, "w", encoding="utf-8") as f:
-        json.dump({"milestone": title, "number": number, "open": open_n, "closed": closed_n,
-                   "admitted": admitted, "items": rows, "verdict": verdict}, f, indent=2, sort_keys=True)
+        json.dump({"milestone": title, "number": number, "open": open_n, "closed": closed_n, "mode": mode,
+                   "admitted": admitted, "items": rows, "to_carry": to_carry, "verdict": verdict},
+                  f, indent=2, sort_keys=True)
         f.write("\n")
 
 for r in admitted:
     print("ADMITTED #%s %s [%s] %s -- the release epic of this train, closed at 06x section 4 step 8"
           % (r["number"], r["kind"], ",".join(r["labels"]), r["title"]))
+for r in to_carry:
+    print("TO CARRY #%s %s [%s] %s -- not must-carry: the autopilot moves it before the tag"
+          % (r["number"], r["kind"], ",".join(r["labels"]), r["title"]))
 if not rows:
-    print("PASS  milestone %s (#%s): 0 open besides its release epic, %d closed -- the cut may proceed" % (title, number, closed_n))
+    if mode == "must-carry":
+        print("PASS  milestone %s (#%s): 0 open must-carry issue(s); %d item(s) to carry before the tag"
+              % (title, number, len(to_carry)))
+    else:
+        print("PASS  milestone %s (#%s): 0 open besides its release epic, %d closed -- the cut may proceed" % (title, number, closed_n))
     sys.exit(0)
 
 for r in rows:
@@ -143,9 +172,13 @@ for r in rows:
 for r in rows:
     print("  remedy #%s: close it, or carry it: gh %s edit %s --milestone <next> && gh %s comment %s --body \"slipped_from: %s\""
           % (r["number"], r["kind"], r["number"], r["kind"], r["number"], title))
-print("RED   milestone %s (#%s): %d open item(s) -- no tag until each is closed or carried" % (title, number, len(rows)))
+if mode == "must-carry":
+    print("RED   milestone %s (#%s): %d open must-carry issue(s) -- they BLOCK the cut and are never carried"
+          % (title, number, len(rows)))
+else:
+    print("RED   milestone %s (#%s): %d open item(s) -- no tag until each is closed or carried" % (title, number, len(rows)))
 sys.exit(1)
-' "$1" "$2" "${3:-}"
+' "$1" "$2" "${3:-}" "${4:-}"
 }
 
 # fetch_live REPO TITLE DIR -- writes DIR/milestones.jsonl and DIR/items.jsonl.
@@ -212,7 +245,7 @@ st_judge() {
     t="$4"
     shift 4
     rc=0
-    judge_from_dir "$fx" "$t" "" > "${fx}/out" 2>&1 || rc=$?
+    judge_from_dir "$fx" "$t" "" "${ST_MODE:-}" > "${fx}/out" 2>&1 || rc=$?
     st_check "$c" "$want" "$rc" "${fx}/out" "$@"
 }
 
@@ -330,6 +363,28 @@ self_test() {
     { st_item 20 7 issue open epic "EPIC: release train M — a"; st_item 10 7 issue open P1; } > "${fx}/items.jsonl"
     st_judge "$fx" S23 1 M "ADMITTED #20" "#10 issue [P1] item 10" "1 open item(s)"
 
+    # --must-carry (#3459 part 2): only open ISSUES labelled must-carry block. The STRICT rows above
+    # still hold unchanged: cut_tag() runs strict AFTER the carry, so nothing is left behind.
+    # S24 = S3 inverted in the new scope: one open PR is NOT a blocker, it is listed TO CARRY
+    st_ms M 7 1 5 > "${fx}/milestones.jsonl"
+    st_item 11 7 pr open release > "${fx}/items.jsonl"
+    ST_MODE=must-carry st_judge "$fx" S24 0 M "TO CARRY #11 pr [release] item 11" "0 open must-carry issue(s); 1 item(s) to carry"
+    # S25 = its twin: one open must-carry ISSUE blocks, named with its label, and is never carried
+    st_item 30 7 issue open must-carry > "${fx}/items.jsonl"
+    ST_MODE=must-carry st_judge "$fx" S25 1 M "#30 issue [must-carry] item 30" "1 open must-carry issue(s) -- they BLOCK the cut"
+    # S26 an unlabelled open issue does not block the must-carry run: it is carried
+    st_item 10 7 issue open P1 > "${fx}/items.jsonl"
+    ST_MODE=must-carry st_judge "$fx" S26 0 M "TO CARRY #10 issue [P1] item 10"
+    # S27 (cop ruling) the SAME unlabelled issue left in the milestone at the TAG is RED: strict is the
+    #     post-carry verification, and a tagged milestone with an open item is never clean
+    st_judge "$fx" S27 1 M "#10 issue [P1] item 10" "1 open item(s)"
+    # S28 a PULL REQUEST labelled must-carry does not block: the universe is ISSUES
+    st_item 31 7 pr open must-carry > "${fx}/items.jsonl"
+    ST_MODE=must-carry st_judge "$fx" S28 0 M "TO CARRY #31 pr [must-carry]"
+    # S29 the release epic stays ADMITTED in must-carry mode, whatever its labels
+    st_item 20 7 issue open epic "EPIC: release train M — schedule" > "${fx}/items.jsonl"
+    ST_MODE=must-carry st_judge "$fx" S29 0 M "ADMITTED #20 issue [epic]" "0 open must-carry issue(s); 0 item(s) to carry"
+
     # S16 --json records the verdict and the items
     st_ms M 7 1 5 > "${fx}/milestones.jsonl"
     st_item 10 7 issue open bug > "${fx}/items.jsonl"
@@ -416,7 +471,11 @@ main() {
     shift
     repo="paiml/aprender"
     json_out=""
+    mode="strict"
     while [ $# -gt 0 ]; do
+        case "$1" in
+            --must-carry) mode="must-carry"; shift; continue ;;
+        esac
         [ $# -ge 2 ] || usage
         case "$1" in
             --repo) repo="$2" ;;
@@ -446,7 +505,7 @@ main() {
         exit 2
     fi
     rc=0
-    judge_from_dir "$input_dir" "$title" "$json_out" || rc=$?
+    judge_from_dir "$input_dir" "$title" "$json_out" "$mode" || rc=$?
     exit "$rc"
 }
 
diff --git a/scripts/check_tag_step_gated.sh b/scripts/check_tag_step_gated.sh
index 4545b2f01..37118af34 100755
--- a/scripts/check_tag_step_gated.sh
+++ b/scripts/check_tag_step_gated.sh
@@ -14,8 +14,14 @@
 #     gate rc 0 -> tag is cut
 #     gate rc 1 -> no tag, no publish   (the milestone holds open items)
 #     gate rc 2 -> no tag               (Unknown; never a silent pass)
-# --self-test then removes the gate call to build a MUTANT and requires this guard to
-# go RED on it. A guard that cannot fail on the defect it names is theater.
+# #3459 part 2 made the tag path three steps (must-carry gate, carry, STRICT gate), so the
+# stubs answer each call separately and record the ORDER they ran in:
+#     must-carry rc 1/2 -> no tag AND nothing carried (a blocker is never carried around)
+#     carry rc 2        -> no tag
+#     all clean         -> the carry ran BEFORE the strict gate, and the tag is cut
+# --self-test then builds MUTANTS (gate calls removed, verdicts discarded, the carry call
+# removed) and requires this guard to go RED on each. It also runs the carry script's own
+# case table, which lives in scripts/release/ where guard_tree cannot discover it.
 #
 #   check_tag_step_gated.sh              judge scripts/release/autopilot.sh
 #   check_tag_step_gated.sh --self-test  case table + the gate-removed mutant
@@ -27,15 +33,18 @@ ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
 rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
 SUBJECT="$ROOT/scripts/release/autopilot.sh"
 
-# run_cut_tag <autopilot> <gate-rc> -- extract cut_tag(), run it with stubs, print a
-# transcript (SAY/DIE/GIT-TAG/GIT-PUSH lines). Returns 2 if the function is missing.
+# run_cut_tag <autopilot> <strict-rc> [<must-carry-rc> [<carry-rc>]] -- extract cut_tag(), run
+# it with stubs, print a transcript (SAY/DIE/GIT-TAG/GIT-PUSH lines, then the CALL order).
+# Returns 2 if the function is missing.
 run_cut_tag() {
-    local ap=$1 grc=$2 d fn
+    local ap=$1 grc=$2 mrc=${3:-0} crc=${4:-0} d fn
     d=$(mktemp -d) || return 2
     fn=$(awk '/^cut_tag\(\) \{/,/^\}/' "$ap")
     [ -n "$fn" ] || { rmtree "$d"; return 2; }
-    mkdir -p "$d/scripts"
-    printf '#!/usr/bin/env bash\nexit %s\n' "$grc" > "$d/scripts/check_milestone_cut.sh"
+    mkdir -p "$d/scripts/release"
+    printf '#!/usr/bin/env bash\nif [ "${2:-}" = --must-carry ]; then echo CALL-MUST-CARRY >> %q; exit %s; fi\necho CALL-STRICT >> %q; exit %s\n' \
+        "$d/calls" "$mrc" "$d/calls" "$grc" > "$d/scripts/check_milestone_cut.sh"
+    printf '#!/usr/bin/env bash\necho CALL-CARRY >> %q\nexit %s\n' "$d/calls" "$crc" > "$d/scripts/release/carry_milestone_items.sh"
     {
         printf 'set -uo pipefail\n'
         printf 'REPO_ROOT=%q\nLOG=%q\n' "$d" "$d/log"
@@ -47,6 +56,7 @@ run_cut_tag() {
     } > "$d/harness.sh"
     bash "$d/harness.sh" 2>&1
     cat "$d/log" 2>/dev/null
+    printf 'ORDER %s\n' "$(tr '\n' ' ' < "$d/calls" 2>/dev/null)"
     rmtree "$d"
 }
 
@@ -70,6 +80,21 @@ judge() {
     if grep -q 'GIT-TAG' <<< "$out"; then
         printf 'FAIL  gate rc=2 (Unknown) -> A TAG WAS CUT ANYWAY\n%s\n' "$out" >&2; bad=1
     else printf 'ok    gate rc=2 (Unknown) -> no tag\n'; fi
+    # #3459 part 2: the must-carry gate, the carry, and their ORDER
+    out=$(run_cut_tag "$ap" 0) || true
+    if grep -q '^ORDER CALL-MUST-CARRY CALL-CARRY CALL-STRICT $' <<< "$out" && grep -q 'GIT-TAG' <<< "$out"; then
+        printf 'ok    all clean -> must-carry, then the carry, then STRICT, then the tag\n'
+    else printf 'FAIL  all clean did not run must-carry -> carry -> strict -> tag\n%s\n' "$out" >&2; bad=1; fi
+    for m in 1 2; do
+        out=$(run_cut_tag "$ap" 0 "$m") || true
+        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-CARRY' <<< "$out"; then
+            printf 'FAIL  must-carry rc=%s -> a tag was cut or items were CARRIED around a blocker\n%s\n' "$m" "$out" >&2; bad=1
+        else printf 'ok    must-carry rc=%s -> nothing carried, no tag\n' "$m"; fi
+    done
+    out=$(run_cut_tag "$ap" 0 0 2) || true
+    if grep -q 'GIT-TAG' <<< "$out"; then
+        printf 'FAIL  carry rc=2 -> A TAG WAS CUT over a failed carry\n%s\n' "$out" >&2; bad=1
+    else printf 'ok    carry rc=2 -> no tag\n'; fi
     return "$bad"
 }
 
@@ -106,6 +131,30 @@ if [ "${1:-}" = "--self-test" ]; then
         ok "mutant 2: gate verdict discarded -> RED"
     fi
 
+    # M4: the MUST-CARRY verdict discarded -> items are carried around a blocker and a tag is cut.
+    sed 's#\(bash "$REPO_ROOT/scripts/check_milestone_cut.sh" "$v" --must-carry >> "$LOG" 2>&1\) || rc=$?#\1 || true#' "$SUBJECT" > "$d/m4.sh"
+    if cmp -s "$SUBJECT" "$d/m4.sh"; then
+        nok "MUTANT 4 could not be built -- the must-carry call line did not match; vacuous"
+    elif judge "$d/m4.sh" > "$d/m4.out" 2>&1; then
+        nok "MUTANT 4 (must-carry verdict discarded) PASSED"
+    else
+        ok "mutant 4: must-carry verdict discarded -> RED"
+    fi
+    # M5: the carry call deleted -> a milestone is judged strict without anything having been moved.
+    sed '/carry_milestone_items\.sh" "\$v"/d' "$SUBJECT" > "$d/m5.sh"
+    if cmp -s "$SUBJECT" "$d/m5.sh"; then
+        nok "MUTANT 5 could not be built -- the carry call line did not match; vacuous"
+    elif judge "$d/m5.sh" > "$d/m5.out" 2>&1; then
+        nok "MUTANT 5 (carry call deleted) PASSED"
+    else
+        ok "mutant 5: carry call deleted -> RED"
+    fi
+    # the carry script's own case table: it lives in scripts/release/, where guard_tree cannot see it
+    if bash "$ROOT/scripts/release/carry_milestone_items.sh" --self-test > "$d/carry.out" 2>&1; then
+        ok "carry_milestone_items.sh case table ($(grep -c '^ok ' "$d/carry.out") rows)"
+    else
+        nok "carry_milestone_items.sh case table FAILED"; cat "$d/carry.out" >&2
+    fi
     # M3: cut_tag() removed entirely -> ENV (2), never a pass.
     awk '/^cut_tag\(\) \{/,/^\}/ {next} {print}' "$SUBJECT" > "$d/m3.sh"
     judge "$d/m3.sh" > "$d/m3.out" 2>&1; rc=$?
diff --git a/scripts/release/autopilot.sh b/scripts/release/autopilot.sh
index 8827a1a20..fa57edb0b 100755
--- a/scripts/release/autopilot.sh
+++ b/scripts/release/autopilot.sh
@@ -134,8 +134,27 @@ fi
 #   2 = the gate could not judge (Unknown)    -> no tag. Never a silent pass.
 # scripts/check_tag_step_gated.sh runs this function against stubs and requires each
 # of those three paths, plus a gate-call-removed MUTANT, to behave as stated.
+#
+# #3459 part 2 (cop ruling 2026-09-24): three steps, in this order, all ahead of `git tag`:
+#   (a) --must-carry: an open ISSUE labelled must-carry BLOCKS the cut. It is never carried.
+#   (b) carry_milestone_items.sh MOVES every other open item (to the next release when its epic
+#       lists it, else to backlog, one comment each). It runs only when (a) is clean.
+#   (c) STRICT: the milestone now holds nothing open but its release epic. An item the carry
+#       missed, or one that reappeared, is RED here: a tagged milestone is never left with an
+#       open item.
 cut_tag() {
     local v=$1 t=$2 mc=$3 rc=0
+    bash "$REPO_ROOT/scripts/check_milestone_cut.sh" "$v" --must-carry >> "$LOG" 2>&1 || rc=$?
+    case "$rc" in
+        0) say "MUST-CARRY $v: no open must-carry issue (check_milestone_cut.sh --must-carry rc=0)" ;;
+        1) die "milestone $v holds open must-carry issue(s) -- nothing carried, no tag (check_milestone_cut.sh --must-carry rc=1)" ;;
+        *) die "milestone $v could not be judged for must-carry (rc=$rc) -- nothing carried, no tag; Unknown is not a pass" ;;
+    esac
+    rc=0
+    bash "$REPO_ROOT/scripts/release/carry_milestone_items.sh" "$v" >> "$LOG" 2>&1 || rc=$?
+    [ "$rc" -eq 0 ] || die "carrying the open items out of $v failed (carry_milestone_items.sh rc=$rc) -- no tag"
+    say "CARRIED the non-must-carry open items out of $v"
+    rc=0
     bash "$REPO_ROOT/scripts/check_milestone_cut.sh" "$v" >> "$LOG" 2>&1 || rc=$?
     case "$rc" in
         0) say "MILESTONE-GATE $v clean at the cut (check_milestone_cut.sh rc=0)" ;;
diff --git a/scripts/release/carry_milestone_items.sh b/scripts/release/carry_milestone_items.sh
new file mode 100755
index 000000000..0654d0f96
--- /dev/null
+++ b/scripts/release/carry_milestone_items.sh
@@ -0,0 +1,214 @@
+#!/usr/bin/env bash
+# carry_milestone_items.sh <milestone-title> [--repo O/R] [--dry-run] | --self-test
+#
+# #3459 part 2 (cop ruling 2026-09-24): only open ISSUES labelled `must-carry` block a release cut
+# (check_milestone_cut.sh --must-carry). Every OTHER open item of the milestone is MOVED here, before
+# the tag, so nothing is silently left behind:
+#   * to the NEXT release's milestone when that release's epic ("EPIC: release train <next>", label
+#     epic) references it (#N in its body);
+#   * otherwise to the `backlog` milestone (the operator's backlog rule).
+# Each move gets ONE comment: "slipped_from: <milestone> -- carried to <target> at the <milestone>
+# cut by the release autopilot (<why>)". The release epic of THIS train is never moved: it closes
+# after publish. The autopilot's cut_tag() runs this between the must-carry gate and the STRICT
+# gate; the strict gate is what verifies the milestone is empty at the tag.
+#
+# REFUSES (exit 1, nothing moved) while any open must-carry issue remains: carrying around a
+# blocker would hide the reason the cut must wait. Exit 2 = could not act (gh/python3 missing, a
+# read or a write failed, no or ambiguous milestone, no next milestone, no backlog milestone). A
+# partial carry is reported by name and is exit 2, never 0.
+#
+# --dry-run prints the moves without writing. --self-test stubs gh on PATH (no network) and runs
+# the case table.
+set -uo pipefail
+SELF="$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")"
+BACKLOG="backlog"
+
+usage() { printf 'usage: %s <milestone-title> [--repo O/R] [--dry-run] | --self-test\n' "$(basename "$0")" >&2; exit 2; }
+
+# plan DIR TITLE -> lines "MOVE <kind> <number> <target> <why>", or "BLOCK <number>" / "ENV <msg>"
+plan() {
+    python3 - "$1" "$2" "$BACKLOG" <<'PY'
+import json, re, sys
+d, title, backlog = sys.argv[1:]
+def rows(name):
+    out = []
+    with open(d + "/" + name, encoding="utf-8") as f:
+        for raw in f:
+            raw = raw.strip()
+            if raw:
+                out.append(json.loads(raw))
+    return out
+def semver(t):
+    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", t or "")
+    return tuple(int(x) for x in m.groups()) if m else None
+ms = rows("milestones.jsonl")
+cur = [m for m in ms if m.get("title") == title]
+if len(cur) != 1:
+    print("ENV milestone %s matches %d milestone(s)" % (title, len(cur))); sys.exit(0)
+if not any(m.get("title") == backlog for m in ms):
+    print("ENV no '%s' milestone to carry into" % backlog); sys.exit(0)
+v = semver(title)
+if v is None:
+    print("ENV milestone title %s is not X.Y.Z" % title); sys.exit(0)
+later = sorted((semver(m["title"]), m["title"]) for m in ms
+               if m.get("state") == "open" and semver(m.get("title")) and semver(m["title"]) > v)
+if not later:
+    print("ENV no open milestone after %s" % title); sys.exit(0)
+nxt = later[0][1]
+listed = set()
+for e in rows("next_epics.jsonl"):
+    names = [l.get("name") for l in (e.get("labels") or [])]
+    t = e.get("title", "")
+    rest = t[len("EPIC: release train " + nxt):]
+    if "epic" in names and t.startswith("EPIC: release train " + nxt) and (rest == "" or rest[0].isspace()):
+        listed |= {int(n) for n in re.findall(r"#(\d+)\b", e.get("body") or "")}
+epic_prefix = "EPIC: release train " + title
+for it in sorted(rows("items.jsonl"), key=lambda i: i.get("number", 0)):
+    n = it.get("number")
+    kind = "pr" if "pull_request" in it else "issue"
+    labels = [l.get("name") for l in (it.get("labels") or [])]
+    t = it.get("title", "")
+    rest = t[len(epic_prefix):]
+    if kind == "issue" and "epic" in labels and t.startswith(epic_prefix) and (rest == "" or rest[0].isspace()):
+        continue                                   # this train's epic: closes after publish
+    if kind == "issue" and "must-carry" in labels:
+        print("BLOCK %s" % n); continue
+    if n in listed:
+        print("MOVE %s %s %s the %s epic lists it" % (kind, n, nxt, nxt))
+    else:
+        print("MOVE %s %s %s not must-carry, and not listed by the %s epic" % (kind, n, backlog, nxt))
+PY
+}
+
+fetch() { # REPO TITLE DIR -> milestones.jsonl, items.jsonl, next_epics.jsonl
+    local repo=$1 title=$2 dir=$3 number
+    gh api --paginate --jq '.[]' "repos/${repo}/milestones?state=all&per_page=100" > "$dir/milestones.jsonl" || return 1
+    number=$(python3 -c 'import json,sys
+n=[json.loads(l)["number"] for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("title")==sys.argv[2]]
+print(n[0] if len(n)==1 else "")' "$dir/milestones.jsonl" "$title") || return 1
+    [ -n "$number" ] || { : > "$dir/items.jsonl"; : > "$dir/next_epics.jsonl"; return 0; }
+    gh api --paginate --jq '.[]' "repos/${repo}/issues?milestone=${number}&state=open&per_page=100" > "$dir/items.jsonl" || return 1
+    gh api --paginate --jq '.[]' "repos/${repo}/issues?labels=epic&state=open&per_page=100" > "$dir/next_epics.jsonl" || return 1
+}
+
+carry() { # REPO TITLE DRY
+    local repo=$1 title=$2 dry=$3 dir p rc=0 moved=0 failed="" kind n target why
+    dir=$(mktemp -d) || return 2
+    fetch "$repo" "$title" "$dir" || { rm -rf -- "${dir:?}"; echo "ENV: a gh read of $repo failed" >&2; return 2; }
+    p=$(plan "$dir" "$title") || { rm -rf -- "${dir:?}"; echo "ENV: the carry plan could not be computed" >&2; return 2; }
+    rm -rf -- "${dir:?}"
+    if grep -q '^ENV ' <<< "$p"; then sed -n 's/^ENV /ENV: /p' <<< "$p" >&2; return 2; fi
+    if grep -q '^BLOCK ' <<< "$p"; then
+        printf 'REFUSE: %s open must-carry issue(s) block the %s cut; NOTHING was carried: %s\n' \
+            "$(grep -c '^BLOCK ' <<< "$p")" "$title" "$(sed -n 's/^BLOCK /#/p' <<< "$p" | tr '\n' ' ')"
+        return 1
+    fi
+    while read -r _ kind n target why; do
+        [ -n "${n:-}" ] || continue
+        if [ "$dry" = 1 ]; then printf 'WOULD CARRY %s #%s -> %s (%s)\n' "$kind" "$n" "$target" "$why"; continue; fi
+        if gh "$kind" edit "$n" --repo "$repo" --milestone "$target" > /dev/null \
+           && gh "$kind" comment "$n" --repo "$repo" \
+                --body "slipped_from: $title -- carried to $target at the $title cut by the release autopilot ($why)" > /dev/null; then
+            printf 'CARRIED %s #%s -> %s (%s)\n' "$kind" "$n" "$target" "$why"; moved=$((moved + 1))
+        else
+            failed="$failed #$n"; rc=2
+        fi
+    done < <(grep '^MOVE ' <<< "$p")
+    if [ "$rc" -ne 0 ]; then printf 'PARTIAL: %s carried, FAILED:%s -- the strict gate will be RED\n' "$moved" "$failed" >&2; return 2; fi
+    printf 'DONE  %s item(s) carried out of %s\n' "$moved" "$title"
+    return 0
+}
+
+self_test() {
+    local d stub rc bad=0 out
+    d=$(mktemp -d) || return 2
+    case "$d" in /tmp/?*) ;; *) echo "self-test: bad temp dir $d"; return 2 ;; esac
+    stub="$d/bin"; mkdir -p "$stub"
+    # a stub gh: serves the fixture reads and RECORDS every write (edit/comment) to $STUB_DIR/writes
+    cat > "$stub/gh" <<'STUB'
+#!/usr/bin/env bash
+case "$1" in
+  api) case "$*" in
+         *'milestones?state=all'*) cat "$STUB_DIR/milestones.jsonl" ;;
+         *'issues?milestone=7&state=open'*) cat "$STUB_DIR/items.jsonl" ;;
+         *'issues?labels=epic&state=open'*) cat "$STUB_DIR/epics.jsonl" ;;
+         *) echo "stub gh: unexpected read $*" >&2; exit 9 ;;
+       esac ;;
+  issue|pr) [ "${FAIL_ON:-}" = "$3" ] && exit 1; echo "$*" >> "$STUB_DIR/writes" ;;
+  *) echo "stub gh: unexpected $*" >&2; exit 9 ;;
+esac
+STUB
+    chmod +x "$stub/gh"
+    ms() { printf '{"title":"%s","number":%s,"state":"%s"}\n' "$1" "$2" "$3"; }
+    item() { printf '{"number":%s,"state":"open","title":"%s","labels":[{"name":"%s"}]%s}\n' "$1" "$2" "$3" "${4:-}"; }
+    fixture() { # base fixture: M=0.70.0 (#7), next 0.71.0 whose epic lists #12, backlog exists
+        { ms 0.70.0 7 open; ms 0.71.0 9 open; ms 0.69.1 5 closed; ms backlog 11 open; } > "$d/milestones.jsonl"
+        { item 20 "EPIC: release train 0.70.0 — schedule" epic
+          item 10 "an unlabelled issue" P1
+          item 12 "listed by the next epic" bug
+          item 13 "a pull request" release ',"pull_request":{"url":"u"}'; } > "$d/items.jsonl"
+        printf '{"number":40,"title":"EPIC: release train 0.71.0 — schedule","labels":[{"name":"epic"}],"body":"carries #12 and #99"}\n' > "$d/epics.jsonl"
+        : > "$d/writes"
+    }
+    run() { STUB_DIR="$d" PATH="$stub:$PATH" bash "$SELF" 0.70.0 --repo o/r "$@" > "$d/out" 2>&1; }
+    ok() { printf 'ok    %s\n' "$1"; }
+    nok() { printf 'FAIL  %s\n' "$1"; sed 's/^/        /' "$d/out"; bad=1; }
+
+    fixture; run; rc=$?
+    if [ "$rc" = 0 ] && grep -q '^issue edit 12 --repo o/r --milestone 0.71.0$' "$d/writes" \
+       && grep -q '^issue edit 10 --repo o/r --milestone backlog$' "$d/writes" \
+       && grep -q '^pr edit 13 --repo o/r --milestone backlog$' "$d/writes" \
+       && ! grep -q ' 20 ' "$d/writes" && [ "$(grep -c ' comment ' "$d/writes")" = 3 ] \
+       && grep -q 'slipped_from: 0.70.0 -- carried to 0.71.0 at the 0.70.0 cut' "$d/writes"; then
+        ok "every non-must-carry item moves: listed -> next release, others -> backlog, one comment each; the epic stays"
+    else nok "the carry moved the wrong set (rc=$rc)"; cat "$d/writes"; fi
+
+    fixture; item 30 "a must-carry blocker" must-carry >> "$d/items.jsonl"; run; rc=$?
+    if [ "$rc" = 1 ] && [ ! -s "$d/writes" ] && grep -q 'REFUSE: 1 open must-carry issue(s) block the 0.70.0 cut; NOTHING was carried: #30' "$d/out"; then
+        ok "an open must-carry issue REFUSES the carry, and nothing is written"
+    else nok "a must-carry blocker did not refuse cleanly (rc=$rc)"; fi
+
+    fixture; run --dry-run; rc=$?
+    if [ "$rc" = 0 ] && [ ! -s "$d/writes" ] && grep -q 'WOULD CARRY issue #12 -> 0.71.0' "$d/out"; then
+        ok "--dry-run writes nothing"
+    else nok "--dry-run wrote or misplanned (rc=$rc)"; fi
+
+    fixture; FAIL_ON=10 run; rc=$?
+    if [ "$rc" = 2 ] && grep -q 'PARTIAL: .* FAILED: #10' "$d/out"; then
+        ok "a failed write is a named PARTIAL, exit 2, never DONE"
+    else nok "a failed write was not reported (rc=$rc)"; fi
+
+    fixture; { ms 0.70.0 7 open; ms 0.71.0 9 open; } > "$d/milestones.jsonl"; run; rc=$?
+    if [ "$rc" = 2 ] && [ ! -s "$d/writes" ] && grep -q "no 'backlog' milestone" "$d/out"; then
+        ok "no backlog milestone is 'cannot act' (2), nothing written"
+    else nok "a missing backlog milestone was not refused (rc=$rc)"; fi
+
+    fixture; { ms 0.70.0 7 open; ms backlog 11 open; } > "$d/milestones.jsonl"; run; rc=$?
+    if [ "$rc" = 2 ] && [ ! -s "$d/writes" ] && grep -q 'no open milestone after 0.70.0' "$d/out"; then
+        ok "no next milestone is 'cannot act' (2), nothing written"
+    else nok "a missing next milestone was not refused (rc=$rc)"; fi
+
+    fixture; printf '{"number":41,"title":"EPIC: release train 0.71.0x — other","labels":[{"name":"epic"}],"body":"#10"}\n' > "$d/epics.jsonl"; run; rc=$?
+    if [ "$rc" = 0 ] && grep -q '^issue edit 10 --repo o/r --milestone backlog$' "$d/writes" && grep -q '^issue edit 12 --repo o/r --milestone backlog$' "$d/writes"; then
+        ok "another train's epic (prefix 0.71.0x) lists nothing for 0.71.0"
+    else nok "a prefix-matching epic was read as the next release's (rc=$rc)"; fi
+
+    rm -rf -- "${d:?}"
+    [ "$bad" -eq 0 ] && { echo "SELF-TEST PASSED"; return 0; }
+    echo "SELF-TEST FAILED"; return 1
+}
+
+[ $# -gt 0 ] || { self_test; exit $?; }
+[ "${1:-}" = "--self-test" ] && { self_test; exit $?; }
+case "$1" in -h|--help) sed -n '2,20p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;; -*) usage ;; esac
+title=$1; shift; repo="paiml/aprender"; dry=0
+while [ $# -gt 0 ]; do
+    case "$1" in
+        --repo) [ $# -ge 2 ] || usage; repo=$2; shift 2 ;;
+        --dry-run) dry=1; shift ;;
+        *) usage ;;
+    esac
+done
+command -v gh > /dev/null || { echo "ENV: gh is not on PATH" >&2; exit 2; }
+command -v python3 > /dev/null || { echo "ENV: python3 is not on PATH" >&2; exit 2; }
+carry "$repo" "$title" "$dry"; exit $?
```
