You are one of 1 independent reviewers. Judge whether this diff does what its ticket says, and nothing the ticket forbids. Try to REFUTE it: default to FAIL when a test asserts the opposite of the ticket, when a gate is weakened, when a receipt claim is not backed by the diff, or when the change does something the ticket does not ask for. Every finding needs file, line, claim and grounding (cited = you quote the diff; measured = you ran a command; asserted = neither). Return PASS only if you found nothing that refutes it.

## Ticket(s) PMAT-4225 — the diff is judged against ALL of them
### PMAT-4225
📊 Status for: PMAT-4225

   Title: check_crux_greedy_rows: list kill -TERM / kill -9 forms explicitly in the inline-kill case table (#4225)
   Status: Planned
   Priority: Low
   Progress: 0%
   GitHub: #4225



## Diff (origin/feat/4209-greedy-teardown...HEAD)
```diff
diff --git a/docs/roadmaps/roadmap.yaml b/docs/roadmaps/roadmap.yaml
index bcb1d4cb6..1f2a036f6 100644
--- a/docs/roadmaps/roadmap.yaml
+++ b/docs/roadmaps/roadmap.yaml
@@ -21515,3 +21515,21 @@ roadmap:
   labels:
   - kind:code
   notes: null
+- id: PMAT-4225
+  github_issue: 4225
+  item_type: task
+  title: 'check_crux_greedy_rows: list kill -TERM / kill -9 forms explicitly in the inline-kill case table (#4225)'
+  status: planned
+  priority: low
+  assigned_to: null
+  created: 2026-09-24T09:10:00Z
+  updated: 2026-09-24T09:10:00Z
+  spec: null
+  acceptance_criteria:
+  - The row-6c case table lists kill -TERM and kill -9 as must-match; a scan regex that drops flag forms turns the table RED.
+  phases: []
+  subtasks: []
+  estimated_effort: null
+  labels:
+  - kind:code
+  notes: null
diff --git a/scripts/check_crux_greedy_rows.sh b/scripts/check_crux_greedy_rows.sh
index 1f9e4541d..923149065 100755
--- a/scripts/check_crux_greedy_rows.sh
+++ b/scripts/check_crux_greedy_rows.sh
@@ -298,13 +298,13 @@ nred=$(printf '%s\n' "$got" | grep -c '^row .* REFUSED cell teardown FAILED: ser
 inline_kill() { grep -nE '^[^#]*(^|[^[:alnum:]_-])(p?kill|killall)([^[:alnum:]_-]|$)' "$@"; }
 ct_ok=1
 for l in "  [ \"\$HAVE_LLAMA\" = 1 ] && printf 'kill \"\$(cat %q)\" 2> /dev/null\\n' \"\$d/x.pid\" >> \"\$cell\"" \
-         'kill "$pid"' '  pkill -f llama-server' 'x; killall llama-server'; do
+         'kill "$pid"' 'kill -TERM "$pid"' 'kill -9 $(cat p.pid)' '  pkill -f llama-server' 'x; killall llama-server'; do
   printf '%s\n' "$l" | inline_kill > /dev/null || { ct_ok=0; broke "inline-kill scan missed: $l"; }
 done
 for l in '# a timeout kill runs the trap too' '  crux_teardown_trap "$cell" "$d/t" "$d/p.pid"' 'skill=1' 'kill_seam=x' '--no-kill'; do
   printf '%s\n' "$l" | inline_kill > /dev/null && { ct_ok=0; broke "inline-kill scan false positive: $l"; }
 done
-[ "$ct_ok" = 1 ] && ok "inline-kill scan: case table (4 must-match, 5 must-not-match)"
+[ "$ct_ok" = 1 ] && ok "inline-kill scan: case table (6 must-match incl. kill -TERM / kill -9, 5 must-not-match)"
 hits=$(inline_kill "$ROOT"/scripts/lib/crux_cells_*.sh)
 [ -z "$hits" ] && ok "no scripts/lib/crux_cells_*.sh stops a server inline: the teardown trap is the only way" \
   || { broke "inline server kill in a crux cell lib (route it through crux_teardown_trap):"; printf '%s\n' "$hits" | sed 's/^/        /'; }
```
