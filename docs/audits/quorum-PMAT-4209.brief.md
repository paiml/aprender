You are one of 1 independent reviewers. Judge whether this diff does what its ticket says, and nothing the ticket forbids. Try to REFUTE it: default to FAIL when a test asserts the opposite of the ticket, when a gate is weakened, when a receipt claim is not backed by the diff, or when the change does something the ticket does not ask for. Every finding needs file, line, claim and grounding (cited = you quote the diff; measured = you ran a command; asserted = neither). Return PASS only if you found nothing that refutes it.

## Ticket(s) PMAT-4209 — the diff is judged against ALL of them
### PMAT-4209
📊 Status for: PMAT-4209

   Title: CRUX greedy cell: route the llama-server stop through crux_cell_teardown (no inline kill) (#4209)
   Status: Planned
   Priority: Medium
   Progress: 0%
   GitHub: #4209



## Diff (origin/chore/0.69.1-merge-back...HEAD)
```diff
diff --git a/docs/roadmaps/roadmap.yaml b/docs/roadmaps/roadmap.yaml
index 30b80993b..bcb1d4cb6 100644
--- a/docs/roadmaps/roadmap.yaml
+++ b/docs/roadmaps/roadmap.yaml
@@ -21497,3 +21497,21 @@ roadmap:
   labels:
   - kind:code
   notes: null
+- id: PMAT-4209
+  github_issue: 4209
+  item_type: task
+  title: 'CRUX greedy cell: route the llama-server stop through crux_cell_teardown (no inline kill) (#4209)'
+  status: planned
+  priority: medium
+  assigned_to: null
+  created: 2026-09-24T08:30:51Z
+  updated: 2026-09-24T08:30:51Z
+  spec: null
+  acceptance_criteria:
+  - Greedy cell installs crux_teardown_trap over llama-server.pid; rows read crux_teardown_why and a non-clean teardown makes every greedy row RED; no inline server kill in scripts/lib/crux_cells_*.sh (must-RED row with planted mutant); greedy + serve-code tables stay green.
+  phases: []
+  subtasks: []
+  estimated_effort: null
+  labels:
+  - kind:code
+  notes: null
diff --git a/scripts/check_crux_greedy_rows.sh b/scripts/check_crux_greedy_rows.sh
index cfc2c5711..1f9e4541d 100755
--- a/scripts/check_crux_greedy_rows.sh
+++ b/scripts/check_crux_greedy_rows.sh
@@ -16,6 +16,9 @@
 #   3. llama.cpp unavailable (HAVE_LLAMA=0) → llama rows refused with LLAMA_WHY; apr OFF refused naming why
 #   4. apr prints no `tokens`  → apr OFF refused by name
 #   5. MUTANT: the apr-ON refusal dropped from the lib → the no-flag row goes missing, and that is caught
+#   6. the llama-server is stopped by the cell's teardown trap (#4209): state `clean`, its pid gone from /proc
+#   6b. a teardown that cannot prove the server gone (the kill seam a no-op) → EVERY row REFUSED naming it
+#   6c. no inline server kill in scripts/lib/crux_cells_*.sh (case table + a planted mutant of the old kill line)
 #
 # Exit: 0 every row behaved · 1 a row broke · 2 ENV.
 set -uo pipefail
@@ -121,9 +124,12 @@ ran=gpu; fb=false; [ -n "${STUB_APR_FELL_BACK:-}" ] && { ran=cpu; fb=true; }
 if [ -n "${STUB_APR_NO_TOKENS:-}" ]; then printf '{"text": "4"}\n'; else printf '{"text": "4", "tokens": %s, "finish_reason": "stop", "backend": {"requested": "gpu", "ran": "%s", "fell_back": %s}}\n' "$ids" "$ran" "$fb"; fi
 SH
 chmod +x "$BIN/llama-server" "$BIN/apr"
+# the teardown's GPU check reads a stub nvidia-smi that lists no compute app: this table owns no GPU
+printf '#!/bin/sh\nexit 0\n' > "$TMP/shim/nvidia-smi"; chmod +x "$TMP/shim/nvidia-smi"
+export CRUX_NVIDIA_SMI="$TMP/shim/nvidia-smi" CRUX_TEARDOWN_GPU_POLLS=2
 
 # The dogfood's own cell helpers, extracted by name.
-python3 - "$DOGFOOD" "$TMP/helpers.sh" <<'PY'
+python3 - "$DOGFOOD" "$TMP/helpers.sh" "$ROOT/scripts/lib/crux_cells_serve_code.sh" <<'PY'
 import re, sys
 s = open(sys.argv[1]).read()
 out = []
@@ -132,6 +138,13 @@ for name in ("cell_add", "serve_wait_line", "free_port", "run_cell"):
     if not m:
         sys.exit("helper %s() not found in the dogfood: update this check" % name)
     out.append(m.group(0))
+# the teardown trap the dogfood sources from the serve/code lib BEFORE the greedy lib (#4209)
+s = open(sys.argv[3]).read()
+for name in ("crux_teardown_trap", "crux_teardown_why"):
+    m = re.search(r"^%s\(\) \{.*?^\}\n" % name, s, re.S | re.M)
+    if not m:
+        sys.exit("helper %s() not found in crux_cells_serve_code.sh: update this check" % name)
+    out.append(m.group(0))
 open(sys.argv[2], "w").write("\n".join(out))
 PY
 [ $? -eq 0 ] || { printf '%s: ENV - could not extract the dogfood cell helpers\n' "$PROG" >&2; exit 2; }
@@ -264,6 +277,42 @@ run_case mutant "$TMP/mutant-lib.sh"
 got=$(rows mutant | grep -c '^row apr on')
 [ "$got" = 0 ] && ok "MUTANT (apr ON refusal dropped) is caught: the no-flag table's ON row is gone" || broke "MUTANT not caught ($got apr-ON rows)"
 
+# Row 6: the server is stopped by the teardown trap, and PROVEN gone (#4209)
+g="$TMP/up/abc123abc123/greedy"
+st=$(cat "$g/teardown.state" 2>/dev/null); spid=$(cat "$g/llama-server.pid" 2>/dev/null)
+if [ "$st" = clean ] && [ -n "$spid" ] && [ ! -d "/proc/$spid" ] && grep -q '^trap .*crux_cell_teardown.sh' "$g/cell-greedy.sh"; then
+  ok "the llama-server is stopped by the cell's teardown trap: state clean, pid $spid gone"
+else
+  broke "greedy teardown: state '$st', pid '$spid' (alive: $([ -d "/proc/${spid:-x}" ] && echo yes || echo no))"
+fi
+# Row 6b: the kill seam a no-op → the server survives TERM and KILL → the teardown FAILS → every row RED
+run_case tdfail "$LIB" STUB_APR_THINKING=1 CRUX_TEARDOWN_KILL=true
+pkill -f "$TMP/bin/llama-server" 2>/dev/null
+got=$(rows tdfail)
+n=$(printf '%s\n' "$got" | grep -c '^row ')
+nred=$(printf '%s\n' "$got" | grep -c '^row .* REFUSED cell teardown FAILED: server pid(s) ')
+[ "$n" -ge 6 ] && [ "$n" = "$nred" ] && ok "a teardown that cannot prove the server gone makes EVERY greedy row RED ($nred/$n)" \
+  || { broke "teardown FAILED rows: $nred of $n refused"; printf '%s\n' "$got" | sed 's/^/        /'; }
+
+# Row 6c: no crux cell lib stops a server with an inline kill — the trap is the one way (#4209). Case table first.
+inline_kill() { grep -nE '^[^#]*(^|[^[:alnum:]_-])(p?kill|killall)([^[:alnum:]_-]|$)' "$@"; }
+ct_ok=1
+for l in "  [ \"\$HAVE_LLAMA\" = 1 ] && printf 'kill \"\$(cat %q)\" 2> /dev/null\\n' \"\$d/x.pid\" >> \"\$cell\"" \
+         'kill "$pid"' '  pkill -f llama-server' 'x; killall llama-server'; do
+  printf '%s\n' "$l" | inline_kill > /dev/null || { ct_ok=0; broke "inline-kill scan missed: $l"; }
+done
+for l in '# a timeout kill runs the trap too' '  crux_teardown_trap "$cell" "$d/t" "$d/p.pid"' 'skill=1' 'kill_seam=x' '--no-kill'; do
+  printf '%s\n' "$l" | inline_kill > /dev/null && { ct_ok=0; broke "inline-kill scan false positive: $l"; }
+done
+[ "$ct_ok" = 1 ] && ok "inline-kill scan: case table (4 must-match, 5 must-not-match)"
+hits=$(inline_kill "$ROOT"/scripts/lib/crux_cells_*.sh)
+[ -z "$hits" ] && ok "no scripts/lib/crux_cells_*.sh stops a server inline: the teardown trap is the only way" \
+  || { broke "inline server kill in a crux cell lib (route it through crux_teardown_trap):"; printf '%s\n' "$hits" | sed 's/^/        /'; }
+cp "$LIB" "$TMP/mutant-kill.sh"
+printf '  [ "$HAVE_LLAMA" = 1 ] && printf %s "$d/llama-server.pid" >> "$cell"\n' "'kill \"\$(cat %q)\" 2> /dev/null\\n'" >> "$TMP/mutant-kill.sh"
+[ -n "$(inline_kill "$TMP/mutant-kill.sh")" ] && ok "MUTANT (the old inline kill line planted back) is caught by the scan" \
+  || broke "MUTANT inline kill NOT caught"
+
 printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
 [ "$FAIL" -eq 0 ] || exit 1
 exit 0
diff --git a/scripts/lib/crux_cells_greedy.sh b/scripts/lib/crux_cells_greedy.sh
index d60a4170e..ebf1a9658 100644
--- a/scripts/lib/crux_cells_greedy.sh
+++ b/scripts/lib/crux_cells_greedy.sh
@@ -20,12 +20,19 @@
 # Each is one `kind: "greedy"` manifest row, key (model_sha256, host, prompt_id, thinking), whose `tokens`
 # file is the `raw` object: {generated_ids, generated_text, greedy, special, max_tokens}. Both engines use the
 # same max_tokens ($GREEDY_MAXTOK).
+#
+# The llama-server is stopped by crux_teardown_trap (scripts/lib/crux_cells_serve_code.sh), the cell's EXIT/TERM/INT
+# trap that PROVES it gone from /proc and nvidia-smi before the lock drops (#4209) — never an inline `kill` line,
+# which a timeout skips and which cannot see device memory. Anything but a `clean` teardown makes EVERY row RED.
 greedy_cells() {
   local d="$WORK/$SHA12/greedy" cell port pid th content think_flag=0 aprflag
   mkdir -p "$d"
   "$APR" run --help 2>/dev/null | grep -q -- '--thinking' && think_flag=1
   cell="$d/cell-greedy.sh"
+  # a pid file left by an earlier run would name a pid the kernel may have reused: the trap must see only this cell's
+  rm -f -- "$d/llama-server.pid" "$d/teardown.state"
   printf '#!/usr/bin/env bash\n# one CRUX greedy cell: apr + llama.cpp greedy ids on the identical GGUF\n' > "$cell"
+  crux_teardown_trap "$cell" "$d/teardown.state" "$d/llama-server.pid"
   if [ "$HAVE_LLAMA" = 1 ]; then
     port=$(free_port)
     { printf '%q ' "$LLAMA_SERVER" -m "$M" --port "$port" --host 127.0.0.1 -c "$CTX" -ngl "$NGL" "${LLAMA_DEV[@]}" \
@@ -55,21 +62,23 @@ greedy_cells() {
       fi
     done
   done
-  [ "$HAVE_LLAMA" = 1 ] && printf 'kill "$(cat %q)" 2> /dev/null; wait "$(cat %q)" 2> /dev/null\n' \
-    "$d/llama-server.pid" "$d/llama-server.pid" >> "$cell"
   printf 'exit 0\n' >> "$cell"
   run_cell "$cell"
+  local td_why
+  td_why=$(crux_teardown_why "$d/teardown.state")
   python3 - "$MANIFEST" "$SHA" "$HOST" "$BACKEND" "$d" "$GREEDY_MAXTOK" "$HAVE_LLAMA" "${LLAMA_WHY:-}" \
-    "${CELL_WHY:-}" "$think_flag" $GREEDY_PIDS <<'PY'
+    "${CELL_WHY:-}" "$think_flag" "$td_why" $GREEDY_PIDS <<'PY'
 import json, os, sys
-m, sha, host, backend, d, maxtok, llama_ok, llama_why, cell_why, think_flag = sys.argv[1:11]
-pids = sys.argv[11:]
+m, sha, host, backend, d, maxtok, llama_ok, llama_why, cell_why, think_flag, cell_fault = sys.argv[1:12]
+pids = sys.argv[12:]
 NO_ON = ("#3723: this apr has no `run --thinking` flag — realizar routes every Qwen3/Qwen3.5 to the no-think "
          "template, and a pre-rendered thinking-ON prompt is escaped (zero-width space inside its special tokens), "
          "so apr cannot generate greedily with thinking ON")
 
 
 def row(engine, pid, th, path, refused, source="apr"):
+    if cell_fault:  # the teardown did not prove the server gone: no row of this cell is a measurement (#4209)
+        refused = cell_fault
     r = {"kind": "greedy", "engine": engine, "model_sha256": sha, "host": host, "backend": backend, "prompt_id": pid,
          "thinking": th, "prompt_source": source, "max_tokens": int(maxtok), "tokens": None, "logits": None,
          "refused": refused}
```
