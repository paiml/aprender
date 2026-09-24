You are one of 1 independent reviewers. Judge whether this diff does what its ticket says, and nothing the ticket forbids. Try to REFUTE it: default to FAIL when a test asserts the opposite of the ticket, when a gate is weakened, when a receipt claim is not backed by the diff, or when the change does something the ticket does not ask for. Every finding needs file, line, claim and grounding (cited = you quote the diff; measured = you ran a command; asserted = neither). Return PASS only if you found nothing that refutes it.

## Ticket(s) PMAT-4036 — the diff is judged against ALL of them
### PMAT-4036
📊 Status for: PMAT-4036

   Title: CRUX: reference cache keyed by model sha + prompt ids + oracle version; only the apr legs run per host (+ one source-weight engine load per mode, #4033 lever c)
   Status: InProgress
   Priority: High
   Progress: 50%
   GitHub: #4036



## Receipt

## Receipt PMAT-4036
# PMAT-4036 implementation receipt: CRUX reference cache + one source-weight load per mode

Ticket: paiml/aprender#4036, part of #4033, lever c. Branch `feat/4036-crux-reference-cache`, off #4046's head `9345331b9`.
Author: aprender-83 (claude-opus-5-5).

## What changed

- `scripts/lib/crux_ref_cache.py` (new) caches the rows of the **pure comparators**: hf, vLLM and llamafile.
  - **Key.** Each entry is keyed on model sha256, thinking mode, backend, engine, row verb, prompt id, and also:
    - the prompt object's sha256;
    - the cap the engine is given: the mode's global cap for run/chat, the prompt's own for serve/code;
    - temperature, seed and context;
    - the oracle's package versions;
    - the HF source;
    - a digest of every harness file that produces a row, the lib itself included.
  - **Scope.** The host is not in the key. llama.cpp is refused by name, because its llama-server renders apr's raw-prompt serve routes.
  - **Outcomes.** There are three, and no fourth:
    - hit: inject the rows, and the engines do not run;
    - miss: run the mode and store its clean rows;
    - stale: refuse every reference row, which is RED and never reused. An entry is stale when any of three independent defenses fails:
      - its **content digest** fails: the WHOLE entry (key, rows, files map, origin) is sealed at store time, so any field added, removed or changed anywhere is caught (quorum round 4, lane 1; the origin since quorum round 6, lane 2);
      - an **artifact's sha256** moved;
      - a row's **pointer** is not a plain name that the entry's own `files{}` holds, even if the digest was re-sealed (quorum round 3, lane 2);
      - `files{}` holds a file **no row points at** (quorum round 5, lane 2);
      - an **artifact field** (`stdout`, `stderr`) is neither null nor a `refcache:` pointer, even re-sealed (quorum round 8, lane 2).
  - **Exit contract.** `lookup` exits 0 hit, 10 miss, 11 stale. A keying refusal (1) or a crash (2) is none of the three, and the dogfood declines the run on it. A crash once shared the miss code, so a corrupted cache was silently recomputed (quorum round 3, lane 2).
- `scripts/crux_inference_dogfood.sh` changes:
  - `--reference-cache <dir>` adds a lookup per mode and an inject or store after the mode.
  - `crux_batched` + `plugin_batch_cells`: a source-weight driver that answers `gen-batch --help` loads once per mode per interface. That is one cell for run+chat in-process and one for serve+code through the engine's own server, instead of once per prompt × verb. `CRUX_NO_BATCH=1` keeps the per-cell path.
  - The meta records the cache result for each mode.
- `scripts/lib/crux_cells_serve_code.sh`: the per-prompt serve/code paths skip a batched engine (`crux_lib_batched`).
- `scripts/crux_sweep_shards.sh`: `--reference-cache` is passed through to the certified shards only.

## Measured on real engines (gx10, GB10)

**Setup.**
- One certified shard: Qwen3.5-2B-Q4_K_M (`aaf42c8b…`), thinking OFF.
- The 3 known-answer controls, verbs run/chat/serve/code.
- Release binary X2 `d8a6df53a`, sha256 `53e506d466bb0262…0632`.
- Run under `/tmp/apr-gpu.lock`.
- The receipts are in `evidence/crux/4036-measure/<leg>/gx10-gpu.json`, and the wall times in `evidence/crux/4036-measure/times.txt`.

| Leg | Commit | Wall time | Receipt |
|---|---|---|---|
| `cold`: baseline, one engine load per cell | `1b483fff6` | 1037 s (17.3 min) | 32/32 GREEN |
| `batched`: batching, cold | `3488e25af` | 306 s (5.1 min) | 32/32 GREEN |
| `final-cold` | `78df5a98c` | 289 s (4.8 min) | 32/32 GREEN |
| `final-warm`: cache hit, hf + vLLM skipped | `78df5a98c` | 130 s (2.2 min) | 32/32 GREEN |
| `head-cold` | `1eba4178c` (reviewed head) | 373 s (6.2 min) | 32/32 GREEN |
| `head-warm`: cache hit | `1eba4178c` (reviewed head) | 158 s (2.6 min) | 32/32 GREEN |

**Comparison with the baseline.**
- Run-to-run variance on the same code: `final-*` vs `head-*` is 289 vs 373 s cold and 130 vs 158 s warm. The round-1 fix changes only key contents, not what runs. Summary: 17.3 min → 4.8–6.2 min batched, and → 2.2–2.6 min on a warm cache.
- Every GREEN leg's receipt is identical to the baseline **cell for cell**: same 32 keys, same 32 verdicts.
- Batching changed no reference answer: all 20 hf/vLLM rows have the same text and refusal state, unbatched vs batched. The rows of both legs are in `evidence/crux/4036-measure/reference-texts.json` (`identical_ignoring_batch_id: true`).

**A defect this measurement found.** The `warm` leg at `1b483fff6` cached llama.cpp too. It took 74 s, and its receipt was RED on 14 of 32 cells, all of them apr serve cells on raw-prompt routes ("no reference renderer was given"). Fixed in `78df5a98c`: llama.cpp is never cached. The failure was RED, never a false GREEN.

## Hermetic tables (real dogfood + real judge, stub engines)

**`scripts/check_crux_ref_cache.sh` — 17/17**

1. Cold stores.
2. Warm makes 0 reference calls, the receipt is identical cell for cell, and every injected row is provenanced.
3. MUST-RED: an answer-*preserving* edit to a stored entry is refused STALE, and every cell is RED.
4. A device change is a hit; a package version change is a miss, recomputed.
5. A refused row is never stored.
6. MUTANT, stale check disabled: the mutant fills its own cache, then REUSES the edited entry (0 calls, GREEN). Row 3 is what catches it.
7. llama.cpp is refused.
8. A new global cap is a miss for run/chat.
9. MUST-RED: a row's artifact pointer is rewritten to a `../` traversal while the key and `files{}` stay intact, and the content digest is re-sealed, so only the pointer rule can refuse it. It is STALE, with 0 engine calls, and every cell is RED.
10. The exit contract: miss 10, keying refusal 1, crash 2.
11. MUST-DECLINE: a lookup that crashes inside the real dogfood declines the run (rc 2, no receipt). It is never recomputed silently.
12. An edited pointer that resolves to the SAME hashed file, with the digest re-sealed, is STALE on the real lib, and REUSED by a mutant with the row↔file rules disabled (the pointer rule and its converse). Those rules are what refuse it.
13. A field REMOVED from a stored row, either `stdout` or `backend` (which no per-field rule reads and whose removal orphans no file), is STALE by the content digest. A mutant without the digest REUSES the `backend` edit.
14. An unreferenced file added to `files{}`, hashed and with the digest re-sealed, is STALE (quorum round 5, lane 2). A mutant without the converse rule REUSES it.
15. MUST-RED: an origin-only edit, not re-sealed, is STALE. The seal once covered only {key, rows, files}, and this edit passed (quorum round 6, lane 2).
16. A driver that answers one item TWICE: the cell is RED, the entry is never stored (an entry is exactly one row), and the next run is a MISS measured again (quorum round 7, lane 2).
17. An artifact field (`stdout`) holds a bare string, its file is dropped and the digest re-sealed. It is STALE by the field-shape rule: an artifact field is null or a `refcache:` pointer (quorum round 8, lane 2).

**`scripts/check_crux_plugin_batch.sh` — 6/6**

- 1 load for 2 items.
- `CRUX_NO_BATCH` gives the same receipt and the same rows, row for row.
- A driver without the capability keeps the per-prompt path.
- MUST-RED: a dropped item is refused by name.
- MUST-RED: an item the batch answered TWICE (one row wrong) is refused by name, and that refusal is the row the judge keeps. The count guard once saw only a missing row (quorum round 7, lane 2). The per-prompt paths got the same guard.
- MUTANT, never batch: caught by the load count.

**Unchanged on this diff:** greedy 12/12, ollama-in-lock 6/6, serve_code 66/66, judge 153/153, oracles PASS.

## Not covered

- **A coherent forgery is not covered.** Someone who can write the cache dir can rewrite an artifact, update its `files{}` hash and re-seal the content digest, and lookup will HIT (quorum round 5, lane 2, measured).
  - Nothing stored beside the entry can refuse a writer who recomputes every seal.
  - That same writer can edit the dogfood, the judge or the receipts.
  - The checks cover corruption, partial edits and leftovers from another harness, not forgery. Forgery resistance would need a key held outside the cache.
  - The PMAT-4036 acceptance criterion originally said "any edit". It has been narrowed to this threat model, stated here rather than overclaimed.

- Entries stored before the content digest existed carry none, so they read as stale (RED) and must be deleted. The only such caches are the gx10 measurement directories of this ticket.

- ON-mode vLLM serve rows are refused with a 400 until #4029 lands. Refusals are never cached, so ON modes stay misses, and run in full, until then.
- The cache is shared between hosts by copying the directory. No transport is part of this change.

## Diff (origin/chore/0.69.1-merge-back...HEAD)
```diff
diff --git a/docs/audits/impl-PMAT-4036-receipt.md b/docs/audits/impl-PMAT-4036-receipt.md
new file mode 100644
index 000000000..f679ae88b
--- /dev/null
+++ b/docs/audits/impl-PMAT-4036-receipt.md
@@ -0,0 +1,103 @@
+# PMAT-4036 implementation receipt: CRUX reference cache + one source-weight load per mode
+
+Ticket: paiml/aprender#4036, part of #4033, lever c. Branch `feat/4036-crux-reference-cache`, off #4046's head `9345331b9`.
+Author: aprender-83 (claude-opus-5-5).
+
+## What changed
+
+- `scripts/lib/crux_ref_cache.py` (new) caches the rows of the **pure comparators**: hf, vLLM and llamafile.
+  - **Key.** Each entry is keyed on model sha256, thinking mode, backend, engine, row verb, prompt id, and also:
+    - the prompt object's sha256;
+    - the cap the engine is given: the mode's global cap for run/chat, the prompt's own for serve/code;
+    - temperature, seed and context;
+    - the oracle's package versions;
+    - the HF source;
+    - a digest of every harness file that produces a row, the lib itself included.
+  - **Scope.** The host is not in the key. llama.cpp is refused by name, because its llama-server renders apr's raw-prompt serve routes.
+  - **Outcomes.** There are three, and no fourth:
+    - hit: inject the rows, and the engines do not run;
+    - miss: run the mode and store its clean rows;
+    - stale: refuse every reference row, which is RED and never reused. An entry is stale when any of three independent defenses fails:
+      - its **content digest** fails: the WHOLE entry (key, rows, files map, origin) is sealed at store time, so any field added, removed or changed anywhere is caught (quorum round 4, lane 1; the origin since quorum round 6, lane 2);
+      - an **artifact's sha256** moved;
+      - a row's **pointer** is not a plain name that the entry's own `files{}` holds, even if the digest was re-sealed (quorum round 3, lane 2);
+      - `files{}` holds a file **no row points at** (quorum round 5, lane 2);
+      - an **artifact field** (`stdout`, `stderr`) is neither null nor a `refcache:` pointer, even re-sealed (quorum round 8, lane 2).
+  - **Exit contract.** `lookup` exits 0 hit, 10 miss, 11 stale. A keying refusal (1) or a crash (2) is none of the three, and the dogfood declines the run on it. A crash once shared the miss code, so a corrupted cache was silently recomputed (quorum round 3, lane 2).
+- `scripts/crux_inference_dogfood.sh` changes:
+  - `--reference-cache <dir>` adds a lookup per mode and an inject or store after the mode.
+  - `crux_batched` + `plugin_batch_cells`: a source-weight driver that answers `gen-batch --help` loads once per mode per interface. That is one cell for run+chat in-process and one for serve+code through the engine's own server, instead of once per prompt × verb. `CRUX_NO_BATCH=1` keeps the per-cell path.
+  - The meta records the cache result for each mode.
+- `scripts/lib/crux_cells_serve_code.sh`: the per-prompt serve/code paths skip a batched engine (`crux_lib_batched`).
+- `scripts/crux_sweep_shards.sh`: `--reference-cache` is passed through to the certified shards only.
+
+## Measured on real engines (gx10, GB10)
+
+**Setup.**
+- One certified shard: Qwen3.5-2B-Q4_K_M (`aaf42c8b…`), thinking OFF.
+- The 3 known-answer controls, verbs run/chat/serve/code.
+- Release binary X2 `d8a6df53a`, sha256 `53e506d466bb0262…0632`.
+- Run under `/tmp/apr-gpu.lock`.
+- The receipts are in `evidence/crux/4036-measure/<leg>/gx10-gpu.json`, and the wall times in `evidence/crux/4036-measure/times.txt`.
+
+| Leg | Commit | Wall time | Receipt |
+|---|---|---|---|
+| `cold`: baseline, one engine load per cell | `1b483fff6` | 1037 s (17.3 min) | 32/32 GREEN |
+| `batched`: batching, cold | `3488e25af` | 306 s (5.1 min) | 32/32 GREEN |
+| `final-cold` | `78df5a98c` | 289 s (4.8 min) | 32/32 GREEN |
+| `final-warm`: cache hit, hf + vLLM skipped | `78df5a98c` | 130 s (2.2 min) | 32/32 GREEN |
+| `head-cold` | `1eba4178c` (reviewed head) | 373 s (6.2 min) | 32/32 GREEN |
+| `head-warm`: cache hit | `1eba4178c` (reviewed head) | 158 s (2.6 min) | 32/32 GREEN |
+
+**Comparison with the baseline.**
+- Run-to-run variance on the same code: `final-*` vs `head-*` is 289 vs 373 s cold and 130 vs 158 s warm. The round-1 fix changes only key contents, not what runs. Summary: 17.3 min → 4.8–6.2 min batched, and → 2.2–2.6 min on a warm cache.
+- Every GREEN leg's receipt is identical to the baseline **cell for cell**: same 32 keys, same 32 verdicts.
+- Batching changed no reference answer: all 20 hf/vLLM rows have the same text and refusal state, unbatched vs batched. The rows of both legs are in `evidence/crux/4036-measure/reference-texts.json` (`identical_ignoring_batch_id: true`).
+
+**A defect this measurement found.** The `warm` leg at `1b483fff6` cached llama.cpp too. It took 74 s, and its receipt was RED on 14 of 32 cells, all of them apr serve cells on raw-prompt routes ("no reference renderer was given"). Fixed in `78df5a98c`: llama.cpp is never cached. The failure was RED, never a false GREEN.
+
+## Hermetic tables (real dogfood + real judge, stub engines)
+
+**`scripts/check_crux_ref_cache.sh` — 17/17**
+
+1. Cold stores.
+2. Warm makes 0 reference calls, the receipt is identical cell for cell, and every injected row is provenanced.
+3. MUST-RED: an answer-*preserving* edit to a stored entry is refused STALE, and every cell is RED.
+4. A device change is a hit; a package version change is a miss, recomputed.
+5. A refused row is never stored.
+6. MUTANT, stale check disabled: the mutant fills its own cache, then REUSES the edited entry (0 calls, GREEN). Row 3 is what catches it.
+7. llama.cpp is refused.
+8. A new global cap is a miss for run/chat.
+9. MUST-RED: a row's artifact pointer is rewritten to a `../` traversal while the key and `files{}` stay intact, and the content digest is re-sealed, so only the pointer rule can refuse it. It is STALE, with 0 engine calls, and every cell is RED.
+10. The exit contract: miss 10, keying refusal 1, crash 2.
+11. MUST-DECLINE: a lookup that crashes inside the real dogfood declines the run (rc 2, no receipt). It is never recomputed silently.
+12. An edited pointer that resolves to the SAME hashed file, with the digest re-sealed, is STALE on the real lib, and REUSED by a mutant with the row↔file rules disabled (the pointer rule and its converse). Those rules are what refuse it.
+13. A field REMOVED from a stored row, either `stdout` or `backend` (which no per-field rule reads and whose removal orphans no file), is STALE by the content digest. A mutant without the digest REUSES the `backend` edit.
+14. An unreferenced file added to `files{}`, hashed and with the digest re-sealed, is STALE (quorum round 5, lane 2). A mutant without the converse rule REUSES it.
+15. MUST-RED: an origin-only edit, not re-sealed, is STALE. The seal once covered only {key, rows, files}, and this edit passed (quorum round 6, lane 2).
+16. A driver that answers one item TWICE: the cell is RED, the entry is never stored (an entry is exactly one row), and the next run is a MISS measured again (quorum round 7, lane 2).
+17. An artifact field (`stdout`) holds a bare string, its file is dropped and the digest re-sealed. It is STALE by the field-shape rule: an artifact field is null or a `refcache:` pointer (quorum round 8, lane 2).
+
+**`scripts/check_crux_plugin_batch.sh` — 6/6**
+
+- 1 load for 2 items.
+- `CRUX_NO_BATCH` gives the same receipt and the same rows, row for row.
+- A driver without the capability keeps the per-prompt path.
+- MUST-RED: a dropped item is refused by name.
+- MUST-RED: an item the batch answered TWICE (one row wrong) is refused by name, and that refusal is the row the judge keeps. The count guard once saw only a missing row (quorum round 7, lane 2). The per-prompt paths got the same guard.
+- MUTANT, never batch: caught by the load count.
+
+**Unchanged on this diff:** greedy 12/12, ollama-in-lock 6/6, serve_code 66/66, judge 153/153, oracles PASS.
+
+## Not covered
+
+- **A coherent forgery is not covered.** Someone who can write the cache dir can rewrite an artifact, update its `files{}` hash and re-seal the content digest, and lookup will HIT (quorum round 5, lane 2, measured).
+  - Nothing stored beside the entry can refuse a writer who recomputes every seal.
+  - That same writer can edit the dogfood, the judge or the receipts.
+  - The checks cover corruption, partial edits and leftovers from another harness, not forgery. Forgery resistance would need a key held outside the cache.
+  - The PMAT-4036 acceptance criterion originally said "any edit". It has been narrowed to this threat model, stated here rather than overclaimed.
+
+- Entries stored before the content digest existed carry none, so they read as stale (RED) and must be deleted. The only such caches are the gx10 measurement directories of this ticket.
+
+- ON-mode vLLM serve rows are refused with a 400 until #4029 lands. Refusals are never cached, so ON modes stay misses, and run in full, until then.
+- The cache is shared between hosts by copying the directory. No transport is part of this change.
diff --git a/docs/roadmaps/roadmap.yaml b/docs/roadmaps/roadmap.yaml
index ca3ad7c2c..2bd2d7af3 100644
--- a/docs/roadmaps/roadmap.yaml
+++ b/docs/roadmaps/roadmap.yaml
@@ -21444,3 +21444,29 @@ roadmap:
   estimated_effort: null
   labels: []
   notes: null
+- id: PMAT-4036
+  github_issue: 4036
+  item_type: task
+  title: 'CRUX: reference cache keyed by model sha + prompt ids + oracle version; only the apr legs run per host (+ one
+    source-weight engine load per mode, #4033 lever c)'
+  status: in_progress
+  priority: high
+  assigned_to: aprender-83
+  created: 2026-09-23T18:00:00Z
+  updated: 2026-09-23T18:40:00Z
+  spec: null
+  acceptance_criteria:
+  - 'A reference row is cached under a key bound to model sha256, thinking mode, backend, engine, verb, prompt sha256,
+    sampling, oracle version and the producing harness files; the host is not in the key'
+  - 'On a miss the mode recomputes and stores only clean rows; a stale entry (corrupted, edited in part or in any field,
+    even answer-preserving; NOT a coherent re-seal by a cache writer, which is out of scope and stated) is RED,
+    never reused — a must-RED falsifier in scripts/check_crux_ref_cache.sh'
+  - 'Only pure comparators are cached (hf, vLLM, llamafile); llama.cpp is refused because it renders apr''s serve routes'
+  - 'Source-weight engines load once per mode per interface (gen-batch); rows identical row for row to the per-cell path'
+  - 'The saving is measured in minutes against the 0.69.1 baseline on real engines, receipts identical cell for cell'
+  phases: []
+  subtasks: []
+  estimated_effort: null
+  labels:
+  - kind:code
+  notes: null
diff --git a/evidence/crux/4036-measure/times.txt b/evidence/crux/4036-measure/times.txt
new file mode 100644
index 000000000..8fa0bbb28
--- /dev/null
+++ b/evidence/crux/4036-measure/times.txt
@@ -0,0 +1,11 @@
+cold rc=0 seconds=1037
+warm rc=1 seconds=74
+MEASURE_DONE
+batched rc=0 seconds=306
+BATCHED_DONE
+final-cold rc=0 seconds=289
+final-warm rc=0 seconds=130
+FINAL_DONE
+head-cold rc=0 seconds=373
+head-warm rc=0 seconds=158
+HEAD_DONE
diff --git a/scripts/check_crux_plugin_batch.sh b/scripts/check_crux_plugin_batch.sh
new file mode 100755
index 000000000..a7ce88849
--- /dev/null
+++ b/scripts/check_crux_plugin_batch.sh
@@ -0,0 +1,241 @@
+#!/usr/bin/env bash
+# check_crux_plugin_batch.sh: the case table for batching the source-weight engines (#4036 lever 2,
+# plugin_batch_cells in scripts/crux_inference_dogfood.sh). Hermetic: stub apr, hf and llamafile drivers inside a
+# COPY of scripts/, a private lock file for /tmp/apr-gpu.lock. No GPU, no model.
+#
+# The stub hf driver logs one LOAD per invocation, i.e. per engine load, and writes one row per item.
+# Rows (the REAL dogfood and the REAL judge, --verbs run,chat on the positive control):
+#   1. batched   → hf loaded ONCE for the mode's 2 in-process items; the receipt is GREEN
+#   2. unbatched (CRUX_NO_BATCH=1) → hf loaded once PER ITEM (2); the receipt is identical cell for cell to row 1,
+#                  and hf's rows are the same row for row (every field but the per-load `batch` id and the paths)
+#   3. a driver without gen-batch → the per-prompt path, one load per item: batching is a capability, never assumed
+#   4. MUST-RED: the batch returns no row for one item → that item is REFUSED by name, the cell RED, never absent
+#  4b. MUST-RED: the batch returns TWO rows for one item (one wrong) → refused by name, the cell RED, never a pick
+#   5. MUTANT: crux_batched always false → row 1 sees 2 loads, not 1: the table catches a batch that never happens
+#
+# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
+set -uo pipefail
+
+ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
+PROG=check_crux_plugin_batch
+for t in python3 flock; do
+  command -v "$t" >/dev/null 2>&1 || { printf '%s: ENV - %s is missing\n' "$PROG" "$t" >&2; exit 2; }
+done
+for f in scripts/crux_inference_dogfood.sh scripts/lib/crux_inference_judge.py scripts/crux_inference_prompts.v2.json; do
+  [ -f "$ROOT/$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
+done
+
+TMP=$(mktemp -d) || exit 2
+_rm_tmp() {
+  local w
+  [ -n "${KEEP_TMP:-}" ] && { echo "kept $TMP"; return; }
+  for w in $(sed -n 's/^work kept: //p' "$TMP"/*.log 2>/dev/null); do
+    [ -n "$w" ] && [ "$w" != / ] || continue
+    case "${w:-}" in
+      /tmp/?*) rm -rf -- "${w:?}" || : ;;
+      *) : ;;
+    esac
+  done
+  case "${TMP:-}" in
+    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
+    *) : ;;
+  esac
+}
+trap _rm_tmp EXIT
+unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
+mkdir -p "$TMP/shim"
+printf '#!/bin/sh\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done\n[ $# -gt 0 ] && shift\nexec "$@"\n' > "$TMP/shim/choom"
+chmod +x "$TMP/shim/choom"
+export PATH="$TMP/shim:$PATH"
+
+PASS=0
+FAIL=0
+ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
+broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }
+
+# The stub driver, shared by hf (batch-capable unless STUB_NO_BATCH=1) and llamafile (never batch-capable).
+cat > "$TMP/stub_driver.py" <<'PY'
+import json, os, sys
+eng, argv = sys.argv[1], sys.argv[2:]
+cmd = argv[0] if argv else ""
+batchable = eng == "hf" and not os.environ.get("STUB_NO_BATCH")
+if cmd == "probe":
+    print("%s=1.0.0 transformers=5.0.0 torch=2.0.0 device=stub" % eng); sys.exit(0)
+if cmd == "gen-batch" and "--help" in argv:
+    sys.exit(0 if batchable else 2)
+if cmd not in ("gen", "gen-batch") or (cmd == "gen-batch" and not batchable):
+    sys.exit(2)
+opt = {argv[i][2:]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("--")}
+if cmd == "gen":
+    items = [{"prompt_id": opt["prompt-id"], "verb": opt["verb"], "thinking": opt["thinking"]}]
+else:
+    items = [json.loads(l) for l in open(opt["batch"]) if l.strip()]
+open(os.environ["STUB_CALLS"], "a").write("LOAD %s %d\n" % (eng, len(items)))
+sha = opt["model-sha256"]
+for it in items:
+    if it["verb"] == os.environ.get("STUB_DROP", "-") and eng == "hf":
+        continue
+    d = os.path.join(os.environ["CRUX_WORK"], sha[:12], it["verb"])
+    os.makedirs(d, exist_ok=True)
+    stem = "%s-%s-%s" % (eng, it["prompt_id"], it["thinking"])
+    out = os.path.join(d, stem + ".json")
+    json.dump({"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\n",
+               "reported": {"thinking_requested": it["thinking"], "device": "stub"}}, open(out, "w"))
+    open(os.path.join(d, stem + ".err"), "w").close()
+    row = {"kind": "gen", "engine": eng, "model_sha256": sha, "host": opt["host"], "verb": it["verb"],
+           "thinking": it["thinking"], "backend": opt["backend"], "prompt_id": it["prompt_id"], "rc": 0,
+           "stdout": out, "stderr": os.path.join(d, stem + ".err"), "refused": None}
+    if eng == "hf":
+        row["source"] = {"repo": "stub/src", "revision": "0" * 40, "dtype": "bfloat16"}
+        row["batch"] = {"id": str(os.getpid()), "size": len(items)}
+    open(os.environ["CRUX_MANIFEST"], "a").write(json.dumps(row) + "\n")
+    if it["verb"] == os.environ.get("STUB_DUP", "-") and eng == "hf":
+        wrong = os.path.join(d, stem + ".dup.json")
+        json.dump({"text": "<answer>5</answer>", "reported": {"device": "stub"}}, open(wrong, "w"))
+        open(os.environ["CRUX_MANIFEST"], "a").write(json.dumps(dict(row, stdout=wrong)) + "\n")
+PY
+
+mk_tree() { # mk_tree <dir>: the real scripts with the two drivers stubbed
+  local t="$1" eng
+  mkdir -p "$t"
+  cp -r "$ROOT/scripts" "$t/scripts"
+  for eng in hf llamafile; do
+    printf '#!/usr/bin/env bash\nexec python3 %q %s "$@"\n' "$TMP/stub_driver.py" "$eng" > "$t/scripts/crux_engine_$eng.sh"
+    chmod +x "$t/scripts/crux_engine_$eng.sh"
+  done
+}
+
+BIN="$TMP/bin"; mkdir -p "$BIN"
+cat > "$BIN/apr" <<'SH'
+#!/usr/bin/env bash
+case "${1:-}" in
+  --version) echo "apr 0.0.0 (stub)" ;;
+  run)
+    case " $* " in *" --help "*) echo "  --thinking <MODE>"; exit 0 ;; esac
+    printf '{"text": "<answer>4</answer>", "tokens_generated": 1, "tok_per_sec": 1.0, "finish_reason": "stop", "backend": {"requested": "gpu", "ran": "gpu", "fell_back": false}}\n' ;;
+  chat)
+    while IFS= read -r _turn; do printf 'You: \nAssistant: <answer>4</answer>\n'; done
+    printf 'You: \nGoodbye!\n' ;;
+  *) echo "stub apr: unhandled '$*'" >&2; exit 1 ;;
+esac
+SH
+chmod +x "$BIN/apr"
+MODEL="$TMP/model.gguf"; printf 'GGUF-stub-model' > "$MODEL"
+MSHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
+printf 'sources:\n  %s:\n    hf: {repo: stub/src, revision: "%s", dtype: bfloat16}\n' "$MSHA" "$(printf '0%.0s' $(seq 40))" > "$TMP/hf-sources.yaml"
+python3 - "$ROOT/scripts/crux_inference_prompts.v2.json" "$MSHA" "$TMP/cert.json" <<'PY' || exit 2
+import hashlib, json, sys
+prompts, sha, out = sys.argv[1:4]
+json.dump({"schema": "crux-prompt-certification/v1", "prompts": "scripts/crux_inference_prompts.v2.json",
+           "prompts_sha256": hashlib.sha256(open(prompts, "rb").read()).hexdigest(), "admitted": {},
+           "admitted_by_sha": {sha: ["ctl-2plus2"]}, "admitted_by_sha_thinking": {sha: {"off": ["ctl-2plus2"]}}},
+          open(out, "w"))
+PY
+
+run_row() { # run_row <name> <tree> [env...]
+  local name="$1" tree="$2"; shift 2
+  : > "$TMP/$name.calls"; : > "$TMP/$name.lock"
+  ( cd "$tree" && env STUB_CALLS="$TMP/$name.calls" CRUX_GPU_LOCK="$TMP/$name.lock" GPUQ_BIN=/nonexistent/gpu-q \
+      CRUX_HF_SOURCES="$TMP/hf-sources.yaml" DOGFOOD_ALLOW_UNPINNED=1 APR="$BIN/apr" "$@" \
+      timeout 300 bash scripts/crux_inference_dogfood.sh 0.0.0 --model "$MODEL" --engines apr,hf,llamafile \
+        --verbs run,chat --prompts scripts/crux_inference_prompts.v2.json --certification "$TMP/cert.json" \
+        --only-prompts ctl-2plus2 --thinking-modes off \
+        --host stub --out "$TMP/$name.out" --timeout 60 --keep-work ) > "$TMP/$name.log" 2>&1
+  echo "$?" > "$TMP/$name.rc"
+}
+
+# summary <name>: "<hf loads> | <cell verdicts>"
+summary() {
+  python3 - "$TMP/$1.out/stub-gpu.json" "$TMP/$1.calls" <<'PY'
+import json, sys
+loads = sum(1 for l in open(sys.argv[2]) if l.startswith("LOAD hf "))
+try:
+    d = json.load(open(sys.argv[1]))
+except (OSError, ValueError):
+    print("%d | NO-RECEIPT" % loads); sys.exit()
+print("%d | %s" % (loads, " ".join(sorted("%s/%s=%s" % (c["key"]["verb"], c["key"]["prompt_id"], c["verdict"])
+                                          for c in d.get("cells", []))) or "no-cells"))
+PY
+}
+
+hf_rows() { # hf_rows <name>: hf's rows, with the per-load batch id and the work-dir paths removed
+  python3 - "$(sed -n 's/^work kept: //p' "$TMP/$1.log" | tail -1)" <<'PY'
+import json, os, sys
+w = sys.argv[1]
+rows = []
+for l in open(os.path.join(w, "manifest.jsonl")):
+    r = json.loads(l)
+    if r.get("kind") == "gen" and r.get("engine") == "hf":
+        r.pop("batch", None)
+        for f in ("stdout", "stderr"):
+            if r.get(f):
+                r[f] = os.path.relpath(r[f], w)
+        rows.append(json.dumps(r, sort_keys=True))
+print("\n".join(sorted(rows)))
+PY
+}
+
+printf '%s: the source-weight engines, one load per batch (#4036)\n' "$PROG"
+T="$TMP/tree"; mk_tree "$T"
+
+run_row batched "$T"
+b=$(summary batched)
+case "$b" in
+  "1 | "*=GREEN*) printf '%s' "$b" | grep -q '=RED' && broke "batched: a RED cell ($b)" \
+                  || ok "batched: hf loaded ONCE for the mode's in-process items, receipt GREEN ($b)" ;;
+  *) broke "batched: $b (rc $(cat "$TMP/batched.rc")); log $TMP/batched.log"; tail -4 "$TMP/batched.log" | sed 's/^/        /' ;;
+esac
+
+run_row unbatched "$T" CRUX_NO_BATCH=1
+u=$(summary unbatched)
+if [ "${u%% |*}" = 2 ] && [ "${u#* | }" = "${b#* | }" ] && [ "$(hf_rows batched)" = "$(hf_rows unbatched)" ] \
+   && [ -n "$(hf_rows batched)" ]; then
+  ok "CRUX_NO_BATCH: hf loaded per item (2), receipt identical cell for cell, hf rows identical row for row ($u)"
+else
+  broke "unbatched: $u vs batched $b; rows equal: $([ "$(hf_rows batched)" = "$(hf_rows unbatched)" ] && echo yes || echo no)"
+fi
+
+run_row nocap "$T" STUB_NO_BATCH=1
+n=$(summary nocap)
+[ "${n%% |*}" = 2 ] && [ "${n#* | }" = "${b#* | }" ] \
+  && ok "a driver without gen-batch keeps the per-prompt path: 2 loads, same receipt ($n)" \
+  || broke "no gen-batch capability: $n"
+
+run_row drop "$T" STUB_DROP=chat
+dr=$(summary drop)
+if printf '%s' "$dr" | grep -q 'chat/ctl-2plus2=RED' \
+   && grep -q 'gen-batch exited 0 without a row for this item' "$(sed -n 's/^work kept: //p' "$TMP/drop.log" | tail -1)/manifest.jsonl"; then
+  ok "MUST-RED: an item the batch returned no row for is refused by name, its cell RED ($dr)"
+else
+  broke "dropped item: $dr"
+fi
+
+# Row 5b: MUST-RED — the batch returns TWO rows for one item (one right, one wrong). The count guard only saw a
+# MISSING row, so this passed, the judge kept whichever row came last, and the cache replayed both (quorum round 7,
+# lane 2, measured). Now the item is refused by name, the refusal is the row the judge keeps, and the cell is RED.
+run_row dup "$T" STUB_DUP=chat
+du=$(summary dup)
+if printf '%s' "$du" | grep -q 'chat/ctl-2plus2=RED' \
+   && grep -q 'rows for ONE item is refused' "$(sed -n 's/^work kept: //p' "$TMP/dup.log" | tail -1)/manifest.jsonl"; then
+  ok "MUST-RED: an item the batch answered TWICE is refused by name, its cell RED ($du)"
+else
+  broke "duplicate item: $du"
+fi
+
+MT="$TMP/mutant"; mk_tree "$MT"
+python3 - "$MT/scripts/crux_inference_dogfood.sh" <<'PY'
+import sys
+p = sys.argv[1]
+s = open(p).read()
+a = 'crux_batched() { # crux_batched <engine>: its rows this mode come from plugin_batch_cells, not the per-prompt cells\n'
+assert s.count(a) == 1, "mutation anchor moved: update this check with the dogfood"
+open(p, "w").write(s.replace(a, a + "  return 1\n"))
+PY
+run_row mutant "$MT"
+m=$(summary mutant)
+[ "${m%% |*}" != 1 ] && ok "MUTANT (crux_batched never true) is caught: $m, not 1 load" \
+  || broke "MUTANT not caught: $m"
+
+printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
+[ "$FAIL" -eq 0 ] || exit 1
+exit 0
diff --git a/scripts/check_crux_ref_cache.sh b/scripts/check_crux_ref_cache.sh
new file mode 100755
index 000000000..da4a25bed
--- /dev/null
+++ b/scripts/check_crux_ref_cache.sh
@@ -0,0 +1,558 @@
+#!/usr/bin/env bash
+# check_crux_ref_cache.sh: the case table for the CRUX reference cache (#4036, scripts/lib/crux_ref_cache.py and
+# its hook in scripts/crux_inference_dogfood.sh). Hermetic: stub apr, hf and llamafile drivers stand in for the
+# engines inside a COPY of scripts/, and a private lock file stands in for /tmp/apr-gpu.lock. No GPU, no model.
+#
+# The rows run the REAL dogfood and the REAL judge, and read the receipt, never the cache's own log:
+#   1. cold run (empty cache)   → every reference engine ran, their clean rows were stored, the receipt is GREEN
+#   2. warm run (same cache)    → NO reference engine ran, the receipt is the same cell for cell (key, verdict),
+#                                 and every reference row says it came from the cache and names its origin
+#   3. MUST-RED: one stored artifact edited, its answer still RIGHT → NO reference engine ran, every cell is RED,
+#                                 and the receipt's refusal says STALE. Any edit is RED, not only a wrong answer
+#                                 (a wrong one the judge would catch anyway, which is why this edit keeps it right).
+#   4. oracle key: another host's device on the probe is a HIT (the host is not in the key); a new package
+#                                 version is a MISS, recomputed GREEN, never matched against the old truth
+#   5. a refused reference row is never stored → the next run is a MISS for that engine, not a hit
+#   6. MUTANT: the stale check disabled (why_stale always None) → row 3's receipt is no longer RED: the table
+#                                 catches it
+#   7. llama.cpp is refused by name: it is apr's reference renderer on the serve routes, so it always runs
+#   8. the global cap moved (a bigger budget elsewhere in the prompt set) → a MISS for run/chat, recomputed
+#   9. MUST-RED: a row's artifact pointer rewritten to a `../` traversal (key and files{} intact) → STALE, RED
+#  10. the lookup's exit contract: hit 0 · miss 10 · stale 11; a keying refusal (1) and a crash (2) are neither
+#  11. MUST-DECLINE: a lookup that crashes inside the real dogfood declines the run, never a silent recompute
+#  12. an edited pointer that resolves to the SAME hashed file (digest re-sealed): STALE on the real lib, REUSED by a
+#      mutant without the row<->file rules (pointer + converse) — they, not a sha256, digest or crash, refuse it
+#  13. a field REMOVED from a stored row (stdout; backend, which no per-field rule reads): STALE by the entry's
+#      content digest; a mutant without the digest REUSES the backend edit
+#  14. an unreferenced file added to files{} (hashed, digest re-sealed) is STALE; a mutant without the converse
+#      rule REUSES it
+#  15. MUST-RED: an origin-only edit, not re-sealed, is STALE (the seal covers the whole entry)
+#  16. a driver answering one item TWICE: the cell is RED, the entry is never stored, the next run is a MISS
+#  17. an artifact field (stdout) as a bare string, its file dropped, digest re-sealed: STALE by the field-shape rule
+#  NOT covered, by design: a coherent re-seal by a cache writer (see the lib's THREAT MODEL)
+#
+# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
+set -uo pipefail
+
+ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
+PROG=check_crux_ref_cache
+for t in python3 flock; do
+  command -v "$t" >/dev/null 2>&1 || { printf '%s: ENV - %s is missing\n' "$PROG" "$t" >&2; exit 2; }
+done
+for f in scripts/crux_inference_dogfood.sh scripts/lib/crux_ref_cache.py scripts/lib/crux_inference_judge.py \
+         scripts/crux_inference_prompts.v2.json scripts/lib/crux_prompt_certify.py; do
+  [ -f "$ROOT/$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
+done
+
+TMP=$(mktemp -d) || exit 2
+_rm_tmp() {
+  local w
+  [ -n "${KEEP_TMP:-}" ] && { echo "kept $TMP"; return; }
+  # the dogfood's own work dirs, kept by --keep-work so row 2 can read the manifest
+  for w in $(sed -n 's/^work kept: //p' "$TMP"/*.log 2>/dev/null); do
+    [ -n "$w" ] && [ "$w" != / ] || continue
+    case "${w:-}" in
+      /tmp/?*) rm -rf -- "${w:?}" || : ;;
+      *) : ;;
+    esac
+  done
+  case "${TMP:-}" in
+    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
+    *) : ;;
+  esac
+}
+trap _rm_tmp EXIT
+unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
+# run_cell's OOM-victim marking is shimmed: this table measures the cache, and an unprivileged sandbox denies choom
+mkdir -p "$TMP/shim"
+printf '#!/bin/sh\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done\n[ $# -gt 0 ] && shift\nexec "$@"\n' > "$TMP/shim/choom"
+chmod +x "$TMP/shim/choom"
+export PATH="$TMP/shim:$PATH"
+
+PASS=0
+FAIL=0
+ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
+broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }
+
+# ---- the tree: a copy of scripts/ with the two reference drivers stubbed ------------------------------------
+# mk_tree <dir>: the real scripts, stub drivers. Every stub call appends "CALL <engine> <verb> <prompt>" to $STUB_CALLS.
+mk_tree() {
+  local t="$1" eng
+  mkdir -p "$t"
+  cp -r "$ROOT/scripts" "$t/scripts"
+  for eng in hf llamafile; do
+    cat > "$t/scripts/crux_engine_$eng.sh" <<SH
+#!/usr/bin/env bash
+# stub $eng driver for check_crux_ref_cache.sh: STUB_PROBE_$eng overrides the probe line; STUB_REFUSE_$eng=1
+# makes every gen row a refusal.
+eng=$eng
+SH
+    cat >> "$t/scripts/crux_engine_$eng.sh" <<'SH'
+case "${1:-}" in
+  probe) pv="STUB_PROBE_$eng"; printf '%s\n' "${!pv:-$eng=1.0.0 transformers=5.0.0 torch=2.0.0 device=stub-$HOSTNAME}"; exit 0 ;;
+  gen) shift ;;
+  *) exit 2 ;;
+esac
+while [ $# -gt 0 ]; do
+  case "$1" in
+    --model-sha256) sha="$2"; shift 2 ;;
+    --verb) verb="$2"; shift 2 ;;
+    --prompt-id) pid="$2"; shift 2 ;;
+    --thinking) think="$2"; shift 2 ;;
+    --backend) backend="$2"; shift 2 ;;
+    --host) host="$2"; shift 2 ;;
+    --interface) shift 2 ;;
+    *) if [ "${2:-}" != "" ] && [ "${2#--}" = "$2" ]; then shift 2; else shift; fi ;;
+  esac
+done
+echo "CALL $eng $verb $pid" >> "$STUB_CALLS"
+d="$CRUX_WORK/${sha:0:12}/$verb"; mkdir -p "$d"
+out="$d/$eng-$pid-$think.json"
+printf '{"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\\n", "reported": {"thinking_requested": "%s", "device": "stub"}}\n' "$think" > "$out"
+: > "$d/$eng-$pid-$think.err"
+rv="STUB_REFUSE_$eng"
+python3 - "$CRUX_MANIFEST" "$eng" "$sha" "$host" "$verb" "$think" "$backend" "$pid" "$out" "$d/$eng-$pid-$think.err" "${!rv:-}" <<'PY'
+import json, os, sys
+m, eng, sha, host, verb, think, backend, pid, out, err, refuse = sys.argv[1:12]
+row = {"kind": "gen", "engine": eng, "model_sha256": sha, "host": host, "verb": verb, "thinking": think,
+       "backend": backend, "prompt_id": pid, "rc": 0, "stdout": out, "stderr": err,
+       "refused": "stub refusal" if refuse else None}
+if eng == "hf":
+    row["source"] = {"repo": "stub/src", "revision": "0" * 40, "dtype": "bfloat16"}
+open(m, "a").write(json.dumps(row) + "\n")
+if os.environ.get("STUB_DUP_" + eng):
+    open(m, "a").write(json.dumps(row) + "\n")
+PY
+SH
+    chmod +x "$t/scripts/crux_engine_$eng.sh"
+  done
+}
+
+BIN="$TMP/bin"; mkdir -p "$BIN"
+cat > "$BIN/apr" <<'SH'
+#!/usr/bin/env bash
+case "${1:-}" in
+  --version) echo "apr 0.0.0 (stub)" ;;
+  run)
+    case " $* " in *" --help "*) echo "  --thinking <MODE>"; exit 0 ;; esac
+    printf '{"text": "<answer>4</answer>", "tokens_generated": 1, "tok_per_sec": 1.0, "finish_reason": "stop", "backend": {"requested": "gpu", "ran": "gpu", "fell_back": false}}\n'
+    printf '[DEBUG] formatted_prompt="q"\n' >&2 ;;
+  *) echo "stub apr: unhandled '$*'" >&2; exit 1 ;;
+esac
+SH
+chmod +x "$BIN/apr"
+MODEL="$TMP/model.gguf"; printf 'GGUF-stub-model' > "$MODEL"
+MSHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
+printf 'sources:\n  %s:\n    hf: {repo: stub/src, revision: "%s", dtype: bfloat16}\n' "$MSHA" "$(printf '0%.0s' $(seq 40))" > "$TMP/hf-sources.yaml"
+# A fixture certification over the REAL v2 prompt bytes that admits the positive control for the stub model, OFF.
+# The judge checks it through the certifier's own `check`, so a drift in the prompt set is an ENV refusal here.
+python3 - "$ROOT/scripts/crux_inference_prompts.v2.json" "$MSHA" "$TMP/cert.json" <<'PY' || exit 2
+import hashlib, json, sys
+prompts, sha, out = sys.argv[1:4]
+json.dump({"schema": "crux-prompt-certification/v1", "prompts": "scripts/crux_inference_prompts.v2.json",
+           "prompts_sha256": hashlib.sha256(open(prompts, "rb").read()).hexdigest(), "admitted": {},
+           "admitted_by_sha": {sha: ["ctl-2plus2"]}, "admitted_by_sha_thinking": {sha: {"off": ["ctl-2plus2"]}}},
+          open(out, "w"))
+PY
+
+# run_row <name> <tree> <cache> [env...]: one real dogfood run; receipt at $TMP/<name>.out/stub-gpu.json
+run_row() {
+  local name="$1" tree="$2" cache="$3"; shift 3
+  : > "$TMP/$name.calls"; : > "$TMP/$name.lock"
+  ( cd "$tree" && env STUB_CALLS="$TMP/$name.calls" CRUX_GPU_LOCK="$TMP/$name.lock" GPUQ_BIN=/nonexistent/gpu-q \
+      CRUX_HF_SOURCES="$TMP/hf-sources.yaml" DOGFOOD_ALLOW_UNPINNED=1 APR="$BIN/apr" "$@" \
+      timeout 300 bash scripts/crux_inference_dogfood.sh 0.0.0 --model "$MODEL" --engines apr,hf,llamafile \
+        --verbs run --prompts scripts/crux_inference_prompts.v2.json --certification "${RUN_CERT:-$TMP/cert.json}" \
+        --only-prompts ctl-2plus2 --thinking-modes off \
+        --host stub --out "$TMP/$name.out" --timeout 60 --reference-cache "$cache" --keep-work ) > "$TMP/$name.log" 2>&1
+  echo "$?" > "$TMP/$name.rc"
+}
+
+# summary <name>: "<engine calls> | <cell verdicts> | <reference rows from cache>/<reference rows>"
+summary() {
+  python3 - "$TMP/$1.out/stub-gpu.json" "$TMP/$1.calls" "$TMP/$1.out" <<'PY'
+import glob, json, os, sys
+rec, calls = sys.argv[1], sys.argv[2]
+n = sum(1 for _ in open(calls))
+try:
+    d = json.load(open(rec))
+except (OSError, ValueError):
+    print("%d | NO-RECEIPT | -" % n); sys.exit()
+cells = sorted("%s/%s/%s=%s" % (c["key"]["verb"], c["key"]["thinking"], c["key"]["prompt_id"], c["verdict"])
+               for c in d.get("cells", []))
+print("%d | %s | %s" % (n, " ".join(cells) or "no-cells", d.get("declined") or d.get("declined_because") or ""))
+PY
+}
+
+printf '%s: the CRUX reference cache (#4036)\n' "$PROG"
+T="$TMP/tree"; mk_tree "$T"
+CACHE="$TMP/cache"
+
+run_row cold "$T" "$CACHE"
+cold=$(summary cold)
+stored=$(find "$CACHE" -name entry.json 2>/dev/null | wc -l)
+case "$cold" in
+  "2 | "*"=GREEN"*) [ "$stored" -eq 2 ] && ok "cold: both reference engines ran, 2 entries stored, receipt GREEN ($cold)" \
+                    || broke "cold: $stored entries stored, want 2 ($cold)" ;;
+  *) broke "cold: $cold (rc $(cat "$TMP/cold.rc")); log $TMP/cold.log"; tail -5 "$TMP/cold.log" | sed 's/^/        /' ;;
+esac
+
+run_row warm "$T" "$CACHE"
+warm=$(summary warm)
+# every reference row of the warm manifest: from the cache, host rewritten to this run, origin named, artifact local
+prov=$(python3 - "$(sed -n 's/^work kept: //p' "$TMP/warm.log" | tail -1)" <<'PY'
+import json, os, sys
+bad, n = [], 0
+for l in open(os.path.join(sys.argv[1], "manifest.jsonl")):
+    r = json.loads(l)
+    if r.get("kind") != "gen" or r.get("engine") == "apr":
+        continue
+    n += 1
+    rc = r.get("reference_cache") or {}
+    if not (rc.get("digest") and (rc.get("origin") or {}).get("host") == "stub" and r.get("host") == "stub"
+            and r["stdout"].startswith(sys.argv[1] + "/refcache/") and os.path.isfile(r["stdout"])):
+        bad.append(r["engine"])
+print("%d rows, not provenanced: %s" % (n, ",".join(bad) or "none"))
+PY
+)
+if [ "${warm%% |*}" = 0 ] && [ "${warm#* | }" = "${cold#* | }" ] && [ "$prov" = "2 rows, not provenanced: none" ] \
+   && grep -q '"result": "hit"' "$TMP/warm.out/stub-gpu.json"; then
+  ok "warm: 0 reference engine calls, receipt identical cell for cell, $prov, meta records the hit ($warm)"
+else
+  broke "warm: $warm vs cold $cold; $prov (rc $(cat "$TMP/warm.rc")); log $TMP/warm.log"; grep -i 'reference cache' "$TMP/warm.log" | sed 's/^/        /'
+fi
+
+# Row 3: MUST-RED. Edit one stored artifact (the truth itself).
+art=$(find "$CACHE" -path '*/files/*' -name '*stdout*' | head -1)
+cp -r "$CACHE" "$TMP/cache-tampered"
+art_t="$TMP/cache-tampered/${art#"$CACHE"/}"
+# The edit keeps the answer RIGHT: only the stale check can make this RED, not the judge seeing a wrong truth.
+printf '{"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\\n", "reported": {"device": "edited"}}\n' > "$art_t"
+run_row stale "$T" "$TMP/cache-tampered"
+stale=$(summary stale)
+case "$stale" in
+  "0 | "*"=RED"*) if grep -q 'is STALE' "$TMP/stale.out/stub-gpu.json" && ! printf '%s' "$stale" | grep -q '=GREEN'; then
+                    ok "MUST-RED: an edited entry (answer still right) is refused STALE in the receipt, 0 engine calls, every cell RED ($stale)"
+                  else broke "stale: RED but not every cell, or the refusal does not say STALE ($stale)"; fi ;;
+  *) broke "stale: $stale — a tampered reference was not RED"; grep -i 'reference cache' "$TMP/stale.log" | sed 's/^/        /' ;;
+esac
+
+# Row 4: a new oracle version is a MISS, recomputed.
+# The device on the probe line is NOT in the key (a host moved), the package version IS (the oracle moved).
+run_row newhost "$T" "$CACHE" "STUB_PROBE_hf=hf=1.0.0 transformers=5.0.0 torch=2.0.0 device=another-host"
+nh=$(summary newhost)
+run_row newprobe "$T" "$CACHE" "STUB_PROBE_hf=hf=1.0.0 transformers=5.1.0 torch=2.0.0 device=stub"
+np=$(summary newprobe)
+case "$nh|$np" in
+  "0 | "*"=GREEN"*"|2 | "*"=GREEN"*) ok "oracle key: another device is a HIT ($nh); a new transformers is a MISS, recomputed GREEN ($np)" ;;
+  *) broke "oracle key: device change $nh; version change $np" ;;
+esac
+
+# Row 5: a refused reference row is never stored.
+run_row refuse "$T" "$TMP/cache-refuse" STUB_REFUSE_llamafile=1
+lf=$(find "$TMP/cache-refuse" -name entry.json -exec grep -l '"engine": "llamafile"' {} + 2>/dev/null | wc -l)
+run_row refuse2 "$T" "$TMP/cache-refuse"
+r2=$(summary refuse2)
+if [ "$lf" -eq 0 ] && [ "${r2%% |*}" = 2 ]; then
+  ok "a refused llamafile row was not stored; the next run was a MISS and ran both engines ($r2)"
+else
+  broke "refused row: $lf llamafile entries stored, next run $r2"
+fi
+
+# Row 6: MUTANT — the stale check disabled. Row 3's receipt must stop being RED, or the table is blind.
+MT="$TMP/mutant"; mk_tree "$MT"
+python3 - "$MT/scripts/lib/crux_ref_cache.py" <<'PY'
+import sys
+p = sys.argv[1]
+s = open(p).read()
+a = '    """None when the entry at edir is intact for key; otherwise why it may not be reused."""\n'
+assert s.count(a) == 1, "mutation anchor moved: update this check with the lib"
+open(p, "w").write(s.replace(a, a + "    return None\n"))
+PY
+# The lib is in its own harness digest, so the mutant keys differently from the real lib: it must fill ITS OWN cache
+# and have ITS entry tampered. Otherwise it misses, recomputes and is GREEN for the wrong reason (measured: the row
+# passed vacuously that way once the lib joined the digest). Required: 0 engine calls AND non-RED — the edited truth
+# really reused.
+run_row mutant-cold "$MT" "$TMP/cache-mutant"
+mart=$(find "$TMP/cache-mutant" -path '*/files/*' -name '*stdout*' | head -1)
+if [ -n "$mart" ]; then
+  printf '{"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\\n", "reported": {"device": "edited"}}\n' > "$mart"
+fi
+run_row mutant "$MT" "$TMP/cache-mutant"
+mu=$(summary mutant)
+case "$mu" in
+  "0 | "*=RED*) broke "MUTANT (stale check disabled) still RED: the table cannot see the stale check ($mu)" ;;
+  "0 | "*=GREEN*) ok "MUTANT (stale check disabled) REUSES the edited entry, 0 engine calls, GREEN: row 3 is what catches it ($mu)" ;;
+  *) broke "MUTANT row is vacuous: the mutant never hit its own cache ($mu; artifact ${mart:-none})" ;;
+esac
+
+# Row 7: llama.cpp is never cached. Its llama-server renders apr's raw-prompt serve routes; a hit that switched it
+# off refused 14 of apr's own serve cells on gx10 (2026-09-23). The lib refuses it by name.
+why=$(cd "$T" && python3 scripts/lib/crux_ref_cache.py lookup --cache "$TMP/c7" --work "$TMP" --manifest /dev/null \
+  --model-sha "$MSHA" --thinking off --backend gpu --host stub --engines llama.cpp --verbs run \
+  --oracle "llama.cpp=b1" --temperature 0 --seed 42 --context 4096 --max-tokens 1024 --root "$T" \
+  --out-rows "$TMP/c7.rows" 2>&1)
+rc=$?
+case "$rc:$why" in
+  1:*"reference renderer"*) ok "llama.cpp cannot be cached: refused by name (it renders apr's serve routes)" ;;
+  *) broke "llama.cpp cacheable? rc $rc: $why" ;;
+esac
+
+# Row 8: the run/chat key carries the cap the engine is GIVEN — the mode's global cap. A prompt set whose largest
+# budget moved (another prompt edited) changes it; the old truth was generated under a different cap (quorum round 1).
+T2="$TMP/tree-cap"; mk_tree "$T2"
+python3 - "$T2/scripts/crux_inference_prompts.v2.json" "$MSHA" "$TMP/cert-cap.json" <<'PY' || exit 2
+import hashlib, json, sys
+p, sha, out = sys.argv[1:4]
+d = json.load(open(p))
+other = next(x for x in d["prompts"] if x["id"] != "ctl-2plus2" and isinstance(x.get("max_tokens"), dict))
+other["max_tokens"]["off"] = max(int(x["max_tokens"]["off"]) for x in d["prompts"] if isinstance(x.get("max_tokens"), dict)) * 2
+json.dump(d, open(p, "w"), indent=2)
+json.dump({"schema": "crux-prompt-certification/v1", "prompts": "scripts/crux_inference_prompts.v2.json",
+           "prompts_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "admitted": {},
+           "admitted_by_sha": {sha: ["ctl-2plus2"]}, "admitted_by_sha_thinking": {sha: {"off": ["ctl-2plus2"]}}},
+          open(out, "w"))
+PY
+RUN_CERT="$TMP/cert-cap.json" run_row newcap "$T2" "$CACHE"
+nc=$(summary newcap)
+case "$nc" in
+  "2 | "*"=GREEN"*) ok "a new global cap is a MISS for run/chat, recomputed GREEN ($nc)" ;;
+  *) broke "new global cap: $nc (a hit would reuse a truth generated under another cap)" ;;
+esac
+
+# Row 9: MUST-RED — a row's artifact POINTER rewritten to a traversal, key and files{} untouched and the content
+# digest RE-SEALED, so only the pointer rule can refuse it (quorum round 3, lane 2 measured the traversal passing as
+# intact and reading outside the cache). It is STALE: 0 engine calls, every cell RED.
+cp -r "$CACHE" "$TMP/cache-ptr"
+python3 - "$TMP/cache-ptr" <<'PY' || exit 2
+import glob, hashlib, json, sys
+# every entry: the cache holds entries of several keys by now (rows 4 and 8), and only this run's must be hit
+for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
+    e = json.load(open(p))
+    e["rows"][0]["stdout"] = "refcache:" + "../" * 8 + "etc/hostname"
+    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
+                                                    sort_keys=True).encode()).hexdigest()  # RE-SEALED: only the pointer rule may refuse it
+    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
+PY
+run_row ptr "$T" "$TMP/cache-ptr"
+pt=$(summary ptr)
+case "$pt" in
+  "0 | "*"=RED"*) if grep -q "is not one of the entry's own hashed files" "$TMP/ptr.out/stub-gpu.json" && ! printf '%s' "$pt" | grep -q '=GREEN'; then
+                    ok "MUST-RED: a traversal pointer (key and files{} intact) is STALE, 0 engine calls, every cell RED ($pt)"
+                  else broke "pointer: RED but not for the pointer ($pt)"; fi ;;
+  *) broke "pointer: $pt — an edited pointer was reused or recomputed, not refused" ;;
+esac
+
+# Row 10: the lookup's exit contract — hit 0, miss 10, stale 11, and anything else is NOT one of them: a keying
+# refusal exits 1 and a crash 2 (a crash once shared the miss code, so a corrupted cache was silently recomputed).
+lk() { python3 "$T/scripts/lib/crux_ref_cache.py" lookup --cache "$1" --work "$2" --manifest /dev/null \
+  --model-sha "$MSHA" --thinking off --backend gpu --host stub --engines hf --verbs "$3" --oracle "hf=transformers=5.0.0" \
+  --temperature 0 --seed 42 --context 4096 --max-tokens 1024 --root "$T" --out-rows "$TMP/lk.rows" > /dev/null 2>&1; echo $?; }
+W10="$TMP/w10"; mkdir -p "$W10"; printf 'run ctl-2plus2\n' > "$W10/pids-all.txt"; printf '{}' > "$W10/prompt-ctl-2plus2.json"
+printf '["ctl-2plus2", ["serve run"]]\n' > "$W10/serve-prompts.jsonl"
+r_miss=$(lk "$TMP/c10" "$W10" run); r_key=$(lk "$TMP/c10" "$W10" run,serve); r_crash=$(lk "$TMP/c10" "$TMP/no-such-work" run)
+[ "$r_miss:$r_key:$r_crash" = "10:1:2" ] && ok "exit contract: miss 10, keying refusal 1 (no cap file), crash 2 — none of the latter two is a miss" \
+  || broke "exit contract: miss $r_miss (want 10), keying $r_key (want 1), crash $r_crash (want 2)"
+
+# Row 11: MUST-DECLINE — a lookup that CRASHES inside the real dogfood declines the run; it is never read as a miss
+# and recomputed GREEN.
+T3="$TMP/tree-crash"; mk_tree "$T3"
+python3 - "$T3/scripts/lib/crux_ref_cache.py" <<'PY'
+import sys
+p = sys.argv[1]
+s = open(p).read()
+a = "def cmd_lookup(a):\n"
+assert s.count(a) == 1, "crash anchor moved: update this check with the lib"
+open(p, "w").write(s.replace(a, a + "    raise RuntimeError('planted crash')\n"))
+PY
+run_row crash "$T3" "$TMP/cache-crash"
+cr=$(summary crash)
+if [ "$(cat "$TMP/crash.rc")" = 2 ] && grep -q 'neither hit, miss nor stale' "$TMP/crash.log" && [ "${cr#* | }" = "NO-RECEIPT | -" ]; then
+  ok "a crashing lookup DECLINES the run (rc 2, no receipt), never a silent recompute"
+else
+  broke "crashing lookup: rc $(cat "$TMP/crash.rc"), $cr"
+fi
+
+# Row 12: the row<->file correspondence is what refuses an edited pointer. `refcache:./<same name>` resolves to the
+# SAME hashed file, with the digest re-sealed, so no sha256, digest or crash can catch it; the pointer rule (a row's
+# pointer is a plain name files{} holds) and its converse (files{} holds only what a row points at) each do. Real lib:
+# STALE. MUTANT with both disabled (its own cache, since the lib is in the digest): REUSED, 0 calls, GREEN.
+dot_ptr() { # dot_ptr <cache>: every entry's first row points at ./<its own file>
+  python3 - "$1" <<'PY' || exit 2
+import glob, hashlib, json, sys
+for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
+    e = json.load(open(p))
+    v = e["rows"][0]["stdout"]
+    e["rows"][0]["stdout"] = "refcache:./" + v[len("refcache:"):]
+    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
+                                                    sort_keys=True).encode()).hexdigest()  # RE-SEALED: only the pointer rule may refuse it
+    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
+PY
+}
+cp -r "$CACHE" "$TMP/cache-dot"; dot_ptr "$TMP/cache-dot"
+run_row dot "$T" "$TMP/cache-dot"
+dt=$(summary dot)
+MP="$TMP/mutant-ptr"; mk_tree "$MP"
+python3 - "$MP/scripts/lib/crux_ref_cache.py" <<'PY'
+import sys
+p = sys.argv[1]
+s = open(p).read()
+a = "            if not rel or rel != os.path.basename(rel) or rel in (\".\", \"..\") or rel not in (ent.get(\"files\") or {}):"
+assert s.count(a) == 1, "pointer-rule anchor moved: update this check with the lib"
+s = s.replace(a, "            if False:")
+b = "    if extra:"
+assert s.count(b) == 1, "converse-rule anchor moved: update this check with the lib"
+open(p, "w").write(s.replace(b, "    if False:"))
+PY
+run_row mptr-cold "$MP" "$TMP/cache-mptr"; dot_ptr "$TMP/cache-mptr"
+run_row mptr "$MP" "$TMP/cache-mptr"
+mp=$(summary mptr)
+case "$dt|$mp" in
+  "0 | "*"=RED"*"|0 | "*"=GREEN"*) ok "an equivalent-but-edited pointer: STALE on the real lib ($dt); REUSED with the row<->file rules disabled ($mp)" ;;
+  *) broke "pointer rule: real lib $dt, mutant $mp" ;;
+esac
+
+# Row 13: a FIELD REMOVED from a stored row. Deleting `stdout` passed every per-field rule (quorum round 4, lane 1,
+# measured); the entry's content digest now refuses any field added, removed or changed. MUST-RED on `stdout`;
+# and `backend` — a field no per-field rule reads, whose removal orphans no file — is STALE on the real lib but
+# REUSED by a mutant without the content digest, so the digest (not another rule, not the judge) refuses it.
+del_field() { # del_field <cache> <field>: every entry's first row loses <field>; nothing else is touched
+  python3 - "$1" "$2" <<'PY' || exit 2
+import glob, json, sys
+for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
+    e = json.load(open(p))
+    e["rows"][0].pop(sys.argv[2], None)
+    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
+PY
+}
+cp -r "$CACHE" "$TMP/cache-del"; del_field "$TMP/cache-del" stdout
+run_row del "$T" "$TMP/cache-del"
+dl=$(summary del)
+cp -r "$CACHE" "$TMP/cache-dels"; del_field "$TMP/cache-dels" backend
+run_row dels "$T" "$TMP/cache-dels"
+ds=$(summary dels)
+MC="$TMP/mutant-content"; mk_tree "$MC"
+python3 - "$MC/scripts/lib/crux_ref_cache.py" <<'PY'
+import sys
+p = sys.argv[1]
+s = open(p).read()
+a = "    if not ent.get(\"content_sha256\") or content_sha256(ent) != ent[\"content_sha256\"]:"
+assert s.count(a) == 1, "content-digest anchor moved: update this check with the lib"
+open(p, "w").write(s.replace(a, "    if False:"))
+PY
+run_row mcon-cold "$MC" "$TMP/cache-mcon"; del_field "$TMP/cache-mcon" backend
+run_row mcon "$MC" "$TMP/cache-mcon"
+mc=$(summary mcon)
+case "$dl|$ds|$mc" in
+  "0 | "*"=RED"*"|0 | "*"=RED"*"|0 | "*"=GREEN"*)
+    if grep -q 'its content changed since it was stored' "$TMP/del.out/stub-gpu.json" "$TMP/dels.out/stub-gpu.json"; then
+      ok "a REMOVED field is STALE (stdout: $dl; backend: $ds); a mutant without the content digest REUSES the backend edit ($mc)"
+    else broke "field removal: RED but not for the content digest"; fi ;;
+  *) broke "field removal: stdout $dl, stderr $ds, mutant $mc" ;;
+esac
+
+# Row 14: an EXTRA file smuggled into files{} (hashed correctly, digest re-sealed) that no row points at is STALE:
+# an entry carries only the files its rows need (quorum round 5, lane 2, measured the addition passing).
+cp -r "$CACHE" "$TMP/cache-extra"
+python3 - "$TMP/cache-extra" <<'PY' || exit 2
+import glob, hashlib, json, os, sys
+for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
+    e = json.load(open(p))
+    extra = os.path.join(os.path.dirname(p), "files", "smuggled.txt")
+    open(extra, "w").write("not a row's artifact\n")
+    e["files"]["smuggled.txt"] = hashlib.sha256(open(extra, "rb").read()).hexdigest()
+    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
+                                                    sort_keys=True).encode()).hexdigest()
+    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
+PY
+run_row extra "$T" "$TMP/cache-extra"
+ex=$(summary extra)
+# ...and a MUTANT without the converse rule alone REUSES it: no other rule sees an extra, correctly hashed file.
+MX="$TMP/mutant-extra"; mk_tree "$MX"
+python3 - "$MX/scripts/lib/crux_ref_cache.py" <<'PY'
+import sys
+p = sys.argv[1]
+s = open(p).read()
+b = "    if extra:"
+assert s.count(b) == 1, "converse-rule anchor moved: update this check with the lib"
+open(p, "w").write(s.replace(b, "    if False:"))
+PY
+run_row mext-cold "$MX" "$TMP/cache-mext"
+python3 - "$TMP/cache-mext" <<'PY' || exit 2
+import glob, hashlib, json, os, sys
+for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
+    e = json.load(open(p))
+    extra = os.path.join(os.path.dirname(p), "files", "smuggled.txt")
+    open(extra, "w").write("not a row's artifact\n")
+    e["files"]["smuggled.txt"] = hashlib.sha256(open(extra, "rb").read()).hexdigest()
+    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
+                                                    sort_keys=True).encode()).hexdigest()
+    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
+PY
+run_row mext "$MX" "$TMP/cache-mext"
+mx=$(summary mext)
+case "$ex|$mx" in
+  "0 | "*"=RED"*"|0 | "*"=GREEN"*) grep -q 'which no row points at' "$TMP/extra.out/stub-gpu.json" \
+                  && ok "an unreferenced file in files{} (hashed, re-sealed) is STALE ($ex); a mutant without the converse rule REUSES it ($mx)" \
+                  || broke "extra file: RED but not for the unreferenced file ($ex)" ;;
+  *) broke "extra file: real lib $ex, mutant $mx" ;;
+esac
+
+# Row 15: MUST-RED — an ORIGIN-only edit, NOT re-sealed (the provenance a hit reports). The seal once covered only
+# {key, rows, files}, so this passed as a HIT and served a false origin (quorum round 6, lane 2, measured).
+cp -r "$CACHE" "$TMP/cache-origin"
+python3 - "$TMP/cache-origin" <<'PY' || exit 2
+import glob, json, sys
+for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
+    e = json.load(open(p))
+    e["origin"]["host"] = "not-the-host-that-measured-it"
+    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
+PY
+run_row origin "$T" "$TMP/cache-origin"
+og=$(summary origin)
+case "$og" in
+  "0 | "*"=RED"*) grep -q 'its content changed since it was stored' "$TMP/origin.out/stub-gpu.json" \
+                  && ok "an origin-only edit (not re-sealed) is STALE ($og)" \
+                  || broke "origin edit: RED but not for the seal ($og)" ;;
+  *) broke "origin edit: $og — a false provenance was served" ;;
+esac
+
+# Row 16: a driver that appends TWO rows for one item: the cell refuses it (RED, never a pick) and the cache never
+# stores it — an entry is exactly one row — so the next run is a MISS that measures again (quorum round 7, lane 2).
+run_row dup "$T" "$TMP/cache-dup" STUB_DUP_hf=1
+dp=$(summary dup)
+hfe=$(find "$TMP/cache-dup" -name entry.json -exec grep -l '"engine": "hf"' {} + 2>/dev/null | wc -l)
+run_row dup2 "$T" "$TMP/cache-dup"
+d2=$(summary dup2)
+case "$dp|$hfe|$d2" in
+  *"=RED"*"|0|2 | "*"=GREEN"*) ok "an item answered TWICE: its cell RED ($dp), never stored (0 hf entries), next run a MISS measured again ($d2)" ;;
+  *) broke "duplicate rows: first $dp, hf entries $hfe, next $d2" ;;
+esac
+
+# Row 17: an artifact FIELD in the wrong shape — `stdout` a bare string (not a refcache: pointer), its file dropped
+# from files{} and the digest RE-SEALED, so neither the digest nor the pointer rules see it. The judge would open
+# the string as a relative path. The field-shape rule refuses it (quorum round 8, lane 2, measured as a HIT).
+cp -r "$CACHE" "$TMP/cache-shape"
+python3 - "$TMP/cache-shape" <<'PY' || exit 2
+import glob, hashlib, json, sys
+for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
+    e = json.load(open(p))
+    old = e["rows"][0]["stdout"]
+    e["rows"][0]["stdout"] = "FABRICATED-NOT-A-POINTER"
+    e["files"].pop(old[len("refcache:"):], None)
+    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
+                                                    sort_keys=True).encode()).hexdigest()
+    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
+PY
+run_row shape "$T" "$TMP/cache-shape"
+sh=$(summary shape)
+case "$sh" in
+  "0 | "*"=RED"*) grep -q 'an artifact field is null or a refcache: pointer' "$TMP/shape.out/stub-gpu.json" \
+                  && ok "an artifact field in the wrong shape (bare string, re-sealed) is STALE ($sh)" \
+                  || broke "field shape: RED but not for the shape ($sh)" ;;
+  *) broke "field shape: $sh — a bare-string artifact field was served" ;;
+esac
+
+printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
+[ "$FAIL" -eq 0 ] || exit 1
+exit 0
diff --git a/scripts/crux_inference_dogfood.sh b/scripts/crux_inference_dogfood.sh
index d8ea90c9e..b6422f46e 100755
--- a/scripts/crux_inference_dogfood.sh
+++ b/scripts/crux_inference_dogfood.sh
@@ -96,6 +96,12 @@ THINK_MODES="off,on"
 ONLY_PROMPTS=""
 GREEDY_PIDS=""
 GREEDY_MAXTOK=256
+# --reference-cache <dir> (#4036): the pure comparators' rows (hf, vllm, llamafile) are host-independent
+# truth, cached per (model, mode, engine, verb, prompt) under a key bound to the oracle version, the sampling and
+# the harness files (scripts/lib/crux_ref_cache.py). A mode whose every reference entry is cached runs ONLY apr and
+# injects the cached rows; any absent entry runs the whole mode and stores its clean rows; a STALE entry refuses
+# every reference row of the mode (RED), never reused. Off unless given.
+REF_CACHE=""
 MODELS=()
 while [ $# -gt 0 ]; do
   case "$1" in
@@ -115,6 +121,7 @@ while [ $# -gt 0 ]; do
     --greedy) GREEDY=1; shift ;;
     --greedy-prompts) [ $# -ge 2 ] || decline "--greedy-prompts needs a value"; GREEDY_PIDS="${2//,/ }"; shift 2 ;;
     --greedy-max-tokens) [ $# -ge 2 ] || decline "--greedy-max-tokens needs a value"; GREEDY_MAXTOK="$2"; shift 2 ;;
+    --reference-cache) [ $# -ge 2 ] || decline "--reference-cache needs a value"; REF_CACHE="$2"; shift 2 ;;
     -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
     -*) decline "unknown argument '$1'" ;;
     *) [ -z "$VERSION" ] || decline "one version, got '$VERSION' and '$1'"; VERSION="$1"; shift ;;
@@ -211,7 +218,7 @@ PLUGIN_ENGINES=(hf llamafile vllm)
 # evidence/crux/hf-sources.yaml sidecar: hf by design, vllm because 0.30.0 cannot read a
 # local GGUF (#3952). A model with no declared source is a refused row for each of them.
 source_engine() { case "$1" in hf|vllm) return 0 ;; *) return 1 ;; esac; }
-declare -A EXT_OK EXT_WHY EXT_SCRIPT EXT_PROBE
+declare -A EXT_OK EXT_WHY EXT_SCRIPT EXT_PROBE EXT_BATCH
 for eng in "${PLUGIN_ENGINES[@]}"; do
   want "$eng" || continue
   EXT_OK[$eng]=0
@@ -231,6 +238,13 @@ for eng in "${PLUGIN_ENGINES[@]}"; do
   if [ "$prc" -eq 0 ] && [ -n "$probe_out" ]; then
     EXT_OK[$eng]=1
     EXT_PROBE[$eng]=$(printf '%s\n' "$probe_out" | head -1)
+    # #4036 lever 2: a source-weight driver with `gen-batch` serves every item of a mode from ONE engine load.
+    # Measured on the 0.69.1 lambda sweep (aprender-36, #4033): vLLM was 64% of CRUX time, ~55 s of cold start per
+    # prompt x verb. CRUX_NO_BATCH=1 keeps the one-load-per-cell path (the A/B baseline).
+    if [ -z "${CRUX_NO_BATCH:-}" ] && source_engine "$eng" \
+       && "${ext_run[@]}" "${EXT_SCRIPT[$eng]}" gen-batch --help > /dev/null 2>&1; then
+      EXT_BATCH[$eng]=1
+    fi
   else
     EXT_WHY[$eng]="probe exit $prc: $(head -c 300 /tmp/crux-probe-$$-$eng.err 2>/dev/null | tr '\n' ' ')"
   fi
@@ -343,6 +357,7 @@ PY
 ) || decline "prompt set $PROMPTS: $(tr '\n' ' ' < "$WORK/prompts.err" 2>/dev/null | cut -c1-300)"
 pids_for() { printf '%s\n' "$PIDS_ALL" | sed -n "s/^$1 //p" | tr '\n' ' '; }
 PIDS=$(pids_for run)
+printf '%s\n' "$PIDS_ALL" > "$WORK/pids-all.txt"
 # The global cap for the run/chat cells: v1's `max_tokens`, else the largest per-prompt `off` budget.
 MAXTOK=$(python3 -c 'import json,sys
 d = json.load(open(sys.argv[1]))
@@ -455,6 +470,92 @@ print(n)
 PY
 }
 
+crux_batched() { # crux_batched <engine>: its rows this mode come from plugin_batch_cells, not the per-prompt cells
+  [ "${EXT_BATCH[$1]:-0}" = 1 ] && want "$1" && [ "${EXT_OK[$1]:-0}" = 1 ] \
+    && ! { source_engine "$1" && [ -n "$HF_MODEL_WHY" ]; }
+}
+
+# plugin_batch_cells <engine>: this mode's items for one batched engine, as TWO cells, each one engine load under
+# the GPU lock: run+chat in-process, then serve run+serve stream+code through the engine's own server (the driver
+# refuses a batch that mixes them). The items are exactly the per-prompt cells' (same messages, same max_tokens:
+# the global cap for run/chat, the prompt's own for serve/code), so a row is the row a per-prompt cell writes, plus
+# its `batch` id. An item the driver returned no row for is refused by name, never absent.
+plugin_batch_cells() {
+  local eng="$1" d="$WORK/$SHA12/batch-$eng" group items cell ext_run v pid mt
+  mkdir -p "$d" || return 1
+  case "${EXT_SCRIPT[$eng]}" in *.py) ext_run=(python3) ;; *) ext_run=(bash) ;; esac
+  for group in inproc serve; do
+    items="$d/$group.items.jsonl"; : > "$items"
+    python3 - "$WORK" "$VERBS" "$group" "$THINK" "$MAXTOK" "$items" <<'PY' || return 1
+import json, os, sys
+work, verbs, group, think, maxtok, out = sys.argv[1:7]
+verbs = verbs.split(",")
+items = []
+if group == "inproc":
+    for line in open(os.path.join(work, "pids-all.txt")):
+        parts = line.split()
+        if len(parts) == 2 and parts[0] in verbs:
+            items.append((parts[0], parts[1], maxtok))
+else:
+    own = lambda pid: open(os.path.join(work, "maxtok-%s.txt" % pid)).read().strip()
+    sp = os.path.join(work, "serve-prompts.jsonl")
+    if "serve" in verbs and os.path.exists(sp):
+        pairs = [json.loads(l) for l in open(sp) if l.strip()]
+        for want in ("serve run", "serve stream"):
+            items += [(want, pid, own(pid)) for pid, vs in pairs if want in vs]
+    cp = os.path.join(work, "code-prompts.txt")
+    if "code" in verbs and os.path.exists(cp):
+        items += [("code", pid.strip(), own(pid.strip())) for pid in open(cp) if pid.strip()]
+with open(out, "w") as f:
+    for verb, pid, mt in items:
+        f.write(json.dumps({"prompt_id": pid, "verb": verb, "messages": os.path.join(work, "messages-%s.json" % pid),
+                            "thinking": think, "max_tokens": int(mt)}) + "\n")
+PY
+    [ -s "$items" ] || continue
+    cell="$d/cell-$group.sh"
+    printf '#!/usr/bin/env bash\n# one CRUX batch cell (#4036): every %s item of this mode through ONE %s load\n' "$group" "$eng" > "$cell"
+    local -A before=()
+    while IFS=$'\t' read -r v pid; do before["$v|$pid"]=$(VERB_KEY="$v" rows_for "$eng" "$pid"); done \
+      < <(python3 -c 'import json,sys; [print("%s\t%s" % (i["verb"], i["prompt_id"])) for i in map(json.loads, open(sys.argv[1]))]' "$items")
+    cell_add "$cell" "$d/$group.driver" "${ext_run[@]}" "${EXT_SCRIPT[$eng]}" gen-batch --batch "$items" \
+      --model "$M" --model-sha256 "$SHA" --backend "$BACKEND" --host "$HOST" \
+      --seed "$SEED" --temperature "$TEMP" --context "$CTX" "${HF_SRC[@]}"
+    printf 'exit 0\n' >> "$cell"
+    run_cell "$cell"
+    local now
+    while IFS=$'\t' read -r v pid; do
+      now=$(VERB_KEY="$v" rows_for "$eng" "$pid")
+      if [ -n "$CELL_WHY" ]; then VERB_KEY="$v" emit_gen "$eng" "$pid" "" "" "" "$CELL_WHY"
+      elif [ "$now" -le "${before["$v|$pid"]}" ]; then
+        VERB_KEY="$v" emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen-batch exited $(cat "$d/$group.driver.rc" 2>/dev/null || echo '?') without a row for this item: $(tail -c 200 "$d/$group.driver.err" 2>/dev/null | tr '\n' ' ')"
+      elif [ "$now" -gt $(( ${before["$v|$pid"]} + 1 )) ]; then
+        # appended LAST, so it is the row the judge keeps for this engine: the item is refused, never a pick
+        VERB_KEY="$v" emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen-batch: a driver that returned $(( now - ${before["$v|$pid"]} )) rows for ONE item is refused: the judge would keep whichever came last, and the reference cache would replay both (quorum round 7, lane 2)"
+      fi
+    done < <(python3 -c 'import json,sys; [print("%s\t%s" % (i["verb"], i["prompt_id"])) for i in map(json.loads, open(sys.argv[1]))]' "$items")
+  done
+}
+
+ref_cache_args() { # ref_cache_args: REF_ENGINES + REF_ARGS for the current model and mode (#4036)
+  # An engine is cached only when it would really run here AND says which build it is: an unversioned oracle
+  # cannot be keyed, so it simply runs. llama.cpp always runs: its llama-server is apr's reference renderer on the
+  # serve routes (a hit that switched it off refused 14 apr serve cells on gx10). ollama: not in the sweep.
+  REF_ENGINES=(); REF_ARGS=()
+  local e v src=""
+  for e in "${PLUGIN_ENGINES[@]}"; do
+    want "$e" && [ "${EXT_OK[$e]:-0}" = 1 ] || continue
+    source_engine "$e" && [ -n "$HF_MODEL_WHY" ] && continue
+    v=${EXT_PROBE[$e]:-}
+    [ -n "$v" ] || continue
+    REF_ENGINES+=("$e"); REF_ARGS+=(--oracle "$e=$v")
+  done
+  [ "${#HF_SRC[@]}" -gt 0 ] && src=$(python3 -c 'import json,sys; print(json.dumps({"repo": sys.argv[1], "revision": sys.argv[2], "dtype": sys.argv[3]}))' "$hf_repo" "$hf_rev" "$hf_dtype")
+  REF_ARGS+=(--cache "$REF_CACHE" --work "$WORK" --manifest "$MANIFEST" --model-sha "$SHA" --thinking "$THINK"
+    --backend "$BACKEND" --host "$HOST" --engines "$(IFS=,; printf '%s' "${REF_ENGINES[*]}")" --verbs "$VERBS"
+    --source "$src" --temperature "$TEMP" --seed "$SEED" --context "$CTX" --max-tokens "$MAXTOK" --root "$ROOT"
+    --harness-git-sha "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)")
+}
+
 free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
 
 # llama.cpp's OWN rendering and tokenization of each prompt: the reference apr's
@@ -644,6 +745,32 @@ PY
     cp -- "$f" "$base.txt"
   done
 
+  REF_SKIP=0
+  REF_ENGINES=()
+  if [ -n "$REF_CACHE" ]; then
+    ref_cache_args
+    if [ "${#REF_ENGINES[@]}" -gt 0 ]; then
+      python3 scripts/lib/crux_ref_cache.py lookup "${REF_ARGS[@]}" --out-rows "$WORK/$SHA12/refcache-rows.jsonl" \
+        > "$WORK/$SHA12/refcache-lookup.log" 2>&1
+      rrc=$?
+      sed 's/^/  /' "$WORK/$SHA12/refcache-lookup.log"
+      case "$rrc" in
+        0) REF_SKIP=1; rres=hit ;;
+        11) REF_SKIP=1; rres=stale ;;
+        10) rres=miss ;;
+        # a keying refusal or a crash is none of the three: never a silent recompute of a corrupted cache
+        *) decline "reference cache lookup failed (rc $rrc, neither hit, miss nor stale): $(tail -1 "$WORK/$SHA12/refcache-lookup.log")" ;;
+      esac
+      printf '%s\t%s\t%s\t%s\n' "$SHA" "$THINK" "$rres" "$(IFS=,; printf '%s' "${REF_ENGINES[*]}")" >> "$WORK/refcache.tsv"
+    fi
+  fi
+  # A hit (or a stale entry) runs this mode with the cached engines REMOVED: they neither run nor emit, and the
+  # cached (or refused) rows are appended after the verb loop. Restored before the next mode and the greedy rows.
+  if [ "$REF_SKIP" = 1 ]; then
+    REF_SAVED_ENGINES=$ENGINES
+    ENGINES=$(for e in ${ENGINES//,/ }; do case " ${REF_ENGINES[*]} " in *" $e "*) ;; *) printf '%s\n' "$e" ;; esac; done | paste -sd,)
+  fi
+
   for VERB in ${VERBS//,/ }; do
   VERB_KEY=$VERB
   if [ "$VERB" = serve ]; then
@@ -707,6 +834,7 @@ PY
     for eng in "${PLUGIN_ENGINES[@]}"; do
       want "$eng" && [ "${EXT_OK[$eng]}" = 1 ] || continue
       source_engine "$eng" && [ -n "$HF_MODEL_WHY" ] && continue
+      crux_batched "$eng" && continue
       case "${EXT_SCRIPT[$eng]}" in *.py) ext_run=(python3) ;; *) ext_run=(bash) ;; esac
       # llamafile's `--cli` ignores the thinking switch and does not bound output
       # with -n (infra-3c, measured on 0.10.6); its server honours both.
@@ -733,16 +861,29 @@ PY
     elif want ollama; then emit_gen ollama "$pid" "" "" "" "${OL_REFUSED:-$OLLAMA_WHY}"; fi
     for eng in "${PLUGIN_ENGINES[@]}"; do
       want "$eng" || continue
+      crux_batched "$eng" && continue
       if [ "${EXT_OK[$eng]}" != 1 ]; then emit_gen "$eng" "$pid" "" "" "" "${EXT_WHY[$eng]}"; continue; fi
       if source_engine "$eng" && [ -n "$HF_MODEL_WHY" ]; then emit_gen "$eng" "$pid" "" "" "" "$HF_MODEL_WHY"; continue; fi
       before=${before_run[$eng]}
       if [ -n "$CELL_WHY" ]; then emit_gen "$eng" "$pid" "" "" "" "$CELL_WHY"
       elif [ "$(rows_for "$eng" "$pid")" -le "$before" ]; then
         emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen exited $(cat "$d/$eng-$pid.driver.rc" 2>/dev/null || echo '?') without appending a row: $(tail -c 200 "$d/$eng-$pid.driver.err" 2>/dev/null | tr '\n' ' ')"
+      elif [ "$(rows_for "$eng" "$pid")" -gt $(( before + 1 )) ]; then
+        emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen: a driver that returned $(( $(rows_for "$eng" "$pid") - before )) rows for ONE item is refused: the judge would keep whichever came last, and the reference cache would replay both (quorum round 7, lane 2)"
       fi
     done
   done
   done
+  for eng in "${PLUGIN_ENGINES[@]}"; do
+    crux_batched "$eng" || continue
+    plugin_batch_cells "$eng" || decline "the $eng batch cell could not be built for $NAME"
+  done
+  if [ "$REF_SKIP" = 1 ]; then
+    cat "$WORK/$SHA12/refcache-rows.jsonl" >> "$MANIFEST"
+    ENGINES=$REF_SAVED_ENGINES
+  elif [ "${#REF_ENGINES[@]}" -gt 0 ]; then
+    python3 scripts/lib/crux_ref_cache.py store "${REF_ARGS[@]}" 2>&1 | sed 's/^/  /'
+  fi
   done  # thinking modes
   SHA12=$SHA12_MODEL
 
@@ -798,6 +939,18 @@ meta = {
 }
 json.dump(meta, open(out, "w"), indent=2)
 PY
+if [ -s "$WORK/refcache.tsv" ]; then
+  python3 - "$WORK/meta.json" "$WORK/refcache.tsv" "$REF_CACHE" <<'PY' || decline "could not record the reference cache in meta.json"
+import json, sys
+meta = json.load(open(sys.argv[1]))
+modes = [dict(zip(("model_sha256", "thinking", "result", "engines"), l.rstrip("\n").split("\t")))
+         for l in open(sys.argv[2]) if l.strip()]
+meta["reference_cache"] = {"dir": sys.argv[3], "modes": modes,
+                           "note": "hit = the reference rows were measured by an earlier run (origin in each row's "
+                                   "reference_cache), not on this host; stale = every reference row refused (#4036)"}
+json.dump(meta, open(sys.argv[1], "w"), indent=2)
+PY
+fi
 # $OUT_DIR reaches here from `--out` (default evidence/crux/$VERSION). A receipt dir may
 # legitimately be absolute -- operators point it at /mnt -- so absoluteness is allowed and
 # only a `..` segment, which walks out of wherever the caller meant, is refused.
diff --git a/scripts/crux_sweep_shards.sh b/scripts/crux_sweep_shards.sh
index 9f8ed9fc8..97b979e21 100755
--- a/scripts/crux_sweep_shards.sh
+++ b/scripts/crux_sweep_shards.sh
@@ -4,7 +4,7 @@
 #
 #   bash scripts/crux_sweep_shards.sh <version> --host <id> --apr <binary> --out <dir>
 #        [--backend gpu|cpu] [--scope controls|admitted] [--models-dir <dir>]... [--certification <receipt>]
-#        [--greedy-model <gguf>]... [--greedy-only] [--dry-run]
+#        [--greedy-model <gguf>]... [--greedy-only] [--reference-cache <dir>] [--dry-run]
 #
 # WHY PER (MODEL, MODE). The certification admits prompts per quant sha AND thinking mode
 # (admitted_by_sha_thinking); a prompt run outside its admission is RED at the judge by design. So each shard is
@@ -21,6 +21,11 @@
 # budgets are impractical on a CPU lane, and F9's CPU reference needs greedy rows alone. The receipt then has no
 # judged cell, so the judge DECLINES it (exit 2, "no cell was measured") while its greedy[] carries the rows: a
 # greedy-only receipt is F9 evidence, never a CRUX verdict, and the plan file says so.
+# --reference-cache <dir> (#4036): the certified shards reuse the source-weight engines' rows (hf, vLLM) that an
+# earlier run stored there, keyed by model sha + mode + prompt + oracle version + harness; a host then runs apr and
+# llama.cpp only (llama.cpp always runs: it is apr's reference renderer on the serve routes). A miss runs the mode in full and stores; a stale entry is RED. Greedy shards never read it: their
+# rows decode apr's ids through a llama-server on THIS host. Share the dir between hosts by copying it
+# (rsync -a lambda:<dir>/ gx10:<dir>/); each entry is written by one atomic rename, so two writers cannot tear one.
 set -uo pipefail
 cd "$(dirname "$0")/.." || exit 2
 PROG=crux_sweep_shards
@@ -28,7 +33,7 @@ die() { printf '%s: %s\n' "$PROG" "$1" >&2; exit 2; }
 
 VERSION=""; HOST=""; APR_BIN=""; OUT=""; BACKEND=gpu; SCOPE=controls; DRY=0; GREEDY_ONLY=0; MERGE_ONLY=0
 CERT="evidence/crux/0.69.1/prompt-certification.json"; PROMPTS="scripts/crux_inference_prompts.v2.json"
-MODEL_DIRS=(); GREEDY_MODELS=()
+MODEL_DIRS=(); GREEDY_MODELS=(); REF_CACHE_ARGS=()
 while [ $# -gt 0 ]; do
   case "$1" in
     --host) HOST="$2"; shift 2 ;;
@@ -39,6 +44,7 @@ while [ $# -gt 0 ]; do
     --models-dir) MODEL_DIRS+=("$2"); shift 2 ;;
     --certification) CERT="$2"; shift 2 ;;
     --greedy-model) GREEDY_MODELS+=("$2"); shift 2 ;;
+    --reference-cache) REF_CACHE_ARGS=(--reference-cache "$2"); shift 2 ;;
     --greedy-only) GREEDY_ONLY=1; shift ;;
     --merge-only) MERGE_ONLY=1; shift ;;
     --dry-run) DRY=1; shift ;;
@@ -111,7 +117,7 @@ run_shard() { # run_shard <name> <dogfood args...> — one dogfood run; echoes i
 [ "$MERGE_ONLY" = 1 ] || : > "$OUT/shards.tsv"
 while [ "$MERGE_ONLY" = 0 ] && IFS=$'\t' read -r kind sha mode path ids; do
   [ "$kind" = RUN ] && [ "$GREEDY_ONLY" = 0 ] || continue
-  run_shard "${sha:0:12}-$mode" --model "$path" --engines apr,llama.cpp,vllm,hf --verbs run,chat,serve,code \
+  run_shard "${sha:0:12}-$mode" --model "$path" --engines apr,llama.cpp,vllm,hf --verbs run,chat,serve,code "${REF_CACHE_ARGS[@]}" \
     --thinking-modes "$mode" --only-prompts "$ids"
 done < "$PLAN"
 CTL=$(python3 -c 'import json,sys; print(next(p["id"] for p in json.load(open(sys.argv[1]))["prompts"] if p.get("control")))' "$PROMPTS")
diff --git a/scripts/lib/crux_cells_serve_code.sh b/scripts/lib/crux_cells_serve_code.sh
index 0810d73ec..e06f95962 100644
--- a/scripts/lib/crux_cells_serve_code.sh
+++ b/scripts/lib/crux_cells_serve_code.sh
@@ -41,6 +41,9 @@
 crux_plugin_engines() { # the plugin engine list: the driver's, else the pre-#3952 pair
   if declare -p PLUGIN_ENGINES > /dev/null 2>&1; then printf '%s\n' "${PLUGIN_ENGINES[@]}"; else printf 'hf\nllamafile\n'; fi
 }
+crux_lib_batched() { # the driver runs this engine's items as a batch cell (#4036); a standalone caller never does
+  declare -F crux_batched > /dev/null 2>&1 && crux_batched "$1"
+}
 crux_is_source_engine() { # engines that run the SOURCE weights and need an HF source declared
   if declare -F source_engine > /dev/null 2>&1; then source_engine "$1"; else [ "$1" = hf ]; fi
 }
@@ -52,6 +55,7 @@ crux_plugin_lines() {
   for eng in $(crux_plugin_engines); do
     want "$eng" && [ "${EXT_OK[$eng]:-0}" = 1 ] || continue
     crux_is_source_engine "$eng" && [ -n "$HF_MODEL_WHY" ] && continue
+    crux_lib_batched "$eng" && continue
     case "${EXT_SCRIPT[$eng]}" in *.py) ext_run=(python3) ;; *) ext_run=(bash) ;; esac
     ext_extra=()
     [ "$eng" = llamafile ] && ext_extra=(--interface server)
@@ -76,12 +80,15 @@ crux_plugin_rows() {
   for pid in "$@"; do
     for eng in $(crux_plugin_engines); do
       want "$eng" || continue
+      crux_lib_batched "$eng" && continue
       if [ "${EXT_OK[$eng]:-0}" != 1 ]; then emit_gen "$eng" "$pid" "" "" "" "${EXT_WHY[$eng]:-engine unavailable}"; continue; fi
       if crux_is_source_engine "$eng" && [ -n "$HF_MODEL_WHY" ]; then emit_gen "$eng" "$pid" "" "" "" "$HF_MODEL_WHY"; continue; fi
       b="${before_ref[$eng-$pid]:-0}"
       if [ -n "$CELL_WHY" ]; then emit_gen "$eng" "$pid" "" "" "" "$CELL_WHY"
       elif [ "$(rows_for "$eng" "$pid")" -le "$b" ]; then
         emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen exited $(cat "$d/$eng-$pid.driver.rc" 2> /dev/null || echo '?') without appending a row: $(tail -c 200 "$d/$eng-$pid.driver.err" 2> /dev/null | tr '\n' ' ')"
+      elif [ "$(rows_for "$eng" "$pid")" -gt $((b + 1)) ]; then
+        emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen: a driver that returned $(( $(rows_for "$eng" "$pid") - b )) rows for ONE item is refused: the judge would keep whichever came last, and the reference cache would replay both (quorum round 7, lane 2)"
       fi
     done
   done
diff --git a/scripts/lib/crux_ref_cache.py b/scripts/lib/crux_ref_cache.py
new file mode 100644
index 000000000..da6ecbf95
--- /dev/null
+++ b/scripts/lib/crux_ref_cache.py
@@ -0,0 +1,414 @@
+"""crux_ref_cache.py: the CRUX reference cache (#4036, part of #4033 lever c).
+
+A reference answer (hf, vLLM, llamafile) to a prompt is a property of the ORACLE and the QUESTION, not of the
+host that asked it: the same model file, prompt, sampling and engine build give the same truth on lambda and on gx10.
+So a host that already measured the references can hand them to the next host, and that host runs only the apr legs.
+Measured on the 0.69.1 lambda sweep (aprender-36, #4033): the reference engines are 84% of CRUX wall time.
+
+llama.cpp is NOT cached, and asking to cache it is refused: its llama-server is also apr's REFERENCE RENDERER
+on the serve verb (the raw-prompt routes, POST /generate and the like, get their prompt rendered by it). A
+cache hit that switched llama.cpp off refused 14 of apr's own serve cells on gx10 (2026-09-23), RED rather
+than a false GREEN, and no saving. Only pure comparators are cached: engines no apr cell depends on.
+
+KEY. One entry per (model sha256, thinking mode, backend, engine, row verb, prompt id), bound to everything that could
+change the answer. The key holds:
+  - the prompt object's sha256 (messages, max_tokens, verbs);
+  - the mode's max_tokens for that prompt, and the protocol's temperature, seed and context;
+  - the oracle's version: a plugin's probe reduced to its package versions;
+  - the HF source (repo, revision, dtype) for the source-weight engines;
+  - a digest of every harness file that PRODUCES a reference row: the dogfood, its cell libs and the engine's driver.
+The host is NOT in the key; that is the point. The backend IS: a CPU-lane reference does not vouch for a GPU lane.
+
+THREAT MODEL. The integrity checks catch a cache that is CORRUPTED, EDITED IN PART or LEFT FROM ANOTHER HARNESS:
+a byte of an artifact, any field of an entry (added, removed or changed), a pointer, an unreferenced file. They do
+NOT stop a deliberate forger who can write the cache dir and re-seal every hash — nothing stored beside the entry
+can, and that writer can equally edit the dogfood, the judge or the receipt. Forgery resistance would need a key
+held outside the cache; it is out of scope for #4036 (quorum round 5, lane 2), and stated here rather than claimed.
+
+THREE OUTCOMES, never a fourth:
+  hit    every expected reference entry is present and intact: the rows are injected, and the engines do not run.
+  miss   any entry is absent: the mode runs every engine as before, and its clean rows are stored afterwards.
+         Absence is the only thing a changed oracle or harness can produce, since it changes the digest.
+  stale  an entry is present at its digest but no longer matches it: an edited key, a tampered row, an artifact
+         whose sha256 moved, or a refused row that should never have been stored. It is RED and NEVER reused. Every
+         reference row of that mode is injected REFUSED and names the entry, so the judge has no oracle and the
+         cell goes RED. Recovery is deleting the entry by hand, after reading why it went stale.
+
+Only rows that answered (rc 0, not refused) are stored. A refusal is not truth, so it is recomputed every run.
+
+  crux_ref_cache.py lookup  --cache D --work W --manifest M <key args> --out-rows F   exit 0 hit · 10 miss · 11 stale
+Every other exit (1 a keying refusal, 2 a crash) is NONE of the three, and the dogfood declines the run on it: a
+crash that shared the miss code would turn a corrupted cache into a silent recompute (quorum round 3, lane 2).
+  crux_ref_cache.py store   --cache D --work W --manifest M <key args>                exit 0 · prints "stored N"
+Key args: --model-sha --thinking --backend --host --engines e1,e2 --verbs v1,v2 --oracle eng=version (repeat) --source JSON
+          --temperature --seed --context --max-tokens <the mode's global cap> --root <repo root>
+"""
+import argparse
+import datetime
+import hashlib
+import json
+import os
+import shutil
+import sys
+import tempfile
+
+SCHEMA = "crux-ref-cache/v1"
+HIT, MISS, STALE, CRASH = 0, 10, 11, 2
+CACHED_ENGINES = ("hf", "vllm", "llamafile")
+NOT_CACHEABLE = {"llama.cpp": "its llama-server is apr's reference renderer on the serve routes, so it must run "
+                              "wherever apr runs (#4036, measured on gx10)"}
+SOURCE_ENGINES = ("hf", "vllm")
+ARTIFACT_FIELDS = ("stdout", "stderr")  # the row fields the judge opens as files
+
+# The harness files that PRODUCE a reference row. The judge's files (crux_inference_judge.py, crux_oracles.py, the
+# certifier, the smoke scope) are not here: they read rows and never change one.
+COMMON_FILES = (
+    "scripts/lib/crux_ref_cache.py",  # its injection and provenance shape the rows the judge reads
+    "scripts/crux_inference_dogfood.sh",
+    "scripts/lib/crux_cells_serve_code.sh",
+    "scripts/lib/crux_cell_teardown.sh",
+    "scripts/lib/crux_openai_client.py",
+    "scripts/lib/crux_proc.py",
+    "scripts/lib/crux_pty_chat.py",
+    "scripts/lib/crux_serve_routes.py",
+    "scripts/lib/crux_sse.py",
+    "scripts/llama_pin.toml",
+    "scripts/llama_bin.sh",
+)
+ENGINE_FILES = {
+    "hf": ("scripts/crux_engine_hf.sh", "scripts/lib/crux_hf_verify.py", "scripts/crux_hf/engine.py",
+           "scripts/crux_hf/pyproject.toml", "scripts/crux_hf/uv.lock"),
+    "vllm": ("scripts/crux_engine_vllm.sh", "scripts/lib/crux_hf_verify.py", "scripts/crux_vllm/engine.py",
+             "scripts/crux_vllm/pyproject.toml", "scripts/crux_vllm/uv.lock"),
+    "llamafile": ("scripts/crux_engine_llamafile.sh",),
+}
+# A probe line carries the device, the capability and the lock path of THIS host. Only the package versions say
+# which oracle answered, so only they enter the key.
+PROBE_KEYS = ("vllm", "transformers", "torch", "tokenizers", "jinja2", "llamafile")
+
+
+def sha256_file(path):
+    h = hashlib.sha256()
+    with open(path, "rb") as f:
+        for block in iter(lambda: f.read(1 << 20), b""):
+            h.update(block)
+    return h.hexdigest()
+
+
+def oracle_id(engine, version):
+    if engine in ("hf", "vllm"):
+        toks = [t for t in version.split() if "=" in t and t.split("=", 1)[0] in PROBE_KEYS]
+        return " ".join(sorted(toks)) or version
+    return version
+
+
+def harness_digest(root, engine):
+    h = hashlib.sha256()
+    for rel in COMMON_FILES + ENGINE_FILES.get(engine, ()):
+        p = os.path.join(root, rel)
+        h.update(rel.encode() + b"\0")
+        h.update((sha256_file(p) if os.path.isfile(p) else "absent").encode() + b"\n")
+    return h.hexdigest()
+
+
+def expected(work, verbs):
+    """The (row verb, prompt id) pairs this run asks every engine for, from the dogfood's own prompt files, limited
+    to the verbs the run was given (`serve` makes the `serve run` and `serve stream` rows)."""
+    return [(v, pid) for v, pid in _all_pairs(work) if (v.split()[0] if v.startswith("serve") else v) in verbs]
+
+
+def _all_pairs(work):
+    pairs = []
+    with open(os.path.join(work, "pids-all.txt")) as f:
+        for line in f:
+            parts = line.split()
+            if len(parts) == 2:
+                pairs.append((parts[0], parts[1]))
+    sp = os.path.join(work, "serve-prompts.jsonl")
+    if os.path.exists(sp):
+        for line in open(sp):
+            if line.strip():
+                pid, verbs = json.loads(line)
+                pairs.extend((v, pid) for v in verbs)
+    cp = os.path.join(work, "code-prompts.txt")
+    if os.path.exists(cp):
+        pairs.extend(("code", pid.strip()) for pid in open(cp) if pid.strip())
+    return pairs
+
+
+def entry_key(a, engine, verb, pid, oracles, harness):
+    # The cap the engine is GIVEN (quorum round 1, PMAT-4036): run and chat cells get the mode's global cap (the
+    # largest budget in the whole prompt set, so another prompt's edit moves it), serve and code the prompt's own.
+    # A cap that cannot be read is a refusal, never a key with max_tokens None.
+    if verb in ("run", "chat"):
+        maxtok = str(a.max_tokens)
+    else:
+        maxtok_f = os.path.join(a.work, "maxtok-%s-%s.txt" % (pid, a.thinking))
+        if not os.path.isfile(maxtok_f):
+            sys.exit("crux_ref_cache: %s has no per-mode cap file %s: an entry without its max_tokens cannot be keyed"
+                     % (pid, maxtok_f))
+        maxtok = open(maxtok_f).read().strip()
+    return {
+        "schema": SCHEMA,
+        "model_sha256": a.model_sha,
+        "thinking": a.thinking,
+        "backend": a.backend,
+        "engine": engine,
+        "verb": verb,
+        "prompt_id": pid,
+        "prompt_sha256": sha256_file(os.path.join(a.work, "prompt-%s.json" % pid)),
+        "sampling": {"temperature": a.temperature, "seed": a.seed, "context": a.context,
+                     "max_tokens": maxtok},
+        "oracle": oracle_id(engine, oracles[engine]),
+        "source": json.loads(a.source or "null") if engine in SOURCE_ENGINES else None,
+        "harness_sha256": harness[engine],
+    }
+
+
+def digest(key):
+    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
+
+
+def content_sha256(ent):
+    """The WHOLE entry — every field but this seal itself: key, rows, files, origin — as one digest, recorded at store
+    and checked at lookup. A rule per field only sees the fields it names: deleting a row's `stdout` passed every one
+    of them (quorum round 4, lane 1), and the origin was left outside a {key, rows, files} seal (quorum round 6,
+    lane 2, measured). Any field added, removed or changed anywhere in the entry moves this digest."""
+    return hashlib.sha256(json.dumps({k: v for k, v in ent.items() if k != "content_sha256"},
+                                     sort_keys=True).encode()).hexdigest()
+
+
+def entry_dir(cache, d):
+    return os.path.join(cache, d[:2], d)
+
+
+def plan(a):
+    oracles = dict(o.split("=", 1) for o in a.oracle)
+    engines = [e for e in a.engines.split(",") if e]
+    for e in engines:
+        if e in NOT_CACHEABLE:
+            sys.exit("crux_ref_cache: %s cannot be cached: %s" % (e, NOT_CACHEABLE[e]))
+        if e not in CACHED_ENGINES:
+            sys.exit("crux_ref_cache: %s is not a reference engine (cached: %s)" % (e, ", ".join(CACHED_ENGINES)))
+        if e not in oracles:
+            sys.exit("crux_ref_cache: no --oracle version for %s: an unversioned oracle cannot be keyed" % e)
+    harness = {e: harness_digest(a.root, e) for e in engines}
+    out = []
+    for e in engines:
+        for verb, pid in expected(a.work, a.verbs.split(",")):
+            k = entry_key(a, e, verb, pid, oracles, harness)
+            out.append((k, digest(k)))
+    return out
+
+
+def row_paths(row):
+    """(container, field) for every string field of the row that names a file: top level and one level down."""
+    for k, v in row.items():
+        if isinstance(v, str) and v.startswith("/"):
+            yield row, k
+        elif isinstance(v, dict):
+            for k2, v2 in v.items():
+                if isinstance(v2, str) and v2.startswith("/"):
+                    yield v, k2
+
+
+def why_stale(key, d, edir):
+    """None when the entry at edir is intact for key; otherwise why it may not be reused."""
+    try:
+        ent = json.load(open(os.path.join(edir, "entry.json")))
+    except (OSError, ValueError) as e:
+        return "entry.json unreadable (%s)" % e.__class__.__name__
+    if not ent.get("content_sha256") or content_sha256(ent) != ent["content_sha256"]:
+        return "its content changed since it was stored (a key, row or files field was edited, added or removed)"
+    if ent.get("key") != key:
+        diff = sorted(f for f in set(key) | set(ent.get("key") or {}) if key.get(f) != (ent.get("key") or {}).get(f))
+        return "its key no longer matches its digest (fields: %s)" % ", ".join(diff)
+    if digest(ent["key"]) != d:
+        return "its key hashes to a different digest"
+    rows = ent.get("rows") or []
+    if len(rows) != 1:
+        return "it holds %d rows; an entry is exactly one row for its (engine, verb, prompt)" % len(rows)
+    for r in rows:
+        ident = (r.get("model_sha256"), r.get("thinking"), r.get("engine"), r.get("verb"), r.get("prompt_id"))
+        if ident != (key["model_sha256"], key["thinking"], key["engine"], key["verb"], key["prompt_id"]):
+            return "a row's identity %r contradicts the key" % (ident,)
+        if r.get("refused") or r.get("rc") != 0:
+            return "it holds a refused or failed row, which is never truth"
+        # An artifact field is null or a cache pointer — never another shape: the judge opens it as a path, so a bare
+        # string would be read relative to wherever it runs (quorum round 8, lane 2, measured under a re-seal).
+        for f in ARTIFACT_FIELDS:
+            v = r.get(f)
+            if v is not None and not (isinstance(v, str) and v.startswith("refcache:")):
+                return "a row's %s is %r: an artifact field is null or a refcache: pointer" % (f, v)
+        for box, f in row_paths_rel(r):
+            rel = box[f][len("refcache:"):]
+            # A pointer is a plain name the entry's own files{} holds — never a path: `../` would read or write outside
+            # the cache and the work dir, and a name files{} does not hold is a row no sha256 vouches for.
+            if not rel or rel != os.path.basename(rel) or rel in (".", "..") or rel not in (ent.get("files") or {}):
+                return "a row's artifact pointer %r is not one of the entry's own hashed files" % box[f]
+        for box, f in row_paths(r):
+            return "a row still names an absolute path %r: its artifact was never stored" % box[f]
+    used = {box[f][len("refcache:"):] for r in rows for box, f in row_paths_rel(r)}
+    extra = sorted(set(ent.get("files") or {}) - used)
+    if extra:
+        return "files{} holds %s, which no row points at: an entry carries only the files its rows need" % extra[0]
+    for rel, want in (ent.get("files") or {}).items():
+        if not rel or rel != os.path.basename(rel) or rel in (".", ".."):
+            return "files{} names %r, which is not a plain file name" % rel
+        p = os.path.join(edir, "files", rel)
+        if not os.path.isfile(p):
+            return "artifact %s is missing" % rel
+        if sha256_file(p) != want:
+            return "artifact %s changed since it was stored (sha256)" % rel
+    return None
+
+
+def cmd_lookup(a):
+    entries = plan(a)
+    hits, misses, stale = [], [], []
+    for k, d in entries:
+        edir = entry_dir(a.cache, d)
+        if not os.path.exists(edir):
+            misses.append((k, d))
+            continue
+        w = why_stale(k, d, edir)
+        (stale if w else hits).append((k, d, w))
+    if not entries:
+        print("reference cache: nothing to look up")
+        return MISS
+    if not stale and misses:
+        print("reference cache MISS: %d of %d entries absent (e.g. %s %s %s): every engine runs"
+              % (len(misses), len(entries), misses[0][0]["engine"], misses[0][0]["verb"], misses[0][0]["prompt_id"]))
+        return MISS
+    out = open(a.out_rows, "w")
+    for k, d, _ in hits:
+        if stale:
+            continue
+        edir = entry_dir(a.cache, d)
+        ent = json.load(open(os.path.join(edir, "entry.json")))
+        dest = os.path.join(a.work, "refcache", d)
+        os.makedirs(dest, exist_ok=True)
+        for r in ent["rows"]:
+            for box, f in list(row_paths_rel(r)):
+                rel = box[f][len("refcache:"):]
+                shutil.copyfile(os.path.join(edir, "files", rel), os.path.join(dest, rel))
+                box[f] = os.path.join(dest, rel)
+            r["host"] = a.host
+            r["reference_cache"] = {"digest": d, "origin": ent.get("origin")}
+            out.write(json.dumps(r) + "\n")
+    if stale:
+        k0, d0, w0 = stale[0]
+        why = ("reference cache entry %s (%s %s %s) is STALE: %s; a stale entry is RED and never reused (#4036), "
+               "delete it by hand after reading why" % (d0[:16], k0["engine"], k0["verb"], k0["prompt_id"], w0))
+        for k, d in entries:
+            out.write(json.dumps({"kind": "gen", "engine": k["engine"], "prompt_id": k["prompt_id"], "rc": None,
+                                  "stdout": None, "stderr": None, "refused": why, "model_sha256": k["model_sha256"],
+                                  "host": a.host, "verb": k["verb"], "thinking": k["thinking"], "backend": a.backend,
+                                  "reference_cache": {"digest": d, "stale": d in {s[1] for s in stale}}}) + "\n")
+        out.close()
+        print("reference cache STALE: %d of %d entries (%s); every reference row of this mode is refused"
+              % (len(stale), len(entries), w0))
+        return STALE
+    out.close()
+    print("reference cache HIT: %d entries, %d engines skipped this mode" % (len(hits), len({k["engine"] for k, _, _ in hits})))
+    return HIT
+
+
+def row_paths_rel(row):
+    for k, v in row.items():
+        if isinstance(v, str) and v.startswith("refcache:"):
+            yield row, k
+        elif isinstance(v, dict):
+            for k2, v2 in v.items():
+                if isinstance(v2, str) and v2.startswith("refcache:"):
+                    yield v, k2
+
+
+def cmd_store(a):
+    rows = [json.loads(line) for line in open(a.manifest) if line.strip()]
+    stored = skipped = 0
+    for k, d in plan(a):
+        edir = entry_dir(a.cache, d)
+        if os.path.exists(edir):
+            continue
+        mine = [r for r in rows if r.get("kind") == "gen" and r.get("model_sha256") == k["model_sha256"]
+                and r.get("thinking") == k["thinking"] and r.get("engine") == k["engine"]
+                and r.get("verb") == k["verb"] and r.get("prompt_id") == k["prompt_id"]
+                and "reference_cache" not in r]
+        # exactly ONE clean row: several rows for one (engine, verb, prompt) are ambiguous — the judge keeps whichever
+        # came last — so they are never stored as truth (quorum round 7, lane 2)
+        if len(mine) != 1 or any(r.get("refused") or r.get("rc") != 0 for r in mine):
+            skipped += 1
+            continue
+        os.makedirs(os.path.dirname(edir), exist_ok=True)
+        tmp = tempfile.mkdtemp(prefix=".tmp-", dir=os.path.dirname(edir))
+        os.makedirs(os.path.join(tmp, "files"))
+        files, ok = {}, True
+        for i, r in enumerate(mine):
+            for box, f in list(row_paths(r)):
+                p = box[f]
+                if not p.startswith(a.work.rstrip("/") + "/"):
+                    continue
+                if not os.path.isfile(p):
+                    ok = False
+                    break
+                rel = "%d-%s-%s" % (i, f, os.path.basename(p))
+                shutil.copyfile(p, os.path.join(tmp, "files", rel))
+                files[rel] = sha256_file(os.path.join(tmp, "files", rel))
+                box[f] = "refcache:" + rel
+        if not ok:
+            shutil.rmtree(tmp)
+            skipped += 1
+            continue
+        ent = {"key": k, "rows": mine, "files": files,
+               "origin": {"host": a.host, "backend": a.backend, "harness_git_sha": a.harness_git_sha or None,
+                          "stored_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}}
+        ent["content_sha256"] = content_sha256(ent)
+        json.dump(ent, open(os.path.join(tmp, "entry.json"), "w"), indent=1, sort_keys=True)
+        try:
+            os.rename(tmp, edir)
+            stored += 1
+        except OSError:
+            shutil.rmtree(tmp)  # another writer stored the same digest first
+    print("reference cache: stored %d entries, %d not stored (refused, failed or absent rows are never truth)"
+          % (stored, skipped))
+    return 0
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
+    p.add_argument("cmd", choices=("lookup", "store"))
+    p.add_argument("--cache", required=True)
+    p.add_argument("--work", required=True)
+    p.add_argument("--manifest", required=True)
+    p.add_argument("--model-sha", required=True)
+    p.add_argument("--thinking", required=True, choices=("off", "on"))
+    p.add_argument("--backend", required=True)
+    p.add_argument("--host", required=True)
+    p.add_argument("--engines", required=True)
+    p.add_argument("--verbs", required=True)
+    p.add_argument("--oracle", action="append", default=[])
+    p.add_argument("--source", default="")
+    p.add_argument("--temperature", required=True)
+    p.add_argument("--seed", required=True)
+    p.add_argument("--context", required=True)
+    p.add_argument("--max-tokens", type=int, required=True, help="the mode's global cap, which run/chat cells are given")
+    p.add_argument("--root", required=True)
+    p.add_argument("--harness-git-sha", default="")
+    p.add_argument("--out-rows")
+    a = p.parse_args()
+    if a.cmd == "lookup":
+        if not a.out_rows:
+            p.error("lookup needs --out-rows")
+        return cmd_lookup(a)
+    return cmd_store(a)
+
+
+if __name__ == "__main__":
+    try:
+        sys.exit(main())
+    except SystemExit:
+        raise
+    except Exception as e:  # noqa: BLE001 — any crash is its own outcome, never a miss
+        sys.stderr.write("crux_ref_cache: CRASH %s: %s\n" % (e.__class__.__name__, e))
+        sys.exit(CRASH)
```
