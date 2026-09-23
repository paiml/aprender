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
    - stale: refuse every reference row, which is RED and never reused. "Stale" includes a row whose artifact pointer is not a plain name that the entry's own hashed `files{}` holds (quorum round 3, lane 2).
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

**`scripts/check_crux_ref_cache.sh` — 12/12**

1. Cold stores.
2. Warm makes 0 reference calls, the receipt is identical cell for cell, and every injected row is provenanced.
3. MUST-RED: an answer-*preserving* edit to a stored entry is refused STALE, and every cell is RED.
4. A device change is a hit; a package version change is a miss, recomputed.
5. A refused row is never stored.
6. MUTANT, stale check disabled: the mutant fills its own cache, then REUSES the edited entry (0 calls, GREEN). Row 3 is what catches it.
7. llama.cpp is refused.
8. A new global cap is a miss for run/chat.
9. MUST-RED: a row's artifact pointer is rewritten to a `../` traversal while the key and `files{}` stay intact. It is STALE, with 0 engine calls, and every cell is RED.
10. The exit contract: miss 10, keying refusal 1, crash 2.
11. MUST-DECLINE: a lookup that crashes inside the real dogfood declines the run (rc 2, no receipt). It is never recomputed silently.
12. An edited pointer that resolves to the SAME hashed file is STALE on the real lib, and REUSED by a mutant with the pointer rule disabled. The rule is what refuses it.

**`scripts/check_crux_plugin_batch.sh` — 5/5**

- 1 load for 2 items.
- `CRUX_NO_BATCH` gives the same receipt and the same rows, row for row.
- A driver without the capability keeps the per-prompt path.
- MUST-RED: a dropped item is refused by name.
- MUTANT, never batch: caught by the load count.

**Unchanged on this diff:** greedy 12/12, ollama-in-lock 6/6, serve_code 66/66, judge 153/153, oracles PASS.

## Not covered

- ON-mode vLLM serve rows are refused with a 400 until #4029 lands. Refusals are never cached, so ON modes stay misses, and run in full, until then.
- The cache is shared between hosts by copying the directory. No transport is part of this change.
