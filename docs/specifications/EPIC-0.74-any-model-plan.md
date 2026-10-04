# EPIC 0.74.0 "Any Model": plan (paiml/aprender#4001)

**Status:** plan for operator review. Nothing is applied: no child issues, milestone moves, or closes.
**Ticket:** PMAT-4001 · **kind:** docs · **Ratchet:** slice 5 of 5 (close-out: ≥ 80% cleared) of DEBT-RATCHET-001 (#3997, PR #4003)

The operator's words, quoted on #4001: *".74 is unified model support style in the ticket alfredo metnioned, i.e.
quicker ability to support ANY model and we leverage llama.cpp and rachets in each release"*. Alfredo's tickets are
#3423, #3422 and #3418, and standing policy prioritizes them.

## 1. Baselines (re-measured 2026-09-23, `origin/main` @ `49fe19c28`, using #3422's own commands)

| Fact | #3422/#3418 said | Measured today |
|---|---|---|
| Shared-block adoption: calls to the attention blocks (`standard_single_head_attention \| parallel_multihead_attention \| … \| reshape_for_parallel_heads`) and FFN blocks (`ffn_block:: \| adaptive_ffn::`) in the production forward files | 0 in `forward_qwen35.rs` and `forward_qwen3_moe.rs` (09-17, `1d7dcc5e9`) | **0 in every production forward file**: `single.rs`, `forward_cached.rs`, `forward_qwen35.rs`, `forward_qwen3_moe*.rs`, `gemma_dispatch.rs`, `forward_single_profiled.rs`. The only calls are in a test (`batch_tests_tiled_single.rs`, 7) |
| `forward_qwen35.rs` non-comment lines | 939 | **1,474** (+57% in 6 days). The per-architecture cost is **growing** |
| Files matching on quant type in `aprender-serve/src` | "~30" | **42** (files matching `match … (qtype\|quant_type\|ggml_type\|dtype)`). The grep is a proxy, and R-1 replaces it with the registry's own census |
| llama.cpp's Qwen3-MoE architecture file (reference) | 179 lines (`src/models/qwen3moe.cpp` at `3173a5647`) | cited from #3418, not re-measured here |

## 2. Exit bar (from #4001, sharpened here for the quorum)

| Bar | Unit | Baseline | 0.74 threshold |
|---|---|---|---|
| **E-1** a new architecture costs ≤ N lines | non-comment lines added by the PR that adds a **previously unsupported** architecture from llama.cpp's supported set, **measured on a real addition during 0.74** | `forward_qwen35.rs` = 1,474 lines for one architecture | **N = 300** (proposed; the quorum decides), counted by `git diff --numstat` on that PR, excluding tests and the architecture's config file |
| **E-2** every certified architecture routes through shared blocks | the share of production forward files whose attention and FFN are shared-block calls | **0 of 8** | 8 of 8 (or the file is deleted) |
| **E-3** one quant dispatch | files outside the dispatch module that match on quant type | 42 (proxy) | **0**, enforced by a guard |
| **E-4** new-architecture correctness is proven against llama.cpp | a CRUX-style oracle cell: apr vs llama.cpp on the same GGUF, **official chat template**, positive control | not built | green for the E-1 architecture, with a planted-wrong-weights negative control |

## 3. Rows

| Row | Item | done_when | Baseline | First-green proof |
|---|---|---|---|---|
| **R-1** | #3418 / PP-QUANT (#3421, #3420): one quant-type registry (`ggml_type_traits`-style) and one dispatch | `scripts/check_quant_dispatch.sh` finds 0 quant-type `match` outside the registry module, with a shrink-only baseline until then | 42 files (proxy) | the guard is RED on today's tree at baseline − 1, and GREEN at the baseline. A planted new `match qtype` outside the registry is RED. **Quorum fixes:** the guard has a `--self-test`; it fails closed when it scans 0 files; and deleting a `match` without routing through the registry is RED (the registry's own census must list the quant type) |
| **R-2** | #3422 / PP-ARCH (#3424): shared attention/FFN blocks adopted by every production forward path | a census of production forward files: each one calls the shared blocks, or it is deleted | 0 of 8 | the census RED today, GREEN when done; a forward file reintroducing a private attention loop is RED. **Quorum fix:** the census is by symbol, i.e. it counts calls to the shared blocks. "Or it is deleted" counts only when the deleted file's architecture still passes its ladder rung through the shared path, so a deletion without a replacement is RED |
| **R-3** | PP-TENSOR (#3428): a tensor with no bytes is a different type from one with bytes (MoE / lazy tensors) | #3428's own acceptance | OPEN | #3428's must-RED case |
| **R-4** | #3423: the consolidation epic's own rows | #3423's rows | OPEN (milestone "Inference dispatch & architecture consolidation") | per row |
| **R-5** | **The real addition** (E-1): add one previously unsupported llama.cpp architecture during 0.74 | the PR merges within N lines, and its E-4 oracle cell is green, **with a planted-corrupt-weights negative control that must go RED** (quorum fix) | none | the addition itself. The candidate is chosen by the quorum (Q2) |
| **R-6** | The llama.cpp oracle harness (E-4) | per-architecture cell: token-level agreement over a fixed prompt set on the official template, with a positive control and a negative control | not built; llama.cpp is pinned in infra (#911) | the positive control (a known-good architecture) is green, and the negative (planted wrong weights) is RED |
| **R-7** | **Ratchet slice 5 of 5: close-out, ≥ 80% cleared** | the DEBT-RATCHET-001 slice-5 gates (all pillars at their 0.74 floors) | see #4003 | see #4003 |

## 4. Ratchet slice 5 of 5 (from #4003 §3, proposed): the ≥ 80% close-out

| Pillar | 0.74 floor |
|---|---|
| A: P₀ bp | ≥ 9,364 (= baseline + 80% of the gap to 9,500); `P_cuda` ≥ `B_cuda + 3·s_cuda` (window: decision 7) |
| B-1: E2 call sites | ≥ 435 (total ≥ 510) |
| B-2: contracts with no falsifier | 0 |
| C: ONT rows bound | 27 / 27 |
| D-1 / D-2 / D-3 | 0 / 0 / 0 (armed at zero) |

## 5. Open questions for the quorum to DECIDE

- **Q1. N for E-1.** Recommendation: **N = 300** non-comment, non-test lines. llama.cpp's reference is 179 for
  Qwen3-MoE; 300 leaves room for Rust's explicitness without allowing a new forward path.
- **Q2. Which architecture is the E-1 addition?** It must be in llama.cpp's supported set, unsupported by apr today,
  and small enough to fit on the fleet. Recommendation: pick from llama.cpp's architecture list at the pinned
  commit. **The quorum lanes propose candidates and the plan does not pre-select**, because the candidate list is a
  fact about the pinned llama.cpp that lanes can read.
- **Q3. Order.** Recommendation: R-1 (quant registry) before R-2 (shared blocks), because a shared block that still
  matches on quant type inherits #3418's duplication. R-5 last, as the measurement.

## 6. Commands

```bash
cd crates/aprender-serve/src && for f in gguf/inference/forward/*.rs; do echo "$f attn=$(grep -cE 'standard_single_head_attention|parallel_multihead_attention|parallel_batched_qk_scores|standard_softmax|online_softmax|tiled_single_head_attention|reshape_for_parallel_heads' $f) ffn=$(grep -cE 'ffn_block::|adaptive_ffn::' $f)"; done
grep -vcE '^\s*//|^\s*$' crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs     # 1474
grep -rlE 'match .*(qtype|quant_type|ggml_type|dtype)' crates/aprender-serve/src | wc -l       # 42
for i in 3423 3422 3418 3421 3420 3424 3428; do gh issue view $i -R paiml/aprender --json state,milestone; done
```


## Quorum record: decision quorum, 2026-09-23 (aprender-cb)

**Lanes (ADVISORY: single family, all gemini):** gemini-3.1-pro-high, gemini-3.8-flash-high, gemini-3.7-flash-high,
all returning PASS-with-changes. gpt-oss returned 429. 3/3 exited 3 on foreign ref motion, with every clone
byte-identical. Conversations: `9be51392`, `19bd8d9e`, `436c3c42`.

| Q | Decision (tally) | Applied as |
|---|---|---|
| Q1 | **N = 300** non-comment, non-test lines, 3/3 | E-1 |
| Q2 | **No consensus.** Lane 1 could not read llama.cpp's list and said so. Lane 2: **StarCoder2** (backups StableLM, Granite). Lane 3: **Command-R** (backups MiniCPM3, StarCoder2). **aprender-cb checked the tree:** `starcoder2`, `stablelm`, `granite` and `minicpm3` already appear as architecture strings in apr's config/format code, with **0** forward files. `cohere`/Command-R has **0** hits in `crates/aprender-serve/src`. All five exist in `~/src/llama.cpp/src/models/` (local checkout `60b06ab9a`, not compared with `scripts/llama_pin.toml`) | **Proposed: Command-R (cohere)**, because apr has no footprint for it, so the N-line count measures a whole addition. **StarCoder2 is the fallback** if Command-R does not fit the fleet at a certified quant. **This goes to the operator; the quorum did not decide it** |
| Q3 | **R-1 (quant registry) → R-2 (shared blocks) → R-5 last**, 3/3 | §5 |

**Must-fix items applied:**
- R-2's deletion escape is closed;
- R-1's guard gets a self-test, fails closed on 0 files, and a deletion without the registry is RED;
- R-5's negative control.

**Must-fix items carried to step 2 as child-issue acceptance:**
- exact `done_when` commands;
- R-3's measured behaviour baseline (not "OPEN");
- #3423 decomposed into rows with baselines and controls;
- the slice-5 baselines and commands (#4003 §7);
- aligning the llama.cpp reference commit (`3173a5647`, cited from #3418) with `scripts/llama_pin.toml`.

