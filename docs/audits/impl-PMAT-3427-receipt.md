# IMPL receipt — PMAT-3427 — PP-QUANT-001 research + 0.69 must-carry plan

Marks: `[V]` verified by the command shown · `[V-diff]` read from a fix diff · `[C]` computed from a cited artifact · `[A]` asserted · `[X]` third-party · `[U]` unverified.

## Identity
- ticket PMAT-3427 (#3427), kind `docs`, branch `PMAT-3427-pp-quant-001-plan`, base `origin/main` `a672677f4`
- llama.cpp `3173a5647` `[X]`; release asset under test `apr-v0.68.0-x86_64-unknown-linux-gnu-cpu.tar.gz` (`sha256sum -c` OK)
- operator brief v4 (2026-09-17). Its design reference (v3) was not on disk: rows that depend on v3-only definitions (T4, Q4, Q6, X1, tables T/E/B, the quorum grid) were not run and nothing was invented for them.

## Harness
- `kind-gate.sh PMAT-3427` → exit 3, "kind=docs is not implemented (owned refusal)"; `model-gate.sh` → exit 2. The paiml-implement rail was therefore **not** used: no worker, no delegate, no quorum lane, 0 subagents. This PR is ordinary docs work by the orchestrator.
- `verify.sh` from a fresh clone of paiml-implement `origin/main` @ `1d34e47`: **63/66** — red: `orchestrator-lock`, `bashrs-gate`, `case-tables` (`t12-install-statusline.sh`); exactly the three waived, none other; wall 1179 s `[V]`
- Operator waiver (brief v4 §0.1) covers `orchestrator-lock`, `bashrs-gate`, `case-tables` only.

## Row A — train state `[V]`
`git tag --sort=-creatordate | head -1` → `v0.68.0` (2026-09-17T06:35:55Z). `v0.69.0` not tagged ⇒ slice M rides 0.69.0; T-0 waits; andon at 2026-09-20T06:35Z. Milestones at session start: 0.68.0 open 2 / closed 93 (shipped with open issues → #3445) · 0.69.0 open 116 / closed 14 · 0.70.0 open 233 / closed 13.

## Row B — #3091 on the published 0.68.0 binary `[V]` — NOT FIXED
| file (`unsloth/Qwen3.5-0.8B-GGUF`) | sha256[0:16] | `apr run <f> --prompt "2+2=" --max-tokens 8 --no-gpu` |
|---|---|---|
| `Qwen3.5-0.8B-UD-IQ2_XXS.gguf` (338 MB) | `a369165c8ec45a92` | exit 8 · `tensor_byte_size` · type 16 |
| `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (559 MB) | `3177ebd67afe4438` | exit 8 · `tensor_byte_size` · type 23 |
| `Qwen3.5-0.8B-Q4_K_M.gguf` (control) | — | exit 0 · `2+2=4` |

## Row C — Pareto of the five cited defects `[V-diff]`
| issue | fix PR | what the diff does | class | placeholder removed? |
|---|---|---|---|---|
| #1749 | #1751 | new `bench_moe.rs`; `apr bench` routes MoE GGUFs to the MoE path | placeholder | no — re-route |
| #1789 | #1790, #1806 | `validate_matmul_weight_shape` rejects empty data; HTTP 501 for `qwen3_moe` | placeholder | no — guard |
| #2535 | #2541 | `validate_quantized_tensors` checks attention only when arch is MoE | placeholder | no — guard |
| #3341 | #3405 | `SUPPORTED_EXPERT_QTYPES` + drift test; adds `QTYPE_LABELS` (13 rows) | qtype list | n/a |
| #3091 reopen | open | `tensor_byte_size` has 11 arms | qtype list | n/a |

Placeholder 3 of 5, qtype list 2 of 5 ⇒ brief v4 §2: **T1–T3 join M directly after M1.** The placeholder is `dense_ffn_placeholder` (`byte_size: 0`, `qtype: 0` = F32), built in `gguf/transformer.rs::load_quantized_layer_moe_skeleton` and `gguf/qwen3_moe_load.rs:819`; a second instance exists for tied `lm_head` (`gguf/loader_apr_quantized.rs`). Consumers of `.ffn_{up,gate,down}_weight`: 63 non-test files in `aprender-serve/src` `[V]`; 14 contain none of `moe|is_empty()|placeholder|validate_matmul_weight_shape` (6 `gguf/inference`, 4 `gguf/cuda`, 1 `gpu/adapters`, 3 other) — a file-level heuristic, `[U]` as a semantic claim.

Correction recorded: the 09:20Z comment on #3418 classed #3091 as "arch" (4/1 split) from PR bodies; the diffs and the reopen give 3/2.

## Row D — Table G: non-test files in `aprender-serve/src` naming a `GGUF_TYPE_*` tensor constant `[V]`
`git grep -l -E 'GGUF_TYPE_[A-Z0-9_]+' -- 'crates/aprender-serve/src/*.rs'` minus test files → 27. No two handled-sets of size ≥ 10 agree across loader and CPU paths.

| gguf/types.rs | BF16 F16 F32 Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0 | 13 | 0 | loader |
| gguf/metadata.rs | BF16 F16 F32 Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0 | 13 | 16 | loader |
| gguf/loader.rs | BF16 F16 F32 Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0 | 13 | 0 | loader |
| gguf/inference/forward/acceleration.rs | BF16 F16 F32 Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0 | 13 | 7 | cpu |
| convert/q4k_converter_helpers.rs | BF16 F16 F32 Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0 | 13 | 0 | loader |
| gguf/transformer.rs | BF16 F16 F32 Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_K Q6_K Q8_0 | 12 | 2 | loader |
| gguf/inference/matmul.rs | BF16 F16 F32 Q4_0 Q4_1 Q4_K Q5_0 Q5_K Q6_K Q8_0 | 10 | 0 | cpu |
| gguf/inference/matmul_fused.rs | BF16 F16 F32 Q4_0 Q4_1 Q4_K Q5_0 Q5_K Q6_K Q8_0 | 10 | 9 | cpu |
| gguf/inference/fused_matmul_into.rs | BF16 F16 F32 Q4_0 Q4_1 Q4_K Q5_0 Q5_K Q6_K Q8_0 | 10 | 1 | cpu |
| gguf/embedding.rs | Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0 | 10 | 0 | cpu |
| gpu/adapters/wgpu_adapter.rs | F16 F32 Q4_K Q5_K Q6_K | 5 | 0 | wgpu |
| gguf/qwen3_moe_load.rs | F32 Q4_K Q6_K | 3 | 4 | loader |
| gguf/inference/forward/single.rs | Q4_K Q5_K Q6_K | 3 | 0 | cpu |
| gguf/inference/forward/results.rs | Q4_K Q5_K Q6_K | 3 | 0 | cpu |
| gguf/inference/forward/batch.rs | Q4_K Q5_K Q6_K | 3 | 0 | cpu |
| gguf/cuda/expert_swiglu_cuda.rs | Q4_K Q6_K Q8_0 | 3 | 7 | cuda |
| gguf/transformer_quantized_layer_field.rs | Q4_0 Q4_K | 2 | 0 | loader |
| gguf/quantized.rs | Q4_K Q6_K | 2 | 0 | cpu |
| gguf/model_owned_quantized.rs | Q4_K Q8_0 | 2 | 0 | loader |
| gguf/model_gguf_transformer.rs | Q4_K Q6_K | 2 | 0 | loader |
| gguf/cuda/matmul.rs | Q4_K Q6_K | 2 | 3 | cuda |
| gguf/loader_gguf_model_02.rs | Q3_K | 1 | 0 | loader |
| gguf/inference/forward/traced.rs | Q4_K | 1 | 1 | cpu |
| gguf/inference/forward/forward_qwen35.rs | F32 | 1 | 0 | cpu |
| gguf/inference/forward/ffn_block.rs | Q4_K | 1 | 0 | cpu |
| gguf/cuda/moe_ffn_forward_layer_cuda.rs | F32 | 1 | 3 | cuda |
| gguf/cuda/cuda.rs | Q4_K | 1 | 2 | cuda |

Reconciliation against PR #3420's `dispatch_sites` (30): 24 in both, union 33. Tree-only: `gguf/cuda/moe_ffn_forward_layer_cuda.rs`, `gguf/inference/forward/forward_qwen35.rs`, `gguf/loader_gguf_model_02.rs` (they name only F32 / Q3_K; the PR's oracle keys on Q4_K, Q4_0 and `match qtype`). Contract-only: `api/model_source.rs`, `apr/cache.rs`, `convert/q4k_conversion_stats.rs`, `cuda/executor/layers/cublas_prefill/gemm.rs`, `cuda/executor/layers/gemv_dispatch.rs`, `cuda/types.rs` (they match on `qtype` without naming a constant). Bare-integer dispatch: regex upper bound 558 non-test lines over 10+ crates `[U]`.

Byte-size functions keyed on a raw qtype: 3 (`gguf/transformer.rs:435`, `convert/q4k_conversion_stats.rs:6`, `convert/q4k_converter_helpers.rs:20`). ggml tensor-type enums: 3 (serve 16 ids, core 12, compute 15); further `Quant*`/`Quantization*` enums: 14.

## Row D — Table S: `enum ggml_type` at llama.cpp `3173a5647` `[X]`, sizes from gguf-py `GGML_QUANT_SIZES` at the same sha
43 rows = 35 live + 8 removed. Live ids named by no aprender enum: 13.

| id | name | family | blck_size | type_size | serve enum | core enum | compute enum | `tensor_byte_size` arm |
|---|---|---|---|---|---|---|---|---|
| 0 | F32 | float | 1 | 4 | Y | Y | Y | Y |
| 1 | F16 | float | 1 | 2 | Y | Y | Y | Y |
| 2 | Q4_0 | affine | 32 | 18 | Y | Y | Y | Y |
| 3 | Q4_1 | affine | 32 | 20 | Y | Y | Y | Y |
| 4 | Q4_2 | removed | - | - | · | · | · | · |
| 5 | Q4_3 | removed | - | - | · | · | · | · |
| 6 | Q5_0 | affine | 32 | 22 | Y | · | Y | Y |
| 7 | Q5_1 | affine | 32 | 24 | Y | · | Y | · |
| 8 | Q8_0 | affine | 32 | 34 | Y | Y | Y | Y |
| 9 | Q8_1 | affine | 32 | 40 | Y | · | Y | · |
| 10 | Q2_K | k-quant | 256 | 84 | Y | · | Y | Y |
| 11 | Q3_K | k-quant | 256 | 110 | Y | · | Y | · |
| 12 | Q4_K | k-quant | 256 | 144 | Y | Y | Y | Y |
| 13 | Q5_K | k-quant | 256 | 176 | Y | · | Y | Y |
| 14 | Q6_K | k-quant | 256 | 210 | Y | Y | Y | Y |
| 15 | Q8_K | k-quant | 256 | 292 | · | · | Y | · |
| 16 | IQ2_XXS | codebook | 256 | 66 | Y | · | · | · |
| 17 | IQ2_XS | codebook | 256 | 74 | Y | · | · | · |
| 18 | IQ3_XXS | codebook | 256 | 98 | · | · | · | · |
| 19 | IQ1_S | codebook | 256 | 50 | · | · | · | · |
| 20 | IQ4_NL | codebook | 32 | 18 | · | · | · | · |
| 21 | IQ3_S | codebook | 256 | 110 | · | · | · | · |
| 22 | IQ2_S | codebook | 256 | 82 | · | · | · | · |
| 23 | IQ4_XS | codebook | 256 | 136 | · | · | · | · |
| 24 | I8 | int | 1 | 1 | · | Y | · | · |
| 25 | I16 | int | 1 | 2 | · | Y | · | · |
| 26 | I32 | int | 1 | 4 | · | Y | · | · |
| 27 | I64 | int | 1 | 8 | · | Y | · | · |
| 28 | F64 | float | 1 | 8 | · | Y | · | · |
| 29 | IQ1_M | codebook | 256 | 56 | · | · | · | · |
| 30 | BF16 | float | 1 | 2 | Y | · | Y | Y |
| 31 | Q4_0_4_4 | removed | - | - | · | · | · | · |
| 32 | Q4_0_4_8 | removed | - | - | · | · | · | · |
| 33 | Q4_0_8_8 | removed | - | - | · | · | · | · |
| 34 | TQ1_0 | ternary | 256 | 54 | · | · | · | · |
| 35 | TQ2_0 | ternary | 256 | 66 | · | · | · | · |
| 36 | IQ4_NL_4_4 | removed | - | - | · | · | · | · |
| 37 | IQ4_NL_4_8 | removed | - | - | · | · | · | · |
| 38 | IQ4_NL_8_8 | removed | - | - | · | · | · | · |
| 39 | MXFP4 | microscale-fp | 32 | 17 | · | · | · | · |
| 40 | NVFP4 | microscale-fp | 64 | 36 | · | · | · | · |
| 41 | Q1_0 | affine | 128 | 18 | · | · | · | · |
| 42 | Q2_0 | affine | 64 | 18 | · | · | · | · |

## Rows F–I — writes (each read back)
- #3091 repro comment · #3420 review comment · pmat `--path` leak → paiml/paiml-mcp-agent-toolkit#1386 (47 paths returned, 0 inside `--path`, truth 0)
- minted: epic #3428; M3 #3429 · M1 #3430 · M2 #3431 · M4 #3432 · T1 #3433 · T2 #3434 (0.69.0); T3 #3435 · Q2 #3436–#3440 (later closed into #3423) · Q3 #3441 · Q5 #3442 · P4 #3443 (0.70.0); S1 #3445 · §1.5 landing #3446 (0.69.0). #3444 closed as a duplicate of #3422.
- `Ĵ` on every ticket is 24 turns `[A]`: `estimate.sh aprender` reports 47 measured ledger rows and pools none — a harness finding, not a measurement.

## Row J — 0.69.0 hygiene `[V]` (`gh api graphql`, milestone 6, open issues)
124 open: 109 no linked PR · 10 minted this run · 3 epics · 1 open PR (#3204→#3205) · 1 linked PR merged with the issue still open (#2753→#2809). Authors: noahgift 118, alfredodeza 5, guyernest 1. `updatedAt` is ≤ 7 days on every row, so it measures triage edits, not work. Only tickets minted by this run were placed; the rest is one HRQ comment on #3421. The target of ≤ 15 open at T-0 is not reachable by this run's authority.

## HRQ rows
1. Placement conflict: #3423 puts PP-QUANT Phase 2/3 on a dedicated non-cadence milestone; brief v4 makes slice M 0.69.0 must-carry. #3436–#3440 restate Phase 2 and fold into whichever placement stands.
2. #3091's milestone (open in shipped 0.68.0) — human-authored ticket, not moved.
3. 109 no-PR rows in 0.69.0 one day before its due date.
4. Train rule ("dates never slip for scope", 12 h hold) vs must-carry ("T-0 waits", 72 h andon) — §1.1a records the exception; APR-RELEASE-001 has no §1.5 on `main` (#3446).

## Run 2 (same ticket, 2026-09-17) — tables from `origin/main` @ `7eb81a531`, design reference v3 (sha256 prefix `147c53d2d75a7f9d`, verified before use)

### Gate answer: does the rail admit `paiml-agy-delegate --mode plan` lanes under PMAT-3427? **No — STOP(rail).**
```
kind-gate: kind=docs is not implemented (AUTO-IMPL-SKILL-001-triage §2 — owned refusal); nothing was planned, dispatched, or written for PMAT-3427   (exit 3)
model-gate: usage/vacuity: unsupported kind 'docs' for PMAT-3427   (exit 2)
```
`kind-gate` refuses the ticket, not a lane type, so there is no carve-out for plan lanes. `model-gate` never admits, so no author model is admitted for this ticket, which `lane-reduce.sh --author-model` needs to refuse same-family review. Nothing in the `SubagentStart` hook consults either gate, so a lane would physically start; it would be dispatched past a Phase-0 refusal. Not done: no lane was launched, the ticket was not relabelled, no review of mine stands in for a lane. **There is no quorum grid.**

### Hand-written ticket state (operator: leave as is; delete when the harness gets a `kind=docs` route — harness row 6)
- `/run/user/1000/paiml-implement/495cc289-a575-4a44-b9b4-5358ebe242e3/1827347351/active-ticket` (repo key of the `/tmp/apr-head` worktree)
- `/run/user/1000/paiml-implement/495cc289-a575-4a44-b9b4-5358ebe242e3/124170554/active-ticket` (repo key of the session checkout; the edit hook keys on the session cwd, not on the edited file)
Both contain `PMAT-3427`, written by hand because `kind-gate` refused before Phase 1 could write them. No others were created.

### Row counts
| table | rows | how |
|---|---|---|
| T — tensor carriers (serve structs with a raw-integer qtype field) | 5 | struct scan; constructors = literal `Name {` sites |
| S — spec ids | 43 (35 live + 8 removed) | llama.cpp `3173a5647` + gguf-py sizes |
| E — `Ggml*`/`Quant*`/`GgufValue*` enums, non-test, workspace | 38 | 3 merge · 10 merge-candidate (≥ 5 ggml-named variants) · 24 keep? · 1 keep |
| B — raw-integer qtype fields and params | 53 (24 fields, 29 params; serve 34, core 8, apr-cli 7, cbtop 2, compute 1, train 1) | `(qtype|quant_type|ggml_type|tensor_type|dtype|weight_type): u8|u16|u32|i32|usize` |
| G — dispatch sites, file:line, over the 33-file union | 231 in 28 files (151 arm/ref · 54 `.qtype` compare · 24 `match` · 2 bare-integer) | 5 union files are not sites: 2 whole-file test modules, 3 comment/const only |

**T∩B (decides X1):** 14 of the 34 serve rows of Table B (41 %) sit in a file that uses a Table T carrier; 9 of 24 files (38 %). X1's falsifier is "< 20 % ⇒ parallel", so it does **not** fire: T1 before Q2 stands. File-level measure; not a quorum ruling.

The 558-line literal-arm upper bound (v3 Table G's `qtype | not-qtype` resolution) was **not** resolved; only 2 bare-integer sites inside the 33-file union were classified. It stays `[U]`.

### Table T
| carrier | defined in | size/type fields | can hold (len=0, qtype=0)? | literal constructor sites | validating `new`? | files using it | files indexing `.data` | …with no length/empty check in file |
|---|---|---|---|---|---|---|---|---|
| `QuantizedTensorRef` | gguf/quantized.rs | offset: usize, byte_size: usize, num_elements: usize, qtype: u32 | yes | 29 | no | 6 | 0 | 0 |
| `OwnedQuantizedTensor` | gguf/quantized.rs | data: Vec<u8>, in_dim: usize, out_dim: usize, qtype: u32 | yes | 26 | no | 25 | 7 | 0 |
| `TensorInfo` | gguf/types.rs | n_dims: u32, dims: Vec<u64>, qtype: u32, offset: u64 | yes | 3 | no | 5 | 0 | 0 |
| `RawTensor` | convert/mod.rs | data: Vec<u8> | yes | 2 | no | 4 | 0 | 0 |
| `Qwen3MoeQuantizedLayer` | gguf/qwen3_moe_load.rs |  | yes | 4 | no | 10 | 0 | 0 |

Every carrier can hold `(len = 0, qtype = 0)`: fields are plain integers and `Vec`/offsets, and none has a validating constructor. The two "indexing" columns are a file-level regex and found an `is_empty()`/length comparison somewhere in each indexing file — that is weak evidence of a guard, not proof of one; the consumer count that matters for T1 is the 63-file figure in Row C.

### Table E
| enum | file:line | variants | ggml-named variants | verdict |
|---|---|---|---|---|
| `QuantType` | aprender-registry/src/format.rs:568 | 30 | 30 | merge-candidate (names ≥5 ggml types) |
| `GgmlQuantType` | aprender-serve/src/gguf/types.rs:101 | 16 | 16 | merge → GgmlType |
| `GgmlType` | aprender-compute/src/inference/gguf.rs:27 | 15 | 15 | merge → GgmlType |
| `QuantTag` | apr-cli/src/commands/auto_quant.rs:26 | 10 | 10 | merge-candidate (names ≥5 ggml types) |
| `QuantType` | aprender-compute/src/brick/tracing/quant_type.rs:10 | 10 | 10 | merge-candidate (names ≥5 ggml types) |
| `GgmlType` | aprender-core/src/format/gguf/types.rs:55 | 12 | 9 | merge → GgmlType |
| `QuantFormat` | aprender-cbtop/src/quantize/format.rs:12 | 10 | 8 | merge-candidate (names ≥5 ggml types) |
| `QuantType` | aprender-compute/src/tuner/types.rs:15 | 8 | 8 | merge-candidate (names ≥5 ggml types) |
| `QuantType` | aprender-qa-runner/src/conversion.rs:330 | 9 | 8 | merge-candidate (names ≥5 ggml types) |
| `WeightQuantType` | aprender-serve/src/cuda/types.rs:99 | 8 | 8 | merge-candidate (names ≥5 ggml types) |
| `QuantType` | aprender-serve/src/testing/mod.rs:92 | 8 | 8 | merge-candidate (names ≥5 ggml types) |
| `QuantizationType` | aprender-train/src/ecosystem/realizar/quantization.rs:9 | 8 | 8 | merge-candidate (names ≥5 ggml types) |
| `QuantFormat` | aprender-cuda-edge/src/quant_oracle/boundary.rs:11 | 6 | 6 | merge-candidate (names ≥5 ggml types) |
| `QuantKernel` | aprender-cgp/src/profilers/quant.rs:9 | 5 | 4 | keep? (different concept — confirm) |
| `QuantType` | aprender-core/src/format/quantize.rs:39 | 5 | 4 | keep? (different concept — confirm) |
| `GgufValue` | aprender-cbtop/src/quantize/gguf.rs:24 | 13 | 3 | keep? (different concept — confirm) |
| `AprQuantizationType` | aprender-serve/src/apr_transformer/loader.rs:303 | 3 | 3 | keep? (different concept — confirm) |
| `Quantization` | aprender-train-inspect/src/convert.rs:113 | 3 | 3 | keep? (different concept — confirm) |
| `KvQuantType` | aprender-serve/src/paged_kv/mod_compute_prefix.rs:254 | 3 | 2 | keep? (different concept — confirm) |
| `QuantizedKvData` | aprender-serve/src/paged_kv/mod_compute_prefix.rs:410 | 3 | 2 | keep? (different concept — confirm) |
| `GgufQuantization` | aprender-train/src/hf_pipeline/export/gguf_writer.rs:13 | 3 | 2 | keep? (different concept — confirm) |
| `GGUFQuantType` | aprender-train/src/quant/gguf_quant/quant_type.rs:5 | 2 | 2 | keep? (different concept — confirm) |
| `QuantScheme` | apr-cli/src/commands/quantize.rs:27 | 4 | 1 | keep? (different concept — confirm) |
| `QuantizationType` | aprender-core/src/format/converter_types_expectations.rs:149 | 4 | 1 | keep? (different concept — confirm) |
| `QuantizationType` | aprender-registry/src/lineage/mod.rs:53 | 5 | 1 | keep? (different concept — confirm) |
| `AutoQuantError` | apr-cli/src/commands/auto_quant.rs:100 | 3 | 0 | keep? (different concept — confirm) |
| `QuantizeArgvVerdict` | apr-cli/src/commands/quantize_flag_parity.rs:49 | 2 | 0 | keep? (different concept — confirm) |
| `QuantScheme` | aprender-cbtop/src/grammar/transform.rs:7 | 3 | 0 | keep? (different concept — confirm) |
| `QuantizationType` | aprender-core/src/demo/mod.rs:344 | 4 | 0 | keep? (different concept — confirm) |
| `GgufValueType` | aprender-core/src/format/gguf/types.rs:23 | 13 | 0 | keep (metadata value-type id space) |
| `GgufValue` | aprender-core/src/format/gguf/types.rs:84 | 16 | 0 | keep? (different concept — confirm) |
| `QuantizationType` | aprender-core/src/stack/mod.rs:223 | 6 | 0 | keep? (different concept — confirm) |
| `QuantizationError` | aprender-orchestrate/src/oracle/rag/quantization/error.rs:7 | 6 | 0 | keep? (different concept — confirm) |
| `QuantFamily` | aprender-serve/src/quantize/format_trait.rs:31 | 2 | 0 | keep? (different concept — confirm) |
| `QuantMethod` | aprender-train/src/config/cli/quant_merge.rs:76 | 2 | 0 | keep? (different concept — confirm) |
| `QuantPublishError` | aprender-train/src/hf_pipeline/export/publish_pipeline.rs:66 | 2 | 0 | keep? (different concept — confirm) |
| `QuantGranularity` | aprender-train/src/quant/granularity/types.rs:7 | 3 | 0 | keep? (different concept — confirm) |
| `QuantMode` | aprender-train/src/quant/granularity/types.rs:19 | 2 | 0 | keep? (different concept — confirm) |

### Table B
| file:line | kind | struct | name | int type |
|---|---|---|---|---|
| apr-cli/src/commands/hex.rs:301 | field | GgufTensorEntry | dtype | u32 |
| apr-cli/src/commands/ptx_map.rs:61 | param | - | qtype | u32 |
| apr-cli/src/commands/ptx_map_print_kernel.rs:565 | param | - | qtype | u32 |
| apr-cli/src/commands/ptx_map_print_kernel.rs:601 | param | - | qtype | u32 |
| apr-cli/src/commands/rosetta_validate.rs:372 | param | - | dtype | u8 |
| apr-cli/src/commands/serve/server.rs:15 | param | - | qtype | u32 |
| apr-cli/src/commands/sliding_window_entropy.rs:295 | param | - | dtype | u32 |
| aprender-cbtop/src/quantize/gguf.rs:50 | field | GgufTensorInfo | dtype | u32 |
| aprender-cbtop/src/quantize/mod.rs:117 | param | - | ggml_type | u32 |
| aprender-compute/src/contracts.rs:211 | field | WeightBufferError | ggml_type | u32 |
| aprender-core/src/format/converter/write_model_config.rs:136 | param | - | dtype | u32 |
| aprender-core/src/format/gguf/api.rs:261 | field | GgufRawTensor | dtype | u32 |
| aprender-core/src/format/gguf/builder.rs:21 | field | - | dtype | u32 |
| aprender-core/src/format/gguf/dequantize.rs:254 | field | - | dtype | u32 |
| aprender-core/src/format/gguf/merge.rs:55 | field | TensorPlan | dtype | u32 |
| aprender-core/src/format/gguf/reader.rs:325 | field | GgufTensorMeta | dtype | u32 |
| aprender-core/src/format/safetensors.rs:9 | param | - | dtype | u32 |
| aprender-core/src/format/tensors.rs:571 | param | - | dtype | u32 |
| aprender-serve/src/api/model_source.rs:206 | param | - | qtype | u32 |
| aprender-serve/src/apr/beat_fail_closed_config.rs:26 | param | - | dtype | u8 |
| aprender-serve/src/apr/cache.rs:11 | param | - | qtype | u32 |
| aprender-serve/src/apr/cache.rs:81 | field | - | qtype | u32 |
| aprender-serve/src/apr/weight.rs:60 | field | - | qtype | u32 |
| aprender-serve/src/apr/weight.rs:453 | param | - | qtype | u32 |
| aprender-serve/src/apr_transformer/mod_dequant_q4k_apr.rs:41 | param | - | dtype | u8 |
| aprender-serve/src/contract_gate.rs:486 | param | - | dtype | u8 |
| aprender-serve/src/convert/mod.rs:427 | field | RawTensor | dtype | u32 |
| aprender-serve/src/convert/q4k_conversion_stats.rs:6 | param | - | qtype | u32 |
| aprender-serve/src/convert/q4k_converter_helpers.rs:20 | param | - | qtype | u32 |
| aprender-serve/src/cuda/executor/weights.rs:152 | field | - | qtype | u32 |
| aprender-serve/src/cuda/executor/weights.rs:299 | field | - | qtype | u32 |
| aprender-serve/src/gguf/cuda/expert_swiglu_cuda.rs:182 | field | - | qtype | u32 |
| aprender-serve/src/gguf/cuda/loading.rs:282 | field | - | qtype | u32 |
| aprender-serve/src/gguf/cuda/weights_preload_gpu.rs:303 | field | - | qtype | u32 |
| aprender-serve/src/gguf/dtype.rs:16 | param | - | qtype | u32 |
| aprender-serve/src/gguf/dtype.rs:47 | param | - | qtype | u32 |
| aprender-serve/src/gguf/embedding.rs:27 | param | - | qtype | u32 |
| aprender-serve/src/gguf/embedding.rs:49 | param | - | qtype | u32 |
| aprender-serve/src/gguf/inference/matmul_fused.rs:543 | param | - | qtype | u32 |
| aprender-serve/src/gguf/quantized.rs:34 | field | QuantizedTensorRef | qtype | u32 |
| aprender-serve/src/gguf/quantized.rs:103 | field | OwnedQuantizedTensor | qtype | u32 |
| aprender-serve/src/gguf/qwen3_moe_load.rs:107 | param | - | qtype | u32 |
| aprender-serve/src/gguf/qwen3_moe_load.rs:127 | param | - | qtype | u32 |
| aprender-serve/src/gguf/qwen3_moe_load.rs:452 | field | Qwen3MoeQuantizedLayer | qtype | u32 |
| aprender-serve/src/gguf/qwen3_moe_load.rs:476 | param | - | qtype | u32 |
| aprender-serve/src/gguf/transformer.rs:435 | param | - | qtype | u32 |
| aprender-serve/src/gguf/types.rs:329 | field | TensorInfo | qtype | u32 |
| aprender-serve/src/gpu/backend.rs:79 | field | MockBackend | qtype | u32 |
| aprender-serve/src/gpu/backend.rs:211 | field | MockBackend | qtype | u32 |
| aprender-serve/src/gpu/mock_backend.rs:126 | field | MockBackend | qtype | u32 |
| aprender-serve/src/infer/inference_result.rs:512 | param | - | qtype | u32 |
| aprender-serve/src/infer/mod.rs:38 | param | - | qtype | u32 |
| aprender-train/src/hf_pipeline/export/gguf_verify/types.rs:26 | field | GgufTensorInfo | dtype | u32 |

### Table G — file:line
| file:line | fn | form | ids on the line |
|---|---|---|---|
| api/model_source.rs:207 | `gguf_qtype_name` | match |  |
| apr/cache.rs:12 | `cached_apr_gpu_gemv_supported_qtype` | bare-int |  |
| apr/cache.rs:88 | `dispatch_quantized_gemv` | bare-int |  |
| apr/cache.rs:89 | `dispatch_quantized_gemv` | match |  |
| apr/cache.rs:91 | `dispatch_quantized_gemv` | match |  |
| convert/q4k_conversion_stats.rs:7 | `ggml_tensor_byte_size` | match |  |
| convert/q4k_converter_helpers.rs:22 | `ggml_tensor_byte_size_h` | arm/ref | BF16 F16 F32 Q2_K Q3_K |
| convert/q4k_converter_helpers.rs:23 | `ggml_tensor_byte_size_h` | arm/ref | Q4_0 Q4_1 Q4_K Q5_0 Q5_1 |
| convert/q4k_converter_helpers.rs:24 | `ggml_tensor_byte_size_h` | arm/ref | Q5_K Q6_K Q8_0 |
| convert/q4k_converter_helpers.rs:31 | `ggml_tensor_byte_size_h` | match |  |
| convert/q4k_converter_helpers.rs:32 | `ggml_tensor_byte_size_h` | arm/ref | F32 |
| convert/q4k_converter_helpers.rs:33 | `ggml_tensor_byte_size_h` | arm/ref | BF16 F16 |
| convert/q4k_converter_helpers.rs:34 | `ggml_tensor_byte_size_h` | arm/ref | Q4_0 |
| convert/q4k_converter_helpers.rs:35 | `ggml_tensor_byte_size_h` | arm/ref | Q4_1 |
| convert/q4k_converter_helpers.rs:36 | `ggml_tensor_byte_size_h` | arm/ref | Q5_0 |
| convert/q4k_converter_helpers.rs:37 | `ggml_tensor_byte_size_h` | arm/ref | Q5_1 |
| convert/q4k_converter_helpers.rs:38 | `ggml_tensor_byte_size_h` | arm/ref | Q8_0 |
| convert/q4k_converter_helpers.rs:39 | `ggml_tensor_byte_size_h` | arm/ref | Q2_K |
| convert/q4k_converter_helpers.rs:40 | `ggml_tensor_byte_size_h` | arm/ref | Q3_K |
| convert/q4k_converter_helpers.rs:41 | `ggml_tensor_byte_size_h` | arm/ref | Q4_K |
| convert/q4k_converter_helpers.rs:42 | `ggml_tensor_byte_size_h` | arm/ref | Q5_K |
| convert/q4k_converter_helpers.rs:43 | `ggml_tensor_byte_size_h` | arm/ref | Q6_K |
| cuda/executor/layers/cublas_prefill/gemm.rs:25 | `get_or_cache_fp8_weight` | match |  |
| cuda/executor/layers/cublas_prefill/gemm.rs:26 | `get_or_cache_fp8_weight` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:27 | `get_or_cache_fp8_weight` | arm/ref | Q6K |
| cuda/executor/layers/cublas_prefill/gemm.rs:426 | `get_or_cache_fp16_weight` | match |  |
| cuda/executor/layers/cublas_prefill/gemm.rs:427 | `get_or_cache_fp16_weight` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:428 | `get_or_cache_fp16_weight` | arm/ref | Q6K |
| cuda/executor/layers/cublas_prefill/gemm.rs:546 | `cublas_prefill_gemm` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:559 | `cublas_prefill_gemm` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:572 | `cublas_prefill_gemm` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:585 | `cublas_prefill_gemm` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:620 | `cublas_prefill_gemm` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:650 | `cublas_prefill_gemm` | match |  |
| cuda/executor/layers/cublas_prefill/gemm.rs:651 | `cublas_prefill_gemm` | arm/ref | Q4K |
| cuda/executor/layers/cublas_prefill/gemm.rs:652 | `cublas_prefill_gemm` | arm/ref | Q6K |
| cuda/executor/layers/gemv_dispatch.rs:42 | `gemv_dispatch` | match |  |
| cuda/executor/layers/gemv_dispatch.rs:43 | `gemv_dispatch` | arm/ref | Q4_0 |
| cuda/executor/layers/gemv_dispatch.rs:44 | `gemv_dispatch` | arm/ref | Q4_1 |
| cuda/executor/layers/gemv_dispatch.rs:45 | `gemv_dispatch` | arm/ref | Q5_0 |
| cuda/executor/layers/gemv_dispatch.rs:46 | `gemv_dispatch` | arm/ref | Q4K |
| cuda/executor/layers/gemv_dispatch.rs:47 | `gemv_dispatch` | arm/ref | Q5K |
| cuda/executor/layers/gemv_dispatch.rs:48 | `gemv_dispatch` | arm/ref | Q6K |
| cuda/executor/layers/gemv_dispatch.rs:49 | `gemv_dispatch` | arm/ref | Q8_0 |
| cuda/executor/layers/gemv_dispatch.rs:50 | `gemv_dispatch` | arm/ref | F32 |
| cuda/types.rs:301 | `bind` | match |  |
| cuda/types.rs:302 | `bind` | arm/ref | Q4K |
| cuda/types.rs:303 | `bind` | arm/ref | Q5K |
| cuda/types.rs:304 | `bind` | arm/ref | Q6K |
| cuda/types.rs:305 | `bind` | arm/ref | Q8_0 |
| cuda/types.rs:306 | `bind` | arm/ref | Q4_0 |
| cuda/types.rs:307 | `bind` | arm/ref | Q5_0 |
| cuda/types.rs:308 | `bind` | arm/ref | Q4_1 |
| cuda/types.rs:309 | `bind` | arm/ref | F32 |
| gguf/cuda/cuda.rs:382 | `fused_matmul_cuda` | cmp | Q4_K |
| gguf/cuda/expert_swiglu_cuda.rs:191 | `matvec_qtype_cuda` | match |  |
| gguf/cuda/expert_swiglu_cuda.rs:192 | `matvec_qtype_cuda` | arm/ref | Q4_K |
| gguf/cuda/expert_swiglu_cuda.rs:198 | `matvec_qtype_cuda` | arm/ref | Q6_K |
| gguf/cuda/matmul.rs:22 | `fused_matmul_cuda_with_key` | cmp | Q4_K Q6_K |
| gguf/cuda/matmul.rs:47 | `fused_matmul_cuda_with_key` | cmp | Q4_K |
| gguf/cuda/matmul.rs:53 | `fused_matmul_cuda_with_key` | cmp | Q4_K |
| gguf/cuda/moe_ffn_forward_layer_cuda.rs:68 | `moe_ffn_forward_layer_cuda` | cmp | F32 |
| gguf/cuda/moe_ffn_forward_layer_cuda.rs:231 | `moe_ffn_forward_layer_cuda_with_router` | cmp | F32 |
| gguf/embedding.rs:29 | `quant_block_layout` | arm/ref | Q2_K Q3_K Q4_0 Q4_1 Q4_K |
| gguf/embedding.rs:30 | `quant_block_layout` | arm/ref | Q5_0 Q5_1 Q5_K Q6_K Q8_0 |
| gguf/embedding.rs:32 | `quant_block_layout` | match |  |
| gguf/embedding.rs:34 | `quant_block_layout` | arm/ref | Q4_0 |
| gguf/embedding.rs:35 | `quant_block_layout` | arm/ref | Q8_0 |
| gguf/embedding.rs:36 | `quant_block_layout` | arm/ref | Q4_1 |
| gguf/embedding.rs:37 | `quant_block_layout` | arm/ref | Q5_0 |
| gguf/embedding.rs:38 | `quant_block_layout` | arm/ref | Q5_1 |
| gguf/embedding.rs:40 | `quant_block_layout` | arm/ref | Q4_K |
| gguf/embedding.rs:41 | `quant_block_layout` | arm/ref | Q5_K |
| gguf/embedding.rs:42 | `quant_block_layout` | arm/ref | Q6_K |
| gguf/embedding.rs:43 | `quant_block_layout` | arm/ref | Q2_K |
| gguf/embedding.rs:44 | `quant_block_layout` | arm/ref | Q3_K |
| gguf/inference/forward/acceleration.rs:89 | `dequantize_weight` | arm/ref | BF16 F16 F32 Q2_K |
| gguf/inference/forward/acceleration.rs:90 | `dequantize_weight` | arm/ref | Q3_K Q4_0 Q4_1 Q5_0 Q5_1 |
| gguf/inference/forward/acceleration.rs:91 | `dequantize_weight` | arm/ref | Q8_0 |
| gguf/inference/forward/acceleration.rs:103 | `dequantize_weight` | match |  |
| gguf/inference/forward/acceleration.rs:104 | `dequantize_weight` | arm/ref | Q4_K |
| gguf/inference/forward/acceleration.rs:117 | `dequantize_weight` | arm/ref | Q5_K |
| gguf/inference/forward/acceleration.rs:129 | `dequantize_weight` | arm/ref | Q6_K |
| gguf/inference/forward/acceleration.rs:143 | `dequantize_weight` | arm/ref | F32 |
| gguf/inference/forward/acceleration.rs:152 | `dequantize_weight` | arm/ref | F16 |
| gguf/inference/forward/acceleration.rs:160 | `dequantize_weight` | arm/ref | BF16 |
| gguf/inference/forward/acceleration.rs:172 | `dequantize_weight` | arm/ref | Q4_0 |
| gguf/inference/forward/acceleration.rs:173 | `dequantize_weight` | arm/ref | Q4_1 |
| gguf/inference/forward/acceleration.rs:174 | `dequantize_weight` | arm/ref | Q5_0 |
| gguf/inference/forward/acceleration.rs:175 | `dequantize_weight` | arm/ref | Q5_1 |
| gguf/inference/forward/acceleration.rs:176 | `dequantize_weight` | arm/ref | Q8_0 |
| gguf/inference/forward/acceleration.rs:178 | `dequantize_weight` | arm/ref | Q2_K |
| gguf/inference/forward/acceleration.rs:179 | `dequantize_weight` | arm/ref | Q3_K |
| gguf/inference/forward/batch.rs:14 | `-` | arm/ref | Q4_K Q5_K |
| gguf/inference/forward/batch.rs:15 | `-` | arm/ref | Q6_K |
| gguf/inference/forward/ffn_block.rs:790 | `matvec_into_honest` | cmp | Q4_K |
| gguf/inference/forward/ffn_block.rs:799 | `matvec_honest` | cmp | Q4_K |
| gguf/inference/forward/ffn_block.rs:840 | `ffn_up_gate_honest` | cmp | Q4_K |
| gguf/inference/forward/forward_qwen35.rs:407 | `load_f32_vec` | cmp | F32 |
| gguf/inference/forward/results.rs:31 | `scratch_q8k_up_gate` | cmp | Q4_K |
| gguf/inference/forward/results.rs:43 | `scratch_q8k_up_gate` | cmp |  |
| gguf/inference/forward/results.rs:46 | `scratch_q8k_up_gate` | cmp | Q4_K |
| gguf/inference/forward/results.rs:47 | `scratch_q8k_up_gate` | cmp | Q5_K |
| gguf/inference/forward/results.rs:48 | `scratch_q8k_up_gate` | cmp | Q6_K |
| gguf/inference/forward/results.rs:62 | `scratch_q8k_up_gate` | cmp | Q4_K |
| gguf/inference/forward/results.rs:80 | `scratch_q8k_up_gate` | cmp | Q4_K |
| gguf/inference/forward/results.rs:124 | `scratch_q8k_down_projection` | cmp | Q4_K |
| gguf/inference/forward/results.rs:286 | `scratch_attention_block` | cmp | Q4_K |
| gguf/inference/forward/results.rs:459 | `scratch_gelu_ffn` | cmp | Q4_K |
| gguf/inference/forward/results.rs:461 | `scratch_gelu_ffn` | cmp | Q4_K |
| gguf/inference/forward/results.rs:639 | `forward_single_with_scratch` | cmp | Q4_K |
| gguf/inference/forward/single.rs:13 | `-` | arm/ref | Q4_K Q5_K Q6_K |
| gguf/inference/forward/traced.rs:270 | `forward_traced` | cmp | Q4_K |
| gguf/inference/fused_matmul_into.rs:35 | `fused_matmul_into` | match |  |
| gguf/inference/fused_matmul_into.rs:36 | `fused_matmul_into` | arm/ref | Q4_0 |
| gguf/inference/fused_matmul_into.rs:42 | `fused_matmul_into` | arm/ref | Q8_0 |
| gguf/inference/fused_matmul_into.rs:49 | `fused_matmul_into` | arm/ref | Q4_K |
| gguf/inference/fused_matmul_into.rs:56 | `fused_matmul_into` | arm/ref | Q5_K |
| gguf/inference/fused_matmul_into.rs:63 | `fused_matmul_into` | arm/ref | Q6_K |
| gguf/inference/fused_matmul_into.rs:101 | `fused_gate_up_matmul_into` | cmp |  |
| gguf/inference/fused_matmul_into.rs:105 | `fused_gate_up_matmul_into` | match |  |
| gguf/inference/fused_matmul_into.rs:106 | `fused_gate_up_matmul_into` | arm/ref | Q4_K |
| gguf/inference/fused_matmul_into.rs:117 | `fused_gate_up_matmul_into` | arm/ref | Q5_K |
| gguf/inference/fused_matmul_into.rs:128 | `fused_gate_up_matmul_into` | arm/ref | Q6_K |
| gguf/inference/fused_matmul_into.rs:266 | `fused_rmsnorm_matmul` | cmp | Q4_0 |
| gguf/inference/fused_matmul_into.rs:359 | `fused_rmsnorm_lm_head` | cmp | Q4_0 |
| gguf/inference/fused_matmul_into.rs:388 | `fused_rmsnorm_ffn_up_gate` | cmp | Q4_0 |
| gguf/inference/fused_matmul_into.rs:389 | `fused_rmsnorm_ffn_up_gate` | cmp | Q4_0 |
| gguf/inference/fused_matmul_into.rs:432 | `qkv_matmul_q8k_into` | cmp | Q4_K |
| gguf/inference/fused_matmul_into.rs:459 | `qkv_matmul_q8k_into` | cmp | Q4_K |
| gguf/inference/fused_matmul_into.rs:468 | `qkv_matmul_q8k_into` | cmp | Q4_K |
| gguf/inference/fused_matmul_into.rs:477 | `qkv_matmul_q8k_into` | cmp | Q4_K |
| gguf/inference/fused_matmul_into.rs:503 | `dequantize_weight_for_cuda` | match |  |
| gguf/inference/fused_matmul_into.rs:505 | `dequantize_weight_for_cuda` | arm/ref | F32 |
| gguf/inference/fused_matmul_into.rs:514 | `dequantize_weight_for_cuda` | arm/ref | F16 |
| gguf/inference/fused_matmul_into.rs:526 | `dequantize_weight_for_cuda` | arm/ref | BF16 |
| gguf/inference/fused_matmul_into.rs:537 | `dequantize_weight_for_cuda` | arm/ref | Q4_0 |
| gguf/inference/fused_matmul_into.rs:538 | `dequantize_weight_for_cuda` | arm/ref | Q4_1 |
| gguf/inference/fused_matmul_into.rs:539 | `dequantize_weight_for_cuda` | arm/ref | Q5_0 |
| gguf/inference/fused_matmul_into.rs:540 | `dequantize_weight_for_cuda` | arm/ref | Q8_0 |
| gguf/inference/fused_matmul_into.rs:541 | `dequantize_weight_for_cuda` | arm/ref | Q4_K |
| gguf/inference/fused_matmul_into.rs:542 | `dequantize_weight_for_cuda` | arm/ref | Q5_K |
| gguf/inference/fused_matmul_into.rs:543 | `dequantize_weight_for_cuda` | arm/ref | Q6_K |
| gguf/inference/matmul_fused.rs:127 | `fused_matmul` | cmp | F32 |
| gguf/inference/matmul_fused.rs:133 | `fused_matmul` | cmp | BF16 |
| gguf/inference/matmul_fused.rs:141 | `fused_matmul` | cmp | F16 |
| gguf/inference/matmul_fused.rs:149 | `fused_matmul` | cmp | Q4_0 Q8_0 |
| gguf/inference/matmul_fused.rs:154 | `fused_matmul` | cmp | Q4_1 Q5_0 |
| gguf/inference/matmul_fused.rs:155 | `fused_matmul` | cmp | Q4_1 |
| gguf/inference/matmul_fused.rs:160 | `fused_matmul` | cmp | Q4_1 |
| gguf/inference/matmul_fused.rs:182 | `fused_matmul` | cmp |  |
| gguf/inference/matmul_fused.rs:184 | `fused_matmul` | cmp |  |
| gguf/inference/matmul_fused.rs:189 | `fused_matmul` | cmp |  |
| gguf/inference/matmul_fused.rs:269 | `fused_matmul_q4_q8` | cmp | Q4_0 |
| gguf/inference/matmul_fused.rs:304 | `fused_matmul_k_quants` | match |  |
| gguf/inference/matmul_fused.rs:305 | `fused_matmul_k_quants` | arm/ref | Q4_K |
| gguf/inference/matmul_fused.rs:306 | `fused_matmul_k_quants` | arm/ref | Q5_K |
| gguf/inference/matmul_fused.rs:307 | `fused_matmul_k_quants` | arm/ref | Q6_K |
| gguf/inference/matmul_fused.rs:322 | `fused_matmul_k_quants` | match |  |
| gguf/inference/matmul_fused.rs:323 | `fused_matmul_k_quants` | arm/ref | Q4_K |
| gguf/inference/matmul_fused.rs:324 | `fused_matmul_k_quants` | arm/ref | Q5_K |
| gguf/inference/matmul_fused.rs:325 | `fused_matmul_k_quants` | arm/ref | Q6_K |
| gguf/inference/matmul_fused.rs:357 | `fused_matmul_cuda` | match |  |
| gguf/inference/matmul_fused.rs:358 | `fused_matmul_cuda` | arm/ref | Q4_K |
| gguf/inference/matmul_fused.rs:359 | `fused_matmul_cuda` | arm/ref | Q5_K |
| gguf/inference/matmul_fused.rs:360 | `fused_matmul_cuda` | arm/ref | Q6_K |
| gguf/inference/matmul_fused.rs:366 | `fused_matmul_cuda` | cmp | Q4_K |
| gguf/inference/matmul_fused.rs:367 | `fused_matmul_cuda` | cmp | Q5_K |
| gguf/inference/matmul_fused.rs:368 | `fused_matmul_cuda` | cmp | Q6_K |
| gguf/inference/matmul_fused.rs:394 | `fused_matmul_cuda` | match |  |
| gguf/inference/matmul_fused.rs:395 | `fused_matmul_cuda` | arm/ref | Q4_K |
| gguf/inference/matmul_fused.rs:402 | `fused_matmul_cuda` | arm/ref | Q5_K |
| gguf/inference/matmul_fused.rs:409 | `fused_matmul_cuda` | arm/ref | Q6_K |
| gguf/inference/matmul_fused.rs:512 | `validate_matmul_weight_shape` | cmp | F32 |
| gguf/inference/matmul.rs:8 | `-` | arm/ref | BF16 F16 F32 Q4_0 |
| gguf/inference/matmul.rs:9 | `-` | arm/ref | Q4_1 Q4_K Q5_0 Q5_K Q6_K Q8_0 |
| gguf/loader.rs:10 | `-` | arm/ref | BF16 F16 |
| gguf/loader.rs:11 | `-` | arm/ref | F32 Q2_K Q3_K Q4_0 Q4_1 Q4_K |
| gguf/loader.rs:12 | `-` | arm/ref | Q5_0 Q5_1 Q5_K Q6_K Q8_0 |
| gguf/metadata.rs:59 | `get_tensor_f32` | match |  |
| gguf/metadata.rs:60 | `get_tensor_f32` | arm/ref | F32 |
| gguf/metadata.rs:84 | `get_tensor_f32` | arm/ref | Q4_0 |
| gguf/metadata.rs:115 | `get_tensor_f32` | arm/ref | Q8_0 |
| gguf/metadata.rs:145 | `get_tensor_f32` | arm/ref | Q2_K |
| gguf/metadata.rs:174 | `get_tensor_f32` | arm/ref | Q3_K |
| gguf/metadata.rs:203 | `get_tensor_f32` | arm/ref | Q4_K |
| gguf/metadata.rs:232 | `get_tensor_f32` | arm/ref | Q5_K |
| gguf/metadata.rs:261 | `get_tensor_f32` | arm/ref | Q6_K |
| gguf/metadata.rs:290 | `get_tensor_f32` | arm/ref | F16 |
| gguf/metadata.rs:311 | `get_tensor_f32` | arm/ref | BF16 |
| gguf/metadata.rs:334 | `get_tensor_f32` | arm/ref | Q4_1 |
| gguf/metadata.rs:364 | `get_tensor_f32` | arm/ref | Q5_0 |
| gguf/metadata.rs:394 | `get_tensor_f32` | arm/ref | Q5_1 |
| gguf/model_gguf_transformer.rs:103 | `test_owned_quantized_model_clone` | arm/ref | Q4_K |
| gguf/model_gguf_transformer.rs:169 | `test_owned_quantized_model_debug` | arm/ref | Q4_K |
| gguf/model_gguf_transformer.rs:392 | `test_owned_quantized_model_with_bias` | arm/ref | Q4_K |
| gguf/model_gguf_transformer.rs:446 | `test_owned_quantized_model_clone_preserves_data` | arm/ref | Q6_K |
| gguf/model_gguf_transformer.rs:464 | `test_owned_quantized_model_clone_preserves_data` | arm/ref | Q6_K |
| gguf/qwen3_moe_load.rs:84 | `-` | arm/ref | Q4_K Q6_K |
| gguf/qwen3_moe_load.rs:193 | `validate_moe_layer_tensors` | cmp | F32 |
| gguf/qwen3_moe_load.rs:462 | `matvec_for_qtype` | match |  |
| gguf/qwen3_moe_load.rs:463 | `matvec_for_qtype` | arm/ref | Q4_K |
| gguf/qwen3_moe_load.rs:464 | `matvec_for_qtype` | arm/ref | Q6_K |
| gguf/qwen3_moe_load.rs:546 | `moe_ffn_forward_layer` | cmp | F32 |
| gguf/qwen3_moe_load.rs:717 | `moe_ffn_forward_layer_with_router` | cmp | F32 |
| gguf/transformer.rs:12 | `-` | arm/ref | BF16 F16 F32 Q2_K Q4_0 |
| gguf/transformer.rs:13 | `-` | arm/ref | Q4_1 Q4_K Q5_0 Q5_K Q6_K Q8_0 |
| gguf/transformer.rs:384 | `load_quantized_layer_moe_skeleton` | arm/ref | F32 |
| gguf/transformer.rs:448 | `k_quant_bytes` | match |  |
| gguf/transformer.rs:449 | `k_quant_bytes` | arm/ref | F32 |
| gguf/transformer.rs:457 | `k_quant_bytes` | arm/ref | BF16 F16 |
| gguf/transformer.rs:458 | `k_quant_bytes` | arm/ref | Q4_0 |
| gguf/transformer.rs:459 | `k_quant_bytes` | arm/ref | Q8_0 |
| gguf/transformer.rs:460 | `k_quant_bytes` | arm/ref | Q2_K |
| gguf/transformer.rs:461 | `k_quant_bytes` | arm/ref | Q4_1 |
| gguf/transformer.rs:462 | `k_quant_bytes` | arm/ref | Q5_0 |
| gguf/transformer.rs:463 | `k_quant_bytes` | arm/ref | Q4_K |
| gguf/transformer.rs:464 | `k_quant_bytes` | arm/ref | Q5_K |
| gguf/transformer.rs:465 | `k_quant_bytes` | arm/ref | Q6_K |
| gguf/transformer.rs:491 | `resolve_qtype` | arm/ref | Q4_0 |
| gguf/transformer.rs:498 | `resolve_qtype` | arm/ref | Q8_0 |
| gpu/adapters/wgpu_adapter.rs:279 | `raw_q4k_weights` | cmp | Q4_K |
| gpu/adapters/wgpu_adapter.rs:285 | `raw_q4k_weights` | cmp | Q4_K |
| gpu/adapters/wgpu_adapter.rs:288 | `raw_q4k_weights` | cmp | Q4_K |
| gpu/adapters/wgpu_adapter.rs:291 | `raw_q4k_weights` | cmp | Q4_K |
| gpu/adapters/wgpu_adapter.rs:309 | `dequant_tensor_public` | match |  |
| gpu/adapters/wgpu_adapter.rs:310 | `dequant_tensor_public` | arm/ref | Q4_K |
| gpu/adapters/wgpu_adapter.rs:311 | `dequant_tensor_public` | arm/ref | Q6_K |
| gpu/adapters/wgpu_adapter.rs:312 | `dequant_tensor_public` | arm/ref | Q5_K |
| gpu/adapters/wgpu_adapter.rs:313 | `dequant_tensor_public` | arm/ref | F32 |
| gpu/adapters/wgpu_adapter.rs:320 | `dequant_tensor_public` | arm/ref | F16 |

### Operator decisions executed (2026-09-17)
1. #3436–#3440 closed as duplicates of #3423 Phase 2; mapping comment on #3421. Their roadmap entries are not mirrored.
2. #3091 not moved; scope note posted. Exit fixtures (URL pinned to revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`, bytes, sha256) posted on #3429 and #3432.
3. Per-row verdict table for 0.69.0 posted on #3421 (124 open rows at read time: 11 stay, 2 stay-at-risk, 1 close-check, 110 remove-milestone). Nothing human-authored moved.
4. §1.1a accepted; #3446 makes it permanent.

### CI defects in this PR's first push, both mine, both fixed by a new commit (rerun = 0)
- `guard-tree` / `check_roadmap_fragment_required.sh`: `pmat work add` writes the monolith; entries must arrive as `docs/roadmaps/entries/<ID>.yaml`. Fixed with `roadmap_fragments.py adopt` + `aggregate --write`.
- `workspace-test` / `readme_contract::test_documented_paths_exist`: §1.1a cited a contract file that does not exist yet as a backticked path. Reworded without a path.

## Not done
Quorum ×3 on Q1, T1, T2, X1 — STOP(rail), see the gate answer above · resolution of the 558 literal-arm lines into `qtype | not-qtype` · unguarded-consumer count at function level (Table T's columns are file-level).

verdict: STOPPED(rail) — tables, review, tickets, milestone verdicts delivered; the harness refuses kind=docs, so no quorum lane ran and the PR stays draft
IMPL-PMAT-3427-RECEIPT-END
