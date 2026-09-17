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
- minted: epic #3428; M3 #3429 · M1 #3430 · M2 #3431 · M4 #3432 · T1 #3433 · T2 #3434 (0.69.0); T3 #3435 · Q2 #3436–#3440 · Q3 #3441 · Q5 #3442 · P4 #3443 (0.70.0); S1 #3445 · §1.5 landing #3446 (0.69.0). #3444 closed as a duplicate of #3422.
- `Ĵ` on every ticket is 24 turns `[A]`: `estimate.sh aprender` reports 47 measured ledger rows and pools none — a harness finding, not a measurement.

## Row J — 0.69.0 hygiene `[V]` (`gh api graphql`, milestone 6, open issues)
124 open: 109 no linked PR · 10 minted this run · 3 epics · 1 open PR (#3204→#3205) · 1 linked PR merged with the issue still open (#2753→#2809). Authors: noahgift 118, alfredodeza 5, guyernest 1. `updatedAt` is ≤ 7 days on every row, so it measures triage edits, not work. Only tickets minted by this run were placed; the rest is one HRQ comment on #3421. The target of ≤ 15 open at T-0 is not reachable by this run's authority.

## HRQ rows
1. Placement conflict: #3423 puts PP-QUANT Phase 2/3 on a dedicated non-cadence milestone; brief v4 makes slice M 0.69.0 must-carry. #3436–#3440 restate Phase 2 and fold into whichever placement stands.
2. #3091's milestone (open in shipped 0.68.0) — human-authored ticket, not moved.
3. 109 no-PR rows in 0.69.0 one day before its due date.
4. Train rule ("dates never slip for scope", 12 h hold) vs must-carry ("T-0 waits", 72 h andon) — §1.1a records the exception; APR-RELEASE-001 has no §1.5 on `main` (#3446).

## Not done
Quorum ×3 and the T1–X2 verdict grid (v3 missing; harness rail refused `kind=docs`) · per-call-site (file:line) Table G — delivered per file · tables T/E/B (defined only in v3).

verdict: PARTIAL(escalate) — research, review, tickets and milestone delivered; quorum grid not run
IMPL-PMAT-3427-RECEIPT-END
