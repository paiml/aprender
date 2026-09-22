# PMAT-3751 — implementation receipt

Ticket: #3751 (0.69.1; owner routed by the cop, 2026-09-21T17:55:49Z, together with #3757).
Branch `PMAT-3751-exact-f2-references`, stacked on #3714 R1 (`01989323d`), which introduced `quantize::with_fp32_activations`.
This PR is `Refs #3751`, not `Closes`: done_when 3 has its falsifier, and the fix is the cop's ruling (below).

## done_when 1: CPU(Q8_K) vs CPU(FP32), every model, both hosts
Harness `infer/q8k_reference_drift_tests.rs` (env `APR_3751_MODELS`). Each model runs its OWN CPU forward (dense, qwen35 hybrid, qwen3moe) over `chat-france`, `chat-zorblat` (aprender-37's turn) and `plain`, once as shipped and once under `with_fp32_activations`, then 16 greedy tokens.

| host | rows | min cosine < 0.999 | greedy answer changes | lowest per-model min |
|---|---|---|---|---|
| lambda | 39 (13 files × 3) | 20 | 4 | Qwen3.5-9B 0.833554, Qwen3-30B-A3B-Instruct-2507 0.872551, Qwen3-Coder-30B-A3B 0.925310 |
| gx10 | 39 | 18 | 7 | Qwen3.5-9B 0.838791, Qwen3-30B-A3B-Instruct-2507 0.872471, Qwen3-Coder-30B-A3B 0.925414 |

The two hosts' answer-change sets are **different pairs**. Under Q8_K, the CPU answer depends on the host. Every row: `evidence/3751/q8k-vs-fp32-{lambda,gx10}.txt`.

## done_when 2: every validator site, its reference, and what changed
| # | site | reference before | reference now | basis |
|---|---|---|---|---|
| 1 | dense runtime F2 (`validate_gpu_first_token`), **serial** probe | Q8_K | **FP32** | all 24 serial rows (gx10 18, lambda 6) are at least as close to FP32. The worst position's 1 − cos is a median 2.5× smaller (1.15× to 72×), and no row rejects against either reference (`dense-f2-reference-choice-*.txt`) |
| 1 | dense runtime F2, **batched** probe | Q8_K | Q8_K (kept) | lambda qwen2.5 rows track neither reference uniformly. Two Q8_K rejects clear under FP32, and one Q8_K accept becomes an FP32 reject at pos 3. coder-0.5b (all 3 prompts) and coder-7b (both chat prompts) reject against both at pos 1 (cos 0.415–0.712 and 0.587–0.599), which are the prefill defects aprender-37 is fixing. **On aprender-37's stack** (#3727/#3728/#3759/#3785, lambda, run on this harness): with `FP8_PREFILL=0` the batched path accepts 6/6 against both, and FP32 is closer on every row (Q8_K 0.9938–0.9998, FP32 0.9998–0.99996). The two remaining default-path (FP8) REJECTs reject against BOTH references and are FP8 precision (#3785). So Batched → FP32 is the follow-up once that stack is on main |
| 2 | load-time parity gate (`mod_parity_gate.rs`, 1 token, floor 0.98) | Q8_K | Q8_K (kept) | GPU ≥ 0.999531 vs Q8_K and ≥ 0.999760 vs FP32 on all 6 files, same argmax. FP32 is closer on 5 of 6; Qwen3-8B is closer to Q8_K. Margin ~40× in 1 − cos (`parity-gate-reference-choice-lambda.txt`) |
| 3 | qwen35 hybrid F2 (`forward_qwen35.rs`) | Q8_K | **FP32** | Qwen3.5-9B, aprender-37's chat turn, lambda. Before (`5615e7afe`): REJECT at pos 23, cos 0.8338. After (`86e54afcc`): **passed on 28 positions** (`qwen35-9b-chat-f2-lambda.txt`) |
| 4 | wgpu gate (`gguf_gpu_generate.rs`) | Q8_K | Q8_K (not the cause) | rejects against BOTH references. The reference moves the cosine by ≤ 0.0066, and 4 of 6 rows fail at step 1 = position 0. A wgpu forward defect, posted on #3757 (`wgpu-reference-choice-lambda.txt`) |
| 5 | qwen3moe F2 (`qwen3_moe_dispatch.rs`) | — | FP32 (from #3714) | GPU vs FP32 CPU 1.000000 at every position |

The rejection line now names the reference: `… validated via serial prefill against the FP32-activation CPU reference`.

### Cost of the FP32 reference (dense serial F2, per run, no receipt)
| host | per-row FP32 / Q8_K | total | why |
|---|---|---|---|
| lambda | 0.94×–1.06× | flat | `fused_q4k_dot_simd` has an AVX2 FP32 path |
| gx10 | median 1.17× (0.79×–1.78×) | +24% (244.2 → 303.8 s over 18 rows, load average 26) | aarch64 takes the scalar `fused_q4k_dot`. There is no NEON FP32 path (follow-up) |

## done_when 3: the falsifier
`APR_3751_ASSERT=1` with `APR_3751_MODELS` makes `q8k_reference_drift` FAIL, naming every (model, prompt) whose greedy answer changes. Proved on lambda:
- **RED** on qwen2.5-coder-0.5b: `plain` parts at generated token 1, Q8_K `[1096, 4124, 18484, …]` vs FP32 `[1096, 3493, 572, …]`.
- **GREEN** on qwen2.5-1.5b (no divergent row).

**RULED (b)** by the cop (aprender-3e, 2026-09-22T03:14Z), on the issue body as an amendment so a quorum judges the ruling and not our messages: these pairs are the accepted Q8_K activation approximation for 0.69.1, the same trade llama.cpp makes on its Q4_K dot, and done_when 3 is met by the falsifier existing and being RED-provable on demand, not by the divergences being zero. **(a)** — a CPU default of FP32 activations — is DEFERRED, not dropped: filed as #3811, which carries this falsifier and is blocked first on a NEON FP32 Q4_K dot for aarch64 (`fused_q4k_dot_simd` has an AVX2 arm only, which is why this row's FP32 reference costs gx10 +24% and lambda nothing; x86's cost of the flip is PMAT-305's −17% decode).

## Mutants
| mutant | result |
|---|---|
| `reference_uses_fp32_activations` returns true for `Batched` instead of `Serial` | `f2_reference_precision_follows_the_probe_path` RED; restored tree GREEN, with and without `--features cuda` |
| drift falsifier on a known-divergent file | RED (above); GREEN control on a clean file |

## End to end: `apr run --gpu`, released 0.69.0 (`5615e7afe`, Q8_K dense reference) vs this branch
lambda (`ef8963016`), `--prompt` = aprender-37's Zorblat question, `--max-tokens 16 --format json`, under `gpu-q` (`e2e-apr-run-gpu-lambda.txt`). Neither binary contains #3672 (`a9502d992`), so both auto-template the prompt with the inner ChatML escaped (aprender-37, measured). The F2 probe here is that token list, NOT the harness's `chat-zorblat` sequence. This is an A/B of the two binaries on one input, and does not re-run the harness rows. **The harness rows themselves are unaffected**: they build tokens via `mapped.model.encode(text)` directly, never through `apr run`'s auto-templating, so the double-templating bug never reached them. aprender-37 confirmed this independently on main (`a9502d992` + their stack): `apr run coder-7b --chat` encodes exactly the harness's 18-token chat-france sequence and the production F2 prints the same 0.8134 cosine reject at position 1 that the harness measured. The harness rows are the production cells, value for value.

| file | probe path | 0.69.0 | this branch |
|---|---|---|---|
| Qwen3-1.7B | serial | GPU, "…Lima", 4.97 s | GPU, same text, 4.44 s |
| Qwen3-8B | serial | GPU, "Lima", 13.64 s | GPU, "Lima", 13.48 s |
| qwen2.5-1.5b | batched (unchanged) | GPU, "Lima", 7.38 s | GPU, "Lima", 5.68 s |

## Checks
- `cargo fmt --all -- --check`, `cargo deny check advisories`: clean.
- `cargo test -p aprender-contracts --lib`: 1689 passed.
- `cargo test -p aprender-serve --features cuda --lib pmat3477_f2_prefill_path_tests`: 3 passed.
