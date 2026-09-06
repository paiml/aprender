---
status: partial
ticket: PMAT-1065
row: L0-1
issue: 3017
epic: 2873
priority: P0
branch: agent/L0-1
pr: not yet opened (P0–P2 in flight; blocked_by G-10 complete; the write-set guard G-11 must land first for the queue)
model: claude-fable-5-1 (orchestrator) · N-lane root-cause quorum on agy (three model families, recorded below when returned)
tokens_used: orchestrator [U]
wall_clock_s: 3600 (basis=session clock; [U] precision)
turns: 9
---
# impl receipt — PMAT-1065 (PP-066 row L0-1, #3017, P0): Qwen2.5-1.5B cuda ≠ cpu, and apr runs cpu under --gpu

## RED first (card item 2) — recorded BEFORE any kernel edit
`evidence/parity/l0-1/lambda/RECORD.md` + the two `apr parity --json` files. Host lambda (RTX 4090), binary the 0.65.2 post-publish cuda install (`c642576eecb62daa`), 78 positions:
| model | positions < 0.98 | min cosine | at | max |Δlogit| there | verdict under the horizon rule |
|---|---|---|---|---|---|
| qwen2.5-coder-1.5b-instruct-q4_k_m | 1 | **0.9508** | 0 (token 785, the prompt's first token — not BOS) | 11.97 | **RED** |
| qwen2.5-coder-7b-instruct-q4_k_m | 0 | 0.9986 | 0 | 0.78 | GREEN |
`bash scripts/check_model_parity.sh --self-test` rows 1–2 are exactly these two records; row 3 is the must-RED twin (`tests/fixtures/parity/defective/one-position-at-0.5.json`); row 4 refuses < 64 positions (I8). The driver's numbers (0.9418 / 5.38) are not lambda's; gx10 is reached through `make fleet-verify ROW=L0-1` (G-11b) once it lands — [U] until then.

## Landed on the branch (items 1, 2, 7-part)
- `scripts/derive_model_manifest.sh` → `evidence/models/supported.yaml`: 18 models, every entry cites file:line in README/BEATS/book/dogfood receipts/perf-matrix; `--check` refuses a hand-typed entry (6-row case table).
- `scripts/check_model_parity.sh` (C14): `--manifest` runs `apr parity` per manifest model present on the host over the 78-token corpus prompt, judges min cosine over ≥ 64 positions against `evidence/parity/thresholds.yaml`; UNMEASURED reported (RED when README cites the model); `SKIP_PARITY_GATE` prints `override:` and refuses to pass (REG-15); `--judge` for a recorded run; 6-row case table.
- `evidence/parity/thresholds.yaml`: 0.98 = `PARITY_GATE_COSINE_MIN` (`crates/aprender-serve/src/gguf/cuda/mod.rs:803`), itself [U] — item 5 replaces it with n ≥ 5 per known-good pair.
- `.pr/L0-1/accept.sh` re-runs every A_i.

## What the tree already does (cited; the five-whys start here)
- The load-time gate `crates/aprender-serve/src/gguf/cuda/mod_parity_gate.rs::parity_gate` measures ONE token (BOS, position 0) — the exact position that fails — and when cosine ∈ [0.90, 0.98) retries once on `executor.force_high_precision_ffn()` (the PMAT-798 comment: the fused gate+up+SwiGLU FFN quantizes activations to Q8_1 and costs first-token cosine on massive-activation models). So the model is admitted on the unfused path while `apr parity` (fused) measures 0.9508: the retry masks the op.
- On gate failure the CLI prints `[CUDA init failed: …, falling back to CPU]` (`chat_generate_session_02.rs:471`; `run_resolve_tokenizer.rs:132`) — a forced `--gpu` silently becomes CPU: the second half of the P0.
- `SKIP_PARITY_GATE` is set SILENTLY by `commands/comparison.rs:204` and `commands/diff_benchmark_report.rs:82`; the staging contract `gpu-multi-backend-parity-v1.yaml:231` forbids it in production and nothing enforces that.

## Five whys (hypothesis; the N-lane quorum judges it — see the dispatch ledger)
1. Why is 1.5B RED? Position 0 cosine 0.9508. 2. Why position 0 only? The first token carries the massive-activation outlier Qwen2.5-1.5B is known for. 3. Why does the outlier hurt on GPU? The fused gate+up+SwiGLU path quantizes activations to Q8_1 (PMAT-798), clipping the outlier channel. 4. Why does the gate still pass? It retries on the unfused FFN for the BOS token and admits the model, but the session's generation path and `apr parity` still use the fused kernel. 5. Why does apr then run CPU under --gpu? When the gate fails outright the CLI catches the load error and downgrades silently. → OP: Q8_1 activation quantization in the fused FFN at position 0; POLICY: the gate's retry result is neither recorded nor applied; the CLI downgrade is silent.

## Dispatch ledger
| dispatch | agent | lane | width | agy conversations | families | note |
|---|---|---|---|---|---|---|
| root-cause quorum | paiml-agy-delegate `a3b9e53386bd9cf4c` (opus) | quorum, mode plan | 3 | b69448cb-aa3b-4c01-990c-bb843c0f4df2 · 0298ed4c-c0a2-4535-a1b9-26bafaefef4a · 36a93c83-32a4-4f93-b89e-f5de170f3a7b | **gemini-3.1-pro ×3 — NOT three families** (a gap: the driver requires three model families for an N-lane row; the delegate did not vary them — kaizen line, re-run with distinct families before P2) | num_turns=1 each: no lane ran a command; every citation is from the staged prompt or memory |

## Root-cause quorum: 3/3 root-cause-HYPOTHESIS (no lane could close it) — verified against the tree
| finding | lanes | verification | consequence |
|---|---|---|---|
| position 0 of `apr parity` is the prompt's first token (token_id 785), not the BOS (151643) the load-time gate measures | 3/3 | **confirmed** from the evidence JSON (`metrics[0].token_id`) | a gate PASS and a parity RED are consistent; my RECORD.md said "(BOS)" — corrected. The gate measures a token the user never sends. |
| the fused gate+up+SwiGLU (Q8_1) FFN path is OFF by default: `gpu_profile.rs:238` `auto_q4k` returns `Mwv` unconditionally (FALSIFY-Q4K-ADA-PARITY-001), `:137` derives `fused_gate_up` from it, `:815` asserts it false for Mwv | lane 1; delegate re-read the lines | **confirmed by the delegate's read**; lambda's login env carries no `FUSED_GATE_UP` (checked) | the Q8_1 hypothesis of lanes 2/3 (my five-whys step 3) is REFUTED on default config: 0.9508 was measured on the unfused path. The 1.5B/7B asymmetry on the same unfused path is UNEXPLAINED — the discriminating experiment is outstanding |
| `force_high_precision_ffn` persists for the session (`executor_api.rs:392-394` mutates the live profile and drops the decode graph) | 3/3 | cited, not re-run | the retry, when it fires, does apply to generation |
| the retry decision is printed only under `verbose()` (`mod_parity_gate.rs:92-97`); no session records which precision path served it | 3/3 | cited | the one fix every lane supports, zero throughput risk: REG-15's `selected:` line + effective-config `parity:{…}` (Q3 option a) |
| fix class | split: lane 3 kernel-only; lanes 1/2 both | — | with the fused path off by default, (b) is moot for THIS defect; (a) is the surviving policy fix; the kernel question waits for the experiment |
| the threshold | 3/3: n ≥ 5 known-good model×host pairs; 0.98 (one zero-context token) and `parity.rs:41`'s 0.95 (full-sequence) measure different things | — | item 5's measurement design |
| the driver's 0.9418 / 5.38 | 3/3 guess gx10/GB10 sm_121; none tested the non-coder variant or another prompt | — | [U]; `make fleet-verify ROW=L0-1` on gx10 when G-11b lands |

**Outstanding discriminating experiment (all three lanes name it):** a per-layer `APR_GPU_STAGE_DUMP` at position 0 (token 785) on the 1.5B, CPU vs GPU per stage, to separate attention-at-position-0 / RMSNorm / LM-head GEMV / FFN; plus the same token at position 1. Not run this session.

## Next (P2/P3, not started)
item 3 the discriminating experiment (apr parity with the unfused FFN forced — no switch exists today; a flag is part of the fix) and the fix; item 4 REG-15 admission in apr-cli (forced backend never downgrades; `selected: cpu (reason: parity FAILED …)`; effective-config `parity:{…}`; SKIP_PARITY_GATE printed as override, receipts INVALID-CORRECTNESS, asserted unset in dogfood and `ci / gate`; the two silent `set_var` sites removed); item 5 the threshold measurement; item 6 contract `apr-gpu-cpu-parity-v1.yaml`; item 7 C14 wired in `apr-dogfood --release`, C4, R-8; relabel every (1.5B, cuda) receipt INVALID-CORRECTNESS citing #3017.

## Verdict
PARTIAL — RED-first recorded on lambda; gx10 [U]; the fix not started.
