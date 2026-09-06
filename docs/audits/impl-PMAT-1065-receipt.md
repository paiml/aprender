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
| qwen2.5-coder-1.5b-instruct-q4_k_m | 1 | **0.9508** | 0 (BOS) | 11.97 | **RED** |
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
| dispatch | agent | lane | width | note |
|---|---|---|---|---|
| root-cause quorum | paiml-agy-delegate (opus) | quorum, mode plan | 3 (three model families recorded in the lane files) | appended when returned |

## Next (P2/P3, not started)
item 3 the discriminating experiment (apr parity with the unfused FFN forced — no switch exists today; a flag is part of the fix) and the fix; item 4 REG-15 admission in apr-cli (forced backend never downgrades; `selected: cpu (reason: parity FAILED …)`; effective-config `parity:{…}`; SKIP_PARITY_GATE printed as override, receipts INVALID-CORRECTNESS, asserted unset in dogfood and `ci / gate`; the two silent `set_var` sites removed); item 5 the threshold measurement; item 6 contract `apr-gpu-cpu-parity-v1.yaml`; item 7 C14 wired in `apr-dogfood --release`, C4, R-8; relabel every (1.5B, cuda) receipt INVALID-CORRECTNESS citing #3017.

## Verdict
PARTIAL — RED-first recorded on lambda; gx10 [U]; the fix not started.
