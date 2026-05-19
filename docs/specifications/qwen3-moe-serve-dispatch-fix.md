# Qwen3-MoE serve-dispatch fix (paiml/aprender#1789)

**Status (2026-05-19)**: SCOPE + CONTRACT in place; implementation in flight on branch `fix/1789-qwen3-moe-serve-dispatch`.

**Cross-refs**:
- Contract: [`contracts/qwen3-moe-serve-dispatch-v1.yaml`](../../contracts/qwen3-moe-serve-dispatch-v1.yaml)
- Companion issue: [paiml/claude-code-parity-apr M280](https://github.com/paiml/claude-code-parity-apr/issues/280) (CCPA suspension pending this fix)
- Shallow guard PR: [paiml/aprender#1790](https://github.com/paiml/aprender/pull/1790) — defensive matmul guard (MERGED 2026-05-18)
- Companion-side contract bump: [paiml/aprender#1794](https://github.com/paiml/aprender/pull/1794) — claude-code-parity-apr-v1.yaml v1.32.0 with CCPA-019 + CCPA-020 (MERGED)

## TL;DR

`apr serve`'s `/v1/chat/completions` handler at `crates/aprender-serve/src/api/cuda_chat_backend.rs:361` calls `Arc<Model>::generate()`. For Qwen3-MoE GGUFs, this goes through the **dense FFN matmul path** (which expects `ffn_up.weight` populated), not the **per-expert MoE path** (which uses `ffn_up_exps.weight` indexed by router output). The fix already exists at `crates/aprender-serve/src/infer/inference_result.rs:225` — but it's only wired into the **`apr run` CLI**, not the **`apr serve` HTTP API**.

`apr code` (used by claude-code-parity-apr Phase 6) dispatches through HTTP, so it hits the bug.

## Root cause (5-whys)

1. **Why does `apr code` produce 0/20 student pass against Qwen3-Coder-30B-MoE under CCPA Phase 6?** Because the HTTP chat-completions handler panics on the first inference call. Now (post-#1790) it returns `RealizarError::InvalidShape` cleanly instead of panicking, but still produces no inference output.
2. **Why does it panic / error?** Because `fused_matmul_f32` is called with a weight tensor whose `data.len() == 0`. The defensive guard at #1790 catches this and returns an actionable error, but the underlying dispatch is wrong.
3. **Why is the weight data empty?** Because the Qwen3-MoE GGUF stores FFN weights per-expert (under `blk.{L}.ffn_up_exps.weight`, shape `[num_experts, intermediate, hidden]`), not under the dense names (`blk.{L}.ffn_up.weight`). The dense names exist in the model index but reference zero-byte slices.
4. **Why is the dense path being taken at all?** Because `cuda_chat_backend.rs::openai_chat_completions_handler` calls `Arc<Model>::generate()` unconditionally, which calls `Model::forward()` (dense), which calls the dense FFN matmul on `ffn_up.weight`.
5. **Why doesn't the handler route to the existing MoE path?** Because `infer::run_inference` (where the MoE branching at `inference_result.rs:225` exists) is invoked only by the CLI binary (`apr run`), not by the HTTP handler. The HTTP handler builds its own dispatch on top of `Arc<Model>`, bypassing the MoE-aware routing entirely.

**Root cause** (level 5): The HTTP chat-completions handler and the CLI inference entry point have **two separate dispatch trees**. The CLI tree is MoE-aware; the HTTP tree is not. Both call into the same low-level GGUF tensor stores, but only one of them knows about per-expert dispatch.

## Affected code paths

### MoE-aware path (CORRECT, exists, used by `apr run`)

```
src/bin/apr.rs::main()
  → infer::run_inference(model_path, prompt, config)
    crates/aprender-serve/src/infer/inference_result.rs:225
      if canonical_arch == "qwen3_moe" {
          run_qwen3_moe_generate(&mapped, &model, &input_tokens, &gen_config)
            crates/aprender-serve/src/infer/qwen3_moe_generate.rs:56
              forward_qwen3_moe(token_ids, moe_layers, num_experts,
                                num_experts_per_tok, moe_intermediate, data)
                crates/aprender-serve/src/gguf/inference/forward/forward_qwen3_moe.rs:69
      } else {
          run_gguf_generate(...)  // dense
      }
```

### Dense-only path (BUG, used by `apr serve` chat-completions)

```
HTTP POST /v1/chat/completions
  → cuda_chat_backend.rs::openai_chat_completions_handler()
    crates/aprender-serve/src/api/cuda_chat_backend.rs:361
      let generated = match model.generate(&prompt, &config)  // ← Arc<Model>::generate
        crates/aprender-serve/src/layers/model_model.rs:130
          self.forward(...)  // dense FFN, expects ffn_up.weight populated
            → fused_matmul_f32(..., ffn_up_weight, ...)
              data.len() == 0  (for MoE)
              → RealizarError::InvalidShape (post-#1790 guard)
```

## Three dispatch options (engineering trade-off)

### Option A: Detection + clear-error short-circuit (1-2 hours)

**Smallest viable step.** In `cuda_chat_backend.rs::openai_chat_completions_handler`, BEFORE calling `Arc<Model>::generate()`, check if the model's canonical architecture is `qwen3_moe`. If yes, return `(StatusCode::NOT_IMPLEMENTED, Json(error))` with body:

```json
{
  "error": {
    "type": "moe_dispatch_not_implemented",
    "message": "qwen3_moe-arch GGUFs are not yet supported via /v1/chat/completions. Use `apr run` CLI for MoE inference, or wait for aprender#1789 phase-2 wire-up. See contracts/qwen3-moe-serve-dispatch-v1.yaml.",
    "issue": "https://github.com/paiml/aprender/issues/1789"
  }
}
```

**Pros**: ships in 1-2 hours; eliminates cryptic matmul errors at the API surface; gives operators a clear signal. Discharges FALSIFY-QWEN3_MOE_SERVE_DISPATCH_V1_002.

**Cons**: doesn't actually enable MoE inference. CCPA Phase 6 still gets 0/20 student pass, but with a clean error class instead of a noisy one. Doesn't discharge V1_001 / V1_003 / V1_004.

### Option B: Wire `run_inference` into chat handler (1-2 days)

**Mid-effort.** Refactor `cuda_chat_backend.rs` to detect GGUF-backed models (those with a `MappedGGUFModel` in `AppState`) + dispatch through `infer::run_inference` instead of `Arc<Model>::generate()`. The MoE branching at `inference_result.rs:225` is reused as-is. Requires:

1. Adding `Arc<MappedGGUFModel>` (or equivalent) to the chat handler's `AppState`-accessible context (the GGUF state is currently held server-wide but not threaded into per-request dispatch).
2. Adapting `run_inference` to be callable in a streaming context (it currently returns `Result<Vec<f32>>` for the final logits; chat completions need a token-by-token interface). The CLI version uses an autoregressive loop internally; expose that loop's per-token callback.
3. Mapping the `GenerationConfig` from the chat request schema (`max_tokens`, `temperature`, `top_p`, `top_k`, etc.) into the `infer::GenerationConfig` struct.

**Pros**: actual MoE inference works via HTTP. Discharges V1_001 + V1_003. Re-uses tested CLI code.

**Cons**: 1-2 days of careful surgery to unify the two dispatch trees without breaking the dense path. Risk of regressing existing chat-completions tests.

### Option C: Make `Arc<Model>::generate` MoE-aware (medium-large refactor)

**Highest-effort, most-correct.** Push the MoE branching down into `Model::generate()` / `Model::forward()` so both CLI and HTTP paths "just work" without HTTP knowing about GGUF specifics. Requires the `Model` abstraction to hold optional `Qwen3MoeQuantizedLayer` references + conditionally dispatch in `forward()`.

**Pros**: cleanest separation; matches the long-term shape (one model abstraction, dispatch internal).

**Cons**: largest blast radius — touches the `Model` trait used by every inference call site. Risk of subtle behavior changes in non-MoE paths. Probably 3-5 days including tests.

## Recommended dispatch

**Ship Option A in this PR.** Defer Option B to a follow-up PR after Option A merges + CCPA companion-side validates the clean error class via Phase 6 bench against MoE. Option C is a future architectural cleanup.

Rationale: Option A is *strictly* better than the current state (no panic, clear error, contract-discharging), ships fast, and unblocks the CCPA suspension's "we can at least classify the failure mode cleanly" requirement. Option B is the proper fix but needs a separate review cycle because it touches the dispatch architecture.

## Implementation plan (Option A)

### Step 1: Detect MoE architecture in chat handler

In `crates/aprender-serve/src/api/cuda_chat_backend.rs`, before line 361 (the `model.generate(&prompt, &config)` call), insert architecture detection:

```rust
// Detect qwen3_moe arch before dispatching to dense Model::generate.
// See aprender#1789 + contracts/qwen3-moe-serve-dispatch-v1.yaml.
if let Some(gguf_state) = state.gguf_state.as_ref() {
    if let Ok(arch) = gguf_state.canonical_arch() {
        if arch == "qwen3_moe" {
            tracing::warn!(
                "qwen3_moe arch detected at /v1/chat/completions; \
                 MoE dispatch via HTTP not yet implemented (aprender#1789). \
                 Returning NOT_IMPLEMENTED."
            );
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(ChatCompletionError {
                    error: ErrorDetail {
                        kind: "moe_dispatch_not_implemented".into(),
                        message: "qwen3_moe-arch GGUFs are not yet supported \
                                  via /v1/chat/completions. Use `apr run` CLI \
                                  for MoE inference, or wait for aprender#1789 \
                                  phase-2 wire-up.".into(),
                        issue: Some(
                            "https://github.com/paiml/aprender/issues/1789".into()
                        ),
                    },
                }),
            ).into_response();
        }
    }
}
```

### Step 2: Add `ChatCompletionError` + `ErrorDetail` types

If not already present, add structured error types matching the OpenAI API error schema.

### Step 3: Unit test in `cuda_chat_backend.rs::tests`

```rust
#[tokio::test]
async fn qwen3_moe_returns_not_implemented() {
    let state = mock_app_state_with_moe_arch();
    let request = mock_chat_request("hello");
    let response = openai_chat_completions_handler(State(state), Json(request)).await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body = extract_error_body(response).await;
    assert_eq!(body.error.kind, "moe_dispatch_not_implemented");
}
```

### Step 4: Integration test (deferred to Option B)

A real Qwen3-MoE GGUF integration test requires fixture infrastructure that's heavier than this PR should carry. Defer to Option B.

### Step 5: Run the falsifier locally

```bash
cargo test -p aprender-serve --lib api::cuda_chat_backend::tests::qwen3_moe_returns_not_implemented
```

## Companion-side integration (post-merge)

After this PR merges, the companion (claude-code-parity-apr) can:

1. Re-dispatch Phase 6 against Qwen3-Coder-30B-MoE with the same fixture set.
2. Confirm that `evidence/under-contract/scores.json` now shows the `moe_dispatch_not_implemented` driver-error class instead of the previous opaque panic / `InvalidShape` class.
3. Update `evidence/phase-6/1.5b-calibration-run.md` with the new evidence class.
4. Stay suspended pending Option B (which is what actually un-blocks meaningful CCPA measurement).

## Open questions

- Q1: Does `AppState` have a clean way to expose the loaded GGUF's `canonical_arch`? Need to check `crates/aprender-serve/src/api/server_state.rs`.
- Q2: Is the chat-completions handler's response type already an `impl IntoResponse`, or does it need to be widened to support both success + error variants?
- Q3: Does an `ErrorDetail` / `ChatCompletionError` type already exist in `cuda_chat_backend.rs`? Reuse if so.

Answers to Q1-Q3 will be resolved during Step 1-2 implementation.
