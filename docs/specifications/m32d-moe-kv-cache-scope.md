# M32d — KV cache for the qwen3_moe inference path

**Status (2026-05-19)**: SCOPE doc. Implementation deferred pending operator go/no-go.

**Cross-refs**:
- Contract gate: [`contracts/qwen3-moe-serve-dispatch-v1.yaml`](../../contracts/qwen3-moe-serve-dispatch-v1.yaml) v1.1.1 — V1_004 (CCPA Phase 6 bench non-zero student pass rate) is BLOCKED on this work.
- Empirical evidence: [paiml/claude-code-parity-apr `evidence/phase-6/30b-moe-empirical-2026-05-19.md`](https://github.com/paiml/claude-code-parity-apr/blob/main/evidence/phase-6/30b-moe-empirical-2026-05-19.md) — 5 Phase 6 dispatches all timed out at the per-turn budget.
- Predecessor PRs (this fix chain): #1806 (Option A), #1807 (Option B), #1812 (apr-cli wire + HTTP timeout env), #1814 (max_tokens cap env), #1819 (V1_001 cargo test). All MERGED.
- Cousin contract: `qwen3-moe-forward-v1` v1.2.0 (the MoE forward pass itself, dense-side reference).

## Problem statement

`run_qwen3_moe_generate` is full-prefill-per-token: every output token re-embeds and re-runs the entire prompt + previously-generated context through all 48 layers + per-expert FFN dispatch. This is O(N²) per generated token (N = current sequence length). Empirical measurement on Qwen3-Coder-30B-A3B-Instruct-Q4_K_M:

- Sustained throughput: ~0.5 tok/s on warm cache
- 256-token generation: ~9 min wall (theoretical with no growth; quadratic in practice → far worse multi-turn)
- CCPA Phase 6 bench per-turn budget: 600-2000s — insufficient for even a single turn

The dense path already has KV cache (`OwnedQuantizedKVCache` + `forward_single_with_cache`). The MoE path does NOT — `forward_qwen3_moe` is the only inference primitive for qwen3_moe, and it takes the full token sequence each call.

## Goal

Add KV-cache-aware incremental decoding for the qwen3_moe path. After this work:

- Initial prefill: existing `forward_qwen3_moe(token_ids, ...)` populates the cache for positions `0..prompt_len`
- Subsequent tokens: new `forward_single_qwen3_moe_with_cache(token_id, cache, position, moe_layers, ...)` runs ONE token through attention (using cached K/V) + ONE per-expert FFN dispatch
- Expected throughput: 5-15 tok/s on 30B-MoE (10-30× speedup), bounded by per-expert FFN compute rather than re-prefill cost

## Reference: dense path

The dense path's equivalent function is `OwnedQuantizedModel::forward_single_with_cache` at `crates/aprender-serve/src/gguf/inference/forward/debug.rs:441`:

```rust
pub fn forward_single_with_cache(
    &self,
    token_id: u32,
    cache: &mut OwnedQuantizedKVCache,
    position: usize,
) -> Result<Vec<f32>>
```

Body (615-line file; the `_with_cache` variant is lines 441–~600). Key steps:

1. Embed single token
2. (Optional) absolute position embedding
3. For each transformer layer:
   - Fused RMSNorm + QKV projection (single-token Q, K, V)
   - QKV bias add
   - RoPE on Q, K (using `position` as the offset)
   - **`cache.append(layer_idx, &k, &v)`** — append new K/V to cache
   - **`k_all = cache.get_k(layer_idx); v_all = cache.get_v(layer_idx)`** — read all K/V up to `position+1`
   - Attention: `softmax(q @ k_all^T / sqrt(d_k)) @ v_all`
   - Attention output projection
   - Residual add
   - FFN (norm + gate × up + SwiGLU + down + residual) — **this is the part that differs for MoE**
4. After all layers: `cache.advance()` — bump cache len for next token
5. Final norm + LM head matmul → logits

## KV cache API surface

From `crates/aprender-serve/src/gguf/runtime.rs:123`:

```rust
pub struct OwnedQuantizedKVCache {
    k_cache: Vec<Vec<f32>>,  // [num_layers][seq_len × kv_dim]
    v_cache: Vec<Vec<f32>>,
    len: usize,              // current seq length
    max_seq_len: usize,
    _hidden_dim: usize,      // kv_dim
}

pub fn new(num_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self;
pub fn from_config(config: &GGUFConfig, max_seq_len: usize) -> Self;
pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]);
pub fn advance(&mut self);                                // call AFTER each token
pub fn append_kv(&mut self, layer: usize, k_all: &[f32], v_all: &[f32]);  // batch variant
pub fn advance_by(&mut self, n: usize);                   // batch variant
pub fn rollback_to(&mut self, new_len: usize, kv_dim: usize);
pub fn get_k(&self, layer: usize) -> &[f32];              // all K up to len
pub fn get_v(&self, layer: usize) -> &[f32];
pub fn len(&self) -> usize;
```

No changes needed to `OwnedQuantizedKVCache` itself — its existing API is sufficient for both dense and MoE paths.

## Implementation steps

### Step 1: New forward function

Add to `crates/aprender-serve/src/gguf/inference/forward/forward_qwen3_moe.rs`:

```rust
impl OwnedQuantizedModel {
    /// Single-token MoE forward with KV cache (M32d).
    ///
    /// Mirrors `forward_single_with_cache`'s structure but replaces the
    /// dense FFN block with qwen3_moe's expert dispatch (mirrors the FFN
    /// block from `forward_qwen3_moe`).
    pub fn forward_single_qwen3_moe_with_cache(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
    ) -> Result<Vec<f32>>
}
```

### Step 2: Attention block (copy from dense path)

The attention block from `forward_single_with_cache` is reusable as-is (it doesn't depend on FFN type). Lift the per-layer attention sub-block from `debug.rs:441-~520` into a private helper:

```rust
fn attention_layer_with_cache(
    &self,
    hidden: &mut Vec<f32>,
    layer: &OwnedQuantizedLayer,
    layer_idx: usize,
    cache: &mut OwnedQuantizedKVCache,
    position: usize,
    attn_out_buffer: &mut Vec<f32>,
    use_rmsnorm: bool,
) -> Result<()>
```

Both `forward_single_with_cache` and `forward_single_qwen3_moe_with_cache` call this helper. Eliminates code duplication.

### Step 3: MoE FFN block (lift from forward_qwen3_moe)

The MoE FFN block at `forward_qwen3_moe.rs:~180-260` (router → top-k expert select → per-expert FFN) operates on a single token's hidden state. Lift into a private helper:

```rust
fn moe_ffn_layer(
    &self,
    hidden: &mut [f32],
    moe_layer: &Qwen3MoeQuantizedLayer,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate: usize,
    data: &[u8],
) -> Result<()>
```

Both `forward_qwen3_moe` (in a `for token_idx` loop) and `forward_single_qwen3_moe_with_cache` (single call) use this helper.

### Step 4: Wire into run_qwen3_moe_generate

In `crates/aprender-serve/src/infer/qwen3_moe_generate.rs:run_qwen3_moe_generate`:

```rust
// Existing: full-prefill on prompt
let mut all_tokens = input_tokens.to_vec();
let mut cache = OwnedQuantizedKVCache::from_config(model.config(), max_seq_len);
let prompt_logits = model.forward_qwen3_moe_with_cache_prefill(
    &all_tokens, moe_layers, num_experts, num_experts_per_tok, moe_intermediate, data, &mut cache,
)?;
let mut next_token = greedy_sample(&prompt_logits);
all_tokens.push(next_token);

// New: per-token incremental decode
while !stop && all_tokens.len() < max_seq_len && all_tokens.len() - input_tokens.len() < max_tokens {
    let pos = all_tokens.len() - 1;
    let logits = model.forward_single_qwen3_moe_with_cache(
        next_token, &mut cache, pos,
        moe_layers, num_experts, num_experts_per_tok, moe_intermediate, data,
    )?;
    next_token = greedy_sample(&logits);
    all_tokens.push(next_token);
    cache.advance();  // bump cache len; required for next iteration's get_k/get_v slices
}
```

`forward_qwen3_moe_with_cache_prefill` is a small adapter that runs `forward_qwen3_moe` on the full prompt AND populates the cache layer-by-layer in the same pass. Could also be implemented as N sequential single-token calls but that defeats the purpose of prefill.

### Step 5: Tests

5a. **Numerical equivalence test** (cargo test, `#[ignore]`'d, env-gated):

```rust
#[test]
#[ignore = "requires Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH"]
fn moe_kv_cache_matches_full_prefill_on_first_8_tokens() {
    // Generate 8 tokens via:
    //   (a) run_qwen3_moe_generate WITH cache (post-M32d default)
    //   (b) run_qwen3_moe_generate WITHOUT cache (legacy full-prefill-per-token)
    // Assert greedy outputs identical token-by-token.
    // Tolerate ULP-level float differences on logits (atol=1e-3 on argmax-safe class).
}
```

5b. **Token-rate measurement** (operator-dispatched bench):

```bash
# Pre-M32d:
time apr serve run qwen3-moe.gguf  # → manual chat-completion → ~0.5 tok/s

# Post-M32d:
time apr serve run qwen3-moe.gguf  # → manual chat-completion → expect 5-15 tok/s
```

5c. **CCPA Phase 6 V1_004 discharge** (operator-dispatched):

```bash
APR_MODEL=/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
PHASE6_COMPLIANCE_ENFORCED=1 \
PHASE6_MAX_TURNS=20 PHASE6_WALL_SECONDS=3600 \
APR_TIMEOUT_S=900 APR_AGENT_HTTP_TIMEOUT_S=1500 \
APR_AGENT_MAX_TOKENS_CAP=1024 \
bash scripts/phase-6-bench.sh
```

Expected: student_pass_rate > 0 on at least some fixtures. Total wall: ~10 hours.

## Effort estimate

| Step | Time | Risk |
|---|---|---|
| 1. Function skeleton | 30 min | low |
| 2. Lift attention helper (refactor dense path) | 2 hr | medium (must not regress dense KV cache) |
| 3. Lift MoE FFN helper (refactor full-prefill MoE path) | 2 hr | medium (per-expert routing logic) |
| 4. Wire run_qwen3_moe_generate | 1 hr | low |
| 5a. Numerical equivalence test | 2 hr | high (float-equivalence is hard) |
| 5b. Perf measurement | 30 min | low |

**Total: 8 hours focused engineering** + buffer for unexpected issues. Realistically a 1-2 day deliverable for a focused engineer with full context on the dense KV cache path.

## Risk surface

1. **Numerical equivalence**: KV cache vs full-prefill compute attention in different orders. Sums-of-products on float32 are non-associative. The greedy-sample test (5a) is the right check but may fail at temperature > 0 with non-greedy sampling.
2. **Dense path regression**: Step 2 refactors `forward_single_with_cache` to call a new helper. Must preserve byte-identical behavior on the dense path. Mitigated by existing `forward/single_tests.rs` battery (16+ tests).
3. **RoPE position offset**: cache holds K's AFTER RoPE has been applied. Computing fresh Q for new token needs the SAME RoPE base + freq scheme. Easy to misplace.
4. **GQA expansion**: Qwen3's grouped query attention has `num_kv_heads < num_heads`. K/V are smaller than Q. Cache shape matches `num_kv_heads * head_dim`. Already handled by `from_config`. Verify with a Qwen3-Coder-specific test.
5. **Expert routing under cache**: NONE — expert routing reads from current hidden state only; doesn't depend on cache state. Step 3 lift is purely mechanical.
6. **Streaming SSE for free**: per-token incremental forward exposes a natural emit-per-token point. The `apr serve` chat handler can pipe these into an SSE response. Out-of-scope for THIS contract but listed as the natural follow-up (one-line addition once KV cache lands).

## What this is NOT

- NOT a perf-tuning exercise. The 5-15 tok/s target is the no-tuning baseline. Further wins from batched matmul, expert kernel fusion, etc. are separate work.
- NOT a streaming SSE delivery (see Risk #6).
- NOT a GPU acceleration (see `qwen3-moe-forward-gpu-v1` contract + M-GPU-MOE-2.x). CPU is the floor.

## Operator decision required

Choose ONE:

- **(a) Greenlight in-session implementation**: 8-hour focused work; Claude attempts steps 1-5a; ships as 1-2 PRs depending on size. Risk: numerical equivalence test may not pass cleanly on first try; iteration cycles add 2-4 hours.
- **(b) Schedule for engineer-driven follow-up**: defer to a focused engineering session with full dense-path context. Likely 1-2 day deliverable. No risk to existing dense KV cache.
- **(c) Skip M32d and accept V1_004 stays blocked**: rely on smaller MoE student models (7B-13B coder GGUFs, if available) or alternate measurement strategies. V1_004 contract row stays open indefinitely.

Reference numbers for decision:
- Current state: 0% student pass on Phase 6 bench (no KV cache); meter validated but engine slow
- (a) outcome if successful: V1_004 discharges with ~10 hour bench wall; companion-side suspension lifts
- (b) outcome: same as (a) but cleaner timeline; ~1-2 weeks calendar
- (c) outcome: V1_004 stays open; project-level milestone (compliance_cost_ratio measurement) waits for engine improvements outside this contract
