# M32d — KV cache for the qwen3_moe inference path

**Status (2026-05-20)**: ACTIVE — **Option (b) Engineer-driven follow-up** chosen by operator. Scheduled for 1-2 week calendar delivery. See [Engineer playbook](#engineer-playbook-option-b) below.

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

## Operator decision

**CHOSEN 2026-05-20: Option (b) — Engineer-driven follow-up.** Calendar target 1-2 weeks. See [Engineer playbook](#engineer-playbook-option-b) below for the day-by-day workplan, acceptance criteria, hand-off checklist, and risk gates.

Decision rationale (for the record):

- **Option (a) Greenlight in-session** was passed over because the 8-hour focused work has numerical-equivalence risk that's hard to validate without a dedicated test fixture. Iteration cost on float-equivalence bugs (sums-of-products non-associative; subtle RoPE-position bugs) historically multiplies session time.
- **Option (c) Skip M32d** was passed over because V1_004 is on the critical path for un-suspending the CCPA project (compliance_cost_ratio measurement). Skipping leaves the meter validated but the engine unable to drive it.
- **Option (b) Engineer-driven** chosen: dedicated engineer with full dense-path context, multi-day calendar, in-repo CI/test cycles. Lower per-hour intensity but higher quality bar. Cleaner outcome.

Historical reference numbers (kept for context):
- Current state: 0% student pass on Phase 6 bench (no KV cache); meter validated but engine slow
- Post-M32d expected: 5-15 tok/s on 30B-MoE; V1_004 discharges with ~10 hour bench wall; companion-side suspension lifts

---

## Engineer playbook (Option b)

**Audience**: One engineer with familiarity with the aprender inference stack (or willing to ramp up via the dense-path reference). NOT a Claude in-session task.

**Calendar target**: 1-2 weeks (5-10 working days, depending on whether numerical-equivalence iteration adds cycles).

**Hand-off criteria**: M32d is "done" when ALL of the following are true:

1. `forward_single_qwen3_moe_with_cache` ships in `crates/aprender-serve/src/gguf/inference/forward/forward_qwen3_moe.rs`.
2. `run_qwen3_moe_generate` (in `crates/aprender-serve/src/infer/qwen3_moe_generate.rs`) uses the cache-aware path after initial prefill.
3. New cargo test `moe_kv_cache_matches_full_prefill_on_first_8_tokens` passes against a real Qwen3-MoE GGUF (env-gated, `#[ignore]` by default — mirror of `qwen3_moe_serve_dispatch_v1` from #1819).
4. Existing dense-path tests in `crates/aprender-serve/src/gguf/inference/forward/single_tests.rs` (16+ tests) still pass — no regression from Step 2's helper lift.
5. Empirical throughput on Qwen3-Coder-30B-A3B-Instruct-Q4_K_M: ≥ 5 tok/s sustained (vs ~0.5 tok/s pre-M32d).
6. Companion-side CCPA Phase 6 bench produces non-zero student pass rate when dispatched against post-M32d binary (V1_004 discharge — paiml/claude-code-parity-apr operator-coordinated).

### Day-by-day plan

**Day 1 — Ramp-up + ground truth (4-6 hours)**

- Read `crates/aprender-serve/src/gguf/runtime.rs:123` (`OwnedQuantizedKVCache` struct + tests at lines 325-450).
- Read `crates/aprender-serve/src/gguf/inference/forward/debug.rs:441-~600` (`forward_single_with_cache` — the dense reference).
- Read `crates/aprender-serve/src/gguf/inference/forward/forward_qwen3_moe.rs:69-~280` (the existing full-prefill MoE forward).
- Run the existing dense-path tests:
  ```bash
  cargo test -p aprender-serve --lib --features cuda gguf::inference::forward::single_tests
  ```
  Confirm 16+ tests pass. Baseline.
- Build + run the V1_001 test (#1819) to confirm the current MoE path produces tokens:
  ```bash
  QWEN3_MOE_GGUF_PATH=/path/to/qwen3-moe.gguf \
    cargo test --test qwen3_moe_serve_dispatch_v1 \
    -p aprender-serve --features cuda --release -- --ignored --nocapture
  ```
  Should pass in ~10s wall.
- Commit a `WIP: M32d ramp-up notes` private branch (not for review) with personal notes on the dense path's attention structure (RoPE handling, GQA expansion, fused norm+QKV).

**Day 2 — Refactor: lift attention helper from dense path (6 hours)**

- New private helper on `OwnedQuantizedModel`:
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
- Extract this from `forward_single_with_cache` (debug.rs:441). The body becomes the lifted helper; the original function reduces to: embed → loop layers calling `attention_layer_with_cache` + `ffn_block_dense` → final norm → LM head.
- **Critical invariant**: this refactor must not change ANY output of dense `forward_single_with_cache`. Verify by running `single_tests.rs` before AND after — diff must be zero failures.
- One PR for this refactor alone — keeps blast radius small.

**Day 3 — Refactor: lift MoE FFN helper from full-prefill path (4 hours)**

- New private helper on `OwnedQuantizedModel`:
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
- Extract from `forward_qwen3_moe.rs:~180-260` (the router + top-k + per-expert SwiGLU block). The body becomes the lifted helper; the original `forward_qwen3_moe` reduces to: embed → loop tokens × layers calling `attention_layer_full_prefill` (NOT cache; existing) + `moe_ffn_layer` → final norm → LM head.
- Verify forward_qwen3_moe still returns identical logits — the V1_001 cargo test (#1819) is the regression check.
- Second PR.

**Day 4-5 — New function: `forward_single_qwen3_moe_with_cache` (8-10 hours)**

- Skeleton (from scope above):
  ```rust
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
  ```
- Body:
  1. Single-token embed
  2. Optional absolute-position add
  3. Pre-allocate `attn_out_buffer`
  4. For each layer:
     - `attention_layer_with_cache(...)` (Day 2 helper) — handles QKV proj, RoPE, cache append, attention with cached K/V, attn out proj, residual
     - `moe_ffn_layer(...)` (Day 3 helper) — handles FFN norm, router, top-k expert routing, per-expert SwiGLU, residual
  5. Final norm
  6. LM head matmul → logits
- The function should be ~80-120 LOC since both helpers do the heavy lifting.

**Day 6 — Wire into `run_qwen3_moe_generate` (3-4 hours)**

- In `crates/aprender-serve/src/infer/qwen3_moe_generate.rs`:
  - Build cache: `let mut cache = OwnedQuantizedKVCache::from_config(model.config(), max_seq_len)`.
  - Prefill path: call a new `forward_qwen3_moe_with_cache_prefill` adapter that runs the full prompt through `forward_qwen3_moe` AND populates the cache layer-by-layer.
    - Simplest: have `forward_qwen3_moe` take an optional `&mut Option<&mut OwnedQuantizedKVCache>`; when Some, append K/V per layer per token during the forward pass.
    - Alternative (heavier): N sequential calls to `forward_single_qwen3_moe_with_cache`. Slower but doesn't require touching `forward_qwen3_moe` signature.
  - Decode loop: per token, call `forward_single_qwen3_moe_with_cache(token, &mut cache, position, ...)` + `cache.advance()`.
- Third PR.

**Day 7 — Tests (4-6 hours)**

- New cargo test `crates/aprender-serve/tests/moe_kv_cache_equivalence.rs`:
  ```rust
  #[test]
  #[ignore = "requires Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH"]
  fn moe_kv_cache_matches_full_prefill_on_first_8_tokens() {
      let Some(path) = std::env::var("QWEN3_MOE_GGUF_PATH").ok() else {
          eprintln!("SKIP: QWEN3_MOE_GGUF_PATH unset");
          return;
      };
      // Mirror the V1_001 test's setup pattern.
      // Generate 8 tokens twice:
      //   (a) via run_qwen3_moe_generate (cache-on, post-M32d default)
      //   (b) via legacy full-prefill loop (cache-off, pre-M32d behavior)
      // Assert greedy outputs identical token-by-token.
      // Tolerate ULP-level float drift on logits (atol=1e-3 on argmax-safe class).
  }
  ```
- Sanity check: existing V1_001 test (#1819) still passes — confirms the chat-completions wire still produces tokens after KV cache wires in.
- Perf measurement: tag a release build, dispatch a 256-token chat completion against Qwen3-Coder-30B-A3B-Instruct-Q4_K_M, log per-token wall time. Target ≥ 5 tok/s sustained.
- Fourth PR (could combine with Day 6 if tests are tight).

**Day 8-10 — V1_004 discharge dispatch (operator-coordinated; no engineer work after Day 7 PR merges)**

- Operator updates `/home/noah/.local/bin/apr` to post-M32d binary.
- Operator dispatches Phase 6 bench:
  ```bash
  APR_MODEL=/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  PHASE6_COMPLIANCE_ENFORCED=1 \
  PHASE6_MAX_TURNS=20 PHASE6_WALL_SECONDS=3600 \
  APR_TIMEOUT_S=900 APR_AGENT_HTTP_TIMEOUT_S=1500 \
  APR_AGENT_MAX_TOKENS_CAP=1024 \
  bash scripts/phase-6-bench.sh 2>&1 | tee /tmp/phase-6-30b-post-m32d.log
  ```
- Expected wall: ~10 hours. Possibly overnight.
- Acceptance: `evidence/under-contract/scores.json` shows `student_pass_rate > 0` on at least one fixture.
- Repeat with `PHASE6_COMPLIANCE_ENFORCED=0` (control mode, ~10 hr) to get the ratio.
- The pair of scores.json files lets the companion-side analyzer compute the meaningful `compliance_cost_ratio`.

### PR layout (recommended)

| PR | Title | Files touched | Lines |
|----|-------|---------------|-------|
| 1 | `refactor: lift attention helper out of forward_single_with_cache (M32d prep)` | `gguf/inference/forward/debug.rs` + new helper file | ~150 net |
| 2 | `refactor: lift moe_ffn_layer helper out of forward_qwen3_moe (M32d prep)` | `gguf/inference/forward/forward_qwen3_moe.rs` + new helper file | ~120 net |
| 3 | `feat(M32d): KV cache for qwen3_moe inference path` | new `forward_single_qwen3_moe_with_cache` + `run_qwen3_moe_generate` wire | ~200 net |
| 4 | `test(M32d): numerical-equivalence + V1_001 regression + perf measurement` | `tests/moe_kv_cache_equivalence.rs` + perf-log helper | ~150 net |

PRs 1-2 are pure refactors that should not change ANY observable behavior — they exist to keep PR 3's diff small and reviewable.

### Risk gates

Each PR must pass a gate before next PR starts:

- **After PR 1**: `cargo test -p aprender-serve --lib gguf::inference::forward::single_tests --features cuda` shows zero new failures. Dense path is byte-identical.
- **After PR 2**: V1_001 cargo test (`#1819`) passes against the real GGUF. MoE full-prefill path is byte-identical.
- **After PR 3**: New `moe_kv_cache_equivalence` test passes greedy token-equivalence over first 8 tokens. If float drift causes a token mismatch, fix RoPE position handling first (most common cause).
- **After PR 4**: Perf number ≥ 5 tok/s sustained on 30B-MoE. If lower, profile per-layer; expert routing should be <10% of per-token cost.

### Open questions for the engineer

These weren't resolved in the scope investigation; engineer should answer during Day 1 ramp-up:

1. **Prefill efficiency**: option A (modify `forward_qwen3_moe` to populate cache during prefill) vs option B (N sequential `forward_single_qwen3_moe_with_cache` calls for prefill). A is faster but touches more code. B is cleaner but slower. Recommend A if the modification is small.
2. **`forward_qwen3_moe_gpu` parity**: there's a GPU variant at `forward_qwen3_moe_gpu.rs:99`. Does it need a `_with_cache` variant too? Probably NO for this contract (V1_004 is CPU-only), but check if any caller flips to GPU after KV cache lands.
3. **Cache rollback semantics**: `OwnedQuantizedKVCache::rollback_to` exists — relevant for resampling / beam search. Not needed for V1_004 discharge (greedy decoding only) but document if the engineer encounters it.
4. **Multi-turn chat**: the chat handler treats each chat completion as a fresh session — cache is created per-request. Is there a place to reuse cache across turns? Not in V1_004 scope but useful for token cost reduction.

### Cross-team coordination

- **Reviewer for PR 1-2 (refactors)**: anyone with dense KV cache context.
- **Reviewer for PR 3 (core M32d)**: ideally someone who's touched `OwnedQuantizedKVCache` before (commit blame `runtime.rs`).
- **Reviewer for PR 4 (tests)**: low expertise needed; the equivalence test is self-checking.
- **CCPA companion side**: paiml/claude-code-parity-apr operator dispatches V1_004 discharge bench. No engineer work after PR 4 merges.

### Closing the loop

After V1_004 discharge bench succeeds:

1. Update `contracts/qwen3-moe-serve-dispatch-v1.yaml` v1.1.1 → v1.2.0 with V1_004 status: "DISCHARGED <date>".
2. Update `docs/specifications/m32d-moe-kv-cache-scope.md` (this doc): Status → "SHIPPED + V1_004 DISCHARGED <date>".
3. CCPA-side: ship a companion mechanical PR (M286 or similar) updating `evidence/phase-6/30b-moe-empirical-2026-05-19.md` with the post-M32d evidence + lifting the M280 suspension formally.
4. Optional follow-up contract: `qwen3-moe-streaming-sse-v1` for the per-token SSE delivery (one-liner Risk #6 mentioned).
