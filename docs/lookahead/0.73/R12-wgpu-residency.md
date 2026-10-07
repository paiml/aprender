# Row 12: wgpu device residency (attention, RoPE, LM head, argmax)

Read at origin/main 11f844a772. All `W:` cites are
`crates/aprender-compute/src/backends/gpu/device/linalg/wgsl_forward.rs`. No build.

## Host round trips today, per decode token
`WgslForwardPass::forward_model` (W:754) and `forward_layer` (W:819):
| step | where | cite |
|---|---|---|
| embedding lookup | host | W:768 ("Embedding lookup (CPU)") |
| hidden upload | host to device, each layer | W:830 `write_buffer(hidden_buf)` |
| Q, K, V projections | device | W:877 |
| Q, K, V read back | 3 `map_async` per layer | W:910, W:925, W:940 |
| RoPE | host, f64 loop | W:968-1000 |
| KV cache append | host `Vec<f32>` | W:1010-1011 |
| attention scores, softmax, weighted V | host | W:1025, W:1036 (`attention_scores`, `attention_weighted_v` at W:1157, W:1193) |
| attention out upload | host to device | W:1046 |
| O proj, residual, norm, FFN | device | to W:1130 |
| layer out read back | 1 `map_async` | W:1137 |
| output RMSNorm | host | W:790 |
| LM head | host matmul, `vocab x hidden` f32 | W:796-815 |
| argmax / sampling | host | T5-T7 in `E4-census.md` |
So per token: 4L read backs and 2L uploads for L layers, plus the whole LM head on the
host. For a 28-layer model that is 112 synchronous `map_async` waits a token.

## What the tree already has
- RoPE: `rope_shader()` (W:304) and `encode_batch_rope` (W:1818), used by the training
  path only.
- Attention: `CAUSAL_ATTENTION_SHADER` (`shaders/advanced.rs:18`) and
  `encode_attention` (W:1859), training path only, and only over its own `seq_len`
  activations, with no KV cache.
- LM head: blocked, per its own comment, by "vocab > 65535 dispatch limit" (W:799) and
  the [N,K] layout (PMAT-346, PMAT-751). The 2D dispatch the fix needs is already in
  tree: `device/backward.rs:1001-1006` splits `total_wg` as `X = min(n, 65535)`,
  `Y = ceil(n / 65535)`.
- Argmax: no WGSL argmax or max-reduce shader in `shaders/` (`reductions.rs` has
  rmsnorm, silu-mul, residual and rope).

## Finding R12-1: RoPE theta is hardcoded to Qwen2's value on the inference path
`let rope_theta = 1_000_000.0f64; // Qwen2 rope_theta` (W:970). There is no setter;
`rms_norm_eps` got one for the same bug class (#4056, W:113, W:570), theta did not.
It is reached by `apr serve --backend wgpu` (`apr-cli/src/commands/serve/handlers.rs:140`,
`:545`, which set only eps at `:771`) and by batch inference
(`aprender-serve/src/infer/batch_wgpu.rs:74`, `:90`, eps only at `:161`). The family
contracts declare other values: `llama.yaml:19` 500000.0, `mistral.yaml:19`,
`gemma.yaml:20`, `deepseek.yaml:19` 10000.0 (`crates/apr-cli/contracts/model-families/`).
With E4 N2 (serve does no architecture check), a Llama or Mistral model served on wgpu
gets the wrong rotation with no error. E1 would catch it as a cosine FAIL on C1-C3 for
any non-Qwen2 model, and it would read as a wgpu bug, not a config bug.
Fix, ahead of row 12 and small: `set_rope_theta` mirroring `set_rms_norm_eps`, set at
the three call sites from the model config; carry theta into the device RoPE too.
Falsifier FALSIFY-R12-THETA-001: a wgpu forward with theta 1e4 and one with 1e6 on the
same input must give different Q after RoPE, and the wgpu Q must match the CPU forward
for a Llama-config model. Whether `gguf_gpu_generate` (T5) reaches W:970 is [U].

## Draft order for the row-12 ticket (each step one PR, each keeps E1 parity)
1. R12-1 theta setter (correctness, before any speed work).
2. RoPE on device: reuse `rope_shader()` for M=1 decode with `position` as a uniform.
   Saves the Q and K read backs only if step 3 lands with it; alone it saves nothing.
3. Device KV cache plus decode attention: a per-layer KV buffer (`init_kv_cache` at
   W:621 exists, read what it allocates [U]) and an M=1 attention kernel, or
   `CAUSAL_ATTENTION_SHADER` with a KV-length uniform. Removes 3L read backs and L
   uploads. This is the step that ends HYBRID for the layer loop.
4. LM head on device: `upload_weight_transposed()` (PMAT-751) plus the 2D dispatch
   from `backward.rs:1001-1006`. Output RMSNorm moves with it.
5. Argmax on device: a two-pass max-reduce (per-workgroup max and index, then one
   workgroup), so a greedy token reads back 4 bytes, not `vocab x 4`. Sampling (T8,
   row 17) stays on the host and reads back the logits only when it needs them.
   Device argmax is for the greedy route only; a sampled route keeps the logits
   read back (R17-2 in `R17-wgpu-sampling.md`), and row 17 lands first.
6. Embedding on device: a gather from the uploaded table, last, since it is one row.

## Falsifiers
| id | claim | how |
|---|---|---|
| F12-1 | after step 3, a decode token does at most 1 read back (the token id, after 5) | count `map_async` per token in a trace build; want <= 1 with 4 and 5 landed |
| F12-2 | each step keeps E1 cosine >= 0.995 vs apr CPU fp32_act | E1 receipt per step |
| F12-3 | the 2D LM head dispatch covers every vocab row | vocab 151936 (Qwen2): all logits equal the CPU LM head |
| F12-4 | device argmax equals host argmax, ties to the lowest index | planted tie at 0 and vocab-1 |
| F12-5 | RQ-4: a cell reports GPU only when steps 3-5 have landed for its route | E6 join reads the residency field |

## Open
- Which routes (T5 generate, T6 serve, T7 batch) share `forward_layer`: all three. T5
  reaches it from both generate routes and both probes (R10-4: gguf_gpu_generate.rs:282, :337, :780, :840).
- `init_kv_cache` (W:621): what it allocates and whether any route uses it [U].
