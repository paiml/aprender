# Flash Attention

> **Status**: Implemented
>
> See: `src/cuda/executor/layer.rs`, `src/cuda/executor/kv_cache.rs`

## Overview

Realizar implements FlashAttention via trueno-gpu's `AttentionKernel`, achieving O(N) memory complexity instead of the standard O(N^2) materialized attention matrix. The implementation runs on CUDA with automatic CPU fallback for edge cases.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   flash_attention()                        │
│  src/cuda/executor/layer.rs                                │
├───────────────────────────────────────────────────────────┤
│  1. Pre-launch validation (shared memory bounds)           │
│  2. trueno-gpu AttentionKernel PTX generation              │
│  3. GPU buffer allocation (Q, K, V, output)                │
│  4. Kernel launch with tiled execution                     │
│  5. Output transfer back to host                           │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│               flash_attention_cached()                     │
│  src/cuda/executor/kv_cache.rs                             │
├───────────────────────────────────────────────────────────┤
│  if seq_len >= head_dim:                                   │
│      → GPU flash attention (tiled, O(N) memory)            │
│  else:                                                     │
│      → CPU scaled dot-product attention (fallback)         │
└───────────────────────────────────────────────────────────┘
```

## Pre-Launch Validation

The trueno-gpu `AttentionKernel` uses shared memory for K tile loading. The kernel accesses `K[col * head_dim + dot_idx]`, requiring `head_dim^2` elements in shared memory, but only `tile_q * head_dim` elements are allocated. When `seq_len < head_dim`, `tile_q < head_dim`, causing out-of-bounds shared memory access.

Realizar validates this before launch:

```rust
// Shared memory overflow check
let tile_q = seq_len.min(TILE_SIZE);
let required = tile_q * head_dim;
let accessed = head_dim * head_dim; // K dot-loop worst case

if accessed > required {
    return Err(GpuError::InvalidParameter(format!(
        "flash_attention: shared memory overflow — \
         tile_q={tile_q} < head_dim={head_dim}"
    )));
}
```

## CPU Fallback

When `seq_len < head_dim` (common for the first token in autoregressive generation), the implementation automatically falls back to CPU-based scaled dot-product attention:

```rust
// CPU fallback: standard causal attention per head
let scale = 1.0 / (head_dim as f32).sqrt();
for head in 0..num_heads {
    for row in 0..seq_len {
        // Compute attention scores with causal mask
        let mut scores = vec![f32::NEG_INFINITY; seq_len];
        for col in 0..=row {
            scores[col] = dot(Q[row], K[col]) * scale;
        }
        // Softmax + weighted sum of V
        let weights = softmax(&scores);
        output[row] = weighted_sum(&weights, &V);
    }
}
```

This fallback is numerically equivalent to the GPU kernel and only activates for the first few tokens until `seq_len >= head_dim`.

## Memory Complexity

| Approach | Memory | Notes |
|----------|--------|-------|
| Standard attention | O(N^2) | Materializes full NxN attention matrix |
| Flash attention | O(N) | Tiled computation, no materialization |
| Reduction | 32x at seq_len=512 | Measured in `imp_801_flash_attention_falsification` |

## CUDA Context Safety

Flash attention kernel launches include null device pointer validation to prevent GPU context poisoning:

```rust
validate_device_ptr(q_buf.device_ptr(), "flash_attention Q")?;
validate_device_ptr(k_buf.device_ptr(), "flash_attention K")?;
validate_device_ptr(v_buf.device_ptr(), "flash_attention V")?;
validate_device_ptr(out_buf.device_ptr(), "flash_attention output")?;
```

If any pointer is null (0x0), the kernel launch is refused with a descriptive error instead of crashing the GPU.

## Test Coverage

- **988 CUDA executor tests** pass including flash attention variants
- **5 cached flash attention tests** with `head_dim=8` exercise the CPU fallback path
- **Parity tests** verify GPU output matches CPU reference within 1e-4 tolerance

## Examples

```bash
# FlashAttention O(N) memory verification
cargo run --example parity_039_flash_attention --features cuda

# Falsification of FlashAttention claims
cargo run --example imp_801_flash_attention_falsification --features cuda
```

## See Also

- [Memory Management](../gpu/memory-management.md) - GPU buffer pooling
- [CUDA PTX Generation](../gpu/cuda-ptx.md) - How trueno-gpu generates PTX
- [KV Cache Management](../transformer/kv-cache.md) - Cached attention integration
