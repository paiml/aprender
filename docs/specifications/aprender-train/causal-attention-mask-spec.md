# Causal Attention Mask Specification

## Status: CRITICAL BUG — entrenar#310

**Root cause #2 of convergence gap (combined with entrenar#309 init fix).**

## 1. Problem Statement

`CudaTransformerBlock::compute_attention_cuda()` in `cuda_block.rs` applies
row-wise softmax to the full `[num_heads, seq, seq]` attention score matrix
without a causal mask. Every token attends to every other token, including
future tokens.

```rust
// cuda_block.rs:855-875 — NO CAUSAL MASK
batched_softmax_forward(
    &scores_view,
    &mut self.scratch.attn_scores,
    total_rows,  // num_heads * seq
    seq,         // full row width — includes future positions
    stream,
)?;
```

For autoregressive (causal LM) training, position `i` must only attend to
positions `j <= i`. Without this mask, the model learns bidirectional
representations that are useless for left-to-right text generation.

## 2. Evidence

### 2.1 Albor PyTorch Canary (with init fix applied)

| Step | PyTorch (causal) | Entrenar v17 (no causal) | Gap |
|------|-----------------|------------------------|-----|
| 0 | loss=10.586 | loss=10.593 | 0.07% (parity) |
| 1K | val_ppl=50.0 | val_ppl=729.4 | **14.6x** |

Forward pass is correct (step-0 parity). Backward pass diverges because
gradients teach the model to use future context that won't be available
at inference time.

### 2.2 Gradient Norm Comparison

PyTorch raw gnorm (batch=1): 7.41
Entrenar gnorm (batch=4, GA=8): 0.654

After accounting for GA scaling: ratio ≈ 1.4x — consistent with
bidirectional attention producing different gradient magnitudes than causal.

## 3. Literature

### 3.1 Original Transformer

[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — "Attention Is All
You Need" §3.1:

> "We need to prevent leftward information flow in the decoder to preserve the
> auto-regressive property. We implement this [...] by masking out (setting to
> −∞) all values in the input of the softmax which correspond to illegal
> connections."

The causal mask is a **lower triangular matrix**:
```
M = [[0,   -∞,  -∞,  -∞],
     [0,    0,  -∞,  -∞],
     [0,    0,   0,  -∞],
     [0,    0,   0,   0]]
```

### 3.2 FlashAttention

[Dao et al. (2022)](https://arxiv.org/abs/2205.14135) — FlashAttention fuses
the causal mask with tiled softmax computation. Masked tiles (upper triangle)
are skipped entirely, providing ~2x speedup. The causal flag is a single
boolean parameter in the kernel launch.

### 3.3 PyTorch Implementation

[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html):

```python
# When is_causal=True:
attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
attn_weight[~attn_mask] = float("-inf")
```

HuggingFace LlamaAttention uses `is_causal=True` by default.

### 3.4 trueno Existing Support

trueno already has causal attention kernels:
- `AttentionKernel::new(max_seq, head_dim).with_causal()` → `"flash_attention_causal"`
- Located in `trueno-gpu/src/kernels/attention/flash/mod.rs`
- Includes `causal: bool` field and tiled implementation

These kernels are NOT used by entrenar's `CudaTransformerBlock`.

## 4. Specification

### 4.1 Causal Mask (Minimum Viable Fix)

Before softmax in `compute_attention_cuda()`:

```
For each head h, position i, key position j:
    if j > i:
        scores[h * seq + i, j] = -inf  (f32::NEG_INFINITY)
```

This can be implemented as:
1. **Separate kernel** (simplest): one kernel pass to apply mask, then existing softmax
2. **Fused causal softmax** (optimal): modified softmax kernel that skips `j > i`
3. **Use trueno flash attention** (best long-term): replace entire QKV pipeline

### 4.2 Backward Pass

The causal mask must also be applied in the backward pass:
- `backward_attention()` in `cuda_block.rs` computes `dA @ V^T` and `A^T @ dO`
- The gradient of masked positions (where `A[i,j] = 0`) should also be zero
- If using softmax backward: the mask is implicitly handled (softmax output was zero)
- If using fused attention: backward must also be causal

### 4.3 NF4 Path

The NF4 training path (`CudaNf4TransformerBlock`) also has
`compute_attention_cuda()` at line 2984. The same fix must be applied.

## 5. Files to Modify

| File | Line | Change |
|------|------|--------|
| `transformer/cuda_block.rs` | 855 | Add causal mask before softmax (fp32 path) |
| `transformer/cuda_block.rs` | 2984 | Add causal mask before softmax (NF4 path) |
| `autograd/cuda_forward/activations.rs` | 224 | Optional: create `batched_causal_softmax_forward` |

## 6. Provable Contract

See `contracts/causal-attention-mask-v1.yaml` (C-CAUSAL-001).

5 falsification tests:
- FALSIFY-CAUSAL-001: Future tokens have zero attention weight
- FALSIFY-CAUSAL-002: Last token sees all positions
- FALSIFY-CAUSAL-003: Changing future token doesn't affect past output
- FALSIFY-CAUSAL-004: Attention matrix is lower triangular
- FALSIFY-CAUSAL-005: Step-1K convergence within 2x of PyTorch

## 7. References

- [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — Attention Is All You Need
- [Dao et al. (2022)](https://arxiv.org/abs/2205.14135) — FlashAttention
- [Dao (2023)](https://arxiv.org/abs/2307.08691) — FlashAttention-2
- [PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [HuggingFace LlamaAttention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- entrenar#310, entrenar#309
