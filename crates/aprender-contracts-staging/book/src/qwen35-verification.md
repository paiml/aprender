# Qwen 3.5 Verification

This chapter describes the end-to-end verification of the Qwen 3.5 hybrid
architecture using provable-contracts. Seven dedicated contracts compose
into a complete model proof covering the full inference pipeline from
token embedding through logit generation.

## Architecture Overview

Qwen 3.5 is a hybrid transformer that alternates two types of sequence
modeling layers:

- **Standard attention layers** with QK-norm and grouped-query attention
  (GQA), using sliding window masks for efficient long-sequence inference.
- **Gated Delta Network (GDN) layers** providing linear-complexity
  alternatives via causal conv1d and delta-rule state updates.

Both layer types share a common SwiGLU feed-forward network (FFN) sublayer
and use RMSNorm pre-normalization with residual connections.

### Key Architectural Constants (Qwen3.5-9B)

| Parameter | Value |
|-----------|-------|
| Hidden dimension (d_model) | 4096 |
| Attention heads (n_h) | 16 |
| KV heads (n_kv) | 4 (GQA ratio 4:1) |
| Head dimension (d_k) | 256 |
| Total layers (L) | 48 |
| FFN intermediate | 12288 (3x d_model, SwiGLU) |
| Vocabulary (V) | 152064 |
| RoPE base frequency | 1000000.0 |

## The Seven Contracts

### 1. Sliding Window Attention (`sliding-window-attention-v1`)

Defines the bounded-context window mask algebra for efficient attention.

**Key equations:**
- Window mask: `M_ij = 1 if |i - j| <= w, 0 otherwise`
- Causal window mask: `M_ij = M_window AND M_causal`
- Multi-layer receptive field: `R(L) = min(L * w, seq_len)`

**Dependencies:** softmax-kernel, attention-kernel

### 2. RoPE Extrapolation (`rope-extrapolation-v1`)

NTK-aware scaling and YaRN interpolation for extending context windows
beyond training length.

**Key equations:**
- Base frequency: `θ_i = base^(-2i/d)`
- NTK scaled base: `base' = base * (α * s / (2π))^(d/(d-2))`
- YaRN mixed frequency: `f'_i = (1 - λ_i) * f_i/s + λ_i * f_i`

**Dependencies:** rope-kernel

### 3. Embedding Algebra (`embedding-algebra-v1`)

Token embedding and unembedding with tied weight semantics.

**Key equations:**
- Embedding lookup: `embed(t) = E[t, :] for t ∈ [0, V)`
- Unembedding: `logits = h @ E^T / τ` (weight tying)
- Vocabulary bounds: `∀t in input: 0 <= t < V`

**Dependencies:** none (base contract)

### 4. Inference Pipeline (`inference-pipeline-v1`)

End-to-end prefill/decode composition through the hybrid layer stack.

**Key equations:**
- Prefill: `logits = unembed(norm(compose(embed(tokens), blocks)))`
- Decode step: `logit = unembed(norm(compose_cached(embed(token), blocks)))`
- Layer composition: `h_L = block_L ∘ ... ∘ block_1 (h_0)`

**Dependencies:** softmax, attention, GDN, embedding-algebra, rmsnorm

### 5. Hybrid Forward Pass (`qwen35-hybrid-forward-v1`)

The per-block forward computation with attention/GDN routing.

**Key equations:**
- Attention sublayer: `y = x + attn(qk_norm(q_proj(rmsnorm(x))), kv_proj(rmsnorm(x)))`
- GDN sublayer: `y = x + gdn(conv1d(rmsnorm(x)))`
- FFN sublayer: `y = x + swiglu(rmsnorm(x))`
- Hybrid block: `block_l(x) = ffn(attn_or_gdn_l(x))`

**Dependencies:** attention, GDN, rmsnorm, swiglu, qk-norm,
hybrid-layer-dispatch

### 6. Attention Scaling (`attention-scaling-v1`)

Numerical stability through 1/sqrt(d_k) normalization and QK-norm.

**Key equations:**
- Scaled dot product: `score(Q, K) = Q @ K^T / sqrt(d_k)`
- Variance preservation: `Var(score_ij) ≈ 1` for unit-variance inputs
- Score bound with QK-norm: `|score_ij| <= sqrt(d_k)`

**Dependencies:** softmax, qk-norm

### 7. End-to-End Verification (`qwen35-e2e-verification-v1`)

The capstone contract composing all sub-contracts into a complete model
proof. This is the top of the dependency DAG.

**Key equations:**
- Parameter count: `P = V*d + L*(d_attn + d_ffn + d_norm) + d_final ≈ 9.05B`
- FLOPs per token: `F ≈ 2P` (forward pass)
- Memory: `M = M_weights + M_kv + M_activations`
- Verification ladder: `coverage(contracts) = verified / total = 1.0`

**Dependencies:** 8 sub-contracts (hybrid-forward, shapes, pipeline,
embedding, sliding-window, rope-extrapolation, attention-scaling,
kv-cache-sizing)

## Dependency DAG

```
softmax ← attention ← sliding-window-attention
       ← cross-entropy        ↑
       ← sampling       qk-norm ← attention-scaling
       ← gqa                   ↑
                        rmsnorm ← qwen35-hybrid-forward ← e2e
silu ← swiglu ─────────────────↑
matmul ← gqa             conv1d ← gated-delta-net ──────↑
rope ← rope-extrapolation       hybrid-dispatch ────────↑
                          embedding-algebra ← inference-pipeline
model-config-algebra ← qwen35-shapes ──────────────────↑
                     ← kv-cache-sizing ─────────────────↑
```

### Full Dependency Tree

```
qwen35-e2e-verification-v1
├── qwen35-hybrid-forward-v1
│   ├── attention-kernel-v1 → softmax-kernel-v1
│   ├── gated-delta-net-v1 → conv1d-kernel-v1
│   ├── rmsnorm-kernel-v1
│   ├── swiglu-kernel-v1 → silu-kernel-v1 + matmul-kernel-v1
│   ├── qk-norm-v1 → rmsnorm-kernel-v1
│   └── hybrid-layer-dispatch-v1 → model-config-algebra-v1
├── qwen35-shapes-v1 → model-config-algebra-v1
├── inference-pipeline-v1
│   ├── softmax-kernel-v1
│   ├── attention-kernel-v1
│   ├── gated-delta-net-v1
│   ├── embedding-algebra-v1
│   └── rmsnorm-kernel-v1
├── embedding-algebra-v1
├── sliding-window-attention-v1
│   ├── softmax-kernel-v1
│   └── attention-kernel-v1
├── rope-extrapolation-v1 → rope-kernel-v1
├── attention-scaling-v1
│   ├── softmax-kernel-v1
│   └── qk-norm-v1 → rmsnorm-kernel-v1
└── kv-cache-sizing-v1 → model-config-algebra-v1
```

## Verification Coverage

| Metric | Count |
|--------|-------|
| Qwen 3.5 contracts | 7 |
| Equations | 41 |
| Proof obligations | 50 |
| Falsification tests | 50 |
| Kani harnesses | 14 |
| Binding entries | 41 |

All 50 proof obligations have corresponding falsification tests. The
end-to-end verification contract achieves 100% obligation coverage
across all 7 Qwen 3.5 contracts.

## Using the Verification Suite

```bash
# Validate all Qwen 3.5 contracts
for f in contracts/sliding-window-attention-v1.yaml \
         contracts/rope-extrapolation-v1.yaml \
         contracts/embedding-algebra-v1.yaml \
         contracts/inference-pipeline-v1.yaml \
         contracts/qwen35-hybrid-forward-v1.yaml \
         contracts/attention-scaling-v1.yaml \
         contracts/qwen35-e2e-verification-v1.yaml; do
  pv validate "$f"
done

# Generate property tests for the full suite
pv probar contracts/qwen35-e2e-verification-v1.yaml \
    --binding contracts/aprender/binding.yaml

# Run traceability audit
pv audit contracts/qwen35-e2e-verification-v1.yaml \
    --binding contracts/aprender/binding.yaml

# Visualize the dependency graph
pv graph contracts/

# Cross-contract obligation coverage
pv coverage contracts/ --binding contracts/aprender/binding.yaml
```

## Design Principles

1. **Compositional verification**: Each component is verified independently.
   Shape invariants compose through residual connections. The e2e contract
   proves that composition preserves all invariants.

2. **Popperian falsification**: Every proof obligation has a falsification
   test that would fail if the obligation were violated. Tests are
   designed to fail fast on contract violations.

3. **Dependency-ordered verification**: The topological sort of the DAG
   gives a natural build order. Verify foundations (softmax, rmsnorm)
   first, then composites (attention, SwiGLU), then the pipeline, then
   the capstone.

4. **Quantitative bounds**: Where possible, obligations include concrete
   numeric bounds (parameter count within [9.0B, 9.2B], variance ≈ 1.0,
   entropy bounds). This makes falsification tests crisp.
