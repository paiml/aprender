# Kernel Contract Registry

107 kernel contracts ship in `contracts/`, organized by tier, plus
57 per-crate contracts in `contracts/{aprender,entrenar,trueno,forjar}/`.

## Naming Convention

```
<operation>-kernel-v<version>.yaml
```

Model architecture and verification contracts omit the `-kernel` suffix:

```
<concept>-v<version>.yaml
```

## Contract ID Convention

Falsification test IDs follow: `FALSIFY-<PREFIX>-NNN`

| Contract | Prefix | Equations | Obligations |
|----------|--------|-----------|-------------|
| softmax-kernel | SM | 1 | 6 |
| rmsnorm-kernel | RN | 1 | 5 |
| rope-kernel | RP | 1 | 4 |
| activation-kernel | ACT | 3 | 6 |
| attention-kernel | ATT | 1 | 5 |
| matmul-kernel | MM | 2 | 5 |
| flash-attention | FA | 1 | 4 |
| swiglu-kernel | SG | 2 | 4 |
| gqa-kernel | GQ | 1 | 5 |
| layernorm-kernel | LN | 2 | 6 |
| silu-kernel | SI | 2 | 5 |
| cross-entropy-kernel | CE | 2 | 5 |
| adamw-kernel | AW | 4 | 5 |
| ssm-kernel | SSM | 3 | 5 |
| conv1d-kernel | CV | 1 | 5 |
| batchnorm-kernel | BN | 3 | 5 |
| kmeans-kernel | KM | 3 | 5 |
| pagerank-kernel | PR | 2 | 5 |
| lbfgs-kernel | LB | 3 | 5 |
| cma-es-kernel | CMA | 3 | 5 |
| model-config-algebra | MCA | 5 | 6 |
| qk-norm | QKN | 1 | 7 |
| tensor-shape-flow | TSF | 5 | 6 |
| roofline-model | RM | 4 | 5 |
| gated-delta-net | GDN | 5 | 5 |
| format-parity | FP | 4 | 5 |
| shannon-entropy | SE | 2 | 4 |
| f16-conversion | F16 | 2 | 4 |
| kernel-launch-budget | KL | 3 | 4 |
| tensor-inventory | TI | 5 | 6 |
| performance-grading | PG | 3 | 5 |
| q4k-q6k-superblock | QS | 5 | 7 |
| sampling-algorithms | SA | 4 | 5 |
| validated-tensor | VT | 3 | 4 |
| hybrid-layer-dispatch | HL | 6 | 6 |
| qwen35-shapes | Q35 | 5 | 7 |
| kv-cache-sizing | KV | 5 | 6 |
| kv-cache-equivalence | KCE | 4 | 4 |
| backend-dispatch | BD | 4 | 5 |
| lora-algebra | LA | 5 | 6 |
| quantization-ordering | QO | 4 | 5 |
| sliding-window-attention | SWA | 5 | 7 |
| rope-extrapolation | REXT | 6 | 8 |
| embedding-algebra | EMB | 6 | 7 |
| inference-pipeline | INF | 6 | 7 |
| qwen35-hybrid-forward | QHF | 6 | 7 |
| attention-scaling | ASCL | 6 | 7 |
| qwen35-e2e-verification | QE2E | 6 | 7 |

## QA Gate ID Convention

`F-<PREFIX>-NNN` (matches certeza format).

---

## Tier 1 — Core Kernels (7 contracts)

Foundation operations with no contract dependencies.

| Contract | Description | Key Equation |
|----------|-------------|--------------|
| softmax-kernel-v1 | Log-sum-exp safe softmax | `s_i = exp(x_i - max) / Σ exp(x_j - max)` |
| rmsnorm-kernel-v1 | Root mean square normalization | `RMSNorm(x) = x / RMS(x) * γ` |
| rope-kernel-v1 | Rotary position embedding | `RoPE(x, m) = x * cos(mθ) + rotate(x) * sin(mθ)` |
| activation-kernel-v1 | GeLU / ReLU / SiLU activations | `GeLU(x) = x * Φ(x)` |
| attention-kernel-v1 | Scaled dot-product attention | `Attn(Q,K,V) = softmax(QK^T/√d_k)V` |
| matmul-kernel-v1 | GEMM / GEMV | `C_{ij} = Σ_k A_{ik} B_{kj}` |
| flash-attention-v1 | Tiled attention with online softmax | `FlashAttn = tiled_softmax(QK^T/√d_k) V` |

## Tier 2 — Compound Kernels (6 contracts)

Compose Tier 1 operations.

| Contract | Description | Dependencies |
|----------|-------------|--------------|
| swiglu-kernel-v1 | SwiGLU gated MLP | silu, matmul |
| gqa-kernel-v1 | Grouped-query attention | softmax, matmul |
| layernorm-kernel-v1 | Layer normalization | — |
| silu-kernel-v1 | SiLU / Swish activation | — |
| cross-entropy-kernel-v1 | Cross-entropy loss | softmax |
| adamw-kernel-v1 | AdamW optimizer | — |

## Tier 3 — Extended Algorithms (7 contracts)

Research and classical algorithms.

| Contract | Description | Dependencies |
|----------|-------------|--------------|
| ssm-kernel-v1 | Mamba state-space model | — |
| conv1d-kernel-v1 | Causal 1D convolution | — |
| batchnorm-kernel-v1 | Batch normalization | — |
| kmeans-kernel-v1 | K-means clustering | — |
| pagerank-kernel-v1 | PageRank iteration | — |
| lbfgs-kernel-v1 | L-BFGS optimization | — |
| cma-es-kernel-v1 | CMA-ES evolution strategy | — |

## Model Architecture (21 contracts)

Structural and configuration contracts for model inference.

| Contract | Description | Dependencies |
|----------|-------------|--------------|
| model-config-algebra-v1 | Config parameter algebra | — |
| qk-norm-v1 | Query-key normalization | rmsnorm |
| tensor-shape-flow-v1 | Tensor shape propagation | — |
| roofline-model-v1 | Compute/memory roofline | — |
| gated-delta-net-v1 | Gated Delta Network | conv1d |
| format-parity-v1 | Format equivalence checks | — |
| shannon-entropy-v1 | Shannon entropy metrics | — |
| f16-conversion-v1 | FP16 conversion algebra | — |
| kernel-launch-budget-v1 | GPU kernel launch budgets | — |
| tensor-inventory-v1 | Tensor inventory tracking | — |
| performance-grading-v1 | Performance tier grading | — |
| q4k-q6k-superblock-v1 | Q4K/Q6K superblock layout | — |
| sampling-algorithms-v1 | Top-p, top-k, temperature | softmax |
| validated-tensor-v1 | Validated tensor newtypes | — |
| hybrid-layer-dispatch-v1 | Attention/GDN layer routing | — |
| qwen35-shapes-v1 | Qwen3.5-9B concrete shapes | model-config-algebra |
| kv-cache-sizing-v1 | KV cache memory sizing | model-config-algebra |
| kv-cache-equivalence-v1 | KV cache format equivalence | — |
| backend-dispatch-v1 | CPU/GPU backend dispatch | — |
| lora-algebra-v1 | LoRA adapter algebra | — |
| quantization-ordering-v1 | Quantization format ordering | — |

## Qwen 3.5 Verification (7 contracts)

End-to-end verification of the Qwen 3.5 hybrid architecture.

| Contract | Description | Dependencies |
|----------|-------------|--------------|
| sliding-window-attention-v1 | Bounded-context window mask | softmax, attention |
| rope-extrapolation-v1 | NTK/YaRN context extension | rope |
| embedding-algebra-v1 | Token embed/unembed algebra | — |
| inference-pipeline-v1 | Prefill/decode pipeline | softmax, attention, GDN, embedding, rmsnorm |
| qwen35-hybrid-forward-v1 | Hybrid attention/GDN forward | attention, GDN, rmsnorm, swiglu, qk-norm, dispatch |
| attention-scaling-v1 | 1/√d_k scaling + QK-norm | softmax, qk-norm |
| qwen35-e2e-verification-v1 | Full model verification | 8 sub-contracts (capstone) |

---

## Qwen 3.5 Verification DAG

The end-to-end verification contract composes all Qwen 3.5 kernel contracts
into a complete model proof. The dependency graph:

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
│   └── hybrid-layer-dispatch-v1
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
│   └── qk-norm-v1
└── kv-cache-sizing-v1 → model-config-algebra-v1
```

## Totals (Core Kernel Contracts)

| Metric | Count |
|--------|-------|
| Contracts | 107 |
| Equations | 356 |
| Proof Obligations | 550 |
| Falsification Tests | 590 |
| Kani Harnesses | 211 |
| Obligation Coverage | 100% |

## Binding Registries

| Crate | Bindings | Policy | Coverage |
|-------|----------|--------|----------|
| aprender | 301 | AllImplemented | 100% |
| entrenar | 96 | WarnOnGaps | 84% |
| realizar | 23 | WarnOnGaps | 100% |
| trueno | 22 | AllImplemented | 100% |
