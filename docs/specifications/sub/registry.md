# Sub-spec: Kernel Contract Registry

**Parent:** [pv-spec.md](../pv-spec.md) Section 10

---

## 1. Registry Organization

Contracts live in `contracts/` with project-specific subdirectories:

```
contracts/
+-- softmax-kernel-v1.yaml         Tier 1 foundation
+-- rmsnorm-kernel-v1.yaml         Tier 1 foundation
+-- rope-kernel-v1.yaml            Tier 1 foundation
+-- activation-kernel-v1.yaml      Tier 1 foundation (SwiGLU)
+-- attention-kernel-v1.yaml       Tier 2 composite
+-- matmul-kernel-v1.yaml          Tier 2 composite
+-- flash-attention-v1.yaml        Tier 2 composite
+-- ...                            (165 total)
+-- aprender/                      aprender-specific
|   +-- binding.yaml               301 binding entries
|   +-- tensor-layout-v1.yaml
|   +-- layer-parity-v1.yaml
|   +-- ...
+-- entrenar/                      training-specific
|   +-- cuda-classify-training-v1.yaml
|   +-- qlora-hyperparameters-v1.yaml
+-- trueno/                        SIMD/CUDA-specific
+-- trueno-gpu/                    GPU-specific
+-- forjar/                        forging-specific
```

---

## 2. Kernel Equivalence Classes

Transformer architectures share kernel subsets. Five equivalence
classes cover all major architectures:

### Class A: Llama / Mistral / Yi

```
GQA + RMSNorm + SiLU + SwiGLU + RoPE
```

Contracts: gqa-kernel-v1, rmsnorm-kernel-v1, silu-kernel-v1,
swiglu-kernel-v1, rope-kernel-v1

### Class B: GPT-2 / BERT / RoBERTa

```
MHA + LayerNorm + GELU + Absolute Position
```

Contracts: attention-kernel-v1, layernorm-kernel-v1, gelu-kernel-v1,
absolute-position-v1

### Class C: BLOOM / MPT

```
MHA + LayerNorm + GELU + ALiBi
```

Contracts: attention-kernel-v1, layernorm-kernel-v1, gelu-kernel-v1,
alibi-kernel-v1

### Class D: Gemma

```
LayerNorm + GELU + SiLU + GQA
```

Contracts: layernorm-kernel-v1, gelu-kernel-v1, silu-kernel-v1,
gqa-kernel-v1

### Class E: Qwen

```
RMSNorm + SwiGLU + GQA
```

Contracts: rmsnorm-kernel-v1, swiglu-kernel-v1, gqa-kernel-v1

---

## 3. Contract Tiers

### Tier 1: Foundation Kernels

No dependencies. Building blocks for everything else.

| Contract | Paper | Key Property |
|---|---|---|
| softmax-kernel-v1 | Bridle 1990 | Output sums to 1.0 |
| rmsnorm-kernel-v1 | Zhang & Sennrich 2019 | Unit RMS before scaling |
| rope-kernel-v1 | Su et al. 2021 | Rotation structure, periodicity |
| silu-kernel-v1 | Elfwing et al. 2018 | x * sigmoid(x) |
| swiglu-kernel-v1 | Shazeer 2020 | Gate * up structure |
| gelu-kernel-v1 | Hendrycks & Gimpel 2016 | Gaussian-weighted activation |
| layernorm-kernel-v1 | Ba et al. 2016 | Zero mean, unit variance |
| batchnorm-kernel-v1 | Ioffe & Szegedy 2015 | Batch statistics |
| embedding-lookup-v1 | — | Index bounds, sparsity |
| cross-entropy-kernel-v1 | — | Loss >= 0, gradient correct |
| linear-projection-v1 | — | y = Wx + b |
| dropout-v1 | Srivastava et al. 2014 | Mask * scale invariance |

### Tier 2: Composite Kernels

Depend on Tier 1 contracts.

| Contract | Depends On | Key Property |
|---|---|---|
| attention-kernel-v1 | softmax, matmul | QK^T/sqrt(d_k) |
| gqa-kernel-v1 | attention | Group query heads |
| matmul-kernel-v1 | — | C_ij = sum_k A_ik * B_kj |
| flash-attention-v1 | attention, softmax | Tiled + online softmax |
| sliding-window-attention-v1 | attention | Window mask |
| qk-norm-v1 | rmsnorm, attention | Pre-attention norm |

### Tier 3: System Kernels

| Contract | Purpose |
|---|---|
| kv-cache-equivalence-v1 | Cached vs uncached parity |
| kv-cache-sizing-v1 | Memory budget bounds |
| sampling-algorithms-v1 | Top-p, top-k, temperature |
| inference-pipeline-v1 | End-to-end pipeline |
| streaming-tpot-v1 | Token-per-output-token latency |

### Tier 4: Training Kernels

| Contract | Purpose |
|---|---|
| adamw-kernel-v1 | Weight update formula |
| loss-functions-v1 | Cross-entropy, MSE, MAE |
| lora-algebra-v1 | Low-rank adaptation math |
| classification-finetune-v1 | Classification head training |
| optimization-v1 | Learning rate scheduling |

### Tier 5: Classical ML

| Contract | Algorithm |
|---|---|
| kmeans-kernel-v1 | Lloyd's algorithm |
| pagerank-kernel-v1 | Power iteration |
| pca-v1 | Eigendecomposition |
| svm-v1 | Support vector classification |
| decision-tree-v1 | CART splitting |
| random-forest-v1 | Bagged trees |
| naive-bayes-v1 | Bayes' theorem |
| lbfgs-kernel-v1 | Two-loop recursion |
| gbm-v1 | Gradient boosting |

### Tier 6: Model-Specific

| Contract | Model Family |
|---|---|
| qwen2-shapes-v1 | Qwen2 / Qwen2.5 |
| qwen2-e2e-verification-v1 | Qwen2 end-to-end |
| qwen3-shapes-v1 | Qwen3 |
| qwen3-e2e-verification-v1 | Qwen3 end-to-end |
| qwen3moe-shapes-v1 | Qwen3 MoE |
| qwen3moe-e2e-verification-v1 | Qwen3 MoE end-to-end |
| qwen35-shapes-v1 | Qwen3.5 |
| qwen35-hybrid-forward-v1 | Qwen3.5 hybrid |

### Tier 7: Performance (KAIZEN)

KAIZEN contracts are not kernel contracts — they are performance
obligation contracts created during optimization sprints. Located in
`contracts/entrenar/` and `contracts/trueno-gpu/`.

Examples: buffer pre-allocation, zero-copy transfers, kernel fusion
decisions, GPU residency obligations.

---

## 4. Gap Analysis

### Current Coverage

```
pv coverage contracts/ --binding contracts/aprender/binding.yaml
```

| Metric | Value |
|---|---|
| Total contracts | 165 |
| Total binding entries | 442 (aprender 301, entrenar 96, realizar 23, trueno 22) |
| Implemented bindings | 295 (98.0%) |
| Partial bindings | 0 |
| Not implemented | 6 |
| ALLOWED_GAPS (build.rs) | 3 equations from ssm-kernel-v1 |

Note: The binding.yaml header comment (289/282/3/4) is stale.
Actual counts from `grep -c "status: ..."` differ.

### Known Gaps

| Contract | Gap | Priority |
|---|---|---|
| ssm-kernel-v1 | 3 equations not implemented (ALLOWED_GAPS) | Low |
| arch-constraints-v1 | Broken YAML: missing `rule` field in falsification_tests | High |
| flash-attention-v1 | No Kani harnesses | High |
| matmul-kernel-v1 | 3/7 obligations unproven | High |

---

## 5. Production Incident Contracts

Four contracts born from production incidents:

| Contract | Incident | Root Cause |
|---|---|---|
| tensor-layout-v1 | PMAT-234 | SafeTensors 94.5% zeros passed checks |
| layer-parity-v1 | PMAT-232 | 7B GPU garbage, no CPU comparison |
| kernel-fusion-v1 | PAR-077 | Fused kernel existed but never wired |
| quantized-dot-product-v1 | PAR-001 | SIMD kernels had no reference |

These demonstrate the Kaizen principle: when a production incident
reveals a failure mode not covered by the contract, update the contract
BEFORE fixing the code.
