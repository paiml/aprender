# Sub-spec: Contract Gap Analysis

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 10

---

## ML/Systems Contract Gaps

Systematic gap analysis against the full ML systems landscape.
Current registry: 271 scored contracts (896 equations). 9 domains analyzed.
Updated 2026-04-03 with infrastructure contracts (MCP, CLI, HTTP).

### 1. Training Infrastructure (major gap)

AdamW and L-BFGS exist but almost nothing else from the optimizer ecosystem.

**Missing:**
- Adam variants (Adadelta, Lion, Muon)
- Learning rate schedulers (cosine with warmup, one-cycle)
- Gradient clipping (global norm vs per-param)
- Gradient accumulation correctness
- **Distributed training allreduce** (ring-allreduce, NCCL semantics)

### 2. Quantization (partial gap)

Q4K/Q6K superblocks and quantization ordering exist.

**Missing:**
- **GPTQ** (block-wise second-order quantization)
- **AWQ** (activation-aware weight quantization)
- **INT8 GEMM** semantics
- **FP8 (e4m3/e5m2 interchange)** — small state space, tractable for Kani
- **QLoRA dequantization** kernels

### 3. Attention Variants (partially covered)

Flash attention and sliding window exist.

**Missing:**
- **Flash-Decoding v2** (split-KV parallelization)
- **Ring Attention** (sequence parallelism across GPUs)
- **MLA** (DeepSeek multi-head latent attention with KV compression)

### 4. Memory Management (biggest practical gap)

KV cache sizing and equivalence exist.

**Missing:**
- **PagedAttention** (vLLM block-table semantics) — pointer aliasing, Kani-tractable
- **Prefix caching** invariants
- **Speculative decoding** correctness (draft/verify loop) — highest Kani leverage

### 5. Numerical Precision (mostly missing)

F16 conversion exists.

**Missing:**
- **BF16 rounding behavior**
- **FP8 scaled tensor semantics** — directly needed for trueno
- **Stochastic rounding** contracts

### 6. Tokenization / Data Pipeline (entirely absent)

Zero contracts. Major class of silent correctness failure.

**Missing:**
- **BPE merge correctness** — associativity and round-trip provable
- Sequence packing (multi-document attention mask algebra)
- Data collation invariants

### 7. SIMD / Hardware Abstraction (partial)

Roofline and kernel launch budget exist.

**Missing:**
- **Warp divergence bounds**
- **Memory coalescing contracts**
- **WGPU/Metal** compute semantics

### 8. Post-Training / Alignment (almost entirely absent)

LoRA algebra exists.

**Missing:**
- **DPO/RLHF loss** correctness (log-ratio clipping, KL terms) — single equation, high value
- **PPO/GRPO** policy update invariants
- **Reward model** contracts

### 9. Inference Serving (mostly missing)

Sampling algorithms exist.

**Missing:**
- **Continuous batching** correctness (iteration-level scheduling)
- **Tensor parallelism** (column/row linear split contracts)
- **CUDA graph capture** constraints

---

## Shape Contract Gaps

The repo has 7 shape-aware contracts (tensor-shape-flow, validated-tensor,
qwen35-shapes, kv-cache-sizing, model-config-algebra, format-parity,
tensor-inventory). Analysis of what's missing:

### Shape Algebra is Point-in-Time, Not Compositional

Existing contracts verify individual tensor shapes or model-specific
consistency. Missing: a *compositional* shape type system — proofs that
shapes propagate correctly through arbitrary operator sequences.

### Broadcast Semantics Uncontracted

NumPy/PyTorch broadcast rules (right-align, size-1-expand) have known
edge cases. No contract captures when broadcast is valid, when it
silently produces wrong shapes, or rank promotion rules.

### Batch Dimension Contracts Absent

The difference between `[B, S, H]` and `[S, B, H]` is invisible to most
shape checks but silently produces wrong results. No contracts around
batch dimension position invariants.

### Dynamic Shape Reasoning Missing

Contracts for shapes depending on runtime values — sequence length after
padding removal, variable-length batches, dynamic KV cache slicing.
Hardest shape bugs in practice.

### Contiguity and Stride Contracts Absent

A tensor can have correct logical shape but wrong physical layout
(non-contiguous after transpose). No contracts for when a kernel requires
contiguous input, what `.contiguous()` guarantees, or stride-based view
correctness. Directly relevant to trueno SIMD alignment assumptions.

### Operator-Level Shape Inference Missing

Each operator (conv, matmul, gather, scatter, einsum) has a shape
inference rule from its math. The repo has matmul/conv1d kernel contracts
but not their *shape inference* contracts.

### Einsum Contraction Correctness Absent

Index contraction rules (what gets summed, what survives, dimension
compatibility) are provable properties with no contract.

---

## Highest-Leverage Additions (ranked by impact x Kani tractability)

| Priority | Contract | Why |
|---|---|---|
| 1 | **Speculative decoding** | Tractable BMC, clear acceptance criterion, not yet formally verified |
| 2 | **FP8 e4m3/e5m2 interchange** | Small state space, directly needed for trueno |
| 3 | **DPO loss** | Single log-ratio clipping equation with known failure modes |
| 4 | **BPE tokenization** | Merge rule associativity and round-trip are provable |
| 5 | **PagedAttention block table** | Pointer aliasing invariants — exactly what Kani excels at |
| 6 | **Shape type algebra** | Formalizes dependent-type shape safety for Kani verification |
| 7 | **Broadcast semantics** | Small rule set, tractable, prevents silent shape bugs |
| 8 | **Operator shape inference** | 6 core transforms (reshape, permute, broadcast, slice, gather, scatter) |
| 9 | **Allreduce correctness** | Ring-allreduce associativity/commutativity provable |
| 10 | **Continuous batching** | Scheduling invariants for inference serving |

---

## References

1. Kwon, W. et al. (2023). "Efficient Memory Management for LLM Serving with PagedAttention." SOSP.
2. Leviathan, Y. et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
3. Frantar, E. et al. (2023). "GPTQ: Accurate Post-Training Quantization." ICLR.
4. Lin, J. et al. (2024). "AWQ: Activation-aware Weight Quantization." MLSys.
5. Rafailov, R. et al. (2023). "Direct Preference Optimization." NeurIPS.
6. Sennrich, R. et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." ACL.
