# MoE Inference: From GGUF Packed Tensors to Contract-First Expert Dispatch

This chapter traces the full arc of bringing Mixture of Experts inference
to aprender-serve (realizar): how we loaded GGUF packed 3D tensors, hit
four distinct bugs on the way to coherent output, and built three provable
contracts that prevent regression.

The model under study is **Qwen3-Coder-30B-A3B-Instruct-Q4\_K\_M** -- 128
experts, top-8 routing, SwiGLU activation per expert, 48 decoder layers.
Active parameters per token: 3.3B of 30.5B total (10.8%).

## 1. Mixture of Experts Architecture

A dense transformer feeds every hidden state through a single FFN. MoE
replaces that FFN with N parallel expert FFNs and a learned router that
selects the top-k:

```
hidden ─── router(W_gate) ──── softmax ──── top-k ──┐
                                                     │
            ┌─────────┐  ┌─────────┐       ┌────────┴──────┐
            │ expert 0 │  │ expert 1 │ ...  │ expert 127    │
            │ SwiGLU   │  │ SwiGLU   │      │ SwiGLU        │
            └────┬─────┘  └────┬─────┘      └────┬──────────┘
                 │              │                  │
                 └──── weighted sum (renormalized) ┘
                                │
                             output
```

Each expert is a standard SwiGLU FFN:

```
expert_forward(x):
    gate_out = x @ gate_proj.T          // [moe_intermediate]
    up_out   = x @ up_proj.T            // [moe_intermediate]
    swiglu   = SiLU(gate_out) * up_out  // [moe_intermediate]
    output   = swiglu @ down_proj.T     // [hidden_dim]
```

For Qwen3-Coder-30B-A3B: hidden=2048, moe\_intermediate=768, so each
expert has 3 x 2048 x 768 = 4,718,592 parameters. With 128 experts per
layer across 48 layers, the total expert parameter count dominates the
model at ~29B of ~30.5B.

The router selects the top-8 experts per token. Weights are renormalized
(`norm_topk_prob`) so that the 8 selected weights sum to 1.0. This means
only 8/128 = 6.25% of expert compute is active per token.

### Why MoE is memory-bound, not compute-bound

The fundamental tension: you must **load all 128 experts** into memory
(bandwidth cost proportional to total parameters) but only **compute with 8**
(FLOP cost proportional to active parameters). This makes MoE inference
memory-bandwidth-bound on every known hardware target. The roofline model
confirms:

```
tok/s = min(bandwidth / bytes_per_token, compute / flops_per_token)
```

For Q4K quantization on a 200 GB/s memory bus, the bandwidth limit is the
binding constraint. This is why llama.cpp achieves 92 tok/s while our
initial CPU implementation reached 1.76 tok/s -- the gap is entirely in
memory access patterns, not arithmetic.

## 2. GGUF Packed 3D Tensor Format

Dense models store one 2D weight tensor per layer per projection. MoE
models in GGUF pack all 128 experts into a single 3D tensor per projection
type per layer:

| GGUF tensor name | GGUF dims (ne) | After `dims.reverse()` |
|---|---|---|
| `blk.L.ffn_gate_exps.weight` | `[hidden, intermediate, num_experts]` | `[num_experts, intermediate, hidden]` |
| `blk.L.ffn_up_exps.weight` | `[hidden, intermediate, num_experts]` | `[num_experts, intermediate, hidden]` |
| `blk.L.ffn_down_exps.weight` | `[intermediate, hidden, num_experts]` | `[num_experts, hidden, intermediate]` |
| `blk.L.ffn_gate_inp.weight` | `[hidden, num_experts]` | `[num_experts, hidden]` |

The GGUF parser reads dimensions in storage order and calls `dims.reverse()`
to convert from GGUF column-major convention to row-major logical shape.
After reversal, `dims[0]` is always `num_experts` for 3D expert tensors.

The binary layout is contiguous: expert 0's data occupies bytes
`[0..expert_stride)`, expert 1 occupies `[expert_stride..2*expert_stride)`,
and so on, where `expert_stride = total_bytes / num_experts`.

For Q4K quantization, this means each expert's gate/up projection occupies
884,736 bytes (768 rows x 8 super blocks x 144 bytes per Q4K block), and
the total packed tensor for 128 experts is ~113 MB per projection per layer.

## 3. The Four Bugs

Every bug below was caught during the first end-to-end run of
`apr run qwen3-coder-30b-q4k.gguf --prompt "def fibonacci(n):"`. The model
produced garbage tokens. Contract-first debugging (running `apr qa` first,
then `apr tensors`, then `apr trace`) identified each root cause.

### Bug 1: GGUF 3D Tensor Loading

**Symptom:** `Expert tensor not found: model.layers.0.mlp.experts.0.gate_proj.weight`

**Root cause:** The GGUF file does not contain per-expert tensors. It stores
a single packed 3D tensor named `blk.0.ffn_gate_exps.weight`. The tensor
loader expected the HuggingFace per-expert naming convention
(`model.layers.L.mlp.experts.E.gate_proj.weight`) and failed to find it.

**Fix:** Add Format B path in the MoE tensor loader -- detect the `_exps`
suffix, load the packed 3D tensor, and slice it by stride:

```rust
// Format B: GGUF packed 3D tensors
let gate_3d = source.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?;
let expert_stride = gate_3d.data.len() / num_experts;
for e in 0..num_experts {
    let offset = e * expert_stride;
    let expert_data = &gate_3d.data[offset..offset + expert_stride];
    // ... use expert_data directly (zero-copy with Arc<Mmap>)
}
```

**Contract coverage:** FALSIFY-MOE-007 ("GGUF packed 3D expert tensors load
correctly") would have caught this if written first.

### Bug 2: Expert Count from Reversed Dims

**Symptom:** `num_experts = 2048` (should be 128)

**Root cause:** After `dims.reverse()`, the code read `dims[2]` for
num\_experts. But `dims.reverse()` puts the expert count at `dims[0]`:

```
Before reverse: [hidden=2048, intermediate=768, num_experts=128]
After reverse:  [num_experts=128, intermediate=768, hidden=2048]
                 ^^^^^^^^ dims[0], NOT dims[2]
```

Reading `dims[2]` returned `hidden=2048`, creating 2048 "experts" with
wrong shapes.

**Fix:** Read `dims[0]` after reversal for the expert count. This is
consistent with the GGUF convention documented in `tensor-layout-v1.yaml`
(LAYOUT-001): GGUF dims are reversed at the import boundary.

### Bug 3: Architecture Constraint Mismatch

**Symptom:** `UnsupportedArchitecture("qwen3moe")`

**Root cause:** The GGUF metadata field `general.architecture` contained
`qwen3moe` (no underscore), but the architecture constraint table expected
`qwen3_moe` (with underscore). This is a case-sensitivity and naming
convention mismatch between llama.cpp's metadata writer and our parser.

**Fix:** Normalize architecture strings by stripping underscores before
matching:

```rust
fn normalize_arch(s: &str) -> String {
    s.to_lowercase().replace('_', "")
}
```

**Contract coverage:** The `arch-constraints-v1` contract now lists both
variants.

### Bug 4: Double Down-Projection

**Symptom:** Coherent-looking but numerically wrong output. Layer-by-layer
trace showed divergence starting at the first MoE FFN block.

**Root cause:** The MoE expert FFN (`expert_forward`) correctly applied
gate, up, SwiGLU, and down projections. But the caller (`moe_forward`)
applied down\_proj a second time to the weighted sum:

```rust
// BUG: double down_proj application
fn moe_forward(hidden, experts, routes) {
    let mut output = zeros(hidden_dim);
    for (idx, weight) in routes {
        let expert_out = expert_forward(hidden, &experts[idx]);
        // expert_out is already [hidden_dim] after down_proj
        output += weight * expert_out;
    }
    // WRONG: down_proj applied AGAIN
    output = output @ down_proj.T;  // <-- second application
}
```

The fix was removing the redundant down\_proj in the caller. The expert
SwiGLU FFN is self-contained: `x -> gate/up -> SwiGLU -> down -> output`.
The weighted sum of expert outputs is the final MoE output.

**This bug is subtle** because the output shape is still correct
(`[hidden_dim]` in, `[hidden_dim]` out). Only numerical comparison against
a reference implementation (HF transformers `modeling_qwen3_moe.py`) exposed it.

## 4. Zero-Copy Stride-Based Dispatch

The initial working implementation copied each expert's weight data into a
fresh `Vec<u8>` before calling the Q4K matmul kernel. For 128 experts x
3 projections x 884 KB per expert, this allocated **113 MB per layer per
token**. Across 48 layers, that is 5.4 GB of heap allocation per generated
token.

The fix: `PackedMoeRef` -- a reference type that borrows directly from the
memory-mapped GGUF file:

```rust
/// Zero-copy reference into a packed 3D MoE tensor.
/// Borrows from Arc<Mmap>, no allocation on expert access.
struct PackedMoeRef {
    data: Arc<Mmap>,          // shared mmap of the GGUF file
    tensor_offset: usize,     // byte offset of the 3D tensor in the file
    expert_stride: usize,     // bytes per expert = total_bytes / num_experts
    num_experts: usize,
}

impl PackedMoeRef {
    /// Returns a slice into the mmap for expert `idx`. Zero allocation.
    fn expert_data(&self, idx: usize) -> &[u8] {
        let offset = self.tensor_offset + idx * self.expert_stride;
        &self.data[offset..offset + self.expert_stride]
    }
}
```

The Q4K matmul kernel (`fused_q4k_parallel_matvec`) accepts `&[u8]`
directly -- no ownership transfer needed. The expert slice is valid for
the lifetime of the `Arc<Mmap>`, which lives for the duration of the
model session.

**Contract:** `moe-stride-dispatch-v1` requires bit-identical output
between the copy-based and stride-based paths (FALSIFY-STRIDE-001) and
zero heap allocation in the expert dispatch hot path (FALSIFY-STRIDE-002).

## 5. Performance Analysis: 1.76 tok/s vs 92 tok/s

After fixing all four bugs and implementing stride-based dispatch, the
measured throughput on the Blackwell GB10 (CPU path, 8-core ARM):

| Implementation | tok/s | Notes |
|---|---|---|
| llama.cpp (Q4\_K\_M, CPU) | 92 | Fused multi-expert SIMD kernel, NUMA-aware |
| aprender (Q4\_K\_M, CPU) | 1.76 | Per-expert sequential matvec, rayon parallel rows |

The 52x gap has three root causes, identified via `apr profile`:

### Root Cause 1: Per-Expert Kernel Launch Overhead

llama.cpp dispatches a single `ggml_mul_mat` call for all active experts
using stride-based indexing inside the kernel. aprender dispatches 8
separate `fused_q4k_parallel_matvec` calls (one per active expert), each
incurring rayon thread pool synchronization overhead.

**Impact:** 8 rayon dispatches x 3 projections x 48 layers = 1,152 rayon
synchronization points per token. Each rayon `par_chunks_mut` dispatch costs
~5 us of overhead, totaling ~5.8 ms of pure synchronization per token.

### Root Cause 2: Cache Thrashing

Each expert's gate projection is 884 KB. The 8 active experts' gate
projections total 7.1 MB, which exceeds the L2 cache (4 MB on GB10 ARM
cores). llama.cpp processes multiple experts in a single kernel pass,
keeping the input vector hot in L1. Our per-expert dispatch reloads the
input vector 24 times per layer (8 experts x 3 projections).

### Root Cause 3: No SIMD Fusion Across Experts

llama.cpp's `mul_mat_vec_q` kernel uses NEON/AVX2 intrinsics with expert
indexing built into the inner loop. Our implementation treats each expert
as an independent matmul, missing the opportunity to fuse the gate+up
projections and to amortize SIMD setup across experts.

### Remediation Plan (Contract-Driven)

The three MoE contracts define a phased remediation:

| Phase | Contract | Target | Status |
|---|---|---|---|
| Phase 5 | `moe-stride-dispatch-v1` | >= 10 tok/s | Implemented (1.76 -- gate not met) |
| Phase 6 | `moe-apr-q4k-inference-v1` FALSIFY-MOE-009 | >= 25 tok/s (4x parity) | Planned: fused multi-expert kernel |
| Phase 7 | `moe-cuda-kernel-v1` | >= 61 tok/s (1.5x parity) | Planned: single CUDA launch |

## 6. Three Provable Contracts

### Contract 1: `moe-apr-q4k-inference-v1`

Covers the full MoE inference pipeline: tensor loading (Format A per-expert
and Format B packed 3D), routing (softmax + top-k), expert FFN (SwiGLU with
Q4K dequant), and weighted sum.

**Key proof obligations:**

- All 128 experts loaded per layer, no partial loading
- Router produces exactly top\_k indices in `[0, num_experts)`
- Selected weights sum to 1.0 when `norm_topk_prob=true`
- SwiGLU activation used (not GELU, not ReLU)
- Q4K row-major layout (LAYOUT-002) enforced

**Falsification tests:**

| ID | Rule | Prediction | If Fails |
|---|---|---|---|
| FALSIFY-MOE-001 | Expert count | 128 x 3 x 48 = 18,432 tensors | Loop bounds wrong |
| FALSIFY-MOE-002 | Softmax stable | No NaN in routing | Missing max-subtract |
| FALSIFY-MOE-003 | Top-k correct | Returns exactly top-8 | Sort order or k boundary |
| FALSIFY-MOE-004 | Expert non-zero | Dequantized output non-zero | Q4K offset wrong |
| FALSIFY-MOE-005 | No OOM | Peak RSS < 24 GB for 17 GB model | Experts loaded as F32 |
| FALSIFY-MOE-006 | Coherent output | Python keywords in output | Layer trace needed |
| FALSIFY-MOE-007 | 3D unpack | 128 experts from ffn\_gate\_exps | dims.reverse() wrong |
| FALSIFY-MOE-008 | Throughput | >= 10 tok/s | Profile with apr profile |
| FALSIFY-MOE-009 | 4x parity | >= 25 tok/s | Fused kernel required |

### Contract 2: `moe-stride-dispatch-v1`

Covers zero-copy expert access via stride arithmetic on memory-mapped data.

**Key equation:**

```
expert_matmul_strided(packed_data, input, in_dim, out_dim, expert_idx, expert_stride):
    offset = expert_idx * expert_stride
    result = fused_q4k_parallel_matvec(&packed_data[offset..offset+expert_stride],
                                       input, in_dim, out_dim)
```

**Falsification tests:**

| ID | Rule | Prediction | If Fails |
|---|---|---|---|
| FALSIFY-STRIDE-001 | Bit-identical | Strided == copy-based output | Offset calculation wrong |
| FALSIFY-STRIDE-002 | Zero allocation | No Vec\<u8\> in hot path | Residual .to\_vec() |
| FALSIFY-STRIDE-003 | Throughput | >= 10 tok/s | Copy was not the bottleneck |

**Kani harness:** KANI-STRIDE-001 proves that for all `expert_idx` in
`[0, 128)`, `offset + stride <= data.len()`. Bound: 128, strategy:
`bounded_int`, solver: CaDiCaL.

### Contract 3: `moe-cuda-kernel-v1`

Specifies the target CUDA implementation: a single kernel launch per MoE
layer that indexes all top-k experts via stride offsets in GPU memory.

**Key design (from llama.cpp `mmvq.cu` and vLLM `fused_moe.py`):**

- Grid dimensions: `(nblocks_rows, top_k)` -- one block column per active expert
- Expert data accessed via `expert_id * stride_channel` offset
- Q4K/Q6K dot product uses existing `vec_dot_q4k` device function
- Output accumulated with expert weights in shared memory

**Target:** 61 tok/s (1.5x parity with llama.cpp's 92 tok/s).

## 7. Reference Implementations

Three reference implementations were consulted during development and
used for numerical validation:

### HuggingFace transformers `modeling_qwen3_moe.py`

The authoritative reference for correctness. Key patterns adopted:

- `Qwen3MoeSparseMoeBlock.forward()`: router softmax, top-k selection,
  weight renormalization, per-expert forward, weighted sum
- Expert FFN: `gate_proj`, `up_proj`, `down_proj` with `silu` activation
- `norm_topk_prob=True`: renormalize selected weights to sum to 1.0

This implementation confirmed Bug 4 (double down\_proj) -- the HF code
applies down\_proj inside the expert and does not re-apply it in the
sparse block.

### llama.cpp `ggml-cuda/mmvq.cu`

The performance reference. Key patterns studied:

- `mul_mat_vec_q_moe`: single kernel launch with `stride_channel_x`
  indexing for expert data
- GGUF packed 3D tensor layout: expert data is contiguous along the
  outermost dimension
- `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` naming convention

This implementation informed the stride-based dispatch design in
`moe-stride-dispatch-v1`.

### vLLM `fused_moe.py`

The throughput reference for batched inference. Key patterns noted:

- Token sorting by expert assignment for L2 cache reuse
- Fused triton kernel for gate+up+SwiGLU in a single pass
- Block-sparse routing for workload balancing

These patterns are earmarked for Phase 8 (batched MoE inference) but
were not needed for the single-token decode path.

## 8. Lessons Learned

**1. GGUF 3D tensors are not documented.** The packed expert format is a
llama.cpp implementation detail inferred from tensor names and dim ordering.
The contract (`moe-apr-q4k-inference-v1`) now documents both Format A and
Format B explicitly.

**2. `dims.reverse()` is a footgun.** After reversal, every dimension index
means something different. The fix is to name dimensions immediately after
parsing -- never index into a reversed array by magic number:

```rust
let (num_experts, intermediate, hidden) = match reversed_dims.as_slice() {
    [e, i, h] => (*e, *i, *h),
    _ => return Err(anyhow!("expected 3D tensor")),
};
```

**3. Shape-preserving bugs are the hardest.** Bug 4 (double down\_proj)
produced `[hidden_dim]` output with the correct shape. No shape assertion
could catch it. Only numerical comparison against a reference model exposed
the error. The contract now includes FALSIFY-MOE-006 (coherent output test)
as a mandatory gate.

**4. Memory bandwidth is the MoE bottleneck, not compute.** The 52x gap
between llama.cpp and our initial implementation is not algorithmic -- both
do the same matrix-vector products. The difference is entirely in how many
times the data is touched (cache reuse) and how many synchronization points
exist (kernel launch overhead). Contracts must include throughput gates, not
just correctness gates.
