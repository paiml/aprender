# Case Study: PTX Parity Validation (GH-219)

This chapter documents the PTX parity validation system—a **compile-time Poka-Yoke** that catches GPU kernel generation bugs before they reach runtime. It validates that batched kernels maintain structural parity with their single-vector reference implementations.

## The Problem: Batched Kernels Diverge Silently

When we added batched prefill (processing all prompt tokens in one GPU pass), we created batched variants of 6 GPU kernels. Each batched kernel must implement the same mathematical operation as its single-vector reference, just for M vectors instead of 1.

But three classes of bugs can creep in silently:

| Bug Class | Example | Impact |
|-----------|---------|--------|
| Missing batch dispatch | No `ctaid.y` in batched RmsNorm | All vectors processed as batch=0 |
| u64 shared memory | `ld.shared.u64 [%rd4]` instead of `[%r4]` | Wrong shared memory addressing on some GPUs |
| Wrong dispatch strategy | grid_y for GEMV instead of register_unroll | Poor memory coalescing, 10x slowdown |

**Real bug found (GH-219):** `BatchedQ6KGemvKernel` had 3 dequantization bugs:
1. Wrong thread-to-value mapping (contiguous vs strided)
2. Wrong ql/qh addressing (naive linear vs Q6K super-block layout)
3. Wrong bit combination (`ql+4*qh-32` vs `ql|(qh<<4)-32`)

These bugs produced garbage output—but only for Q6K quantized models, and only during batched prefill. Serial prefill worked perfectly, making the bug extremely hard to catch with traditional testing.

## The Solution: Structural PTX Analysis

Instead of testing numerical outputs (which requires models and is flaky), we validate the **structure** of generated PTX assembly at compile time.

### The KernelParity Trait

```rust
/// Implemented by every batched kernel in trueno-gpu
pub trait KernelParity: Kernel {
    /// Expected batch dispatch mechanism
    fn expected_dispatch() -> BatchDispatch;

    /// The reference (single-vector) kernel for comparison
    type Reference: Kernel;

    /// Validate structural parity between batched and reference PTX
    fn validate_batch_dispatch(&self) -> ParityReport;
}
```

### Two Dispatch Strategies

**grid_y (ctaid.y)** — For elementwise kernels (RmsNorm, ResidualAdd, RoPE, SwiGLU):

```
// Single-vector: grid.x covers the hidden dimension
kernel_rmsnorm<<<grid_x, block>>>(input, output, eps);

// Batched: grid.y selects which vector in the batch
kernel_batched_rmsnorm<<<(grid_x, batch_size), block>>>(input, output, eps);
// PTX: mov.u32 %r_batch, %ctaid.y;
```

**register_unroll (m_dim)** — For quantized GEMV kernels (Q4K, Q6K):

```
// Single-vector: one output row per thread block
kernel_q4k_gemv<<<n_rows, block>>>(input, weights, output);

// Batched: M output rows, each block handles one row across all batch elements
kernel_batched_q4k_gemv<<<n_rows, block>>>(input, weights, output, m_dim);
// PTX: ld.param.u32 %r_m, [m_dim];
```

### What Gets Validated

For each kernel pair, the validator checks:

1. **Batch dispatch mechanism exists** — The PTX contains `%ctaid.y` (grid_y) or `m_dim` parameter (register_unroll)
2. **No u64 shared memory addressing** — `st.shared` and `ld.shared` instructions use `[%r...]` (32-bit), not `[%rd...]` (64-bit)
3. **Dispatch strategy matches expectation** — Elementwise kernels use grid_y, GEMV kernels use register_unroll

## The 6 Kernel Pairs

| # | Batched Kernel | Reference | Strategy | Validates |
|---|----------------|-----------|----------|-----------|
| 1 | `BatchedVectorizedRmsNormKernel` | `VectorizedRmsNormKernel` | grid_y | Attention/FFN layer norm |
| 2 | `BatchedQ4KGemvKernel` | `Q4KGemvKernel` | register_unroll | QKV/output/FFN projections |
| 3 | `BatchedQ6KGemvKernel` | `Q6KGemvKernel` | register_unroll | Q6K quantized models |
| 4 | `BatchedResidualAddKernel` | `ResidualAddKernel` | grid_y | Skip connections |
| 5 | `BatchedRopeKernel` | `RopeKernel` | grid_y | Rotary position embeddings |
| 6 | `BatchedSwigluKernel` | `SwigluKernel` | grid_y | FFN activation |

## Integration: `apr qa` Gate 6

The validation runs automatically as part of the QA suite:

```bash
# Runs all 7 gates including PTX parity
apr qa model.gguf --verbose

# Output:
# Running PTX parity validation...
#   ✓ PASS PTX Parity 6/6 kernel pairs passed PTX parity
#        14ms
```

The gate:
1. Detects GGUF format from magic bytes (first 8 bytes, not the full file)
2. Extracts model dimensions from GGUF metadata (`GGUFConfig::from_gguf`)
3. Instantiates all 6 batched kernels with those dimensions
4. Runs structural PTX validation on each
5. Reports pass/fail with specific violations

### Skip flag

```bash
# Skip PTX parity if not needed (e.g., CPU-only testing)
apr qa model.gguf --skip-ptx-parity
```

## Running the Example

```bash
# With CUDA (validates actual PTX)
cargo run -p apr-cli --example ptx_parity_validation --features inference,cuda

# Without CUDA (shows structure only)
cargo run -p apr-cli --example ptx_parity_validation --features inference
```

Output:
```
═══════════════════════════════════════════════════════════════════
     GH-219: PTX Parity Validation — Poka-Yoke for GPU Kernels
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ Demo 1: Qwen2.5-Coder-1.5B (Q4K) — 6 Kernel Pairs             │
└─────────────────────────────────────────────────────────────────┘

  Model dimensions:
    hidden_dim:       1536
    intermediate_dim: 8960
    num_heads:        12
    head_dim:         128

  ┌──────────────────────────────────┬──────────┬──────────────────┐
  │ Kernel Pair                      │ Status   │ Dispatch         │
  ├──────────────────────────────────┼──────────┼──────────────────┤
  │ BatchedRmsNorm ↔ RmsNorm        │ PASS     │ grid_y           │
  │ BatchedQ4KGemv ↔ Q4KGemv       │ PASS     │ register_unroll  │
  │ BatchedQ6KGemv ↔ Q6KGemv       │ PASS     │ register_unroll  │
  │ BatchedResidualAdd ↔ ResidualAdd│ PASS     │ grid_y           │
  │ BatchedRoPE ↔ RoPE             │ PASS     │ grid_y           │
  │ BatchedSwiGLU ↔ SwiGLU         │ PASS     │ grid_y           │
  └──────────────────────────────────┴──────────┴──────────────────┘

  6/6 kernel pairs passed PTX parity
```

## Toyota Way Principles Applied

### Poka-Yoke (Mistake-Proofing)

The validation runs at **compile time** (PTX generation), not at runtime. You cannot ship a broken batched kernel because the QA gate catches it before the model runs.

### Jidoka (Stop the Line)

If any kernel pair fails validation, `apr qa` fails the entire suite. You cannot ship a model with broken PTX parity.

### Genchi Genbutsu (Go and See)

The `--verbose` flag shows exactly which PTX instruction violated parity, with the specific line from the generated assembly. No guessing—you see the actual problem.

## Lessons Learned

1. **Test structure, not output** — Numerical output tests are flaky and require models. Structural PTX analysis is deterministic and fast (14ms for all 6 pairs).
2. **Two dispatch strategies exist for a reason** — Elementwise ops are embarrassingly parallel (grid_y). GEMV is memory-bound and benefits from register unrolling across the batch dimension.
3. **Copy dequant logic exactly** — When writing a batched variant of a quantized kernel, copy the dequantization logic verbatim from the reference. The Q6K bug came from rewriting it "more cleanly."

## References

- GH-219: PTX Parity Validation issue
- `trueno-gpu/src/kernels/parity_impls.rs` — KernelParity implementations (27 tests)
- `realizar/src/ptx_parity.rs` — Wrapper module with KernelDimensions and PtxParityReport
- `crates/apr-cli/src/commands/qa.rs` — Gate 6 implementation
- Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press.
