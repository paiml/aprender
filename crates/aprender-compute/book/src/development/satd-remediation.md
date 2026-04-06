# SATD Remediation Guide

This guide documents the process and patterns for identifying and fixing **Self-Admitted Technical Debt (SATD)** in trueno-gpu kernels.

## What is SATD?

Self-Admitted Technical Debt (SATD) refers to code where developers have explicitly acknowledged shortcuts or incomplete implementations through comments. Common SATD markers include:

- `// TODO`
- `// FIXME`
- `// HACK`
- `// Simplified`
- `// Placeholder`
- `// Exit after one iteration for simplicity`

## The Stubbed Loop Pattern

The most critical SATD pattern in GPU kernels is the **stubbed loop**:

```rust
// ANTI-PATTERN: Stubbed Loop (SATD)
let counter = ctx.mov_u32_imm(0);
ctx.label("loop_start");
let done = ctx.setp_ge_u32(counter, max);
ctx.branch_if(done, "loop_end");

// ... loop body ...

let _next = ctx.add_u32(counter, 1);  // INCREMENT DISCARDED!
ctx.branch("loop_end");                // WRONG: exits immediately!

ctx.label("loop_end");
```

**Why it's dangerous:**
- Loop executes only once regardless of input size
- Produces mathematically incorrect results
- Silently fails on real data (works on trivial test cases)

## The Fix Pattern

Correct loop implementation uses in-place updates:

```rust
// CORRECT: Proper Loop
let counter = ctx.mov_u32_imm(0);
ctx.label("loop_start");
let done = ctx.setp_ge_u32(counter, max);
ctx.branch_if(done, "loop_end");

// ... loop body ...

ctx.add_u32_inplace(counter, 1);  // IN-PLACE UPDATE
ctx.branch("loop_start");          // BRANCH BACK TO LOOP

ctx.label("loop_end");
```

## TRUENO-SATD-001 Fixes

The following SATD issues were identified and fixed:

### 1. quantize.rs: K-loop (Lines 232-233)

**Before:**
```rust
let _k_next = ctx.add_u32(k_block, 1);
ctx.branch("k_block_done");  // Simplified - single iteration
```

**After:**
```rust
ctx.add_u32_inplace(k_block, 1);
ctx.branch("k_block_loop");
```

### 2. quantize.rs: Shuffle Broadcast (Line 226)

**Before:**
```rust
let broadcast_sum = ctx.shfl_down_f32(block_sum, 0, mask);  // No-op!
```

**After:**
```rust
let broadcast_sum = ctx.shfl_idx_f32(block_sum, 0, mask);  // Proper broadcast
```

**Why:** `shfl_down(x, 0)` is a no-op (shifts by 0). Use `shfl_idx(x, 0)` to broadcast lane 0's value.

### 3. softmax.rs: Max-Reduce (Lines 214-215)

**Before:**
```rust
let _next_stride = ctx.add_u32(stride_reg, 0);  // placeholder
ctx.branch("max_reduce_done");  // Exit after one iteration
```

**After:**
```rust
ctx.shr_u32_inplace(stride_reg, 1);  // Halve stride
ctx.branch("max_reduce_loop");        // Loop back
```

### 4. softmax.rs: Sum-Reduce

Similar fix applied to sum reduction loop.

## Testing SATD Fixes (EXTREME TDD)

Every SATD fix requires falsifiable tests:

```rust
#[test]
fn test_kloop_branches_back_to_loop_start() {
    let kernel = QuantizeKernel::new(64, 64, 128);
    let ptx = kernel.emit_ptx();

    let has_loop_back = ptx.contains("bra k_block_loop");

    assert!(
        has_loop_back,
        "FALSIFIED: K-loop does not branch back to loop start"
    );
}
```

## Running the Example

Verify SATD fixes with:

```bash
cargo run --example satd_kernels
```

Expected output:
```
╔══════════════════════════════════════════════════════════════╗
║     SATD Remediation: Fixed Kernel Examples                  ║
╚══════════════════════════════════════════════════════════════╝

K-loop fix verified: ✓ PASS
Shuffle fix verified: ✓ PASS
Max-reduce fix verified: ✓ PASS
Sum-reduce fix verified: ✓ PASS
Stride halving verified: ✓ PASS
```

## In-Place Update Methods

The PTX builder provides in-place update methods for loops:

| Method | Purpose |
|--------|---------|
| `add_u32_inplace(dst, imm)` | Increment loop counter |
| `add_f32_inplace(dst, src)` | Accumulate float value |
| `shr_u32_inplace(dst, imm)` | Halve stride (tree reduction) |
| `fma_f32_inplace(dst, a, b)` | GEMM accumulation |

## Prevention Checklist

Before committing GPU kernel code:

1. **Search for SATD comments:** `grep -r "Simplified\|TODO\|FIXME" src/kernels/`
2. **Verify loop structure:** Branch targets should be `loop_start`, not `loop_done`
3. **Check in-place updates:** Loop counters use `_inplace` methods
4. **Run SATD tests:** `cargo test test_kloop test_shuffle test_reduce`
5. **Run example:** `cargo run --example satd_kernels`

## References

- **Specification:** `docs/specifications/fix-stubbed-kernel-loops-enhanced-monitoring-pixel-level-gpu-stress-testing-probar.md`
- **Academic:** Potdar & Shihab (2014), "An exploratory study on self-admitted technical debt"
- **Related:** PARITY-040 (WMMA Infrastructure), PARITY-041 (Q4_K GGML Format)
