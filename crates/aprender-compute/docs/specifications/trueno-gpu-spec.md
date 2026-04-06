# Trueno GPU: Pure Rust First-Principles GPU Compute Specification

**Version**: 1.2
**Date**: 2025-12-16
**Status**: SPECIFICATION - Ready for Implementation
**Priority**: P1 - Performance Critical Path
**Crate**: `trueno-gpu` (sub-crate of trueno ecosystem)
**Philosophy**: Own the Stack - Zero C Dependencies in Hot Path
**Review Status**: Toyota Way Engineering Review Complete (35 citations)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-10 | Batuta Team | Initial specification with 25 peer-reviewed citations |
| 1.1 | 2025-12-10 | Batuta Team | Toyota Way review: +Poka-Yoke, +Bank conflicts, +ILP, +10 citations [46-55] |
| 1.2 | 2025-12-16 | Batuta Team | Added Q5_K (PARITY-116) and Q6_K (PARITY-117) kernel specifications |

---

## Executive Summary

This specification defines a **pure Rust GPU compute library** built from first principles. Unlike wrapper crates (cudarc, cust) that FFI to NVIDIA libraries, trueno-gpu generates PTX directly from Rust, enabling:

1. **Zero C/C++ in the hot path** - Pure Rust kernel generation
2. **Compile-time safety** - Rust's type system for GPU memory safety
3. **Portability** - Same Rust code targets CUDA, Metal, Vulkan, WebGPU
4. **Auditability** - No opaque binary blobs, full source visibility

### Core Thesis

> **Hypothesis**: A pure Rust GPU compute stack can achieve ≥80% of cuBLAS performance while providing compile-time memory safety, zero undefined behavior, and complete auditability - properties impossible with C/C++ FFI approaches.

### Toyota Way Engineering Principles

1. **Genchi Genbutsu** (Go and See): Direct PTX generation, not wrapper abstractions
2. **Jidoka** (Automation with Human Touch): Compile-time GPU memory safety with Rust's type system
3. **Kaizen** (Continuous Improvement): Iterative kernel optimization with microbenchmarks
4. **Heijunka** (Level Loading): Uniform workload distribution across warps and banks
5. **Muda Elimination**: Zero unnecessary memory copies, zero register spills, zero runtime overhead
6. **Poka-Yoke** (Mistake Proofing): Rust typestates make invalid GPU states unrepresentable at compile time

---

## 1. Architecture Overview

### 1.1 The Memory Wall Problem [21]

Per Wulf & McKee's seminal paper [21], the fundamental bottleneck in modern computing is **memory bandwidth**, not compute. This is especially acute for LLM inference:

```
Roofline Analysis (phi-2 on RTX 4090):
├── Peak Compute: 82.6 TFLOPS (FP32)
├── Peak Bandwidth: 1,008 GB/s (GDDR6X)
├── Arithmetic Intensity Ceiling: 82 FLOP/byte
│
├── LLM Decode Phase:
│   ├── Arithmetic Intensity: ~62 FLOP/byte [31]
│   └── Status: MEMORY-BOUND
│
└── Implication: Bandwidth optimization > Compute optimization
```

**Design Principle**: Every architectural decision prioritizes memory bandwidth efficiency over raw compute throughput.

### 1.2 Crate Structure

```
trueno-gpu/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── ptx/                # PTX code generation
│   │   ├── mod.rs
│   │   ├── builder.rs      # PTX IR builder
│   │   ├── emit.rs         # PTX text emission
│   │   └── optimize.rs     # Peephole optimizations
│   ├── driver/             # CUDA driver API (minimal FFI)
│   │   ├── mod.rs
│   │   ├── context.rs      # CUcontext management
│   │   ├── module.rs       # CUmodule loading
│   │   ├── stream.rs       # CUstream async ops
│   │   └── memory.rs       # Device memory allocation
│   ├── kernels/            # Hand-optimized kernels
│   │   ├── mod.rs
│   │   ├── gemm.rs         # Matrix multiplication
│   │   ├── softmax.rs      # Numerically stable softmax
│   │   ├── layernorm.rs    # Fused LayerNorm
│   │   ├── attention.rs    # FlashAttention-style
│   │   └── quantize.rs     # Q4_K dequant + matmul
│   ├── memory/             # Memory management
│   │   ├── mod.rs
│   │   ├── pool.rs         # Memory pool allocator
│   │   ├── transfer.rs     # H2D/D2H transfers
│   │   └── unified.rs      # Unified memory support
│   └── backend/            # Multi-backend abstraction
│       ├── mod.rs
│       ├── cuda.rs         # NVIDIA CUDA
│       ├── metal.rs        # Apple Metal (future)
│       └── vulkan.rs       # Vulkan compute (future)
└── benches/
    ├── gemm_bench.rs
    ├── bandwidth_bench.rs
    └── kernel_bench.rs
```

### 1.3 Design Constraints

| Constraint | Rationale | Citation |
|------------|-----------|----------|
| No cuBLAS FFI in hot path | Auditability, reproducibility | [5, 6] |
| Pure Rust PTX generation | Compile-time safety | [32] |
| Explicit memory management | Predictable performance | [12] |
| Coalesced access patterns | Memory bandwidth [21] | [33] |
| Warp-uniform control flow | Avoid divergence penalty | [34, 35] |
| **ILP over Occupancy** | Hide latency via instruction parallelism, not just thread parallelism | **[46]** |
| **Bank conflict avoidance** | Shared memory accesses must not serialize | **[48, 49]** |
| **Register pressure management** | Avoid spilling to local memory (Muda) | **[47]** |

---

## 2. PTX Code Generation

### 2.1 PTX Overview [36]

PTX (Parallel Thread Execution) is NVIDIA's virtual ISA, providing:
- **Forward compatibility**: PTX compiled for SM 7.5 runs on SM 12.0
- **Optimization opportunity**: JIT compiler optimizes for target architecture
- **Portability**: Architecture-independent intermediate representation

```
Compilation Pipeline:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Rust Kernel │───▶│  PTX Text   │───▶│ cubin/SASS  │───▶│ GPU Execute │
│   (trueno)  │    │  (trueno)   │    │  (driver)   │    │  (hardware) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     Pure Rust         Pure Rust         NVIDIA JIT         Hardware
```

### 2.2 PTX Builder API

```rust
//! Pure Rust PTX generation - no LLVM, no nvcc
//! Generates PTX 8.0+ compatible with SM 7.0+

use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType, PtxReg};

/// Build a vector addition kernel in pure Rust
pub fn build_vector_add() -> PtxModule {
    let mut module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .address_size(64);

    let kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a_ptr")      // .param .u64 a_ptr
        .param(PtxType::U64, "b_ptr")      // .param .u64 b_ptr
        .param(PtxType::U64, "c_ptr")      // .param .u64 c_ptr
        .param(PtxType::U32, "n")          // .param .u32 n
        .body(|ctx| {
            // Calculate global thread index
            let tid = ctx.special_reg(PtxReg::TidX);
            let ctaid = ctx.special_reg(PtxReg::CtaIdX);
            let ntid = ctx.special_reg(PtxReg::NtidX);

            let idx = ctx.mad_lo_u32(ctaid, ntid, tid);  // idx = ctaid * ntid + tid

            // Bounds check (predicated)
            let n = ctx.load_param_u32("n");
            let pred = ctx.setp_ge_u32(idx, n);
            ctx.branch_if(pred, "exit");

            // Load inputs (coalesced access pattern) [33]
            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            let offset = ctx.mul_wide_u32(idx, 4);  // 4 bytes per f32
            let a_addr = ctx.add_u64(a_ptr, offset);
            let b_addr = ctx.add_u64(b_ptr, offset);
            let c_addr = ctx.add_u64(c_ptr, offset);

            let a_val = ctx.ld_global_f32(a_addr);
            let b_val = ctx.ld_global_f32(b_addr);

            // Compute
            let c_val = ctx.add_f32(a_val, b_val);

            // Store result
            ctx.st_global_f32(c_addr, c_val);

            ctx.label("exit");
            ctx.ret();
        });

    module.add_kernel(kernel);
    module
}
```

**Register Pressure Management [47]**: The PTX builder tracks register liveness to prevent spilling to slow local memory (a form of Muda):

```rust
/// Register allocator with liveness analysis
/// Per Xiao et al. [47] - prevents register spills
pub struct RegisterAllocator {
    /// Live ranges for each virtual register
    live_ranges: HashMap<VirtualReg, LiveRange>,
    /// Physical register pool (limited per SM)
    available: BitSet<256>,  // Max 256 registers per thread
    /// Spill count - should be zero (Muda)
    spill_count: usize,
}

impl RegisterAllocator {
    /// Allocate register with liveness-aware coloring
    pub fn allocate(&mut self, vreg: VirtualReg, live_range: LiveRange) -> PhysicalReg {
        // Find physical register not conflicting with overlapping live ranges
        for preg in self.available.iter() {
            if !self.conflicts(preg, &live_range) {
                self.live_ranges.insert(vreg, live_range);
                return PhysicalReg(preg);
            }
        }

        // Spill to local memory (Muda - should not happen with good ILP [46])
        self.spill_count += 1;
        warn!("Register spill detected - consider reducing ILP or tile size");
        self.spill_to_local(vreg)
    }

    /// Report register pressure metrics
    pub fn pressure_report(&self) -> RegisterPressure {
        RegisterPressure {
            max_live: self.max_simultaneous_live(),
            spill_count: self.spill_count,
            utilization: self.live_ranges.len() as f64 / 256.0,
        }
    }
}
```

### 2.3 Generated PTX Output

```ptx
// Generated by trueno-gpu v0.1.0
// Pure Rust PTX generation - no external dependencies

.version 8.0
.target sm_70
.address_size 64

.visible .entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
) {
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;

    // Thread index calculation
    mov.u32     %r1, %tid.x;
    mov.u32     %r2, %ctaid.x;
    mov.u32     %r3, %ntid.x;
    mad.lo.s32  %r4, %r2, %r3, %r1;      // idx = ctaid * ntid + tid

    // Bounds check
    ld.param.u32 %r5, [n];
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra    exit;

    // Load parameters
    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];

    // Calculate addresses (coalesced)
    mul.wide.u32 %rd4, %r4, 4;
    add.s64     %rd5, %rd1, %rd4;
    add.s64     %rd6, %rd2, %rd4;
    add.s64     %rd7, %rd3, %rd4;

    // Load, compute, store
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32     %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

exit:
    ret;
}
```

---

## 3. Memory Coalescing Strategy [33]

### 3.1 The Coalescing Problem

Per NVIDIA documentation and academic research [33], global memory transactions are coalesced at the warp level (32 threads). Misaligned or scattered accesses cause:

- **32x bandwidth waste** in worst case (32 separate transactions)
- **L2 cache pollution** from unused bytes in cache lines
- **Increased memory latency** from multiple round-trips

### 3.2 Coalesced Access Pattern

```rust
/// Memory access pattern analyzer
/// Ensures all global memory accesses are coalesced per [33]
pub struct CoalescingAnalyzer;

impl CoalescingAnalyzer {
    /// Verify access pattern is coalesced for warp of 32 threads
    ///
    /// Coalesced: Thread i accesses address base + i * element_size
    /// - All threads access consecutive memory locations
    /// - Base address is 128-byte aligned (L2 cache line)
    pub fn verify_coalesced<T>(
        base_addr: u64,
        thread_indices: &[u32; 32],
        element_size: usize,
    ) -> CoalescingResult {
        // Check alignment
        if base_addr % 128 != 0 {
            return CoalescingResult::Unaligned {
                addr: base_addr,
                required: 128
            };
        }

        // Check consecutive access
        for (i, &tid) in thread_indices.iter().enumerate() {
            let expected = i as u32;
            if tid != expected {
                return CoalescingResult::Scattered {
                    thread: i,
                    expected,
                    actual: tid,
                };
            }
        }

        CoalescingResult::Coalesced {
            transactions: 1,  // Single 128-byte transaction
            efficiency: 1.0,
        }
    }
}

/// GEMM kernel with explicit coalescing
/// Per tile-based GEMM methodology [37]
pub fn gemm_coalesced_kernel() -> PtxKernel {
    PtxKernel::new("gemm_coalesced")
        .shared_memory(TILE_SIZE * TILE_SIZE * 4 * 2)  // A and B tiles
        .body(|ctx| {
            // Thread block loads tile of A into shared memory
            // Coalesced: Each thread loads consecutive elements
            //
            // Warp 0, Thread 0-31: A[row][0:31]
            // Warp 1, Thread 0-31: A[row+1][0:31]
            // ...

            let warp_id = ctx.div_u32(ctx.tid_x(), 32);
            let lane_id = ctx.rem_u32(ctx.tid_x(), 32);

            // Coalesced load: lane_id determines column
            let a_col = lane_id;
            let a_row = ctx.add_u32(ctx.block_row(), warp_id);

            // Calculate global address (coalesced pattern)
            let a_offset = ctx.mad_lo_u32(a_row, ctx.lda(), a_col);
            let a_addr = ctx.add_u64(ctx.a_ptr(), ctx.mul_wide_u32(a_offset, 4));

            // Load to shared memory
            let a_val = ctx.ld_global_f32(a_addr);
            let smem_offset = ctx.mad_lo_u32(warp_id, TILE_SIZE, lane_id);
            ctx.st_shared_f32(smem_offset, a_val);

            ctx.bar_sync(0);  // __syncthreads()

            // ... compute using shared memory tiles ...
        })
}
```

### 3.3 Shared Memory Bank Conflicts [48, 49]

Shared memory is organized into **32 banks** (4-byte width each). When multiple threads in a warp access the same bank, accesses serialize causing **bank conflicts**:

```
Bank Conflict Example (32-way worst case):
┌────────────────────────────────────────────────────────────────┐
│ Shared Memory Layout (naive [32][32] tile)                      │
├────────────────────────────────────────────────────────────────┤
│ Bank 0:  [0][0], [0][32], [1][0], [1][32], ...                 │
│ Bank 1:  [0][1], [0][33], [1][1], [1][33], ...                 │
│ ...                                                             │
│ Bank 31: [0][31], [0][63], [1][31], [1][63], ...               │
├────────────────────────────────────────────────────────────────┤
│ Column 0 access by all 32 threads:                              │
│   Thread 0 → [0][0] → Bank 0                                   │
│   Thread 1 → [1][0] → Bank 0  ← CONFLICT!                      │
│   Thread 2 → [2][0] → Bank 0  ← CONFLICT!                      │
│   ...                                                           │
│   Thread 31 → [31][0] → Bank 0  ← 32-way serialization!        │
└────────────────────────────────────────────────────────────────┘
```

**Solution 1: Padding** - Add 1 element to row stride:

```rust
/// Bank-conflict-free shared memory allocation with padding
/// Per Volkov [46] and Ruetsch & Micikevicius [48]
pub struct SharedMemoryTile<const ROWS: usize, const COLS: usize> {
    /// Padded layout: [ROWS][COLS + 1] eliminates conflicts
    data: [[f32; COLS + 1]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> SharedMemoryTile<ROWS, COLS> {
    /// Access element with automatic padding adjustment
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        // Stride is COLS + 1, so consecutive rows map to different banks
        self.data[row][col]
    }

    /// Padded stride for PTX address calculation
    pub const fn stride() -> usize {
        COLS + 1  // 33 instead of 32
    }
}

/// PTX builder for padded shared memory
pub fn gemm_bank_conflict_free() -> PtxKernel {
    // Allocate [32][33] instead of [32][32]
    const TILE_SIZE: u32 = 32;
    const PADDED_STRIDE: u32 = 33;

    PtxKernel::new("gemm_bank_conflict_free")
        .shared_memory(TILE_SIZE * PADDED_STRIDE * 4)  // f32 = 4 bytes
        .body(|ctx| {
            let row = ctx.div_u32(ctx.tid_x(), TILE_SIZE);
            let col = ctx.rem_u32(ctx.tid_x(), TILE_SIZE);

            // Padded offset: row * PADDED_STRIDE + col
            let smem_offset = ctx.mad_lo_u32(row, PADDED_STRIDE, col);
            let smem_addr = ctx.mul_u32(smem_offset, 4);  // 4 bytes per f32

            // Load to shared memory (bank-conflict-free)
            ctx.st_shared_f32(smem_addr, ctx.ld_global_f32(ctx.global_addr()));
            ctx.bar_sync(0);

            // Column access is now conflict-free:
            // Thread 0 → row 0, col 0 → offset 0 → Bank 0
            // Thread 1 → row 1, col 0 → offset 33 → Bank 1  ← Different bank!
            // Thread 2 → row 2, col 0 → offset 66 → Bank 2  ← Different bank!
        })
}
```

**Solution 2: Swizzling** - XOR-based bank remapping (more complex but no memory waste):

```rust
/// XOR-based swizzling for bank conflict avoidance
/// Per Nath & Tomov [49]
pub fn swizzle_index(row: u32, col: u32) -> u32 {
    // XOR row with col to spread accesses across banks
    let swizzled_col = col ^ (row % 32);
    row * 32 + swizzled_col
}

/// PTX builder with swizzled addressing
pub fn gemm_swizzled() -> PtxKernel {
    PtxKernel::new("gemm_swizzled")
        .shared_memory(32 * 32 * 4)  // No padding needed
        .body(|ctx| {
            let row = ctx.div_u32(ctx.tid_x(), 32);
            let col = ctx.rem_u32(ctx.tid_x(), 32);

            // Swizzled column: col XOR (row % 32)
            let row_mod = ctx.and_u32(row, 31);  // row % 32
            let swizzled_col = ctx.xor_u32(col, row_mod);

            // Address with swizzled column
            let smem_offset = ctx.mad_lo_u32(row, 32, swizzled_col);
            // ... load/store with swizzled addresses
        })
}
```

**Design Decision**: trueno-gpu uses **padding by default** (simpler, proven effective [48]) with swizzling as an optional optimization for memory-constrained scenarios.

---

## 4. Warp Divergence Mitigation [34, 35]

### 4.1 The Divergence Problem

SIMT execution requires all threads in a warp to execute the same instruction. Branch divergence causes serialization:

```
Divergence Example (32-way worst case):
┌─────────────────────────────────────────────────────────┐
│ Warp Execution (32 threads)                              │
├─────────────────────────────────────────────────────────┤
│ if (tid < 16) {        // 16 threads take this path     │
│     path_A();          // Execute with 16 threads masked │
│ } else {               // 16 threads take this path     │
│     path_B();          // Execute with 16 threads masked │
│ }                                                        │
├─────────────────────────────────────────────────────────┤
│ Total instructions: 2x (both paths executed)            │
│ Efficiency: 50%                                          │
└─────────────────────────────────────────────────────────┘
```

Per Fung et al. [34], divergence can cause **27-125x slowdown** depending on architecture.

### 4.2 Divergence-Free Patterns

```rust
/// Divergence-free softmax using warp shuffle
/// Avoids all control flow divergence per [34]
pub fn softmax_warp_shuffle() -> PtxKernel {
    PtxKernel::new("softmax_warp_shuffle")
        .body(|ctx| {
            // Load value for this thread
            let val = ctx.ld_global_f32(ctx.input_addr());

            // Warp-level max reduction (no divergence)
            // Uses shuffle instructions [38]
            let mut max_val = val;
            for offset in [16, 8, 4, 2, 1] {
                let other = ctx.shfl_down_f32(max_val, offset);
                max_val = ctx.max_f32(max_val, other);  // No branch!
            }
            // Broadcast max to all lanes
            max_val = ctx.shfl_idx_f32(max_val, 0);

            // Compute exp(val - max) - numerically stable
            let shifted = ctx.sub_f32(val, max_val);
            let exp_val = ctx.ex2_approx_f32(ctx.mul_f32(shifted, LOG2_E));

            // Warp-level sum reduction (no divergence)
            let mut sum = exp_val;
            for offset in [16, 8, 4, 2, 1] {
                let other = ctx.shfl_down_f32(sum, offset);
                sum = ctx.add_f32(sum, other);
            }
            sum = ctx.shfl_idx_f32(sum, 0);

            // Final division
            let result = ctx.div_f32(exp_val, sum);
            ctx.st_global_f32(ctx.output_addr(), result);
        })
}

/// Predicated execution instead of branches
/// Converts control flow to data flow [35]
pub fn relu_predicated() -> PtxKernel {
    PtxKernel::new("relu_predicated")
        .body(|ctx| {
            let val = ctx.ld_global_f32(ctx.addr());

            // Instead of: if (val < 0) val = 0;
            // Use predicated select (no divergence):
            let zero = ctx.const_f32(0.0);
            let pred = ctx.setp_lt_f32(val, zero);
            let result = ctx.selp_f32(zero, val, pred);  // result = pred ? zero : val

            ctx.st_global_f32(ctx.addr(), result);
        })
}
```

---

## 5. Quantized GEMM Kernel [22, 23, 24]

### 5.1 Q4_K Dequantize-Fused GEMM

Per AWQ [23] and GPTQ [24], quantized inference requires fused dequantization to avoid memory bandwidth bottleneck:

```rust
/// Q4_K GEMM with fused dequantization
/// Per block quantization methodology [22, 23, 24]
///
/// Memory layout (Q4_K, block_size=32):
/// ┌─────────────────────────────────────────┐
/// │ Block Header (2 bytes)                   │
/// │   - scale: f16 (1 byte effective)        │
/// │   - min: f16 (1 byte effective)          │
/// ├─────────────────────────────────────────┤
/// │ Quantized values (16 bytes)              │
/// │   - 32 × 4-bit values packed             │
/// └─────────────────────────────────────────┘
///
/// Dequantization: val = scale * quant + min
pub fn q4k_gemm_fused() -> PtxKernel {
    PtxKernel::new("q4k_gemm_fused")
        .shared_memory(TILE_K * TILE_N * 4)  // Dequantized tile in shared
        .body(|ctx| {
            // Each warp processes one block of 32 weights
            let block_idx = ctx.div_u32(ctx.global_tid(), 32);
            let lane = ctx.rem_u32(ctx.global_tid(), 32);

            // Load block header (scale, min)
            let header_addr = ctx.add_u64(
                ctx.weights_ptr(),
                ctx.mul_wide_u32(block_idx, Q4K_BLOCK_SIZE)
            );
            let scale = ctx.ld_global_f16_to_f32(header_addr);
            let min_val = ctx.ld_global_f16_to_f32(ctx.add_u64(header_addr, 2));

            // Load packed 4-bit values (2 values per byte)
            let data_addr = ctx.add_u64(header_addr, 4);  // Skip header
            let byte_idx = ctx.div_u32(lane, 2);
            let nibble_idx = ctx.rem_u32(lane, 2);

            let packed = ctx.ld_global_u8(ctx.add_u64(data_addr, byte_idx as u64));

            // Extract 4-bit value (no branch - use shift/mask)
            let shift = ctx.mul_u32(nibble_idx, 4);
            let quant = ctx.and_u32(ctx.shr_u32(packed, shift), 0xF);

            // Fused dequantization: val = scale * quant + min
            let quant_f32 = ctx.cvt_f32_u32(quant);
            let dequant = ctx.fma_f32(scale, quant_f32, min_val);

            // Store to shared memory for GEMM
            ctx.st_shared_f32(ctx.mul_u32(lane, 4), dequant);
            ctx.bar_sync(0);

            // ... GEMM computation on dequantized tile ...
        })
}
```

### 5.2 Memory Bandwidth Analysis

```
Q4_K vs FP32 Memory Traffic:
┌────────────────────────────────────────────────────────┐
│ Model: phi-2 (2.7B parameters)                          │
├────────────────────────────────────────────────────────┤
│ FP32:  2.7B × 4 bytes = 10.8 GB                        │
│ Q4_K:  2.7B × 0.5 bytes = 1.35 GB                      │
│ Reduction: 8x                                           │
├────────────────────────────────────────────────────────┤
│ RTX 4090 Bandwidth: 1,008 GB/s                         │
│ FP32 floor: 10.8 GB / 1,008 GB/s = 10.7ms              │
│ Q4_K floor: 1.35 GB / 1,008 GB/s = 1.3ms               │
│ Speedup potential: 8x (memory-bound) [21]              │
└────────────────────────────────────────────────────────┘
```

### 5.3 Q5_K Fused GEMM Kernel (PARITY-116)

Q5_K provides 5-bit quantization with improved accuracy over Q4_K:

```
Q5_K Super-block Layout (176 bytes for 256 values):
┌────────────────────────────────────────────────────────┐
│ Offset 0-1:   d (f16 super-block scale)                │
│ Offset 2-3:   dmin (f16 super-block min)               │
│ Offset 4-15:  scales (12 bytes, packed 6-bit × 8)      │
│ Offset 16-143: qs (128 bytes, 256 × 4-bit low values)  │
│ Offset 144-175: qh (32 bytes, 256 × 1-bit high values) │
├────────────────────────────────────────────────────────┤
│ Dequantization: val = d × scale_b × (ql + 16×qh) - dmin × min_b │
│ Where ql is 4-bit (0-15), qh is 1-bit (0 or 1)         │
│ Combined 5-bit range: 0-31                             │
└────────────────────────────────────────────────────────┘
```

```rust
/// Q5_K quantized GEMM kernel
/// Per PARITY-116 specification
use trueno_gpu::kernels::{Q5KKernel, Kernel};

let kernel = Q5KKernel::new(1024, 1024, 4096);
let ptx = kernel.emit_ptx();

// Key features:
// - Nested super-block and sub-block loops
// - Loads both ql (4-bit) and qh (1-bit high) values
// - Fused dequantization with scale/min extraction
```

### 5.4 Q6_K Fused GEMM Kernel (PARITY-117)

Q6_K provides 6-bit quantization for highest accuracy among K-quant formats:

```
Q6_K Super-block Layout (210 bytes for 256 values):
┌────────────────────────────────────────────────────────┐
│ Offset 0-127:   ql (128 bytes, 256 × 4-bit low values) │
│ Offset 128-191: qh (64 bytes, 256 × 2-bit high values) │
│ Offset 192-207: scales (16 bytes, 16 × 8-bit scales)   │
│ Offset 208-209: d (f16 super-block scale)              │
├────────────────────────────────────────────────────────┤
│ Dequantization: val = d × scale_b × (ql + 4×qh - 32)   │
│ Where ql is 4-bit (0-15), qh is 2-bit (0-3)            │
│ Combined 6-bit signed range: -32 to 31                 │
└────────────────────────────────────────────────────────┘
```

```rust
/// Q6_K quantized GEMM kernel
/// Per PARITY-117 specification
use trueno_gpu::kernels::{Q6KKernel, Kernel};

let kernel = Q6KKernel::new(1024, 1024, 4096);
let ptx = kernel.emit_ptx();

// Key features:
// - 16 sub-blocks of 16 values (vs Q4/Q5's 8 sub-blocks of 32)
// - 2-bit high value extraction (vs Q5's 1-bit)
// - Signed offset (-32) for symmetric quantization
```

### 5.5 Quantization Format Comparison

| Format | Bits | Block Size | Bytes/256 | Accuracy | Use Case |
|--------|------|------------|-----------|----------|----------|
| Q4_K | 4 | 256 | 144 | Good | Default inference |
| Q5_K | 5 | 256 | 176 | Better | Quality-sensitive |
| Q6_K | 6 | 256 | 210 | Best | Maximum accuracy |

---

## 6. Tensor Core Integration [39, 40, 41]

### 6.1 WMMA (Warp Matrix Multiply-Accumulate)

Tensor Cores provide **8x** throughput for FP16 matrix multiplication [39]:

```rust
/// Tensor Core WMMA wrapper
/// Per NVIDIA Tensor Core analysis [39, 40]
pub fn tensor_core_gemm() -> PtxKernel {
    PtxKernel::new("tensor_core_gemm")
        .body(|ctx| {
            // WMMA dimensions: 16x16x16 (Volta/Turing)
            // Each warp computes: D = A × B + C
            // where A: 16×16 FP16, B: 16×16 FP16, C/D: 16×16 FP32

            // Declare WMMA fragments
            ctx.wmma_fragment_a("frag_a", 16, 16, 16, "row", "f16");
            ctx.wmma_fragment_b("frag_b", 16, 16, 16, "col", "f16");
            ctx.wmma_fragment_c("frag_c", 16, 16, 16, "f32");
            ctx.wmma_fragment_d("frag_d", 16, 16, 16, "f32");

            // Load fragments from shared memory
            ctx.wmma_load_a("frag_a", ctx.a_smem_ptr(), 16);
            ctx.wmma_load_b("frag_b", ctx.b_smem_ptr(), 16);
            ctx.wmma_fill_c("frag_c", 0.0);

            // Tensor Core MMA operation
            ctx.wmma_mma("frag_d", "frag_a", "frag_b", "frag_c");

            // Store result
            ctx.wmma_store_d(ctx.d_smem_ptr(), "frag_d", 16);
        })
}
```

### 6.2 Mixed Precision Strategy [30, 41]

Per mixed precision training literature [30]:

```rust
/// Mixed precision accumulation for numerical stability
/// Per Micikevicius et al. [30] and Fasi et al. [41]
pub struct MixedPrecisionConfig {
    /// Input/weight precision
    pub compute_type: PrecisionType,  // FP16 or BF16
    /// Accumulator precision
    pub accumulate_type: PrecisionType,  // FP32
    /// Output precision
    pub output_type: PrecisionType,  // FP16 or FP32
    /// Loss scaling for gradients (training only)
    pub loss_scale: Option<f32>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            compute_type: PrecisionType::FP16,
            accumulate_type: PrecisionType::FP32,  // Prevent overflow [41]
            output_type: PrecisionType::FP16,
            loss_scale: None,
        }
    }
}
```

---

## 7. FlashAttention-Style Kernel [16, 42]

### 7.1 IO-Aware Attention

Per Dao et al. [16], standard attention is memory-bound due to materializing the N×N attention matrix:

```
Standard Attention Memory:
├── Q, K, V: O(N × d) each
├── S = Q × K^T: O(N²)      ← Memory bottleneck!
├── P = softmax(S): O(N²)   ← Memory bottleneck!
└── O = P × V: O(N × d)

FlashAttention Memory:
├── Q, K, V: O(N × d) each
├── S, P: O(B_r × B_c)      ← Tiled, fits in SRAM!
└── O: O(N × d)
└── Reduction: O(N²) → O(N × d)
```

### 7.2 Tiled Attention Kernel

```rust
/// FlashAttention-style tiled attention
/// Per Dao et al. [16] - IO-aware algorithm
pub fn flash_attention_kernel() -> PtxKernel {
    const BLOCK_SIZE: u32 = 64;  // Tile size
    const HEAD_DIM: u32 = 64;    // Head dimension

    PtxKernel::new("flash_attention")
        .shared_memory((BLOCK_SIZE * HEAD_DIM * 4) * 3)  // Q, K, V tiles
        .body(|ctx| {
            // Outer loop: iterate over K, V blocks
            // Inner loop: iterate over Q blocks
            // Never materialize full N×N matrix

            // Load Q tile to shared memory
            let q_tile = ctx.load_tile_to_shared(
                ctx.q_ptr(),
                ctx.block_row(),
                BLOCK_SIZE,
                HEAD_DIM
            );

            // Initialize output accumulator and softmax stats
            let mut o_acc = ctx.zeros_f32(HEAD_DIM);
            let mut m_prev = ctx.const_f32(f32::NEG_INFINITY);  // Running max
            let mut l_prev = ctx.const_f32(0.0);  // Running sum of exp

            // Iterate over K, V blocks (tiled)
            for kv_block in 0..ctx.num_kv_blocks() {
                // Load K, V tiles
                let k_tile = ctx.load_tile_to_shared(
                    ctx.k_ptr(),
                    kv_block,
                    BLOCK_SIZE,
                    HEAD_DIM
                );
                let v_tile = ctx.load_tile_to_shared(
                    ctx.v_ptr(),
                    kv_block,
                    BLOCK_SIZE,
                    HEAD_DIM
                );

                ctx.bar_sync(0);

                // Compute S = Q × K^T (in shared memory)
                let s_tile = ctx.gemm_shared(q_tile, k_tile.transpose());

                // Online softmax update [16]
                // m_new = max(m_prev, rowmax(S))
                // l_new = exp(m_prev - m_new) * l_prev + rowsum(exp(S - m_new))
                // O_new = (l_prev * exp(m_prev - m_new) * O_prev + exp(S - m_new) × V) / l_new
                let s_max = ctx.row_max(s_tile);
                let m_new = ctx.max_f32(m_prev, s_max);

                let scale_prev = ctx.exp_f32(ctx.sub_f32(m_prev, m_new));
                let p_tile = ctx.exp_f32(ctx.sub_f32(s_tile, m_new));
                let l_new = ctx.add_f32(
                    ctx.mul_f32(scale_prev, l_prev),
                    ctx.row_sum(p_tile)
                );

                // Update output accumulator
                let pv = ctx.gemm_shared(p_tile, v_tile);
                o_acc = ctx.div_f32(
                    ctx.add_f32(
                        ctx.mul_f32(ctx.mul_f32(l_prev, scale_prev), o_acc),
                        pv
                    ),
                    l_new
                );

                m_prev = m_new;
                l_prev = l_new;
            }

            // Store final output
            ctx.store_tile_from_shared(ctx.o_ptr(), o_acc);
        })
}
```

---

## 8. Memory Pool Allocator [12]

### 8.1 PagedAttention-Inspired Memory Management

Per Kwon et al. [12], KV-cache fragmentation is a major source of memory waste:

```rust
/// GPU memory pool with fragmentation tracking
/// Per PagedAttention [12] methodology
pub struct GpuMemoryPool {
    /// Total pool size in bytes
    total_bytes: u64,
    /// Page size (default: 256KB)
    page_size: u64,
    /// Free page bitmap
    free_pages: BitVec,
    /// Allocation metadata
    allocations: HashMap<AllocationId, AllocationInfo>,
}

impl GpuMemoryPool {
    /// Allocate with fragmentation tracking
    pub fn allocate(&mut self, size: u64) -> Result<GpuBuffer, MemoryError> {
        let pages_needed = (size + self.page_size - 1) / self.page_size;

        // Find contiguous free pages
        let start_page = self.find_contiguous_pages(pages_needed)?;

        // Mark pages as allocated
        for i in start_page..(start_page + pages_needed) {
            self.free_pages.set(i as usize, false);
        }

        // Track allocation
        let id = AllocationId::new();
        self.allocations.insert(id, AllocationInfo {
            start_page,
            num_pages: pages_needed,
            size,
            timestamp: Instant::now(),
        });

        Ok(GpuBuffer {
            id,
            ptr: self.page_to_ptr(start_page),
            size,
        })
    }

    /// Calculate fragmentation percentage
    /// Per KV-cache waste metric [12]
    pub fn fragmentation_pct(&self) -> f64 {
        let total_pages = self.free_pages.len();
        let free_pages = self.free_pages.count_ones();
        let used_pages = total_pages - free_pages;

        if used_pages == 0 {
            return 0.0;
        }

        // Find largest contiguous free region
        let largest_free = self.largest_contiguous_free();
        let free_bytes = free_pages as u64 * self.page_size;
        let largest_bytes = largest_free as u64 * self.page_size;

        // Fragmentation = 1 - (largest_free / total_free)
        if free_bytes == 0 {
            0.0
        } else {
            (1.0 - (largest_bytes as f64 / free_bytes as f64)) * 100.0
        }
    }
}
```

---

## 9. Roofline Model Integration [19, 31]

### 9.1 Automatic Roofline Analysis

Per Williams et al. [19] and LLM Inference survey [31]:

```rust
/// Roofline model for kernel performance analysis
/// Per Williams et al. [19]
pub struct RooflineModel {
    /// Peak compute (FLOPS)
    peak_flops: f64,
    /// Peak memory bandwidth (bytes/s)
    peak_bandwidth: f64,
    /// Ridge point (FLOPS/byte)
    ridge_point: f64,
}

impl RooflineModel {
    pub fn new(gpu: &GpuInfo) -> Self {
        let peak_flops = gpu.peak_flops_fp32();
        let peak_bandwidth = gpu.peak_bandwidth_bytes();
        Self {
            peak_flops,
            peak_bandwidth,
            ridge_point: peak_flops / peak_bandwidth,
        }
    }

    /// Analyze kernel performance
    pub fn analyze(&self, kernel: &KernelProfile) -> RooflineAnalysis {
        let arithmetic_intensity = kernel.flops as f64 / kernel.bytes_accessed as f64;

        let theoretical_peak = if arithmetic_intensity < self.ridge_point {
            // Memory-bound: peak = bandwidth × intensity
            self.peak_bandwidth * arithmetic_intensity
        } else {
            // Compute-bound: peak = peak_flops
            self.peak_flops
        };

        let achieved = kernel.flops as f64 / kernel.duration.as_secs_f64();
        let efficiency = achieved / theoretical_peak;

        RooflineAnalysis {
            arithmetic_intensity,
            theoretical_peak_flops: theoretical_peak,
            achieved_flops: achieved,
            efficiency,
            bound: if arithmetic_intensity < self.ridge_point {
                Bound::Memory
            } else {
                Bound::Compute
            },
            optimization_suggestions: self.suggest_optimizations(
                arithmetic_intensity,
                efficiency,
            ),
        }
    }

    fn suggest_optimizations(&self, ai: f64, eff: f64) -> Vec<String> {
        let mut suggestions = Vec::new();

        if ai < self.ridge_point {
            // Memory-bound
            suggestions.push("Kernel is memory-bound. Consider:".into());
            suggestions.push("  - Data compression (quantization) [22, 23]".into());
            suggestions.push("  - Memory coalescing [33]".into());
            suggestions.push("  - Shared memory tiling [16]".into());
        } else if eff < 0.5 {
            // Compute-bound but low efficiency
            suggestions.push("Kernel is compute-bound but underutilized. Consider:".into());
            suggestions.push("  - Increase occupancy".into());
            suggestions.push("  - Reduce warp divergence [34, 35]".into());
            suggestions.push("  - Use Tensor Cores [39, 40]".into());
        }

        suggestions
    }
}
```

---

## 10. Safety Guarantees

### 10.1 Rust's Type System for GPU Safety

Unlike C++ CUDA, trueno-gpu leverages Rust's type system:

```rust
/// GPU buffer with lifetime tracking
/// Prevents use-after-free at compile time
pub struct GpuBuffer<'ctx, T> {
    ptr: DevicePtr<T>,
    len: usize,
    _context: PhantomData<&'ctx GpuContext>,
}

impl<'ctx, T> GpuBuffer<'ctx, T> {
    /// Buffer is tied to context lifetime
    /// Cannot outlive the GPU context that created it
    pub fn new(ctx: &'ctx GpuContext, len: usize) -> Self {
        let ptr = ctx.allocate::<T>(len);
        Self {
            ptr,
            len,
            _context: PhantomData,
        }
    }
}

/// Stream with command ordering guarantees
pub struct GpuStream<'ctx> {
    handle: CuStream,
    /// Pending operations - must complete before stream is dropped
    pending: Vec<PendingOp>,
    _context: PhantomData<&'ctx GpuContext>,
}

impl<'ctx> Drop for GpuStream<'ctx> {
    fn drop(&mut self) {
        // Synchronize on drop - no dangling async operations
        self.synchronize();
    }
}

/// Launch configuration with compile-time bounds checking
pub struct LaunchConfig<const BLOCK_X: u32, const BLOCK_Y: u32, const BLOCK_Z: u32> {
    grid: (u32, u32, u32),
}

impl<const BX: u32, const BY: u32, const BZ: u32> LaunchConfig<BX, BY, BZ> {
    /// Block dimensions are const generics - verified at compile time
    pub const fn new(grid_x: u32, grid_y: u32, grid_z: u32) -> Self {
        // Compile-time assertion
        const { assert!(BX * BY * BZ <= 1024, "Block size exceeds maximum") };
        Self {
            grid: (grid_x, grid_y, grid_z),
        }
    }
}
```

### 10.2 Poka-Yoke: Typestate Pattern for GPU State Machines [50, 51]

**Poka-Yoke** (mistake-proofing) ensures invalid GPU states are **unrepresentable at compile time**. The typestate pattern encodes the GPU stream state machine in Rust's type system:

```rust
/// GPU stream state machine using typestates
/// Per Strom & Yemini [50] and Aldrich et al. [51]
///
/// State transitions:
///   Idle ──launch()──▶ Recording ──sync()──▶ Idle
///                           │
///                     submit()
///                           ▼
///                      Submitted ──wait()──▶ Idle

/// Marker types for stream states (zero-sized)
pub mod states {
    pub struct Idle;       // Ready to record commands
    pub struct Recording;  // Actively recording commands
    pub struct Submitted;  // Commands submitted, awaiting completion
}

/// GPU stream with compile-time state tracking
pub struct GpuStream<'ctx, State> {
    handle: CuStream,
    _context: PhantomData<&'ctx GpuContext>,
    _state: PhantomData<State>,
}

impl<'ctx> GpuStream<'ctx, states::Idle> {
    /// Create new stream in Idle state
    pub fn new(ctx: &'ctx GpuContext) -> Self {
        Self {
            handle: ctx.create_stream(),
            _context: PhantomData,
            _state: PhantomData,
        }
    }

    /// Begin recording commands - transitions Idle → Recording
    pub fn begin(self) -> GpuStream<'ctx, states::Recording> {
        GpuStream {
            handle: self.handle,
            _context: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<'ctx> GpuStream<'ctx, states::Recording> {
    /// Launch kernel - only valid in Recording state
    pub fn launch_kernel<K: Kernel>(
        &mut self,
        kernel: &K,
        config: LaunchConfig,
    ) -> &mut Self {
        unsafe { cuLaunchKernel(self.handle, kernel.ptr(), config) };
        self
    }

    /// Transfer H2D - only valid in Recording state
    pub fn copy_h2d<T>(
        &mut self,
        dst: &GpuBuffer<'ctx, T>,
        src: &[T],
    ) -> &mut Self {
        unsafe { cuMemcpyHtoDAsync(dst.ptr(), src.as_ptr(), self.handle) };
        self
    }

    /// Submit commands - transitions Recording → Submitted
    pub fn submit(self) -> GpuStream<'ctx, states::Submitted> {
        GpuStream {
            handle: self.handle,
            _context: PhantomData,
            _state: PhantomData,
        }
    }

    /// Synchronize immediately - transitions Recording → Idle
    pub fn sync(self) -> GpuStream<'ctx, states::Idle> {
        unsafe { cuStreamSynchronize(self.handle) };
        GpuStream {
            handle: self.handle,
            _context: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<'ctx> GpuStream<'ctx, states::Submitted> {
    /// Wait for completion - transitions Submitted → Idle
    pub fn wait(self) -> GpuStream<'ctx, states::Idle> {
        unsafe { cuStreamSynchronize(self.handle) };
        GpuStream {
            handle: self.handle,
            _context: PhantomData,
            _state: PhantomData,
        }
    }

    /// Check if complete (non-blocking)
    pub fn is_complete(&self) -> bool {
        unsafe { cuStreamQuery(self.handle) == CUDA_SUCCESS }
    }
}

// Compile-time error examples:
//
// let stream = GpuStream::new(&ctx);  // Idle
// stream.launch_kernel(&k, cfg);       // ERROR: no method `launch_kernel` for Idle
//
// let recording = stream.begin();      // Recording
// recording.wait();                    // ERROR: no method `wait` for Recording
```

**Benefits of Typestate Pattern**:
1. **Invalid states impossible**: Can't launch kernel on Idle stream (compile error)
2. **No runtime checks**: State encoded in types, zero overhead
3. **Self-documenting**: State machine visible in function signatures
4. **Prevents resource leaks**: Submitted stream must be waited on

---

## 11. Performance Targets

### 11.1 Benchmark Targets

| Kernel | Metric | Target | cuBLAS Reference | Gap Tolerance |
|--------|--------|--------|------------------|---------------|
| SGEMM 4096×4096 | TFLOPS | ≥65 | 82 | ≤20% |
| Q4_K GEMM 4096×4096 | TFLOPS | ≥40 | N/A (custom) | N/A |
| Softmax 32K | GB/s | ≥800 | 900 | ≤12% |
| FlashAttn 2K×64 | TFLOPS | ≥50 | 70 (FA2) | ≤30% |
| LayerNorm 4096 | GB/s | ≥900 | 950 | ≤6% |

### 11.2 CTA WMMA GEMM Performance (RTX 4090, measured 2026-04-04)

Pure Rust PTX tensor core GEMM (FP16 input, FP32 accumulation):

| Size | Previous | Optimized | Improvement | cuBLAS | Gap |
|------|----------|-----------|-------------|--------|-----|
| 128×128 | 9.6µs | 5.7µs | **1.68x** | 3.4µs | 1.7x |
| 256×256 | 17.3µs | 9.1µs | **1.90x** | 4.3µs | 2.1x |
| 512×512 | 35.7µs / 7.5 TFLOP/s | 23.2µs / 11.6 TFLOP/s | **1.54x** | 7.7µs | 3.0x |

**Optimizations applied (PERF-CTA-001 through PERF-CTA-006):**
1. Fixed WMMA MMA layout: `row.col` → `row.row` to match RowMajor B loads
2. Fully unrolled cooperative load (eliminated 8 loop branches per thread per K-tile)
3. Warp-uniform branching (warps 0-1 load A, warps 2-3 load B, no divergence)
4. Interior tile fast path (skip boundary checks for tiles fully within matrix)
5. Removed `.maxnreg 64` constraint (let JIT optimize register allocation)
6. Pre-computed per-element byte addresses outside K-loop (3 inst/element vs 6)

### 11.2 Quality Gates

```rust
/// Benchmark regression detection
/// Per Hoefler & Belli [17]
pub struct PerformanceGate {
    pub baseline: BenchmarkResult,
    pub regression_threshold: f64,  // 5% default
}

impl PerformanceGate {
    pub fn check(&self, current: &BenchmarkResult) -> GateResult {
        let regression = (self.baseline.median - current.median) / self.baseline.median;

        if regression > self.regression_threshold {
            GateResult::Failed {
                baseline: self.baseline.median,
                current: current.median,
                regression_pct: regression * 100.0,
            }
        } else {
            GateResult::Passed
        }
    }
}
```

---

## 12. Implementation Roadmap

### 12.1 Sprint Planning: TRUENO-GPU-001

**Sprint Goal**: PTX generation + basic GEMM kernel achieving ≥50% cuBLAS performance

**Duration**: 3 weeks

| ID | Task | Effort | Acceptance Criteria |
|----|------|--------|---------------------|
| TG-001 | PTX builder API | 3 days | Generate valid PTX for vector_add |
| TG-002 | CUDA driver FFI (minimal) | 2 days | cuModuleLoad, cuLaunchKernel working |
| TG-003 | Memory management | 2 days | Allocate, transfer, free with no leaks |
| TG-004 | SGEMM naive kernel | 2 days | Correct output, any performance |
| TG-005 | SGEMM tiled kernel | 3 days | ≥30 TFLOPS (shared memory tiling) |
| TG-006 | SGEMM coalesced | 2 days | ≥50 TFLOPS (memory coalescing) |
| TG-007 | Benchmark harness | 1 day | Criterion.rs + roofline analysis |

### 12.2 Future Sprints

- **TRUENO-GPU-002**: Q4_K kernels, Tensor Core WMMA
- **TRUENO-GPU-003**: FlashAttention, fused kernels
- **TRUENO-GPU-004**: Multi-GPU, async pipelining
- **TRUENO-GPU-005**: Metal backend (Apple Silicon)

---

## 13. References

### Memory and Bandwidth

[19] S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65-76, 2009. DOI: 10.1145/1498765.1498785

[21] W. A. Wulf and S. A. McKee, "Hitting the Memory Wall: Implications of the Obvious," *ACM SIGARCH Computer Architecture News*, vol. 23, no. 1, pp. 20-24, 1995. DOI: 10.1145/216585.216588

[31] Y. Yuan et al., "LLM Inference Unveiled: Survey and Roofline Model Insights," arXiv:2402.16363, 2024.

[33] NVIDIA Corporation, "CUDA C++ Best Practices Guide: Memory Coalescing," NVIDIA Documentation, 2024.

### Quantization

[22] G. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," in *ICML*, 2023. arXiv:2211.10438

[23] J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," in *MLSys*, 2024. arXiv:2306.00978

[24] E. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," in *ICLR*, 2023. arXiv:2210.17323

### Warp Divergence and SIMT

[34] W. W. L. Fung, I. Sham, G. Yuan, and T. M. Aamodt, "Warp-Level Divergence in GPUs: Characterization, Impact, and Mitigation," in *HPCA*, 2014.

[35] S. Damani et al., "GPU Subwarp Interleaving," in *HPCA*, 2022.

[38] J. Luitjens, "Faster Parallel Reductions on Kepler," NVIDIA Developer Blog, 2014.

### PTX and Compilation

[32] Rust-GPU Project, "Rust CUDA: Ecosystem of libraries and tools for writing GPU code in Rust," GitHub, 2025. https://github.com/Rust-GPU/Rust-CUDA

[36] NVIDIA Corporation, "PTX ISA Version 8.0," NVIDIA Documentation, 2024.

[37] S. Chetlur et al., "cuDNN: Efficient Primitives for Deep Learning," arXiv:1410.0759, 2014.

### Tensor Cores

[30] P. Micikevicius et al., "Mixed Precision Training," in *ICLR*, 2018. arXiv:1710.03740

[39] S. Markidis et al., "NVIDIA Tensor Core Programmability, Performance & Precision," in *IPDPSW*, 2018.

[40] H. Ootomo and R. Yokota, "Recovering Single Precision Accuracy from Tensor Cores While Surpassing the FP32 Theoretical Peak Performance," *IJHPCA*, vol. 36, no. 4, pp. 475-491, 2022.

[41] M. Fasi, N. J. Higham, M. Mikaitis, and S. Pranesh, "Numerical Behavior of NVIDIA Tensor Cores," *PeerJ Computer Science*, 7:e330, 2021.

### Attention and Transformers

[12] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *SOSP '23*, 2023. DOI: 10.1145/3600006.3613165

[16] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *NeurIPS*, 2022. arXiv:2205.14135

[42] T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691, 2023.

### Benchmarking Methodology

[5] J. Vitek and T. Kalibera, "Repeatability, Reproducibility, and Rigor in Systems Research," in *EMSOFT*, 2011. DOI: 10.1145/2038642.2038650

[6] C. Collberg and T. A. Proebsting, "Repeatability in Computer Systems Research," *Communications of the ACM*, vol. 59, no. 3, pp. 62-69, 2016. DOI: 10.1145/2812803

[17] T. Hoefler and R. Belli, "Scientific Benchmarking of Parallel Computing Systems," in *SC '15*, 2015. DOI: 10.1145/2807591.2807644

### Additional References

[43] A. Ivanov et al., "Data Movement Is All You Need: A Case Study on Optimizing Transformers," in *MLSys*, 2021.

[44] R. Pope et al., "Efficiently Scaling Transformer Inference," in *MLSys*, 2023.

[45] Y. Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," in *ICML*, 2023.

### Toyota Way Engineering Review (v1.1 Additions)

[46] V. Volkov, "Better Performance at Lower Occupancy," in *GPU Technology Conference (GTC)*, 2010. [ILP over Occupancy - seminal work showing instruction-level parallelism beats high occupancy]

[47] S. Xiao and W. Feng, "Inter-Block GPU Communication via Fast Barrier Synchronization," in *IEEE IPDPS*, 2010. DOI: 10.1109/IPDPS.2010.5470477 [Register pressure and liveness analysis for GPU kernels]

[48] G. Ruetsch and P. Micikevicius, "Optimizing Matrix Transpose in CUDA," NVIDIA Technical Report, 2009. [Bank conflict avoidance via padding - foundational CUDA optimization]

[49] R. Nath and S. Tomov, "An Improved MAGMA GEMM for Fermi Graphics Processing Units," *International Journal of High Performance Computing Applications*, vol. 24, no. 4, pp. 511-515, 2010. DOI: 10.1177/1094342010385729 [XOR-based swizzling for bank conflict elimination]

[50] R. E. Strom and S. Yemini, "Typestate: A Programming Language Concept for Enhancing Software Reliability," *IEEE Transactions on Software Engineering*, vol. SE-12, no. 1, pp. 157-171, 1986. DOI: 10.1109/TSE.1986.6312929 [Original typestate paper - compile-time state machine verification]

[51] J. Aldrich, V. Kostadinov, and C. Chambers, "Alias Annotations for Program Understanding," in *OOPSLA '02*, 2002. DOI: 10.1145/582419.582448 [Typestate extensions for object-oriented languages]

[52] G. C. Necula, "Proof-Carrying Code," in *POPL '97*, 1997. DOI: 10.1145/263699.263712 [Foundation for typed intermediate representations - Jidoka inspiration]

[53] X. Leroy, "Formal Verification of a Realistic Compiler," *Communications of the ACM*, vol. 52, no. 7, pp. 107-115, 2009. DOI: 10.1145/1538788.1538814 [CompCert - verified compilation, typed IR validation]

[54] J. A. Stratton et al., "Parboil: A Revised Benchmark Suite for Scientific and Commercial Throughput Computing," *IMPACT Technical Report*, 2012. [GPU kernel benchmarks with bank conflict analysis]

[55] NVIDIA Corporation, "CUDA Occupancy Calculator," NVIDIA Developer Tools Documentation, 2024. [Official tool for register pressure and occupancy analysis]

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-10 | Batuta Team | Initial specification with 25+ peer-reviewed citations |
| 1.1 | 2025-12-10 | Batuta Team | Toyota Way review: +Poka-Yoke typestates (10.2), +Bank conflicts (3.3), +Register pressure (2.2), +ILP over Occupancy, +10 citations [46-55] |

**Next Steps**:
1. Create `trueno-gpu` sub-crate in trueno workspace
2. Implement PTX builder (TG-001)
3. Write acceptance tests for each kernel
4. Begin GEMM optimization journey

---

## 14. Dimension-Independent Kernel Architecture (trueno#203)

### 14.1 Current Architecture: Dynamic PTX with Baked Dimensions

The current kernel generation system bakes matrix dimensions (M, K, N) directly into PTX source code. Each unique shape produces a distinct PTX module that must be JIT-compiled by the CUDA driver:

```
Current Flow:
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Kernel Request  │───▶│ Generate PTX     │───▶│ cuModuleLoadData │
│  (M=2560, K=896) │    │ with baked dims  │    │ (JIT compile)    │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                        50+ kernel variants              │
                        (one per unique shape)            ▼
                                                 ┌──────────────┐
                                                 │ GPU Execution │
                                                 └──────────────┘
```

**Problems with this approach**:
- **50+ kernel variants**: Every new dimension triplet generates a new PTX module
- **JIT overhead**: Each variant requires runtime PTX-to-SASS compilation (~50-200ms per kernel)
- **Blackwell JIT bug (trueno#200)**: `cuModuleLoadDataEx` fails during active GPU work on sm_121
- **Cache pressure**: Large number of compiled kernels consumes GPU code cache

### 14.2 Target Architecture: Dimension-Independent Kernels

Kernels accept M, K, N as runtime parameters via kernel arguments rather than compile-time constants. This reduces the total kernel count to approximately **15 kernel types**:

| # | Kernel Type | Direction | Notes |
|---|-------------|-----------|-------|
| 1 | GEMM (tiled) | Forward | General matrix multiply |
| 2 | GEMM (NF4 fused) | Forward | Dequant + matmul fused |
| 3 | Softmax | Forward | Warp shuffle reduction |
| 4 | LayerNorm (RMS) | Forward | Fused RMSNorm |
| 5 | RoPE | Forward | Rotary position embedding |
| 6 | QK-Norm | Forward | Per-head RMS normalization |
| 7 | SiLU | Forward | Activation function |
| 8 | GEMM backward | Backward | dL/dW, dL/dx |
| 9 | Softmax backward | Backward | Jacobian-vector product |
| 10 | LayerNorm backward | Backward | dL/dgamma, dL/dbeta, dL/dx |
| 11 | RoPE backward | Backward | Reverse rotation |
| 12 | LoRA forward | Forward | Low-rank adapter matmul |
| 13 | LoRA backward | Backward | Adapter gradient accumulation |
| 14 | Cross-entropy loss | Forward | Log-softmax + NLL |
| 15 | Adam update | Optimizer | Fused parameter update |

```
Target Flow:
┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  build.rs       │───▶│ nvcc --cubin          │───▶│ include_bytes!() │
│  (compile time) │    │ 15 kernels × 3 archs  │    │ (embedded blobs) │
└─────────────────┘    └──────────────────────┘    └──────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  Kernel Request  │───▶│ cuModuleLoadData     │───▶│ cuLaunchKernel   │
│  (M=2560, K=896) │    │ (pre-compiled cubin) │    │ (M,K,N as args)  │
└─────────────────┘    └──────────────────────┘    └──────────────────┘
                        Zero JIT compilation          Dims via params
```

### 14.3 Pre-Compilation Pipeline

```bash
# build.rs pseudocode
for arch in [sm_80, sm_89, sm_121] {
    for kernel in [gemm, softmax, layernorm, ...] {
        nvcc --cubin --gpu-architecture={arch} {kernel}.ptx -o {kernel}_{arch}.cubin
    }
}
```

The pre-compiled cubin blobs are embedded via `include_bytes!()`:
```rust
const GEMM_SM80: &[u8] = include_bytes!("cubins/gemm_sm80.cubin");
const GEMM_SM89: &[u8] = include_bytes!("cubins/gemm_sm89.cubin");
const GEMM_SM121: &[u8] = include_bytes!("cubins/gemm_sm121.cubin");
```

### 14.4 Blackwell Compatibility

This architecture **eliminates the Blackwell JIT bug entirely**:
- No `cuModuleLoadDataEx` calls at runtime (no PTX JIT)
- Pre-compiled cubins load via `cuModuleLoadData` which reads binary, no compilation
- Safe to load modules at any time, even during active GPU work
- Forward and backward kernels both work without pre-warming

### 14.5 Provable Contract

Contract file: `provable-contracts/contracts/dimension-independent-kernels-v1.yaml`

Key assertions:
- FALSIFY-DIK-001: All 15 kernel types accept M/K/N as runtime params
- FALSIFY-DIK-002: Zero `cuModuleLoadDataEx` calls after initialization
- FALSIFY-DIK-003: cubin blobs present for sm_80, sm_89, sm_121
- FALSIFY-DIK-004: Dimension-independent GEMM produces identical results to baked-dimension GEMM
- FALSIFY-DIK-005: Backward kernels work during active GPU training (no JIT crash)

---

## 15. WGSL Backward Shaders for Training (WGPU)

### 15.1 Motivation

The CUDA PTX backward kernels (Section 14) solve the Blackwell JIT bug for NVIDIA hardware, but training on AMD, Intel Arc, and Apple Silicon GPUs requires a different approach. WGSL compute shaders running on the `wgpu` backend provide cross-platform training support via Vulkan, Metal, DX12, and WebGPU.

### 15.2 Architecture

```
trueno/src/backends/gpu/
├── shaders/
│   ├── basic_ops.rs      # Forward pass shaders (GEMV, MATMUL, RMSNorm, SiLU, RoPE)
│   └── backward.rs       # Backward pass shaders (7 WGSL compute shaders)
└── device/
    ├── mod.rs             # GpuDevice (forward dispatch)
    └── backward.rs        # Backward dispatch functions (7 GPU dispatch methods)
```

### 15.3 Backward Ops (7 WGSL Compute Shaders)

| Shader | Workgroup Size | Description |
|--------|---------------|-------------|
| `silu_backward` | 256 | SiLU activation gradient: `grad * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))` |
| `gemm_backward_a` | 16x16 | Weight gradient dL/dA via tiled shared-memory GEMM |
| `gemm_backward_b` | 16x16 | Input gradient dL/dB via tiled shared-memory GEMM |
| `rmsnorm_backward` | 256 | RMSNorm gradient with per-row reduction |
| `rope_backward` | 256 | Rotary position embedding gradient (inverse rotation) |
| `adamw_step` | 256 | AdamW optimizer: first/second moment update + decoupled weight decay |
| `nf4_dequant` | 256 | NF4 4-bit dequantization via lookup table (for QLoRA training) |

### 15.4 Verification

All shaders verified on AMD Radeon Pro W5700X via Vulkan backend.

**Contract**: `bashrs/provable-contracts/contracts/wgpu-training-v1.yaml`

8 FALSIFY tests, all PASS:
- Numerical correctness vs CPU reference (f32 tolerance < 1e-5)
- Dimension handling (non-power-of-2 sizes)
- Buffer management (upload/download round-trip)

### 15.5 Cross-Platform GPU Training Path

With both forward (Section 12) and backward (this section) WGSL shaders, trueno provides a complete training loop on non-NVIDIA hardware:

```
Forward Pass (WGSL):  RMSNorm → GEMV/GEMM → SiLU → RoPE → Attention → Loss
Backward Pass (WGSL): Loss grad → RMSNorm backward → GEMM backward → SiLU backward → RoPE backward
Optimizer (WGSL):     AdamW step (with NF4 dequant for QLoRA)
```

This eliminates the NVIDIA-only dependency for training workloads, enabling the Sovereign AI Stack to train models on AMD Radeon, Intel Arc, and Apple M-series GPUs.
