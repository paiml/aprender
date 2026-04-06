# Tiling Compute Blocks (TCB) Specification

**Version**: 1.0.0
**Status**: DRAFT
**Author**: Trueno Engineering
**Date**: 2026-01-15
**PMAT Roadmap ID**: `TILING-SPEC-001`
**PMAT Tracking**: `pmat work continue TILING-SPEC-001`
**Spec Path**: `docs/specifications/tiling-compute-blocks.md`

**Canonical References**:
- PROBAR-SPEC-009 (Brick Architecture)
- CBTOP-SPEC-001 (ComputeBrick Profiling)
- TUNER-SPEC-001 (ML-Tuner for ComputeBricks)
- SHOWCASE-BRICK-001 (Qwen2.5-Coder Showcase)
- SPEC-024 (Popperian Falsification Protocol)

---

## Table of Contents

| § | Section | Status |
|---|---------|--------|
| [0](#executive-summary) | Executive Summary | - |
| [1](#1-scientific-foundations) | Scientific Foundations | - |
| [2](#2-architecture-overview) | Architecture Overview | - |
| [3](#3-tiling-geometry) | Tiling Geometry | - |
| [4](#4-hierarchical-mapping) | Hierarchical Mapping | - |
| [5](#5-memory-coalescing-strategy) | Memory Coalescing Strategy | - |
| [6](#6-implementation-patterns) | Implementation Patterns | - |
| [7](#7-100-point-popperian-falsification) | 100-Point Popperian Falsification | - |
| [8](#8-pmat-tickets) | PMAT Tickets | - |

---

## Executive Summary

**Tiling Compute Blocks (TCB)** represent the fundamental unit of work partitioning within high-performance `ComputeBrick` kernels. While a `ComputeBrick` defines a logical operation (e.g., Q4_K MatMul), a TCB defines the physical execution strategy—how data is partitioned across the hardware's memory hierarchy (Registers, Shared Memory, L1/L2/L3 caches).

**Core Insight**: The 37x performance gap identified in the B4 Investigation was largely due to **lack of tiling** in the CPU path and **suboptimal tiling** (low occupancy) in early GPU kernels. This specification formalizes the tiling strategies that enabled the >200 tok/s target for Qwen2.5-Coder.

**Design Philosophy**: "Cache-First, Computation-Second." Execution is planned around memory movement, ensuring that every byte loaded from Global Memory is reused maximum times within the TCB.

---

## 1. Scientific Foundations

### 1.1 Temporal and Spatial Locality

The effectiveness of Tiling is grounded in the memory hierarchy literature:

| Citation | Contribution | Relevance |
|----------|--------------|-----------|
| **[1] Lam et al. (1991). "The Cache Performance and Optimizations of Blocked Algorithms."** ASPLOS | Optimal tile size selection for caches | CPU Tiling (AVX2/AVX-512) |
| **[2] Whaley & Dongarra (1998). "Automatically Tuned Linear Algebra Software (ATLAS)."** | Empirical search for tiling parameters | Basis for ML-Tuner integration |
| **[3] Volkov (2010). "Better Performance at Lower Occupancy."** GTC | GPU register pressure vs. tiling | GPU TCB geometry |
| **[4] Dao et al. (2022). "FlashAttention."** NeurIPS | Tiling attention via online softmax | Tiling for Attention Bricks |

### 1.2 The Tiling Identity

For a matrix multiplication $C = AB$, a TCB of size $(M_{tile}, N_{tile}, K_{tile})$ ensures that:
- $M_{tile} 	imes K_{tile}$ of $A$ is loaded once.
- $K_{tile} 	imes N_{tile}$ of $B$ is loaded once.
- $M_{tile} 	imes N_{tile}$ compute operations are performed in high-speed memory.

The goal is to maximize the **Arithmetic Intensity**:
$$AI = \frac{2 \cdot M_{tile} \cdot N_{tile} \cdot K_{tile}}{M_{tile} \cdot K_{tile} + K_{tile} \cdot N_{tile}}$$

---

## 2. Architecture Overview

### 2.1 The TCB Hierarchy

Tiling occurs at three distinct levels, each corresponding to a hardware memory tier:

1.  **Macro-Tile (L3/Global Memory)**: Partitioning the full workload across multiple CPU sockets or GPU SMs.
2.  **Midi-Tile (L2/Shared Memory)**: Partitioning work within a thread block (GPU) or a Rayon task (CPU).
3.  **Micro-Tile (Registers)**: The smallest unit of work processed by SIMD instructions or CUDA warps.

### 2.2 TCB Structure (Rust)

```rust
/// Dimensions for a Tiling Compute Block
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TcbGeometry {
    /// Items processed in M dimension (rows)
    pub m: u32,
    /// Items processed in N dimension (columns)
    pub n: u32,
    /// Reduction dimension (inner product)
    pub k: u32,
    /// Alignment requirement (typically 16 for SIMD/Float4)
    pub alignment: u32,
}

/// A Tiling Compute Block instance
pub struct TilingComputeBlock<'a, T> {
    pub geometry: TcbGeometry,
    pub a_slice: &'a [T],
    pub b_slice: &'a [T],
    pub c_slice: &'a mut [T],
    pub shared_mem: Option<u32>,
}
```

---

## 3. Tiling Geometry

### 3.1 GPU Standard Geometries (NVIDIA Ada/Lovelace)

| Kernel Type | Macro-Tile (M,N,K) | Midi-Tile (Block) | Micro-Tile (Warp) |
|-------------|-------------------|-------------------|-------------------|
| Q4_K MatVec | (1, 4096, 256)    | (1, 256, 1)       | (1, 32, 1)        |
| Q4_K MatMul | (128, 128, 32)    | (32, 32, 1)       | (8, 8, 1)         |
| Softmax     | (1, 32000, 1)     | (1, 1024, 1)      | (1, 32, 1)        |

### 3.2 CPU Standard Geometries (AVX-512)

| Kernel Type | L3 Tile | L2 Tile | Register Tile |
|-------------|---------|---------|---------------|
| MatMul      | 512x512 | 128x128 | 16x4 (zmm0-15)|
| RmsNorm     | N/A     | 4096    | 16 (zmm0)     |

---

## 4. Hierarchical Mapping

### 4.1 Index Calculation

The TCB must handle boundary conditions (when $N \% N_{tile} \neq 0$) without performance-killing branches in the inner loop.

```rust
#[inline]
pub fn get_tcb_offset(
    block_idx: u32, 
    geometry: TcbGeometry, 
    stride: u32
) -> usize {
    let row = (block_idx / (stride / geometry.n)) * geometry.m;
    let col = (block_idx % (stride / geometry.n)) * geometry.n;
    (row * stride + col) as usize
}
```

---

## 5. Memory Coalescing Strategy

### 5.1 Transposed Weights Pattern (TCB-01)

As discovered in SHOWCASE-BRICK-001, tiling only works if the memory layout is optimized for the micro-tile:

- **GPU**: Weights must be stored in `QuantBlock` format (e.g., 32 nibbles + 1 scale) to ensure a single `float4` load services an entire warp micro-tile.
- **CPU**: Weights should be de-interleaved into SIMD-width chunks (e.g., 16 floats for AVX-512) to allow `vmovaps` (aligned packed move).

### 5.2 Shared Memory Swizzling (TCB-02)

To avoid Shared Memory bank conflicts (which serialize access), TCBs must employ XOR-swizzling when mapping 2D tiles to linear shared memory addresses.

**Pattern**: `idx_swizzled = idx ^ (idx >> 5)` (for 32-bank architectures).

### 5.3 Tile Quantization Alignment (TCB-03)

Quantized formats impose strict alignment constraints. A TCB must align with the *Superblock* size of the quantization format:

| Format | Superblock Size | TCB constraint |
|--------|-----------------|----------------|
| Q4_0   | 32              | $K_{tile} \% 32 == 0$ |
| Q4_K   | 256             | $K_{tile} \% 256 == 0$ |
| Q8_0   | 32              | $K_{tile} \% 32 == 0$ |

**Violation**: If $K_{tile} < 256$ for Q4_K, the kernel must load *unwanted* neighbors or perform unaligned reads, destroying throughput.

---

## 6. Implementation Patterns

### 6.1 The "Tile-and-Reduce" Pattern

Used for MatMul where the inner $K$ dimension is larger than high-speed memory.

```rust
// Pseudocode for TCB loop
for k_offset in (0..K).step_by(geometry.k) {
    // 1. Load TCB from Global to Shared/L1
    load_tile_a(shared_a, a_global, k_offset);
    load_tile_b(shared_b, b_global, k_offset);
    
    // 2. Synchronize (Barrier)
    sync_threads();
    
    // 3. Compute Micro-tiles from Shared/L1 to Registers
    compute_micro_tiles(acc, shared_a, shared_b);
    
    // 4. Synchronize before next load
    sync_threads();
}
```

### 6.2 CPU Prefetching with Tiling

Link TCB geometry to hardware prefetcher limits.

```rust
// AVX-512 Prefetch Distance: ~64-128 bytes ahead
let prefetch_dist = geometry.k * 2;
prefetch(a_ptr + prefetch_dist, PrefetchLocality::T0);
```

---

## 7. 100-Point Popperian Falsification

### 7.1 Falsification Categories

| Category | Points | Description |
|----------|--------|-------------|
| **F301-F320** | 20 | Geometry Correctness |
| **F321-F340** | 20 | Boundary Handling |
| **F341-F360** | 20 | Memory Coalescing |
| **F361-F380** | 20 | Hierarchical Efficiency |
| **F381-F400** | 20 | Thread-Safety & Race Conditions |

### 7.2 Selected Falsification Criteria

| ID | Criterion | Threshold | Test Method |
|----|-----------|-----------|-------------|
| F301 | TCB Output Equivalence | Bit-exact vs. Scalar | Property-based test |
| F302 | Tile Size Power of 2 | Required for CUDA | Static analysis |
| F310 | **Occupancy Sweet Spot** | >50% theoretical | `cuda-occupancy-calculator` |
| F311 | **Wave Quantization** | Tail wave > 80% full | Kernel Trace Analysis |
| F312 | **Superblock Alignment** | $K_{tile} \ge 256$ (Q4_K) | Static Assert / Check |
| F313 | **Bank Conflict Rate** | < 1% conflicts | Nsight Compute |
| F314 | **Register Spill** | 0 bytes local mem | `ptxas` analysis |
| F315 | **Warp Divergence** | < 5% divergent branches | Nsight Compute |
| F321 | Odd-Sized Matrix Handling | No out-of-bounds | Fuzzing (M,N,K ∈ [1, 1025]) |
| F322 | Zero-Padding Efficiency | < 5% cycle waste | Cycle counting (RDTSCP) |
| F341 | Memory Coalescing (GPU) | 100% efficient | Nsight Compute Trace |
| F342 | Cache Line Alignment (CPU) | 0 misaligned loads | `perf stat -e mem_inst_retired.split_loads` |
| F361 | L2 Cache Hit Rate | > 90% for TCB-01 | PMU Counters |
| F362 | Register Pressure | < 64 regs/thread | `ptxas --verbose` |
| F381 | Shared Memory Race | 0 Race conditions | `cuda-memcheck --tool racecheck` |
| F382 | Rayon Thread Collision | 0 Data races | `cargo +nightly tsan` |

---

## 8. PMAT Tickets

| ID | Title | Priority | Status |
|----|-------|----------|--------|
| TILE-001 | Implement TcbGeometry struct | P0 | TODO |
| TILE-002 | Add hierarchical index calculator | P0 | TODO |
| TILE-003 | Optimize Q4_K MatVec with TCB-01 | P1 | TODO |
| TILE-004 | Implement AVX-512 register tiling | P1 | TODO |
| TILE-005 | Add F321-F340 (Boundary) tests | P0 | TODO |

---

*Document generated by PMAT specification framework.*
*Falsification checklist aligned with SPEC-024 (Popperian Protocol).*