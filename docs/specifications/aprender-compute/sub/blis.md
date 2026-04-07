# Sub-spec: BLIS GEMM Engine

**Parent:** [trueno-spec.md](../trueno-spec.md) Section 18

---

## 1. Overview

`src/blis/` implements BLIS-style blocked GEMM (Goto & van de Geijn 2008) with cache hierarchy optimization. This is the core matrix multiplication engine for CPU backends.

## 2. Block Sizes

| Parameter | Value | Purpose |
|-----------|-------|---------|
| MR | 8 | Micro-kernel rows (register tile) |
| NR | 6 | Micro-kernel columns |
| KC | 256 | L1 cache block (inner dimension) |
| MC | 72 | L2 cache block (rows of A) |
| NC | 4096 | L3 cache block (columns of B) |

## 3. Micro-kernels

| Kernel | ISA | Tile | File |
|--------|-----|------|------|
| `microkernel_8x6_avx2` | AVX2+FMA | 8x6 | `src/blis/` |
| `microkernel_8x6_true_asm` | AVX2 inline asm | 8x6 | `src/blis/` |
| `microkernel_8x8_neon` | ARM NEON | 8x8 | `src/blis/` |

## 4. Packing

`pack_a()` and `pack_b()` reorder matrices into cache-friendly MR-wide and NR-wide panels. `PrepackedB` caches packed weight matrices across inference calls (amortizes packing cost).

## 5. Toyota Production System Integration

- **Jidoka:** `JidokaGuard` validates numerical output against scalar reference. Stops on NaN or divergence >1e-3.
- **Heijunka:** `HeijunkaScheduler` distributes MC-blocks across threads for balanced parallel GEMM.
- **Kaizen:** `BlisProfiler` with `BlisLevelStats` tracks timing at every BLIS hierarchy level (L3→L2→L1→micro).

## 6. Backend Cost Model

`BackendCostModel` uses roofline analysis to select between BLIS GEMM, wgpu GEMM, and CUDA GEMM based on matrix dimensions and hardware capabilities.

## 7. API

```rust
// Standard GEMM (writes result into c in-place)
blis::gemm(m, n, k, a, b, c) -> Result<(), TruenoError>

// Profiled GEMM (profiler collects per-level timing)
blis::gemm_profiled(m, n, k, a, b, c, &mut profiler) -> Result<(), TruenoError>

// Parallel GEMM with prepacked weights
blis::gemm_blis_parallel(m, n, k, a, b, c) -> Result<(), TruenoError>
```
