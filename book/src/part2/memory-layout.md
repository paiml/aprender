# Memory Layout

The Sovereign AI Stack enforces a row-major tensor layout across all components. This is a critical architectural decision documented as LAYOUT-002 that affects aprender, realizar, and all model conversion pipelines.

## LAYOUT-002: Row-Major Mandate

All tensors in the stack use row-major (C-style) memory layout. External formats that use column-major layout are transposed at import time.

```
External Formats                    Stack Internal (Row-Major)
----------------                    -------------------------
SafeTensors (row-major) ----------> APR v2 --> realizar --> output
                         (native)       ^
GGUF (column-major) ---------------/
                    (transposed by aprender)
```

## Why Row-Major

Three factors drive this decision:

1. **PyTorch/SafeTensors compatibility** -- HuggingFace models are natively row-major. No conversion needed for the most common import path.

2. **Cache efficiency** -- Row-major matches C memory layout. When iterating over rows (the common case in matrix-vector products), data is contiguous in memory, maximizing L1/L2 cache utilization.

3. **Kernel simplicity** -- Realizar's fused quantization kernels (`fused_q4k_parallel_matvec`, `fused_q6k_parallel_matvec`) assume row-major layout. A single layout eliminates runtime branching.

## Component Responsibilities

| Component | Role |
|-----------|------|
| **aprender** | Transposes GGUF column-major data to row-major during `apr import` |
| **realizar** | Assumes row-major layout in all inference kernels |
| **trueno** | Provides both column-major and row-major kernels; APR code uses row-major |

## Diagnosing Layout Bugs

If model output produces garbage text like `"olumbia+lsi nunca/localENTS"` instead of coherent language, the root cause is almost always a layout mismatch: column-major data fed to a row-major kernel.

**Fix:** Ensure the model was converted through aprender's GGUF converter, which transposes weight matrices to row-major.

## Cache-Friendly Access Patterns

Row-major layout means elements in the same row are contiguous:

```
Row-major [3x4]:
  [a b c d | e f g h | i j k l]
   row 0     row 1     row 2

Column-major [3x4]:
  [a e i | b f j | c g k | d h l]
   col 0   col 1   col 2   col 3
```

For a matrix-vector product `y = Wx`, each output element computes `dot(row_i, x)`. In row-major layout, `row_i` is a contiguous memory span, which the CPU prefetcher handles efficiently.

## Quantized Tensor Layout

Quantized formats (Q4K, Q6K) store data in 256-element blocks. Each block contains scales, minimums, and quantized values packed together. The block layout is row-major at the block level:

| Format | Block Size | Bytes per Block | Per-Row Blocks |
|--------|-----------|----------------|----------------|
| Q4K | 256 elements | 144 bytes | ceil(dim / 256) |
| Q6K | 256 elements | 210 bytes | ceil(dim / 256) |

## APR v2 Format

The APR v2 binary format stores tensors with 64-byte alignment for zero-copy memory mapping. Metadata (including layout information) is padded to 64-byte boundaries:

```
[header] [metadata (64-byte aligned)] [tensor data (64-byte aligned)]
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
