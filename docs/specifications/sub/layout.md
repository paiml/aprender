# Sub-spec: LAYOUT-002 Row-Major Mandate

**Parent:** [trueno-spec.md](../trueno-spec.md) Section 9

---

## 1. The Rule

The Sovereign AI Stack uses **row-major layout exclusively** for APR/GGUF data. Trueno provides both row-major and column-major Q4K/Q6K kernels, but column-major is for internal BLAS-style operations only.

## 2. Kernel Selection

| Layout | Use Case | Consumers |
|--------|----------|-----------|
| **Row-major** | APR format, SafeTensors, PyTorch | aprender, realizar |
| Column-major | Internal BLAS ops, transposed matmul | Advanced/internal only |

**For APR/GGUF data, ALWAYS use row-major kernels.**

## 3. Data Pipeline

Trueno does NOT handle layout conversion. Aprender transposes during import:

```
GGUF (column-major) → aprender transpose → APR (row-major) → realizar → trueno row-major kernels
```

Aprender converter: `src/format/converter/write.rs`

## 4. Diagnosing Layout Bugs

If inference produces garbage output (e.g., `"olumbia+lsi nunca/localENTS"`):

1. Check if column-major kernel was called with row-major data
2. Verify APR file was created via `apr import` (not raw GGUF passthrough)
3. Cross-reference: `aprender/CLAUDE.md` LAYOUT-002, `realizar/CLAUDE.md` LAYOUT-002

## 5. Fused Q4K

See `book/src/advanced/phase15-fused-q4k.md` for the fused dequant+dot kernel specification targeting 2x Ollama throughput. The fused kernel eliminates intermediate materialization of dequantized weights.

## 6. Q4K Super-Block Format

Super-blocks are 256 elements (144 bytes). `in_dim` MUST be a multiple of 256 or `matmul_q4k_f32_scalar` panics. This is a hard constraint from the GGML quantization format.
