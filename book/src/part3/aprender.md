# Aprender

Aprender is the ML library for the Sovereign AI Stack, providing training algorithms, model formats, and format conversion utilities.

## Key Features

- **Algorithms**: Linear regression, logistic regression, k-means, decision trees, random forests, gradient boosting, SVM, KNN, Naive Bayes, PCA
- **Formats**: APR v2 native format, SafeTensors import, GGUF import
- **Quantization**: Q4_K, Q5_K, Q6_K encoding with row-padded super-blocks

## LAYOUT-002: Row-Major Mandate

**Critical**: Aprender handles all layout conversion for the Sovereign AI Stack.

### Format Conversion Architecture

```
┌─────────────────────────────────────────────────────────┐
│         APRENDER FORMAT CONVERTER                        │
│         src/format/converter/write.rs                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  SafeTensors (row-major) ───(pass-through)───► APR v2   │
│                                                          │
│  GGUF (column-major) ───(TRANSPOSE)───► APR v2          │
│                         dequant→transpose→requant        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `transpose_q4k_for_matmul` | `mod.rs:1273` | GGUF Q4K → row-major Q4K |
| `transpose_q6k_for_matmul` | `mod.rs:1311` | GGUF Q6K → row-major Q6K |
| `quantize_q4_k_matrix` | `mod.rs:1195` | Row-padded Q4K encoding |

### Transpose Process

1. **Dequantize**: Q4K bytes → F32 floats
2. **Transpose**: `[rows, cols]` → `[cols, rows]`
3. **Re-quantize**: F32 → Q4K with row-padded super-blocks

### Usage

```bash
# Import GGUF with automatic transpose
apr import model.gguf -o model.apr

# Import SafeTensors (no transpose needed)
apr import model.safetensors -o model.apr
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
