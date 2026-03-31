# Appendices B through G (Archived)

> Archived from qwen2.5-coder-showcase-demo.md, Appendices B-G (lines 2854-2965).

## Appendix B: PMAT Work Tickets

| Ticket | Title | Status |
|--------|-------|--------|
| T-QA-001 | Coverage Infrastructure | Done |
| T-QA-002 | CLI Refactor (Extreme TDD) | Done |
| T-QA-003 | CUDA Live Testing | Done |
| T-QA-007-016 | Coverage Gaps | Done |
| T-QA-017 | CUDA Heavy Integration | Done (PMAT-116) |
| T-QA-018-022 | Resource Efficiency | Done |
| PMAT-116 | SafeTensors GPU Inference | Done (Zero SATD) |
| PMAT-085 | File Health: optim/mod.rs | Done (2848->2022 lines) |
| PMAT-206 | GH-189: APR BpeTokenizer Special Tokens | Done (realizar v0.6.11) |
| PMAT-235 | Validated Tensor Newtypes (Poka-Yoke) | Done |
| PMAT-237 | Pre-Dispatch Contract Gate | Done |

---

## Appendix C: Open GitHub Issues

> **Toyota Way:** These are NOT "tech debt." These are **known defects** honestly documented.

### P0 Defects

**All historical P0 defects RESOLVED.** See [v9-1.5b-results.md](v9-1.5b-results.md) for details.

| Issue | Title | Status |
|-------|-------|--------|
| #162 | Pulled models don't show in `apr list` | Open (cache directory mismatch) |

### P1/P2 Open

| Issue | Title | Priority | Status |
|-------|-------|----------|--------|
| #159 | Convolution Layout Optimization | P2 | Open |
| #149 | Lottery Ticket Hypothesis pruning | P2 | Open |
| #144 | Synthetic noise generation | P3 | Open |
| #141 | Y7: GPU Performance Benchmarks | P2 | Open |

---

## Appendix F: Q4_K Quantization Format Specification

### F.1 Overview (from llama.cpp)

Q4_K is a mixed-precision 4-bit quantization format used by GGUF. Each **superblock** contains 256 elements.

**Source:** `llama.cpp/ggml/src/ggml-quants.c`

### F.2 Superblock Structure (144 bytes per 256 elements)

| Field | Bytes | Description |
|-------|-------|-------------|
| `d` | 2 | Scale factor (f16) |
| `dmin` | 2 | Minimum value (f16) |
| `scales` | 12 | Per-block scales (6-bit packed) |
| `qs` | 128 | Quantized values (4-bit packed, 256 elements) |
| **Total** | **144** | Per superblock |

### F.3 Dequantization Algorithm

```c
// From llama.cpp/ggml/src/ggml-quants.c
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k) {
    for (int i = 0; i < nb; i++) {
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        // Dequantize: y = d * scale * q - min * scale_min
        for (int j = 0; j < QK_K/2; ++j) {
            y[j]        = d * sc[0] * (q[j] & 0xF) - min * m[0];
            y[j + QK_K/2] = d * sc[1] * (q[j] >> 4)  - min * m[1];
        }
    }
}
```

### F.4 Size Calculation

For a weight matrix `[out_dim, in_dim]`:
```
num_superblocks = out_dim * ceil(in_dim / 256)
total_bytes = num_superblocks * 144
```

---

## Appendix G: SafeTensors Format Specification

### G.1 File Layout

```
+------------------------------------------+
| Header Length (8 bytes, u64 LE)          |
+------------------------------------------+
| JSON Metadata (variable length)          |
|   - Tensor names -> {dtype, shape, offsets} |
|   - Optional __metadata__ section        |
+------------------------------------------+
| Tensor Data (contiguous, aligned)        |
+------------------------------------------+
```

### G.2 Supported Data Types

| Type | Bytes | Description |
|------|-------|-------------|
| `BF16` | 2 | Brain float 16 (primary for 7B) |
| `F32` | 4 | 32-bit float |
| `F16` | 2 | 16-bit float |
| `I8` | 1 | 8-bit signed int |

---
