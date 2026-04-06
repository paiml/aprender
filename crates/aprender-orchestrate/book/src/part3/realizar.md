# Realizar

Realizar is the pure-Rust ML inference engine for the Sovereign AI Stack. It provides high-performance model serving with fused quantized kernels.

## Key Features

- **Format Support**: APR v2, GGUF, SafeTensors
- **Quantization**: Q4_K, Q5_K, Q6_K, Q8_0 with fused dequant+matmul
- **Performance**: Ollama-parity throughput targets (100+ tok/s CPU, 500+ GPU)
- **Architecture**: Qwen2, LLaMA, Mistral, Phi model families

## LAYOUT-002: Row-Major Mandate

**Critical**: Realizar exclusively uses row-major tensor layout.

All GGUF models must be converted to APR format using aprender's converter, which transposes data from GGUF's column-major layout to row-major.

```bash
# Correct workflow
apr import model.gguf -o model.apr
realizar run model.apr --prompt "Hello"

# WRONG - bypasses layout conversion
realizar run model.gguf  # May produce garbage output
```

### Fused Kernels (Row-Major Only)

| Kernel | Purpose | File |
|--------|---------|------|
| `fused_q4k_parallel_matvec` | Q4_K matmul | `src/quantize/fused_k.rs` |
| `fused_q6k_parallel_matvec` | Q6_K matmul | `src/quantize/parallel_k.rs` |

**Never use** trueno's `*_colmajor` variants for APR/GGUF data.

### Garbage Output Diagnosis

If output looks like `"olumbia+lsi nunca/localENTS"`:
1. Check that model was converted via `apr import`
2. Verify APR file (not raw GGUF) is being loaded
3. See `CLAUDE.md` LAYOUT-002 section for details

---

**Navigate:** [Table of Contents](../SUMMARY.md)
