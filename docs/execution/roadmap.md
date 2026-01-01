# PMAT Development Roadmap

## Current Sprint: v0.9.0 Autograd Engine - PyTorch-Compatible Automatic Differentiation
- **Duration**: 2025-11-25 to 2025-12-16
- **Priority**: P0

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated

## Current Sprint: v0.4.0 llama.cpp Performance Parity
- **Duration**: 2026-01-01 to 2026-01-15
- **Priority**: P0
- **Quality Gates**: Complexity ≤ 20, SATD = 0, Coverage ≥ 80%
- **Target**: Achieve <2x gap vs llama.cpp (currently 42x gap)

### Benchmark Baseline (2026-01-01)
| Model | llama.cpp | realizar | Gap |
|-------|-----------|----------|-----|
| TinyLlama-1.1B | 554.89 tok/s | 13.1 tok/s | 42.4x |
| Qwen2-0.5B | 595.22 tok/s | 13.8 tok/s | 43.1x |
| Phi-2 | 290.16 tok/s | 13.1 tok/s | 22.1x |

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| PAR-001 | Fix Q4_K dequantization (garbage output bug) | 🟡 IN PROGRESS | High | P0 |
| PAR-002 | Debug CUDA driver error 700 in attention kernel | 🔴 TODO | High | P0 |
| PAR-003 | Fix CUDA Q4_K matvec PTX module load failure | 🔴 TODO | High | P0 |
| PAR-004 | Implement flash attention on GPU path | 🔴 TODO | High | P1 |
| PAR-005 | Move quantized matmul to GPU (currently CPU dequant) | 🔴 TODO | Medium | P1 |
| PAR-006 | Add GEMM kernel fusion for FFN layers | 🔴 TODO | Medium | P2 |
| PAR-007 | Implement memory pool for GPU allocations | 🔴 TODO | Medium | P2 |
| PAR-008 | Achieve M2 milestone (>24 tok/s, <10x gap) | 🔴 TODO | - | P1 |
| PAR-009 | Achieve M3 milestone (>48 tok/s, <5x gap) | 🔴 TODO | - | P1 |
| PAR-010 | Achieve M4 milestone (>192 tok/s, <1.25x gap) | 🔴 TODO | - | P2 |

### Investigation Notes

**PAR-001: Q4_K Dequantization Bug (2026-01-01)**

Fixes applied to `realizar/src/quantize.rs`:
1. Fixed `extract_scale_min()` to match llama.cpp's `get_scale_min_k4()` packing scheme
2. Removed incorrect `/63.0` normalization (d/dmin already normalized in GGUF header)

Current status: Output still garbage ("uolauola...") - issue may be elsewhere:
- Tensor layout mismatch?
- Forward pass matrix operation ordering?
- RoPE position encoding?
- KV cache initialization?

Next steps:
- [ ] Add numerical verification against llama.cpp dequant output
- [ ] Trace first layer output to find divergence point
- [ ] Compare RoPE implementation

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated

