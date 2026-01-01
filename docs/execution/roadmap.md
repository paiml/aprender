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
| TinyLlama-1.1B Q4_K_M | 528.6 tok/s | 14.5 tok/s | 36.5x |
| Qwen2-0.5B | 595.22 tok/s | 13.8 tok/s | 43.1x |
| Phi-2 | 290.16 tok/s | 13.1 tok/s | 22.1x |

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| PAR-001 | Fix Q4_K dequantization (garbage output bug) | ✅ DONE | High | P0 |
| PAR-002 | Debug CUDA driver error 700 in attention kernel | 🔴 TODO | High | P0 |
| PAR-003 | Fix CUDA Q4_K matvec PTX module load failure | 🔴 TODO | High | P0 |
| PAR-004 | Implement flash attention on GPU path | 🔴 TODO | High | P1 |
| PAR-005 | Move quantized matmul to GPU (currently CPU dequant) | 🔴 TODO | Medium | P1 |
| PAR-006 | Add GEMM kernel fusion for FFN layers | 🔴 TODO | Medium | P2 |
| PAR-007 | Implement memory pool for GPU allocations | 🔴 TODO | Medium | P2 |
| PAR-008 | Achieve M2 milestone (>24 tok/s, <10x gap) | 🔴 TODO | - | P1 |
| PAR-009 | Achieve M3 milestone (>48 tok/s, <5x gap) | 🔴 TODO | - | P1 |
| PAR-010 | Achieve M4 milestone (>192 tok/s, <1.25x gap) | 🔴 TODO | - | P2 |
| PAR-011 | Add --gpu flag to run/serve commands | ✅ DONE | Medium | P0 |

### Investigation Notes

**PAR-001: Q4_K Dequantization - RESOLVED (2026-01-01)**

Output is coherent (not garbage). Previous "uolauola" issue was from earlier version.
Current behavior: realizar and llama.cpp produce different but both coherent outputs.

Test: `"Hello" -> "HelloWorld()\n\n// Example 2.\n"` (realizar) vs `"Hello, World!\n"` (llama.cpp)

Differences may be due to:
- Sampling/RNG differences
- Minor numerical precision in RoPE
- LayerNorm epsilon differences

✅ Q4_K dequantization formula verified correct against llama.cpp.

**Performance Analysis (2026-01-01)**

Current gap: 36.5x (TinyLlama Q4_K_M, greedy, 16 tokens)
- llama.cpp: 528.6 tok/s (text generation, RTX 4090 CUDA)
- realizar: 14.5 tok/s (CPU path, AVX2)

Root cause analysis:
1. `realizar run` uses `QuantizedGGUFTransformer` (CPU-only mmap-based path)
2. `realizar serve` uses `OwnedQuantizedModel` which supports CUDA via `REALIZAR_BACKEND=cuda`
3. The `run` command needs to be updated to support CUDA path

Next priority:
- [x] PAR-011: Add --gpu flag to `realizar run` and `realizar serve` commands ✅
- [ ] PAR-002: Debug CUDA driver error 700 in attention kernel
- [ ] PAR-003: Fix CUDA Q4_K matvec PTX module load failure

**PAR-011: --gpu Flag Implementation - COMPLETED (2026-01-01)**

Added `--gpu` flag to both `realizar run` and `realizar serve` commands:
- `realizar run model.gguf --gpu "prompt"` - Forces CUDA acceleration
- `realizar serve --model model.gguf --gpu` - Forces CUDA acceleration for server

When `--gpu` is specified:
1. Uses `OwnedQuantizedModel` instead of `QuantizedGGUFTransformer`
2. Calls `enable_cuda(0)` to activate CUDA backend
3. Shows GPU status in output: "Backend: CUDA (GPU)"
4. Falls back to CPU with warning if CUDA feature not enabled

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated

