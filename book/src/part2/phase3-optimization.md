# Phase 3: Optimization

Phase 3 analyzes transpiled code for compute-intensive patterns and selects optimal execution backends using Mixture-of-Experts (MoE) routing.

## Overview

After transpilation produces Rust code, the optimization phase identifies opportunities for hardware acceleration:

```
Transpiled .rs files
       │
       ▼
┌──────────────────┐
│ Pattern Scanner  │ ← Scan for matmul, reduce, iter patterns
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MoE Router      │ ← BackendSelector::select_with_moe()
│  (5× PCIe Rule)  │
└────────┬─────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
 Scalar SIMD  GPU     ← Per-pattern recommendation
```

## The 5x PCIe Dispatch Rule

Based on Gregg & Hazelwood (2011), GPU dispatch is only beneficial when:

```
compute_time > 5 × transfer_time
```

This prevents wasteful GPU dispatch for small workloads where PCIe transfer overhead dominates. The `--gpu-threshold` flag controls the matrix size cutoff (default: 500).

## Compute Pattern Classification

| Pattern | Complexity | Recommended Backend |
|---------|-----------|-------------------|
| `matmul`/`gemm`/`dot_product` | High | GPU (if above threshold) |
| `.sum()`/`.fold()`/`reduce` | Medium | SIMD |
| `.iter().map()`/`.zip()` | Low | Scalar |

## Cargo Profile Optimization

The optimizer writes `[profile.release]` settings to `Cargo.toml`:

| Profile | `opt-level` | LTO | `codegen-units` | Strip |
|---------|-------------|-----|-----------------|-------|
| Fast | 2 | off | 16 | — |
| Balanced | 3 | thin | 4 | — |
| Aggressive | 3 | full | 1 | symbols |

## Jidoka Integration

If optimization analysis fails (e.g., output directory missing), the phase is marked as failed in the workflow state machine. Subsequent phases (Validation, Build) will refuse to run until the issue is resolved.

## CLI Reference

See [`batuta optimize`](../part6/cli-optimize.md) for full command documentation.

---

**Previous:** [Phase 2: Transpilation](./phase2-transpilation.md)
**Next:** [Phase 4: Validation](./phase4-validation.md)
