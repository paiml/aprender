# Motivation

## The Problem

ML kernel implementations are derived from research papers, but the derivation
chain is invisible:

```
Paper (LaTeX) → Developer's head → Code → Tests → Ship
```

The developer's head is an unauditable black box. When a SIMD kernel produces
wrong results six months later, nobody can trace back to which equation was
violated or which paper assumption was broken.

## Evidence from Production

The earliest contracts were born from production incidents in aprender:

| Contract | Root Cause | Incident |
|----------|-----------|----------|
| `tensor-layout-v1.yaml` | SafeTensors 94.5% zeros passed structural checks | PMAT-234 |
| `layer-parity-v1.yaml` | 7B GPU produced garbage, no way to compare with CPU | PMAT-232 |
| `kernel-fusion-v1.yaml` | Fused kernel existed but was never wired in | PAR-077 |
| `quantized-dot-product-v1.yaml` | SIMD kernels had no reference to compare against | PAR-001 |

Every incident would have been prevented if the mathematical specification had
been the source of truth, not the code.

## The Solution

Make the derivation chain explicit, auditable, and **provable**:

```
Paper (arXiv) → Equations → Contract (YAML) → Trait (Rust) → Kernel (SIMD) → Tests (probar) → Proof (Kani)
       ↑                         ↑                                                ↑                ↑
   peer-reviewed           machine-parseable                                 falsifiable    formally verified
```

Every link in this chain is a concrete artifact in version control.
The final link -- Kani bounded model checking -- is what elevates this from
"really good testing" to "actual proof."
