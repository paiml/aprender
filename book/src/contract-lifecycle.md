# Contract Lifecycle

## Versioning

Contracts follow semantic versioning:

- **MAJOR:** Breaking change to equations, tolerance tightening, new required
  proof obligations. Consumers MUST update.
- **MINOR:** New optional proof obligations, new SIMD dispatch entries,
  additional falsification tests. Consumers SHOULD update.
- **PATCH:** Typo fixes, clarifications, additional references. No code changes
  needed.

## Evolution

```
v1.0.0  Initial contract from paper
  ↓
v1.1.0  Add SIMD dispatch for new ISA (e.g., AVX-512 VNNI)
  ↓
v1.2.0  Add falsification test from production incident
  ↓
v2.0.0  New paper with better algorithm (e.g., FlashAttention replaces naive attention)
```

## The Kaizen Principle

> When a production incident reveals a failure mode not covered by the
> contract, the contract MUST be updated before the code is fixed.

This is the tensor-layout-v1.yaml lesson: PMAT-234 revealed that semantic
validation (data quality) was missing. The contract was updated to v2.0.0
with semantic validation rules BEFORE the code was patched.
