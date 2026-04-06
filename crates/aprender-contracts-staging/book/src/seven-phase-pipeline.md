# The Seven-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Phase 1: EXTRACT       arXiv PDF                                   │
│  ──────────────────      ↓                                          │
│  Read paper.             LaTeX equations                             │
│  Identify governing      ↓                                          │
│  equations and           Canonical math form                         │
│  proof obligations.      (with domains, codomains, invariants)       │
│                                                                     │
│  Phase 2: SPECIFY        Canonical math                              │
│  ──────────────────      ↓                                          │
│  Encode equations,       YAML contract                               │
│  tolerances, dispatch    (machine-parseable, version-controlled)     │
│  tables, and             ↓                                          │
│  falsification tests     Enforcement rules                           │
│  into YAML.                                                         │
│                                                                     │
│  Phase 3: SCAFFOLD       YAML contract                               │
│  ──────────────────      ↓                                          │
│  Generate Rust trait     pub trait FooKernel { ... }                  │
│  with proof obligations  ↓                                          │
│  as doc-comments.        #[test] fn falsify_001() { ... }            │
│  Generate failing tests. (ALL TESTS FAIL — no implementation yet)   │
│                                                                     │
│  Phase 4: IMPLEMENT      Trait + failing tests                       │
│  ──────────────────      ↓                                          │
│  Write scalar reference  fn foo_scalar(...) → ... { ... }            │
│  first (ground truth).   ↓                                          │
│  Then SIMD variants.     fn foo_avx2(...) → ... { ... }              │
│  SIMD must match scalar  fn foo_avx512(...) → ... { ... }            │
│  within ULP tolerance.                                               │
│                                                                     │
│  Phase 5: FALSIFY        Implementation + tests                      │
│  ──────────────────      ↓                                          │
│  Run probar property     proptest: random inputs, check invariants   │
│  tests. Run certeza      ↓                                          │
│  quality gates.          metamorphic: scaled input → scaled output?  │
│  Verify SIMD parity.     ↓                                          │
│  Verify cross-format     certeza: coverage ≥95%, mutation score ≥80% │
│  isolation.              (Level 3: high confidence, not proof)       │
│                                                                     │
│  Phase 6: VERIFY         Implementation + probar tests               │
│  ──────────────────      ↓                                          │
│  Write Kani proof        #[kani::proof] harnesses for each           │
│  harnesses.              proof obligation.                           │
│  Run `cargo kani`.       ↓                                          │
│  Kani explores ALL       kani::any() = symbolic (all possible)       │
│  execution paths up      ↓                                          │
│  to the kernel's         VERIFIED: property holds for ALL inputs     │
│  natural bound.          within bound (Level 4: bounded proof)       │
│                                                                     │
│  Phase 7: PROVE          Kani proofs + YAML contracts                │
│  ──────────────────      ↓                                          │
│  Lift selected           Lean 4 theorem stubs (sorry)                │
│  obligations into        ↓                                          │
│  Lean 4 dependent        Machine-checked proofs over ℝ               │
│  type theory.            ↓                                          │
│  Prove unconditionally   PROVED: property holds for ALL inputs       │
│  over mathematical       unconditionally (Level 5: unbounded proof)  │
│  reals via Mathlib.                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Invariant: Every Phase Produces an Artifact

| Phase | Input | Output Artifact | Location |
|-------|-------|-----------------|----------|
| Extract | arXiv PDF | Canonical math + proof obligations | `contracts/<name>/MATH.md` |
| Specify | Canonical math | YAML contract | `contracts/<name>-v1.yaml` |
| Scaffold | YAML contract | Rust trait + failing tests | `src/<name>/trait.rs`, `src/<name>/contract_tests.rs` |
| Implement | Trait | Scalar + SIMD kernels | `src/<name>/scalar.rs`, `src/<name>/simd.rs` |
| Falsify | Implementation | probar tests + certeza report | `src/<name>/probar_tests.rs` |
| Verify | Implementation | Kani proof harnesses + verification report | `src/<name>/kani_proofs.rs` |
| **Prove** | **YAML + Kani proofs** | **Lean 4 theorems (proved or sorry)** | **`lean/ProvableContracts/Theorems/<name>/`** |

No phase is complete until its artifact is committed.
**A contract is not "provable" until Phase 6 Kani harnesses pass.**
Phase 7 Lean proofs are optional but provide the strongest guarantee:
unbounded certainty over mathematical reals.
