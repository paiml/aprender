# Phase 7: Prove -- Unbounded Verification via Lean 4

Phase 7 completes the verification hierarchy by lifting selected proof
obligations into Lean 4's dependent type theory, producing proofs valid
**for all inputs unconditionally** -- achieving what Kani bounded model
checking cannot.

```
Phase 5: Falsify   →  Probabilistic confidence  (proptest/probar)
Phase 6: Verify    →  Bounded certainty         (Kani BMC)
Phase 7: Prove     →  Unbounded certainty       (Lean 4)
```

## The Bounded Verification Gap

Kani proves `softmax_non_negative` for all `f32` vectors up to length N=8. But:

- What about N=65536 (a real transformer sequence length)?
- What about the algebraic identity `Σᵢ softmax(xᵢ) = 1` independent of precision?
- What about compositional properties across the verification DAG?

These require **unbounded proofs over mathematical reals**, which is precisely
what Lean 4 + Mathlib provide.

## Verification Hierarchy

Each proof obligation gains a verification level representing the highest level
at which it has been discharged:

| Level | Method | Guarantees | Tool |
|-------|--------|-----------|------|
| L0 | Untested | None | -- |
| L1 | Unit tests | Spot checks | `cargo test` |
| L2 | Property tests | Probabilistic falsification | probar / proptest |
| L3 | Bounded model checking | Exhaustive within bounds | Kani |
| L4 | Theorem proving | Unbounded, unconditional | Lean 4 |

Obligations can be discharged at multiple levels simultaneously. Higher levels
subsume lower ones logically but serve different practical purposes (L2 catches
regressions fast; L4 provides mathematical certainty).

## Contract Schema: The `lean` Block

Each proof obligation can include a `lean:` block specifying its Lean 4 metadata:

```yaml
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|Σ σ(x)_i - 1.0| < ε"
    tolerance: 1.0e-6
    lean:
      theorem: Softmax.partition_of_unity
      module: ProvableContracts.Softmax
      status: proved       # proved | sorry | wip | not-applicable
      depends_on:
        - Real.exp_pos
        - Finset.sum_div_distrib
      mathlib_imports:
        - Mathlib.Analysis.SpecialFunctions.ExpDeriv
        - Mathlib.Algebra.BigOperators.Group.Finset
      notes: "Proof over reals; f32 gap addressed by error-bound lemma"
```

The top-level `verification_summary` tracks aggregate L4 coverage:

```yaml
verification_summary:
  total_obligations: 6
  l2_property_tested: 6
  l3_kani_proved: 3
  l4_lean_proved: 5
  l4_sorry_count: 0
  l4_not_applicable: 1
```

## Lean 4 Project Structure

The `lean/` directory contains the theorem-proving layer:

```
lean/
├── lakefile.lean                      # Lake build with Mathlib dependency
├── lean-toolchain                     # Lean 4 version pin (v4.29.0-rc4)
├── lake-manifest.json                 # Pinned Mathlib dependency commits
├── forjar.yaml                        # Reproducible toolchain install
├── ProvableContracts.lean             # Root module (imports all 22 submodules)
├── ProvableContracts/
│   ├── Basic.lean                     # RVec, sum definitions
│   ├── Defs/
│   │   ├── Softmax.lean              # softmax, log_softmax
│   │   ├── RMSNorm.lean             # rms, rmsnorm
│   │   ├── LayerNorm.lean           # mean, variance, layernorm
│   │   ├── Sigmoid.lean             # sigmoid
│   │   ├── CrossEntropy.lean        # cross_entropy, log_softmax
│   │   └── Transpose.lean           # transpose (matrix involution)
│   └── Theorems/
│       ├── Softmax/
│       │   ├── PartitionOfUnity.lean  # Σ softmax(x)_i = 1 (proved)
│       │   ├── NonNegativity.lean     # softmax(x)_i > 0 (proved)
│       │   ├── Bounded.lean           # 0 < softmax(x)_i < 1 (proved)
│       │   ├── Monotonicity.lean      # x_i > x_j → σ(x)_i > σ(x)_j (proved)
│       │   └── ShiftInvariance.lean   # σ(x + c·1) = σ(x) (proved)
│       ├── RMSNorm/
│       │   ├── DenominatorPositive.lean  # √(mean(x²) + ε) > 0 (proved)
│       │   └── ScaleInvariance.lean      # rmsnorm(αx) = sign(α)·rmsnorm(x) (proved)
│       ├── LayerNorm/
│       │   ├── DenominatorPositive.lean  # √(var(x) + ε) > 0 (proved)
│       │   └── ShiftInvariance.lean      # LN(x + c) = LN(x) (proved)
│       ├── Sigmoid/
│       │   ├── Bounded.lean              # 0 < σ(x) < 1 (proved)
│       │   └── Symmetry.lean             # σ(-x) = 1 - σ(x) (proved)
│       ├── CrossEntropy/
│       │   ├── NonNegativity.lean        # H(p,q) ≥ 0 (proved)
│       │   └── LogSoftmaxBound.lean      # log(softmax(x)_i) ≤ 0 (proved)
│       └── Transpose/
│           └── Involution.lean           # (Aᵀ)ᵀ = A (proved)
└── test/
```

All 14 theorems are **fully proved** (zero `sorry`). The proofs type-check
against Lean 4 v4.29.0-rc4 + Mathlib4 master (2123 build jobs, 0 errors).

## CLI Commands

### `pv lean <contract.yaml> [--output-dir <dir>]`

Generate Lean 4 definitions and theorem stubs from a contract:

```bash
$ pv lean contracts/softmax-kernel-v1.yaml --output-dir lean/

  lean/ProvableContracts/Defs/Softmax.lean
  lean/ProvableContracts/Theorems/Softmax/partition_of_unity.lean
  lean/ProvableContracts/Theorems/Softmax/softmax_pos.lean
  lean/ProvableContracts/Theorems/Softmax/softmax_bounded.lean
  lean/ProvableContracts/Theorems/Softmax/monotone.lean
  lean/ProvableContracts/Theorems/Softmax/shift_invariance.lean

Generated 6 Lean files.
```

Without `--output-dir`, prints to stdout for inspection.

### `pv lean-status [<path>]`

Report Lean proof status across contracts:

```bash
$ pv lean-status contracts/

Contract                       Oblgs Proved Sorry WIP N/A
────────────────────────────────────────────────────────────
Cross-entropy kernel — log-s       2      2     0   0   0
LayerNorm kernel — layer nor       2      2     0   0   0
RMSNorm kernel — root mean s       2      2     0   0   0
SiLU kernel — sigmoid linear       1      1     0   0   0
Softmax kernel — numerically       5      5     0   0   0
Matrix transpose kernel — AV       2      2     0   0   0
────────────────────────────────────────────────────────────
Total                             14     14     0   0   0
L4 Coverage: 100% (14/14)   Sorry Debt: 0
```

## Bridging the Real-Float Gap

Lean proofs operate over `ℝ` (mathematical reals). Rust code operates over
`f32`/`f64`. The strategy is **layered proofs**:

1. **Ideal layer** (Lean): Prove the property over `ℝ` unconditionally
2. **Error layer** (Lean): Prove an error bound for IEEE 754 rounding
3. **Concrete layer** (Kani): Verify the Rust implementation matches the
   error-bounded specification for concrete bit-widths

```lean
-- Ideal layer (PROVED)
theorem partition_of_unity {n : ℕ} (x : RVec (n + 1)) :
    ∑ i : Fin (n + 1), softmax x i = 1 := by
  simp only [softmax]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (sum_exp_pos x))

-- Error layer (future)
theorem softmax_partition_f32_error (x : Vector Float32 n)
    (hx : ∀ i, |x i| ≤ 88) :
    |Finset.univ.sum (softmax_f32 x) - 1| ≤ n * Float32.epsilon := by
  sorry
```

## Obligation Triage: What Gets a Lean Proof

| Category | Priority | Rationale |
|----------|:--------:|-----------|
| Universal algebraic identities | **High** | `Σ softmax = 1`, RMSNorm idempotence |
| Compositional correctness | **High** | Verification DAG composition |
| Monotonicity / ordering | **High** | Natural in Mathlib's `Order` |
| Numerical error bounds | Medium | Requires careful epsilon arithmetic |
| Equivalence proofs | Medium | Flash attention = standard attention |
| Performance / hardware | Skip | Empirical by nature |
| Shape / dimension checks | Skip | Well-served by Rust types + Kani |

## Reproducible Lean Toolchain via forjar

The `lean/forjar.yaml` orchestrates the full toolchain install reproducibly:

```bash
# Install elan + Lean toolchain + Mathlib cache
make lean-install

# Build and type-check all proofs
make lean-build

# Quick check (direct lake build, requires elan in PATH)
make lean-check

# Clean build artifacts
make lean-clean
```

The forjar pipeline has 4 stages: `elan-install` → `lean-toolchain` → `mathlib-cache` → `lean-build`, each idempotent and safe for re-runs.

## Examples

```bash
# Generate Lean stubs for a contract
cargo run --example lean_codegen -- contracts/softmax-kernel-v1.yaml

# Report L4 coverage across all contracts
cargo run --example lean_status -- contracts/

# Run all proofs and verify they type-check
cargo run --example lean_proofs
```
