# Sub-spec: The Seven-Phase Pipeline

**Parent:** [pv-spec.md](../pv-spec.md) Section 4

---

## 1. Phase Overview

```
Phase 1: EXTRACT    arXiv PDF -> canonical math + proof obligations
Phase 2: SPECIFY    Canonical math -> YAML contract
Phase 3: SCAFFOLD   YAML contract -> Rust trait + failing tests
Phase 4: IMPLEMENT  Failing tests -> scalar, SIMD, PTX kernels
Phase 5: FALSIFY    Implementation -> probar property tests + certeza
Phase 6: VERIFY     Implementation -> Kani bounded model checking
Phase 7: PROVE      Implementation -> Lean 4 theorem proving
```

---

## 2. Phase 1: Extract

### Input
An arXiv paper containing a mathematical operation used in ML.

### Process
1. Identify governing equation(s) — the forward pass computation
2. Identify domain and codomain — shapes, types
3. Extract proof obligations — invariants, bounds, equivalences
4. Identify numerical stability requirements
5. Note assumptions and boundary conditions

### Output
Canonical math form with proof obligation table.

### Example: Softmax

**Governing equation (stable):**
```
softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

**Proof obligations:**

| ID | Type | Property |
|---|---|---|
| SM-INV-001 | Invariant | Output sums to 1.0 |
| SM-INV-002 | Invariant | All outputs positive |
| SM-INV-003 | Bound | Output in (0,1) |
| SM-EQV-001 | Equivalence | Shift invariance |
| SM-MON-001 | Monotonicity | Order preservation |
| SM-BND-001 | Bound | Argmax >= 1/n |

---

## 3. Phase 2: Specify

### Translation Rules

| Math Concept | YAML Field |
|---|---|
| Governing equation | `equations.<name>.formula` |
| Domain | `equations.<name>.domain` |
| Codomain | `equations.<name>.codomain` |
| Invariant | `proof_obligations[].type: invariant` |
| Equivalence | `proof_obligations[].type: equivalence` |
| Bound | `proof_obligations[].type: bound` |
| Tolerance | `proof_obligations[].tolerance` |
| Falsification | `falsification_tests[]` |

### The Critical Rule

> Every YAML entry must be traceable to a specific equation in the paper.
> If you cannot point to a formula, you cannot write a contract.
> If you cannot write a contract, you cannot write a falsification test.
> If you cannot write a falsification test, you are guessing.

---

## 4. Phase 3: Scaffold

`pv scaffold` generates:

1. **Rust trait** — each equation becomes a method, proof obligations
   become doc-comments with `INVARIANT:` or `REQUIRES:` prefix.
2. **Failing tests** — each proof obligation becomes a `#[test]` with
   `todo!()` body. ALL tests MUST fail before implementation.

### The Rule

> All scaffold tests MUST fail. If a test passes before implementation,
> either the test is wrong or the implementation already exists.

---

## 5. Phase 4: Implement

### Order

```
1. Scalar reference (ground truth, close to paper)
2. Tests pass for scalar
3. AVX2 variant
4. AVX2 parity tests pass (ULP tolerance)
5. PTX kernel (inline assembly string)
6. PTX structural tests pass
7. Dispatch table updated in YAML
```

### The Scalar Reference is Sacrosanct

The scalar implementation is the mathematical reference. SIMD/PTX may
diverge only within the contract's ULP tolerance. If they disagree
beyond tolerance, the optimized variant is wrong — not the scalar.

---

## 6. Phase 5: Falsify

### probar Property Tests

Each proof obligation maps to a property test:

```rust
#[probar::property]
fn prop_softmax_sums_to_one(xs: Vec<f32>) -> bool {
    let result = softmax_scalar(&xs);
    (result.iter().sum::<f32>() - 1.0).abs() < 1e-6
}
```

### Metamorphic Relations

Test properties relating different inputs/outputs:

```rust
#[probar::metamorphic]
fn mr_softmax_preserves_order(xs: Vec<f32>) {
    let result = softmax_scalar(&xs);
    for i in 0..xs.len() {
        for j in 0..xs.len() {
            if xs[i] > xs[j] {
                assert!(result[i] > result[j]);
            }
        }
    }
}
```

### SIMD Parity (Universal)

Every contract gets a SIMD parity test:

```rust
#[probar::property]
fn prop_simd_matches_scalar(data: Vec<f32>) -> bool {
    let scalar = softmax_scalar(&data);
    let simd = softmax_avx2(&data);
    scalar.iter().zip(simd.iter()).all(|(s, a)| {
        (s.to_bits() as i32 - a.to_bits() as i32).unsigned_abs() <= ULP_TOLERANCE
    })
}
```

### certeza Quality Gates

```yaml
qa_gate:
  id: "F-SOFTMAX-001"
  checks:
    - "All normalization tests pass (SM-INV-001)"
    - "Shift invariance holds (SM-EQV-001)"
    - "SIMD matches scalar (FALSIFY-SM-003)"
  pass_criteria: "All falsification tests pass"
  falsification: "Introduce off-by-one in max reduction -- gate must catch"
```

---

## 7. Phase 6: Verify (Kani)

### Three Verification Strategies

#### Strategy 1: Exhaustive

For integer arithmetic and structural properties. Zero false positives.

**Best for:** quantized dot products, bsums, format dispatch, shape
validation, index bounds.

```rust
#[kani::proof]
#[kani::unwind(33)]
fn verify_bsums_exact() {
    let activations: [i8; 32] = kani::any();
    let precomputed: i32 = activations.iter().map(|&x| x as i32).sum();
    let mut online: i32 = 0;
    for i in 0..32 { online += activations[i] as i32; }
    assert_eq!(precomputed, online);
}
```

#### Strategy 2: Stub Float Transcendentals

For `exp()`, `sqrt()`, `log()`. Stub with bounded approximations
preserving structural properties.

**Best for:** softmax, rmsnorm, swiglu.

```rust
#[cfg(kani)]
fn exp_stub(x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result > 0.0);
    kani::assume(result.is_finite());
    result
}

#[kani::proof]
#[kani::stub(f32::exp, exp_stub)]
#[kani::unwind(17)]
fn verify_softmax_positivity() { /* ... */ }
```

#### Strategy 3: Compositional (Function Contracts)

For composite kernels. Verify sub-kernels independently, compose via
`#[kani::stub_verified]`.

**Best for:** attention, transformer layers, multi-step pipelines.

```rust
#[kani::requires(input.len() >= 1)]
#[kani::ensures(|r| r.iter().all(|&x| x > 0.0))]
pub fn softmax_verified(input: &[f32]) -> Vec<f32> { /* ... */ }

#[kani::proof]
#[kani::stub_verified(softmax_verified)]
fn verify_attention_normalized_weights() { /* ... */ }
```

### SIMD Parity Proofs

The highest-value Kani harness: prove SIMD matches scalar for ALL
inputs. Kani supports all SIMD intrinsics.

### Negative Verification

```rust
#[kani::proof]
#[kani::should_panic]
fn verify_rejects_invalid_input() {
    let bad_data: Vec<f32> = /* symbolic with >50% zeros */;
    ValidatedEmbedding::new(bad_data).unwrap(); // MUST panic
}
```

### Kani Limitations

| Property | Kani | Alternative |
|---|---|---|
| Numerical accuracy of exp() | Over-approximated | probar L3 |
| Unbounded vector lengths | Bounded to N | probar L3 |
| Concurrent dispatch | Not supported | Integration tests |
| GPU kernel correctness | Not supported | apr parity tool |

---

## 8. Phase 7: Prove (Lean 4)

### Motivation

Kani proves for all inputs <= size N. Lean 4 proves for ALL inputs
unconditionally, over mathematical reals.

### What Gets Lifted to Lean

1. **Universal algebraic identities** — softmax partition of unity
2. **Structural properties** — RMSNorm homogeneity
3. **Compositional theorems** — attention weight normalization

### What Stays in Kani

1. SIMD parity (hardware-specific)
2. Integer arithmetic (already exact)
3. Index bounds (implementation-specific)

### Lean Status Tracking

Each obligation has a `lean.status` field:
- `proved` — machine-checked Lean theorem
- `sorry` — stated but not proven
- `wip` — work in progress
- `not-applicable` — implementation-specific, stays in Kani

### Integration

```bash
pv lean contracts/softmax-kernel-v1.yaml --output-dir lean/
pv lean-status contracts/
```

Generates `.lean` files with definitions and theorem stubs.
`pv lean-status` reports proved/sorry/wip counts.
