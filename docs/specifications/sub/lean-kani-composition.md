# Sub-spec: Lean 4 + Kani Composition

**Parent:** [pv-spec.md](../pv-spec.md) Section 2 (Verification Ladder)

---

## 1. Why Both

Lean 4 and Kani solve **different problems**. Neither subsumes the other.

| | Lean 4 | Kani |
|---|---|---|
| **Domain** | Mathematical reals (ℝ) | Rust f32/f64/i32/usize |
| **Scope** | ALL inputs, unbounded | ALL inputs up to size N |
| **Proves** | The algorithm is correct | The code matches the algorithm |
| **Misses** | Float precision, overflow, NaN | Inputs larger than bound N |
| **Tool** | Lean 4 + Mathlib | cargo kani (CBMC backend) |
| **Artifact** | `.lean` theorem (no sorry) | `#[kani::proof]` harness (verified) |

### The Gap Between Them

Lean proves `∀ x ∈ ℝⁿ, Σᵢ softmax(xᵢ) = 1`. Beautiful. But ℝ is not f32.

The Rust implementation uses `f32::exp()`, which:
- Overflows at `x > 88.7` (returns `Inf`)
- Underflows at `x < -87.3` (returns `0.0`)
- Has ~7 digits of precision (ULP errors accumulate)
- Encounters NaN, Inf, subnormals

The Lean proof says nothing about any of this. It assumes infinite precision.

Kani verifies the **actual Rust code** — with all its f32 edge cases — but
only for vectors up to size N (typically 8-32). For ML kernels with fixed
block sizes (Q4_K = 256 elements, SIMD width = 8), bounded verification
at the natural bound IS exhaustive.

### Compositional Verification

The key insight: **Lean proves the math, Kani proves the code, and the
`stub_float` strategy bridges them.**

```
Lean 4: "Σᵢ exp(xᵢ)/Z = 1 for all x ∈ ℝⁿ"
          ↓ (mathematical truth)
          ↓
Kani stub_float: "If exp() returns ANY positive finite value,
                   the surrounding code still produces outputs
                   that sum to 1.0 ± ε"
          ↓ (structural invariant)
          ↓
Kani exhaustive: "For ALL f32 vectors of length 1-8,
                   softmax_1d_alloc() produces correct output"
          ↓ (implementation verification)
          ↓
proptest:  "For 10,000 random f32 vectors of length 1-65536,
            softmax_1d_alloc() produces correct output"
          ↓ (statistical confidence)
          ↓
#[contract]: "At every call site in debug builds,
              !logits.is_empty() && logits are finite"
          ↓ (runtime enforcement)
```

## 2. How They Compose

### 2.1 Same Obligation, Different Levels

Each proof obligation in the YAML maps to BOTH a Lean theorem and a Kani
harness:

```yaml
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|Σ σ(x)_i - 1.0| < ε"
    lean:
      theorem: Softmax.partition_of_unity
      status: proved
    # Lean proves: exact identity Σ = 1 over ℝ

kani_harnesses:
  - id: KANI-SM-001
    obligation: SM-INV-001
    property: "Softmax sums to 1.0 for small vectors"
    bound: 8
    strategy: stub_float
    # Kani proves: f32 implementation within ε for |x| ≤ 8
```

The obligation is ONE claim. Lean gives L5 (unbounded, ideal).
Kani gives L4 (bounded, real). Both are required because they cover
different failure modes.

### 2.2 The stub_float Bridge

Kani cannot reason about `f32::exp()` — it's a transcendental
implemented in libm assembly. The `stub_float` strategy replaces it:

```rust
// Kani stub: exp() returns any positive finite f32
fn stub_exp(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r > 0.0 && r.is_finite());
    r
}
```

This is sound because:
1. **Lean proves** that `exp(x) > 0` for all real x
2. **Lean proves** that `exp` is finite for finite x (bounded intervals)
3. Kani's stub **assumes** exactly what Lean proved
4. Kani then verifies the **surrounding code** (max-subtract, sum,
   divide) preserves the invariant regardless of exp's exact value

This is compositional verification: Lean discharges the transcendental,
Kani verifies the structural code.

### 2.3 What Each Catches

| Failure mode | Lean | Kani | proptest | #[contract] |
|---|---|---|---|---|
| Wrong formula | ✓ (if modeled) | | | |
| f32 overflow in exp() | | ✓ exhaustive | ✓ statistical | |
| NaN propagation | | ✓ | ✓ | |
| Off-by-one in loop | | ✓ | ✓ | |
| Division by zero | | ✓ | ✓ | |
| Empty input panic | | ✓ | | ✓ |
| SIMD ≠ scalar | | ✓ | ✓ | |
| Algorithm correctness | ✓ | | | |
| Precision loss at N=65536 | | | ✓ (L3) | |

## 3. Obligation Routing

Not every obligation needs both Lean AND Kani. The routing depends on
the nature of the claim:

### Always Both (algebraic + numeric)
- "Softmax sums to 1" — Lean proves the algebra, Kani catches f32 drift
- "RMSNorm preserves direction" — Lean proves the math, Kani catches underflow
- "Cross-entropy is non-negative" — Lean proves log convexity, Kani catches NaN

### Lean Only (pure math, no code path)
- "Transpose is an involution" — A^T^T = A is algebraic, no float issues
- "Cholesky uniqueness" — existence/uniqueness theorem, no implementation
- "FFT butterfly symmetry" — combinatorial identity

### Kani Only (implementation detail, not mathematical)
- "SIMD path matches scalar path" — no math to prove, just code equivalence
- "No buffer overflow in AVX2 kernel" — memory safety, not math
- "Q4_K block decode is lossless" — bit manipulation, not calculus
- "GPU shader matches CPU path" — cross-platform equivalence

### Neither (runtime only)
- "API returns error on invalid input" — tested by #[contract] + unit tests
- "Serialization roundtrips" — proptest sufficient

## 4. The Verification DAG

Obligations form a DAG. Kani can use `#[kani::stub_verified]` to compose
verified sub-kernels:

```
┌─────────────────────────────────────┐
│ attention = softmax(Q·K^T/√d) · V  │  ← L3 + L4 (Kani compositional)
│                                     │
│   uses:                             │
│   ├── softmax  ← L5 (Lean) + L4 (Kani stub_float)
│   ├── matmul   ← L4 (Kani exhaustive) + L5 (Lean)
│   └── scale    ← L4 (Kani bounded_int)
└─────────────────────────────────────┘
```

Kani's compositional strategy: when verifying `attention`, softmax is
replaced by `#[kani::stub_verified(softmax)]` which assumes softmax's
postconditions without re-verifying softmax itself. This keeps
verification tractable while maintaining soundness — because softmax
was already verified at L4+L5.

## 5. CI Enforcement

```yaml
# .github/workflows/lean.yml
lean-verify:
  steps:
    - lake build           # All Lean compiles
    - grep -r "sorry" lean/ && exit 1  # No sorry
    - pv proof-status      # Report L5 coverage

# .github/workflows/kani.yml
kani-verify:
  steps:
    - cargo kani --harness "verify_*" --solver cadical
    - pv proof-status      # Report L4 coverage

# .github/workflows/quality-gate.yml
quality-gate:
  needs: [lean-verify, kani-verify]
  steps:
    - pmat score --gate 70  # Composite score includes L4+L5 coverage
```

## 6. Current Status

| Domain | Lean (L5) | Kani (L4) | Gap |
|---|---|---|---|
| Softmax | 5 theorems | 3 harnesses (stub) | Harnesses need kernel wiring |
| Elementwise | 3 theorems | 0 harnesses | Need Kani harnesses |
| BLAS | 2 theorems | 0 harnesses | Need Kani harnesses |
| LayerNorm | 5 theorems | 0 harnesses | Need Kani harnesses |
| CrossEntropy | 4 theorems | 0 harnesses | Need Kani harnesses |
| RMSNorm | 4 theorems | 0 harnesses | Need Kani harnesses |
| AdamW | 3 theorems | 0 harnesses | Need Kani harnesses |
| FFT | 5 theorems | 0 harnesses | Need Kani harnesses |
| MatMul | 3 theorems | 0 harnesses | Need Kani harnesses |

**Lean: 64 theorems, 0 sorry.** Comprehensive.
**Kani: 45 harnesses defined, 0 verified (all `unimplemented!`).** Scaffolding only.

The Lean side is production. The Kani side is infrastructure without
kernel wiring. Priority: wire up the 3 softmax harnesses as the template,
then replicate across domains.

## 7. References

- Kani Model Checker: <https://model-checking.github.io/kani/>
- Lean 4 + Mathlib: <https://leanprover-community.github.io/>
- SPARK/Ada Proof Discharge: Burns & Wellings, "Concurrent and Real-Time Programming in Ada" (2007)
- Dafny Verification Conditions: Leino, "Dafny: An Automatic Program Verifier" (2010)
- The Lean 4 Theorem Prover: Moura & Ullrich (2021)
