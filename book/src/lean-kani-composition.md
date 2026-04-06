# Lean 4 + Kani: Why Both, How They Compose

## The Two-Prover Model

provable-contracts uses **two** independent verification tools because they
prove fundamentally different things:

| | Lean 4 | Kani |
|---|---|---|
| Proves | The **algorithm** is correct | The **code** is correct |
| Domain | Mathematical reals (infinite precision) | Rust f32/i32 (actual hardware) |
| Scope | ALL inputs, unbounded | ALL inputs up to size N |
| Catches | Wrong formula, missing invariant | Overflow, NaN, code path bugs |

**Neither subsumes the other.** Lean says "softmax sums to 1 over the reals."
Kani says "the Rust f32 implementation of softmax produces outputs that sum
to 1.0 within tolerance, for ALL possible f32 vectors of length 1-8."

## The Float Gap

Lean proves over real numbers. Rust runs on f32. The gap includes:

- `f32::exp(89.0)` = `Inf` (overflow)
- `f32::exp(-88.0)` = `0.0` (underflow)
- `1.0 + 1e-8 - 1.0` = `0.0` (precision loss)
- `0.0 / 0.0` = `NaN` (undefined)

Lean's `Σ exp(xᵢ)/Z = 1` assumes none of this happens. Kani catches
all of it in the actual Rust code.

## The stub_float Bridge

The key composition technique. Kani's `stub_float` strategy replaces
transcendentals with symbolic values constrained by Lean's proven properties:

```rust
// Kani stub: exp() returns any positive finite f32
// Sound because Lean proved: ∀ x ∈ ℝ, exp(x) > 0
fn stub_exp(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r > 0.0 && r.is_finite());
    r
}
```

Then Kani verifies the surrounding code:
```rust
#[kani::proof]
#[kani::stub(f32::exp, stub_exp)]
fn verify_softmax_sums_to_one() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 8);
    let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
    kani::assume(input.iter().all(|x| x.is_finite()));

    let output = softmax_1d_alloc(&input);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}
```

This is **compositional**: Lean discharges `exp(x) > 0`, Kani uses
that as an assumption and verifies the division/accumulation code.

## Obligation Routing

Not every obligation needs both tools:

| Both Lean + Kani | Lean only | Kani only |
|---|---|---|
| Softmax sums to 1 | Transpose involution | SIMD = scalar |
| RMSNorm preserves direction | Cholesky uniqueness | No buffer overflow |
| Cross-entropy non-negative | FFT butterfly symmetry | Q4_K decode lossless |

**Rule:** If the claim involves algebra AND float arithmetic, use both.
If pure math, Lean only. If implementation detail, Kani only.

## The Verification DAG

Kani's `#[kani::stub_verified]` enables compositional verification:

```
attention = softmax(Q * K^T / sqrt(d)) * V
  uses: softmax (L5 Lean + L4 Kani)
  uses: matmul  (L4 Kani + L5 Lean)
  uses: scale   (L4 Kani)
```

When verifying `attention`, softmax is stubbed with its proven
postconditions. This keeps verification tractable.

## Current Status

- **Lean:** 64 theorems, 0 sorry. Production.
- **Kani:** 45 harnesses scaffolded, 0 wired to kernels. Infrastructure only.
- **Bridge:** `stub_float` stubs defined, not yet connected to Lean postconditions.

Next step: wire the 3 softmax Kani harnesses to call actual `softmax_1d_alloc`,
verify with `cargo kani`, then replicate across domains.

## Further Reading

- Full design: `docs/specifications/sub/lean-kani-composition.md`
- Kani Model Checker: <https://model-checking.github.io/kani/>
- Lean 4 + Mathlib: <https://leanprover-community.github.io/>
