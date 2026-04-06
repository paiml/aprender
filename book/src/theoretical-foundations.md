# Theoretical Foundations

## Popperian Falsificationism

> "A theory is scientific if and only if it makes falsifiable predictions."
> -- Karl Popper, *The Logic of Scientific Discovery* (1959)

A contract is not a wish list. Every rule must have a **falsification test** -- a
concrete experiment that would **disprove** the implementation's correctness if
the implementation is wrong. If a contract rule cannot be falsified, it is not a
contract rule; it is documentation.

**Application:** Every YAML contract entry has:
- `prediction`: what the correct implementation guarantees
- `falsification_test`: code that would PASS if the implementation is WRONG
- `if_fails`: what a failure means (root cause diagnosis)

## Toyota Production System (TPS)

**Poka-Yoke** (mistake-proofing): Make it impossible to do wrong.

> "The most effective approach to mistake-proofing is to design the process so
> that mistakes cannot be made."
> -- Shigeo Shingo, *Zero Quality Control* (1986)

**Application:** Rust's type system as Poka-Yoke. `ValidatedTensor<T>` types
that can only be constructed through validation. The compiler rejects wrong
states -- it's not a runtime check, it's a physical impossibility.

**Jidoka** (automation with a human touch): Stop the line on first defect.

**Application:** Load-time parity gates. If GPU output diverges from CPU
reference, the model constructor returns an error. No garbage reaches inference.

**Genchi Genbutsu** (go and see): Verify by direct observation, not reports.

**Application:** Falsification tests run the actual code with known-bad input.
We don't trust that validation works -- we prove it catches bad data.

## Type-Driven Development

> "Make illegal states unrepresentable."
> -- Edwin Brady, *Type-Driven Development with Idris* (2017)

And the Haskell community's extension:

> "Parse, Don't Validate."
> -- Alexis King (2019)

**Application:** Raw `Vec<f32>` is parsed into `ValidatedWeight` (with
invariant checks) at the system boundary. All internal code operates on
validated types. The gap between "data exists" and "data is correct" is closed
by construction.

## Equation-Driven Development (EDD)

The batuta oracle's `quality-edd` recipe formalizes this cycle:

```
Equation → Failing Test → Implementation → Verification → Falsification
```

This is TDD with a mathematical preamble. The equation comes first. The test
is derived from the equation's invariants. The implementation must satisfy the
test. The falsification demonstrates conditions under which the implementation
would break (and proves it doesn't).

## Design by Contract (DbC)

> "A contract defines the obligations and benefits of each party."
> -- Bertrand Meyer, *Object-Oriented Software Construction* (1988)

**Preconditions:** What the caller guarantees (input shapes, finite values).
**Postconditions:** What the kernel guarantees (output shape, numerical bounds).
**Invariants:** What holds throughout (energy conservation, normalization).

## Bounded Model Checking (Kani)

> "Testing shows the presence of bugs, not their absence."
> -- Edsger Dijkstra

Property-based testing (probar/proptest) checks 10,000 random inputs. That's
Level 2 on the verification ladder. It catches most bugs but cannot guarantee
correctness for ALL inputs. Kani closes this gap.

**Kani** (Amazon, open source) is a bounded model checker for Rust. Instead of
sampling random inputs, it symbolically explores **every possible execution
path** up to a given bound. If Kani says "verified," the property holds for ALL
inputs within that bound -- not 10,000 of them, ALL of them.

**How it works:**
1. `kani::any::<T>()` creates a **symbolic value** representing every possible
   bit pattern of type `T`.
2. `kani::assume(predicate)` constrains the symbolic space (preconditions).
3. `assert!(postcondition)` must hold for every surviving path.
4. Kani compiles Rust to CBMC (C Bounded Model Checker) IR and exhaustively
   explores the state space using SAT/SMT solvers (CaDiCaL, kissat, Z3).

**Why Kani for ML kernels:**
- **SIMD intrinsics are fully supported.** All `simd_add`, `simd_mul`,
  `simd_shuffle`, `simd_reduce_*` operations are modeled precisely. This means
  we can prove SIMD kernels match scalar references for ALL inputs, not just
  sampled ones.
- **Integer arithmetic is exact.** Quantized dot products (Q4_K, Q6_K, Q8_0)
  use integer sub-block sums. Kani proves these exactly -- no tolerance needed.
- **Function contracts** (`#[kani::requires]`, `#[kani::ensures]`) map directly
  to DbC preconditions/postconditions from the Design by Contract section.
- **Compositional verification** via `#[kani::stub_verified]` -- prove each
  kernel in isolation, then compose proofs for the full transformer layer.

**Limitations (honest):**
- **Float transcendentals are over-approximated.** `exp()`, `sqrt()`, `sin()`
  return nondeterministic values in valid ranges. For softmax (`exp`) and
  RMSNorm (`sqrt`), we must stub these with bounded approximations or verify
  the integer/structural logic separately.
- **Bounded, not unbounded.** Verification holds for vectors up to size N
  (set by `#[kani::unwind]`). We choose N to cover all realistic kernel
  invocation sizes.
- **State space explosion.** Large bounds + many branches = slow. Practical
  limit is ~256 elements for most kernels, which covers one super-block.

**The key insight:** Kani's limitations align perfectly with our kernel
structure. Quantized kernels operate on fixed-size super-blocks (256 elements).
SIMD operates on fixed-width lanes (8xf32 for AVX2, 16xf32 for AVX-512).
These are naturally bounded -- Kani can verify them exhaustively.

**Production precedent:**
- AWS Firecracker: 27 Kani harnesses verified VirtIO and rate limiter.
  Found a rounding bug allowing guests to exceed I/O bandwidth by 0.01%.
- AWS s2n-quic: 30+ harnesses across QUIC protocol. Same harness runs as
  fuzz test OR Kani proof via Bolero framework.
- Rust standard library: Ongoing community verification effort using Kani.
