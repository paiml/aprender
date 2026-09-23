# The Verification Ladder

Every proof obligation in a contract is verified at multiple levels. Higher
levels subsume lower ones. The goal is to push every obligation as high as
practically possible.

<!-- generated from ProofLevel; do not edit -->
| Level | Method |
|-------|--------|
| L5 | Lean 4 theorem proved + every binding verified implemented |
| L4 | Lean 4 theorem proved |
| L3 | Kani bounded model check |
| L2 | Falsification tests cover the obligations |
| L1 | Contract YAML with equations |

L4 and L5 are self-declared until PVL-001 EV-8b lands: the level is computed from the contract's own YAML, not from a checked Lean discharge summary.
<!-- end generated from ProofLevel -->

## Where Each Tool Lives

| Obligation Type | Types (rustc) | L2 (probar/proptest) | L3 (Kani) | L4 (Lean) |
|----------------|-----------------|-------------------|-----------------|----------------|
| Shape correctness | `ValidatedTensor` newtype | N/A (compile-time) | N/A (compile-time) | N/A |
| Softmax sums to 1 | N/A | proptest random vectors | `#[kani::proof]` all vectors <= 16 | `partition_of_unity` (proved) |
| SIMD = scalar | N/A | proptest random data | `#[kani::proof]` all data <= 256 | N/A (empirical) |
| No overflow | N/A | proptest edge cases | Kani automatic (checks ALL paths) | N/A |
| Quantized bsums correct | N/A | proptest random blocks | `#[kani::proof]` all blocks (integer-exact) | N/A |
| Format isolation | `#[test]` cross-format | N/A | `#[kani::proof]` + `#[kani::should_panic]` | N/A |

## The Provability Claim

When we say a kernel is "provable," we mean:

1. **Types:** the type system prevents invalid construction (Poka-Yoke).
2. **L2:** falsification tests (probar/proptest) have exercised the property for
   10,000+ random inputs.
3. **L3:** Kani has exhaustively verified the property for ALL inputs up
   to the kernel's natural bound (super-block size, SIMD width, etc.).

For fixed-size kernel operations -- which is what ML inference IS -- bounded
verification at the natural bound IS exhaustive. A Q4_K super-block is always
256 elements. Verifying for all 256-element inputs IS verifying for all inputs.

Phase 7 (L4) extends this to **unbounded proofs** via Lean 4 for
algebraic identities like `Σ softmax(x)_i = 1` that hold regardless of vector
length. See [Phase 7: Prove](./phase-7-prove.md) for details.
