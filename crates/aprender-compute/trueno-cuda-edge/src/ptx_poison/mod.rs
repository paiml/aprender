//! PTX poison mutation testing.
//!
//! Mutation testing injects faults into PTX assembly to verify that the
//! test suite can detect them. If a mutation survives (tests still pass),
//! the tests have insufficient coverage.
//!
//! # Mutation Operators
//!
//! 1. `FlipAddSub`: Replace `add` with `sub`
//! 2. `FlipMulDiv`: Replace `mul` with `div`
//! 3. `FlipMulLoHi`: Replace `.lo` with `.hi` in mul
//! 4. `InvertPredicate`: Flip comparison operators
//! 5. `RemoveBarrier`: Delete `bar.sync` instructions
//! 6. `ZeroRegister`: Replace register with `%r0`
//! 7. `WidenPrecision`: Change `.f32` to `.f64`
//! 8. `SwapMemorySpace`: Replace `.shared` with `.global`
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::ptx_poison::{PtxMutator, PtxVerifier, MINIMAL_VALID_PTX};
//!
//! // Apply mutation
//! let src = "add.f32 %r1, %r2, %r3;";
//! let mutated = PtxMutator::FlipAddSub.apply(src).unwrap();
//! assert!(mutated.contains("sub.f32"));
//!
//! // Verify PTX structure
//! let verifier = PtxVerifier::new();
//! assert!(verifier.verify(MINIMAL_VALID_PTX).is_ok());
//! assert!(verifier.verify("").is_err());
//! ```

pub mod mutator;
pub mod trap;
pub mod verifier;

pub use mutator::{default_mutators, PtxMutator};
pub use trap::{MutantResult, PoisonTrapReport, PtxPoisonTrapConfig};
pub use verifier::{PtxVerificationError, PtxVerifier, VerifiedPtx, MINIMAL_VALID_PTX};
