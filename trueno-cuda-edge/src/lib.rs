//! # trueno-cuda-edge
//!
//! GPU edge-case test framework for the Trueno ecosystem.
//!
//! This crate provides comprehensive tools for testing GPU code edge cases:
//!
//! - **Null Fuzzing**: Inject null pointers into kernel arguments
//! - **Shared Memory Probing**: Detect overflows and bank conflicts
//! - **Lifecycle Chaos**: Stress-test context creation/destruction
//! - **Quantization Oracle**: Verify CPU/GPU parity across formats
//! - **PTX Poison**: Mutation testing for PTX assembly
//! - **Supervision**: Erlang-style worker crash recovery
//!
//! ## Feature Flags
//!
//! - `cuda`: Enable CUDA driver FFI (requires NVIDIA driver)
//! - `full`: All features enabled
//!
//! ## Quick Start
//!
//! ```rust
//! use trueno_cuda_edge::{
//!     null_fuzzer::NonNullDevicePtr,
//!     shmem_prober::{ComputeCapability, check_allocation},
//!     supervisor::{SupervisionTree, SupervisionStrategy},
//!     falsification::FalsificationReport,
//! };
//!
//! // Guard type rejects null pointers
//! assert!(NonNullDevicePtr::<u8>::new(0).is_err());
//!
//! // Validate shared memory allocation
//! let ampere = ComputeCapability::new(8, 0);
//! assert!(check_allocation(ampere, 100 * 1024).is_ok());
//!
//! // Supervision tree
//! let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);
//!
//! // Track test coverage
//! let report = FalsificationReport::new();
//! assert_eq!(report.coverage(), 0.0);
//! ```
//!
//! ## PMAT Tickets
//!
//! This crate implements:
//!
//! | Ticket | Module | Description |
//! |--------|--------|-------------|
//! | TCE-001 | `null_fuzzer` | `NonNullDevicePtr` guard type |
//! | TCE-002 | `null_fuzzer` | Null injection strategies |
//! | TCE-003 | `null_fuzzer` | Propagation tracking |
//! | TCE-004 | `shmem_prober` | Boundary sentinel probing |
//! | TCE-005 | `shmem_prober` | Bank conflict injection |
//! | TCE-006 | `lifecycle_chaos` | Context chaos scenarios |
//! | TCE-007 | `lifecycle_chaos` | Leak detection |
//! | TCE-008 | `quant_oracle` | CPU/GPU parity checking |
//! | TCE-009 | `quant_oracle` | Boundary value generation |
//! | TCE-010 | `ptx_poison` | PTX mutation operators |
//! | TCE-011 | `ptx_poison` | PTX structural verification |
//! | TCE-012 | `supervisor` | Worker supervision tree |

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod falsification;
pub mod harness;
pub mod lifecycle_chaos;
pub mod null_fuzzer;
pub mod ptx_poison;
pub mod quant_oracle;
pub mod shmem_prober;
pub mod supervisor;

pub use error::{EdgeError, Result, Severity};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test_imports() {
        // Verify all modules are accessible
        let _ = null_fuzzer::NonNullDevicePtr::<u8>::new(1).unwrap();
        let _ = shmem_prober::ComputeCapability::new(8, 0);
        let _ = supervisor::SupervisionStrategy::OneForOne;
        let _ = lifecycle_chaos::ChaosScenario::all();
        let _ = quant_oracle::QuantFormat::Q4K;
        let _ = ptx_poison::default_mutators();
        let _ = falsification::all_claims();
        let _ = harness::is_worker_process();
    }
}
