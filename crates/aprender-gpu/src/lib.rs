//! # trueno-gpu: Pure Rust PTX Generation for NVIDIA CUDA
//!
//! Generate PTX assembly directly from Rust - no LLVM, no nvcc, no external dependencies.
//!
//! ## Philosophy
//!
//! **Own the Stack** - Build everything from first principles for complete control,
//! auditability, and reproducibility.
//!
//! ## Quick Start
//!
//! ```rust
//! use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};
//!
//! // Build a vector addition kernel
//! let module = PtxModule::new()
//!     .version(8, 0)
//!     .target("sm_70")
//!     .address_size(64);
//!
//! let ptx_source = module.emit();
//! assert!(ptx_source.contains(".version 8.0"));
//! ```
//!
//! ## Modules
//!
//! - [`ptx`] - PTX code generation (builder pattern)
//! - [`driver`] - CUDA driver API (minimal FFI, optional)
//! - [`kernels`] - Hand-optimized GPU kernels
//! - [`memory`] - GPU memory management
//! - [`backend`] - Multi-backend abstraction

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![deny(unsafe_op_in_unsafe_fn)]
// ============================================================================
// Development-phase lint allows - to be addressed incrementally
// ============================================================================
// Allow dead code during development - will be used as API expands
#![allow(dead_code)]
// Allow precision loss in non-critical floating point calculations
#![allow(clippy::cast_precision_loss)]
// Allow possible truncation - we handle 64-bit correctly
#![allow(clippy::cast_possible_truncation)]
// Allow format push string - not a critical performance path
#![allow(clippy::format_push_string)]
// Allow doc markdown for code references - these are placeholders
#![allow(clippy::doc_markdown)]
// Allow missing errors doc during initial development
#![allow(clippy::missing_errors_doc)]
// Allow unnecessary literal bound for backend trait
#![allow(clippy::unnecessary_literal_bound)]
// Allow manual div_ceil - will use std when stabilized
#![allow(clippy::manual_div_ceil)]
// Allow missing panics doc during initial development
#![allow(clippy::missing_panics_doc)]
// Allow cast_lossless - we intentionally use as for u32->u64
#![allow(clippy::cast_lossless)]
// Allow uninlined format args - stylistic preference
#![allow(clippy::uninlined_format_args)]
// Allow map_unwrap_or - more readable with map().unwrap_or()
#![allow(clippy::map_unwrap_or)]
// Allow redundant closure for method calls - clearer intent
#![allow(clippy::redundant_closure_for_method_calls)]
// Allow unused self - methods will use self as API expands
#![allow(clippy::unused_self)]
// Allow expect_used in tests and non-critical paths
#![allow(clippy::expect_used)]
// Allow too_many_lines during development - will be refactored
#![allow(clippy::too_many_lines)]
// Allow needless_range_loop - clearer intent in some algorithms
#![allow(clippy::needless_range_loop)]
// Allow float_cmp in tests where exact comparison is intended
#![allow(clippy::float_cmp)]
// Allow unused comparisons - some are defensive checks
#![allow(unused_comparisons)]
// Allow unwrap_used in tests
#![allow(clippy::unwrap_used)]
// Allow cast_sign_loss - we know values are positive
#![allow(clippy::cast_sign_loss)]
// Allow field_reassign_with_default - clearer test setup
#![allow(clippy::field_reassign_with_default)]
// Allow panic in tests
#![allow(clippy::panic)]
// Allow manual_range_contains - clearer in assertions
#![allow(clippy::manual_range_contains)]
// Allow default_constructed_unit_structs
#![allow(clippy::default_constructed_unit_structs)]
// Allow clone_on_copy - clearer intent
#![allow(clippy::clone_on_copy)]
// Allow absurd_extreme_comparisons - defensive checks
#![allow(clippy::absurd_extreme_comparisons)]
// Allow no_effect_underscore_binding - intentional in tests
#![allow(clippy::no_effect_underscore_binding)]
// Allow must_use_candidate - methods may return values not always needed
#![allow(clippy::must_use_candidate)]
// Allow manual_find - clearer intent in some cases
#![allow(clippy::manual_find)]
// Allow type_complexity - complex return types for tuples
#![allow(clippy::type_complexity)]
// Allow range_plus_one - clearer in some contexts
#![allow(clippy::range_plus_one)]
// Allow map_clone - clearer intent
#![allow(clippy::map_clone)]
// Allow manual_is_multiple_of - not yet stabilized
#![allow(clippy::manual_is_multiple_of)]
// Allow items_after_statements - const definitions in kernels
#![allow(clippy::items_after_statements)]
// Allow doc_lazy_continuation - doc formatting
#![allow(clippy::doc_lazy_continuation)]
// Allow useless_vec in tests - clearer intent
#![allow(clippy::useless_vec)]
// Allow similar names - k_h vs kt_h are semantically distinct (key vs key-transposed)
#![allow(clippy::similar_names)]
// Allow many single char names - standard matrix notation (a, b, m, n, k)
#![allow(clippy::many_single_char_names)]
// Allow doc nested refdefs - acceptable in list items
#![allow(clippy::doc_nested_refdefs)]
// Allow cloned instead of copied - semantic clarity
#![allow(clippy::cloned_instead_of_copied)]
// Allow too many arguments - GPU APIs require many parameters
#![allow(clippy::too_many_arguments)]
// Allow explicit lifetimes - clearer for complex lifetime relationships
#![allow(clippy::elidable_lifetime_names)]
// Allow manual slice size calculation - clearer intent
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::large_stack_arrays)]

pub mod backend;
/// CUDA driver FFI — feature-gated behind `cuda`. Default build has zero unsafe.
/// Will be deleted entirely once memory/resident is migrated to wgpu (§26 Phase 3).
#[cfg(feature = "cuda")]
pub mod driver;
/// PMAT-291: Tensor compute graph for GPU inference (reduces 430 dispatches to ~15)
pub mod graph;
/// PTX kernel generators — feature-gated behind `cuda`. Safe Rust (no unsafe blocks)
/// but produces PTX text that requires CUDA driver to execute. Dead code without `cuda`.
#[cfg(feature = "cuda")]
pub mod kernels;
/// GPU memory management — feature-gated behind `cuda` (uses driver FFI).
#[cfg(feature = "cuda")]
pub mod memory;
pub mod monitor;
/// PTX instruction builder — feature-gated behind `cuda`. Safe Rust.
#[cfg(feature = "cuda")]
pub mod ptx;

/// Error types for trueno-gpu operations
pub mod error;

/// E2E visual testing framework for GPU kernels
pub mod testing;

pub use error::{GpuError, Result};
pub use monitor::{cuda_device_count, cuda_monitoring_available, CudaDeviceInfo, CudaMemoryInfo};

// NOTE: ComputeBrick is available from the trueno crate, not trueno-gpu
// This is because trueno optionally depends on trueno-gpu (not vice versa)
// Usage: `use trueno::brick::{ComputeBrick, ComputeBackend, TokenBudget};`
// See: trueno/src/brick.rs for the full brick architecture

#[cfg(test)]
mod tests {
    #[test]
    fn test_crate_compiles() {
        // Smoke test - crate compiles
        let _ = super::error::Result::<()>::Ok(());
    }
}
