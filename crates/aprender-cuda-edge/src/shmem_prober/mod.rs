//! Shared memory boundary probing and bank conflict analysis.
//!
//! CUDA shared memory is a limited per-SM resource with specific access
//! patterns that can cause bank conflicts. This module provides tools to:
//!
//! - Validate allocations against hardware limits
//! - Detect out-of-bounds writes via sentinel guards
//! - Inject and measure bank conflicts
//!
//! # Compute Capabilities
//!
//! | Arch | SM | Shared Memory |
//! |------|-----|---------------|
//! | Kepler | 3.x | 48 KB |
//! | Maxwell | 5.x | 48 KB |
//! | Pascal | 6.x | 48 KB |
//! | Volta/Turing | 7.x | 96 KB |
//! | Ampere | 8.x | 164 KB |
//! | Hopper | 9.x | 228 KB |
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::shmem_prober::{ComputeCapability, shared_memory_limit, check_allocation};
//!
//! let ampere = ComputeCapability::new(8, 0);
//! assert_eq!(shared_memory_limit(ampere), 164 * 1024);
//! assert!(check_allocation(ampere, 100 * 1024).is_ok());
//! assert!(check_allocation(ampere, 200 * 1024).is_err());
//! ```

pub mod bank_conflict;
pub mod boundary;
pub mod overflow;

pub use bank_conflict::{BankConflictInjector, BankConflictResult};
pub use boundary::{
    check_sentinels, compute_sentinel_offsets, BoundaryProbeReport, BoundaryViolation,
    SharedMemoryRegion, SENTINEL_AFTER, SENTINEL_BEFORE, SENTINEL_SIZE,
};
pub use overflow::{check_allocation, shared_memory_limit, AccessPattern, ComputeCapability};
