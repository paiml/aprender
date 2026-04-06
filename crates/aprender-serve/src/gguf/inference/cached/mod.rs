//! Cached model wrappers for efficient GPU inference
//!
//! This module contains:
//! - `single.rs`: OwnedQuantizedModelCached (RefCell-based, single-threaded)
//! - `sync.rs`: OwnedQuantizedModelCachedSync (Mutex-based, thread-safe)
//! - `weights.rs`: DequantizedWeightCache for GPU GEMM

mod single;
mod sync;
mod weights;

#[cfg(test)]
mod sync_tests;

pub use single::OwnedQuantizedModelCached;
pub use sync::OwnedQuantizedModelCachedSync;
pub use weights::{DequantizedFFNWeights, DequantizedWeightCache};
