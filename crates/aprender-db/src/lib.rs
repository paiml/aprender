//! # Trueno-DB: GPU-First Embedded Analytics Database
//!
//! **Version**: 0.1.0 (Phase 1 MVP)
//!
//! Trueno-DB is a GPU-aware, compute-intensity-based embedded analytics database
//! designed for high-performance aggregations with graceful degradation from
//! GPU → SIMD → Scalar.
//!
//! ## Design Principles (Toyota Way Aligned)
//!
//! - **Muda elimination**: Kernel fusion minimizes `PCIe` transfers
//! - **Poka-Yoke safety**: Out-of-core execution prevents VRAM OOM
//! - **Genchi Genbutsu**: Physics-based cost model (5x rule for GPU dispatch)
//! - **Jidoka**: Backend equivalence tests (GPU == SIMD == Scalar)
//!
//! ## Example Usage (Phase 1 MVP)
//!
//! ```rust,no_run
//! use trueno_db::storage::StorageEngine;
//!
//! // Load Parquet file
//! let storage = StorageEngine::load_parquet("data/events.parquet")?;
//!
//! // Iterate over 128MB morsels (out-of-core execution)
//! for morsel in storage.morsels() {
//!     println!("Morsel: {} rows", morsel.num_rows());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod backend;
pub mod error;
pub mod experiment;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod kv;
pub mod query;
pub mod storage;
pub mod topk;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

pub use error::{Error, Result};

/// Database instance
pub struct Database {
    _private: (),
}

/// Backend selection strategy
#[derive(Debug, Clone, Copy)]
pub enum Backend {
    /// Cost-based dispatch (arithmetic intensity)
    CostBased,
    /// Force GPU execution
    Gpu,
    /// Force SIMD execution
    Simd,
}

impl Database {
    /// Create a new database builder
    #[must_use]
    pub fn builder() -> DatabaseBuilder {
        DatabaseBuilder::default()
    }
}

/// Database builder
#[derive(Default)]
pub struct DatabaseBuilder {
    _private: (),
}

impl DatabaseBuilder {
    /// Set backend selection strategy
    #[must_use]
    pub const fn backend(self, _backend: Backend) -> Self {
        self
    }

    /// Set morsel size for out-of-core execution (Poka-Yoke)
    #[must_use]
    pub const fn morsel_size_mb(self, _size: usize) -> Self {
        self
    }

    /// Build the database
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails
    pub const fn build(self) -> Result<Database> {
        Ok(Database { _private: () })
    }
}
