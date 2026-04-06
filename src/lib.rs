//! alimentar - Data Loading, Distribution and Tooling in Pure Rust
//!
//! A sovereign-first data loading library for the paiml AI stack.
//! Provides HuggingFace-compatible functionality without mandatory cloud
//! dependency.
//!
//! # Design Principles
//!
//! 1. **Sovereign-first** - Local storage default, no mandatory cloud
//!    dependency
//! 2. **Pure Rust** - No Python, no FFI (WASM-compatible)
//! 3. **Zero-copy** - Arrow `RecordBatch` throughout
//! 4. **Ecosystem aligned** - Arrow 53, Parquet 53
//!
//! # Quick Start
//!
//! ```no_run
//! use alimentar::{ArrowDataset, DataLoader};
//!
//! // Load a parquet file
//! let dataset = ArrowDataset::from_parquet("data/train.parquet").unwrap();
//!
//! // Create a data loader
//! let loader = DataLoader::new(dataset).batch_size(32).shuffle(true);
//!
//! // Iterate over batches
//! for batch in loader {
//!     println!("Batch with {} rows", batch.num_rows());
//! }
//! ```
// unsafe_code is forbidden except where explicitly allowed (e.g., mmap module)
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::approx_constant)]
#![allow(clippy::len_zero)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::float_cmp)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::needless_collect)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::similar_names)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::iter_on_single_items)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::cloned_ref_to_slice_refs)]
#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;
#[cfg(feature = "tokio-runtime")]
pub mod async_prefetch;
pub mod backend;
/// CLI module for command-line interface
#[cfg(feature = "cli")]
pub mod cli;
pub mod dataloader;
pub mod dataset;
pub mod datasets;
#[cfg(feature = "doctest")]
pub mod doctest;
pub mod drift;
pub mod error;
pub mod federated;
pub mod format;
#[cfg(feature = "hf-hub")]
pub mod hf_hub;
pub mod imbalance;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod parallel;
pub mod quality;
pub mod registry;
#[cfg(feature = "repl")]
pub mod repl;
pub mod serve;
pub mod sketch;
pub mod split;
pub mod streaming;
pub mod tensor;
pub mod transform;
/// TUI dataset viewer module
pub mod tui;
#[cfg(feature = "shuffle")]
pub mod weighted;
// Re-exports for convenience
// Re-export arrow types commonly needed
pub use arrow::{
    array::RecordBatch,
    datatypes::{Schema, SchemaRef},
};
#[cfg(feature = "tokio-runtime")]
pub use async_prefetch::{AsyncPrefetchBuilder, AsyncPrefetchDataset, SyncPrefetchDataset};
pub use dataloader::DataLoader;
pub use dataset::{ArrowDataset, CsvOptions, Dataset, JsonOptions};
#[cfg(feature = "doctest")]
pub use doctest::{DocTest, DocTestCorpus, DocTestParser};
pub use drift::{ColumnDrift, DriftDetector, DriftReport, DriftSeverity, DriftTest};
pub use error::{Error, Result};
pub use federated::{
    FederatedSplitCoordinator, FederatedSplitStrategy, GlobalSplitReport, NodeSplitInstruction,
    NodeSplitManifest, NodeSummary, SplitQualityIssue,
};
#[cfg(feature = "shuffle")]
pub use imbalance::resample;
pub use imbalance::{
    sqrt_inverse_weights, ClassDistribution, ImbalanceDetector, ImbalanceMetrics,
    ImbalanceRecommendation, ImbalanceReport, ImbalanceSeverity, ResampleStrategy,
};
#[cfg(feature = "mmap")]
pub use mmap::{MmapDataset, MmapDatasetBuilder};
pub use parallel::{ParallelDataLoader, ParallelDataLoaderBuilder};
pub use quality::{
    ColumnQuality, QualityChecker, QualityIssue, QualityProfile, QualityReport, TextColumnStats,
};
pub use sketch::{
    Centroid, DDSketch, DataSketch, DistributedDriftDetector, SketchDriftResult, SketchType,
    TDigest,
};
pub use split::DatasetSplit;
pub use transform::{
    Cast, Chain, Drop, FillNull, FillStrategy, Filter, Map, NormMethod, Normalize, Rename, Select,
    Skip, Sort, SortOrder, Take, Transform, Unique,
};
#[cfg(feature = "shuffle")]
pub use transform::{Fim, FimFormat, FimTokens, Sample, Shuffle};
pub use tui::{DatasetAdapter, DatasetViewer, RowDetailView, SchemaInspector, TuiError, TuiResult};
#[cfg(feature = "shuffle")]
pub use weighted::WeightedDataLoader;
