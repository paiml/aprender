//! HuggingFace Dataset Fetcher and Collator
//!
//! Provides dataset loading and batching for distillation training.
//!
//! # Features
//!
//! - Streaming support for large datasets
//! - Parquet file loading
//! - Dynamic padding and batching
//! - Teacher output caching

mod batch;
mod cache;
mod collator;
mod dataset_impl;
mod example;
mod fetcher;
mod options;
mod split;

#[cfg(test)]
mod tests;

// Re-export all public types for API compatibility
pub use batch::Batch;
pub use cache::{CacheStats, TeacherCache};
pub use collator::DistillationCollator;
pub use dataset_impl::Dataset;
pub use example::Example;
pub use fetcher::HfDatasetFetcher;
pub use options::DatasetOptions;
pub use split::Split;
