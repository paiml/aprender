//! ML-driven compression algorithm selection for trueno-zram.
//!
//! This crate provides entropy-based analysis and machine learning models
//! to select the optimal compression algorithm for each memory page.
//!
//! ## GPU Routing
//!
//! The [`BatchClassifier`] implements intelligent routing of compression
//! workloads to scalar, SIMD, or GPU backends based on the 5x PCIe rule.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_precision_loss)] // Entropy calculations use f64 approximations
#![allow(clippy::doc_markdown)] // PCIe and other technical terms

pub mod classifier;
pub mod entropy;
pub mod model;

pub use classifier::{
    BatchClassifier, ComputeBackend, PageClassifier, GPU_BATCH_THRESHOLD, SIMD_BATCH_THRESHOLD,
};
pub use entropy::{EntropyCalculator, EntropyLevel};
