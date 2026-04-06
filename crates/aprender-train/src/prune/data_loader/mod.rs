//! Calibration data loader for pruning
//!
//! Provides data loading utilities for collecting activation statistics
//! during calibration for pruning methods like Wanda and SparseGPT.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::prune::{CalibrationDataLoader, CalibrationDataConfig};
//!
//! let config = CalibrationDataConfig::new()
//!     .with_num_samples(128)
//!     .with_batch_size(4);
//!
//! let loader = CalibrationDataLoader::new(config);
//! for batch in loader.iter() {
//!     // Process batch for calibration
//! }
//! ```

mod config;
mod iter;
mod loader;

#[cfg(test)]
mod tests;

pub use config::CalibrationDataConfig;
pub use loader::CalibrationDataLoader;

// CalibrationDataIter is publicly exported for users who need to reference
// the iterator type returned by CalibrationDataLoader::iter().
#[allow(unused_imports)]
pub use iter::CalibrationDataIter;
