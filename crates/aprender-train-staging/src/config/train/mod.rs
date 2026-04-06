//! Single-command training from YAML configuration
//!
//! This module provides the main entry points for declarative training via YAML configs.

#[cfg(feature = "parquet")]
mod arrow;
mod batches;
mod demo;
mod loader;

// Re-export public API (allow unused for external consumers)
#[allow(unused_imports)]
pub use batches::{load_training_batches, rebatch};
#[allow(unused_imports)]
pub use demo::create_demo_batches;
pub use loader::{load_config, train_from_yaml, try_load_apr_for_inference};

// Conditionally export parquet/json loaders for non-WASM
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
#[allow(unused_imports)]
pub use arrow::arrow_array_to_f32;
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
pub use batches::load_json_batches;
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
#[allow(unused_imports)]
pub use batches::load_parquet_batches;

#[cfg(test)]
mod tests;
