//! Training batch loading from various data formats

mod loader;
mod rebatch;
pub mod streaming;

#[cfg(not(target_arch = "wasm32"))]
mod json;
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
mod parquet;

#[cfg(test)]
mod tests;

pub use loader::load_training_batches;
pub use rebatch::rebatch;

#[cfg(not(target_arch = "wasm32"))]
pub use json::load_json_batches;
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
pub use parquet::load_parquet_batches;
