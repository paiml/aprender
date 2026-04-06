//! Tests for training configuration and batch loading

mod config_loading;
mod demo_batches;
mod rebatch;
mod train_yaml;

#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
mod arrow_conversion;
#[cfg(not(target_arch = "wasm32"))]
mod json_batches;
#[cfg(not(target_arch = "wasm32"))]
mod training_batches;
