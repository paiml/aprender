//! Main batch loading entry point

use crate::config::schema::TrainSpec;
use crate::error::{Error, Result};
use crate::train::Batch;

#[cfg(not(target_arch = "wasm32"))]
use super::json::load_json_batches;
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
use super::parquet::load_parquet_batches;

/// The documented on-disk schema for `--task pretrain` JSON training data.
///
/// Quoted verbatim in every load failure so the user is never left guessing
/// what the loader wanted.
pub(crate) const JSON_SCHEMA_HINT: &str = "expected JSON of the form \
     {\"examples\":[{\"input\":[f32,..],\"target\":[f32,..]}, ..]} \
     or a bare array [{\"input\":[..],\"target\":[..]}, ..]";

/// Load training batches from the dataset named by the config.
///
/// Supported formats: JSON (see [`JSON_SCHEMA_HINT`]) and, when the `parquet`
/// feature is enabled, Parquet via alimentar.
///
/// # Errors
///
/// Returns [`Error::ConfigError`] when the dataset is missing, is in a format
/// this build cannot read, or cannot be parsed. It NEVER substitutes synthetic
/// data for a dataset it failed to read: a training run that silently trains on
/// fabricated examples and reports success is worse than one that refuses to
/// start.
pub fn load_training_batches(spec: &TrainSpec) -> Result<Vec<Batch>> {
    let data_path = &spec.data.train;
    let batch_size = spec.data.batch_size;

    // Check if data file exists
    if !data_path.exists() {
        return Err(Error::ConfigError(format!(
            "Training data not found at '{}'. Training cannot proceed without it.",
            data_path.display()
        )));
    }

    // Load data using alimentar (only on non-WASM)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let ext = data_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

        match ext.as_str() {
            #[cfg(feature = "parquet")]
            "parquet" => load_parquet_batches(data_path, batch_size),
            #[cfg(not(feature = "parquet"))]
            "parquet" => Err(Error::ConfigError(format!(
                "Cannot read Parquet training data '{}': this build lacks the 'parquet' feature. \
                 Rebuild with --features parquet, or convert the dataset to JSON ({JSON_SCHEMA_HINT}).",
                data_path.display()
            ))),
            "json" => load_json_batches(data_path, batch_size),
            _ => Err(Error::ConfigError(format!(
                "Unsupported training data format '{ext}' for '{}'. Supported: {}. \
                 Convert the dataset to JSON — {JSON_SCHEMA_HINT}.",
                data_path.display(),
                supported_formats()
            ))),
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = batch_size;
        Err(Error::ConfigError(
            "Data loading is not available in WASM builds; training cannot proceed.".to_string(),
        ))
    }
}

/// Human-readable list of the dataset formats this build can actually read.
#[cfg(not(target_arch = "wasm32"))]
fn supported_formats() -> &'static str {
    #[cfg(feature = "parquet")]
    {
        "json, parquet"
    }
    #[cfg(not(feature = "parquet"))]
    {
        "json"
    }
}
