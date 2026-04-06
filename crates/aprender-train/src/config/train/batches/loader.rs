//! Main batch loading entry point

use super::super::demo::create_demo_batches;
use crate::config::schema::TrainSpec;
use crate::error::Result;
use crate::train::Batch;

#[cfg(not(target_arch = "wasm32"))]
use super::json::load_json_batches;
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
use super::parquet::load_parquet_batches;

/// Load training batches from data file using alimentar
///
/// Supports parquet, JSON, and CSV formats via alimentar.
/// Falls back to demo data if the file doesn't exist (for testing).
pub fn load_training_batches(spec: &TrainSpec) -> Result<Vec<Batch>> {
    let data_path = &spec.data.train;
    let batch_size = spec.data.batch_size;

    // Check if data file exists
    if !data_path.exists() {
        eprintln!("Warning: Training data not found at '{}', using demo data", data_path.display());
        return Ok(create_demo_batches(batch_size));
    }

    // Load data using alimentar (only on non-WASM)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let ext = data_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

        match ext.as_str() {
            #[cfg(feature = "parquet")]
            "parquet" => load_parquet_batches(data_path, batch_size),
            #[cfg(not(feature = "parquet"))]
            "parquet" => {
                eprintln!(
                    "Warning: Parquet support requires the 'parquet' feature, using demo data"
                );
                Ok(create_demo_batches(batch_size))
            }
            "json" => load_json_batches(data_path, batch_size),
            _ => {
                eprintln!("Warning: Unsupported data format '{ext}', using demo data");
                Ok(create_demo_batches(batch_size))
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        eprintln!("Warning: Data loading not available in WASM, using demo data");
        Ok(create_demo_batches(batch_size))
    }
}
