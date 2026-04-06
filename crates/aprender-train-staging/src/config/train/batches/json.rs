//! JSON batch loading

use super::super::demo::create_demo_batches;
use crate::error::{Error, Result};
use crate::train::Batch;
use crate::Tensor;
use std::path::Path;

/// Load batches from JSON file
pub fn load_json_batches(path: &Path, batch_size: usize) -> Result<Vec<Batch>> {
    println!("  Loading JSON: {}", path.display());

    // Try to load as JSON array of {input, target} objects
    let content = std::fs::read_to_string(path).map_err(|e| {
        Error::ConfigError(format!("Failed to read JSON {}: {}", path.display(), e))
    })?;

    #[derive(serde::Deserialize)]
    struct Example {
        input: Vec<f32>,
        target: Vec<f32>,
    }

    #[derive(serde::Deserialize)]
    struct DataFile {
        examples: Vec<Example>,
    }

    // Try structured format first
    if let Ok(data) = serde_json::from_str::<DataFile>(&content) {
        println!("  Loaded {} examples from JSON", data.examples.len());
        let batches: Vec<Batch> = data
            .examples
            .chunks(batch_size.max(1))
            .map(|chunk| {
                let input_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.input.clone()).collect();
                let target_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.target.clone()).collect();
                Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect();
        return Ok(batches);
    }

    // Try array of examples
    if let Ok(examples) = serde_json::from_str::<Vec<Example>>(&content) {
        println!("  Loaded {} examples from JSON array", examples.len());
        let batches: Vec<Batch> = examples
            .chunks(batch_size.max(1))
            .map(|chunk| {
                let input_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.input.clone()).collect();
                let target_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.target.clone()).collect();
                Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect();
        return Ok(batches);
    }

    eprintln!("Warning: Could not parse JSON data format, using demo data");
    Ok(create_demo_batches(batch_size))
}
