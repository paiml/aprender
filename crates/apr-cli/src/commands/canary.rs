//! Canary command implementation
//!
//! Implements APR-SPEC §4.8.5: Canary Inputs for Regression Testing
//!
//! Canary tests capture model tensor statistics and allow regression
//! testing after transformations like quantization or pruning.
//!
//! # Usage
//!
//! ```bash
//! # Create canary test suite from original model
//! apr canary create model.apr --input reference.wav --output canary.json
//!
//! # Check optimized model against canary
//! apr canary check model-optimized.apr --canary canary.json
//! ```

use crate::error::{CliError, Result};
use crate::CanaryCommands;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Canary test data (serialized to JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryData {
    /// Model name or path
    pub model_name: String,
    /// Tensor statistics for each tensor
    pub tensors: BTreeMap<String, TensorCanary>,
    /// Total number of tensors
    pub tensor_count: usize,
    /// Creation timestamp
    pub created_at: String,
}

/// Canary data for a single tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCanary {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Number of elements
    pub count: usize,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
}

/// Canary check result
#[derive(Debug, Clone)]
pub struct CanaryCheckResult {
    pub tensor_name: String,
    pub passed: bool,
    pub mean_drift: f32,
    pub std_drift: f32,
    pub shape_match: bool,
    pub message: Option<String>,
}

/// Run the canary command
pub(crate) fn run(command: CanaryCommands) -> Result<()> {
    match command {
        CanaryCommands::Create {
            file,
            input,
            output,
        } => create_canary(&file, &input, &output),
        CanaryCommands::Check { file, canary } => check_canary(&file, &canary),
    }
}

/// Create a canary test from a model
fn create_canary(model_path: &Path, _input_path: &Path, output_path: &Path) -> Result<()> {
    use aprender::format::TensorStats;
    use aprender::serialization::safetensors::load_safetensors;

    println!("{}", "=== APR Canary Create ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Output: {}", output_path.display());
    println!();

    // Validate model exists
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    // Load model tensors
    println!("{}", "Loading model...".yellow());
    let (metadata, raw_data) =
        load_safetensors(model_path).map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    // Compute tensor statistics
    println!("{}", "Computing tensor statistics...".yellow());
    let mut tensors = BTreeMap::new();

    for (name, info) in &metadata {
        // Extract tensor data
        let start = info.data_offsets[0];
        let end = info.data_offsets[1];
        let tensor_bytes = &raw_data[start..end];

        // Convert to f32
        let data: Vec<f32> = tensor_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(bytes)
            })
            .collect();

        // Compute statistics
        let stats = TensorStats::compute(name, &data);

        tensors.insert(
            name.clone(),
            TensorCanary {
                shape: info.shape.clone(),
                count: data.len(),
                mean: stats.mean,
                std: stats.std,
                min: stats.min,
                max: stats.max,
            },
        );
    }

    // Build canary data
    let canary = CanaryData {
        model_name: model_path.display().to_string(),
        tensor_count: tensors.len(),
        tensors,
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    // Write to JSON file
    let json = serde_json::to_string_pretty(&canary)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to serialize canary data: {e}")))?;

    fs::write(output_path, json)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write canary file: {e}")))?;

    println!();
    println!("{}", "=== Canary Created ===".cyan().bold());
    println!("Tensors captured: {}", canary.tensor_count);
    println!("Output file: {}", output_path.display());
    println!();
    println!("{}", "Canary test created successfully".green().bold());

    Ok(())
}

/// Check a model against a canary test
fn check_canary(model_path: &Path, canary_path: &Path) -> Result<()> {
    use aprender::format::TensorStats;
    use aprender::serialization::safetensors::load_safetensors;

    const MEAN_THRESHOLD: f32 = 0.1; // 10% drift allowed
    const STD_THRESHOLD: f32 = 0.2; // 20% std drift allowed

    println!("{}", "=== APR Canary Check ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Canary: {}", canary_path.display());
    println!();

    // Validate files exist
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }
    if !canary_path.exists() {
        return Err(CliError::FileNotFound(canary_path.to_path_buf()));
    }

    // Load canary data
    let canary_json = fs::read_to_string(canary_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read canary file: {e}")))?;
    let canary: CanaryData = serde_json::from_str(&canary_json)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse canary file: {e}")))?;

    // Load model tensors
    println!("{}", "Loading model...".yellow());
    let (metadata, raw_data) =
        load_safetensors(model_path).map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    // Compare tensors
    println!("{}", "Comparing tensors...".yellow());
    println!();

    let mut results: Vec<CanaryCheckResult> = Vec::new();
    let mut passed_count = 0;
    let mut failed_count = 0;

    for (name, expected) in &canary.tensors {
        let result = if let Some(info) = metadata.get(name) {
            // Extract tensor data
            let start = info.data_offsets[0];
            let end = info.data_offsets[1];
            let tensor_bytes = &raw_data[start..end];

            // Convert to f32
            let data: Vec<f32> = tensor_bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(bytes)
                })
                .collect();

            // Compute statistics
            let stats = TensorStats::compute(name, &data);

            // Check shape match
            let shape_match = info.shape == expected.shape;

            // Calculate drift
            let mean_drift = if expected.mean.abs() > 1e-6 {
                ((stats.mean - expected.mean) / expected.mean).abs()
            } else {
                (stats.mean - expected.mean).abs()
            };

            let std_drift = if expected.std.abs() > 1e-6 {
                ((stats.std - expected.std) / expected.std).abs()
            } else {
                (stats.std - expected.std).abs()
            };

            let passed = shape_match && mean_drift <= MEAN_THRESHOLD && std_drift <= STD_THRESHOLD;

            let message = if !passed {
                Some(if !shape_match {
                    format!(
                        "Shape mismatch: expected {:?}, got {:?}",
                        expected.shape, info.shape
                    )
                } else if mean_drift > MEAN_THRESHOLD {
                    format!(
                        "Mean drift {:.1}% exceeds threshold {:.1}%",
                        mean_drift * 100.0,
                        MEAN_THRESHOLD * 100.0
                    )
                } else {
                    format!(
                        "Std drift {:.1}% exceeds threshold {:.1}%",
                        std_drift * 100.0,
                        STD_THRESHOLD * 100.0
                    )
                })
            } else {
                None
            };

            CanaryCheckResult {
                tensor_name: name.clone(),
                passed,
                mean_drift,
                std_drift,
                shape_match,
                message,
            }
        } else {
            CanaryCheckResult {
                tensor_name: name.clone(),
                passed: false,
                mean_drift: f32::MAX,
                std_drift: f32::MAX,
                shape_match: false,
                message: Some("Tensor not found in model".to_string()),
            }
        };

        if result.passed {
            passed_count += 1;
        } else {
            failed_count += 1;
        }
        results.push(result);
    }

    // Display results
    println!("{}", "=== Canary Check Results ===".cyan().bold());
    println!();

    for result in &results {
        let status = if result.passed {
            "PASS".green()
        } else {
            "FAIL".red()
        };

        println!("[{}] {}", status, result.tensor_name);

        if !result.passed {
            if let Some(ref msg) = result.message {
                println!("       {}", msg.yellow());
            }
        }
    }

    println!();
    println!(
        "Results: {} passed, {} failed out of {} tensors",
        passed_count, failed_count, canary.tensor_count
    );

    if failed_count == 0 {
        println!();
        println!(
            "{}",
            "Canary check PASSED - model within tolerance"
                .green()
                .bold()
        );
        Ok(())
    } else {
        println!();
        println!(
            "{}",
            "Canary check FAILED - model drifted beyond tolerance"
                .red()
                .bold()
        );
        Err(CliError::ValidationFailed(format!(
            "{} of {} tensors failed canary check",
            failed_count, canary.tensor_count
        )))
    }
}
