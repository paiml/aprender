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
use clap::Subcommand;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Canary subcommands
#[derive(Subcommand, Clone, Debug)]
pub(crate) enum CanaryCommands {
    /// Create a canary test
    Create {
        /// Model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Input file (e.g. wav)
        #[arg(long)]
        input: PathBuf,

        /// Output json file
        #[arg(long)]
        output: PathBuf,
    },
    /// Check against a canary test
    Check {
        /// Model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Canary json file
        #[arg(long)]
        canary: PathBuf,
    },
}

/// Canary test data (serialized to JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CanaryData {
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
pub(crate) struct TensorCanary {
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
pub(crate) struct CanaryCheckResult {
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
        load_safetensors(model_path).map_err(|e| CliError::ValidationFailed(e.clone()))?;

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

/// Drift thresholds for canary checks.
const MEAN_THRESHOLD: f32 = 0.1; // 10% drift allowed
const STD_THRESHOLD: f32 = 0.2; // 20% std drift allowed

/// Check a model against a canary test
fn check_canary(model_path: &Path, canary_path: &Path) -> Result<()> {
    use aprender::serialization::safetensors::load_safetensors;

    print_canary_check_header(model_path, canary_path);
    validate_paths_exist(model_path, canary_path)?;

    let canary = load_canary_data(canary_path)?;

    println!("{}", "Loading model...".yellow());
    let (metadata, raw_data) =
        load_safetensors(model_path).map_err(|e| CliError::ValidationFailed(e.clone()))?;

    println!("{}", "Comparing tensors...".yellow());
    println!();

    let results = compare_all_tensors(&canary, &metadata, &raw_data);
    display_canary_results(&results, canary.tensor_count)
}

/// Print the canary check header.
fn print_canary_check_header(model_path: &Path, canary_path: &Path) {
    println!("{}", "=== APR Canary Check ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Canary: {}", canary_path.display());
    println!();
}

/// Validate that both paths exist.
fn validate_paths_exist(model_path: &Path, canary_path: &Path) -> Result<()> {
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }
    if !canary_path.exists() {
        return Err(CliError::FileNotFound(canary_path.to_path_buf()));
    }
    Ok(())
}

/// Load and parse canary data from JSON file.
fn load_canary_data(canary_path: &Path) -> Result<CanaryData> {
    let canary_json = fs::read_to_string(canary_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read canary file: {e}")))?;
    serde_json::from_str(&canary_json)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse canary file: {e}")))
}

/// Compare all tensors from canary against model tensors.
fn compare_all_tensors(
    canary: &CanaryData,
    metadata: &BTreeMap<String, aprender::serialization::safetensors::TensorMetadata>,
    raw_data: &[u8],
) -> Vec<CanaryCheckResult> {
    canary
        .tensors
        .iter()
        .map(|(name, expected)| {
            metadata.get(name).map_or_else(
                || missing_tensor_result(name),
                |info| compare_single_tensor(name, expected, info, raw_data),
            )
        })
        .collect()
}

/// Create result for a missing tensor.
fn missing_tensor_result(name: &str) -> CanaryCheckResult {
    CanaryCheckResult {
        tensor_name: name.to_string(),
        passed: false,
        mean_drift: f32::MAX,
        std_drift: f32::MAX,
        shape_match: false,
        message: Some("Tensor not found in model".to_string()),
    }
}

/// Compare a single tensor against expected canary values.
fn compare_single_tensor(
    name: &str,
    expected: &TensorCanary,
    info: &aprender::serialization::safetensors::TensorMetadata,
    raw_data: &[u8],
) -> CanaryCheckResult {
    use aprender::format::TensorStats;

    let data = extract_tensor_data(info, raw_data);
    let stats = TensorStats::compute(name, &data);

    let shape_match = info.shape == expected.shape;
    let mean_drift = compute_relative_drift(stats.mean, expected.mean);
    let std_drift = compute_relative_drift(stats.std, expected.std);

    let passed = shape_match && mean_drift <= MEAN_THRESHOLD && std_drift <= STD_THRESHOLD;
    let message = build_failure_message(passed, shape_match, mean_drift, std_drift, expected, info);

    CanaryCheckResult {
        tensor_name: name.to_string(),
        passed,
        mean_drift,
        std_drift,
        shape_match,
        message,
    }
}

/// Extract f32 tensor data from raw bytes.
fn extract_tensor_data(
    info: &aprender::serialization::safetensors::TensorMetadata,
    raw_data: &[u8],
) -> Vec<f32> {
    let start = info.data_offsets[0];
    let end = info.data_offsets[1];
    let tensor_bytes = &raw_data[start..end];

    tensor_bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
            f32::from_le_bytes(bytes)
        })
        .collect()
}

/// Compute relative drift, handling near-zero expected values.
fn compute_relative_drift(actual: f32, expected: f32) -> f32 {
    if expected.abs() > 1e-6 {
        ((actual - expected) / expected).abs()
    } else {
        (actual - expected).abs()
    }
}

/// Build failure message if check failed.
fn build_failure_message(
    passed: bool,
    shape_match: bool,
    mean_drift: f32,
    std_drift: f32,
    expected: &TensorCanary,
    info: &aprender::serialization::safetensors::TensorMetadata,
) -> Option<String> {
    if passed {
        return None;
    }

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
}

/// Display canary check results and return final status.
fn display_canary_results(results: &[CanaryCheckResult], tensor_count: usize) -> Result<()> {
    println!("{}", "=== Canary Check Results ===".cyan().bold());
    println!();

    let mut passed_count = 0;
    let mut failed_count = 0;

    for result in results {
        if result.passed {
            passed_count += 1;
            println!("[{}] {}", "PASS".green(), result.tensor_name);
            println!(
                "       mean_drift: {:.4}, std_drift: {:.4}, shape_match: {}",
                result.mean_drift, result.std_drift, result.shape_match
            );
        } else {
            failed_count += 1;
            println!("[{}] {}", "FAIL".red(), result.tensor_name);
            println!(
                "       mean_drift: {:.4}, std_drift: {:.4}, shape_match: {}",
                result.mean_drift, result.std_drift, result.shape_match
            );
            if let Some(ref msg) = result.message {
                println!("       {}", msg.yellow());
            }
        }
    }

    println!();
    println!("Results: {passed_count} passed, {failed_count} failed out of {tensor_count} tensors");

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
            "{failed_count} of {tensor_count} tensors failed canary check",
        )))
    }
}
