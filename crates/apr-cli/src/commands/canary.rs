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
pub enum CanaryCommands {
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

/// Loaded tensor data: name → (f32 values, shape)
type TensorDataMap = BTreeMap<String, (Vec<f32>, Vec<usize>)>;

/// Load tensor data from any supported format (APR, GGUF, SafeTensors).
/// Uses Rosetta Stone format detection to dispatch to the appropriate reader.
fn load_tensor_data(model_path: &Path) -> Result<TensorDataMap> {
    use aprender::format::rosetta::FormatType;

    let format = FormatType::from_magic(model_path)
        .or_else(|_| FormatType::from_extension(model_path))
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;

    match format {
        FormatType::SafeTensors => load_tensor_data_safetensors(model_path),
        FormatType::Gguf => load_tensor_data_gguf(model_path),
        FormatType::Apr => load_tensor_data_apr(model_path),
    }
}

/// Load tensor data from SafeTensors format.
fn load_tensor_data_safetensors(path: &Path) -> Result<TensorDataMap> {
    use aprender::serialization::safetensors::load_safetensors;

    let (metadata, raw_data) =
        load_safetensors(path).map_err(|e| CliError::ValidationFailed(e.clone()))?;

    let mut result = BTreeMap::new();
    for (name, info) in &metadata {
        let start = info.data_offsets[0];
        let end = info.data_offsets[1];
        let tensor_bytes = &raw_data[start..end];

        let data: Vec<f32> = tensor_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(bytes)
            })
            .collect();

        result.insert(name.clone(), (data, info.shape.clone()));
    }
    Ok(result)
}

/// Load tensor data from GGUF format.
fn load_tensor_data_gguf(path: &Path) -> Result<TensorDataMap> {
    use aprender::format::gguf::reader::GgufReader;

    let data = fs::read(path)?;
    let reader = GgufReader::from_bytes(data)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

    let mut result = BTreeMap::new();
    let tensor_names: Vec<String> = reader.tensors.iter().map(|t| t.name.clone()).collect();

    for name in &tensor_names {
        if let Ok((f32_data, shape)) = reader.get_tensor_f32(name) {
            result.insert(name.clone(), (f32_data, shape));
        }
    }
    Ok(result)
}

/// Load tensor data from APR v2 format.
fn load_tensor_data_apr(path: &Path) -> Result<TensorDataMap> {
    use aprender::format::v2::AprV2Reader;

    let data = fs::read(path)?;
    let reader = AprV2Reader::from_bytes(&data)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse APR: {e}")))?;

    let mut result = BTreeMap::new();
    let names = reader.tensor_names();

    for name in &names {
        if let Some(entry) = reader.get_tensor(name) {
            let shape = entry.shape.clone();
            if let Some(f32_data) = reader.get_tensor_as_f32(name) {
                result.insert((*name).to_string(), (f32_data, shape));
            }
        }
    }
    Ok(result)
}

/// Create a canary test from a model
fn create_canary(model_path: &Path, _input_path: &Path, output_path: &Path) -> Result<()> {
    use aprender::format::TensorStats;

    println!("{}", "=== APR Canary Create ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Output: {}", output_path.display());
    println!();

    // Validate model exists
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    // Load model tensors (auto-detects APR, GGUF, SafeTensors)
    println!("{}", "Loading model...".yellow());
    let tensor_data = load_tensor_data(model_path)?;

    // Compute tensor statistics
    println!("{}", "Computing tensor statistics...".yellow());
    let mut tensors = BTreeMap::new();

    for (name, (data, shape)) in &tensor_data {
        let stats = TensorStats::compute(name, data);

        tensors.insert(
            name.clone(),
            TensorCanary {
                shape: shape.clone(),
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
    print_canary_check_header(model_path, canary_path);
    validate_paths_exist(model_path, canary_path)?;

    let canary = load_canary_data(canary_path)?;

    println!("{}", "Loading model...".yellow());
    let tensor_data = load_tensor_data(model_path)?;

    println!("{}", "Comparing tensors...".yellow());
    println!();

    let results = compare_all_tensors_generic(&canary, &tensor_data);
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

/// Compare all tensors from canary against model tensors (format-agnostic).
fn compare_all_tensors_generic(
    canary: &CanaryData,
    tensor_data: &TensorDataMap,
) -> Vec<CanaryCheckResult> {
    canary
        .tensors
        .iter()
        .map(|(name, expected)| {
            tensor_data.get(name).map_or_else(
                || missing_tensor_result(name),
                |(data, shape)| compare_single_tensor_generic(name, expected, data, shape),
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

/// Compare a single tensor against expected canary values (format-agnostic).
fn compare_single_tensor_generic(
    name: &str,
    expected: &TensorCanary,
    data: &[f32],
    shape: &[usize],
) -> CanaryCheckResult {
    use aprender::format::TensorStats;

    let stats = TensorStats::compute(name, data);

    let shape_match = shape == expected.shape.as_slice();
    let mean_drift = compute_relative_drift(stats.mean, expected.mean);
    let std_drift = compute_relative_drift(stats.std, expected.std);

    let passed = shape_match && mean_drift <= MEAN_THRESHOLD && std_drift <= STD_THRESHOLD;
    let message =
        build_failure_message_generic(passed, shape_match, mean_drift, std_drift, expected, shape);

    CanaryCheckResult {
        tensor_name: name.to_string(),
        passed,
        mean_drift,
        std_drift,
        shape_match,
        message,
    }
}

/// Compute relative drift, handling near-zero expected values.
fn compute_relative_drift(actual: f32, expected: f32) -> f32 {
    if expected.abs() > 1e-6 {
        ((actual - expected) / expected).abs()
    } else {
        (actual - expected).abs()
    }
}

/// Build failure message if check failed (format-agnostic).
fn build_failure_message_generic(
    passed: bool,
    shape_match: bool,
    mean_drift: f32,
    std_drift: f32,
    expected: &TensorCanary,
    actual_shape: &[usize],
) -> Option<String> {
    if passed {
        return None;
    }

    Some(if !shape_match {
        format!(
            "Shape mismatch: expected {:?}, got {:?}",
            expected.shape, actual_shape
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "canary_tests.rs"]
mod tests;
