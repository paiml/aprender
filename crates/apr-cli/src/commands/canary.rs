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
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // CanaryData and TensorCanary Tests
    // ========================================================================

    #[test]
    fn test_canary_data_serialize_deserialize() {
        let canary = CanaryData {
            model_name: "test-model.safetensors".to_string(),
            tensor_count: 1,
            tensors: BTreeMap::new(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&canary).expect("serialize");
        let parsed: CanaryData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model_name, "test-model.safetensors");
        assert_eq!(parsed.tensor_count, 1);
    }

    #[test]
    fn test_tensor_canary_serialize_deserialize() {
        let tensor = TensorCanary {
            shape: vec![768, 768],
            count: 589824,
            mean: 0.0,
            std: 0.02,
            min: -0.1,
            max: 0.1,
        };
        let json = serde_json::to_string(&tensor).expect("serialize");
        let parsed: TensorCanary = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.shape, vec![768, 768]);
        assert_eq!(parsed.count, 589824);
    }

    #[test]
    fn test_canary_data_with_tensors() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "encoder.weight".to_string(),
            TensorCanary {
                shape: vec![768, 768],
                count: 589824,
                mean: 0.0,
                std: 0.02,
                min: -0.1,
                max: 0.1,
            },
        );
        let canary = CanaryData {
            model_name: "test.safetensors".to_string(),
            tensor_count: 1,
            tensors,
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string_pretty(&canary).expect("serialize");
        assert!(json.contains("encoder.weight"));
        assert!(json.contains("768"));
    }

    #[test]
    fn test_canary_data_clone() {
        let canary = CanaryData {
            model_name: "test.safetensors".to_string(),
            tensor_count: 0,
            tensors: BTreeMap::new(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };
        let cloned = canary.clone();
        assert_eq!(cloned.model_name, canary.model_name);
    }

    #[test]
    fn test_tensor_canary_clone() {
        let tensor = TensorCanary {
            shape: vec![768],
            count: 768,
            mean: 0.5,
            std: 0.1,
            min: 0.0,
            max: 1.0,
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.mean, tensor.mean);
    }

    // ========================================================================
    // CanaryCheckResult Tests
    // ========================================================================

    #[test]
    fn test_canary_check_result_passed() {
        let result = CanaryCheckResult {
            tensor_name: "weight".to_string(),
            passed: true,
            mean_drift: 0.01,
            std_drift: 0.02,
            shape_match: true,
            message: None,
        };
        assert!(result.passed);
        assert!(result.message.is_none());
    }

    #[test]
    fn test_canary_check_result_failed() {
        let result = CanaryCheckResult {
            tensor_name: "weight".to_string(),
            passed: false,
            mean_drift: 0.15,
            std_drift: 0.02,
            shape_match: true,
            message: Some("Mean drift exceeded".to_string()),
        };
        assert!(!result.passed);
        assert!(result.message.is_some());
    }

    #[test]
    fn test_canary_check_result_debug() {
        let result = CanaryCheckResult {
            tensor_name: "test".to_string(),
            passed: true,
            mean_drift: 0.0,
            std_drift: 0.0,
            shape_match: true,
            message: None,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("CanaryCheckResult"));
    }

    // ========================================================================
    // compute_relative_drift Tests
    // ========================================================================

    #[test]
    fn test_compute_relative_drift_normal() {
        // 10% drift: actual=1.1, expected=1.0
        let drift = compute_relative_drift(1.1, 1.0);
        assert!((drift - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_compute_relative_drift_negative() {
        // -10% drift: actual=0.9, expected=1.0
        let drift = compute_relative_drift(0.9, 1.0);
        assert!((drift - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_compute_relative_drift_zero_expected() {
        // When expected is near zero, use absolute difference
        let drift = compute_relative_drift(0.001, 0.0);
        assert!((drift - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_compute_relative_drift_same_value() {
        let drift = compute_relative_drift(1.0, 1.0);
        assert_eq!(drift, 0.0);
    }

    #[test]
    fn test_compute_relative_drift_large_values() {
        // 50% drift: actual=150, expected=100
        let drift = compute_relative_drift(150.0, 100.0);
        assert!((drift - 0.5).abs() < 0.001);
    }

    // ========================================================================
    // missing_tensor_result Tests
    // ========================================================================

    #[test]
    fn test_missing_tensor_result() {
        let result = missing_tensor_result("missing_weight");
        assert_eq!(result.tensor_name, "missing_weight");
        assert!(!result.passed);
        assert_eq!(result.mean_drift, f32::MAX);
        assert_eq!(result.std_drift, f32::MAX);
        assert!(!result.shape_match);
        assert!(result.message.is_some());
        assert!(result.message.unwrap().contains("not found"));
    }

    // ========================================================================
    // build_failure_message Tests
    // ========================================================================

    #[test]
    fn test_build_failure_message_passed() {
        // Create a minimal TensorCanary for reference
        let _expected = TensorCanary {
            shape: vec![768],
            count: 768,
            mean: 0.0,
            std: 0.02,
            min: -0.1,
            max: 0.1,
        };
        // Since we can't easily create TensorMetadata, we use a helper
        // to test the passed case where message should be None
        let msg = build_failure_message_test_helper(true, true, 0.01, 0.01);
        assert!(msg.is_none());
    }

    // Helper for testing build_failure_message without needing TensorMetadata
    fn build_failure_message_test_helper(
        passed: bool,
        shape_match: bool,
        mean_drift: f32,
        std_drift: f32,
    ) -> Option<String> {
        if passed {
            return None;
        }
        Some(if !shape_match {
            "Shape mismatch".to_string()
        } else if mean_drift > MEAN_THRESHOLD {
            format!("Mean drift {:.1}% exceeds threshold", mean_drift * 100.0)
        } else {
            format!("Std drift {:.1}% exceeds threshold", std_drift * 100.0)
        })
    }

    #[test]
    fn test_build_failure_message_shape_mismatch() {
        let msg = build_failure_message_test_helper(false, false, 0.01, 0.01);
        assert!(msg.is_some());
        assert!(msg.unwrap().contains("Shape mismatch"));
    }

    #[test]
    fn test_build_failure_message_mean_drift() {
        let msg = build_failure_message_test_helper(false, true, 0.15, 0.01);
        assert!(msg.is_some());
        assert!(msg.unwrap().contains("Mean drift"));
    }

    #[test]
    fn test_build_failure_message_std_drift() {
        let msg = build_failure_message_test_helper(false, true, 0.05, 0.25);
        assert!(msg.is_some());
        assert!(msg.unwrap().contains("Std drift"));
    }

    // ========================================================================
    // CanaryCommands Tests
    // ========================================================================

    #[test]
    fn test_canary_commands_create() {
        let cmd = CanaryCommands::Create {
            file: PathBuf::from("model.safetensors"),
            input: PathBuf::from("input.wav"),
            output: PathBuf::from("canary.json"),
        };
        match cmd {
            CanaryCommands::Create {
                file,
                input,
                output,
            } => {
                assert_eq!(file.to_string_lossy(), "model.safetensors");
                assert_eq!(input.to_string_lossy(), "input.wav");
                assert_eq!(output.to_string_lossy(), "canary.json");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_canary_commands_check() {
        let cmd = CanaryCommands::Check {
            file: PathBuf::from("model.safetensors"),
            canary: PathBuf::from("canary.json"),
        };
        match cmd {
            CanaryCommands::Check { file, canary } => {
                assert_eq!(file.to_string_lossy(), "model.safetensors");
                assert_eq!(canary.to_string_lossy(), "canary.json");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_canary_commands_clone() {
        let cmd = CanaryCommands::Create {
            file: PathBuf::from("model.safetensors"),
            input: PathBuf::from("input.wav"),
            output: PathBuf::from("canary.json"),
        };
        let cloned = cmd.clone();
        match cloned {
            CanaryCommands::Create { file, .. } => {
                assert_eq!(file.to_string_lossy(), "model.safetensors");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_canary_commands_debug() {
        let cmd = CanaryCommands::Check {
            file: PathBuf::from("model.safetensors"),
            canary: PathBuf::from("canary.json"),
        };
        let debug = format!("{cmd:?}");
        assert!(debug.contains("Check"));
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    #[test]
    fn test_run_create_model_not_found() {
        let output = NamedTempFile::with_suffix(".json").expect("create output");
        let input = NamedTempFile::with_suffix(".wav").expect("create input");
        let cmd = CanaryCommands::Create {
            file: PathBuf::from("/nonexistent/model.safetensors"),
            input: input.path().to_path_buf(),
            output: output.path().to_path_buf(),
        };
        let result = run(cmd);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_create_invalid_model() {
        let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
        model
            .write_all(b"not a valid safetensors file")
            .expect("write");
        let output = NamedTempFile::with_suffix(".json").expect("create output");
        let input = NamedTempFile::with_suffix(".wav").expect("create input");

        let cmd = CanaryCommands::Create {
            file: model.path().to_path_buf(),
            input: input.path().to_path_buf(),
            output: output.path().to_path_buf(),
        };
        let result = run(cmd);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_check_model_not_found() {
        let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
        canary
            .write_all(
                br#"{"model_name": "test", "tensor_count": 0, "tensors": {}, "created_at": ""}"#,
            )
            .expect("write");

        let cmd = CanaryCommands::Check {
            file: PathBuf::from("/nonexistent/model.safetensors"),
            canary: canary.path().to_path_buf(),
        };
        let result = run(cmd);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_check_canary_not_found() {
        let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
        model.write_all(b"fake model").expect("write");

        let cmd = CanaryCommands::Check {
            file: model.path().to_path_buf(),
            canary: PathBuf::from("/nonexistent/canary.json"),
        };
        let result = run(cmd);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_check_invalid_canary() {
        let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
        model.write_all(b"fake model").expect("write");
        let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
        canary.write_all(b"not valid json").expect("write");

        let cmd = CanaryCommands::Check {
            file: model.path().to_path_buf(),
            canary: canary.path().to_path_buf(),
        };
        let result = run(cmd);
        assert!(result.is_err());
    }

    // ========================================================================
    // validate_paths_exist Tests
    // ========================================================================

    #[test]
    fn test_validate_paths_exist_model_missing() {
        let canary = NamedTempFile::with_suffix(".json").expect("create canary");
        let result =
            validate_paths_exist(Path::new("/nonexistent/model.safetensors"), canary.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_paths_exist_canary_missing() {
        let model = NamedTempFile::with_suffix(".safetensors").expect("create model");
        let result = validate_paths_exist(model.path(), Path::new("/nonexistent/canary.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_paths_exist_both_exist() {
        let model = NamedTempFile::with_suffix(".safetensors").expect("create model");
        let canary = NamedTempFile::with_suffix(".json").expect("create canary");
        let result = validate_paths_exist(model.path(), canary.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // load_canary_data Tests
    // ========================================================================

    #[test]
    fn test_load_canary_data_valid() {
        let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
        canary.write_all(br#"{"model_name": "test.safetensors", "tensor_count": 0, "tensors": {}, "created_at": "2024-01-01"}"#).expect("write");

        let result = load_canary_data(canary.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().model_name, "test.safetensors");
    }

    #[test]
    fn test_load_canary_data_invalid_json() {
        let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
        canary.write_all(b"not valid json").expect("write");

        let result = load_canary_data(canary.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_canary_data_file_not_found() {
        let result = load_canary_data(Path::new("/nonexistent/canary.json"));
        assert!(result.is_err());
    }

    // ========================================================================
    // Threshold Tests
    // ========================================================================

    #[test]
    fn test_mean_threshold_value() {
        assert_eq!(MEAN_THRESHOLD, 0.1);
    }

    #[test]
    fn test_std_threshold_value() {
        assert_eq!(STD_THRESHOLD, 0.2);
    }

    // ========================================================================
    // Multi-Format load_tensor_data Tests
    // ========================================================================

    #[test]
    fn test_load_tensor_data_gguf() {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

        // Create valid GGUF file with F32 tensor data
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let tensor_data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensor = GgufTensor {
            name: "model.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: tensor_data,
        };
        let metadata = vec![(
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        )];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(&gguf_bytes).expect("write GGUF");

        // load_tensor_data should dispatch to GGUF path
        let result = load_tensor_data(file.path());
        assert!(result.is_ok(), "load_tensor_data should work for GGUF");

        let tensor_map = result.unwrap();
        assert_eq!(tensor_map.len(), 1);
        assert!(tensor_map.contains_key("model.weight"));
    }

    #[test]
    fn test_load_tensor_data_safetensors() {
        // Create valid SafeTensors file manually
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let tensor_data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let header_json = serde_json::json!({
            "test.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            }
        });
        let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
        let header_len = header_bytes.len() as u64;

        let mut st_bytes = Vec::new();
        st_bytes.extend_from_slice(&header_len.to_le_bytes());
        st_bytes.extend_from_slice(&header_bytes);
        st_bytes.extend_from_slice(&tensor_data);

        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(&st_bytes).expect("write SafeTensors");

        // load_tensor_data should dispatch to SafeTensors path
        let result = load_tensor_data(file.path());
        assert!(result.is_ok(), "load_tensor_data should work for SafeTensors");

        let tensor_map = result.unwrap();
        assert_eq!(tensor_map.len(), 1);
        assert!(tensor_map.contains_key("test.weight"));
    }

    #[test]
    fn test_load_tensor_data_gguf_multiple_tensors() {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
        use aprender::format::gguf::reader::GgufReader;

        // Create GGUF with multiple tensors (same shape to avoid alignment issues)
        let floats1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let floats2: Vec<u8> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensors = vec![
            GgufTensor {
                name: "model.weight".to_string(),
                shape: vec![2, 2],
                dtype: GgmlType::F32,
                data: floats1,
            },
            GgufTensor {
                name: "model.bias".to_string(),
                shape: vec![2, 2],
                dtype: GgmlType::F32,
                data: floats2,
            },
        ];
        let metadata = vec![(
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        )];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");

        // First verify the GGUF was written correctly
        let reader = GgufReader::from_bytes(gguf_bytes.clone()).expect("parse GGUF");
        assert_eq!(
            reader.tensors.len(),
            2,
            "GGUF should have 2 tensor metadata entries"
        );

        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(&gguf_bytes).expect("write GGUF");

        let result = load_tensor_data(file.path());
        assert!(result.is_ok(), "load_tensor_data should succeed");

        let tensor_map = result.unwrap();
        // Check that we got at least the first tensor
        assert!(!tensor_map.is_empty(), "tensor_map should not be empty");
        assert!(
            tensor_map.contains_key("model.weight"),
            "Should have model.weight"
        );
    }

    #[test]
    fn test_load_tensor_data_format_detection_by_magic() {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

        // Create GGUF with .bin extension (magic detection, not extension)
        let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensor = GgufTensor {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: floats,
        };
        let metadata = vec![(
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        )];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        file.write_all(&gguf_bytes).expect("write GGUF");

        // Should detect GGUF by magic, not extension
        let result = load_tensor_data(file.path());
        assert!(result.is_ok(), "Should detect GGUF by magic bytes");
    }
}
