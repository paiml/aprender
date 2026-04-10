//! Tensor Layout Contract Validation (Issue #4)
//!
//! Implements automated validation against aprender's tensor-layout-v1.yaml contract.
//! This contract is THE SOURCE OF TRUTH for GGUF/SafeTensors→APR tensor conversion.
//!
//! # Validation Rules
//!
//! - F-LAYOUT-CONTRACT-001: All 2D weights are transposed
//! - F-LAYOUT-CONTRACT-002: lm_head shape matches kernel expectation (CRITICAL)
//! - F-LAYOUT-CONTRACT-003: 1D tensors unchanged
//! - F-LAYOUT-CONTRACT-004: Byte size matches kernel expectation
//! - F-LAYOUT-CONTRACT-005: No garbage output from lm_head
//!
//! # References
//!
//! - Contract file: `../aprender/contracts/tensor-layout-v1.yaml`
//! - Spec: Section E.8 of qwen2.5-coder-showcase-demo.md
//! - GH-202: lm_head shape bug that caused garbage output

// Debug format {:?} cannot be inlined
#![allow(clippy::uninlined_format_args)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Default path to the tensor layout contract relative to this repo.
pub const DEFAULT_CONTRACT_PATH: &str = "../aprender/contracts/tensor-layout-v1.yaml";

// ============================================================================
// Contract types (deserialized from YAML)
// ============================================================================

/// Top-level tensor layout contract.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorLayoutContract {
    /// Contract metadata.
    pub metadata: ContractMetadata,

    /// Format conventions (gguf, apr, safetensors).
    pub formats: HashMap<String, FormatConvention>,

    /// Kernel convention defining weight shapes.
    pub kernel: KernelConvention,

    /// Per-tensor specifications.
    pub tensors: HashMap<String, TensorSpec>,

    /// Validation rules for automated testing.
    pub validation_rules: Vec<ValidationRule>,

    /// Semantic validation configuration.
    #[serde(default)]
    pub semantic_validation: Option<SemanticValidation>,
}

/// Contract metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct ContractMetadata {
    /// Contract version.
    pub version: String,
    /// Creation date.
    pub created: String,
    /// Last update date.
    pub updated: String,
    /// Author.
    pub author: String,
    /// Description.
    pub description: String,
}

/// Format convention (layout and shape convention).
#[derive(Debug, Clone, Deserialize)]
pub struct FormatConvention {
    /// Layout: "row-major" or "column-major".
    pub layout: String,
    /// Shape convention description.
    pub shape_convention: String,
    /// Additional notes.
    #[serde(default)]
    pub note: Option<String>,
}

/// Kernel convention - source of truth for shapes.
#[derive(Debug, Clone, Deserialize)]
pub struct KernelConvention {
    /// Kernel function signature.
    pub signature: String,
    /// Weight shape convention.
    pub weight_shape: String,
    /// Computation description.
    pub computation: String,
    /// Byte calculation formula.
    pub byte_calculation: String,
    /// Block sizes for quantized types.
    pub block_sizes: HashMap<String, u32>,
    /// Elements per super-block.
    #[serde(rename = "QK_K")]
    pub qk_k: u32,
}

/// Per-tensor specification.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorSpec {
    /// GGUF tensor name.
    pub gguf_name: String,
    /// APR tensor name.
    pub apr_name: String,
    /// GGUF shape as string (e.g., "[hidden, vocab]").
    pub gguf_shape: String,
    /// APR shape as string (e.g., "[vocab, hidden]").
    pub apr_shape: String,
    /// Whether tensor needs transposition.
    pub transpose: bool,
    /// Kernel that uses this tensor.
    pub kernel: String,
    /// Kernel output dimension expression.
    #[serde(default)]
    pub kernel_out_dim: Option<String>,
    /// Kernel input dimension expression.
    #[serde(default)]
    pub kernel_in_dim: Option<String>,
    /// Validation expression.
    #[serde(default)]
    pub validation: Option<String>,
    /// Whether this is a critical tensor.
    #[serde(default)]
    pub critical: bool,
    /// Additional notes.
    #[serde(default)]
    pub note: Option<String>,
}

/// Validation rule from contract.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ValidationRule {
    /// Rule ID (e.g., "F-LAYOUT-CONTRACT-001").
    pub id: String,
    /// Rule name.
    pub name: String,
    /// Rule description.
    pub description: String,
    /// Severity: P0, P1, P2.
    pub severity: String,
    /// Whether this is critical.
    #[serde(default)]
    pub critical: bool,
    /// Reference ticket.
    #[serde(default)]
    pub reference: Option<String>,
}

/// Semantic validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct SemanticValidation {
    /// Density validation config.
    #[serde(default)]
    pub density: Option<DensityConfig>,
    /// Numeric validation config.
    #[serde(default)]
    pub numeric: Option<NumericConfig>,
    /// Distribution validation config.
    #[serde(default)]
    pub distribution: Option<DistributionConfig>,
}

/// Density validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DensityConfig {
    /// Max zero percentage for embeddings.
    pub embedding_max_zero_pct: f64,
    /// Max zero percentage for weights.
    pub weight_max_zero_pct: f64,
}

/// Numeric validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct NumericConfig {
    /// Allow NaN values.
    pub allow_nan: bool,
    /// Allow Inf values.
    pub allow_inf: bool,
}

/// Distribution validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DistributionConfig {
    /// Minimum L2 norm.
    pub min_l2_norm: f64,
    /// Require variation in values.
    pub require_variation: bool,
}

// ============================================================================
// Contract loader
// ============================================================================

/// Load the tensor layout contract from the default path.
///
/// # Errors
///
/// Returns an error if the contract file cannot be read or parsed.
pub fn load_contract() -> Result<TensorLayoutContract> {
    load_contract_from(DEFAULT_CONTRACT_PATH)
}

/// Load the tensor layout contract from a specific path.
///
/// # Errors
///
/// Returns an error if the contract file cannot be read or parsed.
pub fn load_contract_from<P: AsRef<Path>>(path: P) -> Result<TensorLayoutContract> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| {
        Error::Execution(format!(
            "Failed to read tensor layout contract from {}: {e}",
            path.display()
        ))
    })?;

    serde_yaml::from_str(&content).map_err(|e| {
        Error::Execution(format!(
            "Failed to parse tensor layout contract from {}: {e}",
            path.display()
        ))
    })
}

// ============================================================================
// Validation result types
// ============================================================================

/// Result of validating a tensor against the contract.
#[derive(Debug, Clone, Serialize)]
pub struct TensorValidationResult {
    /// Tensor name.
    pub tensor_name: String,
    /// Rule ID that was checked.
    pub rule_id: String,
    /// Whether validation passed.
    pub passed: bool,
    /// Details about the validation.
    pub details: String,
    /// Expected value/shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// Actual value/shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
}

/// Result of validating an entire model against the contract.
#[derive(Debug, Clone, Serialize)]
pub struct ModelValidationResult {
    /// Model path that was validated.
    pub model_path: PathBuf,
    /// Overall pass/fail status.
    pub passed: bool,
    /// Number of rules checked.
    pub rules_checked: usize,
    /// Number of rules passed.
    pub rules_passed: usize,
    /// Number of rules failed.
    pub rules_failed: usize,
    /// Individual tensor validation results.
    pub tensor_results: Vec<TensorValidationResult>,
    /// Critical failures (P0 violations).
    pub critical_failures: Vec<String>,
}

// ============================================================================
// Validation functions
// ============================================================================

/// Maximum SafeTensors header size (10MB should cover any model)
const MAX_HEADER_SIZE: usize = 10 * 1024 * 1024;

/// Validate a model file against the tensor layout contract.
///
/// # Arguments
///
/// * `model_path` - Path to the APR model file or directory
/// * `contract` - The loaded tensor layout contract
///
/// # Returns
///
/// Validation result with per-tensor details.
///
/// # Errors
///
/// This function does not currently return errors; all validation failures
/// are reported in the `ModelValidationResult`. The `Result` wrapper is
/// reserved for future I/O errors when parsing APR model files.
pub fn validate_model(
    model_path: &Path,
    contract: &TensorLayoutContract,
) -> Result<ModelValidationResult> {
    // Early returns for missing path or no safetensors files
    if let Some(early_result) = check_model_path_preconditions(model_path) {
        return Ok(early_result);
    }

    // Collect all tensor metadata and run validations
    let (results, critical_failures) = run_all_validations(model_path, contract);

    let rules_failed = results.iter().filter(|r| !r.passed).count();
    let rules_passed = results.iter().filter(|r| r.passed).count();

    Ok(ModelValidationResult {
        model_path: model_path.to_path_buf(),
        passed: critical_failures.is_empty() && rules_failed == 0,
        rules_checked: results.len(),
        rules_passed,
        rules_failed,
        tensor_results: results,
        critical_failures,
    })
}

/// Check model path preconditions, returning early result if validation cannot proceed
fn check_model_path_preconditions(model_path: &Path) -> Option<ModelValidationResult> {
    if !model_path.exists() {
        return Some(ModelValidationResult {
            model_path: model_path.to_path_buf(),
            passed: false,
            rules_checked: 0,
            rules_passed: 0,
            rules_failed: 1,
            tensor_results: vec![TensorValidationResult {
                tensor_name: "N/A".to_string(),
                rule_id: "FILE-EXISTS".to_string(),
                passed: false,
                details: format!("Model file not found: {}", model_path.display()),
                expected: Some("File exists".to_string()),
                actual: Some("File not found".to_string()),
            }],
            critical_failures: vec!["Model file not found".to_string()],
        });
    }

    let safetensors_files = find_safetensors_files(model_path);
    if safetensors_files.is_empty() {
        return Some(ModelValidationResult {
            model_path: model_path.to_path_buf(),
            passed: false,
            rules_checked: 0,
            rules_passed: 0,
            rules_failed: 1,
            tensor_results: vec![TensorValidationResult {
                tensor_name: "N/A".to_string(),
                rule_id: "FILE-FORMAT".to_string(),
                passed: false,
                details: format!("No SafeTensors files found in: {}", model_path.display()),
                expected: Some("At least one .safetensors file".to_string()),
                actual: Some("No SafeTensors files".to_string()),
            }],
            critical_failures: vec!["No SafeTensors files found".to_string()],
        });
    }

    None
}

/// Run all validation checks and collect results
fn run_all_validations(
    model_path: &Path,
    contract: &TensorLayoutContract,
) -> (Vec<TensorValidationResult>, Vec<String>) {
    let mut results = Vec::new();
    let mut critical_failures = Vec::new();

    let all_tensors = collect_tensor_metadata(model_path, &mut results);
    let config = find_and_load_config(model_path);

    // Validate lm_head (F-LAYOUT-CONTRACT-002 - CRITICAL)
    validate_lm_head(
        &all_tensors,
        &config,
        contract,
        &mut results,
        &mut critical_failures,
    );

    // Validate 2D tensors (F-LAYOUT-CONTRACT-001)
    validate_2d_tensors(contract, &all_tensors, &config, &mut results);

    // Validate 1D tensors (F-LAYOUT-CONTRACT-003)
    validate_1d_tensors(contract, &all_tensors, &config, &mut results);

    (results, critical_failures)
}

/// Collect tensor metadata from all SafeTensors files
fn collect_tensor_metadata(
    model_path: &Path,
    results: &mut Vec<TensorValidationResult>,
) -> HashMap<String, Vec<usize>> {
    let safetensors_files = find_safetensors_files(model_path);
    let mut all_tensors = HashMap::new();

    for file in &safetensors_files {
        match read_safetensors_metadata(file) {
            Ok(tensors) => all_tensors.extend(tensors),
            Err(e) => {
                results.push(TensorValidationResult {
                    tensor_name: file.display().to_string(),
                    rule_id: "PARSE-ERROR".to_string(),
                    passed: false,
                    details: format!("Failed to read SafeTensors metadata: {e}"),
                    expected: None,
                    actual: None,
                });
            }
        }
    }

    all_tensors
}

/// Validate lm_head shape (F-LAYOUT-CONTRACT-002 - GH-202 critical check)
fn validate_lm_head(
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &LayoutModelConfig,
    contract: &TensorLayoutContract,
    results: &mut Vec<TensorValidationResult>,
    critical_failures: &mut Vec<String>,
) {
    if let Some(lm_head_shape) = all_tensors.get("lm_head.weight") {
        let validation = validate_lm_head_shape(lm_head_shape, config, contract);
        if !validation.passed && validation.rule_id == "F-LAYOUT-CONTRACT-002" {
            critical_failures.push(validation.details.clone());
        }
        results.push(validation);
    }
}

include!("layout_model_config.rs");
