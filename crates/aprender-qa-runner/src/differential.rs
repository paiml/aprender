//! Differential Testing (GH-188, PMAT-114, PMAT-192)
//!
//! Implements differential testing capabilities:
//! - Tensor diff between models (rosetta diff-tensors)
//! - Inference comparison (rosetta compare-inference)
//! - Performance benchmarking (profile --diff-benchmark)
//! - Trace payload comparison (trace --payload --reference)
//!
//! # Toyota Way Principle
//!
//! "Genchi Genbutsu" (Go and see) - Don't trust that two implementations
//! are equivalent; verify by running both and comparing outputs.

use crate::error::{Error, Result};
use crate::provenance::{
    add_derived, create_source_provenance, get_apr_cli_version, load_provenance, save_provenance,
    validate_provenance, Provenance,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Result of `apr rosetta inspect --json` (T-GH192-01)
///
/// Parses model metadata including tensor count, tensor names,
/// and architecture parameters needed for cardinality and name-set gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectResult {
    /// Total number of tensors in the model
    pub tensor_count: usize,
    /// List of all tensor names
    #[serde(default)]
    pub tensor_names: Vec<String>,
    /// Number of attention heads (from model config)
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    /// Number of key-value heads (GQA/MQA config)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Hidden size / embedding dimension
    #[serde(default)]
    pub hidden_size: Option<usize>,
    /// Model architecture name (e.g., "Qwen2ForCausalLM")
    #[serde(default)]
    pub architecture: Option<String>,
}

/// Run `apr rosetta inspect --json <model>` and parse the result
///
/// Falls back to text-mode parsing for tensor count if JSON is unavailable.
///
/// # Errors
///
/// Returns an error if the apr command fails to execute.
pub fn run_inspect(model_path: &Path, apr_binary: &str) -> Result<InspectResult> {
    // Retry on ETXTBSY (os error 26): a transient condition on Linux where
    // fork() inherits write fds from other threads, causing execve() to fail.
    let output = {
        let mut attempts = 0;
        loop {
            match Command::new(apr_binary)
                .arg("rosetta")
                .arg("inspect")
                .arg(model_path)
                .arg("--json")
                .output()
            {
                Ok(output) => break output,
                Err(e) if e.raw_os_error() == Some(26) && attempts < 3 => {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(Error::ExecutionFailed {
                        command: "apr rosetta inspect --json".to_string(),
                        reason: e.to_string(),
                    });
                }
            }
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Try JSON parsing first
    if output.status.success() {
        if let Ok(result) = serde_json::from_str::<InspectResult>(&stdout) {
            return Ok(result);
        }
    }

    // Fall back to text output parsing
    parse_inspect_text(&stdout)
}

/// Try to parse a tensor count from a line (e.g., "Tensors: 338" or "tensor_count: 338")
fn parse_tensor_count_line(line: &str) -> Option<usize> {
    line.strip_prefix("Tensors:")
        .or_else(|| line.strip_prefix("tensor_count:"))
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Try to extract a tensor name from a dimension line
///
/// Lines like "model.layers.0.self_attn.q_proj.weight [4096, 4096]"
fn try_extract_tensor_name(line: &str) -> Option<String> {
    if !line.contains('[') || !line.contains(']') || line.starts_with('{') {
        return None;
    }
    let name = line.split_whitespace().next()?;
    if name.contains('.') {
        Some(name.to_string())
    } else {
        None
    }
}

/// Architecture metadata field prefixes.
/// Parsed from `apr rosetta inspect` output lines.
const ARCH_FIELD_PREFIXES: &[&str] = &[
    "num_attention_heads:",
    "num_key_value_heads:",
    "hidden_size:",
    "architecture:",
];

/// Parse architecture metadata fields from a single line.
///
/// Uses prefix matching against `ARCH_FIELD_PREFIXES`, then dispatches
/// to the appropriate field setter based on which prefix matched.
fn parse_architecture_line(line: &str, result: &mut InspectResult) {
    let Some((idx, val)) = ARCH_FIELD_PREFIXES
        .iter()
        .enumerate()
        .find_map(|(i, prefix)| line.strip_prefix(prefix).map(|v| (i, v.trim())))
    else {
        return;
    };
    match idx {
        0 => result.num_attention_heads = val.parse().ok(),
        1 => result.num_key_value_heads = val.parse().ok(),
        2 => result.hidden_size = val.parse().ok(),
        3 => result.architecture = Some(val.to_string()),
        _ => {}
    }
}

/// Parse text-mode output from `apr rosetta inspect`
///
/// Extracts tensor count and tensor names from human-readable output.
fn parse_inspect_text(output: &str) -> Result<InspectResult> {
    let mut result = InspectResult {
        tensor_count: 0,
        tensor_names: Vec::new(),
        num_attention_heads: None,
        num_key_value_heads: None,
        hidden_size: None,
        architecture: None,
    };

    for line in output.lines() {
        let line = line.trim();

        if let Some(count) = parse_tensor_count_line(line) {
            result.tensor_count = count;
        }

        if let Some(name) = try_extract_tensor_name(line) {
            result.tensor_names.push(name);
        }

        parse_architecture_line(line, &mut result);
    }

    // If we found tensor names but no explicit count, use the name count
    if result.tensor_count == 0 && !result.tensor_names.is_empty() {
        result.tensor_count = result.tensor_names.len();
    }

    Ok(result)
}

/// Result of tensor diff operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDiffResult {
    /// Total tensors compared
    pub total_tensors: usize,
    /// Tensors with shape mismatches
    pub mismatched_tensors: usize,
    /// Tensors with transposed dimensions (GGML vs standard)
    pub transposed_tensors: usize,
    /// Details of each mismatch
    pub mismatches: Vec<TensorMismatch>,
    /// Whether the diff passed (no critical mismatches)
    pub passed: bool,
}

/// A single tensor mismatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMismatch {
    /// Tensor name
    pub name: String,
    /// Shape in model A
    pub shape_a: Vec<usize>,
    /// Shape in model B
    pub shape_b: Vec<usize>,
    /// Type of mismatch
    pub mismatch_type: TensorMismatchType,
}

/// Type of tensor mismatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorMismatchType {
    /// Dimensions are transposed (e.g., [4096, 32000] vs [32000, 4096])
    Transposed,
    /// Dimensions are completely different
    ShapeMismatch,
    /// Tensor missing in one model
    Missing,
}

impl TensorMismatchType {
    /// Get the gate ID for this mismatch type
    #[must_use]
    #[allow(clippy::match_same_arms)] // ShapeMismatch and Missing share the same gate intentionally
    pub fn gate_id(&self) -> &'static str {
        match self {
            Self::Transposed => "F-ROSETTA-DIFF-001",
            Self::ShapeMismatch => "F-ROSETTA-DIFF-002",
            Self::Missing => "F-ROSETTA-DIFF-002",
        }
    }
}

/// Configuration for differential testing
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Path to APR CLI binary
    pub apr_binary: String,
    /// Filter pattern for tensor names
    pub filter: Option<String>,
    /// Only show mismatches
    pub mismatches_only: bool,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
}

impl Default for DiffConfig {
    /// Create a default configuration with apr binary, no filter, and 1e-5 tolerance
    fn default() -> Self {
        Self {
            apr_binary: "apr".to_string(),
            filter: None,
            mismatches_only: true,
            tolerance: 1e-5,
        }
    }
}

/// Differential test executor
pub struct DifferentialExecutor {
    config: DiffConfig,
}

impl DifferentialExecutor {
    /// Create a new differential executor
    #[must_use]
    pub fn new(config: DiffConfig) -> Self {
        Self { config }
    }

    /// Run tensor diff between two models
    ///
    /// Uses `apr rosetta diff-tensors` to compare tensor layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if the apr command fails to execute or returns non-zero.
    pub fn diff_tensors(&self, model_a: &Path, model_b: &Path) -> Result<TensorDiffResult> {
        let mut cmd = Command::new(&self.config.apr_binary);
        cmd.arg("rosetta")
            .arg("diff-tensors")
            .arg(model_a)
            .arg(model_b)
            .arg("--json");

        if self.config.mismatches_only {
            cmd.arg("--mismatches-only");
        }

        if let Some(filter) = &self.config.filter {
            cmd.arg("--filter").arg(filter);
        }

        let output = cmd.output().map_err(|e| Error::ExecutionFailed {
            command: format!("{cmd:?}"),
            reason: e.to_string(),
        })?;

        if !output.status.success() {
            // Try to parse error from stderr
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::ExecutionFailed {
                command: "apr rosetta diff-tensors".to_string(),
                reason: stderr.to_string(),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_diff_output(&stdout)
    }

    /// Parse diff-tensors JSON output
    fn parse_diff_output(&self, output: &str) -> Result<TensorDiffResult> {
        // Try to parse as JSON first
        if let Ok(result) = serde_json::from_str::<TensorDiffResult>(output) {
            return Ok(result);
        }

        // Fall back to parsing text output
        let mut mismatches = Vec::new();
        let mut transposed_count = 0;

        for line in output.lines() {
            if line.contains("TRANSPOSED") || line.contains("⚠️") {
                // Parse tensor name and shapes from line
                // Format: "tensor_name: [a, b] vs [b, a] ⚠️ TRANSPOSED"
                if let Some((name, _shapes)) = line.split_once(':') {
                    let name = name.trim().to_string();
                    // Extract shapes (simplified parsing)
                    let mismatch = TensorMismatch {
                        name,
                        shape_a: vec![],
                        shape_b: vec![],
                        mismatch_type: TensorMismatchType::Transposed,
                    };
                    mismatches.push(mismatch);
                    transposed_count += 1;
                }
            }
        }

        Ok(TensorDiffResult {
            total_tensors: 0, // Not available from text output
            mismatched_tensors: mismatches.len(),
            transposed_tensors: transposed_count,
            passed: mismatches.is_empty(),
            mismatches,
        })
    }

    /// Compare inference between two models token-by-token
    ///
    /// Uses `apr rosetta compare-inference` to verify output equivalence.
    ///
    /// # Errors
    ///
    /// Returns an error if the apr command fails to execute.
    pub fn compare_inference(
        &self,
        model_a: &Path,
        model_b: &Path,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<InferenceComparisonResult> {
        let output = Command::new(&self.config.apr_binary)
            .arg("rosetta")
            .arg("compare-inference")
            .arg(model_a)
            .arg(model_b)
            .arg("--prompt")
            .arg(prompt)
            .arg("--max-tokens")
            .arg(max_tokens.to_string())
            .arg("--tolerance")
            .arg(self.config.tolerance.to_string())
            .arg("--json")
            .output()
            .map_err(|e| Error::ExecutionFailed {
                command: "apr rosetta compare-inference".to_string(),
                reason: e.to_string(),
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_inference_output(&stdout, output.status.success())
    }

    /// Parse compare-inference output
    fn parse_inference_output(
        &self,
        output: &str,
        success: bool,
    ) -> Result<InferenceComparisonResult> {
        // Try JSON parsing first
        if let Ok(result) = serde_json::from_str::<InferenceComparisonResult>(output) {
            return Ok(result);
        }

        // Fall back to basic result
        Ok(InferenceComparisonResult {
            total_tokens: 0,
            matching_tokens: 0,
            max_logit_diff: 0.0,
            passed: success,
            token_comparisons: vec![],
        })
    }
}

include!("differential_types.rs");
include!("differential_format_conversion.rs");

#[cfg(test)]
#[path = "differential_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "differential_tests_b.rs"]
mod tests_b;
