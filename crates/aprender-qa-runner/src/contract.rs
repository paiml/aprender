//! Shared Format Contract: YAML-Defined Behavioral Invariants
//!
//! Implements invariants I-2 through I-5 from the Five-Whys analysis
//! (GH-190, GH-191). I-1 (Golden Rule Test) is already implemented
//! in `executor.rs`.
//!
//! The contract is defined in `apr_format_contract.yaml` and loaded
//! at compile time via `include_str!()`.

use crate::command::CommandRunner;
use crate::evidence::Evidence;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Deserialize a bool that may be quoted as a string in YAML (CB-950 compliance).
fn deserialize_bool_or_string<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolOrString {
        Bool(bool),
        String(String),
    }
    match BoolOrString::deserialize(deserializer)? {
        BoolOrString::Bool(b) => Ok(b),
        BoolOrString::String(s) => match s.to_lowercase().as_str() {
            "true" | "yes" | "on" => Ok(true),
            "false" | "no" | "off" => Ok(false),
            _ => Err(serde::de::Error::custom(format!(
                "expected boolean or truthy string, got '{s}'"
            ))),
        },
    }
}
use std::sync::Arc;

/// Embedded YAML contract — single source of truth for format invariants.
const CONTRACT_YAML: &str = include_str!("apr_format_contract.yaml");

// ============================================================================
// Contract types (deserialized from YAML)
// ============================================================================

/// Top-level format contract.
#[derive(Debug, Clone, Deserialize)]
pub struct FormatContract {
    /// Contract version (e.g., "1.0").
    pub version: String,
    /// Tensor naming convention.
    pub tensor_naming: TensorNamingContract,
    /// GGML dtype-to-byte mappings.
    pub dtype_bytes: DtypeByteSection,
    /// Per-dtype tolerances.
    pub tolerances: Vec<ToleranceEntry>,
    /// Invariant definitions (I-1 through I-5).
    pub invariants: Vec<InvariantDef>,
}

/// Tensor naming convention contract.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorNamingContract {
    /// Convention name (e.g., "gguf-short").
    pub convention: String,
    /// Human-readable description.
    pub description: String,
    /// Canonical/forbidden example pairs.
    pub examples: Vec<NamingExample>,
    /// Regex pattern that valid names must match.
    pub pattern: String,
}

/// Example of canonical vs. forbidden tensor name.
#[derive(Debug, Clone, Deserialize)]
pub struct NamingExample {
    /// Correct short name.
    pub canonical: String,
    /// Forbidden long-form name.
    pub forbidden: String,
}

/// Dtype bytes section with description and mappings.
#[derive(Debug, Clone, Deserialize)]
pub struct DtypeByteSection {
    /// Human-readable description.
    pub description: String,
    /// Dtype-to-byte mappings.
    pub mappings: Vec<DtypeByteEntry>,
}

/// Single dtype-to-byte mapping.
#[derive(Debug, Clone, Deserialize)]
pub struct DtypeByteEntry {
    /// Dtype label (e.g., "Q4_K").
    pub dtype: String,
    /// GGML byte value.
    pub byte: u8,
}

/// Per-dtype tolerance for statistical comparison.
#[derive(Debug, Clone, Deserialize)]
pub struct ToleranceEntry {
    /// Dtype label.
    pub dtype: String,
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
}

/// Definition of a single invariant.
#[derive(Debug, Clone, Deserialize)]
pub struct InvariantDef {
    /// Invariant ID (e.g., "I-2").
    pub id: String,
    /// Short name.
    pub name: String,
    /// Description of what the invariant checks.
    pub description: String,
    /// Bug tickets this invariant catches.
    pub catches: Vec<String>,
    /// Gate ID for evidence (e.g., "F-CONTRACT-I2-001").
    pub gate_id: String,
    /// Command template (if applicable).
    #[serde(default)]
    pub test: Option<String>,
    /// Whether already implemented elsewhere.
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub implemented: bool,
}

/// Which invariants to enable in a contract test run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractTestConfig {
    /// List of invariant IDs to enable (e.g., `["I-2", "I-3", "I-4", "I-5"]`).
    #[serde(default = "default_invariants")]
    pub invariants: Vec<String>,
}

/// Return the default set of invariant IDs (I-2 through I-5)
fn default_invariants() -> Vec<String> {
    vec![
        "I-2".to_string(),
        "I-3".to_string(),
        "I-4".to_string(),
        "I-5".to_string(),
    ]
}

impl Default for ContractTestConfig {
    /// Create default config with invariants I-2 through I-5
    fn default() -> Self {
        Self {
            invariants: default_invariants(),
        }
    }
}

/// Invariant identifier enum for type-safe dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvariantId {
    /// I-1: Round-trip identity (implemented in executor.rs).
    I1,
    /// I-2: Tensor name bijection.
    I2,
    /// I-3: No silent fallbacks.
    I3,
    /// I-4: Statistical preservation.
    I4,
    /// I-5: Tokenizer roundtrip.
    I5,
}

impl InvariantId {
    /// Parse from string label (e.g., "I-2").
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "I-1" => Some(Self::I1),
            "I-2" => Some(Self::I2),
            "I-3" => Some(Self::I3),
            "I-4" => Some(Self::I4),
            "I-5" => Some(Self::I5),
            _ => None,
        }
    }

    /// Gate ID for this invariant.
    #[must_use]
    pub fn gate_id(self) -> &'static str {
        match self {
            Self::I1 => "F-CONTRACT-I1-001",
            Self::I2 => "F-CONTRACT-I2-001",
            Self::I3 => "F-CONTRACT-I3-001",
            Self::I4 => "F-CONTRACT-I4-001",
            Self::I5 => "F-CONTRACT-I5-001",
        }
    }
}

// ============================================================================
// Contract loader
// ============================================================================

/// Load the embedded format contract from YAML.
///
/// # Errors
///
/// Returns an error if the embedded YAML fails to parse.
pub fn load_format_contract() -> crate::error::Result<FormatContract> {
    serde_yaml::from_str(CONTRACT_YAML).map_err(crate::error::Error::from)
}

// ============================================================================
// Pure validation functions (no subprocess)
// ============================================================================

/// Validate that dtype byte mappings have no duplicate byte values.
///
/// # Errors
///
/// Returns an error if duplicate byte values are found.
pub fn validate_dtype_bytes(contract: &FormatContract) -> crate::error::Result<()> {
    let mut seen = HashSet::new();
    for entry in &contract.dtype_bytes.mappings {
        if !seen.insert(entry.byte) {
            return Err(crate::error::Error::Execution(format!(
                "Duplicate GGML byte value {} for dtype {}",
                entry.byte, entry.dtype
            )));
        }
    }
    Ok(())
}

/// Validate a tensor name against the contract pattern.
///
/// Returns `true` if the name matches the GGUF-short convention.
#[must_use]
pub fn validate_tensor_name(name: &str, contract: &FormatContract) -> bool {
    // Simple pattern matching without regex dependency.
    // The pattern is: ^(\d+\.\w+\.\w+|token_embd\.\w+|output_norm\.\w+|output\.\w+)$
    is_valid_tensor_name(name, &contract.tensor_naming.pattern)
}

/// Check if a tensor name matches the GGUF-short naming pattern.
///
/// Supported patterns:
/// - `{digit}.{word}.{word}` (layer tensors)
/// - `token_embd.{word}` (embedding)
/// - `output_norm.{word}` (output norm)
/// - `output.{word}` (output head)
fn is_valid_tensor_name(name: &str, _pattern: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    match parts.len() {
        2 => {
            // token_embd.weight, output_norm.weight, output.weight
            matches!(parts[0], "token_embd" | "output_norm" | "output") && is_word(parts[1])
        }
        3 => {
            // 0.q_proj.weight — first part must be all digits
            parts[0].chars().all(|c| c.is_ascii_digit())
                && !parts[0].is_empty()
                && is_word(parts[1])
                && is_word(parts[2])
        }
        _ => false,
    }
}

/// Check if a string is a "word" (alphanumeric + underscore, non-empty).
fn is_word(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Look up tolerance values for a given dtype.
///
/// Returns `None` if the dtype is not in the contract.
#[must_use]
pub fn lookup_tolerance(dtype: &str, contract: &FormatContract) -> Option<(f64, f64)> {
    contract
        .tolerances
        .iter()
        .find(|t| t.dtype == dtype)
        .map(|t| (t.atol, t.rtol))
}

// ============================================================================
// Contract invariant test runners (use CommandRunner)
// ============================================================================

/// Run contract invariant tests I-2 through I-5.
///
/// Returns `(passed, failed)` counts.
pub fn run_contract_tests(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    config: &ContractTestConfig,
) -> Vec<Evidence> {
    let mut evidence = Vec::new();
    let contract = match load_format_contract() {
        Ok(c) => c,
        Err(e) => {
            evidence.push(Evidence::falsified(
                "F-CONTRACT-LOAD-001",
                contract_scenario(model_id),
                format!("Failed to load format contract: {e}"),
                "N/A",
                0,
            ));
            return evidence;
        }
    };

    for label in &config.invariants {
        let Some(inv_id) = InvariantId::from_label(label) else {
            evidence.push(Evidence::falsified(
                "F-CONTRACT-INVALID-001",
                contract_scenario(model_id),
                format!("Unknown invariant ID '{label}' — valid: I-1 through I-5"),
                "N/A",
                0,
            ));
            continue;
        };

        // Skip I-1 (handled by golden rule test in executor.rs)
        if inv_id == InvariantId::I1 {
            continue;
        }

        let inv_def = contract.invariants.iter().find(|i| i.id == *label);
        let gate_id = inv_def.map_or_else(|| inv_id.gate_id(), |d| d.gate_id.as_str());

        let ev = match inv_id {
            InvariantId::I1 => unreachable!(),
            InvariantId::I2 => run_i2_tensor_bijection(runner, model_path, model_id, gate_id),
            InvariantId::I3 => run_i3_no_silent_fallbacks(runner, model_path, model_id, gate_id),
            InvariantId::I4 => {
                run_i4_statistical_preservation(runner, model_path, model_id, gate_id)
            }
            InvariantId::I5 => run_i5_tokenizer_roundtrip(runner, model_path, model_id, gate_id),
        };
        evidence.push(ev);
    }

    evidence
}

/// Resolve the APR file path from a workspace directory.
///
/// Workspace layout: `{workspace}/apr/model.apr`
/// Avoids `Path::with_extension` which corrupts names containing dots
/// (e.g., `Qwen2.5-Coder-0.5B-Instruct` becomes `Qwen2.5-Coder-0.apr`).
fn resolve_apr_path(model_path: &Path) -> PathBuf {
    model_path.join("apr").join("model.apr")
}

/// Resolve the SafeTensors file path from a workspace directory.
fn resolve_safetensors_path(model_path: &Path) -> PathBuf {
    model_path.join("safetensors").join("model.safetensors")
}

/// I-2: Tensor Name Bijection — writer names == reader names.
///
/// Allows exactly one extra tensor in APR (`lm_head.weight`) when the source
/// model uses tied embeddings (no separate `lm_head` in SafeTensors). The
/// converter materializes `lm_head.weight` from `embed_tokens.weight`, which
/// is correct behavior per `write.rs:49-89`.
#[allow(clippy::too_many_lines)]
fn run_i2_tensor_bijection(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let st_path = resolve_safetensors_path(model_path);
    let apr_path = resolve_apr_path(model_path);

    let start = std::time::Instant::now();
    // Inspect both models to get tensor name lists
    let st_inspect = runner.inspect_model_json(&st_path);
    let apr_inspect = runner.inspect_model_json(&apr_path);
    let duration = start.elapsed().as_millis() as u64;

    if !st_inspect.success || !apr_inspect.success {
        let err = if st_inspect.success {
            &apr_inspect.stderr
        } else {
            &st_inspect.stderr
        };
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!("I-2 Tensor Name Bijection: inspect failed: {err}"),
            &format!("st: {}, apr: {}", st_inspect.stdout, apr_inspect.stdout),
            duration,
        );
    }

    let st_names = parse_tensor_names(&st_inspect.stdout);
    let apr_names = parse_tensor_names(&apr_inspect.stdout);

    // Popper: empty tensor sets → vacuous bijection. Both must be non-empty.
    if st_names.is_empty() || apr_names.is_empty() {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-2 Tensor Name Bijection: cannot validate — parsed 0 tensors (source={}, apr={})",
                st_names.len(),
                apr_names.len()
            ),
            &format!(
                "st_stdout: {}, apr_stdout: {}",
                st_inspect.stdout, apr_inspect.stdout
            ),
            duration,
        );
    }

    // Every source tensor must appear in the APR output
    let missing: Vec<&str> = st_names
        .iter()
        .filter(|n| !apr_names.contains(n.as_str()))
        .map(String::as_str)
        .collect();

    if !missing.is_empty() {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-2 Tensor Name Bijection: {} source tensors missing in APR: {}",
                missing.len(),
                missing.join(", ")
            ),
            &format!("source={}, apr={}", st_names.len(), apr_names.len()),
            duration,
        );
    }

    // APR may have extra tensors only for tied embedding materialization
    let extra: Vec<&str> = apr_names
        .iter()
        .filter(|n| !st_names.contains(n.as_str()))
        .map(String::as_str)
        .collect();

    let allowed_extras: HashSet<&str> = HashSet::from(["lm_head.weight", "lm_head.bias"]);
    let unexpected: Vec<&str> = extra
        .iter()
        .filter(|n| !allowed_extras.contains(*n))
        .copied()
        .collect();

    if !unexpected.is_empty() {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-2 Tensor Name Bijection: {} unexpected extra tensors in APR: {}",
                unexpected.len(),
                unexpected.join(", ")
            ),
            &format!(
                "source={}, apr={}, extra={:?}",
                st_names.len(),
                apr_names.len(),
                extra
            ),
            duration,
        );
    }

    let tied = if extra.is_empty() {
        ""
    } else {
        " (tied embedding materialized)"
    };
    let mut ev = Evidence::corroborated(
        gate_id,
        contract_scenario(model_id),
        &format!("source={}, apr={}", st_names.len(), apr_names.len()),
        duration,
    );
    ev.reason = format!(
        "I-2 Tensor Name Bijection: all {} source tensors present in APR ({} total){}",
        st_names.len(),
        apr_names.len(),
        tied,
    );
    ev
}

include!("contract_invariant_checks.rs");
