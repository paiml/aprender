//! Playbook definition and parsing
//!
//! Playbooks define test scenarios in YAML format.

use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use crate::error::{Error, Result};

/// Deserialize a bool that may be quoted as a string in YAML (CB-950 compliance)
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

// ── Playbook Naming Convention (PMAT-266) ────────────────────────────────────
//
// Playbook filenames MUST follow the pattern:
//   {family}-{size}[-{tier}].playbook.yaml
//
// Examples:
//   qwen2.5-coder-0.5b-mvp.playbook.yaml   → family="qwen2.5-coder", size="0.5b", tier="mvp"
//   llama3.2-1b.playbook.yaml              → family="llama3.2", size="1b", tier=None
//   deepseek-coder-v2-16b-full.playbook.yaml → family="deepseek-coder-v2", size="16b", tier="full"
//
// Size patterns: {digits}[.{digits}]b (e.g., 0.5b, 1b, 7b, 70b)
// Tier patterns: dim-smoke, mvp, smoke, quick, ci, full, nightly, release

/// Regex pattern for playbook naming convention
/// Matches: {family}-{size}[-{tier}].playbook.yaml
/// - family: one or more segments separated by `-` (letters, digits, dots)
/// - size: digits optionally with decimal, followed by `b` (e.g., 0.5b, 1b, 7b)
/// - tier (optional): dim-smoke, mvp, smoke, quick, ci, full, nightly, release
static PLAYBOOK_NAME_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    // Static regex pattern verified at compile time — expect is safe here
    #[allow(clippy::expect_used)]
    Regex::new(
        r"^(?P<family>(?:[a-z0-9]+\.?)+(?:-[a-z0-9]+\.?)*)-(?P<size>\d+(?:\.\d+)?b)(?:-(?P<tier>dim-smoke|mvp|smoke|quick|ci|full|nightly|release))?\.playbook\.yaml$"
    ).expect("PLAYBOOK_NAME_REGEX is a valid, compile-time-verified regex pattern")
});

/// Valid tier values for playbook naming
pub const VALID_TIERS: &[&str] = &[
    "dim-smoke",
    "mvp",
    "smoke",
    "quick",
    "ci",
    "full",
    "nightly",
    "release",
];

/// Parsed components from a playbook filename
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaybookNameParts {
    /// Model family (e.g., "qwen2.5-coder", "llama3.2")
    pub family: String,
    /// Model size (e.g., "0.5b", "7b", "70b")
    pub size: String,
    /// Optional tier (e.g., "mvp", "full", "nightly")
    pub tier: Option<String>,
}

/// Filename reconstruction from parsed playbook name components
impl PlaybookNameParts {
    /// Reconstruct the canonical filename from parts
    #[must_use]
    #[allow(clippy::option_if_let_else)]
    pub fn to_filename(&self) -> String {
        match &self.tier {
            Some(tier) => {
                format!("{}-{}-{}.playbook.yaml", self.family, self.size, tier)
            }
            None => format!("{}-{}.playbook.yaml", self.family, self.size),
        }
    }
}

/// Validate a playbook filename against the naming convention (PMAT-266)
///
/// # Arguments
/// * `filename` - The filename to validate (not the full path)
///
/// # Returns
/// * `Ok(PlaybookNameParts)` if valid
/// * `Err` with descriptive message if invalid
///
/// # Errors
///
/// Returns an error if the filename doesn't match the naming convention.
pub fn validate_playbook_name(filename: &str) -> Result<PlaybookNameParts> {
    let captures = PLAYBOOK_NAME_REGEX.captures(filename).ok_or_else(|| {
        Error::Validation(format!(
            "Playbook filename '{filename}' does not match naming convention: \
             {{family}}-{{size}}[-{{tier}}].playbook.yaml\n\
             Examples: qwen2.5-coder-0.5b-mvp.playbook.yaml, llama3.2-7b.playbook.yaml"
        ))
    })?;

    Ok(PlaybookNameParts {
        family: captures["family"].to_string(),
        size: captures["size"].to_string(),
        tier: captures.name("tier").map(|m| m.as_str().to_string()),
    })
}

/// Extract and validate playbook name from a full path
///
/// # Errors
///
/// Returns an error if the path has no filename or doesn't match the naming convention.
pub fn validate_playbook_path(path: impl AsRef<Path>) -> Result<PlaybookNameParts> {
    let path = path.as_ref();
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Validation(format!("Invalid playbook path: {}", path.display())))?;

    validate_playbook_name(filename)
}

/// Model size category for resource management (§3.4 Resource-Aware Scheduling)
///
/// These categories enforce worker limits to prevent OOM conditions when testing
/// large models. The executor MUST respect these limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SizeCategory {
    /// < 1B params: 4 workers, can run in parallel with others
    #[default]
    Tiny,
    /// 1-2B params: 4 workers, can run in parallel with tiny models
    Small,
    /// 2-4B params: 2 workers, should run alone or with tiny/small
    Medium,
    /// 4-10B params: 1 worker, must run alone
    Large,
    /// 10-30B params: 1 worker, must run alone, may need swap
    Xlarge,
    /// > 30B params: 1 worker, requires careful resource management
    Huge,
}

/// Resource limits and concurrency rules based on model size
impl SizeCategory {
    /// Maximum workers allowed for this model size
    #[must_use]
    pub const fn max_workers(&self) -> usize {
        match self {
            Self::Tiny | Self::Small => 4,
            Self::Medium => 2,
            Self::Large | Self::Xlarge | Self::Huge => 1,
        }
    }

    /// Estimated memory requirement in GB (rough heuristic)
    #[must_use]
    pub const fn estimated_memory_gb(&self) -> usize {
        match self {
            Self::Tiny => 2,
            Self::Small => 4,
            Self::Medium => 8,
            Self::Large => 16,
            Self::Xlarge => 32,
            Self::Huge => 64,
        }
    }

    /// Can run concurrently with other playbooks
    #[must_use]
    pub const fn can_run_concurrent(&self) -> bool {
        matches!(self, Self::Tiny | Self::Small)
    }

    /// Parse a size category from a string.
    ///
    /// Accepts lowercase category names: tiny, small, medium, large, xlarge, huge.
    ///
    /// # Errors
    ///
    /// Returns an error if the string doesn't match a valid category.
    pub fn from_str_lowercase(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "xlarge" => Ok(Self::Xlarge),
            "huge" => Ok(Self::Huge),
            _ => Err(Error::Validation(format!(
                "Invalid size category: {s}. Valid: tiny, small, medium, large, xlarge, huge"
            ))),
        }
    }
}

/// A complete playbook for model qualification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Playbook {
    /// Playbook name
    pub name: String,
    /// Version
    pub version: String,
    /// Model configuration
    pub model: ModelConfig,
    /// Test matrix configuration
    pub test_matrix: TestMatrix,
    /// Property test definitions
    #[serde(default)]
    pub property_tests: Vec<PropertyTest>,
    /// Falsification gates
    #[serde(default)]
    pub falsification_gates: Vec<FalsificationGate>,
    /// State machine definition (optional)
    #[serde(default)]
    pub state_machine: Option<StateMachine>,
    /// Differential tests (GH-188, PMAT-114)
    #[serde(default)]
    pub differential_tests: Option<DifferentialTestConfig>,
    /// Profile CI assertions (PMAT-192)
    #[serde(default)]
    pub profile_ci: Option<ProfileCiConfig>,
    /// Trace payload testing (APR-TRACE-001)
    #[serde(default)]
    pub trace_payload: Option<TracePayloadConfig>,
    /// Contract invariant tests (GH-190/191 Five-Whys)
    #[serde(default)]
    pub contract_tests: Option<crate::contract::ContractTestConfig>,
    /// Ollama parity tests (GH-6/AC-2)
    #[serde(default)]
    pub ollama_parity: Option<OllamaParityConfig>,
    /// Transformation tests (quantize, import, prune, distill)
    #[serde(default)]
    pub transformations: Option<TransformationConfig>,
}

/// Playbook loading, parsing, scenario generation, and resource management
impl Playbook {
    /// Load a playbook from a YAML file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Parse a playbook from YAML string.
    ///
    /// Validates required fields after parsing:
    /// - `model.hf_repo` must be non-empty
    /// - `test_matrix.modalities` must be non-empty
    /// - `test_matrix.backends` must be non-empty
    /// - `model.formats` must be non-empty
    /// - `test_matrix.scenario_count` must be > 0
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is invalid or required fields are missing.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let playbook: Self = serde_yaml::from_str(yaml).map_err(Error::from)?;
        playbook.validate()?;
        Ok(playbook)
    }

    /// Post-parse validation (Jidoka: reject invalid playbooks early).
    fn validate(&self) -> Result<()> {
        if self.model.hf_repo.trim().is_empty() {
            return Err(Error::Validation(
                "model.hf_repo must not be empty".to_string(),
            ));
        }
        if self.test_matrix.modalities.is_empty() {
            return Err(Error::Validation(
                "test_matrix.modalities must not be empty".to_string(),
            ));
        }
        if self.test_matrix.backends.is_empty() {
            return Err(Error::Validation(
                "test_matrix.backends must not be empty".to_string(),
            ));
        }
        if self.model.formats.is_empty() {
            return Err(Error::Validation(
                "model.formats must not be empty".to_string(),
            ));
        }
        if self.test_matrix.scenario_count == 0 {
            return Err(Error::Validation(
                "test_matrix.scenario_count must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert to YAML string
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(Error::from)
    }

    /// Generate all scenarios from this playbook
    #[must_use]
    pub fn generate_scenarios(&self) -> Vec<QaScenario> {
        let mut scenarios = Vec::new();
        let mut seed: u64 = 0;

        let model_id = ModelId::new(&self.model.hf_org(), &self.model.hf_name());

        // Use custom prompts from test_matrix if provided, otherwise fall back
        let default_prompt = "What is 2+2?".to_string();
        let prompts: &[String] = self
            .test_matrix
            .prompts
            .as_deref()
            .unwrap_or_else(|| std::slice::from_ref(&default_prompt));

        for modality in &self.test_matrix.modalities {
            for backend in &self.test_matrix.backends {
                for format in &self.model.formats {
                    for i in 0..self.test_matrix.scenario_count {
                        let prompt = prompts[i % prompts.len()].clone();
                        scenarios.push(QaScenario::new(
                            model_id.clone(),
                            *modality,
                            *backend,
                            *format,
                            prompt,
                            seed,
                        ));
                        seed = seed.wrapping_add(1);
                    }
                }
            }
        }

        scenarios
    }

    /// Get total expected test count
    #[must_use]
    pub fn total_tests(&self) -> usize {
        self.test_matrix.modalities.len()
            * self.test_matrix.backends.len()
            * self.model.formats.len()
            * self.test_matrix.scenario_count
    }

    /// Get the model ID for this playbook
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        ModelId::new(&self.model.hf_org(), &self.model.hf_name())
    }

    /// Get the effective maximum workers based on model size (§3.4)
    ///
    /// This ENFORCES resource limits - the executor MUST use this value
    /// and cannot exceed it. Large models get fewer workers to prevent OOM.
    #[must_use]
    pub fn effective_max_workers(&self, requested: usize) -> usize {
        let size_limit = self.model.size_category.max_workers();
        requested.min(size_limit)
    }

    /// Get the model's size category
    #[must_use]
    pub fn size_category(&self) -> SizeCategory {
        self.model.size_category
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// HuggingFace repository
    pub hf_repo: String,
    /// Optional local path
    pub local_path: Option<String>,
    /// Supported formats
    #[serde(default = "default_formats")]
    pub formats: Vec<Format>,
    /// Quantizations to test
    #[serde(default = "default_quantizations")]
    pub quantizations: Vec<String>,
    /// Model size category for resource-aware scheduling (§3.4)
    /// Defaults to `small` which allows 4 workers.
    /// IMPORTANT: Large models (7B+) MUST set this to `large` or higher
    /// to prevent OOM conditions during parallel testing.
    #[serde(default)]
    pub size_category: SizeCategory,

    // ── PMAT-269: Expected architectural parameters from family YAML ────────
    /// Expected hidden dimension (from family YAML size_variants)
    #[serde(default)]
    pub expected_hidden_dim: Option<u32>,
    /// Expected number of layers (from family YAML size_variants)
    #[serde(default)]
    pub expected_num_layers: Option<u32>,
    /// Expected number of attention heads (from family YAML size_variants)
    #[serde(default)]
    pub expected_num_heads: Option<u32>,
    /// Expected number of KV heads for GQA (from family YAML size_variants)
    #[serde(default)]
    pub expected_num_kv_heads: Option<u32>,
    /// Expected vocabulary size (from family YAML size_variants)
    #[serde(default)]
    pub expected_vocab_size: Option<u32>,
    /// Expected intermediate/FFN dimension (from family YAML size_variants)
    #[serde(default)]
    pub expected_intermediate_dim: Option<u32>,
    /// Model family identifier for contract lookup
    #[serde(default)]
    pub family: Option<String>,
    /// Size variant identifier (e.g., "0.5b", "7b")
    #[serde(default)]
    pub size_variant: Option<String>,
}

include!("playbook_config_types.rs");
include!("playbook_fingerprint_config.rs");
include!("playbook_transformation_types.rs");
