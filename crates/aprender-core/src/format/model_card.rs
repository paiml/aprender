//! Model Card for .apr format (spec §11)
//!
//! Provides ML model documentation following best practices from:
//! - Mitchell et al. (2019) "Model Cards for Model Reporting"
//! - Hugging Face Model Card specification
//!
//! # Toyota Way Principles
//!
//! - **Standardized Work**: Every model must be self-describing
//! - **Jidoka**: Build quality in through complete documentation
//! - **Genchi Genbutsu**: Go and see - provenance enables debugging

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model card metadata embedded in .apr files.
///
/// Designed for dual compatibility:
/// - APR sovereign format (full control)
/// - Hugging Face ecosystem (interoperability)
///
/// # Example
///
/// ```rust
/// use aprender::format::model_card::{ModelCard, TrainingDataInfo};
///
/// let card = ModelCard::new("my-model", "1.0.0")
///     .with_author("user@host")
///     .with_description("A test model");
///
/// assert_eq!(card.version, "1.0.0");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCard {
    // === Identity ===
    /// Unique model identifier (e.g., "aprender-shell-markov-3gram-20251127")
    pub model_id: String,

    /// Human-readable model name
    pub name: String,

    /// Semantic version (e.g., "1.2.3")
    pub version: String,

    // === Provenance ===
    /// Model author or organization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,

    /// Creation timestamp (ISO 8601)
    pub created_at: String,

    /// Training framework version (e.g., "aprender 0.10.0")
    pub framework_version: String,

    /// Rust toolchain used (e.g., "1.82.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rust_version: Option<String>,

    // === Description ===
    /// Short description (one line)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// License (SPDX identifier)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    // === Training Details ===
    /// Training dataset description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_data: Option<TrainingDataInfo>,

    /// Hyperparameters used
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub hyperparameters: HashMap<String, serde_json::Value>,

    /// Training/evaluation metrics
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metrics: HashMap<String, serde_json::Value>,

    // === Technical Details ===
    /// Model architecture type (e.g., "`MarkovModel`", "`LinearRegression`")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,

    /// Number of parameters (for complexity estimation)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_count: Option<u64>,

    /// Target hardware platforms
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_hardware: Vec<String>,

    /// Custom metadata (extensible)
    #[serde(default, skip_serializing_if = "HashMap::is_empty", flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ModelCard {
    /// Create a new model card with required fields.
    #[must_use]
    pub fn new(model_id: impl Into<String>, version: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let name = model_id.clone();

        Self {
            model_id,
            name,
            version: version.into(),
            author: None,
            created_at: Self::now_iso8601(),
            framework_version: format!("aprender {}", env!("CARGO_PKG_VERSION")),
            rust_version: option_env!("RUSTC_VERSION").map(String::from),
            description: None,
            license: None,
            training_data: None,
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            architecture: None,
            param_count: None,
            target_hardware: vec!["cpu".to_string()],
            extra: HashMap::new(),
        }
    }

    /// Set the model name (human-readable).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the author.
    #[must_use]
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the license (SPDX identifier).
    #[must_use]
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Set the architecture type.
    #[must_use]
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Set the parameter count.
    #[must_use]
    pub fn with_param_count(mut self, count: u64) -> Self {
        self.param_count = Some(count);
        self
    }

    /// Set training data info.
    #[must_use]
    pub fn with_training_data(mut self, data: TrainingDataInfo) -> Self {
        self.training_data = Some(data);
        self
    }

    /// Add a hyperparameter.
    #[must_use]
    pub fn with_hyperparameter(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.hyperparameters.insert(key.into(), value.into());
        self
    }

    /// Add a metric.
    #[must_use]
    pub fn with_metric(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metrics.insert(key.into(), value.into());
        self
    }

    /// Get current time as ISO 8601 string.
    fn now_iso8601() -> String {
        // Simple implementation without external deps
        // Format: 2025-11-27T12:30:00Z
        let now = std::time::SystemTime::now();
        let duration = now
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = duration.as_secs();

        // Convert to date/time components
        let days = secs / 86400;
        let time_secs = secs % 86400;
        let hours = time_secs / 3600;
        let minutes = (time_secs % 3600) / 60;
        let seconds = time_secs % 60;

        // Calculate year/month/day from days since epoch
        let (year, month, day) = days_to_ymd(days);

        format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
    }

    /// Export to Hugging Face model card format (YAML front matter + Markdown).
    #[must_use]
    pub fn to_huggingface(&self) -> String {
        use std::fmt::Write;

        let mut output = String::from("---\n");

        // License
        if let Some(license) = &self.license {
            let _ = writeln!(output, "license: {}", license.to_lowercase());
        }

        // Pipeline tag
        output.push_str("pipeline_tag: text-generation\n");

        // Tags
        output.push_str("tags:\n");
        if let Some(arch) = &self.architecture {
            let _ = writeln!(output, "  - {}", arch.to_lowercase());
        }
        output.push_str("  - aprender\n");
        output.push_str("  - rust\n");

        // Model index
        output.push_str("model-index:\n");
        let _ = writeln!(output, "  - name: {}", self.model_id);
        if !self.metrics.is_empty() {
            output.push_str("    results:\n");
            output.push_str("      - task:\n");
            output.push_str("          type: text-generation\n");
            output.push_str("        metrics:\n");
            for (key, value) in &self.metrics {
                let _ = writeln!(output, "          - name: {key}");
                output.push_str("            type: custom\n");
                let _ = writeln!(output, "            value: {value}");
            }
        }

        output.push_str("---\n\n");

        // Title
        let _ = writeln!(output, "# {}\n", self.name);

        // Description
        if let Some(desc) = &self.description {
            let _ = writeln!(output, "{desc}\n");
        }

        // Training data
        if let Some(data) = &self.training_data {
            output.push_str("## Training Data\n\n");
            let _ = writeln!(output, "- **Source:** {}", data.name);
            if let Some(samples) = data.samples {
                let _ = writeln!(output, "- **Samples:** {samples}");
            }
            if let Some(hash) = &data.hash {
                let _ = writeln!(output, "- **Hash:** `{hash}`");
            }
            output.push('\n');
        }

        // Framework
        output.push_str("## Framework\n\n");
        let _ = writeln!(output, "- **Version:** {}", self.framework_version);
        if let Some(rust) = &self.rust_version {
            let _ = writeln!(output, "- **Rust:** {rust}");
        }

        output
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for ModelCard {
    fn default() -> Self {
        Self::new("unnamed", "0.0.0")
    }
}

/// Training data information for model card.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingDataInfo {
    /// Dataset name or source path
    pub name: String,

    /// Number of samples/commands
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<u64>,

    /// Content hash for reproducibility (SHA-256)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

impl TrainingDataInfo {
    /// Create new training data info.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            samples: None,
            hash: None,
        }
    }

    /// Set sample count.
    #[must_use]
    pub fn with_samples(mut self, count: u64) -> Self {
        self.samples = Some(count);
        self
    }

    /// Set content hash.
    #[must_use]
    pub fn with_hash(mut self, hash: impl Into<String>) -> Self {
        self.hash = Some(hash.into());
        self
    }
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    // Simplified calculation (not accounting for leap seconds)
    let mut remaining = days as i64;
    let mut year = 1970i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let leap = is_leap_year(year);
    let months = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    for days_in_month in months {
        if remaining < days_in_month {
            break;
        }
        remaining -= days_in_month;
        month += 1;
    }

    let day = remaining as u32 + 1;

    (year as u32, month, day)
}

/// Check if year is a leap year.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

include!("generating.rs");
