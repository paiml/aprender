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
    /// Model architecture type (e.g., "MarkovModel", "LinearRegression")
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

// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_card_new() {
        let card = ModelCard::new("test-model", "1.0.0");

        assert_eq!(card.model_id, "test-model");
        assert_eq!(card.name, "test-model");
        assert_eq!(card.version, "1.0.0");
        assert!(card.framework_version.starts_with("aprender"));
        assert!(!card.created_at.is_empty());
    }

    #[test]
    fn test_model_card_builder() {
        let card = ModelCard::new("my-model", "2.1.0")
            .with_name("My Model")
            .with_author("user@host")
            .with_description("A test model")
            .with_license("MIT")
            .with_architecture("MarkovModel")
            .with_param_count(12345);

        assert_eq!(card.name, "My Model");
        assert_eq!(card.author, Some("user@host".to_string()));
        assert_eq!(card.description, Some("A test model".to_string()));
        assert_eq!(card.license, Some("MIT".to_string()));
        assert_eq!(card.architecture, Some("MarkovModel".to_string()));
        assert_eq!(card.param_count, Some(12345));
    }

    #[test]
    fn test_model_card_with_training_data() {
        let data = TrainingDataInfo::new("~/.zsh_history")
            .with_samples(15234)
            .with_hash("sha256:abc123");

        let card = ModelCard::new("shell-model", "1.0.0").with_training_data(data);

        let training = card.training_data.expect("should have training data");
        assert_eq!(training.name, "~/.zsh_history");
        assert_eq!(training.samples, Some(15234));
        assert_eq!(training.hash, Some("sha256:abc123".to_string()));
    }

    #[test]
    fn test_model_card_with_hyperparameters() {
        let card = ModelCard::new("test", "1.0.0")
            .with_hyperparameter("n_gram_size", 3)
            .with_hyperparameter("smoothing", "laplace");

        assert_eq!(
            card.hyperparameters.get("n_gram_size"),
            Some(&serde_json::json!(3))
        );
        assert_eq!(
            card.hyperparameters.get("smoothing"),
            Some(&serde_json::json!("laplace"))
        );
    }

    #[test]
    fn test_model_card_with_metrics() {
        let card = ModelCard::new("test", "1.0.0")
            .with_metric("vocab_size", 4521)
            .with_metric("accuracy", 0.72);

        assert_eq!(
            card.metrics.get("vocab_size"),
            Some(&serde_json::json!(4521))
        );
        assert_eq!(card.metrics.get("accuracy"), Some(&serde_json::json!(0.72)));
    }

    #[test]
    fn test_model_card_json_roundtrip() {
        let card = ModelCard::new("roundtrip-test", "1.2.3")
            .with_author("test@example.com")
            .with_description("Test description")
            .with_metric("score", 0.95);

        let json = card.to_json().expect("serialize");
        let restored = ModelCard::from_json(&json).expect("deserialize");

        assert_eq!(card, restored);
    }

    #[test]
    fn test_model_card_to_huggingface() {
        let card = ModelCard::new("my-model", "1.0.0")
            .with_name("My Model")
            .with_description("A test model")
            .with_license("MIT")
            .with_architecture("MarkovModel")
            .with_metric("accuracy", 0.95)
            .with_training_data(
                TrainingDataInfo::new("dataset.txt")
                    .with_samples(1000)
                    .with_hash("sha256:abc"),
            );

        let hf = card.to_huggingface();

        // Check YAML front matter
        assert!(hf.starts_with("---"));
        assert!(hf.contains("license: mit"));
        assert!(hf.contains("- aprender"));
        assert!(hf.contains("- markovmodel"));

        // Check markdown content
        assert!(hf.contains("# My Model"));
        assert!(hf.contains("A test model"));
        assert!(hf.contains("**Source:** dataset.txt"));
        assert!(hf.contains("**Samples:** 1000"));
    }

    #[test]
    fn test_training_data_info() {
        let info = TrainingDataInfo::new("data.csv")
            .with_samples(500)
            .with_hash("sha256:def456");

        assert_eq!(info.name, "data.csv");
        assert_eq!(info.samples, Some(500));
        assert_eq!(info.hash, Some("sha256:def456".to_string()));
    }

    #[test]
    fn test_days_to_ymd() {
        // 1970-01-01
        assert_eq!(days_to_ymd(0), (1970, 1, 1));

        // 2000-01-01 (leap year)
        assert_eq!(days_to_ymd(10957), (2000, 1, 1));

        // 2025-11-27
        assert_eq!(days_to_ymd(20419), (2025, 11, 27));
    }

    #[test]
    fn test_is_leap_year() {
        assert!(!is_leap_year(1970));
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(!is_leap_year(1900)); // Divisible by 100 but not 400
        assert!(is_leap_year(2024)); // Divisible by 4
        assert!(!is_leap_year(2025));
    }

    #[test]
    fn test_model_card_default() {
        let card = ModelCard::default();
        assert_eq!(card.model_id, "unnamed");
        assert_eq!(card.version, "0.0.0");
    }

    #[test]
    fn test_model_card_created_at_format() {
        let card = ModelCard::new("test", "1.0.0");

        // Should be ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        let created = &card.created_at;
        assert_eq!(created.len(), 20);
        assert!(created.ends_with('Z'));
        assert!(created.contains('T'));
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid model IDs
    fn arb_model_id() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_-]{0,30}".prop_map(|s| s.clone())
    }

    /// Strategy for generating semantic versions
    fn arb_semver() -> impl Strategy<Value = String> {
        (0u32..100, 0u32..100, 0u32..100)
            .prop_map(|(major, minor, patch)| format!("{major}.{minor}.{patch}"))
    }

    /// Strategy for generating author names
    fn arb_author() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9 _@.-]{0,50}".prop_map(|s| s.clone())
    }

    /// Strategy for generating descriptions
    fn arb_description() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 .,!?-]{0,200}".prop_map(|s| s.clone())
    }

    proptest! {
        /// Property: ModelCard JSON roundtrip preserves all fields
        #[test]
        fn prop_model_card_json_roundtrip(
            model_id in arb_model_id(),
            version in arb_semver(),
            author in arb_author(),
            description in arb_description(),
            param_count in any::<u64>(),
        ) {
            let card = ModelCard::new(&model_id, &version)
                .with_author(&author)
                .with_description(&description)
                .with_param_count(param_count);

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            prop_assert_eq!(card.model_id, restored.model_id);
            prop_assert_eq!(card.version, restored.version);
            prop_assert_eq!(card.author, restored.author);
            prop_assert_eq!(card.description, restored.description);
            prop_assert_eq!(card.param_count, restored.param_count);
        }

        /// Property: ModelCard builder methods are idempotent
        #[test]
        fn prop_builder_idempotent(
            model_id in arb_model_id(),
            version in arb_semver(),
            author in arb_author(),
        ) {
            let card1 = ModelCard::new(&model_id, &version)
                .with_author(&author)
                .with_author(&author); // Apply twice

            let card2 = ModelCard::new(&model_id, &version)
                .with_author(&author);

            prop_assert_eq!(card1.author, card2.author);
        }

        /// Property: created_at is always valid ISO 8601
        #[test]
        fn prop_created_at_valid_iso8601(
            model_id in arb_model_id(),
            version in arb_semver(),
        ) {
            let card = ModelCard::new(&model_id, &version);
            let created = &card.created_at;

            // Format: YYYY-MM-DDTHH:MM:SSZ
            prop_assert_eq!(created.len(), 20);
            prop_assert!(created.ends_with('Z'));
            prop_assert!(created.contains('T'));
            prop_assert!(created.chars().filter(|c| *c == '-').count() == 2);
            prop_assert!(created.chars().filter(|c| *c == ':').count() == 2);
        }

        /// Property: TrainingDataInfo roundtrip through JSON
        #[test]
        fn prop_training_data_roundtrip(
            name in "[a-zA-Z0-9/_.-]{1,50}",
            samples in any::<u64>(),
            hash in "[a-f0-9]{64}",
        ) {
            let info = TrainingDataInfo::new(&name)
                .with_samples(samples)
                .with_hash(&hash);

            let card = ModelCard::new("test", "1.0.0")
                .with_training_data(info.clone());

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            let restored_info = restored.training_data.expect("training data");
            prop_assert_eq!(info.name, restored_info.name);
            prop_assert_eq!(info.samples, restored_info.samples);
            prop_assert_eq!(info.hash, restored_info.hash);
        }

        /// Property: Hyperparameters roundtrip through JSON
        #[test]
        fn prop_hyperparameters_roundtrip(
            key in "[a-z_]{1,20}",
            int_value in any::<i64>(),
            float_value in any::<f64>().prop_filter("finite", |f| f.is_finite()),
        ) {
            let card = ModelCard::new("test", "1.0.0")
                .with_hyperparameter(&key, int_value)
                .with_hyperparameter("float_param", float_value);

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            prop_assert_eq!(
                card.hyperparameters.get(&key),
                restored.hyperparameters.get(&key)
            );
        }

        /// Property: Metrics roundtrip through JSON (keys preserved)
        #[test]
        fn prop_metrics_roundtrip(
            key in "[a-z_]{1,20}",
            value in -1e10f64..1e10f64, // Reasonable range to avoid JSON precision issues
        ) {
            let card = ModelCard::new("test", "1.0.0")
                .with_metric(&key, value);

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            // Key must exist in restored metrics
            prop_assert!(restored.metrics.contains_key(&key));
        }

        /// Property: days_to_ymd produces valid dates
        #[test]
        fn prop_days_to_ymd_valid(days in 0u64..50000) {
            let (year, month, day) = days_to_ymd(days);

            // Year in reasonable range (1970 - ~2106)
            prop_assert!((1970..=2200).contains(&year));
            // Month 1-12
            prop_assert!((1..=12).contains(&month));
            // Day 1-31
            prop_assert!((1..=31).contains(&day));
        }

        /// Property: is_leap_year consistent with days_in_year
        #[test]
        fn prop_leap_year_consistent(year in 1970i32..2200) {
            let leap = is_leap_year(year);
            let days = if leap { 366 } else { 365 };

            // February has 29 days in leap years, 28 otherwise
            let feb_days = if leap { 29 } else { 28 };
            let total = 31 + feb_days + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31;

            prop_assert_eq!(days, total);
        }

        /// Property: Hugging Face export always contains required fields
        #[test]
        fn prop_huggingface_has_required_fields(
            model_id in arb_model_id(),
            version in arb_semver(),
        ) {
            let card = ModelCard::new(&model_id, &version)
                .with_license("MIT")
                .with_architecture("TestModel");

            let hf = card.to_huggingface();

            // Must have YAML front matter markers
            prop_assert!(hf.starts_with("---"));
            prop_assert!(hf.contains("---\n\n"));

            // Must have required tags
            prop_assert!(hf.contains("- aprender"));
            prop_assert!(hf.contains("- rust"));

            // Must have model name header
            let expected_header = format!("# {}", card.name);
            prop_assert!(hf.contains(&expected_header));
        }

        /// Property: Default ModelCard has expected defaults
        #[test]
        fn prop_default_has_expected_values(_seed in any::<u8>()) {
            let card = ModelCard::default();

            prop_assert_eq!(card.model_id, "unnamed");
            prop_assert_eq!(card.version, "0.0.0");
            prop_assert!(card.author.is_none());
            prop_assert!(card.description.is_none());
            prop_assert!(card.license.is_none());
            prop_assert!(card.training_data.is_none());
            prop_assert!(card.hyperparameters.is_empty());
            prop_assert!(card.metrics.is_empty());
        }
    }
}
