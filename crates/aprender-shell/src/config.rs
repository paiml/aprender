//! Configuration and resource limits for aprender-shell
//!
//! Follows Toyota Way principle *Heijunka* (Level Loading):
//! Balance workloads to prevent resource exhaustion.

use std::time::Duration;

/// Configuration with safety limits.
///
/// These limits protect against resource exhaustion from malformed input
/// or corrupted data.
///
/// # Example
/// ```
/// use aprender_shell::config::ShellConfig;
///
/// let config = ShellConfig::default();
/// assert_eq!(config.suggest_timeout_ms, 100);
/// assert_eq!(config.max_suggestions, 10);
///
/// let custom = ShellConfig::default()
///     .with_suggest_timeout_ms(50)
///     .with_max_suggestions(5);
/// assert_eq!(custom.suggest_timeout_ms, 50);
/// assert_eq!(custom.max_suggestions, 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ShellConfig {
    /// Maximum time for suggestion generation (ms)
    pub suggest_timeout_ms: u64,

    /// Maximum model file size (bytes)
    pub max_model_size: usize,

    /// Maximum history file size (bytes)
    pub max_history_size: usize,

    /// Maximum number of suggestions to return
    pub max_suggestions: usize,

    /// Maximum prefix length to process
    pub max_prefix_length: usize,

    /// Minimum prefix length to process
    pub min_prefix_length: usize,

    /// Minimum quality score for suggestions (0.0 to 1.0)
    pub min_quality_score: f32,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            suggest_timeout_ms: 100,
            max_model_size: 100 * 1024 * 1024,   // 100 MB
            max_history_size: 500 * 1024 * 1024, // 500 MB
            max_suggestions: 10,
            max_prefix_length: 500,
            min_prefix_length: 2,
            min_quality_score: 0.3,
        }
    }
}

impl ShellConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the suggestion timeout in milliseconds.
    #[must_use]
    pub fn with_suggest_timeout_ms(mut self, timeout: u64) -> Self {
        self.suggest_timeout_ms = timeout;
        self
    }

    /// Set the maximum model size in bytes.
    #[must_use]
    pub fn with_max_model_size(mut self, size: usize) -> Self {
        self.max_model_size = size;
        self
    }

    /// Set the maximum history file size in bytes.
    #[must_use]
    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Set the maximum number of suggestions.
    #[must_use]
    pub fn with_max_suggestions(mut self, count: usize) -> Self {
        self.max_suggestions = count;
        self
    }

    /// Set the maximum prefix length.
    #[must_use]
    pub fn with_max_prefix_length(mut self, length: usize) -> Self {
        self.max_prefix_length = length;
        self
    }

    /// Set the minimum prefix length.
    #[must_use]
    pub fn with_min_prefix_length(mut self, length: usize) -> Self {
        self.min_prefix_length = length;
        self
    }

    /// Set the minimum quality score (0.0 to 1.0).
    #[must_use]
    pub fn with_min_quality_score(mut self, score: f32) -> Self {
        self.min_quality_score = score.clamp(0.0, 1.0);
        self
    }

    /// Get the suggestion timeout as a Duration.
    #[must_use]
    pub fn suggest_timeout(&self) -> Duration {
        Duration::from_millis(self.suggest_timeout_ms)
    }

    /// Check if a model file size is within limits.
    #[must_use]
    pub fn is_model_size_valid(&self, size: usize) -> bool {
        size <= self.max_model_size
    }

    /// Check if a history file size is within limits.
    #[must_use]
    pub fn is_history_size_valid(&self, size: usize) -> bool {
        size <= self.max_history_size
    }

    /// Check if a prefix length is within limits.
    #[must_use]
    pub fn is_prefix_valid(&self, prefix: &str) -> bool {
        let len = prefix.len();
        len >= self.min_prefix_length && len <= self.max_prefix_length
    }

    /// Truncate prefix to max length if needed.
    #[must_use]
    pub fn truncate_prefix<'a>(&self, prefix: &'a str) -> &'a str {
        if prefix.len() > self.max_prefix_length {
            // Find the last valid UTF-8 boundary before max_prefix_length
            let mut end = self.max_prefix_length;
            while end > 0 && !prefix.is_char_boundary(end) {
                end -= 1;
            }
            &prefix[..end]
        } else {
            prefix
        }
    }
}

/// Preset configurations for different use cases.
impl ShellConfig {
    /// Fast configuration for interactive use.
    ///
    /// Lower timeouts and limits for snappy response.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            suggest_timeout_ms: 50,
            max_model_size: 50 * 1024 * 1024,    // 50 MB
            max_history_size: 100 * 1024 * 1024, // 100 MB
            max_suggestions: 5,
            max_prefix_length: 200,
            min_prefix_length: 2,
            min_quality_score: 0.5,
        }
    }

    /// Thorough configuration for batch processing.
    ///
    /// Higher limits for comprehensive results.
    #[must_use]
    pub fn thorough() -> Self {
        Self {
            suggest_timeout_ms: 500,
            max_model_size: 500 * 1024 * 1024,    // 500 MB
            max_history_size: 1024 * 1024 * 1024, // 1 GB
            max_suggestions: 20,
            max_prefix_length: 1000,
            min_prefix_length: 1,
            min_quality_score: 0.1,
        }
    }
}

use crate::model::MarkovModel;
use crate::quality::suggestion_quality_score;
use crate::security::is_sensitive_command;
use std::time::Instant;

/// Suggestion with graceful degradation and timeout handling.
///
/// This function applies all hardening measures:
/// - Prefix truncation if too long
/// - Timeout-based suggestion generation
/// - Quality filtering
/// - Security filtering
/// - Result limiting
///
/// # Arguments
/// * `prefix` - The command prefix to complete
/// * `model` - Optional model reference (returns empty if None)
/// * `config` - Configuration with limits
///
/// # Example
/// ```
/// use aprender_shell::config::{ShellConfig, suggest_with_fallback};
/// use aprender_shell::model::MarkovModel;
///
/// let config = ShellConfig::fast();
/// // Without model - returns empty
/// let suggestions = suggest_with_fallback("git ", None, &config);
/// assert!(suggestions.is_empty());
/// ```
pub fn suggest_with_fallback(
    prefix: &str,
    model: Option<&MarkovModel>,
    config: &ShellConfig,
) -> Vec<(String, f32)> {
    let model = match model {
        Some(m) => m,
        None => return vec![],
    };

    if !is_prefix_processable(prefix, config) {
        return vec![];
    }

    let prefix = config.truncate_prefix(prefix);
    let raw_suggestions = model.suggest(prefix, config.max_suggestions * 2);

    filter_suggestions(raw_suggestions, config)
}

/// Check if prefix meets minimum length requirements.
fn is_prefix_processable(prefix: &str, config: &ShellConfig) -> bool {
    prefix.len() >= config.min_prefix_length
}

/// Filter and score suggestions with timeout, security, and quality checks.
fn filter_suggestions(
    raw_suggestions: Vec<(String, f32)>,
    config: &ShellConfig,
) -> Vec<(String, f32)> {
    let deadline = Instant::now() + config.suggest_timeout();
    let mut results = Vec::with_capacity(config.max_suggestions);

    for (suggestion, score) in raw_suggestions {
        if should_stop_filtering(&results, &deadline, config) {
            break;
        }

        if let Some(scored) = process_suggestion(&suggestion, score, config) {
            results.push(scored);
        }
    }

    results
}

/// Check if filtering should stop due to timeout or result limit.
fn should_stop_filtering(
    results: &[(String, f32)],
    deadline: &Instant,
    config: &ShellConfig,
) -> bool {
    Instant::now() > *deadline || results.len() >= config.max_suggestions
}

/// Process a single suggestion: security and quality filtering.
fn process_suggestion(suggestion: &str, score: f32, config: &ShellConfig) -> Option<(String, f32)> {
    if is_sensitive_command(suggestion) {
        return None;
    }

    let quality = suggestion_quality_score(suggestion);
    if quality < config.min_quality_score {
        return None;
    }

    Some((suggestion.to_string(), score * quality))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = ShellConfig::default();
        assert_eq!(config.suggest_timeout_ms, 100);
        assert_eq!(config.max_model_size, 100 * 1024 * 1024);
        assert_eq!(config.max_history_size, 500 * 1024 * 1024);
        assert_eq!(config.max_suggestions, 10);
        assert_eq!(config.max_prefix_length, 500);
        assert_eq!(config.min_prefix_length, 2);
    }

    #[test]
    fn test_builder_pattern() {
        let config = ShellConfig::new()
            .with_suggest_timeout_ms(50)
            .with_max_suggestions(5)
            .with_min_quality_score(0.5);

        assert_eq!(config.suggest_timeout_ms, 50);
        assert_eq!(config.max_suggestions, 5);
        assert!((config.min_quality_score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_score_clamped() {
        let config = ShellConfig::new().with_min_quality_score(1.5);
        assert!((config.min_quality_score - 1.0).abs() < f32::EPSILON);

        let config = ShellConfig::new().with_min_quality_score(-0.5);
        assert!((config.min_quality_score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_suggest_timeout_duration() {
        let config = ShellConfig::new().with_suggest_timeout_ms(100);
        assert_eq!(config.suggest_timeout(), Duration::from_millis(100));
    }

    #[test]
    fn test_model_size_validation() {
        let config = ShellConfig::new().with_max_model_size(1024);
        assert!(config.is_model_size_valid(512));
        assert!(config.is_model_size_valid(1024));
        assert!(!config.is_model_size_valid(2048));
    }

    #[test]
    fn test_history_size_validation() {
        let config = ShellConfig::new().with_max_history_size(1024);
        assert!(config.is_history_size_valid(512));
        assert!(!config.is_history_size_valid(2048));
    }

    #[test]
    fn test_prefix_validation() {
        let config = ShellConfig::new()
            .with_min_prefix_length(2)
            .with_max_prefix_length(10);

        assert!(!config.is_prefix_valid("a")); // Too short
        assert!(config.is_prefix_valid("ab")); // Minimum
        assert!(config.is_prefix_valid("hello")); // Within range
        assert!(config.is_prefix_valid("0123456789")); // Maximum
        assert!(!config.is_prefix_valid("01234567890")); // Too long
    }

    #[test]
    fn test_truncate_prefix() {
        let config = ShellConfig::new().with_max_prefix_length(5);

        assert_eq!(config.truncate_prefix("abc"), "abc");
        assert_eq!(config.truncate_prefix("abcde"), "abcde");
        assert_eq!(config.truncate_prefix("abcdefgh"), "abcde");
    }

    #[test]
    fn test_truncate_prefix_utf8_boundary() {
        let config = ShellConfig::new().with_max_prefix_length(5);

        // "日本" is 6 bytes (3 bytes each), truncating to 5 should give first char
        let jp = "日本";
        let truncated = config.truncate_prefix(jp);
        assert!(truncated.len() <= 5);
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn test_fast_preset() {
        let config = ShellConfig::fast();
        assert_eq!(config.suggest_timeout_ms, 50);
        assert_eq!(config.max_suggestions, 5);
    }

    #[test]
    fn test_thorough_preset() {
        let config = ShellConfig::thorough();
        assert_eq!(config.suggest_timeout_ms, 500);
        assert_eq!(config.max_suggestions, 20);
    }

    #[test]
    fn test_clone_and_eq() {
        let config1 = ShellConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1, config2);
    }

    // =========================================================================
    // suggest_with_fallback Tests
    // =========================================================================

    #[test]
    fn test_suggest_without_model_returns_empty() {
        let config = ShellConfig::default();
        let suggestions = suggest_with_fallback("git ", None, &config);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_with_short_prefix_returns_empty() {
        let config = ShellConfig::default().with_min_prefix_length(3);

        let mut model = MarkovModel::new(3);
        model.train(&["git status".to_string()]);

        let suggestions = suggest_with_fallback("g", Some(&model), &config);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_with_model_returns_results() {
        let config = ShellConfig::default();

        let mut model = MarkovModel::new(3);
        model.train(&[
            "git status".to_string(),
            "git commit".to_string(),
            "git push".to_string(),
        ]);

        let suggestions = suggest_with_fallback("git ", Some(&model), &config);
        // Should have at least one suggestion
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_suggest_respects_max_suggestions() {
        let config = ShellConfig::default().with_max_suggestions(2);

        let mut model = MarkovModel::new(3);
        model.train(&[
            "git status".to_string(),
            "git commit".to_string(),
            "git push".to_string(),
            "git pull".to_string(),
            "git fetch".to_string(),
        ]);

        let suggestions = suggest_with_fallback("git ", Some(&model), &config);
        assert!(suggestions.len() <= 2);
    }

    #[test]
    fn test_suggest_filters_sensitive_commands() {
        let config = ShellConfig::default().with_min_quality_score(0.0);

        let mut model = MarkovModel::new(3);
        model.train(&[
            "git status".to_string(),
            "export SECRET=abc".to_string(),
            "curl -u admin:pass http://localhost".to_string(),
        ]);

        let suggestions = suggest_with_fallback("export ", Some(&model), &config);
        // Should not contain sensitive commands
        for (suggestion, _) in &suggestions {
            assert!(!suggestion.contains("SECRET="));
        }
    }
}
