//! Per-Layer Merge Granularity for Model Merging (GH-452)
//!
//! Provides fine-grained control over how individual layers are merged
//! when combining multiple models. Supports per-layer strategy overrides
//! via a YAML-like configuration format.
//!
//! # References
//!
//! - [Wortsman et al. 2022] "Model soups: averaging weights of multiple
//!   fine-tuned models improves accuracy without increasing inference time"
//! - [Yadav et al. 2023] "TIES-Merging: Resolving Interference When Merging
//!   Models"
//!
//! # Toyota Way Principles
//!
//! - **Standardization**: Declarative YAML config for reproducible merges
//! - **Poka-Yoke**: Validation prevents misconfigured merges before execution
//! - **Heijunka**: Per-layer weights enable load-leveled contribution

use std::collections::HashMap;

use crate::error::{AprenderError, Result};

/// Valid merge strategy names
const VALID_STRATEGIES: &[&str] = &[
    "average",
    "weighted_average",
    "slerp",
    "ties",
    "dare",
    "passthrough",
];

/// Configuration for per-layer merge granularity
///
/// Allows different merge strategies and weights for different layer
/// patterns, enabling fine-grained control over model combination.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::online::per_layer_merge::LayerMergeConfig;
///
/// let config = LayerMergeConfig {
///     layer_rules: vec![
///         LayerRule {
///             layer_pattern: "attn".to_string(),
///             strategy: "slerp".to_string(),
///             weights: Some(vec![0.7, 0.3]),
///             scale: None,
///         },
///     ],
///     default_strategy: "average".to_string(),
///     default_weights: vec![0.5, 0.5],
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LayerMergeConfig {
    /// Ordered list of layer rules (first match wins)
    pub layer_rules: Vec<LayerRule>,
    /// Fallback strategy when no rule matches
    pub default_strategy: String,
    /// Default weights for models when no rule-specific weights exist
    pub default_weights: Vec<f64>,
}

/// A single rule mapping a layer name pattern to a merge strategy
///
/// Patterns use simple substring matching with support for `*` wildcards.
/// For example, `"layers.0."` matches any tensor name containing that
/// substring, while `"attn*weight"` matches names containing "attn"
/// followed (possibly later) by "weight".
#[derive(Debug, Clone)]
pub struct LayerRule {
    /// Pattern to match against tensor names (substring or wildcard `*`)
    pub layer_pattern: String,
    /// Merge strategy for matched tensors
    pub strategy: String,
    /// Optional per-model weights (overrides default_weights)
    pub weights: Option<Vec<f64>>,
    /// Optional scaling factor applied after merge
    pub scale: Option<f64>,
}

/// Top-level YAML merge configuration
///
/// Parsed from a YAML-like config file specifying models to merge,
/// output path, default strategy, and optional per-layer rules.
#[derive(Debug, Clone)]
pub struct MergeYamlConfig {
    /// Source models to merge
    pub models: Vec<ModelSource>,
    /// Output path for merged model
    pub output: String,
    /// Default merge strategy
    pub default_strategy: String,
    /// Optional per-layer rules
    pub layers: Option<Vec<LayerRule>>,
}

/// A model source for merging
#[derive(Debug, Clone)]
pub struct ModelSource {
    /// Path to the model file
    pub path: String,
    /// Optional weight for this model in the merge
    pub weight: Option<f64>,
}

/// Report summarizing a merge operation
#[derive(Debug, Clone)]
pub struct LayerMergeReport {
    /// Total tensors processed during merge
    pub tensors_processed: usize,
    /// Count of tensors matched by each rule pattern
    pub rules_matched: HashMap<String, usize>,
}

impl LayerMergeReport {
    /// Create a new empty report
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors_processed: 0,
            rules_matched: HashMap::new(),
        }
    }

    /// Record a tensor being processed, optionally matched by a rule
    pub fn record_tensor(&mut self, matched_pattern: Option<&str>) {
        self.tensors_processed += 1;
        if let Some(pattern) = matched_pattern {
            *self.rules_matched.entry(pattern.to_string()).or_insert(0) += 1;
        }
    }

    /// Total number of tensors that matched at least one rule
    #[must_use]
    pub fn total_matched(&self) -> usize {
        self.rules_matched.values().sum()
    }

    /// Total number of tensors that used the default strategy
    #[must_use]
    pub fn total_defaulted(&self) -> usize {
        self.tensors_processed.saturating_sub(self.total_matched())
    }
}

impl Default for LayerMergeReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Match a tensor name against an ordered list of layer rules
///
/// Returns the first matching rule. Patterns support:
/// - Substring match: `"attn"` matches `"model.layers.0.self_attn.q_proj.weight"`
/// - Wildcard `*`: `"layers.0.*weight"` matches names containing `"layers.0."`
///   followed (possibly later) by `"weight"`
/// - Escaped dot `\\.`: treated as a literal `.` in the pattern
///
/// # Arguments
///
/// * `tensor_name` - The fully qualified tensor name
/// * `rules` - Ordered slice of layer rules to try
///
/// # Returns
///
/// Reference to the first matching `LayerRule`, or `None` if no rule matches.
pub fn match_layer_rule<'a>(tensor_name: &str, rules: &'a [LayerRule]) -> Option<&'a LayerRule> {
    rules
        .iter()
        .find(|rule| pattern_matches(tensor_name, &rule.layer_pattern))
}

/// Simple pattern matching supporting substring and `*` wildcards
///
/// Splits the pattern on `*` and checks that all segments appear in order
/// within the target string. Escaped dots (`\\.`) are treated as literal dots.
fn pattern_matches(name: &str, pattern: &str) -> bool {
    // Normalize escaped dots to literal dots for matching
    let normalized = pattern.replace("\\.", ".");

    if !normalized.contains('*') {
        // Pure substring match
        return name.contains(&normalized);
    }

    // Wildcard match: split on * and ensure all parts appear in order
    let parts: Vec<&str> = normalized.split('*').collect();
    let mut search_from = 0;

    for part in &parts {
        if part.is_empty() {
            continue;
        }
        match name[search_from..].find(part) {
            Some(pos) => {
                search_from += pos + part.len();
            }
            None => return false,
        }
    }
    true
}

/// Parser state machine section tracking
#[derive(Debug, PartialEq)]
enum ParserSection {
    Root,
    Models,
    ModelEntry,
    Layers,
    LayerEntry,
}

/// Parse a YAML-like merge configuration string
///
/// Supports a simplified YAML subset with the following structure:
///
/// ```yaml
/// models:
///   - path: /path/to/model1.apr
///     weight: 0.6
///   - path: /path/to/model2.apr
///     weight: 0.4
/// output: /path/to/merged.apr
/// default_strategy: average
/// layers:
///   - layer_pattern: "attn"
///     strategy: slerp
///     weights: [0.7, 0.3]
///     scale: 1.0
/// ```
///
/// # Errors
///
/// Returns `AprenderError::ValidationError` if the YAML is malformed
/// or missing required fields.
pub fn parse_merge_yaml(yaml_str: &str) -> Result<MergeYamlConfig> {
    let mut models: Vec<ModelSource> = Vec::new();
    let mut output = String::new();
    let mut default_strategy = String::new();
    let mut layer_rules: Vec<LayerRule> = Vec::new();

    let mut section = ParserSection::Root;
    let mut current_model_path = String::new();
    let mut current_model_weight: Option<f64> = None;
    let mut current_layer_pattern = String::new();
    let mut current_layer_strategy = String::new();
    let mut current_layer_weights: Option<Vec<f64>> = None;
    let mut current_layer_scale: Option<f64> = None;

    for line in yaml_str.lines() {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Detect top-level section headers (e.g., "models:" at column 0)
        if !line.starts_with(' ') && !line.starts_with('\t') && !trimmed.starts_with('-') {
            // Flush pending entries before switching sections
            flush_model_entry(
                &section,
                &mut models,
                &mut current_model_path,
                &mut current_model_weight,
            );
            flush_layer_entry(
                &section,
                &mut layer_rules,
                &mut current_layer_pattern,
                &mut current_layer_strategy,
                &mut current_layer_weights,
                &mut current_layer_scale,
            );

            if trimmed.ends_with(':') && !trimmed.contains(": ") {
                let key = trimmed.trim_end_matches(':');
                match key {
                    "models" => {
                        section = ParserSection::Models;
                        continue;
                    }
                    "layers" => {
                        section = ParserSection::Layers;
                        continue;
                    }
                    _ => {}
                }
            }

            // Root-level "key: value" pairs
            if let Some(val) = parse_kv(trimmed, "output") {
                output = unquote(val);
                section = ParserSection::Root;
            } else if let Some(val) = parse_kv(trimmed, "default_strategy") {
                default_strategy = unquote(val);
                section = ParserSection::Root;
            }
            continue;
        }

        // Indented content — belongs to current section
        match section {
            ParserSection::Models | ParserSection::ModelEntry => {
                if trimmed.starts_with("- ") || trimmed == "-" {
                    // Flush previous model entry
                    flush_model_entry(
                        &section,
                        &mut models,
                        &mut current_model_path,
                        &mut current_model_weight,
                    );
                    section = ParserSection::ModelEntry;

                    // Handle inline "- path: value"
                    let after_dash = trimmed.trim_start_matches('-').trim();
                    if let Some(val) = parse_kv(after_dash, "path") {
                        current_model_path = unquote(val);
                    }
                } else if let Some(val) = parse_kv(trimmed, "path") {
                    current_model_path = unquote(val);
                } else if let Some(val) = parse_kv(trimmed, "weight") {
                    current_model_weight = val.trim().parse::<f64>().ok();
                }
            }
            ParserSection::Layers | ParserSection::LayerEntry => {
                if trimmed.starts_with("- ") || trimmed == "-" {
                    // Flush previous layer entry
                    flush_layer_entry(
                        &section,
                        &mut layer_rules,
                        &mut current_layer_pattern,
                        &mut current_layer_strategy,
                        &mut current_layer_weights,
                        &mut current_layer_scale,
                    );
                    section = ParserSection::LayerEntry;

                    let after_dash = trimmed.trim_start_matches('-').trim();
                    if let Some(val) = parse_kv(after_dash, "layer_pattern") {
                        current_layer_pattern = unquote(val);
                    }
                } else if let Some(val) = parse_kv(trimmed, "layer_pattern") {
                    current_layer_pattern = unquote(val);
                } else if let Some(val) = parse_kv(trimmed, "strategy") {
                    current_layer_strategy = unquote(val);
                } else if let Some(val) = parse_kv(trimmed, "weights") {
                    current_layer_weights = Some(parse_float_list(val));
                } else if let Some(val) = parse_kv(trimmed, "scale") {
                    current_layer_scale = val.trim().parse::<f64>().ok();
                }
            }
            ParserSection::Root => {
                // Indented root content — try key: value
                if let Some(val) = parse_kv(trimmed, "output") {
                    output = unquote(val);
                } else if let Some(val) = parse_kv(trimmed, "default_strategy") {
                    default_strategy = unquote(val);
                }
            }
        }
    }

    // Flush final pending entries
    flush_model_entry(
        &section,
        &mut models,
        &mut current_model_path,
        &mut current_model_weight,
    );
    flush_layer_entry(
        &section,
        &mut layer_rules,
        &mut current_layer_pattern,
        &mut current_layer_strategy,
        &mut current_layer_weights,
        &mut current_layer_scale,
    );

    if output.is_empty() {
        return Err(AprenderError::ValidationError {
            message: "merge config missing required field: output".to_string(),
        });
    }

    if default_strategy.is_empty() {
        return Err(AprenderError::ValidationError {
            message: "merge config missing required field: default_strategy".to_string(),
        });
    }

    let layers = if layer_rules.is_empty() {
        None
    } else {
        Some(layer_rules)
    };

    Ok(MergeYamlConfig {
        models,
        output,
        default_strategy,
        layers,
    })
}

/// Validate a parsed merge configuration
///
/// Checks:
/// - At least 2 models are specified
/// - All strategy names are recognized
/// - Weights (if provided) are non-negative and finite
/// - Layer patterns are non-empty
///
/// # Errors
///
/// Returns `AprenderError::ValidationError` with a descriptive message
/// on the first validation failure.
pub fn validate_merge_config(config: &MergeYamlConfig) -> Result<()> {
    if config.models.len() < 2 {
        return Err(AprenderError::ValidationError {
            message: format!(
                "merge requires at least 2 models, got {}",
                config.models.len()
            ),
        });
    }

    // Validate default strategy
    if !is_valid_strategy(&config.default_strategy) {
        return Err(AprenderError::ValidationError {
            message: format!(
                "unknown default strategy '{}', valid: {}",
                config.default_strategy,
                VALID_STRATEGIES.join(", ")
            ),
        });
    }

    // Validate model paths
    for (i, model) in config.models.iter().enumerate() {
        if model.path.is_empty() {
            return Err(AprenderError::ValidationError {
                message: format!("model {} has empty path", i),
            });
        }
        if let Some(w) = model.weight {
            if !w.is_finite() || w < 0.0 {
                return Err(AprenderError::ValidationError {
                    message: format!(
                        "model {} weight must be non-negative and finite, got {}",
                        i, w
                    ),
                });
            }
        }
    }

    // Validate output path
    if config.output.is_empty() {
        return Err(AprenderError::ValidationError {
            message: "output path is empty".to_string(),
        });
    }

    // Validate layer rules
    if let Some(ref rules) = config.layers {
        for (i, rule) in rules.iter().enumerate() {
            if rule.layer_pattern.is_empty() {
                return Err(AprenderError::ValidationError {
                    message: format!("layer rule {} has empty pattern", i),
                });
            }
            if !is_valid_strategy(&rule.strategy) {
                return Err(AprenderError::ValidationError {
                    message: format!(
                        "layer rule {} has unknown strategy '{}', valid: {}",
                        i,
                        rule.strategy,
                        VALID_STRATEGIES.join(", ")
                    ),
                });
            }
            if let Some(ref weights) = rule.weights {
                for (j, &w) in weights.iter().enumerate() {
                    if !w.is_finite() {
                        return Err(AprenderError::ValidationError {
                            message: format!("layer rule {} weight[{}] is not finite: {}", i, j, w),
                        });
                    }
                }
            }
            if let Some(s) = rule.scale {
                if !s.is_finite() {
                    return Err(AprenderError::ValidationError {
                        message: format!("layer rule {} scale is not finite: {}", i, s),
                    });
                }
            }
        }
    }

    Ok(())
}

/// Check whether a strategy name is recognized
fn is_valid_strategy(name: &str) -> bool {
    VALID_STRATEGIES.contains(&name)
}

/// Parse a `key: value` pair, returning the value if the key matches
fn parse_kv<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let trimmed = line.trim();
    let prefix_with_space = format!("{}: ", key);
    let prefix_bare = format!("{}:", key);

    if trimmed.starts_with(&prefix_with_space) {
        Some(trimmed[prefix_with_space.len()..].trim())
    } else if trimmed == prefix_bare {
        // Key with no value (e.g., "models:")
        None
    } else if trimmed.starts_with(&prefix_bare) {
        let rest = &trimmed[prefix_bare.len()..];
        if rest.is_empty() {
            None
        } else {
            Some(rest.trim())
        }
    } else {
        None
    }
}

/// Remove surrounding quotes from a string value
fn unquote(s: &str) -> String {
    let trimmed = s.trim();
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        if trimmed.len() >= 2 {
            trimmed[1..trimmed.len() - 1].to_string()
        } else {
            trimmed.to_string()
        }
    } else {
        trimmed.to_string()
    }
}

/// Parse a bracketed list of floats like `[0.7, 0.3]`
fn parse_float_list(s: &str) -> Vec<f64> {
    let inner = s
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .collect()
}

/// Flush a pending model entry into the models list
fn flush_model_entry(
    section: &ParserSection,
    models: &mut Vec<ModelSource>,
    path: &mut String,
    weight: &mut Option<f64>,
) {
    if matches!(section, ParserSection::ModelEntry) && !path.is_empty() {
        models.push(ModelSource {
            path: std::mem::take(path),
            weight: weight.take(),
        });
    }
}

/// Flush a pending layer rule entry into the rules list
fn flush_layer_entry(
    section: &ParserSection,
    rules: &mut Vec<LayerRule>,
    pattern: &mut String,
    strategy: &mut String,
    weights: &mut Option<Vec<f64>>,
    scale: &mut Option<f64>,
) {
    if matches!(section, ParserSection::LayerEntry) && !pattern.is_empty() {
        rules.push(LayerRule {
            layer_pattern: std::mem::take(pattern),
            strategy: if strategy.is_empty() {
                "average".to_string()
            } else {
                std::mem::take(strategy)
            },
            weights: weights.take(),
            scale: scale.take(),
        });
    }
}

#[cfg(test)]
#[path = "per_layer_merge_tests.rs"]
mod tests;
