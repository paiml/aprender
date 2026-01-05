//! Template-based synthetic data generation.
//!
//! Uses parameterized templates with slot filling to generate
//! variations of commands or text. Fast and controllable, but
//! limited diversity compared to learned approaches.
//!
//! # Example
//!
//! ```
//! use aprender::synthetic::template::{TemplateGenerator, Template};
//!
//! let mut gen = TemplateGenerator::new();
//! gen.add_template(Template::new("git {action} {target}")
//!     .with_slot("action", &["commit", "push", "pull"])
//!     .with_slot("target", &["origin", "upstream"]));
//!
//! let samples = gen.generate_samples(5, 42);
//! assert_eq!(samples.len(), 5);
//! ```

use super::{SyntheticConfig, SyntheticGenerator};
use crate::error::Result;
use std::collections::HashMap;

/// A template with named slots for variation.
#[derive(Debug, Clone)]
pub struct Template {
    /// Template pattern with {`slot_name`} placeholders.
    pattern: String,
    /// Slot name -> possible values.
    slots: HashMap<String, Vec<String>>,
    /// Weight for selection (higher = more likely).
    weight: f32,
}

impl Template {
    /// Create a new template from a pattern string.
    ///
    /// Slots are marked with `{slot_name}` syntax.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::synthetic::template::Template;
    ///
    /// let template = Template::new("git {cmd} -m {msg}")
    ///     .with_slot("cmd", &["commit", "stash"])
    ///     .with_slot("msg", &["fix", "update"]);
    /// ```
    #[must_use]
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            slots: HashMap::new(),
            weight: 1.0,
        }
    }

    /// Add a slot with possible values.
    #[must_use]
    pub fn with_slot(mut self, name: &str, values: &[&str]) -> Self {
        self.slots.insert(
            name.to_string(),
            values.iter().map(|s| (*s).to_string()).collect(),
        );
        self
    }

    /// Set the weight for this template.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    /// Get the pattern string.
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Get all slot names.
    #[must_use]
    pub fn slot_names(&self) -> Vec<&String> {
        self.slots.keys().collect()
    }

    /// Get values for a slot.
    #[must_use]
    pub fn slot_values(&self, name: &str) -> Option<&Vec<String>> {
        self.slots.get(name)
    }

    /// Get the weight.
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Calculate total number of possible combinations.
    #[must_use]
    pub fn combination_count(&self) -> usize {
        if self.slots.is_empty() {
            return 1;
        }
        self.slots.values().map(|v| v.len().max(1)).product()
    }

    /// Fill the template with specific slot values.
    #[must_use]
    pub fn fill(&self, values: &HashMap<String, String>) -> String {
        let mut result = self.pattern.clone();
        for (name, value) in values {
            let placeholder = format!("{{{name}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Generate a filled template using indexed slot values.
    ///
    /// Uses modular arithmetic to map index to slot combination.
    #[must_use]
    pub fn fill_indexed(&self, index: usize) -> String {
        if self.slots.is_empty() {
            return self.pattern.clone();
        }

        let mut result = self.pattern.clone();
        let mut remaining = index;

        // Sort slot names for deterministic ordering
        let mut slot_names: Vec<_> = self.slots.keys().collect();
        slot_names.sort();

        for name in slot_names {
            if let Some(values) = self.slots.get(name) {
                if !values.is_empty() {
                    let value_idx = remaining % values.len();
                    remaining /= values.len();

                    let placeholder = format!("{{{name}}}");
                    result = result.replace(&placeholder, &values[value_idx]);
                }
            }
        }

        result
    }
}

/// Simple RNG for deterministic template generation.
#[derive(Debug, Clone)]
struct TemplateRng {
    state: u64,
}

impl TemplateRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next() as usize) % max
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }
}

/// Configuration for template generation.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateConfig {
    /// Whether to use weighted template selection.
    pub use_weights: bool,
    /// Whether to ensure unique outputs.
    pub unique_outputs: bool,
    /// Maximum attempts to generate unique output.
    pub max_unique_attempts: usize,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            use_weights: true,
            unique_outputs: true,
            max_unique_attempts: 100,
        }
    }
}

impl TemplateConfig {
    /// Create new template configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to use weighted selection.
    #[must_use]
    pub fn with_use_weights(mut self, use_weights: bool) -> Self {
        self.use_weights = use_weights;
        self
    }

    /// Set whether to ensure unique outputs.
    #[must_use]
    pub fn with_unique_outputs(mut self, unique: bool) -> Self {
        self.unique_outputs = unique;
        self
    }

    /// Set max attempts for unique generation.
    #[must_use]
    pub fn with_max_unique_attempts(mut self, attempts: usize) -> Self {
        self.max_unique_attempts = attempts.max(1);
        self
    }
}

/// Template-based synthetic data generator.
///
/// Generates variations by filling slots in templates.
#[derive(Debug, Clone)]
pub struct TemplateGenerator {
    templates: Vec<Template>,
    config: TemplateConfig,
}

impl Default for TemplateGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateGenerator {
    /// Create a new template generator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            config: TemplateConfig::default(),
        }
    }

    /// Create a generator with configuration.
    #[must_use]
    pub fn with_config(config: TemplateConfig) -> Self {
        Self {
            templates: Vec::new(),
            config,
        }
    }

    /// Add a template.
    pub fn add_template(&mut self, template: Template) {
        self.templates.push(template);
    }

    /// Add a template (builder pattern).
    #[must_use]
    pub fn with_template(mut self, template: Template) -> Self {
        self.templates.push(template);
        self
    }

    /// Get all templates.
    #[must_use]
    pub fn templates(&self) -> &[Template] {
        &self.templates
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &TemplateConfig {
        &self.config
    }

    /// Calculate total possible combinations across all templates.
    #[must_use]
    pub fn total_combinations(&self) -> usize {
        self.templates.iter().map(Template::combination_count).sum()
    }

    /// Generate samples from templates.
    #[must_use]
    pub fn generate_samples(&self, count: usize, seed: u64) -> Vec<String> {
        if self.templates.is_empty() {
            return Vec::new();
        }

        let mut rng = TemplateRng::new(seed);
        let mut results = Vec::with_capacity(count);
        let mut seen = std::collections::HashSet::new();

        let total_weight: f32 = if self.config.use_weights {
            self.templates.iter().map(Template::weight).sum()
        } else {
            self.templates.len() as f32
        };

        for _ in 0..count {
            let mut attempts = 0;
            loop {
                // Select template
                let template = if self.config.use_weights {
                    self.select_weighted(&mut rng, total_weight)
                } else {
                    &self.templates[rng.next_usize(self.templates.len())]
                };

                // Generate from template
                let idx = rng.next_usize(template.combination_count().max(1));
                let sample = template.fill_indexed(idx);

                // Check uniqueness
                if !self.config.unique_outputs || !seen.contains(&sample) {
                    seen.insert(sample.clone());
                    results.push(sample);
                    break;
                }

                attempts += 1;
                if attempts >= self.config.max_unique_attempts {
                    // Give up on uniqueness, accept duplicate
                    results.push(sample);
                    break;
                }
            }
        }

        results
    }

    /// Select a template based on weights.
    fn select_weighted(&self, rng: &mut TemplateRng, total_weight: f32) -> &Template {
        let threshold = rng.next_f32() * total_weight;
        let mut cumulative = 0.0;

        for template in &self.templates {
            cumulative += template.weight();
            if cumulative >= threshold {
                return template;
            }
        }

        // Fallback to last template
        self.templates
            .last()
            .expect("templates should not be empty")
    }

    /// Create common shell command templates.
    #[must_use]
    pub fn shell_commands() -> Self {
        Self::new()
            .with_template(
                Template::new("git {cmd}")
                    .with_slot("cmd", &["status", "log", "diff", "branch", "fetch"])
                    .with_weight(2.0),
            )
            .with_template(
                Template::new("git {cmd} {target}")
                    .with_slot("cmd", &["checkout", "merge", "rebase", "push", "pull"])
                    .with_slot("target", &["main", "master", "develop", "origin/main"]),
            )
            .with_template(Template::new("git commit -m \"{msg}\"").with_slot(
                "msg",
                &[
                    "fix: bug",
                    "feat: add feature",
                    "docs: update",
                    "refactor: clean up",
                ],
            ))
            .with_template(
                Template::new("cargo {cmd}")
                    .with_slot("cmd", &["build", "test", "run", "check", "clippy", "fmt"])
                    .with_weight(1.5),
            )
            .with_template(
                Template::new("cargo {cmd} --{flag}")
                    .with_slot("cmd", &["build", "test", "run"])
                    .with_slot("flag", &["release", "all-features", "verbose"]),
            )
            .with_template(
                Template::new("npm {cmd}")
                    .with_slot("cmd", &["install", "run", "test", "build", "start"]),
            )
            .with_template(
                Template::new("docker {cmd} {target}")
                    .with_slot("cmd", &["run", "build", "pull", "push", "stop"])
                    .with_slot("target", &["nginx", "postgres", "redis", "."]),
            )
    }
}

impl SyntheticGenerator for TemplateGenerator {
    type Input = String;
    type Output = String;

    fn generate(&self, seeds: &[String], config: &SyntheticConfig) -> Result<Vec<String>> {
        let target = config.target_count(seeds.len());

        // Use seeds as additional context for generation
        // For templates, we generate based on template patterns
        let mut results = self.generate_samples(target, config.seed);

        // Filter by quality threshold
        results.retain(|sample| {
            // Template outputs are always valid syntactically
            // Quality is based on length and structure
            let quality = if sample.len() >= 3 { 0.8 } else { 0.3 };
            quality >= config.quality_threshold
        });

        Ok(results)
    }

    fn quality_score(&self, generated: &String, _seed: &String) -> f32 {
        // Template outputs are syntactically correct by construction
        // Score based on length and structure
        if generated.is_empty() {
            return 0.0;
        }
        if generated.len() < 3 {
            return 0.3;
        }

        // Check for common patterns
        let has_command = generated.split_whitespace().next().is_some();
        let word_count = generated.split_whitespace().count();

        let length_score = (word_count as f32 / 5.0).min(1.0);
        let structure_score = if has_command { 0.8 } else { 0.2 };

        0.5 * length_score + 0.5 * structure_score
    }

    fn diversity_score(&self, batch: &[String]) -> f32 {
        if batch.len() < 2 {
            return 1.0;
        }

        // Count unique samples
        let unique: std::collections::HashSet<_> = batch.iter().collect();
        unique.len() as f32 / batch.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // EXTREME TDD: Template Tests
    // ============================================================================

    #[test]
    fn test_template_new() {
        let t = Template::new("git {cmd}");
        assert_eq!(t.pattern(), "git {cmd}");
        assert!(t.slots.is_empty());
        assert!((t.weight() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_template_with_slot() {
        let t = Template::new("git {cmd}").with_slot("cmd", &["status", "log"]);

        assert_eq!(t.slot_names().len(), 1);
        let values = t.slot_values("cmd").expect("should have values");
        assert_eq!(values.len(), 2);
        assert!(values.contains(&"status".to_string()));
    }

    #[test]
    fn test_template_with_weight() {
        let t = Template::new("test").with_weight(2.5);
        assert!((t.weight() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_template_weight_non_negative() {
        let t = Template::new("test").with_weight(-1.0);
        assert!((t.weight() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_template_combination_count() {
        let t = Template::new("git {cmd} {target}")
            .with_slot("cmd", &["a", "b", "c"])
            .with_slot("target", &["x", "y"]);

        assert_eq!(t.combination_count(), 6); // 3 * 2
    }

    #[test]
    fn test_template_combination_count_no_slots() {
        let t = Template::new("git status");
        assert_eq!(t.combination_count(), 1);
    }

    #[test]
    fn test_template_fill() {
        let t = Template::new("git {cmd} {target}")
            .with_slot("cmd", &["push"])
            .with_slot("target", &["origin"]);

        let mut values = HashMap::new();
        values.insert("cmd".to_string(), "push".to_string());
        values.insert("target".to_string(), "origin".to_string());

        assert_eq!(t.fill(&values), "git push origin");
    }

    #[test]
    fn test_template_fill_indexed() {
        let t = Template::new("{a} {b}")
            .with_slot("a", &["x", "y"])
            .with_slot("b", &["1", "2"]);

        // With sorted keys: a, b
        // Index 0: a[0]=x, b[0]=1 -> "x 1"
        // Index 1: a[1]=y, b[0]=1 -> "y 1"
        // Index 2: a[0]=x, b[1]=2 -> "x 2"
        // Index 3: a[1]=y, b[1]=2 -> "y 2"

        let s0 = t.fill_indexed(0);
        let s1 = t.fill_indexed(1);
        let s2 = t.fill_indexed(2);
        let s3 = t.fill_indexed(3);

        // All should be valid combinations
        let valid = ["x 1", "x 2", "y 1", "y 2"];
        assert!(valid.contains(&s0.as_str()));
        assert!(valid.contains(&s1.as_str()));
        assert!(valid.contains(&s2.as_str()));
        assert!(valid.contains(&s3.as_str()));
    }

    // ============================================================================
    // EXTREME TDD: TemplateConfig Tests
    // ============================================================================

    #[test]
    fn test_template_config_default() {
        let config = TemplateConfig::default();
        assert!(config.use_weights);
        assert!(config.unique_outputs);
        assert_eq!(config.max_unique_attempts, 100);
    }

    #[test]
    fn test_template_config_builder() {
        let config = TemplateConfig::new()
            .with_use_weights(false)
            .with_unique_outputs(false)
            .with_max_unique_attempts(50);

        assert!(!config.use_weights);
        assert!(!config.unique_outputs);
        assert_eq!(config.max_unique_attempts, 50);
    }

    #[test]
    fn test_template_config_min_attempts() {
        let config = TemplateConfig::new().with_max_unique_attempts(0);
        assert_eq!(config.max_unique_attempts, 1);
    }

    // ============================================================================
    // EXTREME TDD: TemplateGenerator Tests
    // ============================================================================

    #[test]
    fn test_template_generator_new() {
        let gen = TemplateGenerator::new();
        assert!(gen.templates().is_empty());
    }

    #[test]
    fn test_template_generator_add_template() {
        let mut gen = TemplateGenerator::new();
        gen.add_template(Template::new("test"));
        assert_eq!(gen.templates().len(), 1);
    }

    #[test]
    fn test_template_generator_with_template() {
        let gen = TemplateGenerator::new()
            .with_template(Template::new("a"))
            .with_template(Template::new("b"));

        assert_eq!(gen.templates().len(), 2);
    }

    #[test]
    fn test_template_generator_total_combinations() {
        let gen = TemplateGenerator::new()
            .with_template(Template::new("{a}").with_slot("a", &["1", "2"]))
            .with_template(Template::new("{b}").with_slot("b", &["x", "y", "z"]));

        assert_eq!(gen.total_combinations(), 5); // 2 + 3
    }

    #[test]
    fn test_template_generator_generate_samples() {
        let gen = TemplateGenerator::new()
            .with_template(Template::new("git {cmd}").with_slot("cmd", &["status", "log", "diff"]));

        let samples = gen.generate_samples(5, 42);
        assert_eq!(samples.len(), 5);

        for sample in &samples {
            assert!(sample.starts_with("git "));
        }
    }

    #[test]
    fn test_template_generator_generate_samples_empty() {
        let gen = TemplateGenerator::new();
        let samples = gen.generate_samples(5, 42);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_template_generator_deterministic() {
        let gen = TemplateGenerator::new()
            .with_template(Template::new("cmd {arg}").with_slot("arg", &["a", "b", "c", "d", "e"]));

        let samples1 = gen.generate_samples(10, 42);
        let samples2 = gen.generate_samples(10, 42);

        assert_eq!(samples1, samples2);
    }

    #[test]
    fn test_template_generator_different_seeds() {
        let gen = TemplateGenerator::new().with_template(
            Template::new("{a} {b}")
                .with_slot("a", &["x", "y", "z"])
                .with_slot("b", &["1", "2", "3"]),
        );

        let samples1 = gen.generate_samples(10, 42);
        let samples2 = gen.generate_samples(10, 123);

        // Should produce different sequences
        assert_ne!(samples1, samples2);
    }

    #[test]
    fn test_template_generator_unique_outputs() {
        let config = TemplateConfig::new().with_unique_outputs(true);
        let gen = TemplateGenerator::with_config(config)
            .with_template(Template::new("{a}").with_slot("a", &["x", "y", "z", "w", "v"]));

        let samples = gen.generate_samples(5, 42);

        // All should be unique
        let unique: std::collections::HashSet<_> = samples.iter().collect();
        assert_eq!(unique.len(), samples.len());
    }

    #[test]
    fn test_template_generator_shell_commands() {
        let gen = TemplateGenerator::shell_commands();
        assert!(!gen.templates().is_empty());

        let samples = gen.generate_samples(20, 42);
        assert_eq!(samples.len(), 20);

        // Should have variety of commands
        let has_git = samples.iter().any(|s| s.starts_with("git"));
        let has_cargo = samples.iter().any(|s| s.starts_with("cargo"));

        assert!(has_git || has_cargo);
    }

    // ============================================================================
    // EXTREME TDD: SyntheticGenerator Trait Tests
    // ============================================================================

    #[test]
    fn test_template_synthetic_generator_trait() {
        let gen = TemplateGenerator::shell_commands();
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let seeds = vec!["git status".to_string()];
        let result = gen.generate(&seeds, &config);

        assert!(result.is_ok());
        assert!(!result.expect("should succeed").is_empty());
    }

    #[test]
    fn test_template_quality_score() {
        let gen = TemplateGenerator::new();

        let score = gen.quality_score(&"git status".to_string(), &String::new());
        assert!(score > 0.5);

        let score = gen.quality_score(&String::new(), &String::new());
        assert!((score - 0.0).abs() < f32::EPSILON);

        let score = gen.quality_score(&"ab".to_string(), &String::new());
        assert!(score < 0.5);
    }

    #[test]
    fn test_template_diversity_score() {
        let gen = TemplateGenerator::new();

        // All unique
        let batch = vec![
            "git status".to_string(),
            "cargo build".to_string(),
            "npm install".to_string(),
        ];
        let diversity = gen.diversity_score(&batch);
        assert!((diversity - 1.0).abs() < f32::EPSILON);

        // Some duplicates
        let batch = vec![
            "git status".to_string(),
            "git status".to_string(),
            "cargo build".to_string(),
        ];
        let diversity = gen.diversity_score(&batch);
        assert!(diversity < 1.0);
        assert!(diversity > 0.5);
    }
}
