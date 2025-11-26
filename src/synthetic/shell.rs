//! Shell Autocomplete Synthetic Data Generator.
//!
//! Generates synthetic training data for shell command autocomplete SLMs.
//! Uses template substitution, argument permutation, and context variation
//! to augment limited shell command corpora.
//!
//! # References
//!
//! - Jia & Liang (2016). Data Recombination for Neural Semantic Parsing. ACL.
//! - Section 4 of AutoML with Synthetic Data Specification.

use std::collections::{HashMap, HashSet};

use super::{SyntheticConfig, SyntheticGenerator};
use crate::error::Result;

// ============================================================================
// ShellSample: Command with context
// ============================================================================

/// A shell command sample with execution context.
///
/// Represents a single training example for shell autocomplete, including
/// command history, current input prefix, and working directory context.
///
/// # Example
///
/// ```
/// use aprender::synthetic::shell::ShellSample;
///
/// let sample = ShellSample::new("git st", "git status")
///     .with_history(vec!["cd project".to_string()])
///     .with_cwd("/home/user/project");
///
/// assert_eq!(sample.prefix(), "git st");
/// assert_eq!(sample.completion(), "git status");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ShellSample {
    /// Previous commands in the session (context).
    history: Vec<String>,
    /// Current partial input (prefix to complete).
    prefix: String,
    /// Completed command (label).
    completion: String,
    /// Working directory context.
    cwd: String,
}

impl ShellSample {
    /// Create a new shell sample.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Current partial input
    /// * `completion` - Full completed command
    #[must_use]
    pub fn new(prefix: impl Into<String>, completion: impl Into<String>) -> Self {
        Self {
            history: Vec::new(),
            prefix: prefix.into(),
            completion: completion.into(),
            cwd: String::new(),
        }
    }

    /// Add command history context.
    #[must_use]
    pub fn with_history(mut self, history: Vec<String>) -> Self {
        self.history = history;
        self
    }

    /// Set working directory context.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = cwd.into();
        self
    }

    /// Get the prefix.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Get the completion.
    #[must_use]
    pub fn completion(&self) -> &str {
        &self.completion
    }

    /// Get the history.
    #[must_use]
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Get the working directory.
    #[must_use]
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// Extract the command name (first token).
    #[must_use]
    pub fn command_name(&self) -> Option<&str> {
        self.completion.split_whitespace().next()
    }

    /// Extract arguments (tokens after command name).
    #[must_use]
    pub fn arguments(&self) -> Vec<&str> {
        self.completion.split_whitespace().skip(1).collect()
    }

    /// Check if the completion starts with the prefix.
    #[must_use]
    pub fn is_valid_completion(&self) -> bool {
        self.completion.starts_with(&self.prefix)
    }
}

// ============================================================================
// ShellGrammar: Command validation
// ============================================================================

/// Shell command grammar for validation.
///
/// Validates that generated commands follow expected patterns
/// and use known command names, subcommands, and options.
///
/// # Example
///
/// ```
/// use aprender::synthetic::shell::ShellGrammar;
///
/// let grammar = ShellGrammar::common_commands();
///
/// assert!(grammar.is_valid_command("git status"));
/// assert!(grammar.is_valid_command("cargo build --release"));
/// assert!(!grammar.is_valid_command(""));
/// ```
#[derive(Debug, Clone)]
pub struct ShellGrammar {
    /// Known command names.
    commands: HashSet<String>,
    /// Subcommands for each command: command -> [subcommands].
    subcommands: HashMap<String, HashSet<String>>,
    /// Common options that apply to many commands.
    common_options: HashSet<String>,
}

impl ShellGrammar {
    /// Create an empty grammar.
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: HashSet::new(),
            subcommands: HashMap::new(),
            common_options: HashSet::new(),
        }
    }

    /// Create a grammar with common shell commands.
    #[must_use]
    pub fn common_commands() -> Self {
        let mut grammar = Self::new();

        // Git commands
        grammar.add_command("git");
        grammar.add_subcommands(
            "git",
            &[
                "status", "commit", "push", "pull", "checkout", "branch", "merge", "rebase", "log",
                "diff", "add", "reset", "stash", "fetch", "clone", "init",
            ],
        );

        // Cargo commands
        grammar.add_command("cargo");
        grammar.add_subcommands(
            "cargo",
            &[
                "build", "run", "test", "check", "clippy", "fmt", "doc", "publish", "new", "init",
                "add", "remove", "update", "bench", "clean",
            ],
        );

        // npm commands
        grammar.add_command("npm");
        grammar.add_subcommands(
            "npm",
            &[
                "install", "run", "test", "start", "build", "publish", "init", "update", "audit",
                "ci", "pack",
            ],
        );

        // Docker commands
        grammar.add_command("docker");
        grammar.add_subcommands(
            "docker",
            &[
                "run", "build", "push", "pull", "ps", "images", "exec", "stop", "start", "rm",
                "rmi", "logs", "compose",
            ],
        );

        // Common Unix commands
        for cmd in &[
            "ls", "cd", "cp", "mv", "rm", "mkdir", "rmdir", "cat", "grep", "find", "chmod",
            "chown", "curl", "wget", "ssh", "scp", "tar", "zip", "unzip", "make", "python", "node",
        ] {
            grammar.add_command(cmd);
        }

        // Common options
        grammar.add_common_options(&[
            "-h",
            "--help",
            "-v",
            "--version",
            "-q",
            "--quiet",
            "-f",
            "--force",
            "-r",
            "--recursive",
            "-n",
            "--dry-run",
        ]);

        grammar
    }

    /// Add a known command.
    pub fn add_command(&mut self, command: &str) {
        self.commands.insert(command.to_string());
    }

    /// Add subcommands for a command.
    pub fn add_subcommands(&mut self, command: &str, subs: &[&str]) {
        let entry = self.subcommands.entry(command.to_string()).or_default();
        for sub in subs {
            entry.insert((*sub).to_string());
        }
    }

    /// Add common options.
    pub fn add_common_options(&mut self, options: &[&str]) {
        for opt in options {
            self.common_options.insert((*opt).to_string());
        }
    }

    /// Check if a command string is valid.
    ///
    /// A command is valid if:
    /// - It's non-empty
    /// - The first token is a known command OR starts with known command
    /// - If it has subcommands defined, the second token should be a known subcommand
    #[must_use]
    pub fn is_valid_command(&self, command: &str) -> bool {
        let tokens: Vec<&str> = command.split_whitespace().collect();
        if tokens.is_empty() {
            return false;
        }

        let cmd_name = tokens[0];

        // Check if command is known
        if !self.commands.contains(cmd_name) {
            return false;
        }

        // If we have subcommand definitions and there's a second token
        if let Some(subs) = self.subcommands.get(cmd_name) {
            if tokens.len() > 1 {
                let second = tokens[1];
                // Allow if it's a known subcommand or starts with '-' (option)
                if !subs.contains(second) && !second.starts_with('-') {
                    return false;
                }
            }
        }

        true
    }

    /// Check if a token is a known option.
    #[must_use]
    pub fn is_valid_option(&self, option: &str) -> bool {
        option.starts_with('-') && (self.common_options.contains(option) || option.len() <= 20)
    }

    /// Get all known commands.
    #[must_use]
    pub fn commands(&self) -> &HashSet<String> {
        &self.commands
    }

    /// Get subcommands for a command.
    #[must_use]
    pub fn get_subcommands(&self, command: &str) -> Option<&HashSet<String>> {
        self.subcommands.get(command)
    }
}

impl Default for ShellGrammar {
    fn default() -> Self {
        Self::common_commands()
    }
}

// ============================================================================
// ShellSyntheticGenerator: Generates synthetic shell samples
// ============================================================================

/// Configuration for shell synthetic generation.
#[derive(Debug, Clone)]
pub struct ShellGeneratorConfig {
    /// Enable template-based generation.
    pub enable_template: bool,
    /// Enable argument permutation.
    pub enable_permutation: bool,
    /// Enable context variation.
    pub enable_context_variation: bool,
    /// Maximum arguments to permute.
    pub max_permute_args: usize,
}

impl Default for ShellGeneratorConfig {
    fn default() -> Self {
        Self {
            enable_template: true,
            enable_permutation: true,
            enable_context_variation: true,
            max_permute_args: 3,
        }
    }
}

/// Synthetic data generator for shell autocomplete.
///
/// Implements three generation strategies:
/// 1. Template substitution: Replace arguments with variants
/// 2. Argument permutation: Reorder and add/remove options
/// 3. Context variation: Modify history and cwd
///
/// # Example
///
/// ```
/// use aprender::synthetic::shell::{ShellSyntheticGenerator, ShellSample};
/// use aprender::synthetic::{SyntheticGenerator, SyntheticConfig};
///
/// let gen = ShellSyntheticGenerator::new();
/// let seeds = vec![
///     ShellSample::new("git st", "git status"),
///     ShellSample::new("cargo b", "cargo build"),
/// ];
/// let config = SyntheticConfig::default().with_augmentation_ratio(1.0);
///
/// let synthetic = gen.generate(&seeds, &config).expect("generation failed");
/// assert!(!synthetic.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ShellSyntheticGenerator {
    /// Grammar for command validation.
    grammar: ShellGrammar,
    /// Generator configuration.
    config: ShellGeneratorConfig,
    /// Argument substitutions: arg -> [variants].
    substitutions: HashMap<String, Vec<String>>,
}

impl ShellSyntheticGenerator {
    /// Create a new generator with default settings.
    #[must_use]
    pub fn new() -> Self {
        let mut gen = Self {
            grammar: ShellGrammar::common_commands(),
            config: ShellGeneratorConfig::default(),
            substitutions: HashMap::new(),
        };
        gen.init_default_substitutions();
        gen
    }

    /// Create with custom grammar.
    #[must_use]
    pub fn with_grammar(mut self, grammar: ShellGrammar) -> Self {
        self.grammar = grammar;
        self
    }

    /// Create with custom config.
    #[must_use]
    pub fn with_config(mut self, config: ShellGeneratorConfig) -> Self {
        self.config = config;
        self
    }

    /// Add argument substitutions.
    pub fn add_substitution(&mut self, arg: &str, variants: &[&str]) {
        self.substitutions.insert(
            arg.to_string(),
            variants.iter().map(|s| (*s).to_string()).collect(),
        );
    }

    /// Initialize default substitutions for common arguments.
    fn init_default_substitutions(&mut self) {
        // Git branches
        self.add_substitution("main", &["master", "develop", "feature"]);
        self.add_substitution("master", &["main", "develop", "feature"]);

        // Build targets
        self.add_substitution("--release", &["--debug", ""]);
        self.add_substitution("--debug", &["--release", ""]);

        // Common paths
        self.add_substitution(".", &["./", "../", "src/"]);
        self.add_substitution("src/", &["lib/", "tests/", "examples/"]);

        // Commit message patterns
        self.add_substitution("fix:", &["feat:", "docs:", "refactor:", "test:"]);
        self.add_substitution("feat:", &["fix:", "docs:", "refactor:", "chore:"]);
    }

    /// Generate via template substitution.
    fn generate_from_template(&self, seed: &ShellSample, rng_seed: u64) -> Vec<ShellSample> {
        if !self.config.enable_template {
            return Vec::new();
        }

        let mut results = Vec::new();
        let args = seed.arguments();

        for (i, arg) in args.iter().enumerate() {
            if let Some(variants) = self.substitutions.get(*arg) {
                for (vi, variant) in variants.iter().enumerate() {
                    // Use deterministic selection based on rng_seed
                    if (rng_seed + i as u64 + vi as u64) % 3 == 0 {
                        let mut new_args = args.clone();
                        if variant.is_empty() {
                            new_args.remove(i);
                        } else {
                            new_args[i] = variant.as_str();
                        }

                        if let Some(cmd) = seed.command_name() {
                            let new_completion = if new_args.is_empty() {
                                cmd.to_string()
                            } else {
                                format!("{} {}", cmd, new_args.join(" "))
                            };

                            // Generate matching prefix
                            let prefix_len = seed.prefix().len().min(new_completion.len());
                            let new_prefix = &new_completion[..prefix_len];

                            results.push(
                                ShellSample::new(new_prefix, &new_completion)
                                    .with_history(seed.history().to_vec())
                                    .with_cwd(seed.cwd()),
                            );
                        }
                    }
                }
            }
        }

        results
    }

    /// Generate via argument permutation.
    fn permute_arguments(&self, seed: &ShellSample, rng_seed: u64) -> Vec<ShellSample> {
        if !self.config.enable_permutation {
            return Vec::new();
        }

        let mut results = Vec::new();
        let args = seed.arguments();

        if args.len() < 2 || args.len() > self.config.max_permute_args {
            return results;
        }

        // Generate one permutation by swapping first two args
        let mut new_args: Vec<&str> = args.clone();
        new_args.swap(0, 1);

        if let Some(cmd) = seed.command_name() {
            let new_completion = format!("{} {}", cmd, new_args.join(" "));

            // Only use if different from original
            if new_completion != seed.completion() {
                // Add common option based on rng
                let final_completion = if rng_seed % 2 == 0 {
                    format!("{new_completion} --verbose")
                } else {
                    new_completion
                };

                let prefix_len = seed.prefix().len().min(final_completion.len());
                let new_prefix = final_completion[..prefix_len].to_string();

                results.push(
                    ShellSample::new(new_prefix, &final_completion)
                        .with_history(seed.history().to_vec())
                        .with_cwd(seed.cwd()),
                );
            }
        }

        results
    }

    /// Generate via context variation.
    fn vary_context(&self, seed: &ShellSample, rng_seed: u64) -> Vec<ShellSample> {
        if !self.config.enable_context_variation {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Vary working directory
        let cwd_variants = ["/home/user/project", "/tmp", "./src", "/var/log"];
        let cwd_idx = (rng_seed as usize) % cwd_variants.len();
        let new_cwd = cwd_variants[cwd_idx];

        if new_cwd != seed.cwd() {
            results.push(
                ShellSample::new(seed.prefix(), seed.completion())
                    .with_history(seed.history().to_vec())
                    .with_cwd(new_cwd),
            );
        }

        // Vary history
        let history_additions = ["ls -la", "pwd", "cd ..", "clear"];
        let hist_idx = (rng_seed as usize + 1) % history_additions.len();

        let mut new_history = seed.history().to_vec();
        new_history.push(history_additions[hist_idx].to_string());

        results.push(
            ShellSample::new(seed.prefix(), seed.completion())
                .with_history(new_history)
                .with_cwd(seed.cwd()),
        );

        results
    }

    /// Calculate semantic similarity between two commands (Jaccard similarity).
    fn semantic_similarity(a: &str, b: &str) -> f32 {
        let tokens_a: HashSet<&str> = a.split_whitespace().collect();
        let tokens_b: HashSet<&str> = b.split_whitespace().collect();

        if tokens_a.is_empty() && tokens_b.is_empty() {
            return 1.0;
        }
        if tokens_a.is_empty() || tokens_b.is_empty() {
            return 0.0;
        }

        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();

        intersection as f32 / union as f32
    }

    /// Calculate context coherence score.
    fn context_coherence(sample: &ShellSample) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Bonus for having history
        if !sample.history().is_empty() {
            score += 0.2;
        }

        // Bonus for having cwd
        if !sample.cwd().is_empty() {
            score += 0.2;
        }

        // Bonus for valid completion relationship
        if sample.is_valid_completion() {
            score += 0.1;
        }

        score.min(1.0)
    }
}

impl Default for ShellSyntheticGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticGenerator for ShellSyntheticGenerator {
    type Input = ShellSample;
    type Output = ShellSample;

    fn generate(
        &self,
        seeds: &[ShellSample],
        config: &SyntheticConfig,
    ) -> Result<Vec<ShellSample>> {
        let target = config.target_count(seeds.len());
        let mut results = Vec::with_capacity(target);
        let mut seen: HashSet<String> = HashSet::new();

        for (seed_idx, seed) in seeds.iter().enumerate() {
            let rng_seed = config.seed.wrapping_add(seed_idx as u64);

            // Strategy 1: Template substitution
            let template_samples = self.generate_from_template(seed, rng_seed);

            // Strategy 2: Argument permutation
            let permuted = self.permute_arguments(seed, rng_seed);

            // Strategy 3: Context variation
            let context_varied = self.vary_context(seed, rng_seed);

            // Combine and filter
            for sample in template_samples
                .into_iter()
                .chain(permuted)
                .chain(context_varied)
            {
                // Deduplicate
                if seen.contains(sample.completion()) {
                    continue;
                }

                // Quality filter
                let quality = self.quality_score(&sample, seed);
                if config.meets_quality(quality) {
                    // Grammar validation
                    if self.grammar.is_valid_command(sample.completion()) {
                        seen.insert(sample.completion().to_string());
                        results.push(sample);

                        if results.len() >= target {
                            return Ok(results);
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &ShellSample, seed: &ShellSample) -> f32 {
        // Semantic similarity (Jaccard)
        let semantic_sim = Self::semantic_similarity(generated.completion(), seed.completion());

        // Grammar validity (binary)
        let grammar_valid = if self.grammar.is_valid_command(generated.completion()) {
            1.0
        } else {
            0.0
        };

        // Context coherence
        let context_coherent = Self::context_coherence(generated);

        // Weighted combination (spec: 0.4 semantic + 0.4 grammar + 0.2 context)
        0.4 * semantic_sim + 0.4 * grammar_valid + 0.2 * context_coherent
    }

    fn diversity_score(&self, batch: &[ShellSample]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Count unique command patterns
        let unique_commands: HashSet<_> = batch.iter().filter_map(|s| s.command_name()).collect();

        let unique_completions: HashSet<_> = batch.iter().map(ShellSample::completion).collect();

        let cmd_diversity = unique_commands.len() as f32 / batch.len() as f32;
        let completion_diversity = unique_completions.len() as f32 / batch.len() as f32;

        (cmd_diversity + completion_diversity) / 2.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ShellSample Tests
    // ========================================================================

    #[test]
    fn test_shell_sample_new() {
        let sample = ShellSample::new("git st", "git status");
        assert_eq!(sample.prefix(), "git st");
        assert_eq!(sample.completion(), "git status");
        assert!(sample.history().is_empty());
        assert!(sample.cwd().is_empty());
    }

    #[test]
    fn test_shell_sample_with_history() {
        let sample =
            ShellSample::new("cargo b", "cargo build").with_history(vec!["cd project".to_string()]);
        assert_eq!(sample.history().len(), 1);
        assert_eq!(sample.history()[0], "cd project");
    }

    #[test]
    fn test_shell_sample_with_cwd() {
        let sample = ShellSample::new("ls", "ls -la").with_cwd("/home/user");
        assert_eq!(sample.cwd(), "/home/user");
    }

    #[test]
    fn test_shell_sample_command_name() {
        let sample = ShellSample::new("git", "git status --short");
        assert_eq!(sample.command_name(), Some("git"));

        let empty = ShellSample::new("", "");
        assert_eq!(empty.command_name(), None);
    }

    #[test]
    fn test_shell_sample_arguments() {
        let sample = ShellSample::new("cargo", "cargo build --release --target wasm32");
        let args = sample.arguments();
        assert_eq!(args, vec!["build", "--release", "--target", "wasm32"]);
    }

    #[test]
    fn test_shell_sample_is_valid_completion() {
        let valid = ShellSample::new("git st", "git status");
        assert!(valid.is_valid_completion());

        let invalid = ShellSample::new("cargo", "git status");
        assert!(!invalid.is_valid_completion());
    }

    #[test]
    fn test_shell_sample_clone() {
        let sample = ShellSample::new("ls", "ls -la")
            .with_history(vec!["pwd".to_string()])
            .with_cwd("/tmp");
        let cloned = sample.clone();
        assert_eq!(sample, cloned);
    }

    #[test]
    fn test_shell_sample_debug() {
        let sample = ShellSample::new("test", "test command");
        let debug = format!("{sample:?}");
        assert!(debug.contains("ShellSample"));
        assert!(debug.contains("prefix"));
    }

    // ========================================================================
    // ShellGrammar Tests
    // ========================================================================

    #[test]
    fn test_grammar_new_empty() {
        let grammar = ShellGrammar::new();
        assert!(grammar.commands().is_empty());
    }

    #[test]
    fn test_grammar_add_command() {
        let mut grammar = ShellGrammar::new();
        grammar.add_command("mycommand");
        assert!(grammar.commands().contains("mycommand"));
    }

    #[test]
    fn test_grammar_add_subcommands() {
        let mut grammar = ShellGrammar::new();
        grammar.add_command("git");
        grammar.add_subcommands("git", &["status", "commit"]);

        let subs = grammar.get_subcommands("git").expect("should have subs");
        assert!(subs.contains("status"));
        assert!(subs.contains("commit"));
    }

    #[test]
    fn test_grammar_common_commands() {
        let grammar = ShellGrammar::common_commands();

        // Check git
        assert!(grammar.commands().contains("git"));
        let git_subs = grammar
            .get_subcommands("git")
            .expect("git should have subs");
        assert!(git_subs.contains("status"));
        assert!(git_subs.contains("commit"));

        // Check cargo
        assert!(grammar.commands().contains("cargo"));
        let cargo_subs = grammar
            .get_subcommands("cargo")
            .expect("cargo should have subs");
        assert!(cargo_subs.contains("build"));
        assert!(cargo_subs.contains("test"));

        // Check common commands
        assert!(grammar.commands().contains("ls"));
        assert!(grammar.commands().contains("cd"));
    }

    #[test]
    fn test_grammar_is_valid_command_known() {
        let grammar = ShellGrammar::common_commands();

        assert!(grammar.is_valid_command("git status"));
        assert!(grammar.is_valid_command("cargo build"));
        assert!(grammar.is_valid_command("ls -la"));
        assert!(grammar.is_valid_command("docker run"));
    }

    #[test]
    fn test_grammar_is_valid_command_with_options() {
        let grammar = ShellGrammar::common_commands();

        assert!(grammar.is_valid_command("git -h"));
        assert!(grammar.is_valid_command("cargo --version"));
        assert!(grammar.is_valid_command("git status --short"));
    }

    #[test]
    fn test_grammar_is_valid_command_empty() {
        let grammar = ShellGrammar::common_commands();
        assert!(!grammar.is_valid_command(""));
        assert!(!grammar.is_valid_command("   "));
    }

    #[test]
    fn test_grammar_is_valid_command_unknown() {
        let grammar = ShellGrammar::common_commands();
        assert!(!grammar.is_valid_command("unknowncommand"));
        assert!(!grammar.is_valid_command("notacommand --flag"));
    }

    #[test]
    fn test_grammar_is_valid_command_bad_subcommand() {
        let grammar = ShellGrammar::common_commands();
        // "notasub" is not a known git subcommand
        assert!(!grammar.is_valid_command("git notasub"));
    }

    #[test]
    fn test_grammar_is_valid_option() {
        let grammar = ShellGrammar::common_commands();

        assert!(grammar.is_valid_option("-h"));
        assert!(grammar.is_valid_option("--help"));
        assert!(grammar.is_valid_option("--version"));
        assert!(grammar.is_valid_option("--some-flag")); // Unknown but valid format
        assert!(!grammar.is_valid_option("notanoption"));
    }

    #[test]
    fn test_grammar_default() {
        let grammar = ShellGrammar::default();
        assert!(grammar.commands().contains("git"));
    }

    #[test]
    fn test_grammar_clone() {
        let grammar = ShellGrammar::common_commands();
        let cloned = grammar.clone();
        assert_eq!(grammar.commands().len(), cloned.commands().len());
    }

    // ========================================================================
    // ShellGeneratorConfig Tests
    // ========================================================================

    #[test]
    fn test_generator_config_default() {
        let config = ShellGeneratorConfig::default();
        assert!(config.enable_template);
        assert!(config.enable_permutation);
        assert!(config.enable_context_variation);
        assert_eq!(config.max_permute_args, 3);
    }

    // ========================================================================
    // ShellSyntheticGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_new() {
        let gen = ShellSyntheticGenerator::new();
        assert!(!gen.substitutions.is_empty());
    }

    #[test]
    fn test_generator_with_grammar() {
        let mut grammar = ShellGrammar::new();
        grammar.add_command("custom");

        let gen = ShellSyntheticGenerator::new().with_grammar(grammar);
        assert!(gen.grammar.commands().contains("custom"));
    }

    #[test]
    fn test_generator_with_config() {
        let config = ShellGeneratorConfig {
            enable_template: false,
            ..Default::default()
        };
        let gen = ShellSyntheticGenerator::new().with_config(config);
        assert!(!gen.config.enable_template);
    }

    #[test]
    fn test_generator_add_substitution() {
        let mut gen = ShellSyntheticGenerator::new();
        gen.add_substitution("myarg", &["variant1", "variant2"]);

        assert!(gen.substitutions.contains_key("myarg"));
        assert_eq!(gen.substitutions["myarg"].len(), 2);
    }

    #[test]
    fn test_generator_semantic_similarity() {
        // Identical
        let sim = ShellSyntheticGenerator::semantic_similarity("git status", "git status");
        assert!((sim - 1.0).abs() < f32::EPSILON);

        // Partial overlap
        let sim = ShellSyntheticGenerator::semantic_similarity("git status", "git commit");
        assert!(sim > 0.0 && sim < 1.0);

        // No overlap
        let sim = ShellSyntheticGenerator::semantic_similarity("cargo build", "npm install");
        assert!((sim - 0.0).abs() < f32::EPSILON);

        // Empty
        let sim = ShellSyntheticGenerator::semantic_similarity("", "");
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_context_coherence() {
        // Base sample
        let base = ShellSample::new("git", "git status");
        let score = ShellSyntheticGenerator::context_coherence(&base);
        assert!(score >= 0.5);

        // With history
        let with_hist =
            ShellSample::new("git", "git status").with_history(vec!["cd repo".to_string()]);
        let score_hist = ShellSyntheticGenerator::context_coherence(&with_hist);
        assert!(score_hist > score);

        // With cwd
        let with_cwd = ShellSample::new("git", "git status").with_cwd("/home/user");
        let score_cwd = ShellSyntheticGenerator::context_coherence(&with_cwd);
        assert!(score_cwd > score);

        // Valid completion bonus
        let valid = ShellSample::new("git st", "git status");
        let invalid = ShellSample::new("cargo", "git status");
        assert!(
            ShellSyntheticGenerator::context_coherence(&valid)
                > ShellSyntheticGenerator::context_coherence(&invalid)
        );
    }

    #[test]
    fn test_generator_generate_basic() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![
            ShellSample::new("git st", "git status"),
            ShellSample::new("cargo b", "cargo build"),
        ];
        let config = SyntheticConfig::default().with_augmentation_ratio(2.0);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Should generate some samples (exact count depends on strategies)
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generator_generate_empty_seeds() {
        let gen = ShellSyntheticGenerator::new();
        let seeds: Vec<ShellSample> = vec![];
        let config = SyntheticConfig::default();

        let result = gen.generate(&seeds, &config).expect("generation failed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_generator_generate_deduplicates() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![
            ShellSample::new("git st", "git status"),
            ShellSample::new("git st", "git status"), // Duplicate
        ];
        let config = SyntheticConfig::default().with_augmentation_ratio(1.0);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Check no duplicate completions
        let completions: HashSet<_> = result.iter().map(ShellSample::completion).collect();
        assert_eq!(completions.len(), result.len());
    }

    #[test]
    fn test_generator_quality_score() {
        let gen = ShellSyntheticGenerator::new();

        let seed = ShellSample::new("git st", "git status");
        let similar = ShellSample::new("git st", "git status --short");
        let different = ShellSample::new("cargo", "cargo build");

        let score_similar = gen.quality_score(&similar, &seed);
        let score_different = gen.quality_score(&different, &seed);

        // Similar should have higher quality
        assert!(score_similar > score_different);
    }

    #[test]
    fn test_generator_quality_score_invalid_grammar() {
        let gen = ShellSyntheticGenerator::new();

        let seed = ShellSample::new("git", "git status");
        let invalid = ShellSample::new("unk", "unknowncommand");

        let score = gen.quality_score(&invalid, &seed);

        // Grammar component should be 0
        assert!(score < 0.5);
    }

    #[test]
    fn test_generator_diversity_score() {
        let gen = ShellSyntheticGenerator::new();

        // Empty batch
        assert!((gen.diversity_score(&[]) - 0.0).abs() < f32::EPSILON);

        // Single sample
        let single = vec![ShellSample::new("git", "git status")];
        assert!((gen.diversity_score(&single) - 1.0).abs() < f32::EPSILON);

        // Diverse batch
        let diverse = vec![
            ShellSample::new("git", "git status"),
            ShellSample::new("cargo", "cargo build"),
            ShellSample::new("npm", "npm install"),
        ];
        let div_score = gen.diversity_score(&diverse);
        assert!((div_score - 1.0).abs() < f32::EPSILON);

        // Homogeneous batch (same command)
        let homogeneous = vec![
            ShellSample::new("git", "git status"),
            ShellSample::new("git", "git status"),
            ShellSample::new("git", "git status"),
        ];
        let homo_score = gen.diversity_score(&homogeneous);
        assert!(homo_score < 1.0);
    }

    #[test]
    fn test_generator_default() {
        let gen = ShellSyntheticGenerator::default();
        assert!(gen.grammar.commands().contains("git"));
    }

    #[test]
    fn test_generator_template_substitution() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("cargo build", "cargo build --release");

        let results = gen.generate_from_template(&seed, 0);

        // Should generate some variants with --debug or without the flag
        // (depends on substitution rules and rng)
        // We mainly test it doesn't panic
        assert!(results.len() <= 10); // Reasonable upper bound
    }

    #[test]
    fn test_generator_permute_arguments() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("git checkout", "git checkout main develop");

        let results = gen.permute_arguments(&seed, 0);

        // Should have swapped version
        for r in &results {
            assert_ne!(r.completion(), seed.completion());
        }
    }

    #[test]
    fn test_generator_permute_single_arg() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("git", "git status");

        let results = gen.permute_arguments(&seed, 0);

        // Single arg can't be permuted
        assert!(results.is_empty());
    }

    #[test]
    fn test_generator_vary_context() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("ls", "ls -la").with_cwd("/original");

        let results = gen.vary_context(&seed, 0);

        // Should have at least history variation
        assert!(!results.is_empty());

        // Check that context was varied
        let has_different_cwd = results.iter().any(|r| r.cwd() != seed.cwd());
        let has_different_history = results.iter().any(|r| r.history() != seed.history());
        assert!(has_different_cwd || has_different_history);
    }

    #[test]
    fn test_generator_respects_quality_threshold() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![ShellSample::new("git st", "git status")];

        // High threshold - fewer results
        let high_config = SyntheticConfig::default()
            .with_augmentation_ratio(5.0)
            .with_quality_threshold(0.95);
        let high_result = gen
            .generate(&seeds, &high_config)
            .expect("generation failed");

        // Low threshold - more results
        let low_config = SyntheticConfig::default()
            .with_augmentation_ratio(5.0)
            .with_quality_threshold(0.1);
        let low_result = gen
            .generate(&seeds, &low_config)
            .expect("generation failed");

        assert!(low_result.len() >= high_result.len());
    }

    #[test]
    fn test_generator_respects_target_count() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![
            ShellSample::new("git st", "git status"),
            ShellSample::new("cargo b", "cargo build"),
            ShellSample::new("npm i", "npm install"),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(0.5) // Target: 1.5 -> 1 sample
            .with_quality_threshold(0.1);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Should stop at target (may be fewer if not enough generated)
        assert!(result.len() <= 2);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_pipeline() {
        let gen = ShellSyntheticGenerator::new();

        let seeds = vec![
            ShellSample::new("git", "git status")
                .with_history(vec!["cd repo".to_string()])
                .with_cwd("/home/user/repo"),
            ShellSample::new("cargo", "cargo build --release").with_cwd("/home/user/project"),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0)
            .with_quality_threshold(0.3);

        let synthetic = gen.generate(&seeds, &config).expect("generation failed");

        // Verify generated samples are valid
        for sample in &synthetic {
            // Should have non-empty completion
            assert!(!sample.completion().is_empty());

            // Should be grammatically valid (generator filters invalid)
            assert!(gen.grammar.is_valid_command(sample.completion()));

            // Quality score should be in valid range
            let quality = gen.quality_score(sample, &seeds[0]);
            assert!((0.0..=1.0).contains(&quality));
        }

        // Verify diversity
        let diversity = gen.diversity_score(&synthetic);
        assert!((0.0..=1.0).contains(&diversity));
    }

    #[test]
    fn test_deterministic_generation() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![ShellSample::new("git st", "git status")];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0)
            .with_seed(12345);

        let result1 = gen.generate(&seeds, &config).expect("generation failed");
        let result2 = gen.generate(&seeds, &config).expect("generation failed");

        // Same seed should produce same results
        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_eq!(r1.completion(), r2.completion());
        }
    }
}
