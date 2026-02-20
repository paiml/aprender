//! Shell Autocomplete Synthetic Data Generator.
//!
//! Generates synthetic training data for shell command autocomplete SLMs.
//! Uses template substitution, argument permutation, and context variation
//! to augment limited shell command corpora.
//!
//! # References
//!
//! - Jia & Liang (2016). Data Recombination for Neural Semantic Parsing. ACL.
//! - Section 4 of `AutoML` with Synthetic Data Specification.

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

include!("shell_generator_impl.rs");
include!("shell_part_03.rs");
