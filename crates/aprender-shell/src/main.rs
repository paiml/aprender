//! aprender-shell: AI-powered shell completion trained on your history
//!
//! Train a personalized autocomplete model on your shell history in seconds.
//! 100% local, private, and fast.

use aprender_shell::{
    filter_sensitive_suggestions, load_model_graceful, sanitize_prefix, synthetic, HistoryParser,
    MarkovModel, PagedMarkovModel, ShellError, SyntheticPipeline,
};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "aprender-shell")]
#[command(about = "AI-powered shell completion trained on your history")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model on your shell history (full retrain)
    Train {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Output model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        output: String,

        /// N-gram size (2-5)
        #[arg(short, long, default_value = "3")]
        ngram: usize,

        /// Memory limit in MB (enables paged storage for large histories)
        #[arg(long)]
        memory_limit: Option<usize>,

        /// Encrypt model with password (AES-256-GCM)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Incrementally update model with new commands (fast)
    Update {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Model file to update
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Be quiet (for hooks)
        #[arg(short, long)]
        quiet: bool,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Get completions for a prefix
    Suggest {
        /// The current command prefix (use `-- PREFIX` for prefixes starting with -)
        #[arg(allow_hyphen_values = true)]
        prefix: String,

        /// Model file to use
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Number of suggestions
        #[arg(short = 'c', visible_short_alias = 'k', long, default_value = "5")]
        count: usize,

        /// Memory limit in MB (for paged models)
        #[arg(long)]
        memory_limit: Option<usize>,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Show model statistics
    Stats {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Memory limit in MB (for paged models)
        #[arg(long)]
        memory_limit: Option<usize>,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Export model for sharing
    Export {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Output path
        output: PathBuf,
    },

    /// Import a shared model
    Import {
        /// Model file to import
        input: PathBuf,

        /// Destination
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        output: String,
    },

    /// Generate ZSH widget code
    ZshWidget,

    /// Generate Bash widget code
    BashWidget,

    /// Generate Fish shell widget code
    FishWidget,

    /// Uninstall widget from shell config
    Uninstall {
        /// Target ZSH shell (~/.zshrc)
        #[arg(long)]
        zsh: bool,

        /// Target Bash shell (~/.bashrc)
        #[arg(long)]
        bash: bool,

        /// Target Fish shell (~/.config/fish/config.fish)
        #[arg(long)]
        fish: bool,

        /// Keep model file (don't delete ~/.aprender-shell.model)
        #[arg(long)]
        keep_model: bool,

        /// Show what would be removed without doing it
        #[arg(long)]
        dry_run: bool,
    },

    /// Validate model accuracy using holdout evaluation
    Validate {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// N-gram size (2-5)
        #[arg(short, long, default_value = "3")]
        ngram: usize,

        /// Train/test split ratio (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        ratio: f32,
    },

    /// Augment training data with synthetic commands
    Augment {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Output model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        output: String,

        /// N-gram size (2-5)
        #[arg(short, long, default_value = "3")]
        ngram: usize,

        /// Augmentation ratio (synthetic/original, e.g., 0.5 = 50% more data)
        #[arg(short = 'a', long, default_value = "0.5")]
        augmentation_ratio: f32,

        /// Minimum quality threshold (0.0-1.0)
        #[arg(short, long, default_value = "0.7")]
        quality_threshold: f32,

        /// Enable diversity monitoring
        #[arg(long)]
        monitor_diversity: bool,

        /// Use CodeEDA for code-aware augmentation
        #[arg(long)]
        use_code_eda: bool,
    },

    /// Analyze command patterns and extract features
    Analyze {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Show top N commands by category
        #[arg(short, long, default_value = "10")]
        top: usize,
    },

    /// Auto-tune hyperparameters using aprender's AutoML
    Tune {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Number of trials to run
        #[arg(short, long, default_value = "10")]
        trials: usize,

        /// Train/test split ratio (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        ratio: f32,
    },

    /// Inspect model metadata (model card, version)
    Inspect {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Output format (json, yaml, huggingface)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Publish model to Hugging Face Hub (GH-100)
    Publish {
        /// Model file to publish
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Repository ID (org/name format)
        #[arg(short, long)]
        repo: String,

        /// Commit message
        #[arg(short, long, default_value = "Upload model via aprender-shell")]
        commit: String,

        /// Create repository if it doesn't exist
        #[arg(long, default_value = "true")]
        create: bool,

        /// Make repository private
        #[arg(long)]
        private: bool,
    },

    /// Stream mode: read prefixes from stdin, output suggestions (GH-95)
    ///
    /// Keeps model in memory for sub-millisecond latency.
    /// Use with shell coprocess for zero-latency suggestions.
    Stream {
        /// Model file to use
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Number of suggestions per request
        #[arg(short = 'c', long, default_value = "5")]
        count: usize,

        /// Output format (lines, json, tab)
        #[arg(short, long, default_value = "lines")]
        format: String,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Daemon mode: Unix socket server for sub-ms suggestions (GH-95)
    ///
    /// Starts a background server that keeps the model in memory.
    /// Connect via Unix socket for instant suggestions.
    Daemon {
        /// Model file to use
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Unix socket path
        #[arg(short, long, default_value = "/tmp/aprender-shell.sock")]
        socket: PathBuf,

        /// Number of suggestions per request
        #[arg(short = 'c', long, default_value = "5")]
        count: usize,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,

        /// Run in foreground (don't daemonize)
        #[arg(long)]
        foreground: bool,
    },

    /// Stop the running daemon
    DaemonStop {
        /// Unix socket path
        #[arg(short, long, default_value = "/tmp/aprender-shell.sock")]
        socket: PathBuf,
    },

    /// Check daemon status
    DaemonStatus {
        /// Unix socket path
        #[arg(short, long, default_value = "/tmp/aprender-shell.sock")]
        socket: PathBuf,
    },
}

fn expand_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

include!("cli_dispatch.rs");
include!("suggestion_helpers.rs");
include!("remove.rs");
include!("diversity.rs");
include!("main_cmd_tune_name.rs");
include!("daemon_action.rs");
include!("main_cmd_daemon.rs");
