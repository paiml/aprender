//! Core CLI types - Cli, Command, and basic argument structs

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use super::extended::{
    AuditArgs, BenchArgs, CompletionArgs, ExperimentsArgs, FinetuneArgs, InspectArgs, MonitorArgs,
    PublishArgs,
};
use super::init::InitArgs;
use super::quant_merge::{MergeArgs, QuantizeArgs};
use super::research::ResearchArgs;
use super::types::OutputFormat;

/// Entrenar: Training & Optimization Library
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(name = "entrenar")]
#[command(author = "PAIML")]
#[command(version)]
#[command(
    about = "Training & Optimization Library with autograd, LoRA, quantization, and model merging"
)]
pub struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Command,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    pub quiet: bool,
}

/// Available commands
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum Command {
    /// Train a model from YAML configuration
    Train(TrainArgs),

    /// Validate a configuration file without training
    Validate(ValidateArgs),

    /// Display information about a configuration
    Info(InfoArgs),

    /// Initialize a new YAML Mode training manifest
    Init(InitArgs),

    /// Quantize a model
    Quantize(QuantizeArgs),

    /// Merge multiple models
    Merge(MergeArgs),

    /// Academic research artifacts and workflows
    Research(ResearchArgs),

    /// Generate shell completions
    Completion(CompletionArgs),

    /// Run inference benchmarks
    Bench(BenchArgs),

    /// Inspect data or model statistics
    Inspect(InspectArgs),

    /// Audit model for bias and fairness
    Audit(AuditArgs),

    /// Monitor model for drift
    Monitor(MonitorArgs),

    /// Publish a trained model to HuggingFace Hub
    Publish(PublishArgs),

    /// Fine-tune a classification model (plan/apply workflow)
    Finetune(FinetuneArgs),

    /// Query the experiment store (runs, metrics, artifacts)
    Experiments(ExperimentsArgs),
}

/// Arguments for the train command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct TrainArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Override output directory
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Resume training from checkpoint
    #[arg(short, long)]
    pub resume: Option<PathBuf>,

    /// Override number of epochs
    #[arg(short, long)]
    pub epochs: Option<usize>,

    /// Override batch size
    #[arg(short, long)]
    pub batch_size: Option<usize>,

    /// Override learning rate
    #[arg(short, long)]
    pub lr: Option<f32>,

    /// Dry run (validate config but don't train)
    #[arg(long)]
    pub dry_run: bool,

    /// Save checkpoint every N steps
    #[arg(long)]
    pub save_every: Option<usize>,

    /// Log metrics every N steps
    #[arg(long)]
    pub log_every: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Arguments for the validate command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ValidateArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Show detailed validation report
    #[arg(short, long)]
    pub detailed: bool,
}

/// Arguments for the info command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct InfoArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Output format (text, json, yaml)
    #[arg(short, long, default_value = "text")]
    pub format: OutputFormat,
}

/// Parse CLI arguments from a string slice (for testing)
pub fn parse_args<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)
}

/// Apply command-line overrides to a TrainSpec
pub fn apply_overrides(spec: &mut crate::config::TrainSpec, args: &TrainArgs) {
    if let Some(output_dir) = &args.output_dir {
        spec.training.output_dir = output_dir.clone();
    }
    if let Some(epochs) = args.epochs {
        spec.training.epochs = epochs;
    }
    if let Some(batch_size) = args.batch_size {
        spec.data.batch_size = batch_size;
    }
    if let Some(lr) = args.lr {
        spec.optimizer.lr = lr;
    }
    if let Some(save_every) = args.save_every {
        spec.training.save_interval = save_every;
    }
    // Note: log_every and seed are CLI-only options not persisted in config
}
