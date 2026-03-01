
/// Training pipeline subcommands (forjar-style plan/apply).
///
/// Thin CLI wrappers around entrenar's training plan/apply infrastructure.
#[derive(Subcommand, Debug)]
pub enum TrainCommands {
    /// Generate a training plan without touching the GPU.
    ///
    /// Validates data quality, checks model compatibility, builds HPO search space,
    /// estimates resource usage, and runs pre-flight checks. Outputs a serializable
    /// plan manifest (text, JSON, or YAML).
    ///
    /// Analogous to `forjar plan` — shows what will happen before committing GPU time.
    Plan {
        /// Path to training data (JSONL) — required for --task classify
        #[arg(long, value_name = "FILE")]
        data: Option<PathBuf>,
        /// Model size: "0.5B", "9B", "7B", "13B"
        #[arg(long, default_value = "0.5B")]
        model_size: String,
        /// Path to model weights directory
        #[arg(long, value_name = "DIR")]
        model_path: Option<PathBuf>,
        /// Number of output classes
        #[arg(long, default_value = "5")]
        num_classes: usize,
        /// Task type: classify, pretrain
        #[arg(long, default_value = "classify")]
        task: String,
        /// YAML training config (for --task pretrain)
        #[arg(long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Output directory for checkpoints
        #[arg(short, long, default_value = "/tmp/training-output")]
        output: PathBuf,
        /// HPO strategy: tpe, grid, random, manual
        #[arg(long, default_value = "tpe")]
        strategy: String,
        /// HPO budget (number of trials)
        #[arg(long, default_value = "20")]
        budget: usize,
        /// Scout mode: 1 epoch per trial for fast exploration
        #[arg(long)]
        scout: bool,
        /// Maximum epochs per trial
        #[arg(long, default_value = "3")]
        max_epochs: usize,
        /// Manual learning rate (only used with --strategy manual)
        #[arg(long)]
        learning_rate: Option<f32>,
        /// Manual LoRA rank (only used with --strategy manual)
        #[arg(long)]
        lora_rank: Option<usize>,
        /// Manual batch size (only used with --strategy manual)
        #[arg(long)]
        batch_size: Option<usize>,
        /// Validation data file (JSONL)
        #[arg(long, value_name = "FILE")]
        val_data: Option<PathBuf>,
        /// Test data file (JSONL)
        #[arg(long, value_name = "FILE")]
        test_data: Option<PathBuf>,
        /// Output format: text, json, yaml
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Execute a training plan (allocate GPU, run trials).
    ///
    /// Reads a previously generated plan (YAML/JSON) and executes it:
    /// - Manual strategy: single training run with specified hyperparameters
    /// - HPO strategy: multiple trials with automatic hyperparameter tuning
    ///
    /// Analogous to `forjar apply` — commits resources and executes the plan.
    Apply {
        /// Path to a saved plan file (YAML or JSON from `apr train plan`)
        #[arg(long, value_name = "FILE")]
        plan: Option<PathBuf>,

        /// YAML training config (for --task pretrain)
        #[arg(long, value_name = "FILE")]
        config: Option<PathBuf>,

        /// Task type: classify, pretrain
        #[arg(long, default_value = "classify")]
        task: String,

        // ── Inline plan params (used when no --plan file is given) ─────
        /// Path to training data (JSONL)
        #[arg(long, value_name = "FILE")]
        data: Option<PathBuf>,
        /// Model size: "0.5B", "9B", "7B", "13B"
        #[arg(long, default_value = "0.5B")]
        model_size: String,
        /// Path to model weights directory
        #[arg(long, value_name = "DIR")]
        model_path: Option<PathBuf>,
        /// Number of output classes
        #[arg(long, default_value = "5")]
        num_classes: usize,
        /// Output directory for checkpoints and leaderboard
        #[arg(short, long, default_value = "/tmp/training-output")]
        output: PathBuf,
        /// HPO strategy: tpe, grid, random, manual
        #[arg(long, default_value = "tpe")]
        strategy: String,
        /// HPO budget (number of trials)
        #[arg(long, default_value = "20")]
        budget: usize,
        /// Scout mode: 1 epoch per trial
        #[arg(long)]
        scout: bool,
        /// Maximum epochs per trial
        #[arg(long, default_value = "3")]
        max_epochs: usize,
        /// Manual learning rate (only used with --strategy manual)
        #[arg(long)]
        learning_rate: Option<f32>,
        /// Manual LoRA rank (only used with --strategy manual)
        #[arg(long)]
        lora_rank: Option<usize>,
        /// Manual batch size (only used with --strategy manual)
        #[arg(long)]
        batch_size: Option<usize>,
    },
}
