
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

        // ── Distributed training params (tickets #131-#140, aprender #393) ──
        /// Enable distributed data-parallel training
        #[arg(long)]
        distributed: bool,
        /// Total number of workers (default: auto-detect GPUs)
        #[arg(long, value_name = "N")]
        world_size: Option<usize>,
        /// This worker's global rank (default: 0 = coordinator)
        #[arg(long, value_name = "N")]
        rank: Option<usize>,
        /// Coordinator address for distributed training (default: 0.0.0.0:9000)
        #[arg(long, value_name = "HOST:PORT")]
        coordinator_addr: Option<String>,

        // ── Reproducibility params (R-084 C-DETERM-001) ──
        /// Enable bitwise deterministic training (CUBLAS_WORKSPACE_CONFIG, cuDNN deterministic)
        #[arg(long)]
        deterministic: bool,
        /// Random seed for reproducibility (default: from YAML or 42)
        #[arg(long, value_name = "N")]
        seed: Option<u64>,
    },

    /// Watch a training run with automatic restart on crash and hang detection.
    ///
    /// Monitors a running or to-be-started training process:
    /// - Detects crashes (SIGABRT, SIGSEGV, OOM) and restarts with backoff
    /// - Detects hangs via heartbeat/training_state.json staleness
    /// - Captures GPU state and crash diagnostics
    /// - Auto-enables CUDA_LAUNCH_BLOCKING on async crash pattern
    ///
    /// Sovereign Rust replacement for train-guard.sh.
    Watch {
        /// YAML training config to run and watch
        #[arg(long, value_name = "FILE")]
        config: PathBuf,

        /// Maximum number of restart attempts
        #[arg(long, default_value = "5")]
        max_restarts: usize,

        /// Heartbeat staleness threshold in seconds
        #[arg(long, default_value = "300")]
        heartbeat_timeout: u64,

        /// Initial backoff delay in seconds
        #[arg(long, default_value = "30")]
        backoff_initial: u64,

        /// Maximum backoff delay in seconds
        #[arg(long, default_value = "600")]
        backoff_max: u64,
    },

    /// Generate hyperparameter sweep configs from a base YAML.
    ///
    /// Creates N training configs with varied hyperparameters using grid
    /// or random search. Each config is a complete YAML that can be
    /// passed to `apr train apply --task pretrain --config <file>`.
    ///
    /// Sovereign Rust replacement for hyperparam-sweep.py.
    Sweep {
        /// Base YAML training config to sweep from
        #[arg(long, value_name = "FILE")]
        config: PathBuf,

        /// Search strategy: grid or random
        #[arg(long, default_value = "random")]
        strategy: String,

        /// Number of configs to generate (random) or max combinations (grid)
        #[arg(long, default_value = "10")]
        num_configs: usize,

        /// Output directory for generated configs
        #[arg(long, default_value = "sweeps/")]
        output_dir: PathBuf,

        /// Seed for random search reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Archive a checkpoint into a release bundle.
    ///
    /// Packages model weights, config, training state, and metadata
    /// into a self-contained directory with integrity manifest.
    Archive {
        /// Path to checkpoint directory
        #[arg(value_name = "CHECKPOINT_DIR")]
        checkpoint_dir: PathBuf,

        /// Output archive directory
        #[arg(short, long, value_name = "DIR")]
        output: PathBuf,

        /// Release version tag (e.g., "v1.0")
        #[arg(long)]
        version: Option<String>,

        /// Release notes
        #[arg(long)]
        notes: Option<String>,
    },

    /// Submit multi-adapter training jobs to a cluster (GPU-SHARE Phase 3).
    ///
    /// Reads a cluster.yaml config, places adapter jobs across nodes using
    /// the greedy placement algorithm, and generates launch commands.
    Submit {
        /// Path to cluster config YAML
        #[arg(long, value_name = "FILE")]
        cluster: PathBuf,

        /// Model checkpoint path (.apr)
        #[arg(long, value_name = "FILE")]
        model: PathBuf,

        /// Adapter specs: DATA:CHECKPOINT pairs (one per adapter)
        #[arg(long = "adapter", value_name = "DATA:CHECKPOINT")]
        adapters: Vec<String>,

        /// LoRA rank
        #[arg(long, default_value = "16")]
        rank: u32,

        /// Number of training epochs
        #[arg(long, default_value = "3")]
        epochs: u32,

        /// Estimated VRAM budget per adapter (MB)
        #[arg(long, default_value = "6000")]
        budget_mb: u64,

        /// Dry run: show placement and commands without executing
        #[arg(long)]
        dry_run: bool,
    },

    /// Show cluster status: nodes, GPUs, adapter capacity (GPU-SHARE Phase 3).
    ///
    /// Reads a cluster.yaml config and displays node health, VRAM availability,
    /// and adapter placement capacity.
    ClusterStatus {
        /// Path to cluster config YAML
        #[arg(long, value_name = "FILE")]
        cluster: PathBuf,
    },
}
