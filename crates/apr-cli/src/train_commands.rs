
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
        /// Path to training data (JSONL). Only read by `apr finetune --task classify`.
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
        /// Task type: pretrain (causal LM). Classification fine-tuning is
        /// `apr finetune --task classify`, not this command.
        #[arg(long, default_value = "pretrain")]
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

        /// Task type: pretrain (causal LM). Classification fine-tuning is
        /// `apr finetune --task classify`, not this command.
        #[arg(long, default_value = "pretrain")]
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
        /// Output directory for checkpoints and leaderboard.
        ///
        /// When given it OVERRIDES `training.output_dir` in the YAML config.
        /// When omitted, the config's `training.output_dir` is used (default
        /// `./checkpoints`). The directory is created if it does not exist.
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
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

        // ── Profiling params (PMAT-486) ──
        /// Enable StepProfiler for per-phase wall-clock timing (KAIZEN-047)
        #[arg(long)]
        profile: bool,
        /// StepProfiler report interval (every N steps, default: 50)
        #[arg(long, value_name = "N", default_value = "50")]
        profile_interval: usize,
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

    /// Run successive halving HPO on sweep configs (C-HPO-001).
    ///
    /// Takes a directory of sweep configs (from `apr train sweep`), runs each
    /// for `--steps-per-round` steps, kills the worst half by val_ppl, doubles
    /// steps, and repeats for `--rounds` rounds. Reports the winner with
    /// μTransfer-scaled LR for the target model width.
    ///
    /// References: Hyperband (Li et al. 2018, arXiv:1603.06560),
    /// μTransfer (Yang et al. 2022, arXiv:2203.03466).
    Halving {
        /// Directory containing sweep-*.yaml configs (from `apr train sweep`)
        #[arg(long, value_name = "DIR")]
        sweep_dir: PathBuf,

        /// Number of halving rounds (default: 3)
        #[arg(long, default_value = "3")]
        rounds: usize,

        /// Training steps in first round (doubles each round)
        #[arg(long, default_value = "500")]
        steps_per_round: usize,

        /// Proxy model hidden_size (for μTransfer scaling)
        #[arg(long, default_value = "512")]
        source_width: usize,

        /// Target model hidden_size (for μTransfer scaling)
        #[arg(long, default_value = "1024")]
        target_width: usize,

        /// Output JSON file for results
        #[arg(long, default_value = "sweeps/hpo-results.json")]
        output: PathBuf,
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
        #[arg(long = "release-version")]
        release_version: Option<String>,

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

    /// Distillation hyperparameter sweeps, strategy comparison and cost analysis.
    ///
    /// Was the `aprender-train-bench` binary (APR-MONO Rule 1). Distinct from
    /// top-level `apr bench`, which measures inference throughput of a model
    /// file; this measures distillation *hyperparameters*.
    Bench {
        #[command(subcommand)]
        action: TrainBenchCommands,
    },

    /// End-to-end knowledge distillation driven by a distill config file.
    ///
    /// Was the `aprender-train-distill` binary (APR-MONO Rule 1). Distinct
    /// from top-level `apr distill`, which is a flag-driven teacher/student
    /// run over apr's own YAML schema; this reads entrenar's native
    /// `DistillConfig` schema and adds estimate / validate / export.
    Distill {
        #[command(subcommand)]
        action: TrainDistillCommands,
    },

    /// Inspect a training checkpoint: architecture, layers, memory, integrity.
    ///
    /// Was the `aprender-train-inspect` binary (APR-MONO Rule 1). Distinct
    /// from top-level `apr inspect`, which reads `.apr` metadata; this reads
    /// SafeTensors training checkpoints and answers training questions
    /// (per-layer parameter counts, optimizer-state memory at a batch size).
    Inspect {
        #[command(subcommand)]
        action: TrainInspectCommands,
    },

    /// Plan, compare and merge LoRA/QLoRA adapters.
    ///
    /// Was the `aprender-train-lora` binary (APR-MONO Rule 1). `apr finetune
    /// --merge` merges an adapter into a `.apr` base at scale 1.0; this is the
    /// SafeTensors/PEFT merge path with a settable `--scale`.
    Lora {
        #[command(subcommand)]
        action: TrainLoraCommands,
    },

    /// Interactive REPL for model exploration and distillation experiments.
    ///
    /// Was the `aprender-train-shell` binary (APR-MONO Rule 1).
    Shell {
        /// Load session from file
        #[arg(short, long, value_name = "FILE")]
        session: Option<PathBuf>,

        /// Execute a single command and exit (instead of entering the REPL)
        #[arg(short, long, value_name = "COMMAND")]
        command: Option<String>,

        /// Output format: table, json, or compact
        #[arg(long, default_value = "table", value_name = "FORMAT")]
        format: String,

        /// Disable colored output
        #[arg(long)]
        no_color: bool,
    },
}

/// Output-shaping arguments every rehomed `apr train` tool accepts.
///
/// The pre-migration binaries flattened `entrenar_common::CommonArgs` into
/// their top-level parser. `--quiet` / `--verbose` are already global on `apr`
/// and are folded in by `crate::commands::train_tools::common_cli`, so only the
/// two remaining flags are redeclared here. Redeclaring `-q` / `-v` would
/// collide with `apr`'s global short flags.
#[derive(clap::Args, Debug, Clone)]
pub struct TrainToolArgs {
    /// Output format: table, json, or compact
    #[arg(long, default_value = "table", value_name = "FORMAT")]
    pub format: String,

    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,
}

/// `apr train bench` — distillation benchmarking (was `aprender-train-bench`).
#[derive(Subcommand, Debug)]
pub enum TrainBenchCommands {
    /// Sweep the distillation temperature hyperparameter
    Temperature {
        /// Start of range
        #[arg(long, default_value = "1.0")]
        start: f32,

        /// End of range
        #[arg(long, default_value = "8.0")]
        end: f32,

        /// Step size
        #[arg(long, default_value = "0.5")]
        step: f32,

        /// Runs per configuration
        #[arg(long, default_value = "3")]
        runs: usize,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Sweep the alpha (KD vs task loss) hyperparameter
    Alpha {
        /// Start of range
        #[arg(long, default_value = "0.1")]
        start: f32,

        /// End of range
        #[arg(long, default_value = "0.9")]
        end: f32,

        /// Step size
        #[arg(long, default_value = "0.1")]
        step: f32,

        /// Runs per configuration
        #[arg(long, default_value = "3")]
        runs: usize,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Compare distillation strategies
    Compare {
        /// Strategies to compare (kd, progressive, attention, combined, all)
        #[arg(long, value_delimiter = ',', default_value = "all")]
        strategies: Vec<String>,

        /// Runs per strategy. Accepted for compatibility and ignored: the
        /// comparison harness is deterministic, so repeats are identical.
        #[arg(long, default_value = "5")]
        runs: usize,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Run ablation study (baseline, +KD, +progressive, +attention)
    Ablation {
        /// Base configuration file. Accepted for compatibility and ignored:
        /// the ablation ladder is fixed in code.
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Analyze cost vs performance trade-offs
    CostPerformance {
        /// GPU type for cost calculation (a100-80gb, a100-40gb, v100, t4)
        #[arg(long, default_value = "a100-80gb")]
        gpu: String,

        /// Path to benchmark results file (JSON). Accepted for compatibility
        /// and ignored: the analysis runs on generated sample points.
        #[arg(long, value_name = "FILE")]
        results: Option<PathBuf>,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Recommend configurations based on constraints
    Recommend {
        /// Maximum GPU-hours
        #[arg(long)]
        max_gpu_hours: Option<f64>,

        /// Maximum cost in USD
        #[arg(long)]
        max_cost: Option<f64>,

        /// Minimum accuracy required (0.0 - 1.0)
        #[arg(long)]
        min_accuracy: Option<f64>,

        /// Maximum memory in GB
        #[arg(long)]
        max_memory: Option<f64>,

        /// GPU type for cost calculation (a100-80gb, a100-40gb, v100, t4)
        #[arg(long, default_value = "a100-80gb")]
        gpu: String,

        #[command(flatten)]
        common: TrainToolArgs,
    },
}

/// `apr train distill` — config-driven distillation (was `aprender-train-distill`).
#[derive(Subcommand, Debug)]
pub enum TrainDistillCommands {
    /// Run the distillation pipeline
    Run {
        /// Path to configuration file
        #[arg(short, long, value_name = "FILE")]
        config: PathBuf,

        /// Override the output directory named in the config
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,

        /// Dry run (validate + estimate memory, don't train)
        #[arg(long)]
        dry_run: bool,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Estimate memory requirements for a teacher/student pair
    Estimate {
        /// Teacher model ID
        #[arg(long, value_name = "ID")]
        teacher: String,

        /// Student model ID (defaults to the teacher)
        #[arg(long, value_name = "ID")]
        student: Option<String>,

        /// Batch size
        #[arg(long, default_value = "32")]
        batch_size: u32,

        /// Sequence length
        #[arg(long, default_value = "512")]
        seq_len: usize,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Validate a distillation configuration file
    Validate {
        /// Path to configuration file
        #[arg(short, long, value_name = "FILE")]
        config: PathBuf,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Export a trained model to another format
    ///
    /// This is the one migrated subcommand that cannot flatten
    /// [`TrainToolArgs`]: its own `-f` / `--format` names the *model* format,
    /// and the deleted binary put the display `--format` one level up, on the
    /// top-level parser. Two args cannot share the name at one level. The
    /// display format is reached through `apr --json` instead — and nothing is
    /// lost, because `run_export` prints progress lines only and never reads
    /// the display format.
    Export {
        /// Input model path (SafeTensors)
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// Output model format: safetensors, gguf, apr
        #[arg(short, long, default_value = "safetensors", value_name = "FORMAT")]
        format: String,

        /// Output path
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        /// Quantization: none, q8_0, q4_0
        #[arg(long, default_value = "none", value_name = "QUANT")]
        quantize: String,

        /// Disable colored output
        #[arg(long)]
        no_color: bool,
    },
}

/// `apr train inspect` — checkpoint inspection (was `aprender-train-inspect`).
#[derive(Subcommand, Debug)]
pub enum TrainInspectCommands {
    /// Show model information (format, architecture, parameters)
    Info {
        /// Path to model file
        #[arg(value_name = "FILE")]
        path: PathBuf,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Show layer-by-layer breakdown
    ///
    /// The deleted binary's own `-v` / `--verbose` (list every tensor name and
    /// shape after the per-layer table) is served by `apr`'s global
    /// `-v` / `--verbose`, which has the same two spellings and the same
    /// meaning here. Redeclaring it would collide with the global flag.
    Layers {
        /// Path to model file
        #[arg(value_name = "FILE")]
        path: PathBuf,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Estimate training memory requirements
    Memory {
        /// Path to model file
        #[arg(value_name = "FILE")]
        path: PathBuf,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: u32,

        /// Sequence length
        #[arg(short, long, default_value = "512")]
        seq_len: usize,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Validate model integrity
    Validate {
        /// Path to model file
        #[arg(value_name = "FILE")]
        path: PathBuf,

        /// Enable strict validation
        #[arg(long)]
        strict: bool,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Convert model format
    Convert {
        /// Input model path
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output format: safetensors, gguf, apr
        #[arg(short, long, value_name = "FORMAT")]
        to: String,

        /// Output path
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        /// Quantization: q4_0, q8_0, f16, none
        #[arg(long, default_value = "none", value_name = "QUANT")]
        quantize: String,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Compare two models side by side
    Compare {
        /// First model path
        #[arg(value_name = "MODEL1")]
        model1: PathBuf,

        /// Second model path
        #[arg(value_name = "MODEL2")]
        model2: PathBuf,

        #[command(flatten)]
        common: TrainToolArgs,
    },
}

/// `apr train lora` — LoRA/QLoRA planning and merging (was `aprender-train-lora`).
#[derive(Subcommand, Debug)]
pub enum TrainLoraCommands {
    /// Plan an optimal LoRA configuration for a VRAM budget
    ///
    /// Two short flags from the deleted binary could not be carried over, and
    /// one of them never worked:
    ///
    /// * `-m` was declared **twice** there — auto-derived for `--model` and
    ///   explicitly for `--method` — which clap rejects outright
    ///   ("Short option names must be unique"). Here `-m` means `--method`,
    ///   matching `apr tune -m` and `apr finetune -m`; `--model` is long-only.
    /// * `-v` meant `--vram` there. On `apr`, `-v` is the global `--verbose`.
    ///   `--vram` is long-only.
    Plan {
        /// Model size in parameters (e.g. "7B", "350M") or an exact number
        #[arg(long, value_name = "SIZE")]
        model: String,

        /// Available VRAM in GB
        #[arg(long)]
        vram: f64,

        /// Fine-tuning method: full, lora, qlora, auto
        #[arg(short = 'm', long, default_value = "auto")]
        method: String,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Compare full / LoRA / QLoRA fine-tuning for a VRAM budget
    ///
    /// `--vram` is long-only here: the deleted binary's `-v` is `apr`'s global
    /// `--verbose`.
    Compare {
        /// Model size in parameters (e.g. "7B", "350M") or an exact number
        #[arg(long, value_name = "SIZE")]
        model: String,

        /// Available VRAM in GB
        #[arg(long, default_value = "24")]
        vram: f64,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Merge a LoRA adapter into a base model
    Merge {
        /// Path to base model
        #[arg(short, long, value_name = "FILE")]
        base: PathBuf,

        /// Path to LoRA adapter
        #[arg(short, long, value_name = "FILE")]
        adapter: PathBuf,

        /// Output path
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        /// Scale factor applied to the adapter delta
        #[arg(short, long, default_value = "1.0")]
        scale: f32,

        #[command(flatten)]
        common: TrainToolArgs,
    },

    /// Inspect LoRA adapter structure
    Inspect {
        /// Path to adapter file
        #[arg(value_name = "FILE")]
        path: PathBuf,

        #[command(flatten)]
        common: TrainToolArgs,
    },
}
