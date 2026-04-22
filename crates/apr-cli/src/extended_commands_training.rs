// Training + data-pipeline subcommands (extracted from `extended_commands.rs`
// to keep the PMAT-689 file-size invariant).
//
// `TrainingCommands` is flattened into `ExtendedCommands` via
// `#[command(flatten)]` so each variant is still invoked top-level
// (`apr train`, `apr pretrain`, `apr tokenize`, `apr data`, `apr pipeline`,
// `apr diagnose`).

/// Training + data-pipeline subcommands.
#[derive(Subcommand, Debug)]
pub enum TrainingCommands {
    /// Training pipeline (plan/apply) — forjar-style pre-flight validation
    #[cfg(feature = "training")]
    Train {
        #[command(subcommand)]
        command: TrainCommands,
    },
    /// Pretraining loop driver (SHIP-TWO-001 MODEL-2).
    ///
    /// Wires the pretraining loop shape defined by
    /// `contracts/training-loop-pretrain-v1.yaml`. Executes a synthetic
    /// decreasing-loss drive by default so GATE-TRAIN-005 / -007 / -008
    /// divergence-and-NaN guards can be exercised without an actual
    /// 370M compute run. Real corpus wiring is a follow-up ticket.
    #[cfg(feature = "training")]
    Pretrain {
        /// Dataset path (tokenized shard index or raw corpus).
        #[arg(long, value_name = "PATH")]
        dataset: PathBuf,
        /// Tokenizer directory (vocab.json + merges.txt).
        #[arg(long, value_name = "DIR")]
        tokenizer: PathBuf,
        /// Run output directory — checkpoints + metadata go to `{run_dir}/ckpt/`.
        #[arg(long, value_name = "DIR")]
        run_dir: PathBuf,
        /// Training regime — finetune (MODEL-1) or from-scratch (MODEL-2 cold start).
        /// Per contract training-loop-pretrain-v1 §hyperparameter_defaults,
        /// this atomically flips (regime, lr_max, warmup_steps, target_val_loss)
        /// unless explicit --lr / --warmup-steps / --target-val-loss override.
        #[arg(long, value_enum, default_value = "finetune")]
        mode: PretrainMode,
        /// Peak learning rate after warmup. Omit to inherit mode default
        /// (finetune: 5e-5, from-scratch: 3e-4).
        #[arg(long)]
        lr: Option<f32>,
        /// Warmup + cosine decay total steps.
        #[arg(long, default_value = "1000")]
        num_steps: usize,
        /// Number of warmup steps. Omit to inherit mode default
        /// (finetune: 100, from-scratch: 1000).
        #[arg(long)]
        warmup_steps: Option<usize>,
        /// Micro-batch size.
        #[arg(long, default_value = "16")]
        batch_size: usize,
        /// Sequence length per example.
        #[arg(long, default_value = "1024")]
        seq_length: usize,
        /// Steps per epoch — controls per-epoch artifact cadence.
        #[arg(long, default_value = "100")]
        steps_per_epoch: usize,
        /// GATE-TRAIN-006 fixed RNG seed.
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Target val_loss. Omit to inherit mode default
        /// (finetune: 2.2, from-scratch: 3.0).
        #[arg(long)]
        target_val_loss: Option<f32>,
        /// Vocabulary size (required for `--mode from-scratch` INV-TRAIN-005
        /// regime-dependent cap: 2·ln(vocab_size)). MODEL-2 uses 50257.
        #[arg(long, default_value = "50257")]
        vocab_size: u32,
        /// Synthetic-drive only — do not attempt real compute, exercise loop gates only.
        /// INV-TRAIN-010: absent = real compute (drive_real), present = synthetic (drive_synthetic).
        #[arg(long, action = clap::ArgAction::SetTrue)]
        synthetic: bool,
        /// Training backend. Grammar (contract gpu-training-backend-v1
        /// INV-GPUTRAIN-001): `^(cpu|cuda(:[0-9]|:1[0-5])?|auto)$`.
        /// Default `auto` uses CUDA if available, else CPU (the only
        /// spelling that may fall back silently — all other values
        /// hard-fail on missing runtime per GATE-GPUTRAIN-002).
        #[arg(long, default_value = "auto")]
        device: String,
        /// Opt into shard-stream cycling when the planned token budget
        /// exceeds corpus total_tokens. Contract:
        /// `contracts/pretraining-corpus-v1.yaml` v2.0.0 §FALSIFY-CORPUS-004
        /// / GATE-CORPUS-PREFLIGHT. Absent (default): dispatch refuses
        /// to start on over-dispatch. Present: emits a single INFO log
        /// at the first cycle boundary and continues (INV-TRAIN-011
        /// path a). This is the path that converts task #141's silent
        /// `(1.0, 1.0)` placeholder into either a refusal or an
        /// explicit operator opt-in.
        #[arg(long, action = clap::ArgAction::SetTrue)]
        allow_shard_cycle: bool,
    },
    /// Tokenizer training pipeline (plan/apply) — BPE vocabulary learning
    Tokenize {
        #[command(subcommand)]
        command: TokenizeCommands,
    },
    /// Data quality pipeline (audit, split, balance) — powered by alimentar
    Data {
        #[command(subcommand)]
        command: DataCommands,
    },
    /// Pipeline orchestration (plan/apply/status) — wraps forjar DAG engine
    Pipeline {
        #[command(subcommand)]
        command: PipelineCommands,
    },
    /// Automated Five Whys diagnosis on a training checkpoint
    Diagnose {
        /// Path to checkpoint directory
        #[arg(value_name = "CHECKPOINT_DIR")]
        checkpoint_dir: PathBuf,
        /// Test data file (JSONL) for evaluation
        #[arg(long, value_name = "FILE")]
        data: Option<PathBuf>,
        /// Model size hint: "0.5B", "tiny"
        #[arg(long)]
        model_size: Option<String>,
        /// Number of output classes (default: 5)
        #[arg(long, default_value = "5")]
        num_classes: usize,
    },
}
