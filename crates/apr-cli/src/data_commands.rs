
/// Parse an overlap threshold, rejecting values outside the documented
/// 0.0-1.0 range.
///
/// A threshold above 1.0 makes the per-sample overlap test unsatisfiable and
/// silently turns the AC-016 contamination gate into an unconditional pass;
/// below 0.0 it flags everything. Neither is a meaningful ratio.
fn parse_unit_interval(raw: &str) -> Result<f64, String> {
    let value: f64 = raw
        .parse()
        .map_err(|_| format!("'{raw}' is not a number"))?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!(
            "'{raw}' is outside the valid range 0.0-1.0 (an overlap ratio)"
        ));
    }
    Ok(value)
}

/// Parse an n-gram size, rejecting 0.
///
/// A zero-width window is not a window: `--ngram 0` reached
/// `slice::windows(0)` inside the decontamination scan and aborted the
/// process with "window size must be non-zero" (exit 101).
fn parse_ngram_size(raw: &str) -> Result<usize, String> {
    let value: usize = raw
        .parse()
        .map_err(|_| format!("'{raw}' is not a non-negative integer"))?;
    if value == 0 {
        return Err("n-gram size must be >= 1; a zero-width window compares nothing".to_string());
    }
    Ok(value)
}

/// Data quality pipeline subcommands (powered by alimentar).
///
/// Two groups live here:
///
/// 1. The five ML-dataset-hygiene commands defined inline below (`audit`,
///    `split`, `decontaminate`, `dedup`, `balance`), implemented in
///    `commands/data.rs` against JSONL classification datasets.
/// 2. Everything the `alimentar` binary used to expose, flattened in from
///    `alimentar::cli::Commands` — `convert`, `info`, `head`, `schema`,
///    `mix`, `fim`, `dedup-text`, `filter-text`, `view`, `import`, `hub`,
///    `registry`, `drift`, `quality`, `fed`, `doctest`, `repl`.
///
/// Group 2 was published as a binary named `alimentar`, which COLLIDES on
/// crates.io with an unrelated `alimentar` crate that ships a bin of the same
/// name — so `cargo install aprender-data` and `cargo install alimentar` fought
/// over one path in `~/.cargo/bin`. The binary is gone; the commands are here.
///
/// The only name that had to change is alimentar's `dedup`, which is spelled
/// `dedup-text` because `apr data dedup` already means something else (exact
/// whole-row dedup of a JSONL file, versus text-column dedup of an Arrow
/// dataset). See the note on `alimentar::cli::Commands::Dedup`.
#[derive(Subcommand, Debug)]
pub enum DataCommands {
    /// Audit a JSONL classification dataset for quality issues
    Audit {
        /// Path to JSONL data file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Number of output classes (for label range validation)
        #[arg(long, default_value = "5")]
        num_classes: usize,
        /// Input text column name
        #[arg(long, default_value = "input")]
        input_column: String,
        /// Label column name
        #[arg(long, default_value = "label")]
        label_column: String,
        /// Preamble prefix to detect (e.g., "#!/")
        #[arg(long, default_value = "#!/")]
        preamble_prefix: Option<String>,
    },
    /// Stratified train/val/test split preserving class proportions
    Split {
        /// Path to JSONL data file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Training set fraction
        #[arg(long, default_value = "0.8")]
        train: f64,
        /// Validation set fraction
        #[arg(long, default_value = "0.1")]
        val: f64,
        /// Test set fraction
        #[arg(long, default_value = "0.1")]
        test: f64,
        /// Label column name for stratification
        #[arg(long, default_value = "label")]
        label_column: String,
        /// Random seed for deterministic split
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Output directory for split files
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Check training data for benchmark contamination via n-gram overlap
    Decontaminate {
        /// Path to training JSONL data file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Reference benchmark JSONL files to check against
        #[arg(long, required = true, num_args = 1..)]
        reference: Vec<PathBuf>,
        /// N-gram size for overlap detection (must be >= 1)
        #[arg(long, default_value = "10", value_parser = parse_ngram_size)]
        ngram: usize,
        /// Overlap threshold (0.0-1.0) above which a sample is flagged
        #[arg(long, default_value = "0.5", value_parser = parse_unit_interval)]
        threshold: f64,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Remove exact duplicate rows from a JSONL dataset
    Dedup {
        /// Path to JSONL data file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Output file path for the deduplicated dataset
        #[arg(short, long)]
        output: PathBuf,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Resample dataset to address class imbalance
    Balance {
        /// Path to JSONL data file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Rebalancing strategy: oversample, undersample, sqrt-inverse
        #[arg(long, default_value = "oversample")]
        strategy: String,
        /// Label column name
        #[arg(long, default_value = "label")]
        label_column: String,
        /// Number of classes (for sqrt-inverse weight computation)
        #[arg(long)]
        num_classes: Option<usize>,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Output file path (required for oversample/undersample)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Dataset tooling: convert, inspect, mix, filter, import, registry,
    /// drift, quality, federated splits (formerly the `alimentar` binary)
    #[command(flatten)]
    Toolbox(alimentar::cli::Commands),
}
