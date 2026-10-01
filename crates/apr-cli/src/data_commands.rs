
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
/// Thin CLI wrappers around alimentar's data utilities.
#[derive(Subcommand, Debug)]
pub enum DataCommands {
    /// Prepare the TweetEval abortion stance benchmark as aprender JSONL
    TweetEvalStance {
        /// Output directory for JSONL splits and benchmark-manifest.json
        #[arg(short, long, value_name = "DIR")]
        output: PathBuf,
        /// Split layout: canonical (train/validation/test) or setfit (train/test)
        #[arg(long, value_enum, default_value_t = TweetEvalStanceProfile::Canonical)]
        profile: TweetEvalStanceProfile,
        /// Existing canonical TweetEval abortion directory (disables download)
        #[arg(long, value_name = "DIR")]
        source: Option<PathBuf>,
        /// Pinned TweetEval git revision used for provenance and downloads
        #[arg(long, default_value = crate::commands::data_tweeteval::CANONICAL_REVISION)]
        revision: String,
        /// Replace benchmark files already present in the output directory
        #[arg(long)]
        force: bool,
    },
    /// Select a balanced few-shot training subset and write its replayable manifest
    ///
    /// Reads an ATTESTED benchmark directory: the label map, per-class counts, split
    /// digests, exclusion record and profile all come from the `dataset_attestation`
    /// section of its `benchmark-manifest.json`, never from hardcoded constants. A
    /// compatibility-profile, mixed, stale-schema or forged directory is a typed error
    /// before a single row is selected.
    Select {
        /// Attested benchmark directory: canonical train/validation/test JSONL plus
        /// benchmark-manifest.json, as written by `apr data tweet-eval-stance`
        #[arg(long, value_name = "DIR")]
        data: PathBuf,
        /// Examples per class — one of the contracted few-shot sizes 8, 16, 32, 64
        #[arg(long, value_name = "N")]
        shots: u32,
        /// Root seed. REQUIRED, with no default: it must be one of the ten contracted
        /// benchmark seeds 13, 17, 23, 29, 31, 37, 41, 43, 47, 53 unless --any-seed is
        /// given. 42 is deliberately NOT one of them, which is why there is no default:
        /// a defaulted seed would quietly produce an off-protocol selection
        #[arg(long, value_name = "SEED")]
        seed: u64,
        /// Accept a seed outside the ten contracted benchmark seeds. The mode is
        /// recorded in the manifest and in --json output, so an experimental selection
        /// cannot later be mistaken for a benchmark cell
        #[arg(long = "any-seed")]
        any_seed: bool,
        /// Directory to write selection-manifest.json into (default: --data)
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
        /// Replace an existing selection-manifest.json
        #[arg(long)]
        force: bool,
    },
    /// Replay a selection into a bounded, deterministic contrastive pair stream
    ///
    /// There is deliberately no --seed here: the root seed is part of the selection
    /// manifest, and re-supplying it at the pair stage would create two sources of truth
    /// for one replay tuple. Pairs are regenerated from that tuple rather than stored —
    /// only the pair-manifest hash is persisted, and `--dump` is the explicit audit path.
    Pairs {
        /// The selection-manifest.json written by `apr data select`
        #[arg(long, value_name = "FILE")]
        selection: PathBuf,
        /// The same attested canonical directory the selection was drawn from; the
        /// manifest is strictly replayed against it before any pair is emitted
        #[arg(long, value_name = "DIR")]
        data: PathBuf,
        /// Pairs per epoch. Omit for the contracted default — the closed-form
        /// oversampling count clamped by --hard-cap. A value ABOVE --hard-cap is an
        /// ERROR naming both numbers, never a silent clamp
        #[arg(long, value_name = "N")]
        budget: Option<u64>,
        /// Upper bound on the per-epoch budget (default: the crate's contracted hard
        /// cap). It clamps the DEFAULT budget and BINDS an explicit --budget
        #[arg(long = "hard-cap", value_name = "N")]
        hard_cap: Option<u64>,
        /// Write one JSON line per pair, in stream order, to this path (audit only)
        #[arg(long, value_name = "FILE")]
        dump: Option<PathBuf>,
        /// Replace an existing --dump file
        #[arg(long)]
        force: bool,
    },
    /// Every alimentar data command: convert, info, head, schema, mix, fim,
    /// filter-text, view, import, hub, registry, drift, quality, fed, doctest,
    /// extract, merge.
    ///
    /// APR-MONO consolidated alimentar in-tree, but its capability stayed
    /// reachable only through the standalone `alimentar` binary -- `apr data`
    /// shipped 5 commands against alimentar's 20, so 18 had no route through
    /// `apr` at all. This dispatches the SAME `alimentar::cli::dispatch`, so
    /// there is one implementation behind two names rather than a second clap
    /// tree that can drift from the first.
    #[command(subcommand, name = "x")]
    Alimentar(alimentar::cli::Commands),

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
}

/// Supported layouts for the TweetEval abortion stance benchmark.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TweetEvalStanceProfile {
    /// Original TweetEval train/validation/test splits (recommended).
    Canonical,
    /// SetFit wrapper layout: train plus validation+test merged as test.
    Setfit,
}
