
/// Data quality pipeline subcommands (powered by alimentar).
///
/// Thin CLI wrappers around alimentar's data utilities.
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
