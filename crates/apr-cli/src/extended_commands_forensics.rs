// Forensics / parity / visualization subcommands (extracted from
// `extended_commands.rs` to keep the PMAT-689 file-size invariant).
//
// `ForensicsCommands` is flattened into `ExtendedCommands` via
// `#[command(flatten)]` so each variant is still invoked top-level
// (`apr probar`, `apr compare-hf`, `apr hex`, `apr tree`, `apr flow`,
// `apr qualify`).

/// Forensics / parity / visualization subcommands.
#[derive(Subcommand, Debug)]
pub enum ForensicsCommands {
    /// Export for probar visual regression testing (PMAT-481)
    Probar {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Output directory for test artifacts
        #[arg(short, long, default_value = "./probar-export")]
        output: PathBuf,
        /// Export format: json, png, or both
        #[arg(long, default_value = "both")]
        format: String,
        /// Golden reference directory for comparison
        #[arg(long)]
        golden: Option<PathBuf>,
        /// Filter layers by name pattern
        #[arg(long)]
        layer: Option<String>,
        /// Exit non-zero on golden divergence (CI mode, PMAT-481)
        #[arg(long)]
        assert: bool,
        /// Cosine similarity threshold for golden comparison (default: 0.98)
        #[arg(long, default_value = "0.98")]
        tolerance: f32,
    },
    /// Compare APR model against HuggingFace source
    #[command(name = "compare-hf")]
    CompareHf {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// HuggingFace repo ID (e.g., openai/whisper-tiny)
        #[arg(long)]
        hf: String,
        /// Filter tensors by name pattern
        #[arg(long)]
        tensor: Option<String>,
        /// Comparison threshold (default: 1e-5)
        #[arg(long, default_value = "1e-5")]
        threshold: f64,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Format-aware binary forensics (10X better than xxd)
    Hex {
        /// Path to model file (APR, GGUF, or SafeTensors)
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Filter tensors by name pattern
        #[arg(long)]
        tensor: Option<String>,
        /// Limit bytes/values to display
        #[arg(long, default_value = "64")]
        limit: usize,
        /// Show tensor statistics
        #[arg(long)]
        stats: bool,
        /// List tensor names only
        #[arg(long)]
        list: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Annotated file header (magic, version, tensor count, metadata)
        #[arg(long)]
        header: bool,
        /// Q4K/Q6K/Q8_0 super-block structure with field annotations
        #[arg(long)]
        blocks: bool,
        /// Value histogram + entropy + kurtosis analysis
        #[arg(long)]
        distribution: bool,
        /// Layout contract verification overlay per tensor
        #[arg(long)]
        contract: bool,
        /// Per-region byte entropy analysis
        #[arg(long)]
        entropy: bool,
        /// Raw bytes (like xxd but format-aware, with ASCII column)
        #[arg(long)]
        raw: bool,
        /// Start at byte offset (supports 0x prefix for hex)
        #[arg(long, default_value = "0")]
        offset: String,
        /// Bytes per row for raw output (default: 16)
        #[arg(long, default_value = "16")]
        width: usize,
        /// Slice range for partial tensor reads (e.g., 0:3 for first 3 elements)
        #[arg(long)]
        slice: Option<String>,
    },
    /// Model architecture tree view
    Tree {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Filter by component pattern
        #[arg(long)]
        filter: Option<String>,
        /// Output format: ascii, dot, mermaid, json
        #[arg(long, default_value = "ascii")]
        format: String,
        /// Show tensor sizes
        #[arg(long)]
        sizes: bool,
        /// Maximum tree depth
        #[arg(long)]
        depth: Option<usize>,
    },
    /// Data flow visualization
    Flow {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Filter by layer pattern
        #[arg(long)]
        layer: Option<String>,
        /// Component to visualize: full, encoder, decoder, etc.
        #[arg(long, default_value = "full")]
        component: String,
        /// Verbose output with statistics
        #[arg(short, long)]
        verbose: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Cross-subcommand smoke test (does every tool handle this model?)
    Qualify {
        /// Path to model file (APR, GGUF, or SafeTensors)
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Testing tier: smoke (Phase 1), standard (+contracts), full (+playbook)
        #[arg(long, default_value = "smoke")]
        tier: String,
        /// Timeout per gate in seconds
        #[arg(long, default_value = "120")]
        timeout: u64,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Show subcommand output (disable stdout suppression)
        #[arg(short, long)]
        verbose: bool,
        /// Skip specific gates (comma-separated)
        #[arg(long, value_delimiter = ',')]
        skip: Option<Vec<String>>,
    },
}
