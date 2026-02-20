
#[derive(Subcommand, Debug)]
pub enum ToolCommands {
    /// Qwen2.5-Coder showcase demo
    Showcase {
        /// Run all steps with auto-verification
        #[arg(long)]
        auto_verify: bool,
        /// Run specific step
        #[arg(long)]
        step: Option<String>,
        /// Model tier: tiny (0.5B), small (1.5B), medium (7B), large (32B)
        #[arg(long, default_value = "small")]
        tier: String,
        /// Model directory
        #[arg(long, default_value = "./models")]
        model_dir: PathBuf,
        /// Baselines to compare: llama-cpp,ollama
        #[arg(long, default_value = "llama-cpp,ollama")]
        baseline: String,
        /// Enable ZRAM compression
        #[arg(long)]
        zram: bool,
        /// Number of benchmark runs (spec: minimum 30)
        #[arg(long, default_value = "30")]
        runs: usize,
        /// Force GPU acceleration
        #[arg(long)]
        gpu: bool,
        /// Output results as JSON
        #[arg(long)]
        json: bool,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
        /// Quiet mode (errors only)
        #[arg(short, long)]
        quiet: bool,
    },
    /// Rosetta Stone - Universal model format converter (PMAT-ROSETTA-001)
    Rosetta {
        #[command(subcommand)]
        action: RosettaCommands,
    },
    /// Publish model to HuggingFace Hub (APR-PUB-001)
    Publish {
        /// Directory containing model files to publish
        #[arg(value_name = "DIRECTORY")]
        directory: PathBuf,
        /// HuggingFace repository ID (e.g., paiml/whisper-apr-tiny)
        #[arg(value_name = "REPO_ID")]
        repo_id: String,
        /// Model display name
        #[arg(long)]
        model_name: Option<String>,
        /// License (SPDX identifier, default: mit)
        #[arg(long, default_value = "mit")]
        license: String,
        /// Pipeline tag (e.g., automatic-speech-recognition, text-generation)
        #[arg(long, default_value = "text-generation")]
        pipeline_tag: String,
        /// Library name (e.g., whisper-apr, aprender)
        #[arg(long)]
        library_name: Option<String>,
        /// Additional tags (comma-separated)
        #[arg(long, value_delimiter = ',')]
        tags: Option<Vec<String>>,
        /// Commit message
        #[arg(long)]
        message: Option<String>,
        /// Dry run (preview without uploading)
        #[arg(long)]
        dry_run: bool,
    },
    /// Model Oracle: identify family, size, constraints, and contract compliance
    ///
    /// Three modes:
    ///   apr oracle <FILE>         - Analyze local model file
    ///   apr oracle hf://org/repo  - Query HuggingFace API
    ///   apr oracle --family qwen2 - Describe contract from YAML
    Oracle {
        /// Model file path or hf:// URI
        #[arg(value_name = "SOURCE")]
        source: Option<String>,
        /// Show contract for a model family (e.g., qwen2, llama, whisper, bert)
        #[arg(long)]
        family: Option<String>,
        /// Filter to a specific size variant (e.g., 0.5b, 7b)
        #[arg(long)]
        size: Option<String>,
        /// Run full contract compliance check
        #[arg(long)]
        compliance: bool,
        /// List all tensor shapes
        #[arg(long)]
        tensors: bool,
        /// Show statistical analysis (GQA, memory, FFN, FLOPS)
        #[arg(long)]
        stats: bool,
        /// Show architecture explanations with literature references
        #[arg(long)]
        explain: bool,
        /// Show kernel compatibility report (quantization, TPS estimates)
        #[arg(long)]
        kernels: bool,
        /// Cross-validate contract against HuggingFace config.json
        #[arg(long)]
        validate: bool,
        /// Enable all analysis sections (stats + explain + kernels + validate)
        #[arg(long)]
        full: bool,
    },
}
