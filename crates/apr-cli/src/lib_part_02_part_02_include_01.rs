    /// PTX analysis and bug detection (trueno-explain: register pressure, roofline, 15+ bug detectors)
    #[command(name = "ptx")]
    Ptx {
        /// Path to a PTX source file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Analyze a named kernel from trueno-gpu
        #[arg(long, short)]
        kernel: Option<String>,
        /// Strict mode (no performance whitelist)
        #[arg(long)]
        strict: bool,
        /// Show only bug analysis (skip register/memory/roofline)
        #[arg(long)]
        bugs: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Verbose output (include PTX source listing)
        #[arg(short, long)]
        verbose: bool,
    },
    /// ML tuning: LoRA/QLoRA configuration and memory planning (GH-176)
    Tune {
        /// Path to model file (optional if using --model)
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Tuning method: auto, full, lora, qlora
        #[arg(long, short = 'm', default_value = "auto")]
        method: String,
        /// LoRA rank (default: auto-selected)
        #[arg(long, short = 'r')]
        rank: Option<u32>,
        /// Available VRAM in GB
        #[arg(long, default_value = "16.0")]
        vram: f64,
        /// Only plan configuration, don't train
        #[arg(long)]
        plan: bool,
        /// Model size for planning (e.g., "7B", "1.5B")
        #[arg(long, value_name = "SIZE")]
        model: Option<String>,
        /// Freeze base model weights
        #[arg(long)]
        freeze_base: bool,
        /// Training data file (JSONL format)
        #[arg(long, value_name = "FILE")]
        train_data: Option<PathBuf>,
        /// Output as JSON (for CI integration)
        #[arg(long)]
        json: bool,
    },
    /// ComputeBrick pipeline monitor (cbtop)
    Cbtop {
        /// Model name (e.g., qwen2.5-coder-1.5b)
        #[arg(long)]
        model: Option<String>,
        /// Attach to running realizar process
        #[arg(long)]
        attach: Option<String>,
        /// Path to GGUF model file for real profiling
        #[arg(long, value_name = "MODEL")]
        model_path: Option<PathBuf>,
        /// Run in headless mode (no TUI, for CI/automation)
        #[arg(long)]
        headless: bool,
        /// Output JSON format (requires --headless)
        #[arg(long)]
        json: bool,
        /// Output file path (requires --headless)
        #[arg(long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// CI mode: exit with code 1 if thresholds not met
        #[arg(long)]
        ci: bool,
        /// Minimum throughput threshold in tok/s (for --ci)
        #[arg(long, value_name = "TOK_S")]
        throughput: Option<f64>,
        /// Minimum brick score threshold 0-100 (for --ci)
        #[arg(long, value_name = "SCORE")]
        brick_score: Option<u32>,
        /// Number of warmup iterations before measurement
        #[arg(long, default_value = "10")]
        warmup: usize,
        /// Number of measurement iterations
        #[arg(long, default_value = "100")]
        iterations: usize,
        /// PAR-100: Enable speculative decoding benchmark
        #[arg(long)]
        speculative: bool,
        /// PAR-100: Number of tokens to draft speculatively (default: 4)
        #[arg(long, default_value = "4")]
        speculation_k: usize,
        /// PAR-099: Path to draft model for speculative decoding
        #[arg(long, value_name = "DRAFT_MODEL")]
        draft_model: Option<PathBuf>,
        /// PAR-102: Number of concurrent requests
        #[arg(long, default_value = "1")]
        concurrent: usize,
        /// Use simulated data (for CI testing only)
        #[arg(long)]
        simulated: bool,
    },
    /// Export for probar visual testing
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
