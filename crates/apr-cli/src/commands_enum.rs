
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run model directly (auto-download, cache, execute)
    Run {
        /// Model source: local path, hf://org/repo, or URL
        #[arg(value_name = "SOURCE")]
        source: String,
        /// Text prompt (positional): `apr run model.gguf "What is 2+2?"`
        #[arg(value_name = "PROMPT")]
        positional_prompt: Option<String>,
        /// Input file (audio, text, etc.)
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Text prompt for generation (for LLM models)
        #[arg(short, long)]
        prompt: Option<String>,
        /// Maximum tokens to generate (default: 32)
        #[arg(short = 'n', long, default_value = "32")]
        max_tokens: usize,
        /// Enable streaming output
        #[arg(long)]
        stream: bool,
        /// Language code (for ASR models)
        #[arg(short, long)]
        language: Option<String>,
        /// Task (transcribe, translate)
        #[arg(short, long)]
        task: Option<String>,
        /// Output format (text, json, srt, vtt)
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
        /// Disable GPU acceleration
        #[arg(long, conflicts_with = "gpu")]
        no_gpu: bool,
        /// Force GPU acceleration
        #[arg(long, conflicts_with = "no_gpu")]
        gpu: bool,
        /// Offline mode: block all network access (Sovereign AI compliance)
        #[arg(long)]
        offline: bool,
        /// Benchmark mode: output performance metrics (tok/s, latency)
        #[arg(long)]
        benchmark: bool,
        /// Enable inference tracing (APR-TRACE-001)
        #[arg(long)]
        trace: bool,
        /// Trace specific steps only (comma-separated)
        #[arg(long, value_delimiter = ',')]
        trace_steps: Option<Vec<String>>,
        /// Verbose tracing (show tensor values)
        #[arg(long)]
        trace_verbose: bool,
        /// Save trace output to JSON file
        #[arg(long, value_name = "FILE")]
        trace_output: Option<PathBuf>,
        /// Trace detail level (none, basic, layer, payload)
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,
        /// Shorthand for --trace --trace-level payload (tensor value inspection)
        #[arg(long)]
        trace_payload: bool,
        /// Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        profile: bool,
        /// Apply chat template for Instruct models (GAP-UX-001)
        ///
        /// Wraps prompt in ChatML format for Qwen2, LLaMA, Mistral Instruct models.
        /// Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        #[arg(long)]
        chat: bool,
        /// Sampling temperature (0.0 = greedy, default: 0.0)
        #[arg(long, default_value = "0.0")]
        temperature: f32,
        /// Top-k sampling (default: 1 = greedy)
        #[arg(long, default_value = "1")]
        top_k: usize,
        /// Batch mode: read prompts from JSONL, output results as JSONL.
        /// Model loads once, processes all prompts sequentially.
        /// Each input line: {"prompt": "...", "task_id": "..."}
        /// Chat template is applied automatically.
        #[arg(long, value_name = "FILE")]
        batch_jsonl: Option<PathBuf>,
        /// Show verbose output (model loading, backend info)
        #[arg(short, long)]
        verbose: bool,
    },
    /// Inference server (plan/run)
    Serve {
        #[command(subcommand)]
        command: ServeCommands,
    },
    /// Inspect model metadata, vocab, and structure
    Inspect {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Show vocabulary details
        #[arg(long)]
        vocab: bool,
        /// Show filter/security details
        #[arg(long)]
        filters: bool,
        /// Show weight statistics
        #[arg(long)]
        weights: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Simple debugging output ("drama" mode available)
    Debug {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Theatrical "drama" mode output
        #[arg(long)]
        drama: bool,
        /// Show hex dump
        #[arg(long)]
        hex: bool,
        /// Extract ASCII strings
        #[arg(long)]
        strings: bool,
        /// Limit output lines
        #[arg(long, default_value = "256")]
        limit: usize,
    },
    /// Validate model integrity and quality
    Validate {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Show 100-point quality assessment
        #[arg(long)]
        quality: bool,
        /// Strict validation (fail on warnings)
        #[arg(long)]
        strict: bool,
        /// Minimum score to pass (0-100)
        #[arg(long)]
        min_score: Option<u8>,
    },
    /// Compare two models
    Diff {
        /// First model file
        #[arg(value_name = "FILE1")]
        file1: PathBuf,
        /// Second model file
        #[arg(value_name = "FILE2")]
        file2: PathBuf,
        /// Show weight-level differences
        #[arg(long)]
        weights: bool,
        /// Compare actual tensor values with statistical analysis
        #[arg(long)]
        values: bool,
        /// Filter tensors by name pattern (for --values)
        #[arg(long)]
        filter: Option<String>,
        /// Maximum number of tensors to compare (for --values)
        #[arg(long, default_value = "10")]
        limit: usize,
        /// Account for transpose when comparing (GGUF col-major vs APR row-major)
        #[arg(long)]
        transpose_aware: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// List tensor names and shapes
    Tensors {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Show tensor statistics (mean, std, min, max)
        #[arg(long)]
        stats: bool,
        /// Filter tensors by name pattern
        #[arg(long)]
        filter: Option<String>,
        /// Limit number of tensors shown (0 = unlimited)
        #[arg(long, default_value = "0")]
        limit: usize,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Layer-by-layer trace analysis
    Trace {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Filter layers by name pattern
        #[arg(long)]
        layer: Option<String>,
        /// Compare with reference model
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Verbose output with per-layer stats
        #[arg(short, long)]
        verbose: bool,
        /// Trace payload through model
        #[arg(long)]
        payload: bool,
        /// Diff mode
        #[arg(long)]
        diff: bool,
        /// Interactive mode
        #[arg(long)]
        interactive: bool,
    },
    /// Check for best practices and conventions
    Lint {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Explain errors, architecture, tensors, and kernel dispatch
    Explain {
        /// Error code, model file path, or family name (auto-detected)
        #[arg(value_name = "CODE_OR_FILE")]
        code_or_file: Option<String>,
        /// Path to .apr model file (optional context for --tensor)
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Explain a specific tensor
        #[arg(long)]
        tensor: Option<String>,
        /// Explain kernel dispatch pipeline for architecture
        #[arg(long)]
        kernel: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Show kernel contract details and proof obligations
        #[arg(short, long)]
        verbose: bool,
        /// Show per-kernel proof status from contract tests
        #[arg(long)]
        proof_status: bool,
    },
    /// Manage canary tests for regression
    Canary {
        #[command(subcommand)]
        command: CanaryCommands,
    },
    /// Export model to other formats
    Export {
        /// Path to .apr model file
        #[arg(value_name = "FILE", required_unless_present = "list_formats")]
        file: Option<PathBuf>,
        /// Output format (safetensors, gguf, mlx, onnx, openvino, coreml)
        #[arg(long, default_value = "safetensors")]
        format: String,
        /// Output file/directory path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Apply quantization during export (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// List all supported export formats
        #[arg(long)]
        list_formats: bool,
        /// Batch export to multiple formats (comma-separated: gguf,mlx,safetensors)
        #[arg(long)]
        batch: Option<String>,
        /// Output in JSON format
        #[arg(long)]
        json: bool,
        /// Plan mode (validate inputs, show export plan, no execution)
        #[arg(long)]
        plan: bool,
    },
    /// Import from external formats (hf://org/repo, local files, URLs)
    Import {
        /// Source: hf://org/repo, local file, or URL
        #[arg(value_name = "SOURCE")]
        source: String,
        /// Output .apr file path (default: derived from source name)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Model architecture (whisper, llama, bert, qwen2, qwen3, gpt2, starcoder, gpt-neox, opt, phi, gemma, falcon, mamba, t5, auto)
        #[arg(long, default_value = "auto")]
        arch: String,
        /// Quantization (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// Strict mode: reject unverified architectures and fail on validation errors
        #[arg(long)]
        strict: bool,
        /// Preserve Q4K quantization for fused kernel inference (GGUF only)
        /// Uses realizar's Q4K converter instead of dequantizing to F32
        #[arg(long)]
        preserve_q4k: bool,
        /// PMAT-232: External tokenizer.json for weights-only GGUF files.
        /// Required if the GGUF has no embedded tokenizer vocabulary.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// F-GT-001: Enforce provenance chain. Rejects pre-baked GGUF imports
        /// (only SafeTensors sources allowed). Ensures single-provenance testing.
        #[arg(long)]
        enforce_provenance: bool,
        /// GH-223: Allow import without config.json (default: error).
        /// Without config.json, hyperparameters like rope_theta are inferred from
        /// tensor shapes and may be wrong, producing garbage output.
        #[arg(long)]
        allow_no_config: bool,
    },
    /// Download and cache model from HuggingFace (Ollama-like UX)
    Pull {
        /// Model reference (alias, hf:// URI, or org/repo)
        #[arg(value_name = "MODEL")]
        model_ref: String,
        /// Force re-download even if cached
        #[arg(long)]
        force: bool,
    },
    /// List cached models
    #[command(name = "list", alias = "ls")]
    List,
    /// Remove model from cache
    #[command(name = "rm", alias = "remove")]
    Rm {
        /// Model reference to remove
        #[arg(value_name = "MODEL")]
        model_ref: String,
    },
    /// Convert/optimize model
    Convert {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Quantize to format (int8, int4, fp16, q4k)
        #[arg(long)]
        quantize: Option<String>,
        /// Compress output (none, zstd, zstd-max, lz4)
        #[arg(long)]
        compress: Option<String>,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },
    /// Compile model into standalone executable (APR-SPEC §4.16)
    Compile {
        /// Input .apr model file
        #[arg(value_name = "FILE", required_unless_present = "list_targets")]
        file: Option<PathBuf>,
        /// Output binary path (default: derived from model name)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Target triple (e.g., x86_64-unknown-linux-musl)
        #[arg(long)]
        target: Option<String>,
        /// Quantize weights before embedding (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// Release mode (optimized)
        #[arg(long)]
        release: bool,
        /// Strip debug symbols
        #[arg(long)]
        strip: bool,
        /// Enable LTO (Link-Time Optimization)
        #[arg(long)]
        lto: bool,
        /// List available compilation targets
        #[arg(long)]
        list_targets: bool,
    },
    /// Merge multiple models
    Merge {
        /// Model files to merge
        #[arg(value_name = "FILES", num_args = 2..)]
        files: Vec<PathBuf>,
        /// Merge strategy (average, weighted, slerp, ties, dare)
        #[arg(long, default_value = "average")]
        strategy: String,
        /// Output file path (optional in --plan mode)
        #[arg(short, long, required_unless_present = "plan")]
        output: Option<PathBuf>,
        /// Weights for weighted merge (comma-separated, e.g., "0.7,0.3")
        #[arg(long, value_delimiter = ',')]
        weights: Option<Vec<f32>>,
        /// Base model for TIES/DARE (task vectors computed as delta from base)
        #[arg(long)]
        base_model: Option<PathBuf>,
        /// DARE drop probability (default: 0.9)
        #[arg(long, default_value = "0.9")]
        drop_rate: f32,
        /// TIES trim density threshold (default: 0.2)
        #[arg(long, default_value = "0.2")]
        density: f32,
        /// RNG seed for DARE (default: 42)
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Plan mode (validate inputs, show merge plan, no execution)
        #[arg(long)]
        plan: bool,
    },
    /// Quantize model weights (GH-243)
    Quantize {
        /// Input model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Quantization scheme: int8, int4, fp16, q4k
        #[arg(long, short = 's', default_value = "int4")]
        scheme: String,
        /// Output file path (required unless --plan)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format override (apr, gguf, safetensors)
        #[arg(long)]
        format: Option<String>,
        /// Batch quantization (comma-separated schemes)
        #[arg(long)]
        batch: Option<String>,
        /// Plan mode (estimate only, no execution)
        #[arg(long)]
        plan: bool,
        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },
    /// Model optimization commands (fine-tune, prune, distill)
    #[command(flatten)]
    ModelOps(ModelOpsCommands),
    /// Interactive terminal UI
    Tui {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
    },
    /// Model self-test: 10-stage pipeline integrity check (APR-TRACE-001)
    Check {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// GPU status and VRAM reservation management (GPU-SHARE-001)
    #[cfg(feature = "training")]
    Gpu {
        /// Show reservations as JSON
        #[arg(long)]
        json: bool,
    },
    /// Extended analysis, profiling, QA, and visualization commands
    #[command(flatten)]
    Extended(ExtendedCommands),
}
