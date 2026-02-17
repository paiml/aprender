
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
        /// Show verbose output (model loading, backend info)
        #[arg(short, long)]
        verbose: bool,
    },
    /// Start inference server (REST API, streaming, metrics)
    Serve {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Disable CORS
        #[arg(long)]
        no_cors: bool,
        /// Disable Prometheus metrics endpoint
        #[arg(long)]
        no_metrics: bool,
        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,
        /// Force GPU acceleration (requires CUDA)
        #[arg(long)]
        gpu: bool,
        /// Enable batched GPU inference for 2X+ throughput
        #[arg(long)]
        batch: bool,
        /// Enable inference tracing (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        trace: bool,
        /// Trace detail level (none, basic, layer)
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,
        /// Enable inline Roofline profiling (adds X-Profile headers)
        #[arg(long)]
        profile: bool,
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
    /// Explain errors, architecture, and tensors
    Explain {
        /// Explain a specific error code
        #[arg(value_name = "CODE")]
        code: Option<String>,
        /// Path to .apr model file (optional context)
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Explain a specific tensor
        #[arg(long)]
        tensor: Option<String>,
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
    },
    /// Import from external formats (hf://org/repo, local files, URLs)
    Import {
        /// Source: hf://org/repo, local file, or URL
        #[arg(value_name = "SOURCE")]
        source: String,
        /// Output .apr file path (default: derived from source name)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Model architecture (whisper, llama, bert, auto)
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
    /// Merge multiple models
    Merge {
        /// Model files to merge
        #[arg(value_name = "FILES", num_args = 2..)]
        files: Vec<PathBuf>,
        /// Merge strategy (average, weighted, slerp, ties, dare)
        #[arg(long, default_value = "average")]
        strategy: String,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
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
    /// Fine-tune model with LoRA/QLoRA (GH-244)
    Finetune {
        /// Input model file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Fine-tuning method: auto, full, lora, qlora
        #[arg(long, short = 'm', default_value = "auto")]
        method: String,
        /// LoRA rank (default: auto-selected)
        #[arg(long, short = 'r')]
        rank: Option<u32>,
        /// Available VRAM in GB
        #[arg(long, default_value = "16.0")]
        vram: f64,
        /// Plan mode (estimate only)
        #[arg(long)]
        plan: bool,
        /// Training data file (JSONL format)
        #[arg(long, short = 'd', value_name = "FILE")]
        data: Option<PathBuf>,
        /// Output path (adapter dir or merged model)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Adapter path for merge mode
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Merge adapter into base model
        #[arg(long)]
        merge: bool,
        /// Training epochs
        #[arg(long, default_value = "3")]
        epochs: u32,
        /// Learning rate
        #[arg(long, default_value = "0.0002")]
        learning_rate: f64,
        /// Model size for planning (e.g., "7B", "1.5B")
        #[arg(long, value_name = "SIZE")]
        model_size: Option<String>,
    },
    /// Prune model (structured/unstructured pruning) (GH-247)
    Prune {
        /// Input model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Pruning method: magnitude, structured, depth, width, wanda, sparsegpt
        #[arg(long, short = 'm', default_value = "magnitude")]
        method: String,
        /// Target pruning ratio (0-1)
        #[arg(long, default_value = "0.5")]
        target_ratio: f32,
        /// Sparsity level (0-1)
        #[arg(long, default_value = "0.0")]
        sparsity: f32,
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Layers to remove for depth pruning (e.g., "20-24")
        #[arg(long)]
        remove_layers: Option<String>,
        /// Analyze mode (identify pruning opportunities)
        #[arg(long)]
        analyze: bool,
        /// Plan mode (estimate only)
        #[arg(long)]
        plan: bool,
        /// Calibration data file
        #[arg(long, value_name = "FILE")]
        calibration: Option<PathBuf>,
    },
    /// Knowledge distillation (teacher → student) (GH-247)
    Distill {
        /// Teacher model file
        #[arg(value_name = "TEACHER")]
        teacher: PathBuf,
        /// Student model file
        #[arg(long, value_name = "FILE")]
        student: Option<PathBuf>,
        /// Training data file
        #[arg(long, short = 'd', value_name = "FILE")]
        data: Option<PathBuf>,
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Distillation strategy: standard, progressive, ensemble
        #[arg(long, default_value = "standard")]
        strategy: String,
        /// Temperature for softmax scaling
        #[arg(long, default_value = "3.0")]
        temperature: f64,
        /// Alpha weight for KL vs task loss
        #[arg(long, default_value = "0.7")]
        alpha: f64,
        /// Training epochs
        #[arg(long, default_value = "3")]
        epochs: u32,
        /// Plan mode (estimate only)
        #[arg(long)]
        plan: bool,
    },
    /// Interactive terminal UI
    Tui {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
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
    /// Interactive chat with language model
    Chat {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Sampling temperature (0 = greedy, higher = more random)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Nucleus sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Maximum tokens to generate per response
        #[arg(long, default_value = "512")]
        max_tokens: usize,
        /// System prompt to set model behavior
        #[arg(long)]
        system: Option<String>,
        /// Show inspection info (top-k probs, tokens/sec)
        #[arg(long)]
        inspect: bool,
        /// Disable GPU acceleration (use CPU)
        #[arg(long)]
        no_gpu: bool,
        /// Force GPU acceleration (requires CUDA)
        #[arg(long)]
        gpu: bool,
        /// Enable inference tracing (APR-TRACE-001)
        #[arg(long)]
        trace: bool,
        /// Trace specific steps only (comma-separated)
        #[arg(long, value_delimiter = ',')]
        trace_steps: Option<Vec<String>>,
        /// Verbose tracing
        #[arg(long)]
        trace_verbose: bool,
        /// Save trace output to JSON file
        #[arg(long, value_name = "FILE")]
        trace_output: Option<PathBuf>,
        /// Trace detail level (none, basic, layer, payload)
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,
        /// Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        profile: bool,
    },
    /// Benchmark throughput (spec H12: >= 10 tok/s)
    Bench {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Number of warmup iterations
        #[arg(long, default_value = "3")]
        warmup: usize,
        /// Number of measurement iterations
        #[arg(long, default_value = "5")]
        iterations: usize,
        /// Max tokens to generate per iteration
        #[arg(long, default_value = "32")]
        max_tokens: usize,
        /// Test prompt
        #[arg(long)]
        prompt: Option<String>,
        /// Use realizar for fast inference (vs aprender baseline)
        #[arg(long)]
        fast: bool,
        /// Benchmark specific brick
        #[arg(long)]
        brick: Option<String>,
    },
    /// Evaluate model perplexity (spec H13: PPL <= 20)
    Eval {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Dataset: wikitext-2, lambada, or custom
        #[arg(long, default_value = "wikitext-2")]
        dataset: String,
        /// Custom text (when dataset=custom)
        #[arg(long)]
        text: Option<String>,
        /// Maximum tokens to evaluate
        #[arg(long, default_value = "512")]
        max_tokens: usize,
        /// Perplexity threshold for pass/fail
        #[arg(long, default_value = "20.0")]
        threshold: f32,
    },
    /// Deep profiling with Roofline analysis
    Profile {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Layer-by-layer granular analysis
        #[arg(long)]
        granular: bool,
        /// Output format (human, json, flamegraph)
        #[arg(long, default_value = "human")]
        format: String,
        /// Focus on specific operation
        #[arg(long)]
        focus: Option<String>,
        /// Detect naive implementations
        #[arg(long)]
        detect_naive: bool,
        /// GFLOPS threshold for naive detection
        #[arg(long, default_value = "10.0")]
        threshold: f64,
        /// Compare against HuggingFace baseline
        #[arg(long)]
        compare_hf: Option<String>,
        /// Measure energy consumption (requires RAPL)
        #[arg(long)]
        energy: bool,
        /// Compute performance grade (vs Ollama baseline)
        #[arg(long)]
        perf_grade: bool,
        /// Show call graph
        #[arg(long)]
        callgraph: bool,
        /// Exit non-zero if naive implementation detected
        #[arg(long)]
        fail_on_naive: bool,
        /// Output file path for flamegraph SVG (GH-174, PMAT-182)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        // PMAT-192: CI Assertion Mode (GH-180)
        /// Enable CI mode with assertion checks (exits 1 on failure)
        #[arg(long)]
        ci: bool,
        /// Minimum throughput in tok/s (CI assertion, exits 1 if below)
        #[arg(long)]
        assert_throughput: Option<f64>,
        /// Maximum p99 latency in ms (CI assertion, exits 1 if above)
        #[arg(long)]
        assert_p99: Option<f64>,
        /// Maximum p50 latency in ms (CI assertion, exits 1 if above)
        #[arg(long)]
        assert_p50: Option<f64>,
        /// Warmup passes before measurement (default: 3)
        #[arg(long, default_value = "3")]
        warmup: usize,
        /// Measurement passes (default: 10)
        #[arg(long, default_value = "10")]
        measure: usize,
        /// Number of tokens to generate per measurement pass (default: 32)
        #[arg(long, default_value = "32")]
        tokens: usize,
        /// Compare against Ollama baseline (runs ollama for comparison)
        #[arg(long)]
        ollama: bool,
        /// Disable GPU (force CPU-only profiling)
        #[arg(long)]
        no_gpu: bool,
        /// Compare against another model format (F-PROFILE-011)
        #[arg(long, value_name = "FILE")]
        compare: Option<PathBuf>,
    },
    /// Falsifiable QA checklist for model releases
    Qa {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Minimum throughput threshold in tok/s
        #[arg(long, value_name = "TPS")]
        assert_tps: Option<f64>,
        /// Minimum speedup vs Ollama
        #[arg(long, value_name = "SPEEDUP")]
        assert_speedup: Option<f64>,
        /// Minimum GPU vs CPU speedup (F-PERF-042)
        #[arg(long, value_name = "SPEEDUP")]
        assert_gpu_speedup: Option<f64>,
        /// Skip golden output test
        #[arg(long)]
        skip_golden: bool,
        /// Skip throughput benchmark
        #[arg(long)]
        skip_throughput: bool,
        /// Skip Ollama parity comparison
        #[arg(long)]
        skip_ollama: bool,
        /// Skip GPU vs CPU speedup test (F-PERF-042)
        #[arg(long)]
        skip_gpu_speedup: bool,
        /// Skip tensor contract validation (PMAT-235)
        #[arg(long)]
        skip_contract: bool,
        /// Skip cross-format parity test (F-QUAL-032)
        #[arg(long)]
        skip_format_parity: bool,
        /// Skip PTX parity validation (GH-219)
        #[arg(long)]
        skip_ptx_parity: bool,
        /// SafeTensors model path for cross-format parity test (F-QUAL-032)
        #[arg(long, value_name = "PATH")]
        safetensors_path: Option<PathBuf>,
        /// Number of benchmark iterations
        #[arg(long, default_value = "10")]
        iterations: usize,
        /// Number of warmup iterations
        #[arg(long, default_value = "3")]
        warmup: usize,
        /// Maximum tokens to generate
        #[arg(long, default_value = "32")]
        max_tokens: usize,
        /// Output as JSON (for CI integration)
        #[arg(long)]
        json: bool,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
        /// Minimum number of gates that must execute (fail if fewer)
        #[arg(long, value_name = "N")]
        min_executed: Option<usize>,
        /// Previous QA report for regression detection
        #[arg(long, value_name = "FILE")]
        previous_report: Option<PathBuf>,
        /// Maximum allowed performance regression ratio (default: 0.10 = 10%)
        #[arg(long, value_name = "RATIO")]
        regression_threshold: Option<f64>,
        /// Skip GPU state isolation test
        #[arg(long)]
        skip_gpu_state: bool,
        /// Skip metadata plausibility validation (Bug 210, GH-222)
        #[arg(long)]
        skip_metadata: bool,
    },
    /// GPU/CPU parity check (PMAT-232: genchi genbutsu — see where GPU diverges)
    Parity {
        /// Path to GGUF model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Prompt text (default: "What is 2+2?")
        #[arg(short, long, default_value = "What is 2+2?")]
        prompt: String,
        /// Assert parity (exit non-zero on divergence)
        #[arg(long)]
        assert: bool,
    },
    /// Model-to-PTX source mapping (Mieruka: make GPU kernel dispatch visible)
    #[command(name = "ptx-map")]
    PtxMap {
        /// Path to GGUF model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Filter to specific kernel (e.g., --kernel Q4KGemv)
        #[arg(long)]
        kernel: Option<String>,
        /// Reverse lookup: kernel name -> which layers/steps use it
        #[arg(long)]
        reverse: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Full PTX snippets and detailed analysis
        #[arg(short, long)]
        verbose: bool,
        /// Show batched prefill kernel variants instead of decode
        #[arg(long)]
        prefill: bool,
    },
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
