/// Attention implementation under test for `apr kernel parity --impl`.
///
/// `Flash2` names the pinned `hf-kernels-community:flash-attn2@<sha>` CUDA
/// kernel. This binary embeds no such kernel, so selecting it is REFUSED —
/// never quietly answered by `Tiled` under flash2's name, which is the
/// fabricated-provenance failure CRUX-L-02 exists to prevent.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum KernelImpl {
    /// In-tree tiled online-softmax kernel (`realizar::brick::FlashAttentionBrick`).
    Tiled,
    /// Pinned hf-kernels-community flash-attn2 CUDA kernel (not embedded here).
    Flash2,
}

/// Reference implementation for `apr kernel parity --ref`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum KernelRef {
    /// Materialised-score softmax attention, computed in f32 on the CPU.
    Naive,
}

/// 2-D projection for `apr debug embed-viz --projection`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum EmbedProjection {
    /// Exact PCA onto the top 2 principal components (deterministic).
    Pca,
    /// Seeded Johnson–Lindenstrauss random projection (deterministic in --seed).
    Random,
    /// Not implemented in this binary — selecting it is refused, not substituted.
    Umap,
}

/// Extended CLI commands (analysis, profiling, QA, benchmarks, and advanced tools).
///
/// Flattened into `Commands` via `#[command(flatten)]` so all subcommands remain
/// top-level from the user's perspective (e.g., `apr chat`, `apr profile`).
#[derive(Subcommand, Debug)]
pub enum ExtendedCommands {
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
        #[arg(long, value_name = "LEVEL", default_value = "basic", value_parser = TRACE_LEVEL_VALUES)]
        trace_level: String,
        /// Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        profile: bool,
        // PMAT-488 / #2583: shared `--backend` declaration (see `BackendArg`).
        #[command(flatten)]
        backend: BackendArg,
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
        /// Comma-separated latency percentile points for JSON output
        /// (CRUX-E-07). Default: `50,95,99`. Values must be in (0, 100].
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "50,95,99",
            value_parser = crate::commands::bench::parse_percentile
        )]
        percentiles: Vec<f64>,
    },
    /// Evaluate model perplexity (spec H13: PPL <= 20) or classification metrics
    Eval {
        /// Path to model file or checkpoint directory
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
        #[arg(long, default_value = "20.0",
              value_parser = commands::threshold_arg::parse_tolerance_f32)]
        threshold: f32,
        /// Task type: omit for perplexity, "classify" for classification eval
        #[arg(long)]
        task: Option<String>,
        /// Test data file (JSONL) for classification evaluation
        #[arg(long, value_name = "FILE")]
        data: Option<PathBuf>,
        /// Model size hint: "0.5B", "tiny" (for classification eval)
        #[arg(long)]
        model_size: Option<String>,
        /// Number of output classes (default: 5)
        #[arg(long, default_value = "5")]
        num_classes: usize,
        /// Generate HuggingFace model card (README.md) in checkpoint dir
        #[arg(long)]
        generate_card: bool,
        /// Device for inference: "cpu" (default) or "cuda" (GPU-accelerated, ALB-089).
        /// Applies to --task humaneval/mbpp; perplexity evaluation is CPU-only.
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "cuda"])]
        device: String,
        /// Number of samples per problem for pass@k (ALB-088, default: 1)
        #[arg(long, default_value = "1")]
        samples: usize,
        /// Sampling temperature (0.0 = greedy, 0.8 = standard for pass@k>1)
        #[arg(long, default_value = "0.0")]
        temperature: f32,
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
        /// Achieved-GFLOPS floor below which the run is reported as naive
        #[arg(long, default_value = "10.0",
              value_parser = commands::threshold_arg::parse_tolerance)]
        threshold: f64,
        /// [NOT IMPLEMENTED — accepted and ignored] Compare against HuggingFace baseline
        #[arg(long)]
        compare_hf: Option<String>,
        /// [NOT IMPLEMENTED — accepted and ignored] Measure energy consumption (requires RAPL)
        #[arg(long)]
        energy: bool,
        /// Compute performance grade (vs Ollama baseline)
        #[arg(long)]
        perf_grade: bool,
        /// [NOT IMPLEMENTED — accepted and ignored] Show call graph
        #[arg(long)]
        callgraph: bool,
        /// Exit non-zero if naive implementation detected (implies --detect-naive)
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
        #[arg(long, value_parser = commands::threshold_arg::parse_tolerance)]
        assert_throughput: Option<f64>,
        /// Maximum p99 latency in ms (CI assertion, exits 1 if above)
        #[arg(long, value_parser = commands::threshold_arg::parse_tolerance)]
        assert_p99: Option<f64>,
        /// Maximum p50 latency in ms (CI assertion, exits 1 if above)
        #[arg(long, value_parser = commands::threshold_arg::parse_tolerance)]
        assert_p50: Option<f64>,
        /// Warmup passes before measurement (default: 3)
        #[arg(long, default_value = "3")]
        warmup: usize,
        /// Measurement passes (default: 10)
        #[arg(long, default_value = "10")]
        measure: usize,
        /// Tokens generated per measurement pass — GPU and --ollama paths only;
        /// the CPU per-operation profiler measures one forward pass per pass
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
        #[arg(long, value_name = "TPS",
              value_parser = commands::threshold_arg::parse_tolerance)]
        assert_tps: Option<f64>,
        /// Minimum speedup vs Ollama
        #[arg(long, value_name = "SPEEDUP",
              value_parser = commands::threshold_arg::parse_tolerance)]
        assert_speedup: Option<f64>,
        /// Minimum GPU vs CPU speedup (F-PERF-042)
        #[arg(long, value_name = "SPEEDUP",
              value_parser = commands::threshold_arg::parse_tolerance)]
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
        #[arg(long, value_name = "RATIO",
              value_parser = commands::threshold_arg::parse_fraction)]
        regression_threshold: Option<f64>,
        /// Skip GPU state isolation test
        #[arg(long)]
        skip_gpu_state: bool,
        /// Skip metadata plausibility validation (Bug 210, GH-222)
        #[arg(long)]
        skip_metadata: bool,
        /// Skip GPU capability match gate (GH-280)
        #[arg(long)]
        skip_capability: bool,
        /// Assert classifier head presence and shape (F-CLASS-004)
        #[arg(long)]
        assert_classifier_head: bool,
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
    /// PTX analysis and bug detection (register pressure, roofline)
    ///
    /// #2399 finding 1: on a build without the analyzer this line is the only
    /// thing a user sees before running the command, so it has to say so.
    #[cfg_attr(feature = "trueno-explain", command(name = "ptx"))]
    #[cfg_attr(
        not(feature = "trueno-explain"),
        command(
            name = "ptx",
            about = "PTX analysis and bug detection [unavailable in this build: cargo install aprender --features ptx]"
        )
    )]
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
    /// ML tuning: LoRA/QLoRA configuration, memory planning, and HPO (GH-176, SPEC-TUNE-2026-001)
    #[cfg(feature = "training")]
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
        /// Task type for HPO: classify (SPEC-TUNE-2026-001)
        #[arg(long)]
        task: Option<String>,
        /// Number of HPO trials (default: 10)
        #[arg(long, default_value = "10")]
        budget: usize,
        /// HPO search strategy: tpe, grid, random
        #[arg(long, default_value = "tpe")]
        strategy: String,
        /// HPO scheduler: asha, median, none
        #[arg(long, default_value = "asha")]
        scheduler: String,
        /// Scout mode: 1 epoch per trial for fast exploration
        #[arg(long)]
        scout: bool,
        /// Training data file for HPO (JSONL format)
        #[arg(long, value_name = "FILE")]
        data: Option<PathBuf>,
        /// Number of output classes for classification
        #[arg(long, default_value = "5")]
        num_classes: usize,
        /// Model size hint for HPO (e.g., "0.5B", "1.5B")
        #[arg(long)]
        model_size: Option<String>,
        /// Warm-start from scout phase results directory
        #[arg(long, value_name = "DIR")]
        from_scout: Option<PathBuf>,
        /// Maximum epochs per trial (full mode, default: 20)
        #[arg(long, default_value = "20")]
        max_epochs: usize,
        /// Maximum wall-clock time (e.g., "8h", "30m")
        #[arg(long)]
        time_limit: Option<String>,
    },
    /// Attach live TUI to a running training session
    #[cfg(feature = "training")]
    Monitor {
        /// Experiment output directory (same as finetune -o)
        #[arg(value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Refresh interval in milliseconds
        #[arg(long, default_value = "500")]
        refresh_ms: u64,
        /// Compact display mode
        #[arg(long)]
        compact: bool,
        /// Output JSON lines instead of TUI (for LLM agents and CI)
        #[arg(long)]
        json: bool,
        /// Output format: tui (default), json, text
        #[arg(long, default_value = "tui")]
        format: String,
    },
    /// List, show, and compare training experiment runs
    #[cfg(feature = "training")]
    Runs {
        #[command(subcommand)]
        command: RunsCommands,
    },
    /// Interactive experiment browser (TUI with loss curves)
    #[cfg(feature = "training")]
    Experiment {
        #[command(subcommand)]
        command: ExperimentCommands,
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
        #[arg(long, requires = "headless")]
        json: bool,
        /// Output file path (requires --headless)
        #[arg(long, value_name = "FILE", requires = "headless")]
        output: Option<PathBuf>,
        /// CI mode: exit non-zero if thresholds are not met or the report status is FAIL
        #[arg(long)]
        ci: bool,
        /// Minimum throughput threshold in tok/s (for --ci)
        #[arg(long, value_name = "TOK_S",
              value_parser = commands::threshold_arg::parse_tolerance)]
        throughput: Option<f64>,
        /// Minimum brick score threshold 0-100 (for --ci)
        #[arg(long, value_name = "SCORE")]
        brick_score: Option<u32>,
        /// Number of warmup iterations before measurement (must be >= 1)
        #[arg(long, default_value = "10", value_parser = parse_cbtop_warmup)]
        warmup: usize,
        /// Number of measurement iterations (must be >= 1)
        #[arg(long, default_value = "100", value_parser = parse_cbtop_iterations)]
        iterations: usize,
        /// PAR-100: Enable speculative decoding benchmark
        #[arg(long)]
        speculative: bool,
        /// PAR-100: Number of tokens to draft speculatively (default: 4, must be >= 1)
        #[arg(long, default_value = "4", value_parser = parse_cbtop_speculation_k)]
        speculation_k: usize,
        /// PAR-099: Path to draft model for speculative decoding
        #[arg(long, value_name = "DRAFT_MODEL")]
        draft_model: Option<PathBuf>,
        /// PAR-102: Number of concurrent requests (must be >= 1)
        #[arg(long, default_value = "1", value_parser = parse_cbtop_concurrent)]
        concurrent: usize,
        /// Use simulated data (for CI testing only)
        #[arg(long)]
        simulated: bool,
    },
    /// Test harness for web, LLM, media and replay — powered by probador.
    ///
    /// Named for what it tests, not for the act of testing. `probar` is Spanish
    /// for "to try"; it named the VERB, so `apr probar --help` told a reader
    /// nothing about the subject. This follows the precedent already set by
    /// `apr data` ("Data quality pipeline ... powered by alimentar"): a plain
    /// noun for the user-facing command, the Spanish name kept for the engine
    /// and credited in the description.
    ///
    /// The harness covers four distinct things, and the subcommands group by
    /// what is UNDER TEST rather than by verb:
    ///
    ///   web     the WASM/browser build and its runtime behaviour
    ///           (serve, build, watch, comply, stress)
    ///   llm     inference correctness, throughput and cost against an endpoint
    ///           (test, load, bench, sweep, score, experiment, data-audit)
    ///   media   rendered output against ground truth
    ///           (av-sync, audio, video, animation)
    ///   replay  the runner itself — recording, state machines, reporting
    ///           (record, playbook, coverage, report)
    ///
    /// Routed today: `tensor` (PMAT-481 visual regression) and `llm bench`
    /// (GH-876 Milestone 2). The rest land as they are delegated to the
    /// probador library. Renaming now costs one path — after those land it is
    /// a breaking change across the whole testing surface.
    ///
    /// `apr probar` stays as a hidden alias so existing scripts keep working.
    #[command(alias = "probar")]
    Test {
        #[command(subcommand)]
        command: TestSubcommand,
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
        #[arg(long, default_value = "1e-5",
              value_parser = commands::threshold_arg::parse_tolerance)]
        threshold: f64,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// CRUX-K-11: parse Ollama-style Modelfile DSL into apr config.
    Modelfile {
        #[command(subcommand)]
        command: ModelfileSubcommand,
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
        ///
        /// #2394 finding 15: this was a `String` that the dispatcher parsed
        /// with `.unwrap_or(TreeFormat::Ascii)`, so `--format bogusvalue`
        /// silently rendered ascii and exited 0 — a typo'd `--format josn` in
        /// a pipeline produced a tree instead of JSON, with no warning. Parsing
        /// at the CLI boundary makes the unparseable value unrepresentable
        /// downstream: clap rejects it before any command runs.
        #[arg(long, default_value = "ascii")]
        format: crate::commands::tree::TreeFormat,
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
        #[arg(long, value_parser = commands::threshold_arg::parse_tolerance_f32)]
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
        /// Initial weights from a pretrained APR file
        /// (contract `apr-pretrain-from-init-v1`). Per spec §49's
        /// MODEL-2 pretrained-init pivot: when present, load weights
        /// from `<PATH>` instead of random-init. Composes with
        /// `--mode finetune` (canonical) or `--mode from-scratch`
        /// (allowed but non-canonical — emits a warning). Missing,
        /// corrupted, or arch-mismatched APR files exit non-zero
        /// before step 1 (no silent random-init fallback).
        #[arg(long, value_name = "PATH")]
        init: Option<PathBuf>,
        /// SPEC §83 P0-J: bypass the Chinchilla compute-optimal hard
        /// gate (`chinchilla-gate-v1`). Default is fail-fast when
        /// D/N < 10× (severely under-provisioned per Hoffmann et al.
        /// 2022). Pass this flag to acknowledge the under-provisioning
        /// and proceed anyway (e.g. for ablation studies, resumed
        /// runs, or smoke tests).
        #[arg(long, action = clap::ArgAction::SetTrue)]
        force_under_provisioned: bool,
        /// SPEC §84 P2-F: shared held-out validation shard.
        ///
        /// When provided, the val-loss eval reads `HELD_OUT_BATCHES`
        /// batches from this separate `.bin`-shards directory instead
        /// of stealing the first 16 batches of `--dataset`. This makes
        /// `val_loss` comparable across runs whose `--dataset`
        /// composition changes (P2-C's audit-falsified result was
        /// confounded by val sets being drawn from different corpus
        /// distributions — qwen-v2 = codeparrot only, qwen-v3 =
        /// codeparrot + the-stack-dedup).
        ///
        /// Path semantics: directory of `.bin` shards (same format as
        /// `--dataset`). Operator tokenizes the held-out corpus
        /// independently via `apr tokenize encode-corpus --max-docs N`
        /// to a separate output dir, then passes that dir here. The
        /// shard contract is `contracts/dataset-thestack-python-v1.yaml`.
        ///
        /// When omitted, falls back to the historical "first 16
        /// batches of --dataset" behaviour for backwards compatibility.
        #[arg(long, value_name = "DIR")]
        val_shard: Option<PathBuf>,
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
    /// Lint an Ollama /api/chat response for schema + NDJSON invariants (CRUX-C-04)
    OllamaChatLint {
        /// Path to captured /api/chat response (JSON object, or NDJSON if --stream)
        #[arg(long, value_name = "FILE")]
        response_file: PathBuf,
        /// Treat input as NDJSON stream (one frame per line)
        #[arg(long)]
        stream: bool,
    },
    /// Lint an Ollama /api/chat function-calling response (CRUX-I-04)
    OllamaToolsLint {
        /// Path to captured /api/chat response (JSON object, or NDJSON if --stream)
        #[arg(long, value_name = "FILE")]
        response_file: PathBuf,
        /// Captured request JSON, required unless --stream — supplies the
        /// tool-name allowlist (every called tool name must appear in
        /// request.tools[*].function.name)
        #[arg(long, value_name = "FILE")]
        request_file: Option<PathBuf>,
        /// Treat input as NDJSON stream (one frame per line)
        #[arg(long)]
        stream: bool,
    },
    /// Lint a captured DRY-sampling observation (CRUX-C-23)
    DrySamplingLint {
        /// Path to observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured AWQ quality/compression/flags observation (CRUX-B-08)
    AwqLint {
        /// Path to captured AWQ observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured FP8 (E4M3) round-trip + SM-capability observation (CRUX-B-11)
    Fp8Lint {
        /// Path to captured observation JSON (frobenius, capability blocks)
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured NF4 codebook/roundtrip/storage/parity observation (CRUX-B-10)
    Nf4Lint {
        /// Path to captured NF4 observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured GPTQ compression/cosine/flags observation (CRUX-B-09)
    GptqLint {
        /// Path to captured GPTQ observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured CUDA OOM postmortem report (CRUX-F-13)
    OomLint {
        /// Path to captured OOM postmortem JSON (e.g. /tmp/apr-oom-<ts>.json)
        #[arg(long, value_name = "FILE")]
        report_file: PathBuf,
        /// Optional captured stderr log to verify the OOM_REPORT breadcrumb
        #[arg(long, value_name = "FILE")]
        stderr_file: Option<PathBuf>,
    },
    /// Lint a captured NCCL failure-diagnostics JSON from stderr (CRUX-F-15)
    NcclDiagLint {
        /// Path to captured stderr JSON diagnostic
        #[arg(long, value_name = "FILE")]
        diag_file: PathBuf,
        /// Optional observed exit code (gate: >= 128 = NCCL class)
        #[arg(long, value_name = "I32")]
        exit_code: Option<i32>,
        /// Require the `suggest` field to cite an nvidia.com / NVIDIA/nccl URL
        #[arg(long)]
        require_doc_link: bool,
    },
    /// Lint an externally captured ReAct loop trace JSON (CRUX-I-06 — no apr producer yet)
    ReactTraceLint {
        /// Path to captured trace JSON
        #[arg(long, value_name = "FILE")]
        trace_file: PathBuf,
        /// Optional max_iterations budget the trace was produced under
        #[arg(long, value_name = "N")]
        max_iterations: Option<i64>,
        /// Require the scratchpad to parse cleanly as Thought/Action/Observation blocks
        #[arg(long)]
        require_grammar: bool,
    },
    /// Lint a captured `$APR_TRACE_DIR` hang stack-dump directory (CRUX-F-14)
    HangTraceLint {
        /// Path to the captured trace directory
        #[arg(long, value_name = "DIR")]
        trace_dir: PathBuf,
        /// Inspection mode: `timeout` (expects per-rank dumps) or `success` (expects empty dir)
        #[arg(long, value_name = "MODE", default_value = "timeout")]
        mode: String,
        /// Expected world_size when mode=timeout (number of rank{N}.py.txt files)
        #[arg(long, value_name = "N", default_value_t = 2)]
        world_size: usize,
        /// Actual exit code from the run under inspection (for exit-code gate)
        #[arg(long, value_name = "I32")]
        exit_code: Option<i32>,
        /// Expected exit code (typically 124 for timeout, 1 for other error, 0 for success)
        #[arg(long, value_name = "I32")]
        expected_exit_code: Option<i32>,
    },
    /// Lint two externally captured DDP metrics JSONs, N=1 and N=k (CRUX-D-11 — no apr producer yet)
    DdpMetricsLint {
        /// Path to N=1 metrics JSON
        #[arg(long, value_name = "FILE")]
        metrics_1gpu_file: PathBuf,
        /// Path to N=world_size metrics JSON
        #[arg(long, value_name = "FILE")]
        metrics_ngpu_file: PathBuf,
        /// World size used for --metrics-ngpu-file run (>= 2)
        #[arg(long, value_name = "N")]
        world_size: i64,
        /// Scaling-efficiency floor (default 0.85, PyTorch DDP convention)
        #[arg(long, value_name = "F", default_value_t = 0.85,
              value_parser = commands::threshold_arg::parse_fraction)]
        scaling_floor: f64,
        /// Loss-parity relative tolerance (default 0.01)
        #[arg(long, value_name = "F", default_value_t = 0.01,
              value_parser = commands::threshold_arg::parse_tolerance)]
        loss_tolerance: f64,
    },
    /// Dataset inspection tools (CRUX-H-13)
    Dataset {
        #[command(subcommand)]
        command: DatasetCommands,
    },
    /// Kernel-level parity measurements (CRUX-L-02)
    Kernel {
        #[command(subcommand)]
        command: KernelCommands,
    },
    /// Lint an audio-inspect JSON body, e.g. from
    /// `apr dataset audio-inspect clip.wav --format json -o audio.json` (CRUX-H-13)
    AudioInspectLint {
        /// Path to the JSON body written by `apr dataset audio-inspect --format json`
        #[arg(long, value_name = "FILE")]
        json_file: PathBuf,
        /// Optional expected sample_rate (typically the `--resample-to` arg)
        #[arg(long, value_name = "U32")]
        expected_sample_rate: Option<u32>,
        /// Optional expected channel count (1 = mono after --mono)
        #[arg(long, value_name = "U32")]
        expected_channels: Option<u32>,
    },
    /// Lint attention parity + provenance JSON, e.g. from
    /// `apr kernel parity --impl tiled --ref naive --json -o parity.json` (CRUX-L-02)
    AttnParityLint {
        /// Parity JSON body (`max_abs_diff`, `cosine_sim`), as written by
        /// `apr kernel parity --json`
        #[arg(long, value_name = "FILE")]
        parity_file: Option<PathBuf>,
        /// Provenance JSON body (`attn_impl`, `kernel_source`, `fallback`).
        /// `apr kernel parity --json` writes both gates' fields into one body,
        /// so the same file may be passed here and to --parity-file
        #[arg(long, value_name = "FILE")]
        provenance_file: Option<PathBuf>,
        /// head_dim refusal JSON, as written by
        /// `apr kernel parity --impl flash2 --head-dim 96 --json` (which exits non-zero)
        #[arg(long, value_name = "FILE")]
        head_dim_error_file: Option<PathBuf>,
        /// Max absolute diff tolerance (default 5e-3, FlashAttention-2 bound)
        #[arg(long, value_name = "F", default_value_t = 5e-3,
              value_parser = commands::threshold_arg::parse_tolerance)]
        tol_abs: f64,
        /// Min cosine similarity floor (default 0.9999)
        #[arg(long, value_name = "F", default_value_t = 0.9999,
              value_parser = commands::threshold_arg::parse_cosine)]
        tol_cos: f64,
    },
    /// Lint an externally captured attention dump (CRUX-F-17 — no apr producer yet)
    AttnVizLint {
        /// Path to attention dump in JSON form (4-D [layers][heads][rows][cols] floats)
        #[arg(long, value_name = "FILE")]
        attn_file: Option<PathBuf>,
        /// Path to HTML heatmap output
        #[arg(long, value_name = "FILE")]
        html_file: Option<PathBuf>,
        /// Minimum <svg|<canvas open-tag count expected in HTML (|layers|*|heads|)
        #[arg(long, value_name = "N", default_value_t = 1)]
        expected_heatmaps: usize,
        /// Row-softmax normalization tolerance (default 1e-5)
        #[arg(long, value_name = "F64", default_value_t = 1e-5,
              value_parser = commands::threshold_arg::parse_tolerance)]
        tolerance: f64,
        /// Causal-mask zero epsilon (default 1e-9)
        #[arg(long, value_name = "F64", default_value_t = 1e-9,
              value_parser = commands::threshold_arg::parse_tolerance)]
        epsilon: f64,
    },
    /// Lint an externally captured check-finite error and/or coverage JSON (CRUX-F-11 — no apr producer yet)
    CheckFiniteLint {
        /// Externally captured check-finite stderr JSON from a poisoned model
        #[arg(long, value_name = "FILE")]
        error_file: Option<PathBuf>,
        /// Externally captured check-finite layer-coverage JSON
        #[arg(long, value_name = "FILE")]
        list_file: Option<PathBuf>,
        /// Minimum layer-coverage count when `--list-file` is supplied (default 100)
        #[arg(long, value_name = "N", default_value_t = 100)]
        min_layers: usize,
    },
    /// Lint an embedding-projection CSV, e.g. from
    /// `apr debug embed-viz --model model.apr --seed 42 -o emb.csv` (CRUX-F-18)
    EmbedVizLint {
        /// Path to the `token_id,token_str,x,y` CSV written by `apr debug embed-viz`
        #[arg(long, value_name = "FILE")]
        csv_file: PathBuf,
        /// Expected row count == vocab_size (optional)
        #[arg(long, value_name = "N")]
        expected_vocab_size: Option<usize>,
        /// Second CSV from a rerun at the same --seed, for the determinism gate (optional)
        #[arg(long, value_name = "FILE")]
        csv_file_b: Option<PathBuf>,
    },
    /// Lint an externally captured token-selection JSONL trace (CRUX-F-19 — no apr producer yet)
    ExplainTokenLint {
        /// Path to captured JSONL body (one sampled-token record per line)
        #[arg(long, value_name = "FILE")]
        jsonl_file: PathBuf,
        /// Tolerance for `Σ post_prob ≈ 1.0` (default 1e-5)
        #[arg(long, value_name = "F64", default_value_t = 1e-5,
              value_parser = commands::threshold_arg::parse_tolerance)]
        tolerance: f64,
        /// Assert greedy decoding: sampled_id must equal argmax(pre_prob)
        #[arg(long)]
        require_greedy: bool,
    },
    /// Lint a captured GPU memory Chrome Trace Event Format JSON (CRUX-F-07)
    GpuMemtraceLint {
        /// Path to an externally captured GPU-memory Chrome Trace JSON (no apr producer yet)
        #[arg(long, value_name = "FILE")]
        trace_file: PathBuf,
    },
    /// Lint a captured KV-cache utilization timeline (CRUX-F-06)
    KvTimelineLint {
        /// Path to an externally captured KV-cache timeline JSON body (no apr producer yet)
        #[arg(long, value_name = "FILE")]
        timeline_file: PathBuf,
        /// Preemption threshold (default 0.95, vLLM canonical)
        #[arg(long, value_name = "FRACTION", default_value_t = 0.95,
              value_parser = commands::threshold_arg::parse_fraction)]
        preempt_threshold: f64,
    },
    /// Lint a captured OTLP/JSON ExportTraceServiceRequest body (CRUX-K-08).
    ///
    /// At least one gate flag is required: every check is opt-in, so a bare
    /// invocation would check nothing and exit 0 for any parseable JSON.
    OtlpLint {
        /// Path to captured OTLP/JSON export body
        #[arg(long, value_name = "FILE")]
        otlp_file: PathBuf,
        /// Require at least one `apr.inference` span to be present
        #[arg(long)]
        require_apr_span: bool,
        /// Require gen_ai.* and apr.tokens.* attribute keys on some span
        #[arg(long)]
        require_genai_attrs: bool,
        /// Verify W3C trace-context propagation: expect this 32-hex traceId
        #[arg(long, value_name = "HEX32")]
        expect_trace_id: Option<String>,
    },
    /// Lint a captured Prometheus /metrics response (CRUX-K-07)
    PrometheusLint {
        /// Path to captured /metrics response body (text/plain; version=0.0.4)
        #[arg(long, value_name = "FILE")]
        metrics_file: PathBuf,
        /// Optional captured Content-Type header to verify against version=0.0.4
        #[arg(long, value_name = "HEADER")]
        content_type: Option<String>,
        /// Require the K-07 metric set (apr_num_requests_running, ...) to be present
        #[arg(long)]
        require_k07_metrics: bool,
    },
    /// Lint a captured OpenAI tool-use response (CRUX-C-11)
    ToolUseLint {
        /// Path to captured OpenAI tool-use response JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a GBNF grammar-constrained observation (CRUX-C-10)
    GbnfLint {
        /// Path to captured GBNF observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a typical-p sampling observation (CRUX-C-22)
    TypicalPLint {
        /// Path to captured typical-p observation JSON, with any of the
        /// sections range/identity/mass/sort/renorm
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Gradient-norm telemetry analysis (CRUX-F-09)
    GradNorm {
        /// Path to JSON file of per-step grad-norm records
        #[arg(long, value_name = "FILE")]
        history_file: PathBuf,
        /// Maximum allowed clipped grad-norm (for cap-violation check)
        #[arg(long, value_name = "M",
              value_parser = commands::threshold_arg::parse_tolerance)]
        max_grad_norm: Option<f64>,
        /// Rolling-median window size for spike detection (in steps)
        #[arg(long, default_value = "16")]
        spike_window: usize,
        /// Multiplier threshold for spike detection
        #[arg(long, default_value = "10.0",
              value_parser = commands::threshold_arg::parse_tolerance)]
        spike_multiplier: f64,
    },
    /// Lint a captured registry byte-quota observation (CRUX-A-22)
    RegistryQuotaLint {
        /// Path to captured quota/atomic/ceiling observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured imatrix calibration observation (CRUX-B-07)
    ImatrixLint {
        /// Path to captured imatrix observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured /v1/embeddings observation (CRUX-C-13)
    EmbeddingsLint {
        /// Path to captured /v1/embeddings observation JSON, with any of the
        /// sections shape/determinism/usage/flag
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured Hub+local unified-search merge observation (CRUX-A-23)
    UnifiedSearchLint {
        /// Path to captured unified-search observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured `apr rm` / externally captured gc blob-GC observation (CRUX-A-25)
    RmGcLint {
        /// Path to captured rm/gc observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured APR_MODELS shared-cache observation (CRUX-A-21)
    SharedCacheLint {
        /// Path to captured dedup/permission observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Perplexity classifier (CRUX-E-02)
    Ppl {
        /// JSON file containing an array of per-token natural-log
        /// probabilities (e.g. `[-1.2, -0.5, -2.1, ...]`). Required.
        #[arg(long, value_name = "FILE")]
        log_probs_file: PathBuf,
    },
    /// Validate dequant→requant metadata preservation (CRUX-B-19)
    QuantPreservationLint {
        /// Reference GGUF (pre-roundtrip)
        #[arg(long, value_name = "REF.gguf")]
        reference: PathBuf,
        /// Requantized GGUF (post-roundtrip)
        #[arg(long, value_name = "REQ.gguf")]
        requant: PathBuf,
    },
    /// Split a safetensors file into shards + weight-map index (CRUX-B-05)
    Shard {
        /// Single-file safetensors model to split
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Maximum size of each shard (e.g. 5GB, 500MB, 1.5GiB)
        #[arg(long, value_name = "SIZE", default_value = "5GB")]
        max_shard_size: String,
        /// Output directory for shards + model.safetensors.index.json
        #[arg(short, long, value_name = "DIR")]
        output: PathBuf,
        /// #2392: Overwrite an existing shard set in the output directory
        #[arg(short, long)]
        force: bool,
    },
    /// Reconstruct a single safetensors file from a sharded directory (CRUX-B-05)
    Unshard {
        /// Sharded directory containing model.safetensors.index.json
        #[arg(value_name = "DIR")]
        input: PathBuf,
        /// Output single-file safetensors path
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
        /// #2392: Overwrite an existing output file (refused without it)
        #[arg(short, long)]
        force: bool,
    },
    /// Publishing, conversion, and analysis tools
    #[command(flatten)]
    Tools(ToolCommands),
    /// Score a query/passage pair (or rank multiple passages) with a BERT
    /// cross-encoder loaded from an APR v2 file (GH-326 Phase 3).
    ///
    /// Wraps `aprender_core::models::bert::CrossEncoder::load_from_reader`
    /// + `score()`. The APR must contain the canonical HF BERT tensor
    /// names (see `models::bert::expected_bert_tensor_names`).
    ///
    /// Tokenisation is NOT applied here — caller passes pre-tokenised
    /// `input_ids` + `token_type_ids` as comma-delimited u32 lists. A
    /// dedicated tokeniser-aware mode is Phase 3b follow-up scope.
    Rerank {
        /// Path to the APR file containing the cross-encoder weights.
        #[arg(value_name = "MODEL")]
        model: PathBuf,
        /// Pre-tokenised input ids (comma-separated `u32`s). Mutually
        /// exclusive with `--query`+`--passage`+`--vocab` (Phase 3b).
        /// Example: `--input-ids 101,2024,102,3456,102` for `[CLS] q [SEP] p [SEP]`.
        #[arg(long, value_name = "IDS")]
        input_ids: Option<String>,
        /// Pre-tokenised token-type ids (comma-separated `u32`s).
        /// Same length as `--input-ids`. 0 for query side, 1 for passage.
        #[arg(long, value_name = "IDS")]
        token_type_ids: Option<String>,
        /// Phase 3b — query text. Pair with `--passage` + `--vocab` to enable
        /// in-process WordPiece tokenisation. The tokeniser builds
        /// `[CLS] query [SEP] passage [SEP]` with `token_type_ids = 0` for
        /// the query side and `1` for the passage side.
        #[arg(long, value_name = "TEXT")]
        query: Option<String>,
        /// Phase 3b — passage text. Required when `--query` is supplied
        /// in single-pair mode (use `--passages` for batch ranking).
        #[arg(long, value_name = "TEXT")]
        passage: Option<String>,
        /// Phase 5 — batch ranking mode (#326). Passage candidates to
        /// score against `--query`. May be supplied multiple times:
        /// `apr rerank model.apr --query "..." --passages "p1" --passages "p2"`.
        /// Mutually exclusive with `--passage`. Output is one
        /// `score[i]` line per passage in input order, OR a JSON array
        /// of `{passage, logit, score}` objects sorted by descending
        /// score when `--sort` is set.
        #[arg(long, value_name = "TEXT")]
        passages: Vec<String>,
        /// Phase 5 — sort batch output by descending score (highest
        /// relevance first). Only meaningful with `--passages` and
        /// `--json`. Default: preserve input order.
        #[arg(long)]
        sort: bool,
        /// Phase 5 — limit to top-K passages after sorting. Implies
        /// `--sort`. Default 0 (no limit).
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Phase 3b — path to a WordPiece `vocab.txt` (one token per line,
        /// line index = token id). Required when `--query` is supplied.
        /// Must contain entries for `[CLS]`, `[SEP]`, and `[UNK]`.
        /// Phase 4 accepts HuggingFace `tokenizer.json` (extension-detected).
        #[arg(long, value_name = "FILE")]
        vocab: Option<PathBuf>,
        /// Override hidden_dim (default: 384 / MiniLM-L-6).
        #[arg(long, default_value_t = 384)]
        hidden_dim: usize,
        /// Override num_layers (default: 6 / MiniLM-L-6).
        #[arg(long, default_value_t = 6)]
        num_layers: usize,
        /// Override num_heads (default: 12 / MiniLM-L-6).
        #[arg(long, default_value_t = 12)]
        num_heads: usize,
        /// Override intermediate_dim (default: 1536 / MiniLM-L-6).
        #[arg(long, default_value_t = 1536)]
        intermediate_dim: usize,
        /// Override vocab_size (default: 30522 / bert-base-uncased).
        #[arg(long, default_value_t = 30522)]
        vocab_size: usize,
        /// Override max_position_embeddings (default: 512).
        #[arg(long, default_value_t = 512)]
        max_position_embeddings: usize,
        /// Override type_vocab_size (default: 2).
        #[arg(long, default_value_t = 2)]
        type_vocab_size: usize,
        /// Number of labels in the classifier head (default: 1 for
        /// regression-style relevance scoring).
        #[arg(long, default_value_t = 1)]
        num_labels: usize,
        /// Load the optional BERT pooler dense layer (default: true).
        /// Cross-encoders that skip the pooler should pass `--with-pooler false`.
        ///
        /// Takes an optional value: `--with-pooler` (bare) and an omitted flag
        /// both mean true; `--with-pooler false` / `--with-pooler=false` turn
        /// the pooler off. A bare `bool` here would compile to a SetTrue switch
        /// and make the documented `false` unreachable.
        #[arg(
            long,
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
        )]
        with_pooler: bool,
        /// Emit the raw logit instead of the sigmoid-mapped relevance score.
        #[arg(long)]
        raw_logit: bool,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Produce sentence embeddings from a BERT bi-encoder (GH-326 Phase 6).
    ///
    /// First-stage dense retrieval companion to `apr rerank`. Loads an
    /// encoder-only BertModel (e.g. `sentence-transformers/all-MiniLM-L6-v2`),
    /// tokenises the input text with WordPiece, runs the full encoder
    /// forward, then pools the hidden states with one of:
    ///   `--pool cls`  — take the [CLS] hidden state
    ///   `--pool mean` — mean over non-padding token positions (default;
    ///                   sentence-transformers convention)
    /// Optionally L2-normalises the result (`--normalize`, default true,
    /// matches sentence-transformers).
    Embed {
        /// Path to the APR file containing the encoder weights (BertModel).
        #[arg(value_name = "MODEL")]
        model: PathBuf,
        /// Text to encode. Repeatable: `apr embed model.apr --text "a" --text "b" --vocab tok.json`.
        #[arg(long, value_name = "TEXT")]
        text: Vec<String>,
        /// Phase 7 (GH-326) — read texts from a file, one per line.
        /// Concatenated with `--text` inputs in order: `--text` first,
        /// then `--text-file` rows. Blank lines and lines starting
        /// with `#` are skipped. Useful for RAG-style first-stage
        /// retrieval where the second-stage rerank candidate set
        /// (50-100 documents) is the embed input.
        #[arg(long, value_name = "FILE")]
        text_file: Option<PathBuf>,
        /// Path to a WordPiece `vocab.txt` or HF `tokenizer.json`.
        #[arg(long, value_name = "FILE")]
        vocab: PathBuf,
        /// Pooling strategy (`cls` or `mean`). Default: `mean`
        /// (matches sentence-transformers convention).
        #[arg(long, default_value = "mean")]
        pool: String,
        /// L2-normalise the output embedding. Default: true (matches
        /// sentence-transformers convention). Pass `--normalize false`
        /// to keep raw magnitudes.
        ///
        /// Takes an optional value: `--normalize` (bare) and an omitted flag
        /// both mean true; `--normalize false` / `--normalize=false` keep the
        /// raw magnitudes. A bare `bool` here would compile to a SetTrue switch
        /// and make the documented `false` unreachable.
        ///
        /// Because the value is optional, do not place a bare `--normalize`
        /// immediately before the MODEL positional — write
        /// `apr embed MODEL --normalize` or `--normalize=true MODEL`.
        #[arg(
            long,
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
        )]
        normalize: bool,
        /// Override hidden_dim (default: 384 / MiniLM).
        #[arg(long, default_value_t = 384)]
        hidden_dim: usize,
        /// Override num_layers (default: 6 / MiniLM-L-6).
        #[arg(long, default_value_t = 6)]
        num_layers: usize,
        /// Override num_heads.
        #[arg(long, default_value_t = 12)]
        num_heads: usize,
        /// Override intermediate_dim.
        #[arg(long, default_value_t = 1536)]
        intermediate_dim: usize,
        /// Override vocab_size.
        #[arg(long, default_value_t = 30522)]
        vocab_size: usize,
        /// Override max_position_embeddings.
        #[arg(long, default_value_t = 512)]
        max_position_embeddings: usize,
        /// Override type_vocab_size.
        #[arg(long, default_value_t = 2)]
        type_vocab_size: usize,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
}

/// Subcommands for `apr dataset` — dataset inspection (aprender#2377 finding 3).
///
/// `audio-inspect` is the PRODUCER for `apr audio-inspect-lint`: CRUX-H-13
/// shipped the lint with help pointing at a command the binary did not have,
/// so its gates had never run on real data.
#[derive(Subcommand, Debug)]
pub enum DatasetCommands {
    /// Decode an uncompressed RIFF/WAVE file and report its measured shape and
    /// amplitude extrema — the observation `apr audio-inspect-lint` reads.
    ///
    /// Supports PCM u8/i16/i24/i32 and IEEE float32. Compressed containers
    /// (FLAC, MP3, Ogg) and codecs it cannot decode are REFUSED with a non-zero
    /// exit; no resampling and no channel mixdown are performed, so the reported
    /// `sample_rate` and `channels` are always the file's own.
    AudioInspect {
        /// Path to the .wav file to decode
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Output format: `json` for the lint-readable body, `text` for humans
        #[arg(long, value_name = "FORMAT", default_value = "text",
              value_parser = ["json", "text"])]
        format: String,
        /// Write the observation here instead of stdout
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Overwrite an existing --output file (refused without it)
        #[arg(short, long)]
        force: bool,
    },
}

/// Subcommands for `apr kernel` — kernel-level measurements (aprender#2377 finding 3).
///
/// `parity` is the PRODUCER for `apr attn-parity-lint`: CRUX-L-02 shipped the
/// lint with help pointing at `apr kernel parity`, which did not exist.
#[derive(Subcommand, Debug)]
pub enum KernelCommands {
    /// Measure a tiled attention kernel against a naive reference on seeded
    /// Q/K/V, emitting the parity + provenance body `apr attn-parity-lint` reads.
    ///
    /// `--impl tiled` runs the in-tree `realizar::brick::FlashAttentionBrick`
    /// online-softmax kernel. `--impl flash2` means the pinned
    /// `hf-kernels-community:flash-attn2@<sha>` CUDA kernel, which this binary
    /// does not embed: asking for it is REFUSED with a non-zero exit rather
    /// than answered by a different kernel under a borrowed name.
    Parity {
        /// Attention implementation under test
        #[arg(long = "impl", value_name = "IMPL", value_enum, default_value_t = KernelImpl::Tiled)]
        kernel: KernelImpl,
        /// Reference implementation to compare against
        #[arg(long = "ref", value_name = "REF", value_enum, default_value_t = KernelRef::Naive)]
        reference: KernelRef,
        /// KV cache length to attend over
        #[arg(long, value_name = "N", default_value_t = 128)]
        seq_len: usize,
        /// Number of query heads
        #[arg(long, value_name = "N", default_value_t = 8)]
        num_heads: usize,
        /// Number of key/value heads (GQA groups when smaller than --num-heads)
        #[arg(long, value_name = "N", default_value_t = 8)]
        num_kv_heads: usize,
        /// Per-head dimension. flash2 dispatches only 64 or 128
        #[arg(long, value_name = "N", default_value_t = 64)]
        head_dim: usize,
        /// Seed pinning the Q/K/V draw, so a run is reproducible
        #[arg(long, value_name = "N", default_value_t = 0)]
        seed: u64,
        /// Emit the observation as JSON (required to capture it for the lint)
        #[arg(long)]
        json: bool,
        /// Write the observation here instead of stdout
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Overwrite an existing --output file (refused without it)
        #[arg(short, long)]
        force: bool,
    },
}

#[cfg(feature = "training")]
/// Subcommands for `apr runs` — experiment run management (ALB-050/051)
#[derive(Subcommand, Debug)]
pub enum RunsCommands {
    /// List all training experiment runs (with inline loss sparklines)
    Ls {
        /// Directory to scan for experiments (default: current dir)
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Read from global experiment registry (~/.entrenar/experiments.db)
        #[arg(long)]
        global: bool,
        /// Filter by status: all, pending, running, completed, failed, cancelled
        #[arg(long, default_value = "all")]
        status: String,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Maximum number of runs to show
        #[arg(long, default_value = "50")]
        limit: usize,
    },
    /// Show detailed metrics for a specific run (with braille loss curve)
    Show {
        /// Run ID
        #[arg(value_name = "RUN_ID")]
        run_id: String,
        /// Directory containing experiment DB
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Read from global registry
        #[arg(long)]
        global: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Compare two runs side-by-side (loss curves, config diff, metrics)
    Diff {
        /// First run ID
        #[arg(value_name = "RUN_A")]
        run_a: String,
        /// Second run ID
        #[arg(value_name = "RUN_B")]
        run_b: String,
        /// Directory containing experiment DB
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Read from global registry
        #[arg(long)]
        global: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

#[cfg(feature = "training")]
/// Subcommands for `apr experiment` — interactive experiment browser (ALB-024)
#[derive(Subcommand, Debug)]
pub enum ExperimentCommands {
    /// Browse experiment history with interactive TUI (loss curves, params)
    View {
        /// Path to experiment database file
        #[arg(long, value_name = "FILE")]
        db: Option<PathBuf>,
        /// Read from global experiment registry (~/.entrenar/experiments.db)
        #[arg(long)]
        global: bool,
        /// Output as JSON (non-interactive)
        #[arg(long)]
        json: bool,
    },
}

/// CRUX-K-11: Subcommands for `apr modelfile`.
#[derive(Subcommand, Debug)]
pub enum ModelfileSubcommand {
    /// Parse an Ollama-style Modelfile and emit the parsed config.
    ///
    /// Grammar: `FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`, `LICENSE`,
    /// `MESSAGE`, `ADAPTER` directives. Triple-quoted blocks supported.
    /// Directive names are case-insensitive. Unknown directives raise
    /// `file:line:col` errors.
    Parse {
        /// Path to the Modelfile
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Output format: `json` or `human`
        #[arg(long, default_value = "json")]
        format: String,
    },
}

/// GH-876: Subcommands for `apr test` — consolidates the probador testing
/// framework under `apr`. Milestone 1 shipped `tensor`; Milestone 2 adds
/// `llm bench`, the benchmark lifecycle harness. Remaining probador
/// subcommands follow as separate PRs that delegate to the same library.
///
/// MILESTONE 2 IS NOT A PORT. `crates/aprender-test-lib/src/llm/` already
/// carries the entire llm module — 11 files, `pub mod llm` unconditional in
/// lib.rs — so the code has shipped inside every `apr` binary since APR-MONO
/// and nothing could reach it. That is a dark capability (F11), and the fix is
/// a CLI surface over the library already present, not 11,000 lines of new
/// code.
///
/// It matters because the tool was built for exactly this job: measuring
/// aprender's inference throughput against a comparator. Without the surface,
/// a benchmark gets hand-rolled instead — which is what happened on
/// 2026-08-24, in a protocol whose own rule is "never dogfood by hand".
#[derive(Subcommand, Debug)]
pub enum TestSubcommand {
    /// Export tensor activations for visual regression testing (PMAT-481).
    ///
    /// Generates JSON/PNG per-layer test artifacts that can be compared
    /// against a golden reference directory to detect regressions in
    /// model behavior after weight updates, quantization, or refactors.
    Tensor {
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
        #[arg(long, default_value = "0.98",
              value_parser = commands::threshold_arg::parse_cosine_f32)]
        tolerance: f32,
    },

    /// LLM inference testing: correctness, load testing, and reporting.
    ///
    /// Delegates to the in-tree `aprender-test-lib` llm module, which is where
    /// this measurement logic has always lived. It was not previously reachable
    /// from `apr`: apr-cli's only key on that crate was a DEV-dependency
    /// WITHOUT its `llm` feature, and `commands/serve_loadtest.rs` — the one
    /// file that used it — was absent from `commands/mod.rs`, so it never
    /// compiled.
    Llm {
        #[command(subcommand)]
        command: LlmSubcommand,
    },
}

/// GH-876 Milestone 2 — `apr test llm <SUB>`.
#[derive(Subcommand, Debug)]
pub enum LlmSubcommand {
    /// Run the full benchmark lifecycle: start, warm up, measure, compare,
    /// tear down.
    ///
    /// Drives an OpenAI-compatible endpoint, so both a runtime under test and
    /// a comparator are measured by the SAME client with the SAME prompts —
    /// which is what makes the two numbers comparable at all.
    Bench {
        /// Endpoint to measure.
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// Model name sent in the request body. Most OpenAI-compatible
        /// servers ignore it; `apr serve` and vLLM do not.
        #[arg(short, long, default_value = "default")]
        model: String,
        /// Command that starts the runtime. Omit to measure something already
        /// running.
        #[arg(long)]
        start: Option<String>,
        /// Seconds to wait for the endpoint to become healthy.
        #[arg(long, default_value = "120")]
        health_timeout: u64,
        /// Warm-up seconds, discarded from the measurement.
        ///
        /// NOT decoration. A first measurement of apr on GB10 with two warmups
        /// produced a BIMODAL series whose median sat in the empty gap between
        /// the two modes, inflating a 1.21x result to 2.91x.
        #[arg(long, default_value = "10")]
        warmup: u64,
        /// Measurement seconds per run.
        #[arg(short, long, default_value = "30")]
        duration: u64,
        /// Concurrent request streams.
        #[arg(short, long, default_value = "1")]
        concurrency: usize,
        /// Number of measured runs; the report aggregates across them.
        #[arg(long, default_value = "3")]
        runs: usize,
        /// Cooldown seconds between runs.
        #[arg(long, default_value = "5")]
        cooldown: u64,
        /// Label recorded in the report, e.g. `apr-cuda` or `llamacpp-39173bcac`.
        #[arg(long, default_value = "apr")]
        runtime_name: String,
        /// Prior report to compare against.
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Fractional regression that fails the run, e.g. 0.10 for 10%.
        #[arg(long)]
        fail_on_regression: Option<f64>,
        /// Write the JSON report here.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Use streaming responses (measures TTFT and TPOT).
        #[arg(long)]
        stream: bool,
        /// Prompt profile: micro (10 in / 1 out), short (32/32),
        /// medium (128/128), long (512/256).
        ///
        /// The workload IS half the measurement. Two runtimes compared on
        /// different profiles are not compared at all, so this is a declared
        /// knob with a recorded value rather than a constant buried in a
        /// helper.
        #[arg(long, default_value = "medium")]
        profile: String,
        /// Prompt file (JSON array of chat requests). Overrides `--profile`.
        #[arg(long)]
        prompts: Option<PathBuf>,
    },
}

/// Reject `0` for an `apr cbtop` knob that shapes the measurement.
///
/// Shared by `--iterations`, `--warmup`, `--concurrent` and `--speculation-k`.
/// All four decide what the emitted report is a report OF, and a zero in any
/// of them yields a populated report describing something other than the run
/// the caller asked for.
fn parse_cbtop_positive(s: &str, unit: &str, why: &str) -> std::result::Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|_| format!("`{s}` is not a valid {unit}"))?;
    if n == 0 {
        return Err(format!("must be at least 1 — {why}"));
    }
    Ok(n)
}

/// Parse `apr cbtop --iterations`, rejecting 0.
///
/// With zero measurement iterations every brick keeps zero samples, so its
/// measured time is 0.0µs, its gap factor is 0.00x and it scores a perfect
/// 100/A — a green report attesting to measurements that never ran. Reject the
/// value where the user typed it rather than emitting the fabricated report.
fn parse_cbtop_iterations(s: &str) -> std::result::Result<usize, String> {
    parse_cbtop_positive(
        s,
        "iteration count",
        "a zero-iteration run measures nothing and would report every brick as a perfect \
         100/A from zero samples",
    )
}

/// Parse `apr cbtop --warmup`, rejecting 0.
///
/// #2731. `--warmup` and `--iterations` are the two flags that decide how the
/// number was measured, and only one of them had a validator: `--iterations 0`
/// exited 2 while `--warmup 0` exited 0 and reported its result as a
/// measurement.
///
/// Zero warmup does not empty the report the way zero iterations does — it
/// FILLS it, from the cold-start regime: first-touch allocation, CUDA context
/// creation, kernel JIT / graph capture, cache-cold weights. The arithmetic is
/// real and the regime is wrong, which is materially harder to notice than a
/// missing number. This project has cold-vs-warm deltas on record measured in
/// orders of magnitude, so a zero-warmup run and a ten-warmup run are not two
/// samples of one quantity.
///
/// Rejecting it here is the same remedy the sibling flag already had:
/// refuse the value where the user typed it rather than emit a report whose
/// regime nobody can recover afterwards.
fn parse_cbtop_warmup(s: &str) -> std::result::Result<usize, String> {
    parse_cbtop_positive(
        s,
        "warmup iteration count",
        "a zero-warmup run measures the cold-start regime (allocation, CUDA context, kernel \
         JIT, cache-cold weights), not steady state, and reports it as though it were",
    )
}

/// Parse `apr cbtop --concurrent`, rejecting 0.
///
/// PAR-102 builds the batch as `(0..config.concurrent)`, so zero issues no
/// requests at all and the aggregate-throughput report is computed over an
/// empty batch.
fn parse_cbtop_concurrent(s: &str) -> std::result::Result<usize, String> {
    parse_cbtop_positive(
        s,
        "concurrency",
        "a zero-concurrency run issues no requests, so the aggregate throughput it reports is \
         computed over an empty batch",
    )
}

/// Parse `apr cbtop --speculation-k`, rejecting 0.
///
/// PAR-100 drafts `k` tokens per step; with `k = 0` nothing is drafted and the
/// run is no longer the speculative-decoding benchmark it is labelled as.
fn parse_cbtop_speculation_k(s: &str) -> std::result::Result<usize, String> {
    parse_cbtop_positive(
        s,
        "draft length",
        "drafting zero tokens is not speculative decoding, so the run would be labelled \
         speculative while measuring the ordinary path",
    )
}
