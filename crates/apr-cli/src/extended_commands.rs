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
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,
        /// Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        profile: bool,
        /// PMAT-488: Compute backend override (cuda, cpu, wgpu)
        #[arg(long, value_name = "BACKEND")]
        backend: Option<String>,
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
        #[arg(long, value_delimiter = ',', default_value = "50,95,99")]
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
        #[arg(long, default_value = "20.0")]
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
        /// Device for inference: "cpu" (default) or "cuda" (GPU-accelerated, ALB-089)
        #[arg(long, default_value = "cpu")]
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
    /// Forensics / parity / visualization subcommands (probar, compare-hf,
    /// hex, tree, flow, qualify).
    #[command(flatten)]
    Forensics(ForensicsCommands),
    /// Training and data-pipeline subcommands (train, pretrain, tokenize,
    /// data, pipeline, diagnose).
    #[command(flatten)]
    Training(TrainingCommands),
    /// CRUX-series lint subcommands (ollama-chat-lint, dry-sampling-lint,
    /// awq-lint, oom-lint, tool-use-lint, gbnf-lint).
    #[command(flatten)]
    Lint(LintCommands),
    /// Publishing, conversion, and analysis tools
    #[command(flatten)]
    Tools(ToolCommands),
}


include!("extended_commands_runs.rs");

include!("extended_commands_lints.rs");

include!("extended_commands_training.rs");

include!("extended_commands_forensics.rs");
