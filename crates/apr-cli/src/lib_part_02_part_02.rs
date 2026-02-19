
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
        /// Skip GPU capability match gate (GH-280)
        #[arg(long)]
        skip_capability: bool,
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
