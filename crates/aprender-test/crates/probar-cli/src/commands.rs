//! CLI command definitions using clap

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Probador: CLI for Probar - Rust-native testing framework for WASM games
#[derive(Parser, Debug)]
#[command(name = "probador")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Quiet mode (suppress non-error output)
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Color output (auto, always, never)
    #[arg(long, default_value = "auto", global = true)]
    pub color: ColorArg,

    /// Subcommand to run
    #[command(subcommand)]
    pub command: Commands,
}

/// CLI subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run tests
    Test(TestArgs),

    /// Record test execution
    Record(RecordArgs),

    /// Generate reports
    Report(ReportArgs),

    /// Generate coverage heatmaps
    Coverage(CoverageArgs),

    /// Initialize a new Probar project
    Init(InitArgs),

    /// Show configuration
    Config(ConfigArgs),

    /// Start WASM development server
    Serve(ServeArgs),

    /// Build WASM package
    Build(BuildArgs),

    /// Watch for changes and rebuild
    Watch(WatchArgs),

    /// Run state machine playbooks
    Playbook(PlaybookArgs),

    /// Run WASM compliance checks (C001-C010)
    ///
    /// Validates WASM application against Probar's compliance checklist:
    /// - C001: Code execution verified (not just mocked HTML)
    /// - C002: Console errors cause test failure
    /// - C003: Custom elements tested
    /// - C004: Both threading/non-threading modes tested
    /// - C005: Low memory scenario tested
    /// - C006: COOP/COEP headers present
    /// - C007: Replay hash matches
    /// - C008: Proper cache handling
    /// - C009: WASM under size limit
    /// - C010: No panic paths in WASM
    Comply(ComplyArgs),

    /// Verify audio-visual synchronization against EDL ground truth
    ///
    /// Extracts audio from rendered video, detects tick onsets, and compares
    /// against EDL declarations. Fails when drift exceeds tolerance.
    AvSync(AvSyncArgs),

    /// Verify audio quality (levels, clipping, silence)
    ///
    /// Extracts audio from rendered video and analyzes peak/RMS levels,
    /// detects digital clipping, and identifies silence regions.
    Audio(AudioArgs),

    /// Verify video quality (codec, resolution, FPS, duration)
    ///
    /// Probes video files with ffprobe and validates metadata against
    /// expected properties.
    Video(VideoArgs),

    /// Verify animation timing and easing curves
    ///
    /// Compares declared animation events against actual timing data
    /// from rendered output.
    Animation(AnimationArgs),

    /// Run browser/WASM stress tests (Section H: Points 116-125)
    ///
    /// Validates system stability under concurrency stress:
    /// - atomics: `SharedArrayBuffer` lock contention (> 10k ops/sec)
    /// - worker-msg: Worker message throughput (> 5k msg/sec)
    /// - render: Render loop stability (60 FPS under load)
    /// - trace: Renacer tracing overhead (< 5%)
    /// - full: All stress tests combined
    Stress(StressArgs),

    /// LLM inference testing: correctness, load testing, and reporting
    ///
    /// Test OpenAI-compatible LLM inference endpoints (realizar, ollama, llama.cpp):
    /// - test: Run correctness tests from a YAML config
    /// - load: Run concurrent load tests with latency/throughput metrics
    /// - report: Generate Markdown/JSON reports from results
    Llm(LlmArgs),
}

/// Arguments for the av-sync command
#[derive(Parser, Debug)]
pub struct AvSyncArgs {
    /// AV sync subcommand
    #[command(subcommand)]
    pub subcommand: AvSyncSubcommand,
}

/// AV sync subcommands
#[derive(Subcommand, Debug)]
pub enum AvSyncSubcommand {
    /// Check a single video against its EDL
    Check(AvSyncCheckArgs),

    /// Batch check a directory of videos
    Report(AvSyncReportArgs),
}

/// Arguments for av-sync check
#[derive(Parser, Debug)]
pub struct AvSyncCheckArgs {
    /// Path to the video file
    pub video: PathBuf,

    /// Path to EDL JSON file (default: <video>.edl.json)
    #[arg(long)]
    pub edl: Option<PathBuf>,

    /// Tolerance in milliseconds (default: 20)
    #[arg(long, default_value = "20")]
    pub tolerance_ms: f64,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: AvSyncOutputFormat,

    /// Show per-tick details
    #[arg(long)]
    pub detailed: bool,
}

/// Arguments for av-sync report
#[derive(Parser, Debug)]
pub struct AvSyncReportArgs {
    /// Directory containing rendered videos and EDL files
    pub dir: PathBuf,

    /// Tolerance in milliseconds (default: 20)
    #[arg(long, default_value = "20")]
    pub tolerance_ms: f64,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: AvSyncOutputFormat,

    /// Output file (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Output format for av-sync commands
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum AvSyncOutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output
    Json,
}

/// Arguments for the audio command
#[derive(Parser, Debug)]
pub struct AudioArgs {
    /// Audio subcommand
    #[command(subcommand)]
    pub subcommand: AudioSubcommand,
}

/// Audio subcommands
#[derive(Subcommand, Debug)]
pub enum AudioSubcommand {
    /// Check audio quality of a single video
    Check(AudioCheckArgs),
}

/// Arguments for audio check
#[derive(Parser, Debug)]
pub struct AudioCheckArgs {
    /// Path to the video file
    pub video: PathBuf,

    /// Sample rate for extraction (default: 48000)
    #[arg(long, default_value = "48000")]
    pub sample_rate: u32,

    /// Minimum RMS level in dBFS (default: -40)
    #[arg(long, default_value = "-40", allow_hyphen_values = true)]
    pub min_rms_dbfs: f64,

    /// Maximum peak level in dBFS (default: -0.1)
    #[arg(long, default_value = "-0.1", allow_hyphen_values = true)]
    pub max_peak_dbfs: f64,

    /// Fail if clipping detected (default: true)
    #[arg(long, default_value = "true")]
    pub no_clipping: bool,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: OutputFormat,
}

/// Arguments for the video command
#[derive(Parser, Debug)]
pub struct VideoArgs {
    /// Video subcommand
    #[command(subcommand)]
    pub subcommand: VideoSubcommand,
}

/// Video subcommands
#[derive(Subcommand, Debug)]
pub enum VideoSubcommand {
    /// Check video quality of a single file
    Check(VideoCheckArgs),
}

/// Arguments for video check
#[derive(Parser, Debug)]
pub struct VideoCheckArgs {
    /// Path to the video file
    pub video: PathBuf,

    /// Expected width
    #[arg(long)]
    pub width: Option<u32>,

    /// Expected height
    #[arg(long)]
    pub height: Option<u32>,

    /// Expected FPS
    #[arg(long)]
    pub fps: Option<f64>,

    /// Expected codec
    #[arg(long)]
    pub codec: Option<String>,

    /// Minimum duration in seconds
    #[arg(long)]
    pub min_duration: Option<f64>,

    /// Maximum duration in seconds
    #[arg(long)]
    pub max_duration: Option<f64>,

    /// Require audio stream
    #[arg(long)]
    pub require_audio: bool,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: OutputFormat,
}

/// Arguments for the animation command
#[derive(Parser, Debug)]
pub struct AnimationArgs {
    /// Animation subcommand
    #[command(subcommand)]
    pub subcommand: AnimationSubcommand,
}

/// Animation subcommands
#[derive(Subcommand, Debug)]
pub enum AnimationSubcommand {
    /// Verify animation timeline against observed events
    Check(AnimationCheckArgs),
}

/// Arguments for animation check
#[derive(Parser, Debug)]
pub struct AnimationCheckArgs {
    /// Path to the animation timeline JSON
    pub timeline: PathBuf,

    /// Path to the observed events JSON
    pub observed: PathBuf,

    /// Tolerance in milliseconds (default: 20)
    #[arg(long, default_value = "20")]
    pub tolerance_ms: f64,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: OutputFormat,
}

/// Output format for check commands
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum OutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output
    Json,
}

/// Arguments for the stress command
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct StressArgs {
    /// Stress test mode
    #[arg(long, default_value = "atomics")]
    pub mode: String,

    /// Test duration in seconds
    #[arg(short, long, default_value = "30")]
    pub duration: u64,

    /// Number of concurrent workers
    #[arg(short, long, default_value = "4")]
    pub concurrency: u32,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub output: String,

    /// Run atomics stress test
    #[arg(long)]
    pub atomics: bool,

    /// Run worker message stress test
    #[arg(long)]
    pub worker_msg: bool,

    /// Run render loop stress test
    #[arg(long)]
    pub render: bool,

    /// Run tracing overhead stress test
    #[arg(long)]
    pub trace: bool,

    /// Run full system stress test
    #[arg(long)]
    pub full: bool,
}

impl StressArgs {
    /// Get the stress mode from arguments
    #[must_use]
    pub fn get_mode(&self) -> String {
        if self.atomics {
            "atomics".to_string()
        } else if self.worker_msg {
            "worker-msg".to_string()
        } else if self.render {
            "render".to_string()
        } else if self.trace {
            "trace".to_string()
        } else if self.full {
            "full".to_string()
        } else {
            self.mode.clone()
        }
    }
}

/// Arguments for the llm command
#[derive(Parser, Debug)]
pub struct LlmArgs {
    /// LLM subcommand
    #[command(subcommand)]
    pub subcommand: LlmSubcommand,
}

/// LLM subcommands
#[derive(Subcommand, Debug)]
pub enum LlmSubcommand {
    /// Run correctness tests against an LLM endpoint
    Test(LlmTestArgs),
    /// Run concurrent load test against an LLM endpoint
    Load(LlmLoadArgs),
    /// Run full benchmark lifecycle (start, warmup, measure, compare, teardown)
    Bench(LlmBenchArgs),
    /// Generate reports from test results
    Report(LlmReportArgs),
    /// ML experiment tracking with data audits, budget gates, and early stopping
    Experiment(ExperimentArgs),
    /// Pre-flight data quality audit for training data
    DataAudit(DataAuditArgs),
    /// Sweep concurrency levels to find optimal operating point
    Sweep(LlmSweepArgs),
    /// Generate synthetic JSONL dataset for workload-driven benchmarking
    GenDataset(LlmGenDatasetArgs),
    /// Compute weighted performance scores for runtime comparison
    Score(LlmScoreArgs),
}

/// Arguments for `probador llm test`
#[derive(Parser, Debug)]
pub struct LlmTestArgs {
    /// Path to the YAML test configuration file
    #[arg(short, long)]
    pub config: PathBuf,

    /// Base URL of the LLM API server
    #[arg(short, long)]
    pub url: String,

    /// Model name to include in requests
    #[arg(short, long, default_value = "default")]
    pub model: String,

    /// Runtime name for reporting (e.g., realizar, ollama, llamacpp)
    #[arg(long, default_value = "unknown")]
    pub runtime_name: String,

    /// Output file path for JSON results
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for `probador llm load`
#[derive(Parser, Debug)]
pub struct LlmLoadArgs {
    /// Base URL of the LLM API server
    #[arg(short, long)]
    pub url: String,

    /// Model name to include in requests
    #[arg(short, long, default_value = "default")]
    pub model: String,

    /// Number of concurrent workers
    #[arg(short, long, default_value = "4")]
    pub concurrency: usize,

    /// Test duration (e.g., 30s, 2m, 1h)
    #[arg(short, long, default_value = "30s")]
    pub duration: String,

    /// Runtime name for reporting
    #[arg(long, default_value = "unknown")]
    pub runtime_name: String,

    /// Prompt profile: micro, short, medium, long
    #[arg(long)]
    pub prompt_profile: Option<String>,

    /// Path to YAML prompt file
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,

    /// Warmup duration before measurement (e.g., 5s, 10s). Default: no warmup.
    #[arg(long, default_value = "0s")]
    pub warmup: String,

    /// Output file path for JSON results
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Use SSE streaming for real per-token timing (TTFT, TPOT, ITL).
    /// Use --stream false to disable.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub stream: bool,

    /// Target request rate (req/s). Omit for max throughput (closed-loop).
    #[arg(long)]
    pub rate: Option<f64>,

    /// Rate distribution: poisson (default) or constant. Only used with --rate.
    #[arg(long, default_value = "poisson")]
    pub rate_distribution: String,

    /// Number of transformer layers in the model (e.g., 28 for Qwen 1.5B).
    /// Computes per-layer decode time for cross-runtime comparison.
    #[arg(long)]
    pub num_layers: Option<u32>,

    /// Inline correctness validation: none, basic, contains:X, pattern:X
    #[arg(long, default_value = "none")]
    pub validate: String,

    /// Exit non-zero if quality pass rate drops below this threshold (e.g., 0.95)
    #[arg(long)]
    pub fail_on_quality: Option<f64>,

    /// Multiplier of median ITL for spike detection (default: 5.0)
    #[arg(long, default_value = "5.0")]
    pub spike_threshold: f64,

    /// Enable GPU telemetry collection via nvidia-smi
    #[arg(long)]
    pub gpu_telemetry: bool,

    /// GPU telemetry polling interval (e.g., 1s, 2s)
    #[arg(long, default_value = "1s")]
    pub gpu_poll_interval: String,

    /// Expected GPU clock speed in MHz for throttle detection (auto-detect if omitted)
    #[arg(long)]
    pub expected_clock_mhz: Option<u32>,

    /// Skip the pre-flight health check (not recommended)
    #[arg(long)]
    pub skip_health_check: bool,

    /// Path to JSONL dataset file for workload-driven benchmarking
    #[arg(long)]
    pub dataset: Option<PathBuf>,

    /// Override `max_tokens` for all requests (e.g., --max-tokens 128)
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Max tokens distribution: uniform:MIN,MAX or fixed:N.
    /// Creates heterogeneous traffic with staggered completion times.
    /// Example: --max-tokens-distribution uniform:16,128
    #[arg(long)]
    pub max_tokens_distribution: Option<String>,
}

/// Arguments for `probador llm bench` (full benchmark lifecycle)
#[derive(Parser, Debug)]
pub struct LlmBenchArgs {
    /// Base URL of the LLM API server
    #[arg(short, long)]
    pub url: String,

    /// Model name to include in requests
    #[arg(short, long, default_value = "default")]
    pub model: String,

    /// Shell command to start the server (optional)
    #[arg(long)]
    pub start: Option<String>,

    /// Maximum time to wait for server readiness (e.g., 120s)
    #[arg(long, default_value = "120s")]
    pub health_timeout: String,

    /// Prompt profile: micro, short, medium, long
    #[arg(long, default_value = "medium")]
    pub prompt_profile: String,

    /// Path to YAML prompt file (overrides --prompt-profile)
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,

    /// Warmup duration before measurement (e.g., 10s)
    #[arg(long, default_value = "10s")]
    pub warmup: String,

    /// Per-run measurement duration (e.g., 60s)
    #[arg(short, long, default_value = "60s")]
    pub duration: String,

    /// Number of concurrent workers
    #[arg(short, long, default_value = "1")]
    pub concurrency: usize,

    /// Number of measurement runs
    #[arg(long, default_value = "3")]
    pub runs: usize,

    /// Cooldown between runs (e.g., 5s)
    #[arg(long, default_value = "5s")]
    pub cooldown: String,

    /// Baseline JSON file for regression detection
    #[arg(long)]
    pub baseline: Option<PathBuf>,

    /// Percentage threshold for regression detection (exit 1 if exceeded)
    #[arg(long)]
    pub fail_on_regression: Option<f64>,

    /// Runtime name for reporting (e.g., apr-gguf-gpu)
    #[arg(long, default_value = "unknown")]
    pub runtime_name: String,

    /// Output file path for JSON results
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Use SSE streaming for real per-token timing (TTFT, TPOT, ITL).
    /// Use --stream false to disable.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub stream: bool,

    /// Trace level for brick profiler data (GH-114): brick, step, layer
    #[arg(long)]
    pub trace_level: Option<String>,

    /// Number of transformer layers in the model (e.g., 28 for Qwen 1.5B).
    /// Computes per-layer decode time for cross-runtime comparison.
    #[arg(long)]
    pub num_layers: Option<u32>,
}

/// Arguments for `probador llm report`
#[derive(Parser, Debug)]
pub struct LlmReportArgs {
    /// Directory containing JSON result files
    #[arg(short, long)]
    pub results: PathBuf,

    /// Output path for the performance Markdown table
    #[arg(short, long, default_value = "performance.md")]
    pub output: PathBuf,

    /// Also update a README.md with the latest results
    #[arg(long)]
    pub update_readme: Option<PathBuf>,
}

/// Arguments for `probador llm score`
#[derive(Parser, Debug)]
pub struct LlmScoreArgs {
    /// Directory containing probador JSON result files
    #[arg(short, long)]
    pub results: PathBuf,

    /// Filter by concurrency level (scores computed per-concurrency)
    #[arg(short, long)]
    pub concurrency: Option<usize>,

    /// Filter results by platform (matches `runtime_name` substring)
    #[arg(long)]
    pub platform: Option<String>,

    /// Output file (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format: json, markdown, table
    #[arg(long, default_value = "table")]
    pub format: String,

    /// Exit non-zero if any runtime scores below this grade (e.g., C+)
    #[arg(long)]
    pub fail_on_grade: Option<String>,

    /// Include per-layer decode efficiency scores
    #[arg(long)]
    pub by_layer: bool,

    /// Include per-prompt-profile scores (short/medium/long)
    #[arg(long)]
    pub by_profile: bool,

    /// Include correctness scores (from inline quality validation)
    #[arg(long)]
    pub by_correctness: bool,

    /// Include per-output-length scores (short/medium/long output)
    #[arg(long)]
    pub by_output_length: bool,

    /// Include VRAM memory efficiency scores
    #[arg(long)]
    pub by_memory: bool,

    /// Include cold start time scores
    #[arg(long)]
    pub by_cold_start: bool,

    /// Include power efficiency (tok/s per watt) scores
    #[arg(long)]
    pub by_power: bool,

    /// Include concurrency scaling curve scores
    #[arg(long)]
    pub by_scaling: bool,
}

/// Arguments for `probador llm experiment`
#[derive(Parser, Debug)]
pub struct ExperimentArgs {
    /// Experiment subcommand
    #[command(subcommand)]
    pub subcommand: ExperimentSubcommand,
}

/// Experiment subcommands
#[derive(Subcommand, Debug)]
pub enum ExperimentSubcommand {
    /// Initialize a new experiment
    Init(ExperimentInitArgs),
    /// Show experiment status
    Status(ExperimentStatusArgs),
    /// Compare two runs within an experiment
    Compare(ExperimentCompareArgs),
}

/// Arguments for `probador llm experiment init`
#[derive(Parser, Debug)]
pub struct ExperimentInitArgs {
    /// Experiment name
    pub name: String,

    /// Description of the experiment
    #[arg(short, long)]
    pub description: Option<String>,

    /// Maximum GPU-hours budget
    #[arg(long)]
    pub max_gpu_hours: Option<f64>,

    /// Maximum cost budget (USD)
    #[arg(long)]
    pub max_cost_usd: Option<f64>,

    /// Cost per GPU-hour for budget calculation
    #[arg(long, default_value = "3.50")]
    pub cost_per_gpu_hour: f64,

    /// Output file for experiment state
    #[arg(short, long, default_value = "experiment.json")]
    pub output: PathBuf,
}

/// Arguments for `probador llm experiment status`
#[derive(Parser, Debug)]
pub struct ExperimentStatusArgs {
    /// Path to experiment JSON file
    #[arg(short, long, default_value = "experiment.json")]
    pub file: PathBuf,
}

/// Arguments for `probador llm experiment compare`
#[derive(Parser, Debug)]
pub struct ExperimentCompareArgs {
    /// Path to experiment JSON file
    #[arg(short = 'f', long, default_value = "experiment.json")]
    pub file: PathBuf,

    /// First run ID
    pub run_a: String,

    /// Second run ID
    pub run_b: String,

    /// Metric to compare (e.g., `eval_accuracy`, `eval_loss`)
    #[arg(short, long, default_value = "eval_loss")]
    pub metric: String,

    /// Whether lower values are better (true for loss, false for accuracy)
    #[arg(long)]
    pub lower_is_better: bool,
}

/// Arguments for `probador llm data-audit`
#[derive(Parser, Debug)]
pub struct DataAuditArgs {
    /// Path to JSONL training data file
    pub file: PathBuf,

    /// Maximum class imbalance ratio before failing (e.g., 3.0 = 3:1)
    #[arg(long, default_value = "3.0")]
    pub max_imbalance: f64,
}

/// Arguments for `probador llm sweep`
#[derive(Parser, Debug)]
pub struct LlmSweepArgs {
    /// Base URL of the LLM API server
    #[arg(short, long)]
    pub url: String,

    /// Model name to include in requests
    #[arg(short, long, default_value = "default")]
    pub model: String,

    /// Concurrency levels to sweep (comma-separated)
    #[arg(long, default_value = "1,2,4,8,16")]
    pub concurrency_levels: String,

    /// Per-level test duration (e.g., 30s)
    #[arg(short, long, default_value = "30s")]
    pub duration: String,

    /// Warmup duration per level (e.g., 5s)
    #[arg(long, default_value = "5s")]
    pub warmup: String,

    /// Use SSE streaming
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub stream: bool,

    /// Runtime name for reporting
    #[arg(long, default_value = "unknown")]
    pub runtime_name: String,

    /// P99 latency multiplier vs c=1 to declare saturated
    #[arg(long, default_value = "2.0")]
    pub saturation_threshold: f64,

    /// Stop sweep early when saturated
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub early_stop: bool,

    /// Prompt profile: micro, short, medium, long
    #[arg(long)]
    pub prompt_profile: Option<String>,

    /// Path to YAML prompt file
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,

    /// Output file for sweep results JSON
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Number of transformer layers
    #[arg(long)]
    pub num_layers: Option<u32>,
}

/// Arguments for `probador llm gen-dataset`
#[derive(Parser, Debug)]
pub struct LlmGenDatasetArgs {
    /// Distribution type: uniform, lognormal
    #[arg(long, default_value = "lognormal")]
    pub distribution: String,

    /// Mean input token count
    #[arg(long, default_value = "128")]
    pub input_mean: f64,

    /// Stddev of input token count
    #[arg(long, default_value = "64")]
    pub input_stddev: f64,

    /// Mean output (`max_tokens`) count
    #[arg(long, default_value = "128")]
    pub output_mean: f64,

    /// Stddev of output (`max_tokens`) count
    #[arg(long, default_value = "96")]
    pub output_stddev: f64,

    /// Number of entries to generate
    #[arg(long, default_value = "1000")]
    pub count: usize,

    /// Output JSONL file path
    #[arg(short, long, default_value = "dataset.jsonl")]
    pub output: PathBuf,
}

/// Arguments for the test command
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct TestArgs {
    /// Filter tests by pattern
    #[arg(short, long)]
    pub filter: Option<String>,

    /// Number of parallel test jobs
    #[arg(short = 'j', long, default_value = "0")]
    pub parallel: usize,

    /// Enable coverage collection
    #[arg(long)]
    pub coverage: bool,

    /// Enable mutation testing
    #[arg(long)]
    pub mutants: bool,

    /// Fail fast on first error
    #[arg(long)]
    pub fail_fast: bool,

    /// Watch mode - rerun on changes
    #[arg(short, long)]
    pub watch: bool,

    /// Test timeout in milliseconds
    #[arg(long, default_value = "30000")]
    pub timeout: u64,

    /// Output directory for results
    #[arg(short, long, default_value = "target/probar")]
    pub output: PathBuf,

    /// Skip compile check before running tests
    /// By default, probar runs `cargo test --no-run` to verify compilation
    /// before executing playbook tests. Use this flag to bypass that check.
    #[arg(long)]
    pub skip_compile: bool,
}

/// Arguments for the record command
#[derive(Parser, Debug)]
pub struct RecordArgs {
    /// Test to record
    pub test: String,

    /// Output format
    #[arg(short, long, default_value = "gif")]
    pub format: RecordFormat,

    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Frame rate for recording (for GIF/MP4)
    #[arg(long, default_value = "10")]
    pub fps: u8,

    /// Recording quality (1-100)
    #[arg(long, default_value = "80")]
    pub quality: u8,
}

/// Recording output format
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum RecordFormat {
    /// Animated GIF
    #[default]
    Gif,
    /// PNG screenshots
    Png,
    /// SVG vector graphics
    Svg,
    /// MP4 video
    Mp4,
}

/// Arguments for the report command
#[derive(Parser, Debug)]
pub struct ReportArgs {
    /// Report format
    #[arg(short, long, default_value = "html")]
    pub format: ReportFormat,

    /// Output directory
    #[arg(short, long, default_value = "target/probar/reports")]
    pub output: PathBuf,

    /// Open report in browser after generation
    #[arg(long)]
    pub open: bool,
}

/// Report output format
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ReportFormat {
    /// HTML report
    #[default]
    Html,
    /// `JUnit` XML
    Junit,
    /// LCOV coverage
    Lcov,
    /// Cobertura XML coverage
    Cobertura,
    /// JSON
    Json,
}

/// Arguments for the coverage command
#[derive(Parser, Debug)]
pub struct CoverageArgs {
    /// Output PNG file path
    #[arg(long)]
    pub png: Option<PathBuf>,

    /// Output JSON file path
    #[arg(long)]
    pub json: Option<PathBuf>,

    /// Color palette (viridis, magma, heat)
    #[arg(long, default_value = "viridis")]
    pub palette: PaletteArg,

    /// Include legend in PNG output
    #[arg(long)]
    pub legend: bool,

    /// Highlight coverage gaps in red
    #[arg(long)]
    pub gaps: bool,

    /// Title for the heatmap
    #[arg(long)]
    pub title: Option<String>,

    /// PNG width in pixels
    #[arg(long, default_value = "800")]
    pub width: u32,

    /// PNG height in pixels
    #[arg(long, default_value = "600")]
    pub height: u32,

    /// Coverage data input file (JSON)
    #[arg(short, long)]
    pub input: Option<PathBuf>,
}

/// Color palette argument
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum PaletteArg {
    /// Viridis (colorblind-friendly)
    #[default]
    Viridis,
    /// Magma (dark to bright)
    Magma,
    /// Heat (black-red-yellow-white)
    Heat,
}

/// Arguments for the init command
#[derive(Parser, Debug)]
pub struct InitArgs {
    /// Project directory (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Force initialization even if files exist
    #[arg(short, long)]
    pub force: bool,
}

/// Arguments for the config command
#[derive(Parser, Debug)]
pub struct ConfigArgs {
    /// Show current configuration
    #[arg(long)]
    pub show: bool,

    /// Set a configuration value (key=value)
    #[arg(long)]
    pub set: Option<String>,

    /// Reset to default configuration
    #[arg(long)]
    pub reset: bool,
}

/// Arguments for the serve command
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ServeArgs {
    /// Subcommand for serve (tree, viz, score)
    #[command(subcommand)]
    pub subcommand: Option<ServeSubcommand>,

    /// Directory to serve (default: current directory)
    #[arg(short = 'd', long = "dir", default_value = ".")]
    pub directory: PathBuf,

    /// HTTP port to listen on
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// WebSocket port for hot reload
    #[arg(long, default_value = "8081")]
    pub ws_port: u16,

    /// Open browser automatically
    #[arg(long)]
    pub open: bool,

    /// Enable CORS for cross-origin requests
    #[arg(long)]
    pub cors: bool,

    /// Enable Cross-Origin Isolation (COOP/COEP headers)
    ///
    /// Required for `SharedArrayBuffer` and parallel WASM with Web Workers.
    /// Sets Cross-Origin-Opener-Policy: same-origin and
    /// Cross-Origin-Embedder-Policy: require-corp headers.
    #[arg(long)]
    pub cross_origin_isolated: bool,

    /// Enable debug mode with verbose request/response logging
    #[arg(long)]
    pub debug: bool,

    /// Enable content linting (HTML/CSS/JS validation)
    #[arg(long)]
    pub lint: bool,

    /// Enable file watching for hot reload (default: true)
    #[arg(long, default_value = "true")]
    pub watch: bool,

    /// Validate module imports before serving
    ///
    /// Scans HTML files for JS/WASM imports and verifies they resolve
    /// with correct MIME types. Fails if any imports are broken.
    #[arg(long)]
    pub validate: bool,

    /// Monitor requests and warn about issues (404s, MIME mismatches)
    #[arg(long)]
    pub monitor: bool,

    /// Exclude directories from validation (e.g., `node_modules`)
    ///
    /// Can be specified multiple times: --exclude `node_modules` --exclude vendor
    #[arg(long, value_name = "DIR")]
    pub exclude: Vec<String>,
}

/// Serve subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum ServeSubcommand {
    /// Display file tree of served directory
    Tree(TreeArgs),

    /// Interactive TUI visualization of served files
    Viz(VizArgs),

    /// Generate project testing score (0-100)
    Score(ScoreArgs),
}

/// Arguments for the tree subcommand
#[derive(Parser, Debug, Clone)]
pub struct TreeArgs {
    /// Directory to display (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Maximum depth to display (0 = root only)
    #[arg(long)]
    pub depth: Option<usize>,

    /// Filter files by glob pattern (e.g., "*.html")
    #[arg(long)]
    pub filter: Option<String>,

    /// Show file sizes
    #[arg(long, default_value = "true")]
    pub sizes: bool,

    /// Show MIME types
    #[arg(long, default_value = "true")]
    pub mime_types: bool,
}

/// Arguments for the viz subcommand
#[derive(Parser, Debug, Clone)]
pub struct VizArgs {
    /// Directory to visualize (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// HTTP port for TUI server
    #[arg(short, long, default_value = "8080")]
    pub port: u16,
}

/// Arguments for the score subcommand
#[derive(Parser, Debug, Clone)]
pub struct ScoreArgs {
    /// Project directory to score (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Minimum score threshold (exit non-zero if below)
    #[arg(long)]
    pub min: Option<u32>,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: ScoreOutputFormat,

    /// Output HTML report to file
    #[arg(long)]
    pub report: Option<PathBuf>,

    /// Show detailed breakdown of all criteria
    #[arg(long)]
    pub detailed: bool,

    /// Append score to history file (JSONL)
    #[arg(long)]
    pub history: Option<PathBuf>,

    /// Show score trend over time
    #[arg(long)]
    pub trend: bool,

    /// Run LIVE browser validation (starts server, launches headless browser)
    ///
    /// This actually tests if the app works rather than just checking for files.
    /// Requires Chrome/Chromium installed. Recommended for accurate scoring.
    #[arg(long)]
    pub live: bool,

    /// Port for live validation server (default: random available port)
    #[arg(long, default_value = "0")]
    pub port: u16,
}

/// Output format for score command
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ScoreOutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output for CI integration
    Json,
}

/// Arguments for the build command
#[derive(Parser, Debug)]
pub struct BuildArgs {
    /// Package directory (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Build target (web, bundler, nodejs, no-modules)
    #[arg(short, long, default_value = "web")]
    pub target: WasmTarget,

    /// Build in release mode
    #[arg(long)]
    pub release: bool,

    /// Output directory (default: pkg)
    #[arg(short, long)]
    pub out_dir: Option<PathBuf>,

    /// Enable profiling (adds names section to WASM)
    #[arg(long)]
    pub profiling: bool,

    // ========================================================================
    // Zero-Artifact Architecture (PROBAR-SPEC-009-P7)
    // ========================================================================
    /// Generate web artifacts from brick definitions (PROBAR-SPEC-009-P7)
    ///
    /// Path to Rust file containing #[brick] definitions.
    /// Generates: index.html, style.css, main.js, worker.js, audio-worklet.js
    #[arg(long)]
    pub bricks: Option<PathBuf>,

    /// Application name for generated artifacts
    #[arg(long, default_value = "app")]
    pub app_name: String,

    /// WASM module path for generated HTML
    #[arg(long, default_value = "./pkg/app.js")]
    pub wasm_module: String,

    /// Model path for worker (if applicable)
    #[arg(long)]
    pub model_path: Option<String>,

    /// Page title for generated HTML
    #[arg(long)]
    pub title: Option<String>,

    /// Verify generated artifacts match brick assertions
    #[arg(long)]
    pub verify: bool,
}

/// WASM build target
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum WasmTarget {
    /// ES modules for web browsers
    #[default]
    Web,
    /// `CommonJS` for bundlers like webpack
    Bundler,
    /// `Node.js` modules
    Nodejs,
    /// No ES modules (legacy)
    NoModules,
}

impl WasmTarget {
    /// Get `wasm-pack` target string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::Bundler => "bundler",
            Self::Nodejs => "nodejs",
            Self::NoModules => "no-modules",
        }
    }
}

/// Arguments for the watch command
#[derive(Parser, Debug)]
pub struct WatchArgs {
    /// Package directory to watch (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Also start the dev server
    #[arg(long)]
    pub serve: bool,

    /// Server port (when --serve is used)
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// WebSocket port for hot reload
    #[arg(long, default_value = "8081")]
    pub ws_port: u16,

    /// Build target
    #[arg(short, long, default_value = "web")]
    pub target: WasmTarget,

    /// Build in release mode
    #[arg(long)]
    pub release: bool,

    /// Debounce delay in milliseconds
    #[arg(long, default_value = "500")]
    pub debounce: u64,
}

/// Arguments for the playbook command
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct PlaybookArgs {
    /// Playbook YAML file(s) to run
    #[arg(required = true)]
    pub files: Vec<PathBuf>,

    /// Validate playbook without running
    #[arg(long)]
    pub validate: bool,

    /// Export state machine diagram
    #[arg(long, value_enum)]
    pub export: Option<DiagramFormat>,

    /// Output file for diagram export
    #[arg(long)]
    pub export_output: Option<PathBuf>,

    /// Run mutation testing (M1-M5)
    #[arg(long)]
    pub mutate: bool,

    /// Mutation classes to run (e.g., M1,M2)
    #[arg(long, value_delimiter = ',')]
    pub mutation_classes: Option<Vec<String>>,

    /// Fail fast on first error
    #[arg(long)]
    pub fail_fast: bool,

    /// Continue on step failure
    #[arg(long)]
    pub continue_on_error: bool,

    /// Output format for results
    #[arg(short, long, default_value = "text")]
    pub format: PlaybookOutputFormat,

    /// Output directory for results
    #[arg(short, long, default_value = "target/probar/playbooks")]
    pub output: PathBuf,
}

/// Diagram export format
#[derive(ValueEnum, Clone, Debug)]
pub enum DiagramFormat {
    /// DOT format (Graphviz)
    Dot,
    /// SVG format
    Svg,
}

/// Output format for playbook results
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum PlaybookOutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output
    Json,
    /// `JUnit` XML
    Junit,
}

/// Arguments for the comply command
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ComplyArgs {
    /// Comply subcommand (check, migrate, diff, enforce, report)
    #[command(subcommand)]
    pub subcommand: Option<ComplySubcommand>,

    /// Directory to check (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Specific checks to run (e.g., C001,C002)
    #[arg(long, value_delimiter = ',')]
    pub checks: Option<Vec<String>>,

    /// Fail on first non-compliance
    #[arg(long)]
    pub fail_fast: bool,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: ComplyOutputFormat,

    /// Maximum WASM binary size in bytes (for C009)
    #[arg(long, default_value = "5242880")]
    pub max_wasm_size: usize,

    /// Enable strict mode (all checks must pass)
    #[arg(long)]
    pub strict: bool,

    /// Generate compliance report file
    #[arg(long)]
    pub report: Option<PathBuf>,

    /// Show detailed check results
    #[arg(long)]
    pub detailed: bool,
}

/// Comply subcommands (per PROBAR-SPEC-011 Section 3.1)
#[derive(Subcommand, Debug, Clone)]
pub enum ComplySubcommand {
    /// Check WASM testing compliance
    Check(ComplyCheckArgs),

    /// Migrate to latest probador standards
    Migrate(ComplyMigrateArgs),

    /// Show changelog between versions
    Diff(ComplyDiffArgs),

    /// Install WASM quality hooks
    Enforce(ComplyEnforceArgs),

    /// Generate compliance report
    Report(ComplyReportArgs),
}

/// Arguments for comply check subcommand
#[derive(Parser, Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ComplyCheckArgs {
    /// Directory to check (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Exit with error if non-compliant
    #[arg(long)]
    pub strict: bool,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: ComplyOutputFormat,

    /// Specific checks to run (e.g., C001,C002)
    #[arg(long, value_delimiter = ',')]
    pub checks: Option<Vec<String>>,

    /// Show detailed check results
    #[arg(long)]
    pub detailed: bool,
}

/// Arguments for comply migrate subcommand
#[derive(Parser, Debug, Clone)]
pub struct ComplyMigrateArgs {
    /// Directory to migrate (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Target version to migrate to
    #[arg(long)]
    pub version: Option<String>,

    /// Preview changes without applying
    #[arg(long)]
    pub dry_run: bool,

    /// Force migration even with uncommitted changes
    #[arg(long)]
    pub force: bool,
}

/// Arguments for comply diff subcommand
#[derive(Parser, Debug, Clone)]
pub struct ComplyDiffArgs {
    /// From version
    #[arg(long)]
    pub from: Option<String>,

    /// To version
    #[arg(long)]
    pub to: Option<String>,

    /// Show only breaking changes
    #[arg(long)]
    pub breaking_only: bool,
}

/// Arguments for comply enforce subcommand
#[derive(Parser, Debug, Clone)]
pub struct ComplyEnforceArgs {
    /// Directory to enforce (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Skip confirmation
    #[arg(long)]
    pub yes: bool,

    /// Remove hooks instead of installing
    #[arg(long)]
    pub disable: bool,
}

/// Arguments for comply report subcommand
#[derive(Parser, Debug, Clone)]
pub struct ComplyReportArgs {
    /// Directory to report on (default: current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: ComplyReportFormat,

    /// Output file (default: stdout)
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// Output format for comply report
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ComplyReportFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output
    Json,
    /// Markdown
    Markdown,
    /// HTML
    Html,
}

/// Output format for comply command
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ComplyOutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output for CI integration
    Json,
    /// `JUnit` XML for CI systems
    Junit,
}

/// Color argument for CLI
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ColorArg {
    /// Automatic color detection
    #[default]
    Auto,
    /// Always use colors
    Always,
    /// Never use colors
    Never,
}

impl From<ColorArg> for crate::config::ColorChoice {
    fn from(arg: ColorArg) -> Self {
        match arg {
            ColorArg::Auto => Self::Auto,
            ColorArg::Always => Self::Always,
            ColorArg::Never => Self::Never,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    mod cli_tests {
        use super::*;

        #[test]
        fn test_parse_test_command() {
            let cli = Cli::parse_from(["probar", "test"]);
            assert!(matches!(cli.command, Commands::Test(_)));
        }

        #[test]
        fn test_parse_test_with_filter() {
            let cli = Cli::parse_from(["probar", "test", "--filter", "game::*"]);
            if let Commands::Test(args) = cli.command {
                assert_eq!(args.filter, Some("game::*".to_string()));
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_test_with_parallel() {
            let cli = Cli::parse_from(["probar", "test", "-j", "4"]);
            if let Commands::Test(args) = cli.command {
                assert_eq!(args.parallel, 4);
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_test_with_coverage() {
            let cli = Cli::parse_from(["probar", "test", "--coverage"]);
            if let Commands::Test(args) = cli.command {
                assert!(args.coverage);
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_test_with_fail_fast() {
            let cli = Cli::parse_from(["probar", "test", "--fail-fast"]);
            if let Commands::Test(args) = cli.command {
                assert!(args.fail_fast);
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_record_command() {
            let cli = Cli::parse_from(["probar", "record", "test_login"]);
            if let Commands::Record(args) = cli.command {
                assert_eq!(args.test, "test_login");
            } else {
                panic!("expected Record command");
            }
        }

        #[test]
        fn test_parse_record_with_format() {
            let cli = Cli::parse_from(["probar", "record", "test_login", "--format", "png"]);
            if let Commands::Record(args) = cli.command {
                assert!(matches!(args.format, RecordFormat::Png));
            } else {
                panic!("expected Record command");
            }
        }

        #[test]
        fn test_parse_report_command() {
            let cli = Cli::parse_from(["probar", "report"]);
            assert!(matches!(cli.command, Commands::Report(_)));
        }

        #[test]
        fn test_parse_report_with_format() {
            let cli = Cli::parse_from(["probar", "report", "--format", "lcov"]);
            if let Commands::Report(args) = cli.command {
                assert!(matches!(args.format, ReportFormat::Lcov));
            } else {
                panic!("expected Report command");
            }
        }

        #[test]
        fn test_parse_init_command() {
            let cli = Cli::parse_from(["probar", "init"]);
            assert!(matches!(cli.command, Commands::Init(_)));
        }

        #[test]
        fn test_parse_config_command() {
            let cli = Cli::parse_from(["probar", "config", "--show"]);
            if let Commands::Config(args) = cli.command {
                assert!(args.show);
            } else {
                panic!("expected Config command");
            }
        }

        #[test]
        fn test_global_verbose_flag() {
            let cli = Cli::parse_from(["probar", "-vvv", "test"]);
            assert_eq!(cli.verbose, 3);
        }

        #[test]
        fn test_global_quiet_flag() {
            let cli = Cli::parse_from(["probar", "-q", "test"]);
            assert!(cli.quiet);
        }

        #[test]
        fn test_global_color_flag() {
            let cli = Cli::parse_from(["probar", "--color", "never", "test"]);
            assert!(matches!(cli.color, ColorArg::Never));
        }
    }

    mod format_tests {
        use super::*;

        #[test]
        fn test_record_format_default() {
            let format = RecordFormat::default();
            assert!(matches!(format, RecordFormat::Gif));
        }

        #[test]
        fn test_report_format_default() {
            let format = ReportFormat::default();
            assert!(matches!(format, ReportFormat::Html));
        }

        #[test]
        fn test_color_arg_conversion() {
            use crate::config::ColorChoice;

            let auto: ColorChoice = ColorArg::Auto.into();
            assert!(matches!(auto, ColorChoice::Auto));

            let always: ColorChoice = ColorArg::Always.into();
            assert!(matches!(always, ColorChoice::Always));

            let never: ColorChoice = ColorArg::Never.into();
            assert!(matches!(never, ColorChoice::Never));
        }
    }

    mod record_format_tests {
        use super::*;

        #[test]
        fn test_default() {
            let format = RecordFormat::default();
            assert!(matches!(format, RecordFormat::Gif));
        }

        #[test]
        fn test_all_variants() {
            let _ = RecordFormat::Gif;
            let _ = RecordFormat::Png;
            let _ = RecordFormat::Svg;
            let _ = RecordFormat::Mp4;
        }

        #[test]
        fn test_debug() {
            let debug = format!("{:?}", RecordFormat::Gif);
            assert!(debug.contains("Gif"));
        }

        #[test]
        fn test_clone() {
            let format = RecordFormat::Mp4;
            let cloned = format;
            assert!(matches!(cloned, RecordFormat::Mp4));
        }
    }

    mod report_format_tests {
        use super::*;

        #[test]
        fn test_default() {
            let format = ReportFormat::default();
            assert!(matches!(format, ReportFormat::Html));
        }

        #[test]
        fn test_all_variants() {
            let _ = ReportFormat::Html;
            let _ = ReportFormat::Junit;
            let _ = ReportFormat::Lcov;
            let _ = ReportFormat::Cobertura;
            let _ = ReportFormat::Json;
        }

        #[test]
        fn test_debug() {
            let debug = format!("{:?}", ReportFormat::Junit);
            assert!(debug.contains("Junit"));
        }
    }

    mod test_args_tests {
        use super::*;

        #[test]
        fn test_defaults() {
            // Verify TestArgs can be created with defaults via clap
            let args = TestArgs {
                filter: None,
                parallel: 0,
                coverage: false,
                mutants: false,
                fail_fast: false,
                watch: false,
                timeout: 30000,
                output: PathBuf::from("target/probar"),
                skip_compile: false,
            };
            assert!(!args.coverage);
            assert_eq!(args.timeout, 30000);
        }

        #[test]
        fn test_debug() {
            let args = TestArgs {
                filter: Some("test_*".to_string()),
                parallel: 4,
                coverage: true,
                mutants: false,
                fail_fast: true,
                watch: false,
                timeout: 5000,
                output: PathBuf::from("target"),
                skip_compile: false,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("TestArgs"));
        }

        #[test]
        fn test_skip_compile_flag() {
            let args = TestArgs {
                filter: None,
                parallel: 0,
                coverage: false,
                mutants: false,
                fail_fast: false,
                watch: false,
                timeout: 30000,
                output: PathBuf::from("target/probar"),
                skip_compile: true,
            };
            assert!(args.skip_compile);
        }
    }

    mod record_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = RecordArgs {
                test: "my_test".to_string(),
                format: RecordFormat::Gif,
                output: None,
                fps: 10,
                quality: 80,
            };
            assert_eq!(args.test, "my_test");
            assert_eq!(args.fps, 10);
        }

        #[test]
        fn test_debug() {
            let args = RecordArgs {
                test: "test".to_string(),
                format: RecordFormat::Png,
                output: Some(PathBuf::from("out.png")),
                fps: 30,
                quality: 100,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("RecordArgs"));
        }
    }

    mod report_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = ReportArgs {
                format: ReportFormat::Lcov,
                output: PathBuf::from("coverage"),
                open: true,
            };
            assert!(args.open);
        }

        #[test]
        fn test_debug() {
            let args = ReportArgs {
                format: ReportFormat::Html,
                output: PathBuf::from("reports"),
                open: false,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("ReportArgs"));
        }
    }

    mod init_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = InitArgs {
                path: PathBuf::from("."),
                force: false,
            };
            assert!(!args.force);
        }
    }

    mod config_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = ConfigArgs {
                show: false,
                set: None,
                reset: false,
            };
            assert!(!args.show);
        }
    }

    mod cli_additional_tests {
        use super::*;

        #[test]
        fn test_cli_debug() {
            let cli = Cli {
                verbose: 0,
                quiet: false,
                color: ColorArg::Auto,
                command: Commands::Config(ConfigArgs {
                    show: true,
                    set: None,
                    reset: false,
                }),
            };
            let debug = format!("{cli:?}");
            assert!(debug.contains("Cli"));
        }
    }

    mod coverage_tests {
        use super::*;

        #[test]
        fn test_parse_coverage_command() {
            let cli = Cli::parse_from(["probar", "coverage"]);
            assert!(matches!(cli.command, Commands::Coverage(_)));
        }

        #[test]
        fn test_parse_coverage_with_png() {
            let cli = Cli::parse_from(["probar", "coverage", "--png", "output.png"]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.png, Some(PathBuf::from("output.png")));
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_palette() {
            let cli = Cli::parse_from(["probar", "coverage", "--palette", "magma"]);
            if let Commands::Coverage(args) = cli.command {
                assert!(matches!(args.palette, PaletteArg::Magma));
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_legend() {
            let cli = Cli::parse_from(["probar", "coverage", "--legend"]);
            if let Commands::Coverage(args) = cli.command {
                assert!(args.legend);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_gaps() {
            let cli = Cli::parse_from(["probar", "coverage", "--gaps"]);
            if let Commands::Coverage(args) = cli.command {
                assert!(args.gaps);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_title() {
            let cli = Cli::parse_from(["probar", "coverage", "--title", "My Coverage"]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.title, Some("My Coverage".to_string()));
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_dimensions() {
            let cli = Cli::parse_from(["probar", "coverage", "--width", "1024", "--height", "768"]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.width, 1024);
                assert_eq!(args.height, 768);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_full_options() {
            let cli = Cli::parse_from([
                "probar",
                "coverage",
                "--png",
                "heatmap.png",
                "--palette",
                "heat",
                "--legend",
                "--gaps",
                "--title",
                "Test Coverage",
                "--width",
                "1920",
                "--height",
                "1080",
            ]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.png, Some(PathBuf::from("heatmap.png")));
                assert!(matches!(args.palette, PaletteArg::Heat));
                assert!(args.legend);
                assert!(args.gaps);
                assert_eq!(args.title, Some("Test Coverage".to_string()));
                assert_eq!(args.width, 1920);
                assert_eq!(args.height, 1080);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_palette_default() {
            let palette = PaletteArg::default();
            assert!(matches!(palette, PaletteArg::Viridis));
        }

        #[test]
        fn test_coverage_args_defaults() {
            let args = CoverageArgs {
                png: None,
                json: None,
                palette: PaletteArg::default(),
                legend: false,
                gaps: false,
                title: None,
                width: 800,
                height: 600,
                input: None,
            };
            assert_eq!(args.width, 800);
            assert_eq!(args.height, 600);
            assert!(matches!(args.palette, PaletteArg::Viridis));
        }

        #[test]
        fn test_coverage_args_debug() {
            let args = CoverageArgs {
                png: Some(PathBuf::from("test.png")),
                json: None,
                palette: PaletteArg::Magma,
                legend: true,
                gaps: true,
                title: Some("Test".to_string()),
                width: 640,
                height: 480,
                input: None,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("CoverageArgs"));
        }
    }

    mod playbook_tests {
        use super::*;

        #[test]
        fn test_parse_playbook_command() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml"]);
            assert!(matches!(cli.command, Commands::Playbook(_)));
        }

        #[test]
        fn test_parse_playbook_multiple_files() {
            let cli = Cli::parse_from(["probar", "playbook", "a.yaml", "b.yaml", "c.yaml"]);
            if let Commands::Playbook(args) = cli.command {
                assert_eq!(args.files.len(), 3);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_validate() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--validate"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.validate);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_export_dot() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--export", "dot"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(matches!(args.export, Some(DiagramFormat::Dot)));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_export_svg() {
            let cli = Cli::parse_from([
                "probar",
                "playbook",
                "test.yaml",
                "--export",
                "svg",
                "--export-output",
                "diagram.svg",
            ]);
            if let Commands::Playbook(args) = cli.command {
                assert!(matches!(args.export, Some(DiagramFormat::Svg)));
                assert_eq!(args.export_output, Some(PathBuf::from("diagram.svg")));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_mutate() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--mutate"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.mutate);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_mutation_classes() {
            let cli = Cli::parse_from([
                "probar",
                "playbook",
                "test.yaml",
                "--mutate",
                "--mutation-classes",
                "M1,M2,M3",
            ]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.mutate);
                let classes = args.mutation_classes.expect("mutation classes");
                assert_eq!(classes.len(), 3);
                assert!(classes.contains(&"M1".to_string()));
                assert!(classes.contains(&"M2".to_string()));
                assert!(classes.contains(&"M3".to_string()));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_fail_fast() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--fail-fast"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.fail_fast);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_continue_on_error() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--continue-on-error"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.continue_on_error);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_format_json() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--format", "json"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(matches!(args.format, PlaybookOutputFormat::Json));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_output_dir() {
            let cli =
                Cli::parse_from(["probar", "playbook", "test.yaml", "--output", "results/pb"]);
            if let Commands::Playbook(args) = cli.command {
                assert_eq!(args.output, PathBuf::from("results/pb"));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_playbook_args_defaults() {
            let args = PlaybookArgs {
                files: vec![PathBuf::from("test.yaml")],
                validate: false,
                export: None,
                export_output: None,
                mutate: false,
                mutation_classes: None,
                fail_fast: false,
                continue_on_error: false,
                format: PlaybookOutputFormat::default(),
                output: PathBuf::from("target/probar/playbooks"),
            };
            assert!(!args.validate);
            assert!(!args.mutate);
            assert!(matches!(args.format, PlaybookOutputFormat::Text));
        }

        #[test]
        fn test_playbook_args_debug() {
            let args = PlaybookArgs {
                files: vec![PathBuf::from("login.yaml")],
                validate: true,
                export: Some(DiagramFormat::Svg),
                export_output: Some(PathBuf::from("out.svg")),
                mutate: true,
                mutation_classes: Some(vec!["M1".to_string(), "M2".to_string()]),
                fail_fast: true,
                continue_on_error: false,
                format: PlaybookOutputFormat::Json,
                output: PathBuf::from("output"),
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("PlaybookArgs"));
        }

        #[test]
        fn test_diagram_format_debug() {
            let dot_debug = format!("{:?}", DiagramFormat::Dot);
            assert!(dot_debug.contains("Dot"));

            let svg_debug = format!("{:?}", DiagramFormat::Svg);
            assert!(svg_debug.contains("Svg"));
        }

        #[test]
        fn test_playbook_output_format_default() {
            let format = PlaybookOutputFormat::default();
            assert!(matches!(format, PlaybookOutputFormat::Text));
        }

        #[test]
        fn test_playbook_output_format_all_variants() {
            let _ = PlaybookOutputFormat::Text;
            let _ = PlaybookOutputFormat::Json;
            let _ = PlaybookOutputFormat::Junit;
        }
    }

    mod av_sync_tests {
        use super::*;

        #[test]
        fn test_parse_av_sync_check() {
            let cli = Cli::parse_from(["probar", "av-sync", "check", "video.mp4"]);
            assert!(matches!(cli.command, Commands::AvSync(_)));
        }

        #[test]
        fn test_parse_av_sync_check_with_edl() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "check",
                "video.mp4",
                "--edl",
                "video.edl.json",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert_eq!(check_args.edl, Some(PathBuf::from("video.edl.json")));
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_check_with_tolerance() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "check",
                "video.mp4",
                "--tolerance-ms",
                "30",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert!((check_args.tolerance_ms - 30.0).abs() < f64::EPSILON);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_report() {
            let cli = Cli::parse_from(["probar", "av-sync", "report", "/output/dir"]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Report(report_args) = args.subcommand {
                    assert_eq!(report_args.dir, PathBuf::from("/output/dir"));
                } else {
                    panic!("expected Report subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_check_detailed() {
            let cli = Cli::parse_from(["probar", "av-sync", "check", "video.mp4", "--detailed"]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert!(check_args.detailed);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_report_with_output() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "report",
                "/output",
                "-o",
                "report.json",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Report(report_args) = args.subcommand {
                    assert_eq!(report_args.output, Some(PathBuf::from("report.json")));
                } else {
                    panic!("expected Report subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_av_sync_output_format_default() {
            let format = AvSyncOutputFormat::default();
            assert!(matches!(format, AvSyncOutputFormat::Text));
        }

        #[test]
        fn test_parse_av_sync_check_json_format() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "check",
                "video.mp4",
                "--format",
                "json",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert!(matches!(check_args.format, AvSyncOutputFormat::Json));
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }
    }

    mod audio_tests {
        use super::*;

        #[test]
        fn test_parse_audio_check() {
            let cli = Cli::parse_from(["probar", "audio", "check", "video.mp4"]);
            assert!(matches!(cli.command, Commands::Audio(_)));
        }

        #[test]
        fn test_parse_audio_check_with_options() {
            let cli = Cli::parse_from([
                "probar",
                "audio",
                "check",
                "video.mp4",
                "--min-rms-dbfs",
                "-30",
                "--sample-rate",
                "44100",
            ]);
            if let Commands::Audio(args) = cli.command {
                if let AudioSubcommand::Check(check_args) = args.subcommand {
                    assert!((check_args.min_rms_dbfs - (-30.0)).abs() < f64::EPSILON);
                    assert_eq!(check_args.sample_rate, 44100);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected Audio command");
            }
        }

        #[test]
        fn test_output_format_default() {
            let format = OutputFormat::default();
            assert!(matches!(format, OutputFormat::Text));
        }
    }

    mod video_tests {
        use super::*;

        #[test]
        fn test_parse_video_check() {
            let cli = Cli::parse_from(["probar", "video", "check", "video.mp4"]);
            assert!(matches!(cli.command, Commands::Video(_)));
        }

        #[test]
        fn test_parse_video_check_with_expectations() {
            let cli = Cli::parse_from([
                "probar",
                "video",
                "check",
                "video.mp4",
                "--width",
                "1920",
                "--height",
                "1080",
                "--fps",
                "24",
                "--codec",
                "h264",
                "--require-audio",
            ]);
            if let Commands::Video(args) = cli.command {
                if let VideoSubcommand::Check(check_args) = args.subcommand {
                    assert_eq!(check_args.width, Some(1920));
                    assert_eq!(check_args.height, Some(1080));
                    assert!((check_args.fps.unwrap() - 24.0).abs() < f64::EPSILON);
                    assert_eq!(check_args.codec.as_deref(), Some("h264"));
                    assert!(check_args.require_audio);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected Video command");
            }
        }
    }

    mod animation_tests {
        use super::*;

        #[test]
        fn test_parse_animation_check() {
            let cli = Cli::parse_from([
                "probar",
                "animation",
                "check",
                "timeline.json",
                "observed.json",
            ]);
            assert!(matches!(cli.command, Commands::Animation(_)));
        }

        #[test]
        fn test_parse_animation_check_with_tolerance() {
            let cli = Cli::parse_from([
                "probar",
                "animation",
                "check",
                "timeline.json",
                "observed.json",
                "--tolerance-ms",
                "30",
            ]);
            if let Commands::Animation(args) = cli.command {
                if let AnimationSubcommand::Check(check_args) = args.subcommand {
                    assert!((check_args.tolerance_ms - 30.0).abs() < f64::EPSILON);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected Animation command");
            }
        }
    }

    mod stress_args_tests {
        use super::*;

        fn make_stress_args(
            atomics: bool,
            worker_msg: bool,
            render: bool,
            trace: bool,
            full: bool,
            mode: &str,
        ) -> StressArgs {
            StressArgs {
                mode: mode.to_string(),
                duration: 30,
                concurrency: 4,
                output: "text".to_string(),
                atomics,
                worker_msg,
                render,
                trace,
                full,
            }
        }

        #[test]
        fn test_get_mode_atomics() {
            let args = make_stress_args(true, false, false, false, false, "default");
            assert_eq!(args.get_mode(), "atomics");
        }

        #[test]
        fn test_get_mode_worker_msg() {
            let args = make_stress_args(false, true, false, false, false, "default");
            assert_eq!(args.get_mode(), "worker-msg");
        }

        #[test]
        fn test_get_mode_render() {
            let args = make_stress_args(false, false, true, false, false, "default");
            assert_eq!(args.get_mode(), "render");
        }

        #[test]
        fn test_get_mode_trace() {
            let args = make_stress_args(false, false, false, true, false, "default");
            assert_eq!(args.get_mode(), "trace");
        }

        #[test]
        fn test_get_mode_full() {
            let args = make_stress_args(false, false, false, false, true, "default");
            assert_eq!(args.get_mode(), "full");
        }

        #[test]
        fn test_get_mode_default() {
            let args = make_stress_args(false, false, false, false, false, "custom-mode");
            assert_eq!(args.get_mode(), "custom-mode");
        }

        #[test]
        fn test_stress_args_debug() {
            let args = make_stress_args(true, false, false, false, false, "atomics");
            let debug = format!("{args:?}");
            assert!(debug.contains("StressArgs"));
        }

        #[test]
        fn test_parse_stress_command() {
            let cli = Cli::parse_from(["probar", "stress"]);
            assert!(matches!(cli.command, Commands::Stress(_)));
        }

        #[test]
        fn test_parse_stress_with_duration() {
            let cli = Cli::parse_from(["probar", "stress", "--duration", "60"]);
            if let Commands::Stress(args) = cli.command {
                assert_eq!(args.duration, 60);
            } else {
                panic!("expected Stress command");
            }
        }

        #[test]
        fn test_parse_stress_with_concurrency() {
            let cli = Cli::parse_from(["probar", "stress", "--concurrency", "8"]);
            if let Commands::Stress(args) = cli.command {
                assert_eq!(args.concurrency, 8);
            } else {
                panic!("expected Stress command");
            }
        }

        #[test]
        fn test_parse_stress_with_atomics_flag() {
            let cli = Cli::parse_from(["probar", "stress", "--atomics"]);
            if let Commands::Stress(args) = cli.command {
                assert!(args.atomics);
                assert_eq!(args.get_mode(), "atomics");
            } else {
                panic!("expected Stress command");
            }
        }
    }
}
