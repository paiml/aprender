//! CLI argument definitions for the batuta binary.
//!
//! Contains the `Cli` struct and `Commands` enum that define the
//! command-line interface via clap. The Oracle subcommand args are
//! in `main_oracle_args.rs` to keep file sizes manageable.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::cli;
use crate::main_oracle_args::OracleArgs;
use crate::pacha;

/// CLI argument parsing re-exports from pipeline_cmds.
pub(crate) use cli::pipeline_cmds::{OptimizationProfile, ReportFormat};

#[derive(Parser)]
#[command(name = "batuta")]
#[command(version, about = "Sovereign AI orchestration: agents, ML serving, code analysis, and transpilation", long_about = None)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Enable debug output
    #[arg(short, long, global = true)]
    pub debug: bool,

    /// Skip stack drift check (emergency use only - hidden)
    #[arg(long, global = true, hide = true)]
    pub unsafe_skip_drift_check: bool,

    /// Enforce strict drift checking (blocks on any drift)
    /// Default: tolerant in local dev (warn only), strict in CI
    #[arg(long, global = true)]
    pub strict: bool,

    /// Allow drift warnings without blocking (explicit tolerance)
    #[arg(long, global = true)]
    pub allow_drift: bool,
}

/// MCP transport mode
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub(crate) enum McpTransport {
    /// Standard I/O (JSON-RPC over stdin/stdout)
    Stdio,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Commands {
    /// Initialize a new Batuta project
    Init {
        /// Source project path
        #[arg(long, default_value = ".")]
        source: PathBuf,

        /// Output directory for Rust project
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Analyze source codebase (Phase 1: Analysis)
    Analyze {
        /// Project path
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Generate TDG score
        #[arg(long)]
        tdg: bool,

        /// Detect languages
        #[arg(long)]
        languages: bool,

        /// Analyze dependencies
        #[arg(long)]
        dependencies: bool,
    },

    /// Transpile source code to Rust (Phase 2: Transpilation)
    Transpile {
        /// Enable incremental transpilation
        #[arg(long)]
        incremental: bool,

        /// Use caching for unchanged files
        #[arg(long)]
        cache: bool,

        /// Specific modules to transpile
        #[arg(long, value_delimiter = ',')]
        modules: Option<Vec<String>>,

        /// Generate Ruchy instead of Rust
        #[arg(long)]
        ruchy: bool,

        /// Start REPL after transpilation
        #[arg(long)]
        repl: bool,
    },

    /// Optimize transpiled code (Phase 3: Optimization)
    Optimize {
        /// Enable GPU acceleration
        #[arg(long)]
        enable_gpu: bool,

        /// Enable SIMD vectorization
        #[arg(long, default_value = "true")]
        enable_simd: bool,

        /// Optimization profile
        #[arg(long, value_enum, default_value = "balanced")]
        profile: OptimizationProfile,

        /// GPU dispatch threshold (matrix size)
        #[arg(long, default_value = "500")]
        gpu_threshold: usize,
    },

    /// Validate semantic equivalence (Phase 4: Validation)
    Validate {
        /// Trace syscalls for comparison
        #[arg(long)]
        trace_syscalls: bool,

        /// Generate diff output
        #[arg(long)]
        diff_output: bool,

        /// Run original test suite
        #[arg(long)]
        run_original_tests: bool,

        /// Run benchmarks
        #[arg(long)]
        benchmark: bool,
    },

    /// Build Rust binary (Phase 5: Deployment)
    Build {
        /// Build in release mode
        #[arg(long)]
        release: bool,

        /// Target platform
        #[arg(long)]
        target: Option<String>,

        /// Build for WebAssembly
        #[arg(long)]
        wasm: bool,
    },

    /// Generate migration report
    Report {
        /// Output file path
        #[arg(long, default_value = "migration_report.html")]
        output: PathBuf,

        /// Report format
        #[arg(long, value_enum, default_value = "html")]
        format: ReportFormat,
    },

    /// Show workflow status
    Status,

    /// Reset workflow state
    Reset {
        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
    },

    /// Pattern and reference finder (PARF) for code analysis
    Parf {
        /// Path to analyze
        #[arg(default_value = "src")]
        path: PathBuf,

        /// Find all references to a symbol
        #[arg(long)]
        find: Option<String>,

        /// Detect code patterns
        #[arg(long)]
        patterns: bool,

        /// Analyze dependencies
        #[arg(long)]
        dependencies: bool,

        /// Find dead code
        #[arg(long)]
        dead_code: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: cli::parf::ParfOutputFormat,

        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Oracle Mode - Query the Sovereign AI Stack for recommendations
    Oracle {
        #[command(flatten)]
        args: OracleArgs,
    },

    /// PAIML Stack dependency orchestration
    Stack {
        #[command(subcommand)]
        command: cli::stack::StackCommand,
    },

    /// HuggingFace Hub integration
    Hf {
        #[command(subcommand)]
        command: cli::hf::HfCommand,
    },

    /// MCP (Model Context Protocol) server for tool integration
    Mcp {
        /// Transport mode
        #[arg(value_enum, default_value = "stdio")]
        transport: McpTransport,
    },

    /// Pacha model registry (ollama-like model management)
    Pacha {
        #[command(subcommand)]
        command: pacha::PachaCommand,
    },

    /// Data Platforms integration (Databricks, Snowflake, AWS, HuggingFace)
    Data {
        #[command(subcommand)]
        command: cli::data::DataCommand,
    },

    /// Visualization frameworks ecosystem (Gradio, Streamlit, Panel, Dash)
    Viz {
        #[command(subcommand)]
        command: cli::viz::VizCommand,
    },

    /// Experiment tracking frameworks comparison (MLflow replacement)
    Experiment {
        #[command(subcommand)]
        command: cli::experiment::ExperimentCommand,
    },

    /// Content creation tooling (prompt emission)
    Content {
        #[command(subcommand)]
        command: cli::content::ContentCommand,
    },

    /// Serve ML models via Realizar inference server
    ///
    /// Examples:
    ///   batuta serve pacha://llama3:8b
    ///   batuta serve ./model.gguf --port 8080
    ///   batuta serve --openai-api
    Serve {
        /// Model reference (pacha://name:version, hf://org/model, or path)
        #[arg(value_name = "MODEL")]
        model: Option<String>,

        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Enable OpenAI-compatible API at /v1/*
        #[arg(long, default_value = "true")]
        openai_api: bool,

        /// Enable hot-reload on model changes
        #[arg(long)]
        watch: bool,

        /// Start Banco local AI workbench API instead of Realizar server
        #[arg(long)]
        banco: bool,
    },

    /// Deploy ML models to production targets
    ///
    /// Examples:
    ///   batuta deploy docker pacha://llama3:8b
    ///   batuta deploy lambda my-model:v1.0
    ///   batuta deploy k8s --replicas 3
    Deploy {
        #[command(subcommand)]
        command: cli::deploy::DeployCommand,
    },

    /// Run Popperian Falsification Checklist (Sovereign AI Assurance Protocol)
    ///
    /// Evaluates a project against the 108-item checklist for scientific rigor.
    /// Toyota Way: Jidoka (automated gates), Genchi Genbutsu (evidence-based)
    ///
    /// Examples:
    ///   batuta falsify .                     # Evaluate current project
    ///   batuta falsify --critical-only       # Only CRITICAL items
    ///   batuta falsify --format json         # JSON output for CI/CD
    Falsify {
        /// Project path to evaluate
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Only evaluate CRITICAL architectural invariants (AI-01 through AI-05)
        #[arg(long)]
        critical_only: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: cli::falsify::FalsifyOutputFormat,

        /// Output file (default: stdout)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Fail on any check below this grade
        #[arg(long, default_value = "kaizen-required")]
        min_grade: String,

        /// Show verbose evidence for each check
        #[arg(long)]
        verbose: bool,
    },

    /// Proactive Bug Hunting (Popperian Falsification-Driven Defect Discovery)
    ///
    /// Systematically attempt to falsify "this code is correct" using
    /// mutation testing, SBFL, static analysis, fuzzing, and concolic execution.
    ///
    /// Examples:
    ///   batuta bug-hunter analyze .
    ///   batuta bug-hunter hunt --stack-trace crash.log
    ///   batuta bug-hunter falsify --target src/lib.rs
    ///   batuta bug-hunter fuzz --target-unsafe
    ///   batuta bug-hunter ensemble .
    #[command(name = "bug-hunter")]
    BugHunter {
        #[command(subcommand)]
        command: cli::bug_hunter::BugHunterCommand,
    },

    /// Run deterministic YAML pipelines with BLAKE3 caching
    ///
    /// Examples:
    ///   batuta playbook run pipeline.yaml
    ///   batuta playbook validate pipeline.yaml
    ///   batuta playbook status pipeline.yaml
    ///   batuta playbook run pipeline.yaml --force -p model=large
    Playbook {
        #[command(subcommand)]
        command: cli::playbook::PlaybookCommand,
    },

    /// Sovereign agent runtime (perceive-reason-act loop).
    ///
    /// Examples:
    ///   batuta agent run --manifest agent.toml --prompt "Hello"
    ///   batuta agent validate --manifest agent.toml
    ///   batuta agent chat --manifest agent.toml
    #[cfg(feature = "agents")]
    Agent {
        #[command(subcommand)]
        command: cli::agent::AgentCommand,
    },

    /// Interactive AI coding assistant (sovereign-first).
    ///
    /// Sovereign AI coding assistant — all inference local via realizar.
    ///
    /// Launch an agentic coding session with file read/write/edit,
    /// shell execution, code search, and streaming LLM output.
    /// All inference runs locally (GGUF/APR). No cloud. No API keys.
    ///
    /// Examples:
    ///   batuta code --model path/to/model.gguf
    ///   batuta code -p "Fix the auth bug"
    #[cfg(feature = "agents")]
    Code {
        /// Path to local GGUF or APR model file (prefers .apr format).
        #[arg(long)]
        model: Option<PathBuf>,

        /// Project directory (loads APR.md/CLAUDE.md from this path).
        #[arg(long, default_value = ".")]
        project: PathBuf,

        /// Resume previous session (by ID, or most recent for cwd).
        #[arg(long)]
        resume: Option<Option<String>>,

        /// Initial prompt (non-interactive if provided with -p).
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,

        /// Non-interactive: print response and exit.
        #[arg(long, short)]
        print: bool,

        /// Maximum turns before stopping.
        #[arg(long, default_value = "50")]
        max_turns: u32,

        /// Path to agent manifest (advanced; overrides defaults).
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
}
