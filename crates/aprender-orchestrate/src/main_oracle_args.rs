//! Oracle subcommand arguments extracted for QA-002 compliance.
//!
//! The Oracle subcommand has many flags; this struct keeps them
//! organized and keeps `main_cli.rs` within the 500-line limit.

use crate::cli;

/// All arguments for the `batuta oracle` subcommand.
#[derive(clap::Args)]
pub(crate) struct OracleArgs {
    /// Natural language query (e.g., "How do I train a model?")
    pub query: Option<String>,

    /// Get component recommendation for a problem
    #[arg(long)]
    pub recommend: bool,

    /// Specify problem type for recommendation
    #[arg(long)]
    pub problem: Option<String>,

    /// Data size for recommendations (e.g., "1M", "100K")
    #[arg(long)]
    pub data_size: Option<String>,

    /// Show integration pattern between components
    #[arg(long)]
    pub integrate: Option<String>,

    /// List capabilities of a component
    #[arg(long)]
    pub capabilities: Option<String>,

    /// List all stack components
    #[arg(long)]
    pub list: bool,

    /// Show component details
    #[arg(long)]
    pub show: Option<String>,

    /// Enter interactive mode
    #[arg(short, long)]
    pub interactive: bool,

    /// Use RAG-based retrieval from indexed documentation
    #[arg(long)]
    pub rag: bool,

    /// Index/reindex stack documentation for RAG
    #[arg(long)]
    pub rag_index: bool,

    /// Force reindex (clear cache first)
    #[arg(long)]
    pub rag_index_force: bool,

    /// Show RAG index statistics
    #[arg(long)]
    pub rag_stats: bool,

    /// Enable RAG profiling output (timing breakdown)
    #[arg(long)]
    pub rag_profile: bool,

    /// Enable RAG tracing (detailed query execution trace)
    #[arg(long)]
    pub rag_trace: bool,

    /// Generate an answer using retrieved chunks as context (requires ANTHROPIC_API_KEY)
    #[arg(long)]
    pub answer: bool,

    /// Model for answer generation (default: claude-haiku-4-5-20251001)
    #[arg(long, default_value = "claude-haiku-4-5-20251001")]
    pub answer_model: String,

    /// Show RAG dashboard (TUI)
    #[cfg(feature = "native")]
    #[arg(long)]
    pub rag_dashboard: bool,

    /// List all cookbook recipes
    #[arg(long)]
    pub cookbook: bool,

    /// Show a specific recipe by ID
    #[arg(long)]
    pub recipe: Option<String>,

    /// Find recipes by tag (e.g., "wasm", "ml", "distributed")
    #[arg(long)]
    pub recipes_by_tag: Option<String>,

    /// Find recipes by component (e.g., "aprender", "trueno")
    #[arg(long)]
    pub recipes_by_component: Option<String>,

    /// Search recipes by keyword
    #[arg(long)]
    pub search_recipes: Option<String>,

    /// Show local workspace status (~/src PAIML projects)
    #[arg(long)]
    pub local: bool,

    /// Show only dirty (uncommitted) projects
    #[arg(long)]
    pub dirty: bool,

    /// Show publish order for local projects
    #[arg(long)]
    pub publish_order: bool,

    /// Search functions via PMAT quality-annotated code search
    #[arg(long)]
    pub pmat_query: bool,

    /// Project path for PMAT query (defaults to current directory)
    #[arg(long)]
    pub pmat_project_path: Option<String>,

    /// Maximum number of PMAT results to return
    #[arg(long, default_value = "10")]
    pub pmat_limit: usize,

    /// Minimum TDG grade filter (A, B, C, D, F)
    #[arg(long)]
    pub pmat_min_grade: Option<String>,

    /// Maximum cyclomatic complexity filter
    #[arg(long)]
    pub pmat_max_complexity: Option<u32>,

    /// Include source code in PMAT results
    #[arg(long)]
    pub pmat_include_source: bool,

    /// Search across all local PAIML projects
    #[arg(long)]
    pub pmat_all_local: bool,

    /// Generate a Coursera reading asset (banner, reflection, key-concepts, vocabulary)
    #[arg(long, value_enum)]
    pub asset: Option<cli::oracle::CourseraAssetType>,

    /// Transcript file or directory for Coursera asset generation
    #[arg(long)]
    pub transcript: Option<std::path::PathBuf>,

    /// Output file for banner PNG/SVG
    #[arg(short, long)]
    pub output: Option<std::path::PathBuf>,

    /// Override topic for arXiv citation lookup
    #[arg(long)]
    pub topic: Option<String>,

    /// Course title for banner generation
    #[arg(long)]
    pub course_title: Option<String>,

    /// Enrich results with relevant arXiv papers (builtin curated database)
    #[arg(long)]
    pub arxiv: bool,

    /// Fetch live arXiv papers instead of builtin database
    #[arg(long)]
    pub arxiv_live: bool,

    /// Maximum arXiv papers to show (default: 3)
    #[arg(long, default_value = "3")]
    pub arxiv_max: usize,

    /// Output format
    #[arg(long, value_enum, default_value = "text")]
    pub format: cli::oracle::OracleOutputFormat,
}
