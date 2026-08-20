use clap::Args;
use std::path::PathBuf;

/// Arguments for the `pv query` command — extracted for file health compliance.
#[derive(Args, Clone, Debug)]
pub struct QueryArgs {
    /// Search query string
    pub query: String,
    /// Directory containing contract YAML files
    #[arg(long, default_value = "contracts")]
    pub contract_dir: PathBuf,
    /// Use regex matching instead of semantic search
    #[arg(long)]
    pub regex: bool,
    /// Use literal substring matching
    #[arg(long)]
    pub literal: bool,
    /// Force case-sensitive matching
    #[arg(long)]
    pub case_sensitive: bool,
    /// Maximum number of results
    #[arg(short, long, default_value = "10")]
    pub limit: usize,
    /// Filter by obligation type
    #[arg(long)]
    pub obligation: Option<String>,
    /// Filter to contracts depending on this stem
    #[arg(long)]
    pub depends_on: Option<String>,
    /// Filter to contracts depended on by this stem
    #[arg(long)]
    pub depended_by: Option<String>,
    /// Show only contracts with unproven obligations
    #[arg(long)]
    pub unproven: bool,
    /// Minimum score threshold
    #[arg(long)]
    pub min_score: Option<f64>,
    /// Minimum proof level (L1-L5)
    #[arg(long)]
    pub min_level: Option<String>,
    /// Include contract scores in output
    #[arg(long)]
    pub score: bool,
    /// Include dependency graph info
    #[arg(long)]
    pub graph: bool,
    /// Include paper references
    #[arg(long)]
    pub paper: bool,
    /// Include proof level (L1-L5)
    #[arg(long)]
    pub proof_status: bool,
    /// Include binding status per equation
    #[arg(long)]
    pub binding_info: bool,
    /// Show only contracts with unimplemented bindings
    #[arg(long)]
    pub binding_gaps: bool,
    /// Show last git modification date
    #[arg(long)]
    pub diff: bool,
    /// Show dependency pagerank score
    #[arg(long)]
    pub pagerank: bool,
    /// Show cross-project call sites
    #[arg(long)]
    pub call_sites: bool,
    /// Show contract violations in consumer projects
    #[arg(long)]
    pub violations: bool,
    /// Show cross-project coverage matrix
    #[arg(long)]
    pub coverage_map: bool,
    /// Filter cross-project results to a named project
    #[arg(long)]
    pub project: Option<String>,
    /// Add an explicit project path to the cross-project scan
    #[arg(long)]
    pub include_project: Option<PathBuf>,
    /// Force full cross-project scan
    #[arg(long)]
    pub all_projects: bool,
    /// Filter by contract tier (1-7)
    #[arg(long)]
    pub tier: Option<u8>,
    /// Filter by kernel equivalence class (A-E)
    #[arg(long, value_name = "CLASS")]
    pub class: Option<char>,
    /// Filter: kernel|registry|model-family|pattern|schema
    #[arg(long)]
    pub kind: Option<String>,
    /// Force rebuild of the contract index
    #[arg(long)]
    pub rebuild_index: bool,
    /// Path to binding registry YAML
    #[arg(long)]
    pub binding: Option<PathBuf>,
    /// Output format: text, json, or markdown
    #[arg(short, long, default_value = "text")]
    pub format: String,
    /// Exit with status 1 if no results match
    #[arg(long)]
    pub exit_code: bool,
}
