//! Types for the contract query engine.

use crate::schema::ContractKind;
use serde::{Deserialize, Serialize};

/// How to interpret the query string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchMode {
    /// BM25 semantic ranking over corpus (default).
    Semantic,
    /// Regex match against all string fields.
    Regex,
    /// Exact substring match (case-insensitive by default).
    Literal,
}

/// A single entry in the contract index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractEntry {
    pub stem: String,
    pub path: String,
    pub description: String,
    pub equations: Vec<String>,
    pub obligation_types: Vec<String>,
    pub properties: Vec<String>,
    pub references: Vec<String>,
    pub depends_on: Vec<String>,
    pub is_registry: bool,
    #[serde(default)]
    pub kind: ContractKind,
    pub obligation_count: usize,
    pub falsification_count: usize,
    pub kani_count: usize,
    /// Concatenated searchable text for BM25.
    pub corpus_text: String,
}

/// Parameters controlling a query.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct QueryParams {
    pub query: String,
    pub mode: SearchMode,
    pub case_sensitive: bool,
    pub limit: usize,
    pub obligation_filter: Option<String>,
    pub min_score: Option<f64>,
    pub depends_on: Option<String>,
    pub depended_by: Option<String>,
    pub unproven_only: bool,
    pub show_score: bool,
    pub show_graph: bool,
    pub show_paper: bool,
    pub show_proof_status: bool,
    pub show_binding: bool,
    pub binding_path: Option<String>,
    pub binding_gaps_only: bool,
    pub show_diff: bool,
    pub show_pagerank: bool,
    pub show_call_sites: bool,
    pub show_violations: bool,
    pub show_coverage_map: bool,
    pub min_level: Option<String>,
    /// Filter cross-project results to a named project (e.g., "aprender").
    pub project_filter: Option<String>,
    /// Explicit additional project path to include in cross-project scan.
    pub include_project: Option<String>,
    /// Force cross-project scan even when no cross-project flags are set.
    pub all_projects: bool,
    /// Filter by contract tier (1-7).
    pub tier_filter: Option<u8>,
    /// Filter by kernel equivalence class (A-E).
    pub class_filter: Option<char>,
    /// Filter by contract kind (kernel, registry, model-family, pattern, schema).
    pub kind_filter: Option<ContractKind>,
}

impl Default for QueryParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            mode: SearchMode::Semantic,
            case_sensitive: false,
            limit: 10,
            obligation_filter: None,
            min_score: None,
            depends_on: None,
            depended_by: None,
            unproven_only: false,
            show_score: false,
            show_graph: false,
            show_paper: false,
            show_proof_status: false,
            show_binding: false,
            binding_path: None,
            binding_gaps_only: false,
            show_diff: false,
            show_pagerank: false,
            show_call_sites: false,
            show_violations: false,
            show_coverage_map: false,
            min_level: None,
            project_filter: None,
            include_project: None,
            all_projects: false,
            tier_filter: None,
            class_filter: None,
            kind_filter: None,
        }
    }
}

/// A single query result with relevance score and optional enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub rank: usize,
    pub stem: String,
    pub path: String,
    pub relevance: f64,
    pub description: String,
    pub kind: ContractKind,
    pub equations: Vec<String>,
    pub obligation_count: usize,
    pub references: Vec<String>,
    pub depends_on: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub depended_by: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<ScoreInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_status: Option<ProofStatusInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bindings: Vec<EquationBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<DiffInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pagerank: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub call_sites: Vec<CallSiteInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<ViolationInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub coverage_map: Vec<ProjectCoverage>,
}

/// Cross-project call site for a contract.
#[derive(Debug, Clone, Serialize)]
pub struct CallSiteInfo {
    pub project: String,
    pub file: String,
    pub line: u32,
    pub equation: Option<String>,
}

/// A contract violation detected in a consumer project.
#[derive(Debug, Clone, Serialize)]
pub struct ViolationInfo {
    pub project: String,
    pub kind: String,
    pub detail: String,
}

/// Per-project coverage summary for a contract.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectCoverage {
    pub project: String,
    pub call_sites: usize,
    pub binding_refs: usize,
    pub binding_implemented: usize,
    pub binding_total: usize,
}

/// Inline score info for enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct ScoreInfo {
    pub composite: f64,
    pub grade: String,
    pub spec_depth: f64,
    pub falsification: f64,
    pub kani: f64,
    pub lean: f64,
    pub binding: f64,
}

/// Inline proof status info for enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct ProofStatusInfo {
    pub level: String,
    pub obligations: u32,
    pub falsification_tests: u32,
    pub kani_harnesses: u32,
    pub lean_proved: u32,
}

/// Git diff summary for a contract.
#[derive(Debug, Clone, Serialize)]
pub struct DiffInfo {
    pub last_modified: String,
    pub days_ago: u64,
    pub commit_hash: String,
}

/// Binding status for a single equation.
#[derive(Debug, Clone, Serialize)]
pub struct EquationBinding {
    pub equation: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_path: Option<String>,
}

/// Output of a query execution.
#[derive(Debug, Clone, Serialize)]
pub struct QueryOutput {
    pub query: String,
    pub total_matches: usize,
    pub results: Vec<QueryResult>,
}
