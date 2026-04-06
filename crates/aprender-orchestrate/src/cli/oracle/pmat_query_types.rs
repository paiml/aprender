//! Types for PMAT Query integration.

/// A single result from `pmat query --format json`.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PmatQueryResult {
    pub file_path: String,
    pub function_name: String,
    #[serde(default)]
    pub signature: String,
    #[serde(default)]
    pub doc_comment: Option<String>,
    #[serde(default)]
    pub start_line: usize,
    #[serde(default)]
    pub end_line: usize,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub tdg_score: f64,
    #[serde(default)]
    pub tdg_grade: String,
    #[serde(default)]
    pub complexity: u32,
    #[serde(default)]
    pub big_o: String,
    #[serde(default)]
    pub satd_count: u32,
    #[serde(default)]
    pub loc: usize,
    #[serde(default)]
    pub relevance_score: f64,
    #[serde(default)]
    pub source: Option<String>,
    /// Project name (set during cross-project search).
    #[serde(default)]
    pub project: Option<String>,
    /// RAG documentation backlinks found for this result's file.
    #[serde(default)]
    pub rag_backlinks: Vec<String>,
}

impl Default for PmatQueryResult {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            function_name: String::new(),
            signature: String::new(),
            doc_comment: None,
            start_line: 0,
            end_line: 0,
            language: String::new(),
            tdg_score: 0.0,
            tdg_grade: String::new(),
            complexity: 0,
            big_o: String::new(),
            satd_count: 0,
            loc: 0,
            relevance_score: 0.0,
            source: None,
            project: None,
            rag_backlinks: Vec::new(),
        }
    }
}

/// Options for invoking `pmat query`.
#[derive(Debug, Clone)]
pub struct PmatQueryOptions {
    pub query: String,
    pub project_path: Option<String>,
    pub limit: usize,
    pub min_grade: Option<String>,
    pub max_complexity: Option<u32>,
    pub include_source: bool,
}

/// Quality distribution summary computed from a set of results.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QualitySummary {
    pub grades: std::collections::HashMap<String, usize>,
    pub avg_complexity: f64,
    pub total_satd: u32,
    pub complexity_range: (u32, u32),
}

/// A fused result that can be either a PMAT function hit or a RAG document hit.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum FusedResult {
    #[serde(rename = "function")]
    Function(Box<PmatQueryResult>),
    #[serde(rename = "document")]
    Document { component: String, source: String, score: f64, content: String },
}
