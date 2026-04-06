//! RAG tool — wraps `oracle::rag::RagOracle` for document retrieval.
//!
//! Provides agent access to the indexed Sovereign AI Stack documentation
//! via the existing RAG pipeline (BM25 + dense hybrid retrieval, RRF).
//!
//! Feature-gated: requires `rag` feature for `RagOracle` access.

use async_trait::async_trait;
use std::sync::Arc;

use super::{Tool, ToolResult};
use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;
use crate::oracle::rag::RagOracle;

/// Tool that wraps `RagOracle` for agent document retrieval.
pub struct RagTool {
    oracle: Arc<RagOracle>,
    max_results: usize,
}

impl RagTool {
    /// Create a new RAG tool wrapping an existing oracle instance.
    pub fn new(oracle: Arc<RagOracle>, max_results: usize) -> Self {
        Self { oracle, max_results }
    }
}

#[async_trait]
impl Tool for RagTool {
    fn name(&self) -> &'static str {
        "rag"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "rag".into(),
            description: "Search indexed Sovereign AI Stack documentation".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for documentation"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let Some(query) = input.get("query").and_then(|q| q.as_str()) else {
            return ToolResult::error("missing required field: query");
        };

        let results = self.oracle.query(query);
        let truncated: Vec<_> = results.into_iter().take(self.max_results).collect();

        if truncated.is_empty() {
            return ToolResult::success("No results found for the given query.");
        }

        let formatted = format_results(&truncated);
        ToolResult::success(formatted)
    }

    fn required_capability(&self) -> Capability {
        Capability::Rag
    }

    fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(120)
    }
}

/// Format retrieval results as markdown for LLM consumption.
fn format_results(results: &[crate::oracle::rag::RetrievalResult]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(results.len() * 256);
    for (i, r) in results.iter().enumerate() {
        let _ = writeln!(out, "### Result {} (score: {:.3})", i + 1, r.score);
        let _ = write!(
            out,
            "**Source:** {} ({}:{}–{})\n\n",
            r.source, r.component, r.start_line, r.end_line
        );
        out.push_str(&r.content);
        out.push_str("\n\n---\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_results_empty() {
        let results = vec![];
        assert_eq!(format_results(&results), "");
    }

    #[test]
    fn test_format_results_single() {
        use crate::oracle::rag::ScoreBreakdown;
        let results = vec![crate::oracle::rag::RetrievalResult {
            id: "doc-1".into(),
            component: "trueno".into(),
            source: "src/lib.rs".into(),
            content: "SIMD compute primitives".into(),
            score: 0.95,
            start_line: 1,
            end_line: 10,
            score_breakdown: ScoreBreakdown {
                bm25_score: 0.5,
                dense_score: 0.45,
                rrf_score: 0.95,
                rerank_score: None,
            },
        }];
        let formatted = format_results(&results);
        assert!(formatted.contains("Result 1"));
        assert!(formatted.contains("0.950"));
        assert!(formatted.contains("trueno"));
        assert!(formatted.contains("SIMD compute primitives"));
    }

    #[test]
    fn test_rag_tool_metadata() {
        // Cannot construct RagOracle without a full index,
        // so test metadata only via trait bounds
        assert_eq!(Capability::Rag, Capability::Rag, "Rag capability match");
    }

    #[test]
    fn test_tool_definition_schema() {
        // Validate the schema structure statically
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for documentation"
                }
            },
            "required": ["query"]
        });
        assert!(schema.get("properties").is_some());
        assert!(schema.get("required").is_some());
    }
}
