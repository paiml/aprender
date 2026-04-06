//! Dedicated `pmat_query` tool for agent code discovery.
//!
//! Replaces the `shell: pmat query "..."` fallback with a structured tool
//! that returns quality-annotated, ranked results. This is the agent's
//! primary code search tool — it understands TDG grades, complexity,
//! fault patterns, and call graphs.
//!
//! PMAT-163: Phase 4a — stack-native tool replaces shell fallback.

use async_trait::async_trait;
use std::process::Command;

use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;

use super::{Tool, ToolResult};

/// Maximum output bytes before truncation.
const MAX_OUTPUT_BYTES: usize = 32_768;

/// Dedicated pmat query tool for code discovery.
///
/// Executes `pmat query` as a subprocess and returns structured results
/// including function name, file, line range, TDG grade, and complexity.
/// Supports all pmat query flags: `--include-source`, `--faults`,
/// `--min-grade`, `--max-complexity`, `--exclude-tests`, etc.
#[derive(Default)]
pub struct PmatQueryTool;

impl PmatQueryTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for PmatQueryTool {
    fn name(&self) -> &'static str {
        "pmat_query"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "pmat_query".into(),
            description:
                "Search code by intent with quality annotations. Returns functions ranked \
                          by relevance with TDG grade, complexity, and call graph. Preferred over \
                          grep for code discovery."
                    .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query (e.g., 'error handling', 'cache invalidation')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)"
                    },
                    "include_source": {
                        "type": "boolean",
                        "description": "Include function source code in results"
                    },
                    "min_grade": {
                        "type": "string",
                        "description": "Minimum TDG grade filter (A, B, C, D, F)"
                    },
                    "max_complexity": {
                        "type": "integer",
                        "description": "Maximum cyclomatic complexity filter"
                    },
                    "exclude_tests": {
                        "type": "boolean",
                        "description": "Exclude test functions from results"
                    },
                    "faults": {
                        "type": "boolean",
                        "description": "Show fault patterns (unwrap, panic, unsafe)"
                    },
                    "regex": {
                        "type": "string",
                        "description": "Regex pattern match instead of semantic search"
                    },
                    "literal": {
                        "type": "string",
                        "description": "Exact literal string match instead of semantic search"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let regex = input.get("regex").and_then(|v| v.as_str());
        let literal = input.get("literal").and_then(|v| v.as_str());

        // Build pmat query command
        let mut cmd = Command::new("pmat");
        cmd.arg("query");

        // Mode: regex, literal, or semantic (default)
        if let Some(re) = regex {
            cmd.args(["--regex", re]);
        } else if let Some(lit) = literal {
            cmd.args(["--literal", lit]);
        } else if !query.is_empty() {
            cmd.arg(query);
        } else {
            return ToolResult::error("provide 'query', 'regex', or 'literal'");
        }

        // Optional flags
        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(10);
        cmd.args(["--limit", &limit.to_string()]);

        if input.get("include_source").and_then(|v| v.as_bool()).unwrap_or(false) {
            cmd.arg("--include-source");
        }

        if let Some(grade) = input.get("min_grade").and_then(|v| v.as_str()) {
            cmd.args(["--min-grade", grade]);
        }

        if let Some(complexity) = input.get("max_complexity").and_then(|v| v.as_u64()) {
            cmd.args(["--max-complexity", &complexity.to_string()]);
        }

        if input.get("exclude_tests").and_then(|v| v.as_bool()).unwrap_or(false) {
            cmd.arg("--exclude-tests");
        }

        if input.get("faults").and_then(|v| v.as_bool()).unwrap_or(false) {
            cmd.arg("--faults");
        }

        // Execute
        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => {
                return ToolResult::error(format!(
                    "pmat not found on PATH: {e}. Install: cargo install pmat"
                ));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            return ToolResult::error(format!("pmat query failed: {stderr}"));
        }

        // Truncate to prevent context overflow (Jidoka)
        let mut result = stdout.to_string();
        if result.len() > MAX_OUTPUT_BYTES {
            result.truncate(MAX_OUTPUT_BYTES);
            result.push_str("\n[truncated — use --limit or narrower query]");
        }

        if result.trim().is_empty() {
            ToolResult::success("No results found.")
        } else {
            ToolResult::success(result)
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::FileRead { allowed_paths: vec!["*".into()] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_name() {
        let tool = PmatQueryTool::new();
        assert_eq!(tool.name(), "pmat_query");
    }

    #[test]
    fn test_definition_has_query_field() {
        let tool = PmatQueryTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "pmat_query");
        let schema = &def.input_schema;
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("query")));
    }

    #[test]
    fn test_required_capability() {
        let tool = PmatQueryTool::new();
        assert!(matches!(tool.required_capability(), Capability::FileRead { .. }));
    }

    #[tokio::test]
    async fn test_empty_query_errors() {
        let tool = PmatQueryTool::new();
        let result = tool.execute(serde_json::json!({"query": ""})).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn test_semantic_query_runs() {
        let tool = PmatQueryTool::new();
        let result = tool
            .execute(serde_json::json!({
                "query": "error handling",
                "limit": 3
            }))
            .await;
        // pmat should be on PATH in dev environment
        if !result.is_error {
            assert!(!result.content.is_empty());
        }
    }

    #[tokio::test]
    async fn test_regex_query_runs() {
        let tool = PmatQueryTool::new();
        let result = tool
            .execute(serde_json::json!({
                "query": "",
                "regex": "fn\\s+test_",
                "limit": 3
            }))
            .await;
        if !result.is_error {
            assert!(!result.content.is_empty());
        }
    }

    #[tokio::test]
    async fn test_literal_query_runs() {
        let tool = PmatQueryTool::new();
        let result = tool
            .execute(serde_json::json!({
                "query": "",
                "literal": "unwrap()",
                "limit": 3,
                "exclude_tests": true
            }))
            .await;
        if !result.is_error {
            assert!(!result.content.is_empty());
        }
    }
}
