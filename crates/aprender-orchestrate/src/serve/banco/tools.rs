//! Tool calling framework — OpenAI-compatible function calling with self-healing.
//!
//! Tools are registered functions that models can invoke during chat.
//! Built-in tools: code_execution (sandbox), calculator.
//! Custom tools can be registered via the API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// Tool definition — describes a callable function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: serde_json::Value,
    /// Whether this tool is enabled.
    pub enabled: bool,
    /// Privacy tier requirement (None = all tiers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_tier: Option<String>,
}

/// A tool call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A tool call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Tool registry — manages available tools.
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, ToolDefinition>>,
}

impl ToolRegistry {
    /// Create registry with built-in tools.
    #[must_use]
    pub fn new() -> Self {
        let mut tools = HashMap::new();

        // Built-in: calculator
        tools.insert(
            "calculator".to_string(),
            ToolDefinition {
                name: "calculator".to_string(),
                description: "Evaluate a mathematical expression".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                        }
                    },
                    "required": ["expression"]
                }),
                enabled: true,
                required_tier: None,
            },
        );

        // Built-in: code_execution
        tools.insert(
            "code_execution".to_string(),
            ToolDefinition {
                name: "code_execution".to_string(),
                description: "Execute code in a sandboxed environment".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "bash", "rust"],
                            "description": "Programming language"
                        },
                        "code": {
                            "type": "string",
                            "description": "Code to execute"
                        }
                    },
                    "required": ["language", "code"]
                }),
                enabled: true,
                required_tier: None,
            },
        );

        // Built-in: web_search (Standard tier only)
        tools.insert(
            "web_search".to_string(),
            ToolDefinition {
                name: "web_search".to_string(),
                description: "Search the web for information".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }),
                enabled: false, // disabled by default — requires Standard tier
                required_tier: Some("Standard".to_string()),
            },
        );

        Self { tools: RwLock::new(tools) }
    }

    /// List all tools.
    #[must_use]
    pub fn list(&self) -> Vec<ToolDefinition> {
        let store = self.tools.read().unwrap_or_else(|e| e.into_inner());
        let mut tools: Vec<ToolDefinition> = store.values().cloned().collect();
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }

    /// List enabled tools for a given privacy tier.
    #[must_use]
    pub fn list_for_tier(&self, tier: &str) -> Vec<ToolDefinition> {
        self.list()
            .into_iter()
            .filter(|t| t.enabled)
            .filter(|t| {
                t.required_tier.as_ref().is_none_or(|req| req == tier || tier == "Standard")
            })
            .collect()
    }

    /// Get a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<ToolDefinition> {
        self.tools.read().unwrap_or_else(|e| e.into_inner()).get(name).cloned()
    }

    /// Register a custom tool.
    pub fn register(&self, tool: ToolDefinition) {
        if let Ok(mut store) = self.tools.write() {
            store.insert(tool.name.clone(), tool);
        }
    }

    /// Enable/disable a tool.
    pub fn set_enabled(&self, name: &str, enabled: bool) -> bool {
        if let Ok(mut store) = self.tools.write() {
            if let Some(tool) = store.get_mut(name) {
                tool.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Execute a tool call. Returns the result.
    #[must_use]
    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        match call.name.as_str() {
            "calculator" => execute_calculator(call),
            "code_execution" => execute_code_sandbox(call),
            "web_search" => ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                content: String::new(),
                error: Some("Web search not implemented in sovereign mode".to_string()),
            },
            _ => ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                content: String::new(),
                error: Some(format!("Unknown tool: {}", call.name)),
            },
        }
    }

    /// Execute with self-healing retry.
    /// If the call fails, injects the error and allows the caller to re-prompt.
    /// Returns (result, retry_messages) where retry_messages is the error context for re-prompting.
    pub fn execute_with_retry(&self, call: &ToolCall, max_retries: usize) -> ToolCallOutcome {
        let result = self.execute(call);

        if result.error.is_some() && max_retries > 0 {
            // Return the error context for the caller to inject and re-prompt
            let error_context = format!(
                "Tool call to '{}' failed: {}. Please fix the arguments and try again.",
                call.name,
                result.error.as_deref().unwrap_or("unknown error")
            );
            ToolCallOutcome {
                result,
                should_retry: true,
                error_context: Some(error_context),
                retries_remaining: max_retries - 1,
            }
        } else {
            ToolCallOutcome {
                result,
                should_retry: false,
                error_context: None,
                retries_remaining: 0,
            }
        }
    }
}

/// Outcome of a tool call with retry information.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallOutcome {
    pub result: ToolResult,
    pub should_retry: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_context: Option<String>,
    pub retries_remaining: usize,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in calculator — evaluates simple math expressions.
fn execute_calculator(call: &ToolCall) -> ToolResult {
    let expr = call.arguments.get("expression").and_then(|v| v.as_str()).unwrap_or("");

    let result = eval_math(expr);

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        content: match &result {
            Ok(val) => val.to_string(),
            Err(_) => String::new(),
        },
        error: result.err().map(|e| e.to_string()),
    }
}

/// Simple math expression evaluator (supports +, -, *, /, parentheses).
fn eval_math(expr: &str) -> Result<f64, String> {
    let tokens: Vec<char> = expr.chars().filter(|c| !c.is_whitespace()).collect();
    if tokens.is_empty() {
        return Err("Empty expression".to_string());
    }
    let mut pos = 0;
    let result = parse_expr(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(format!("Unexpected character at position {pos}"));
    }
    Ok(result)
}

fn parse_expr(tokens: &[char], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_term(tokens, pos)?;
    while *pos < tokens.len() && (tokens[*pos] == '+' || tokens[*pos] == '-') {
        let op = tokens[*pos];
        *pos += 1;
        let right = parse_term(tokens, pos)?;
        left = if op == '+' { left + right } else { left - right };
    }
    Ok(left)
}

fn parse_term(tokens: &[char], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_factor(tokens, pos)?;
    while *pos < tokens.len() && (tokens[*pos] == '*' || tokens[*pos] == '/') {
        let op = tokens[*pos];
        *pos += 1;
        let right = parse_factor(tokens, pos)?;
        if op == '/' && right == 0.0 {
            return Err("Division by zero".to_string());
        }
        left = if op == '*' { left * right } else { left / right };
    }
    Ok(left)
}

fn parse_factor(tokens: &[char], pos: &mut usize) -> Result<f64, String> {
    if *pos >= tokens.len() {
        return Err("Unexpected end of expression".to_string());
    }

    // Negation
    if tokens[*pos] == '-' {
        *pos += 1;
        let val = parse_factor(tokens, pos)?;
        return Ok(-val);
    }

    // Parentheses
    if tokens[*pos] == '(' {
        *pos += 1;
        let val = parse_expr(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos] != ')' {
            return Err("Missing closing parenthesis".to_string());
        }
        *pos += 1;
        return Ok(val);
    }

    // Number
    let start = *pos;
    while *pos < tokens.len() && (tokens[*pos].is_ascii_digit() || tokens[*pos] == '.') {
        *pos += 1;
    }
    if start == *pos {
        return Err(format!("Expected number at position {start}"));
    }
    let num_str: String = tokens[start..*pos].iter().collect();
    num_str.parse::<f64>().map_err(|e| e.to_string())
}

/// Code execution sandbox (dry-run — actual sandbox requires jugar-probar).
fn execute_code_sandbox(call: &ToolCall) -> ToolResult {
    let language = call.arguments.get("language").and_then(|v| v.as_str()).unwrap_or("unknown");
    let code = call.arguments.get("code").and_then(|v| v.as_str()).unwrap_or("");

    // Dry-run: echo what would be executed
    let content = format!(
        "{{\"stdout\": \"[sandbox dry-run] Would execute {language} code ({} chars)\", \"stderr\": \"\", \"exit_code\": 0}}",
        code.len()
    );

    ToolResult { tool_call_id: call.id.clone(), name: call.name.clone(), content, error: None }
}
