//! `AprMcpServer` — JSON-RPC 2.0 dispatcher for aprender MCP tools.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools;
use crate::types::{JsonRpcRequest, JsonRpcResponse, ToolCallResult, ToolDefinition};

/// MCP server exposing the `apr` CLI as tools.
///
/// M1: `initialize`, `tools/list`, `tools/call` with `apr.version`.
#[derive(Debug, Default)]
pub struct AprMcpServer {}

impl AprMcpServer {
    /// Construct a new server.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Dispatch a single JSON-RPC request.
    ///
    /// The dispatcher enforces two protocol-level invariants before routing:
    /// FALSIFY-MCP-005 (`jsonrpc` must be exactly `"2.0"` or the response is
    /// `-32600 Invalid Request`) and FALSIFY-MCP-007 (an `initialize` whose
    /// `params.protocolVersion` mismatches ours returns `-32602 Invalid Params`
    /// instead of advancing to tools/list).
    #[must_use]
    pub fn handle_request(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        if request.jsonrpc != "2.0" {
            return JsonRpcResponse::error(
                request.id.clone(),
                -32600,
                format!(
                    "Invalid Request: jsonrpc must be \"2.0\", got \"{}\"",
                    request.jsonrpc
                ),
            );
        }

        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call(request),
            other => JsonRpcResponse::error(
                request.id.clone(),
                -32601,
                format!("Method not found: {other}"),
            ),
        }
    }

    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        // FALSIFY-MCP-007: if the client advertises a protocolVersion, it must
        // match ours. Missing field is permitted (some clients omit it on the
        // very first handshake); only a *mismatch* is rejected.
        if let Some(client_version) = request
            .params
            .get("protocolVersion")
            .and_then(|v| v.as_str())
        {
            if client_version != crate::PROTOCOL_VERSION {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    format!(
                        "Unsupported protocolVersion: client requested \"{}\", server speaks \"{}\"",
                        client_version,
                        crate::PROTOCOL_VERSION
                    ),
                );
            }
        }

        JsonRpcResponse::success(
            request.id.clone(),
            serde_json::json!({
                "protocolVersion": crate::PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": crate::SERVER_NAME,
                    "version": env!("CARGO_PKG_VERSION"),
                },
            }),
        )
    }

    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tools: Vec<ToolDefinition> = self.tool_definitions();
        JsonRpcResponse::success(request.id.clone(), serde_json::json!({ "tools": tools }))
    }

    fn handle_tools_call(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let name = request.params.get("name").and_then(|v| v.as_str());
        let arguments = request
            .params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        let result = match name {
            Some(tools::version::NAME) => tools::version::call(&arguments),
            Some(tools::validate::NAME) => tools::validate::call(&arguments),
            Some(tools::tensors::NAME) => tools::tensors::call(&arguments),
            Some(tools::bench::NAME) => tools::bench::call(&arguments),
            Some(tools::qa::NAME) => tools::qa::call(&arguments),
            Some(tools::trace::NAME) => tools::trace::call(&arguments),
            Some(tools::serve::NAME) => tools::serve::call(&arguments),
            Some(other) => ToolCallResult::error(format!("Unknown tool: {other}")),
            None => ToolCallResult::error("Missing tool name"),
        };

        JsonRpcResponse::success(
            request.id.clone(),
            serde_json::to_value(result).unwrap_or_else(|_| serde_json::json!({})),
        )
    }

    /// All tool definitions registered on this server.
    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            tools::version_tool_definition(),
            tools::validate_tool_definition(),
            tools::tensors_tool_definition(),
            tools::bench_tool_definition(),
            tools::qa_tool_definition(),
            tools::trace_tool_definition(),
            tools::serve_tool_definition(),
        ]
    }

    /// Run the server over stdio (blocking).
    ///
    /// Reads one JSON-RPC request per line from stdin, writes one response per
    /// line to stdout. Parse errors map to JSON-RPC code -32700.
    ///
    /// # Errors
    /// Returns an error if stdin/stdout I/O fails.
    #[cfg(feature = "native")]
    pub fn run_stdio(&mut self) -> anyhow::Result<()> {
        use std::io::{self, BufRead, Write};

        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut out = stdout.lock();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
                Ok(req) => self.handle_request(&req),
                Err(e) => JsonRpcResponse::error(None, -32700, format!("Parse error: {e}")),
            };

            let json = serde_json::to_string(&response)?;
            writeln!(out, "{json}")?;
            out.flush()?;
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    fn make_request(method: &str, params: serde_json::Value) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: method.to_string(),
            params,
        }
    }

    /// FALSIFY-MCP-001: initialize returns protocolVersion "2024-11-05".
    #[test]
    fn initialize_returns_protocol_version() {
        let mut server = AprMcpServer::new();
        let req = make_request("initialize", serde_json::json!({}));
        let resp = server.handle_request(&req);

        assert!(resp.error.is_none());
        let result = resp.result.expect("result present");
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "aprender-mcp");
        assert!(result["capabilities"]["tools"].is_object());
    }

    /// FALSIFY-MCP-002 (progressive): tools/list returns every tool that has
    /// shipped so far. Full 8-tool set lands when M2 completes.
    #[test]
    fn tools_list_returns_registered_tools() {
        let mut server = AprMcpServer::new();
        let req = make_request("tools/list", serde_json::json!({}));
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        let tools = result["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        for expected in [
            "apr.version",
            "apr.validate",
            "apr.tensors",
            "apr.bench",
            "apr.qa",
            "apr.trace",
            "apr.serve",
        ] {
            assert!(names.contains(&expected), "{expected} registered");
        }

        for tool in tools {
            assert_eq!(tool["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn tools_call_version_returns_metadata() {
        let mut server = AprMcpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({ "name": "apr.version", "arguments": {} }),
        );
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        let text = result["content"][0]["text"].as_str().expect("text");
        let parsed: serde_json::Value = serde_json::from_str(text).expect("json");
        assert_eq!(parsed["server"], "aprender-mcp");
        assert_eq!(parsed["protocol_version"], "2024-11-05");
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let mut server = AprMcpServer::new();
        let req = make_request("tools/explode", serde_json::json!({}));
        let resp = server.handle_request(&req);

        assert!(resp.result.is_none());
        let err = resp.error.expect("error present");
        assert_eq!(err.code, -32601);
    }

    /// `apr.validate` without `model_path` must return `isError: true` via
    /// the argument-validation branch (no subprocess spawn).
    #[test]
    fn tools_call_validate_missing_model_path_is_error() {
        let mut server = AprMcpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({ "name": "apr.validate", "arguments": {} }),
        );
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().expect("text");
        assert!(text.contains("model_path"));
    }

    #[test]
    fn tools_call_unknown_tool_returns_is_error() {
        let mut server = AprMcpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({ "name": "apr.nonexistent" }),
        );
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn tools_call_missing_name_returns_is_error() {
        let mut server = AprMcpServer::new();
        let req = make_request("tools/call", serde_json::json!({}));
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn id_is_echoed_back() {
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!("req-42")),
            method: "initialize".to_string(),
            params: serde_json::json!({}),
        };
        let resp = server.handle_request(&req);
        assert_eq!(resp.id, Some(serde_json::json!("req-42")));
    }
}
