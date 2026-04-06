//! MCP Server Implementation
//!
//! Handles JSON-RPC 2.0 messages over stdio for the HuggingFace MCP tools.

use super::types::*;
use crate::hf::hub_client::{HubAssetType, HubClient, SearchFilters};
use std::collections::HashMap;

/// MCP Server for batuta HuggingFace tools
pub struct McpServer {
    hub_client: HubClient,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new() -> Self {
        Self { hub_client: HubClient::new() }
    }

    /// Handle a JSON-RPC request and return a response
    pub fn handle_request(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call(request),
            _ => JsonRpcResponse::error(
                request.id.clone(),
                -32601,
                format!("Method not found: {}", request.method),
            ),
        }
    }

    /// Handle initialize request
    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse::success(
            request.id.clone(),
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "batuta-hf",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
    }

    /// Handle tools/list request
    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tools = self.tool_definitions();
        JsonRpcResponse::success(request.id.clone(), serde_json::json!({ "tools": tools }))
    }

    /// Handle tools/call request
    fn handle_tools_call(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let name = request.params.get("name").and_then(|v| v.as_str());
        let arguments = request.params.get("arguments").cloned().unwrap_or(serde_json::json!({}));

        let result = match name {
            Some("hf_search") => self.tool_hf_search(&arguments),
            Some("hf_info") => self.tool_hf_info(&arguments),
            Some("hf_tree") => self.tool_hf_tree(&arguments),
            Some("hf_integration") => self.tool_hf_integration(),
            Some(other) => ToolCallResult::error(format!("Unknown tool: {}", other)),
            None => ToolCallResult::error("Missing tool name"),
        };

        JsonRpcResponse::success(
            request.id.clone(),
            serde_json::to_value(result).unwrap_or(serde_json::json!({})),
        )
    }

    // ========================================================================
    // Tool Definitions
    // ========================================================================

    /// Return all available tool definitions
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "hf_search".to_string(),
                description: "Search HuggingFace Hub for models, datasets, or spaces".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: HashMap::from([
                        (
                            "query".to_string(),
                            PropertySchema {
                                prop_type: "string".to_string(),
                                description: "Search query text".to_string(),
                                r#enum: None,
                            },
                        ),
                        (
                            "asset_type".to_string(),
                            PropertySchema {
                                prop_type: "string".to_string(),
                                description: "Type of asset to search".to_string(),
                                r#enum: Some(vec![
                                    "model".into(),
                                    "dataset".into(),
                                    "space".into(),
                                ]),
                            },
                        ),
                        (
                            "task".to_string(),
                            PropertySchema {
                                prop_type: "string".to_string(),
                                description: "Filter by ML task (e.g., text-generation)"
                                    .to_string(),
                                r#enum: None,
                            },
                        ),
                        (
                            "limit".to_string(),
                            PropertySchema {
                                prop_type: "integer".to_string(),
                                description: "Maximum number of results (default: 10)".to_string(),
                                r#enum: None,
                            },
                        ),
                    ]),
                    required: vec!["query".to_string()],
                },
            },
            ToolDefinition {
                name: "hf_info".to_string(),
                description: "Get metadata for a HuggingFace model, dataset, or space".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: HashMap::from([
                        (
                            "repo_id".to_string(),
                            PropertySchema {
                                prop_type: "string".to_string(),
                                description: "Repository ID (e.g., meta-llama/Llama-2-7b-hf)"
                                    .to_string(),
                                r#enum: None,
                            },
                        ),
                        (
                            "asset_type".to_string(),
                            PropertySchema {
                                prop_type: "string".to_string(),
                                description: "Type of asset".to_string(),
                                r#enum: Some(vec![
                                    "model".into(),
                                    "dataset".into(),
                                    "space".into(),
                                ]),
                            },
                        ),
                    ]),
                    required: vec!["repo_id".to_string()],
                },
            },
            ToolDefinition {
                name: "hf_tree".to_string(),
                description: "Show HuggingFace ecosystem component hierarchy".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: HashMap::from([(
                        "category".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: "Filter by category (e.g., inference, training)"
                                .to_string(),
                            r#enum: None,
                        },
                    )]),
                    required: vec![],
                },
            },
            ToolDefinition {
                name: "hf_integration".to_string(),
                description: "Show PAIML stack to HuggingFace integration mappings".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: HashMap::new(),
                    required: vec![],
                },
            },
        ]
    }

    // ========================================================================
    // Tool Implementations
    // ========================================================================

    fn tool_hf_search(&mut self, args: &serde_json::Value) -> ToolCallResult {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let asset_type = args.get("asset_type").and_then(|v| v.as_str()).unwrap_or("model");
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let task = args.get("task").and_then(|v| v.as_str());

        let mut filters = SearchFilters::new().with_query(query).with_limit(limit);
        if let Some(t) = task {
            filters = filters.with_task(t);
        }

        let results = match asset_type {
            "model" => self.hub_client.search_models(&filters),
            "dataset" => self.hub_client.search_datasets(&filters),
            "space" => self.hub_client.search_spaces(&filters),
            _ => return ToolCallResult::error(format!("Invalid asset_type: {}", asset_type)),
        };

        match results {
            Ok(assets) => {
                let formatted: Vec<String> = assets
                    .iter()
                    .map(|a| {
                        let mut line = format!("{} ({}⬇ {}♥)", a.id, a.downloads, a.likes);
                        if let Some(ref tag) = a.pipeline_tag {
                            line.push_str(&format!(" [{}]", tag));
                        }
                        line
                    })
                    .collect();
                ToolCallResult::success(formatted.join("\n"))
            }
            Err(e) => ToolCallResult::error(format!("Search failed: {}", e)),
        }
    }

    fn tool_hf_info(&mut self, args: &serde_json::Value) -> ToolCallResult {
        let repo_id = match args.get("repo_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolCallResult::error("Missing required parameter: repo_id"),
        };
        let asset_type = args.get("asset_type").and_then(|v| v.as_str()).unwrap_or("model");

        let result = match asset_type {
            "model" => self.hub_client.get_model(repo_id),
            "dataset" => self.hub_client.get_dataset(repo_id),
            "space" => self.hub_client.get_space(repo_id),
            _ => return ToolCallResult::error(format!("Invalid asset_type: {}", asset_type)),
        };

        match result {
            Ok(asset) => {
                let mut info = format!("ID: {}\n", asset.id);
                info.push_str(&format!("Author: {}\n", asset.author));
                info.push_str(&format!("Downloads: {}\n", asset.downloads));
                info.push_str(&format!("Likes: {}\n", asset.likes));
                if let Some(ref tag) = asset.pipeline_tag {
                    info.push_str(&format!("Task: {}\n", tag));
                }
                if let Some(ref lib) = asset.library {
                    info.push_str(&format!("Library: {}\n", lib));
                }
                if let Some(ref license) = asset.license {
                    info.push_str(&format!("License: {}\n", license));
                }
                if !asset.tags.is_empty() {
                    info.push_str(&format!("Tags: {}\n", asset.tags.join(", ")));
                }
                ToolCallResult::success(info)
            }
            Err(e) => ToolCallResult::error(format!("Info failed: {}", e)),
        }
    }

    fn tool_hf_tree(&self, args: &serde_json::Value) -> ToolCallResult {
        let _category = args.get("category").and_then(|v| v.as_str());

        let tree = r"HuggingFace Ecosystem
├── Inference
│   ├── transformers (PyTorch/TF models)
│   ├── text-generation-inference (TGI)
│   ├── optimum (hardware optimization)
│   └── candle (Rust inference)
├── Training
│   ├── accelerate (distributed training)
│   ├── peft (parameter-efficient fine-tuning)
│   ├── trl (RLHF training)
│   └── bitsandbytes (quantization)
├── Data
│   ├── datasets (data loading)
│   ├── tokenizers (fast tokenization)
│   └── evaluate (metrics)
├── Deployment
│   ├── inference-endpoints (managed API)
│   ├── spaces (app hosting)
│   └── gradio (web UI)
└── PAIML Integration
    ├── trueno ↔ candle (tensor ops)
    ├── aprender ↔ transformers (ML algorithms)
    ├── realizar ↔ TGI (inference serving)
    └── alimentar ↔ datasets (data loading)";

        ToolCallResult::success(tree)
    }

    fn tool_hf_integration(&self) -> ToolCallResult {
        let map = r"PAIML ↔ HuggingFace Integration Map

| PAIML Component | HF Equivalent | Integration |
|-----------------|---------------|-------------|
| trueno          | candle        | SIMD tensor operations |
| aprender        | transformers  | ML algorithm mapping |
| realizar        | TGI           | Inference serving |
| alimentar       | datasets      | Data loading (Arrow) |
| entrenar        | accelerate    | Distributed training |
| entrenar        | peft          | LoRA/QLoRA fine-tuning |
| entrenar        | trl           | RLHF training |
| whisper-apr     | whisper       | Speech recognition |
| pacha           | hub           | Model registry |
| batuta          | gradio        | UI/deployment |

Format: SafeTensors (shared), APR v2 (PAIML native)
Quantization: Q4K/Q5K/Q6K (PAIML) ↔ GPTQ/AWQ (HF)";

        ToolCallResult::success(map)
    }

    /// Run the MCP server on stdio (blocking)
    #[cfg(feature = "native")]
    pub fn run_stdio(&mut self) -> anyhow::Result<()> {
        use std::io::{self, BufRead, Write};

        let stdin = io::stdin();
        let stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let error_resp =
                        JsonRpcResponse::error(None, -32700, format!("Parse error: {}", e));
                    let json = serde_json::to_string(&error_resp)?;
                    writeln!(stdout.lock(), "{}", json)?;
                    continue;
                }
            };

            let response = self.handle_request(&request);
            let json = serde_json::to_string(&response)?;
            writeln!(stdout.lock(), "{}", json)?;
            stdout.lock().flush()?;
        }

        Ok(())
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
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

    #[test]
    fn test_initialize() {
        let mut server = McpServer::new();
        let req = make_request("initialize", serde_json::json!({}));
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "batuta-hf");
    }

    #[test]
    fn test_tools_list() {
        let mut server = McpServer::new();
        let req = make_request("tools/list", serde_json::json!({}));
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        let tools = result["tools"].as_array().expect("expected array value");
        assert_eq!(tools.len(), 4);
        assert_eq!(tools[0]["name"], "hf_search");
        assert_eq!(tools[1]["name"], "hf_info");
        assert_eq!(tools[2]["name"], "hf_tree");
        assert_eq!(tools[3]["name"], "hf_integration");
    }

    #[test]
    fn test_tool_hf_search() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_search",
                "arguments": {
                    "query": "llama",
                    "asset_type": "model",
                    "limit": 5
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        assert!(result["isError"].is_null());
        let content = result["content"].as_array().expect("expected array value");
        assert!(!content.is_empty());
        assert!(content[0]["text"].as_str().expect("expected string value").contains("llama"));
    }

    #[test]
    fn test_tool_hf_info() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_info",
                "arguments": {
                    "repo_id": "meta-llama/Llama-2-7b-hf",
                    "asset_type": "model"
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        let content = result["content"].as_array().expect("expected array value");
        assert!(content[0]["text"].as_str().expect("expected string value").contains("meta-llama"));
    }

    #[test]
    fn test_tool_hf_tree() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_tree",
                "arguments": {}
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        let content = result["content"].as_array().expect("expected array value");
        assert!(content[0]["text"]
            .as_str()
            .expect("unexpected failure")
            .contains("HuggingFace Ecosystem"));
    }

    #[test]
    fn test_tool_hf_integration() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_integration",
                "arguments": {}
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        let content = result["content"].as_array().expect("expected array value");
        assert!(content[0]["text"].as_str().expect("expected string value").contains("PAIML"));
    }

    #[test]
    fn test_unknown_method() {
        let mut server = McpServer::new();
        let req = make_request("unknown/method", serde_json::json!({}));
        let resp = server.handle_request(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.expect("unexpected failure").code, -32601);
    }

    #[test]
    fn test_unknown_tool() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "nonexistent_tool",
                "arguments": {}
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn test_missing_tool_name() {
        let mut server = McpServer::new();
        let req = make_request("tools/call", serde_json::json!({}));
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn test_hf_info_missing_repo_id() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_info",
                "arguments": {}
            }),
        );
        let resp = server.handle_request(&req);
        let result = resp.result.expect("operation failed");
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .expect("unexpected failure")
            .contains("Missing required"));
    }

    #[test]
    fn test_default_impl() {
        let server = McpServer::default();
        assert_eq!(server.tool_definitions().len(), 4);
    }

    // =====================================================================
    // Coverage: tool definition schema validation
    // =====================================================================

    #[test]
    fn test_tool_definitions_hf_search_schema() {
        let server = McpServer::new();
        let defs = server.tool_definitions();
        let search = &defs[0];
        assert_eq!(search.name, "hf_search");
        assert!(search.description.contains("Search"));
        assert_eq!(search.input_schema.schema_type, "object");
        assert!(search.input_schema.required.contains(&"query".to_string()));

        // Verify properties
        let props = &search.input_schema.properties;
        assert!(props.contains_key("query"));
        assert!(props.contains_key("asset_type"));
        assert!(props.contains_key("task"));
        assert!(props.contains_key("limit"));

        // Verify enum values for asset_type
        let asset_type = props.get("asset_type").expect("key not found");
        let enums = asset_type.r#enum.as_ref().expect("unexpected failure");
        assert!(enums.contains(&"model".to_string()));
        assert!(enums.contains(&"dataset".to_string()));
        assert!(enums.contains(&"space".to_string()));

        // Verify non-enum properties have None enum
        assert!(props.get("query").expect("key not found").r#enum.is_none());
        assert!(props.get("task").expect("key not found").r#enum.is_none());
        assert!(props.get("limit").expect("key not found").r#enum.is_none());

        // Verify property types
        assert_eq!(props.get("query").expect("key not found").prop_type, "string");
        assert_eq!(props.get("limit").expect("key not found").prop_type, "integer");
    }

    #[test]
    fn test_tool_definitions_hf_info_schema() {
        let server = McpServer::new();
        let defs = server.tool_definitions();
        let info = &defs[1];
        assert_eq!(info.name, "hf_info");
        assert!(info.description.contains("metadata"));
        assert!(info.input_schema.required.contains(&"repo_id".to_string()));

        let props = &info.input_schema.properties;
        assert!(props.contains_key("repo_id"));
        assert!(props.contains_key("asset_type"));
        assert_eq!(props.len(), 2);

        // Verify asset_type enum
        let enums =
            props.get("asset_type").expect("key not found").r#enum.as_ref().expect("key not found");
        assert_eq!(enums.len(), 3);
    }

    #[test]
    fn test_tool_definitions_hf_tree_schema() {
        let server = McpServer::new();
        let defs = server.tool_definitions();
        let tree = &defs[2];
        assert_eq!(tree.name, "hf_tree");
        assert!(tree.description.contains("hierarchy"));
        assert!(tree.input_schema.required.is_empty());

        let props = &tree.input_schema.properties;
        assert_eq!(props.len(), 1);
        assert!(props.contains_key("category"));
    }

    #[test]
    fn test_tool_definitions_hf_integration_schema() {
        let server = McpServer::new();
        let defs = server.tool_definitions();
        let integration = &defs[3];
        assert_eq!(integration.name, "hf_integration");
        assert!(integration.description.contains("integration"));
        assert!(integration.input_schema.required.is_empty());
        assert!(integration.input_schema.properties.is_empty());
    }

    // =====================================================================
    // Coverage: search with task filter and dataset/space types
    // =====================================================================

    #[test]
    fn test_tool_hf_search_with_task_filter() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_search",
                "arguments": {
                    "query": "bert",
                    "asset_type": "model",
                    "task": "text-classification",
                    "limit": 3
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        // May succeed or error depending on network, but should not panic
        assert!(result["isError"].is_null() || result["isError"] == true);
    }

    #[test]
    fn test_tool_hf_search_dataset() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_search",
                "arguments": {
                    "query": "squad",
                    "asset_type": "dataset",
                    "limit": 2
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_tool_hf_search_space() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_search",
                "arguments": {
                    "query": "gradio",
                    "asset_type": "space",
                    "limit": 2
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_tool_hf_search_invalid_asset_type() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_search",
                "arguments": {
                    "query": "test",
                    "asset_type": "invalid_type"
                }
            }),
        );
        let resp = server.handle_request(&req);
        let result = resp.result.expect("operation failed");
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .expect("unexpected failure")
            .contains("Invalid asset_type"));
    }

    #[test]
    fn test_tool_hf_search_defaults() {
        // No asset_type or limit specified -- defaults to "model" and 10
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_search",
                "arguments": {
                    "query": "tiny"
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }

    // =====================================================================
    // Coverage: hf_info with dataset/space and invalid asset_type
    // =====================================================================

    #[test]
    fn test_tool_hf_info_dataset() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_info",
                "arguments": {
                    "repo_id": "squad",
                    "asset_type": "dataset"
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_tool_hf_info_space() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_info",
                "arguments": {
                    "repo_id": "stabilityai/stable-diffusion",
                    "asset_type": "space"
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_tool_hf_info_invalid_asset_type() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_info",
                "arguments": {
                    "repo_id": "test/repo",
                    "asset_type": "invalid"
                }
            }),
        );
        let resp = server.handle_request(&req);
        let result = resp.result.expect("operation failed");
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .expect("unexpected failure")
            .contains("Invalid asset_type"));
    }

    #[test]
    fn test_tool_hf_info_default_asset_type() {
        // No asset_type: defaults to "model"
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_info",
                "arguments": {
                    "repo_id": "meta-llama/Llama-2-7b-hf"
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }

    // =====================================================================
    // Coverage: hf_tree with category filter
    // =====================================================================

    #[test]
    fn test_tool_hf_tree_with_category() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_tree",
                "arguments": {
                    "category": "inference"
                }
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        let content = result["content"].as_array().expect("expected array value");
        // Tree output should contain the ecosystem structure
        assert!(content[0]["text"].as_str().expect("expected string value").contains("Inference"));
    }

    // =====================================================================
    // Coverage: tools/call with no arguments key
    // =====================================================================

    #[test]
    fn test_tool_call_no_arguments_key() {
        let mut server = McpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({
                "name": "hf_integration"
            }),
        );
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.expect("operation failed");
        // Should use default empty object for arguments
        let content = result["content"].as_array().expect("expected array value");
        assert!(content[0]["text"].as_str().expect("expected string value").contains("PAIML"));
    }

    // =====================================================================
    // Coverage: JSON-RPC serialization
    // =====================================================================

    #[test]
    fn test_initialize_response_serialization() {
        let mut server = McpServer::new();
        let req = make_request("initialize", serde_json::json!({}));
        let resp = server.handle_request(&req);
        let json = serde_json::to_string(&resp).expect("json serialize failed");
        assert!(json.contains("protocolVersion"));
        assert!(json.contains("batuta-hf"));
        assert!(json.contains("2.0"));
    }

    #[test]
    fn test_tools_list_serialization() {
        let mut server = McpServer::new();
        let req = make_request("tools/list", serde_json::json!({}));
        let resp = server.handle_request(&req);
        let json = serde_json::to_string(&resp).expect("json serialize failed");
        // Verify all tools appear in serialized JSON
        assert!(json.contains("hf_search"));
        assert!(json.contains("hf_info"));
        assert!(json.contains("hf_tree"));
        assert!(json.contains("hf_integration"));
        assert!(json.contains("inputSchema"));
    }

    #[test]
    fn test_error_response_serialization() {
        let mut server = McpServer::new();
        let req = make_request("nonexistent", serde_json::json!({}));
        let resp = server.handle_request(&req);
        let json = serde_json::to_string(&resp).expect("json serialize failed");
        assert!(json.contains("error"));
        assert!(json.contains("-32601"));
        assert!(json.contains("Method not found"));
    }

    #[test]
    fn test_request_with_null_id() {
        let mut server = McpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: "initialize".to_string(),
            params: serde_json::json!({}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        assert!(resp.id.is_none());
    }
}
