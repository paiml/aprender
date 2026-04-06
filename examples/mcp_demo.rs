//! MCP Tooling Demo
//!
//! Demonstrates Model Context Protocol concepts for ML tool servers.
//!
//! The PAIML stack includes:
//! - **pmcp** v1.8.6: Rust SDK for MCP servers/clients
//! - **pforge** v0.1.4: Declarative YAML-based MCP framework
//!
//! # Run
//!
//! ```bash
//! cargo run --example mcp_demo
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// MCP Protocol Types (simplified from pmcp)
// ============================================================================

/// Tool definition following MCP specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: JsonSchema,
}

/// JSON Schema for tool parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: HashMap<String, PropertySchema>,
    #[serde(default)]
    pub required: Vec<String>,
}

/// Property schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    pub prop_type: String,
    pub description: String,
}

/// Tool call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Content block in tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
}

// ============================================================================
// pforge-style Tool Definitions (YAML → Rust)
// ============================================================================

/// Tool handler trait (like pforge native handlers)
pub trait ToolHandler: Send + Sync {
    fn call(&self, args: serde_json::Value) -> Result<ToolResult, String>;
}

/// Train model tool - integrates with Entrenar
struct TrainModelTool;

impl ToolHandler for TrainModelTool {
    fn call(&self, args: serde_json::Value) -> Result<ToolResult, String> {
        let config_path = args["config_path"].as_str().ok_or("Missing config_path")?;
        let epochs = args["epochs"].as_u64().unwrap_or(10);

        // In real implementation, this would call entrenar
        let result = format!(
            "Training started:\n  Config: {}\n  Epochs: {}\n  Status: completed\n  Loss: 0.0234\n  Accuracy: 0.9812",
            config_path, epochs
        );

        Ok(ToolResult { content: vec![Content::Text { text: result }], is_error: None })
    }
}

/// Quantize model tool - integrates with Entrenar quantization
struct QuantizeModelTool;

impl ToolHandler for QuantizeModelTool {
    fn call(&self, args: serde_json::Value) -> Result<ToolResult, String> {
        let model_path = args["model_path"].as_str().ok_or("Missing model_path")?;
        let bits = args["bits"].as_u64().unwrap_or(4);

        let result = format!(
            "Quantization complete:\n  Model: {}\n  Bits: {}\n  Original size: 13.5 GB\n  Quantized size: {} GB\n  Output: {}.q{}",
            model_path,
            bits,
            13.5 / (32.0 / bits as f64),
            model_path.trim_end_matches(".safetensors"),
            bits
        );

        Ok(ToolResult { content: vec![Content::Text { text: result }], is_error: None })
    }
}

/// Run inference tool - integrates with Realizar
struct InferenceTool;

impl ToolHandler for InferenceTool {
    fn call(&self, args: serde_json::Value) -> Result<ToolResult, String> {
        let prompt = args["prompt"].as_str().ok_or("Missing prompt")?;
        let max_tokens = args["max_tokens"].as_u64().unwrap_or(256);
        let temperature = args["temperature"].as_f64().unwrap_or(0.7);

        let result = format!(
            "Inference result:\n  Prompt: \"{}\"\n  Max tokens: {}\n  Temperature: {}\n  Response: \"{}\"",
            prompt,
            max_tokens,
            temperature,
            simulate_inference(prompt)
        );

        Ok(ToolResult { content: vec![Content::Text { text: result }], is_error: None })
    }
}

fn simulate_inference(prompt: &str) -> String {
    if prompt.contains("Rust") {
        "Rust is a systems programming language focused on safety, concurrency, and performance."
            .to_string()
    } else if prompt.contains("ML") || prompt.contains("machine learning") {
        "Machine learning enables computers to learn from data without explicit programming."
            .to_string()
    } else {
        "I'm an AI assistant ready to help with your questions.".to_string()
    }
}

/// Query database tool - integrates with Trueno-DB
struct QueryDatabaseTool;

impl ToolHandler for QueryDatabaseTool {
    fn call(&self, args: serde_json::Value) -> Result<ToolResult, String> {
        let sql = args["sql"].as_str().ok_or("Missing sql")?;

        let result = format!(
            "Query executed:\n  SQL: {}\n  Rows returned: 42\n  Execution time: 3.2ms",
            sql
        );

        Ok(ToolResult { content: vec![Content::Text { text: result }], is_error: None })
    }
}

// ============================================================================
// MCP Server (simplified from pmcp)
// ============================================================================

/// MCP Server holding tool registry
pub struct McpServer {
    name: String,
    version: String,
    tools: HashMap<String, (Tool, Box<dyn ToolHandler>)>,
}

impl McpServer {
    pub fn new(name: &str, version: &str) -> Self {
        Self { name: name.to_string(), version: version.to_string(), tools: HashMap::new() }
    }

    pub fn register_tool(&mut self, tool: Tool, handler: Box<dyn ToolHandler>) {
        self.tools.insert(tool.name.clone(), (tool, handler));
    }

    pub fn list_tools(&self) -> Vec<&Tool> {
        self.tools.values().map(|(t, _)| t).collect()
    }

    pub fn call_tool(&self, call: &ToolCall) -> Result<ToolResult, String> {
        let (_, handler) =
            self.tools.get(&call.name).ok_or_else(|| format!("Unknown tool: {}", call.name))?;

        handler.call(call.arguments.clone())
    }
}

// ============================================================================
// pforge YAML Config Representation
// ============================================================================

/// pforge.yaml configuration (demonstrates YAML → Rust mapping)
#[derive(Debug, Serialize, Deserialize)]
pub struct PforgeConfig {
    pub forge: ForgeConfig,
    pub tools: Vec<ToolConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForgeConfig {
    pub name: String,
    pub version: String,
    pub transport: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolConfig {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    pub description: String,
    pub handler: Option<HandlerConfig>,
    pub params: Option<HashMap<String, ParamConfig>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandlerConfig {
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParamConfig {
    #[serde(rename = "type")]
    pub param_type: String,
    pub required: bool,
    #[serde(default)]
    pub description: String,
}

// ============================================================================
// Demo
// ============================================================================

fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("              MCP Tooling Demo (pmcp + pforge)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // 1. Show pforge YAML configuration
    println!("1. pforge YAML Configuration (pforge.yaml)");
    println!("───────────────────────────────────────────────────────────────────");

    let config = PforgeConfig {
        forge: ForgeConfig {
            name: "paiml-ml-server".to_string(),
            version: "0.1.0".to_string(),
            transport: "stdio".to_string(),
            description: Some("PAIML ML tools MCP server".to_string()),
        },
        tools: vec![
            ToolConfig {
                tool_type: "native".to_string(),
                name: "train_model".to_string(),
                description: "Train model from YAML config".to_string(),
                handler: Some(HandlerConfig { path: "handlers::train_model".to_string() }),
                params: Some(HashMap::from([
                    (
                        "config_path".to_string(),
                        ParamConfig {
                            param_type: "string".to_string(),
                            required: true,
                            description: "Path to training YAML".to_string(),
                        },
                    ),
                    (
                        "epochs".to_string(),
                        ParamConfig {
                            param_type: "integer".to_string(),
                            required: false,
                            description: "Number of epochs".to_string(),
                        },
                    ),
                ])),
            },
            ToolConfig {
                tool_type: "native".to_string(),
                name: "quantize".to_string(),
                description: "Quantize model to N-bit".to_string(),
                handler: Some(HandlerConfig { path: "handlers::quantize".to_string() }),
                params: Some(HashMap::from([
                    (
                        "model_path".to_string(),
                        ParamConfig {
                            param_type: "string".to_string(),
                            required: true,
                            description: "Path to model".to_string(),
                        },
                    ),
                    (
                        "bits".to_string(),
                        ParamConfig {
                            param_type: "integer".to_string(),
                            required: false,
                            description: "Quantization bits (4 or 8)".to_string(),
                        },
                    ),
                ])),
            },
        ],
    };

    let yaml = serde_yaml_ng::to_string(&config)?;
    println!("{}", yaml);

    // 2. Create MCP server with tools
    println!("\n2. MCP Server Initialization");
    println!("───────────────────────────────────────────────────────────────────");

    let mut server = McpServer::new("paiml-ml-server", "0.1.0");

    // Register train_model tool
    let train_tool = Tool {
        name: "train_model".to_string(),
        description: "Train model using Entrenar YAML configuration".to_string(),
        input_schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: HashMap::from([
                (
                    "config_path".to_string(),
                    PropertySchema {
                        prop_type: "string".to_string(),
                        description: "Path to training YAML config".to_string(),
                    },
                ),
                (
                    "epochs".to_string(),
                    PropertySchema {
                        prop_type: "integer".to_string(),
                        description: "Number of training epochs".to_string(),
                    },
                ),
            ]),
            required: vec!["config_path".to_string()],
        },
    };
    server.register_tool(train_tool, Box::new(TrainModelTool));

    // Register quantize tool
    let quantize_tool = Tool {
        name: "quantize".to_string(),
        description: "Quantize model to 4-bit or 8-bit".to_string(),
        input_schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: HashMap::from([
                (
                    "model_path".to_string(),
                    PropertySchema {
                        prop_type: "string".to_string(),
                        description: "Path to model file".to_string(),
                    },
                ),
                (
                    "bits".to_string(),
                    PropertySchema {
                        prop_type: "integer".to_string(),
                        description: "Quantization bits".to_string(),
                    },
                ),
            ]),
            required: vec!["model_path".to_string()],
        },
    };
    server.register_tool(quantize_tool, Box::new(QuantizeModelTool));

    // Register inference tool
    let inference_tool = Tool {
        name: "generate".to_string(),
        description: "Generate text using Realizar inference".to_string(),
        input_schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: HashMap::from([
                (
                    "prompt".to_string(),
                    PropertySchema {
                        prop_type: "string".to_string(),
                        description: "Input prompt".to_string(),
                    },
                ),
                (
                    "max_tokens".to_string(),
                    PropertySchema {
                        prop_type: "integer".to_string(),
                        description: "Maximum tokens to generate".to_string(),
                    },
                ),
                (
                    "temperature".to_string(),
                    PropertySchema {
                        prop_type: "number".to_string(),
                        description: "Sampling temperature".to_string(),
                    },
                ),
            ]),
            required: vec!["prompt".to_string()],
        },
    };
    server.register_tool(inference_tool, Box::new(InferenceTool));

    // Register query tool
    let query_tool = Tool {
        name: "query".to_string(),
        description: "Query Trueno-DB with SQL".to_string(),
        input_schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: HashMap::from([(
                "sql".to_string(),
                PropertySchema {
                    prop_type: "string".to_string(),
                    description: "SQL query to execute".to_string(),
                },
            )]),
            required: vec!["sql".to_string()],
        },
    };
    server.register_tool(query_tool, Box::new(QueryDatabaseTool));

    println!("Server: {} v{}", server.name, server.version);
    println!("Registered tools:");
    for tool in server.list_tools() {
        println!("  - {}: {}", tool.name, tool.description);
    }

    // 3. Execute tool calls
    println!("\n3. Tool Execution");
    println!("───────────────────────────────────────────────────────────────────");

    // Train model
    let train_call = ToolCall {
        name: "train_model".to_string(),
        arguments: serde_json::json!({
            "config_path": "configs/lora-finetune.yaml",
            "epochs": 5
        }),
    };

    println!("\nCalling: train_model");
    match server.call_tool(&train_call) {
        Ok(result) => {
            for content in &result.content {
                let Content::Text { text } = content;
                println!("{}", text);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // Quantize model
    let quantize_call = ToolCall {
        name: "quantize".to_string(),
        arguments: serde_json::json!({
            "model_path": "models/llama-7b.safetensors",
            "bits": 4
        }),
    };

    println!("\nCalling: quantize");
    match server.call_tool(&quantize_call) {
        Ok(result) => {
            for content in &result.content {
                let Content::Text { text } = content;
                println!("{}", text);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // Generate text
    let generate_call = ToolCall {
        name: "generate".to_string(),
        arguments: serde_json::json!({
            "prompt": "What is Rust programming?",
            "max_tokens": 100,
            "temperature": 0.7
        }),
    };

    println!("\nCalling: generate");
    match server.call_tool(&generate_call) {
        Ok(result) => {
            for content in &result.content {
                let Content::Text { text } = content;
                println!("{}", text);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // Query database
    let query_call = ToolCall {
        name: "query".to_string(),
        arguments: serde_json::json!({
            "sql": "SELECT * FROM experiments WHERE accuracy > 0.95"
        }),
    };

    println!("\nCalling: query");
    match server.call_tool(&query_call) {
        Ok(result) => {
            for content in &result.content {
                let Content::Text { text } = content;
                println!("{}", text);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // 4. Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                          Summary");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("MCP Stack Components:");
    println!("  • pmcp v1.8.6   - Rust SDK for MCP (low-level)");
    println!("  • pforge v0.1.4 - Declarative YAML framework (high-level)");
    println!();
    println!("Integration Points:");
    println!("  • train_model → Entrenar (YAML Mode Training)");
    println!("  • quantize    → Entrenar (4-bit/8-bit quantization)");
    println!("  • generate    → Realizar (GGUF inference)");
    println!("  • query       → Trueno-DB (SQL analytics)");
    println!();
    println!("Usage:");
    println!("  # Create MCP server with pforge");
    println!("  pforge new my-ml-server");
    println!("  cd my-ml-server");
    println!("  pforge serve");
    println!();
    println!("  # Or use pmcp directly for custom implementations");
    println!("  cargo add pmcp");

    Ok(())
}
