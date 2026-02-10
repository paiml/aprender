//! Tool Calling Demo (GH-160, PMAT-186)
//!
//! Demonstrates OpenAI-compatible tool calling with apr serve.
//!
//! # Usage
//!
//! First start the server:
//! ```bash
//! apr serve model.gguf --port 8080
//! ```
//!
//! Then run this example:
//! ```bash
//! cargo run --example tool_calling_demo
//! ```
//!
//! # Tool Calling Flow
//!
//! 1. Client sends request with `tools` array defining available functions
//! 2. Model generates response, optionally invoking tools
//! 3. If model invokes tool, response includes `tool_calls` with `finish_reason: "tool_calls"`
//! 4. Client executes tool and sends result back with `tool_call_id`
//! 5. Model generates final response incorporating tool result

use serde::{Deserialize, Serialize};

/// Tool definition (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: FunctionDef,
}

#[derive(Debug, Serialize)]
struct FunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Chat message
#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

/// Chat completion request
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

/// Chat completion response
#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
    id: String,
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Choice {
    message: ChatMessage,
    finish_reason: Option<String>,
}

fn main() {
    println!("=== Tool Calling Demo (GH-160) ===\n");

    // Define available tools
    let tools = vec![
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: "Get the current weather for a location".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. 'New York'"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "calculate".to_string(),
                description: "Perform a mathematical calculation".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate, e.g. '2 + 2'"
                        }
                    },
                    "required": ["expression"]
                }),
            },
        },
    ];

    // Create request with tools
    let request = ChatRequest {
        model: "qwen".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some("What's the weather in Tokyo?".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Some(tools),
        max_tokens: Some(100),
    };

    println!("Request:");
    println!("{}\n", serde_json::to_string_pretty(&request).expect("JSON serialization"));

    // Example response with tool call
    let example_response = ChatResponse {
        id: "chatcmpl-123".to_string(),
        choices: vec![Choice {
            message: ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_abc123".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "get_weather".to_string(),
                        arguments: r#"{"location": "Tokyo", "unit": "celsius"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
            },
            finish_reason: Some("tool_calls".to_string()),
        }],
    };

    println!("Example Response (with tool call):");
    println!(
        "{}\n",
        serde_json::to_string_pretty(&example_response).expect("JSON serialization")
    );

    // Show multi-turn conversation with tool result
    println!("Multi-turn conversation with tool result:\n");

    let messages_with_tool_result = vec![
        ChatMessage {
            role: "user".to_string(),
            content: Some("What's the weather in Tokyo?".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_abc123".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "get_weather".to_string(),
                    arguments: r#"{"location": "Tokyo"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
        },
        ChatMessage {
            role: "tool".to_string(),
            content: Some(
                r#"{"temperature": 22, "condition": "sunny", "humidity": 65}"#.to_string(),
            ),
            tool_calls: None,
            tool_call_id: Some("call_abc123".to_string()),
        },
    ];

    let follow_up_request = ChatRequest {
        model: "qwen".to_string(),
        messages: messages_with_tool_result,
        tools: None, // Tools already defined in context
        max_tokens: Some(100),
    };

    println!("Follow-up request with tool result:");
    println!(
        "{}\n",
        serde_json::to_string_pretty(&follow_up_request).expect("JSON serialization")
    );

    println!("---");
    println!("To test with a real server:");
    println!("  1. Start: apr serve model.gguf --port 8080");
    println!("  2. POST to http://localhost:8080/v1/chat/completions");
    println!("\nNote: Model must be trained/fine-tuned to output tool call JSON.");
}
