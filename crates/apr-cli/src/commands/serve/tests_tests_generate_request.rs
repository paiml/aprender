
// ========================================================================
// P. GenerateRequest / GenerateResponse Tests
// ========================================================================

#[test]
fn test_generate_request_full_deserialization() {
    let json = r#"{
        "prompt": "Hello world",
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": true,
        "stop": [".", "!"]
    }"#;
    let req: GenerateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "Hello world");
    assert_eq!(req.max_tokens, 128);
    assert!((req.temperature - 0.7).abs() < f32::EPSILON);
    assert!(req.stream);
    assert_eq!(req.stop.len(), 2);
    assert_eq!(req.stop[0], ".");
    assert_eq!(req.stop[1], "!");
}

#[test]
fn test_generate_request_minimal() {
    // Only prompt is required, everything else has defaults
    let json = r#"{"prompt": "test"}"#;
    let req: GenerateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "test");
    assert_eq!(req.max_tokens, 256);
    assert!((req.temperature - 1.0).abs() < f32::EPSILON);
    assert!(!req.stream);
    assert!(req.stop.is_empty());
}

#[test]
fn test_generate_response_serialization() {
    let resp = GenerateResponse {
        text: "Hello there".to_string(),
        tokens_generated: 5,
        finish_reason: "stop".to_string(),
        latency_ms: 123,
    };

    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"text\":\"Hello there\""));
    assert!(json.contains("\"tokens_generated\":5"));
    assert!(json.contains("\"finish_reason\":\"stop\""));
    assert!(json.contains("\"latency_ms\":123"));
}

#[test]
fn test_generate_response_roundtrip() {
    let original = GenerateResponse {
        text: "Output text".to_string(),
        tokens_generated: 42,
        finish_reason: "length".to_string(),
        latency_ms: 567,
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: GenerateResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.text, "Output text");
    assert_eq!(parsed.tokens_generated, 42);
    assert_eq!(parsed.finish_reason, "length");
    assert_eq!(parsed.latency_ms, 567);
}

// ========================================================================
// Q. StreamEvent Comprehensive Tests
// ========================================================================

#[test]
fn test_stream_event_token_fields() {
    let event = StreamEvent::token("world", 42);
    assert_eq!(event.event, "token");
    assert_eq!(event.data, "world");
    assert_eq!(event.token_id, Some(42));
}

#[test]
fn test_stream_event_done_fields() {
    let event = StreamEvent::done("length", 100);
    assert_eq!(event.event, "done");
    assert!(event.data.contains("\"finish_reason\":\"length\""));
    assert!(event.data.contains("\"tokens_generated\":100"));
    assert!(event.token_id.is_none());
}

#[test]
fn test_stream_event_error_fields() {
    let event = StreamEvent::error("Out of memory");
    assert_eq!(event.event, "error");
    assert_eq!(event.data, "Out of memory");
    assert!(event.token_id.is_none());
}

#[test]
fn test_stream_event_to_sse_format() {
    let event = StreamEvent::token("hi", 1);
    let sse = event.to_sse();
    // SSE format: "event: <type>\ndata: <data>\n\n"
    assert!(sse.starts_with("event: token\n"));
    assert!(sse.contains("data: hi\n"));
    assert!(sse.ends_with("\n\n"));
}

#[test]
fn test_stream_event_done_sse_format() {
    let event = StreamEvent::done("stop", 5);
    let sse = event.to_sse();
    assert!(sse.starts_with("event: done\n"));
    assert!(sse.contains("data: "));
    assert!(sse.ends_with("\n\n"));
}

#[test]
fn test_stream_event_serialization() {
    let event = StreamEvent::token("test", 99);
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event\":\"token\""));
    assert!(json.contains("\"data\":\"test\""));
    assert!(json.contains("\"token_id\":99"));
}

#[test]
fn test_stream_event_done_serialization_skips_token_id() {
    let event = StreamEvent::done("stop", 10);
    let json = serde_json::to_string(&event).unwrap();
    // token_id is None -> should be skipped
    assert!(!json.contains("token_id"));
}

// ========================================================================
// R. TranscribeRequest / TranscribeResponse Tests
// ========================================================================

#[test]
fn test_transcribe_request_defaults() {
    let json = r#"{}"#;
    let req: TranscribeRequest = serde_json::from_str(json).unwrap();
    assert!(req.language.is_none());
    assert_eq!(req.task, "transcribe");
}

#[test]
fn test_transcribe_request_with_language() {
    let json = r#"{"language": "fr", "task": "translate"}"#;
    let req: TranscribeRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.language.as_deref(), Some("fr"));
    assert_eq!(req.task, "translate");
}

#[test]
fn test_transcribe_response_roundtrip() {
    let original = TranscribeResponse {
        text: "Hello world".to_string(),
        language: "en".to_string(),
        duration_seconds: 3.5,
        latency_ms: 200,
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: TranscribeResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.text, "Hello world");
    assert_eq!(parsed.language, "en");
    assert!((parsed.duration_seconds - 3.5).abs() < f64::EPSILON);
    assert_eq!(parsed.latency_ms, 200);
}

// ========================================================================
// S. ServerInfo Tests
// ========================================================================

#[test]
fn test_server_info_serialization() {
    let info = ServerInfo {
        name: "apr-serve".to_string(),
        version: "0.25.1".to_string(),
        model_id: "my-model".to_string(),
    };
    let json = serde_json::to_string(&info).unwrap();
    assert!(json.contains("\"name\":\"apr-serve\""));
    assert!(json.contains("\"version\":\"0.25.1\""));
    assert!(json.contains("\"model_id\":\"my-model\""));
}

#[test]
fn test_server_info_roundtrip() {
    let original = ServerInfo {
        name: "test".to_string(),
        version: "1.2.3".to_string(),
        model_id: "model-abc".to_string(),
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: ServerInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "test");
    assert_eq!(parsed.version, "1.2.3");
    assert_eq!(parsed.model_id, "model-abc");
}

// ========================================================================
// T. ChatMessage Tests
// ========================================================================

#[test]
fn test_chat_message_user_message() {
    let json = r#"{"role": "user", "content": "Hello"}"#;
    let msg: ChatMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.role, "user");
    assert_eq!(msg.content.as_deref(), Some("Hello"));
    assert!(msg.tool_calls.is_none());
    assert!(msg.tool_call_id.is_none());
    assert!(msg.name.is_none());
}

#[test]
fn test_chat_message_system_message() {
    let json = r#"{"role": "system", "content": "You are helpful."}"#;
    let msg: ChatMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.role, "system");
    assert_eq!(msg.content.as_deref(), Some("You are helpful."));
}

#[test]
fn test_chat_message_assistant_with_tool_calls() {
    let json = r#"{
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "calc", "arguments": "{\"x\":1}"}
        }]
    }"#;
    let msg: ChatMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.role, "assistant");
    assert!(msg.content.is_none());
    assert!(msg.tool_calls.is_some());
    let calls = msg.tool_calls.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].function.name, "calc");
}

#[test]
fn test_chat_message_serialization_skips_none() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: Some("Hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(!json.contains("tool_calls"));
    assert!(!json.contains("tool_call_id"));
    assert!(!json.contains("name"));
}

// ========================================================================
// U. ChatCompletionRequest / Response Tests
// ========================================================================

#[test]
fn test_chat_completion_request_minimal() {
    let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, ""); // default
    assert_eq!(req.messages.len(), 1);
    assert!(req.tools.is_none());
    assert!(req.tool_choice.is_none());
    assert!(req.max_tokens.is_none());
    assert!(!req.stream);
    assert!(req.temperature.is_none());
    assert!(req.top_p.is_none());
}

#[test]
fn test_chat_completion_request_full() {
    let json = r#"{
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100,
        "stream": true,
        "temperature": 0.5,
        "top_p": 0.9
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "gpt-4");
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.max_tokens, Some(100));
    assert!(req.stream);
    assert!((req.temperature.unwrap() - 0.5).abs() < f32::EPSILON);
    assert!((req.top_p.unwrap() - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_chat_completion_response_roundtrip() {
    let original = ChatCompletionResponse {
        id: "chatcmpl-001".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "apr".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: Some("Hello!".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
        }),
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: ChatCompletionResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "chatcmpl-001");
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].message.content.as_deref(), Some("Hello!"));
    assert_eq!(parsed.usage.as_ref().unwrap().total_tokens, 6);
}

// ========================================================================
// V. TokenUsage Tests
// ========================================================================

#[test]
fn test_token_usage_roundtrip() {
    let usage = TokenUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let json = serde_json::to_string(&usage).unwrap();
    let parsed: TokenUsage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.prompt_tokens, 10);
    assert_eq!(parsed.completion_tokens, 20);
    assert_eq!(parsed.total_tokens, 30);
}

// ========================================================================
// W. ToolChoice Tests
// ========================================================================

#[test]
fn test_tool_choice_mode_auto() {
    let json = r#""auto""#;
    let choice: ToolChoice = serde_json::from_str(json).unwrap();
    match choice {
        ToolChoice::Mode(mode) => assert_eq!(mode, "auto"),
        _ => panic!("Expected Mode variant"),
    }
}

#[test]
fn test_tool_choice_mode_none() {
    let json = r#""none""#;
    let choice: ToolChoice = serde_json::from_str(json).unwrap();
    match choice {
        ToolChoice::Mode(mode) => assert_eq!(mode, "none"),
        _ => panic!("Expected Mode variant"),
    }
}

#[test]
fn test_tool_choice_specific_function() {
    let json = r#"{"type": "function", "function": {"name": "get_temp"}}"#;
    let choice: ToolChoice = serde_json::from_str(json).unwrap();
    match choice {
        ToolChoice::Function {
            tool_type,
            function,
        } => {
            assert_eq!(tool_type, "function");
            assert_eq!(function.name, "get_temp");
        }
        _ => panic!("Expected Function variant"),
    }
}

// ========================================================================
// X. format_tools_prompt Tests
// ========================================================================

#[test]
fn test_format_tools_prompt_empty() {
    let result = format_tools_prompt(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_format_tools_prompt_single_tool() {
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: FunctionDef {
            name: "search".to_string(),
            description: Some("Search the web".to_string()),
            parameters: None,
        },
    }];
    let prompt = format_tools_prompt(&tools);
    assert!(prompt.contains("### search"));
    assert!(prompt.contains("Search the web"));
    assert!(prompt.contains("tool_call"));
}

#[test]
fn test_format_tools_prompt_multiple_tools() {
    let tools = vec![
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "tool_a".to_string(),
                description: Some("First tool".to_string()),
                parameters: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "tool_b".to_string(),
                description: Some("Second tool".to_string()),
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        },
    ];
    let prompt = format_tools_prompt(&tools);
    assert!(prompt.contains("### tool_a"));
    assert!(prompt.contains("### tool_b"));
    assert!(prompt.contains("First tool"));
    assert!(prompt.contains("Second tool"));
    assert!(prompt.contains("Parameters:"));
}
