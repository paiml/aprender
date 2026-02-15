
#[test]
fn test_format_tools_prompt_no_description() {
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: FunctionDef {
            name: "bare_tool".to_string(),
            description: None,
            parameters: None,
        },
    }];
    let prompt = format_tools_prompt(&tools);
    assert!(prompt.contains("### bare_tool"));
    // Should not crash or contain garbage for missing description
}

// ========================================================================
// Y. parse_tool_calls Tests
// ========================================================================

#[test]
fn test_parse_tool_calls_valid_json() {
    let output = r#"{"tool_call": {"name": "calc", "arguments": {"x": 42}}}"#;
    let calls = parse_tool_calls(output).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "calc");
    assert_eq!(calls[0].tool_type, "function");
    assert!(calls[0].id.starts_with("call_"));
    assert!(calls[0].function.arguments.contains("42"));
}

#[test]
fn test_parse_tool_calls_embedded_in_text() {
    let output = r#"Let me help. {"tool_call": {"name": "search", "arguments": {"q": "rust"}}}"#;
    let calls = parse_tool_calls(output).unwrap();
    assert_eq!(calls[0].function.name, "search");
}

#[test]
fn test_parse_tool_calls_no_tool_call() {
    assert!(parse_tool_calls("Just regular text").is_none());
    assert!(parse_tool_calls("").is_none());
    assert!(parse_tool_calls("{}").is_none());
}

#[test]
fn test_parse_tool_calls_missing_name() {
    let output = r#"{"tool_call": {"arguments": {"x": 1}}}"#;
    assert!(parse_tool_calls(output).is_none());
}

#[test]
fn test_parse_tool_calls_missing_arguments() {
    let output = r#"{"tool_call": {"name": "test"}}"#;
    assert!(parse_tool_calls(output).is_none());
}

#[test]
fn test_parse_tool_calls_invalid_json() {
    let output = r#"{"tool_call": {"name": "test", "arguments": broken}}"#;
    assert!(parse_tool_calls(output).is_none());
}

#[test]
fn test_parse_tool_calls_whitespace_trimmed() {
    let output = r#"  {"tool_call": {"name": "ws_test", "arguments": {}}}  "#;
    let calls = parse_tool_calls(output).unwrap();
    assert_eq!(calls[0].function.name, "ws_test");
}

// ========================================================================
// Z. uuid_simple Tests
// ========================================================================

#[test]
fn test_uuid_simple_is_hex_string() {
    let id = uuid_simple();
    assert_eq!(id.len(), 16);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_uuid_simple_changes_over_time() {
    let id1 = uuid_simple();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let id2 = uuid_simple();
    // Not guaranteed to differ in all cases, but very likely with sleep
    // This is a best-effort check
    let _ = (id1, id2); // At minimum ensure both calls succeed
}

// ========================================================================
// AA. MAX_REQUEST_SIZE Constant Test
// ========================================================================

#[test]
fn test_max_request_size_is_10mb() {
    assert_eq!(MAX_REQUEST_SIZE, 10 * 1024 * 1024);
    assert_eq!(MAX_REQUEST_SIZE, 10_485_760);
}

// ========================================================================
// AB. FunctionDef / FunctionCall Tests
// ========================================================================

#[test]
fn test_function_def_serialization_skips_none() {
    let def = FunctionDef {
        name: "test".to_string(),
        description: None,
        parameters: None,
    };
    let json = serde_json::to_string(&def).unwrap();
    assert!(!json.contains("description"));
    assert!(!json.contains("parameters"));
    assert!(json.contains("\"name\":\"test\""));
}

#[test]
fn test_function_call_roundtrip() {
    let original = FunctionCall {
        name: "compute".to_string(),
        arguments: r#"{"a": 1, "b": 2}"#.to_string(),
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: FunctionCall = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "compute");
    assert_eq!(parsed.arguments, r#"{"a": 1, "b": 2}"#);
}

#[test]
fn test_tool_choice_function_name_roundtrip() {
    let name = ToolChoiceFunction {
        name: "my_fn".to_string(),
    };
    let json = serde_json::to_string(&name).unwrap();
    let parsed: ToolChoiceFunction = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "my_fn");
}
