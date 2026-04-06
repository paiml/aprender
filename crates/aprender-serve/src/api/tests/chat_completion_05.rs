
#[test]
fn test_chat_completion_request_debug() {
    let req = ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };
    let debug = format!("{:?}", req);
    assert!(debug.contains("ChatCompletionRequest"));
}

// ============================================================================
// ChatCompletionResponse coverage
// ============================================================================

#[test]
fn test_chat_completion_response_debug() {
    let resp = ChatCompletionResponse {
        id: "id".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };
    let debug = format!("{:?}", resp);
    assert!(debug.contains("ChatCompletionResponse"));
}

#[test]
fn test_chat_completion_response_with_traces() {
    let resp = ChatCompletionResponse {
        id: "id".to_string(),
        object: "chat.completion".to_string(),
        created: 123,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
            total_tokens: 15,
        },
        brick_trace: Some(TraceData {
            level: "brick".to_string(),
            operations: 1,
            total_time_us: 100,
            breakdown: vec![],
            provenance: TraceProvenance::Estimated,
        }),
        step_trace: None,
        layer_trace: None,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    assert!(json.contains("brick"));
    let parsed: ChatCompletionResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.brick_trace.is_some());
}

// ============================================================================
// ErrorResponse coverage
// ============================================================================

#[test]
fn test_error_response_serde_roundtrip() {
    let err = ErrorResponse {
        error: "Something failed".to_string(),
    };
    let json = serde_json::to_string(&err).expect("JSON serialization failed");
    assert!(json.contains("Something failed"));
    let parsed: ErrorResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.error, "Something failed");
}
