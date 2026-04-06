
#[test]
fn test_chat_completion_response_serialize_cov() {
    let resp = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1677652288,
        model: "phi-2".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("chatcmpl-123"));
    assert!(json.contains("Hello!"));
}

#[test]
fn test_chat_completion_chunk_serialize_cov() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-chunk-123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1677652288,
        model: "phi-2".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: Some("Hi".to_string()),
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).expect("serialize");
    assert!(json.contains("chunk"));
    assert!(json.contains("Hi"));
}

// =========================================================================
// Additional Coverage Tests: Usage struct
// =========================================================================

#[test]
fn test_usage_debug_clone_cov() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };
    let debug = format!("{:?}", usage);
    assert!(debug.contains("Usage"));
    assert!(debug.contains("100"));

    let cloned = usage.clone();
    assert_eq!(cloned.prompt_tokens, usage.prompt_tokens);
    assert_eq!(cloned.total_tokens, usage.total_tokens);
}

// =========================================================================
// Additional Coverage Tests: ChatDelta struct
// =========================================================================
