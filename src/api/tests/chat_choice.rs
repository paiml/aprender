
#[test]
fn test_chat_choice_clone_debug() {
    let choice = ChatChoice {
        index: 1,
        message: ChatMessage {
            role: "user".to_string(),
            content: "test".to_string(),
            name: Some("John".to_string()),
        },
        finish_reason: "length".to_string(),
    };
    let cloned = choice.clone();
    assert_eq!(cloned.index, choice.index);
    assert_eq!(cloned.message.name, Some("John".to_string()));

    let debug = format!("{:?}", choice);
    assert!(debug.contains("ChatChoice"));
}

// =============================================================================
// CompletionChoice and ReloadResponse Tests
// =============================================================================

#[test]
fn test_completion_choice_serialization() {
    let choice = CompletionChoice {
        text: "Generated text here".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("Generated text here"));
    assert!(json.contains("stop"));

    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.text, "Generated text here");
    assert_eq!(parsed.finish_reason, "stop");
}

#[test]
fn test_completion_choice_with_logprobs() {
    let choice = CompletionChoice {
        text: "test".to_string(),
        index: 1,
        logprobs: Some(serde_json::json!({"token_logprobs": [-0.5, -1.2]})),
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("token_logprobs"));
    assert!(json.contains("-0.5"));

    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.logprobs.is_some());
}

#[test]
fn test_completion_choice_clone_debug() {
    let choice = CompletionChoice {
        text: "debug".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };
    let cloned = choice.clone();
    assert_eq!(cloned.text, choice.text);

    let debug = format!("{:?}", choice);
    assert!(debug.contains("CompletionChoice"));
}

#[test]
fn test_reload_response_serialization() {
    let response = ReloadResponse {
        success: true,
        message: "Model reloaded successfully".to_string(),
        reload_time_ms: 1234,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("true"));
    assert!(json.contains("Model reloaded successfully"));
    assert!(json.contains("1234"));

    let parsed: ReloadResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.success);
    assert_eq!(parsed.reload_time_ms, 1234);
}

#[test]
fn test_reload_response_failure() {
    let response = ReloadResponse {
        success: false,
        message: "Failed to load model: file not found".to_string(),
        reload_time_ms: 50,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: ReloadResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(!parsed.success);
    assert!(parsed.message.contains("file not found"));
}

#[test]
fn test_reload_response_clone_debug() {
    let response = ReloadResponse {
        success: true,
        message: "OK".to_string(),
        reload_time_ms: 100,
    };
    let cloned = response.clone();
    assert_eq!(cloned.success, response.success);

    let debug = format!("{:?}", response);
    assert!(debug.contains("ReloadResponse"));
}

// =============================================================================
// Usage Tests
// =============================================================================

#[test]
fn test_usage_serialization() {
    let usage = Usage {
        prompt_tokens: 150,
        completion_tokens: 300,
        total_tokens: 450,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    assert!(json.contains("150"));
    assert!(json.contains("300"));
    assert!(json.contains("450"));

    let parsed: Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.prompt_tokens, 150);
    assert_eq!(parsed.completion_tokens, 300);
    assert_eq!(parsed.total_tokens, 450);
}

#[test]
fn test_usage_clone_debug() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let cloned = usage.clone();
    assert_eq!(cloned.total_tokens, usage.total_tokens);

    let debug = format!("{:?}", usage);
    assert!(debug.contains("Usage"));
}

// =============================================================================
// ChatMessage Extended Tests
// =============================================================================

#[test]
fn test_chat_message_with_name() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello from Alice".to_string(),
        name: Some("Alice".to_string()),
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    assert!(json.contains("Alice"));

    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.name, Some("Alice".to_string()));
}

#[test]
fn test_chat_message_all_roles() {
    let roles = ["system", "user", "assistant", "function", "tool"];
    for role in roles {
        let msg = ChatMessage {
            role: role.to_string(),
            content: format!("Message from {}", role),
            name: None,
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.role, role);
    }
}

#[test]
fn test_chat_message_unicode() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello 世界! 🌍 مرحبا Привет".to_string(),
        name: Some("用户".to_string()),
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.content.contains("世界"));
    assert!(parsed.content.contains("🌍"));
    assert_eq!(parsed.name, Some("用户".to_string()));
}
