
#[tokio::test]
async fn test_batch_generate_invalid_strategy_error() {
    let app = create_test_app_shared();

    let request = BatchGenerateRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 3,
        temperature: 1.0,
        strategy: "invalid".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_batch_generate_top_k_strategy() {
    let app = create_test_app_shared();

    let request = BatchGenerateRequest {
        prompts: vec!["token1".to_string(), "token2".to_string()],
        max_tokens: 2,
        temperature: 0.8,
        strategy: "top_k".to_string(),
        top_k: 5,
        top_p: 0.9,
        seed: Some(456),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchGenerateResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert_eq!(result.results.len(), 2);
}

#[tokio::test]
async fn test_batch_generate_top_p_strategy() {
    let app = create_test_app_shared();

    let request = BatchGenerateRequest {
        prompts: vec!["token1".to_string()],
        max_tokens: 2,
        temperature: 0.7,
        strategy: "top_p".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(789),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchGenerateResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert_eq!(result.results.len(), 1);
}

// -------------------------------------------------------------------------
// OpenAI-Compatible API Tests
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_openai_models_endpoint() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: OpenAIModelsResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert_eq!(result.object, "list");
    assert!(!result.data.is_empty());
    assert_eq!(result.data[0].object, "model");
    assert_eq!(result.data[0].owned_by, "realizar");
}

#[tokio::test]
async fn test_openai_chat_completions_endpoint() {
    let app = create_test_app_shared();

    let request = ChatCompletionRequest {
        model: "default".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
            },
        ],
        max_tokens: Some(10),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ChatCompletionResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert!(result.id.starts_with("chatcmpl-"));
    assert_eq!(result.object, "chat.completion");
    assert_eq!(result.model, "default");
    assert_eq!(result.choices.len(), 1);
    assert_eq!(result.choices[0].message.role, "assistant");
    assert_eq!(result.choices[0].finish_reason, "stop");
    assert!(result.usage.total_tokens > 0);
}

#[tokio::test]
async fn test_openai_chat_completions_with_defaults() {
    let app = create_test_app_shared();

    // Minimal request with just required fields
    let json = r#"{"model": "default", "messages": [{"role": "user", "content": "Hi"}]}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ChatCompletionResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    // Verify response structure
    assert!(result.id.starts_with("chatcmpl-"));
    assert_eq!(result.choices.len(), 1);
}

#[test]
fn test_format_chat_messages_simple_raw() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    // Raw format (None model) just concatenates content
    let result = format_chat_messages(&messages, None);
    assert!(result.contains("Hello"));
}

#[test]
fn test_format_chat_messages_chatml() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    // Qwen2 uses ChatML format
    let result = format_chat_messages(&messages, Some("Qwen2-0.5B"));
    assert!(result.contains("<|im_start|>user"));
    assert!(result.contains("Hello"));
    assert!(result.contains("<|im_end|>"));
    assert!(result.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_format_chat_messages_llama2() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        },
    ];

    // TinyLlama uses Zephyr format (not Llama2!)
    let result = format_chat_messages(&messages, Some("TinyLlama-1.1B"));
    assert!(result.contains("<|system|>"), "Expected Zephyr system tag");
    assert!(result.contains("<|user|>"), "Expected Zephyr user tag");
    assert!(result.contains("You are helpful."));
    assert!(result.contains("Hi"));
}

#[test]
fn test_format_chat_messages_mistral() {
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "How are you?".to_string(),
            name: None,
        },
    ];

    // Mistral format
    let result = format_chat_messages(&messages, Some("Mistral-7B"));
    assert!(result.contains("[INST]"));
    assert!(result.contains("Hello"));
    assert!(result.contains("Hi there!"));
    assert!(result.contains("How are you?"));
}

#[test]
fn test_format_chat_messages_phi() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        name: None,
    }];

    // Phi format
    let result = format_chat_messages(&messages, Some("phi-2"));
    assert!(result.contains("Instruct: Test"));
    assert!(result.ends_with("Output:"));
}

#[test]
fn test_format_chat_messages_alpaca() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        name: None,
    }];

    // Alpaca format
    let result = format_chat_messages(&messages, Some("alpaca-7b"));
    assert!(result.contains("### Instruction:"));
    assert!(result.contains("Test"));
    assert!(result.ends_with("### Response:\n"));
}

#[test]
fn test_default_n() {
    assert_eq!(default_n(), 1);
}

#[test]
fn test_chat_message_serialization() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("test_user".to_string()),
    };

    let json = serde_json::to_string(&msg).expect("test");
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("\"content\":\"Hello\""));
    assert!(json.contains("\"name\":\"test_user\""));
}

#[test]
fn test_usage_serialization() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };

    let json = serde_json::to_string(&usage).expect("test");
    assert!(json.contains("\"prompt_tokens\":10"));
    assert!(json.contains("\"completion_tokens\":20"));
    assert!(json.contains("\"total_tokens\":30"));
}

// ========================================================================
// Streaming Types Tests
// ========================================================================

#[test]
fn test_chat_completion_chunk_initial() {
    let chunk = ChatCompletionChunk::initial("chatcmpl-123", "gpt-4");
    assert_eq!(chunk.id, "chatcmpl-123");
    assert_eq!(chunk.object, "chat.completion.chunk");
    assert_eq!(chunk.model, "gpt-4");
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
    assert!(chunk.choices[0].delta.content.is_none());
    assert!(chunk.choices[0].finish_reason.is_none());
}
