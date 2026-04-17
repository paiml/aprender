
#[test]
fn test_completion_request_minimal() {
    let json = r#"{"model":"m","prompt":"p"}"#;
    let parsed: CompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.model, "m");
    assert!(parsed.max_tokens.is_none());
    assert!(parsed.temperature.is_none());
}

#[test]
fn test_completion_response_serde() {
    let resp = CompletionResponse {
        id: "cmpl-test-123".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![CompletionChoice {
            text: "Generated text".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: crate::api::Usage {
            prompt_tokens: 5,
            completion_tokens: 3,
            total_tokens: 8,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: CompletionResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "cmpl-test-123");
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].finish_reason, "stop");
    assert_eq!(parsed.usage.total_tokens, 8);
}

#[test]
fn test_completion_choice_with_logprobs() {
    let choice = CompletionChoice {
        text: "Hello".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({"tokens": ["Hello"], "token_logprobs": [-0.5]})),
        finish_reason: "length".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("serialize");
    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.logprobs.is_some());
}

// ============================================================================
// build_trace_data Tests
// ============================================================================

#[test]
fn test_build_trace_data_brick() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("brick"), 1000, 10, 5, 4);
    assert!(brick.is_some());
    assert!(step.is_none());
    assert!(layer.is_none());
    let brick = brick.expect("brick trace");
    assert_eq!(brick.level, "brick");
    assert_eq!(brick.operations, 5);
    assert_eq!(brick.total_time_us, 1000);
    assert_eq!(brick.breakdown.len(), 1);
    assert_eq!(brick.breakdown[0].name, "total_inference");
    assert_eq!(brick.breakdown[0].time_us, 1000);
    assert!(brick.breakdown[0].details.as_ref().expect("details").contains("apr profile"));
}

#[test]
fn test_build_trace_data_step() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("step"), 2000, 20, 10, 8);
    assert!(brick.is_none());
    assert!(step.is_some());
    assert!(layer.is_none());
    let step = step.expect("step trace");
    assert_eq!(step.level, "step");
    assert_eq!(step.breakdown.len(), 1);
    assert_eq!(step.breakdown[0].name, "total_inference");
    assert_eq!(step.breakdown[0].time_us, 2000);
    assert!(step.breakdown[0].details.as_ref().expect("details").contains("apr profile"));
}

#[test]
fn test_build_trace_data_layer() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("layer"), 3000, 10, 5, 6);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());
    let layer = layer.expect("layer trace");
    assert_eq!(layer.level, "layer");
    assert_eq!(layer.operations, 6);
    assert_eq!(layer.breakdown.len(), 1);
    assert_eq!(layer.breakdown[0].name, "total_inference");
    assert_eq!(layer.breakdown[0].time_us, 3000);
    let details = layer.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("6 layers"));
    assert!(details.contains("apr profile"));
}

#[test]
fn test_build_trace_data_none() {
    let (brick, step, layer) = crate::api::build_trace_data(None, 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("unknown"), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

// ============================================================================
// AppState Constructor Tests (non-async, pure logic)
// ============================================================================

#[test]
fn test_app_state_demo_mock() {
    let state = crate::api::AppState::demo_mock();
    assert!(state.is_ok());
}

#[test]
fn test_app_state_with_cache() {
    let state = crate::api::AppState::with_cache(10);
    // Should create state without panicking
    let _ = state;
}

// ============================================================================
// API Types Default/Serde Tests (types.rs coverage)
// ============================================================================

#[test]
fn test_default_max_tokens() {
    assert_eq!(crate::api::default_max_tokens(), 50);
}

#[test]
fn test_default_top_k() {
    assert_eq!(crate::api::default_top_k(), 50);
}

#[cfg(test)]
#[test]
fn test_default_strategy() {
    assert_eq!(crate::api::default_strategy(), "greedy");
}

#[cfg(test)]
#[test]
fn test_default_temperature() {
    assert!((crate::api::default_temperature() - 1.0).abs() < f32::EPSILON);
}

#[cfg(test)]
#[test]
fn test_default_top_p() {
    assert!((crate::api::default_top_p() - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_error_response_serde() {
    let err = crate::api::ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&err).expect("serialize");
    let parsed: crate::api::ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.error, "Something went wrong");
}

#[test]
fn test_health_response_serde() {
    let resp = crate::api::HealthResponse {
        status: "ok".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: crate::api::HealthResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.status, "ok");
}

#[test]
fn test_tokenize_request_serde() {
    let req = crate::api::TokenizeRequest {
        text: "Hello world".to_string(),
        model_id: None,
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: crate::api::TokenizeRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.text, "Hello world");
}

#[test]
fn test_tokenize_response_serde() {
    let resp = crate::api::TokenizeResponse {
        token_ids: vec![1, 2, 3],
        num_tokens: 3,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: crate::api::TokenizeResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.token_ids, vec![1, 2, 3]);
    assert_eq!(parsed.num_tokens, 3);
}

#[test]
fn test_generate_request_defaults() {
    let json = r#"{"prompt":"test","strategy":"greedy"}"#;
    let parsed: crate::api::GenerateRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.prompt, "test");
    assert_eq!(parsed.max_tokens, 50); // default
    assert_eq!(parsed.top_k, 50); // default
}

#[test]
fn test_batch_tokenize_request_serde() {
    let req = crate::api::BatchTokenizeRequest {
        texts: vec!["hello".to_string(), "world".to_string()],
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: crate::api::BatchTokenizeRequest =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.texts.len(), 2);
}

#[test]
fn test_batch_generate_request_defaults() {
    let json = r#"{"prompts":["a","b"]}"#;
    let parsed: crate::api::BatchGenerateRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.prompts.len(), 2);
    assert_eq!(parsed.max_tokens, 50); // default
}

#[test]
fn test_stream_token_event_serde() {
    let evt = crate::api::StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&evt).expect("serialize");
    let parsed: crate::api::StreamTokenEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.token_id, 42);
}

#[test]
fn test_stream_done_event_serde() {
    let evt = crate::api::StreamDoneEvent { num_generated: 10 };
    let json = serde_json::to_string(&evt).expect("serialize");
    let parsed: crate::api::StreamDoneEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.num_generated, 10);
}

#[test]
fn test_models_response_serde() {
    let resp = crate::api::ModelsResponse {
        models: vec![crate::api::ModelInfo {
            id: "model-1".to_string(),
            name: "Test".to_string(),
            description: "A test model".to_string(),
            format: "gguf".to_string(),
            loaded: true,
        }],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: crate::api::ModelsResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.models.len(), 1);
    assert_eq!(parsed.models[0].id, "model-1");
}
