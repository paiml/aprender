
/// Stream generate handler - generates tokens one by one via Server-Sent Events
pub async fn stream_generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    // NOTE: Streaming via CUDA model uses /v1/chat/completions endpoint with stream=true
    // This handler uses the CPU model path; for GPU streaming use OpenAI-compatible endpoint

    // Get model and tokenizer
    let (model, tokenizer) = state.get_model(request.model_id.as_deref()).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Tokenize prompt
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Prompt cannot be empty".to_string(),
            }),
        ));
    }

    // Convert to usize for model
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();
    let prompt_len = prompt.len();

    // Build generation config
    let strategy = match request.strategy.as_str() {
        "greedy" => SamplingStrategy::Greedy,
        "top_k" => SamplingStrategy::TopK { k: request.top_k },
        "top_p" => SamplingStrategy::TopP { p: request.top_p },
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid strategy: {}", request.strategy),
                }),
            ));
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);

    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    // Generate all tokens (in future, this will be truly streaming token-by-token)
    let generated = match model.generate(&prompt, &config) {
        Ok(tokens) => tokens,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            ));
        },
    };

    // Convert to u32 with proper overflow handling
    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("Token ID {id} exceeds u32 range"),
                    }),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Create stream that emits tokens one by one
    let tokenizer_clone = tokenizer;
    let stream = async_stream::stream! {
        // Skip prompt tokens, only stream generated tokens
        for &token_id in &token_ids[prompt_len..] {
            // Decode single token
            let text = match tokenizer_clone.decode(&[token_id]) {
                Ok(t) => t,
                Err(_) => String::from("<error>"),
            };

            let event = StreamTokenEvent { token_id, text };
            // Serialization of simple struct should not fail, but handle gracefully
            let data = serde_json::to_string(&event)
                .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());

            yield Ok::<_, Infallible>(Event::default().event("token").data(data));
        }

        // Send done event
        let done_event = StreamDoneEvent {
            num_generated: token_ids.len() - prompt_len,
        };
        // Serialization of simple struct should not fail, but handle gracefully
        let data = serde_json::to_string(&done_event)
            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());
        yield Ok(Event::default().event("done").data(data));
    };

    Ok(Sse::new(stream))
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
#[path = "gpu_handlers_tests.rs"]
mod gpu_handlers_tests;
