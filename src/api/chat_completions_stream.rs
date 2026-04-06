
/// OpenAI-compatible /v1/chat/completions streaming endpoint (SSE)
pub async fn openai_chat_completions_stream_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = state.get_model(model_id).map_err(|e| {
        state.metrics.record_failure();
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let prompt_text = format_chat_messages(&request.messages, Some(&request.model));
    let prompt_ids = tokenizer.encode(&prompt_text);
    if prompt_ids.is_empty() {
        state.metrics.record_failure();
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Messages cannot be empty".to_string(),
            }),
        ));
    }

    let prompt_len = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    // GH-665: Cap max_tokens to prevent hangs on large values
    let max_tokens = request.max_tokens.unwrap_or(256).min(4096);
    let temperature = request.temperature.unwrap_or(0.7);

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p };
    }

    let request_id = format!(
        "chatcmpl-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let generated = model.generate(&prompt, &config).map_err(|e| {
        state.metrics.record_failure();
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let token_ids: Vec<u32> = generated
        .iter()
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();

    let generated_ids = token_ids[prompt_len..].to_vec();
    let model_name = request.model.clone();
    let request_id_clone = request_id.clone();
    let tokenizer_clone = tokenizer;

    let stream = async_stream::stream! {
        let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
        let data = serde_json::to_string(&initial).unwrap_or_default();
        yield Ok(Event::default().data(format!("data: {}\n", data)));

        for &token_id in &generated_ids {
            let text = match tokenizer_clone.decode(&[token_id]) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &text);
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            yield Ok(Event::default().data(format!("data: {}\n", data)));
        }

        let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
        let data = serde_json::to_string(&done).unwrap_or_default();
        yield Ok(Event::default().data(format!("data: {}\n", data)));

        yield Ok(Event::default().data("data: [DONE]\n".to_string()));
    };

    Ok(Sse::new(stream))
}
