
/// Produce char-boundary-safe streaming text deltas from a fully-generated token list.
///
/// Fixes two bugs on this pregenerated SSE path (PMAT-758):
/// 1. **UTF-8 splitting** — decoding one token at a time runs `String::from_utf8_lossy` on
///    an incomplete byte sequence (a byte-level BPE token can be a single byte of a
///    multi-byte char), so emoji / CJK that span tokens emit U+FFFD replacement chars. We
///    decode cumulative prefixes and only advance the emitted offset once the decoded text
///    no longer ends in U+FFFD — i.e. once the multi-byte char is complete (the HuggingFace
///    `TextStreamer` technique).
/// 2. **Stop sequences ignored** — the cumulative text is truncated at the EARLIEST stop via
///    the shared `truncate_at_stop` helper, and emission stops as soon as a stop matches, so
///    the streamed text never contains a stop string.
fn streaming_text_deltas(
    tokenizer: &BPETokenizer,
    token_ids: &[u32],
    stops: Option<&[String]>,
) -> Vec<String> {
    let mut deltas = Vec::new();
    let mut emitted = 0usize;
    for i in 0..token_ids.len() {
        let Ok(raw) = tokenizer.decode(&token_ids[..=i]) else {
            continue;
        };
        let text = crate::api::realize_handlers::truncate_at_stop(raw.clone(), stops);
        let stop_hit = text.len() < raw.len();
        // Hold back a delta that ends mid-multibyte-char (trailing U+FFFD) until it completes.
        if !stop_hit && text.ends_with('\u{FFFD}') {
            continue;
        }
        if text.len() > emitted && text.is_char_boundary(emitted) {
            deltas.push(text[emitted..].to_string());
            emitted = text.len();
        }
        if stop_hit {
            break;
        }
    }
    deltas
}

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

    // PMAT-758: precompute char-safe, stop-truncated deltas BEFORE streaming. The previous
    // per-token `decode(&[token_id])` split multi-byte UTF-8 (emoji/CJK -> U+FFFD) and
    // ignored request.stop entirely. All tokens are already generated here, so we can decode
    // cumulatively and emit only complete-char, pre-stop deltas.
    let deltas = streaming_text_deltas(&tokenizer, &generated_ids, request.stop.as_deref());

    let stream = async_stream::stream! {
        // PMAT-753: pass ONLY the JSON payload to Event::data() — axum's Sse adds the
        // `data: ` field prefix and the `\n\n` terminator itself. A manual `data: ` prefix
        // would double-prefix the wire and break JSON.parse for every spec-compliant client.
        let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
        let data = serde_json::to_string(&initial).unwrap_or_default();
        yield Ok(Event::default().data(data));

        for delta in &deltas {
            let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, delta);
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            yield Ok(Event::default().data(data));
        }

        let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
        let data = serde_json::to_string(&done).unwrap_or_default();
        yield Ok(Event::default().data(data));

        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Sse::new(stream))
}

#[cfg(test)]
mod pmat758_streaming_delta_tests {
    use super::*;

    fn tok(vocab: &[&str]) -> BPETokenizer {
        BPETokenizer::new(
            vocab.iter().map(|s| (*s).to_string()).collect(),
            vec![],
            "<unk>",
        )
        .expect("test tokenizer")
    }

    #[test]
    fn holds_back_multibyte_utf8_until_complete() {
        // 😀 = U+1F600 = bytes F0 9F 98 80, one byte per token. The old per-token
        // decode(&[id]) ran from_utf8_lossy on each single byte -> four U+FFFD. Cumulative
        // decode must hold back until the char completes, emitting a single "😀".
        let t = tok(&["<unk>", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]);
        let deltas = streaming_text_deltas(&t, &[1, 2, 3, 4], None);
        assert_eq!(deltas.concat(), "😀");
        assert!(
            !deltas.concat().contains('\u{FFFD}'),
            "no replacement chars in streamed deltas"
        );
    }

    #[test]
    fn applies_stop_and_halts_emission() {
        // "abXc" with stop ["X"] -> streamed text is "ab", never contains the stop string.
        let t = tok(&["<unk>", "a", "b", "X", "c"]);
        let deltas = streaming_text_deltas(&t, &[1, 2, 3, 4], Some(&["X".to_string()]));
        assert_eq!(deltas.concat(), "ab");
        assert!(!deltas.concat().contains('X'));
    }

    #[test]
    fn no_stop_streams_full_text() {
        let t = tok(&["<unk>", "a", "b", "X", "c"]);
        let deltas = streaming_text_deltas(&t, &[1, 2, 3, 4], None);
        assert_eq!(deltas.concat(), "abXc");
    }
}
