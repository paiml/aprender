
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

/// Resolve the `GenerationConfig` for a streaming chat completion (PMAT-790).
///
/// `temperature == 0` is the canonical OpenAI request for deterministic (greedy) output, and
/// every non-streaming `/v1/chat/completions` backend honors it via the `top_k == 1` greedy
/// path. The streaming handler previously passed the raw `0.0` into `GenerationConfig`, so
/// `model.generate` -> `sample_token` -> `apply_temperature(0.0)` returned an `InvalidShape`
/// error ("Temperature must be a positive finite number") which the handler mapped to HTTP
/// 500 — so EVERY streaming chat completion with `temperature: 0` was broken.
///
/// This helper forces `Greedy` for `temperature == 0` and substitutes a no-op temperature of
/// `1.0` so the sampler never sees a non-positive scale. For positive temperatures the
/// behavior is unchanged: greedy by default, or top-p when `top_p` is set.
fn resolve_stream_generation_config(
    temperature: f32,
    top_p: Option<f32>,
    max_tokens: usize,
) -> GenerationConfig {
    if temperature == 0.0 {
        // Deterministic: greedy argmax, with a safe (no-op) temperature scale.
        return GenerationConfig::default()
            .with_max_tokens(max_tokens)
            .with_temperature(1.0);
    }

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(p) = top_p {
        config.strategy = SamplingStrategy::TopP { p };
    }
    config
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
    let config = resolve_stream_generation_config(
        request.temperature.unwrap_or(0.7),
        request.top_p,
        max_tokens,
    );

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

// PMAT-790: streaming /v1/chat/completions with `temperature: 0` must not 500. The handler
// builds a GenerationConfig and runs it through `model.generate` -> `sample_token` ->
// `apply_temperature`, which rejects a non-positive temperature. `temperature: 0` is the
// canonical OpenAI deterministic request and is honored by every non-streaming backend; it
// must resolve to a runnable, greedy config here too.
#[cfg(test)]
mod pmat790_stream_temperature_zero_tests {
    use super::resolve_stream_generation_config;
    use crate::generate::{sample_token, SamplingStrategy};
    use crate::tensor::Tensor;

    #[test]
    fn temperature_zero_resolves_to_runnable_greedy_config() {
        // FALSIFIER: pre-fix the handler passed temperature 0.0 straight into the config, so
        // `sample_token` -> `apply_temperature(0.0)` returned Err -> the handler answered HTTP
        // 500 for every streaming chat completion with temperature 0. The resolved config must
        // (a) be greedy and (b) sample without error.
        let config = resolve_stream_generation_config(0.0, None, 16);
        assert_eq!(
            config.strategy,
            SamplingStrategy::Greedy,
            "temperature 0 must request deterministic (greedy) decoding"
        );

        // The exact chain the handler runs: sample_token applies temperature then samples.
        // Logit index 2 is the unique argmax, so greedy must pick it.
        let logits = Tensor::from_vec(vec![4], vec![0.1, 0.2, 0.9, 0.3]).expect("tensor");
        let token = sample_token(&logits, &config, 0.5)
            .expect("temperature-0 config must sample without error (was HTTP 500)");
        assert_eq!(token, 2, "greedy must select the argmax token");
    }

    #[test]
    fn temperature_zero_ignores_top_p_and_stays_greedy() {
        // Even when top_p is supplied, temperature 0 means deterministic output (matches the
        // non-streaming backends where temperature 0 forces top_k = 1 regardless of other
        // sampling controls).
        let config = resolve_stream_generation_config(0.0, Some(0.9), 16);
        assert_eq!(config.strategy, SamplingStrategy::Greedy);
    }

    #[test]
    fn positive_temperature_unchanged() {
        // Regression guard: positive temperatures keep prior behavior — greedy by default,
        // top-p when requested — and remain runnable.
        let greedy = resolve_stream_generation_config(0.7, None, 16);
        assert_eq!(greedy.strategy, SamplingStrategy::Greedy);
        assert!((greedy.temperature - 0.7).abs() < 1e-6);

        let nucleus = resolve_stream_generation_config(0.7, Some(0.8), 16);
        assert!(matches!(
            nucleus.strategy,
            SamplingStrategy::TopP { p } if (p - 0.8).abs() < 1e-6
        ));

        let logits = Tensor::from_vec(vec![3], vec![0.1, 2.0, 0.3]).expect("tensor");
        assert!(
            sample_token(&logits, &greedy, 0.5).is_ok(),
            "positive-temperature config must remain runnable"
        );
    }
}
