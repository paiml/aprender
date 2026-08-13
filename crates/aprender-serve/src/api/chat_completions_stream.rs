
/// The deltas of a streamed response, plus whether a stop STRING ended it.
///
/// `stopped` exists because the terminal chunk needs it: #2375 finding 6 shipped
/// a hardcoded `finish_reason: "stop"` partly because the delta builder threw
/// away the one fact that distinguishes "stopped" from "ran out of budget".
struct StreamedText {
    /// Text deltas, in order; `deltas.concat()` is the full (stop-truncated) text.
    deltas: Vec<String>,
    /// True when a stop string matched and truncated the text.
    stopped: bool,
}

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
) -> StreamedText {
    let mut deltas = Vec::new();
    let mut emitted = 0usize;
    let mut stopped = false;
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
            stopped = true;
            break;
        }
    }
    StreamedText { deltas, stopped }
}

/// OpenAI-compatible `/v1/chat/completions/stream` endpoint (SSE).
///
/// aprender#2375(4): this route is mounted unconditionally and printed by the
/// server's own banner, and it answered `404 {"error":"Model registry error: No
/// model available"}` on every `apr serve run model.gguf` — the standard
/// deployment. It resolved the dense f32 [`Model`](crate::layers::Model) through
/// `AppState::get_model`, which is `None` whenever the weights are quantized, so
/// the route was dead on arrival for the whole GGUF/APR fleet while
/// `/v1/chat/completions` on the same process answered 200 with real text.
///
/// It also carried a SECOND, separate implementation of chat completion —
/// its own prompt formatting, sampling config, id format and delta builder —
/// which is how the two paths drifted apart in the first place (this one alone
/// handled `temperature: 0`; the main one alone reached the quantized, cached,
/// CUDA and MoE backends).
///
/// So it is now exactly what its name says: `/v1/chat/completions` with
/// `stream` forced on. Every backend, one wire format, one set of falsifiers.
pub async fn openai_chat_completions_stream_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(cancel): Extension<CancelToken>,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    request.stream = true;
    openai_chat_completions_handler(State(state), headers, Extension(cancel), Json(request)).await
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
        let deltas = streaming_text_deltas(&t, &[1, 2, 3, 4], None).deltas;
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
        let deltas = streaming_text_deltas(&t, &[1, 2, 3, 4], Some(&["X".to_string()])).deltas;
        assert_eq!(deltas.concat(), "ab");
        assert!(!deltas.concat().contains('X'));
    }

    #[test]
    fn no_stop_streams_full_text() {
        let t = tok(&["<unk>", "a", "b", "X", "c"]);
        let deltas = streaming_text_deltas(&t, &[1, 2, 3, 4], None).deltas;
        assert_eq!(deltas.concat(), "abXc");
    }
}

// PMAT-790 (+ #2375): a dense chat/completions request with `temperature: 0` must not 500.
// The handler builds a GenerationConfig and runs it through `model.generate` ->
// `sample_token` -> `apply_temperature`, which rejects a non-positive temperature.
// `temperature: 0` is the canonical OpenAI deterministic request; it must resolve to a
// runnable, greedy config on EVERY dense backend, which is why the resolver these tests
// drive is now the shared one in `realize_handlers` rather than a stream-only copy.
#[cfg(test)]
mod pmat790_stream_temperature_zero_tests {
    use crate::api::realize_handlers::resolve_dense_generation_config as resolve_stream_generation_config;
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
