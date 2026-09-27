// #3718 done_when 3: a prompt that does not fit the context window says so,
// never a silent cut. The wgpu handler capped `max_tokens` at 4096 and nothing
// else, so a prompt at or past the context window prefilled anyway and the
// decode loop grew the KV cache past what the model was trained on, reporting
// `finish_reason: "stop"`. These two pure functions are the rule it now follows,
// the same rule the CPU (`effective_max_tokens`) and Qwen3.5 (`Session`) paths
// already enforce. The SafeTensors handlers (chat.rs, simple.rs) apply it too,
// through `st_context_budget`. They are not gated on `wgpu`, so default-feature
// CI tests them.

/// Tokens the reply may use once the prompt is in a `context_length` window:
/// `min(requested, context_length - prompt_len)`.
///
/// # Errors
/// `(prompt_len, context_length)` when the prompt leaves no room for even one
/// generated token. That is decided by the request alone, so the caller refuses
/// it as a client error rather than truncating.
#[cfg_attr(not(any(feature = "wgpu", feature = "inference")), allow(dead_code))]
pub(super) fn context_token_budget(
    prompt_len: usize,
    requested: usize,
    context_length: usize,
) -> std::result::Result<usize, (usize, usize)> {
    if prompt_len >= context_length {
        return Err((prompt_len, context_length));
    }
    Ok(requested.min(context_length - prompt_len))
}

/// The OpenAI-shaped 400 body for a prompt refused for length
/// (`error.code = "context_length_exceeded"`), so a client can tell it from a
/// server fault without parsing the message.
#[cfg_attr(not(any(feature = "wgpu", feature = "inference")), allow(dead_code))]
pub(super) fn context_length_exceeded_body(prompt_len: usize, context_length: usize) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": format!(
                "prompt is {prompt_len} tokens; the model's context window is {context_length}, \
                 which leaves no room to generate. The prompt was refused whole, not truncated."
            ),
            "type": "invalid_request_error",
            "param": "messages",
            "code": "context_length_exceeded",
            "prompt_tokens": prompt_len,
            "context_length": context_length,
        }
    })
}

/// OpenAI `finish_reason` for a reply of `generated` tokens under a `max_tokens`
/// budget: `"length"` when the budget ran out, else `"stop"`. Every serve path
/// (wgpu, CUDA, the CUDA->CPU fallback) reports through this one rule, so a cut
/// reply is never labelled a natural stop.
pub(super) fn finish_reason_for(generated: usize, max_tokens: usize) -> &'static str {
    if generated >= max_tokens {
        "length"
    } else {
        "stop"
    }
}

#[cfg(test)]
mod tests_context_budget_3718 {
    use super::{context_length_exceeded_body, context_token_budget, finish_reason_for};

    #[test]
    fn a_reply_that_used_the_whole_budget_is_length_not_stop() {
        assert_eq!(finish_reason_for(64, 64), "length");
        assert_eq!(finish_reason_for(65, 64), "length");
        assert_eq!(finish_reason_for(63, 64), "stop");
        assert_eq!(finish_reason_for(0, 64), "stop");
    }

    #[test]
    fn budget_is_the_request_when_it_fits() {
        assert_eq!(context_token_budget(10, 64, 2048), Ok(64));
    }

    #[test]
    fn budget_is_clamped_to_the_room_left() {
        // 2040 prompt tokens in a 2048 window leave 8, whatever was asked.
        assert_eq!(context_token_budget(2040, 64, 2048), Ok(8));
        assert_eq!(context_token_budget(2047, 4096, 2048), Ok(1));
    }

    #[test]
    fn a_prompt_that_fills_the_window_is_refused_not_cut() {
        assert_eq!(context_token_budget(2048, 64, 2048), Err((2048, 2048)));
        assert_eq!(context_token_budget(9000, 1, 2048), Err((9000, 2048)));
    }

    #[test]
    fn refusal_body_names_the_code_and_both_counts() {
        let body = context_length_exceeded_body(9000, 2048);
        let e = &body["error"];
        assert_eq!(e["code"], "context_length_exceeded");
        assert_eq!(e["type"], "invalid_request_error");
        assert_eq!(e["prompt_tokens"], 9000);
        assert_eq!(e["context_length"], 2048);
        let msg = e["message"].as_str().expect("message is a string");
        assert!(msg.contains("9000") && msg.contains("2048") && msg.contains("not truncated"));
    }
}
