//! OpenAI `logprobs` / `top_logprobs` on `/v1/chat/completions` (#4026).
//!
//! Only the session engine records per-step logprobs, so only the Qwen3.5 arm
//! answers them. Every other arm, and the streaming path, refuses the request
//! by name: a reply without the logprobs it was asked for would read as a
//! model that has none.
//!
//! Each `content` entry is OpenAI's shape (`token`, `logprob`, `bytes`,
//! `top_logprobs`) plus three fields PRM C11 (`sparse-logits-v1`) needs and
//! OpenAI does not carry: `token_id`, the top ids, and `logsumexp_full`, the
//! full-vocab log-sum-exp at T = 1.0 from which the retained mass and the
//! residual tail follow. All values are from the forward's logits, before any
//! repetition penalty or sampler.

use super::ChatCompletionRequest;
use crate::gguf::StepLogprobs;

/// OpenAI's ceiling on `top_logprobs`.
pub(crate) const MAX_TOP_LOGPROBS: usize = 20;

/// `Ok(None)` when the request asks for no logprobs, `Ok(Some(n))` for `n`
/// alternatives per token, `Err` for a request that cannot be answered as
/// asked.
pub(crate) fn requested_top_logprobs(req: &ChatCompletionRequest) -> Result<Option<usize>, String> {
    let wanted = req.logprobs.unwrap_or(false);
    match (wanted, req.top_logprobs) {
        (false, None) => Ok(None),
        (false, Some(_)) => Err("`top_logprobs` requires `logprobs: true` (#4026)".to_string()),
        (true, Some(n)) if n > MAX_TOP_LOGPROBS => Err(format!(
            "`top_logprobs` is {n}; at most {MAX_TOP_LOGPROBS} are supported (#4026)"
        )),
        (true, _) if req.stream => Err(
            "`logprobs` on a streamed chat completion is not recorded yet; send `stream: false` \
             (#4026)"
                .to_string(),
        ),
        (true, n) => Ok(Some(n.unwrap_or(0))),
    }
}

/// The refusal every arm but the session engine returns for `logprobs: true`.
pub(crate) fn unsupported_backend_reason(architecture: Option<&str>) -> String {
    format!(
        "`logprobs` are recorded only by the Qwen3.5 session engine; this model ({}) is served \
         by a backend that cannot record them, so the request is refused rather than answered \
         without them (#4026)",
        architecture.unwrap_or("unknown architecture")
    )
}

/// The `choices[].logprobs` object for the generated `steps`.
///
/// `steps` must already be trimmed to the ids the reply reports; `n_top` of
/// each step's recorded alternatives are emitted.
pub(crate) fn chat_logprobs_json(
    steps: &[StepLogprobs],
    n_top: usize,
    decode: &dyn Fn(u32) -> String,
) -> serde_json::Value {
    let content: Vec<serde_json::Value> = steps
        .iter()
        .map(|s| {
            let top: Vec<serde_json::Value> = s
                .top
                .iter()
                .take(n_top)
                .map(|t| {
                    serde_json::json!({
                        "token": decode(t.token_id),
                        "token_id": t.token_id,
                        "logprob": t.logprob,
                        "bytes": serde_json::Value::Null,
                    })
                })
                .collect();
            serde_json::json!({
                "token": decode(s.chosen),
                "token_id": s.chosen,
                "logprob": s.chosen_logprob,
                "bytes": serde_json::Value::Null,
                "top_logprobs": top,
                "logsumexp_full": s.logsumexp_full,
            })
        })
        .collect();
    serde_json::json!({ "content": content })
}

#[cfg(test)]
#[path = "chat_logprobs_tests.rs"]
mod tests;
