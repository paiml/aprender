//! #4971 (V3-a): OpenAI `logprobs` and `top_logprobs` on `/v1/chat/completions`.
//!
//! ASOC-INV-021 (`contracts/apr-serve-openai-compat-v1.yaml`): with `logprobs:
//! true` the reply carries `choices[0].logprobs.content`, one entry per
//! generated token. A backend that does not compute them answers 501 and names
//! itself, and a request without `logprobs` gets a reply with no `logprobs` key.
//!
//! The entries are put into the serialized reply rather than carried as a field
//! of [`super::ChatChoice`], so a reply without them is byte-identical to the
//! reply before #4971.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::{ChatCompletionRequest, ChatCompletionResponse, ErrorResponse};
use crate::gguf::logprobs::StepLogprobs;
use crate::tokenizer::BPETokenizer;

/// The most `top_logprobs` OpenAI allows, and so the most this server returns.
pub const MAX_TOP_LOGPROBS: u8 = 20;

/// `top_logprobs`: how many of each step's best tokens to return, 0 to
/// [`MAX_TOP_LOGPROBS`].
///
/// Anything else is refused at deserialization with a reason that names the
/// field, so no handler ever sees it (the [`super::ChoiceCount`] pattern).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct TopLogprobs(u8);

impl TopLogprobs {
    /// The number of alternatives each entry carries.
    #[must_use]
    pub fn get(self) -> usize {
        usize::from(self.0)
    }
}

impl<'de> Deserialize<'de> for TopLogprobs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let in_range = value
            .as_u64()
            .and_then(|n| u8::try_from(n).ok())
            .filter(|&n| n <= MAX_TOP_LOGPROBS);
        in_range.map(Self).ok_or_else(|| {
            let got = if value.is_number() {
                format!(", not {value}")
            } else {
                String::new()
            };
            serde::de::Error::custom(format!(
                "{}top_logprobs must be an integer from 0 to {MAX_TOP_LOGPROBS}{got}",
                crate::api::CLIENT_VISIBLE_MARKER
            ))
        })
    }
}

impl ChatCompletionRequest {
    /// How many alternatives each entry carries when the request asked for
    /// logprobs (`top_logprobs`, 0 when absent), else `None` (#4971).
    #[must_use]
    pub fn logprobs_top_n(&self) -> Option<usize> {
        (self.logprobs == Some(true)).then(|| self.top_logprobs.map_or(0, TopLogprobs::get))
    }

    /// `Some(reason)` for `top_logprobs` sent without `logprobs: true` (#4971).
    /// OpenAI refuses it; serving it would return none of what was asked for.
    #[must_use]
    pub fn logprobs_conflict(&self) -> Option<String> {
        (self.top_logprobs.is_some() && self.logprobs != Some(true)).then(|| {
            "top_logprobs requires logprobs: true; send both or neither (#4971)".to_string()
        })
    }

    /// Every cross-field conflict the chat handler refuses with 400 before it
    /// picks a backend: the thinking toggle (#3723), then logprobs (#4971).
    #[must_use]
    pub fn field_conflict(&self) -> Option<String> {
        self.thinking_conflict()
            .or_else(|| self.logprobs_conflict())
    }
}

/// `choices[0].logprobs` of a chat reply, in OpenAI's shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatLogprobs {
    /// One entry per generated token, in order.
    pub content: Vec<ChatTokenLogprob>,
}

/// One generated token: what was chosen, and the best tokens of that step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTokenLogprob {
    /// The chosen token's text.
    pub token: String,
    /// ln P(token) under the logits the choice read.
    pub logprob: f32,
    /// The token's own bytes, which can be part of one character.
    pub bytes: Vec<u8>,
    /// The `top_logprobs` best tokens of this step, best first.
    pub top_logprobs: Vec<ChatTopLogprob>,
}

/// One of a step's best tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTopLogprob {
    /// The token's text.
    pub token: String,
    /// ln P(token) at this step.
    pub logprob: f32,
    /// The token's own bytes.
    pub bytes: Vec<u8>,
}

impl ChatLogprobs {
    /// The wire entries of the engine's step records, one per step.
    #[must_use]
    pub fn from_steps(tokenizer: &BPETokenizer, steps: &[StepLogprobs]) -> Self {
        let content = steps
            .iter()
            .map(|step| ChatTokenLogprob::from_step(tokenizer, step))
            .collect();
        Self { content }
    }
}

impl ChatTokenLogprob {
    /// The wire entry of one step record of the engine.
    #[must_use]
    pub fn from_step(tokenizer: &BPETokenizer, step: &StepLogprobs) -> Self {
        let (token, bytes) = token_text(tokenizer, step.chosen);
        let top_logprobs = step
            .top
            .iter()
            .map(|top| {
                let (token, bytes) = token_text(tokenizer, top.token_id);
                ChatTopLogprob {
                    token,
                    logprob: top.logprob,
                    bytes,
                }
            })
            .collect();
        Self {
            token,
            logprob: step.chosen_logprob,
            bytes,
            top_logprobs,
        }
    }
}

/// #4971: the logprobs of a live stream. The engine sends each token's record
/// ahead of the token, so the record is here by the time the stream receives
/// the token, and taking it never waits. The entries are held until a chunk
/// carries text, and that chunk carries all of them; what is still held when
/// the stream ends goes on its terminal chunk. So every entry is sent once, in
/// the order of the tokens. A finished turn's entries are [`Self::collect`]ed.
pub(crate) struct StreamLogprobs {
    records: tokio::sync::mpsc::UnboundedReceiver<StepLogprobs>,
    held: Vec<ChatTokenLogprob>,
}

impl StreamLogprobs {
    /// The engine's end and the stream's end of one stream's records.
    pub(crate) fn channel() -> (tokio::sync::mpsc::UnboundedSender<StepLogprobs>, Self) {
        let (tx, records) = tokio::sync::mpsc::unbounded_channel();
        let held = Vec::new();
        (tx, Self { records, held })
    }

    /// Holds the entry of `token`, which the stream has just received.
    ///
    /// It does not wait for the record: one that is not here when its token is
    /// was never sent, and waiting would hang the stream if an engine kept the
    /// sender and recorded nothing.
    ///
    /// # Errors
    /// The engine sent no record for `token`, or the record of another token.
    /// The stream then ends with an error rather than go on with entries that
    /// do not match its tokens.
    pub(crate) fn take(&mut self, tokenizer: &BPETokenizer, token: u32) -> Result<(), String> {
        match self.records.try_recv() {
            Ok(step) if step.chosen == token => {
                self.held
                    .push(ChatTokenLogprob::from_step(tokenizer, &step));
                Ok(())
            },
            Ok(step) => Err(format!(
                "token {token} arrived with the logprobs of token {} (#4971)",
                step.chosen
            )),
            Err(_) => Err(format!(
                "token {token} arrived without its logprobs (#4971)"
            )),
        }
    }

    /// The entries of a finished turn's reply `tokens`, whose records were all
    /// sent ahead of them.
    ///
    /// # Errors
    /// As [`Self::take`], for the first token without its own record.
    pub(crate) fn collect(
        mut self,
        tokenizer: &BPETokenizer,
        tokens: &[u32],
    ) -> Result<ChatLogprobs, String> {
        for &token in tokens {
            self.take(tokenizer, token)?;
        }
        Ok(ChatLogprobs { content: self.held })
    }

    /// The entries held until now, for the chunk about to be sent, or `None`
    /// when none are held.
    pub(crate) fn release(&mut self) -> Option<ChatLogprobs> {
        if self.held.is_empty() {
            return None;
        }
        let content = std::mem::take(&mut self.held);
        Some(ChatLogprobs { content })
    }
}

/// A token's text and bytes as a logprobs entry reports them: the bytes it
/// decodes to and their (lossy) text. A token that decodes to no bytes, such
/// as a special token, is reported by its own spelling. An id the vocabulary
/// lacks (a padded row of the LM head) is reported as an empty token.
fn token_text(tokenizer: &BPETokenizer, id: u32) -> (String, Vec<u8>) {
    let bytes = tokenizer.decode_bytes(&[id]).unwrap_or_default();
    if bytes.is_empty() {
        let spelling = tokenizer.get_token(id).unwrap_or_default();
        return (spelling.to_string(), spelling.as_bytes().to_vec());
    }
    (String::from_utf8_lossy(&bytes).into_owned(), bytes)
}

/// The chat reply as JSON, with `choices[0].logprobs` when the request asked
/// for them (#4971). Without them the reply serializes exactly as before, so
/// it has no `logprobs` key.
pub(crate) fn chat_reply(
    response: ChatCompletionResponse,
    logprobs: Option<ChatLogprobs>,
) -> Response {
    let Some(logprobs) = logprobs else {
        return Json(response).into_response();
    };
    match with_logprobs(&response, &logprobs) {
        Some(body) => Json(body).into_response(),
        // Never a 200 without what the request asked for.
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "the reply could not carry its logprobs (#4971)".to_string(),
            }),
        )
            .into_response(),
    }
}

/// `reply` (a chat reply or a stream chunk) as JSON, with `logprobs` as its
/// `choices[0].logprobs`.
pub(crate) fn with_logprobs(
    reply: &impl Serialize,
    logprobs: &ChatLogprobs,
) -> Option<serde_json::Value> {
    let mut body = serde_json::to_value(reply).ok()?;
    let choice = body.pointer_mut("/choices/0")?.as_object_mut()?;
    choice.insert("logprobs".to_string(), serde_json::to_value(logprobs).ok()?);
    Some(body)
}

/// Why `backend` refuses a request that asked for logprobs, or `None` when it
/// did not ask (#4971). A backend that does not compute them answers 501 with
/// this, never a 200 without them.
#[must_use]
pub(crate) fn logprobs_refusal(request: &ChatCompletionRequest, backend: &str) -> Option<String> {
    request.logprobs_top_n().map(|_| {
        format!(
            "`logprobs` is not supported by the {backend} chat backend; the reply would \
             carry none. Refused rather than served (#4971, ASOC-INV-021)."
        )
    })
}

/// [`logprobs_refusal`] for a backend that computes them on its dense turn,
/// streamed or not (the quantized CPU and dense CUDA backends): only its
/// traced loop refuses, as `{backend} (traced)`.
#[must_use]
pub(crate) fn traced_logprobs_refusal(
    request: &ChatCompletionRequest,
    backend: &str,
    traced: bool,
) -> Option<String> {
    if !traced {
        return None;
    }
    logprobs_refusal(request, &format!("{backend} (traced)"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ChatChoice, ChatMessage, Usage};
    use crate::gguf::logprobs::TopLogprob;

    fn request(json: &str) -> Result<ChatCompletionRequest, String> {
        serde_json::from_str(&format!(r#"{{"model":"m","messages":[]{json}}}"#))
            .map_err(|e| e.to_string())
    }

    #[test]
    fn top_logprobs_is_refused_outside_0_to_20_naming_the_field() {
        // (field json, Some(n) accepted / None refused)
        let table: &[(&str, Option<usize>)] = &[
            (r#","logprobs":true,"top_logprobs":0"#, Some(0)),
            (r#","logprobs":true,"top_logprobs":20"#, Some(20)),
            (r#","logprobs":true,"top_logprobs":21"#, None),
            (r#","logprobs":true,"top_logprobs":256"#, None),
            (r#","logprobs":true,"top_logprobs":-1"#, None),
            (r#","logprobs":true,"top_logprobs":2.5"#, None),
            (r#","logprobs":true,"top_logprobs":"3""#, None),
        ];
        for &(json, want) in table {
            match (request(json), want) {
                (Ok(r), Some(n)) => assert_eq!(r.logprobs_top_n(), Some(n), "{json}"),
                (Err(e), None) => assert!(
                    e.contains("[request] top_logprobs must be an integer from 0 to 20"),
                    "{json}: the refusal must name the field: {e}"
                ),
                (got, _) => panic!("{json}: want {want:?}, got {got:?}"),
            }
        }
    }

    #[test]
    fn logprobs_top_n_and_the_conflict_follow_the_two_fields() {
        // (field json, logprobs_top_n, conflict?)
        let table: &[(&str, Option<usize>, bool)] = &[
            ("", None, false),
            (r#","logprobs":false"#, None, false),
            (r#","logprobs":true"#, Some(0), false),
            (r#","logprobs":true,"top_logprobs":3"#, Some(3), false),
            (r#","top_logprobs":2"#, None, true),
            (r#","logprobs":false,"top_logprobs":0"#, None, true),
            (r#","logprobs":true,"top_logprobs":null"#, Some(0), false),
        ];
        for &(json, top_n, conflict) in table {
            let r = request(json).expect("valid request");
            assert_eq!(r.logprobs_top_n(), top_n, "{json}");
            assert_eq!(r.logprobs_conflict().is_some(), conflict, "{json}");
            assert_eq!(r.field_conflict().is_some(), conflict, "{json}");
            if let Some(reason) = r.field_conflict() {
                assert!(reason.contains("top_logprobs"), "{reason}");
            }
        }
    }

    #[test]
    fn a_request_without_logprobs_serializes_without_the_fields() {
        let r = request("").expect("valid request");
        let json = serde_json::to_string(&r).expect("serialize");
        assert!(!json.contains("logprobs"), "{json}");
    }

    fn tokenizer() -> BPETokenizer {
        let vocab = ["<unk>", "Ġhi", "<0xE6>", "<|im_end|>"];
        BPETokenizer::new(
            vocab.iter().map(|s| (*s).to_string()).collect(),
            vec![],
            "<unk>",
        )
        .expect("test tokenizer")
    }

    fn step(chosen: u32, logits: &[f32], n: usize) -> StepLogprobs {
        StepLogprobs::of(0, chosen, logits, n)
    }

    #[test]
    fn entries_carry_each_tokens_text_bytes_and_best_alternatives() {
        let tok = tokenizer();
        let steps = [
            step(1, &[0.0, 3.0, 1.0, 2.0], 2),
            step(2, &[0.0, 0.0, 5.0, 0.0], 1),
        ];
        let lp = ChatLogprobs::from_steps(&tok, &steps);
        assert_eq!(lp.content.len(), 2);

        let first = &lp.content[0];
        assert_eq!(first.token, " hi");
        assert_eq!(first.bytes, b" hi".to_vec());
        assert!(first.logprob <= 0.0);
        let tops: Vec<&str> = first
            .top_logprobs
            .iter()
            .map(|t| t.token.as_str())
            .collect();
        assert_eq!(
            tops,
            [" hi", "<|im_end|>"],
            "best first; a special token by its spelling"
        );
        assert_eq!(first.top_logprobs[1].bytes, b"<|im_end|>".to_vec());
        assert_eq!(first.top_logprobs[0].logprob, first.logprob);

        // A byte token is one byte of a character: its bytes are exact, its text lossy.
        let second = &lp.content[1];
        assert_eq!(second.bytes, vec![0xE6]);
        assert_eq!(second.top_logprobs.len(), 1);
    }

    #[test]
    fn an_id_the_vocabulary_lacks_is_an_empty_token() {
        assert_eq!(token_text(&tokenizer(), 99), (String::new(), Vec::new()));
    }

    fn response() -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: " hi".to_string(),
                    ..Default::default()
                },
                finish_reason: "length".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            brick_trace: None,
            step_trace: None,
            layer_trace: None,
            timings: None,
            used_gpu: None,
        }
    }

    async fn body(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    #[tokio::test]
    async fn the_reply_carries_logprobs_only_when_asked() {
        let plain = body(chat_reply(response(), None)).await;
        assert_eq!(plain, serde_json::to_value(response()).expect("serialize"));
        assert!(plain["choices"][0].get("logprobs").is_none());

        let lp = ChatLogprobs::from_steps(&tokenizer(), &[step(1, &[0.0, 3.0], 1)]);
        let with = body(chat_reply(response(), Some(lp.clone()))).await;
        let content: ChatLogprobs =
            serde_json::from_value(with["choices"][0]["logprobs"].clone()).expect("logprobs");
        assert_eq!(content, lp);
        let mut without = with.clone();
        without["choices"][0]
            .as_object_mut()
            .expect("choice")
            .remove("logprobs");
        assert_eq!(without, plain, "logprobs is the only key the request adds");
    }

    #[test]
    fn backends_without_logprobs_refuse_only_requests_that_ask() {
        let asks = request(r#","logprobs":true,"top_logprobs":2"#).expect("valid");
        let plain = request(r#","logprobs":false"#).expect("valid");
        let reason = logprobs_refusal(&asks, "CUDA").expect("refused");
        assert!(
            reason.contains("CUDA") && reason.contains("logprobs"),
            "{reason}"
        );
        assert_eq!(logprobs_refusal(&plain, "CUDA"), None);
    }

    #[test]
    fn a_dense_backend_refuses_only_its_traced_loop() {
        let mut asks = request(r#","logprobs":true"#).expect("valid");
        // (backend, stream, traced, refused as)
        let table: &[(&str, bool, bool, Option<&str>)] = &[
            ("quantized", false, false, None),
            ("quantized", true, false, None),
            ("quantized", false, true, Some("quantized (traced)")),
            ("quantized", true, true, Some("quantized (traced)")),
            ("CUDA", false, false, None),
            ("CUDA", true, false, None),
            ("CUDA", false, true, Some("CUDA (traced)")),
            ("CUDA", true, true, Some("CUDA (traced)")),
        ];
        for &(backend, stream, traced, want) in table {
            asks.stream = stream;
            let got = traced_logprobs_refusal(&asks, backend, traced);
            match want {
                None => assert_eq!(got, None, "{backend} stream={stream} traced={traced}"),
                Some(path) => assert!(
                    got.as_deref().is_some_and(|r| r.contains(path)),
                    "{backend} stream={stream} traced={traced}: {got:?}"
                ),
            }
        }
        let mut plain = request("").expect("valid");
        plain.stream = true;
        assert_eq!(traced_logprobs_refusal(&plain, "CUDA", true), None);
    }

    #[test]
    fn a_finished_turn_collects_one_entry_per_token_from_records_sent_ahead() {
        let (records, logprobs) = StreamLogprobs::channel();
        for (chosen, logits) in [(1, [0.0, 2.0, 1.0]), (2, [0.0, 1.0, 3.0])] {
            records.send(step(chosen, &logits, 2)).expect("open");
        }
        let lp = logprobs.collect(&tokenizer(), &[1, 2]).expect("collected");
        assert_eq!(lp.content.len(), 2);
        assert!(lp.content.iter().all(|e| e.top_logprobs.len() == 2));
        // The sender is still held: collect read what was sent and did not wait.
        drop(records);
    }

    #[test]
    fn a_token_without_its_own_record_is_an_error_and_never_a_wait() {
        // The engine keeps its sender and sends nothing, as a batched turn
        // would: the first token is refused at once.
        let (records, logprobs) = StreamLogprobs::channel();
        let err = logprobs.collect(&tokenizer(), &[1]).expect_err("no record");
        assert!(err.contains("without its logprobs"), "{err}");
        drop(records);
        let (other, mut held) = StreamLogprobs::channel();
        other.send(step(2, &[0.0, 1.0, 3.0], 1)).expect("open");
        let err = held
            .take(&tokenizer(), 1)
            .expect_err("another token's record");
        assert!(err.contains("logprobs of token 2"), "{err}");
        assert_eq!(held.release(), None, "a refused entry is never held");
    }

    #[test]
    fn a_top_entry_is_the_engines_record() {
        let s = StepLogprobs {
            step: 0,
            chosen: 1,
            chosen_logprob: -0.25,
            top: vec![TopLogprob {
                token_id: 1,
                logit: 4.0,
                logprob: -0.25,
            }],
        };
        let lp = ChatLogprobs::from_steps(&tokenizer(), &[s]);
        assert_eq!(lp.content[0].logprob, -0.25);
        assert_eq!(lp.content[0].top_logprobs[0].logprob, -0.25);
    }
}
