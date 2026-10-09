//! Tool calls on the streaming chat path (#4918).
//!
//! `build_tool_calling_message` turned a model's `<tool_call>` / `<function=…>`
//! markup into OpenAI `tool_calls`, but only on the non-streaming path. Both SSE
//! builders wrote the same markup to the client as `delta.content` and ended the
//! stream `stop`, so an agent client (which always streams) showed the raw call
//! and never ran it.
//!
//! [`ToolCallDetector`] sits after the stop filter in both builders. Text streams
//! live until a marker appears. A tail that could still grow into a marker is held
//! back, never more than `len(longest marker) - 1` bytes. From the marker on, the
//! rest of the output is captured, and at end of stream it is parsed by the SAME
//! `build_tool_calling_message` the non-streaming path uses, so the two paths
//! cannot disagree about what counts as a call. Capture that parses to no call is
//! flushed as content, so nothing is lost.
//!
//! Contract: `contracts/serve-stream-tool-calls-v1.yaml`.

use super::{ChatCompletionRequest, OpenAiTool, ResponseToolCall};
use crate::grammar::ToolChoice;

/// The markers that open a tool call. A bare `{` is deliberately NOT one: a JSON
/// call with no marker streams as content (D-c, pinned by T12).
pub(crate) const TOOL_CALL_MARKERS: [&str; 2] = ["<tool_call>", "<function="];

/// The longest marker prefix the detector may hold back: `len("<tool_call>") - 1`.
pub(crate) const MAX_HELD_BYTES: usize = 10;

/// What a streaming request declared about tools, when detection applies.
#[derive(Debug, Clone)]
pub(crate) struct StreamTools {
    tools: Vec<OpenAiTool>,
    tool_choice: Option<ToolChoice>,
}

impl StreamTools {
    /// `Some` only when the request declares `tools` and `tool_choice` is not
    /// `"none"`. Every other stream must stay byte-identical to the pre-fix one,
    /// and `None` here is what keeps the detector off it.
    pub(crate) fn from_request(request: &ChatCompletionRequest) -> Option<Self> {
        let tools = request.tools.as_ref()?;
        let tool_choice = request
            .tool_choice
            .as_ref()
            .map(super::OpenAiToolChoice::to_grammar);
        if matches!(tool_choice, Some(ToolChoice::None)) {
            return None;
        }
        Some(Self {
            tools: tools.clone(),
            tool_choice,
        })
    }

    pub(crate) fn detector(self) -> ToolCallDetector {
        ToolCallDetector {
            tools: self,
            held: String::new(),
            captured: None,
        }
    }
}

/// What the end of a stream still owes the client.
#[derive(Debug, Default)]
pub(crate) struct DetectorEnd {
    /// Text to send as content: the held tail, or a capture that held no call.
    pub(crate) content: Option<String>,
    /// The calls the capture parsed to; empty when there were none.
    pub(crate) calls: Vec<ResponseToolCall>,
}

/// Separates streamed text from a tool call at the end of a stream.
#[derive(Debug)]
pub(crate) struct ToolCallDetector {
    tools: StreamTools,
    /// A tail of the text that is a proper prefix of a marker.
    held: String,
    /// Everything from the first marker on, once one has appeared.
    captured: Option<String>,
}

impl ToolCallDetector {
    /// Accept one delta; the content that is safe to send now, if any.
    pub(crate) fn push(&mut self, text: &str) -> Option<String> {
        if let Some(captured) = self.captured.as_mut() {
            captured.push_str(text);
            return None;
        }
        let mut buf = std::mem::take(&mut self.held);
        buf.push_str(text);
        if let Some(at) = first_marker(&buf) {
            self.captured = Some(buf[at..].to_string());
            buf.truncate(at);
        } else {
            let keep = marker_prefix_len(&buf);
            self.held = buf.split_off(buf.len() - keep);
        }
        (!buf.is_empty()).then_some(buf)
    }

    /// Bytes currently held back as a possible marker prefix (T7).
    #[cfg(test)]
    pub(crate) fn held_len(&self) -> usize {
        self.held.len()
    }

    /// Close the stream: parse the capture with the non-streaming parser.
    pub(crate) fn finish(self) -> DetectorEnd {
        let held = (!self.held.is_empty()).then_some(self.held);
        let Some(captured) = self.captured else {
            return DetectorEnd {
                content: held,
                calls: Vec::new(),
            };
        };
        let (message, _) = super::openai_handlers::build_tool_calling_message(
            captured.clone(),
            String::new(),
            &self.tools.tools,
            self.tools.tool_choice.as_ref(),
        );
        match message.tool_calls {
            Some(calls) if !calls.is_empty() => DetectorEnd {
                content: None,
                calls,
            },
            // An undeclared tool or a mere mention of a marker: nothing is lost.
            _ => DetectorEnd {
                content: Some(captured),
                calls: Vec::new(),
            },
        }
    }
}

/// The byte offset of the earliest marker in `text`.
fn first_marker(text: &str) -> Option<usize> {
    TOOL_CALL_MARKERS.iter().filter_map(|m| text.find(m)).min()
}

/// The length of the longest suffix of `text` that is a proper prefix of a marker.
fn marker_prefix_len(text: &str) -> usize {
    (1..=MAX_HELD_BYTES.min(text.len()))
        .rev()
        .find(|&k| {
            let start = text.len() - k;
            text.is_char_boundary(start)
                && TOOL_CALL_MARKERS
                    .iter()
                    .any(|m| m.len() > k && m.starts_with(&text[start..]))
        })
        .unwrap_or(0)
}

/// Pass one delta through the detector when the request has one; without one the
/// delta goes out untouched, which is what keeps a tool-less stream byte-identical.
pub(crate) fn detect(detector: &mut Option<ToolCallDetector>, text: String) -> Option<String> {
    match detector {
        Some(d) => d.push(&text),
        None => Some(text),
    }
}

/// End of stream for an optional detector.
pub(crate) fn finish(detector: Option<ToolCallDetector>) -> DetectorEnd {
    detector.map(ToolCallDetector::finish).unwrap_or_default()
}

/// The replayed builder's whole generation through the detector at once: the
/// content deltas to send, then the calls.
pub(crate) fn detect_all(
    tools: Option<StreamTools>,
    deltas: Vec<String>,
) -> (Vec<String>, Vec<ResponseToolCall>) {
    let mut detector = tools.map(StreamTools::detector);
    let mut out: Vec<String> = deltas
        .into_iter()
        .filter_map(|d| detect(&mut detector, d))
        .collect();
    let end = finish(detector);
    out.extend(end.content);
    (out, end.calls)
}

/// The terminal reason: `tool_calls` when the stream carried calls, otherwise what
/// the generation itself did.
pub(crate) fn finish_reason(
    calls: &[ResponseToolCall],
    generated: super::FinishReason,
) -> super::FinishReason {
    if calls.is_empty() {
        generated
    } else {
        super::FinishReason::ToolCalls
    }
}
