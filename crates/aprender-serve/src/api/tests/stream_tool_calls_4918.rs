//! Falsifiers for aprender#4918: a streamed `/v1/chat/completions` turn that
//! makes a tool call carries it as `delta.tool_calls` and ends with
//! `finish_reason: "tool_calls"`, as the non-streaming body does. Before the
//! fix both SSE builders wrote the `<tool_call>` markup into `delta.content`
//! and ended `stop`, so an agent client showed the call and never ran it.
//!
//! Contract: `contracts/serve-stream-tool-calls-v1.yaml`. Each test is one
//! falsifier there and names the mutant that turns it RED. T10 is the live
//! CUDA receipt and is not a unit test.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::api::openai_handlers::{
    build_tool_calling_message, pregenerated_sse_response, true_streaming_sse_response,
};
use crate::api::stream_tool_calls::{self, StreamTools, MAX_HELD_BYTES, TOOL_CALL_MARKERS};
use crate::api::{
    ChatCompletionChunk, ChatCompletionRequest, ChatDelta, FinishReason, OpenAiTool,
    ResponseToolCall,
};
use crate::tokenizer::BPETokenizer;

/// The issue's live delta sequence. The issue prints the first five content
/// deltas and the last one verbatim and elides the middle
/// (`… <parameter=command>\nls -la ~/tmp/\n</parameter>\n</function> …`); the
/// middle is split here the way a byte-level BPE vocabulary splits it, so the
/// markers and closing tags are broken across tokens too.
const ISSUE_DELTAS: &[&str] = &[
    "<tool_call>",
    "\n",
    "<function",
    "=b",
    "ash",
    ">\n",
    "<parameter",
    "=command",
    ">\n",
    "ls",
    " -",
    "la",
    " ~/",
    "tmp",
    "/\n",
    "</parameter",
    ">\n",
    "</function",
    ">",
    "\n</tool_call>",
];

/// T5's fixed token sequence: plain text, a marker split across tokens, a whole
/// Hermes-style call to the declared tool, and text after it. With no tools, or
/// with `tool_choice: "none"`, every byte of it is content, chunked as v0.70.2
/// chunked it. The generator (`scripts/gen_stream_golden_4918.sh`) carries the
/// same list; the two must not drift.
const GOLDEN_PIECES: &[&str] = &[
    "Sure",
    ".",
    " <",
    "tool",
    "_call",
    ">\n",
    "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}",
    "\n</tool_call>",
    " done",
];

/// Written from a checkout of `v0.70.2` by `scripts/gen_stream_golden_4918.sh`.
/// Never regenerate it from a later tree: it is the pre-fix wire format.
const GOLDEN: &str = include_str!("golden/stream_no_tools_v0702.sse");

const REQUEST_ID: &str = "chatcmpl-4918";
const MODEL: &str = "qwen3-coder";

/// The issue's request, verbatim, with `stream` on. `edits` sets top-level
/// keys; a `null` removes the key.
fn issue_request(edits: Value) -> ChatCompletionRequest {
    let mut req = json!({
        "model": "qwen3-coder",
        "messages": [{"role": "user", "content": "what are the directories in ~/tmp/"}],
        "tools": [{"type": "function", "function": {"name": "bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"]}}}],
        "max_tokens": 200,
        "temperature": 0,
        "stream": true
    });
    let obj = req.as_object_mut().expect("request is an object");
    for (k, v) in edits.as_object().expect("edits is an object") {
        if v.is_null() {
            obj.remove(k);
        } else {
            obj.insert(k.clone(), v.clone());
        }
    }
    serde_json::from_value(req).expect("the issue's request parses")
}

fn issue_tools() -> Option<StreamTools> {
    StreamTools::from_request(&issue_request(json!({})))
}

fn declared_tools() -> Vec<OpenAiTool> {
    issue_request(json!({})).tools.expect("the issue declares tools")
}

/// One token per piece. A repeated piece reuses its first id, so the
/// vocabulary has no duplicates.
fn tokenizer_for(pieces: &[&str]) -> (Arc<BPETokenizer>, Vec<u32>) {
    let mut vocab: Vec<String> = Vec::new();
    let mut ids = Vec::new();
    for piece in pieces {
        let id = match vocab.iter().position(|v| v == piece) {
            Some(i) => i,
            None => {
                vocab.push((*piece).to_string());
                vocab.len() - 1
            },
        };
        ids.push(u32::try_from(id).expect("small vocab"));
    }
    let unk = vocab[0].clone();
    let tokenizer = BPETokenizer::new(vocab, vec![], unk.as_str()).expect("build tokenizer");
    (Arc::new(tokenizer), ids)
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read SSE body");
    String::from_utf8(bytes.to_vec()).expect("SSE body is utf-8")
}

/// The live builder's SSE body for `pieces`, one token each.
async fn live_body(pieces: &[&str], stops: Option<&[String]>, tools: Option<StreamTools>) -> String {
    let (tokenizer, ids) = tokenizer_for(pieces);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(ids.len().max(1));
    for id in ids {
        tx.send(Ok(id)).await.expect("send token");
    }
    drop(tx);
    let response = true_streaming_sse_response(
        rx,
        tokenizer,
        REQUEST_ID.to_string(),
        MODEL.to_string(),
        Arc::new(crate::metrics::MetricsCollector::new()),
        std::time::Instant::now(),
        256,
        0,
        None,
        stops,
        tools,
    );
    body_text(response).await
}

/// The replayed builder's SSE body for the same tokens.
async fn replayed_body(
    pieces: &[&str],
    stops: Option<&[String]>,
    tools: Option<StreamTools>,
) -> String {
    let (tokenizer, ids) = tokenizer_for(pieces);
    let response = pregenerated_sse_response(
        ids,
        tokenizer,
        REQUEST_ID.to_string(),
        MODEL.to_string(),
        stops,
        256,
        0,
        tools,
    );
    body_text(response).await
}

/// What a streaming client reassembles from one SSE body.
#[derive(Debug)]
struct Streamed {
    /// Every `delta.content`, in order.
    content: Vec<String>,
    /// Every entry of every `delta.tool_calls`, in order.
    calls: Vec<Value>,
    /// How many frames carried a `delta.tool_calls` key.
    call_frames: usize,
    /// Frame index of the last `delta.tool_calls` frame and of the terminal one.
    call_frame_at: Option<usize>,
    finish_at: Option<usize>,
    finish: String,
}

fn parse(body: &str) -> Streamed {
    let frames: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| p.trim() != "[DONE]")
        .map(|p| serde_json::from_str(p).expect("every SSE payload is JSON"))
        .collect();
    let mut out = Streamed {
        content: Vec::new(),
        calls: Vec::new(),
        call_frames: 0,
        call_frame_at: None,
        finish_at: None,
        finish: String::new(),
    };
    for (i, frame) in frames.iter().enumerate() {
        let delta = &frame["choices"][0]["delta"];
        if let Some(text) = delta["content"].as_str() {
            out.content.push(text.to_string());
        }
        if let Some(calls) = delta.get("tool_calls") {
            out.call_frames += 1;
            out.call_frame_at = Some(i);
            out.calls.extend(calls.as_array().expect("tool_calls is an array").iter().cloned());
        }
        if let Some(reason) = frame["choices"][0]["finish_reason"].as_str() {
            out.finish = reason.to_string();
            out.finish_at = Some(i);
        }
    }
    out
}

/// The calls and finish reason the non-streaming path returns for `text`.
fn non_streaming(text: &str) -> (Vec<ResponseToolCall>, String) {
    let (message, reason) =
        build_tool_calling_message(text.to_string(), "stop".to_string(), &declared_tools(), None);
    (message.tool_calls.unwrap_or_default(), reason)
}

/// A streamed call, compared with a non-streaming one: same id, type, name and
/// arguments; `index` is the streamed call's position.
fn without_index(call: &Value) -> Value {
    let mut call = call.clone();
    call.as_object_mut().expect("a call is an object").remove("index");
    call
}

/// T1/T2: the issue's generation is one call to `bash`, on one frame before the
/// terminal one, in parity with the non-streaming body; no content carries any
/// byte of it (the call starts the generation, so there is no preamble).
fn assert_issue_call(s: &Streamed) {
    assert!(s.content.is_empty(), "a content delta carries bytes of the call: {:?}", s.content);
    assert_eq!(s.call_frames, 1, "the calls arrive on exactly one frame: {s:?}");
    assert_eq!(s.calls.len(), 1, "exactly one call: {:?}", s.calls);
    let call = &s.calls[0];
    assert_eq!(call["index"], 0);
    assert!(call["id"].as_str().is_some_and(|id| !id.is_empty()), "call has an id: {call}");
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "bash");
    let args: Value = serde_json::from_str(
        call["function"]["arguments"].as_str().expect("arguments is a JSON string"),
    )
    .expect("arguments parse as JSON");
    assert_eq!(args, json!({"command": "ls -la ~/tmp/"}));
    assert_eq!(s.finish, "tool_calls");
    assert!(s.call_frame_at < s.finish_at, "the call frame precedes the terminal frame: {s:?}");

    let (calls, reason) = non_streaming(&ISSUE_DELTAS.concat());
    let expected: Vec<Value> = calls.iter().map(|c| serde_json::to_value(c).expect("ser")).collect();
    let got: Vec<Value> = s.calls.iter().map(without_index).collect();
    assert_eq!(got, expected, "streamed calls differ from the non-streaming body's");
    assert_eq!(s.finish, reason, "finish reason differs from the non-streaming body's");
}

// T1 — mutant: the detector passes text through.
#[tokio::test]
async fn t1_live_builder_streams_the_issue_call_as_tool_calls() {
    let s = parse(&live_body(ISSUE_DELTAS, None, issue_tools()).await);
    assert_issue_call(&s);
}

// T2 — mutant: the detector is wired to one builder only.
#[tokio::test]
async fn t2_replayed_builder_streams_the_issue_call_as_tool_calls() {
    let s = parse(&replayed_body(ISSUE_DELTAS, None, issue_tools()).await);
    assert_issue_call(&s);
}

/// T3's generations: each call starts at a marker. Some parse to calls, some
/// to none (an undeclared tool, a mention of the tag).
const T3_TEXTS: &[&str] = &[
    "<tool_call>\n<function=bash>\n<parameter=command>\nls -la ~/tmp/\n</parameter>\n</function>\n</tool_call>",
    "Let me look.\n<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>",
    "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n<function=bash>\n<parameter=command>\ndf -h\n</parameter>\n</function>",
    "I'll run it: <tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls ~/tmp\"}}\n</tool_call>",
    "<tool_call>\n<function=rm>\n<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>",
    "Use the <tool_call> tag to call a tool.",
    "No call here, just text with a < and a <tool and a <func.",
];

/// Split `text` at the given byte offsets (ASCII texts, so every offset is a
/// char boundary).
fn split_at(text: &str, cuts: &[usize]) -> Vec<String> {
    let mut cuts: Vec<usize> = cuts.iter().map(|c| c % (text.len() + 1)).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let mut out = Vec::new();
    let mut from = 0;
    for cut in cuts.into_iter().chain(std::iter::once(text.len())) {
        if cut > from {
            out.push(text[from..cut].to_string());
            from = cut;
        }
    }
    out
}

proptest::proptest! {
    // T3 — mutant: parsing runs per delta.
    #[test]
    fn t3_any_split_emits_the_calls_of_the_captured_text(
        which in 0..T3_TEXTS.len(),
        cuts in proptest::collection::vec(0usize..200, 0..24),
    ) {
        let text = T3_TEXTS[which];
        let deltas = split_at(text, &cuts);
        proptest::prop_assert_eq!(deltas.concat(), text);
        let (content, calls) = stream_tool_calls::detect_all(issue_tools(), deltas);
        let finish = stream_tool_calls::finish_reason(&calls, FinishReason::Stop);

        let at = TOOL_CALL_MARKERS.iter().filter_map(|m| text.find(m)).min();
        let (expected_calls, expected_reason) = match at {
            Some(at) => non_streaming(&text[at..]),
            None => (Vec::new(), "stop".to_string()),
        };
        proptest::prop_assert_eq!(&calls, &expected_calls);
        proptest::prop_assert_eq!(finish.as_str(), expected_reason.as_str());
        let expected_content = match at {
            Some(at) if !expected_calls.is_empty() => &text[..at],
            _ => text,
        };
        proptest::prop_assert_eq!(content.concat(), expected_content);
    }
}

// T4 — mutant: all content is held to the end.
#[tokio::test]
async fn t4_tools_declared_plain_answer_streams_delta_by_delta_and_ends_stop() {
    let pieces = ["The", " directories", " are", " a", " and", " b", "."];
    for (label, body) in [
        ("live", live_body(&pieces, None, issue_tools()).await),
        ("replayed", replayed_body(&pieces, None, issue_tools()).await),
    ] {
        let s = parse(&body);
        assert_eq!(s.content, pieces, "{label}: content is not streamed delta by delta");
        assert_eq!(s.call_frames, 0, "{label}: {s:?}");
        assert_eq!(s.finish, "stop", "{label}");
    }
}

/// `"created":<digits>` → `"created":0`. The generator masks the same way.
fn mask_created(body: &str) -> String {
    const KEY: &str = "\"created\":";
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(at) = rest.find(KEY) {
        out.push_str(&rest[..at + KEY.len()]);
        out.push('0');
        rest = rest[at + KEY.len()..].trim_start_matches(|c: char| c.is_ascii_digit());
    }
    out.push_str(rest);
    out
}

// T5 — mutant: the detector runs unconditionally.
#[tokio::test]
async fn t5_no_tools_or_tool_choice_none_matches_the_v0702_golden() {
    let golden: String = GOLDEN
        .split_inclusive('\n')
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert!(
        GOLDEN.contains("v0.70.2 89e261cda1042ad14dba50034da9a9090abcdecb"),
        "the golden header names the tag and sha it was generated from"
    );
    let pieces = serde_json::to_string(GOLDEN_PIECES).expect("pieces");
    assert!(
        GOLDEN.lines().any(|l| l == format!("# pieces: {pieces}")),
        "GOLDEN_PIECES drifted from the pieces the golden was generated with"
    );
    assert!(golden.contains("== live\n") && golden.contains("== replayed\n"), "golden is empty");
    for (label, request) in [
        ("no tools", issue_request(json!({"tools": null}))),
        ("tool_choice none", issue_request(json!({"tool_choice": "none"}))),
    ] {
        let tools = StreamTools::from_request(&request);
        let live = mask_created(&live_body(GOLDEN_PIECES, None, tools.clone()).await);
        let replayed = mask_created(&replayed_body(GOLDEN_PIECES, None, tools).await);
        let got = format!("== live\n{live}== replayed\n{replayed}");
        assert_eq!(got, golden, "{label}: the SSE output is not v0.70.2's");
    }
}

// T6 — mutant: captured text is dropped.
#[tokio::test]
async fn t6_capture_that_parses_to_no_call_is_flushed_as_content() {
    let undeclared: &[&str] = &[
        "Running:",
        " <tool",
        "_call>\n<function",
        "=rm>\n<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>",
    ];
    let mention: &[&str] = &["Use the ", "<tool_", "call> tag", " to call a tool."];
    for pieces in [undeclared, mention] {
        for (label, body) in [
            ("live", live_body(pieces, None, issue_tools()).await),
            ("replayed", replayed_body(pieces, None, issue_tools()).await),
        ] {
            let s = parse(&body);
            assert_eq!(s.content.concat(), pieces.concat(), "{label}: content was lost");
            assert_eq!(s.call_frames, 0, "{label}: {s:?}");
            assert_eq!(s.finish, "stop", "{label}");
        }
    }
}

// T7 — mutant: the hold-back never releases.
#[test]
fn t7_hold_back_is_bounded_and_releases_a_dead_prefix_at_once() {
    let longest = TOOL_CALL_MARKERS.iter().map(|m| m.len()).max().expect("markers");
    assert_eq!(MAX_HELD_BYTES, longest - 1);
    for marker in TOOL_CALL_MARKERS {
        for k in 1..marker.len() {
            let prefix = &marker[..k];
            let mut detector = issue_tools().expect("tools declared").detector();
            assert_eq!(detector.push(&format!("ab{prefix}")).as_deref(), Some("ab"));
            assert_eq!(detector.held_len(), k, "holds exactly the marker prefix {prefix:?}");
            // `~` continues no marker: the held prefix is released in the same delta.
            assert_eq!(detector.push("~").as_deref(), Some(format!("{prefix}~").as_str()));
            assert_eq!(detector.held_len(), 0);
            let end = detector.finish();
            assert!(end.content.is_none() && end.calls.is_empty(), "{end:?}");
        }
    }
    // A long run of marker-ish text never holds more than the bound.
    let mut detector = issue_tools().expect("tools declared").detector();
    let mut released = String::new();
    for piece in ["<", "<t", "ool", "<fun", "ction", "<tool_ca", "<<<", "<function", "x"] {
        released.extend(detector.push(piece));
        assert!(detector.held_len() <= MAX_HELD_BYTES, "held {} bytes", detector.held_len());
    }
    let end = detector.finish();
    released.extend(end.content);
    assert_eq!(released, "<<tool<function<tool_ca<<<<functionx");
    assert!(end.calls.is_empty());
}

// T8 — mutant: the flush path writes straight to content.
#[tokio::test]
async fn t8_call_completed_by_the_end_of_stream_flush_is_a_call() {
    // The request's own stop sequence makes the stop filter hold the closing
    // `</tool_call>` until the stream ends; only the flush delivers it.
    let held_close = vec!["</tool_call>\n<tool_response>".to_string()];
    let s = parse(&live_body(ISSUE_DELTAS, Some(&held_close), issue_tools()).await);
    assert_issue_call(&s);

    // A trailing newline is held by the `\nHuman:` stop prefix, then flushed.
    let mut pieces = ISSUE_DELTAS.to_vec();
    pieces.push("\n");
    let s = parse(&live_body(&pieces, None, issue_tools()).await);
    assert!(s.content.is_empty(), "the flushed tail leaked as content: {:?}", s.content);
    assert_eq!(s.calls.len(), 1, "{s:?}");
    assert_eq!(s.finish, "tool_calls");
}

// T9 — mutant: the route bypasses the detector.
#[cfg(feature = "gpu")] // create_test_quantized_model is gpu-gated
#[tokio::test]
async fn t9_chat_completions_stream_route_returns_the_same_tool_calls() {
    use crate::api::create_router;
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig};
    use axum::{body::Body, http::Request};
    use tower::util::ServiceExt;

    // The test model's weights are all zero, so every logit ties and greedy
    // picks an end of the vocabulary. Every token but the middle one decodes to
    // the issue's whole call, so one generated token is one call whichever end
    // the tie breaks to. "<unk>" sits in the middle (the prompt encodes to it),
    // and `ignore_eos` keeps the default EOS id 0 from stopping the reply.
    let call = ISSUE_DELTAS.concat();
    let state = || {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: ArchConstraints::from_architecture("llama"),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 256,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            query_pre_attn_scalar: None,
            bos_token_id: None,
            eos_token_id: None,
        };
        let mut vocab = vec![call.clone(); 256];
        vocab[128] = "<unk>".to_string();
        crate::api::AppState::with_quantized_model_and_vocab(
            create_test_quantized_model(&config),
            vocab,
        )
        .expect("build quantized AppState")
    };
    let body = json!({
        "model": "default",
        "messages": [{"role": "user", "content": "what are the directories in ~/tmp/"}],
        "tools": [{"type": "function", "function": {"name": "bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"]}}}],
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": true,
        "stream": true
    })
    .to_string();
    let mut streams = Vec::new();
    for uri in ["/v1/chat/completions", "/v1/chat/completions/stream"] {
        let response = create_router(state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .expect("build request"),
            )
            .await
            .expect("dispatch");
        assert_eq!(response.status(), 200, "{uri}");
        let s = parse(&body_text(response).await);
        assert_eq!(s.call_frames, 1, "{uri}: no delta.tool_calls frame: {s:?}");
        assert_eq!(s.calls[0]["function"]["name"], "bash", "{uri}");
        assert_eq!(s.finish, "tool_calls", "{uri}");
        assert!(s.content.is_empty(), "{uri}: {:?}", s.content);
        streams.push(s.calls);
    }
    assert_eq!(streams[0], streams[1], "the two routes stream different calls");
}

// T11 — mutant: the field always serializes.
#[test]
fn t11_delta_tool_calls_is_omitted_when_empty() {
    let content = ChatCompletionChunk::content(REQUEST_ID, MODEL, "hi");
    let json = serde_json::to_value(&content).expect("serialize");
    assert!(json["choices"][0]["delta"].get("tool_calls").is_none(), "{json}");

    for tool_calls in [None, Some(Vec::new())] {
        let delta = ChatDelta {
            role: None,
            content: Some("hi".to_string()),
            tool_calls,
        };
        let json = serde_json::to_string(&delta).expect("serialize");
        assert_eq!(json, r#"{"content":"hi"}"#);
    }

    let (calls, _) = non_streaming(&ISSUE_DELTAS.concat());
    let chunk = ChatCompletionChunk::tool_calls(REQUEST_ID, MODEL, calls);
    let json = serde_json::to_value(&chunk).expect("serialize");
    let delta = &json["choices"][0]["delta"];
    assert_eq!(delta["tool_calls"][0]["index"], 0, "{json}");
    assert!(delta.get("content").is_none() && delta.get("role").is_none(), "{json}");
    assert!(json["choices"][0]["finish_reason"].is_null(), "{json}");
}

// T12 — mutant: `{` is treated as a marker.
#[tokio::test]
async fn t12_bare_json_call_without_a_marker_streams_as_content() {
    let pieces = ["{\"name\": \"bash\", ", "\"arguments\": ", "{\"command\": \"ls\"}}"];
    for (label, body) in [
        ("live", live_body(&pieces, None, issue_tools()).await),
        ("replayed", replayed_body(&pieces, None, issue_tools()).await),
    ] {
        let s = parse(&body);
        assert_eq!(s.content, pieces, "{label}: bare JSON was held or captured");
        assert_eq!(s.call_frames, 0, "{label}: {s:?}");
        assert_eq!(s.finish, "stop", "{label}");
    }
    // D-c: the non-streaming path reads the same text as a call. This pins the
    // known difference; closing it is a later row.
    let (calls, reason) = non_streaming(&pieces.concat());
    assert_eq!(calls.len(), 1, "the non-streaming path no longer reads bare JSON");
    assert_eq!(reason, "tool_calls");
}
