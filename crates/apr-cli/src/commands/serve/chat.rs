
fn merge_special_tokens_into_vocab(
    added_tokens: Option<&Vec<serde_json::Value>>,
    vocab: &mut Vec<String>,
) -> (Option<u32>, Option<u32>) {
    let mut bos_token_id = None;
    let mut eos_token_id = None;

    let tokens: Vec<(u32, String)> = added_tokens
        .into_iter()
        .flatten()
        .filter_map(parse_special_token)
        .inspect(|(id, content)| {
            let (is_bos, is_eos) = classify_bos_eos(content);
            if is_bos {
                bos_token_id = Some(*id);
            }
            if is_eos {
                eos_token_id = Some(*id);
            }
        })
        .collect();

    if let Some(max_id) = tokens.iter().map(|(id, _)| *id).max() {
        if max_id as usize >= vocab.len() {
            vocab.resize(max_id as usize + 1, "<unused>".to_string());
        }
    }
    for (id, content) in tokens {
        if (id as usize) < vocab.len() {
            vocab[id as usize] = content;
        }
    }

    (bos_token_id, eos_token_id)
}

pub(crate) fn load_safetensors_tokenizer(path: &Path) -> Option<SafeTensorsTokenizerInfo> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Extract vocabulary from model.vocab (token -> id mapping)
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;

    // Build vocab vector sorted by ID (for index-based lookup)
    let mut vocab_vec: Vec<(String, u32)> = vocab_map
        .iter()
        .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
        .collect();
    vocab_vec.sort_by_key(|(_, id)| *id);
    let vocab: Vec<String> = vocab_vec.into_iter().map(|(token, _)| token).collect();

    // Extract BPE merge rules from model.merges
    // Merges are stored as ["ab cd", "ef gh", ...] meaning "merge 'ab' + 'cd'"
    let merges: Vec<(String, String)> = json
        .get("model")?
        .get("merges")?
        .as_array()?
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let parts: Vec<&str> = s.split(' ').collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    // Extract special tokens and merge them into vocabulary (PMAT-099)
    let added_tokens = json.get("added_tokens").and_then(|v| v.as_array());
    let mut vocab = vocab;
    let (bos_token_id, eos_token_id) = merge_special_tokens_into_vocab(added_tokens, &mut vocab);

    // Create BPE tokenizer with vocab and merge rules
    let tokenizer = realizar::tokenizer::BPETokenizer::new(vocab.clone(), merges, "<unk>").ok()?;

    Some(SafeTensorsTokenizerInfo {
        tokenizer: std::sync::Arc::new(tokenizer),
        vocab,
        bos_token_id,
        eos_token_id,
    })
}

/// Parse a chat completion request from raw JSON, with fallback for backwards compatibility (GH-160)
#[cfg(feature = "inference")]
#[allow(clippy::result_large_err)]
fn parse_chat_completion_request(
    request: &serde_json::Value,
) -> std::result::Result<ChatCompletionRequest, axum::response::Response> {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    if let Ok(req) = serde_json::from_value::<ChatCompletionRequest>(request.clone()) {
        return Ok(req);
    }

    // Fallback: extract from raw JSON
    let messages = request.get("messages").and_then(|m| m.as_array());
    if messages.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "Missing messages field"})),
        )
            .into_response());
    }
    let msgs: Vec<ChatMessage> = messages
        .expect("messages presence checked above")
        .iter()
        .filter_map(|m| {
            Some(ChatMessage {
                role: m.get("role")?.as_str()?.to_string(),
                content: m.get("content").and_then(|c| c.as_str()).map(String::from),
                tool_calls: None,
                tool_call_id: m
                    .get("tool_call_id")
                    .and_then(|t| t.as_str())
                    .map(String::from),
                name: m.get("name").and_then(|n| n.as_str()).map(String::from),
            })
        })
        .collect();
    Ok(ChatCompletionRequest {
        model: request
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("default")
            .to_string(),
        messages: msgs,
        tools: request
            .get("tools")
            .and_then(|t| serde_json::from_value(t.clone()).ok()),
        tool_choice: request
            .get("tool_choice")
            .and_then(|t| serde_json::from_value(t.clone()).ok()),
        max_tokens: request
            .get("max_tokens")
            .and_then(|m| m.as_u64())
            .map(|v| v as u32),
        stream: request
            .get("stream")
            .and_then(|s| s.as_bool())
            .unwrap_or(false),
        temperature: request
            .get("temperature")
            .and_then(|t| t.as_f64())
            .map(|v| v as f32),
        top_p: request
            .get("top_p")
            .and_then(|t| t.as_f64())
            .map(|v| v as f32),
    })
}

/// Build a ChatML-formatted prompt from a parsed chat completion request (GH-160)
#[cfg(feature = "inference")]
fn build_chatml_prompt(request: &ChatCompletionRequest, has_tools: bool) -> String {
    let mut prompt = String::new();

    // Add system message with tools if present
    if has_tools {
        let tools_prompt =
            super::types::format_tools_prompt(request.tools.as_deref().unwrap_or(&[]));
        let has_system = request.messages.iter().any(|m| m.role == "system");
        if !has_system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str("You are a helpful assistant.");
            prompt.push_str(&tools_prompt);
            prompt.push_str("<|im_end|>\n");
        }
    }

    for msg in &request.messages {
        prompt.push_str(&format!("<|im_start|>{}\n", msg.role));

        if msg.role == "tool" {
            if let Some(ref tool_call_id) = msg.tool_call_id {
                prompt.push_str(&format!("[Tool Result for {}]\n", tool_call_id));
            }
        }

        if msg.role == "system" && has_tools {
            if let Some(ref content) = msg.content {
                prompt.push_str(content);
            }
            prompt.push_str(&super::types::format_tools_prompt(
                request.tools.as_deref().unwrap_or(&[]),
            ));
        } else if let Some(ref content) = msg.content {
            prompt.push_str(content);
        }

        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                prompt.push_str(&format!(
                    "\n[Tool Call: {} with args {}]",
                    tc.function.name, tc.function.arguments
                ));
            }
        }

        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// SafeTensors chat completions handler (PAR-301, GH-160 Tool Calling)
#[cfg(feature = "inference")]
pub(crate) async fn safetensors_chat_completions_handler(
    axum::extract::State(state): axum::extract::State<SafeTensorsState>,
    axum::Json(request): axum::Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::{sse::Event, IntoResponse, Sse};
    use futures_util::stream;

    // Parse request - try structured first, fallback to raw JSON (GH-160)
    let parsed_request = match parse_chat_completion_request(&request) {
        Ok(req) => req,
        Err(resp) => return resp,
    };

    let max_tokens = parsed_request.max_tokens.unwrap_or(50) as usize;
    let stream_mode = parsed_request.stream;
    let has_tools = parsed_request.tools.as_ref().is_some_and(|t| !t.is_empty());

    // Build prompt from messages (ChatML format)
    let prompt = build_chatml_prompt(&parsed_request, has_tools);

    // Get transformer
    let transformer = match &state.transformer {
        Some(t) => t.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(
                    serde_json::json!({"error": "Inference not available - missing config.json"}),
                ),
            )
                .into_response();
        }
    };

    // Encode prompt using BPE tokenizer (PMAT-093)
    let input_ids = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info.tokenizer.encode(&prompt)
    } else {
        // Fallback: character-level tokenization (no tokenizer.json)
        prompt.chars().map(|c| c as u32).collect()
    };

    // PMAT-103 FIX: Use generate_with_cache for O(n) generation
    // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
    // generate_with_cache() uses KV cache for incremental generation = O(n)
    let start = Instant::now();
    let temperature = request
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0) as f32;
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
        stop_tokens: vec![],
        cancel: realizar::generate::CancelToken::never(),
    };
    let output_ids = {
        // PMAT-189: Handle transformer lock poisoning gracefully
        let t = match transformer.lock() {
            Ok(guard) => guard,
            Err(_poisoned) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": "Transformer state corrupted (lock poisoned). Please restart the server."
                    })),
                )
                    .into_response();
            }
        };
        match t.generate_with_cache(&input_ids, &gen_config) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": format!("Generation failed: {e}")})),
                )
                    .into_response();
            }
        }
    };
    let elapsed = start.elapsed();

    // Decode output using BPE tokenizer (PMAT-093)
    let new_tokens = &output_ids[input_ids.len()..];
    let output_text = if let Some(ref tok_info) = state.tokenizer_info {
        match tok_info.tokenizer.decode(new_tokens) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Warning: BPE decode failed, falling back to vocab lookup: {e}");
                simple_decode(new_tokens, &tok_info.vocab)
            }
        }
    } else {
        new_tokens
            .iter()
            .map(|&id| char::from_u32(id.min(127)).unwrap_or('?'))
            .collect()
    };

    // Clean output (remove any trailing special tokens)
    let output_text = output_text
        .split("<|im_end|>")
        .next()
        .unwrap_or(&output_text)
        .to_string();

    let tokens_generated = new_tokens.len();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let tool_calls = if has_tools {
        super::types::parse_tool_calls(&output_text)
    } else {
        None
    };

    build_chat_response(
        output_text,
        tool_calls,
        stream_mode,
        input_ids.len(),
        tokens_generated,
        elapsed,
        tok_per_sec,
    )
}

/// SafeTensors Ollama `/api/chat` handler (PMAT-923).
///
/// Reuses the SAME generation backend as `/v1/chat/completions`
/// ([`safetensors_chat_completions_handler`]): translate the Ollama request to
/// the OpenAI-chat JSON that handler already consumes, run it, then re-shape the
/// OpenAI response into Ollama's `{message:{role,content}, done}` schema.
#[cfg(feature = "inference")]
pub(crate) async fn safetensors_ollama_chat_handler(
    state: axum::extract::State<SafeTensorsState>,
    axum::Json(req): axum::Json<super::ollama::OllamaChatRequest>,
) -> axum::response::Response {
    let model = super::ollama::model_label(&req.model);
    let stream = req.stream;
    let openai_body = super::ollama::ollama_chat_to_openai(&req);
    let inner = safetensors_chat_completions_handler(state, axum::Json(openai_body)).await;
    // PMAT-928: the SafeTensors backend is batch (`generate_with_cache`), so
    // `stream:true` emits NDJSON framing over the coalesced result (intermediate
    // done:false + terminal done:true); `stream:false` keeps a single object.
    if stream {
        super::ollama::reshape_openai_to_ollama_ndjson(
            super::ollama::OllamaStreamKind::Chat,
            model,
            inner,
        )
        .await
    } else {
        super::ollama::reshape_openai_to_ollama_chat(model, inner).await
    }
}

/// SafeTensors Ollama `/api/generate` handler (PMAT-923).
///
/// Same backend as `/v1/chat/completions`, emitting Ollama's flat
/// `{response, done}` schema.
#[cfg(feature = "inference")]
pub(crate) async fn safetensors_ollama_generate_handler(
    state: axum::extract::State<SafeTensorsState>,
    axum::Json(req): axum::Json<super::ollama::OllamaGenerateRequest>,
) -> axum::response::Response {
    let model = super::ollama::model_label(&req.model);
    let stream = req.stream;
    let openai_body = super::ollama::ollama_generate_to_openai(&req);
    let inner = safetensors_chat_completions_handler(state, axum::Json(openai_body)).await;
    // PMAT-928: NDJSON framing over the batch result when `stream:true`.
    if stream {
        super::ollama::reshape_openai_to_ollama_ndjson(
            super::ollama::OllamaStreamKind::Generate,
            model,
            inner,
        )
        .await
    } else {
        super::ollama::reshape_openai_to_ollama_generate(model, inner).await
    }
}

/// Generate a unique request ID for OpenAI-compatible responses.
#[cfg(feature = "inference")]
fn generate_request_id() -> String {
    format!(
        "chatcmpl-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id()
    )
}

/// Build OpenAI-compatible chat completion response (streaming or non-streaming).
#[cfg(feature = "inference")]
#[allow(clippy::needless_pass_by_value)]
fn build_chat_response(
    output_text: String,
    tool_calls: Option<Vec<super::types::ToolCall>>,
    stream_mode: bool,
    prompt_tokens: usize,
    tokens_generated: usize,
    elapsed: std::time::Duration,
    tok_per_sec: f64,
) -> axum::response::Response {
    use axum::response::{sse::Event, IntoResponse, Sse};
    use futures_util::stream;

    let request_id = generate_request_id();
    let has_tool_calls = tool_calls.is_some();
    let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if stream_mode {
        let delta = if has_tool_calls {
            serde_json::json!({"role": "assistant", "tool_calls": tool_calls})
        } else {
            serde_json::json!({"role": "assistant", "content": output_text})
        };
        let response = serde_json::json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "safetensors",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        });
        let stream = stream::once(async move {
            Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string()))
        });
        Sse::new(stream).into_response()
    } else {
        let message = if has_tool_calls {
            serde_json::json!({"role": "assistant", "content": null, "tool_calls": tool_calls})
        } else {
            serde_json::json!({"role": "assistant", "content": output_text})
        };
        axum::Json(serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": "safetensors",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens_generated,
                "total_tokens": prompt_tokens + tokens_generated
            },
            "latency_ms": elapsed.as_millis(),
            "tok_per_sec": tok_per_sec
        }))
        .into_response()
    }
}

// ============================================================================
// PMAT-125 B3: unit tests for SafeTensors chat/tokenizer pure helpers.
//
// `chat.rs` is `include!`d into the `serve::safetensors` module, so these
// helpers (plus `parse_special_token` / `classify_bos_eos` /
// `merge_special_tokens_into_vocab` defined in `safetensors.rs` itself) are all
// in that scope. The `types` module is a sibling under `serve`, reached via
// `super::super::types`. CPU-only: no transformer, no GPU.
// ============================================================================
#[cfg(all(test, feature = "inference"))]
mod chat_helper_tests {
    use super::*;

    // ---- classify_bos_eos ---------------------------------------------

    #[test]
    fn classify_bos_eos_detects_bos_markers() {
        assert_eq!(classify_bos_eos("<s>"), (true, false));
        assert_eq!(classify_bos_eos("<|bos|>"), (true, false));
    }

    #[test]
    fn classify_bos_eos_detects_eos_markers() {
        assert_eq!(classify_bos_eos("</s>"), (false, true));
        assert_eq!(classify_bos_eos("<|eos|>"), (false, true));
        assert_eq!(classify_bos_eos("<|im_end|>"), (false, true));
    }

    #[test]
    fn classify_bos_eos_plain_token_is_neither() {
        assert_eq!(classify_bos_eos("hello"), (false, false));
    }

    // ---- parse_special_token ------------------------------------------

    #[test]
    fn parse_special_token_extracts_id_and_content() {
        let tok = serde_json::json!({"id": 5, "content": "<|im_end|>"});
        assert_eq!(
            parse_special_token(&tok),
            Some((5, "<|im_end|>".to_string()))
        );
    }

    #[test]
    fn parse_special_token_none_when_missing_fields() {
        assert!(parse_special_token(&serde_json::json!({"id": 1})).is_none());
        assert!(parse_special_token(&serde_json::json!({"content": "x"})).is_none());
        assert!(parse_special_token(&serde_json::json!({"id": "nan", "content": "x"})).is_none());
    }

    // ---- merge_special_tokens_into_vocab ------------------------------

    #[test]
    fn merge_special_tokens_none_added_tokens_keeps_vocab() {
        let mut vocab = vec!["a".to_string(), "b".to_string()];
        let (bos, eos) = merge_special_tokens_into_vocab(None, &mut vocab);
        assert_eq!(vocab, vec!["a", "b"]);
        assert!(bos.is_none());
        assert!(eos.is_none());
    }

    #[test]
    fn merge_special_tokens_overwrites_in_range_and_detects_eos() {
        let added = vec![serde_json::json!({"id": 1, "content": "<|im_end|>"})];
        let mut vocab = vec!["a".to_string(), "b".to_string()];
        let (bos, eos) = merge_special_tokens_into_vocab(Some(&added), &mut vocab);
        assert_eq!(vocab[1], "<|im_end|>");
        assert!(bos.is_none());
        assert_eq!(eos, Some(1));
    }

    #[test]
    fn merge_special_tokens_resizes_vocab_for_high_id() {
        let added = vec![
            serde_json::json!({"id": 4, "content": "</s>"}),
            serde_json::json!({"id": 3, "content": "<s>"}),
        ];
        let mut vocab = vec!["a".to_string()];
        let (bos, eos) = merge_special_tokens_into_vocab(Some(&added), &mut vocab);
        // resized to max_id (4) + 1 = 5 entries; gap filled with "<unused>".
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab[2], "<unused>");
        assert_eq!(vocab[3], "<s>");
        assert_eq!(vocab[4], "</s>");
        assert_eq!(bos, Some(3));
        assert_eq!(eos, Some(4));
    }

    // ---- parse_chat_completion_request --------------------------------

    #[test]
    fn parse_chat_completion_structured_path() {
        let req = serde_json::json!({
            "model": "apr",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": true,
            "temperature": 0.3
        });
        let parsed = parse_chat_completion_request(&req).expect("structured parse");
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(parsed.messages[0].role, "user");
        assert_eq!(parsed.max_tokens, Some(16));
        assert!(parsed.stream);
        assert_eq!(parsed.temperature, Some(0.3));
    }

    #[test]
    fn parse_chat_completion_missing_messages_is_err() {
        let req = serde_json::json!({"model": "apr"});
        let result = parse_chat_completion_request(&req);
        assert!(result.is_err(), "missing messages must yield a 400 response");
        let resp = result.err().expect("error response");
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn parse_chat_completion_fallback_extracts_fields() {
        // A messages entry whose `content` is non-string forces the structured
        // deserialize to fail (content must be Option<String>), exercising the
        // raw-JSON fallback path.
        let req = serde_json::json!({
            "messages": [
                {"role": "user", "content": 12345},
                {"role": "assistant", "content": "ok"}
            ],
            "max_tokens": 8
        });
        let parsed = parse_chat_completion_request(&req).expect("fallback parse");
        assert!(parsed.messages.iter().any(|m| m.role == "assistant"));
        assert_eq!(
            parsed.model, "default",
            "model defaults to 'default' in fallback"
        );
        assert_eq!(parsed.max_tokens, Some(8));
    }

    // ---- build_chatml_prompt ------------------------------------------

    fn user_request(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "apr".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(content.to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            max_tokens: None,
            stream: false,
            temperature: None,
            top_p: None,
        }
    }

    #[test]
    fn build_chatml_prompt_no_tools_basic() {
        let req = user_request("hello");
        let prompt = build_chatml_prompt(&req, false);
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn build_chatml_prompt_with_tools_injects_system() {
        let mut req = user_request("weather?");
        req.tools = Some(vec![super::super::types::Tool {
            tool_type: "function".to_string(),
            function: super::super::types::FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Look up weather".to_string()),
                parameters: None,
            },
        }]);
        let prompt = build_chatml_prompt(&req, true);
        // No system message present → synthesized system block carrying tools.
        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("get_weather"), "tool definition injected");
    }

    #[test]
    fn build_chatml_prompt_tool_role_renders_tool_result() {
        let mut req = user_request("ignored");
        req.messages = vec![ChatMessage {
            role: "tool".to_string(),
            content: Some("sunny".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_42".to_string()),
            name: None,
        }];
        let prompt = build_chatml_prompt(&req, false);
        assert!(prompt.contains("[Tool Result for call_42]"));
        assert!(prompt.contains("sunny"));
    }

    // ---- generate_request_id (chat.rs variant) ------------------------

    #[test]
    fn chat_generate_request_id_prefix() {
        let id = generate_request_id();
        assert!(id.starts_with("chatcmpl-"));
        assert_eq!(id.split('-').count(), 3);
    }

    // ---- build_chat_response ------------------------------------------

    #[tokio::test]
    async fn build_chat_response_non_streaming_json_body() {
        use axum::body::to_bytes;
        let resp = build_chat_response(
            "the answer".to_string(),
            None,
            false,
            5,
            3,
            std::time::Duration::from_millis(10),
            300.0,
        );
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.expect("body");
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "the answer");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 5);
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert_eq!(v["usage"]["total_tokens"], 8);
    }

    #[tokio::test]
    async fn build_chat_response_with_tool_calls_sets_finish_reason() {
        use axum::body::to_bytes;
        let tool_calls = vec![super::super::types::ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: super::super::types::FunctionCall {
                name: "f".to_string(),
                arguments: "{}".to_string(),
            },
        }];
        let resp = build_chat_response(
            String::new(),
            Some(tool_calls),
            false,
            2,
            0,
            std::time::Duration::from_millis(1),
            0.0,
        );
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.expect("body");
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert!(v["choices"][0]["message"]["content"].is_null());
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "f"
        );
    }

    #[tokio::test]
    async fn build_chat_response_streaming_is_sse() {
        let resp = build_chat_response(
            "hi".to_string(),
            None,
            true,
            1,
            1,
            std::time::Duration::from_millis(1),
            1.0,
        );
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/event-stream"),
            "streaming mode is SSE, got {ct}"
        );
    }
}
