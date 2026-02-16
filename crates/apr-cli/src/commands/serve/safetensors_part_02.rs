
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
        tok_info
            .tokenizer
            .decode(new_tokens)
            .unwrap_or_else(|_| simple_decode(new_tokens, &tok_info.vocab))
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
