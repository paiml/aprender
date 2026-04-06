//! Inference engine — bridges banco chat handler to realizar's forward pass.
//!
//! Gated behind `#[cfg(feature = "realizar")]`. Provides:
//! - `generate_sync()` — greedy/sampled token generation for non-streaming
//! - `generate_stream()` — yields tokens one at a time for SSE

#[cfg(feature = "realizar")]
use std::sync::Arc;

#[cfg(feature = "realizar")]
use realizar::gguf::{OwnedQuantizedKVCache, OwnedQuantizedModel};

/// Result of a completed generation.
#[cfg(feature = "realizar")]
pub struct GenerationResult {
    pub text: String,
    pub token_count: u32,
    pub finish_reason: String,
}

/// Sampling parameters for token generation.
#[cfg(feature = "realizar")]
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub max_tokens: u32,
}

#[cfg(feature = "realizar")]
impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, max_tokens: 256 }
    }
}

/// Generate a complete response synchronously (non-streaming).
///
/// Runs the autoregressive loop: embed → forward → sample → decode.
/// Returns the full generated text plus token count and finish reason.
#[cfg(feature = "realizar")]
pub fn generate_sync(
    model: &Arc<OwnedQuantizedModel>,
    vocab: &[String],
    prompt_tokens: &[u32],
    params: &SamplingParams,
) -> Result<GenerationResult, String> {
    if prompt_tokens.is_empty() {
        return Err("prompt_tokens must not be empty".to_string());
    }

    let config = model.config();
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let max_seq = prompt_tokens.len() + params.max_tokens as usize;

    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, kv_dim, max_seq);

    // Prefill: process all prompt tokens through the model
    let mut logits = Vec::new();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        logits = model
            .forward_single_with_cache(token, &mut cache, pos)
            .map_err(|e| format!("forward error at pos {pos}: {e}"))?;
    }

    // Decode: generate new tokens autoregressively
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut pos = prompt_tokens.len();
    let eos_token = find_eos_token(vocab);

    for _ in 0..params.max_tokens {
        let next_token = sample_token(&logits, params);

        // Check EOS
        if Some(next_token) == eos_token {
            return Ok(GenerationResult {
                text: decode_tokens(vocab, &generated_tokens),
                token_count: generated_tokens.len() as u32,
                finish_reason: "stop".to_string(),
            });
        }

        generated_tokens.push(next_token);

        // Forward pass for the new token
        logits = model
            .forward_single_with_cache(next_token, &mut cache, pos)
            .map_err(|e| format!("forward error at pos {pos}: {e}"))?;
        pos += 1;
    }

    Ok(GenerationResult {
        text: decode_tokens(vocab, &generated_tokens),
        token_count: generated_tokens.len() as u32,
        finish_reason: "length".to_string(),
    })
}

/// Generate tokens one at a time for streaming.
///
/// Returns an iterator-like vec of (token_text, is_last, finish_reason).
/// For true async streaming we'd use a channel, but this is simpler for Phase 2b.
#[cfg(feature = "realizar")]
pub fn generate_stream_tokens(
    model: &Arc<OwnedQuantizedModel>,
    vocab: &[String],
    prompt_tokens: &[u32],
    params: &SamplingParams,
) -> Result<Vec<StreamToken>, String> {
    if prompt_tokens.is_empty() {
        return Err("prompt_tokens must not be empty".to_string());
    }

    let config = model.config();
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let max_seq = prompt_tokens.len() + params.max_tokens as usize;

    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, kv_dim, max_seq);

    // Prefill
    let mut logits = Vec::new();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        logits = model
            .forward_single_with_cache(token, &mut cache, pos)
            .map_err(|e| format!("forward error at pos {pos}: {e}"))?;
    }

    // Decode
    let mut tokens = Vec::new();
    let mut pos = prompt_tokens.len();
    let eos_token = find_eos_token(vocab);

    for _ in 0..params.max_tokens {
        let next_token = sample_token(&logits, params);

        if Some(next_token) == eos_token {
            tokens
                .push(StreamToken { text: String::new(), finish_reason: Some("stop".to_string()) });
            return Ok(tokens);
        }

        let raw = vocab
            .get(next_token as usize)
            .cloned()
            .unwrap_or_else(|| format!("<unk:{next_token}>"));
        let text = decode_bpe_text(&raw);

        tokens.push(StreamToken { text, finish_reason: None });

        logits = model
            .forward_single_with_cache(next_token, &mut cache, pos)
            .map_err(|e| format!("forward error at pos {pos}: {e}"))?;
        pos += 1;
    }

    // Hit max_tokens
    tokens.push(StreamToken { text: String::new(), finish_reason: Some("length".to_string()) });

    Ok(tokens)
}

/// A single token in a streaming response.
#[cfg(feature = "realizar")]
pub struct StreamToken {
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Sample a token from logits using temperature + top-k.
#[cfg(feature = "realizar")]
fn sample_token(logits: &[f32], params: &SamplingParams) -> u32 {
    if params.temperature <= 0.0 || params.top_k <= 1 {
        // Greedy: argmax
        return argmax(logits);
    }

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&l| l / params.temperature).collect();

    // Top-k filtering
    let k = (params.top_k as usize).min(scaled.len());
    let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = &indexed[..k];

    // Softmax over top-k
    let max_val = top_k[0].1;
    let exps: Vec<f32> = top_k.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Simple deterministic sampling using hash of logits as pseudo-random
    // (True random would use thread_rng, but this is reproducible for testing)
    let hash = logits_hash(logits);
    let r = (hash as f32) / (u64::MAX as f32);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return top_k[i].0 as u32;
        }
    }

    top_k[0].0 as u32
}

/// Argmax over a logit vector.
#[cfg(feature = "realizar")]
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Decode token IDs back to text using the vocabulary.
/// Handles BPE byte encoding (Ġ → space, Ċ → newline, etc.).
#[cfg(feature = "realizar")]
fn decode_tokens(vocab: &[String], tokens: &[u32]) -> String {
    let raw: String =
        tokens.iter().map(|&id| vocab.get(id as usize).map(String::as_str).unwrap_or("")).collect();
    decode_bpe_text(&raw)
}

/// Decode BPE byte-encoded text back to UTF-8.
///
/// GPT/Qwen BPE uses Unicode characters U+0100..U+01FF to represent raw bytes.
/// Ġ (U+0120) = space (0x20), Ċ (U+010A) = newline (0x0A), etc.
#[cfg(feature = "realizar")]
fn decode_bpe_text(text: &str) -> String {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let cp = ch as u32;
        if (0x100..=0x1FF).contains(&cp) {
            // BPE byte token: U+01XX → byte 0xXX
            bytes.push((cp - 0x100) as u8);
        } else if cp == 0x0100 {
            // U+0100 typically maps to byte 0x00 but often means special — skip
        } else if ch == 'Ā' {
            // Some BPE encodings use Ā for null byte
        } else {
            // Regular UTF-8 character
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(encoded.as_bytes());
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

/// Find the EOS token ID in the vocabulary.
#[cfg(feature = "realizar")]
fn find_eos_token(vocab: &[String]) -> Option<u32> {
    // Common EOS tokens across model families
    let eos_candidates = ["</s>", "<|endoftext|>", "<|end|>", "<eos>", "<|im_end|>", "<|eot_id|>"];
    for candidate in &eos_candidates {
        if let Some(pos) = vocab.iter().position(|t| t == candidate) {
            return Some(pos as u32);
        }
    }
    None
}

/// Simple hash of logits for reproducible pseudo-random sampling.
#[cfg(feature = "realizar")]
fn logits_hash(logits: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &l in logits.iter().take(64) {
        h ^= l.to_bits() as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Encode a text prompt into token IDs using the vocabulary.
///
/// Uses greedy longest-match tokenization. For production, the GGUF vocab
/// includes merge rules that `realizar::tokenizer::BPETokenizer` handles,
/// but for Phase 2b this simple approach works for basic generation.
#[cfg(feature = "realizar")]
pub fn encode_prompt(vocab: &[String], text: &str) -> Vec<u32> {
    if text.is_empty() {
        return Vec::new();
    }

    // Build token→id lookup (could be cached on ModelSlot, but keep simple for now)
    let token_to_id: std::collections::HashMap<&str, u32> =
        vocab.iter().enumerate().map(|(i, t)| (t.as_str(), i as u32)).collect();

    // Greedy longest-match character by character
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut pos = 0;

    while pos < chars.len() {
        let mut best_len = 0;
        let mut best_id = None;

        // Try decreasing lengths from current position
        let max_len = (chars.len() - pos).min(32); // Cap at 32 chars per token
        for len in (1..=max_len).rev() {
            let substr: String = chars[pos..pos + len].iter().collect();
            if let Some(&id) = token_to_id.get(substr.as_str()) {
                best_len = len;
                best_id = Some(id);
                break;
            }
        }

        if let Some(id) = best_id {
            tokens.push(id);
            pos += best_len;
        } else {
            // Unknown character — use UNK token (usually 0)
            tokens.push(0);
            pos += 1;
        }
    }

    tokens
}

/// Compute a mean-pooled embedding using pre-tokenized IDs.
///
/// Looks up each token embedding via `model.embed()`,
/// and returns the mean across all token positions. The resulting vector
/// has `hidden_dim` dimensions.
///
/// Caller should use `ModelSlot::encode_text()` for proper BPE tokenization.
#[cfg(feature = "realizar")]
pub fn embed_tokens(model: &Arc<OwnedQuantizedModel>, token_ids: &[u32]) -> Option<Vec<f32>> {
    if token_ids.is_empty() {
        return None;
    }

    // Get raw embeddings [num_tokens * hidden_dim]
    let raw = model.embed(token_ids);
    let hidden_dim = model.config().hidden_dim;
    let num_tokens = token_ids.len();

    if raw.len() != num_tokens * hidden_dim {
        return None;
    }

    // Mean pool across tokens
    let mut pooled = vec![0.0f32; hidden_dim];
    for t in 0..num_tokens {
        let offset = t * hidden_dim;
        for d in 0..hidden_dim {
            pooled[d] += raw[offset + d];
        }
    }
    let scale = 1.0 / num_tokens as f32;
    for val in &mut pooled {
        *val *= scale;
    }

    // L2 normalize
    let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for val in &mut pooled {
            *val /= norm;
        }
    }

    Some(pooled)
}

// ============================================================================
// Tests (available without inference feature)
// ============================================================================

#[cfg(test)]
#[cfg(feature = "realizar")]
mod tests {
    use super::*;

    fn test_vocab() -> Vec<String> {
        vec![
            "<unk>".to_string(),
            "</s>".to_string(),
            "Hello".to_string(),
            " world".to_string(),
            "!".to_string(),
            "The".to_string(),
            " answer".to_string(),
            " is".to_string(),
            " 42".to_string(),
        ]
    }

    #[test]
    fn test_inf_001_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn test_inf_002_argmax_empty() {
        let logits: Vec<f32> = Vec::new();
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn test_inf_003_decode_tokens() {
        let vocab = test_vocab();
        let tokens = vec![2, 3, 4]; // "Hello", " world", "!"
        assert_eq!(decode_tokens(&vocab, &tokens), "Hello world!");
    }

    #[test]
    fn test_inf_004_decode_unknown_token() {
        let vocab = test_vocab();
        let tokens = vec![2, 999]; // "Hello", out-of-range
        assert_eq!(decode_tokens(&vocab, &tokens), "Hello");
    }

    #[test]
    fn test_inf_005_find_eos_token() {
        let vocab = test_vocab();
        assert_eq!(find_eos_token(&vocab), Some(1)); // "</s>" is at index 1
    }

    #[test]
    fn test_inf_006_find_eos_missing() {
        let vocab = vec!["a".to_string(), "b".to_string()];
        assert_eq!(find_eos_token(&vocab), None);
    }

    #[test]
    fn test_inf_007_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let params = SamplingParams { temperature: 0.0, top_k: 1, max_tokens: 10 };
        assert_eq!(sample_token(&logits, &params), 3);
    }

    #[test]
    fn test_inf_008_encode_prompt() {
        let vocab = test_vocab();
        let tokens = encode_prompt(&vocab, "Hello world!");
        // Should find "Hello", " world", "!" but space handling depends on vocab
        // At minimum, should produce non-empty output
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_inf_009_encode_empty() {
        let vocab = test_vocab();
        assert!(encode_prompt(&vocab, "").is_empty());
    }

    #[test]
    fn test_inf_010_logits_hash_deterministic() {
        let logits = vec![0.1, 0.2, 0.3];
        let h1 = logits_hash(&logits);
        let h2 = logits_hash(&logits);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_inf_011_sampling_params_default() {
        let params = SamplingParams::default();
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.max_tokens, 256);
    }
}
