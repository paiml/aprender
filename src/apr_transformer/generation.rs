//! APR Transformer Generation (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.
//! Token generation with KV cache support.

use super::{AprKVCache, AprTransformer, GenerateConfig};
use crate::error::{RealizarError, Result};

/// GH-330: Check if a token is an end-of-sequence marker.
///
/// Uses the config-provided stop tokens (Design by Contract).
/// Token 0 is always treated as EOS (padding/unknown).
#[inline]
fn is_eos_token(token: u32, stop_tokens: &[u32]) -> bool {
    token == 0 || stop_tokens.contains(&token)
}

/// Sample the next token from logits using temperature scaling
fn sample_from_logits(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        // Greedy: pick argmax
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    } else {
        // Temperature-scaled sampling (currently picks argmax of probs)
        let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = scaled.iter().map(|s| (s - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|e| e / sum).collect();
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    }
}

/// Process prompt tokens and return logits from the last token
fn process_prompt_tokens(
    model: &AprTransformer,
    prompt: &[u32],
    cache: &mut AprKVCache,
    trace: bool,
) -> Result<Vec<f32>> {
    if trace {
        eprintln!("[TRACE] Processing {} prompt tokens...", prompt.len());
    }
    let mut logits = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        let start = std::time::Instant::now();
        logits = model.forward_with_cache(token, cache, pos)?;
        if trace {
            eprintln!("[TRACE] Prompt token {}: {:?}", pos, start.elapsed());
        }
    }
    Ok(logits)
}

/// Generate tokens up to max_tokens or EOS
fn generate_next_tokens(
    model: &AprTransformer,
    cache: &mut AprKVCache,
    output: &mut Vec<u32>,
    initial_logits: Vec<f32>,
    config: &GenerateConfig,
    trace: bool,
) -> Result<()> {
    let mut logits = initial_logits;
    for i in 0..config.max_tokens {
        let next_token = sample_from_logits(&logits, config.temperature);
        output.push(next_token);

        if is_eos_token(next_token, &config.stop_tokens) {
            break;
        }

        // If we need more tokens, process this one to get logits for the next
        if i < config.max_tokens - 1 {
            let start = std::time::Instant::now();
            logits = model.forward_with_cache(next_token, cache, output.len() - 1)?;
            if trace {
                eprintln!(
                    "[TRACE] Gen token {} (pos {}): {:?}",
                    i,
                    output.len() - 1,
                    start.elapsed()
                );
            }
        }
    }
    Ok(())
}

/// Generate tokens using KV cache for efficiency (Y4)
///
/// # Arguments
///
/// * `model` - The APR transformer model
/// * `prompt` - Initial token IDs
/// * `config` - Generation configuration
///
/// # Returns
///
/// Generated token sequence (including prompt)
///
/// # Errors
///
/// Returns error if prompt is empty or forward pass fails.
pub(crate) fn generate_with_cache(
    model: &AprTransformer,
    prompt: &[u32],
    config: &GenerateConfig,
) -> Result<Vec<u32>> {
    if prompt.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Prompt cannot be empty".to_string(),
        });
    }

    let trace = std::env::var("REALIZE_TRACE").is_ok();
    let mut cache = AprKVCache::new(&model.config);
    let mut output = prompt.to_vec();

    let logits = process_prompt_tokens(model, prompt, &mut cache, trace)?;
    generate_next_tokens(model, &mut cache, &mut output, logits, config, trace)?;

    if trace {
        eprintln!(
            "[TRACE] Generation complete. Total output tokens: {}",
            output.len()
        );
    }

    Ok(output)
}

/// Single-token forward pass with optional trace logging.
fn forward_with_trace(
    model: &AprTransformer,
    token: u32,
    cache: &mut AprKVCache,
    pos: usize,
    step: usize,
    trace: bool,
) -> Result<Vec<f32>> {
    let start = std::time::Instant::now();
    let logits = model.forward_with_cache(token, cache, pos)?;
    if trace {
        eprintln!(
            "[TRACE] Gen token {} (pos {}): {:?}",
            step,
            pos,
            start.elapsed()
        );
    }
    Ok(logits)
}

/// Log streaming generation completion.
fn trace_generation_complete(trace: bool, total_tokens: usize) {
    if trace {
        eprintln!(
            "[TRACE] Streaming generation complete. Total output tokens: {}",
            total_tokens
        );
    }
}

/// Generate tokens with streaming callback (GH-284)
///
/// Same as `generate_with_cache` but calls `on_token` after each generated
/// token, enabling true per-token streaming to HTTP clients.
///
/// # Arguments
///
/// * `model` - The APR transformer model
/// * `prompt` - Initial token IDs
/// * `config` - Generation configuration
/// * `on_token` - Callback for each new token. Return `false` to stop early
///   (e.g., client disconnected).
///
/// # Returns
///
/// Generated token sequence (including prompt)
///
/// # Errors
///
/// Returns error if prompt is empty or forward pass fails.
pub(crate) fn generate_with_cache_streaming<F>(
    model: &AprTransformer,
    prompt: &[u32],
    config: &GenerateConfig,
    mut on_token: F,
) -> Result<Vec<u32>>
where
    F: FnMut(u32) -> bool,
{
    if prompt.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Prompt cannot be empty".to_string(),
        });
    }

    let trace = std::env::var("REALIZE_TRACE").is_ok();
    let mut cache = AprKVCache::new(&model.config);
    let mut output = prompt.to_vec();

    let logits = process_prompt_tokens(model, prompt, &mut cache, trace)?;

    // Generate tokens with streaming callback
    let mut logits = logits;
    for i in 0..config.max_tokens {
        let next_token = sample_from_logits(&logits, config.temperature);
        output.push(next_token);

        if is_eos_token(next_token, &config.stop_tokens) {
            break;
        }

        // GH-284: Stream token to client — stop if callback returns false
        if !on_token(next_token) {
            break;
        }

        if i < config.max_tokens - 1 {
            logits = forward_with_trace(model, next_token, &mut cache, output.len() - 1, i, trace)?;
        }
    }

    trace_generation_complete(trace, output.len());

    Ok(output)
}
