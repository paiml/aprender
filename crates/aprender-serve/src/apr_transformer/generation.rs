//! APR Transformer Generation (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.
//! Token generation with KV cache support.

use super::{AprKVCache, AprTransformer, GenerateConfig};
use crate::error::{RealizarError, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// GH-330: Check if a token is an end-of-sequence marker.
///
/// Uses the config-provided stop tokens (Design by Contract).
/// Token 0 is always treated as EOS (padding/unknown).
#[inline]
fn is_eos_token(token: u32, stop_tokens: &[u32]) -> bool {
    token == 0 || stop_tokens.contains(&token)
}

/// Argmax over a logit slice (greedy selection).
#[inline]
fn argmax_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx as u32)
}

/// The next token: greedy (this loop's own argmax) at temperature 0 or top-k 1,
/// otherwise one seeded draw through the shared sampler (#3760).
///
/// PMAT-820 threaded `top_k`/`top_p` through here, but the selection that followed
/// was the ARGMAX of the survivors. The highest logit always survives, so that was
/// the greedy token for every parameter value, and `--temperature`, `--top-k`,
/// `--top-p` and `--seed` changed nothing. It now draws from the filtered
/// distribution with `crate::sampling::draw_seeded`, the draw the GGUF loops make.
fn sample_from_logits(logits: &[f32], config: &GenerateConfig, rng: &mut StdRng) -> u32 {
    if crate::sampling::is_greedy(config.temperature, config.top_k) {
        return argmax_logits(logits);
    }
    crate::sampling::draw_seeded(logits, config.temperature, config.top_k, config.top_p, rng)
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
    let mut rng = StdRng::seed_from_u64(config.seed);
    for i in 0..config.max_tokens {
        // aprender#2376(3): CANCELLATION POLL. The HTTP client may be gone;
        // stop here instead of burning a core to max_tokens for nobody.
        if config.cancel.is_cancelled() {
            break;
        }
        let next_token = sample_from_logits(&logits, config, &mut rng);
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
    let mut rng = StdRng::seed_from_u64(config.seed);
    for i in 0..config.max_tokens {
        // aprender#2376(3): CANCELLATION POLL. The HTTP client may be gone;
        // stop here instead of burning a core to max_tokens for nobody.
        if config.cancel.is_cancelled() {
            break;
        }
        let next_token = sample_from_logits(&logits, config, &mut rng);
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

#[cfg(test)]
mod sampler_tests {
    //! #3760: the APR sampler DRAWS. PMAT-820's rows here asserted that sampling at
    //! temperature > 0 returned the argmax ("byte-identical to legacy"), which pinned
    //! the defect: a sampler that could never draw. The top-k/top-p filter itself is
    //! the shared `crate::sampling::draw`, pinned by the GGUF top-k/top-p rows.
    use super::{argmax_logits, sample_from_logits};
    use crate::apr_transformer::GenerateConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn cfg(temperature: f32, top_k: usize, top_p: f32) -> GenerateConfig {
        GenerateConfig {
            max_tokens: 1,
            temperature,
            top_p,
            top_k,
            ..GenerateConfig::default()
        }
    }

    /// Several comparable logits, so a real draw lands off the argmax often.
    const FLAT: [f32; 6] = [1.0, 0.9, 1.1, 0.95, 1.05, 0.85];

    fn draws(config: &GenerateConfig, seed: u64, n: usize) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| sample_from_logits(&FLAT, config, &mut rng))
            .collect()
    }

    #[test]
    fn a_sampled_step_draws_off_the_argmax() {
        let greedy = argmax_logits(&FLAT);
        let tokens = draws(&cfg(1.0, 0, 1.0), 7, 64);
        assert!(
            tokens.iter().any(|&t| t != greedy),
            "64 draws at temperature 1.0 all returned the argmax: the sampler does not draw"
        );
    }

    #[test]
    fn the_same_seed_reproduces_and_another_seed_changes_the_draws() {
        let config = cfg(1.0, 0, 1.0);
        assert_eq!(draws(&config, 7, 32), draws(&config, 7, 32));
        assert_ne!(
            draws(&config, 7, 32),
            draws(&config, 8, 32),
            "the seed is not used"
        );
    }

    /// Greedy is this loop's own argmax: at temperature 0, or top-k 1 at any temperature.
    #[test]
    fn temperature_zero_or_top_k_one_is_the_argmax() {
        let expected = argmax_logits(&FLAT);
        for seed in 0..8 {
            assert!(draws(&cfg(0.0, 0, 1.0), seed, 8)
                .iter()
                .all(|&t| t == expected));
            assert!(draws(&cfg(1.0, 1, 1.0), seed, 8)
                .iter()
                .all(|&t| t == expected));
            assert!(draws(&cfg(0.0, 40, 0.1), seed, 8)
                .iter()
                .all(|&t| t == expected));
        }
    }

    /// The filter is honoured: top-k 2 only ever yields the two highest logits.
    #[test]
    fn top_k_bounds_the_draw() {
        let tokens = draws(&cfg(5.0, 2, 1.0), 3, 200);
        assert!(tokens.iter().all(|&t| t == 2 || t == 4), "{tokens:?}");
        assert!(
            tokens.contains(&2) && tokens.contains(&4),
            "both survivors are drawn"
        );
    }
}
