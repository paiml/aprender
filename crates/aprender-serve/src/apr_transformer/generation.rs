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

/// Argmax over a logit slice (greedy selection).
#[inline]
fn argmax_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx as u32)
}

/// PMAT-820: Compute the surviving candidate set after applying top-k truncation
/// then top-p (nucleus) truncation to temperature-scaled logits.
///
/// Returns `(index, scaled_logit)` pairs sorted by scaled logit descending, pruned
/// to honor `top_k` and `top_p`. This is the APR-path analogue of the canonical
/// GGUF/MoE sampler filter (`infer::qwen3_moe_generate::sample_from_logits`).
///
/// Neutrality contract (required for byte-identical no-regression):
/// - `top_k == 0` OR `top_k >= scaled.len()` → no top-k pruning.
/// - `top_p >= 1.0` (or `<= 0.0`) → no top-p pruning.
///
/// When BOTH are neutral, every token survives, so the argmax of the survivor
/// set equals the argmax of the full distribution (byte-identical to the
/// pre-PMAT-820 temperature-only behavior).
fn top_k_top_p_survivors(scaled: &[f32], top_k: usize, top_p: f32) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k: keep the highest-k logits. Neutral at top_k == 0 or top_k >= vocab.
    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }

    // Top-p (nucleus): smallest prefix whose cumulative softmax prob >= top_p.
    // Neutral at top_p >= 1.0 or top_p <= 0.0.
    if top_p > 0.0 && top_p < 1.0 {
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_vals: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let total: f32 = exp_vals.iter().sum();
        if total > 0.0 {
            let mut cumulative = 0.0;
            let mut cutoff = indexed.len();
            for (i, &ev) in exp_vals.iter().enumerate() {
                cumulative += ev / total;
                if cumulative >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }
    }

    indexed
}

/// Sample the next token from logits honoring temperature, top-k, and top-p.
///
/// PMAT-820 (APR-format generation path): prior to this fix the APR transformer
/// decode loop applied ONLY temperature and silently dropped `config.top_k` /
/// `config.top_p`, so `apr run model.apr --top-p ... --top-k ...` was full-
/// distribution temperature sampling. This now applies
/// temperature → top-k truncate → top-p nucleus → argmax over the survivor set,
/// matching the GGUF/MoE sampler filter semantics.
///
/// Determinism / no-regression: selection is argmax over the (possibly pruned)
/// survivor set. Because the argmax token always survives truncation, neutral
/// values (`top_k == 0` or `>= vocab`, `top_p >= 1.0`) yield a result that is
/// byte-identical to the pre-fix temperature-only argmax. Greedy
/// (`temperature == 0`) is unaffected.
fn sample_from_logits(logits: &[f32], config: &GenerateConfig) -> u32 {
    if config.temperature == 0.0 {
        // Greedy: pick argmax (unaffected by top-k/top-p).
        return argmax_logits(logits);
    }

    // Temperature scaling.
    let scaled: Vec<f32> = logits.iter().map(|l| l / config.temperature).collect();

    // Honor top-k then top-p, then select argmax over the survivors.
    let survivors = top_k_top_p_survivors(&scaled, config.top_k, config.top_p);
    survivors
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| *idx as u32)
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
        let next_token = sample_from_logits(&logits, config);
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
        let next_token = sample_from_logits(&logits, config);
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
mod top_p_top_k_tests {
    //! PMAT-820: APR-format generation path top_p/top_k obligation.
    //!
    //! Before the fix, `sample_from_logits` took only `temperature` and silently
    //! dropped `config.top_k` / `config.top_p`. These tests pin the corrected
    //! behavior:
    //! - F-APR-SAMPLE-TOPK-001 / F-APR-SAMPLE-TOPP-001: a low-probability token
    //!   that the unfiltered (neutral) sampler keeps reachable is provably
    //!   EXCLUDED once top_k / top_p are honored.
    //! - No-regression: neutral values (top_k == 0 or >= vocab, top_p >= 1.0)
    //!   produce the SAME survivor set / argmax as the old temperature-only path,
    //!   and greedy (temperature == 0) is unaffected.
    use super::{argmax_logits, sample_from_logits, top_k_top_p_survivors};
    use crate::apr_transformer::GenerateConfig;

    fn cfg(temperature: f32, top_k: usize, top_p: f32) -> GenerateConfig {
        GenerateConfig {
            max_tokens: 1,
            temperature,
            top_p,
            top_k,
            repetition_penalty: 1.0,
            trace: false,
            stop_tokens: vec![],
        }
    }

    /// Reference for the PRE-FIX temperature-only behavior: argmax of the full
    /// temperature-scaled distribution (no top-k/top-p pruning). The fixed
    /// sampler MUST equal this for any neutral param combination.
    fn legacy_temperature_only(logits: &[f32], temperature: f32) -> u32 {
        if temperature == 0.0 {
            return argmax_logits(logits);
        }
        let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
        argmax_logits(&scaled)
    }

    // --- Survivor-set falsifiers (test the pruning obligation directly) -------

    /// F-APR-SAMPLE-TOPK-001: top_k=1 must collapse the survivor set to exactly
    /// the single highest-logit token. A low-prob token reachable in the
    /// unfiltered (neutral) set is EXCLUDED. RED if top_k ignored.
    #[test]
    fn topk_one_excludes_low_prob_token() {
        // idx 2 is the clear max; idx 0 is a low-prob token reachable unfiltered.
        let scaled = [1.0_f32, 0.5, 9.0, 2.0];
        let low_prob = 0_usize;

        // Unfiltered (neutral) keeps the low-prob token in the candidate set —
        // this is the pre-fix behavior the bug exhibited.
        let unfiltered = top_k_top_p_survivors(&scaled, 0, 1.0);
        assert!(
            unfiltered.iter().any(|(i, _)| *i == low_prob),
            "neutral survivor set must contain the low-prob token (pre-fix reachable)"
        );

        // top_k=1 prunes everything but the argmax → low-prob token GONE.
        let filtered = top_k_top_p_survivors(&scaled, 1, 1.0);
        assert_eq!(filtered.len(), 1, "top_k=1 keeps exactly one survivor");
        assert_eq!(filtered[0].0, 2, "the survivor is the argmax token");
        assert!(
            !filtered.iter().any(|(i, _)| *i == low_prob),
            "top_k=1 must EXCLUDE the low-prob token (RED if top_k dropped)"
        );
    }

    /// F-APR-SAMPLE-TOPP-001: a tight nucleus (top_p=0.1) on a peaked
    /// distribution must keep only the dominant token and exclude the long tail
    /// (incl. a low-prob token reachable unfiltered). RED if top_p ignored.
    #[test]
    fn topp_tight_nucleus_excludes_tail() {
        // idx 1 dominates after softmax; idxs 0,2,3 are the low-prob tail.
        let scaled = [0.0_f32, 12.0, 0.5, 1.0];
        let low_prob = 3_usize;

        let unfiltered = top_k_top_p_survivors(&scaled, 0, 1.0);
        assert!(
            unfiltered.iter().any(|(i, _)| *i == low_prob),
            "neutral survivor set must contain the low-prob tail token"
        );

        let filtered = top_k_top_p_survivors(&scaled, 0, 0.1);
        assert_eq!(
            filtered.len(),
            1,
            "tight nucleus keeps only the dominant token"
        );
        assert_eq!(filtered[0].0, 1, "nucleus survivor is the dominant token");
        assert!(
            !filtered.iter().any(|(i, _)| *i == low_prob),
            "top_p=0.1 must EXCLUDE the low-prob tail token (RED if top_p dropped)"
        );
    }

    // --- End-to-end sampler honoring config ----------------------------------

    /// With top_k=1, the sampler must return the argmax token regardless of
    /// temperature — confirming the param is threaded end-to-end.
    #[test]
    fn sampler_honors_top_k_one() {
        let logits = [1.0_f32, 0.5, 9.0, 2.0];
        let token = sample_from_logits(&logits, &cfg(1.0, 1, 1.0));
        assert_eq!(token, 2, "top_k=1 → argmax token");
    }

    // --- No-regression: neutral values are byte-identical --------------------

    /// top_k == 0 AND top_p >= 1.0 → byte-identical to the legacy
    /// temperature-only argmax across a range of temperatures.
    #[test]
    fn neutral_params_byte_identical_to_legacy() {
        let logits = [0.2_f32, -1.0, 3.3, 1.1, 0.0, 2.9, -0.4];
        for &temp in &[0.0_f32, 0.5, 0.7, 1.0, 1.3, 2.0] {
            let new = sample_from_logits(&logits, &cfg(temp, 0, 1.0));
            let legacy = legacy_temperature_only(&logits, temp);
            assert_eq!(
                new, legacy,
                "neutral params must match legacy temperature-only at temp={temp}"
            );
        }
    }

    /// top_k >= vocab is also neutral (no pruning possible) → matches legacy.
    #[test]
    fn top_k_ge_vocab_is_neutral() {
        let logits = [0.2_f32, -1.0, 3.3, 1.1];
        let new = sample_from_logits(&logits, &cfg(1.0, logits.len(), 1.0));
        assert_eq!(new, legacy_temperature_only(&logits, 1.0));
        // top_k strictly greater than vocab is equally neutral.
        let new_gt = sample_from_logits(&logits, &cfg(1.0, logits.len() + 100, 1.0));
        assert_eq!(new_gt, legacy_temperature_only(&logits, 1.0));
    }

    /// Greedy (temperature == 0) ignores top-k/top-p entirely and stays argmax.
    #[test]
    fn greedy_unaffected_by_top_k_top_p() {
        let logits = [0.2_f32, -1.0, 3.3, 1.1];
        let expected = argmax_logits(&logits);
        assert_eq!(sample_from_logits(&logits, &cfg(0.0, 1, 0.1)), expected);
        assert_eq!(sample_from_logits(&logits, &cfg(0.0, 0, 1.0)), expected);
    }
}
