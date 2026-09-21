//! The one token sampler every format shares (#3760).
//!
//! The GGUF decode loops drew from the temperature-scaled, top-k/top-p-filtered
//! distribution. `AprTransformer`'s sampler took the ARGMAX of those survivors
//! instead, which is the greedy token for every parameter value, and the
//! SafeTensors `apr run` loop never looked at the sampling parameters at all. So
//! `--temperature`, `--top-k`, `--top-p` and `--seed` did nothing on those formats.
//! Every sampled path now draws through [`draw`], so the formats cannot drift again.
//!
//! Greedy stays each loop's own argmax: the loops break ties differently, and
//! greedy output must stay byte-identical. [`is_greedy`] is the shared predicate
//! that picks between the two.

use rand::rngs::StdRng;
use rand::Rng;

/// The seed a sampled generation uses when its caller names none (#3760).
///
/// Only a sampled step reads it; greedy decoding never touches the RNG.
pub const DEFAULT_SEED: u64 = 42;

/// Whether a generation picks greedily: temperature 0, or a top-k of exactly 1.
///
/// `top_k == 0` means the filter is OFF (llama.cpp, Ollama), which samples.
#[must_use]
pub fn is_greedy(temperature: f32, top_k: usize) -> bool {
    temperature == 0.0 || top_k == 1
}

/// One draw from the seeded RNG: the draw every sampled decode step makes.
///
/// The same `(logits, temperature, top_k, top_p)` and the same RNG state give
/// the same token, so a seed fully determines a sampled generation.
pub fn draw_seeded(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut StdRng,
) -> u32 {
    let r: f32 = rng.random();
    draw(logits, temperature, top_k, top_p, r)
}

/// Draw a token from the temperature-scaled, top-k/top-p-filtered categorical
/// distribution, using the caller's uniform sample `r ∈ [0, 1)`.
///
/// Pure: `r` is the only randomness, so a seeded RNG fully determines the token.
/// `top_k == 0` disables the top-k filter; `top_p` outside `(0, 1)` disables the
/// nucleus filter (a bit-exact no-op at the default 1.0).
#[must_use]
pub fn draw(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, r: f32) -> u32 {
    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Get top-k indices
    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    // `top_k == 0` means "disabled" in llama.cpp and Ollama, and both the
    // Ollama-compat and OpenAI-compat surfaces pass it straight through. An
    // unguarded `truncate(0)` empties `indexed`, so `probs` is empty and the
    // inverse-CDF loop below falls through to `probs.last().map_or(0, ..)` —
    // returning token 0 on EVERY step (`!!!!!!`). Guard exactly as the live
    // MoE sampler does (infer/qwen3_moe_generate.rs:104).
    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }

    // Top-p (nucleus): keep the smallest prefix whose cumulative softmax mass
    // reaches `top_p`. Ported from the live MoE sampler
    // (infer/qwen3_moe_generate.rs:109), which is where the only working
    // implementation lived — the dense path accepted `top_p` in
    // QuantizedGenerateConfig and then silently discarded it, so
    // `--top-p 0.001` was byte-identical to `--top-p 1.0` on every
    // /v1/chat/completions, /api/chat, /api/generate and `apr run` request.
    //
    // The `top_p > 0.0 && top_p < 1.0` guard is what keeps this a BIT-EXACT
    // no-op at the default 1.0: the branch is not entered at all, so the
    // candidate set and the inverse-CDF draw below are unchanged. That
    // matters because the default config sets top_p = 1.0 (runtime.rs:51) —
    // every existing caller must keep its current output token-for-token.
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

    // Softmax over the filtered set
    let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
        .collect();

    // Inverse-CDF draw from the categorical distribution
    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if cumulative >= r {
            return idx as u32;
        }
    }

    probs.last().map_or(0, |(idx, _)| *idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn is_greedy_case_table() {
        assert!(is_greedy(0.0, 0));
        assert!(is_greedy(0.0, 40));
        assert!(is_greedy(0.8, 1));
        assert!(
            !is_greedy(0.8, 0),
            "top_k 0 disables the filter and samples"
        );
        assert!(!is_greedy(0.8, 40));
    }

    /// `r` walks the CDF: the first survivor at r = 0, the last near 1.
    #[test]
    fn draw_walks_the_cdf_of_the_filtered_distribution() {
        let logits = [0.0_f32, 3.0, 1.0, 2.0];
        assert_eq!(
            draw(&logits, 1.0, 0, 1.0, 0.0),
            1,
            "r = 0 takes the top logit"
        );
        assert_eq!(
            draw(&logits, 1.0, 0, 1.0, 0.999_999),
            0,
            "r -> 1 reaches the tail"
        );
        assert_eq!(
            draw(&logits, 1.0, 2, 1.0, 0.999_999),
            3,
            "top_k 2 cuts the tail"
        );
    }

    #[test]
    fn draw_seeded_is_reproducible_and_seed_sensitive() {
        let logits = [1.0_f32, 0.9, 1.1, 0.95, 1.05];
        let run = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..32)
                .map(|_| draw_seeded(&logits, 1.0, 0, 1.0, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }
}
