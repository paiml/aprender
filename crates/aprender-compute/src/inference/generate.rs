//! Token sampling and text generation loop.

use crate::error::TruenoError;
use crate::inference::model::{ForwardArena, KvCache, LlamaModel};

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, top_p: 0.9, seed: 42 }
    }
}

/// Simple xorshift64 PRNG (no external dependency).
pub struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f32) / (u64::MAX as f32)
    }
}

/// Sample a token from logits using temperature + top-k + top-p.
pub fn sample_token(logits: &[f32], params: &SampleParams, rng: &mut Rng) -> u32 {
    let vocab_size = logits.len();

    if params.temperature <= 0.0 {
        // Greedy: argmax
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Apply temperature
    let inv_temp = 1.0 / params.temperature;
    let mut scaled: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v * inv_temp)).collect();

    // Top-k: keep only top-k highest logits
    let k = params.top_k.min(vocab_size);
    if k < vocab_size {
        scaled.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scaled.truncate(k);
    }

    // Softmax over remaining candidates
    let max_logit = scaled.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> =
        scaled.iter().map(|&(i, v)| (i, (v - max_logit).exp())).collect();
    let sum: f32 = probs.iter().map(|x| x.1).sum();
    for p in &mut probs {
        p.1 /= sum;
    }

    // Top-p (nucleus): accumulate until cumulative prob >= top_p
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, prob)) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= params.top_p {
            cutoff = i + 1;
            break;
        }
    }
    probs.truncate(cutoff);

    // Renormalize
    let sum2: f32 = probs.iter().map(|x| x.1).sum();
    for p in &mut probs {
        p.1 /= sum2;
    }

    // Sample from distribution
    let r = rng.next_f32();
    let mut cum = 0.0;
    for &(idx, prob) in &probs {
        cum += prob;
        if r < cum {
            return idx as u32;
        }
    }
    probs.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

/// Generate text tokens autoregressively.
///
/// Takes initial prompt token IDs and generates up to `max_tokens` more.
/// Returns the generated token IDs (not including prompt).
pub fn generate(
    model: &LlamaModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    params: &SampleParams,
    eos_token: u32,
) -> Result<Vec<u32>, TruenoError> {
    let mut kv_cache = KvCache::new(&model.config);
    let mut arena = ForwardArena::new(&model.config);
    let mut rng = Rng::new(params.seed);
    let mut generated = Vec::with_capacity(max_tokens);

    // Prefill: process prompt tokens
    let mut last_logits = Vec::new();
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        last_logits = model.forward(token_id, pos, &mut kv_cache, &mut arena)?;
    }

    if last_logits.is_empty() {
        return Err(TruenoError::InvalidInput("Empty prompt".into()));
    }

    // Decode: generate tokens one at a time
    for pos in (prompt_tokens.len()..).take(max_tokens) {
        let token = sample_token(&last_logits, params, &mut rng);

        if token == eos_token {
            break;
        }
        if pos >= model.config.max_seq_len - 1 {
            break;
        }

        generated.push(token);
        last_logits = model.forward(token, pos, &mut kv_cache, &mut arena)?;
    }

    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let params = SampleParams { temperature: 0.0, ..Default::default() };
        let mut rng = Rng::new(42);
        assert_eq!(sample_token(&logits, &params, &mut rng), 3); // argmax
    }

    #[test]
    fn test_temperature_sampling() {
        let logits = vec![1.0, 2.0, 3.0];
        let params = SampleParams { temperature: 1.0, top_k: 3, top_p: 1.0, seed: 42 };
        let mut rng = Rng::new(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert!(token < 3);
    }

    #[test]
    fn test_top_k_reduces_candidates() {
        let mut logits = vec![0.0f32; 100];
        logits[50] = 10.0;
        logits[51] = 9.0;
        let params = SampleParams { temperature: 1.0, top_k: 2, top_p: 1.0, seed: 42 };
        let mut rng = Rng::new(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert!(token == 50 || token == 51);
    }
}
