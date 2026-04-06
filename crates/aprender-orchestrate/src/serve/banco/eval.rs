//! Model evaluation — perplexity and benchmarks.
//!
//! Uses the existing inference engine to compute perplexity on text samples.
//! Perplexity measures how well the model predicts the next token.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Eval run result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub eval_id: String,
    pub model: String,
    pub metric: String,
    pub value: f64,
    pub tokens_evaluated: usize,
    pub duration_secs: f64,
    pub status: EvalStatus,
}

/// Eval status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalStatus {
    Running,
    Complete,
    Failed,
    NoModel,
}

/// Eval store — tracks evaluation runs.
pub struct EvalStore {
    runs: RwLock<HashMap<String, EvalResult>>,
    counter: std::sync::atomic::AtomicU64,
}

impl EvalStore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            runs: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Record an eval result.
    pub fn record(&self, result: EvalResult) {
        if let Ok(mut store) = self.runs.write() {
            store.insert(result.eval_id.clone(), result);
        }
    }

    /// List all eval runs (most recent first).
    #[must_use]
    pub fn list(&self) -> Vec<EvalResult> {
        let store = self.runs.read().unwrap_or_else(|e| e.into_inner());
        let mut runs: Vec<EvalResult> = store.values().cloned().collect();
        runs.sort_by(|a, b| b.eval_id.cmp(&a.eval_id));
        runs
    }

    /// Get an eval run by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<EvalResult> {
        self.runs.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// Generate a unique eval ID.
    pub fn next_id(&self) -> String {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("eval-{}-{seq}", epoch_secs())
    }
}

/// Compute perplexity on a text sample using the inference engine.
///
/// PPL = exp(-1/N * Σ log P(token_i | context))
///
/// Requires a loaded model with inference feature. Returns None without.
/// Accepts pre-tokenized IDs so the caller can use proper BPE encoding.
#[cfg(feature = "realizar")]
pub fn compute_perplexity(
    model: &Arc<realizar::gguf::OwnedQuantizedModel>,
    token_ids: &[u32],
    max_tokens: usize,
) -> Option<(f64, usize)> {
    if token_ids.len() < 2 {
        return None;
    }

    let config = model.config();
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let eval_len = token_ids.len().min(max_tokens);

    let mut cache =
        realizar::gguf::OwnedQuantizedKVCache::new(config.num_layers, kv_dim, eval_len + 1);

    let mut total_log_prob = 0.0f64;
    let mut count = 0usize;

    for pos in 0..eval_len - 1 {
        let logits = model.forward_single_with_cache(token_ids[pos], &mut cache, pos).ok()?;

        // Softmax to get probabilities
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let next_token = token_ids[pos + 1] as usize;

        if next_token < logits.len() {
            let log_prob = (logits[next_token] - max_logit) as f64 - (exp_sum as f64).ln();
            total_log_prob += log_prob;
            count += 1;
        }
    }

    if count == 0 {
        return None;
    }

    let ppl = (-total_log_prob / count as f64).exp();
    Some((ppl, count))
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
