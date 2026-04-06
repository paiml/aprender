
impl PromptCache {
    /// Create new prompt cache
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            max_entries,
        }
    }

    /// Compute hash for token sequence
    fn hash_tokens(tokens: &[usize]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Find longest matching prefix in cache
    pub fn find_prefix(&mut self, tokens: &[usize]) -> Option<(usize, u64)> {
        // Try progressively shorter prefixes
        for len in (1..=tokens.len()).rev() {
            let prefix = &tokens[..len];
            let hash = Self::hash_tokens(prefix);
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.hit_count += 1;
                entry.last_access = std::time::Instant::now();
                return Some((len, entry.kv_hash));
            }
        }
        None
    }

    /// Add entry to cache
    pub fn add(&mut self, tokens: Vec<usize>, kv_hash: u64) {
        // Evict if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let hash = Self::hash_tokens(&tokens);
        self.entries.insert(
            hash,
            PromptCacheEntry {
                tokens,
                kv_hash,
                hit_count: 0,
                last_access: std::time::Instant::now(),
            },
        );
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some((&key, _)) = self.entries.iter().min_by_key(|(_, v)| v.last_access) {
            self.entries.remove(&key);
        }
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> PromptCacheStats {
        let total_hits: usize = self.entries.values().map(|e| e.hit_count).sum();
        PromptCacheStats {
            entries: self.entries.len(),
            total_hits,
            max_entries: self.max_entries,
        }
    }
}

/// Prompt cache statistics
#[derive(Debug, Clone)]
pub struct PromptCacheStats {
    /// Number of entries in cache
    pub entries: usize,
    /// Total cache hits
    pub total_hits: usize,
    /// Maximum cache size
    pub max_entries: usize,
}

/// Beam search state for a single hypothesis
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token sequence generated so far
    pub tokens: Vec<usize>,
    /// Cumulative log probability
    pub score: f32,
    /// Whether this hypothesis has finished (hit EOS)
    pub finished: bool,
}

impl BeamHypothesis {
    /// Create a new hypothesis starting with given tokens
    pub fn new(tokens: Vec<usize>, score: f32) -> Self {
        Self {
            tokens,
            score,
            finished: false,
        }
    }

    /// Extend hypothesis with a new token
    #[must_use]
    pub fn extend(&self, token: usize, log_prob: f32, is_eos: bool) -> Self {
        let mut new_tokens = self.tokens.clone();
        new_tokens.push(token);
        Self {
            tokens: new_tokens,
            score: self.score + log_prob,
            finished: is_eos,
        }
    }

    /// Get length-normalized score
    pub fn normalized_score(&self, length_penalty: f32) -> f32 {
        let len = self.tokens.len() as f32;
        self.score / len.powf(length_penalty)
    }
}

/// Beam search configuration
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams (hypotheses) to keep
    pub num_beams: usize,
    /// Length penalty (>1.0 favors longer sequences, <1.0 favors shorter)
    pub length_penalty: f32,
    /// Early stopping: stop when num_beams hypotheses are finished
    pub early_stopping: bool,
    /// Number of beams to return
    pub num_return: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            num_beams: 4,
            length_penalty: 1.0,
            early_stopping: true,
            num_return: 1,
        }
    }
}

impl BeamSearchConfig {
    /// Create new beam search config
    pub fn new(num_beams: usize) -> Self {
        Self {
            num_beams,
            ..Default::default()
        }
    }

    /// Set length penalty
    #[must_use]
    pub fn with_length_penalty(mut self, penalty: f32) -> Self {
        self.length_penalty = penalty;
        self
    }

    /// Set early stopping
    #[must_use]
    pub fn with_early_stopping(mut self, early: bool) -> Self {
        self.early_stopping = early;
        self
    }

    /// Set number of sequences to return
    #[must_use]
    pub fn with_num_return(mut self, n: usize) -> Self {
        self.num_return = n;
        self
    }
}

/// Beam search state manager
#[derive(Debug, Clone)]
pub struct BeamSearchState {
    /// Current hypotheses
    pub hypotheses: Vec<BeamHypothesis>,
    /// Finished hypotheses
    pub finished: Vec<BeamHypothesis>,
    /// Configuration
    pub config: BeamSearchConfig,
}

impl BeamSearchState {
    /// Create new beam search state
    pub fn new(config: BeamSearchConfig, initial_tokens: Vec<usize>) -> Self {
        let hypotheses = vec![BeamHypothesis::new(initial_tokens, 0.0)];
        Self {
            hypotheses,
            finished: Vec::new(),
            config,
        }
    }

    /// Process a step with log probabilities for each hypothesis
    ///
    /// # Arguments
    ///
    /// * `log_probs_per_hyp` - Log probabilities for each token, for each hypothesis
    /// * `eos_token` - Optional end-of-sequence token ID
    pub fn step(&mut self, log_probs_per_hyp: &[Vec<f32>], eos_token: Option<usize>) {
        let mut candidates: Vec<BeamHypothesis> = Vec::new();

        for (hyp_idx, hyp) in self.hypotheses.iter().enumerate() {
            if hyp.finished {
                candidates.push(hyp.clone());
                continue;
            }

            let log_probs = &log_probs_per_hyp[hyp_idx];

            // Get top-k tokens for this hypothesis (k = num_beams * 2 for safety)
            let mut indexed: Vec<(usize, f32)> = log_probs
                .iter()
                .enumerate()
                .map(|(i, &lp)| (i, lp))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for &(token, log_prob) in indexed.iter().take(self.config.num_beams * 2) {
                let is_eos = eos_token == Some(token);
                let new_hyp = hyp.extend(token, log_prob, is_eos);

                if is_eos {
                    self.finished.push(new_hyp);
                } else {
                    candidates.push(new_hyp);
                }
            }
        }

        // Select top num_beams hypotheses by normalized score
        candidates.sort_by(|a, b| {
            let score_a = a.normalized_score(self.config.length_penalty);
            let score_b = b.normalized_score(self.config.length_penalty);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.hypotheses = candidates.into_iter().take(self.config.num_beams).collect();
    }

    /// Check if search should stop
    pub fn should_stop(&self) -> bool {
        if self.config.early_stopping && self.finished.len() >= self.config.num_beams {
            return true;
        }
        self.hypotheses.is_empty() || self.hypotheses.iter().all(|h| h.finished)
    }

    /// Get best completed hypotheses
    pub fn best_hypotheses(&self) -> Vec<BeamHypothesis> {
        let mut all: Vec<_> = self
            .finished
            .iter()
            .chain(self.hypotheses.iter())
            .cloned()
            .collect();
        all.sort_by(|a, b| {
            let score_a = a.normalized_score(self.config.length_penalty);
            let score_b = b.normalized_score(self.config.length_penalty);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.into_iter().take(self.config.num_return).collect()
    }
}

/// Streaming generation state
#[derive(Debug)]
pub struct StreamingGenerator {
    /// Tokens generated so far
    pub tokens: Vec<usize>,
    /// Generated text so far
    pub text: String,
    /// Whether generation is complete
    pub finished: bool,
    /// Total tokens generated
    pub total_tokens: usize,
}

impl StreamingGenerator {
    /// Create new streaming generator
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            text: String::new(),
            finished: false,
            total_tokens: 0,
        }
    }

    /// Add a generated token
    pub fn add_token(&mut self, token_id: usize, token_text: Option<&str>) {
        self.tokens.push(token_id);
        if let Some(text) = token_text {
            self.text.push_str(text);
        }
        self.total_tokens += 1;
    }

    /// Mark generation as finished
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Get current token count
    pub fn token_count(&self) -> usize {
        self.total_tokens
    }
}

impl Default for StreamingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Extended generation configuration with advanced sampling options
#[derive(Debug, Clone, Default)]
pub struct AdvancedGenerationConfig {
    /// Base generation config
    pub base: GenerationConfig,
    /// Stop sequence detector
    pub stop_detector: Option<StopSequenceDetector>,
    /// Repetition penalty config
    pub repetition_penalty: Option<RepetitionPenaltyConfig>,
    /// Presence/frequency penalties
    pub presence_frequency: Option<PresenceFrequencyPenalty>,
    /// Logit bias
    pub logit_bias: Option<LogitBias>,
}

impl AdvancedGenerationConfig {
    /// Create with base config
    pub fn new(base: GenerationConfig) -> Self {
        Self {
            base,
            ..Default::default()
        }
    }

    /// Add stop sequences
    #[must_use]
    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.stop_detector = Some(StopSequenceDetector::new().with_stop_strings(stops));
        self
    }

    /// Add repetition penalty
    #[must_use]
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(RepetitionPenaltyConfig::new(penalty));
        self
    }

    /// Add presence/frequency penalties
    #[must_use]
    pub fn with_presence_frequency(mut self, presence: f32, frequency: f32) -> Self {
        self.presence_frequency = Some(PresenceFrequencyPenalty::new(presence, frequency));
        self
    }

    /// Add logit bias
    #[must_use]
    pub fn with_logit_bias(mut self, bias: LogitBias) -> Self {
        self.logit_bias = Some(bias);
        self
    }
}

/// Apply all configured penalties and biases to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - Previously generated tokens
/// * `config` - Advanced generation configuration
///
/// # Returns
///
/// Logits with all penalties applied
pub fn apply_all_penalties(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &AdvancedGenerationConfig,
) -> Tensor<f32> {
    let mut result = logits.clone();

    // Apply repetition penalty
    if let Some(ref rep_config) = config.repetition_penalty {
        result = apply_repetition_penalty(&result, context_tokens, rep_config);
    }

    // Apply presence/frequency penalty
    if let Some(ref pf_config) = config.presence_frequency {
        result = apply_presence_frequency_penalty(&result, context_tokens, pf_config);
    }

    // Apply logit bias
    if let Some(ref bias) = config.logit_bias {
        result = apply_logit_bias(&result, bias);
    }

    result
}
