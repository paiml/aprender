
/// Apply DRY (Don't Repeat Yourself) penalty to logits
///
/// Penalizes tokens that would extend n-gram repetitions in the context.
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - List of previously generated token IDs
/// * `config` - DRY configuration
///
/// # Returns
///
/// Logits with DRY penalty applied
pub fn apply_dry_penalty(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &DryConfig,
) -> Tensor<f32> {
    if !config.is_enabled() || context_tokens.len() < config.allowed_length {
        return logits.clone();
    }

    let data = logits.data();
    let mut penalized = data.to_vec();

    // Get relevant context window
    let window_start = if context_tokens.len() > config.penalty_last_n {
        context_tokens.len() - config.penalty_last_n
    } else {
        0
    };
    let context = &context_tokens[window_start..];

    // For each possible next token, check if it would extend a repetition
    for (token_id, logit) in penalized.iter_mut().enumerate() {
        let match_len = find_ngram_match_length(context, token_id, config.allowed_length);

        if match_len >= config.allowed_length {
            // Apply exponential penalty based on match length
            let penalty =
                config.multiplier * config.base.powi((match_len - config.allowed_length) as i32);
            *logit -= penalty;
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), penalized)
        .expect("Shape should match original logits")
}

/// Check if suffix matches at position and next token follows
#[inline]
fn check_ngram_match(
    context: &[usize],
    suffix: &[usize],
    start: usize,
    suffix_len: usize,
    next_token: usize,
) -> bool {
    let potential_end = start + suffix_len;
    potential_end < context.len()
        && context[start..potential_end] == *suffix
        && context[potential_end] == next_token
}

/// Find the length of the longest n-gram that would be repeated if we add this token
fn find_ngram_match_length(context: &[usize], next_token: usize, min_len: usize) -> usize {
    if context.len() < min_len {
        return 0;
    }

    let mut max_match = 0;

    // Build the sequence ending with the potential next token
    // Then search for earlier occurrences
    for end_pos in min_len..=context.len() {
        let search_start = context.len() - end_pos;
        let suffix = &context[search_start..];

        // Look for this suffix earlier in the context
        for start in 0..(context.len() - end_pos) {
            if check_ngram_match(context, suffix, start, end_pos, next_token) {
                max_match = max_match.max(end_pos + 1);
            }
        }
    }

    max_match
}

// ===== XTC (Exclude Top Choices) Sampling =====

/// XTC (Exclude Top Choices) sampling configuration
///
/// XTC removes the most likely tokens with some probability, forcing the model
/// to explore alternative completions. This can increase creativity and diversity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XtcConfig {
    /// Probability of excluding top tokens (0.0 = disabled, 1.0 = always exclude)
    pub probability: f32,
    /// Threshold for excluding tokens (tokens with prob >= threshold may be excluded)
    pub threshold: f32,
    /// Minimum number of tokens to keep after exclusion
    pub min_keep: usize,
}

impl Default for XtcConfig {
    fn default() -> Self {
        Self {
            probability: 0.0,
            threshold: 0.5,
            min_keep: 1,
        }
    }
}

impl XtcConfig {
    /// Create new XTC config with specified probability
    pub fn new(probability: f32) -> Self {
        Self {
            probability,
            ..Default::default()
        }
    }

    /// Set threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set minimum tokens to keep
    #[must_use]
    pub fn with_min_keep(mut self, min_keep: usize) -> Self {
        self.min_keep = min_keep;
        self
    }

    /// Check if XTC is enabled
    pub fn is_enabled(&self) -> bool {
        self.probability > 0.0
    }
}

/// Apply XTC (Exclude Top Choices) sampling
///
/// XTC randomly excludes top tokens to increase diversity.
///
/// # Arguments
///
/// * `logits` - Raw logits from the model
/// * `config` - XTC configuration
/// * `rng_value` - Random value [0, 1) for stochastic exclusion decision
///
/// # Returns
///
/// Modified logits with top choices potentially excluded
pub fn apply_xtc(logits: &Tensor<f32>, config: &XtcConfig, rng_value: f32) -> Tensor<f32> {
    if !config.is_enabled() || rng_value >= config.probability {
        return logits.clone();
    }

    let data = logits.data();
    if data.len() <= config.min_keep {
        return logits.clone();
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Find tokens above threshold
    let mut excluded_count = 0;
    let mut modified = data.to_vec();

    // Sort by probability descending to find top tokens
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Exclude top tokens above threshold, respecting min_keep
    for (idx, prob) in &indexed {
        if *prob >= config.threshold && data.len() - excluded_count > config.min_keep {
            modified[*idx] = f32::NEG_INFINITY;
            excluded_count += 1;
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), modified).expect("Shape should match original logits")
}

// ===== Eta Sampling =====

/// Eta Sampling (entropy-based truncation)
///
/// Eta sampling dynamically adjusts the truncation threshold based on the
/// entropy of the probability distribution. Higher entropy = more tokens kept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtaConfig {
    /// Eta parameter (controls sensitivity to entropy)
    pub eta: f32,
    /// Minimum probability to keep (absolute floor)
    pub min_p: f32,
}

impl Default for EtaConfig {
    fn default() -> Self {
        Self {
            eta: 0.3,
            min_p: 0.0001,
        }
    }
}

impl EtaConfig {
    /// Create new Eta config
    pub fn new(eta: f32) -> Self {
        Self {
            eta,
            ..Default::default()
        }
    }

    /// Set minimum probability
    #[must_use]
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Check if eta sampling is enabled
    pub fn is_enabled(&self) -> bool {
        self.eta > 0.0
    }
}

/// Apply Eta sampling
///
/// # Arguments
///
/// * `logits` - Raw logits from the model
/// * `config` - Eta configuration
/// * `rng_value` - Random value [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_eta(logits: &Tensor<f32>, config: &EtaConfig, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(crate::error::RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Compute entropy
    let entropy: f32 = -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    // Compute dynamic threshold: eta * exp(-entropy)
    let threshold = (config.eta * (-entropy).exp()).max(config.min_p);

    // Keep tokens above threshold
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p >= threshold)
        .map(|(i, &p)| (i, p))
        .collect();

    // Ensure at least one token
    if indexed.is_empty() {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        return Ok(max_idx);
    }

    // Sort by probability descending
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Renormalize and sample
    let sum_kept: f32 = indexed.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = indexed.iter().map(|(_, p)| p / sum_kept).collect();
    let indices: Vec<usize> = indexed.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(&normalized, &indices, rng_value))
}

// ===== Token Healing =====

/// Token Healing configuration
///
/// Token healing fixes broken token boundaries by backing up and re-tokenizing
/// when a partial token is detected at the prompt boundary.
#[derive(Debug, Clone, Default)]
pub struct TokenHealingConfig {
    /// Enable token healing
    pub enabled: bool,
    /// Maximum characters to back up
    pub max_backup_chars: usize,
}

impl TokenHealingConfig {
    /// Create new token healing config
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            max_backup_chars: 10,
        }
    }

    /// Set max backup characters
    #[must_use]
    pub fn with_max_backup(mut self, chars: usize) -> Self {
        self.max_backup_chars = chars;
        self
    }
}

/// Token healing result
#[derive(Debug, Clone)]
pub struct TokenHealingResult {
    /// Adjusted prompt tokens (may be shorter than original)
    pub adjusted_tokens: Vec<usize>,
    /// Prefix constraint for first generated token
    pub prefix_constraint: Option<String>,
    /// Number of tokens removed from end
    pub tokens_removed: usize,
}

/// Analyze prompt for token healing
///
/// Detects if the last token is a partial token that should be healed.
/// This is a simplified implementation - full implementation requires tokenizer access.
///
/// # Arguments
///
/// * `prompt_tokens` - Original prompt tokens
/// * `last_token_text` - Text of the last token (if available)
///
/// # Returns
///
/// Token healing result with adjusted tokens
pub fn analyze_token_healing(
    prompt_tokens: &[usize],
    last_token_text: Option<&str>,
) -> TokenHealingResult {
    // Simple heuristic: if last token is a partial word (no space, single char),
    // we might want to heal it
    let should_heal = last_token_text.is_some_and(|text| {
        !text.is_empty()
            && !text.starts_with(' ')
            && text.len() <= 3
            && text.chars().all(char::is_alphanumeric)
    });

    if should_heal && !prompt_tokens.is_empty() {
        TokenHealingResult {
            adjusted_tokens: prompt_tokens[..prompt_tokens.len() - 1].to_vec(),
            prefix_constraint: last_token_text.map(String::from),
            tokens_removed: 1,
        }
    } else {
        TokenHealingResult {
            adjusted_tokens: prompt_tokens.to_vec(),
            prefix_constraint: None,
            tokens_removed: 0,
        }
    }
}

// ===== Classifier-Free Guidance (CFG) =====

/// Classifier-Free Guidance configuration
///
/// CFG improves generation quality by comparing conditional and unconditional
/// logits, amplifying the difference to steer generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgConfig {
    /// Guidance scale (1.0 = no guidance, higher = stronger guidance)
    pub scale: f32,
    /// Negative prompt tokens (for unconditional generation)
    pub negative_prompt_tokens: Vec<usize>,
}

impl Default for CfgConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            negative_prompt_tokens: Vec::new(),
        }
    }
}

impl CfgConfig {
    /// Create new CFG config with specified scale
    pub fn new(scale: f32) -> Self {
        Self {
            scale,
            ..Default::default()
        }
    }

    /// Set negative prompt tokens
    #[must_use]
    pub fn with_negative_prompt(mut self, tokens: Vec<usize>) -> Self {
        self.negative_prompt_tokens = tokens;
        self
    }

    /// Check if CFG is enabled
    pub fn is_enabled(&self) -> bool {
        self.scale > 1.0
    }
}
