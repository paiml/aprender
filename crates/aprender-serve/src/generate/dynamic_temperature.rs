
// ============================================================================
// Dynamic Temperature (temp_ext) - Entropy-based temperature adjustment
// ============================================================================

/// Configuration for dynamic temperature (temp_ext)
///
/// Adjusts temperature based on the entropy of the probability distribution.
/// When entropy is low (confident), uses higher temperature to increase diversity.
/// When entropy is high (uncertain), uses lower temperature to focus on likely tokens.
///
/// Reference: llama.cpp `llama_sampler_init_temp_ext`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynTempConfig {
    /// Base temperature
    pub temp: f32,
    /// Range around base temperature (min = temp - delta, max = temp + delta)
    pub delta: f32,
    /// Exponent for entropy mapping (higher = more aggressive adjustment)
    pub exponent: f32,
}

impl Default for DynTempConfig {
    fn default() -> Self {
        Self {
            temp: 1.0,
            delta: 0.0,
            exponent: 1.0,
        }
    }
}

impl DynTempConfig {
    /// Create a new dynamic temperature config
    pub fn new(temp: f32, delta: f32, exponent: f32) -> Self {
        Self {
            temp,
            delta,
            exponent,
        }
    }

    /// Create with just temperature (no dynamic adjustment)
    pub fn static_temp(temp: f32) -> Self {
        Self {
            temp,
            delta: 0.0,
            exponent: 1.0,
        }
    }
}

/// Apply dynamic temperature based on entropy
///
/// The algorithm:
/// 1. Calculate max possible entropy: -log(1/n)
/// 2. Calculate actual entropy: -sum(p * log(p))
/// 3. Normalize entropy to [0, 1]
/// 4. Map to temperature: min_temp + (max_temp - min_temp) * pow(norm_entropy, exponent)
/// 5. Apply calculated temperature to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `config` - Dynamic temperature configuration
///
/// # Returns
///
/// Logits with dynamic temperature applied
pub fn apply_dynamic_temperature(logits: &Tensor<f32>, config: &DynTempConfig) -> Tensor<f32> {
    // If no delta, just apply static temperature
    if config.delta <= 0.0 {
        return apply_temperature(logits, config.temp).unwrap_or_else(|e| {
            eprintln!("[WARN] Temperature application failed ({e}), using raw logits");
            logits.clone()
        });
    }

    let data = logits.data();
    if data.len() <= 1 {
        return logits.clone();
    }

    // Calculate softmax probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = data.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = data
        .iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect();

    // Calculate maximum possible entropy: -log(1/n) = log(n)
    let max_entropy = (data.len() as f32).ln();

    // Calculate actual entropy: -sum(p * log(p))
    let entropy: f32 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Normalize entropy to [0, 1]
    let normalized_entropy = if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Calculate dynamic temperature
    let min_temp = (config.temp - config.delta).max(0.0);
    let max_temp = config.temp + config.delta;
    let dyn_temp = min_temp + (max_temp - min_temp) * normalized_entropy.powf(config.exponent);

    // Apply calculated temperature
    apply_temperature(logits, dyn_temp).unwrap_or_else(|e| {
        eprintln!("[WARN] Dynamic temperature application failed ({e}), using raw logits");
        logits.clone()
    })
}

// ============================================================================
// Infill/FIM Sampler - Fill-in-the-Middle for code completion
// ============================================================================

/// Configuration for infill/FIM (Fill-in-the-Middle) sampling
///
/// Used for code completion where the model generates text to fill a gap.
/// Handles EOG (End-of-Generation) tokens specially to determine when to stop.
///
/// Reference: llama.cpp `llama_sampler_init_infill`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfillConfig {
    /// EOG (End-of-Generation) token IDs
    pub eog_tokens: Vec<usize>,
    /// Ratio threshold: if 3*p_eog*n > p_txt, force EOG
    pub eog_ratio_threshold: f32,
}

impl Default for InfillConfig {
    fn default() -> Self {
        Self {
            eog_tokens: vec![],
            eog_ratio_threshold: 3.0,
        }
    }
}

impl InfillConfig {
    /// Create a new infill config with EOG tokens
    pub fn new(eog_tokens: Vec<usize>) -> Self {
        Self {
            eog_tokens,
            eog_ratio_threshold: 3.0,
        }
    }

    /// Set the EOG ratio threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.eog_ratio_threshold = threshold;
        self
    }
}

/// Result of infill sampling
#[derive(Debug, Clone)]
pub struct InfillResult {
    /// Modified logits (with non-EOG tokens potentially zeroed)
    pub logits: Tensor<f32>,
    /// Whether to force EOG token
    pub force_eog: bool,
    /// Probability sum of text tokens
    pub p_txt: f32,
    /// Probability sum of EOG tokens
    pub p_eog: f32,
}

/// Apply infill sampling logic
///
/// This determines if the model should stop generating (emit EOG) based on
/// the relative probabilities of EOG vs text tokens.
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `config` - Infill configuration
///
/// Compute p_eog and p_txt from probability distribution
fn compute_eog_txt_probs(probs: &[f32], eog_tokens: &[usize]) -> (f32, f32) {
    let mut p_eog: f32 = 0.0;
    let mut p_txt: f32 = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        if eog_tokens.contains(&i) {
            p_eog += p;
        } else {
            p_txt += p;
        }
    }
    (p_eog, p_txt)
}

/// Create logits with only EOG tokens, renormalized
fn create_eog_only_logits(
    data: &[f32],
    probs: &[f32],
    eog_tokens: &[usize],
    shape: &[usize],
) -> Tensor<f32> {
    let mut new_data = vec![f32::NEG_INFINITY; data.len()];
    let mut eog_sum = 0.0;

    for &eog_id in eog_tokens {
        if eog_id < data.len() {
            new_data[eog_id] = data[eog_id];
            eog_sum += probs[eog_id];
        }
    }

    if eog_sum > 0.0 {
        for &eog_id in eog_tokens {
            if eog_id < data.len() && new_data[eog_id] > f32::NEG_INFINITY {
                let normalized_p = probs[eog_id] / eog_sum;
                new_data[eog_id] = normalized_p.ln();
            }
        }
    }

    // Shape and data length match by construction (new_data.len() == data.len())
    Tensor::from_vec(shape.to_vec(), new_data)
        .expect("BUG: EOG logits shape/data mismatch (same shape as input tensor)")
}

/// # Returns
///
/// `InfillResult` with modified logits and EOG decision
pub fn apply_infill_sampling(logits: &Tensor<f32>, config: &InfillConfig) -> InfillResult {
    let data = logits.data();
    if data.is_empty() || config.eog_tokens.is_empty() {
        return InfillResult {
            logits: logits.clone(),
            force_eog: false,
            p_txt: 1.0,
            p_eog: 0.0,
        };
    }

    // Calculate softmax probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = data.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = data
        .iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect();

    let (p_eog, p_txt) = compute_eog_txt_probs(&probs, &config.eog_tokens);

    // Check if we should force EOG: 3 * p_eog * n > p_txt
    let n = data.len() as f32;
    let force_eog = config.eog_ratio_threshold * p_eog * n > p_txt;

    if force_eog {
        InfillResult {
            logits: create_eog_only_logits(data, &probs, &config.eog_tokens, logits.shape()),
            force_eog: true,
            p_txt,
            p_eog,
        }
    } else {
        InfillResult {
            logits: logits.clone(),
            force_eog: false,
            p_txt,
            p_eog,
        }
    }
}

// ============================================================================
// Sampler Chain - Composable sampler pipeline
// ============================================================================

/// Trait for samplers that can be chained together
pub trait Sampler: Send + Sync {
    /// Get the sampler name
    fn name(&self) -> &'static str;

    /// Apply the sampler to logits (in-place modification)
    fn apply(&self, logits: &mut Tensor<f32>, context: &SamplerContext);

    /// Clone the sampler (for use in chains)
    fn clone_box(&self) -> Box<dyn Sampler>;
}

/// Context passed to samplers during application
#[derive(Debug, Clone, Default)]
pub struct SamplerContext {
    /// Previously generated tokens
    pub tokens: Vec<usize>,
    /// Random value for stochastic samplers [0, 1)
    pub rng_value: f32,
    /// Current generation step
    pub step: usize,
}

impl SamplerContext {
    /// Create a new sampler context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tokens
    #[must_use]
    pub fn with_tokens(mut self, tokens: Vec<usize>) -> Self {
        self.tokens = tokens;
        self
    }

    /// Set RNG value
    #[must_use]
    pub fn with_rng(mut self, rng_value: f32) -> Self {
        self.rng_value = rng_value;
        self
    }

    /// Set step
    #[must_use]
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }
}

/// A chain of samplers applied in sequence
pub struct SamplerChain {
    samplers: Vec<Box<dyn Sampler>>,
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplerChain {
    /// Create a new empty sampler chain
    pub fn new() -> Self {
        Self { samplers: vec![] }
    }

    /// Add a sampler to the chain (builder pattern)
    #[must_use]
    pub fn with_sampler<S: Sampler + 'static>(mut self, sampler: S) -> Self {
        self.samplers.push(Box::new(sampler));
        self
    }

    /// Push a boxed sampler to the chain
    pub fn push(&mut self, sampler: Box<dyn Sampler>) {
        self.samplers.push(sampler);
    }

    /// Get the number of samplers in the chain
    pub fn len(&self) -> usize {
        self.samplers.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.samplers.is_empty()
    }

    /// Get sampler names in order
    pub fn names(&self) -> Vec<&'static str> {
        self.samplers.iter().map(|s| s.name()).collect()
    }

    /// Apply all samplers in sequence
    pub fn apply(&self, logits: &mut Tensor<f32>, context: &SamplerContext) {
        for sampler in &self.samplers {
            sampler.apply(logits, context);
        }
    }

    /// Sample a token after applying all samplers
    ///
    /// # Errors
    ///
    /// Returns error if sampling fails
    pub fn sample(&self, logits: &Tensor<f32>, context: &SamplerContext) -> Result<usize> {
        contract_pre_sample!();
        let mut modified = logits.clone();
        self.apply(&mut modified, context);
        sample_greedy(&modified)
    }
}

impl Clone for SamplerChain {
    fn clone(&self) -> Self {
        Self {
            samplers: self.samplers.iter().map(|s| s.clone_box()).collect(),
        }
    }
}

// Concrete sampler implementations for the chain

/// Temperature sampler
#[derive(Debug, Clone)]
pub struct TemperatureSampler {
    /// Temperature value (1.0 = no change)
    pub temp: f32,
}

impl TemperatureSampler {
    /// Create a new temperature sampler
    pub fn new(temp: f32) -> Self {
        Self { temp }
    }
}

impl Sampler for TemperatureSampler {
    fn name(&self) -> &'static str {
        "temperature"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        if let Ok(result) = apply_temperature(logits, self.temp) {
            *logits = result;
        }
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Dynamic temperature sampler
#[derive(Debug, Clone)]
pub struct DynTempSampler {
    /// Dynamic temperature configuration
    pub config: DynTempConfig,
}

impl DynTempSampler {
    /// Create a new dynamic temperature sampler
    pub fn new(config: DynTempConfig) -> Self {
        Self { config }
    }
}
