//! Classification head and corpus loader for fine-tuning
//!
//! Provides a classifier head that attaches to a transformer's hidden states
//! (mean pooling → linear) and a JSONL corpus loader for safety labels.
//!
//! # Contract
//!
//! See `aprender/contracts/classification-finetune-v1.yaml`:
//! - F-CLASS-001: Logit shape == num_classes
//! - F-CLASS-002: Label index < num_classes
//! - F-CLASS-004: Weight shape == hidden_size * num_classes
//!
//! # Architecture
//!
//! ```text
//! hidden_states [seq_len, hidden_size]
//!   → mean pool → [hidden_size]
//!   → linear    → [num_classes]
//!   → softmax   → class probabilities
//! ```

use crate::autograd::{matmul, Tensor};
use crate::transformer::ModelArchitecture;
use serde::Deserialize;
use std::path::Path;

/// Pooling strategy for extracting a fixed-size vector from sequence hidden states.
///
/// Decoders use last-token pooling (autoregressive — last position sees all context).
/// Encoders use CLS pooling (bidirectional — first [CLS] token sees all context).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean pool across all positions (default, architecture-agnostic)
    Mean,
    /// Use last token's hidden state (decoder convention)
    LastToken,
    /// Use first token's [CLS] hidden state (encoder convention: BERT/RoBERTa)
    Cls,
}

impl PoolingStrategy {
    /// Select pooling strategy based on model architecture.
    pub fn from_architecture(arch: ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::Encoder => Self::Cls,
            ModelArchitecture::Decoder => Self::Mean, // keep backward compat
        }
    }
}

/// Classification head: mean pool + linear projection.
///
/// Maps transformer hidden states to class logits.
/// Weight shape: [hidden_size * num_classes] (flattened row-major).
/// Bias shape: [num_classes].
pub struct ClassificationHead {
    /// Linear weight [hidden_size, num_classes] flattened
    pub weight: Tensor,
    /// Bias [num_classes]
    pub bias: Tensor,
    /// Input dimension (model hidden_size)
    hidden_size: usize,
    /// Output dimension (number of classes)
    num_classes: usize,
}

impl ClassificationHead {
    /// Create a new classification head with Xavier-initialized weights.
    ///
    /// # Arguments
    /// * `hidden_size` - Transformer hidden dimension (e.g., 896 for Qwen2-0.5B)
    /// * `num_classes` - Number of output classes (e.g., 5 for shell safety)
    ///
    /// # Contract (F-CLASS-004)
    /// Validates hidden_size > 0 and num_classes >= 2.
    pub fn new(hidden_size: usize, num_classes: usize) -> Self {
        assert!(hidden_size > 0, "F-CLASS-004: hidden_size must be > 0");
        assert!(num_classes >= 2, "F-CLASS-004: num_classes must be >= 2");

        // Xavier uniform initialization: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let scale = (6.0 / (hidden_size + num_classes) as f32).sqrt();
        let mut rng_state: u64 = 42;
        let weight_data: Vec<f32> = (0..hidden_size * num_classes)
            .map(|_| {
                // Simple LCG for deterministic init
                rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect();

        let weight = Tensor::from_vec(weight_data, true);
        let bias = Tensor::zeros(num_classes, true);

        Self { weight, bias, hidden_size, num_classes }
    }

    /// Forward pass: hidden_states → mean pool → linear → logits.
    ///
    /// # Arguments
    /// * `hidden_states` - Transformer output [seq_len * hidden_size] flattened
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Logits tensor [num_classes]
    ///
    /// # Contract (F-CLASS-001)
    /// Output always has exactly num_classes elements.
    pub fn forward(&self, hidden_states: &Tensor, seq_len: usize) -> Tensor {
        // Mean pool across sequence dimension
        let pooled = self.mean_pool(hidden_states, seq_len);

        // Linear: pooled [1, hidden_size] @ weight [hidden_size, num_classes] = [1, num_classes]
        let logits = matmul(&pooled, &self.weight, 1, self.hidden_size, self.num_classes);

        // Add bias
        let logits_data: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits data")
            .iter()
            .zip(self.bias.data().as_slice().expect("contiguous bias data").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        Tensor::from_vec(logits_data, logits.requires_grad())
    }

    /// Mean pool hidden states across sequence length.
    ///
    /// hidden_states: [seq_len * hidden_size] → [hidden_size]
    pub fn mean_pool(&self, hidden_states: &Tensor, seq_len: usize) -> Tensor {
        let data = hidden_states.data();
        let slice = data.as_slice().expect("contiguous hidden states");
        let h = self.hidden_size;

        let mut pooled = vec![0.0f32; h];
        for pos in 0..seq_len {
            let start = pos * h;
            for j in 0..h {
                pooled[j] += slice[start + j];
            }
        }
        let inv_len = 1.0 / seq_len as f32;
        for v in &mut pooled {
            *v *= inv_len;
        }

        Tensor::from_vec(pooled, hidden_states.requires_grad())
    }

    /// CLS pooling: extract the first token's hidden state.
    ///
    /// In BERT-family models, position 0 is the [CLS] token which attends
    /// bidirectionally to all other tokens, making it suitable for classification.
    ///
    /// # Contract (ENC-007)
    /// Output has exactly hidden_size elements (first position of input).
    pub fn cls_pool(&self, hidden_states: &Tensor) -> Tensor {
        let data = hidden_states.data();
        let slice = data.as_slice().expect("contiguous hidden states");
        let h = self.hidden_size;
        Tensor::from_vec(slice[..h].to_vec(), hidden_states.requires_grad())
    }

    /// Last-token pooling: extract the last token's hidden state.
    ///
    /// In decoder models, the last token has seen all prior context through
    /// causal attention, making it suitable for classification.
    pub fn last_token_pool(&self, hidden_states: &Tensor, seq_len: usize) -> Tensor {
        let data = hidden_states.data();
        let slice = data.as_slice().expect("contiguous hidden states");
        let h = self.hidden_size;
        let start = (seq_len - 1) * h;
        Tensor::from_vec(slice[start..start + h].to_vec(), hidden_states.requires_grad())
    }

    /// Pool hidden states using the specified strategy.
    pub fn pool(
        &self,
        hidden_states: &Tensor,
        seq_len: usize,
        strategy: PoolingStrategy,
    ) -> Tensor {
        match strategy {
            PoolingStrategy::Mean => self.mean_pool(hidden_states, seq_len),
            PoolingStrategy::Cls => self.cls_pool(hidden_states),
            PoolingStrategy::LastToken => self.last_token_pool(hidden_states, seq_len),
        }
    }

    /// Forward pass with configurable pooling strategy.
    pub fn forward_with_pooling(
        &self,
        hidden_states: &Tensor,
        seq_len: usize,
        strategy: PoolingStrategy,
    ) -> Tensor {
        let pooled = self.pool(hidden_states, seq_len, strategy);

        let logits = matmul(&pooled, &self.weight, 1, self.hidden_size, self.num_classes);

        let logits_data: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits data")
            .iter()
            .zip(self.bias.data().as_slice().expect("contiguous bias data").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        Tensor::from_vec(logits_data, logits.requires_grad())
    }

    /// Get trainable parameters (weight + bias).
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    /// Get parameters (immutable).
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    /// Number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Total trainable parameter count.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.hidden_size * self.num_classes + self.num_classes
    }
}

// =============================================================================
// CORPUS LOADER
// =============================================================================

/// A single shell safety corpus sample (single-label).
#[derive(Debug, Clone, Deserialize)]
pub struct SafetySample {
    /// Shell script content
    pub input: String,
    /// Safety class index (0-4)
    pub label: usize,
}

impl SafetySample {
    /// Convert the input text to token IDs using byte-level encoding.
    ///
    /// Each byte of the UTF-8 representation is mapped to a `u32` token ID.
    /// This provides a simple, deterministic tokenization suitable for the
    /// classification pipeline. For production use with large vocabularies,
    /// an external tokenizer (BPE, SentencePiece) should be used before
    /// calling `train_step` directly.
    #[must_use]
    pub fn input_ids(&self) -> Vec<u32> {
        self.input.bytes().map(u32::from).collect()
    }
}

/// A pre-tokenized training sample (KAIZEN-028).
///
/// Token IDs are computed once at dataset construction time and reused
/// across all epochs, eliminating redundant BPE encoding.
///
/// # Contract (C-PRETOK-001)
///
/// - **Precondition**: `token_ids` produced by the same tokenizer used for inference
/// - **Postcondition**: `token_ids.len() <= max_seq_len` (truncated at construction)
/// - **Invariant**: `token_ids` is non-empty (at least one token)
#[derive(Debug, Clone)]
pub struct TokenizedSample {
    /// Pre-computed token IDs (truncated to max_seq_len)
    pub token_ids: Vec<u32>,
    /// Safety class index
    pub label: usize,
}

/// A multi-label shell safety corpus sample.
///
/// A script can have multiple active labels (e.g., both non-deterministic AND needs-quoting).
/// Labels are a multi-hot vector: `[0.0, 1.0, 1.0, 0.0, 0.0]` means classes 1 and 2 are active.
#[derive(Debug, Clone, Deserialize)]
pub struct MultiLabelSafetySample {
    /// Shell script content
    pub input: String,
    /// Multi-hot label vector (length == num_classes)
    pub labels: Vec<f32>,
}

impl MultiLabelSafetySample {
    /// Create from a single-label sample by converting to multi-hot encoding.
    pub fn from_single_label(sample: &SafetySample, num_classes: usize) -> Self {
        let mut labels = vec![0.0f32; num_classes];
        if sample.label < num_classes {
            labels[sample.label] = 1.0;
        }
        Self { input: sample.input.clone(), labels }
    }

    /// Active class indices (where labels[i] > 0.5).
    pub fn active_classes(&self) -> Vec<usize> {
        self.labels.iter().enumerate().filter(|(_, &v)| v > 0.5).map(|(i, _)| i).collect()
    }
}

/// Corpus statistics.
#[derive(Debug, Clone)]
pub struct SafetyCorpusStats {
    /// Total samples
    pub total: usize,
    /// Samples per class
    pub class_counts: Vec<usize>,
    /// Average input length (chars)
    pub avg_input_len: usize,
}

/// Load shell safety corpus from JSONL file.
///
/// Each line is `{"input": "...", "label": N}` where N is 0-4.
///
/// # Contract (F-CLASS-002)
/// All labels must be < num_classes.
///
/// # Errors
/// Returns error if file cannot be read or contains invalid labels.
pub fn load_safety_corpus(path: &Path, num_classes: usize) -> crate::Result<Vec<SafetySample>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| crate::Error::Io(format!("Corpus file not found: {}: {e}", path.display())))?;

    let mut samples = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let sample: SafetySample = serde_json::from_str(line).map_err(|e| {
            crate::Error::ConfigError(format!("Invalid JSONL at line {}: {e}", line_num + 1))
        })?;

        // F-CLASS-002: label bounds check
        if sample.label >= num_classes {
            return Err(crate::Error::ConfigError(format!(
                "F-CLASS-002: label {} at line {} out of range (num_classes={num_classes})",
                sample.label,
                line_num + 1,
            )));
        }

        samples.push(sample);
    }

    Ok(samples)
}

/// Compute corpus statistics.
pub fn corpus_stats(samples: &[SafetySample], num_classes: usize) -> SafetyCorpusStats {
    let mut class_counts = vec![0usize; num_classes];
    let mut total_len = 0usize;

    for s in samples {
        if s.label < num_classes {
            class_counts[s.label] += 1;
        }
        total_len += s.input.len();
    }

    SafetyCorpusStats {
        total: samples.len(),
        class_counts,
        avg_input_len: if samples.is_empty() { 0 } else { total_len / samples.len() },
    }
}

/// Load multi-label corpus from JSONL file.
///
/// Supports two formats:
/// - Single-label: `{"input": "...", "label": N}` → converts to multi-hot
/// - Multi-label: `{"input": "...", "labels": [0.0, 1.0, 1.0, 0.0, 0.0]}`
///
/// # Errors
/// Returns error if file cannot be read or labels are invalid.
pub fn load_multi_label_corpus(
    path: &Path,
    num_classes: usize,
) -> crate::Result<Vec<MultiLabelSafetySample>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| crate::Error::Io(format!("Corpus file not found: {}: {e}", path.display())))?;

    let mut samples = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        samples.push(parse_multi_label_line(line, line_num, num_classes)?);
    }

    Ok(samples)
}

/// Parse a single JSONL line as multi-label or single-label sample.
fn parse_multi_label_line(
    line: &str,
    line_num: usize,
    num_classes: usize,
) -> crate::Result<MultiLabelSafetySample> {
    // Try multi-label format first
    if let Ok(sample) = serde_json::from_str::<MultiLabelSafetySample>(line) {
        if sample.labels.len() != num_classes {
            return Err(crate::Error::ConfigError(format!(
                "F-CLASS-001: labels length {} at line {} != num_classes {num_classes}",
                sample.labels.len(),
                line_num + 1,
            )));
        }
        return Ok(sample);
    }

    if let Ok(single) = serde_json::from_str::<SafetySample>(line) {
        if single.label >= num_classes {
            return Err(crate::Error::ConfigError(format!(
                "F-CLASS-002: label {} at line {} out of range (num_classes={num_classes})",
                single.label,
                line_num + 1,
            )));
        }
        return Ok(MultiLabelSafetySample::from_single_label(&single, num_classes));
    }

    Err(crate::Error::ConfigError(format!(
        "Invalid JSONL at line {}: unrecognized format",
        line_num + 1,
    )))
}

/// BCE with logits loss for multi-label classification.
///
/// Per-element: `L_i = max(x_i, 0) - x_i * t_i + log(1 + exp(-|x_i|))`
/// Total: `L = mean(L_i)`
///
/// # Contract (F-CLASS-005)
/// Output is finite (no NaN/Inf).
pub fn bce_with_logits_loss(logits: &Tensor, targets: &[f32], num_classes: usize) -> Tensor {
    let data = logits.data();
    let slice = data.as_slice().expect("contiguous logits");
    assert_eq!(slice.len(), num_classes, "F-CLASS-001: logit shape mismatch");
    assert_eq!(targets.len(), num_classes, "F-CLASS-001: target shape mismatch");

    let total_loss: f32 = slice
        .iter()
        .zip(targets.iter())
        .map(|(&x, &t)| {
            let relu = x.max(0.0);
            relu - x * t + (1.0 + (-x.abs()).exp()).ln()
        })
        .sum::<f32>()
        / num_classes as f32;

    // F-CLASS-005: finite check
    let total_loss = if total_loss.is_finite() { total_loss } else { 100.0 };

    Tensor::from_vec(vec![total_loss], logits.requires_grad())
}

/// Class weight computation strategy for imbalanced datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassWeightStrategy {
    /// All classes weighted equally: w_c = 1.0
    Uniform,
    /// Inverse frequency: w_c = N / (K * n_c)
    InverseFreq,
    /// Square root of inverse frequency: w_c = sqrt(N / (K * n_c))
    SqrtInverse,
}

impl std::str::FromStr for ClassWeightStrategy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "uniform" => Ok(Self::Uniform),
            "inverse_freq" | "inverse" => Ok(Self::InverseFreq),
            "sqrt_inverse" | "sqrt" => Ok(Self::SqrtInverse),
            _ => Err(format!(
                "Unknown class weight strategy: {s}. Use: uniform, inverse_freq, sqrt_inverse"
            )),
        }
    }
}

impl std::fmt::Display for ClassWeightStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform => write!(f, "uniform"),
            Self::InverseFreq => write!(f, "inverse_freq"),
            Self::SqrtInverse => write!(f, "sqrt_inverse"),
        }
    }
}

/// Compute class weights from corpus statistics.
///
/// Weights are normalized so they sum to `num_classes`, preserving
/// the overall loss scale while rebalancing class contributions.
///
/// # Contract (F-TUNE-005)
/// `abs(sum(weights) - num_classes) < 1e-5`
///
/// # Panics
/// Panics if `stats.class_counts.len() != num_classes` or any class has zero samples.
pub fn compute_class_weights(
    stats: &SafetyCorpusStats,
    strategy: ClassWeightStrategy,
    num_classes: usize,
) -> Vec<f32> {
    assert_eq!(
        stats.class_counts.len(),
        num_classes,
        "F-TUNE-005: class_counts.len() != num_classes"
    );

    let n = stats.total as f32;
    let k = num_classes as f32;

    let raw_weights: Vec<f32> = match strategy {
        ClassWeightStrategy::Uniform => vec![1.0; num_classes],
        ClassWeightStrategy::InverseFreq => stats
            .class_counts
            .iter()
            .map(|&count| {
                let count = count.max(1) as f32; // avoid division by zero
                n / (k * count)
            })
            .collect(),
        ClassWeightStrategy::SqrtInverse => stats
            .class_counts
            .iter()
            .map(|&count| {
                let count = count.max(1) as f32;
                (n / (k * count)).sqrt()
            })
            .collect(),
    };

    // Normalize so weights sum to num_classes
    let sum: f32 = raw_weights.iter().sum();
    if sum < 1e-10 {
        return vec![1.0; num_classes];
    }
    let scale = k / sum;
    raw_weights.iter().map(|&w| w * scale).collect()
}

/// Cross-entropy loss for classification.
///
/// # Arguments
/// * `logits` - Raw logits [num_classes]
/// * `target` - Target class index
/// * `num_classes` - Number of classes
///
/// # Returns
/// Scalar loss value (as single-element Tensor)
///
/// # Contract (F-CLASS-005)
/// Output is finite (no NaN/Inf).
pub fn cross_entropy_loss(logits: &Tensor, target: usize, num_classes: usize) -> Tensor {
    let data = logits.data();
    let slice = data.as_slice().expect("contiguous logits");
    assert_eq!(slice.len(), num_classes, "F-CLASS-001: logit shape mismatch");
    assert!(target < num_classes, "F-CLASS-002: label out of range");

    // Numerically stable log-softmax: log(softmax(x_i)) = x_i - max - log(sum(exp(x_j - max)))
    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = slice.iter().map(|&v| (v - max_val).exp()).sum::<f32>().ln() + max_val;
    let loss = -(slice[target] - log_sum_exp);

    // F-CLASS-005: finite check
    let loss = if loss.is_finite() { loss } else { 100.0 };

    Tensor::from_vec(vec![loss], logits.requires_grad())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "classification_tests.rs"]
mod tests;
