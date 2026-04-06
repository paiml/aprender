use super::distillation::softmax_with_temp;
#[allow(clippy::wildcard_imports)]
use super::*;
use crate::autograd::Tensor;
use crate::nn::Module;

impl<E: TransferEncoder> Module for DomainAdapter<E> {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.encode(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.encoder.parameters();
        params.extend(self.discriminator.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.encoder.parameters_mut();
        params.extend(self.discriminator.parameters_mut());
        params
    }
}

/// `LoRA` (Low-Rank Adaptation) configuration.
///
/// `LoRA` freezes pre-trained weights and adds small trainable matrices
/// to specific layers, drastically reducing memory and compute for fine-tuning.
///
/// # Reference
///
/// Hu, E. J., et al. (2021). `LoRA`: Low-Rank Adaptation of Large Language Models.
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Rank of the low-rank matrices (typically 4, 8, or 16)
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f32,
    /// Target module names (e.g., `["q_proj", "v_proj"]`)
    pub target_modules: Vec<String>,
    /// Dropout probability for `LoRA` layers
    pub dropout: f32,
}

impl LoRAConfig {
    /// Create a new `LoRA` configuration.
    ///
    /// # Arguments
    ///
    /// * `rank` - Rank of low-rank matrices (4-64 typical)
    /// * `alpha` - Scaling factor (often same as rank)
    #[must_use]
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            dropout: 0.0,
        }
    }

    /// Set target modules for `LoRA` adaptation.
    #[must_use]
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Set dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Compute the scaling factor.
    #[must_use]
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self::new(8, 8.0)
    }
}

/// `LoRA` adapter weights for a single layer.
///
/// Stores the A and B matrices for low-rank adaptation:
/// W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
#[derive(Debug)]
pub struct LoRAAdapter {
    /// Down-projection matrix A (`input_dim` → rank)
    pub lora_a: Tensor,
    /// Up-projection matrix B (rank → `output_dim`)
    pub lora_b: Tensor,
    /// Configuration
    pub config: LoRAConfig,
}

impl LoRAAdapter {
    /// Create a new `LoRA` adapter for a layer.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input dimension of the layer
    /// * `output_dim` - Output dimension of the layer
    /// * `config` - `LoRA` configuration
    #[must_use]
    pub fn new(input_dim: usize, output_dim: usize, config: LoRAConfig) -> Self {
        // Initialize A with small values (simulating kaiming init), B with zeros
        // This ensures the adapter starts as identity (W' = W + 0)
        // Use small values for A (1/sqrt(input_dim) scale factor)
        let scale = 0.01;
        let a_data: Vec<f32> = (0..config.rank * input_dim)
            .map(|i| {
                // Simple deterministic init that varies by position
                ((i % 7) as f32 - 3.0) * scale
            })
            .collect();
        let lora_a = Tensor::new(&a_data, &[config.rank, input_dim]).requires_grad();
        let lora_b = Tensor::zeros(&[output_dim, config.rank]).requires_grad();

        Self {
            lora_a,
            lora_b,
            config,
        }
    }

    /// Apply the `LoRA` adaptation to a weight matrix.
    ///
    /// Returns W + scaling * (B @ A)
    #[must_use]
    pub fn apply(&self, base_weight: &Tensor) -> Tensor {
        let ba = self.lora_b.matmul(&self.lora_a);
        let scaled = ba.mul_scalar(self.config.scaling());
        base_weight.add(&scaled)
    }

    /// Get the delta weight (B @ A * scaling).
    #[must_use]
    pub fn delta_weight(&self) -> Tensor {
        self.lora_b
            .matmul(&self.lora_a)
            .mul_scalar(self.config.scaling())
    }
}

/// Knowledge Distillation (Hinton et al., 2015).
///
/// Transfers knowledge from a large teacher model to a smaller student model
/// by training the student to match the teacher's soft predictions.
///
/// # Variants
///
/// - **Standard**: Match soft logits with temperature scaling
/// - **Feature**: Match intermediate layer representations
/// - **Attention**: Match attention patterns
/// - **Self**: Use deeper layers to teach shallower layers
///
/// # Reference
///
/// - Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network.
#[derive(Debug, Clone)]
pub struct KnowledgeDistillation {
    temperature: f32,
    alpha: f32,
}

impl KnowledgeDistillation {
    /// Create knowledge distillation with temperature and mixing weight.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Softmax temperature (higher = softer targets)
    /// * `alpha` - Weight of distillation loss vs task loss (0-1)
    #[must_use]
    pub fn new(temperature: f32, alpha: f32) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        assert!((0.0..=1.0).contains(&alpha), "Alpha must be in [0, 1]");
        Self { temperature, alpha }
    }

    /// Compute soft cross-entropy loss between teacher and student logits.
    #[must_use]
    pub fn distillation_loss(&self, student_logits: &[f32], teacher_logits: &[f32]) -> f32 {
        let student_soft = softmax_with_temp(student_logits, self.temperature);
        let teacher_soft = softmax_with_temp(teacher_logits, self.temperature);

        // KL divergence: sum(teacher * log(teacher/student))
        let eps = 1e-10;
        let kl: f32 = teacher_soft
            .iter()
            .zip(student_soft.iter())
            .map(|(&t, &s)| t * ((t + eps) / (s + eps)).ln())
            .sum();

        // Scale by T^2 to match gradient magnitudes
        kl * self.temperature * self.temperature
    }

    /// Compute combined loss: alpha * `distill_loss` + (1-alpha) * `task_loss`.
    #[must_use]
    pub fn combined_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        task_loss: f32,
    ) -> f32 {
        let distill = self.distillation_loss(student_logits, teacher_logits);
        self.alpha * distill + (1.0 - self.alpha) * task_loss
    }

    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

/// Feature Distillation for matching intermediate representations.
#[derive(Debug, Clone)]
pub struct FeatureDistillation {
    /// Loss type for feature matching
    loss_type: FeatureLossType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureLossType {
    /// L2 loss between features
    MSE,
    /// L1 loss between features
    MAE,
    /// Cosine similarity loss
    Cosine,
}

impl FeatureDistillation {
    #[must_use]
    pub fn new(loss_type: FeatureLossType) -> Self {
        Self { loss_type }
    }

    /// Compute feature matching loss between teacher and student features.
    #[must_use]
    pub fn compute_loss(&self, student: &[f32], teacher: &[f32]) -> f32 {
        assert_eq!(student.len(), teacher.len());

        match self.loss_type {
            FeatureLossType::MSE => {
                student
                    .iter()
                    .zip(teacher.iter())
                    .map(|(&s, &t)| (s - t).powi(2))
                    .sum::<f32>()
                    / student.len() as f32
            }
            FeatureLossType::MAE => {
                student
                    .iter()
                    .zip(teacher.iter())
                    .map(|(&s, &t)| (s - t).abs())
                    .sum::<f32>()
                    / student.len() as f32
            }
            FeatureLossType::Cosine => {
                let dot: f32 = student
                    .iter()
                    .zip(teacher.iter())
                    .map(|(&s, &t)| s * t)
                    .sum();
                let norm_s: f32 = student.iter().map(|&s| s * s).sum::<f32>().sqrt();
                let norm_t: f32 = teacher.iter().map(|&t| t * t).sum::<f32>().sqrt();
                let cosine = dot / (norm_s * norm_t + 1e-10);
                1.0 - cosine // Loss is 1 - cosine_similarity
            }
        }
    }
}

/// Attention Transfer (Zagoruyko & Komodakis, 2017).
///
/// Transfers attention maps from teacher to student.
#[derive(Debug, Clone)]
pub struct AttentionTransfer {
    /// Power for attention map computation
    p: usize,
}

impl AttentionTransfer {
    #[must_use]
    pub fn new(p: usize) -> Self {
        Self { p }
    }

    /// Compute attention map: sum over channels of |activation|^p
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn compute_attention_map(
        &self,
        activations: &[f32],
        channels: usize,
        spatial: usize,
    ) -> Vec<f32> {
        let mut attention = vec![0.0_f32; spatial];

        for c in 0..channels {
            for s in 0..spatial {
                let idx = c * spatial + s;
                if idx < activations.len() {
                    attention[s] += activations[idx].abs().powi(self.p as i32);
                }
            }
        }

        // L2 normalize
        let norm: f32 = attention.iter().map(|&a| a * a).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for a in &mut attention {
                *a /= norm;
            }
        }

        attention
    }

    /// Compute attention transfer loss between teacher and student attention maps.
    #[must_use]
    pub fn compute_loss(
        &self,
        student_acts: &[f32],
        teacher_acts: &[f32],
        channels: usize,
        spatial: usize,
    ) -> f32 {
        let student_att = self.compute_attention_map(student_acts, channels, spatial);
        let teacher_att = self.compute_attention_map(teacher_acts, channels, spatial);

        student_att
            .iter()
            .zip(teacher_att.iter())
            .map(|(&s, &t)| (s - t).powi(2))
            .sum::<f32>()
            / spatial as f32
    }
}

/// Self-Distillation (Zhang et al., 2019).
///
/// Uses deeper layers to teach shallower layers within the same network.
#[derive(Debug, Clone)]
pub struct SelfDistillation {
    /// Temperature for soft labels
    temperature: f32,
    /// Layer indices for distillation (deeper -> shallower)
    layer_pairs: Vec<(usize, usize)>,
}

impl SelfDistillation {
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            layer_pairs: Vec::new(),
        }
    }

    /// Add a layer pair (`teacher_layer_idx`, `student_layer_idx`).
    /// Teacher should be deeper (higher index) than student.
    #[must_use]
    pub fn add_layer_pair(mut self, teacher_idx: usize, student_idx: usize) -> Self {
        self.layer_pairs.push((teacher_idx, student_idx));
        self
    }

    #[must_use]
    pub fn layer_pairs(&self) -> &[(usize, usize)] {
        &self.layer_pairs
    }

    /// Compute self-distillation loss for a layer pair.
    #[must_use]
    pub fn layer_loss(&self, student_output: &[f32], teacher_output: &[f32]) -> f32 {
        let student_soft = softmax_with_temp(student_output, self.temperature);
        let teacher_soft = softmax_with_temp(teacher_output, self.temperature);

        let eps = 1e-10;
        teacher_soft
            .iter()
            .zip(student_soft.iter())
            .map(|(&t, &s)| t * ((t + eps) / (s + eps)).ln())
            .sum::<f32>()
            * self.temperature
            * self.temperature
    }
}

/// Online Distillation / Deep Mutual Learning (Zhang et al., 2018).
///
/// Co-trains multiple networks simultaneously, where each network learns from
/// the others. Unlike standard distillation where the teacher is fixed, all
/// networks are trained together and learn from each other.
///
/// # Architecture
///
/// ```text
/// ┌─────────┐        ┌─────────┐
/// │Network 1│◄──────►│Network 2│
/// └────┬────┘        └────┬────┘
///      │ KL loss          │ KL loss
///      ▼                  ▼
/// ┌────────────────────────────┐
/// │     Ground Truth Loss      │
/// └────────────────────────────┘
/// ```
///
/// # Reference
///
/// - Zhang, Y., et al. (2018). Deep Mutual Learning. CVPR.
#[derive(Debug, Clone)]
pub struct OnlineDistillation {
    /// Number of networks in the cohort
    pub(crate) num_networks: usize,
    /// Temperature for KL divergence
    pub(crate) temperature: f32,
    /// Weight for mutual learning loss
    pub(crate) mutual_weight: f32,
}

#[cfg(test)]
#[path = "tests_lora_contract.rs"]
mod tests_lora_contract;
