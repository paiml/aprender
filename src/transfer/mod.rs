//! Transfer Learning module for cross-project knowledge sharing.
//!
//! This module provides infrastructure for transfer learning, enabling
//! knowledge sharing across related tasks (e.g., transpiler ecosystems
//! like depyler, ruchy, bashrs).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │   Python    │     │    Ruby     │     │    Bash     │
//! │   Source    │     │   Source    │     │   Source    │
//! └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
//!        │                   │                   │
//!        ▼                   ▼                   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │          Shared Error Embedding Space               │
//! │     (E0308, E0277, E0425, E0599, ... )             │
//! └─────────────────────────────────────────────────────┘
//!        │                   │                   │
//!        ▼                   ▼                   ▼
//! ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
//! │  depyler     │   │   ruchy      │   │   bashrs     │
//! │  Oracle      │   │   Oracle     │   │   Oracle     │
//! └──────────────┘   └──────────────┘   └──────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use aprender::transfer::{TransferEncoder, MultiTaskHead};
//! use aprender::nn::{Linear, Module};
//!
//! // Create shared encoder
//! let encoder = SharedEncoder::new(512, 256);
//!
//! // Wrap for transfer learning
//! let mut transfer_encoder = TransferableEncoder::new(encoder);
//!
//! // Pre-train on large dataset, then freeze
//! transfer_encoder.freeze_base();
//!
//! // Create task-specific heads
//! let mut multi_task = MultiTaskHead::new(transfer_encoder);
//! multi_task.add_task("depyler", 128);
//! multi_task.add_task("ruchy", 128);
//! ```
//!
//! # References
//!
//! - Yosinski, J., et al. (2014). How transferable are features in deep
//!   neural networks? NeurIPS.
//! - Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language
//!   Models. arXiv:2106.09685.

use crate::autograd::Tensor;
use crate::nn::Linear;
use crate::nn::Module;
use std::collections::HashMap;

/// Trait for encoders that support transfer learning operations.
///
/// Extends the base [`Module`] trait with methods for freezing/unfreezing
/// parameters and extracting intermediate features.
///
/// # Example
///
/// ```ignore
/// use aprender::transfer::TransferEncoder;
///
/// let mut encoder = create_encoder();
///
/// // Pre-train on source domain
/// train(&mut encoder, source_data);
///
/// // Freeze base for transfer
/// encoder.freeze_base();
///
/// // Fine-tune on target domain (only head trains)
/// train(&mut encoder, target_data);
///
/// // Optionally unfreeze for full fine-tuning
/// encoder.unfreeze_base();
/// ```
pub trait TransferEncoder: Module {
    /// Freeze the base encoder parameters.
    ///
    /// After calling this, the encoder's parameters won't be updated during training.
    /// Only task-specific heads will be trainable.
    fn freeze_base(&mut self);

    /// Unfreeze the base encoder parameters.
    ///
    /// Allows full fine-tuning of all parameters.
    fn unfreeze_base(&mut self);

    /// Check if the base encoder is frozen.
    fn is_frozen(&self) -> bool;

    /// Extract intermediate features from the encoder.
    ///
    /// Returns embeddings before any task-specific heads, useful for:
    /// - Feature visualization
    /// - Similarity computation
    /// - Multi-task learning
    fn get_features(&self, x: &Tensor) -> Tensor;
}

/// A wrapper that adds transfer learning capabilities to any Module.
///
/// This struct wraps an existing module and tracks frozen state,
/// providing the [`TransferEncoder`] interface.
#[derive(Debug)]
pub struct TransferableEncoder<M: Module> {
    /// The underlying encoder module
    encoder: M,
    /// Whether parameters are frozen
    frozen: bool,
}

impl<M: Module> TransferableEncoder<M> {
    /// Create a new transferable encoder wrapping the given module.
    pub fn new(encoder: M) -> Self {
        Self {
            encoder,
            frozen: false,
        }
    }

    /// Get a reference to the underlying encoder.
    pub fn encoder(&self) -> &M {
        &self.encoder
    }

    /// Get a mutable reference to the underlying encoder.
    pub fn encoder_mut(&mut self) -> &mut M {
        &mut self.encoder
    }
}

impl<M: Module> Module for TransferableEncoder<M> {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.encoder.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        if self.frozen {
            vec![] // Return empty to prevent updates
        } else {
            self.encoder.parameters()
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        if self.frozen {
            vec![] // Return empty to prevent updates
        } else {
            self.encoder.parameters_mut()
        }
    }

    fn train(&mut self) {
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
    }

    fn training(&self) -> bool {
        self.encoder.training()
    }
}

impl<M: Module> TransferEncoder for TransferableEncoder<M> {
    fn freeze_base(&mut self) {
        self.frozen = true;
    }

    fn unfreeze_base(&mut self) {
        self.frozen = false;
    }

    fn is_frozen(&self) -> bool {
        self.frozen
    }

    fn get_features(&self, x: &Tensor) -> Tensor {
        // Default implementation: forward pass IS the features
        self.encoder.forward(x)
    }
}

/// Multi-task learning head with shared encoder.
///
/// Enables training multiple related tasks simultaneously with a shared
/// backbone encoder and task-specific output heads.
///
/// # Architecture
///
/// ```text
/// ┌──────────────────────────┐
/// │     Shared Encoder       │
/// │  (TransferEncoder)       │
/// └───────────┬──────────────┘
///             │ features
///     ┌───────┼───────┐
///     ▼       ▼       ▼
/// ┌───────┐┌───────┐┌───────┐
/// │ Head1 ││ Head2 ││ Head3 │
/// │(task1)││(task2)││(task3)│
/// └───────┘└───────┘└───────┘
/// ```
///
/// # Example
///
/// ```ignore
/// use aprender::transfer::MultiTaskHead;
///
/// let encoder = create_shared_encoder();
/// let mut multi_task = MultiTaskHead::new(encoder);
///
/// multi_task.add_task("classification", 10);
/// multi_task.add_task("regression", 1);
///
/// let features = multi_task.forward_shared(&input);
/// let class_output = multi_task.forward_task("classification", &features);
/// let reg_output = multi_task.forward_task("regression", &features);
/// ```
#[derive(Debug)]
pub struct MultiTaskHead<E: TransferEncoder> {
    /// Shared encoder for all tasks
    shared_encoder: E,
    /// Task-specific output heads
    task_heads: HashMap<String, Linear>,
    /// Feature dimension from encoder
    feature_dim: usize,
}

impl<E: TransferEncoder> MultiTaskHead<E> {
    /// Create a new multi-task head with the given shared encoder.
    ///
    /// # Arguments
    ///
    /// * `shared_encoder` - The shared encoder for feature extraction
    /// * `feature_dim` - Dimension of features output by the encoder
    pub fn new(shared_encoder: E, feature_dim: usize) -> Self {
        Self {
            shared_encoder,
            task_heads: HashMap::new(),
            feature_dim,
        }
    }

    /// Add a new task with the specified output dimension.
    ///
    /// # Arguments
    ///
    /// * `task_name` - Unique name for the task
    /// * `output_dim` - Output dimension for this task's head
    pub fn add_task(&mut self, task_name: &str, output_dim: usize) {
        let head = Linear::new(self.feature_dim, output_dim);
        self.task_heads.insert(task_name.to_string(), head);
    }

    /// Remove a task head.
    pub fn remove_task(&mut self, task_name: &str) -> Option<Linear> {
        self.task_heads.remove(task_name)
    }

    /// Get list of registered task names.
    pub fn task_names(&self) -> Vec<&String> {
        self.task_heads.keys().collect()
    }

    /// Forward pass through shared encoder only.
    ///
    /// Returns features that can be passed to task-specific heads.
    pub fn forward_shared(&self, input: &Tensor) -> Tensor {
        self.shared_encoder.get_features(input)
    }

    /// Forward pass through a specific task head.
    ///
    /// # Arguments
    ///
    /// * `task_name` - Name of the task
    /// * `features` - Features from `forward_shared`
    ///
    /// # Panics
    ///
    /// Panics if the task name is not registered.
    pub fn forward_task(&self, task_name: &str, features: &Tensor) -> Tensor {
        let head = self
            .task_heads
            .get(task_name)
            .unwrap_or_else(|| panic!("Unknown task: {task_name}"));
        head.forward(features)
    }

    /// Full forward pass for a specific task.
    ///
    /// Combines `forward_shared` and `forward_task` into one call.
    pub fn forward_full(&self, task_name: &str, input: &Tensor) -> Tensor {
        let features = self.forward_shared(input);
        self.forward_task(task_name, &features)
    }

    /// Get the shared encoder.
    pub fn encoder(&self) -> &E {
        &self.shared_encoder
    }

    /// Get the shared encoder mutably.
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.shared_encoder
    }

    /// Freeze the shared encoder for transfer learning.
    pub fn freeze_encoder(&mut self) {
        self.shared_encoder.freeze_base();
    }

    /// Unfreeze the shared encoder for full fine-tuning.
    pub fn unfreeze_encoder(&mut self) {
        self.shared_encoder.unfreeze_base();
    }
}

impl<E: TransferEncoder> Module for MultiTaskHead<E> {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("MultiTaskHead requires task name. Use forward_full(task_name, input) instead.");
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.shared_encoder.parameters();
        for head in self.task_heads.values() {
            params.extend(head.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.shared_encoder.parameters_mut();
        for head in self.task_heads.values_mut() {
            params.extend(head.parameters_mut());
        }
        params
    }
}

/// Domain adaptation for aligning source and target distributions.
///
/// Uses adversarial training to learn domain-invariant representations.
/// The discriminator tries to distinguish source vs target, while the
/// encoder tries to fool it.
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────┐
/// │              Domain Adaptation                       │
/// ├─────────────────────────────────────────────────────┤
/// │  Source Data → Encoder → Features ─┐               │
/// │                                     ├→ Discriminator │
/// │  Target Data → Encoder → Features ─┘    (adversarial)│
/// └─────────────────────────────────────────────────────┘
/// ```
///
/// # Training Strategy
///
/// 1. Forward source and target through shared encoder
/// 2. Train discriminator to distinguish domains
/// 3. Train encoder to fool discriminator (gradient reversal)
/// 4. Optionally train task-specific loss on source labels
#[derive(Debug)]
pub struct DomainAdapter<E: TransferEncoder> {
    /// Shared encoder for both domains
    encoder: E,
    /// Domain discriminator (binary classification)
    discriminator: Linear,
    /// Gradient reversal scale (lambda)
    reversal_scale: f32,
}

impl<E: TransferEncoder> DomainAdapter<E> {
    /// Create a new domain adapter.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Shared encoder for feature extraction
    /// * `feature_dim` - Dimension of encoder output features
    /// * `reversal_scale` - Scale for gradient reversal (default: 1.0)
    pub fn new(encoder: E, feature_dim: usize, reversal_scale: f32) -> Self {
        Self {
            encoder,
            discriminator: Linear::new(feature_dim, 1), // Binary: source(0) vs target(1)
            reversal_scale,
        }
    }

    /// Extract features from input.
    pub fn encode(&self, input: &Tensor) -> Tensor {
        self.encoder.get_features(input)
    }

    /// Predict domain (0=source, 1=target) from features.
    pub fn discriminate(&self, features: &Tensor) -> Tensor {
        self.discriminator.forward(features)
    }

    /// Get the gradient reversal scale.
    pub fn reversal_scale(&self) -> f32 {
        self.reversal_scale
    }

    /// Set the gradient reversal scale.
    ///
    /// Higher values = stronger domain adaptation.
    /// Typically starts small and increases during training.
    pub fn set_reversal_scale(&mut self, scale: f32) {
        self.reversal_scale = scale;
    }

    /// Get the encoder.
    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Get the encoder mutably.
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }
}

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

/// LoRA (Low-Rank Adaptation) configuration.
///
/// LoRA freezes pre-trained weights and adds small trainable matrices
/// to specific layers, drastically reducing memory and compute for fine-tuning.
///
/// # Reference
///
/// Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Rank of the low-rank matrices (typically 4, 8, or 16)
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f32,
    /// Target module names (e.g., `["q_proj", "v_proj"]`)
    pub target_modules: Vec<String>,
    /// Dropout probability for LoRA layers
    pub dropout: f32,
}

impl LoRAConfig {
    /// Create a new LoRA configuration.
    ///
    /// # Arguments
    ///
    /// * `rank` - Rank of low-rank matrices (4-64 typical)
    /// * `alpha` - Scaling factor (often same as rank)
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            dropout: 0.0,
        }
    }

    /// Set target modules for LoRA adaptation.
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Compute the scaling factor.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self::new(8, 8.0)
    }
}

/// LoRA adapter weights for a single layer.
///
/// Stores the A and B matrices for low-rank adaptation:
/// W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
#[derive(Debug)]
pub struct LoRAAdapter {
    /// Down-projection matrix A (input_dim → rank)
    pub lora_a: Tensor,
    /// Up-projection matrix B (rank → output_dim)
    pub lora_b: Tensor,
    /// Configuration
    pub config: LoRAConfig,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter for a layer.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input dimension of the layer
    /// * `output_dim` - Output dimension of the layer
    /// * `config` - LoRA configuration
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

    /// Apply the LoRA adaptation to a weight matrix.
    ///
    /// Returns W + scaling * (B @ A)
    pub fn apply(&self, base_weight: &Tensor) -> Tensor {
        let ba = self.lora_b.matmul(&self.lora_a);
        let scaled = ba.mul_scalar(self.config.scaling());
        base_weight.add(&scaled)
    }

    /// Get the delta weight (B @ A * scaling).
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
    pub fn new(temperature: f32, alpha: f32) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        assert!((0.0..=1.0).contains(&alpha), "Alpha must be in [0, 1]");
        Self { temperature, alpha }
    }

    /// Compute soft cross-entropy loss between teacher and student logits.
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

    /// Compute combined loss: alpha * distill_loss + (1-alpha) * task_loss.
    pub fn combined_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        task_loss: f32,
    ) -> f32 {
        let distill = self.distillation_loss(student_logits, teacher_logits);
        self.alpha * distill + (1.0 - self.alpha) * task_loss
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

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
    pub fn new(loss_type: FeatureLossType) -> Self {
        Self { loss_type }
    }

    /// Compute feature matching loss between teacher and student features.
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
    pub fn new(p: usize) -> Self {
        Self { p }
    }

    /// Compute attention map: sum over channels of |activation|^p
    #[allow(clippy::needless_range_loop)]
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
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            layer_pairs: Vec::new(),
        }
    }

    /// Add a layer pair (teacher_layer_idx, student_layer_idx).
    /// Teacher should be deeper (higher index) than student.
    pub fn add_layer_pair(mut self, teacher_idx: usize, student_idx: usize) -> Self {
        self.layer_pairs.push((teacher_idx, student_idx));
        self
    }

    pub fn layer_pairs(&self) -> &[(usize, usize)] {
        &self.layer_pairs
    }

    /// Compute self-distillation loss for a layer pair.
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
    num_networks: usize,
    /// Temperature for KL divergence
    temperature: f32,
    /// Weight for mutual learning loss
    mutual_weight: f32,
}

impl OnlineDistillation {
    /// Create online distillation with specified number of peer networks.
    ///
    /// # Arguments
    ///
    /// * `num_networks` - Number of networks to co-train (typically 2-4)
    /// * `temperature` - Temperature for softening predictions
    /// * `mutual_weight` - Weight for mutual learning loss (vs task loss)
    pub fn new(num_networks: usize, temperature: f32, mutual_weight: f32) -> Self {
        assert!(
            num_networks >= 2,
            "Need at least 2 networks for mutual learning"
        );
        assert!(temperature > 0.0, "Temperature must be positive");
        Self {
            num_networks,
            temperature,
            mutual_weight,
        }
    }

    /// Compute mutual learning loss for one network given all peer outputs.
    ///
    /// Each network learns from the average of its peers' predictions.
    pub fn mutual_loss(&self, network_idx: usize, all_logits: &[Vec<f32>]) -> f32 {
        assert_eq!(all_logits.len(), self.num_networks);

        let my_logits = &all_logits[network_idx];
        let my_soft = softmax_with_temp(my_logits, self.temperature);

        // Average KL divergence to all other networks
        let mut total_kl = 0.0;
        let mut peer_count = 0;

        for (i, peer_logits) in all_logits.iter().enumerate() {
            if i != network_idx {
                let peer_soft = softmax_with_temp(peer_logits, self.temperature);
                let eps = 1e-10;
                let kl: f32 = peer_soft
                    .iter()
                    .zip(my_soft.iter())
                    .map(|(&p, &s)| p * ((p + eps) / (s + eps)).ln())
                    .sum();
                total_kl += kl * self.temperature * self.temperature;
                peer_count += 1;
            }
        }

        if peer_count > 0 {
            total_kl / peer_count as f32
        } else {
            0.0
        }
    }

    /// Compute combined loss for one network: task_loss + mutual_weight * mutual_loss.
    pub fn combined_loss(
        &self,
        network_idx: usize,
        all_logits: &[Vec<f32>],
        task_loss: f32,
    ) -> f32 {
        let mutual = self.mutual_loss(network_idx, all_logits);
        task_loss + self.mutual_weight * mutual
    }

    /// Compute losses for all networks.
    pub fn all_losses(&self, all_logits: &[Vec<f32>], task_losses: &[f32]) -> Vec<f32> {
        (0..self.num_networks)
            .map(|i| self.combined_loss(i, all_logits, task_losses[i]))
            .collect()
    }

    pub fn num_networks(&self) -> usize {
        self.num_networks
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn mutual_weight(&self) -> f32 {
        self.mutual_weight
    }
}

/// Progressive Distillation (Salimans & Ho, 2022).
///
/// Gradually distills a diffusion model by halving the number of sampling steps.
/// Used to speed up diffusion model inference.
#[derive(Debug, Clone)]
pub struct ProgressiveDistillation {
    /// Current number of steps
    current_steps: usize,
    /// Target number of steps
    target_steps: usize,
    /// Distillation weight
    weight: f32,
}

impl ProgressiveDistillation {
    /// Create progressive distillation from current to target steps.
    pub fn new(current_steps: usize, target_steps: usize, weight: f32) -> Self {
        assert!(
            current_steps > target_steps,
            "Current must be > target steps"
        );
        assert!(target_steps > 0, "Target steps must be positive");
        Self {
            current_steps,
            target_steps,
            weight,
        }
    }

    /// Check if we should halve steps (typically after convergence).
    pub fn should_halve(&self) -> bool {
        self.current_steps > self.target_steps * 2
    }

    /// Halve the number of steps.
    pub fn halve_steps(&mut self) {
        if self.current_steps > self.target_steps {
            self.current_steps /= 2;
        }
    }

    /// Compute distillation loss between teacher (2N steps) and student (N steps).
    pub fn compute_loss(&self, teacher_output: &[f32], student_output: &[f32]) -> f32 {
        assert_eq!(teacher_output.len(), student_output.len());
        let mse: f32 = teacher_output
            .iter()
            .zip(student_output.iter())
            .map(|(&t, &s)| (t - s).powi(2))
            .sum::<f32>()
            / teacher_output.len() as f32;
        self.weight * mse
    }

    pub fn current_steps(&self) -> usize {
        self.current_steps
    }

    pub fn target_steps(&self) -> usize {
        self.target_steps
    }
}

fn softmax_with_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let max = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Prototypical Networks for few-shot learning (Snell et al., 2017).
///
/// Learns a metric space where classification is performed by computing
/// distances to class prototypes (mean embeddings of support examples).
#[derive(Debug, Clone)]
pub struct PrototypicalNetwork {
    /// Distance metric
    distance: DistanceMetric,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
}

impl PrototypicalNetwork {
    pub fn new(distance: DistanceMetric) -> Self {
        Self { distance }
    }

    /// Compute class prototypes from support set embeddings.
    /// support: Vec of (embedding, class_label)
    pub fn compute_prototypes(&self, support: &[(Vec<f32>, usize)]) -> Vec<(usize, Vec<f32>)> {
        use std::collections::HashMap;
        let mut class_sums: HashMap<usize, (Vec<f32>, usize)> = HashMap::new();

        for (emb, class) in support {
            let entry = class_sums
                .entry(*class)
                .or_insert_with(|| (vec![0.0; emb.len()], 0));
            for (i, &v) in emb.iter().enumerate() {
                entry.0[i] += v;
            }
            entry.1 += 1;
        }

        class_sums
            .into_iter()
            .map(|(class, (sum, count))| {
                let proto: Vec<f32> = sum.iter().map(|&s| s / count as f32).collect();
                (class, proto)
            })
            .collect()
    }

    /// Classify query embedding against prototypes.
    pub fn classify(&self, query: &[f32], prototypes: &[(usize, Vec<f32>)]) -> usize {
        let mut best_class = 0;
        let mut best_dist = f32::INFINITY;

        for (class, proto) in prototypes {
            let dist = self.distance(query, proto);
            if dist < best_dist {
                best_dist = dist;
                best_class = *class;
            }
        }
        best_class
    }

    /// Compute class probabilities (softmax of negative distances).
    pub fn predict_proba(
        &self,
        query: &[f32],
        prototypes: &[(usize, Vec<f32>)],
    ) -> Vec<(usize, f32)> {
        let neg_dists: Vec<(usize, f32)> = prototypes
            .iter()
            .map(|(c, p)| (*c, -self.distance(query, p)))
            .collect();

        let max_d = neg_dists
            .iter()
            .map(|(_, d)| *d)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = neg_dists.iter().map(|(_, d)| (d - max_d).exp()).sum();

        neg_dists
            .iter()
            .map(|(c, d)| (*c, (d - max_d).exp() / exp_sum))
            .collect()
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.distance {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
                let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (na * nb + 1e-10)
            }
        }
    }
}

impl Default for PrototypicalNetwork {
    fn default() -> Self {
        Self::new(DistanceMetric::Euclidean)
    }
}

/// Matching Networks for few-shot learning (Vinyals et al., 2016).
#[derive(Debug, Clone)]
pub struct MatchingNetwork {
    temperature: f32,
}

impl MatchingNetwork {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Predict class by attention-weighted combination over support set.
    pub fn predict(&self, query: &[f32], support: &[(Vec<f32>, usize)]) -> usize {
        use std::collections::HashMap;
        let mut class_scores: HashMap<usize, f32> = HashMap::new();

        // Compute attention weights (softmax of cosine similarities)
        let sims: Vec<f32> = support
            .iter()
            .map(|(emb, _)| cosine_similarity(query, emb) / self.temperature)
            .collect();

        let max_sim = sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = sims.iter().map(|&s| (s - max_sim).exp()).sum();
        let weights: Vec<f32> = sims
            .iter()
            .map(|&s| (s - max_sim).exp() / exp_sum)
            .collect();

        for ((_, class), &w) in support.iter().zip(&weights) {
            *class_scores.entry(*class).or_insert(0.0) += w;
        }

        class_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(c, _)| c)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;

    // Simple test encoder for testing
    struct SimpleEncoder {
        linear: Linear,
        training: bool,
    }

    impl SimpleEncoder {
        fn new(input_dim: usize, output_dim: usize) -> Self {
            Self {
                linear: Linear::new(input_dim, output_dim),
                training: true,
            }
        }
    }

    impl Module for SimpleEncoder {
        fn forward(&self, input: &Tensor) -> Tensor {
            self.linear.forward(input)
        }

        fn parameters(&self) -> Vec<&Tensor> {
            self.linear.parameters()
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            self.linear.parameters_mut()
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn training(&self) -> bool {
            self.training
        }
    }

    #[test]
    fn test_transferable_encoder_basic() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);

        let x = Tensor::ones(&[2, 10]);
        let y = transfer.forward(&x);

        assert_eq!(y.shape(), &[2, 5]);
    }

    #[test]
    fn test_transferable_encoder_freeze_unfreeze() {
        let encoder = SimpleEncoder::new(10, 5);
        let mut transfer = TransferableEncoder::new(encoder);

        // Initially not frozen
        assert!(!transfer.is_frozen());
        assert!(!transfer.parameters().is_empty());

        // Freeze
        transfer.freeze_base();
        assert!(transfer.is_frozen());
        assert!(transfer.parameters().is_empty());

        // Unfreeze
        transfer.unfreeze_base();
        assert!(!transfer.is_frozen());
        assert!(!transfer.parameters().is_empty());
    }

    #[test]
    fn test_transferable_encoder_get_features() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);

        let x = Tensor::ones(&[2, 10]);
        let features = transfer.get_features(&x);

        assert_eq!(features.shape(), &[2, 5]);
    }

    #[test]
    fn test_multi_task_head_basic() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);
        let mut multi_task = MultiTaskHead::new(transfer, 5);

        multi_task.add_task("task1", 3);
        multi_task.add_task("task2", 7);

        let x = Tensor::ones(&[2, 10]);

        let out1 = multi_task.forward_full("task1", &x);
        let out2 = multi_task.forward_full("task2", &x);

        assert_eq!(out1.shape(), &[2, 3]);
        assert_eq!(out2.shape(), &[2, 7]);
    }

    #[test]
    fn test_multi_task_head_shared_features() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);
        let mut multi_task = MultiTaskHead::new(transfer, 5);

        multi_task.add_task("task1", 3);

        let x = Tensor::ones(&[2, 10]);
        let features = multi_task.forward_shared(&x);
        let output = multi_task.forward_task("task1", &features);

        assert_eq!(features.shape(), &[2, 5]);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_multi_task_head_freeze_encoder() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);
        let mut multi_task = MultiTaskHead::new(transfer, 5);

        multi_task.add_task("task1", 3);

        // Before freeze: encoder params included
        let params_before = multi_task.parameters().len();

        // Freeze encoder
        multi_task.freeze_encoder();

        // After freeze: only head params
        let params_after = multi_task.parameters().len();

        assert!(params_after < params_before);
    }

    #[test]
    fn test_multi_task_head_task_names() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);
        let mut multi_task = MultiTaskHead::new(transfer, 5);

        multi_task.add_task("classification", 10);
        multi_task.add_task("regression", 1);

        let names = multi_task.task_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&&"classification".to_string()));
        assert!(names.contains(&&"regression".to_string()));
    }

    #[test]
    fn test_domain_adapter_basic() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);
        let adapter = DomainAdapter::new(transfer, 5, 1.0);

        let x = Tensor::ones(&[2, 10]);
        let features = adapter.encode(&x);
        let domain_pred = adapter.discriminate(&features);

        assert_eq!(features.shape(), &[2, 5]);
        assert_eq!(domain_pred.shape(), &[2, 1]);
    }

    #[test]
    fn test_domain_adapter_reversal_scale() {
        let encoder = SimpleEncoder::new(10, 5);
        let transfer = TransferableEncoder::new(encoder);
        let mut adapter = DomainAdapter::new(transfer, 5, 1.0);

        assert!((adapter.reversal_scale() - 1.0).abs() < 1e-6);

        adapter.set_reversal_scale(0.5);
        assert!((adapter.reversal_scale() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_lora_config() {
        let config = LoRAConfig::new(8, 16.0);

        assert_eq!(config.rank, 8);
        assert!((config.alpha - 16.0).abs() < 1e-6);
        assert!((config.scaling() - 2.0).abs() < 1e-6); // 16/8 = 2
    }

    #[test]
    fn test_lora_config_with_modules() {
        let config = LoRAConfig::new(4, 4.0)
            .with_target_modules(vec!["attn".to_string(), "mlp".to_string()])
            .with_dropout(0.1);

        assert_eq!(config.target_modules.len(), 2);
        assert!((config.dropout - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_lora_adapter_creation() {
        let config = LoRAConfig::new(4, 4.0);
        let adapter = LoRAAdapter::new(10, 20, config);

        assert_eq!(adapter.lora_a.shape(), &[4, 10]); // rank x input_dim
        assert_eq!(adapter.lora_b.shape(), &[20, 4]); // output_dim x rank
    }

    #[test]
    fn test_lora_adapter_delta_weight() {
        let config = LoRAConfig::new(4, 4.0);
        let adapter = LoRAAdapter::new(10, 20, config);

        let delta = adapter.delta_weight();

        // Delta should be output_dim x input_dim
        assert_eq!(delta.shape(), &[20, 10]);
    }

    #[test]
    fn test_lora_adapter_initial_zero_delta() {
        // B is initialized to zeros, so BA should be ~zero initially
        let config = LoRAConfig::new(4, 4.0);
        let adapter = LoRAAdapter::new(10, 20, config);

        let delta = adapter.delta_weight();

        // All values should be zero (B starts as zeros)
        for &v in delta.data() {
            assert!(v.abs() < 1e-6, "Delta should be zero initially");
        }
    }

    #[test]
    fn test_transfer_encoder_train_eval() {
        let encoder = SimpleEncoder::new(10, 5);
        let mut transfer = TransferableEncoder::new(encoder);

        assert!(transfer.training());

        transfer.eval();
        assert!(!transfer.training());

        transfer.train();
        assert!(transfer.training());
    }

    // Knowledge Distillation Tests
    #[test]
    fn test_knowledge_distillation_creation() {
        let kd = KnowledgeDistillation::new(4.0, 0.7);
        assert!((kd.temperature() - 4.0).abs() < 1e-6);
        assert!((kd.alpha() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_loss_same_logits() {
        let kd = KnowledgeDistillation::new(2.0, 0.5);
        let logits = vec![2.0, 1.0, 0.0];
        let loss = kd.distillation_loss(&logits, &logits);
        assert!(loss.abs() < 0.01, "Same logits should have ~zero loss");
    }

    #[test]
    fn test_distillation_loss_different_logits() {
        let kd = KnowledgeDistillation::new(2.0, 0.5);
        let student = vec![2.0, 0.0, 0.0];
        let teacher = vec![0.0, 2.0, 0.0];
        let loss = kd.distillation_loss(&student, &teacher);
        assert!(loss > 0.0, "Different logits should have positive loss");
    }

    #[test]
    fn test_combined_loss() {
        let kd = KnowledgeDistillation::new(2.0, 0.6);
        let student = vec![1.0, 1.0, 1.0];
        let teacher = vec![1.0, 1.0, 1.0];
        let task_loss = 0.5;

        let combined = kd.combined_loss(&student, &teacher, task_loss);
        // With same logits, distill loss ~0, so combined ~= (1-0.6) * 0.5 = 0.2
        assert!((combined - 0.2).abs() < 0.1);
    }

    // Feature Distillation Tests
    #[test]
    fn test_feature_distillation_mse() {
        let fd = FeatureDistillation::new(FeatureLossType::MSE);
        let student = vec![1.0, 2.0, 3.0];
        let teacher = vec![1.0, 2.0, 3.0];
        let loss = fd.compute_loss(&student, &teacher);
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_feature_distillation_mae() {
        let fd = FeatureDistillation::new(FeatureLossType::MAE);
        let student = vec![1.0, 2.0, 3.0];
        let teacher = vec![2.0, 3.0, 4.0];
        let loss = fd.compute_loss(&student, &teacher);
        assert!((loss - 1.0).abs() < 1e-6); // Average diff is 1
    }

    #[test]
    fn test_feature_distillation_cosine() {
        let fd = FeatureDistillation::new(FeatureLossType::Cosine);
        let student = vec![1.0, 0.0, 0.0];
        let teacher = vec![1.0, 0.0, 0.0];
        let loss = fd.compute_loss(&student, &teacher);
        assert!(loss.abs() < 1e-6); // Same direction = 0 loss
    }

    // Attention Transfer Tests
    #[test]
    fn test_attention_transfer_creation() {
        let at = AttentionTransfer::new(2);
        let activations = vec![1.0, 2.0, 3.0, 4.0]; // 2 channels, 2 spatial
        let attention = at.compute_attention_map(&activations, 2, 2);
        assert_eq!(attention.len(), 2);
    }

    #[test]
    fn test_attention_transfer_loss() {
        let at = AttentionTransfer::new(2);
        let student = vec![1.0, 2.0, 3.0, 4.0];
        let teacher = vec![1.0, 2.0, 3.0, 4.0];
        let loss = at.compute_loss(&student, &teacher, 2, 2);
        assert!(loss < 1e-6);
    }

    // Self-Distillation Tests
    #[test]
    fn test_self_distillation_creation() {
        let sd = SelfDistillation::new(3.0)
            .add_layer_pair(3, 1)
            .add_layer_pair(4, 2);
        assert_eq!(sd.layer_pairs().len(), 2);
    }

    #[test]
    fn test_self_distillation_layer_loss() {
        let sd = SelfDistillation::new(2.0);
        let student = vec![1.0, 0.0, 0.0];
        let teacher = vec![1.0, 0.0, 0.0];
        let loss = sd.layer_loss(&student, &teacher);
        assert!(loss.abs() < 0.01);
    }

    // Meta-Learning Tests
    #[test]
    fn test_prototypical_network_creation() {
        let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
        let pn2 = PrototypicalNetwork::default();
        assert_eq!(pn.distance, pn2.distance);
    }

    #[test]
    fn test_prototypical_compute_prototypes() {
        let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
        let support = vec![
            (vec![1.0, 0.0], 0),
            (vec![1.0, 0.0], 0),
            (vec![0.0, 1.0], 1),
        ];
        let protos = pn.compute_prototypes(&support);
        assert_eq!(protos.len(), 2);
    }

    #[test]
    fn test_prototypical_classify() {
        let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
        let protos = vec![(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];
        let query = vec![0.9, 0.1];
        let class = pn.classify(&query, &protos);
        assert_eq!(class, 0);
    }

    #[test]
    fn test_prototypical_predict_proba() {
        let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
        let protos = vec![(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];
        let query = vec![1.0, 0.0];
        let probs = pn.predict_proba(&query, &protos);
        assert_eq!(probs.len(), 2);
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matching_network_predict() {
        let mn = MatchingNetwork::new(1.0);
        let support = vec![
            (vec![1.0, 0.0], 0),
            (vec![0.9, 0.1], 0),
            (vec![0.0, 1.0], 1),
        ];
        let query = vec![0.95, 0.05];
        let class = mn.predict(&query, &support);
        assert_eq!(class, 0);
    }

    // Online Distillation Tests
    #[test]
    fn test_online_distillation_creation() {
        let od = OnlineDistillation::new(3, 2.0, 0.5);
        assert_eq!(od.num_networks(), 3);
        assert!((od.temperature() - 2.0).abs() < 1e-6);
        assert!((od.mutual_weight() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_online_distillation_same_logits() {
        let od = OnlineDistillation::new(2, 2.0, 1.0);
        let all_logits = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let loss = od.mutual_loss(0, &all_logits);
        assert!(loss.abs() < 0.01, "Same logits should have ~zero loss");
    }

    #[test]
    fn test_online_distillation_different_logits() {
        let od = OnlineDistillation::new(2, 2.0, 1.0);
        let all_logits = vec![vec![2.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]];
        let loss = od.mutual_loss(0, &all_logits);
        assert!(loss > 0.0, "Different logits should have positive loss");
    }

    #[test]
    fn test_online_distillation_combined_loss() {
        let od = OnlineDistillation::new(2, 2.0, 0.5);
        let all_logits = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let task_loss = 0.4;
        let combined = od.combined_loss(0, &all_logits, task_loss);
        // Same logits = ~zero mutual loss, so combined ~= task_loss
        assert!((combined - 0.4).abs() < 0.1);
    }

    #[test]
    fn test_online_distillation_all_losses() {
        let od = OnlineDistillation::new(3, 2.0, 1.0);
        let all_logits = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let task_losses = vec![0.1, 0.2, 0.3];
        let losses = od.all_losses(&all_logits, &task_losses);
        assert_eq!(losses.len(), 3);
        // Each loss should be >= task_loss due to mutual component
        assert!(losses[0] >= 0.1);
        assert!(losses[1] >= 0.2);
        assert!(losses[2] >= 0.3);
    }

    #[test]
    fn test_online_distillation_three_networks() {
        let od = OnlineDistillation::new(3, 1.0, 1.0);
        let all_logits = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        // Network 0 and 1 are similar, so loss for 0 should be smaller
        // than loss for 2 (which is different from both)
        let loss_0 = od.mutual_loss(0, &all_logits);
        let loss_2 = od.mutual_loss(2, &all_logits);
        // Network 2 differs from both, should have higher loss
        assert!(loss_2 > loss_0 * 0.5);
    }

    // Progressive Distillation Tests
    #[test]
    fn test_progressive_distillation_creation() {
        let pd = ProgressiveDistillation::new(64, 4, 1.0);
        assert_eq!(pd.current_steps(), 64);
        assert_eq!(pd.target_steps(), 4);
    }

    #[test]
    fn test_progressive_distillation_should_halve() {
        let pd = ProgressiveDistillation::new(64, 4, 1.0);
        assert!(pd.should_halve()); // 64 > 4*2

        let pd2 = ProgressiveDistillation::new(8, 4, 1.0);
        assert!(!pd2.should_halve()); // 8 is not > 4*2
    }

    #[test]
    fn test_progressive_distillation_halve_steps() {
        let mut pd = ProgressiveDistillation::new(64, 4, 1.0);
        pd.halve_steps();
        assert_eq!(pd.current_steps(), 32);
        pd.halve_steps();
        assert_eq!(pd.current_steps(), 16);
        pd.halve_steps();
        assert_eq!(pd.current_steps(), 8);
        pd.halve_steps();
        assert_eq!(pd.current_steps(), 4);
        pd.halve_steps(); // Should not go below target
        assert_eq!(pd.current_steps(), 4);
    }

    #[test]
    fn test_progressive_distillation_compute_loss() {
        let pd = ProgressiveDistillation::new(16, 4, 1.0);
        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![1.0, 2.0, 3.0];
        let loss = pd.compute_loss(&teacher, &student);
        assert!(loss.abs() < 1e-6); // Same outputs = zero loss
    }

    #[test]
    fn test_progressive_distillation_loss_with_diff() {
        let pd = ProgressiveDistillation::new(16, 4, 1.0);
        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![2.0, 3.0, 4.0];
        let loss = pd.compute_loss(&teacher, &student);
        // MSE = ((1)^2 + (1)^2 + (1)^2) / 3 = 1.0
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_progressive_distillation_weight() {
        let pd = ProgressiveDistillation::new(16, 4, 0.5);
        let teacher = vec![0.0, 0.0];
        let student = vec![1.0, 1.0];
        let loss = pd.compute_loss(&teacher, &student);
        // MSE = 1.0, weighted = 0.5
        assert!((loss - 0.5).abs() < 1e-6);
    }
}
