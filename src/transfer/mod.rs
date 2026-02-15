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
//!   neural networks? `NeurIPS`.
//! - Hu, E. J., et al. (2021). `LoRA`: Low-Rank Adaptation of Large Language
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

include!("mod_part_02.rs");
include!("mod_part_03.rs");
