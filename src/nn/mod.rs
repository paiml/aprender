//! Neural network modules for deep learning.
//!
//! This module provides PyTorch-compatible neural network building blocks
//! following the API design described in Paszke et al. (2019).
//!
//! # Architecture
//!
//! The nn module is organized around the [`Module`] trait, which defines
//! the interface for all neural network layers:
//!
//! - **Layers**: [`Linear`], [`Conv1d`], [`Conv2d`], [`Flatten`]
//! - **Pooling**: [`MaxPool1d`], [`MaxPool2d`], [`AvgPool2d`], [`GlobalAvgPool2d`]
//! - **Activations**: [`ReLU`], [`Sigmoid`], [`Tanh`], [`GELU`]
//! - **Normalization**: [`BatchNorm1d`], [`LayerNorm`], [`GroupNorm`], [`InstanceNorm`], [`RMSNorm`]
//! - **Regularization**: [`Dropout`], [`Dropout2d`], [`AlphaDropout`]
//! - **Containers**: [`Sequential`], [`ModuleList`], [`ModuleDict`]
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::{Module, Linear, ReLU, Sequential};
//! use aprender::autograd::Tensor;
//!
//! // Build a simple MLP
//! let model = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(ReLU::new())
//!     .add(Linear::new(256, 10));
//!
//! // Forward pass
//! let x = Tensor::randn(&[32, 784]);  // batch of 32
//! let output = model.forward(&x);     // [32, 10]
//!
//! // Get all parameters for optimizer
//! let params = model.parameters();
//! ```
//!
//! # References
//!
//! - Paszke, A., et al. (2019). `PyTorch`: An imperative style, high-performance
//!   deep learning library. `NeurIPS`.
//! - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
//!   deep feedforward neural networks. AISTATS.
//! - He, K., et al. (2015). Delving deep into rectifiers. ICCV.

mod activation;
mod container;
mod conv;
mod dropout;
pub mod functional;
pub mod generation;
pub mod gnn;
mod init;
mod linear;
pub mod loss;
mod module;
mod normalization;
pub mod optim;
pub mod quantization;
mod rnn;
pub mod scheduler;
pub mod self_supervised;
pub mod serialize;
mod transformer;
pub mod vae;

pub use activation::{LeakyReLU, ReLU, Sigmoid, Softmax, Tanh, GELU};
pub use container::{ModuleDict, ModuleList, Sequential};
pub use conv::{
    AvgPool2d, Conv1d, Conv2d, ConvDimensionNumbers, ConvLayout, Flatten, GlobalAvgPool2d,
    KernelLayout, MaxPool1d, MaxPool2d,
};
pub use dropout::{AlphaDropout, DropBlock, DropConnect, Dropout, Dropout2d};
pub use functional as F;
pub use gnn::{AdjacencyMatrix, GATConv, GCNConv, MessagePassing, SAGEAggregation, SAGEConv};
pub use init::{kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform};
pub use linear::Linear;
pub use loss::{
    BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, NLLLoss, Reduction, SmoothL1Loss,
};
pub use module::Module;
pub use normalization::{BatchNorm1d, GroupNorm, InstanceNorm, LayerNorm, RMSNorm};
pub use optim::{Adam, AdamW, Optimizer, RMSprop, SGD};
pub use rnn::{Bidirectional, GRU, LSTM};
pub use scheduler::{
    CosineAnnealingLR, ExponentialLR, LRScheduler, LinearWarmup, PlateauMode, ReduceLROnPlateau,
    StepLR, WarmupCosineScheduler,
};
pub use transformer::{
    generate_causal_mask, ALiBi, GroupedQueryAttention, LinearAttention, MultiHeadAttention,
    PositionalEncoding, RotaryPositionEmbedding, TransformerDecoderLayer, TransformerEncoderLayer,
};
