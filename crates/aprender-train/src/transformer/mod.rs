//! Transformer module with full model implementation and weight loading
//!
//! This module provides:
//! - `Transformer` - Complete transformer model for language modeling
//! - `TransformerConfig` - Model architecture configuration
//! - `load_safetensors_weights` - Load weights from SafeTensors files
//! - `Architecture` - Model architecture type for weight mapping
//! - `MultiHeadAttentionWithLoRA` - Attention with deep LoRA injection
//! - `LoRAProjection` - Linear projection with LoRA adapters

mod attention;
mod block;
mod config;
pub(crate) mod cuda_block;
mod embedding;
mod encoder;
mod encoder_block;
mod feedforward;
pub mod init;
mod model;
mod norm;
pub(crate) mod weights;
#[cfg(feature = "gpu")]
pub mod wgpu_block;

pub use attention::{LoRAProjection, MultiHeadAttention, MultiHeadAttentionWithLoRA};
pub use config::{ModelArchitecture, TransformerConfig};
#[cfg(feature = "cuda")]
pub use cuda_block::{
    BlockWeights, CudaBlock, CudaGradWorkspace, CudaNf4TransformerBlock, CudaTransformerBlock,
    GpuBlockOptimizerState,
};
#[cfg(not(feature = "cuda"))]
pub use cuda_block::{CudaBlock, CudaTransformerBlock};
#[cfg(feature = "cuda")]
pub(crate) use cuda_block::{CudaBlockScratch, CudaLoraGradWorkspace, GpuLoraOptimizerState};
pub use embedding::LearnedPositionEmbedding;
pub use encoder::EncoderModel;
pub use encoder_block::EncoderBlock;
pub use feedforward::EncoderFeedForward;
pub use model::Transformer;
pub use norm::LayerNorm;
pub use weights::{load_safetensors_weights, validate_weights, Architecture};
#[cfg(feature = "gpu")]
pub use wgpu_block::WgpuForwardPass;
