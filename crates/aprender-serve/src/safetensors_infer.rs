//! SafeTensors Inference Support (PAR-301)
//!
//! Provides SafeTensors model loading and inference for HuggingFace models.
//!
//! ## Architecture
//!
//! SafeTensors files contain only tensor weights, so we need:
//! - `config.json` for model architecture (hidden_size, num_layers, etc.)
//! - `tokenizer.json` for text tokenization
//!
//! The converter loads these from sibling files and builds an AprTransformer.

use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::error::{RealizarError, Result};
use crate::safetensors::validation::ValidatedAprTransformer;
#[cfg(not(target_arch = "wasm32"))]
use crate::safetensors::ShardedSafeTensorsModel;
use crate::safetensors::{MappedSafeTensorsModel, SafetensorsConfig};
use std::path::Path;

/// Trait abstracting tensor access for both single-file and sharded SafeTensors models.
///
/// This enables `SafetensorsToAprConverter` to work identically with
/// `MappedSafeTensorsModel` (single file) and `ShardedSafeTensorsModel` (multi-shard).
pub(crate) trait TensorSource {
    fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>>;
    fn has_tensor(&self, name: &str) -> bool;
    fn tensor_names(&self) -> Vec<&str>;
}

#[cfg(not(target_arch = "wasm32"))]
impl TensorSource for MappedSafeTensorsModel {
    fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        MappedSafeTensorsModel::get_tensor_auto(self, name)
    }
    fn has_tensor(&self, name: &str) -> bool {
        MappedSafeTensorsModel::has_tensor(self, name)
    }
    fn tensor_names(&self) -> Vec<&str> {
        MappedSafeTensorsModel::tensor_names(self)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TensorSource for ShardedSafeTensorsModel {
    fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        ShardedSafeTensorsModel::get_tensor_auto(self, name)
    }
    fn has_tensor(&self, name: &str) -> bool {
        ShardedSafeTensorsModel::has_tensor(self, name)
    }
    fn tensor_names(&self) -> Vec<&str> {
        ShardedSafeTensorsModel::tensor_names(self)
    }
}

/// SafeTensors to APR Transformer converter
///
/// Converts HuggingFace SafeTensors models to APR Transformer format.
/// Supports BF16, F16, and F32 weights with automatic conversion to F32.
///
/// # Tensor Naming Conventions
///
/// Supports both HuggingFace and GGUF-style tensor naming:
/// - HuggingFace: `model.embed_tokens.weight`, `model.layers.{i}.self_attn.q_proj.weight`
/// - GGUF-style: `token_embd.weight`, `blk.{i}.attn_q.weight`
pub struct SafetensorsToAprConverter;

include!("safetensors_infer_convert.rs");
include!("safetensors_infer_convert_02.rs");
