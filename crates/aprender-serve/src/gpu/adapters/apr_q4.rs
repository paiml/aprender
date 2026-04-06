//! Q4_0 GPU Adapter - Direct Quantized Inference (PMAT-803)
//!
//! Keeps weights in Q4_0 format on GPU, using `q4_0_gemv_into` kernels.
//! This eliminates the 8x bandwidth overhead of F32 dequantization.
//!
//! # Performance Target
//!
//! >50 tok/s on TinyLlama-1.1B (4x improvement over F32 GPU)
//!
//! # Architecture
//!
//! ```text
//! QuantizedAprTransformerQ4 (CPU, Q4_0 weights)
//!         │
//!         ▼ AprQ4ToGpuAdapter::upload_weights()
//!         │
//! CudaExecutor (GPU, Q4_0 cached)
//!         │
//!         ▼ GpuModelQ4::forward()
//!         │
//! Logits (F32)
//! ```

#![allow(clippy::similar_names)]

use crate::apr_transformer::{AprTransformerConfig, QuantizedAprTransformerQ4};
#[cfg(feature = "cuda")]
use crate::cuda::CudaExecutor;
use crate::error::{RealizarError, Result};
#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

/// Q4_0 quantization type ID (GGML format)
#[cfg(feature = "cuda")]
const Q4_0_TYPE: u32 = 2;

/// Adapter for uploading Q4_0 APR models to GPU
///
/// Unlike `AprToGpuAdapter`, this keeps weights quantized on GPU.
pub struct AprQ4ToGpuAdapter;

impl AprQ4ToGpuAdapter {
    /// Upload Q4_0 weights to CUDA executor
    ///
    /// Weights are cached on GPU by layer name for reuse across forward passes.
    /// No dequantization is performed - weights stay in Q4_0 format.
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer with Q4_0 weights
    /// * `executor` - CUDA executor with weight cache
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    ///
    /// # Errors
    ///
    /// Returns error if GPU upload fails.
    #[cfg(feature = "cuda")]
    pub fn upload_weights(
        apr: &QuantizedAprTransformerQ4,
        executor: &mut CudaExecutor,
    ) -> Result<usize> {
        let mut total_bytes = 0;

        // Upload layer weights (Q4_0 only - norms are kept on CPU)
        for (layer_idx, layer) in apr.layers.iter().enumerate() {
            // QKV projection
            let qkv_name = format!("layer_{layer_idx}.attn.qkv");
            let qkv_bytes = executor
                .load_quantized_weights_with_type(&qkv_name, &layer.qkv_weight.data, Q4_0_TYPE)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to upload {qkv_name}: {e}"),
                })?;
            total_bytes += qkv_bytes;

            // Output projection
            let out_name = format!("layer_{layer_idx}.attn.out");
            let out_bytes = executor
                .load_quantized_weights_with_type(
                    &out_name,
                    &layer.attn_output_weight.data,
                    Q4_0_TYPE,
                )
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to upload {out_name}: {e}"),
                })?;
            total_bytes += out_bytes;

            // FFN up projection
            let up_name = format!("layer_{layer_idx}.ffn.up");
            let up_bytes = executor
                .load_quantized_weights_with_type(&up_name, &layer.ffn_up_weight.data, Q4_0_TYPE)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to upload {up_name}: {e}"),
                })?;
            total_bytes += up_bytes;

            // FFN down projection
            let down_name = format!("layer_{layer_idx}.ffn.down");
            let down_bytes = executor
                .load_quantized_weights_with_type(
                    &down_name,
                    &layer.ffn_down_weight.data,
                    Q4_0_TYPE,
                )
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to upload {down_name}: {e}"),
                })?;
            total_bytes += down_bytes;

            // FFN gate projection (optional, for SwiGLU)
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let gate_name = format!("layer_{layer_idx}.ffn.gate");
                let gate_bytes = executor
                    .load_quantized_weights_with_type(&gate_name, &gate_weight.data, Q4_0_TYPE)
                    .map_err(|e| RealizarError::GpuError {
                        reason: format!("Failed to upload {gate_name}: {e}"),
                    })?;
                total_bytes += gate_bytes;
            }
        }

        // Upload LM head (Q4_0)
        let lm_head_bytes = executor
            .load_quantized_weights_with_type("lm_head", &apr.lm_head_weight.data, Q4_0_TYPE)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to upload lm_head: {e}"),
            })?;
        total_bytes += lm_head_bytes;

        // PAR-023: Upload norm weights to GPU for GPU-resident RMSNorm
        // This eliminates CPU roundtrips for normalization operations
        for (layer_idx, layer) in apr.layers.iter().enumerate() {
            // Attention norm
            let attn_norm_name = format!("apr.layer_{layer_idx}.attn_norm");
            let attn_norm_bytes = executor
                .cache_rmsnorm_gamma(&attn_norm_name, &layer.attn_norm_weight)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to cache {attn_norm_name}: {e}"),
                })?;
            total_bytes += attn_norm_bytes;

            // FFN norm (use ones if not present)
            let ffn_norm_name = format!("apr.layer_{layer_idx}.ffn_norm");
            let ffn_norm = layer
                .ffn_norm_weight
                .as_ref()
                .map_or_else(|| vec![1.0f32; apr.config.hidden_dim], Clone::clone);
            let ffn_norm_bytes = executor
                .cache_rmsnorm_gamma(&ffn_norm_name, &ffn_norm)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to cache {ffn_norm_name}: {e}"),
                })?;
            total_bytes += ffn_norm_bytes;
        }

        // Output norm
        let output_norm_bytes = executor
            .cache_rmsnorm_gamma("apr.output_norm", &apr.output_norm_weight)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to cache apr.output_norm: {e}"),
            })?;
        total_bytes += output_norm_bytes;

        Ok(total_bytes)
    }

    /// Create a GPU model wrapper for Q4_0 inference
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer (for config, embeddings, and norm weights)
    ///
    /// # Returns
    ///
    /// `GpuModelQ4` ready for inference
    #[must_use]
    pub fn create_model(apr: &QuantizedAprTransformerQ4) -> GpuModelQ4 {
        // Extract layer norms (small, keep on CPU)
        let layer_norms: Vec<LayerNorms> = apr
            .layers
            .iter()
            .map(|layer| LayerNorms {
                attn_norm: layer.attn_norm_weight.clone(),
                ffn_norm: layer
                    .ffn_norm_weight
                    .clone()
                    .unwrap_or_else(|| vec![1.0; apr.config.hidden_dim]),
            })
            .collect();

        GpuModelQ4 {
            config: apr.config.clone(),
            token_embedding: apr.token_embedding.clone(),
            output_norm_weight: apr.output_norm_weight.clone(),
            layer_norms,
            num_layers: apr.layers.len(),
            has_gate: apr
                .layers
                .first()
                .is_some_and(|l| l.ffn_gate_weight.is_some()),
        }
    }
}

/// Layer normalization weights (small, kept on CPU)
#[derive(Debug, Clone)]
pub struct LayerNorms {
    /// Attention norm weight
    pub attn_norm: Vec<f32>,
    /// FFN norm weight
    pub ffn_norm: Vec<f32>,
}

/// GPU model for Q4_0 inference
///
/// Lightweight wrapper that holds config and uses CudaExecutor for computation.
/// Large Q4_0 weights are stored on GPU, small norm weights on CPU.
#[derive(Debug, Clone)]
pub struct GpuModelQ4 {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding (F32, CPU copy for fast lookup)
    pub token_embedding: Vec<f32>,
    /// Output norm weight (F32, CPU)
    pub output_norm_weight: Vec<f32>,
    /// Per-layer norm weights (F32, CPU)
    pub layer_norms: Vec<LayerNorms>,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Whether model uses SwiGLU (has gate projection)
    pub has_gate: bool,
}

/// Helper to download GPU buffer to host Vec
#[cfg(feature = "cuda")]
fn gpu_to_host(buf: &GpuBuffer<f32>) -> Result<Vec<f32>> {
    let mut host = vec![0.0f32; buf.len()];
    buf.copy_to_host(&mut host)
        .map_err(|e| RealizarError::GpuError {
            reason: format!("GPU->CPU copy failed: {e}"),
        })?;
    Ok(host)
}

include!("apr_q4_apply_rope_gpu.rs");
include!("activation.rs");
