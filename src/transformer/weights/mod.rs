//! Weight loading module for transformer models
//!
//! This module provides functions to load model weights from SafeTensors files
//! and convert them to the format expected by `Transformer::from_params`.
//!
//! Supports:
//! - Qwen2/Qwen2.5 architecture
//! - LLaMA architecture
//! - Mistral architecture
//!
//! Weight name mapping follows HuggingFace conventions.

mod convert;
mod detect;
pub(crate) mod mapping;

#[cfg(test)]
mod tests;

use crate::error::{Error, Result};
use crate::Tensor;
use std::collections::HashMap;
use std::path::Path;

pub(crate) use convert::tensor_to_f32_vec;
pub(crate) use detect::{
    detect_architecture, find_safetensors_files, parse_checkpoint_step_from_path,
};
pub(crate) use mapping::map_weight_name;

/// Architecture type for weight name mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// LLaMA and LLaMA-2 models
    Llama,
    /// Qwen2 and Qwen2.5 models (includes Qwen2.5-Coder)
    Qwen2,
    /// Mistral models
    Mistral,
    /// RoBERTa / CodeBERT encoder models (ENC-006)
    RoBERTa,
    /// GGUF tensor names (token_embd, blk.N.attn_q, etc.)
    Gguf,
    /// Auto-detect from weight names
    Auto,
}

/// Load transformer weights from SafeTensors file(s)
///
/// # Arguments
///
/// * `model_path` - Path to model directory or single SafeTensors file
/// * `arch` - Model architecture (use Auto to detect from weight names)
///
/// # Returns
///
/// HashMap of parameter names mapped to Tensor values.
/// Names follow the HuggingFace LLaMA convention expected by `Transformer::from_params`.
pub fn load_safetensors_weights(
    model_path: &Path,
    arch: Architecture,
) -> Result<HashMap<String, Tensor>> {
    use safetensors::SafeTensors;

    // Find SafeTensors files
    let st_files = find_safetensors_files(model_path)?;
    if st_files.is_empty() {
        return Err(Error::ConfigError(format!(
            "No SafeTensors files found in {}",
            model_path.display()
        )));
    }

    let mut weights = HashMap::new();
    let mut detected_arch = arch;

    // Process each SafeTensors file
    for st_path in &st_files {
        let data = std::fs::read(st_path).map_err(|e| {
            Error::ConfigError(format!("Failed to read {}: {}", st_path.display(), e))
        })?;

        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            Error::ConfigError(format!("Failed to parse SafeTensors {}: {}", st_path.display(), e))
        })?;

        // Auto-detect architecture from first file
        if detected_arch == Architecture::Auto {
            detected_arch = detect_architecture(&tensors);
            eprintln!("  Detected architecture: {detected_arch:?}");
        }

        // Load and map tensors
        for name in tensors.names() {
            if let Ok(tensor_view) = tensors.tensor(name) {
                // Convert tensor to f32 values
                if let Some(values) = tensor_to_f32_vec(&tensor_view) {
                    // Map name to standard LLaMA convention
                    let mapped_name = map_weight_name(name, detected_arch);
                    let tensor = Tensor::from_vec(values, true);
                    weights.insert(mapped_name, tensor);
                }
            }
        }
    }

    eprintln!("  Loaded {} weight tensors", weights.len());
    Ok(weights)
}

/// Get expected weight count for a transformer model (minimum — without attention biases)
pub fn expected_weight_count(num_layers: usize, has_lm_head: bool) -> usize {
    // Per layer (minimum):
    //   input_layernorm.weight (1)
    //   self_attn: q_proj, k_proj, v_proj, o_proj weights (4)
    //   post_attention_layernorm.weight (1)
    //   mlp: gate_proj, up_proj, down_proj (3)
    // = 9 per layer (without biases)
    //
    // Models with attention biases (e.g. Qwen2) have 3 additional per layer:
    //   self_attn: q_proj.bias, k_proj.bias, v_proj.bias
    // = 12 per layer (with biases)
    //
    // Global:
    //   embed_tokens.weight (1)
    //   norm.weight (1)
    //   lm_head.weight (optional, 1)
    let base = 2 + (num_layers * 9);
    if has_lm_head {
        base + 1
    } else {
        base
    }
}

/// Get expected weight count including attention biases
#[allow(dead_code)]
pub fn expected_weight_count_with_biases(num_layers: usize, has_lm_head: bool) -> usize {
    let base = 2 + (num_layers * 12); // 9 weights + 3 biases per layer
    if has_lm_head {
        base + 1
    } else {
        base
    }
}

/// Validate that loaded weights match expected architecture
#[allow(clippy::implicit_hasher)]
pub fn validate_weights(weights: &HashMap<String, Tensor>, num_layers: usize) -> Result<()> {
    // Check embedding
    if !weights.contains_key("model.embed_tokens.weight") {
        return Err(Error::ConfigError("Missing model.embed_tokens.weight".into()));
    }

    // Check final norm
    if !weights.contains_key("model.norm.weight") {
        return Err(Error::ConfigError("Missing model.norm.weight".into()));
    }

    // Check each layer
    for i in 0..num_layers {
        let layer_prefix = format!("model.layers.{i}");

        // Required layer weights
        let required = [
            ".input_layernorm.weight",
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".post_attention_layernorm.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ];

        for suffix in required {
            let key = format!("{layer_prefix}{suffix}");
            if !weights.contains_key(&key) {
                return Err(Error::ConfigError(format!("Missing {key}")));
            }
        }
    }

    // Check weight count for informational purposes
    let has_lm_head = weights.contains_key("lm_head.weight");
    let expected = expected_weight_count(num_layers, has_lm_head);
    let actual = weights.len();
    if actual < expected {
        // This is a warning, not an error - some models may have extra or fewer weights
        eprintln!(
            "Warning: Expected at least {expected} weights, found {actual} (may have extra bias tensors)"
        );
    }

    Ok(())
}
