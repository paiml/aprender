//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{Architecture, QuantizationType};
use crate::format::gguf::GgufReader;
use crate::format::layout_contract::contract;
use crate::serialization::safetensors::{
    save_safetensors, save_safetensors_typed, save_safetensors_with_metadata,
    save_safetensors_with_metadata_typed, UserMetadata,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// Import shared functions from parent module
use super::{
    calculate_tensor_size, load_model_tensors_provenance, map_tensor_names, quantize_tensors,
    NativeF32Tensors,
};
// NOTE: quantize_q4_k_matrix imported via super:: through mod.rs

// GH-202 FIX: Removed transpose_f32_rowmajor_to_colmajor.
// GGML data layout is C row-major with reversed shape [ne0, ne1].
// No data transposition is needed for GGUF export — only shape reversal.

/// GH-236: Resolve architecture name from APR metadata.
///
/// Checks `architecture` field first, then falls back to `model_type` (for models
/// imported without explicit architecture, e.g. GPT-2 from SafeTensors), then
/// defaults to "qwen2" as the most common architecture.
fn resolve_architecture(apr_metadata: &crate::format::v2::AprV2Metadata) -> &str {
    apr_metadata
        .architecture
        .as_deref()
        .or_else(|| {
            let mt = &apr_metadata.model_type;
            if mt.is_empty() || mt == "unknown" {
                None
            } else {
                Some(mt.as_str())
            }
        })
        .unwrap_or("qwen2")
}

/// GH-277: Resolve GGUF pre-tokenizer type from architecture and model metadata.
///
/// The `tokenizer.ggml.pre` field identifies the pre-tokenizer regex patterns
/// used by llama.cpp for text splitting before BPE. This is NOT the same as
/// the model architecture.
///
/// Reference: llama.cpp/convert_hf_to_gguf.py `get_vocab_base_pre()`
fn resolve_pre_tokenizer_type(arch: &str, model_name: &str) -> &'static str {
    let name_lower = model_name.to_lowercase();
    // SmolLM models use "default" pre-tokenizer despite llama architecture
    if name_lower.contains("smollm") {
        return "default";
    }
    match arch {
        "gpt2" => "gpt-2",
        "qwen2" | "qwen2.5" | "qwen" => "qwen2",
        "llama" | "mistral" => "default",
        "phi" | "phi3" => "default",
        "gemma" | "gemma2" => "default",
        "deepseek" | "deepseek2" => "deepseek",
        "starcoder" | "starcoder2" => "starcoder",
        _ => "default", // Safe fallback per llama.cpp convention
    }
}

/// GH-277: Check if architecture uses RoPE (vs learned position embeddings).
///
/// GPT-2 and StarCoder use learned position embeddings, not RoPE.
/// All other transformer architectures in the GGUF ecosystem use RoPE.
fn uses_rope(arch: &str) -> bool {
    !matches!(arch, "gpt2" | "starcoder")
}

/// GH-277: Contract-driven APR→GGUF tensor name mapping.
///
/// Uses the model family contract's `tensor_template` (APR names) and
/// `gguf_tensor_template` (GGUF names) to build a mapping. This replaces
/// the old hardcoded `hf_to_gguf_name()` function.
///
/// Returns a `GgufNameMapper` that converts APR tensor names to GGUF names.
/// Tensors with `None` in gguf_tensor_template are skipped.
struct GgufNameMapper {
    /// Global tensor mapping: APR name → GGUF name
    global: std::collections::HashMap<String, String>,
    /// Per-layer suffix mapping: APR suffix (after "model.layers.{n}.") → GGUF suffix
    per_layer: std::collections::HashMap<String, String>,
    /// Per-layer suffixes that should be skipped
    skip_suffixes: std::collections::HashSet<String>,
    /// GH-277: If true, transpose 2D Conv1D weight tensors to Linear layout during export.
    transpose_weights: bool,
    /// GH-277: Fusion rules — concatenate multiple APR per-layer tensors into one GGUF tensor.
    /// Each entry: (gguf_suffix, [(apr_suffix, apr_role), ...])
    fusion_rules: Vec<FusionExportRule>,
}

/// A resolved fusion rule: maps APR per-layer suffixes to a single GGUF suffix.
struct FusionExportRule {
    /// GGUF suffix for the fused tensor (e.g., "attn_qkv.weight")
    gguf_suffix: String,
    /// APR per-layer suffixes to concatenate, in order
    apr_suffixes: Vec<String>,
}

/// Build global tensor name mappings (embedding, lm_head, norms, position embedding).
fn build_global_mappings(
    tt: &crate::format::model_family::TensorTemplate,
    gt: &crate::format::model_family::GgufTensorTemplate,
) -> std::collections::HashMap<String, String> {
    let mut global = std::collections::HashMap::new();

    if let Some(gguf_name) = &gt.embedding {
        global.insert(tt.embedding.clone(), gguf_name.clone());
    }
    if let (Some(apr_name), Some(gguf_name)) = (&tt.lm_head, &gt.lm_head) {
        global.insert(apr_name.clone(), gguf_name.clone());
    }
    if let (Some(apr_name), Some(gguf_name)) = (&tt.final_norm, &gt.final_norm_weight) {
        global.insert(apr_name.clone(), gguf_name.clone());
    }
    if let (Some(apr_norm), Some(gguf_bias)) = (&tt.final_norm, &gt.final_norm_bias) {
        let apr_bias = apr_norm.replace(".weight", ".bias");
        global.insert(apr_bias, gguf_bias.clone());
    }
    if let Some(gguf_name) = &gt.position_embedding {
        if let Some(Some(apr_name)) = tt.per_layer.get("position_embedding") {
            global.insert(apr_name.clone(), gguf_name.clone());
        } else {
            global.insert(
                "model.position_embedding.weight".to_string(),
                gguf_name.clone(),
            );
        }
    }
    global
}

/// Resolve fusion rules from GGUF template against tensor template to get APR suffixes.
fn resolve_fusion_rules(
    tt: &crate::format::model_family::TensorTemplate,
    gt: &crate::format::model_family::GgufTensorTemplate,
) -> Vec<FusionExportRule> {
    gt.fuse
        .iter()
        .filter_map(|rule| {
            let apr_suffixes: Vec<String> = rule
                .source_roles
                .iter()
                .filter_map(|role| {
                    tt.per_layer.get(role.as_str()).and_then(|opt| {
                        opt.as_ref().map(|pat| {
                            pat.strip_prefix("model.layers.{n}.")
                                .unwrap_or(pat)
                                .to_string()
                        })
                    })
                })
                .collect();

            if apr_suffixes.len() == rule.source_roles.len() {
                Some(FusionExportRule {
                    gguf_suffix: rule.gguf_suffix.clone(),
                    apr_suffixes,
                })
            } else {
                eprintln!(
                    "[GH-277] WARNING: Fusion rule `{}` has unresolved source roles, skipping",
                    rule.gguf_suffix
                );
                None
            }
        })
        .collect()
}

impl GgufNameMapper {
    /// Build a mapper from a model family config's tensor template + gguf template.
    fn from_contract(config: &crate::format::model_family::ModelFamilyConfig) -> Self {
        let tt = &config.tensor_template;
        let gt = &config.gguf_tensor_template;

        let global = build_global_mappings(tt, gt);
        let mut per_layer = std::collections::HashMap::new();
        let mut skip_suffixes = std::collections::HashSet::new();

        for (role, apr_pattern_opt) in &tt.per_layer {
            let Some(apr_pattern) = apr_pattern_opt else {
                continue;
            };
            if role == "position_embedding" {
                continue;
            }

            let apr_suffix = apr_pattern
                .strip_prefix("model.layers.{n}.")
                .unwrap_or(apr_pattern);

            match gt.per_layer.get(role) {
                Some(Some(gguf_suffix)) => {
                    per_layer.insert(apr_suffix.to_string(), gguf_suffix.clone());
                }
                Some(None) => {
                    skip_suffixes.insert(apr_suffix.to_string());
                }
                None => {}
            }
        }

        let fusion_rules = resolve_fusion_rules(tt, gt);

        Self {
            global,
            per_layer,
            skip_suffixes,
            transpose_weights: gt.transpose_weights,
            fusion_rules,
        }
    }

    /// Map an APR tensor name to its GGUF equivalent.
    /// Returns `None` if the tensor should be skipped (e.g., causal attention mask).
    fn map_name(&self, apr_name: &str) -> Option<String> {
        // Check global mappings first
        if let Some(gguf_name) = self.global.get(apr_name) {
            return Some(gguf_name.clone());
        }

        // Check per-layer: extract layer number and suffix
        if let Some(rest) = apr_name.strip_prefix("model.layers.") {
            if let Some(dot_pos) = rest.find('.') {
                let layer_num = &rest[..dot_pos];
                let suffix = &rest[dot_pos + 1..];

                // Check skip list
                if self.skip_suffixes.contains(suffix) {
                    return None;
                }

                // Look up suffix mapping
                if let Some(gguf_suffix) = self.per_layer.get(suffix) {
                    return Some(format!("blk.{layer_num}.{gguf_suffix}"));
                }
            }
        }

        // Fallback: pass through with warning
        eprintln!(
            "[GH-277] WARNING: No GGUF mapping for tensor '{}', passing through",
            apr_name
        );
        Some(apr_name.to_string())
    }

    /// GH-277: Get fusion rules for this mapper.
    fn fusion_rules(&self) -> &[FusionExportRule] {
        &self.fusion_rules
    }

    /// GH-277: Whether to transpose 2D weight tensors from Conv1D to Linear layout.
    fn needs_transpose(&self) -> bool {
        self.transpose_weights
    }
}

/// Build a `GgufNameMapper` from architecture name by looking up the family contract.
fn build_gguf_mapper(arch: &str) -> GgufNameMapper {
    let registry = crate::format::model_family::build_default_registry();

    // Try to find the family by architecture name
    let family = registry
        .get(arch)
        .or_else(|| registry.detect_from_model_type(arch));

    match family {
        Some(f) => {
            let config = f.config();
            if config.gguf_tensor_template.embedding.is_some() {
                eprintln!(
                    "[GH-277] Using contract-driven GGUF mapping for family '{}'",
                    config.family
                );
                GgufNameMapper::from_contract(config)
            } else {
                eprintln!(
                    "[GH-277] Family '{}' has no gguf_tensor_template, using legacy mapping",
                    config.family
                );
                build_legacy_mapper()
            }
        }
        None => {
            eprintln!(
                "[GH-277] No family contract for arch '{}', using legacy mapping",
                arch
            );
            build_legacy_mapper()
        }
    }
}

/// Build a legacy mapper that replicates the old `hf_to_gguf_name()` behavior
/// for architectures that don't have gguf_tensor_template in their contracts.
fn build_legacy_mapper() -> GgufNameMapper {
    let mut global = std::collections::HashMap::new();
    global.insert(
        "model.embed_tokens.weight".to_string(),
        "token_embd.weight".to_string(),
    );
    global.insert("lm_head.weight".to_string(), "output.weight".to_string());
    global.insert(
        "model.norm.weight".to_string(),
        "output_norm.weight".to_string(),
    );

    let mut per_layer = std::collections::HashMap::new();
    per_layer.insert(
        "self_attn.q_proj.weight".to_string(),
        "attn_q.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.q_proj.bias".to_string(),
        "attn_q.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.k_proj.weight".to_string(),
        "attn_k.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.k_proj.bias".to_string(),
        "attn_k.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.v_proj.weight".to_string(),
        "attn_v.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.v_proj.bias".to_string(),
        "attn_v.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.o_proj.weight".to_string(),
        "attn_output.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.o_proj.bias".to_string(),
        "attn_output.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.qkv_proj.weight".to_string(),
        "attn_qkv.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.qkv_proj.bias".to_string(),
        "attn_qkv.bias".to_string(),
    );
    per_layer.insert(
        "input_layernorm.weight".to_string(),
        "attn_norm.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.q_norm.weight".to_string(),
        "attn_q_norm.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.k_norm.weight".to_string(),
        "attn_k_norm.weight".to_string(),
    );
    per_layer.insert(
        "mlp.gate_proj.weight".to_string(),
        "ffn_gate.weight".to_string(),
    );
    per_layer.insert(
        "mlp.up_proj.weight".to_string(),
        "ffn_up.weight".to_string(),
    );
    per_layer.insert(
        "mlp.down_proj.weight".to_string(),
        "ffn_down.weight".to_string(),
    );
    per_layer.insert(
        "post_attention_layernorm.weight".to_string(),
        "ffn_norm.weight".to_string(),
    );

    GgufNameMapper {
        global,
        per_layer,
        skip_suffixes: std::collections::HashSet::new(),
        transpose_weights: false,
        fusion_rules: Vec::new(),
    }
}

/// GH-277: Detect the number of layers from the tensor names.
/// Scans for "model.layers.{n}." and returns max(n) + 1.
fn detect_num_layers_from_names<'a>(names: impl Iterator<Item = &'a str>) -> usize {
    let mut max_layer = 0usize;
    for name in names {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(n) = rest[..dot_pos].parse::<usize>() {
                    max_layer = max_layer.max(n + 1);
                }
            }
        }
    }
    max_layer
}

/// Transpose a 2D f32 matrix from `[rows, cols]` to `[cols, rows]` (Conv1D → Linear).
///
/// PMAT-285: Delegates to trueno's cache-blocked transpose.
fn transpose_2d_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    trueno::blis::transpose::transpose(rows, cols, data, &mut out)
        .expect("transpose_2d_f32: dimension mismatch");
    out
}

/// Compute the concatenated shape from per-source shapes (concat along dim 0).
/// Returns `None` if shapes are incompatible or empty.
fn compute_fused_shape(shapes: &[Vec<usize>]) -> Option<Vec<usize>> {
    let first = shapes.first()?;
    match first.len() {
        2 => {
            let total_rows: usize = shapes.iter().map(|s| s[0]).sum();
            Some(vec![total_rows, first[1]])
        }
        1 => {
            let total: usize = shapes.iter().map(|s| s[0]).sum();
            Some(vec![total])
        }
        _ => None,
    }
}

/// Convert a row-major shape to GGUF ne-order (reversed for 2D).
fn shape_to_gguf(shape: &[usize]) -> Vec<u64> {
    if shape.len() == 2 {
        vec![shape[1] as u64, shape[0] as u64]
    } else {
        shape.iter().map(|&d| d as u64).collect()
    }
}


include!("export_part_02_include.rs");
include!("fusion.rs");
include!("export_part_04_include.rs");
