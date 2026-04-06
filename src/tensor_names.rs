#![allow(dead_code)]
// GH-311: Generated from tensor-names-v1.yaml by build.rs.
// The include! pulls in the generated enums, normalize_architecture(),
// and name/template lookup functions.
//
// Fallback: if build.rs can't find the YAML (CI/crates.io), it uses
// tensor_names_fallback.rs instead.
include!(concat!(env!("OUT_DIR"), "/tensor_names_generated.rs"));

use crate::error::{RealizarError, Result};
use crate::safetensors_infer::TensorSource;

/// Resolve a global tensor from a source using contract-driven name lookup.
///
/// Tries architecture-specific names first, then GGUF fallbacks, then
/// bare names (stripping "model." prefix).
///
/// # Errors
///
/// Returns error listing all attempted names if tensor is not found.
pub(crate) fn resolve_global<S: TensorSource>(
    source: &S,
    arch: &str,
    role: GlobalTensorRole,
) -> Result<Vec<f32>> {
    contract_pre_name_bijection!();
    let arch_key = normalize_architecture(arch);
    let mut tried = Vec::new();

    // 1. Architecture-specific names
    for name in global_names(arch_key, role) {
        if let Ok(t) = source.get_tensor_auto(name) {
            return Ok(t);
        }
        tried.push(*name);
    }

    // 2. GGUF fallback names
    for name in global_fallback_names(role) {
        if let Ok(t) = source.get_tensor_auto(name) {
            return Ok(t);
        }
        tried.push(*name);
    }

    // 3. Bare names (strip "model." prefix)
    for name in global_names(arch_key, role) {
        if let Some(bare) = name.strip_prefix("model.") {
            if let Ok(t) = source.get_tensor_auto(bare) {
                return Ok(t);
            }
            tried.push(bare);
        }
    }

    // Diagnostic error
    let available = source.tensor_names();
    let sample: Vec<&str> = available.iter().take(5).copied().collect();
    Err(RealizarError::UnsupportedOperation {
        operation: "tensor_names::resolve_global".to_string(),
        reason: format!(
            "Tensor not found for {:?} (arch='{}'). Tried: {:?}. \
             Available tensors ({} total): {:?}{}",
            role,
            arch,
            tried,
            available.len(),
            sample,
            if available.len() > 5 { ", ..." } else { "" }
        ),
    })
}

/// Resolve a global tensor, returning None if not found.
pub(crate) fn resolve_global_optional<S: TensorSource>(
    source: &S,
    arch: &str,
    role: GlobalTensorRole,
) -> Option<Vec<f32>> {
    resolve_global(source, arch, role).ok()
}

/// Check if a global tensor exists in the source.
pub(crate) fn has_global<S: TensorSource>(source: &S, arch: &str, role: GlobalTensorRole) -> bool {
    let arch_key = normalize_architecture(arch);

    for name in global_names(arch_key, role) {
        if source.has_tensor(name) {
            return true;
        }
    }
    for name in global_fallback_names(role) {
        if source.has_tensor(name) {
            return true;
        }
    }
    // Bare names
    for name in global_names(arch_key, role) {
        if let Some(bare) = name.strip_prefix("model.") {
            if source.has_tensor(bare) {
                return true;
            }
        }
    }
    false
}

/// Resolve a per-layer tensor from a source using contract-driven name lookup.
///
/// Substitutes `{n}` in templates with the layer index.
///
/// # Errors
///
/// Returns error listing all attempted names if tensor is not found.
pub(crate) fn resolve_layer<S: TensorSource>(
    source: &S,
    arch: &str,
    layer_idx: usize,
    role: LayerTensorRole,
) -> Result<Vec<f32>> {
    let arch_key = normalize_architecture(arch);
    let mut tried = Vec::new();

    // 1. Architecture-specific templates
    for template in layer_templates(arch_key, role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if let Ok(t) = source.get_tensor_auto(&name) {
            return Ok(t);
        }
        tried.push(name);
    }

    // 2. GGUF fallback templates
    for template in layer_fallback_templates(role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if let Ok(t) = source.get_tensor_auto(&name) {
            return Ok(t);
        }
        tried.push(name);
    }

    // 3. Bare names (strip "model." prefix)
    for template in layer_templates(arch_key, role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if let Some(bare) = name.strip_prefix("model.") {
            if let Ok(t) = source.get_tensor_auto(bare) {
                return Ok(t);
            }
            tried.push(bare.to_string());
        }
    }

    // Diagnostic error
    let available = source.tensor_names();
    let sample: Vec<&str> = available.iter().take(5).copied().collect();
    Err(RealizarError::UnsupportedOperation {
        operation: "tensor_names::resolve_layer".to_string(),
        reason: format!(
            "Tensor not found for {:?} layer {} (arch='{}'). Tried: {:?}. \
             Available tensors ({} total): {:?}{}",
            role,
            layer_idx,
            arch,
            tried,
            available.len(),
            sample,
            if available.len() > 5 { ", ..." } else { "" }
        ),
    })
}

/// Resolve a per-layer tensor, returning None if not found.
pub(crate) fn resolve_layer_optional<S: TensorSource>(
    source: &S,
    arch: &str,
    layer_idx: usize,
    role: LayerTensorRole,
) -> Option<Vec<f32>> {
    resolve_layer(source, arch, layer_idx, role).ok()
}

/// Check if a per-layer tensor exists in the source.
pub(crate) fn has_layer<S: TensorSource>(
    source: &S,
    arch: &str,
    layer_idx: usize,
    role: LayerTensorRole,
) -> bool {
    let arch_key = normalize_architecture(arch);

    for template in layer_templates(arch_key, role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if source.has_tensor(&name) {
            return true;
        }
    }
    for template in layer_fallback_templates(role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if source.has_tensor(&name) {
            return true;
        }
    }
    // Bare names
    for template in layer_templates(arch_key, role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if let Some(bare) = name.strip_prefix("model.") {
            if source.has_tensor(bare) {
                return true;
            }
        }
    }
    false
}

/// Check if a fused tensor exists in the source.
pub(crate) fn has_fused<S: TensorSource>(
    source: &S,
    arch: &str,
    layer_idx: usize,
    role: FusedTensorRole,
) -> bool {
    let arch_key = normalize_architecture(arch);

    for template in fused_templates(arch_key, role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if source.has_tensor(&name) {
            return true;
        }
    }
    for template in fused_fallback_templates(role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if source.has_tensor(&name) {
            return true;
        }
    }
    false
}

/// Resolve a fused tensor (e.g., combined QKV) from a source.
///
/// Returns None if no fused tensor exists for this architecture.
pub(crate) fn resolve_fused<S: TensorSource>(
    source: &S,
    arch: &str,
    layer_idx: usize,
    role: FusedTensorRole,
) -> Option<Vec<f32>> {
    let arch_key = normalize_architecture(arch);

    // 1. Architecture-specific templates
    for template in fused_templates(arch_key, role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if let Ok(t) = source.get_tensor_auto(&name) {
            return Some(t);
        }
    }

    // 2. GGUF fallback templates
    for template in fused_fallback_templates(role) {
        let name = template.replace("{n}", &layer_idx.to_string());
        if let Ok(t) = source.get_tensor_auto(&name) {
            return Some(t);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // FALSIFY-TNAME-004: PhiForCausalLM maps to phi2, Phi3ForCausalLM maps to phi
    #[test]
    fn test_phi_architecture_distinction() {
        assert_eq!(normalize_architecture("PhiForCausalLM"), "phi2");
        assert_eq!(normalize_architecture("Phi3ForCausalLM"), "phi");
        assert_eq!(normalize_architecture("Phi3SmallForCausalLM"), "phi");
    }

    // FALSIFY-TNAME-005: Unknown architecture defaults to llama
    #[test]
    fn test_unknown_architecture_defaults_to_llama() {
        assert_eq!(normalize_architecture("FutureArch2027"), "llama");
        assert_eq!(normalize_architecture(""), "llama");
        assert_eq!(normalize_architecture("SomeRandomModel"), "llama");
    }

    // FALSIFY-TNAME-006: GPT-2 bare names
    #[test]
    fn test_gpt2_global_names() {
        let names = global_names("gpt2", GlobalTensorRole::Embedding);
        assert!(names.contains(&"wte.weight"));
        // GPT-2 names should NOT have "model." prefix
        assert!(!names.iter().any(|n| n.starts_with("model.")));
    }

    // FALSIFY-TNAME-007: Fused QKV for GPT-2
    #[test]
    fn test_gpt2_fused_qkv() {
        let fused = fused_templates("gpt2", FusedTensorRole::FusedQkv);
        assert!(!fused.is_empty(), "GPT-2 should have fused QKV templates");

        // GPT-2 should NOT have separate Q/K/V templates
        let q_templates = layer_templates("gpt2", LayerTensorRole::QProjWeight);
        assert!(
            q_templates.is_empty(),
            "GPT-2 should not have separate Q template"
        );
    }

    // FALSIFY-TNAME-001: Architecture map includes all referenced keys
    #[test]
    fn test_architecture_map_completeness() {
        let known_archs = [
            "llama", "qwen2", "qwen3", "mistral", "gemma", "phi", "phi2", "deepseek", "gpt2",
            "gpt_neox", "bert", "openelm", "falcon", "stablelm",
        ];
        for arch in known_archs {
            // Verify that at least one of the global roles has entries for this arch
            let embed = global_names(arch, GlobalTensorRole::Embedding);
            let norm = global_names(arch, GlobalTensorRole::OutputNormWeight);
            assert!(
                !embed.is_empty() || !norm.is_empty(),
                "Architecture '{}' has no global names defined",
                arch
            );
        }
    }

    // FALSIFY-TNAME-003: Required roles have fallbacks
    #[test]
    fn test_required_roles_have_fallbacks() {
        // Embedding is required — must have fallback
        let fb = global_fallback_names(GlobalTensorRole::Embedding);
        assert!(!fb.is_empty(), "Embedding must have fallback names");

        // Output norm is required — must have fallback
        let fb = global_fallback_names(GlobalTensorRole::OutputNormWeight);
        assert!(!fb.is_empty(), "OutputNormWeight must have fallback names");

        // Layer roles: attn_norm, q_proj, etc. must have fallbacks
        let fb = layer_fallback_templates(LayerTensorRole::AttnNormWeight);
        assert!(
            !fb.is_empty(),
            "AttnNormWeight must have fallback templates"
        );

        let fb = layer_fallback_templates(LayerTensorRole::FfnUpWeight);
        assert!(!fb.is_empty(), "FfnUpWeight must have fallback templates");
    }

    // Verify LLaMA names match the existing hardcoded values
    #[test]
    fn test_llama_names_backward_compatible() {
        let embed = global_names("llama", GlobalTensorRole::Embedding);
        assert!(embed.contains(&"model.embed_tokens.weight"));

        let norm = global_names("llama", GlobalTensorRole::OutputNormWeight);
        assert!(norm.contains(&"model.norm.weight"));

        let q = layer_templates("llama", LayerTensorRole::QProjWeight);
        assert!(q.contains(&"model.layers.{n}.self_attn.q_proj.weight"));
    }

    // Verify Phi-2 uses fc1/fc2 instead of gate_proj/up_proj
    #[test]
    fn test_phi2_mlp_names() {
        let up = layer_templates("phi2", LayerTensorRole::FfnUpWeight);
        assert!(
            up.iter().any(|t| t.contains("fc1")),
            "Phi-2 should use fc1 for FFN up: {:?}",
            up
        );

        let down = layer_templates("phi2", LayerTensorRole::FfnDownWeight);
        assert!(
            down.iter().any(|t| t.contains("fc2")),
            "Phi-2 should use fc2 for FFN down: {:?}",
            down
        );

        let gate = layer_templates("phi2", LayerTensorRole::FfnGateWeight);
        assert!(gate.is_empty(), "Phi-2 should have no gate projection");
    }

    // Verify Phi-2 output norm uses final_layernorm
    #[test]
    fn test_phi2_output_norm() {
        let norm = global_names("phi2", GlobalTensorRole::OutputNormWeight);
        assert!(
            norm.iter().any(|n| n.contains("final_layernorm")),
            "Phi-2 should use final_layernorm: {:?}",
            norm
        );
    }

    // Verify GPT-NeoX fused QKV
    #[test]
    fn test_gpt_neox_fused_qkv() {
        let fused = fused_templates("gpt_neox", FusedTensorRole::FusedQkv);
        assert!(
            fused.iter().any(|t| t.contains("query_key_value")),
            "GPT-NeoX should have query_key_value fused template"
        );
    }

    // Verify all HF model class names map correctly
    #[test]
    fn test_hf_class_name_mapping() {
        assert_eq!(normalize_architecture("LlamaForCausalLM"), "llama");
        assert_eq!(normalize_architecture("Qwen2ForCausalLM"), "qwen2");
        assert_eq!(normalize_architecture("Qwen3ForCausalLM"), "qwen3");
        assert_eq!(normalize_architecture("MistralForCausalLM"), "mistral");
        assert_eq!(normalize_architecture("GemmaForCausalLM"), "gemma");
        assert_eq!(normalize_architecture("GPT2LMHeadModel"), "gpt2");
        assert_eq!(normalize_architecture("GPTNeoXForCausalLM"), "gpt_neox");
        assert_eq!(normalize_architecture("BertModel"), "bert");
        assert_eq!(normalize_architecture("FalconForCausalLM"), "falcon");
    }
}
