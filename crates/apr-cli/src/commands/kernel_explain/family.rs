//! Family YAML parsing, constraints extraction, and kernel class derivation.

use super::{Constraints, FamilyInfo, KernelClass, FAMILY_YAMLS};

/// Extract a YAML string value for a key from raw YAML text (simple line-based parse).
/// Avoids needing serde_yaml for the family constraint extraction.
pub(crate) fn yaml_str(text: &str, key: &str) -> Option<String> {
    let search = format!("{key}:");
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&search) {
            let val = trimmed[search.len()..].trim();
            // Strip inline comments (# ...)
            let val = val.split('#').next().unwrap_or(val).trim();
            // Strip quotes
            let val = val.trim_matches('"').trim_matches('\'');
            if val.is_empty() || val == "null" {
                return None;
            }
            return Some(val.to_string());
        }
    }
    None
}

/// Extract a YAML boolean value for a key.
pub(crate) fn yaml_bool(text: &str, key: &str) -> bool {
    yaml_str(text, key).map_or(false, |v| v == "true")
}

/// Extract YAML list items (lines starting with "  - ").
pub(crate) fn yaml_list(text: &str, key: &str) -> Vec<String> {
    let search = format!("{key}:");
    let mut in_section = false;
    let mut items = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&search) {
            in_section = true;
            continue;
        }
        if in_section {
            if let Some(item) = trimmed.strip_prefix("- ") {
                items.push(item.trim_matches('"').trim_matches('\'').to_string());
            } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                break;
            }
        }
    }
    items
}

/// Extract the constraints section from family YAML text.
pub(crate) fn extract_constraints(yaml_text: &str) -> Constraints {
    // Find the constraints: section and extract from there
    let constraints_section = yaml_text
        .find("\nconstraints:")
        .map(|pos| &yaml_text[pos..])
        .unwrap_or("");

    Constraints {
        attention_type: yaml_str(constraints_section, "attention_type").unwrap_or_default(),
        activation: yaml_str(constraints_section, "activation").unwrap_or_default(),
        norm_type: yaml_str(constraints_section, "norm_type").unwrap_or_default(),
        mlp_type: yaml_str(constraints_section, "mlp_type").unwrap_or_default(),
        positional_encoding: yaml_str(constraints_section, "positional_encoding")
            .unwrap_or_default(),
        has_bias: yaml_bool(constraints_section, "has_bias"),
        tied_embeddings: yaml_bool(constraints_section, "tied_embeddings"),
    }
}

/// Derive kernel class from constraints (pure function).
pub(crate) fn derive_kernel_class(c: &Constraints) -> KernelClass {
    let attn = c.attention_type.as_str();
    let norm = c.norm_type.as_str();
    let act = c.activation.as_str();
    let mlp = c.mlp_type.as_str();
    let pos = c.positional_encoding.as_str();

    // Class A: GQA/MHA + RMSNorm + SiLU + SwiGLU + RoPE
    // MHA is a degenerate case of GQA (kv_heads == q_heads), identical kernel dispatch
    if (attn == "gqa" || attn == "mha")
        && norm == "rmsnorm"
        && act == "silu"
        && mlp == "swiglu"
        && pos == "rope"
    {
        return KernelClass::A;
    }
    // Class F: RMSNorm + GELU + GatedMlp + RoPE (check before B/D)
    if norm == "rmsnorm" && act == "gelu" && mlp == "gated_mlp" && pos == "rope" {
        return KernelClass::F;
    }
    // Class B: MHA + LayerNorm + GELU
    if attn == "mha" && norm == "layernorm" && act == "gelu" {
        return KernelClass::B;
    }
    // Class C: MQA + LayerNorm + GELU + ALiBi
    if attn == "mqa" && norm == "layernorm" && act == "gelu" && pos == "alibi" {
        return KernelClass::C;
    }
    // Class D: mixed LayerNorm variants with non-standard combos
    if norm == "layernorm" && (attn == "gqa" || act == "silu") {
        return KernelClass::D;
    }
    // SSM: State Space Models (no attention mechanism)
    if attn == "ssm" {
        return KernelClass::Ssm;
    }
    // Linear: Linear attention (RWKV WKV recurrence, no softmax)
    if attn == "linear" {
        return KernelClass::Linear;
    }
    // Class E: MoE (would need num_experts field — not in current constraints)
    // Fall through to Unknown for now

    KernelClass::Unknown
}

/// Load all family info from embedded YAML.
pub fn load_families() -> Vec<FamilyInfo> {
    FAMILY_YAMLS
        .iter()
        .map(|(name, yaml_text)| {
            let constraints = extract_constraints(yaml_text);
            let kernel_class = derive_kernel_class(&constraints);
            let display_name =
                yaml_str(yaml_text, "display_name").unwrap_or_else(|| name.to_string());
            let architectures = yaml_list(yaml_text, "architectures");

            FamilyInfo {
                family: (*name).to_string(),
                display_name,
                architectures,
                constraints,
                kernel_class,
            }
        })
        .collect()
}
