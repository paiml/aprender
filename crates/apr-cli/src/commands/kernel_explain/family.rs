//! Family YAML parsing, constraints extraction, and kernel class derivation.

use super::{Constraints, FamilyInfo, KernelClass, FAMILY_YAMLS};

/// Extract a YAML string value for a key from raw YAML text (simple line-based parse).
/// Avoids needing a YAML parser for the family constraint extraction.
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

/// Constraint axes the class rules look at, borrowed from `Constraints`.
struct Axes<'a> {
    attn: &'a str,
    norm: &'a str,
    act: &'a str,
    mlp: &'a str,
    pos: &'a str,
}

impl Axes<'_> {
    /// Class A: GQA/MHA + RMSNorm + SiLU + SwiGLU + RoPE.
    /// MHA is a degenerate case of GQA (kv_heads == q_heads), identical kernel dispatch.
    fn is_a(&self) -> bool {
        matches!(self.attn, "gqa" | "mha")
            && self.norm == "rmsnorm"
            && self.act == "silu"
            && self.mlp == "swiglu"
            && self.pos == "rope"
    }

    /// Class F: RMSNorm + GELU + GatedMlp + RoPE (checked before B/D).
    fn is_f(&self) -> bool {
        self.norm == "rmsnorm"
            && self.act == "gelu"
            && self.mlp == "gated_mlp"
            && self.pos == "rope"
    }

    /// Class B: MHA + LayerNorm + GELU.
    fn is_b(&self) -> bool {
        self.attn == "mha" && self.norm == "layernorm" && self.act == "gelu"
    }

    /// Class C: MQA + LayerNorm + GELU + ALiBi.
    fn is_c(&self) -> bool {
        self.attn == "mqa" && self.norm == "layernorm" && self.act == "gelu" && self.pos == "alibi"
    }

    /// Class D: mixed LayerNorm variants with non-standard combos.
    fn is_d(&self) -> bool {
        self.norm == "layernorm" && (self.attn == "gqa" || self.act == "silu")
    }
}

/// Derive kernel class from constraints (pure function).
pub(crate) fn derive_kernel_class(c: &Constraints) -> KernelClass {
    let axes = Axes {
        attn: c.attention_type.as_str(),
        norm: c.norm_type.as_str(),
        act: c.activation.as_str(),
        mlp: c.mlp_type.as_str(),
        pos: c.positional_encoding.as_str(),
    };

    // Order matters: F is checked before B/D.
    if axes.is_a() {
        KernelClass::A
    } else if axes.is_f() {
        KernelClass::F
    } else if axes.is_b() {
        KernelClass::B
    } else if axes.is_c() {
        KernelClass::C
    } else if axes.is_d() {
        KernelClass::D
    } else if axes.attn == "ssm" {
        // State Space Models (no attention mechanism)
        KernelClass::Ssm
    } else if axes.attn == "linear" {
        // Linear attention (RWKV WKV recurrence, no softmax)
        KernelClass::Linear
    } else {
        // Class E: MoE would need a num_experts field, not in the current constraints.
        KernelClass::Unknown
    }
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
