//! Kernel Explainability: static analysis of model architecture → kernel dispatch.
//!
//! Derives kernel equivalence class from family contract constraints without
//! loading the model or running inference. Pure metadata analysis.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

// ── Family YAML embedding (compile-time) ──────────────────────────────────

macro_rules! embed_family {
    ($name:expr, $path:expr) => {
        (
            $name,
            include_str!(concat!("../../../../contracts/model-families/", $path)),
        )
    };
}

const FAMILY_YAMLS: &[(&str, &str)] = &[
    embed_family!("bert", "bert.yaml"),
    embed_family!("deepseek", "deepseek.yaml"),
    embed_family!("falcon_h1", "falcon_h1.yaml"),
    embed_family!("gemma", "gemma.yaml"),
    embed_family!("gpt2", "gpt2.yaml"),
    embed_family!("llama", "llama.yaml"),
    embed_family!("mamba", "mamba.yaml"),
    embed_family!("mistral", "mistral.yaml"),
    embed_family!("moonshine", "moonshine.yaml"),
    embed_family!("openelm", "openelm.yaml"),
    embed_family!("phi", "phi.yaml"),
    embed_family!("qwen2", "qwen2.yaml"),
    embed_family!("qwen3", "qwen3.yaml"),
    embed_family!("qwen3_5", "qwen3_5.yaml"),
    embed_family!("rwkv7", "rwkv7.yaml"),
    embed_family!("whisper", "whisper.yaml"),
];

// ── Kernel contract embedding ─────────────────────────────────────────────

macro_rules! embed_contract {
    ($name:expr, $path:expr) => {
        (
            $name,
            include_str!(concat!("../../../../contracts/", $path)),
        )
    };
}

const KERNEL_CONTRACTS: &[(&str, &str)] = &[
    embed_contract!("matvec-kernel-v1", "matvec-kernel-v1.yaml"),
    embed_contract!("rope-kernel-v1", "rope-kernel-v1.yaml"),
    embed_contract!("normalization-kernel-v1", "normalization-kernel-v1.yaml"),
    embed_contract!("element-wise-ops-v1", "element-wise-ops-v1.yaml"),
    embed_contract!("softmax-kernel-v1", "softmax-kernel-v1.yaml"),
    embed_contract!("kernel-fusion-v1", "kernel-fusion-v1.yaml"),
    embed_contract!("tensor-layout-v1", "tensor-layout-v1.yaml"),
    embed_contract!("quantized-dot-product-v1", "quantized-dot-product-v1.yaml"),
    embed_contract!("transpose-kernel-v1", "transpose-kernel-v1.yaml"),
];

// ── Kernel class taxonomy (A-F) ───────────────────────────────────────────

/// Kernel equivalence class. Models in the same class dispatch identical
/// kernel pipelines, so once a representative is certified, others only
/// need dimensional smoke verification (G0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum KernelClass {
    A, // GQA + RMSNorm + SiLU + SwiGLU + RoPE
    B, // MHA + LayerNorm + GELU + absolute/none
    C, // MQA + LayerNorm + GELU + ALiBi
    D, // mixed: LayerNorm + SiLU or GQA + LayerNorm
    E, // MoE variants
    F, // RMSNorm + GELU + GatedMlp + RoPE
    Unknown,
}

impl KernelClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::A => "A (GQA + RMSNorm + SiLU + SwiGLU + RoPE)",
            Self::B => "B (MHA + LayerNorm + GELU)",
            Self::C => "C (MQA + LayerNorm + GELU + ALiBi)",
            Self::D => "D (GQA + LayerNorm + GELU/SiLU)",
            Self::E => "E (MoE + GQA + RMSNorm + SwiGLU)",
            Self::F => "F (RMSNorm + GELU + GatedMlp + RoPE)",
            Self::Unknown => "Unknown",
        }
    }

    pub fn letter(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::E => "E",
            Self::F => "F",
            Self::Unknown => "Unknown",
        }
    }
}

// ── Kernel operations ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct KernelOp {
    pub op: &'static str,
    pub kernel: &'static str,
    pub contract: &'static str,
}

fn kernel_ops_for_class(class: KernelClass) -> Vec<KernelOp> {
    let mut ops = vec![
        KernelOp {
            op: "MatVec (Q4K)",
            kernel: "fused_q4k_parallel_matvec",
            contract: "matvec-kernel-v1",
        },
        KernelOp {
            op: "MatVec (Q6K)",
            kernel: "fused_q6k_parallel_matvec",
            contract: "matvec-kernel-v1",
        },
        KernelOp {
            op: "Softmax",
            kernel: "softmax",
            contract: "softmax-kernel-v1",
        },
        KernelOp {
            op: "Kernel Fusion",
            kernel: "fused_matvec_activation",
            contract: "kernel-fusion-v1",
        },
    ];

    match class {
        KernelClass::A => {
            ops.push(KernelOp {
                op: "Attention (GQA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "rms_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "silu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "swiglu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::B => {
            ops.push(KernelOp {
                op: "Attention (MHA)",
                kernel: "mha_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "layer_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "gelu_mlp",
                contract: "element-wise-ops-v1",
            });
        }
        KernelClass::C => {
            ops.push(KernelOp {
                op: "Attention (MQA)",
                kernel: "mqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "layer_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "alibi",
                contract: "element-wise-ops-v1",
            });
        }
        KernelClass::D => {
            ops.push(KernelOp {
                op: "Attention (GQA/MHA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "layer_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "silu/gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::E => {
            ops.push(KernelOp {
                op: "Attention (GQA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "rms_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "MoE Router",
                kernel: "moe_routing",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "silu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "swiglu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::F => {
            ops.push(KernelOp {
                op: "Attention (GQA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "rms_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "gated_mlp",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::Unknown => {}
    }

    ops
}

// ── Constraints extraction ────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize)]
pub struct Constraints {
    pub attention_type: String,
    pub activation: String,
    pub norm_type: String,
    pub mlp_type: String,
    pub positional_encoding: String,
    pub has_bias: bool,
    pub tied_embeddings: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FamilyInfo {
    pub family: String,
    pub display_name: String,
    pub architectures: Vec<String>,
    pub constraints: Constraints,
    pub kernel_class: KernelClass,
}

/// Extract a YAML string value for a key from raw YAML text (simple line-based parse).
/// Avoids needing serde_yaml for the family constraint extraction.
fn yaml_str(text: &str, key: &str) -> Option<String> {
    let search = format!("{key}:");
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&search) {
            let val = trimmed[search.len()..].trim();
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
fn yaml_bool(text: &str, key: &str) -> bool {
    yaml_str(text, key).map_or(false, |v| v == "true")
}

/// Extract YAML list items (lines starting with "  - ").
fn yaml_list(text: &str, key: &str) -> Vec<String> {
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
fn extract_constraints(yaml_text: &str) -> Constraints {
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
fn derive_kernel_class(c: &Constraints) -> KernelClass {
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

/// Model types without their own family YAML that share a kernel pipeline
/// with an existing family. Maps model_type → family name.
const FAMILY_ALIASES: &[(&str, &str)] = &[
    // Class A variants (SiLU + RMSNorm + GQA/MHA + RoPE)
    ("olmo2", "llama"),     // MHA variant
    ("granite", "llama"),   // GQA variant
    ("internlm2", "llama"), // GQA variant
    ("phi3", "llama"),      // Phi-3/4: SiLU + RMSNorm + GQA + RoPE (NOT phi-2)
    ("codellama", "llama"),
    ("stablelm", "llama"),
    ("yi", "llama"),
    ("baichuan", "llama"),
    // Class D variants
    ("phi3small", "phi"), // gegelu + LayerNorm (unique, closest=phi)
    // GELU + RMSNorm variants
    ("starcoder2", "qwen2"), // GELU + RMSNorm + GQA + RoPE (closest match)
    // MoE variants — map to base family with MoE warning
    ("qwen2_moe", "qwen2"),      // MoE: Qwen2 MoE (model_type form)
    ("qwen2moe", "qwen2"),       // MoE: Qwen2 MoE (arch-stripped form)
    ("qwen3_moe", "mistral"),    // MoE: 128 experts (model_type form)
    ("qwen3moe", "mistral"),     // MoE: 128 experts (arch-stripped form)
    ("qwen3_next", "mistral"),   // MoE: 512 experts (model_type form)
    ("qwen3next", "mistral"),    // MoE: 512 experts (arch-stripped form)
    ("deepseek_v2", "deepseek"), // MoE: DeepSeek V2 with expert routing
    ("deepseekv2", "deepseek"),  // MoE: arch-stripped form
    ("mixtral", "mistral"),      // MoE: 8 experts
    // Classic falcon: LayerNorm + GELU + GQA/MHA — closest to bert (Class B)
    ("falcon", "bert"), // Falcon-7B/40B: LayerNorm + GELU (no RMSNorm, no SiLU)
];

/// Resolve a family string or architecture string to `FamilyInfo`.
pub fn resolve_family(input: &str) -> Option<FamilyInfo> {
    let lower = input.to_lowercase();
    let lower = lower.trim();
    if lower.is_empty() {
        return None;
    }

    let families = load_families();

    // Direct family name match
    if let Some(f) = families.iter().find(|f| f.family == lower) {
        return Some(f.clone());
    }

    // Alias match (model types sharing kernel pipeline with existing family)
    if let Some((_, target)) = FAMILY_ALIASES.iter().find(|(alias, _)| *alias == lower) {
        if let Some(f) = families.iter().find(|f| f.family == *target) {
            let mut aliased = f.clone();
            aliased.display_name = format!("{} (via {} kernel pipeline)", lower, f.family);
            return Some(aliased);
        }
    }

    // Architecture match (e.g., "Qwen2ForCausalLM")
    if let Some(f) = families.iter().find(|f| {
        f.architectures
            .iter()
            .any(|a| a.to_lowercase() == lower || a == input)
    }) {
        return Some(f.clone());
    }

    // Architecture string → model_type extraction → alias re-check
    // e.g., "GraniteForCausalLM" → "granite" → alias → llama
    let stripped = strip_arch_suffix(lower);
    if stripped != lower {
        // Try alias with stripped name
        if let Some((_, target)) = FAMILY_ALIASES.iter().find(|(alias, _)| *alias == stripped) {
            if let Some(f) = families.iter().find(|f| f.family == *target) {
                let mut aliased = f.clone();
                aliased.display_name = format!("{stripped} (via {target} kernel pipeline)");
                return Some(aliased);
            }
        }
        // Try direct family match with stripped name
        if let Some(f) = families.iter().find(|f| f.family == stripped) {
            return Some(f.clone());
        }
    }

    // Partial match (e.g., "qwen" matches "qwen2") — require >= 3 chars
    if lower.len() >= 3 {
        return families
            .into_iter()
            .find(|f| f.family.contains(lower) || lower.contains(&f.family));
    }

    None
}

/// Strip HuggingFace architecture suffixes to get model type.
/// E.g., "graniteforCausalLM" → "granite", "phi3smallforcausallm" → "phi3small"
fn strip_arch_suffix(s: &str) -> &str {
    // All known suffixes (lowercase). Order matters: longest first.
    const SUFFIXES: &[&str] = &["forconditionalgeneration", "forcausallm", "model"];
    for suffix in SUFFIXES {
        if let Some(prefix) = s.strip_suffix(suffix) {
            if !prefix.is_empty() {
                return prefix;
            }
        }
    }
    s
}

/// Try to resolve family from a config.json file.
pub fn resolve_from_config_json(path: &Path) -> Option<FamilyInfo> {
    let content = std::fs::read_to_string(path).ok()?;
    // Parse model_type from config.json
    let model_type = extract_json_string(&content, "model_type")?;
    resolve_family(&model_type)
}

/// Simple JSON value extraction (no serde dependency for this hot path).
/// Handles both string values ("silu") and numeric values (1e-06, 8, 1000000.0).
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{key}\"");
    let pos = json.find(&search)?;
    let after = &json[pos + search.len()..];
    // Skip whitespace and colon
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start();

    if let Some(after) = after.strip_prefix('"') {
        // Quoted string value
        let end = after.find('"')?;
        Some(after[..end].to_string())
    } else {
        // Numeric or boolean value — read until comma, newline, or }
        let end = after.find(|c: char| c == ',' || c == '\n' || c == '}' || c == ' ')?;
        let val = after[..end].trim();
        if val.is_empty() || val == "null" {
            None
        } else {
            Some(val.to_string())
        }
    }
}

/// Extract config.json fields relevant to kernel dispatch.
pub fn extract_config_mapping(path: &Path) -> BTreeMap<String, ConfigField> {
    let mut map = BTreeMap::new();
    let Ok(content) = std::fs::read_to_string(path) else {
        return map;
    };

    let fields = [
        ("model_type", "Architecture class dispatch"),
        ("hidden_act", "Activation kernel selection"),
        ("rms_norm_eps", "RMSNorm (not LayerNorm)"),
        ("layer_norm_epsilon", "LayerNorm (not RMSNorm)"),
        ("layer_norm_eps", "LayerNorm (not RMSNorm)"),
        ("norm_epsilon", "Normalization epsilon"),
        ("num_key_value_heads", "GQA vs MHA vs MQA"),
        ("num_kv_heads", "GQA vs MHA (Falcon field name)"),
        ("multi_query", "MQA flag (Falcon-7B)"),
        ("num_attention_heads", "Number of query heads"),
        ("rope_theta", "RoPE positional encoding"),
        ("intermediate_size", "MLP width (SwiGLU detection)"),
        ("hidden_size", "Model hidden dimension"),
        ("num_hidden_layers", "Transformer depth"),
        ("num_local_experts", "MoE expert routing"),
        ("num_experts", "MoE expert routing"),
        ("n_routed_experts", "MoE expert routing (DeepSeek)"),
        ("num_experts_per_tok", "MoE active experts per token"),
    ];

    for (key, rationale) in &fields {
        if let Some(val) = extract_json_string(&content, key) {
            // Enrich rationale with kernel-specific interpretation
            let enriched = enrich_rationale(key, &val, &content);
            map.insert(
                (*key).to_string(),
                ConfigField {
                    value: val,
                    rationale: enriched.unwrap_or_else(|| (*rationale).to_string()),
                },
            );
        }
    }

    map
}

/// Enrich config field rationale with kernel-specific interpretation.
fn enrich_rationale(key: &str, value: &str, json: &str) -> Option<String> {
    match key {
        "hidden_act" => match value {
            "silu" => Some("SiLU activation (not GELU)".to_string()),
            "gelu" | "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => {
                Some(format!("GELU activation: {value} (not SiLU)"))
            }
            _ => Some(format!("Activation: {value}")),
        },
        "rms_norm_eps" => Some("RMSNorm (not LayerNorm)".to_string()),
        "num_key_value_heads" => {
            let num_heads = extract_json_string(json, "num_attention_heads")
                .and_then(|v| v.parse::<u32>().ok());
            let kv_heads = value.parse::<u32>().ok();
            match (num_heads, kv_heads) {
                (Some(h), Some(kv)) if kv == 1 => Some(format!("MQA ({kv} KV head < {h} Q heads)")),
                (Some(h), Some(kv)) if kv < h => Some(format!("GQA ({kv} KV heads < {h} Q heads)")),
                (Some(h), Some(kv)) if kv == h => {
                    Some(format!("MHA ({kv} KV heads == {h} Q heads)"))
                }
                _ => None,
            }
        }
        "rope_theta" => Some("RoPE positional encoding".to_string()),
        "intermediate_size" => {
            let hidden =
                extract_json_string(json, "hidden_size").and_then(|v| v.parse::<f64>().ok());
            let inter = value.parse::<f64>().ok();
            match (hidden, inter) {
                (Some(h), Some(i)) if h > 0.0 => {
                    let ratio = i / h;
                    if ratio > 2.5 {
                        Some(format!("SwiGLU MLP ({i:.0}/{h:.0} = {ratio:.2}x)"))
                    } else {
                        Some(format!("Standard FFN ({i:.0}/{h:.0} = {ratio:.2}x)"))
                    }
                }
                _ => None,
            }
        }
        "num_local_experts" | "num_experts" | "n_routed_experts" => {
            let n: u32 = value.parse().unwrap_or(0);
            if n > 0 {
                Some(format!("MoE with {n} experts (Class E kernel routing)"))
            } else {
                None
            }
        }
        "num_experts_per_tok" => {
            let n: u32 = value.parse().unwrap_or(0);
            if n > 0 {
                Some(format!("{n} active experts per token"))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract the first architecture string from a config.json (e.g., "Starcoder2ForCausalLM").
pub fn extract_architecture_from_config(
    config_mapping: &BTreeMap<String, ConfigField>,
) -> Option<String> {
    // Check if we stored architectures — fall back to model_type
    config_mapping.get("model_type").map(|f| f.value.clone())
}

/// Detect mismatches between config.json values and family constraints.
pub fn detect_constraint_mismatches(
    family: &FamilyInfo,
    config_mapping: &BTreeMap<String, ConfigField>,
) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check activation mismatch
    if let Some(act) = config_mapping.get("hidden_act") {
        let config_act = act.value.to_lowercase();
        let family_act = family.constraints.activation.to_lowercase();
        let config_is_gelu = config_act.contains("gelu");
        let family_is_gelu = family_act.contains("gelu");
        let config_is_silu = config_act == "silu" || config_act == "swish";
        let family_is_silu = family_act == "silu" || family_act == "swish";

        if (config_is_gelu && family_is_silu) || (config_is_silu && family_is_gelu) {
            warnings.push(format!(
                "Activation mismatch: config.json has '{}' but family '{}' uses '{}'",
                act.value, family.family, family.constraints.activation
            ));
        }
        // gegelu (Gated GELU, used by Phi-3-small) is distinct from standard gelu
        if config_act == "gegelu" && family_act != "gegelu" {
            warnings.push(format!(
                "Activation variant: config.json uses 'gegelu' (Gated GELU) but family '{}' uses '{}'. Different kernel.",
                family.family, family.constraints.activation
            ));
        }
    }

    // Check normalization mismatch: config.json norm field vs family constraint
    let has_rms = config_mapping.contains_key("rms_norm_eps");
    let has_ln = config_mapping.contains_key("layer_norm_epsilon")
        || config_mapping.contains_key("layer_norm_eps")
        || config_mapping.contains_key("norm_epsilon");
    let family_norm = family.constraints.norm_type.to_lowercase();
    if has_rms && !has_ln && family_norm == "layernorm" {
        warnings.push(format!(
            "Norm mismatch: config.json has rms_norm_eps (RMSNorm) but family '{}' uses LayerNorm",
            family.family
        ));
    } else if has_ln && !has_rms && family_norm == "rmsnorm" {
        warnings.push(format!(
            "Norm mismatch: config.json has layer_norm_epsilon (LayerNorm) but family '{}' uses RMSNorm",
            family.family
        ));
    }

    // Check attention type mismatch (supports both num_key_value_heads and num_kv_heads)
    let kv_field = config_mapping
        .get("num_key_value_heads")
        .or_else(|| config_mapping.get("num_kv_heads"));
    if let Some(kv) = kv_field {
        if let Some(q) = config_mapping.get("num_attention_heads") {
            let kv_n: u32 = kv.value.parse().unwrap_or(0);
            let q_n: u32 = q.value.parse().unwrap_or(0);
            let config_attn = if kv_n == 1 {
                "mqa"
            } else if kv_n < q_n {
                "gqa"
            } else {
                "mha"
            };
            let family_attn = family.constraints.attention_type.to_lowercase();
            if config_attn != family_attn && !family_attn.is_empty() {
                warnings.push(format!(
                    "Attention mismatch: config.json implies {} but family '{}' uses {}",
                    config_attn.to_uppercase(),
                    family.family,
                    family.constraints.attention_type.to_uppercase()
                ));
            }
        }
    } else if let Some(mq) = config_mapping.get("multi_query") {
        // Falcon-7B uses multi_query: true for MQA
        if mq.value == "true" {
            let family_attn = family.constraints.attention_type.to_lowercase();
            if family_attn != "mqa" && !family_attn.is_empty() {
                warnings.push(format!(
                    "Attention mismatch: config.json has multi_query=true (MQA) but family '{}' uses {}",
                    family.family,
                    family.constraints.attention_type.to_uppercase()
                ));
            }
        }
    }

    // Check MoE: config has experts but family class is not E
    let expert_field = config_mapping
        .get("num_local_experts")
        .or_else(|| config_mapping.get("num_experts"))
        .or_else(|| config_mapping.get("n_routed_experts"));
    if let Some(ef) = expert_field {
        if family.kernel_class != KernelClass::E {
            warnings.push(format!(
                "MoE model ({} experts) mapped to non-MoE class {}. Expert routing kernel not covered.",
                ef.value,
                family.kernel_class.letter()
            ));
        }
    } else if family.kernel_class != KernelClass::E {
        // Detect MoE from alias input name or family name.
        // Only check display_name for aliases (which have "via" in the name).
        // The raw family display_name "Mistral / Mixtral" would false-positive
        // on all non-MoE mistral models.
        let dn = family.display_name.to_lowercase();
        let is_alias = dn.contains(" (via ");
        let alias_name = if is_alias {
            dn.split(" (via ").next().unwrap_or("")
        } else {
            ""
        };
        if family.family.contains("moe")
            || alias_name.contains("moe")
            || alias_name.contains("mixtral")
            || alias_name.contains("mixture")
        {
            warnings.push(format!(
                "MoE architecture detected (from name) but mapped to non-MoE class {}. Expert routing kernel not covered.",
                family.kernel_class.letter()
            ));
        }
    }

    warnings
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigField {
    pub value: String,
    pub rationale: String,
}

// ── Proof status ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ProofLevel {
    Proven,
    Tested,
    Documented,
    Unknown,
}

impl ProofLevel {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Proven => "✓",
            Self::Tested => "◉",
            Self::Documented => "○",
            Self::Unknown => "?",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Proven => "Proven",
            Self::Tested => "Tested",
            Self::Documented => "Documented",
            Self::Unknown => "Unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ContractProof {
    pub contract: String,
    pub level: ProofLevel,
    pub evidence: String,
}

/// Determine proof level for a contract by inspecting its embedded YAML.
pub fn proof_status_for_contract(contract_name: &str) -> ContractProof {
    let yaml_text = KERNEL_CONTRACTS
        .iter()
        .find(|(name, _)| *name == contract_name)
        .map(|(_, text)| *text);

    let Some(text) = yaml_text else {
        return ContractProof {
            contract: contract_name.to_string(),
            level: ProofLevel::Unknown,
            evidence: "No contract YAML found".to_string(),
        };
    };

    // Check for kani harnesses or extensive test references
    let has_kani = text.contains("kani_harness") || text.contains("kani:");
    let has_falsification = text.contains("FALSIFY-") || text.contains("falsification:");
    let has_tests_file = text.contains("tests_file:");
    let has_qa_gate = text.contains("qa_gate:");

    let (level, evidence) = if has_kani {
        (
            ProofLevel::Proven,
            "Kani harness + contract tests".to_string(),
        )
    } else if has_falsification && has_tests_file {
        (
            ProofLevel::Tested,
            format!(
                "Falsification tests in {}",
                yaml_str(text, "tests_file").unwrap_or_default()
            ),
        )
    } else if has_qa_gate || has_falsification {
        (ProofLevel::Tested, "Contract tests".to_string())
    } else {
        (
            ProofLevel::Documented,
            "Contract specification exists".to_string(),
        )
    };

    ContractProof {
        contract: contract_name.to_string(),
        level,
        evidence,
    }
}

/// Get proof status for all kernel ops in a class.
pub fn proof_status_for_class(class: KernelClass) -> Vec<ContractProof> {
    let ops = kernel_ops_for_class(class);
    let mut seen = Vec::new();
    let mut proofs = Vec::new();

    for op in &ops {
        if !seen.contains(&op.contract) {
            seen.push(op.contract);
            proofs.push(proof_status_for_contract(op.contract));
        }
    }

    proofs
}

// ── JSON output ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct KernelExplainJson {
    pub architecture: String,
    pub kernel_class: String,
    pub kernel_class_label: String,
    pub family: String,
    pub display_name: String,
    pub kernel_ops: Vec<KernelOp>,
    pub constraints: Constraints,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub config_mapping: BTreeMap<String, ConfigField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_summary: Option<ProofSummary>,
    pub layout: String,
    pub equivalence_class_families: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ProofSummary {
    pub proven: usize,
    pub tested: usize,
    pub documented: usize,
    pub unknown: usize,
    pub total: usize,
}

pub fn build_json_output(
    family: &FamilyInfo,
    config_mapping: BTreeMap<String, ConfigField>,
    show_proof: bool,
) -> KernelExplainJson {
    let ops = kernel_ops_for_class(family.kernel_class);
    let proofs = if show_proof {
        proof_status_for_class(family.kernel_class)
    } else {
        Vec::new()
    };

    let proven = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Proven)
        .count();
    let tested = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Tested)
        .count();
    let documented = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Documented)
        .count();
    let unknown = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Unknown)
        .count();

    let families = load_families();
    let equivalence_class_families: Vec<String> = families
        .iter()
        .filter(|f| f.kernel_class == family.kernel_class)
        .map(|f| f.family.clone())
        .collect();

    // Prefer actual architecture from config.json over family default
    let arch = extract_architecture_from_config(&config_mapping).unwrap_or_else(|| {
        family
            .architectures
            .first()
            .map_or("Unknown", String::as_str)
            .to_string()
    });

    let warnings = detect_constraint_mismatches(family, &config_mapping);

    KernelExplainJson {
        architecture: arch,
        kernel_class: family.kernel_class.letter().to_string(),
        kernel_class_label: family.kernel_class.label().to_string(),
        family: family.family.clone(),
        display_name: family.display_name.clone(),
        kernel_ops: ops,
        constraints: family.constraints.clone(),
        config_mapping,
        proof_summary: if show_proof {
            Some(ProofSummary {
                proven,
                tested,
                documented,
                unknown,
                total: proofs.len(),
            })
        } else {
            None
        },
        layout: "row_major".to_string(),
        equivalence_class_families,
        warnings,
    }
}

// ── Human-readable output ─────────────────────────────────────────────────

pub fn print_human_output(
    family: &FamilyInfo,
    config_mapping: &BTreeMap<String, ConfigField>,
    verbose: bool,
    show_proof: bool,
) {
    // Prefer actual architecture from config.json over family default
    let config_model_type = extract_architecture_from_config(config_mapping);
    let arch = config_model_type
        .as_deref()
        .or(family.architectures.first().map(String::as_str))
        .unwrap_or("Unknown");
    let ops = kernel_ops_for_class(family.kernel_class);

    println!("Kernel Explainability Report: {}", family.display_name);
    println!("{}", "═".repeat(50));
    println!();
    println!("Architecture:  {arch}");
    println!("Kernel Class:  {}", family.kernel_class.label());
    println!("Family:        {}", family.family);
    println!();

    // Kernel pipeline table
    println!("Kernel Pipeline ({} ops)", ops.len());
    println!(
        "┌─────────────────────────┬────────────────────────────────┬──────────────────────────┐"
    );
    println!(
        "│ Operation               │ Kernel                         │ Contract                 │"
    );
    println!(
        "├─────────────────────────┼────────────────────────────────┼──────────────────────────┤"
    );
    for op in &ops {
        println!(
            "│ {:<23} │ {:<30} │ {:<24} │",
            op.op, op.kernel, op.contract
        );
    }
    println!(
        "└─────────────────────────┴────────────────────────────────┴──────────────────────────┘"
    );

    // Config mapping
    if !config_mapping.is_empty() {
        println!();
        println!("Config.json → Kernel Mapping:");
        for (key, field) in config_mapping {
            println!("  {key}={:<18} → {}", field.value, field.rationale);
        }
    }

    // Constraints (verbose)
    if verbose {
        println!();
        println!("Constraints (from family contract):");
        println!(
            "  attention_type:      {}",
            family.constraints.attention_type
        );
        println!("  activation:          {}", family.constraints.activation);
        println!("  norm_type:           {}", family.constraints.norm_type);
        println!("  mlp_type:            {}", family.constraints.mlp_type);
        println!(
            "  positional_encoding: {}",
            family.constraints.positional_encoding
        );
        println!("  has_bias:            {}", family.constraints.has_bias);
        println!(
            "  tied_embeddings:     {}",
            family.constraints.tied_embeddings
        );
    }

    // Constraint mismatch warnings
    if !config_mapping.is_empty() {
        let mismatches = detect_constraint_mismatches(family, config_mapping);
        for warning in &mismatches {
            println!();
            eprintln!("⚠ WARNING: {warning}");
            eprintln!("  This model is mapped via alias. Kernel selection may differ from the family contract.");
        }
    }

    // Layout
    println!();
    println!("Layout: Row-major (LAYOUT-002 compliant)");
    println!("  GGUF→APR conversion transposes at import time.");
    println!("  Direct GGUF inference uses column-major kernels.");

    // Equivalence class members
    let families = load_families();
    let class_members: Vec<&str> = families
        .iter()
        .filter(|f| f.kernel_class == family.kernel_class)
        .map(|f| f.family.as_str())
        .collect();
    if class_members.len() > 1 {
        println!();
        println!(
            "Equivalence Class {}: {} families",
            family.kernel_class.letter(),
            class_members.len()
        );
        println!("  {}", class_members.join(", "));
    }

    // Proof status
    if show_proof {
        let proofs = proof_status_for_class(family.kernel_class);
        println!();
        println!("Proof Status:");
        for proof in &proofs {
            println!(
                "  {} {:<28} {} ({})",
                proof.level.symbol(),
                proof.contract,
                proof.level.label(),
                proof.evidence
            );
        }

        let proven = proofs
            .iter()
            .filter(|p| p.level == ProofLevel::Proven)
            .count();
        let tested = proofs
            .iter()
            .filter(|p| p.level == ProofLevel::Tested)
            .count();
        let total = proofs.len();
        println!();
        println!(
            "Kernel Class {}: {}/{} contracts verified ({} proven, {} tested).",
            family.kernel_class.letter(),
            proven + tested,
            total,
            proven,
            tested
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_kernel_class_a() {
        let c = Constraints {
            attention_type: "gqa".into(),
            activation: "silu".into(),
            norm_type: "rmsnorm".into(),
            mlp_type: "swiglu".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::A);
    }

    #[test]
    fn test_derive_kernel_class_b() {
        let c = Constraints {
            attention_type: "mha".into(),
            activation: "gelu".into(),
            norm_type: "layernorm".into(),
            mlp_type: "gelu_mlp".into(),
            positional_encoding: "absolute".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::B);
    }

    #[test]
    fn test_derive_kernel_class_f() {
        let c = Constraints {
            attention_type: "gqa".into(),
            activation: "gelu".into(),
            norm_type: "rmsnorm".into(),
            mlp_type: "gated_mlp".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::F);
    }

    #[test]
    fn test_derive_kernel_class_unknown() {
        let c = Constraints::default();
        assert_eq!(derive_kernel_class(&c), KernelClass::Unknown);
    }

    #[test]
    fn test_load_families_not_empty() {
        let families = load_families();
        assert!(!families.is_empty());
        // Qwen2 should be Class A
        let qwen2 = families.iter().find(|f| f.family == "qwen2").unwrap();
        assert_eq!(qwen2.kernel_class, KernelClass::A);
    }

    #[test]
    fn test_resolve_family_direct() {
        let f = resolve_family("qwen2").unwrap();
        assert_eq!(f.family, "qwen2");
        assert_eq!(f.kernel_class, KernelClass::A);
    }

    #[test]
    fn test_resolve_family_architecture() {
        let f = resolve_family("Qwen2ForCausalLM").unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn test_resolve_family_partial() {
        let f = resolve_family("llama").unwrap();
        assert_eq!(f.family, "llama");
    }

    #[test]
    fn test_resolve_family_unknown() {
        let f = resolve_family("nonexistent-family-xyz");
        assert!(f.is_none());
    }

    #[test]
    fn test_kernel_ops_class_a_has_rope() {
        let ops = kernel_ops_for_class(KernelClass::A);
        assert!(ops.iter().any(|o| o.kernel == "rope_forward"));
    }

    #[test]
    fn test_kernel_ops_class_b_no_rope() {
        let ops = kernel_ops_for_class(KernelClass::B);
        assert!(!ops.iter().any(|o| o.kernel == "rope_forward"));
    }

    #[test]
    fn test_proof_status_matvec() {
        let proof = proof_status_for_contract("matvec-kernel-v1");
        assert_ne!(proof.level, ProofLevel::Unknown);
    }

    #[test]
    fn test_proof_status_unknown_contract() {
        let proof = proof_status_for_contract("nonexistent-v1");
        assert_eq!(proof.level, ProofLevel::Unknown);
    }

    #[test]
    fn test_yaml_str_extraction() {
        let yaml = "family: qwen2\ndisplay_name: \"Qwen2 / Qwen2.5-Coder\"\n";
        assert_eq!(yaml_str(yaml, "family"), Some("qwen2".to_string()));
        assert_eq!(
            yaml_str(yaml, "display_name"),
            Some("Qwen2 / Qwen2.5-Coder".to_string())
        );
        assert_eq!(yaml_str(yaml, "missing"), None);
    }

    #[test]
    fn test_yaml_bool_extraction() {
        let yaml = "has_bias: true\ntied_embeddings: false\n";
        assert!(yaml_bool(yaml, "has_bias"));
        assert!(!yaml_bool(yaml, "tied_embeddings"));
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"model_type": "qwen2", "hidden_act": "silu"}"#;
        assert_eq!(
            extract_json_string(json, "model_type"),
            Some("qwen2".to_string())
        );
        assert_eq!(
            extract_json_string(json, "hidden_act"),
            Some("silu".to_string())
        );
        assert_eq!(extract_json_string(json, "missing"), None);
    }

    #[test]
    fn test_deterministic_kernel_class() {
        // FALSIFY-KE-001: Same constraints always produce same kernel class
        let c = Constraints {
            attention_type: "gqa".into(),
            activation: "silu".into(),
            norm_type: "rmsnorm".into(),
            mlp_type: "swiglu".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        let class1 = derive_kernel_class(&c);
        let class2 = derive_kernel_class(&c);
        assert_eq!(class1, class2);
    }

    #[test]
    fn test_class_label_round_trip() {
        for class in [
            KernelClass::A,
            KernelClass::B,
            KernelClass::C,
            KernelClass::D,
            KernelClass::E,
            KernelClass::F,
        ] {
            assert!(!class.label().is_empty());
            assert!(!class.letter().is_empty());
        }
    }

    #[test]
    fn test_equivalence_class_membership() {
        let families = load_families();
        // LLaMA and Qwen2 should be in the same kernel class (A)
        let llama = families.iter().find(|f| f.family == "llama").unwrap();
        let qwen2 = families.iter().find(|f| f.family == "qwen2").unwrap();
        assert_eq!(llama.kernel_class, qwen2.kernel_class);
        assert_eq!(llama.kernel_class, KernelClass::A);
    }

    #[test]
    fn test_gpt2_is_class_b() {
        let families = load_families();
        let gpt2 = families.iter().find(|f| f.family == "gpt2").unwrap();
        assert_eq!(gpt2.kernel_class, KernelClass::B);
    }

    #[test]
    fn test_gemma_is_class_f() {
        let families = load_families();
        let gemma = families.iter().find(|f| f.family == "gemma").unwrap();
        assert_eq!(gemma.kernel_class, KernelClass::F);
    }
}
