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
    A,      // GQA + RMSNorm + SiLU + SwiGLU + RoPE
    B,      // MHA + LayerNorm + GELU + absolute/none
    C,      // MQA + LayerNorm + GELU + ALiBi
    D,      // mixed: LayerNorm + SiLU or GQA + LayerNorm
    E,      // MoE variants
    F,      // RMSNorm + GELU + GatedMlp + RoPE
    Ssm,    // State Space Models (Mamba: selective scan, no attention)
    Linear, // Linear Attention (RWKV: WKV recurrence, no softmax)
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
            Self::Ssm => "SSM (State Space Model + RMSNorm + SiLU)",
            Self::Linear => "Linear (WKV Recurrence + LayerNorm + GELU)",
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
            Self::Ssm => "SSM",
            Self::Linear => "Linear",
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

/// Get kernel ops for a class, optionally enriched with constraint-specific ops.
fn kernel_ops_for_family(class: KernelClass, constraints: &Constraints) -> Vec<KernelOp> {
    let mut ops = kernel_ops_for_class(class);
    // Add RoPE op for families that use RoPE but whose class doesn't include it
    // (e.g., Phi: Class B with RoPE positional encoding)
    let has_rope = ops.iter().any(|o| o.kernel == "rope_forward");
    if !has_rope && constraints.positional_encoding == "rope" {
        ops.push(KernelOp {
            op: "Position Encoding",
            kernel: "rope_forward",
            contract: "rope-kernel-v1",
        });
    }
    ops
}

fn kernel_ops_for_class(class: KernelClass) -> Vec<KernelOp> {
    // Base ops: MatVec always present. Softmax only for attention-based models (not SSM).
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
    ];

    // Softmax is attention-specific — SSM and linear attention models don't use it
    if class != KernelClass::Ssm && class != KernelClass::Linear {
        ops.push(KernelOp {
            op: "Softmax",
            kernel: "softmax",
            contract: "softmax-kernel-v1",
        });
    }

    ops.push(KernelOp {
        op: "Kernel Fusion",
        kernel: "fused_matvec_activation",
        contract: "kernel-fusion-v1",
    });

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
        KernelClass::Ssm => {
            ops.push(KernelOp {
                op: "SSM Scan",
                kernel: "selective_scan",
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
                kernel: "gated_mlp",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Conv1d",
                kernel: "depthwise_conv1d",
                contract: "element-wise-ops-v1",
            });
        }
        KernelClass::Linear => {
            ops.push(KernelOp {
                op: "WKV Recurrence",
                kernel: "wkv_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Token Shift",
                kernel: "token_shift",
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
                op: "Channel Mixing",
                kernel: "channel_mix",
                contract: "element-wise-ops-v1",
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

/// Model types without their own family YAML that share a kernel pipeline
/// with an existing family. Maps model_type → family name.
const FAMILY_ALIASES: &[(&str, &str)] = &[
    // Class A variants (SiLU + RMSNorm + GQA/MHA + RoPE)
    ("olmo", "llama"),      // OLMo v1 (LlamaForCausalLM base)
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
    // Mistral-derived fine-tunes (same architecture: MistralForCausalLM)
    ("codestral", "mistral"),  // Codestral-22B coding model
    ("mathstral", "mistral"),  // Mathstral math model
    ("pixtral", "mistral"),    // Pixtral vision-language model
    ("zephyr", "mistral"),     // HuggingFace Zephyr fine-tune
    ("openchat", "mistral"),   // OpenChat fine-tune
    ("openhermes", "mistral"), // OpenHermes fine-tune
    // Llama-derived fine-tunes (same architecture: LlamaForCausalLM)
    ("nemotron", "llama"), // NVIDIA Nemotron
    ("solar", "llama"),    // Upstage Solar
    ("vicuna", "llama"),   // LMSYS Vicuna
    // Qwen-derived
    ("qwq", "qwen2"), // QwQ reasoning model (Qwen2ForCausalLM)
    // SmolLM: LlamaForCausalLM architecture
    ("smollm", "llama"),  // SmolLM (RMSNorm + SiLU + GQA + RoPE)
    ("smollm2", "llama"), // SmolLM2 (RMSNorm + SiLU + GQA + RoPE)
    // Classic falcon: LayerNorm + GELU + GQA/MHA — closest to bert (Class B)
    ("falcon", "bert"), // Falcon-7B/40B: LayerNorm + GELU (no RMSNorm, no SiLU)
    // Bloom: LayerNorm + GELU + MHA + ALiBi — same kernel dispatch as bert (Class B)
    ("bloom", "bert"),      // bigscience/bloom: MHA + LayerNorm + GELU
    ("bloomz", "bert"),     // bigscience/bloomz instruction-tuned variant
    ("bloom_560m", "bert"), // Bloom size variants
    ("bigscience", "bert"), // Org-name resolution
];

/// Get all family aliases for display in help/error messages.
pub fn family_aliases() -> &'static [(&'static str, &'static str)] {
    FAMILY_ALIASES
}

/// Known HuggingFace architecture class names for common aliases.
/// Used to display the correct architecture instead of the raw alias string.
const ALIAS_ARCHITECTURES: &[(&str, &str)] = &[
    ("bloom", "BloomForCausalLM"),
    ("bloomz", "BloomForCausalLM"),
    ("bloom_560m", "BloomForCausalLM"),
    ("falcon", "FalconForCausalLM"),
    ("mixtral", "MixtralForCausalLM"),
    ("phi3", "Phi3ForCausalLM"),
    ("phi3small", "Phi3SmallForCausalLM"),
    ("codellama", "LlamaForCausalLM"),
    ("vicuna", "LlamaForCausalLM"),
    ("solar", "LlamaForCausalLM"),
    ("nemotron", "LlamaForCausalLM"),
    ("olmo", "OlmoForCausalLM"),
    ("olmo2", "Olmo2ForCausalLM"),
    ("granite", "GraniteForCausalLM"),
    ("internlm2", "InternLM2ForCausalLM"),
    ("yi", "LlamaForCausalLM"),
    ("baichuan", "BaichuanForCausalLM"),
    ("stablelm", "StableLmForCausalLM"),
    ("starcoder2", "Starcoder2ForCausalLM"),
    ("codestral", "MistralForCausalLM"),
    ("mathstral", "MistralForCausalLM"),
    ("pixtral", "LlavaMistralForCausalLM"),
    ("zephyr", "MistralForCausalLM"),
    ("openchat", "MistralForCausalLM"),
    ("openhermes", "MistralForCausalLM"),
    ("qwen2_moe", "Qwen2MoeForCausalLM"),
    ("qwen2moe", "Qwen2MoeForCausalLM"),
    ("qwen3_moe", "Qwen3MoeForCausalLM"),
    ("qwen3moe", "Qwen3MoeForCausalLM"),
    ("deepseek_v2", "DeepseekV2ForCausalLM"),
    ("deepseekv2", "DeepseekV2ForCausalLM"),
    ("qwq", "Qwen2ForCausalLM"),
    ("bigscience", "BloomForCausalLM"),
    ("qwen3_next", "Qwen3ForCausalLM"),
    ("qwen3next", "Qwen3ForCausalLM"),
    ("smollm", "LlamaForCausalLM"),
    ("smollm2", "LlamaForCausalLM"),
];

/// Normalize input: lowercase, trim, replace hyphens/dots with underscores.
/// E.g., "falcon-h1" → "falcon_h1", "qwen3.5" → "qwen3_5"
fn normalize_input(input: &str) -> String {
    input
        .to_lowercase()
        .trim()
        .replace('-', "_")
        .replace('.', "_")
}

/// Secondary normalization: strip all separators between name and version.
/// E.g., "phi_3" → "phi3", "gpt_2" → "gpt2", "rwkv_7" → "rwkv7"
/// Returns None if same as input.
fn compact_input(normalized: &str) -> Option<String> {
    let compact = normalized.replace('_', "");
    if compact != normalized {
        Some(compact)
    } else {
        None
    }
}

/// Resolve a family string or architecture string to `FamilyInfo`.
pub fn resolve_family(input: &str) -> Option<FamilyInfo> {
    let lower = input.to_lowercase();
    let lower = lower.trim();
    if lower.is_empty() {
        return None;
    }

    // Strip non-ASCII characters (emoji, CJK, etc.) — family names are ASCII-only
    let ascii_only: String = lower.chars().filter(|c| c.is_ascii()).collect();
    let ascii_only = ascii_only.trim();
    if ascii_only.is_empty() {
        return None;
    }
    // Use the ASCII-stripped version for all matching
    let lower = ascii_only;

    let families = load_families();
    // Normalized form for matching (hyphens/dots → underscores)
    let normalized = normalize_input(lower);
    // Compact form for matching (all separators removed: phi_3 → phi3)
    let compact = compact_input(&normalized);

    // Direct family name match (try raw, normalized, compact, and cross-compact)
    if let Some(f) = families.iter().find(|f| {
        f.family == lower
            || f.family == normalized
            || compact.as_deref().is_some_and(|c| f.family == c)
            // Cross-compact: compare compact forms of both sides
            // e.g., input "qwen-3-5" compact="qwen35", family "qwen3_5" compact="qwen35"
            || compact
                .as_deref()
                .is_some_and(|c| compact_input(&f.family).as_deref() == Some(c))
    }) {
        return Some(f.clone());
    }

    // Alias match (model types sharing kernel pipeline with existing family)
    // Try raw lowercase → normalized → compact forms
    let alias_match = FAMILY_ALIASES
        .iter()
        .find(|(alias, _)| *alias == lower)
        .or_else(|| {
            FAMILY_ALIASES
                .iter()
                .find(|(alias, _)| *alias == normalized.as_str())
        })
        .or_else(|| {
            compact
                .as_deref()
                .and_then(|c| FAMILY_ALIASES.iter().find(|(alias, _)| *alias == c))
        });
    if let Some((matched_alias, target)) = alias_match {
        if let Some(f) = families.iter().find(|f| f.family == *target) {
            let mut aliased = f.clone();
            aliased.display_name = format!("{} (via {} kernel pipeline)", matched_alias, f.family);
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

    // Partial match against families (e.g., "qwen" matches "qwen2")
    // Try both normalized (with underscores) and compact (without) forms
    // Each form must be >= 3 chars to avoid spurious matches (e.g., "ab" ⊂ "stablelm")
    let search_forms: Vec<&str> = {
        let mut v = vec![];
        if normalized.len() >= 3 {
            v.push(normalized.as_str());
        }
        if let Some(ref c) = compact {
            if c.len() >= 3 {
                v.push(c.as_str());
            }
        }
        v
    };
    if !search_forms.is_empty() {
        for search in &search_forms {
            // Use prefix matching, not substring, to avoid spurious matches
            // (e.g., "mma" ⊂ "gemma" or "lama" ⊂ "llama" via compact form)
            if let Some(f) = families.iter().find(|f| {
                f.family.starts_with(*search) || search.starts_with(f.family.as_str())
            }) {
                return Some(f.clone());
            }
        }
        // Also partial match against aliases (both forms, prefix only)
        for search in &search_forms {
            if let Some((matched_alias, target)) = FAMILY_ALIASES
                .iter()
                .find(|(alias, _)| alias.starts_with(*search) || search.starts_with(alias))
            {
                if let Some(f) = families.iter().find(|f| f.family == *target) {
                    let mut aliased = f.clone();
                    aliased.display_name =
                        format!("{} (via {} kernel pipeline)", matched_alias, f.family);
                    return Some(aliased);
                }
            }
        }
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
/// Returns None if model_type is absent/unresolvable. Returns Err for structural issues.
pub fn resolve_from_config_json(path: &Path) -> Option<FamilyInfo> {
    let content = std::fs::read_to_string(path).ok()?;

    // Reject JSON arrays — config.json must be an object
    let trimmed = content.trim();
    if trimmed.starts_with('[') {
        return None;
    }

    // Parse model_type from config.json
    let model_type = extract_json_string(&content, "model_type");

    // If no model_type, try architectures field as fallback
    let model_type = match model_type {
        Some(mt) => mt,
        None => {
            // Extract first architecture and strip suffix to get family name
            let arch = extract_json_string(&content, "architectures").or_else(|| {
                // architectures is a JSON array — manually extract first element
                let pos = content.find("\"architectures\"")?;
                let after = &content[pos..];
                let bracket = after.find('[')?;
                let inner = &after[bracket + 1..];
                let quote_start = inner.find('"')?;
                let rest = &inner[quote_start + 1..];
                let quote_end = rest.find('"')?;
                Some(rest[..quote_end].to_string())
            })?;
            // Convert "LlamaForCausalLM" → "llama"
            let lower = arch.to_lowercase();
            strip_arch_suffix(&lower).to_string()
        }
    };

    resolve_family(&model_type)
}

/// Simple JSON value extraction (no serde dependency for this hot path).
/// Handles both string values ("silu") and numeric values (1e-06, 8, 1000000.0).
pub fn extract_json_string(json: &str, key: &str) -> Option<String> {
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
    } else if after.starts_with('[') || after.starts_with('{') {
        // Array or object — not a scalar value
        None
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

    // Extract architectures array (for conflict detection)
    // architectures is a JSON array — manually extract first element
    if let Some(pos) = content.find("\"architectures\"") {
        let after = &content[pos..];
        if let Some(bracket) = after.find('[') {
            let inner = &after[bracket + 1..];
            if let Some(quote_start) = inner.find('"') {
                let rest = &inner[quote_start + 1..];
                if let Some(quote_end) = rest.find('"') {
                    let arch = &rest[..quote_end];
                    map.insert(
                        "_architectures".to_string(),
                        ConfigField {
                            value: arch.to_string(),
                            rationale: "HuggingFace architecture class".to_string(),
                        },
                    );
                }
            }
        }
    }

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
        ("tie_word_embeddings", "Weight sharing: embedding ↔ lm_head"),
        ("vocab_size", "Vocabulary size"),
        ("max_position_embeddings", "Maximum sequence length"),
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
            let act = extract_json_string(json, "hidden_act")
                .unwrap_or_default()
                .to_lowercase();
            let is_gelu = act.contains("gelu");
            let is_silu = act == "silu" || act == "swish";
            match (hidden, inter) {
                (Some(h), Some(i)) if h > 0.0 => {
                    let ratio = i / h;
                    // SiLU models use SwiGLU MLP regardless of ratio
                    // (MoE models have lower per-expert intermediate_size)
                    // GELU models use standard GELU FFN
                    let mlp_type = if is_gelu {
                        "GELU FFN"
                    } else if is_silu {
                        "SwiGLU MLP"
                    } else if ratio > 2.5 {
                        "SwiGLU MLP"
                    } else {
                        "Standard FFN"
                    };
                    Some(format!("{mlp_type} ({i:.0}/{h:.0} = {ratio:.2}x)"))
                }
                _ => None,
            }
        }
        "num_local_experts" | "num_experts" | "n_routed_experts" => {
            let n: i32 = value.parse().unwrap_or(0);
            if n > 1 {
                Some(format!("MoE with {n} experts (expert routing kernel)"))
            } else if n == 1 {
                Some("1 expert (dense model, not MoE)".to_string())
            } else if n < 0 {
                Some(format!("Invalid: {n} experts (negative)"))
            } else {
                None
            }
        }
        "num_experts_per_tok" => {
            let n: u32 = value.parse().unwrap_or(0);
            if n > 0 {
                let plural = if n == 1 { "expert" } else { "experts" };
                Some(format!("{n} active {plural} per token"))
            } else {
                None
            }
        }
        "tie_word_embeddings" => match value {
            "true" => Some("Shared: embedding == lm_head (saves memory)".to_string()),
            "false" => Some("Separate embedding and lm_head weights".to_string()),
            _ => None,
        },
        "num_attention_heads" => {
            let kv = extract_json_string(json, "num_key_value_heads")
                .and_then(|v| v.parse::<u32>().ok());
            let n: u32 = value.parse().unwrap_or(0);
            match kv {
                Some(kv_n) if kv_n == 1 => Some(format!("{n} query heads, MQA (1 KV head)")),
                Some(kv_n) if kv_n < n => {
                    let ratio = n / kv_n;
                    Some(format!(
                        "{n} query heads, GQA ({ratio} queries per KV group)"
                    ))
                }
                Some(kv_n) if kv_n == n => Some(format!("{n} heads, MHA (no KV grouping)")),
                _ => None,
            }
        }
        "hidden_size" => {
            let n: u64 = value.parse().unwrap_or(0);
            if n > 0 {
                let params_est = if let Some(layers) =
                    extract_json_string(json, "num_hidden_layers")
                        .and_then(|v| v.parse::<u64>().ok())
                {
                    // Use intermediate_size if available for better estimate,
                    // otherwise fall back to 12*L*d^2 (assumes 4x MLP ratio)
                    let inter = extract_json_string(json, "intermediate_size")
                        .and_then(|v| v.parse::<u64>().ok());
                    // Vocab embeddings: vocab_size * d (+ lm_head if not tied)
                    let vocab = extract_json_string(json, "vocab_size")
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(0);
                    let tied = extract_json_string(json, "tie_word_embeddings")
                        .map_or(false, |v| v == "true");
                    let embed_params = if tied { vocab * n } else { 2 * vocab * n };
                    let est = if let Some(i) = inter {
                        // Per layer: 4d² (QKVO) + 3di (SwiGLU gate+up+down) + 2d (norms)
                        layers * (4 * n * n + 3 * n * i + 2 * n) + embed_params
                    } else {
                        // Rough estimate assuming 4x MLP (standard FFN: 8d²/layer)
                        layers * 12 * n * n + embed_params
                    };
                    if est > 1_000_000_000 {
                        format!(", ~{:.1}B params", est as f64 / 1e9)
                    } else if est > 1_000_000 {
                        format!(", ~{:.0}M params", est as f64 / 1e6)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                Some(format!("Hidden dim {n}{params_est}"))
            } else {
                None
            }
        }
        "num_hidden_layers" => {
            let n: u32 = value.parse().unwrap_or(0);
            if n > 0 {
                Some(format!("{n} transformer layers"))
            } else {
                None
            }
        }
        "vocab_size" => {
            let n: u64 = value.parse().unwrap_or(0);
            let hidden =
                extract_json_string(json, "hidden_size").and_then(|v| v.parse::<u64>().ok());
            if let Some(h) = hidden {
                let embed_mb = (n * h * 2) as f64 / 1_048_576.0; // fp16
                Some(format!("{n} tokens (embedding: {embed_mb:.0} MB at fp16)"))
            } else if n > 0 {
                Some(format!("{n} tokens"))
            } else {
                None
            }
        }
        "max_position_embeddings" => {
            let n: u64 = value.parse().unwrap_or(0);
            if n >= 1_048_576 {
                Some(format!("{n} max seq len (1M+ context)"))
            } else if n >= 524_288 {
                Some(format!("{n} max seq len (512K+ context)"))
            } else if n >= 262_144 {
                Some(format!("{n} max seq len (256K+ context)"))
            } else if n >= 131_072 {
                Some(format!("{n} max seq len (128K+ context)"))
            } else if n >= 32_768 {
                Some(format!("{n} max seq len (32K+ context)"))
            } else if n >= 8_192 {
                Some(format!("{n} max seq len (8K+ context)"))
            } else if n > 0 {
                Some(format!("{n} max seq len"))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract the architecture display string.
/// Priority: config.json _architectures → config.json model_type → alias arch table → family default.
/// For aliases, shows the known HF architecture class, not the target family's.
pub fn extract_architecture_display(
    family: &FamilyInfo,
    config_mapping: &BTreeMap<String, ConfigField>,
) -> String {
    // If config.json has _architectures, prefer that (it's the actual HF arch)
    if let Some(arch) = config_mapping.get("_architectures") {
        return arch.value.clone();
    }
    // If config.json has model_type, use that
    if let Some(mt) = config_mapping.get("model_type") {
        return mt.value.clone();
    }
    // For aliases: look up known HF architecture class
    if family.display_name.contains(" (via ") {
        if let Some(alias_name) = family.display_name.split(" (via ").next() {
            // Check the alias architecture table for the canonical HF class name
            if let Some((_, hf_arch)) = ALIAS_ARCHITECTURES
                .iter()
                .find(|(alias, _)| *alias == alias_name)
            {
                return (*hf_arch).to_string();
            }
            // Fall back to the raw alias name
            return alias_name.to_string();
        }
    }
    // Fall back to family's first architecture
    family
        .architectures
        .first()
        .map_or("Unknown".to_string(), Clone::clone)
}

/// Detect mismatches between config.json values and family constraints.
pub fn detect_constraint_mismatches(
    family: &FamilyInfo,
    config_mapping: &BTreeMap<String, ConfigField>,
) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check model_type vs architectures contradiction (bug 6)
    if let Some(mt) = config_mapping.get("model_type") {
        if let Some(arch) = config_mapping.get("_architectures") {
            let arch_lower = arch.value.to_lowercase();
            let mt_lower = mt.value.to_lowercase();
            let arch_family = strip_arch_suffix(&arch_lower);
            // Normalize both sides: remove underscores for comparison
            // (deepseek_v2 vs deepseekv2 from DeepseekV2ForCausalLM)
            let mt_compact = mt_lower.replace('_', "");
            let arch_compact = arch_family.replace('_', "");
            // If the architecture's family name doesn't match the model_type
            if arch_family != mt_lower
                && arch_compact != mt_compact
                && !arch_lower.starts_with(&mt_lower)
            {
                warnings.push(format!(
                    "model_type '{}' conflicts with architectures ['{}']. Using model_type for dispatch.",
                    mt.value, arch.value
                ));
            }
        }
    }

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
    if has_rms && has_ln {
        warnings.push(
            "Conflicting norm config: both rms_norm_eps (RMSNorm) and layer_norm_epsilon (LayerNorm) present. Only one should exist.".to_string()
        );
    } else if has_rms && !has_ln && family_norm == "layernorm" {
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
            // Detect physically impossible config: KV heads > Q heads
            if kv_n > q_n && q_n > 0 {
                warnings.push(format!(
                    "Invalid attention config: num_key_value_heads ({kv_n}) > num_attention_heads ({q_n}). KV heads cannot exceed query heads."
                ));
            } else if kv_n > 0 && q_n > 0 && q_n % kv_n != 0 {
                // GQA requires query heads divisible by KV heads
                warnings.push(format!(
                    "Invalid GQA config: num_attention_heads ({q_n}) not divisible by num_key_value_heads ({kv_n}). GQA requires even grouping."
                ));
            } else {
                let config_attn = if kv_n == 1 {
                    "mqa"
                } else if q_n > 0 && kv_n < q_n {
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
        let n_experts: i32 = ef.value.parse().unwrap_or(0);
        if n_experts < 0 {
            warnings.push(format!(
                "Invalid config: expert count ({}) is negative.",
                ef.value
            ));
        } else if n_experts > 1 && family.kernel_class != KernelClass::E {
            warnings.push(format!(
                "MoE model ({n_experts} experts) mapped to non-MoE class {}. Expert routing kernel not covered.",
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

    // Check for invalid dimensions (negative or zero values)
    for (key, label) in &[
        ("hidden_size", "Hidden size"),
        ("num_attention_heads", "Attention heads"),
        ("num_hidden_layers", "Hidden layers"),
        ("vocab_size", "Vocabulary size"),
    ] {
        if let Some(field) = config_mapping.get(*key) {
            if let Ok(n) = field.value.parse::<i64>() {
                if n < 0 {
                    warnings.push(format!(
                        "Invalid config: {label} ({key}={n}) is negative. Must be positive."
                    ));
                } else if n == 0 && (*key == "hidden_size" || *key == "num_attention_heads") {
                    warnings.push(format!(
                        "Invalid config: {label} ({key}=0) is zero. Would cause division by zero in kernel dispatch."
                    ));
                }
            }
        }
    }

    // Check for implausible dimensions
    if let Some(field) = config_mapping.get("hidden_size") {
        if let Ok(n) = field.value.parse::<u64>() {
            if n > 100_000 {
                warnings.push(format!(
                    "Implausible hidden_size={n}. Largest known models have hidden_size ~16384."
                ));
            }
        }
    }

    // Check hidden_size divisibility by num_attention_heads (defines head_dim)
    if let (Some(hs), Some(nh)) = (
        config_mapping.get("hidden_size"),
        config_mapping.get("num_attention_heads"),
    ) {
        if let (Ok(h), Ok(n)) = (hs.value.parse::<u64>(), nh.value.parse::<u64>()) {
            if n > 0 && h > 0 && h % n != 0 {
                warnings.push(format!(
                    "Invalid config: hidden_size ({h}) not divisible by num_attention_heads ({n}). Head dimension must be an integer."
                ));
            }
        }
    }

    // Note: tied_embeddings mismatch NOT warned about — it varies by model size
    // within a family (small models often tie, large ones don't) and does not
    // affect kernel dispatch.

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
    mut config_mapping: BTreeMap<String, ConfigField>,
    show_proof: bool,
) -> KernelExplainJson {
    // Remove internal fields (prefixed with _) from public output
    config_mapping.retain(|k, _| !k.starts_with('_'));
    let ops = kernel_ops_for_family(family.kernel_class, &family.constraints);
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

    let arch = extract_architecture_display(family, &config_mapping);

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
    let arch = extract_architecture_display(family, config_mapping);
    let ops = kernel_ops_for_family(family.kernel_class, &family.constraints);

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

    // Config mapping (skip internal fields prefixed with _)
    let visible_fields: Vec<_> = config_mapping
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .collect();
    if !visible_fields.is_empty() {
        println!();
        println!("Config.json → Kernel Mapping:");
        // Calculate alignment width from longest key=value
        let max_kv_len = visible_fields
            .iter()
            .map(|(k, f)| k.len() + 1 + f.value.len())
            .max()
            .unwrap_or(20);
        let pad_to = max_kv_len + 2; // 2 extra spaces before arrow
        for (key, field) in &visible_fields {
            let kv = format!("{key}={}", field.value);
            println!("  {kv:<pad_to$} → {}", field.rationale);
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

    // Constraint mismatch warnings (always run — MoE detection works from alias name too)
    let mismatches = detect_constraint_mismatches(family, config_mapping);
    let is_alias = family.display_name.contains(" (via ");
    for warning in &mismatches {
        println!();
        eprintln!("  WARNING: {warning}");
        if is_alias {
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
    if !class_members.is_empty() {
        println!();
        println!(
            "Equivalence Class {}: {} {}",
            family.kernel_class.letter(),
            class_members.len(),
            if class_members.len() == 1 {
                "family"
            } else {
                "families"
            }
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
