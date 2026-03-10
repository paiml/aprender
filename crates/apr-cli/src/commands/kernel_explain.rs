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
    ("phi4", "llama"),      // Phi-4: same pipeline as Phi-3
    ("codellama", "llama"),
    ("tinyllama", "llama"), // TinyLlama (LlamaForCausalLM)
    ("stablelm", "llama"),
    ("yi", "llama"),
    ("baichuan", "llama"),
    // Class B variants (MHA + LayerNorm + GELU)
    ("gpt_neo", "bert"),     // GPTNeoForCausalLM
    ("gptneo", "bert"),      // compact form
    ("gpt_neox", "bert"),    // GPTNeoXForCausalLM
    ("gptneox", "bert"),     // compact form
    ("gpt_j", "bert"),       // GPTJForCausalLM
    ("gptj", "bert"),        // compact form
    ("gpt_bigcode", "bert"), // GPTBigCodeForCausalLM (StarCoder v1)
    ("gptbigcode", "bert"),  // compact form
    ("starcoder1", "bert"),  // StarCoder v1 explicit
    ("codegen", "bert"),     // CodeGenForCausalLM
    ("xglm", "bert"),        // XGLMForCausalLM
    ("opt", "bert"),         // OPTForCausalLM
    ("galactica", "bert"),   // OPT-based (Meta Galactica)
    ("roberta", "bert"),     // RoBERTa (BERT variant)
    ("deberta", "bert"),     // DeBERTa (BERT variant)
    ("electra", "bert"),     // ELECTRA (BERT variant)
    ("distilbert", "bert"),  // DistilBERT
    // Class D variants
    ("phi3small", "phi"), // gegelu + LayerNorm (unique, closest=phi)
    // Class F variants
    ("codegemma", "gemma"), // CodeGemma (same as Gemma)
    ("gemma2", "gemma"),    // Gemma 2
    ("gemma3", "gemma"),    // Gemma 3
    // GELU + RMSNorm variants
    ("starcoder2", "qwen2"), // GELU + RMSNorm + GQA + RoPE (closest match; warns on act/norm)
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
    // GPT variants
    ("gpt_neo", "GPTNeoForCausalLM"),
    ("gptneo", "GPTNeoForCausalLM"),
    ("gpt_neox", "GPTNeoXForCausalLM"),
    ("gptneox", "GPTNeoXForCausalLM"),
    ("gpt_j", "GPTJForCausalLM"),
    ("gptj", "GPTJForCausalLM"),
    ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    ("gptbigcode", "GPTBigCodeForCausalLM"),
    ("starcoder1", "GPTBigCodeForCausalLM"),
    ("codegen", "CodeGenForCausalLM"),
    ("xglm", "XGLMForCausalLM"),
    ("opt", "OPTForCausalLM"),
    ("galactica", "OPTForCausalLM"),
    ("roberta", "RobertaForMaskedLM"),
    ("deberta", "DebertaV2ForMaskedLM"),
    ("electra", "ElectraForPreTraining"),
    ("distilbert", "DistilBertModel"),
    ("tinyllama", "LlamaForCausalLM"),
    ("phi4", "Phi3ForCausalLM"),
    ("codegemma", "CodeGemmaForCausalLM"),
    ("gemma2", "Gemma2ForCausalLM"),
    ("gemma3", "Gemma3ForCausalLM"),
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
        // Two-pass partial matching:
        // Pass 1: search starts_with alias (search is MORE specific, e.g., "phi3mini" starts_with "phi3")
        //   → Aliases win because they're exact subtypes (phi3 → llama, not phi family)
        // Pass 2: family starts_with search (search is LESS specific, e.g., "qwen" prefix of "qwen2")
        //   → Families win because they're direct matches, not accidental alias prefixes

        // Pass 1: search starts_with alias (search is longer/more specific than alias)
        for search in &search_forms {
            if let Some((matched_alias, target)) = FAMILY_ALIASES
                .iter()
                .find(|(alias, _)| search.starts_with(alias))
            {
                if let Some(f) = families.iter().find(|f| f.family == *target) {
                    let mut aliased = f.clone();
                    aliased.display_name =
                        format!("{} (via {} kernel pipeline)", matched_alias, f.family);
                    return Some(aliased);
                }
            }
        }
        // Pass 2: family starts_with search (search is a prefix of family name)
        for search in &search_forms {
            if let Some(f) = families
                .iter()
                .find(|f| f.family.starts_with(*search) || search.starts_with(f.family.as_str()))
            {
                return Some(f.clone());
            }
        }
        // Pass 3: alias starts_with search (search is a prefix of alias name)
        for search in &search_forms {
            if let Some((matched_alias, target)) = FAMILY_ALIASES
                .iter()
                .find(|(alias, _)| alias.starts_with(*search))
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
        ("moe_intermediate_size", "MoE per-expert MLP width"),
        ("head_dim", "Explicit attention head dimension"),
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
                    // GQA-aware attention: Q+O use full heads, K+V use KV heads
                    let kv_heads = extract_json_string(json, "num_key_value_heads")
                        .and_then(|v| v.parse::<u64>().ok());
                    let head_dim_val = extract_json_string(json, "head_dim")
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or_else(|| {
                            let nh = extract_json_string(json, "num_attention_heads")
                                .and_then(|v| v.parse::<u64>().ok())
                                .unwrap_or(1);
                            if nh > 0 {
                                n / nh
                            } else {
                                0
                            }
                        });
                    let num_heads = extract_json_string(json, "num_attention_heads")
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(1);
                    let kv_dim = kv_heads.map_or(n, |kv| kv * head_dim_val);
                    // Attention: Q(h*hd, d) + K(kv*hd, d) + V(kv*hd, d) + O(d, h*hd)
                    let attn_params = 2 * num_heads * head_dim_val * n + 2 * kv_dim * n;
                    // MoE expert params (if any)
                    let n_experts = extract_json_string(json, "num_local_experts")
                        .or_else(|| extract_json_string(json, "num_experts"))
                        .or_else(|| extract_json_string(json, "n_routed_experts"))
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(0);
                    let moe_inter = extract_json_string(json, "moe_intermediate_size")
                        .and_then(|v| v.parse::<u64>().ok());
                    let est = if let Some(i) = inter {
                        // SwiGLU/SiLU: 3 MLP matrices (gate+up+down)
                        // GELU/ReLU: 2 MLP matrices (up+down, no gate)
                        let act = extract_json_string(json, "hidden_act")
                            .unwrap_or_default()
                            .to_lowercase();
                        let is_gated = act == "silu" || act == "swish" || act.contains("gegelu");
                        let mlp_factor = if is_gated { 3 } else { 2 };
                        let dense_mlp = mlp_factor * n * i;
                        let expert_mlp = if n_experts > 1 {
                            let ei = moe_inter.unwrap_or(i);
                            n_experts * mlp_factor * n * ei // per-expert MLP
                        } else {
                            0
                        };
                        let mlp_total = if n_experts > 1 {
                            expert_mlp + dense_mlp
                        } else {
                            dense_mlp
                        };
                        // Per layer: attention + MLP + 2d (norms)
                        layers * (attn_params + mlp_total + 2 * n) + embed_params
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
                // MHA is a degenerate case of GQA (kv_heads == q_heads) — same kernel dispatch
                let is_mha_gqa_compat = config_attn == "mha" && family_attn == "gqa";
                if config_attn != family_attn && !family_attn.is_empty() && !is_mha_gqa_compat {
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
    // Skip when explicit head_dim is present — some models (e.g., Qwen3.5) use
    // head_dim * num_heads != hidden_size (attention dim != hidden dim)
    let has_explicit_head_dim = config_mapping.contains_key("head_dim");
    if !has_explicit_head_dim {
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

    // ── yaml_str ────────────────────────────────────────────────────────

    #[test]
    fn yaml_str_plain_value() {
        let yaml = "family: qwen2\ndisplay_name: \"Qwen2 / Qwen2.5-Coder\"\n";
        assert_eq!(yaml_str(yaml, "family"), Some("qwen2".to_string()));
        assert_eq!(
            yaml_str(yaml, "display_name"),
            Some("Qwen2 / Qwen2.5-Coder".to_string())
        );
    }

    #[test]
    fn yaml_str_missing_key() {
        assert_eq!(yaml_str("family: bert\n", "missing"), None);
    }

    #[test]
    fn yaml_str_inline_comment_stripped() {
        let yaml = "attention_type: gqa  # MLA dispatches as GQA\n";
        assert_eq!(yaml_str(yaml, "attention_type"), Some("gqa".to_string()));
    }

    #[test]
    fn yaml_str_quoted_values() {
        assert_eq!(
            yaml_str("name: 'single'\n", "name"),
            Some("single".to_string())
        );
        assert_eq!(
            yaml_str("name: \"double\"\n", "name"),
            Some("double".to_string())
        );
    }

    #[test]
    fn yaml_str_null_returns_none() {
        assert_eq!(yaml_str("val: null\n", "val"), None);
    }

    #[test]
    fn yaml_str_empty_value_returns_none() {
        assert_eq!(yaml_str("val:\n", "val"), None);
    }

    #[test]
    fn yaml_str_value_with_only_comment() {
        // "val: # just a comment" → after stripping comment, value is empty → None
        assert_eq!(yaml_str("val: # just a comment\n", "val"), None);
    }

    // ── yaml_bool ───────────────────────────────────────────────────────

    #[test]
    fn yaml_bool_true() {
        assert!(yaml_bool("has_bias: true\n", "has_bias"));
    }

    #[test]
    fn yaml_bool_false() {
        assert!(!yaml_bool("has_bias: false\n", "has_bias"));
    }

    #[test]
    fn yaml_bool_missing_is_false() {
        assert!(!yaml_bool("other: true\n", "has_bias"));
    }

    // ── yaml_list ───────────────────────────────────────────────────────

    #[test]
    fn yaml_list_basic() {
        let yaml =
            "architectures:\n  - Qwen2ForCausalLM\n  - Qwen2ForTokenClassification\nother: val\n";
        let items = yaml_list(yaml, "architectures");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], "Qwen2ForCausalLM");
        assert_eq!(items[1], "Qwen2ForTokenClassification");
    }

    #[test]
    fn yaml_list_stops_at_next_key() {
        let yaml = "items:\n  - a\n  - b\nnext_key: val\n";
        let items = yaml_list(yaml, "items");
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn yaml_list_empty() {
        let yaml = "items:\nother: val\n";
        let items = yaml_list(yaml, "items");
        assert!(items.is_empty());
    }

    #[test]
    fn yaml_list_missing_key() {
        let yaml = "other:\n  - a\n";
        let items = yaml_list(yaml, "items");
        assert!(items.is_empty());
    }

    #[test]
    fn yaml_list_strips_quotes() {
        let yaml = "items:\n  - \"quoted\"\n  - 'single'\n";
        let items = yaml_list(yaml, "items");
        assert_eq!(items[0], "quoted");
        assert_eq!(items[1], "single");
    }

    // ── extract_constraints ─────────────────────────────────────────────

    #[test]
    fn extract_constraints_basic() {
        let yaml = "\nconstraints:\n  attention_type: gqa\n  activation: silu\n  norm_type: rmsnorm\n  mlp_type: swiglu\n  positional_encoding: rope\n  has_bias: false\n  tied_embeddings: true\n";
        let c = extract_constraints(yaml);
        assert_eq!(c.attention_type, "gqa");
        assert_eq!(c.activation, "silu");
        assert_eq!(c.norm_type, "rmsnorm");
        assert_eq!(c.mlp_type, "swiglu");
        assert_eq!(c.positional_encoding, "rope");
        assert!(!c.has_bias);
        assert!(c.tied_embeddings);
    }

    #[test]
    fn extract_constraints_missing_section_defaults() {
        let yaml = "family: test\n";
        let c = extract_constraints(yaml);
        assert_eq!(c.attention_type, "");
        assert!(!c.has_bias);
    }

    #[test]
    fn extract_constraints_inline_comments() {
        let yaml =
            "\nconstraints:\n  attention_type: ssm  # state space model\n  activation: silu\n";
        let c = extract_constraints(yaml);
        assert_eq!(c.attention_type, "ssm");
    }

    // ── normalize_input ─────────────────────────────────────────────────

    #[test]
    fn normalize_input_hyphens() {
        assert_eq!(normalize_input("falcon-h1"), "falcon_h1");
    }

    #[test]
    fn normalize_input_dots() {
        assert_eq!(normalize_input("qwen3.5"), "qwen3_5");
    }

    #[test]
    fn normalize_input_uppercase() {
        assert_eq!(normalize_input("Qwen2ForCausalLM"), "qwen2forcausallm");
    }

    #[test]
    fn normalize_input_mixed() {
        assert_eq!(normalize_input("Phi-3.5-mini"), "phi_3_5_mini");
    }

    #[test]
    fn normalize_input_already_normal() {
        assert_eq!(normalize_input("llama"), "llama");
    }

    // ── compact_input ───────────────────────────────────────────────────

    #[test]
    fn compact_input_removes_underscores() {
        assert_eq!(compact_input("phi_3"), Some("phi3".to_string()));
        assert_eq!(compact_input("gpt_2"), Some("gpt2".to_string()));
        assert_eq!(compact_input("rwkv_7"), Some("rwkv7".to_string()));
    }

    #[test]
    fn compact_input_no_underscores() {
        assert_eq!(compact_input("llama"), None);
        assert_eq!(compact_input("bert"), None);
    }

    #[test]
    fn compact_input_multiple_underscores() {
        assert_eq!(
            compact_input("qwen3_5_mini"),
            Some("qwen35mini".to_string())
        );
    }

    // ── strip_arch_suffix ───────────────────────────────────────────────

    #[test]
    fn strip_forcausallm() {
        assert_eq!(strip_arch_suffix("graniteforcausallm"), "granite");
    }

    #[test]
    fn strip_forconditionalgeneration() {
        assert_eq!(
            strip_arch_suffix("whisperforconditionalgeneration"),
            "whisper"
        );
    }

    #[test]
    fn strip_model_suffix() {
        assert_eq!(strip_arch_suffix("distilbertmodel"), "distilbert");
    }

    #[test]
    fn strip_no_match() {
        assert_eq!(strip_arch_suffix("llama"), "llama");
    }

    #[test]
    fn strip_empty_prefix_skipped() {
        // "forcausallm" alone should NOT strip to empty string
        assert_eq!(strip_arch_suffix("forcausallm"), "forcausallm");
    }

    // ── derive_kernel_class (all 9 classes) ─────────────────────────────

    #[test]
    fn derive_class_a_gqa() {
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
    fn derive_class_a_mha_degenerate() {
        // MHA is degenerate GQA — same kernel dispatch
        let c = Constraints {
            attention_type: "mha".into(),
            activation: "silu".into(),
            norm_type: "rmsnorm".into(),
            mlp_type: "swiglu".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::A);
    }

    #[test]
    fn derive_class_b() {
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
    fn derive_class_c() {
        let c = Constraints {
            attention_type: "mqa".into(),
            activation: "gelu".into(),
            norm_type: "layernorm".into(),
            mlp_type: "gelu_mlp".into(),
            positional_encoding: "alibi".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::C);
    }

    #[test]
    fn derive_class_d_gqa_layernorm() {
        let c = Constraints {
            attention_type: "gqa".into(),
            activation: "gelu".into(),
            norm_type: "layernorm".into(),
            mlp_type: "gelu_mlp".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::D);
    }

    #[test]
    fn derive_class_d_silu_layernorm() {
        let c = Constraints {
            attention_type: "mha".into(),
            activation: "silu".into(),
            norm_type: "layernorm".into(),
            mlp_type: "swiglu".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::D);
    }

    #[test]
    fn derive_class_f() {
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
    fn derive_class_f_checked_before_b() {
        // F requires RMSNorm+GELU+GatedMlp+RoPE. If MHA+RMSNorm+GELU+GatedMlp+RoPE,
        // should be F (not B, because B requires LayerNorm)
        let c = Constraints {
            attention_type: "mha".into(),
            activation: "gelu".into(),
            norm_type: "rmsnorm".into(),
            mlp_type: "gated_mlp".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::F);
    }

    #[test]
    fn derive_class_ssm() {
        let c = Constraints {
            attention_type: "ssm".into(),
            activation: "silu".into(),
            norm_type: "rmsnorm".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::Ssm);
    }

    #[test]
    fn derive_class_linear() {
        let c = Constraints {
            attention_type: "linear".into(),
            activation: "gelu".into(),
            norm_type: "layernorm".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::Linear);
    }

    #[test]
    fn derive_class_unknown_empty() {
        let c = Constraints::default();
        assert_eq!(derive_kernel_class(&c), KernelClass::Unknown);
    }

    #[test]
    fn derive_class_unknown_partial_match() {
        // Partial constraints that don't fit any class
        let c = Constraints {
            attention_type: "gqa".into(),
            activation: "relu".into(),
            norm_type: "rmsnorm".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), KernelClass::Unknown);
    }

    #[test]
    fn derive_deterministic() {
        let c = Constraints {
            attention_type: "gqa".into(),
            activation: "silu".into(),
            norm_type: "rmsnorm".into(),
            mlp_type: "swiglu".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        assert_eq!(derive_kernel_class(&c), derive_kernel_class(&c));
    }

    // ── KernelClass label/letter ────────────────────────────────────────

    #[test]
    fn kernel_class_label_all_variants() {
        let all = [
            KernelClass::A,
            KernelClass::B,
            KernelClass::C,
            KernelClass::D,
            KernelClass::E,
            KernelClass::F,
            KernelClass::Ssm,
            KernelClass::Linear,
            KernelClass::Unknown,
        ];
        for class in all {
            assert!(!class.label().is_empty());
            assert!(!class.letter().is_empty());
        }
    }

    #[test]
    fn kernel_class_letters_correct() {
        assert_eq!(KernelClass::A.letter(), "A");
        assert_eq!(KernelClass::Ssm.letter(), "SSM");
        assert_eq!(KernelClass::Linear.letter(), "Linear");
        assert_eq!(KernelClass::Unknown.letter(), "Unknown");
    }

    #[test]
    fn kernel_class_label_contains_letter() {
        assert!(KernelClass::A.label().starts_with("A "));
        assert!(KernelClass::Ssm.label().starts_with("SSM "));
    }

    // ── kernel_ops_for_class ────────────────────────────────────────────

    #[test]
    fn ops_class_a_has_gqa_rms_silu_swiglu_rope() {
        let ops = kernel_ops_for_class(KernelClass::A);
        assert!(ops.iter().any(|o| o.kernel == "gqa_forward"));
        assert!(ops.iter().any(|o| o.kernel == "rms_norm"));
        assert!(ops.iter().any(|o| o.kernel == "silu"));
        assert!(ops.iter().any(|o| o.kernel == "swiglu"));
        assert!(ops.iter().any(|o| o.kernel == "rope_forward"));
        assert!(ops.iter().any(|o| o.kernel == "softmax"));
    }

    #[test]
    fn ops_class_b_has_mha_layernorm_gelu() {
        let ops = kernel_ops_for_class(KernelClass::B);
        assert!(ops.iter().any(|o| o.kernel == "mha_forward"));
        assert!(ops.iter().any(|o| o.kernel == "layer_norm"));
        assert!(ops.iter().any(|o| o.kernel == "gelu"));
        assert!(ops.iter().any(|o| o.kernel == "gelu_mlp"));
        assert!(!ops.iter().any(|o| o.kernel == "rope_forward"));
    }

    #[test]
    fn ops_class_c_has_mqa_alibi() {
        let ops = kernel_ops_for_class(KernelClass::C);
        assert!(ops.iter().any(|o| o.kernel == "mqa_forward"));
        assert!(ops.iter().any(|o| o.kernel == "alibi"));
    }

    #[test]
    fn ops_class_d_has_gated_mlp() {
        let ops = kernel_ops_for_class(KernelClass::D);
        assert!(ops.iter().any(|o| o.kernel == "gated_mlp"));
        assert!(ops.iter().any(|o| o.kernel == "rope_forward"));
    }

    #[test]
    fn ops_class_e_has_moe_router() {
        let ops = kernel_ops_for_class(KernelClass::E);
        assert!(ops.iter().any(|o| o.kernel == "moe_routing"));
        assert!(ops.iter().any(|o| o.kernel == "swiglu"));
    }

    #[test]
    fn ops_class_f_has_gelu_gated_mlp() {
        let ops = kernel_ops_for_class(KernelClass::F);
        assert!(ops.iter().any(|o| o.kernel == "gelu"));
        assert!(ops.iter().any(|o| o.kernel == "gated_mlp"));
        assert!(ops.iter().any(|o| o.kernel == "rms_norm"));
    }

    #[test]
    fn ops_ssm_no_softmax() {
        let ops = kernel_ops_for_class(KernelClass::Ssm);
        assert!(!ops.iter().any(|o| o.kernel == "softmax"));
        assert!(ops.iter().any(|o| o.kernel == "selective_scan"));
        assert!(ops.iter().any(|o| o.kernel == "depthwise_conv1d"));
    }

    #[test]
    fn ops_linear_no_softmax() {
        let ops = kernel_ops_for_class(KernelClass::Linear);
        assert!(!ops.iter().any(|o| o.kernel == "softmax"));
        assert!(ops.iter().any(|o| o.kernel == "wkv_forward"));
        assert!(ops.iter().any(|o| o.kernel == "token_shift"));
        assert!(ops.iter().any(|o| o.kernel == "channel_mix"));
    }

    #[test]
    fn ops_unknown_minimal() {
        let ops = kernel_ops_for_class(KernelClass::Unknown);
        // Should have base ops only (MatVec Q4K, Q6K, Softmax, Fusion)
        assert_eq!(ops.len(), 4);
    }

    #[test]
    fn ops_all_classes_have_matvec() {
        let all = [
            KernelClass::A,
            KernelClass::B,
            KernelClass::C,
            KernelClass::D,
            KernelClass::E,
            KernelClass::F,
            KernelClass::Ssm,
            KernelClass::Linear,
            KernelClass::Unknown,
        ];
        for class in all {
            let ops = kernel_ops_for_class(class);
            assert!(
                ops.iter().any(|o| o.kernel == "fused_q4k_parallel_matvec"),
                "Class {:?} missing Q4K matvec",
                class
            );
        }
    }

    // ── kernel_ops_for_family (RoPE enrichment) ─────────────────────────

    #[test]
    fn family_ops_phi_gets_rope_enrichment() {
        // Phi is Class B (no RoPE in base ops) but uses RoPE positional encoding
        let c = Constraints {
            attention_type: "mha".into(),
            activation: "gelu".into(),
            norm_type: "layernorm".into(),
            mlp_type: "gelu_mlp".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        let ops = kernel_ops_for_family(KernelClass::B, &c);
        assert!(
            ops.iter().any(|o| o.kernel == "rope_forward"),
            "Phi (class B + rope) should get rope_forward enrichment"
        );
    }

    #[test]
    fn family_ops_no_double_rope() {
        // Class A already has RoPE — enrichment should not duplicate it
        let c = Constraints {
            positional_encoding: "rope".into(),
            ..Default::default()
        };
        let ops = kernel_ops_for_family(KernelClass::A, &c);
        let rope_count = ops.iter().filter(|o| o.kernel == "rope_forward").count();
        assert_eq!(rope_count, 1, "Should not duplicate rope_forward");
    }

    #[test]
    fn family_ops_absolute_no_rope() {
        let c = Constraints {
            positional_encoding: "absolute".into(),
            ..Default::default()
        };
        let ops = kernel_ops_for_family(KernelClass::B, &c);
        assert!(!ops.iter().any(|o| o.kernel == "rope_forward"));
    }

    // ── load_families ───────────────────────────────────────────────────

    #[test]
    fn load_families_count() {
        let families = load_families();
        assert_eq!(families.len(), FAMILY_YAMLS.len());
    }

    #[test]
    fn load_families_expected_classes() {
        let families = load_families();
        let find = |name: &str| families.iter().find(|f| f.family == name).unwrap();

        assert_eq!(find("llama").kernel_class, KernelClass::A);
        assert_eq!(find("qwen2").kernel_class, KernelClass::A);
        assert_eq!(find("qwen3").kernel_class, KernelClass::A);
        assert_eq!(find("mistral").kernel_class, KernelClass::A);
        assert_eq!(find("deepseek").kernel_class, KernelClass::A);
        assert_eq!(find("falcon_h1").kernel_class, KernelClass::A);
        assert_eq!(find("openelm").kernel_class, KernelClass::A);

        assert_eq!(find("bert").kernel_class, KernelClass::B);
        assert_eq!(find("gpt2").kernel_class, KernelClass::B);
        assert_eq!(find("whisper").kernel_class, KernelClass::B);

        assert_eq!(find("gemma").kernel_class, KernelClass::F);

        assert_eq!(find("mamba").kernel_class, KernelClass::Ssm);
        assert_eq!(find("rwkv7").kernel_class, KernelClass::Linear);
    }

    #[test]
    fn load_families_all_have_display_name() {
        for f in &load_families() {
            assert!(
                !f.display_name.is_empty(),
                "Family {} missing display_name",
                f.family
            );
        }
    }

    // ── resolve_family ──────────────────────────────────────────────────

    // Direct matches
    #[test]
    fn resolve_direct_llama() {
        let f = resolve_family("llama").unwrap();
        assert_eq!(f.family, "llama");
        assert_eq!(f.kernel_class, KernelClass::A);
    }

    #[test]
    fn resolve_direct_bert() {
        let f = resolve_family("bert").unwrap();
        assert_eq!(f.family, "bert");
        assert_eq!(f.kernel_class, KernelClass::B);
    }

    #[test]
    fn resolve_direct_mamba() {
        let f = resolve_family("mamba").unwrap();
        assert_eq!(f.family, "mamba");
        assert_eq!(f.kernel_class, KernelClass::Ssm);
    }

    #[test]
    fn resolve_direct_rwkv7() {
        let f = resolve_family("rwkv7").unwrap();
        assert_eq!(f.family, "rwkv7");
        assert_eq!(f.kernel_class, KernelClass::Linear);
    }

    // Normalized matches (hyphens, dots, case)
    #[test]
    fn resolve_normalized_hyphens() {
        let f = resolve_family("falcon-h1").unwrap();
        assert_eq!(f.family, "falcon_h1");
    }

    #[test]
    fn resolve_normalized_dots() {
        let f = resolve_family("qwen3.5").unwrap();
        assert_eq!(f.family, "qwen3_5");
    }

    #[test]
    fn resolve_normalized_uppercase() {
        let f = resolve_family("LLAMA").unwrap();
        assert_eq!(f.family, "llama");
    }

    // Cross-compact matches
    #[test]
    fn resolve_cross_compact_qwen_3_5() {
        // "qwen-3-5" → normalized "qwen_3_5" → compact "qwen35"
        // family "qwen3_5" → compact "qwen35" → match
        let f = resolve_family("qwen-3-5").unwrap();
        assert_eq!(f.family, "qwen3_5");
    }

    // Alias matches
    #[test]
    fn resolve_alias_mixtral() {
        let f = resolve_family("mixtral").unwrap();
        assert_eq!(f.family, "mistral");
        assert!(f.display_name.contains("via"));
    }

    #[test]
    fn resolve_alias_phi3() {
        let f = resolve_family("phi3").unwrap();
        assert_eq!(f.family, "llama");
        assert!(f.display_name.contains("phi3"));
    }

    #[test]
    fn resolve_alias_phi4() {
        let f = resolve_family("phi4").unwrap();
        assert_eq!(f.family, "llama");
    }

    #[test]
    fn resolve_alias_bloom() {
        let f = resolve_family("bloom").unwrap();
        assert_eq!(f.family, "bert");
    }

    #[test]
    fn resolve_alias_falcon() {
        let f = resolve_family("falcon").unwrap();
        assert_eq!(f.family, "bert");
    }

    #[test]
    fn resolve_alias_smollm() {
        let f = resolve_family("smollm").unwrap();
        assert_eq!(f.family, "llama");
    }

    #[test]
    fn resolve_alias_smollm2() {
        let f = resolve_family("smollm2").unwrap();
        assert_eq!(f.family, "llama");
    }

    #[test]
    fn resolve_alias_codegemma() {
        let f = resolve_family("codegemma").unwrap();
        assert_eq!(f.family, "gemma");
    }

    #[test]
    fn resolve_alias_gpt_neo() {
        let f = resolve_family("gpt_neo").unwrap();
        assert_eq!(f.family, "bert");
    }

    #[test]
    fn resolve_alias_gptneo_compact() {
        let f = resolve_family("gptneo").unwrap();
        assert_eq!(f.family, "bert");
    }

    #[test]
    fn resolve_alias_starcoder2() {
        let f = resolve_family("starcoder2").unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn resolve_alias_vicuna() {
        let f = resolve_family("vicuna").unwrap();
        assert_eq!(f.family, "llama");
    }

    #[test]
    fn resolve_alias_qwq() {
        let f = resolve_family("qwq").unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn resolve_alias_qwen2_moe() {
        let f = resolve_family("qwen2_moe").unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn resolve_alias_normalized_gpt_j_hyphen() {
        let f = resolve_family("gpt-j").unwrap();
        assert_eq!(f.family, "bert");
    }

    // Architecture string match
    #[test]
    fn resolve_arch_qwen2forcausallm() {
        let f = resolve_family("Qwen2ForCausalLM").unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn resolve_arch_bertmodel() {
        let f = resolve_family("BertModel").unwrap();
        assert_eq!(f.family, "bert");
    }

    // Architecture string → stripped → alias re-check
    #[test]
    fn resolve_stripped_graniteforcausallm() {
        let f = resolve_family("GraniteForCausalLM").unwrap();
        assert_eq!(f.family, "llama");
    }

    // Partial match (3-pass)
    #[test]
    fn resolve_partial_qwen_matches_qwen2() {
        // "qwen" is prefix of family "qwen2" → Pass 2
        let f = resolve_family("qwen").unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn resolve_partial_phi3mini_via_alias() {
        // "phi3mini" starts_with alias "phi3" → Pass 1
        let f = resolve_family("phi-3-mini").unwrap();
        assert_eq!(f.family, "llama");
        assert!(f.display_name.contains("phi3"));
    }

    #[test]
    fn resolve_partial_gpt_matches_gpt2() {
        let f = resolve_family("gpt").unwrap();
        assert_eq!(f.family, "gpt2");
    }

    // Edge cases
    #[test]
    fn resolve_empty_string() {
        assert!(resolve_family("").is_none());
    }

    #[test]
    fn resolve_whitespace_only() {
        assert!(resolve_family("   ").is_none());
    }

    #[test]
    fn resolve_emoji_stripped() {
        // Non-ASCII chars stripped, leaving empty → None
        assert!(resolve_family("🦙").is_none());
    }

    #[test]
    fn resolve_emoji_with_text() {
        // "🦙llama" → strip non-ASCII → "llama" → match
        let f = resolve_family("🦙llama").unwrap();
        assert_eq!(f.family, "llama");
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve_family("nonexistent-xyz-123").is_none());
    }

    #[test]
    fn resolve_short_input_no_partial() {
        // "ab" is < 3 chars, partial matching skipped
        assert!(resolve_family("ab").is_none());
    }

    // ── extract_json_string ─────────────────────────────────────────────

    #[test]
    fn json_string_value() {
        let json = r#"{"model_type": "qwen2"}"#;
        assert_eq!(
            extract_json_string(json, "model_type"),
            Some("qwen2".to_string())
        );
    }

    #[test]
    fn json_numeric_value() {
        let json = r#"{"hidden_size": 4096, "other": 1}"#;
        assert_eq!(
            extract_json_string(json, "hidden_size"),
            Some("4096".to_string())
        );
    }

    #[test]
    fn json_float_value() {
        let json = r#"{"rms_norm_eps": 1e-06, "x": 1}"#;
        assert_eq!(
            extract_json_string(json, "rms_norm_eps"),
            Some("1e-06".to_string())
        );
    }

    #[test]
    fn json_boolean_value() {
        let json = r#"{"tie_word_embeddings": true, "x": 1}"#;
        assert_eq!(
            extract_json_string(json, "tie_word_embeddings"),
            Some("true".to_string())
        );
    }

    #[test]
    fn json_null_returns_none() {
        let json = r#"{"model_type": null, "x": 1}"#;
        assert_eq!(extract_json_string(json, "model_type"), None);
    }

    #[test]
    fn json_array_returns_none() {
        let json = r#"{"architectures": ["QwenForCausalLM"], "x": 1}"#;
        assert_eq!(extract_json_string(json, "architectures"), None);
    }

    #[test]
    fn json_object_returns_none() {
        let json = r#"{"rope_scaling": {"type": "yarn"}, "x": 1}"#;
        assert_eq!(extract_json_string(json, "rope_scaling"), None);
    }

    #[test]
    fn json_missing_key() {
        let json = r#"{"model_type": "bert"}"#;
        assert_eq!(extract_json_string(json, "hidden_act"), None);
    }

    #[test]
    fn json_whitespace_around_colon() {
        let json = "{\n  \"hidden_size\" : 4096,\n  \"x\": 1\n}";
        assert_eq!(
            extract_json_string(json, "hidden_size"),
            Some("4096".to_string())
        );
    }

    // ── enrich_rationale ────────────────────────────────────────────────

    #[test]
    fn enrich_hidden_act_silu() {
        let r = enrich_rationale("hidden_act", "silu", "{}").unwrap();
        assert!(r.contains("SiLU"));
    }

    #[test]
    fn enrich_hidden_act_gelu() {
        let r = enrich_rationale("hidden_act", "gelu", "{}").unwrap();
        assert!(r.contains("GELU"));
    }

    #[test]
    fn enrich_hidden_act_gelu_new() {
        let r = enrich_rationale("hidden_act", "gelu_new", "{}").unwrap();
        assert!(r.contains("GELU"));
    }

    #[test]
    fn enrich_hidden_act_unknown() {
        let r = enrich_rationale("hidden_act", "relu", "{}").unwrap();
        assert!(r.contains("relu"));
    }

    #[test]
    fn enrich_rms_norm_eps() {
        let r = enrich_rationale("rms_norm_eps", "1e-06", "{}").unwrap();
        assert!(r.contains("RMSNorm"));
    }

    #[test]
    fn enrich_num_kv_heads_gqa() {
        let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 8}"#;
        let r = enrich_rationale("num_key_value_heads", "8", json).unwrap();
        assert!(r.contains("GQA"));
        assert!(r.contains("8"));
        assert!(r.contains("32"));
    }

    #[test]
    fn enrich_num_kv_heads_mqa() {
        let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 1}"#;
        let r = enrich_rationale("num_key_value_heads", "1", json).unwrap();
        assert!(r.contains("MQA"));
    }

    #[test]
    fn enrich_num_kv_heads_mha() {
        let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 32}"#;
        let r = enrich_rationale("num_key_value_heads", "32", json).unwrap();
        assert!(r.contains("MHA"));
    }

    #[test]
    fn enrich_rope_theta() {
        let r = enrich_rationale("rope_theta", "10000.0", "{}").unwrap();
        assert!(r.contains("RoPE"));
    }

    #[test]
    fn enrich_intermediate_size_swiglu() {
        let json = r#"{"hidden_size": 4096, "intermediate_size": 11008, "hidden_act": "silu"}"#;
        let r = enrich_rationale("intermediate_size", "11008", json).unwrap();
        assert!(r.contains("SwiGLU"));
    }

    #[test]
    fn enrich_intermediate_size_gelu() {
        let json = r#"{"hidden_size": 768, "intermediate_size": 3072, "hidden_act": "gelu"}"#;
        let r = enrich_rationale("intermediate_size", "3072", json).unwrap();
        assert!(r.contains("GELU"));
    }

    #[test]
    fn enrich_num_local_experts_moe() {
        let r = enrich_rationale("num_local_experts", "8", "{}").unwrap();
        assert!(r.contains("MoE"));
        assert!(r.contains("8"));
    }

    #[test]
    fn enrich_num_experts_single() {
        let r = enrich_rationale("num_local_experts", "1", "{}").unwrap();
        assert!(r.contains("dense"));
    }

    #[test]
    fn enrich_num_experts_negative() {
        let r = enrich_rationale("num_local_experts", "-1", "{}").unwrap();
        assert!(r.contains("negative"));
    }

    #[test]
    fn enrich_tie_word_embeddings_true() {
        let r = enrich_rationale("tie_word_embeddings", "true", "{}").unwrap();
        assert!(r.contains("Shared"));
    }

    #[test]
    fn enrich_tie_word_embeddings_false() {
        let r = enrich_rationale("tie_word_embeddings", "false", "{}").unwrap();
        assert!(r.contains("Separate"));
    }

    #[test]
    fn enrich_num_attention_heads_gqa() {
        let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 4}"#;
        let r = enrich_rationale("num_attention_heads", "32", json).unwrap();
        assert!(r.contains("GQA"));
    }

    #[test]
    fn enrich_num_attention_heads_mha() {
        let json = r#"{"num_attention_heads": 12, "num_key_value_heads": 12}"#;
        let r = enrich_rationale("num_attention_heads", "12", json).unwrap();
        assert!(r.contains("MHA"));
    }

    #[test]
    fn enrich_hidden_size_with_params() {
        let json = r#"{"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008, "hidden_act": "silu", "vocab_size": 32000, "num_attention_heads": 32, "num_key_value_heads": 8}"#;
        let r = enrich_rationale("hidden_size", "4096", json).unwrap();
        assert!(r.contains("Hidden dim"));
        assert!(r.contains("params"));
    }

    #[test]
    fn enrich_hidden_size_gelu_model() {
        let json = r#"{"hidden_size": 768, "num_hidden_layers": 12, "intermediate_size": 3072, "hidden_act": "gelu", "vocab_size": 30522, "num_attention_heads": 12, "num_key_value_heads": 12}"#;
        let r = enrich_rationale("hidden_size", "768", json).unwrap();
        assert!(r.contains("Hidden dim"));
        assert!(r.contains("params"));
    }

    #[test]
    fn enrich_num_hidden_layers() {
        let r = enrich_rationale("num_hidden_layers", "32", "{}").unwrap();
        assert!(r.contains("32"));
        assert!(r.contains("layers"));
    }

    #[test]
    fn enrich_vocab_size_with_hidden() {
        let json = r#"{"vocab_size": 32000, "hidden_size": 4096}"#;
        let r = enrich_rationale("vocab_size", "32000", json).unwrap();
        assert!(r.contains("32000"));
        assert!(r.contains("MB"));
    }

    #[test]
    fn enrich_max_position_1m() {
        let r = enrich_rationale("max_position_embeddings", "1048576", "{}").unwrap();
        assert!(r.contains("1M+"));
    }

    #[test]
    fn enrich_max_position_128k() {
        let r = enrich_rationale("max_position_embeddings", "131072", "{}").unwrap();
        assert!(r.contains("128K+"));
    }

    #[test]
    fn enrich_max_position_8k() {
        let r = enrich_rationale("max_position_embeddings", "8192", "{}").unwrap();
        assert!(r.contains("8K+"));
    }

    #[test]
    fn enrich_max_position_small() {
        let r = enrich_rationale("max_position_embeddings", "512", "{}").unwrap();
        assert!(r.contains("512"));
        assert!(!r.contains("K+"));
    }

    #[test]
    fn enrich_unknown_key() {
        assert!(enrich_rationale("unknown_key", "val", "{}").is_none());
    }

    #[test]
    fn enrich_num_experts_per_tok() {
        let r = enrich_rationale("num_experts_per_tok", "2", "{}").unwrap();
        assert!(r.contains("2"));
        assert!(r.contains("experts"));
    }

    #[test]
    fn enrich_num_experts_per_tok_one() {
        let r = enrich_rationale("num_experts_per_tok", "1", "{}").unwrap();
        assert!(r.contains("1"));
        assert!(r.contains("expert"));
        // singular
        assert!(!r.contains("experts"));
    }

    // ── detect_constraint_mismatches ────────────────────────────────────

    fn make_family(name: &str, constraints: Constraints) -> FamilyInfo {
        let kernel_class = derive_kernel_class(&constraints);
        FamilyInfo {
            family: name.to_string(),
            display_name: name.to_string(),
            architectures: vec![],
            constraints,
            kernel_class,
        }
    }

    fn make_config(entries: &[(&str, &str)]) -> BTreeMap<String, ConfigField> {
        entries
            .iter()
            .map(|(k, v)| {
                (
                    (*k).to_string(),
                    ConfigField {
                        value: (*v).to_string(),
                        rationale: String::new(),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn mismatch_activation_silu_vs_gelu() {
        let family = make_family(
            "test",
            Constraints {
                activation: "silu".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("hidden_act", "gelu")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(
            warnings.iter().any(|w| w.contains("Activation mismatch")),
            "Expected activation mismatch warning"
        );
    }

    #[test]
    fn mismatch_activation_gelu_vs_silu() {
        let family = make_family(
            "test",
            Constraints {
                activation: "gelu".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("hidden_act", "silu")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("Activation mismatch")));
    }

    #[test]
    fn mismatch_activation_gegelu() {
        let family = make_family(
            "test",
            Constraints {
                activation: "gelu".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("hidden_act", "gegelu")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("gegelu")));
    }

    #[test]
    fn mismatch_norm_rms_vs_layernorm() {
        let family = make_family(
            "test",
            Constraints {
                norm_type: "layernorm".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("rms_norm_eps", "1e-06")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("Norm mismatch")));
    }

    #[test]
    fn mismatch_norm_layernorm_vs_rms() {
        let family = make_family(
            "test",
            Constraints {
                norm_type: "rmsnorm".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("layer_norm_epsilon", "1e-05")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("Norm mismatch")));
    }

    #[test]
    fn mismatch_conflicting_norm_fields() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("rms_norm_eps", "1e-06"), ("layer_norm_epsilon", "1e-05")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("Conflicting norm")));
    }

    #[test]
    fn mismatch_attention_gqa_vs_mha() {
        let family = make_family(
            "test",
            Constraints {
                attention_type: "mha".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("num_key_value_heads", "4"), ("num_attention_heads", "32")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("Attention mismatch")));
    }

    #[test]
    fn mismatch_mha_degenerate_gqa_suppressed() {
        // MHA (kv==q) is degenerate GQA — no warning
        let family = make_family(
            "test",
            Constraints {
                attention_type: "gqa".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("num_key_value_heads", "32"), ("num_attention_heads", "32")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(
            !warnings.iter().any(|w| w.contains("Attention mismatch")),
            "MHA-as-GQA should not warn"
        );
    }

    #[test]
    fn mismatch_kv_greater_than_q() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("num_key_value_heads", "64"), ("num_attention_heads", "32")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("cannot exceed")));
    }

    #[test]
    fn mismatch_invalid_gqa_grouping() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("num_key_value_heads", "5"), ("num_attention_heads", "32")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("not divisible")));
    }

    #[test]
    fn mismatch_multi_query_flag() {
        let family = make_family(
            "test",
            Constraints {
                attention_type: "mha".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("multi_query", "true")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("multi_query")));
    }

    #[test]
    fn mismatch_moe_non_e_class() {
        let family = make_family(
            "test",
            Constraints {
                attention_type: "gqa".into(),
                activation: "silu".into(),
                norm_type: "rmsnorm".into(),
                mlp_type: "swiglu".into(),
                positional_encoding: "rope".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[("num_local_experts", "8")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("MoE")));
    }

    #[test]
    fn mismatch_moe_negative_experts() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("num_local_experts", "-1")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("negative")));
    }

    #[test]
    fn mismatch_negative_hidden_size() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("hidden_size", "-1")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("negative")));
    }

    #[test]
    fn mismatch_zero_hidden_size() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("hidden_size", "0")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("zero")));
    }

    #[test]
    fn mismatch_implausible_hidden_size() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("hidden_size", "999999")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("Implausible")));
    }

    #[test]
    fn mismatch_hidden_not_divisible_by_heads() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("hidden_size", "5120"), ("num_attention_heads", "24")]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("not divisible")));
    }

    #[test]
    fn mismatch_hidden_divisibility_skipped_with_explicit_head_dim() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[
            ("hidden_size", "5120"),
            ("num_attention_heads", "24"),
            ("head_dim", "256"),
        ]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(
            !warnings.iter().any(|w| w.contains("not divisible")),
            "Should skip divisibility check when head_dim is explicit"
        );
    }

    #[test]
    fn mismatch_model_type_arch_conflict() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[
            ("model_type", "llama"),
            ("_architectures", "MistralForCausalLM"),
        ]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("conflicts")));
    }

    #[test]
    fn mismatch_model_type_arch_no_conflict_deepseek() {
        // deepseek_v2 vs DeepseekV2ForCausalLM: compact forms match
        let family = make_family("test", Constraints::default());
        let config = make_config(&[
            ("model_type", "deepseek_v2"),
            ("_architectures", "DeepseekV2ForCausalLM"),
        ]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(
            !warnings.iter().any(|w| w.contains("conflicts")),
            "deepseek_v2 vs DeepseekV2 should not conflict"
        );
    }

    #[test]
    fn mismatch_moe_from_alias_name() {
        let family = FamilyInfo {
            family: "mistral".to_string(),
            display_name: "mixtral (via mistral kernel pipeline)".to_string(),
            architectures: vec![],
            constraints: Constraints::default(),
            kernel_class: KernelClass::A,
        };
        let config = make_config(&[]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(warnings.iter().any(|w| w.contains("MoE")));
    }

    #[test]
    fn no_mismatch_clean_config() {
        let family = make_family(
            "test",
            Constraints {
                attention_type: "gqa".into(),
                activation: "silu".into(),
                norm_type: "rmsnorm".into(),
                ..Default::default()
            },
        );
        let config = make_config(&[
            ("hidden_act", "silu"),
            ("rms_norm_eps", "1e-06"),
            ("num_key_value_heads", "8"),
            ("num_attention_heads", "32"),
            ("hidden_size", "4096"),
        ]);
        let warnings = detect_constraint_mismatches(&family, &config);
        assert!(
            warnings.is_empty(),
            "Expected no warnings, got: {warnings:?}"
        );
    }

    // ── extract_architecture_display ────────────────────────────────────

    #[test]
    fn arch_display_from_config_architectures() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("_architectures", "LlamaForCausalLM")]);
        assert_eq!(
            extract_architecture_display(&family, &config),
            "LlamaForCausalLM"
        );
    }

    #[test]
    fn arch_display_from_model_type() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[("model_type", "llama")]);
        assert_eq!(extract_architecture_display(&family, &config), "llama");
    }

    #[test]
    fn arch_display_alias_uses_alias_arch_table() {
        let family = FamilyInfo {
            family: "llama".to_string(),
            display_name: "vicuna (via llama kernel pipeline)".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string()],
            constraints: Constraints::default(),
            kernel_class: KernelClass::A,
        };
        let config = make_config(&[]);
        assert_eq!(
            extract_architecture_display(&family, &config),
            "LlamaForCausalLM"
        );
    }

    #[test]
    fn arch_display_fallback_to_family_arch() {
        let family = FamilyInfo {
            family: "llama".to_string(),
            display_name: "LLaMA".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string()],
            constraints: Constraints::default(),
            kernel_class: KernelClass::A,
        };
        let config = make_config(&[]);
        assert_eq!(
            extract_architecture_display(&family, &config),
            "LlamaForCausalLM"
        );
    }

    #[test]
    fn arch_display_no_archs() {
        let family = make_family("test", Constraints::default());
        let config = make_config(&[]);
        assert_eq!(extract_architecture_display(&family, &config), "Unknown");
    }

    // ── proof_status ────────────────────────────────────────────────────

    #[test]
    fn proof_all_known_contracts() {
        for (name, _) in KERNEL_CONTRACTS {
            let proof = proof_status_for_contract(name);
            assert_ne!(
                proof.level,
                ProofLevel::Unknown,
                "Contract {name} should be known"
            );
        }
    }

    #[test]
    fn proof_unknown_contract() {
        let proof = proof_status_for_contract("does-not-exist-v1");
        assert_eq!(proof.level, ProofLevel::Unknown);
        assert!(proof.evidence.contains("No contract"));
    }

    #[test]
    fn proof_level_symbols() {
        assert_eq!(ProofLevel::Proven.symbol(), "✓");
        assert_eq!(ProofLevel::Tested.symbol(), "◉");
        assert_eq!(ProofLevel::Documented.symbol(), "○");
        assert_eq!(ProofLevel::Unknown.symbol(), "?");
    }

    #[test]
    fn proof_level_labels() {
        assert_eq!(ProofLevel::Proven.label(), "Proven");
        assert_eq!(ProofLevel::Tested.label(), "Tested");
        assert_eq!(ProofLevel::Documented.label(), "Documented");
        assert_eq!(ProofLevel::Unknown.label(), "Unknown");
    }

    // ── proof_status_for_class ──────────────────────────────────────────

    #[test]
    fn proof_class_deduplicates_contracts() {
        let proofs = proof_status_for_class(KernelClass::A);
        let names: Vec<&str> = proofs.iter().map(|p| p.contract.as_str()).collect();
        // Verify no duplicates
        let mut seen = Vec::new();
        for name in &names {
            assert!(
                !seen.contains(name),
                "Duplicate contract in proof list: {name}"
            );
            seen.push(*name);
        }
    }

    #[test]
    fn proof_class_unknown_has_some_proofs() {
        // Unknown still has base ops (MatVec, Softmax, Fusion)
        let proofs = proof_status_for_class(KernelClass::Unknown);
        assert!(!proofs.is_empty());
    }

    // ── build_json_output ───────────────────────────────────────────────

    #[test]
    fn json_output_basic() {
        let family = resolve_family("llama").unwrap();
        let config = make_config(&[]);
        let json = build_json_output(&family, config, false);
        assert_eq!(json.family, "llama");
        assert_eq!(json.kernel_class, "A");
        assert_eq!(json.layout, "row_major");
        assert!(json.proof_summary.is_none());
    }

    #[test]
    fn json_output_with_proof() {
        let family = resolve_family("llama").unwrap();
        let config = make_config(&[]);
        let json = build_json_output(&family, config, true);
        assert!(json.proof_summary.is_some());
        let ps = json.proof_summary.unwrap();
        assert!(ps.total > 0);
    }

    #[test]
    fn json_output_internal_fields_removed() {
        let family = resolve_family("llama").unwrap();
        let mut config = make_config(&[("_architectures", "LlamaForCausalLM")]);
        config.insert(
            "model_type".to_string(),
            ConfigField {
                value: "llama".to_string(),
                rationale: "test".to_string(),
            },
        );
        let json = build_json_output(&family, config, false);
        assert!(
            !json.config_mapping.contains_key("_architectures"),
            "Internal fields should be stripped"
        );
    }

    #[test]
    fn json_output_equivalence_class() {
        let family = resolve_family("qwen2").unwrap();
        let json = build_json_output(&family, make_config(&[]), false);
        // Class A has multiple families
        assert!(json.equivalence_class_families.len() > 1);
        assert!(json
            .equivalence_class_families
            .contains(&"llama".to_string()));
    }

    // ── FAMILY_ALIASES coverage ─────────────────────────────────────────

    #[test]
    fn all_aliases_resolve_to_valid_family() {
        let families = load_families();
        for (alias, target) in FAMILY_ALIASES {
            assert!(
                families.iter().any(|f| f.family == *target),
                "Alias {alias} → {target}: target family not found in loaded families"
            );
        }
    }

    #[test]
    fn all_aliases_resolvable() {
        for (alias, target) in FAMILY_ALIASES {
            let resolved = resolve_family(alias);
            assert!(
                resolved.is_some(),
                "Alias {alias} should resolve (expected target: {target})"
            );
            assert_eq!(
                resolved.unwrap().family,
                *target,
                "Alias {alias} resolved to wrong family"
            );
        }
    }

    // ── ALIAS_ARCHITECTURES coverage ────────────────────────────────────

    #[test]
    fn all_alias_architectures_have_matching_alias() {
        for (alias_name, _hf_arch) in ALIAS_ARCHITECTURES {
            assert!(
                FAMILY_ALIASES.iter().any(|(a, _)| a == alias_name),
                "ALIAS_ARCHITECTURES has {alias_name} but no matching FAMILY_ALIASES entry"
            );
        }
    }

    // ── Regression tests ────────────────────────────────────────────────

    #[test]
    fn regression_phi3_mini_resolves_to_class_a() {
        // phi-3-mini was resolving to phi (Class B/D) instead of phi3 alias (Class A)
        let f = resolve_family("phi-3-mini").unwrap();
        assert_eq!(f.family, "llama");
        assert_eq!(f.kernel_class, KernelClass::A);
    }

    #[test]
    fn regression_qwen_resolves_to_qwen2_not_moe() {
        // "qwen" was resolving to qwen2_moe alias before family match
        let f = resolve_family("qwen").unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn regression_phi2_resolves_to_phi() {
        // phi-2 is the phi family directly (Class B/D, NOT phi3→llama alias)
        let f = resolve_family("phi").unwrap();
        assert_eq!(f.family, "phi");
    }

    #[test]
    fn regression_gemma2_alias() {
        let f = resolve_family("gemma2").unwrap();
        assert_eq!(f.family, "gemma");
        assert_eq!(f.kernel_class, KernelClass::F);
    }

    #[test]
    fn regression_mamba_is_ssm() {
        let f = resolve_family("mamba").unwrap();
        assert_eq!(f.kernel_class, KernelClass::Ssm);
        let ops = kernel_ops_for_class(KernelClass::Ssm);
        assert!(!ops.iter().any(|o| o.kernel == "softmax"));
    }

    #[test]
    fn regression_rwkv7_is_linear() {
        let f = resolve_family("rwkv7").unwrap();
        assert_eq!(f.kernel_class, KernelClass::Linear);
        let ops = kernel_ops_for_class(KernelClass::Linear);
        assert!(!ops.iter().any(|o| o.kernel == "softmax"));
    }

    // ── Parameter estimate tests ────────────────────────────────────────

    #[test]
    fn param_estimate_gqa_model() {
        // Qwen2.5-7B: hidden=3584, layers=28, inter=18944, heads=28, kv=4, vocab=152064
        let json = r#"{"hidden_size": 3584, "num_hidden_layers": 28, "intermediate_size": 18944, "hidden_act": "silu", "vocab_size": 152064, "num_attention_heads": 28, "num_key_value_heads": 4}"#;
        let r = enrich_rationale("hidden_size", "3584", json).unwrap();
        // Should estimate ~7-8B
        assert!(r.contains("B params"), "Expected B params in: {r}");
    }

    #[test]
    fn param_estimate_gelu_model() {
        // BERT-base: hidden=768, layers=12, inter=3072, heads=12, kv=12, vocab=30522
        let json = r#"{"hidden_size": 768, "num_hidden_layers": 12, "intermediate_size": 3072, "hidden_act": "gelu", "vocab_size": 30522, "num_attention_heads": 12, "num_key_value_heads": 12, "tie_word_embeddings": "true"}"#;
        let r = enrich_rationale("hidden_size", "768", json).unwrap();
        // BERT-base is ~110M
        assert!(r.contains("M params"), "Expected M params in: {r}");
    }

    #[test]
    fn param_estimate_moe_model() {
        // Model with experts should include expert weights
        let json = r#"{"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008, "hidden_act": "silu", "vocab_size": 32000, "num_attention_heads": 32, "num_key_value_heads": 8, "num_local_experts": 8, "moe_intermediate_size": 4096}"#;
        let r = enrich_rationale("hidden_size", "4096", json).unwrap();
        assert!(r.contains("B params"), "MoE model should be large: {r}");
    }

    // ── ConfigField type ────────────────────────────────────────────────

    #[test]
    fn config_field_construction() {
        let field = ConfigField {
            value: "silu".to_string(),
            rationale: "SiLU activation".to_string(),
        };
        assert_eq!(field.value, "silu");
        assert_eq!(field.rationale, "SiLU activation");
    }

    // ── family_aliases() accessor ───────────────────────────────────────

    #[test]
    fn family_aliases_not_empty() {
        assert!(!family_aliases().is_empty());
        assert!(family_aliases().len() >= 50, "Expected at least 50 aliases");
    }

    // ── Constraints struct ──────────────────────────────────────────────

    #[test]
    fn constraints_default_all_empty() {
        let c = Constraints::default();
        assert_eq!(c.attention_type, "");
        assert_eq!(c.activation, "");
        assert_eq!(c.norm_type, "");
        assert_eq!(c.mlp_type, "");
        assert_eq!(c.positional_encoding, "");
        assert!(!c.has_bias);
        assert!(!c.tied_embeddings);
    }

    // ── FamilyInfo struct ───────────────────────────────────────────────

    #[test]
    fn family_info_clone_eq() {
        let f = resolve_family("llama").unwrap();
        let f2 = f.clone();
        assert_eq!(f.family, f2.family);
        assert_eq!(f.kernel_class, f2.kernel_class);
    }

    // ── Integration: round-trip family → class → ops → proof ────────────

    #[test]
    fn integration_llama_full_pipeline() {
        let family = resolve_family("llama").unwrap();
        assert_eq!(family.kernel_class, KernelClass::A);
        let ops = kernel_ops_for_family(family.kernel_class, &family.constraints);
        assert!(ops.len() > 4);
        let proofs = proof_status_for_class(family.kernel_class);
        assert!(!proofs.is_empty());
        let json = build_json_output(&family, BTreeMap::new(), true);
        assert_eq!(json.kernel_class, "A");
        assert!(json.proof_summary.is_some());
    }

    #[test]
    fn integration_bert_full_pipeline() {
        let family = resolve_family("bert").unwrap();
        assert_eq!(family.kernel_class, KernelClass::B);
        let ops = kernel_ops_for_class(KernelClass::B);
        assert!(!ops.iter().any(|o| o.kernel == "rope_forward"));
    }

    #[test]
    fn integration_gemma_full_pipeline() {
        let family = resolve_family("gemma").unwrap();
        assert_eq!(family.kernel_class, KernelClass::F);
        let ops = kernel_ops_for_class(KernelClass::F);
        assert!(ops.iter().any(|o| o.kernel == "gated_mlp"));
        assert!(ops.iter().any(|o| o.kernel == "gelu"));
    }

    #[test]
    fn integration_mamba_full_pipeline() {
        let family = resolve_family("mamba").unwrap();
        assert_eq!(family.kernel_class, KernelClass::Ssm);
        let ops = kernel_ops_for_class(KernelClass::Ssm);
        assert!(ops.iter().any(|o| o.kernel == "selective_scan"));
    }

    #[test]
    fn integration_alias_full_pipeline() {
        let family = resolve_family("mixtral").unwrap();
        assert!(family.display_name.contains("via"));
        let json = build_json_output(&family, BTreeMap::new(), false);
        assert!(json.display_name.contains("via"));
    }

    // ── resolve_from_config_json (filesystem tests) ─────────────────────

    #[test]
    fn resolve_config_json_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, r#"{"model_type": "qwen2", "hidden_size": 4096}"#).unwrap();
        let f = resolve_from_config_json(&path).unwrap();
        assert_eq!(f.family, "qwen2");
    }

    #[test]
    fn resolve_config_json_no_model_type_uses_architectures() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            r#"{"architectures": ["LlamaForCausalLM"], "hidden_size": 4096}"#,
        )
        .unwrap();
        let f = resolve_from_config_json(&path).unwrap();
        assert_eq!(f.family, "llama");
    }

    #[test]
    fn resolve_config_json_missing_file() {
        assert!(resolve_from_config_json(Path::new("/nonexistent/config.json")).is_none());
    }

    #[test]
    fn resolve_config_json_array_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, r#"[{"model_type": "bert"}]"#).unwrap();
        assert!(resolve_from_config_json(&path).is_none());
    }

    #[test]
    fn resolve_config_json_unknown_model_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, r#"{"model_type": "totally_unknown_xyz"}"#).unwrap();
        assert!(resolve_from_config_json(&path).is_none());
    }

    // ── extract_config_mapping (filesystem tests) ───────────────────────

    #[test]
    fn config_mapping_extracts_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            r#"{
                "model_type": "qwen2",
                "hidden_act": "silu",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "rms_norm_eps": 1e-06,
                "rope_theta": 10000.0,
                "architectures": ["Qwen2ForCausalLM"]
            }"#,
        )
        .unwrap();
        let map = extract_config_mapping(&path);
        assert!(map.contains_key("model_type"));
        assert!(map.contains_key("hidden_act"));
        assert!(map.contains_key("hidden_size"));
        assert!(map.contains_key("num_attention_heads"));
        assert!(map.contains_key("num_key_value_heads"));
        assert!(map.contains_key("rms_norm_eps"));
        assert!(map.contains_key("_architectures"));
        // Enriched rationale
        assert!(map["hidden_act"].rationale.contains("SiLU"));
    }

    #[test]
    fn config_mapping_missing_file() {
        let map = extract_config_mapping(Path::new("/nonexistent/config.json"));
        assert!(map.is_empty());
    }

    #[test]
    fn config_mapping_enriches_gqa() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            r#"{"num_attention_heads": 32, "num_key_value_heads": 8}"#,
        )
        .unwrap();
        let map = extract_config_mapping(&path);
        assert!(map["num_key_value_heads"].rationale.contains("GQA"));
    }

    // ── print_human_output (smoke test) ─────────────────────────────────

    #[test]
    fn print_human_output_no_panic() {
        let family = resolve_family("llama").unwrap();
        let config = BTreeMap::new();
        // Just verify it doesn't panic — output goes to stdout
        print_human_output(&family, &config, false, false);
        print_human_output(&family, &config, true, true);
    }
}
