//! Kernel Explainability: static analysis of model architecture → kernel dispatch.
//!
//! Derives kernel equivalence class from family contract constraints without
//! loading the model or running inference. Pure metadata analysis.

use serde::Serialize;

pub mod config;
pub mod family;
pub mod kernel_ops;
pub mod output;
pub mod proof;
pub mod resolve;

#[cfg(test)]
mod tests;

// ── Family YAML embedding (compile-time) ──────────────────────────────────

macro_rules! embed_family {
    ($name:expr, $path:expr) => {
        (
            $name,
            include_str!(concat!("../../../contracts/model-families/", $path)),
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
        ($name, include_str!(concat!("../../../contracts/", $path)))
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

// ── Core types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct KernelOp {
    pub op: &'static str,
    pub kernel: &'static str,
    pub contract: &'static str,
}

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

#[derive(Debug, Clone, Serialize)]
pub struct ConfigField {
    pub value: String,
    pub rationale: String,
}

// ── Family aliases ────────────────────────────────────────────────────────

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

// ── Re-exports for public API ─────────────────────────────────────────────

// Re-export the public functions so callers use the same paths as before
pub use config::{
    extract_config_mapping,
    extract_json_string,
};
pub use family::load_families;
pub use output::{build_json_output, print_human_output};

pub use resolve::{family_aliases, resolve_family, resolve_from_config_json};
