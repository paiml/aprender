//! Kernel Profile-Driven Playbook Bootstrapping
//!
//! Maps architecture constraints from family contracts to kernel operations
//! and targeted test prompts. This enables architecture-aware playbook generation
//! that stress-tests the specific kernels each model family exercises.
//!
//! # Design Philosophy
//!
//! HuggingFace model families exercise different kernel operations:
//! - LLaMA/Qwen: GQA + RMSNorm + SiLU + RoPE
//! - Falcon: MHA + LayerNorm + GELU
//! - GPT-NeoX: MHA + LayerNorm + GELU + RoPE
//!
//! By connecting family contract constraints to kernel ops to targeted prompts,
//! we bootstrap playbooks that exercise the exact code paths each model uses.

use serde::{Deserialize, Serialize};

/// Kernel operation exercised by a model architecture.
///
/// Each variant maps to a specific SIMD kernel in the trueno/realizar stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelOp {
    /// Fused Q4K matrix-vector multiply (quantized inference)
    FusedQ4kMatvec,
    /// Fused Q5K matrix-vector multiply
    FusedQ5kMatvec,
    /// Fused Q6K matrix-vector multiply
    FusedQ6kMatvec,
    /// RMS normalization (LLaMA, Qwen, Mistral families)
    RmsNorm,
    /// Layer normalization (Falcon, GPT-NeoX families)
    LayerNorm,
    /// SiLU activation (LLaMA, Qwen)
    Silu,
    /// GELU activation (Falcon, GPT-NeoX)
    Gelu,
    /// SwiGLU MLP gate (LLaMA, Qwen, Mistral)
    SwiGlu,
    /// Rotary positional encoding
    Rope,
    /// Grouped-query attention (Qwen, LLaMA 3.x, Mistral)
    GroupedQueryAttention,
    /// Multi-head attention (Falcon, GPT-NeoX, older models)
    MultiHeadAttention,
    /// Multi-query attention (Falcon-40B)
    MultiQueryAttention,
    /// Bias addition in linear layers
    BiasAdd,
    /// Tied input/output embeddings (shared weight matrix)
    TiedEmbeddings,
    /// ALiBi positional encoding (Falcon)
    Alibi,
    /// Absolute positional encoding (GPT-2, BERT)
    AbsolutePosition,
    /// Gated MLP: gate ⊙ activation(up) → down (Gemma uses GELU, Moonshine uses SiLU)
    GatedMlp,
}

impl KernelOp {
    /// Serde-compatible snake_case name (matches `#[serde(rename_all = "snake_case")]`).
    #[must_use]
    pub const fn serde_name(&self) -> &'static str {
        match self {
            Self::FusedQ4kMatvec => "fused_q4k_matvec",
            Self::FusedQ5kMatvec => "fused_q5k_matvec",
            Self::FusedQ6kMatvec => "fused_q6k_matvec",
            Self::RmsNorm => "rms_norm",
            Self::LayerNorm => "layer_norm",
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::SwiGlu => "swi_glu",
            Self::Rope => "rope",
            Self::GroupedQueryAttention => "grouped_query_attention",
            Self::MultiHeadAttention => "multi_head_attention",
            Self::MultiQueryAttention => "multi_query_attention",
            Self::BiasAdd => "bias_add",
            Self::TiedEmbeddings => "tied_embeddings",
            Self::Alibi => "alibi",
            Self::AbsolutePosition => "absolute_position",
            Self::GatedMlp => "gated_mlp",
        }
    }

    /// Human-readable description of this kernel operation.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::FusedQ4kMatvec => "Fused Q4K quantized matrix-vector multiply",
            Self::FusedQ5kMatvec => "Fused Q5K quantized matrix-vector multiply",
            Self::FusedQ6kMatvec => "Fused Q6K quantized matrix-vector multiply",
            Self::RmsNorm => "RMS normalization",
            Self::LayerNorm => "Layer normalization",
            Self::Silu => "SiLU activation function",
            Self::Gelu => "GELU activation function",
            Self::SwiGlu => "SwiGLU gated MLP",
            Self::Rope => "Rotary positional encoding",
            Self::GroupedQueryAttention => "Grouped-query attention (GQA)",
            Self::MultiHeadAttention => "Multi-head attention (MHA)",
            Self::MultiQueryAttention => "Multi-query attention (MQA)",
            Self::BiasAdd => "Bias addition in linear layers",
            Self::TiedEmbeddings => "Tied input/output embeddings",
            Self::Alibi => "ALiBi positional encoding",
            Self::AbsolutePosition => "Absolute positional encoding",
            Self::GatedMlp => "Gated MLP (gate-up projection)",
        }
    }
}

/// Display a kernel operation as its human-readable description
impl std::fmt::Display for KernelOp {
    /// Format the kernel op using its description string
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// A category of prompts targeting specific kernel behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCategory {
    /// Category name (e.g., "gqa_multi_turn", "rope_long_context")
    pub name: String,
    /// Why these prompts target specific kernels
    pub rationale: String,
    /// The actual test prompts
    pub prompts: Vec<String>,
    /// Oracle type for evaluating outputs
    pub oracle_type: String,
    /// Suggested max tokens for this category
    pub max_tokens: u32,
}

/// Complete kernel profile for a model family.
///
/// Describes which kernel operations a model architecture exercises
/// and provides targeted prompts to stress-test those operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelProfile {
    /// Model family name (e.g., "qwen2", "llama", "falcon")
    pub family: String,
    /// Kernel operations exercised by this architecture
    pub kernel_ops: Vec<KernelOp>,
    /// Architecture-targeted prompt categories
    pub prompt_categories: Vec<PromptCategory>,
    /// Suggested max tokens based on architecture
    pub suggested_max_tokens: u32,
    /// Whether this architecture supports long context (>4K tokens)
    pub long_context: bool,
}

impl KernelProfile {
    /// Get all prompts from all categories, flattened.
    #[must_use]
    pub fn all_prompts(&self) -> Vec<String> {
        self.prompt_categories
            .iter()
            .flat_map(|c| c.prompts.clone())
            .collect()
    }

    /// Total number of prompts across all categories.
    #[must_use]
    pub fn prompt_count(&self) -> usize {
        self.prompt_categories.iter().map(|c| c.prompts.len()).sum()
    }
}

/// Mirror of `Constraints` from `apr-qa-runner::family_contract`.
///
/// Defined here to avoid circular dependency (runner depends on gen).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchConstraints {
    /// Attention type: "mha", "gqa", or "mqa"
    pub attention_type: Option<String>,
    /// Activation function: "silu" or "gelu" (defaults to silu if unrecognized)
    pub activation: Option<String>,
    /// Norm type: "rmsnorm" or "layernorm"
    pub norm_type: Option<String>,
    /// Whether linear layers have bias terms
    pub has_bias: Option<bool>,
    /// Whether input/output embeddings are shared
    pub tied_embeddings: Option<bool>,
    /// Positional encoding: "rope", "absolute", "alibi"
    pub positional_encoding: Option<String>,
    /// MLP type: "swiglu" or "gated_mlp" (other values produce no MLP-specific kernel op)
    pub mlp_type: Option<String>,
}

/// Mirror of `SizeVariant` from `apr-qa-runner::family_contract`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchSizeVariant {
    /// Human-readable parameter count (e.g., "0.5B", "7B")
    pub parameters: String,
    /// Hidden dimension / d_model
    pub hidden_dim: u32,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of attention heads (None for non-attention architectures like SSM/RWKV)
    #[serde(default)]
    pub num_heads: Option<u32>,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: Option<u32>,
    /// FFN intermediate dimension
    pub intermediate_dim: Option<u32>,
    /// Vocabulary size
    pub vocab_size: Option<u32>,
    /// Maximum sequence length
    pub max_position_embeddings: Option<u32>,
}

/// Build a kernel profile from architecture constraints.
///
/// Maps family contract constraints to kernel operations and generates
/// targeted prompts that exercise the specific kernels each model uses.
#[must_use]
pub fn profile_from_constraints(
    family: &str,
    constraints: &ArchConstraints,
    max_position_embeddings: Option<u32>,
) -> KernelProfile {
    let mut kernel_ops = Vec::new();
    let mut prompt_categories = Vec::new();

    // Always include quantized matvec ops (all models use these)
    kernel_ops.push(KernelOp::FusedQ4kMatvec);
    kernel_ops.push(KernelOp::FusedQ5kMatvec);
    kernel_ops.push(KernelOp::FusedQ6kMatvec);

    // Attention type -> kernel ops + prompts
    // SSM architectures (mamba, rwkv) have no attention mechanism
    match constraints.attention_type.as_deref() {
        Some("gqa") => {
            kernel_ops.push(KernelOp::GroupedQueryAttention);
            prompt_categories.push(gqa_prompts());
        }
        Some("mqa") => {
            kernel_ops.push(KernelOp::MultiQueryAttention);
            prompt_categories.push(mqa_prompts());
        }
        Some("none") => {
            // SSM / non-attention architectures: no attention kernel needed
        }
        _ => {
            kernel_ops.push(KernelOp::MultiHeadAttention);
            prompt_categories.push(mha_prompts());
        }
    }

    // Normalization type
    match constraints.norm_type.as_deref() {
        Some("layernorm") => kernel_ops.push(KernelOp::LayerNorm),
        // Default to RMSNorm (most common in modern architectures)
        _ => kernel_ops.push(KernelOp::RmsNorm),
    }

    // Activation function
    match constraints.activation.as_deref() {
        Some("gelu") => kernel_ops.push(KernelOp::Gelu),
        // Default to SiLU (most common in modern architectures)
        _ => kernel_ops.push(KernelOp::Silu),
    }

    // MLP type: gated MLPs need fused gate+activation kernels
    match constraints.mlp_type.as_deref() {
        Some("swiglu") => kernel_ops.push(KernelOp::SwiGlu),
        Some("gated_mlp") => kernel_ops.push(KernelOp::GatedMlp),
        // GeluMlp = linear + GELU + linear: no gating, covered by matvec + GELU
        _ => {}
    }

    // Positional encoding
    let long_context = match constraints.positional_encoding.as_deref() {
        Some("rope") => {
            kernel_ops.push(KernelOp::Rope);
            let max_pos = max_position_embeddings.unwrap_or(4096);
            if max_pos > 4096 {
                prompt_categories.push(rope_long_context_prompts());
            }
            max_pos > 4096
        }
        Some("alibi") => {
            kernel_ops.push(KernelOp::Alibi);
            false
        }
        Some("absolute") => {
            kernel_ops.push(KernelOp::AbsolutePosition);
            false
        }
        _ => false,
    };

    // Bias in linear layers
    if constraints.has_bias == Some(true) {
        kernel_ops.push(KernelOp::BiasAdd);
        prompt_categories.push(bias_stress_prompts());
    }

    // Tied embeddings
    if constraints.tied_embeddings == Some(true) {
        kernel_ops.push(KernelOp::TiedEmbeddings);
    }

    // Always include arithmetic verification prompts
    prompt_categories.push(arithmetic_prompts());

    // Always include code completion prompts (exercises token generation paths)
    prompt_categories.push(code_prompts());

    let suggested_max_tokens = if long_context { 128 } else { 64 };

    KernelProfile {
        family: family.to_string(),
        kernel_ops,
        prompt_categories,
        suggested_max_tokens,
        long_context,
    }
}

/// Prompts targeting grouped-query attention (GQA).
///
/// GQA shares KV heads across query head groups. Multi-turn and
/// context-dependent prompts stress the KV cache sharing logic.
fn gqa_prompts() -> PromptCategory {
    PromptCategory {
        name: "gqa_multi_turn".to_string(),
        rationale: "GQA shares KV heads across query groups; multi-turn prompts \
                    stress KV cache sharing and head group boundaries"
            .to_string(),
        prompts: vec![
            "Given x=5 and y=3, what is x*y? Then what is the result plus 10?".to_string(),
            "List the first 5 prime numbers. Now sum them.".to_string(),
            "Define a function add(a,b) that returns a+b. What does add(3,4) return?".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 64,
    }
}

/// Prompts targeting multi-head attention (MHA).
///
/// MHA has independent KV heads per query head. Long dependency
/// prompts stress full attention computation.
fn mha_prompts() -> PromptCategory {
    PromptCategory {
        name: "mha_long_dependency".to_string(),
        rationale: "MHA computes independent KV per head; long-range dependency \
                    prompts test full attention matrix computation"
            .to_string(),
        prompts: vec![
            "The capital of France is Paris. The capital of Germany is Berlin. \
             What is the capital of France?"
                .to_string(),
            "Alice has 3 apples. Bob gives her 2 more. Carol takes 1. \
             How many apples does Alice have?"
                .to_string(),
            "If x=10, y=x+5, z=y*2, what is z?".to_string(),
        ],
        oracle_type: "garbage".to_string(),
        max_tokens: 64,
    }
}

/// Prompts targeting multi-query attention (MQA).
fn mqa_prompts() -> PromptCategory {
    PromptCategory {
        name: "mqa_kv_efficiency".to_string(),
        rationale: "MQA uses a single KV head for all query heads; prompts test \
                    that shared KV computation produces correct results"
            .to_string(),
        prompts: vec![
            "What is 7*8? Answer with just the number.".to_string(),
            "Complete: The sum of 15 and 25 is".to_string(),
            "Translate 'hello' to Spanish in one word.".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 32,
    }
}

/// Prompts for RoPE long-context models (>4K tokens).
///
/// Tests that rotary position encoding correctly handles
/// positions beyond the standard 2K-4K range.
fn rope_long_context_prompts() -> PromptCategory {
    PromptCategory {
        name: "rope_long_context".to_string(),
        rationale: "RoPE position encoding must correctly extrapolate to long \
                    sequences; these prompts test position-dependent accuracy"
            .to_string(),
        prompts: vec![
            "Write a detailed step-by-step solution to: What is 123 * 456? \
             Show all intermediate multiplication steps."
                .to_string(),
            "List the numbers from 1 to 20, then sum them all. \
             What is the final sum?"
                .to_string(),
            "Explain the Fibonacci sequence, list the first 10 numbers, \
             then tell me what the 10th Fibonacci number is."
                .to_string(),
        ],
        oracle_type: "garbage".to_string(),
        max_tokens: 256,
    }
}

/// Prompts that stress bias addition in linear layers.
///
/// Models with bias terms have additional addition operations
/// in every linear projection. Precision-sensitive prompts
/// can reveal bias accumulation errors.
fn bias_stress_prompts() -> PromptCategory {
    PromptCategory {
        name: "bias_precision".to_string(),
        rationale: "Bias terms add to every linear projection output; \
                    arithmetic prompts can reveal floating-point accumulation errors"
            .to_string(),
        prompts: vec![
            "What is 0.1 + 0.2? Give a precise answer.".to_string(),
            "Calculate 999 + 1.".to_string(),
            "What is 1000000 - 999999?".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 32,
    }
}

/// Standard arithmetic verification prompts.
fn arithmetic_prompts() -> PromptCategory {
    PromptCategory {
        name: "arithmetic_verification".to_string(),
        rationale: "Arithmetic prompts provide deterministic verification \
                    of model output correctness across all architectures"
            .to_string(),
        prompts: vec![
            "What is 2+2?".to_string(),
            "Calculate 7*8".to_string(),
            "What is 15-7?".to_string(),
            "What is 100/4?".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 32,
    }
}

/// Code completion prompts that exercise token generation paths.
fn code_prompts() -> PromptCategory {
    PromptCategory {
        name: "code_completion".to_string(),
        rationale: "Code completion exercises the full token generation pipeline \
                    including vocabulary lookup and sampling"
            .to_string(),
        prompts: vec![
            "def fibonacci(n):".to_string(),
            "fn main() {".to_string(),
            "Write a Python function that checks if a number is prime.".to_string(),
        ],
        oracle_type: "code_syntax".to_string(),
        max_tokens: 64,
    }
}

#[cfg(test)]
#[path = "kernel_profile_tests.rs"]
mod kernel_profile_tests;
