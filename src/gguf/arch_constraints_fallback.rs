// Per-architecture inference constraints.
//
// FALLBACK for CI/crates.io builds when arch-constraints-v1.yaml is not available.
// See: provable-contracts/contracts/arch-constraints-v1.yaml
//
// GH-323: This file is the fallback snapshot. When the YAML contract is present,
// build.rs generates arch_constraints_generated.rs from it instead.

/// Look up architecture constraints from the GGUF `general.architecture` value.
///
/// FALLBACK — matches arch-constraints-v1.yaml.
/// Unknown architectures fall back to LLaMA-like defaults.
#[must_use]
fn from_architecture_generated(arch: &str) -> ArchConstraints {
    match arch {
        // gpt2.yaml
        "gpt2" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Gelu,
            positional_encoding: PositionalEncoding::Absolute,
            mlp_type: MlpType::GeluMlp,
            weight_layout: WeightLayout::Conv1D,
            has_bias: true,
            tied_embeddings: true,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // llama.yaml
        "llama" | "llama3" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // qwen2.yaml
        "qwen2" | "qwen2.5" | "qwen" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: true,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-6,
        },
        // qwen3.yaml
        "qwen3" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: true,
            default_eps: 1e-6,
        },
        // mistral.yaml
        "mistral" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // phi2 (Phi-1.5/Phi-2)
        "phi2" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Gelu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::GeluMlp,
            weight_layout: WeightLayout::Linear,
            has_bias: true,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // phi.yaml (Phi-3/Phi-3.5)
        "phi" | "phi3" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: true,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // gemma.yaml
        "gemma" | "gemma2" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Gelu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::GatedMlp,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: true,
            has_qk_norm: false,
            default_eps: 1e-6,
        },
        // deepseek.yaml — GH-323: fixed eps from 1e-5 to 1e-6 (matches YAML)
        "deepseek" | "deepseek2" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-6,
        },
        // bert.yaml
        "bert" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Gelu,
            positional_encoding: PositionalEncoding::Absolute,
            mlp_type: MlpType::GeluMlp,
            weight_layout: WeightLayout::Linear,
            has_bias: true,
            tied_embeddings: true,
            has_qk_norm: false,
            default_eps: 1e-12,
        },
        // whisper.yaml
        "whisper" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Gelu,
            positional_encoding: PositionalEncoding::Absolute,
            mlp_type: MlpType::GeluMlp,
            weight_layout: WeightLayout::Linear,
            has_bias: true,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // t5.yaml — PMAT-395: T5 encoder-decoder (realizr#177)
        // T5 uses LayerNorm (not RMSNorm), GELU (DenseReluDense),
        // relative position bias (not RoPE), and tied embeddings.
        "t5" | "encoder-decoder" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Gelu,
            positional_encoding: PositionalEncoding::Relative,
            mlp_type: MlpType::GeluMlp,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: true,
            has_qk_norm: false,
            default_eps: 1e-6,
        },
        // mamba.yaml
        "mamba" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::None,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: true,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // qwen3_5.yaml
        "qwen3_5" | "qwen3.5" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-6,
        },
        // ALB-010: Qwen3 MoE (Qwen3-Coder-30B-A3B: 128 experts, per-head QK norm)
        // and Qwen3.5 MoE (Qwen3.5-35B-A3B: 256 experts, DeltaNet+GQA)
        // Both have per-head Q/K RMSNorm: q_norm.weight/k_norm.weight [head_dim]
        "qwen3_moe" | "qwen3_5_moe" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: true,
            default_eps: 1e-6,
        },
        // falcon_h1.yaml
        "falcon_h1" | "falcon-h1" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-6,
        },
        // openelm.yaml
        "openelm" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-6,
        },
        // moonshine.yaml
        "moonshine" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::GatedMlp,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: true,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // rwkv7.yaml
        "rwkv7" | "rwkv" => ArchConstraints {
            norm_type: NormType::LayerNorm,
            activation: Activation::Gelu,
            positional_encoding: PositionalEncoding::None,
            mlp_type: MlpType::GeluMlp,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
        // Default: LLaMA-like (most common pattern in modern LLMs)
        _ => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-5,
        },
    }
}
