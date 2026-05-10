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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
        },
        // qwen3_5.yaml
        "qwen3_5" | "qwen3.5" | "qwen35" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: false,
            default_eps: 1e-6,
            is_moe: false,
        },
        // ALB-010: Qwen3 MoE (Qwen3-Coder-30B-A3B: 128 experts, per-head QK norm)
        // and Qwen3.5 MoE (Qwen3.5-35B-A3B: 256 experts, DeltaNet+GQA)
        // Both have per-head Q/K RMSNorm: q_norm.weight/k_norm.weight [head_dim]
        // M-GPU-MOE-1.3 (qwen3-moe-forward-gpu-v1 v1.3.0): is_moe=true
        // gates `CudaExecutor::build_indexed_weights` from demanding the
        // dense FFN tensor names that don't exist in MoE GGUF.
        //
        // Both `qwen3_moe` (canonical, post-normalization) and `qwen3moe`
        // (raw GGUF general.architecture string) are matched. The raw
        // form is what reaches `ArchConstraints::from_architecture` from
        // `ValidatedModelConfig::from_apr` (config.rs:404) without going
        // through `normalize_architecture`.
        "qwen3_moe" | "qwen3_5_moe" | "qwen3moe" | "qwen3_5moe" => ArchConstraints {
            norm_type: NormType::RmsNorm,
            activation: Activation::Silu,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
            weight_layout: WeightLayout::Linear,
            has_bias: false,
            tied_embeddings: false,
            has_qk_norm: true,
            default_eps: 1e-6,
            is_moe: true,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
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
            is_moe: false,
        },
    }
}
