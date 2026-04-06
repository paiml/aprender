//! Design by Contract example for realizar.
//!
//! Demonstrates `ValidatedModelConfig` as a Poka-Yoke newtype gate (PMAT-235):
//! invalid model configs are rejected at construction time, before any tensor
//! allocation or inference computation can occur.
//!
//! Run: `cargo run --example design_by_contract`

use realizar::gguf::{
    Activation, ArchConstraints, GGUFConfig, MlpType, NormType, PositionalEncoding,
    ValidatedModelConfig, WeightLayout,
};

fn main() {
    // -----------------------------------------------------------------------
    // 1. Valid config: LLaMA-7B passes all 11 structural invariants
    // -----------------------------------------------------------------------
    let llama_config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8, // 4:1 GQA ratio
        vocab_size: 32_000,
        intermediate_dim: 11_008,
        context_length: 4096,
        rope_theta: 10_000.0,
        eps: 1e-5,
        rope_type: 0, // NORM (adjacent pairs)
        explicit_head_dim: None,
        bos_token_id: Some(128_000),
        eos_token_id: Some(128_001),
    };

    let validated = ValidatedModelConfig::validate(llama_config)
        .expect("LLaMA-7B config should pass validation");
    println!(
        "[PASS] LLaMA-7B: hidden={}, heads={}, kv_heads={}, head_dim={}",
        validated.hidden_dim(),
        validated.num_heads(),
        validated.num_kv_heads(),
        validated.head_dim()
    );

    // -----------------------------------------------------------------------
    // 2. Invalid config: hidden_dim=0 is rejected immediately
    // -----------------------------------------------------------------------
    let bad_config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 0, // <-- violates SI-001
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32_000,
        intermediate_dim: 11_008,
        context_length: 4096,
        rope_theta: 10_000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let err =
        ValidatedModelConfig::validate(bad_config).expect_err("hidden_dim=0 must be rejected");
    println!("[REJECTED] hidden_dim=0: {err}");

    // -----------------------------------------------------------------------
    // 3. Invalid config: num_heads does not divide hidden_dim
    // -----------------------------------------------------------------------
    let misaligned = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 33, // <-- 4096 % 33 != 0
        num_kv_heads: 33,
        vocab_size: 32_000,
        intermediate_dim: 11_008,
        context_length: 4096,
        rope_theta: 10_000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let err = ValidatedModelConfig::validate(misaligned)
        .expect_err("hidden_dim % num_heads != 0 must be rejected");
    println!("[REJECTED] head_dim misaligned: {err}");

    // -----------------------------------------------------------------------
    // 4. ArchConstraints: contract-driven architecture behavior
    // -----------------------------------------------------------------------
    println!("\n--- ArchConstraints comparison ---");

    let qwen2 = ArchConstraints::from_architecture("qwen2");
    println!(
        "Qwen2:  norm={:?}, activation={:?}, pos={:?}, mlp={:?}, layout={:?}, bias={}",
        qwen2.norm_type,
        qwen2.activation,
        qwen2.positional_encoding,
        qwen2.mlp_type,
        qwen2.weight_layout,
        qwen2.has_bias
    );

    let gpt2 = ArchConstraints::from_architecture("gpt2");
    println!(
        "GPT-2:  norm={:?}, activation={:?}, pos={:?}, mlp={:?}, layout={:?}, bias={}",
        gpt2.norm_type,
        gpt2.activation,
        gpt2.positional_encoding,
        gpt2.mlp_type,
        gpt2.weight_layout,
        gpt2.has_bias
    );

    // Verify contract differences
    assert_eq!(qwen2.norm_type, NormType::RmsNorm);
    assert_eq!(gpt2.norm_type, NormType::LayerNorm);
    assert_eq!(qwen2.activation, Activation::Silu);
    assert_eq!(gpt2.activation, Activation::Gelu);
    assert_eq!(qwen2.positional_encoding, PositionalEncoding::Rope);
    assert_eq!(gpt2.positional_encoding, PositionalEncoding::Absolute);
    assert_eq!(qwen2.mlp_type, MlpType::SwiGlu);
    assert_eq!(gpt2.mlp_type, MlpType::GeluMlp);
    assert_eq!(gpt2.weight_layout, WeightLayout::Conv1D);
    assert!(gpt2.needs_transpose());
    assert!(!qwen2.needs_transpose());

    // -----------------------------------------------------------------------
    // 5. Metadata bounds: OOM prevention
    // -----------------------------------------------------------------------
    let oom_config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 100_000, // <-- exceeds 65536 max
        num_layers: 32,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 32_000,
        intermediate_dim: 11_008,
        context_length: 4096,
        rope_theta: 10_000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let err = ValidatedModelConfig::validate(oom_config)
        .expect_err("hidden_dim=100000 exceeds 65536 max");
    println!("\n[REJECTED] OOM prevention: {err}");

    println!("\nAll Design by Contract checks passed.");
}
