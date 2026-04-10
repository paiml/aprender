use super::*;

/// Create Qwen2-style architecture constraints with GQA and SwiGLU
fn qwen_constraints() -> ArchConstraints {
    ArchConstraints {
        attention_type: Some("gqa".to_string()),
        activation: Some("silu".to_string()),
        norm_type: Some("rmsnorm".to_string()),
        has_bias: Some(true),
        tied_embeddings: Some(false),
        positional_encoding: Some("rope".to_string()),
        mlp_type: Some("swiglu".to_string()),
    }
}

/// Create Falcon-style architecture constraints with MHA and ALiBi
fn falcon_constraints() -> ArchConstraints {
    ArchConstraints {
        attention_type: Some("mha".to_string()),
        activation: Some("gelu".to_string()),
        norm_type: Some("layernorm".to_string()),
        has_bias: Some(false),
        tied_embeddings: Some(false),
        positional_encoding: Some("alibi".to_string()),
        mlp_type: Some("gelu_mlp".to_string()),
    }
}

/// Verify Qwen2 profile includes GQA, RmsNorm, SiLU, SwiGLU, RoPE, BiasAdd
#[test]
fn test_qwen_profile_kernel_ops() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));

    assert!(profile
        .kernel_ops
        .contains(&KernelOp::GroupedQueryAttention));
    assert!(profile.kernel_ops.contains(&KernelOp::RmsNorm));
    assert!(profile.kernel_ops.contains(&KernelOp::Silu));
    assert!(profile.kernel_ops.contains(&KernelOp::SwiGlu));
    assert!(profile.kernel_ops.contains(&KernelOp::Rope));
    assert!(profile.kernel_ops.contains(&KernelOp::BiasAdd));
    assert!(profile.kernel_ops.contains(&KernelOp::FusedQ4kMatvec));
    // GQA model should not have MHA
    assert!(!profile.kernel_ops.contains(&KernelOp::MultiHeadAttention));
}

/// Verify Falcon profile includes MHA, LayerNorm, GELU, ALiBi
#[test]
fn test_falcon_profile_kernel_ops() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));

    assert!(profile.kernel_ops.contains(&KernelOp::MultiHeadAttention));
    assert!(profile.kernel_ops.contains(&KernelOp::LayerNorm));
    assert!(profile.kernel_ops.contains(&KernelOp::Gelu));
    assert!(profile.kernel_ops.contains(&KernelOp::Alibi));
    // Falcon should not have GQA, RMSNorm, SiLU, RoPE
    assert!(!profile
        .kernel_ops
        .contains(&KernelOp::GroupedQueryAttention));
    assert!(!profile.kernel_ops.contains(&KernelOp::RmsNorm));
    assert!(!profile.kernel_ops.contains(&KernelOp::Silu));
    assert!(!profile.kernel_ops.contains(&KernelOp::Rope));
}

/// Verify Qwen2 with 32K context is flagged as long_context
#[test]
fn test_qwen_long_context() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(profile.long_context);
}

/// Verify Falcon with 2K context is not flagged as long_context
#[test]
fn test_falcon_no_long_context() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert!(!profile.long_context);
}

/// Helper: check if profile has a prompt category by name.
fn has_category(profile: &KernelProfile, name: &str) -> bool {
    profile.prompt_categories.iter().any(|c| c.name == name)
}

/// Verify Qwen2 profile includes GQA prompts and excludes MHA prompts
#[test]
fn test_qwen_has_gqa_prompts() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(has_category(&profile, "gqa_multi_turn"));
    assert!(!has_category(&profile, "mha_long_dependency"));
}

/// Verify Falcon profile includes MHA prompts and excludes GQA prompts
#[test]
fn test_falcon_has_mha_prompts() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert!(has_category(&profile, "mha_long_dependency"));
    assert!(!has_category(&profile, "gqa_multi_turn"));
}

/// Verify RoPE long-context prompts are added for 32K context
#[test]
fn test_rope_long_context_prompts_added() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(has_category(&profile, "rope_long_context"));
}

/// Verify RoPE short-context profile omits long-context prompts
#[test]
fn test_rope_short_context_no_long_prompts() {
    let mut constraints = qwen_constraints();
    constraints.positional_encoding = Some("rope".to_string());
    let profile = profile_from_constraints("qwen2-small", &constraints, Some(2048));
    assert!(!has_category(&profile, "rope_long_context"));
}

/// Verify bias precision prompts are included when has_bias is true
#[test]
fn test_bias_prompts_when_has_bias() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(4096));
    assert!(has_category(&profile, "bias_precision"));
}

/// Verify bias precision prompts are excluded when has_bias is false
#[test]
fn test_no_bias_prompts_when_no_bias() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert!(!has_category(&profile, "bias_precision"));
}

/// Verify all profiles include arithmetic and code completion categories
#[test]
fn test_always_has_arithmetic_and_code() {
    let profile = profile_from_constraints("test", &ArchConstraints::default(), None);
    assert!(has_category(&profile, "arithmetic_verification"));
    assert!(has_category(&profile, "code_completion"));
}

/// Verify all_prompts returns non-empty list matching prompt_count
#[test]
fn test_all_prompts_flattened() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    let all = profile.all_prompts();
    assert!(!all.is_empty());
    assert_eq!(all.len(), profile.prompt_count());
}

/// Verify Qwen2 profile has at least 15 prompts across all categories
#[test]
fn test_prompt_count() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(profile.prompt_count() > 0);
    // Should have prompts from: gqa, rope_long_context, bias, arithmetic, code
    assert!(profile.prompt_count() >= 15);
}

/// Verify default constraints produce MHA, RMSNorm, SiLU profile without long_context
#[test]
fn test_default_constraints_profile() {
    let profile = profile_from_constraints("unknown", &ArchConstraints::default(), None);
    // Should default to MHA, RMSNorm, SiLU
    assert!(profile.kernel_ops.contains(&KernelOp::MultiHeadAttention));
    assert!(profile.kernel_ops.contains(&KernelOp::RmsNorm));
    assert!(profile.kernel_ops.contains(&KernelOp::Silu));
    assert!(!profile.long_context);
}

/// Verify TiedEmbeddings kernel op is included when tied_embeddings is true
#[test]
fn test_tied_embeddings() {
    let constraints = ArchConstraints {
        tied_embeddings: Some(true),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("test", &constraints, None);
    assert!(profile.kernel_ops.contains(&KernelOp::TiedEmbeddings));
}

/// Verify TiedEmbeddings kernel op is excluded when tied_embeddings is false
#[test]
fn test_no_tied_embeddings() {
    let constraints = ArchConstraints {
        tied_embeddings: Some(false),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("test", &constraints, None);
    assert!(!profile.kernel_ops.contains(&KernelOp::TiedEmbeddings));
}

/// Verify MQA attention type maps to MultiQueryAttention and KV efficiency prompts
#[test]
fn test_mqa_attention() {
    let constraints = ArchConstraints {
        attention_type: Some("mqa".to_string()),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("falcon40b", &constraints, None);
    assert!(profile.kernel_ops.contains(&KernelOp::MultiQueryAttention));
    assert!(has_category(&profile, "mqa_kv_efficiency"));
}

/// Verify KernelOp Display impl produces human-readable names
#[test]
fn test_kernel_op_display() {
    assert_eq!(
        format!("{}", KernelOp::GroupedQueryAttention),
        "Grouped-query attention (GQA)"
    );
    assert_eq!(format!("{}", KernelOp::RmsNorm), "RMS normalization");
}

/// Verify every KernelOp variant has a non-empty description
#[test]
fn test_kernel_op_description_all_variants() {
    let variants = [
        (
            KernelOp::FusedQ4kMatvec,
            "Fused Q4K quantized matrix-vector multiply",
        ),
        (
            KernelOp::FusedQ5kMatvec,
            "Fused Q5K quantized matrix-vector multiply",
        ),
        (
            KernelOp::FusedQ6kMatvec,
            "Fused Q6K quantized matrix-vector multiply",
        ),
        (KernelOp::RmsNorm, "RMS normalization"),
        (KernelOp::LayerNorm, "Layer normalization"),
        (KernelOp::Silu, "SiLU activation function"),
        (KernelOp::Gelu, "GELU activation function"),
        (KernelOp::SwiGlu, "SwiGLU gated MLP"),
        (KernelOp::Rope, "Rotary positional encoding"),
        (
            KernelOp::GroupedQueryAttention,
            "Grouped-query attention (GQA)",
        ),
        (KernelOp::MultiHeadAttention, "Multi-head attention (MHA)"),
        (KernelOp::MultiQueryAttention, "Multi-query attention (MQA)"),
        (KernelOp::BiasAdd, "Bias addition in linear layers"),
        (KernelOp::TiedEmbeddings, "Tied input/output embeddings"),
        (KernelOp::Alibi, "ALiBi positional encoding"),
        (KernelOp::AbsolutePosition, "Absolute positional encoding"),
        (KernelOp::GatedMlp, "Gated MLP (gate-up projection)"),
    ];
    for (op, expected) in variants {
        assert_eq!(op.description(), expected, "Mismatch for {op:?}");
    }
    // Assert we test ALL 17 variants
    assert_eq!(variants.len(), 17, "Test must cover all KernelOp variants");
}

/// Verify profile stores the model family name from constraints
#[test]
fn test_profile_family_name() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert_eq!(profile.family, "qwen2");
}

/// Verify long-context profile suggests 128 max tokens
#[test]
fn test_suggested_max_tokens_long_context() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert_eq!(profile.suggested_max_tokens, 128);
}

/// Verify short-context profile suggests 64 max tokens
#[test]
fn test_suggested_max_tokens_short_context() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert_eq!(profile.suggested_max_tokens, 64);
}

/// Verify KernelOp serializes to snake_case JSON string
#[test]
fn test_kernel_op_serialize() {
    let op = KernelOp::GroupedQueryAttention;
    let json = serde_json::to_string(&op).expect("serialize");
    assert_eq!(json, "\"grouped_query_attention\"");
}

/// Verify KernelOp deserializes from snake_case JSON string
#[test]
fn test_kernel_op_deserialize() {
    let op: KernelOp = serde_json::from_str("\"fused_q4k_matvec\"").expect("deserialize");
    assert_eq!(op, KernelOp::FusedQ4kMatvec);
}

/// Verify ArchConstraints default has all fields set to None
#[test]
fn test_arch_constraints_default() {
    let c = ArchConstraints::default();
    assert!(c.attention_type.is_none());
    assert!(c.activation.is_none());
    assert!(c.norm_type.is_none());
    assert!(c.has_bias.is_none());
    assert!(c.tied_embeddings.is_none());
    assert!(c.positional_encoding.is_none());
    assert!(c.mlp_type.is_none());
}

/// Verify ArchSizeVariant default has zeroed dimensions and empty parameters
#[test]
fn test_arch_size_variant_default() {
    let v = ArchSizeVariant::default();
    assert_eq!(v.hidden_dim, 0);
    assert_eq!(v.num_layers, 0);
    assert_eq!(v.num_heads, None);
    assert!(v.parameters.is_empty());
}

/// Verify all prompt categories use valid oracle types
#[test]
fn test_prompt_category_oracle_types() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    for cat in &profile.prompt_categories {
        assert!(
            ["arithmetic", "garbage", "code_syntax"].contains(&cat.oracle_type.as_str()),
            "Unexpected oracle type: {}",
            cat.oracle_type
        );
    }
}

/// Verify all prompt categories have a positive max_tokens value
#[test]
fn test_prompt_category_max_tokens_positive() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    for cat in &profile.prompt_categories {
        assert!(cat.max_tokens > 0, "max_tokens must be positive");
    }
}

/// Verify every prompt category contains at least one prompt
#[test]
fn test_prompt_category_has_prompts() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    for cat in &profile.prompt_categories {
        assert!(
            !cat.prompts.is_empty(),
            "Category '{}' must have prompts",
            cat.name
        );
    }
}

/// Verify absolute positional encoding maps to AbsolutePosition kernel op
#[test]
fn test_absolute_position_encoding() {
    let constraints = ArchConstraints {
        positional_encoding: Some("absolute".to_string()),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("gpt2", &constraints, None);
    assert!(profile.kernel_ops.contains(&KernelOp::AbsolutePosition));
    assert!(!profile.long_context);
}

/// Verify KernelProfile survives JSON serialize/deserialize roundtrip
#[test]
fn test_kernel_profile_serialize_roundtrip() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    let json = serde_json::to_string(&profile).expect("serialize");
    let deserialized: KernelProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.family, profile.family);
    assert_eq!(deserialized.kernel_ops.len(), profile.kernel_ops.len());
    assert_eq!(
        deserialized.prompt_categories.len(),
        profile.prompt_categories.len()
    );
}

/// Create SSM-style architecture constraints (mamba/rwkv) with no attention
fn ssm_constraints() -> ArchConstraints {
    ArchConstraints {
        attention_type: Some("none".to_string()),
        activation: Some("silu".to_string()),
        norm_type: Some("rmsnorm".to_string()),
        has_bias: Some(false),
        tied_embeddings: Some(true),
        positional_encoding: None,
        mlp_type: Some("swiglu".to_string()),
    }
}

/// SSM architectures must NOT include any attention kernel ops
#[test]
fn test_ssm_no_attention_kernel() {
    let profile = profile_from_constraints("mamba", &ssm_constraints(), None);
    assert!(
        !profile.kernel_ops.contains(&KernelOp::MultiHeadAttention),
        "SSM must not have MHA"
    );
    assert!(
        !profile
            .kernel_ops
            .contains(&KernelOp::GroupedQueryAttention),
        "SSM must not have GQA"
    );
    assert!(
        !profile.kernel_ops.contains(&KernelOp::MultiQueryAttention),
        "SSM must not have MQA"
    );
}

/// SSM profile should still include matvec, norm, activation, mlp ops
#[test]
fn test_ssm_has_non_attention_ops() {
    let profile = profile_from_constraints("mamba", &ssm_constraints(), None);
    assert!(profile.kernel_ops.contains(&KernelOp::FusedQ4kMatvec));
    assert!(profile.kernel_ops.contains(&KernelOp::RmsNorm));
    assert!(profile.kernel_ops.contains(&KernelOp::Silu));
    assert!(profile.kernel_ops.contains(&KernelOp::SwiGlu));
    assert!(profile.kernel_ops.contains(&KernelOp::TiedEmbeddings));
}
