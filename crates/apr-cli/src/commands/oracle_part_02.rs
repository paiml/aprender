
// ============================================================================
// Phase 3: Cross-Validation
// ============================================================================

/// Cross-validation of contract YAML vs HuggingFace config.json
#[derive(Debug, Clone, Serialize)]
pub struct CrossValidation {
    pub matches: Vec<CrossValidationEntry>,
    pub mismatches: Vec<CrossValidationEntry>,
    pub contract_only: Vec<String>,
    pub hf_only: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossValidationEntry {
    pub field: String,
    pub contract_value: String,
    pub hf_value: String,
    pub status: String,
}

/// Cross-validate contract size config against HF config.json
pub fn cross_validate(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
    hf_config: &serde_json::Value,
) -> CrossValidation {
    let mut matches = Vec::new();
    let mut mismatches = Vec::new();
    let mut contract_only = Vec::new();
    let mut hf_only = Vec::new();

    type FieldMapping = (&'static str, &'static str, fn(&ModelSizeConfig) -> String);
    let field_mappings: &[FieldMapping] = &[
        ("hidden_size", "hidden_dim", |s| s.hidden_dim.to_string()),
        ("num_hidden_layers", "num_layers", |s| {
            s.num_layers.to_string()
        }),
        ("num_attention_heads", "num_heads", |s| {
            s.num_heads.to_string()
        }),
        ("num_key_value_heads", "num_kv_heads", |s| {
            s.num_kv_heads.to_string()
        }),
        ("intermediate_size", "intermediate_dim", |s| {
            s.intermediate_dim.to_string()
        }),
        ("vocab_size", "vocab_size", |s| s.vocab_size.to_string()),
        ("max_position_embeddings", "max_position_embeddings", |s| {
            s.max_position_embeddings.to_string()
        }),
    ];

    for (hf_key, contract_key, getter) in field_mappings {
        let contract_val = getter(size);
        if let Some(hf_val) = hf_config.get(hf_key) {
            let hf_str = match hf_val {
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let entry = CrossValidationEntry {
                field: (*contract_key).to_string(),
                contract_value: contract_val.clone(),
                hf_value: hf_str.clone(),
                status: if contract_val == hf_str {
                    "match".to_string()
                } else {
                    "mismatch".to_string()
                },
            };
            if contract_val == hf_str {
                matches.push(entry);
            } else {
                mismatches.push(entry);
            }
        } else {
            contract_only.push(format!("{contract_key}={contract_val}"));
        }
    }

    cross_validate_rope_theta(size, hf_config, &mut matches, &mut mismatches);
    cross_validate_norm_eps(size, hf_config, &mut matches, &mut mismatches);
    cross_validate_model_type(constraints, hf_config, &mut matches);
    collect_untracked_hf_fields(hf_config, &mut hf_only);

    CrossValidation {
        matches,
        mismatches,
        contract_only,
        hf_only,
    }
}

fn cross_validate_rope_theta(
    size: &ModelSizeConfig,
    hf_config: &serde_json::Value,
    matches: &mut Vec<CrossValidationEntry>,
    mismatches: &mut Vec<CrossValidationEntry>,
) {
    let Some(hf_theta) = hf_config
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
    else {
        return;
    };
    let contract_theta = size.rope_theta;
    let status = if (contract_theta - hf_theta).abs() < 1.0 {
        "match"
    } else if (contract_theta - hf_theta).abs() / hf_theta.max(1.0) < 0.01 {
        "approximate"
    } else {
        "mismatch"
    };
    let entry = CrossValidationEntry {
        field: "rope_theta".to_string(),
        contract_value: format!("{contract_theta}"),
        hf_value: format!("{hf_theta}"),
        status: status.to_string(),
    };
    if status == "mismatch" {
        mismatches.push(entry);
    } else {
        matches.push(entry);
    }
}

fn cross_validate_norm_eps(
    size: &ModelSizeConfig,
    hf_config: &serde_json::Value,
    matches: &mut Vec<CrossValidationEntry>,
    mismatches: &mut Vec<CrossValidationEntry>,
) {
    let hf_eps = hf_config
        .get("rms_norm_eps")
        .or_else(|| hf_config.get("layer_norm_eps"))
        .or_else(|| hf_config.get("layer_norm_epsilon"))
        .and_then(serde_json::Value::as_f64);
    let Some(hf_eps_val) = hf_eps else { return };
    let contract_eps = size.norm_eps;
    let status = if (contract_eps - hf_eps_val).abs() < 1e-12 {
        "match"
    } else {
        "mismatch"
    };
    let entry = CrossValidationEntry {
        field: "norm_eps".to_string(),
        contract_value: format!("{contract_eps:.1e}"),
        hf_value: format!("{hf_eps_val:.1e}"),
        status: status.to_string(),
    };
    if status == "mismatch" {
        mismatches.push(entry);
    } else {
        matches.push(entry);
    }
}

fn cross_validate_model_type(
    constraints: &ModelConstraints,
    hf_config: &serde_json::Value,
    matches: &mut Vec<CrossValidationEntry>,
) {
    if let Some(hf_model_type) = hf_config["model_type"].as_str() {
        matches.push(CrossValidationEntry {
            field: "model_type".to_string(),
            contract_value: format!("{:?}", constraints.attention_type),
            hf_value: hf_model_type.to_string(),
            status: "info".to_string(),
        });
    }
}

fn collect_untracked_hf_fields(hf_config: &serde_json::Value, hf_only: &mut Vec<String>) {
    const TRACKED: &[&str] = &[
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "layer_norm_eps",
        "layer_norm_epsilon",
        "model_type",
    ];
    const INTERESTING: &[&str] = &[
        "rope_scaling",
        "sliding_window",
        "attention_dropout",
        "use_cache",
        "tie_word_embeddings",
    ];
    let Some(obj) = hf_config.as_object() else {
        return;
    };
    for key in obj.keys() {
        if !TRACKED.contains(&key.as_str()) && INTERESTING.contains(&key.as_str()) {
            hf_only.push(format!("{key}={}", obj[key]));
        }
    }
}

// ============================================================================
// Phase 4: Architecture Explanation
// ============================================================================

/// Template-based architecture explanations referencing literature
#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureExplanation {
    pub attention_explanation: String,
    pub ffn_explanation: String,
    pub norm_explanation: String,
    pub positional_explanation: String,
    pub scaling_analysis: String,
}

pub fn build_architecture_explanation(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
    stats: &StatisticalAnalysis,
) -> ArchitectureExplanation {
    let attention_explanation = match constraints.attention_type {
        AttentionType::Gqa => {
            let mha_kv_mb = stats.kv_cache_4k_mb / stats.gqa_ratio.max(0.01);
            format!(
                "GQA with ratio {:.2} reduces KV cache by {:.0}% vs MHA (Ainslie et al., 2023). \
                 For this model at 4K context: KV cache = {:.1} MB vs {:.1} MB with MHA.",
                stats.gqa_ratio,
                stats.kv_cache_reduction * 100.0,
                stats.kv_cache_4k_mb,
                mha_kv_mb
            )
        }
        AttentionType::Mha => format!(
            "Multi-Head Attention with {} heads, each of dimension {}. \
             Full KV cache: {:.1} MB at 4K context (no sharing).",
            size.num_heads, size.head_dim, stats.kv_cache_4k_mb
        ),
        AttentionType::Mqa => format!(
            "Multi-Query Attention: single KV head shared across {} query heads. \
             Maximum KV cache reduction. Cache: {:.1} MB at 4K context.",
            size.num_heads, stats.kv_cache_4k_mb
        ),
    };

    let ffn_explanation = match constraints.mlp_type {
        MlpType::SwiGlu => format!(
            "SwiGLU (Shazeer, 2020) uses gated activation: \
             FFN(x) = (W_up * x * SiLU(W_gate * x)) * W_down. \
             Expansion ratio {:.2}x compensates for the gating bottleneck (standard FFN uses 4x).",
            stats.ffn_expansion_ratio
        ),
        MlpType::GatedMlp => format!(
            "GeGLU gated MLP: FFN(x) = (W_up * x * GELU(W_gate * x)) * W_down. \
             Expansion ratio {:.2}x with GELU gating.",
            stats.ffn_expansion_ratio
        ),
        MlpType::GeluMlp => format!(
            "Standard GELU MLP: FFN(x) = W2 * GELU(W1 * x + b1) + b2. \
             Expansion ratio {:.2}x (standard 4x for non-gated architectures).",
            stats.ffn_expansion_ratio
        ),
    };

    let norm_explanation = match constraints.norm_type {
        NormType::RmsNorm => format!(
            "RMSNorm (Zhang & Sennrich, 2019) omits mean-centering, reducing compute by ~50% \
             per normalization with minimal quality loss. eps={:.1e}.",
            size.norm_eps
        ),
        NormType::LayerNorm => format!(
            "LayerNorm (Ba et al., 2016) centers and scales activations. \
             eps={:.1e}. More compute than RMSNorm but standard for encoder models.",
            size.norm_eps
        ),
    };

    let positional_explanation = match constraints.positional_encoding {
        PositionalEncoding::Rope => {
            let extrapolated = size.max_position_embeddings * 4;
            format!(
                "RoPE theta={:.0} supports {} positions natively. \
                 Max wavelength: {:.0} tokens. \
                 YaRN/NTK scaling can extend to ~{}.",
                size.rope_theta,
                size.max_position_embeddings,
                stats.rope_max_wavelength,
                extrapolated
            )
        }
        PositionalEncoding::Absolute => format!(
            "Absolute position embeddings: learned vectors for positions 0..{}. \
             No extrapolation beyond training length.",
            size.max_position_embeddings
        ),
        PositionalEncoding::Alibi => format!(
            "ALiBi (Press et al., 2022): linear attention bias, no learned embeddings. \
             Max context: {} (train-time length).",
            size.max_position_embeddings
        ),
        PositionalEncoding::Relative => format!(
            "Relative positional encoding: encodes pairwise distances. \
             Max context: {}.",
            size.max_position_embeddings
        ),
        PositionalEncoding::None => "No positional encoding: temporal information is carried \
             through recurrent state (RWKV, Mamba). Theoretically infinite context."
            .to_string(),
    };

    let params_b = stats.model_params as f64 / 1e9;
    let layers = size.num_layers;
    let hidden = size.hidden_dim;
    let chinchilla_tokens = (params_b * 20.0 * 1e9) as u64;
    let training_flops = 6.0 * stats.model_params as f64 * chinchilla_tokens as f64;
    let scaling_analysis = format!(
        "At {params_b:.1}B parameters with {layers}L x {hidden}H: \
         estimated training compute ~{training_flops:.1e} FLOPs \
         (Chinchilla-optimal tokens: {:.0}B).",
        chinchilla_tokens as f64 / 1e9
    );

    ArchitectureExplanation {
        attention_explanation,
        ffn_explanation,
        norm_explanation,
        positional_explanation,
        scaling_analysis,
    }
}

// ============================================================================
// Phase 5: Kernel Compatibility
// ============================================================================

/// Kernel compatibility report for realizar runtime
#[derive(Debug, Clone, Serialize)]
pub struct KernelCompatibility {
    pub supported_quantizations: Vec<QuantizationSupport>,
    pub attention_kernel: String,
    pub ffn_kernel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tps_cpu: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tps_gpu: Option<f64>,
    pub memory_required_mb: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizationSupport {
    pub format: String,
    pub supported: bool,
    pub kernel: String,
    pub bits_per_weight: f64,
    pub estimated_size_mb: f64,
}
