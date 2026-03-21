//! Config.json extraction, field enrichment, architecture display, and mismatch detection.

use super::resolve::{strip_arch_suffix, ALIAS_ARCHITECTURES};
use super::{ConfigField, FamilyInfo, KernelClass};
use std::collections::BTreeMap;
use std::path::Path;

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
        (
            "tie_word_embeddings",
            "Weight sharing: embedding <-> lm_head",
        ),
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
pub(crate) fn enrich_rationale(key: &str, value: &str, json: &str) -> Option<String> {
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
                        // Rough estimate assuming 4x MLP (standard FFN: 8d^2/layer)
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
/// Priority: config.json _architectures -> config.json model_type -> alias arch table -> family default.
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
