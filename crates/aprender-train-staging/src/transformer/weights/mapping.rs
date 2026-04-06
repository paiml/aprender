//! Weight name mapping between architectures

use super::Architecture;

/// Map RoBERTa/CodeBERT weight names to entrenar encoder convention (ENC-006).
fn map_roberta_weight_name(name: &str) -> String {
    // Strip roberta. or bert. prefix
    let stripped =
        name.strip_prefix("roberta.").or_else(|| name.strip_prefix("bert.")).unwrap_or(name);

    // Embeddings
    if stripped == "embeddings.word_embeddings.weight" {
        return "encoder.embed_tokens.weight".to_string();
    }
    if stripped == "embeddings.position_embeddings.weight" {
        return "encoder.position_embeddings.weight".to_string();
    }
    if stripped == "embeddings.token_type_embeddings.weight" {
        return "encoder.token_type_embeddings.weight".to_string();
    }
    if stripped == "embeddings.LayerNorm.weight" {
        return "encoder.embeddings_layernorm.weight".to_string();
    }
    if stripped == "embeddings.LayerNorm.bias" {
        return "encoder.embeddings_layernorm.bias".to_string();
    }

    // Encoder layers: encoder.layer.{i}.XXX
    if let Some(rest) = stripped.strip_prefix("encoder.layer.") {
        if let Some((num, layer_rest)) = rest.split_once('.') {
            let mapped = layer_rest
                .replace("attention.self.query", "self_attn.q_proj")
                .replace("attention.self.key", "self_attn.k_proj")
                .replace("attention.self.value", "self_attn.v_proj")
                .replace("attention.output.dense", "self_attn.o_proj")
                .replace("attention.output.LayerNorm", "input_layernorm")
                .replace("intermediate.dense", "mlp.intermediate.dense")
                .replace("output.dense", "mlp.output.dense")
                .replace("output.LayerNorm", "post_attention_layernorm");

            return format!("encoder.layers.{num}.{mapped}");
        }
    }

    // Pooler (optional)
    if stripped.starts_with("pooler.") {
        return format!("encoder.{stripped}");
    }

    // Pass through anything else
    name.to_string()
}

/// Map GGUF tensor names to standard LLaMA/HF convention.
///
/// GGUF uses short names like `token_embd.weight`, `blk.0.attn_q.weight`.
/// The training pipeline expects HF-style names like `model.embed_tokens.weight`.
fn map_gguf_weight_name(name: &str) -> String {
    // Embeddings
    if name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }
    if name == "output_norm.bias" {
        return "model.norm.bias".to_string();
    }
    if name == "output.weight" {
        return "lm_head.weight".to_string();
    }

    // Layer tensors: blk.{N}.{component}.weight
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some((num, layer_rest)) = rest.split_once('.') {
            let mapped = match layer_rest {
                // Attention
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.bias" => "self_attn.o_proj.bias",
                // Norms
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_norm.bias" => "input_layernorm.bias",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_norm.bias" => "post_attention_layernorm.bias",
                // FFN (Qwen2/LLaMA style)
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                other => other,
            };
            return format!("model.layers.{num}.{mapped}");
        }
    }

    // Pass through anything else
    name.to_string()
}

/// Map weight name from source architecture to standard LLaMA convention
///
/// Standard names expected by `Transformer::from_params`:
/// - `model.embed_tokens.weight`
/// - `model.layers.{i}.input_layernorm.weight`
/// - `model.layers.{i}.self_attn.q_proj.weight`
/// - `model.layers.{i}.self_attn.k_proj.weight`
/// - `model.layers.{i}.self_attn.v_proj.weight`
/// - `model.layers.{i}.self_attn.o_proj.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight`
/// - `model.layers.{i}.mlp.gate_proj.weight`
/// - `model.layers.{i}.mlp.up_proj.weight`
/// - `model.layers.{i}.mlp.down_proj.weight`
/// - `model.norm.weight`
/// - `lm_head.weight`
pub(crate) fn map_weight_name(name: &str, arch: Architecture) -> String {
    match arch {
        Architecture::Qwen2 => {
            // Qwen2 weight names are nearly identical to LLaMA convention
            // Main differences:
            // - Qwen2 has bias tensors (which we preserve)
            // - Some Qwen2 models use "attn" instead of "self_attn" (rare)

            // Handle potential "attn" -> "self_attn" mapping
            if name.contains(".attn.") && !name.contains(".self_attn.") {
                name.replace(".attn.", ".self_attn.")
            } else {
                name.to_string()
            }
        }
        Architecture::Mistral | Architecture::Llama => {
            // LLaMA and Mistral use same naming convention as our standard
            name.to_string()
        }
        Architecture::RoBERTa => map_roberta_weight_name(name),
        Architecture::Gguf => map_gguf_weight_name(name),
        Architecture::Auto => {
            // Should not reach here after detection
            name.to_string()
        }
    }
}
