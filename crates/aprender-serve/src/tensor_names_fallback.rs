// Architecture-specific tensor name resolution.
//
// AUTO-GENERATED from tensor-names-v1.yaml by build.rs — DO NOT EDIT.
// See: provable-contracts/contracts/tensor-names-v1.yaml
//
// GH-311: Compile-time provable tensor name resolution.

/// Global tensor roles (one per model, not per layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalTensorRole {
    /// Token embedding matrix [vocab_size, hidden_dim]
    Embedding,
    /// Language model head projection [vocab_size, hidden_dim]
    LmHead,
    /// Final layer normalization bias (optional, LayerNorm models)
    OutputNormBias,
    /// Final layer normalization weight
    OutputNormWeight,
    /// Absolute position embedding (GPT-2 style, optional)
    PositionEmbedding,
}

/// Per-layer tensor roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerTensorRole {
    /// Per-head K RMSNorm gamma (Qwen3)
    AttnKNormWeight,
    /// Pre-attention layer normalization bias (LayerNorm models)
    AttnNormBias,
    /// Pre-attention layer normalization weight
    AttnNormWeight,
    /// Per-head Q RMSNorm gamma (Qwen3)
    AttnQNormWeight,
    /// FFN down projection weight (or fc2 for non-gated architectures)
    FfnDownWeight,
    /// FFN gate projection weight (SwiGLU architectures)
    FfnGateWeight,
    /// Pre-FFN layer normalization bias (LayerNorm models)
    FfnNormBias,
    /// Pre-FFN layer normalization weight
    FfnNormWeight,
    /// FFN up projection weight (or fc1 for non-gated architectures)
    FfnUpWeight,
    /// Key projection bias (Qwen2, Phi)
    KProjBias,
    /// Key projection weight
    KProjWeight,
    /// Output projection weight
    OProjWeight,
    /// Query projection bias (Qwen2, Phi)
    QProjBias,
    /// Query projection weight
    QProjWeight,
    /// Value projection bias (Qwen2, Phi)
    VProjBias,
    /// Value projection weight
    VProjWeight,
}

/// Fused tensor roles (e.g., combined QKV).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedTensorRole {
    /// Fused Q/K/V projection weight (GPT-2 c_attn, GPT-NeoX query_key_value)
    FusedQkv,
    /// Fused Q/K/V projection bias
    FusedQkvBias,
}

/// Normalize a raw architecture string to a canonical key.
///
/// Unknown architectures default to "llama" (safest default).
#[must_use]
pub fn normalize_architecture(raw: &str) -> &'static str {
    match raw {
        "BertForMaskedLM" => "bert",
        "BertModel" => "bert",
        "DeepseekV2ForCausalLM" => "deepseek",
        "FalconForCausalLM" => "falcon",
        "GPT2LMHeadModel" => "gpt2",
        "GPTNeoXForCausalLM" => "gpt_neox",
        "Gemma2ForCausalLM" => "gemma",
        "GemmaForCausalLM" => "gemma",
        "LLaMAForCausalLM" => "llama",
        "LlamaForCausalLM" => "llama",
        "MistralForCausalLM" => "mistral",
        "OpenELMForCausalLM" => "openelm",
        "Phi3ForCausalLM" => "phi",
        "Phi3SmallForCausalLM" => "phi",
        "PhiForCausalLM" => "phi2",
        "Qwen2ForCausalLM" => "qwen2",
        "Qwen3ForCausalLM" => "qwen3",
        "Qwen3MoeForCausalLM" => "qwen3_moe",
        "Qwen3_5MoeForCausalLM" => "qwen3_moe",
        "Qwen3_5MoeForConditionalGeneration" => "qwen3_moe",
        "StableLmForCausalLM" => "stablelm",
        "bert" => "bert",
        "deepseek" => "deepseek",
        "falcon" => "falcon",
        "gemma" => "gemma",
        "gemma2" => "gemma",
        "gpt2" => "gpt2",
        "gpt_neox" => "gpt_neox",
        "llama" => "llama",
        "llama3" => "llama",
        "mistral" => "mistral",
        "openelm" => "openelm",
        "phi" => "phi",
        "phi2" => "phi2",
        "phi3" => "phi",
        "qwen" => "qwen2",
        "qwen2" => "qwen2",
        "qwen2.5" => "qwen2",
        "qwen3" => "qwen3",
        "qwen3_moe" => "qwen3_moe",
        "qwen3_5_moe" => "qwen3_moe",
        "stablelm" => "stablelm",
        _ => "llama",
    }
}

/// Returns the tensor name variants for a global role and architecture.
#[must_use]
pub fn global_names(arch: &str, role: GlobalTensorRole) -> &'static [&'static str] {
    match (arch, role) {
        ("bert", GlobalTensorRole::Embedding) => &["bert.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"],
        ("deepseek", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("falcon", GlobalTensorRole::Embedding) => &["transformer.word_embeddings.weight", "model.embed_tokens.weight"],
        ("gemma", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("gpt2", GlobalTensorRole::Embedding) => &["wte.weight", "transformer.wte.weight"],
        ("gpt_neox", GlobalTensorRole::Embedding) => &["gpt_neox.embed_in.weight"],
        ("llama", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("mistral", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("openelm", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight", "transformer.token_embeddings.weight"],
        ("phi", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("phi2", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("qwen2", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("qwen3", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("qwen3_moe", GlobalTensorRole::Embedding) => &["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"],
        ("stablelm", GlobalTensorRole::Embedding) => &["model.embed_tokens.weight"],
        ("bert", GlobalTensorRole::LmHead) => &["cls.predictions.decoder.weight"],
        ("deepseek", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("falcon", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("gemma", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("gpt2", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("gpt_neox", GlobalTensorRole::LmHead) => &["embed_out.weight"],
        ("llama", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("mistral", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("openelm", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("phi", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("phi2", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("qwen2", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("qwen3", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("qwen3_moe", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("stablelm", GlobalTensorRole::LmHead) => &["lm_head.weight"],
        ("bert", GlobalTensorRole::OutputNormBias) => &["bert.encoder.layer_norm.bias", "encoder.layer_norm.bias"],
        ("falcon", GlobalTensorRole::OutputNormBias) => &["transformer.ln_f.bias"],
        ("gpt2", GlobalTensorRole::OutputNormBias) => &["ln_f.bias", "transformer.ln_f.bias"],
        ("gpt_neox", GlobalTensorRole::OutputNormBias) => &["gpt_neox.final_layer_norm.bias"],
        ("phi2", GlobalTensorRole::OutputNormBias) => &["model.final_layernorm.bias"],
        ("bert", GlobalTensorRole::OutputNormWeight) => &["bert.encoder.layer_norm.weight", "encoder.layer_norm.weight"],
        ("deepseek", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("falcon", GlobalTensorRole::OutputNormWeight) => &["transformer.ln_f.weight", "model.norm.weight"],
        ("gemma", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("gpt2", GlobalTensorRole::OutputNormWeight) => &["ln_f.weight", "transformer.ln_f.weight"],
        ("gpt_neox", GlobalTensorRole::OutputNormWeight) => &["gpt_neox.final_layer_norm.weight"],
        ("llama", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("mistral", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("openelm", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight", "transformer.norm.weight"],
        ("phi", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("phi2", GlobalTensorRole::OutputNormWeight) => &["model.final_layernorm.weight", "model.norm.weight"],
        ("qwen2", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("qwen3", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("qwen3_moe", GlobalTensorRole::OutputNormWeight) => &["model.language_model.norm.weight", "model.norm.weight"],
        ("stablelm", GlobalTensorRole::OutputNormWeight) => &["model.norm.weight"],
        ("bert", GlobalTensorRole::PositionEmbedding) => &["bert.embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"],
        ("gpt2", GlobalTensorRole::PositionEmbedding) => &["wpe.weight", "transformer.wpe.weight"],
        _ => &[],
    }
}

/// Returns the GGUF fallback tensor names for a global role.
#[must_use]
pub fn global_fallback_names(role: GlobalTensorRole) -> &'static [&'static str] {
    match role {
        GlobalTensorRole::Embedding => &["token_embd.weight"],
        GlobalTensorRole::LmHead => &["output.weight"],
        GlobalTensorRole::OutputNormBias => &["output_norm.bias"],
        GlobalTensorRole::OutputNormWeight => &["output_norm.weight"],
        GlobalTensorRole::PositionEmbedding => &[],
    }
}

/// Returns the tensor name templates for a layer role and architecture.
/// Templates contain `{n}` as placeholder for layer index.
#[must_use]
pub fn layer_templates(arch: &str, role: LayerTensorRole) -> &'static [&'static str] {
    match (arch, role) {
        ("qwen3", LayerTensorRole::AttnKNormWeight) => &["model.layers.{n}.self_attn.k_norm.weight"],
        ("qwen3_moe", LayerTensorRole::AttnKNormWeight) => &["model.layers.{n}.self_attn.k_norm.weight", "model.language_model.layers.{n}.self_attn.k_norm.weight"],
        ("bert", LayerTensorRole::AttnNormBias) => &["bert.encoder.layer.{n}.attention.output.LayerNorm.bias"],
        ("falcon", LayerTensorRole::AttnNormBias) => &["transformer.h.{n}.input_layernorm.bias"],
        ("gpt2", LayerTensorRole::AttnNormBias) => &["h.{n}.ln_1.bias", "transformer.h.{n}.ln_1.bias"],
        ("gpt_neox", LayerTensorRole::AttnNormBias) => &["gpt_neox.layers.{n}.input_layernorm.bias"],
        ("phi2", LayerTensorRole::AttnNormBias) => &["model.layers.{n}.input_layernorm.bias", "model.layers.{n}.ln.bias"],
        ("bert", LayerTensorRole::AttnNormWeight) => &["bert.encoder.layer.{n}.attention.output.LayerNorm.weight", "encoder.layer.{n}.attention.output.LayerNorm.weight"],
        ("deepseek", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("falcon", LayerTensorRole::AttnNormWeight) => &["transformer.h.{n}.input_layernorm.weight", "model.layers.{n}.input_layernorm.weight"],
        ("gemma", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("gpt2", LayerTensorRole::AttnNormWeight) => &["h.{n}.ln_1.weight", "transformer.h.{n}.ln_1.weight"],
        ("gpt_neox", LayerTensorRole::AttnNormWeight) => &["gpt_neox.layers.{n}.input_layernorm.weight"],
        ("llama", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("mistral", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("openelm", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.attn_norm.weight", "transformer.layers.{n}.attn_norm.weight"],
        ("phi", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("phi2", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight", "model.layers.{n}.ln.weight"],
        ("qwen2", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("qwen3", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("qwen3_moe", LayerTensorRole::AttnNormWeight) => &["model.language_model.layers.{n}.input_layernorm.weight", "model.layers.{n}.input_layernorm.weight"],
        ("stablelm", LayerTensorRole::AttnNormWeight) => &["model.layers.{n}.input_layernorm.weight"],
        ("qwen3", LayerTensorRole::AttnQNormWeight) => &["model.layers.{n}.self_attn.q_norm.weight"],
        ("qwen3_moe", LayerTensorRole::AttnQNormWeight) => &["model.layers.{n}.self_attn.q_norm.weight", "model.language_model.layers.{n}.self_attn.q_norm.weight"],
        ("bert", LayerTensorRole::FfnDownWeight) => &["bert.encoder.layer.{n}.output.dense.weight", "encoder.layer.{n}.output.dense.weight"],
        ("deepseek", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("falcon", LayerTensorRole::FfnDownWeight) => &["transformer.h.{n}.mlp.dense_4h_to_h.weight", "model.layers.{n}.mlp.dense_4h_to_h.weight"],
        ("gemma", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("gpt2", LayerTensorRole::FfnDownWeight) => &["h.{n}.mlp.c_proj.weight", "transformer.h.{n}.mlp.c_proj.weight"],
        ("gpt_neox", LayerTensorRole::FfnDownWeight) => &["gpt_neox.layers.{n}.mlp.dense_4h_to_h.weight"],
        ("llama", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("mistral", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("openelm", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("phi", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("phi2", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.fc2.weight", "model.layers.{n}.mlp.down_proj.weight"],
        ("qwen2", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("qwen3", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("qwen3_moe", LayerTensorRole::FfnDownWeight) => &["model.language_model.layers.{n}.mlp.shared_expert.down_proj.weight", "model.layers.{n}.mlp.shared_expert.down_proj.weight"],
        ("stablelm", LayerTensorRole::FfnDownWeight) => &["model.layers.{n}.mlp.down_proj.weight"],
        ("deepseek", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("gemma", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("llama", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("mistral", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("openelm", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("phi", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("qwen2", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("qwen3", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("qwen3_moe", LayerTensorRole::FfnGateWeight) => &["model.language_model.layers.{n}.mlp.shared_expert.gate_proj.weight", "model.layers.{n}.mlp.shared_expert.gate_proj.weight"],
        ("stablelm", LayerTensorRole::FfnGateWeight) => &["model.layers.{n}.mlp.gate_proj.weight"],
        ("bert", LayerTensorRole::FfnNormBias) => &["bert.encoder.layer.{n}.output.LayerNorm.bias"],
        ("falcon", LayerTensorRole::FfnNormBias) => &["transformer.h.{n}.post_attention_layernorm.bias"],
        ("gpt2", LayerTensorRole::FfnNormBias) => &["h.{n}.ln_2.bias", "transformer.h.{n}.ln_2.bias"],
        ("gpt_neox", LayerTensorRole::FfnNormBias) => &["gpt_neox.layers.{n}.post_attention_layernorm.bias"],
        ("phi2", LayerTensorRole::FfnNormBias) => &["model.layers.{n}.post_attention_layernorm.bias"],
        ("bert", LayerTensorRole::FfnNormWeight) => &["bert.encoder.layer.{n}.output.LayerNorm.weight", "encoder.layer.{n}.output.LayerNorm.weight"],
        ("deepseek", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("falcon", LayerTensorRole::FfnNormWeight) => &["transformer.h.{n}.post_attention_layernorm.weight", "model.layers.{n}.post_attention_layernorm.weight"],
        ("gemma", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("gpt2", LayerTensorRole::FfnNormWeight) => &["h.{n}.ln_2.weight", "transformer.h.{n}.ln_2.weight"],
        ("gpt_neox", LayerTensorRole::FfnNormWeight) => &["gpt_neox.layers.{n}.post_attention_layernorm.weight"],
        ("llama", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("mistral", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("openelm", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.ffn_norm.weight", "transformer.layers.{n}.ffn_norm.weight"],
        ("phi", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("phi2", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("qwen2", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("qwen3", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("qwen3_moe", LayerTensorRole::FfnNormWeight) => &["model.language_model.layers.{n}.post_attention_layernorm.weight", "model.layers.{n}.post_attention_layernorm.weight"],
        ("stablelm", LayerTensorRole::FfnNormWeight) => &["model.layers.{n}.post_attention_layernorm.weight"],
        ("bert", LayerTensorRole::FfnUpWeight) => &["bert.encoder.layer.{n}.intermediate.dense.weight", "encoder.layer.{n}.intermediate.dense.weight"],
        ("deepseek", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("falcon", LayerTensorRole::FfnUpWeight) => &["transformer.h.{n}.mlp.dense_h_to_4h.weight", "model.layers.{n}.mlp.dense_h_to_4h.weight"],
        ("gemma", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("gpt2", LayerTensorRole::FfnUpWeight) => &["h.{n}.mlp.c_fc.weight", "transformer.h.{n}.mlp.c_fc.weight"],
        ("gpt_neox", LayerTensorRole::FfnUpWeight) => &["gpt_neox.layers.{n}.mlp.dense_h_to_4h.weight"],
        ("llama", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("mistral", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("openelm", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("phi", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("phi2", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.fc1.weight", "model.layers.{n}.mlp.up_proj.weight"],
        ("qwen2", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("qwen3", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("qwen3_moe", LayerTensorRole::FfnUpWeight) => &["model.language_model.layers.{n}.mlp.shared_expert.up_proj.weight", "model.layers.{n}.mlp.shared_expert.up_proj.weight"],
        ("stablelm", LayerTensorRole::FfnUpWeight) => &["model.layers.{n}.mlp.up_proj.weight"],
        ("bert", LayerTensorRole::KProjBias) => &["bert.encoder.layer.{n}.attention.self.key.bias"],
        ("phi", LayerTensorRole::KProjBias) => &["model.layers.{n}.self_attn.k_proj.bias"],
        ("phi2", LayerTensorRole::KProjBias) => &["model.layers.{n}.self_attn.k_proj.bias"],
        ("qwen2", LayerTensorRole::KProjBias) => &["model.layers.{n}.self_attn.k_proj.bias"],
        ("bert", LayerTensorRole::KProjWeight) => &["bert.encoder.layer.{n}.attention.self.key.weight", "encoder.layer.{n}.attention.self.key.weight"],
        ("deepseek", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("falcon", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("gemma", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("llama", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("mistral", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("openelm", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("phi", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("phi2", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("qwen2", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("qwen3", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("qwen3_moe", LayerTensorRole::KProjWeight) => &["model.language_model.layers.{n}.self_attn.k_proj.weight", "model.layers.{n}.self_attn.k_proj.weight"],
        ("stablelm", LayerTensorRole::KProjWeight) => &["model.layers.{n}.self_attn.k_proj.weight"],
        ("bert", LayerTensorRole::OProjWeight) => &["bert.encoder.layer.{n}.attention.output.dense.weight", "encoder.layer.{n}.attention.output.dense.weight"],
        ("deepseek", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("falcon", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.dense.weight", "transformer.h.{n}.self_attention.dense.weight"],
        ("gemma", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("gpt2", LayerTensorRole::OProjWeight) => &["h.{n}.attn.c_proj.weight", "transformer.h.{n}.attn.c_proj.weight"],
        ("gpt_neox", LayerTensorRole::OProjWeight) => &["gpt_neox.layers.{n}.attention.dense.weight"],
        ("llama", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("mistral", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("openelm", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("phi", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("phi2", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.dense.weight", "model.layers.{n}.self_attn.o_proj.weight"],
        ("qwen2", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("qwen3", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("qwen3_moe", LayerTensorRole::OProjWeight) => &["model.language_model.layers.{n}.self_attn.o_proj.weight", "model.layers.{n}.self_attn.o_proj.weight"],
        ("stablelm", LayerTensorRole::OProjWeight) => &["model.layers.{n}.self_attn.o_proj.weight"],
        ("bert", LayerTensorRole::QProjBias) => &["bert.encoder.layer.{n}.attention.self.query.bias"],
        ("phi", LayerTensorRole::QProjBias) => &["model.layers.{n}.self_attn.q_proj.bias"],
        ("phi2", LayerTensorRole::QProjBias) => &["model.layers.{n}.self_attn.q_proj.bias"],
        ("qwen2", LayerTensorRole::QProjBias) => &["model.layers.{n}.self_attn.q_proj.bias"],
        ("bert", LayerTensorRole::QProjWeight) => &["bert.encoder.layer.{n}.attention.self.query.weight", "encoder.layer.{n}.attention.self.query.weight"],
        ("deepseek", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("falcon", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight", "transformer.h.{n}.self_attention.query_key_value.weight"],
        ("gemma", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("llama", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("mistral", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("openelm", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("phi", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("phi2", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("qwen2", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("qwen3", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("qwen3_moe", LayerTensorRole::QProjWeight) => &["model.language_model.layers.{n}.self_attn.q_proj.weight", "model.layers.{n}.self_attn.q_proj.weight"],
        ("stablelm", LayerTensorRole::QProjWeight) => &["model.layers.{n}.self_attn.q_proj.weight"],
        ("bert", LayerTensorRole::VProjBias) => &["bert.encoder.layer.{n}.attention.self.value.bias"],
        ("phi", LayerTensorRole::VProjBias) => &["model.layers.{n}.self_attn.v_proj.bias"],
        ("phi2", LayerTensorRole::VProjBias) => &["model.layers.{n}.self_attn.v_proj.bias"],
        ("qwen2", LayerTensorRole::VProjBias) => &["model.layers.{n}.self_attn.v_proj.bias"],
        ("bert", LayerTensorRole::VProjWeight) => &["bert.encoder.layer.{n}.attention.self.value.weight", "encoder.layer.{n}.attention.self.value.weight"],
        ("deepseek", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("falcon", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("gemma", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("llama", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("mistral", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("openelm", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("phi", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("phi2", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("qwen2", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("qwen3", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        ("qwen3_moe", LayerTensorRole::VProjWeight) => &["model.language_model.layers.{n}.self_attn.v_proj.weight", "model.layers.{n}.self_attn.v_proj.weight"],
        ("stablelm", LayerTensorRole::VProjWeight) => &["model.layers.{n}.self_attn.v_proj.weight"],
        _ => &[],
    }
}

/// Returns the GGUF fallback templates for a layer role.
#[must_use]
pub fn layer_fallback_templates(role: LayerTensorRole) -> &'static [&'static str] {
    match role {
        LayerTensorRole::AttnKNormWeight => &["blk.{n}.attn_k_norm.weight"],
        LayerTensorRole::AttnNormBias => &["blk.{n}.attn_norm.bias"],
        LayerTensorRole::AttnNormWeight => &["blk.{n}.attn_norm.weight"],
        LayerTensorRole::AttnQNormWeight => &["blk.{n}.attn_q_norm.weight"],
        LayerTensorRole::FfnDownWeight => &["blk.{n}.ffn_down.weight"],
        LayerTensorRole::FfnGateWeight => &["blk.{n}.ffn_gate.weight"],
        LayerTensorRole::FfnNormBias => &["blk.{n}.ffn_norm.bias"],
        LayerTensorRole::FfnNormWeight => &["blk.{n}.ffn_norm.weight"],
        LayerTensorRole::FfnUpWeight => &["blk.{n}.ffn_up.weight"],
        LayerTensorRole::KProjBias => &["blk.{n}.attn_k.bias"],
        LayerTensorRole::KProjWeight => &["blk.{n}.attn_k.weight"],
        LayerTensorRole::OProjWeight => &["blk.{n}.attn_output.weight"],
        LayerTensorRole::QProjBias => &["blk.{n}.attn_q.bias"],
        LayerTensorRole::QProjWeight => &["blk.{n}.attn_q.weight"],
        LayerTensorRole::VProjBias => &["blk.{n}.attn_v.bias"],
        LayerTensorRole::VProjWeight => &["blk.{n}.attn_v.weight"],
    }
}

/// Returns the fused tensor name templates for a role and architecture.
#[must_use]
pub fn fused_templates(arch: &str, role: FusedTensorRole) -> &'static [&'static str] {
    match (arch, role) {
        ("falcon", FusedTensorRole::FusedQkv) => &["transformer.h.{n}.self_attention.query_key_value.weight"],
        ("gpt2", FusedTensorRole::FusedQkv) => &["h.{n}.attn.c_attn.weight", "transformer.h.{n}.attn.c_attn.weight"],
        ("gpt_neox", FusedTensorRole::FusedQkv) => &["gpt_neox.layers.{n}.attention.query_key_value.weight"],
        ("falcon", FusedTensorRole::FusedQkvBias) => &["transformer.h.{n}.self_attention.query_key_value.bias"],
        ("gpt2", FusedTensorRole::FusedQkvBias) => &["h.{n}.attn.c_attn.bias", "transformer.h.{n}.attn.c_attn.bias"],
        ("gpt_neox", FusedTensorRole::FusedQkvBias) => &["gpt_neox.layers.{n}.attention.query_key_value.bias"],
        _ => &[],
    }
}

/// Returns the GGUF fallback templates for a fused role.
#[must_use]
pub fn fused_fallback_templates(role: FusedTensorRole) -> &'static [&'static str] {
    match role {
        FusedTensorRole::FusedQkv => &["blk.{n}.attn_qkv.weight"],
        FusedTensorRole::FusedQkvBias => &["blk.{n}.attn_qkv.bias"],
    }
}
