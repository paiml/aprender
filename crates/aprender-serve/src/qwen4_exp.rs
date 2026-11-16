//! `qwen4_exp` (Qwen 4 preview architecture) — S1 of paiml/aprender#7169.
//!
//! RECOGNISE and load as METADATA only. No forward exists for the new ops, so
//! every inference entry refuses BY NAME ([`refusal_reason`]) instead of
//! running a Qwen3.5-shaped forward over weights it cannot interpret.
//!
//! Source of the key list: `Qwen/Qwen3.8-Flash-Next` `config.json`
//! (fixture: `tests/fixtures/qwen4_exp/`).

use serde::Deserialize;

// One source for ids and the refusal text: `apr-format` (shared with `apr import`).
pub use apr_format::qwen4_exp::{is_qwen4_exp, ARCHITECTURE, MODEL_TYPE, TEXT_MODEL_TYPE};

/// `mtp` sub-object (multi-token-prediction head).
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct MtpConfig {
    /// Hybrid (linear + full attention) MTP layers.
    pub hybrid: Option<bool>,
    /// Layer types of the MTP stack.
    pub layer_types: Option<Vec<String>>,
    /// Number of MTP layers.
    pub num_hidden_layers: Option<usize>,
    /// RoPE theta of the MTP layers.
    pub rope_theta: Option<f64>,
}

/// `rope_parameters` sub-object.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct RopeParameters {
    /// Interleaved multimodal RoPE.
    pub mrope_interleaved: Option<bool>,
    /// mrope section sizes.
    pub mrope_section: Option<Vec<usize>>,
    /// Fraction of head_dim that is rotated.
    pub partial_rotary_factor: Option<f32>,
    /// RoPE base.
    pub rope_theta: Option<f64>,
    /// RoPE type.
    pub rope_type: Option<String>,
}

/// `vision_config` sub-object (tower is parsed, not run).
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct VisionConfig {
    /// Transformer depth.
    pub depth: Option<usize>,
    /// Hidden size.
    pub hidden_size: Option<usize>,
    /// Patch size.
    pub patch_size: Option<usize>,
    /// Output hidden size (projects into the text hidden size).
    pub out_hidden_size: Option<usize>,
}

/// Typed `text_config` of a `qwen4_exp` checkpoint. Every new key is a named
/// field so none is silently dropped.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct Qwen4ExpConfig {
    /// Top-level `architectures` (not in `text_config`; filled by the parser).
    #[serde(skip)]
    pub architectures: Vec<String>,
    /// `model_type` of the text config.
    pub model_type: Option<String>,
    /// Hidden size.
    pub hidden_size: Option<usize>,
    /// Decoder layers.
    pub num_hidden_layers: Option<usize>,
    /// Attention heads.
    pub num_attention_heads: Option<usize>,
    /// KV heads.
    pub num_key_value_heads: Option<usize>,
    /// Head dim.
    pub head_dim: Option<usize>,
    /// Vocabulary size.
    pub vocab_size: Option<usize>,
    /// Max positions.
    pub max_position_embeddings: Option<usize>,
    /// RMSNorm epsilon.
    pub rms_norm_eps: Option<f32>,
    /// Tied embeddings.
    pub tie_word_embeddings: Option<bool>,
    /// Per-layer `linear_attention` / `full_attention`.
    pub layer_types: Option<Vec<String>>,
    /// Every Nth layer is full attention.
    pub full_attention_interval: Option<usize>,
    /// Partial rotary factor (also inside `rope_parameters`).
    pub partial_rotary_factor: Option<f32>,
    /// RoPE parameters.
    pub rope_parameters: Option<RopeParameters>,
    // --- Gated DeltaNet (already in tree via Qwen3.5) ---
    /// Conv kernel.
    pub linear_conv_kernel_dim: Option<usize>,
    /// Key head dim.
    pub linear_key_head_dim: Option<usize>,
    /// Value head dim.
    pub linear_value_head_dim: Option<usize>,
    /// Key heads.
    pub linear_num_key_heads: Option<usize>,
    /// Value heads.
    pub linear_num_value_heads: Option<usize>,
    /// SSM state dtype.
    pub mamba_ssm_dtype: Option<String>,
    // --- MoE ---
    /// Routed experts.
    pub num_experts: Option<usize>,
    /// Experts per token (top-k).
    pub num_experts_per_tok: Option<usize>,
    /// Routed expert FFN size.
    pub moe_intermediate_size: Option<usize>,
    /// Shared expert FFN size.
    pub shared_expert_intermediate_size: Option<usize>,
    /// Router aux-loss coefficient.
    pub router_aux_loss_coef: Option<f32>,
    // --- NEW: sparse-attention lightning indexer ---
    /// Indexer token budget.
    pub indexer_budget: Option<usize>,
    /// Indexer KV compress ratio.
    pub indexer_compress_ratio: Option<usize>,
    /// Indexer head dim.
    pub indexer_head_dim: Option<usize>,
    /// Indexer KV heads.
    pub indexer_kv_heads: Option<usize>,
    /// Indexer heads.
    pub indexer_n_heads: Option<usize>,
    // --- NEW: n-gram embedding ---
    /// N-gram order.
    pub ngram_size: Option<usize>,
    /// Base hashed vocabulary.
    pub ngram_vocab_size_base: Option<usize>,
    /// Heads per n-gram.
    pub heads_per_ngram: Option<usize>,
    /// Shards of the n-gram table.
    pub split_ngram_parts: Option<usize>,
    /// Vocabulary divisibility.
    pub make_ngram_vocab_size_divisible_by: Option<usize>,
    // --- NEW: per-layer embedding ---
    /// Layers that carry a PLE block.
    pub ple_layer_ids: Option<Vec<usize>>,
    /// PLE embedding dim.
    pub ple_embed_dim: Option<usize>,
    /// PLE conv kernel.
    pub ple_conv_kernel_size: Option<usize>,
    // --- NEW: multi-stream (hyper-connection) residuals ---
    /// Number of residual streams.
    pub hc_count: Option<usize>,
    /// Low-rank mixer width.
    pub hc_lowrank: Option<usize>,
    // --- NEW: output gate, MTP ---
    /// `sigmoid` output gate on attention.
    pub output_gate_type: Option<String>,
    /// MTP sub-config.
    pub mtp: Option<MtpConfig>,
    /// MTP layer count.
    pub mtp_num_hidden_layers: Option<usize>,
    /// MTP uses its own embeddings.
    pub mtp_use_dedicated_embeddings: Option<bool>,
    /// Vision tower (parsed from the composite config, not `text_config`).
    #[serde(skip)]
    pub vision_config: Option<VisionConfig>,
}

impl Qwen4ExpConfig {
    /// Parse a composite (`text_config`-nested) or flat config.json.
    ///
    /// # Errors
    ///
    /// Invalid JSON, or the config is not a `qwen4_exp` one.
    pub fn parse_config_json(content: &str) -> Result<Self, String> {
        let raw: serde_json::Value =
            serde_json::from_str(content).map_err(|e| format!("invalid config.json: {e}"))?;
        let text = raw.get("text_config").unwrap_or(&raw);
        let mut cfg: Self =
            serde_json::from_value(text.clone()).map_err(|e| format!("bad text_config: {e}"))?;
        cfg.architectures = raw
            .get("architectures")
            .and_then(|a| serde_json::from_value(a.clone()).ok())
            .unwrap_or_default();
        cfg.vision_config = raw
            .get("vision_config")
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        let top_type = raw.get("model_type").and_then(|v| v.as_str());
        if !is_qwen4_exp(top_type.or(cfg.model_type.as_deref()), &cfg.architectures) {
            return Err(format!(
                "not a {MODEL_TYPE} config (model_type={top_type:?}, architectures={:?})",
                cfg.architectures
            ));
        }
        Ok(cfg)
    }

    /// The ops this config needs that this build has no implementation of.
    #[must_use]
    pub fn unsupported_ops(&self) -> Vec<UnsupportedOp> {
        let mut ops = Vec::new();
        if self.indexer_budget.is_some() || self.indexer_n_heads.is_some() {
            ops.push(UnsupportedOp::SparseAttentionIndexer);
        }
        if self.ngram_size.is_some() {
            ops.push(UnsupportedOp::NgramEmbedding);
        }
        if self.ple_layer_ids.as_ref().is_some_and(|l| !l.is_empty()) {
            ops.push(UnsupportedOp::PerLayerEmbedding);
        }
        if self.hc_count.is_some() {
            ops.push(UnsupportedOp::HyperConnectionResidual);
        }
        if self.output_gate_type.as_deref() == Some("sigmoid") {
            ops.push(UnsupportedOp::SigmoidOutputGate);
        }
        if self.mtp.is_some() || self.mtp_num_hidden_layers.unwrap_or(0) > 0 {
            ops.push(UnsupportedOp::MultiTokenPrediction);
        }
        ops
    }
}

/// An op `qwen4_exp` needs that has no forward in this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedOp {
    /// Qwen Sparse Attention lightning indexer (`indexer_*`).
    SparseAttentionIndexer,
    /// Hashed n-gram embedding (`ngram_*`).
    NgramEmbedding,
    /// Per-layer embedding (`ple_*`).
    PerLayerEmbedding,
    /// Multi-stream residual (`hc_*`).
    HyperConnectionResidual,
    /// Sigmoid output gate (`output_gate_type`).
    SigmoidOutputGate,
    /// Multi-token-prediction head (`mtp`).
    MultiTokenPrediction,
}

impl UnsupportedOp {
    /// Stable name used in the refusal.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SparseAttentionIndexer => "sparse-attention indexer",
            Self::NgramEmbedding => "ngram embedding",
            Self::PerLayerEmbedding => "ple",
            Self::HyperConnectionResidual => "hc residual",
            Self::SigmoidOutputGate => "sigmoid output gate",
            Self::MultiTokenPrediction => "mtp",
        }
    }
}

/// The named refusal for inference on `qwen4_exp`.
#[must_use]
pub fn refusal_reason(ops: &[UnsupportedOp]) -> String {
    let names: Vec<&str> = ops.iter().map(|o| o.name()).collect();
    apr_format::qwen4_exp::refusal_reason(&names)
}

/// Refusal for a `qwen4_exp` safetensors config, `None` for any other arch.
#[must_use]
pub fn refusal_for(model_type: Option<&str>, architectures: &[String]) -> Option<String> {
    apr_format::qwen4_exp::refusal_for(model_type, architectures)
}

/// Fail closed if `config` is `qwen4_exp`. Not feature-gated: shared by the CPU
/// converter and the CUDA loader so both refuse by name before reading tensors.
///
/// # Errors
/// `UnsupportedOperation` naming every unsupported op.
pub fn check_config(config: &crate::SafetensorsConfig) -> crate::error::Result<()> {
    match refusal_for(
        config.model_type.as_deref(),
        config.architectures.as_deref().unwrap_or(&[]),
    ) {
        Some(reason) => Err(crate::error::RealizarError::UnsupportedOperation {
            operation: "qwen4_exp_forward".to_string(),
            reason,
        }),
        None => Ok(()),
    }
}

/// Tensor families of a `qwen4_exp` checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorFamily {
    /// Token embedding / lm_head.
    Embedding,
    /// Gated DeltaNet layer tensors.
    LinearAttention,
    /// Full (sparse) attention q/k/v/o and q/k norms.
    FullAttention,
    /// Lightning indexer.
    Indexer,
    /// Router, routed experts, shared expert.
    Moe,
    /// Hyper-connection residual tensors.
    HyperConnection,
    /// N-gram embedding table shards and offsets.
    NgramEmbedding,
    /// Per-layer embedding block (non n-gram part).
    PerLayerEmbedding,
    /// MTP head.
    Mtp,
    /// Vision tower.
    Vision,
}

/// Layer-suffix prefix table, first match wins (`self_attn.indexer.` before `self_attn.`).
const LAYER_PREFIXES: [(&str, TensorFamily); 8] = [
    ("linear_attn.", TensorFamily::LinearAttention),
    ("self_attn.indexer.", TensorFamily::Indexer),
    ("self_attn.", TensorFamily::FullAttention),
    ("mlp.", TensorFamily::Moe),
    ("attn_hyper_connection.", TensorFamily::HyperConnection),
    ("mlp_hyper_connection.", TensorFamily::HyperConnection),
    ("ple.ple_embedding.", TensorFamily::NgramEmbedding),
    ("ple.", TensorFamily::PerLayerEmbedding),
];

/// Classify a name below `layers.<idx>.`; `None` means unknown.
fn classify_layer_suffix(rest: &str) -> Option<TensorFamily> {
    LAYER_PREFIXES
        .iter()
        .find(|(p, _)| rest.starts_with(p))
        .map(|&(_, f)| f)
}

/// Classify a tensor name; `None` means unknown.
#[must_use]
pub fn classify_tensor(name: &str) -> Option<TensorFamily> {
    use TensorFamily as F;
    if name.starts_with("mtp.") {
        return Some(F::Mtp);
    }
    if name.starts_with("model.visual.") {
        return Some(F::Vision);
    }
    if name == "lm_head.weight" || name == "model.language_model.embed_tokens.weight" {
        return Some(F::Embedding);
    }
    let rest = name.strip_prefix("model.language_model.")?;
    if rest.starts_with("hyper_connection_mixer.") {
        return Some(F::HyperConnection);
    }
    let (idx, rest) = rest.strip_prefix("layers.")?.split_once('.')?;
    idx.parse::<usize>().ok()?;
    classify_layer_suffix(rest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const CONFIG: &str = include_str!("../tests/fixtures/qwen4_exp/config.json");
    const INDEX: &str = include_str!("../tests/fixtures/qwen4_exp/model.safetensors.index.json");

    #[test]
    fn config_fixture_parses_with_every_new_key() {
        let c = Qwen4ExpConfig::parse_config_json(CONFIG).expect("parse");
        assert_eq!(c.hidden_size, Some(2560));
        assert_eq!(c.num_hidden_layers, Some(48));
        assert_eq!(c.num_experts, Some(512));
        assert_eq!(c.num_experts_per_tok, Some(10));
        assert_eq!(c.full_attention_interval, Some(4));
        assert_eq!(c.partial_rotary_factor, Some(0.25));
        assert_eq!(c.layer_types.as_ref().map(Vec::len), Some(48));
        assert_eq!(c.indexer_budget, Some(2048));
        assert_eq!(c.indexer_compress_ratio, Some(4));
        assert_eq!(c.indexer_head_dim, Some(128));
        assert_eq!(c.indexer_kv_heads, Some(1));
        assert_eq!(c.indexer_n_heads, Some(4));
        assert_eq!(c.ngram_size, Some(3));
        assert_eq!(c.ngram_vocab_size_base, Some(20_000_000));
        assert_eq!(c.heads_per_ngram, Some(8));
        assert_eq!(c.split_ngram_parts, Some(128));
        assert_eq!(c.make_ngram_vocab_size_divisible_by, Some(128));
        assert_eq!(c.ple_layer_ids, Some(vec![2]));
        assert_eq!(c.ple_embed_dim, Some(2560));
        assert_eq!(c.ple_conv_kernel_size, Some(4));
        assert_eq!(c.hc_count, Some(4));
        assert_eq!(c.hc_lowrank, Some(320));
        assert_eq!(c.output_gate_type.as_deref(), Some("sigmoid"));
        assert_eq!(c.mtp_num_hidden_layers, Some(1));
        assert_eq!(c.mtp.as_ref().and_then(|m| m.hybrid), Some(true));
        let rp = c.rope_parameters.as_ref().expect("rope");
        assert_eq!(rp.mrope_section, Some(vec![11, 11, 10]));
        assert_eq!(rp.mrope_interleaved, Some(true));
        assert_eq!(c.vision_config.as_ref().and_then(|v| v.depth), Some(27));
    }

    #[test]
    fn recognised_as_qwen4_exp_not_a_fallback_family() {
        let c = Qwen4ExpConfig::parse_config_json(CONFIG).expect("parse");
        assert_eq!(c.architectures, vec![ARCHITECTURE.to_string()]);
        assert!(is_qwen4_exp(Some("qwen4_exp"), &[]));
        assert!(is_qwen4_exp(Some("qwen4_exp_text"), &[]));
        assert!(is_qwen4_exp(None, &c.architectures));
        for other in ["qwen2", "qwen3", "qwen3_5", "llama"] {
            assert!(!is_qwen4_exp(Some(other), &[]), "{other}");
        }
        let qwen35 = r#"{"model_type":"qwen3_5","hidden_size":8}"#;
        assert!(Qwen4ExpConfig::parse_config_json(qwen35).is_err());
    }

    #[test]
    fn every_index_tensor_is_classified() {
        let v: serde_json::Value = serde_json::from_str(INDEX).expect("index");
        let map = v["weight_map"].as_object().expect("weight_map");
        let mut fam: BTreeMap<TensorFamily, usize> = BTreeMap::new();
        let mut unknown = Vec::new();
        for name in map.keys() {
            match classify_tensor(name) {
                Some(f) => *fam.entry(f).or_default() += 1,
                None => unknown.push(name.clone()),
            }
        }
        assert!(unknown.is_empty(), "unclassified: {unknown:?}");
        assert_eq!(fam.len(), 10, "all ten families present: {fam:?}");
    }

    #[test]
    fn config_lists_all_six_unsupported_ops_by_name() {
        let c = Qwen4ExpConfig::parse_config_json(CONFIG).expect("parse");
        let ops = c.unsupported_ops();
        assert_eq!(ops.len(), 6);
        let msg = refusal_reason(&ops);
        for n in [
            "sparse-attention indexer",
            "ngram embedding",
            "ple",
            "hc residual",
            "sigmoid output gate",
            "mtp",
        ] {
            assert!(msg.contains(n), "{n} missing from: {msg}");
        }
        assert!(msg.contains("refusal, not a fallback"));
    }

    #[test]
    fn refusal_for_only_fires_on_qwen4_exp() {
        assert!(refusal_for(Some("qwen4_exp_text"), &[]).is_some());
        assert!(refusal_for(Some("qwen3_5"), &[]).is_none());
    }

    #[test]
    fn classify_tensor_unknown_names_are_none() {
        for n in [
            "foo.weight",
            "",
            "model.language_model.layers.0.bogus.weight",
            "model.language_model.layers.x.mlp.gate.weight",
            "model.language_model.bogus",
            "model.language_model.layers.0",
        ] {
            assert_eq!(classify_tensor(n), None, "{n}");
        }
    }

    #[test]
    fn check_config_refuses_qwen4_exp_by_name_and_passes_others() {
        use crate::safetensors::SafetensorsConfig;
        let cfg: SafetensorsConfig = serde_json::from_str(CONFIG).expect("cfg");
        let err = check_config(&cfg).expect_err("must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("qwen4_exp") && msg.contains("refusal, not a fallback"),
            "{msg}"
        );
        assert!(msg.contains("no tensor data was read"), "{msg}");
        assert!(!msg.contains("nothing was loaded"), "{msg}");
        let other: SafetensorsConfig =
            serde_json::from_str(r#"{"model_type":"qwen3","hidden_size":8}"#).expect("cfg");
        assert!(check_config(&other).is_ok());
    }
}
