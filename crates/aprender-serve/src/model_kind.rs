//! APR-EMBED-001 EG-1: what kind of model a file is, and the per-layer facts an embedding model reads from it.
//!
//! The kind is decided by a distinct, EXACT architecture match. The `gemma` prefix match in
//! [`crate::contract_gate::is_gemma_family`] is not widened: `gemma-embedding2` is admitted here, before that
//! gate sees it, and every other `gemma*` string still reaches it unchanged.
//!
//! Per-layer facts come from the file (I-9): the KV-head count of each layer is read from the
//! `attention.head_count_kv` value, scalar or array, and the attention kind of each layer from the width of
//! its query projection. Never from layer-index arithmetic, never from an architecture constant.
//!
//! Contract: `contracts/embeddinggemma2-load-v1.yaml`.

use crate::gguf::{GGUFModel, GGUFValue};

/// The GGUF `general.architecture` string of EmbeddingGemma 2, as the files carry it.
pub const EMBEDDING_GEMMA2_ARCH: &str = "gemma-embedding2";

/// What a model file produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Next-token logits from a causal language-model head.
    Generative,
    /// One pooled, projected vector per input (bidirectional attention, no language-model head).
    Embedding,
}

/// Is `arch` the EmbeddingGemma 2 architecture? Exact and case-insensitive; no prefix.
#[must_use]
pub fn is_embedding_gemma2(arch: &str) -> bool {
    arch.eq_ignore_ascii_case(EMBEDDING_GEMMA2_ARCH)
}

/// The kind of a file, from its architecture string and its `attention.causal` flag.
///
/// An embedding architecture whose file says `causal = true` is a contradiction and is refused by name: it is
/// never served under either kind.
///
/// # Errors
/// Returns the reason when the architecture and the causal flag disagree.
pub fn model_kind(arch: &str, causal: Option<bool>) -> Result<ModelKind, String> {
    if !is_embedding_gemma2(arch) {
        return Ok(ModelKind::Generative);
    }
    match causal {
        Some(true) => Err(format!(
            "'{arch}' is an embedding architecture but the file sets attention.causal = true; \
             refusing a file whose metadata contradicts its architecture"
        )),
        Some(false) | None => Ok(ModelKind::Embedding),
    }
}

/// The kind of a parsed GGUF file, from `general.architecture` and `<arch>.attention.causal`.
///
/// A file with no architecture string is generative: the generative loaders keep their own refusals for it.
///
/// # Errors
/// Returns the reason when the architecture and the causal flag disagree (see [`model_kind`]).
pub fn gguf_model_kind(model: &GGUFModel) -> Result<ModelKind, String> {
    let Some(arch) = model.architecture() else {
        return Ok(ModelKind::Generative);
    };
    let causal = match model.metadata.get(&format!("{arch}.attention.causal")) {
        Some(GGUFValue::Bool(b)) => Some(*b),
        _ => None,
    };
    model_kind(arch, causal)
}

/// The KV-head count of every layer, from the `attention.head_count_kv` value.
///
/// A scalar applies to every layer. An array gives one count per layer and must be exactly `block_count` long.
///
/// # Errors
/// Returns the reason for an array of the wrong length, a zero or non-integer count, or any other value type.
pub fn per_layer_kv_heads(value: &GGUFValue, block_count: usize) -> Result<Vec<usize>, String> {
    match value {
        GGUFValue::Array(items) => {
            if items.len() != block_count {
                return Err(format!(
                    "attention.head_count_kv has {} entries for {block_count} blocks",
                    items.len()
                ));
            }
            items.iter().map(kv_count).collect()
        },
        scalar => Ok(vec![kv_count(scalar)?; block_count]),
    }
}

fn kv_count(value: &GGUFValue) -> Result<usize, String> {
    let n = match value {
        GGUFValue::UInt8(v) => u64::from(*v),
        GGUFValue::UInt16(v) => u64::from(*v),
        GGUFValue::UInt32(v) => u64::from(*v),
        GGUFValue::UInt64(v) => *v,
        GGUFValue::Int32(v) => {
            u64::try_from(*v).map_err(|_| format!("negative KV-head count {v}"))?
        },
        other => {
            return Err(format!(
                "attention.head_count_kv entry is not an integer: {other:?}"
            ))
        },
    };
    match usize::try_from(n) {
        Ok(0) => Err("attention.head_count_kv entry is 0".to_string()),
        Ok(k) => Ok(k),
        Err(_) => Err(format!("KV-head count {n} does not fit usize")),
    }
}

/// Per-layer KV-head counts of a GGUF file, from `attention.head_count_kv`, scalar or array (I-9).
///
/// `None` when the key is absent. `Some(Err)` when the value cannot be read per layer: an array of the
/// wrong length, a zero or non-integer count, or a missing `block_count`.
#[must_use]
pub fn num_kv_heads_per_layer(model: &GGUFModel) -> Option<Result<Vec<usize>, String>> {
    let arch = model.architecture()?;
    let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_HEAD_COUNT_KV);
    let value = model.metadata.get(&key)?;
    let Some(block_count) = model.num_layers() else {
        return Some(Err(
            "block_count missing: cannot read per-layer KV heads".to_string()
        ));
    };
    Some(per_layer_kv_heads(value, block_count))
}

/// The single KV-head count of a model whose layers all share one, from the per-layer counts.
///
/// A generative config carries one KV-head count. Per-layer counts that differ cannot be represented there
/// and are refused by name, never collapsed to the first entry or to the query head count (I-9).
///
/// # Errors
/// Returns the reason when the list is empty or its counts differ.
pub fn uniform_kv_heads(per_layer: &[usize]) -> Result<usize, String> {
    let Some(&first) = per_layer.first() else {
        return Err("no layers: cannot read the KV-head count".to_string());
    };
    if per_layer.iter().any(|&k| k != first) {
        return Err(format!(
            "per-layer attention.head_count_kv differs across layers ({per_layer:?}); a config with one \
             KV-head count cannot represent it (I-9)"
        ));
    }
    Ok(first)
}

/// The attention kind of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAttention {
    /// Sliding-window attention over the declared window.
    Local,
    /// Full bidirectional attention over every real token.
    Global,
}

/// The attention kind of every layer, from the output width of each layer's query projection.
///
/// A layer is global when its query projection is the widest in the file, local otherwise. A file whose
/// layers all have one width carries no local/global split and is refused: this architecture has both.
///
/// # Errors
/// Returns the reason for an empty layer list, a zero width, or a file with a single width.
pub fn layer_attention_kinds(q_out_widths: &[usize]) -> Result<Vec<LayerAttention>, String> {
    let (Some(&widest), Some(&narrowest)) = (q_out_widths.iter().max(), q_out_widths.iter().min())
    else {
        return Err("no attn_q tensors: cannot read the layer attention kinds".to_string());
    };
    if narrowest == 0 {
        return Err("an attn_q tensor has output width 0".to_string());
    }
    if widest == narrowest {
        return Err(format!(
            "every attn_q has output width {widest}: the file carries no local/global split"
        ));
    }
    Ok(q_out_widths
        .iter()
        .map(|&w| {
            if w == widest {
                LayerAttention::Global
            } else {
                LayerAttention::Local
            }
        })
        .collect())
}

#[cfg(test)]
#[path = "model_kind_tests.rs"]
mod tests;
