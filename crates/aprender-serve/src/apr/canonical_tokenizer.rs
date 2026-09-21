//! #3742: the canonical byte-level BPE for the two non-GGUF tokenizer sources, an `.apr`'s
//! embedded tables and a HuggingFace `tokenizer.json`.
//!
//! Both used to feed `bpe_encode`, which runs every merge over the whole text with no
//! pre-tokenizer: the defect #3726 removed from the GGUF path, left here as a sibling copy.
//! Each loader now names the model's pre-tokenizer and builds the same
//! [`ByteLevelBpe`] the GGUF path uses. When it cannot name one, it keeps the legacy
//! encoder and says so once on stderr, naming #3742.

use std::sync::{Arc, Once};

use crate::gguf::byte_level_bpe::{ByteLevelBpe, PreTokenizer};

/// A GPT-2 byte-level vocabulary spells space and newline with the glyphs `Ġ` (U+0120) and
/// `Ċ` (U+010A), and both are tokens. A SentencePiece vocabulary (`▁`) has neither.
pub fn is_byte_level(vocab: &[String], model_type: Option<&str>) -> bool {
    let spm = model_type.is_some_and(|m| {
        matches!(
            m.to_ascii_lowercase().as_str(),
            "llama" | "unigram" | "spm" | "sentencepiece"
        )
    });
    !spm && vocab.iter().any(|t| t == "\u{0120}") && vocab.iter().any(|t| t == "\u{010A}")
}

/// Where the pre-tokenizer of an `.apr` came from, for the path line `apr tokenize encode`
/// prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreSource {
    /// The file's own `tokenizer.pre_type` (written by the importer since GH-277).
    Declared(String),
    /// Inferred from the file's architecture (older conversions carry no name).
    Architecture(String),
}

impl std::fmt::Display for PreSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declared(p) => write!(f, "tokenizer.pre_type = {p}"),
            Self::Architecture(a) => write!(f, "inferred from architecture {a}"),
        }
    }
}

/// The pre-tokenizer of an `.apr`'s embedded byte-level tokenizer, and where it came from.
pub fn apr_pre_tokenizer(meta: &super::AprMetadata) -> Option<(PreTokenizer, PreSource)> {
    if let Some(named) = meta
        .extra
        .get("tokenizer.pre_type")
        .and_then(|v| v.as_str())
    {
        return PreTokenizer::from_gguf_pre(named)
            .map(|p| (p, PreSource::Declared(named.to_string())));
    }
    let arch = meta.architecture.as_deref()?;
    PreTokenizer::for_architecture(arch).map(|p| (p, PreSource::Architecture(arch.to_string())))
}

/// The canonical tokenizer for an `.apr`'s embedded tables, or `None` when they are not
/// byte-level BPE or the pre-tokenizer cannot be named (said once).
pub(crate) fn canonical_for_apr(
    meta: &super::AprMetadata,
    vocab: &[String],
    merges: &[(String, String)],
) -> Option<Arc<ByteLevelBpe>> {
    if !is_byte_level(vocab, meta.get_embedded_model_type().as_deref()) {
        return None;
    }
    let Some((pre, _)) = apr_pre_tokenizer(meta) else {
        warn_legacy_once(&format!(
            "the .apr names no implemented pre-tokenizer (tokenizer.pre_type {:?}, architecture {:?})",
            meta.extra.get("tokenizer.pre_type").and_then(|v| v.as_str()),
            meta.architecture
        ));
        return None;
    };
    let types: Vec<i32> = meta
        .extra
        .get("tokenizer.token_type")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|x| x.as_i64().unwrap_or(1) as i32).collect())
        .unwrap_or_default();
    Some(build(pre, vocab, merges, &types))
}

/// The canonical tokenizer for a HuggingFace `tokenizer.json`, or `None` when its
/// pre-tokenizer is not one this crate implements (said once). `added` are its
/// `added_tokens` as (content, id, special); they take the ids above the base vocabulary and
/// become CONTROL (special) or USER_DEFINED tokens, as llama.cpp's converter makes them.
pub(crate) fn canonical_for_tokenizer_json(
    json: &serde_json::Value,
    id_to_token: &[String],
    merges: &[(String, String)],
    added: &[(String, u32, bool)],
) -> Option<Arc<ByteLevelBpe>> {
    let mut vocab = id_to_token.to_vec();
    let mut types = vec![1i32; vocab.len()];
    for (content, id, special) in added {
        let id = *id as usize;
        if vocab.len() <= id {
            vocab.resize(id + 1, String::new());
            types.resize(id + 1, 1);
        }
        vocab[id].clone_from(content);
        types[id] = if *special { 3 } else { 4 };
    }
    if !is_byte_level(&vocab, None) {
        return None;
    }
    let pattern = json.get("pre_tokenizer").and_then(split_regex);
    let Some(pre) = pattern.as_deref().and_then(PreTokenizer::from_hf_regex) else {
        warn_legacy_once(&format!(
            "tokenizer.json's pre-tokenizer is not implemented (Split regex {pattern:?})"
        ));
        return None;
    };
    Some(build(pre, &vocab, merges, &types))
}

fn build(
    pre: PreTokenizer,
    vocab: &[String],
    merges: &[(String, String)],
    types: &[i32],
) -> Arc<ByteLevelBpe> {
    let joined: Vec<String> = merges.iter().map(|(a, b)| format!("{a} {b}")).collect();
    let refs: Vec<&str> = joined.iter().map(String::as_str).collect();
    ByteLevelBpe::cached(pre, vocab, &refs, types)
}

/// The first `Split` regex in a tokenizer.json `pre_tokenizer` (a bare `Split`, or one inside
/// a `Sequence`).
fn split_regex(node: &serde_json::Value) -> Option<String> {
    if node.get("type").and_then(|t| t.as_str()) == Some("Split") {
        return node
            .get("pattern")
            .and_then(|p| p.get("Regex"))
            .and_then(|r| r.as_str())
            .map(String::from);
    }
    node.get("pretokenizers")
        .and_then(|a| a.as_array())
        .and_then(|a| a.iter().find_map(split_regex))
}

fn warn_legacy_once(why: &str) {
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "warning: tokenizer: {why}; falling back to the legacy BPE, which does NOT reproduce \
             the model's tokenization (#3742)"
        );
    });
}

#[cfg(test)]
#[path = "canonical_tokenizer_tests.rs"]
mod tests;
