//! APR-EMBED-001 EG-2 (M5): what EmbeddingGemma 2 reads.
//!
//! The model card's seven task prompts, the document form, and the token ids of a prompt: the
//! file's SPM-style BPE vocabulary, with BOS and EOS where the file asks for them
//! (`tokenizer.ggml.add_bos_token`, `tokenizer.ggml.add_eos_token`), as llama.cpp frames an
//! embedding input. `apr embed` and the serve routes (EG-3) both build their input here, so
//! their vectors can differ only in the forward.
//!
//! The tokenizer is fail-closed: a file whose vocabulary is not SPM-style BPE, or that lacks
//! merges, a framing flag or the special id a flag asks for, is refused rather than encoded by
//! the greedy fallback, which is not the model's tokenization.

use std::sync::Arc;

use crate::error::{RealizarError, Result};
use crate::gguf::spm_bpe::SpmBpe;
use crate::gguf::{GGUFModel, GGUFValue};

/// The seven query prompts of the model card (M5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Task {
    /// `task: search result | query: …`
    SearchQuery,
    /// `task: question answering | query: …`
    QuestionAnswering,
    /// `task: fact checking | query: …`
    FactChecking,
    /// `task: code retrieval | query: …`
    CodeRetrieval,
    /// `task: classification | query: …`
    Classification,
    /// `task: clustering | query: …`
    Clustering,
    /// `task: sentence similarity | query: …`
    SentenceSimilarity,
}

impl Task {
    /// Every task, in the model card's order.
    pub const ALL: [Self; 7] = [
        Self::SearchQuery,
        Self::QuestionAnswering,
        Self::FactChecking,
        Self::CodeRetrieval,
        Self::Classification,
        Self::Clustering,
        Self::SentenceSimilarity,
    ];

    /// The words after `task: ` in the query prompt.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SearchQuery => "search result",
            Self::QuestionAnswering => "question answering",
            Self::FactChecking => "fact checking",
            Self::CodeRetrieval => "code retrieval",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
            Self::SentenceSimilarity => "sentence similarity",
        }
    }
}

/// How the model is asked to read a text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prompt<'a> {
    /// The text alone. M5: the model card warns this lowers quality, with no error.
    Raw,
    /// `task: {label} | query: {text}`.
    Query(Task),
    /// `title: {title or none} | text: {text}`.
    Document {
        /// The document's title; `none` when absent.
        title: Option<&'a str>,
    },
}

impl Prompt<'_> {
    /// The string the tokenizer reads for `text`.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        match self {
            Self::Raw => text.to_string(),
            Self::Query(task) => format!("task: {} | query: {text}", task.label()),
            Self::Document { title } => {
                format!("title: {} | text: {text}", title.unwrap_or("none"))
            },
        }
    }

    /// A short name for receipts (I-12): `raw`, the task label, or `document`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Query(task) => task.label(),
            Self::Document { .. } => "document",
        }
    }
}

fn refuse(reason: impl Into<String>) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: "embedding_gemma2_tokenizer".to_string(),
        reason: reason.into(),
    }
}

/// A framing flag the file must declare: `tokenizer.ggml.add_{bos,eos}_token`.
fn declared_flag(model: &GGUFModel, key: &str) -> Result<bool> {
    match model.metadata.get(key) {
        Some(GGUFValue::Bool(flag)) => Ok(*flag),
        Some(other) => Err(refuse(format!("{key} is not a bool: {other:?}"))),
        None => Err(refuse(format!(
            "{key} missing; the framing of an embedding input is not guessed"
        ))),
    }
}

/// The special id a framing flag asks for, inside the vocabulary.
fn special_id(id: Option<u32>, name: &str, vocab_len: usize) -> Result<u32> {
    let id = id.ok_or_else(|| {
        refuse(format!(
            "tokenizer.ggml.add_{name}_token is true but tokenizer.ggml.{name}_token_id is missing"
        ))
    })?;
    if id as usize >= vocab_len {
        return Err(refuse(format!(
            "tokenizer.ggml.{name}_token_id {id} is outside the vocabulary of {vocab_len}"
        )));
    }
    Ok(id)
}

/// The tokenizer of an EmbeddingGemma 2 file, built once and used for every text.
#[derive(Debug, Clone)]
pub struct EmbedTokenizer {
    bpe: Arc<SpmBpe>,
    bos: Option<u32>,
    eos: Option<u32>,
}

impl EmbedTokenizer {
    /// Build the tokenizer the file declares.
    ///
    /// # Errors
    /// No vocabulary, a vocabulary that is not SPM-style BPE or has no merges, a missing or
    /// non-bool framing flag, or a missing or out-of-range special id that a flag asks for.
    pub fn from_gguf(model: &GGUFModel) -> Result<Self> {
        let vocab = model
            .vocabulary()
            .ok_or_else(|| refuse("tokenizer.ggml.tokens missing"))?;
        let bpe = SpmBpe::from_gguf(&model.metadata, &vocab)
            .map_err(|refusal| refuse(refusal.to_string()))?;
        let add_bos = declared_flag(model, "tokenizer.ggml.add_bos_token")?;
        let add_eos = declared_flag(model, "tokenizer.ggml.add_eos_token")?;
        let bos = if add_bos {
            Some(special_id(model.bos_token_id(), "bos", vocab.len())?)
        } else {
            None
        };
        let eos = if add_eos {
            Some(special_id(model.eos_token_id(), "eos", vocab.len())?)
        } else {
            None
        };
        Ok(Self { bpe, bos, eos })
    }

    /// The framed token ids of `prompt` applied to `text`.
    #[must_use]
    pub fn tokens(&self, prompt: &Prompt<'_>, text: &str) -> Vec<u32> {
        let body = self.bpe.encode(&prompt.apply(text));
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.extend(self.bos);
        ids.extend(body);
        ids.extend(self.eos);
        ids
    }
}

#[cfg(test)]
#[path = "embedding_gemma2_input_tests.rs"]
mod tests;
