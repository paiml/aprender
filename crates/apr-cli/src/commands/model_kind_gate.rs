//! APR-EMBED-001 EG-1 (I-3): the generative verbs refuse an embedding model by kind and name the verb that
//! serves it. Contract: `contracts/embeddinggemma2-load-v1.yaml` (FALSIFY-EG2L-007).

use crate::error::CliError;
use realizar::gguf::GGUFModel;
use realizar::model_kind::{gguf_model_kind, ModelKind};
use std::path::Path;

/// Refuse a GGUF file that is not a generative model, before a tokenizer or backend is built.
///
/// # Errors
/// `CliError::ModelLoadFailed` (exit 6) for an embedding model, naming `apr embed`, and for a file whose
/// causal flag contradicts its architecture.
pub(crate) fn refuse_non_generative(model: &GGUFModel, verb: &str) -> Result<(), CliError> {
    match gguf_model_kind(model) {
        Ok(ModelKind::Generative) => Ok(()),
        Ok(ModelKind::Embedding) => Err(CliError::ModelLoadFailed(format!(
            "'{}' is an embedding model (ModelKind::Embedding): it has no language-model head, so \
             `apr {verb}` cannot serve it. Use `apr embed`.",
            model.architecture().unwrap_or_default()
        ))),
        Err(reason) => Err(CliError::ModelLoadFailed(reason)),
    }
}

/// [`refuse_non_generative`] on raw file bytes. A header that does not parse is left to the loaders,
/// which keep their own errors.
pub(crate) fn refuse_non_generative_bytes(model_bytes: &[u8], verb: &str) -> Result<(), CliError> {
    match GGUFModel::from_bytes(model_bytes) {
        Ok(model) => refuse_non_generative(&model, verb),
        Err(_) => Ok(()),
    }
}

/// The header of the GGUF file at `path`, read without the tensor data
/// ([`super::model_header::gguf_model_header`]). `None` for a file that is not GGUF or whose header
/// does not parse; the loaders keep those errors.
fn read_gguf_header(path: &Path) -> Option<GGUFModel> {
    if super::model_header::read_prefix(path, 4).ok()?.as_slice() != b"GGUF" {
        return None;
    }
    super::model_header::gguf_model_header(path).ok()
}

/// [`refuse_non_generative`] on a file path, reading only the header.
///
/// APR-EMBED-001 EG-5 (I-3): `run`, `code` and `showcase` open a model by path, so they refuse here
/// before any tokenizer, backend or child process exists.
pub(crate) fn refuse_non_generative_path(path: &Path, verb: &str) -> Result<(), CliError> {
    match read_gguf_header(path) {
        Some(model) => refuse_non_generative(&model, verb),
        None => Ok(()),
    }
}

/// Whether the file at `path` is a GGUF embedding model, from its header alone.
///
/// APR-EMBED-001 EG-5: `apr qa` routes an embedding model to `golden_embedding` and skips its
/// token-stream gates by kind.
pub(crate) fn is_embedding_path(path: &Path) -> bool {
    read_gguf_header(path).is_some_and(|m| matches!(gguf_model_kind(&m), Ok(ModelKind::Embedding)))
}

/// Refuse every GGUF file for a verb that serves neither kind, naming the verb that serves its kind.
///
/// APR-EMBED-001 EG-5 (I-3): `apr rerank` scores with a BERT cross-encoder APR file. Before EG-5 an
/// EmbeddingGemma 2 file failed on the vocabulary or the APR magic and named no verb.
pub(crate) fn refuse_gguf_path(path: &Path, verb: &str) -> Result<(), CliError> {
    let Some(model) = read_gguf_header(path) else {
        return Ok(());
    };
    let arch = model.architecture().unwrap_or_default().to_string();
    let (kind, use_verb) = match gguf_model_kind(&model) {
        Ok(ModelKind::Embedding) => ("an embedding model (ModelKind::Embedding)", "embed"),
        Ok(ModelKind::Generative) => ("a generative model (ModelKind::Generative)", "run"),
        Err(reason) => return Err(CliError::ModelLoadFailed(reason)),
    };
    Err(CliError::ModelLoadFailed(format!(
        "'{arch}' is {kind}: `apr {verb}` takes a BERT cross-encoder APR file, not GGUF. \
         Use `apr {use_verb}`."
    )))
}

#[cfg(test)]
#[path = "model_kind_gate_tests.rs"]
pub(crate) mod tests;
