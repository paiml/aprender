//! APR-EMBED-001 EG-1 (I-3): the generative verbs refuse an embedding model by kind and name the verb that
//! serves it. Contract: `contracts/embeddinggemma2-load-v1.yaml` (FALSIFY-EG2L-007).

use crate::error::CliError;
use realizar::gguf::GGUFModel;
use realizar::model_kind::{gguf_model_kind, ModelKind};

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

#[cfg(test)]
#[path = "model_kind_gate_tests.rs"]
pub(crate) mod tests;
