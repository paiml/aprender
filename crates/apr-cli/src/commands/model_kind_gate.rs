//! APR-EMBED-001 EG-1 (I-3): the generative verbs refuse an embedding model by kind and name the verb that
//! serves it. Contract: `contracts/embeddinggemma2-load-v1.yaml` (FALSIFY-EG2L-007).

use crate::error::CliError;
#[cfg(feature = "inference")]
use realizar::gguf::GGUFModel;
#[cfg(feature = "inference")]
use realizar::model_kind::{gguf_model_kind, ModelKind};
#[cfg(feature = "inference")]
use std::path::Path;

// #7294: without `inference` there is no realizar to read a GGUF header with. The verbs that reach
// this gate cannot serve a GGUF file in that build either, so the gate refuses a GGUF file by name
// rather than letting it pass unchecked; any other format is not this gate's concern.
#[cfg(not(feature = "inference"))]
mod no_inference {
    use crate::error::CliError;
    use std::path::Path;

    fn refuse(verb: &str) -> CliError {
        CliError::ModelLoadFailed(format!(
            "`apr {verb}` cannot check a GGUF file's model kind: this apr was built without the \
             `inference` feature"
        ))
    }

    pub(crate) fn refuse_non_generative_bytes(bytes: &[u8], verb: &str) -> Result<(), CliError> {
        if bytes.starts_with(b"GGUF") {
            Err(refuse(verb))
        } else {
            Ok(())
        }
    }

    pub(crate) fn refuse_non_generative_path(path: &Path, verb: &str) -> Result<(), CliError> {
        match super::super::model_header::read_prefix(path, 4) {
            Ok(p) => refuse_non_generative_bytes(&p, verb),
            Err(_) => Ok(()),
        }
    }

    pub(crate) fn refuse_gguf_path(path: &Path, verb: &str) -> Result<(), CliError> {
        refuse_non_generative_path(path, verb)
    }

    pub(crate) fn is_embedding_path(_path: &Path) -> bool {
        false
    }
}
#[cfg(not(feature = "inference"))]
pub(crate) use no_inference::{
    is_embedding_path, refuse_gguf_path, refuse_non_generative_bytes, refuse_non_generative_path,
};

/// Refuse a GGUF file that is not a generative model, before a tokenizer or backend is built.
///
/// # Errors
/// `CliError::ModelLoadFailed` (exit 6) for an embedding model, naming `apr embed`, and for a file whose
/// causal flag contradicts its architecture.
#[cfg(feature = "inference")]
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
#[cfg(feature = "inference")]
pub(crate) fn refuse_non_generative_bytes(model_bytes: &[u8], verb: &str) -> Result<(), CliError> {
    match GGUFModel::from_bytes(model_bytes) {
        Ok(model) => refuse_non_generative(&model, verb),
        Err(_) => Ok(()),
    }
}

/// The header of the GGUF file at `path`, read without the tensor data
/// ([`super::model_header::gguf_model_header`]). `None` for a file that is not GGUF or whose header
/// does not parse; the loaders keep those errors.
#[cfg(feature = "inference")]
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
#[cfg(feature = "inference")]
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
#[cfg(feature = "inference")]
pub(crate) fn is_embedding_path(path: &Path) -> bool {
    read_gguf_header(path).is_some_and(|m| matches!(gguf_model_kind(&m), Ok(ModelKind::Embedding)))
}

/// Refuse every GGUF file for a verb that serves neither kind, naming the verb that serves its kind.
///
/// APR-EMBED-001 EG-5 (I-3): `apr rerank` scores with a BERT cross-encoder APR file. Before EG-5 an
/// EmbeddingGemma 2 file failed on the vocabulary or the APR magic and named no verb.
#[cfg(feature = "inference")]
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

#[cfg(all(test, feature = "inference"))]
#[path = "model_kind_gate_tests.rs"]
pub(crate) mod tests;

// #7294: the no-`inference` stubs. Without realizar the gate cannot read a GGUF header, so it refuses a
// GGUF file by its magic and passes every other format.
#[cfg(all(test, not(feature = "inference")))]
mod no_inference_tests {
    use super::*;
    use std::io::Write;

    fn temp_with(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(bytes).expect("write");
        file
    }

    #[test]
    fn gguf_magic_bytes_are_refused_with_model_load_failed() {
        let err = refuse_non_generative_bytes(b"GGUF\x03\0\0\0", "run").expect_err("GGUF refused");
        assert!(matches!(err, CliError::ModelLoadFailed(_)), "{err:?}");
        assert!(err.to_string().contains("inference"), "{err}");
    }

    #[test]
    fn non_gguf_bytes_pass() {
        assert!(refuse_non_generative_bytes(b"APR\0rest", "run").is_ok());
        assert!(refuse_non_generative_bytes(b"", "run").is_ok());
    }

    #[test]
    fn gguf_path_is_refused_by_both_path_gates() {
        let file = temp_with(b"GGUF\x03\0\0\0");
        for result in [
            refuse_non_generative_path(file.path(), "run"),
            refuse_gguf_path(file.path(), "rerank"),
        ] {
            assert!(
                matches!(result, Err(CliError::ModelLoadFailed(_))),
                "{result:?}"
            );
        }
    }

    #[test]
    fn non_gguf_and_missing_paths_pass() {
        let file = temp_with(b"APR\0rest");
        assert!(refuse_non_generative_path(file.path(), "run").is_ok());
        assert!(refuse_gguf_path(file.path(), "rerank").is_ok());
        assert!(
            refuse_non_generative_path(std::path::Path::new("/nonexistent/x.gguf"), "run").is_ok()
        );
    }

    #[test]
    fn is_embedding_path_is_false_without_inference() {
        let file = temp_with(b"GGUF\x03\0\0\0");
        assert!(!is_embedding_path(file.path()));
    }
}
