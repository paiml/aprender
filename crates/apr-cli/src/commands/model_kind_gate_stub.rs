//! `model_kind_gate` for a build without the `inference` feature (#7294).
//!
//! The real gate reads a GGUF header through `realizar`, which `--no-default-features` does not link.
//! Without it no GGUF file is read, so no gate can refuse by kind: the kind-based entry points pass, as the
//! real gate does for a file whose header it cannot read. `refuse_gguf_path` still refuses by GGUF magic.

use crate::error::CliError;
use std::path::Path;

/// No header reader is linked, so nothing is refused.
pub(crate) fn refuse_non_generative_path(_path: &Path, _verb: &str) -> Result<(), CliError> {
    Ok(())
}

/// No header reader is linked, so nothing is refused.
pub(crate) fn refuse_non_generative_bytes(
    _model_bytes: &[u8],
    _verb: &str,
) -> Result<(), CliError> {
    Ok(())
}

/// No header reader is linked, so no file reads as an embedding model.
pub(crate) fn is_embedding_path(_path: &Path) -> bool {
    false
}

/// Refuse every GGUF file, by magic bytes alone (#7294, lane finding on EG-5).
///
/// `apr rerank` takes a BERT cross-encoder APR file and never GGUF, whatever its kind. The magic check
/// needs no `realizar`, so the refusal holds without `inference`; only the kind (and so the verb that
/// serves it) is unknown here. A file that is not GGUF or cannot be read passes, as in the real gate.
pub(crate) fn refuse_gguf_path(path: &Path, verb: &str) -> Result<(), CliError> {
    if super::model_header::read_prefix(path, 4).is_ok_and(|magic| magic.as_slice() == b"GGUF") {
        return Err(CliError::ModelLoadFailed(format!(
            "'{}' is a GGUF file: `apr {verb}` takes a BERT cross-encoder APR file, not GGUF. \
             This build has no `inference` feature, so its kind is not read; use `apr embed` for an \
             embedding model or `apr run` for a generative one.",
            path.display()
        )));
    }
    Ok(())
}
