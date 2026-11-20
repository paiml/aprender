//! `model_kind_gate` for a build without the `inference` feature (#7294).
//!
//! The real gate reads a GGUF header through `realizar`, which `--no-default-features` does not link.
//! Without it no GGUF file is read, so no gate can refuse by kind: every entry point passes, exactly as
//! the real gate does for a file whose header it cannot read.

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

/// No header reader is linked, so nothing is refused.
pub(crate) fn refuse_gguf_path(_path: &Path, _verb: &str) -> Result<(), CliError> {
    Ok(())
}
