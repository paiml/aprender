//! Model file validation (CONTRACT: `apr_model_validity`).
//!
//! Jidoka boundary check: validate model file BEFORE entering inference.
//! APR files must have embedded tokenizer. GGUF files must have valid magic.
//! Broken models are rejected with actionable error + re-conversion command.
//!
//! Used at two boundaries:
//! 1. `RealizarDriver::new()` — reject at load time (hard error)
//! 2. `ModelConfig::discover_model()` — deprioritize at discovery (PMAT-150)

use std::io::Read as _;
use std::path::Path;

use crate::agent::result::{AgentError, DriverError};

/// APR v2 magic bytes: "APR\0"
const APR_MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00];
/// GGUF magic bytes: "GGUF"
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Read the first `limit` bytes of a file (avoids reading entire multi-GB models).
fn read_header(path: &Path, limit: usize) -> Result<Vec<u8>, AgentError> {
    let file = std::fs::File::open(path).map_err(|e| {
        AgentError::Driver(DriverError::InferenceFailed(format!("cannot read model file: {e}")))
    })?;
    let mut buf = vec![0u8; limit];
    let n = file.take(limit as u64).read(&mut buf).map_err(|e| {
        AgentError::Driver(DriverError::InferenceFailed(format!("cannot read model header: {e}")))
    })?;
    buf.truncate(n);
    Ok(buf)
}

/// Validate model file at the load boundary (Jidoka: stop on first defect).
///
/// **Contract precondition**: model file must be structurally valid before
/// any inference attempt. For APR files, this means embedded tokenizer data
/// must be present. Violation → clear error with `apr convert` instructions.
pub(crate) fn validate_model_file(path: &Path) -> Result<(), AgentError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Read first 64KB for header validation (fast, no full model load)
    let header = read_header(path, 65536)?;

    if header.len() < 4 {
        return Err(AgentError::Driver(DriverError::InferenceFailed(
            "model file too small (< 4 bytes)".into(),
        )));
    }

    match ext {
        "apr" => validate_apr_header(&header, path),
        "gguf" => validate_gguf_header(&header, path),
        _ => Ok(()), // Unknown formats pass through to realizar
    }
}

/// Quick validation for model discovery — returns true if the model file
/// passes basic structural checks (APR tokenizer present, GGUF magic valid).
///
/// Used by `ModelConfig::discover_model()` to skip broken APR files and
/// fall through to GGUF. Does NOT load the full model.
pub fn is_valid_model_file(path: &Path) -> bool {
    validate_model_file(path).is_ok()
}

/// Validate APR file header and check for embedded tokenizer data.
fn validate_apr_header(header: &[u8], path: &Path) -> Result<(), AgentError> {
    // Check magic bytes
    if header[..4] != APR_MAGIC {
        return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
            "invalid APR file (wrong magic bytes): {}",
            path.display()
        ))));
    }

    // APR v2 binary metadata follows the 4-byte magic.
    // Scan the first 64KB of header for tokenizer indicators.
    //
    // APR embeds tokenizer as JSON keys in metadata:
    //   "tokenizer.merges": [...]    — BPE merge rules
    //   "tokenizer.vocabulary": [...] — token vocabulary
    //   "tokenizer.vocab_size": N    — vocabulary size (always present with tokenizer)
    //
    // PMAT-150: "vocab_size" alone (architecture field) is NOT sufficient.
    // PMAT-154: Must match "tokenizer.merges" or "tokenizer.vocabulary" —
    // the actual embedded tokenizer data, not just metadata fields.
    let header_str = String::from_utf8_lossy(&header[4..]);
    let has_tokenizer = header_str.contains("tokenizer.merges")
        || header_str.contains("tokenizer.vocabulary")
        || header_str.contains("tokenizer.ggml")
        || header_str.contains("bpe_ranks")
        || header_str.contains("token_to_id");

    if !has_tokenizer {
        return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
            "APR file missing embedded tokenizer: {}\n\
             APR format requires a self-contained tokenizer (Jidoka: fail-fast).\n\
             Re-convert with: apr convert {} -o {}",
            path.display(),
            path.with_extension("gguf").display(),
            path.display(),
        ))));
    }

    Ok(())
}

/// Validate GGUF file header magic bytes.
fn validate_gguf_header(header: &[u8], path: &Path) -> Result<(), AgentError> {
    if header[..4] != GGUF_MAGIC {
        return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
            "invalid GGUF file (wrong magic bytes): {}",
            path.display()
        ))));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CONTRACT: apr_model_validity tests (FALSIFY-AC-008) ──

    #[test]
    fn test_apr_without_tokenizer_rejected_at_boundary() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(&APR_MAGIC);
        data.extend_from_slice(&[0u8; 100]);
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_err(), "APR without tokenizer must be rejected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing embedded tokenizer"), "error must mention tokenizer: {err}");
        assert!(err.contains("apr convert"), "error must include fix command: {err}");
    }

    #[test]
    fn test_apr_with_tokenizer_passes_validation() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(&APR_MAGIC);
        // PMAT-154: use real APR tokenizer key format
        data.extend_from_slice(br#"{"tokenizer.merges":["a b"],"tokenizer.vocabulary":["hi"]}"#);
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_ok(), "APR with tokenizer should pass: {result:?}");
    }

    #[test]
    fn test_apr_with_ggml_tokenizer_passes() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(&APR_MAGIC);
        data.extend_from_slice(b"tokenizer.ggml.tokens present in this header");
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_ok(), "APR with tokenizer.ggml should pass: {result:?}");
    }

    #[test]
    fn test_apr_with_vocab_size_only_rejected() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(&APR_MAGIC);
        data.extend_from_slice(
            br#"{"architecture":"qwen2","vocab_size":151936,"hidden_size":1536}"#,
        );
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_err(), "APR with only vocab_size (no tokenizer data) must be rejected");
    }

    #[test]
    fn test_gguf_valid_magic_passes() {
        let tmp = tempfile::NamedTempFile::with_suffix(".gguf").expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&[0u8; 100]);
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_ok(), "valid GGUF should pass: {result:?}");
    }

    #[test]
    fn test_gguf_invalid_magic_rejected() {
        let tmp = tempfile::NamedTempFile::with_suffix(".gguf").expect("tmpfile");
        std::fs::write(tmp.path(), b"NOT_GGUF_DATA_HERE").expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_err(), "invalid GGUF must be rejected");
        assert!(result.unwrap_err().to_string().contains("wrong magic bytes"));
    }

    #[test]
    fn test_empty_file_rejected() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        std::fs::write(tmp.path(), b"").expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_err(), "empty file must be rejected");
    }

    #[test]
    fn test_is_valid_model_file_public_api() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(&APR_MAGIC);
        data.extend_from_slice(&[0u8; 100]);
        std::fs::write(tmp.path(), &data).expect("write");

        assert!(!is_valid_model_file(tmp.path()), "invalid APR should return false");
    }
}
