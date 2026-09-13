//! Error types for apr-cli
//!
//! Toyota Way: Jidoka - Stop and highlight problems immediately.

use std::path::PathBuf;
use std::process::ExitCode;
use thiserror::Error;

/// Result type alias for CLI operations
pub type Result<T> = std::result::Result<T, CliError>;

/// REG-15 model admission (#2971, PMAT-1065), re-exported here (rather than
/// from `lib.rs`) so `tests/reg15_admission.rs` can reach it without
/// depending on `crate::commands::parity_admission` directly. `commands` is
/// a private module, but privacy in Rust is scoped to "the declaring module
/// and its descendants" — since `mod commands;` is declared at the crate
/// root, EVERY module in this crate (including this one) already has
/// visibility into it; only an explicit `pub use` from a `pub` module (this
/// one) makes it reachable from outside the crate.
pub mod parity_admission {
    pub use crate::commands::parity_admission::{
        admit, is_parity_gate_error, override_line, parse_gate_error, Admission, ParityVerdict,
    };
}

/// CLI error types
#[derive(Error, Debug)]
pub enum CliError {
    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// Not a file (e.g., directory)
    #[error("Not a file: {0}")]
    NotAFile(PathBuf),

    /// Invalid APR format
    #[error("Invalid APR format: {0}")]
    InvalidFormat(String),

    /// Malformed non-model input (a JSON report, a YAML config, a CSV…).
    ///
    /// Distinct from [`CliError::InvalidFormat`], whose Display hardcodes
    /// "Invalid APR format" and therefore sends a user hunting for a corrupt
    /// model file when what actually failed to parse was, say, a grad-norm
    /// telemetry history. Shares exit code 4 — the class of failure is the
    /// same ("the input could not be parsed"), only the artifact named is
    /// different, exactly as `FileNotFound`/`NotAFile` already share 3.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Aprender error
    #[error("Aprender error: {0}")]
    Aprender(String),

    /// Model loading failed (used with inference feature)
    #[error("Model load failed: {0}")]
    #[allow(dead_code)]
    ModelLoadFailed(String),

    /// Inference failed (used with inference feature)
    #[error("Inference failed: {0}")]
    #[allow(dead_code)]
    InferenceFailed(String),

    /// Feature disabled (used when optional features are not compiled)
    #[error("Feature not enabled: {0}")]
    #[allow(dead_code)]
    FeatureDisabled(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// HTTP 404 Not Found (GH-356: distinguish from other network errors)
    #[error("HTTP 404 Not Found: {0}")]
    HttpNotFound(String),

    /// An advertised option exists but its implementation does not.
    ///
    /// #2407: `apr trace --reference` printed a stub string and exited 0, so
    /// every caller — including the MCP wrapper — read "did nothing" as
    /// "succeeded". An unimplemented option must fail, not report success.
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// A forced GPU backend was refused because the load-time parity gate
    /// failed (REG-15, PMAT-1065, issue #2971).
    ///
    /// Distinct from [`CliError::FeatureDisabled`], which refuses a request a
    /// build *cannot* honour at all (no accelerator compiled in). This is a
    /// build that CAN honour the request but whose GPU forward pass computed
    /// a different function than the CPU one on this model — #2971's second
    /// half was that this case fell back to the CPU silently instead of
    /// refusing. A forced backend (`--gpu` / `--backend cuda`) must never be
    /// silently downgraded; see `crate::commands::parity_admission`.
    #[error("Parity gate failed: {0}")]
    #[allow(dead_code)]
    ParityFailed(String),
}

impl CliError {
    /// Get exit code for this error
    pub fn exit_code(&self) -> ExitCode {
        contract_pre_exit_code_semantics!();
        contract_pre_error_mapping!();
        contract_pre_exit_code_on_error!();
        ExitCode::from(self.exit_code_value())
    }

    /// The numeric exit code this error maps to.
    ///
    /// `exit_code()` returns a `std::process::ExitCode`, which is opaque — it
    /// has no accessor and no `PartialEq`, so a test cannot assert on it. This
    /// is the same mapping as a plain `u8` so the exit-code convention is
    /// testable in-process.
    pub fn exit_code_value(&self) -> u8 {
        match self {
            Self::FileNotFound(_) | Self::NotAFile(_) => 3,
            Self::InvalidFormat(_) | Self::InvalidInput(_) => 4,
            Self::Io(_) => 7,
            Self::ValidationFailed(_) => 5,
            Self::Aprender(_) => 1,
            Self::ModelLoadFailed(_) => 6,
            Self::InferenceFailed(_) => 8,
            Self::FeatureDisabled(_) => 9,
            Self::NetworkError(_) => 10,
            Self::HttpNotFound(_) => 11,
            // #2407: an advertised option whose implementation is a stub must
            // FAIL, not print something and exit 0. Distinct code so a caller
            // can tell "this build cannot do that" from "your input was wrong".
            Self::NotImplemented(_) => 12,
            // REG-15 (#2971, PMAT-1065): a forced GPU backend refused because
            // the load-time parity gate failed. Distinct from FeatureDisabled
            // (9): that build has no accelerator at all; this build has one,
            // but this model computes a different function on it.
            Self::ParityFailed(_) => 13,
        }
    }
}

impl From<aprender::error::AprenderError> for CliError {
    fn from(e: aprender::error::AprenderError) -> Self {
        Self::Aprender(e.to_string())
    }
}

/// Refuse to clobber an existing output artifact unless `--force` was given.
///
/// #2392 (dogfood 0.63.0, finding 4): `apr convert` and `apr quantize` already
/// refused with exit 5 and "Use --force to overwrite", but `apr export`,
/// `apr merge`, `apr shard` and `apr unshard` wrote straight over whatever was
/// at the output path and exited 0 — and none of them even *had* a `--force`
/// flag, so the guarded behaviour was unreachable. A 9-byte file handed to
/// `apr export -o precious.safetensors` came back as 9717255 bytes of model with
/// no warning. Every write-a-file command now routes its overwrite decision
/// through this one function so the policy cannot drift again.
///
/// # Errors
///
/// Returns [`CliError::ValidationFailed`] (exit 5) when `path` exists and
/// `force` is false.
pub fn refuse_overwrite(path: &std::path::Path, force: bool) -> std::result::Result<(), CliError> {
    if path.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            path.display()
        )));
    }
    Ok(())
}

/// GH-668: reject common system/temp directories. Only resolve dirs that are
/// plausible HuggingFace model checkpoints (not `/`, `/tmp`, `/home`, etc.).
fn is_root_level_dir(path: &std::path::Path) -> bool {
    path.components().count() <= 2
}

/// PMAT-314: the sharded SafeTensors index takes priority over individual
/// shard files, which only contain a subset of tensors and would fail the
/// architecture gate. Falls back to the common single-file names in priority
/// order.
fn find_named_candidate(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        return Some(index);
    }
    const CANDIDATES: [&str; 5] = [
        "model.safetensors",
        "model-00001-of-00001.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00001-of-00003.safetensors",
        "model-00001-of-00004.safetensors",
    ];
    CANDIDATES
        .iter()
        .map(|candidate| dir.join(candidate))
        .find(|p| p.is_file())
}

/// GH-668: skip temp files (`rosetta_temp.apr`, etc.) to avoid inspecting
/// stale artifacts left behind by a previous conversion.
fn is_rosetta_temp_file(path: &std::path::Path) -> bool {
    path.file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with("rosetta_temp"))
}

/// First non-temp file in `dir` with extension `ext`, if any.
fn find_first_file_with_ext(dir: &std::path::Path, ext: &str) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    entries.flatten().map(|entry| entry.path()).find(|p| {
        !is_rosetta_temp_file(p) && p.extension().is_some_and(|e| e == ext) && p.is_file()
    })
}

/// The directory branch of [`resolve_model_path`]: locate the model file a
/// HuggingFace-style checkpoint directory holds.
fn resolve_model_dir(path: &std::path::Path) -> std::result::Result<std::path::PathBuf, CliError> {
    if is_root_level_dir(path) {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    if let Some(p) = find_named_candidate(path) {
        return Ok(p);
    }
    if let Some(p) = find_first_file_with_ext(path, "gguf") {
        return Ok(p);
    }
    if let Some(p) = find_first_file_with_ext(path, "apr") {
        return Ok(p);
    }
    Err(CliError::ValidationFailed(format!(
        "Directory {} does not contain a model file (expected model.safetensors, *.gguf, or *.apr)",
        path.display()
    )))
}

/// Resolve a model path: if given a directory, look for common model files inside.
///
/// HuggingFace models are stored as directories containing `model.safetensors`,
/// `model-00001-of-NNNNN.safetensors`, or `*.gguf`. This function resolves such
/// directories to the actual model file, avoiding "Not a file" errors.
pub fn resolve_model_path(
    path: &std::path::Path,
) -> std::result::Result<std::path::PathBuf, CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        resolve_model_dir(path)
    } else {
        Err(CliError::NotAFile(path.to_path_buf()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ==================== Exit Code Tests ====================

    #[test]
    fn test_file_not_found_exit_code() {
        let err = CliError::FileNotFound(PathBuf::from("/test"));
        assert_eq!(err.exit_code(), ExitCode::from(3));
    }

    #[test]
    fn test_not_a_file_exit_code() {
        let err = CliError::NotAFile(PathBuf::from("/test"));
        assert_eq!(err.exit_code(), ExitCode::from(3));
    }

    #[test]
    fn test_invalid_format_exit_code() {
        let err = CliError::InvalidFormat("bad".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(4));
    }

    #[test]
    fn test_io_error_exit_code() {
        let err = CliError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        assert_eq!(err.exit_code(), ExitCode::from(7));
    }

    #[test]
    fn test_validation_failed_exit_code() {
        let err = CliError::ValidationFailed("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(5));
    }

    #[test]
    fn test_aprender_error_exit_code() {
        let err = CliError::Aprender("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(1));
    }

    #[test]
    fn test_model_load_failed_exit_code() {
        let err = CliError::ModelLoadFailed("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(6));
    }

    #[test]
    fn test_inference_failed_exit_code() {
        let err = CliError::InferenceFailed("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(8));
    }

    #[test]
    fn test_feature_disabled_exit_code() {
        let err = CliError::FeatureDisabled("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(9));
    }

    /// #2407: an unimplemented option must exit non-zero — a stub that exits
    /// 0 is read by every caller as "it worked".
    #[test]
    fn test_not_implemented_exit_code_is_nonzero() {
        let err = CliError::NotImplemented("comparison".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(12));
        assert_ne!(err.exit_code(), ExitCode::SUCCESS);
        assert_eq!(err.to_string(), "Not implemented: comparison");
    }

    #[test]
    fn test_network_error_exit_code() {
        let err = CliError::NetworkError("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(10));
    }

    #[test]
    fn test_http_not_found_exit_code() {
        let err = CliError::HttpNotFound("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(11));
    }

    /// REG-15 (#2971, PMAT-1065): a forced backend refused for a parity-gate
    /// failure gets its own exit code, distinct from `FeatureDisabled` (9).
    #[test]
    fn test_parity_failed_exit_code() {
        let err = CliError::ParityFailed("cosine=0.90 < 0.98".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(13));
        assert_eq!(err.exit_code_value(), 13);
        assert_ne!(
            err.exit_code_value(),
            CliError::FeatureDisabled("x".to_string()).exit_code_value()
        );
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_file_not_found_display() {
        let err = CliError::FileNotFound(PathBuf::from("/model.apr"));
        assert_eq!(err.to_string(), "File not found: /model.apr");
    }

    #[test]
    fn test_not_a_file_display() {
        let err = CliError::NotAFile(PathBuf::from("/dir"));
        assert_eq!(err.to_string(), "Not a file: /dir");
    }

    #[test]
    fn test_invalid_format_display() {
        let err = CliError::InvalidFormat("bad magic".to_string());
        assert_eq!(err.to_string(), "Invalid APR format: bad magic");
    }

    #[test]
    fn test_validation_failed_display() {
        let err = CliError::ValidationFailed("missing field".to_string());
        assert_eq!(err.to_string(), "Validation failed: missing field");
    }

    #[test]
    fn test_aprender_error_display() {
        let err = CliError::Aprender("internal".to_string());
        assert_eq!(err.to_string(), "Aprender error: internal");
    }

    #[test]
    fn test_model_load_failed_display() {
        let err = CliError::ModelLoadFailed("corrupt".to_string());
        assert_eq!(err.to_string(), "Model load failed: corrupt");
    }

    #[test]
    fn test_inference_failed_display() {
        let err = CliError::InferenceFailed("OOM".to_string());
        assert_eq!(err.to_string(), "Inference failed: OOM");
    }

    #[test]
    fn test_feature_disabled_display() {
        let err = CliError::FeatureDisabled("cuda".to_string());
        assert_eq!(err.to_string(), "Feature not enabled: cuda");
    }

    #[test]
    fn test_network_error_display() {
        let err = CliError::NetworkError("timeout".to_string());
        assert_eq!(err.to_string(), "Network error: timeout");
    }

    #[test]
    fn test_http_not_found_display() {
        let err = CliError::HttpNotFound("tokenizer.json".to_string());
        assert_eq!(err.to_string(), "HTTP 404 Not Found: tokenizer.json");
    }

    // ==================== Conversion Tests ====================

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let cli_err: CliError = io_err.into();
        assert!(cli_err.to_string().contains("file missing"));
        assert_eq!(cli_err.exit_code(), ExitCode::from(7));
    }

    #[test]
    fn test_debug_impl() {
        let err = CliError::FileNotFound(PathBuf::from("/test"));
        let debug = format!("{:?}", err);
        assert!(debug.contains("FileNotFound"));
    }

    // ==================== Result Type Alias ====================

    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.expect("value"), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(CliError::InvalidFormat("test".to_string()));
        assert!(result.is_err());
    }

    // ==================== Exit Code Uniqueness ====================

    #[test]
    fn test_all_exit_codes_are_distinct_per_category() {
        // Verify exit codes map to distinct categories
        let codes = vec![
            (
                CliError::FileNotFound(PathBuf::from("a")).exit_code(),
                "file",
            ),
            (
                CliError::InvalidFormat("a".to_string()).exit_code(),
                "format",
            ),
            (
                CliError::Io(std::io::Error::new(std::io::ErrorKind::Other, "")).exit_code(),
                "io",
            ),
            (
                CliError::ValidationFailed("a".to_string()).exit_code(),
                "validation",
            ),
            (CliError::Aprender("a".to_string()).exit_code(), "aprender"),
            (
                CliError::ModelLoadFailed("a".to_string()).exit_code(),
                "model_load",
            ),
            (
                CliError::InferenceFailed("a".to_string()).exit_code(),
                "inference",
            ),
            (
                CliError::FeatureDisabled("a".to_string()).exit_code(),
                "feature",
            ),
            (
                CliError::NetworkError("a".to_string()).exit_code(),
                "network",
            ),
            (
                CliError::HttpNotFound("a".to_string()).exit_code(),
                "http_not_found",
            ),
        ];
        // FileNotFound and NotAFile intentionally share exit code 3
        assert_eq!(codes[0].0, ExitCode::from(3));
    }

    // ==================== resolve_model_path Tests ====================

    #[test]
    fn test_resolve_model_path_nonexistent() {
        let result = resolve_model_path(std::path::Path::new("/nonexistent/path/model.gguf"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CliError::FileNotFound(_)));
    }

    #[test]
    fn test_resolve_model_path_regular_file() {
        // Create a temp file and resolve it
        let tmp = std::env::temp_dir().join("apr-test-resolve.safetensors");
        std::fs::write(&tmp, b"test").expect("write");
        let result = resolve_model_path(&tmp);
        assert!(result.is_ok());
        assert_eq!(result.expect("value"), tmp);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_resolve_model_path_dir_with_safetensors() {
        let dir = std::env::temp_dir().join("apr-test-resolve-dir");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let model_file = dir.join("model.safetensors");
        std::fs::write(&model_file, b"test").expect("write");
        let result = resolve_model_path(&dir);
        assert!(result.is_ok());
        assert_eq!(result.expect("value"), model_file);
        std::fs::remove_file(&model_file).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_resolve_model_path_dir_with_gguf() {
        let dir = std::env::temp_dir().join("apr-test-resolve-gguf");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let model_file = dir.join("model-q4.gguf");
        std::fs::write(&model_file, b"test").expect("write");
        let result = resolve_model_path(&dir);
        assert!(result.is_ok());
        assert_eq!(result.expect("value"), model_file);
        std::fs::remove_file(&model_file).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_resolve_model_path_dir_with_sharded_safetensors() {
        // PMAT-314: Sharded models have index.json that MUST take priority
        // over individual shard files (model-00001-of-00002.safetensors)
        let dir = std::env::temp_dir().join("apr-test-resolve-sharded");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let index_file = dir.join("model.safetensors.index.json");
        let shard_file = dir.join("model-00001-of-00002.safetensors");
        std::fs::write(&index_file, b"{}").expect("write index");
        std::fs::write(&shard_file, b"test").expect("write shard");
        let result = resolve_model_path(&dir);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("value"),
            index_file,
            "index.json must take priority over shard files"
        );
        std::fs::remove_file(&shard_file).ok();
        std::fs::remove_file(&index_file).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_resolve_model_path_empty_dir() {
        let dir = std::env::temp_dir().join("apr-test-resolve-empty");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let result = resolve_model_path(&dir);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CliError::ValidationFailed(_)));
        std::fs::remove_dir(&dir).ok();
    }
}
