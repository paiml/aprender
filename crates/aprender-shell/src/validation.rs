//! Input validation for aprender-shell
//!
//! Follows Toyota Way principle *Poka-yoke* (Error-proofing):
//! Design systems that prevent mistakes.

use crate::error::ShellError;
use crate::model::MarkovModel;
use std::path::Path;

/// Sanitize and validate command prefix input.
///
/// Removes dangerous characters and validates input meets minimum requirements.
///
/// # Arguments
/// * `input` - Raw user input from shell
///
/// # Returns
/// * `Ok(String)` - Sanitized, valid prefix
/// * `Err(ShellError)` - If input is invalid
///
/// # Example
/// ```
/// use aprender_shell::validation::sanitize_prefix;
///
/// assert!(sanitize_prefix("").is_err());
/// assert!(sanitize_prefix("git status").is_ok());
/// assert_eq!(sanitize_prefix("git \0status").expect("valid after sanitize"), "git status");
/// ```
pub fn sanitize_prefix(input: &str) -> Result<String, ShellError> {
    // Remove null bytes (security)
    let sanitized = input.replace('\0', "");

    // Trim whitespace
    let trimmed = sanitized.trim();

    // Reject empty input
    if trimmed.is_empty() {
        return Err(ShellError::InvalidInput {
            message: "Empty prefix".into(),
        });
    }

    // Reject if too short (< 2 chars for meaningful suggestions)
    if trimmed.len() < 2 {
        return Err(ShellError::InvalidInput {
            message: "Prefix too short (minimum 2 characters)".into(),
        });
    }

    // Reject control characters (except tab which is common in shell)
    if trimmed.chars().any(|c| c.is_control() && c != '\t') {
        return Err(ShellError::InvalidInput {
            message: "Invalid control characters in input".into(),
        });
    }

    Ok(trimmed.to_string())
}

/// Load model with graceful error handling.
///
/// Instead of panicking on errors, returns a descriptive ShellError
/// with hints for resolution.
///
/// # Arguments
/// * `path` - Path to the model file
///
/// # Returns
/// * `Ok(MarkovModel)` - Successfully loaded model
/// * `Err(ShellError)` - Descriptive error with hints
pub fn load_model_graceful(path: &Path) -> Result<MarkovModel, ShellError> {
    // Check if file exists first
    if !path.exists() {
        return Err(ShellError::ModelNotFound {
            path: path.to_path_buf(),
            hint: "Run 'aprender-shell train' to create a model".into(),
        });
    }

    // Try to load the model
    match MarkovModel::load(path) {
        Ok(model) => Ok(model),
        Err(e) => {
            let msg = e.to_string();

            // Detect specific error types
            if msg.contains("Checksum")
                || msg.contains("checksum")
                || msg.contains("invalid")
                || msg.contains("corrupt")
            {
                Err(ShellError::ModelCorrupted {
                    path: path.to_path_buf(),
                    hint: "Model file is corrupted. Run 'aprender-shell train' to rebuild".into(),
                })
            } else {
                Err(ShellError::ModelLoadFailed {
                    path: path.to_path_buf(),
                    cause: msg,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // =========================================================================
    // Input Validation Tests (sanitize_prefix)
    // =========================================================================

    #[test]
    fn test_empty_input_rejected() {
        assert!(sanitize_prefix("").is_err());
        let err = sanitize_prefix("").unwrap_err();
        assert!(matches!(err, ShellError::InvalidInput { .. }));
    }

    #[test]
    fn test_whitespace_only_rejected() {
        assert!(sanitize_prefix("   ").is_err());
        assert!(sanitize_prefix("\t\n").is_err());
        assert!(sanitize_prefix("  \t  ").is_err());
    }

    #[test]
    fn test_null_bytes_removed() {
        let result = sanitize_prefix("git \0status").expect("should succeed");
        assert_eq!(result, "git status");
        assert!(!result.contains('\0'));
    }

    #[test]
    fn test_short_input_rejected() {
        assert!(sanitize_prefix("g").is_err());
        let err = sanitize_prefix("a").unwrap_err();
        assert!(matches!(err, ShellError::InvalidInput { .. }));
    }

    #[test]
    fn test_two_char_input_accepted() {
        assert!(sanitize_prefix("gi").is_ok());
        assert!(sanitize_prefix("ls").is_ok());
        assert!(sanitize_prefix("cd").is_ok());
    }

    #[test]
    fn test_double_dash_accepted() {
        assert!(sanitize_prefix("--help").is_ok());
        assert!(sanitize_prefix("-- filename").is_ok());
        assert!(sanitize_prefix("git checkout --").is_ok());
    }

    #[test]
    fn test_control_chars_rejected() {
        // Bell character
        assert!(sanitize_prefix("git\x07status").is_err());
        // Escape character
        assert!(sanitize_prefix("git\x1bstatus").is_err());
    }

    #[test]
    fn test_tab_allowed() {
        // Tab is allowed as it's common in shell
        assert!(sanitize_prefix("git\tstatus").is_ok());
    }

    #[test]
    fn test_valid_commands_accepted() {
        assert!(sanitize_prefix("git status").is_ok());
        assert!(sanitize_prefix("cargo build --release").is_ok());
        assert!(sanitize_prefix("docker ps -a").is_ok());
        assert!(sanitize_prefix("kubectl get pods").is_ok());
    }

    #[test]
    fn test_whitespace_trimmed() {
        let result = sanitize_prefix("  git status  ").expect("should succeed");
        assert_eq!(result, "git status");
    }

    // =========================================================================
    // Model Loading Tests (load_model_graceful)
    // =========================================================================

    #[test]
    fn test_missing_model_graceful_error() {
        let result = load_model_graceful(Path::new("/nonexistent/path/model.bin"));
        assert!(matches!(result, Err(ShellError::ModelNotFound { .. })));

        if let Err(ShellError::ModelNotFound { hint, .. }) = result {
            assert!(hint.contains("train"));
        }
    }

    #[test]
    fn test_corrupted_model_graceful_error() {
        // Create a temp file with garbage data
        let mut tmp = NamedTempFile::new().expect("create temp file");
        tmp.write_all(b"GARBAGE DATA NOT A MODEL")
            .expect("write garbage");
        tmp.flush().expect("flush");

        let result = load_model_graceful(tmp.path());

        // Should be either ModelCorrupted or ModelLoadFailed (not panic!)
        assert!(
            matches!(
                result,
                Err(ShellError::ModelCorrupted { .. }) | Err(ShellError::ModelLoadFailed { .. })
            ),
            "Expected graceful error for corrupted model"
        );
    }

    #[test]
    fn test_valid_model_loads_successfully() {
        // Create a valid model
        let mut model = MarkovModel::new(3);
        model.train(&["git status".to_string(), "git commit".to_string()]);

        let tmp = NamedTempFile::new().expect("create temp file");
        model.save(tmp.path()).expect("save model");

        let result = load_model_graceful(tmp.path());
        assert!(result.is_ok(), "Valid model should load successfully");
    }

    #[test]
    fn test_empty_file_graceful_error() {
        // Create an empty file
        let tmp = NamedTempFile::new().expect("create temp file");

        let result = load_model_graceful(tmp.path());

        // Should not panic, should return an error
        assert!(result.is_err(), "Empty file should fail gracefully");
    }
}
