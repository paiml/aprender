//! Research preregister subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::PreregisterArgs;
use crate::research::{PreRegistration, SignedPreRegistration, TimestampProof};

pub fn run_research_preregister(args: PreregisterArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Creating pre-registration: {}", args.title));

    let mut prereg =
        PreRegistration::new(&args.title, &args.hypothesis, &args.methodology, &args.analysis_plan);

    if let Some(notes) = &args.notes {
        prereg = prereg.with_notes(notes);
    }

    // Create commitment
    let commitment = prereg.commit();
    log(level, LogLevel::Verbose, &format!("  Commitment hash: {}...", &commitment.hash[..32]));

    // Sign if key provided
    let output = if let Some(key_path) = &args.sign_key {
        use ed25519_dalek::SigningKey;

        let key_bytes =
            std::fs::read(key_path).map_err(|e| format!("Failed to read signing key: {e}"))?;

        if key_bytes.len() != 32 {
            return Err("Signing key must be exactly 32 bytes".to_string());
        }

        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        let signing_key = SigningKey::from_bytes(&key_array);

        let mut signed = SignedPreRegistration::sign(&prereg, &signing_key);

        // Add git timestamp if requested
        if args.git_timestamp {
            let output = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .map_err(|e| format!("Failed to get git commit: {e}"))?;

            let commit_hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
            signed = signed.with_timestamp_proof(TimestampProof::git(&commit_hash));
            log(level, LogLevel::Verbose, &format!("  Git timestamp: {commit_hash}"));
        }

        serde_yaml::to_string(&signed).map_err(|e| format!("Serialization error: {e}"))?
    } else {
        // Output commitment without signature
        serde_yaml::to_string(&commitment).map_err(|e| format!("Serialization error: {e}"))?
    };

    std::fs::write(&args.output, &output).map_err(|e| format!("Failed to write file: {e}"))?;

    log(level, LogLevel::Normal, &format!("Pre-registration saved to: {}", args.output.display()));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_test_args(output: std::path::PathBuf) -> PreregisterArgs {
        PreregisterArgs {
            title: "Test Title".to_string(),
            hypothesis: "Test Hypothesis".to_string(),
            methodology: "Test Methodology".to_string(),
            analysis_plan: "Test Analysis Plan".to_string(),
            notes: None,
            output,
            sign_key: None,
            git_timestamp: false,
        }
    }

    #[test]
    fn test_preregister_basic() {
        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let args = make_test_args(output_file.path().to_path_buf());
        let result = run_research_preregister(args, LogLevel::Quiet);
        assert!(result.is_ok());
        // Verify output was created
        let content =
            std::fs::read_to_string(output_file.path()).expect("file read should succeed");
        assert!(content.contains("hash:"));
    }

    #[test]
    fn test_preregister_with_notes() {
        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let mut args = make_test_args(output_file.path().to_path_buf());
        args.notes = Some("Additional notes".to_string());
        let result = run_research_preregister(args, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_preregister_with_signing() {
        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let key_file = NamedTempFile::new().expect("temp file creation should succeed");

        // Write a valid 32-byte key
        std::fs::write(key_file.path(), [1u8; 32]).expect("file write should succeed");

        let mut args = make_test_args(output_file.path().to_path_buf());
        args.sign_key = Some(key_file.path().to_path_buf());

        let result = run_research_preregister(args, LogLevel::Quiet);
        assert!(result.is_ok());

        // Verify signed output
        let content =
            std::fs::read_to_string(output_file.path()).expect("file read should succeed");
        assert!(content.contains("signature:"));
        assert!(content.contains("public_key:"));
    }

    #[test]
    fn test_preregister_invalid_key_size() {
        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let key_file = NamedTempFile::new().expect("temp file creation should succeed");

        // Write invalid key size (not 32 bytes)
        std::fs::write(key_file.path(), [1u8; 16]).expect("file write should succeed");

        let mut args = make_test_args(output_file.path().to_path_buf());
        args.sign_key = Some(key_file.path().to_path_buf());

        let result = run_research_preregister(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("32 bytes"));
    }

    #[test]
    fn test_preregister_missing_key_file() {
        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let mut args = make_test_args(output_file.path().to_path_buf());
        args.sign_key = Some(std::path::PathBuf::from("/nonexistent/key/file"));

        let result = run_research_preregister(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read signing key"));
    }

    #[test]
    fn test_preregister_invalid_output_path() {
        let args = make_test_args(std::path::PathBuf::from("/nonexistent/dir/output.yaml"));
        let result = run_research_preregister(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to write file"));
    }

    #[test]
    fn test_preregister_with_git_timestamp() {
        // Skip if not in a git repo
        let in_git = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if !in_git {
            return;
        }

        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let key_file = NamedTempFile::new().expect("temp file creation should succeed");
        std::fs::write(key_file.path(), [2u8; 32]).expect("file write should succeed");

        let mut args = make_test_args(output_file.path().to_path_buf());
        args.sign_key = Some(key_file.path().to_path_buf());
        args.git_timestamp = true;

        let result = run_research_preregister(args, LogLevel::Quiet);
        assert!(result.is_ok());

        let content =
            std::fs::read_to_string(output_file.path()).expect("file read should succeed");
        assert!(content.contains("timestamp_proof:") || content.contains("GitCommit"));
    }

    #[test]
    fn test_preregister_verbose_logging() {
        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let args = make_test_args(output_file.path().to_path_buf());
        // Just ensure it doesn't panic with verbose logging
        let result = run_research_preregister(args, LogLevel::Verbose);
        assert!(result.is_ok());
    }

    #[test]
    fn test_preregister_with_signing_verbose() {
        let output_file = NamedTempFile::new().expect("temp file creation should succeed");
        let key_file = NamedTempFile::new().expect("temp file creation should succeed");
        std::fs::write(key_file.path(), [3u8; 32]).expect("file write should succeed");

        let mut args = make_test_args(output_file.path().to_path_buf());
        args.sign_key = Some(key_file.path().to_path_buf());

        let result = run_research_preregister(args, LogLevel::Verbose);
        assert!(result.is_ok());
    }
}
