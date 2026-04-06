//! Research verify subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::VerifyArgs;
use crate::research::{PreRegistration, SignedPreRegistration, TimestampProof};

/// Result of signature verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureStatus {
    /// Signature is valid
    Valid,
    /// Signature is invalid
    Invalid,
    /// Verification error occurred
    Error(String),
}

/// Verify signature on a signed pre-registration
pub fn verify_signature(signed: &SignedPreRegistration) -> SignatureStatus {
    match signed.verify() {
        Ok(true) => SignatureStatus::Valid,
        Ok(false) => SignatureStatus::Invalid,
        Err(e) => SignatureStatus::Error(e.to_string()),
    }
}

/// Log git timestamp proof information
pub fn log_git_proof(level: LogLevel, proof: &TimestampProof) {
    if proof.is_git() {
        log(level, LogLevel::Normal, "Git timestamp proof: GitCommit");
        if let Some(commit) = proof.git_commit() {
            log(level, LogLevel::Verbose, &format!("  Commit: {commit}"));
        }
    } else {
        log(level, LogLevel::Normal, "Timestamp proof is not a git commit");
    }
}

/// Verify signed pre-registration content
pub fn verify_signed_content(
    signed: &SignedPreRegistration,
    verify_git: bool,
    level: LogLevel,
) -> Result<(), String> {
    match verify_signature(signed) {
        SignatureStatus::Valid => {
            log(level, LogLevel::Normal, "Signature verification: VALID");
        }
        SignatureStatus::Invalid => {
            log(level, LogLevel::Normal, "Signature verification: INVALID");
            return Err("Signature verification failed".to_string());
        }
        SignatureStatus::Error(e) => {
            return Err(format!("Verification error: {e}"));
        }
    }

    if verify_git {
        if let Some(proof) = &signed.timestamp_proof {
            log_git_proof(level, proof);
        } else {
            log(level, LogLevel::Normal, "No git timestamp proof found");
        }
    }

    log(level, LogLevel::Normal, "Pre-registration verified successfully");
    Ok(())
}

/// Compute and log commitment from pre-registration
pub fn compute_commitment(prereg: &PreRegistration, level: LogLevel) {
    let commitment = prereg.commit();
    log(level, LogLevel::Normal, &format!("Computed commitment: {}...", &commitment.hash[..32]));
}

pub fn run_research_verify(args: VerifyArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Verifying: {}", args.file.display()));

    let content =
        std::fs::read_to_string(&args.file).map_err(|e| format!("Failed to read file: {e}"))?;

    if let Ok(signed) = serde_yaml::from_str::<SignedPreRegistration>(&content) {
        return verify_signed_content(&signed, args.verify_git, level);
    }

    log(level, LogLevel::Normal, "File does not contain a signed pre-registration");

    if let Some(original_path) = &args.original {
        let original_content = std::fs::read_to_string(original_path)
            .map_err(|e| format!("Failed to read original: {e}"))?;

        let prereg: PreRegistration = serde_yaml::from_str(&original_content)
            .map_err(|e| format!("Failed to parse original: {e}"))?;

        compute_commitment(&prereg, level);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_status_eq() {
        assert_eq!(SignatureStatus::Valid, SignatureStatus::Valid);
        assert_eq!(SignatureStatus::Invalid, SignatureStatus::Invalid);
        assert_ne!(SignatureStatus::Valid, SignatureStatus::Invalid);
    }

    #[test]
    fn test_signature_status_error() {
        let err = SignatureStatus::Error("test error".to_string());
        assert_eq!(err, SignatureStatus::Error("test error".to_string()));
        assert_ne!(err, SignatureStatus::Valid);
    }

    #[test]
    fn test_signature_status_debug() {
        let valid = SignatureStatus::Valid;
        assert!(format!("{valid:?}").contains("Valid"));
    }

    #[test]
    fn test_signature_status_clone() {
        let orig = SignatureStatus::Error("test".to_string());
        let cloned = orig.clone();
        assert_eq!(orig, cloned);
    }

    #[test]
    fn test_log_git_proof_with_git_commit() {
        let proof = TimestampProof::GitCommit("abc123def456".to_string());
        // Should not panic
        log_git_proof(LogLevel::Quiet, &proof);
        log_git_proof(LogLevel::Verbose, &proof);
    }

    #[test]
    fn test_log_git_proof_with_non_git() {
        let proof = TimestampProof::Rfc3161(vec![1, 2, 3]);
        // Should not panic
        log_git_proof(LogLevel::Quiet, &proof);
    }

    #[test]
    fn test_log_git_proof_opentimestamps() {
        let proof = TimestampProof::OpenTimestamps(vec![4, 5, 6]);
        // Should not panic, logs "not a git commit"
        log_git_proof(LogLevel::Normal, &proof);
    }

    #[test]
    fn test_compute_commitment() {
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        // Should not panic
        compute_commitment(&prereg, LogLevel::Quiet);
    }

    #[test]
    fn test_verify_signature_with_valid_signed() {
        use ed25519_dalek::SigningKey;
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let signing_key = SigningKey::from_bytes(&[1u8; 32]);
        let signed = SignedPreRegistration::sign(&prereg, &signing_key);
        assert_eq!(verify_signature(&signed), SignatureStatus::Valid);
    }

    #[test]
    fn test_verify_signature_with_invalid_signature() {
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let commitment = prereg.commit();
        // Create signed with invalid signature
        let signed = SignedPreRegistration {
            registration: prereg,
            commitment,
            signature: "0".repeat(128), // Invalid but properly formatted signature
            public_key: "0".repeat(64), // Invalid but properly formatted public key
            timestamp_proof: None,
        };
        // This should be Invalid or Error
        let status = verify_signature(&signed);
        assert!(matches!(status, SignatureStatus::Invalid | SignatureStatus::Error(_)));
    }

    #[test]
    fn test_verify_signature_with_malformed_key() {
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let commitment = prereg.commit();
        // Create signed with malformed key (not valid hex)
        let signed = SignedPreRegistration {
            registration: prereg,
            commitment,
            signature: "not-hex".to_string(),
            public_key: "also-not-hex".to_string(),
            timestamp_proof: None,
        };
        let status = verify_signature(&signed);
        assert!(matches!(status, SignatureStatus::Error(_)));
    }

    #[test]
    fn test_verify_signed_content_valid() {
        use ed25519_dalek::SigningKey;
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let signing_key = SigningKey::from_bytes(&[2u8; 32]);
        let signed = SignedPreRegistration::sign(&prereg, &signing_key);
        let result = verify_signed_content(&signed, false, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_signed_content_invalid() {
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let commitment = prereg.commit();
        let signed = SignedPreRegistration {
            registration: prereg,
            commitment,
            signature: "0".repeat(128),
            public_key: "0".repeat(64),
            timestamp_proof: None,
        };
        let result = verify_signed_content(&signed, false, LogLevel::Quiet);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_signed_content_with_git_proof() {
        use ed25519_dalek::SigningKey;
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let signing_key = SigningKey::from_bytes(&[3u8; 32]);
        let mut signed = SignedPreRegistration::sign(&prereg, &signing_key);
        signed.timestamp_proof = Some(TimestampProof::GitCommit("abc123".to_string()));
        let result = verify_signed_content(&signed, true, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_signed_content_no_git_proof() {
        use ed25519_dalek::SigningKey;
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let signing_key = SigningKey::from_bytes(&[4u8; 32]);
        let signed = SignedPreRegistration::sign(&prereg, &signing_key);
        // signed has no timestamp_proof by default
        let result = verify_signed_content(&signed, true, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_signed_content_error_propagation() {
        let prereg =
            PreRegistration::new("Test Title", "Test Hypothesis", "Test Methods", "Test Analysis");
        let commitment = prereg.commit();
        let signed = SignedPreRegistration {
            registration: prereg,
            commitment,
            signature: "invalid".to_string(),
            public_key: "invalid".to_string(),
            timestamp_proof: None,
        };
        let result = verify_signed_content(&signed, false, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("error"));
    }
}
