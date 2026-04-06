//! Ed25519 manifest signing and verification.
//!
//! Computes BLAKE3 hash of manifest TOML content, signs with
//! pacha Ed25519 key. Verification ensures manifest integrity
//! before agent execution (Jidoka: stop on tampered manifest).
//!
//! Phase 3: Feature-gated under `native` (requires pacha + blake3).

/// Manifest signature containing Ed25519 signature and BLAKE3 hash.
#[derive(Debug, Clone)]
pub struct ManifestSignature {
    /// BLAKE3 hash of the manifest content (hex).
    pub content_hash: String,
    /// Ed25519 signature over the content hash (hex).
    pub signature_hex: String,
    /// Signer identity (optional label).
    pub signer: Option<String>,
}

/// Sign a manifest TOML string with Ed25519.
///
/// Computes BLAKE3 hash of the content, then signs the hash
/// with the provided pacha signing key.
#[cfg(feature = "native")]
pub fn sign_manifest(
    manifest_content: &str,
    signing_key: &pacha::signing::SigningKey,
    signer: Option<&str>,
) -> ManifestSignature {
    let content_hash = blake3::hash(manifest_content.as_bytes());
    let hash_hex = content_hash.to_hex().to_string();

    let signature = signing_key.sign(hash_hex.as_bytes());

    ManifestSignature {
        content_hash: hash_hex,
        signature_hex: signature.to_hex(),
        signer: signer.map(String::from),
    }
}

/// Verify a manifest signature against content and public key.
///
/// Returns Ok(()) if valid, or an error describing the failure.
#[cfg(feature = "native")]
pub fn verify_manifest(
    manifest_content: &str,
    signature: &ManifestSignature,
    verifying_key: &pacha::signing::VerifyingKey,
) -> Result<(), ManifestVerifyError> {
    // Recompute content hash
    let content_hash = blake3::hash(manifest_content.as_bytes());
    let hash_hex = content_hash.to_hex().to_string();

    // Check hash matches
    if hash_hex != signature.content_hash {
        return Err(ManifestVerifyError::HashMismatch {
            expected: signature.content_hash.clone(),
            actual: hash_hex,
        });
    }

    // Decode signature from hex
    let sig = pacha::signing::Signature::from_hex(&signature.signature_hex)
        .map_err(|e| ManifestVerifyError::InvalidSignature(format!("{e}")))?;

    // Verify Ed25519 signature over the hash
    verifying_key
        .verify(hash_hex.as_bytes(), &sig)
        .map_err(|e| ManifestVerifyError::SignatureFailed(format!("{e}")))
}

/// Errors from manifest signature verification.
#[derive(Debug, Clone)]
pub enum ManifestVerifyError {
    /// Content hash doesn't match (manifest was modified).
    HashMismatch {
        /// Hash from the signature file.
        expected: String,
        /// Hash of the actual content.
        actual: String,
    },
    /// Signature bytes are malformed.
    InvalidSignature(String),
    /// Ed25519 signature doesn't match the content+key.
    SignatureFailed(String),
}

impl std::fmt::Display for ManifestVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HashMismatch { expected, actual } => {
                write!(
                    f,
                    "content hash mismatch: expected \
                     {expected}, got {actual}"
                )
            }
            Self::InvalidSignature(msg) => {
                write!(f, "invalid signature: {msg}")
            }
            Self::SignatureFailed(msg) => {
                write!(f, "signature verification failed: {msg}")
            }
        }
    }
}

/// Serialize a manifest signature to TOML sidecar format.
pub fn signature_to_toml(sig: &ManifestSignature) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    out.push_str("[signature]\n");
    let _ = writeln!(out, "content_hash = \"{}\"", sig.content_hash);
    let _ = writeln!(out, "signature = \"{}\"", sig.signature_hex);
    if let Some(ref signer) = sig.signer {
        let _ = writeln!(out, "signer = \"{signer}\"");
    }
    out
}

/// Parse a manifest signature from TOML sidecar content.
pub fn signature_from_toml(toml_str: &str) -> Result<ManifestSignature, String> {
    let table: toml::Value = toml::from_str(toml_str).map_err(|e| format!("TOML parse: {e}"))?;

    let sig = table.get("signature").ok_or("missing [signature] section")?;

    let content_hash =
        sig.get("content_hash").and_then(|v| v.as_str()).ok_or("missing content_hash")?.to_string();

    let signature_hex =
        sig.get("signature").and_then(|v| v.as_str()).ok_or("missing signature")?.to_string();

    let signer = sig.get("signer").and_then(|v| v.as_str()).map(String::from);

    Ok(ManifestSignature { content_hash, signature_hex, signer })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MANIFEST: &str = r#"
name = "test-agent"
version = "0.1.0"

[model]
max_tokens = 4096

[resources]
max_iterations = 20
"#;

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = blake3::hash(TEST_MANIFEST.as_bytes());
        let h2 = blake3::hash(TEST_MANIFEST.as_bytes());
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_changes_on_modification() {
        let h1 = blake3::hash(TEST_MANIFEST.as_bytes());
        let modified = TEST_MANIFEST.replace("20", "30");
        let h2 = blake3::hash(modified.as_bytes());
        assert_ne!(h1, h2);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_sign_and_verify_roundtrip() {
        let key = pacha::signing::SigningKey::generate();
        let vk = key.verifying_key();

        let sig = sign_manifest(TEST_MANIFEST, &key, Some("test"));
        assert!(!sig.content_hash.is_empty());
        assert!(!sig.signature_hex.is_empty());
        assert_eq!(sig.signer, Some("test".into()));

        let result = verify_manifest(TEST_MANIFEST, &sig, &vk);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_verify_fails_on_tampered_content() {
        let key = pacha::signing::SigningKey::generate();
        let vk = key.verifying_key();

        let sig = sign_manifest(TEST_MANIFEST, &key, None);
        let tampered = TEST_MANIFEST.replace("20", "999");

        let result = verify_manifest(&tampered, &sig, &vk);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifestVerifyError::HashMismatch { .. } => {}
            other => {
                panic!("expected HashMismatch, got: {other}")
            }
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_verify_fails_with_wrong_key() {
        let key1 = pacha::signing::SigningKey::generate();
        let key2 = pacha::signing::SigningKey::generate();
        let vk2 = key2.verifying_key();

        let sig = sign_manifest(TEST_MANIFEST, &key1, None);

        let result = verify_manifest(TEST_MANIFEST, &sig, &vk2);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifestVerifyError::SignatureFailed(_) => {}
            other => {
                panic!("expected SignatureFailed, got: {other}")
            }
        }
    }

    #[test]
    fn test_signature_toml_roundtrip() {
        let sig = ManifestSignature {
            content_hash: "abc123".into(),
            signature_hex: "def456".into(),
            signer: Some("alice".into()),
        };

        let toml_str = signature_to_toml(&sig);
        assert!(toml_str.contains("abc123"));
        assert!(toml_str.contains("def456"));

        let parsed = signature_from_toml(&toml_str).expect("parse");
        assert_eq!(parsed.content_hash, "abc123");
        assert_eq!(parsed.signature_hex, "def456");
        assert_eq!(parsed.signer, Some("alice".into()));
    }

    #[test]
    fn test_signature_toml_no_signer() {
        let sig = ManifestSignature {
            content_hash: "hash".into(),
            signature_hex: "sig".into(),
            signer: None,
        };

        let toml_str = signature_to_toml(&sig);
        assert!(!toml_str.contains("signer"));

        let parsed = signature_from_toml(&toml_str).expect("parse");
        assert!(parsed.signer.is_none());
    }

    #[test]
    fn test_verify_error_display() {
        let err = ManifestVerifyError::HashMismatch { expected: "a".into(), actual: "b".into() };
        assert!(format!("{err}").contains("mismatch"));

        let err = ManifestVerifyError::InvalidSignature("bad".into());
        assert!(format!("{err}").contains("bad"));

        let err = ManifestVerifyError::SignatureFailed("nope".into());
        assert!(format!("{err}").contains("nope"));
    }

    #[test]
    fn test_signature_from_toml_malformed() {
        let result = signature_from_toml("not valid toml {{");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("TOML parse"));
    }

    #[test]
    fn test_signature_from_toml_missing_section() {
        let result = signature_from_toml("[other]\nkey = 1\n");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing [signature]"));
    }

    #[test]
    fn test_signature_from_toml_missing_content_hash() {
        let toml = r#"
[signature]
signature = "abc"
"#;
        let result = signature_from_toml(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing content_hash"));
    }

    #[test]
    fn test_signature_from_toml_missing_signature() {
        let toml = r#"
[signature]
content_hash = "abc"
"#;
        let result = signature_from_toml(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing signature"));
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_verify_invalid_signature_hex() {
        let key = pacha::signing::SigningKey::generate();
        let vk = key.verifying_key();

        let sig = ManifestSignature {
            content_hash: blake3::hash(TEST_MANIFEST.as_bytes()).to_hex().to_string(),
            signature_hex: "not-valid-hex!!".into(),
            signer: None,
        };

        let result = verify_manifest(TEST_MANIFEST, &sig, &vk);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifestVerifyError::InvalidSignature(msg) => {
                assert!(!msg.is_empty());
            }
            other => {
                panic!("expected InvalidSignature, got: {other}")
            }
        }
    }

    #[test]
    fn test_signature_to_toml_content() {
        let sig = ManifestSignature {
            content_hash: "deadbeef".into(),
            signature_hex: "cafebabe".into(),
            signer: Some("bob".into()),
        };
        let toml = signature_to_toml(&sig);
        assert!(toml.starts_with("[signature]\n"));
        assert!(toml.contains(r#"content_hash = "deadbeef""#));
        assert!(toml.contains(r#"signature = "cafebabe""#));
        assert!(toml.contains(r#"signer = "bob""#));
    }
}
