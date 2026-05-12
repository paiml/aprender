//! Cosign/sigstore artifact-layout + tamper-detection classifier for
//! `apr publish sign` (CRUX-A-17).
//!
//! Contract: `contracts/crux-A-17-v1.yaml`.
//!
//! Three algorithm-level necessary conditions live here, all pure:
//!
//! 1. `artifact_paths_for(manifest)` computes the well-known `.sig`,
//!    `.crt`, and `.bundle` sidecar paths. Without these three artifacts
//!    existing, `cosign verify-blob` cannot succeed. Full round-trip
//!    discharge of FALSIFY-CRUX-A-17-001 blocks on the external cosign
//!    binary invocation.
//!
//! 2. `verify_content_hash(blob, expected_sha256)` is the sha256 gate
//!    that MUST fail before any PKI check if the blob differs from
//!    the signed one. This proves fail-closed tamper detection at the
//!    hash layer — a necessary condition for FALSIFY-CRUX-A-17-002.
//!
//! 3. `bundle_contains_rekor_payload(bundle_json)` parses the bundle
//!    JSON and returns true iff a `Payload` or `rekorBundle`-shaped key
//!    is present — the inclusion-proof envelope. Necessary condition
//!    for FALSIFY-CRUX-A-17-003. Full rekor-signature verification
//!    blocks on live Rekor transparency-log access.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Trio of detached signing artifacts that `apr publish sign` must
/// produce alongside a signed manifest. Path conventions match cosign's
/// `sign-blob --output-signature/--output-certificate/--bundle` flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigningArtifacts {
    pub sig: PathBuf,
    pub cert: PathBuf,
    pub bundle: PathBuf,
}

impl SigningArtifacts {
    /// True iff all three sidecar files exist on disk. This is the
    /// post-condition a `verify-blob` caller needs before it can run.
    pub fn all_present(&self) -> bool {
        self.sig.is_file() && self.cert.is_file() && self.bundle.is_file()
    }
}

/// Compute the well-known sidecar paths for a given manifest.
///
/// Convention: for `foo.json`, produce `foo.json.sig`, `foo.json.crt`,
/// and `foo.json.bundle`. This keeps the three artifacts adjacent in
/// directory listings and matches the cosign CLI's default flag output.
pub fn artifact_paths_for(manifest: &Path) -> SigningArtifacts {
    let base = manifest.as_os_str().to_os_string();
    let mut sig = base.clone();
    sig.push(".sig");
    let mut cert = base.clone();
    cert.push(".crt");
    let mut bundle = base;
    bundle.push(".bundle");
    SigningArtifacts {
        sig: PathBuf::from(sig),
        cert: PathBuf::from(cert),
        bundle: PathBuf::from(bundle),
    }
}

/// Outcome of a content-hash pre-check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyOutcome {
    /// sha256 matches the recorded expectation — PKI check may proceed.
    ContentHashMatches,
    /// sha256 differs — verify MUST fail closed before any PKI work.
    ContentHashMismatch {
        got_hex: String,
        expected_hex: String,
    },
}

/// Compute the sha256 of a blob and compare against an expected digest.
///
/// This is the necessary-condition gate for FALSIFY-CRUX-A-17-002:
/// tamper detection fails closed. If the bytes differ by even one bit,
/// the hash differs, and the verify pipeline rejects the blob before
/// reaching any X.509 / Rekor / Fulcio work.
pub fn verify_content_hash(blob: &[u8], expected_sha256: &[u8; 32]) -> VerifyOutcome {
    let mut h = Sha256::new();
    h.update(blob);
    let got: [u8; 32] = h.finalize().into();
    if &got == expected_sha256 {
        VerifyOutcome::ContentHashMatches
    } else {
        VerifyOutcome::ContentHashMismatch {
            got_hex: to_hex(&got),
            expected_hex: to_hex(expected_sha256),
        }
    }
}

fn to_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Parse a cosign/rekor bundle JSON string and check whether it carries
/// an inclusion-proof-shaped payload.
///
/// Accepts either the modern cosign bundle (top-level `Payload` +
/// `SignedEntryTimestamp`) or the legacy rekor-bundle key. Does NOT
/// cryptographically verify the inclusion — that requires a live Rekor
/// client call. This classifier only proves the envelope is shaped
/// right, which is a necessary condition for FALSIFY-CRUX-A-17-003.
pub fn bundle_contains_rekor_payload(bundle_json: &str) -> bool {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(bundle_json) else {
        return false;
    };
    let Some(obj) = v.as_object() else {
        return false;
    };
    // Cosign ≥1.10: top-level "Payload" (Rekor SET envelope) plus "Base64Signature".
    if obj.contains_key("Payload") && obj.contains_key("Base64Signature") {
        return true;
    }
    // Legacy bundle shape: nested "rekorBundle".
    if obj.contains_key("rekorBundle") {
        return true;
    }
    // Tolerate case-insensitive fallback (some tooling produces "rekor_bundle").
    if obj.keys().any(|k| k.to_ascii_lowercase().contains("rekor")) {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Artifact path layout =====

    #[test]
    fn artifact_paths_use_dot_suffix_convention() {
        let p = Path::new("/var/artifacts/manifest.json");
        let a = artifact_paths_for(p);
        assert_eq!(a.sig, PathBuf::from("/var/artifacts/manifest.json.sig"));
        assert_eq!(a.cert, PathBuf::from("/var/artifacts/manifest.json.crt"));
        assert_eq!(
            a.bundle,
            PathBuf::from("/var/artifacts/manifest.json.bundle")
        );
    }

    #[test]
    fn artifact_paths_adjacent_to_manifest() {
        // All three sidecars MUST share a parent with the manifest so
        // `cp -r` of the manifest dir carries them along.
        let p = Path::new("/tmp/x/y/m.json");
        let a = artifact_paths_for(p);
        assert_eq!(a.sig.parent(), p.parent());
        assert_eq!(a.cert.parent(), p.parent());
        assert_eq!(a.bundle.parent(), p.parent());
    }

    #[test]
    fn artifact_paths_relative_manifest_is_fine() {
        let p = Path::new("manifest.json");
        let a = artifact_paths_for(p);
        assert_eq!(a.sig, PathBuf::from("manifest.json.sig"));
    }

    #[test]
    fn artifact_paths_are_deterministic() {
        let p = Path::new("/tmp/m.json");
        assert_eq!(artifact_paths_for(p), artifact_paths_for(p));
    }

    // ===== Content-hash tamper detection =====

    fn sha256_of(b: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b);
        h.finalize().into()
    }

    #[test]
    fn content_hash_matches_identical_blob() {
        let blob = b"aprender model manifest v1";
        let expected = sha256_of(blob);
        assert_eq!(
            verify_content_hash(blob, &expected),
            VerifyOutcome::ContentHashMatches
        );
    }

    #[test]
    fn content_hash_rejects_single_byte_flip() {
        // Necessary condition for FALSIFY-CRUX-A-17-002: any mutation
        // must fail the hash gate.
        let blob = b"aprender model manifest v1";
        let expected = sha256_of(blob);
        let mut tampered = blob.to_vec();
        tampered[0] ^= 0x01;
        let r = verify_content_hash(&tampered, &expected);
        assert!(
            matches!(r, VerifyOutcome::ContentHashMismatch { .. }),
            "must reject single-byte flip: {r:?}"
        );
    }

    #[test]
    fn content_hash_rejects_trailing_byte_append() {
        let blob = b"aprender model manifest v1";
        let expected = sha256_of(blob);
        let mut tampered = blob.to_vec();
        tampered.push(0x00);
        assert!(matches!(
            verify_content_hash(&tampered, &expected),
            VerifyOutcome::ContentHashMismatch { .. }
        ));
    }

    #[test]
    fn content_hash_rejects_truncation() {
        let blob = b"aprender model manifest v1";
        let expected = sha256_of(blob);
        let tampered = &blob[..blob.len() - 1];
        assert!(matches!(
            verify_content_hash(tampered, &expected),
            VerifyOutcome::ContentHashMismatch { .. }
        ));
    }

    #[test]
    fn content_hash_rejects_empty_against_nonempty() {
        let original = b"not empty";
        let expected = sha256_of(original);
        assert!(matches!(
            verify_content_hash(&[], &expected),
            VerifyOutcome::ContentHashMismatch { .. }
        ));
    }

    #[test]
    fn content_hash_mismatch_carries_hex_for_debug() {
        let blob = b"a";
        let expected = sha256_of(b"b");
        match verify_content_hash(blob, &expected) {
            VerifyOutcome::ContentHashMismatch {
                got_hex,
                expected_hex,
            } => {
                assert_eq!(got_hex.len(), 64);
                assert_eq!(expected_hex.len(), 64);
                assert_ne!(got_hex, expected_hex);
            }
            other => panic!("expected mismatch, got {other:?}"),
        }
    }

    // ===== Rekor bundle shape =====

    #[test]
    fn bundle_with_cosign_modern_shape_is_recognized() {
        let json = r#"{"Payload":"dGVzdA==","Base64Signature":"c2ln"}"#;
        assert!(bundle_contains_rekor_payload(json));
    }

    #[test]
    fn bundle_with_legacy_rekor_bundle_key_is_recognized() {
        let json = r#"{"rekorBundle": {"SignedEntryTimestamp": "abc"}}"#;
        assert!(bundle_contains_rekor_payload(json));
    }

    #[test]
    fn bundle_with_rekor_snake_case_key_is_recognized() {
        // Some wrappers emit rekor_bundle — we accept it as a courtesy.
        let json = r#"{"rekor_bundle": {"x": "y"}}"#;
        assert!(bundle_contains_rekor_payload(json));
    }

    #[test]
    fn bundle_without_rekor_or_payload_is_rejected() {
        let json = r#"{"foo": "bar"}"#;
        assert!(!bundle_contains_rekor_payload(json));
    }

    #[test]
    fn bundle_with_only_payload_but_no_signature_is_rejected() {
        // Payload alone is not enough — cosign modern bundle requires
        // both Payload and Base64Signature to constitute an envelope.
        let json = r#"{"Payload": "dGVzdA=="}"#;
        assert!(!bundle_contains_rekor_payload(json));
    }

    #[test]
    fn malformed_bundle_json_is_rejected() {
        assert!(!bundle_contains_rekor_payload("{not json"));
        assert!(!bundle_contains_rekor_payload(""));
        assert!(!bundle_contains_rekor_payload("null"));
        assert!(!bundle_contains_rekor_payload("[]"));
    }

    #[test]
    fn bundle_classifier_is_pure() {
        let json = r#"{"rekorBundle": {}}"#;
        let a = bundle_contains_rekor_payload(json);
        let b = bundle_contains_rekor_payload(json);
        assert_eq!(a, b);
    }

    // ===== Cross-concern =====

    #[test]
    fn all_present_false_when_paths_do_not_exist() {
        let a = artifact_paths_for(Path::new("/no/such/path/manifest.json"));
        assert!(!a.all_present());
    }
}
