//! Pre-Registration with hash commitment (ENT-022)
//!
//! Provides cryptographic pre-registration for research protocols
//! with hash commitments and Ed25519 signing.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Pre-registration protocol for research studies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreRegistration {
    /// Protocol title
    pub title: String,
    /// Research hypothesis
    pub hypothesis: String,
    /// Methodology description
    pub methods: String,
    /// Statistical analysis plan
    pub analysis_plan: String,
    /// Additional notes
    pub notes: Option<String>,
}

impl PreRegistration {
    /// Create a new pre-registration
    pub fn new(
        title: impl Into<String>,
        hypothesis: impl Into<String>,
        methods: impl Into<String>,
        analysis_plan: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            hypothesis: hypothesis.into(),
            methods: methods.into(),
            analysis_plan: analysis_plan.into(),
            notes: None,
        }
    }

    /// Add notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Create a cryptographic commitment (SHA-256 hash)
    pub fn commit(&self) -> PreRegistrationCommitment {
        let serialized =
            serde_json::to_string(self).expect("PreRegistration should always serialize");
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let hash = hex::encode(hasher.finalize());

        PreRegistrationCommitment { hash, created_at: chrono::Utc::now() }
    }

    /// Verify a commitment matches this pre-registration
    pub fn verify_commitment(&self, commitment: &PreRegistrationCommitment) -> bool {
        let current = self.commit();
        current.hash == commitment.hash
    }

    /// Reveal and verify against a commitment
    pub fn reveal(
        &self,
        commitment: &PreRegistrationCommitment,
    ) -> Result<PreRegistrationReveal, PreRegistrationError> {
        if !self.verify_commitment(commitment) {
            return Err(PreRegistrationError::HashMismatch);
        }

        Ok(PreRegistrationReveal {
            protocol: self.clone(),
            commitment: commitment.clone(),
            revealed_at: chrono::Utc::now(),
        })
    }
}

/// Cryptographic commitment to a pre-registration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreRegistrationCommitment {
    /// SHA-256 hash of the serialized pre-registration
    pub hash: String,
    /// Timestamp when commitment was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl PreRegistrationCommitment {
    /// Get the hash as bytes
    pub fn hash_bytes(&self) -> Vec<u8> {
        hex::decode(&self.hash).unwrap_or_default()
    }
}

/// Revealed pre-registration with verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRegistrationReveal {
    /// The original protocol
    pub protocol: PreRegistration,
    /// The commitment that was verified
    pub commitment: PreRegistrationCommitment,
    /// Timestamp when revealed
    pub revealed_at: chrono::DateTime<chrono::Utc>,
}

/// Timestamp proof types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimestampProof {
    /// Git commit hash
    GitCommit(String),
    /// RFC 3161 timestamp token
    Rfc3161(Vec<u8>),
    /// OpenTimestamps proof
    OpenTimestamps(Vec<u8>),
}

impl TimestampProof {
    /// Create a git commit proof
    pub fn git(commit_hash: impl Into<String>) -> Self {
        Self::GitCommit(commit_hash.into())
    }

    /// Check if this is a git commit proof
    pub fn is_git(&self) -> bool {
        matches!(self, Self::GitCommit(_))
    }

    /// Get git commit hash if this is a git proof
    pub fn git_commit(&self) -> Option<&str> {
        match self {
            Self::GitCommit(hash) => Some(hash),
            _ => None,
        }
    }
}

/// Pre-registered protocol with Ed25519 signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedPreRegistration {
    /// The pre-registration
    pub registration: PreRegistration,
    /// The commitment hash
    pub commitment: PreRegistrationCommitment,
    /// Ed25519 signature (hex-encoded)
    pub signature: String,
    /// Public key (hex-encoded)
    pub public_key: String,
    /// Optional timestamp proof
    pub timestamp_proof: Option<TimestampProof>,
}

impl SignedPreRegistration {
    /// Sign a pre-registration with an Ed25519 key
    pub fn sign(registration: &PreRegistration, signing_key: &SigningKey) -> Self {
        let commitment = registration.commit();
        let signature = signing_key.sign(commitment.hash.as_bytes());
        let public_key = signing_key.verifying_key();

        Self {
            registration: registration.clone(),
            commitment,
            signature: hex::encode(signature.to_bytes()),
            public_key: hex::encode(public_key.to_bytes()),
            timestamp_proof: None,
        }
    }

    /// Add a timestamp proof
    pub fn with_timestamp_proof(mut self, proof: TimestampProof) -> Self {
        self.timestamp_proof = Some(proof);
        self
    }

    /// Verify the signature
    pub fn verify(&self) -> Result<bool, PreRegistrationError> {
        // Decode public key
        let pk_bytes =
            hex::decode(&self.public_key).map_err(|_err| PreRegistrationError::InvalidPublicKey)?;
        let pk_array: [u8; 32] =
            pk_bytes.try_into().map_err(|_err| PreRegistrationError::InvalidPublicKey)?;
        let public_key = VerifyingKey::from_bytes(&pk_array)
            .map_err(|_err| PreRegistrationError::InvalidPublicKey)?;

        // Decode signature
        let sig_bytes =
            hex::decode(&self.signature).map_err(|_err| PreRegistrationError::InvalidSignature)?;
        let sig_array: [u8; 64] =
            sig_bytes.try_into().map_err(|_err| PreRegistrationError::InvalidSignature)?;
        let signature = Signature::from_bytes(&sig_array);

        // Verify signature
        Ok(public_key.verify(self.commitment.hash.as_bytes(), &signature).is_ok())
    }

    /// Verify both signature and that registration matches commitment
    pub fn verify_full(&self) -> Result<bool, PreRegistrationError> {
        // First verify the signature
        if !self.verify()? {
            return Ok(false);
        }

        // Then verify the registration matches the commitment
        Ok(self.registration.verify_commitment(&self.commitment))
    }
}

/// Pre-registration errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum PreRegistrationError {
    #[error("Hash mismatch: pre-registration does not match commitment")]
    HashMismatch,
    #[error("Invalid public key")]
    InvalidPublicKey,
    #[error("Invalid signature")]
    InvalidSignature,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_registration() -> PreRegistration {
        PreRegistration::new(
            "Effect of Treatment A on Outcome B",
            "Treatment A will improve Outcome B by at least 20%",
            "Randomized controlled trial with 100 participants",
            "Two-sample t-test with alpha=0.05",
        )
    }

    #[test]
    fn test_commit_creates_hash() {
        let reg = create_test_registration();
        let commitment = reg.commit();

        assert!(!commitment.hash.is_empty());
        assert_eq!(commitment.hash.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }

    #[test]
    fn test_commit_is_deterministic() {
        let reg = create_test_registration();
        let commitment1 = reg.commit();
        let commitment2 = reg.commit();

        assert_eq!(commitment1.hash, commitment2.hash);
    }

    #[test]
    fn test_reveal_verifies_hash() {
        let reg = create_test_registration();
        let commitment = reg.commit();
        let reveal = reg.reveal(&commitment);

        assert!(reveal.is_ok());
        let reveal = reveal.expect("operation should succeed");
        assert_eq!(reveal.protocol, reg);
    }

    #[test]
    fn test_reveal_fails_on_tamper() {
        let reg = create_test_registration();
        let commitment = reg.commit();

        // Tamper with the registration
        let tampered = PreRegistration::new(
            "Effect of Treatment A on Outcome B",
            "Treatment A will improve Outcome B by at least 50%", // Changed!
            "Randomized controlled trial with 100 participants",
            "Two-sample t-test with alpha=0.05",
        );

        let result = tampered.reveal(&commitment);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PreRegistrationError::HashMismatch));
    }

    #[test]
    fn test_timestamp_proof_git() {
        let proof = TimestampProof::git("abc123def456");

        assert!(proof.is_git());
        assert_eq!(proof.git_commit(), Some("abc123def456"));
    }

    #[test]
    fn test_timestamp_proof_rfc3161() {
        let proof = TimestampProof::Rfc3161(vec![1, 2, 3, 4]);

        assert!(!proof.is_git());
        assert_eq!(proof.git_commit(), None);
    }

    #[test]
    fn test_ed25519_signing() {
        let reg = create_test_registration();

        // Generate a signing key
        let signing_key = SigningKey::from_bytes(&[1u8; 32]);

        // Sign the pre-registration
        let signed = SignedPreRegistration::sign(&reg, &signing_key);

        assert!(!signed.signature.is_empty());
        assert!(!signed.public_key.is_empty());
        assert_eq!(signed.signature.len(), 128); // Ed25519 sig = 64 bytes = 128 hex
        assert_eq!(signed.public_key.len(), 64); // Ed25519 pk = 32 bytes = 64 hex
    }

    #[test]
    fn test_ed25519_verification() {
        let reg = create_test_registration();
        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let signed = SignedPreRegistration::sign(&reg, &signing_key);

        assert!(signed.verify().expect("operation should succeed"));
        assert!(signed.verify_full().expect("operation should succeed"));
    }

    #[test]
    fn test_ed25519_verification_fails_on_tamper() {
        let reg = create_test_registration();
        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let mut signed = SignedPreRegistration::sign(&reg, &signing_key);

        // Tamper with the commitment hash
        signed.commitment.hash = "0".repeat(64);

        // Signature verification should fail
        assert!(!signed.verify().expect("operation should succeed"));
    }

    #[test]
    fn test_signed_with_timestamp() {
        let reg = create_test_registration();
        let signing_key = SigningKey::from_bytes(&[1u8; 32]);

        let signed = SignedPreRegistration::sign(&reg, &signing_key)
            .with_timestamp_proof(TimestampProof::git("deadbeef"));

        assert!(signed.timestamp_proof.is_some());
        assert!(signed.timestamp_proof.as_ref().expect("operation should succeed").is_git());
    }

    #[test]
    fn test_commitment_hash_bytes() {
        let reg = create_test_registration();
        let commitment = reg.commit();

        let bytes = commitment.hash_bytes();
        assert_eq!(bytes.len(), 32); // SHA-256 = 32 bytes
    }

    #[test]
    fn test_registration_with_notes() {
        let reg = create_test_registration().with_notes("Additional protocol considerations");

        assert_eq!(reg.notes, Some("Additional protocol considerations".to_string()));

        // Notes should affect the hash
        let reg_without_notes = create_test_registration();
        assert_ne!(reg.commit().hash, reg_without_notes.commit().hash);
    }

    #[test]
    fn test_different_registrations_different_hashes() {
        let reg1 = create_test_registration();
        let reg2 = PreRegistration::new(
            "Different Study",
            "Different hypothesis",
            "Different methods",
            "Different analysis",
        );

        assert_ne!(reg1.commit().hash, reg2.commit().hash);
    }
}
