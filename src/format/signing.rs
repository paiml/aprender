//! Digital signing support for .ald format (§5.2)
//!
//! Provides Ed25519 signatures for dataset integrity and authenticity
//! verification.

use crate::error::{Error, Result};

/// Ed25519 public key size (32 bytes)
pub const PUBLIC_KEY_SIZE: usize = 32;

/// Ed25519 signature size (64 bytes)
pub const SIGNATURE_SIZE: usize = 64;

/// Signing key pair for creating signatures
#[cfg(feature = "format-signing")]
#[derive(Clone)]
pub struct SigningKeyPair {
    signing_key: ed25519_dalek::SigningKey,
}

#[cfg(feature = "format-signing")]
impl std::fmt::Debug for SigningKeyPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Show first 8 bytes of public key as hex for identification
        let pk = self.public_key_bytes();
        write!(
            f,
            "SigningKeyPair {{ public_key: {:02x}{:02x}{:02x}{:02x}..., secret_key: [REDACTED] }}",
            pk[0], pk[1], pk[2], pk[3]
        )
    }
}

#[cfg(feature = "format-signing")]
impl SigningKeyPair {
    /// Generate a new random signing key pair
    ///
    /// # Errors
    ///
    /// Returns error if random number generation fails.
    pub fn generate() -> Result<Self> {
        let mut key_bytes = [0u8; 32];
        getrandom::getrandom(&mut key_bytes)
            .map_err(|e| Error::Format(format!("RNG error: {e}")))?;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&key_bytes);
        Ok(Self { signing_key })
    }

    /// Create a signing key pair from existing secret key bytes
    #[must_use]
    pub fn from_bytes(secret_key: [u8; 32]) -> Self {
        Self {
            signing_key: ed25519_dalek::SigningKey::from_bytes(&secret_key),
        }
    }

    /// Get the public key bytes for this key pair
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Get the secret key bytes
    #[must_use]
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Sign a message
    ///
    /// Returns a 64-byte Ed25519 signature.
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        use ed25519_dalek::Signer;
        self.signing_key.sign(message).to_bytes()
    }
}

/// Verify an Ed25519 signature
///
/// # Arguments
///
/// * `message` - The signed message
/// * `signature` - The 64-byte signature
/// * `public_key` - The 32-byte public key
///
/// # Errors
///
/// Returns error if the signature is invalid.
#[cfg(feature = "format-signing")]
pub fn verify(message: &[u8], signature: &[u8; 64], public_key: &[u8; 32]) -> Result<()> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let verifying_key = VerifyingKey::from_bytes(public_key)
        .map_err(|e| Error::Format(format!("Invalid public key: {e}")))?;

    let sig = Signature::from_bytes(signature);

    verifying_key
        .verify(message, &sig)
        .map_err(|_| Error::Format("Signature verification failed".to_string()))
}

/// Dataset signature block
#[derive(Debug, Clone)]
pub struct SignatureBlock {
    /// The Ed25519 signature (64 bytes)
    pub signature: [u8; 64],
    /// The signer's public key (32 bytes)
    pub public_key: [u8; 32],
}

impl SignatureBlock {
    /// Size of the signature block in bytes (64 + 32 = 96)
    pub const SIZE: usize = SIGNATURE_SIZE + PUBLIC_KEY_SIZE;

    /// Create a new signature block by signing data
    #[cfg(feature = "format-signing")]
    #[must_use]
    pub fn sign(data: &[u8], key_pair: &SigningKeyPair) -> Self {
        Self {
            signature: key_pair.sign(data),
            public_key: key_pair.public_key_bytes(),
        }
    }

    /// Verify the signature against the given data
    ///
    /// # Errors
    ///
    /// Returns error if signature verification fails.
    #[cfg(feature = "format-signing")]
    pub fn verify(&self, data: &[u8]) -> Result<()> {
        verify(data, &self.signature, &self.public_key)
    }

    /// Serialize to bytes (96 bytes: signature + public_key)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[..SIGNATURE_SIZE].copy_from_slice(&self.signature);
        buf[SIGNATURE_SIZE..].copy_from_slice(&self.public_key);
        buf
    }

    /// Deserialize from bytes
    ///
    /// # Errors
    ///
    /// Returns error if buffer is too small.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(Error::Format(format!(
                "Signature block too small: {} bytes, expected {}",
                buf.len(),
                Self::SIZE
            )));
        }

        let mut signature = [0u8; SIGNATURE_SIZE];
        let mut public_key = [0u8; PUBLIC_KEY_SIZE];
        signature.copy_from_slice(&buf[..SIGNATURE_SIZE]);
        public_key.copy_from_slice(&buf[SIGNATURE_SIZE..Self::SIZE]);

        Ok(Self {
            signature,
            public_key,
        })
    }
}

#[cfg(all(test, feature = "format-signing"))]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify_roundtrip() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let message = b"Hello, World! This is a test message.";

        let signature = key_pair.sign(message);
        let public_key = key_pair.public_key_bytes();

        verify(message, &signature, &public_key).expect("verification failed");
    }

    #[test]
    fn test_wrong_message_fails() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let message = b"Original message";
        let wrong_message = b"Wrong message";

        let signature = key_pair.sign(message);
        let public_key = key_pair.public_key_bytes();

        let result = verify(wrong_message, &signature, &public_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let key_pair1 = SigningKeyPair::generate().expect("keygen 1 failed");
        let key_pair2 = SigningKeyPair::generate().expect("keygen 2 failed");
        let message = b"Test message";

        let signature = key_pair1.sign(message);
        let wrong_public_key = key_pair2.public_key_bytes();

        let result = verify(message, &signature, &wrong_public_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_block_roundtrip() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let data = b"Dataset content to sign";

        let block = SignatureBlock::sign(data, &key_pair);
        block.verify(data).expect("verify failed");

        // Test serialization
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), SignatureBlock::SIZE);

        let restored = SignatureBlock::from_bytes(&bytes).expect("parse failed");
        restored.verify(data).expect("restored verify failed");
    }

    #[test]
    fn test_from_bytes_keypair() {
        let key_pair1 = SigningKeyPair::generate().expect("keygen failed");
        let secret_bytes = key_pair1.secret_key_bytes();

        let key_pair2 = SigningKeyPair::from_bytes(secret_bytes);
        assert_eq!(key_pair1.public_key_bytes(), key_pair2.public_key_bytes());

        // Both should produce identical signatures
        let message = b"Test determinism";
        assert_eq!(key_pair1.sign(message), key_pair2.sign(message));
    }

    #[test]
    fn test_different_keys_produce_different_signatures() {
        let key_pair1 = SigningKeyPair::generate().expect("keygen 1 failed");
        let key_pair2 = SigningKeyPair::generate().expect("keygen 2 failed");
        let message = b"Same message";

        let sig1 = key_pair1.sign(message);
        let sig2 = key_pair2.sign(message);

        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_signature_block_from_bytes_too_small() {
        let buf = [0u8; 10]; // Too small
        let result = SignatureBlock::from_bytes(&buf);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("too small"));
    }

    #[test]
    fn test_constants() {
        assert_eq!(PUBLIC_KEY_SIZE, 32);
        assert_eq!(SIGNATURE_SIZE, 64);
        assert_eq!(SignatureBlock::SIZE, 96);
    }

    #[test]
    fn test_signing_key_pair_clone() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let cloned = key_pair.clone();
        assert_eq!(key_pair.public_key_bytes(), cloned.public_key_bytes());
        assert_eq!(key_pair.secret_key_bytes(), cloned.secret_key_bytes());
    }

    #[test]
    fn test_signing_key_pair_debug() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let debug = format!("{:?}", key_pair);
        assert!(debug.contains("SigningKeyPair"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn test_signature_block_clone() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let data = b"Test data";
        let block = SignatureBlock::sign(data, &key_pair);
        let cloned = block.clone();
        assert_eq!(cloned.signature, block.signature);
        assert_eq!(cloned.public_key, block.public_key);
    }

    #[test]
    fn test_signature_block_debug() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let block = SignatureBlock::sign(b"test", &key_pair);
        let debug = format!("{:?}", block);
        assert!(debug.contains("SignatureBlock"));
    }

    #[test]
    fn test_verify_invalid_public_key() {
        let message = b"Test message";
        let signature = [0u8; 64];
        // All zeros is not a valid public key for Ed25519
        let invalid_pk = [0u8; 32];
        let result = verify(message, &signature, &invalid_pk);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_message_signing() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let message = b"";
        let signature = key_pair.sign(message);
        let public_key = key_pair.public_key_bytes();
        verify(message, &signature, &public_key).expect("empty message verify failed");
    }

    #[test]
    fn test_large_message_signing() {
        let key_pair = SigningKeyPair::generate().expect("keygen failed");
        let message = vec![0xABu8; 1024 * 1024]; // 1MB
        let signature = key_pair.sign(&message);
        let public_key = key_pair.public_key_bytes();
        verify(&message, &signature, &public_key).expect("large message verify failed");
    }
}
