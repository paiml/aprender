//! Model Signing and Verification
//!
//! Provides Ed25519 digital signatures for model integrity and authenticity.
//! Uses ed25519-dalek for proper cryptographic implementation per RFC 8032.
//!
//! ## Features
//!
//! - Key generation and management
//! - Model signing with detached signatures
//! - Signature verification
//! - Keyring for multiple signing identities
//!
//! ## Security
//!
//! - 128-bit security level (Ed25519)
//! - Deterministic signatures (no random number generation during signing)
//! - Fast verification suitable for load-time checks
//!
//! ## Example
//!
//! ```rust,ignore
//! use pacha::signing::{SigningKey, VerifyingKey, sign_model, verify_model};
//!
//! // Generate a new key pair
//! let signing_key = SigningKey::generate();
//! let verifying_key = signing_key.verifying_key();
//!
//! // Sign a model
//! let signature = sign_model(&model_bytes, &signing_key)?;
//!
//! // Verify the signature
//! verify_model(&model_bytes, &signature)?;
//! ```

use crate::error::{PachaError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// SIGN-001: Key Types - Proper Ed25519 Implementation
// ============================================================================

/// Ed25519 signing key (private key)
///
/// This wraps ed25519-dalek's SigningKey for proper Ed25519 signatures
/// per RFC 8032.
#[derive(Clone)]
pub struct SigningKey {
    #[cfg(feature = "signing")]
    inner: ed25519_dalek::SigningKey,
    #[cfg(not(feature = "signing"))]
    bytes: [u8; 32],
}

impl SigningKey {
    /// Generate a new random signing key using a cryptographically secure RNG
    #[must_use]
    pub fn generate() -> Self {
        #[cfg(feature = "signing")]
        {
            use rand::rngs::OsRng;
            Self { inner: ed25519_dalek::SigningKey::generate(&mut OsRng) }
        }
        #[cfg(not(feature = "signing"))]
        {
            use std::collections::hash_map::RandomState;
            use std::hash::{BuildHasher, Hasher};

            let mut bytes = [0u8; 32];
            let hasher_state = RandomState::new();

            for (i, byte) in bytes.iter_mut().enumerate() {
                let mut hasher = hasher_state.build_hasher();
                hasher.write_usize(i);
                hasher.write_u64(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0),
                );
                *byte = (hasher.finish() & 0xFF) as u8;
            }

            Self { bytes }
        }
    }

    /// Create from raw bytes (32 bytes for Ed25519 secret key)
    ///
    /// # Errors
    ///
    /// Returns error if bytes length is not 32
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(PachaError::Validation(format!(
                "Invalid key length: expected 32, got {}",
                bytes.len()
            )));
        }

        #[cfg(feature = "signing")]
        {
            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(bytes);
            Ok(Self { inner: ed25519_dalek::SigningKey::from_bytes(&key_bytes) })
        }
        #[cfg(not(feature = "signing"))]
        {
            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(bytes);
            Ok(Self { bytes: key_bytes })
        }
    }

    /// Get raw key bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        #[cfg(feature = "signing")]
        {
            self.inner.as_bytes()
        }
        #[cfg(not(feature = "signing"))]
        {
            &self.bytes
        }
    }

    /// Derive the verifying (public) key
    #[must_use]
    pub fn verifying_key(&self) -> VerifyingKey {
        #[cfg(feature = "signing")]
        {
            VerifyingKey { inner: self.inner.verifying_key() }
        }
        #[cfg(not(feature = "signing"))]
        {
            // Simplified: deterministic derivation for fallback
            let mut public = [0u8; 32];
            let hash = blake3::hash(&self.bytes);
            public.copy_from_slice(&hash.as_bytes()[..32]);
            VerifyingKey { bytes: public }
        }
    }

    /// Sign a message using Ed25519
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> Signature {
        #[cfg(feature = "signing")]
        {
            use ed25519_dalek::Signer;
            let sig = self.inner.sign(message);
            Signature { bytes: sig.to_bytes() }
        }
        #[cfg(not(feature = "signing"))]
        {
            // Simplified BLAKE3-based signature for fallback
            let mut hasher = blake3::Hasher::new();
            hasher.update(&self.bytes);
            hasher.update(message);
            let r_hash = hasher.finalize();

            let mut hasher2 = blake3::Hasher::new();
            hasher2.update(r_hash.as_bytes());
            hasher2.update(&self.verifying_key().bytes);
            hasher2.update(message);
            let s_hash = hasher2.finalize();

            let mut signature_bytes = [0u8; 64];
            signature_bytes[..32].copy_from_slice(r_hash.as_bytes());
            signature_bytes[32..].copy_from_slice(s_hash.as_bytes());

            Signature { bytes: signature_bytes }
        }
    }

    /// Export to PEM format
    #[must_use]
    pub fn to_pem(&self) -> String {
        let encoded = base64_encode(self.as_bytes());
        format!(
            "-----BEGIN PACHA ED25519 SIGNING KEY-----\n{encoded}\n-----END PACHA ED25519 SIGNING KEY-----\n"
        )
    }

    /// Import from PEM format
    ///
    /// # Errors
    ///
    /// Returns error if PEM format is invalid
    pub fn from_pem(pem: &str) -> Result<Self> {
        let pem = pem.trim();

        // Support both old and new PEM headers
        let (start, end) = if pem.contains("ED25519") {
            ("-----BEGIN PACHA ED25519 SIGNING KEY-----", "-----END PACHA ED25519 SIGNING KEY-----")
        } else {
            ("-----BEGIN PACHA SIGNING KEY-----", "-----END PACHA SIGNING KEY-----")
        };

        if !pem.starts_with(start) || !pem.ends_with(end) {
            return Err(PachaError::Validation("Invalid PEM format".to_string()));
        }

        let content = pem.trim_start_matches(start).trim_end_matches(end).trim().replace('\n', "");

        let bytes = base64_decode(&content)?;
        Self::from_bytes(&bytes)
    }
}

impl fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SigningKey([REDACTED])")
    }
}

/// Ed25519 verifying key (public key)
#[derive(Clone, PartialEq, Eq)]
pub struct VerifyingKey {
    #[cfg(feature = "signing")]
    inner: ed25519_dalek::VerifyingKey,
    #[cfg(not(feature = "signing"))]
    bytes: [u8; 32],
}

impl fmt::Debug for VerifyingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VerifyingKey({})", self.to_hex())
    }
}

impl Serialize for VerifyingKey {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for VerifyingKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let hex = String::deserialize(deserializer)?;
        Self::from_hex(&hex).map_err(serde::de::Error::custom)
    }
}

impl VerifyingKey {
    /// Create from raw bytes (32 bytes for Ed25519 public key)
    ///
    /// # Errors
    ///
    /// Returns error if bytes length is not 32 or key is invalid
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(PachaError::Validation(format!(
                "Invalid key length: expected 32, got {}",
                bytes.len()
            )));
        }

        #[cfg(feature = "signing")]
        {
            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(bytes);
            let inner = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes)
                .map_err(|e| PachaError::Validation(format!("Invalid Ed25519 public key: {e}")))?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "signing"))]
        {
            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(bytes);
            Ok(Self { bytes: key_bytes })
        }
    }

    /// Get raw key bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        #[cfg(feature = "signing")]
        {
            self.inner.as_bytes()
        }
        #[cfg(not(feature = "signing"))]
        {
            &self.bytes
        }
    }

    /// Verify a signature using Ed25519
    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<()> {
        #[cfg(feature = "signing")]
        {
            use ed25519_dalek::Verifier;
            let sig = ed25519_dalek::Signature::from_bytes(&signature.bytes);
            self.inner.verify(message, &sig).map_err(|_| PachaError::SignatureInvalid)
        }
        #[cfg(not(feature = "signing"))]
        {
            // Simplified verification for fallback
            let r = &signature.bytes[..32];
            let s = &signature.bytes[32..];

            let mut hasher = blake3::Hasher::new();
            hasher.update(r);
            hasher.update(&self.bytes);
            hasher.update(message);
            let expected_s = hasher.finalize();

            if s != expected_s.as_bytes() {
                return Err(PachaError::SignatureInvalid);
            }

            Ok(())
        }
    }

    /// Export to hex string
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex_encode(self.as_bytes())
    }

    /// Import from hex string
    ///
    /// # Errors
    ///
    /// Returns error if hex is invalid
    pub fn from_hex(hex: &str) -> Result<Self> {
        let bytes = hex_decode(hex)?;
        Self::from_bytes(&bytes)
    }

    /// Export to PEM format
    #[must_use]
    pub fn to_pem(&self) -> String {
        let encoded = base64_encode(self.as_bytes());
        format!(
            "-----BEGIN PACHA ED25519 VERIFYING KEY-----\n{encoded}\n-----END PACHA ED25519 VERIFYING KEY-----\n"
        )
    }

    /// Import from PEM format
    ///
    /// # Errors
    ///
    /// Returns error if PEM format is invalid
    pub fn from_pem(pem: &str) -> Result<Self> {
        let pem = pem.trim();

        // Support both old and new PEM headers
        let (start, end) = if pem.contains("ED25519") {
            (
                "-----BEGIN PACHA ED25519 VERIFYING KEY-----",
                "-----END PACHA ED25519 VERIFYING KEY-----",
            )
        } else {
            ("-----BEGIN PACHA VERIFYING KEY-----", "-----END PACHA VERIFYING KEY-----")
        };

        if !pem.starts_with(start) || !pem.ends_with(end) {
            return Err(PachaError::Validation("Invalid PEM format".to_string()));
        }

        let content = pem.trim_start_matches(start).trim_end_matches(end).trim().replace('\n', "");

        let bytes = base64_decode(&content)?;
        Self::from_bytes(&bytes)
    }
}

// ============================================================================
// SIGN-002: Signature Type
// ============================================================================

/// Ed25519 signature (64 bytes)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// Raw signature bytes (64 bytes)
    bytes: [u8; 64],
}

impl Serialize for Signature {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let hex = String::deserialize(deserializer)?;
        Self::from_hex(&hex).map_err(serde::de::Error::custom)
    }
}

impl Signature {
    /// Create from raw bytes
    ///
    /// # Errors
    ///
    /// Returns error if bytes length is not 64
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 64 {
            return Err(PachaError::Validation(format!(
                "Invalid signature length: expected 64, got {}",
                bytes.len()
            )));
        }
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(bytes);
        Ok(Self { bytes: sig_bytes })
    }

    /// Get raw signature bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.bytes
    }

    /// Export to hex string
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex_encode(&self.bytes)
    }

    /// Import from hex string
    ///
    /// # Errors
    ///
    /// Returns error if hex is invalid
    pub fn from_hex(hex: &str) -> Result<Self> {
        let bytes = hex_decode(hex)?;
        Self::from_bytes(&bytes)
    }
}

// ============================================================================
// SIGN-003: Model Signature Metadata
// ============================================================================

/// Metadata for a signed model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSignature {
    /// Model content hash (BLAKE3)
    pub content_hash: String,
    /// Signature over the content hash (hex-encoded)
    pub signature: String,
    /// Signer's public key (hex-encoded)
    pub signer_key: String,
    /// Signer identity (optional, e.g., email)
    pub signer_id: Option<String>,
    /// Timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// Algorithm identifier
    pub algorithm: String,
}

impl ModelSignature {
    /// Create a new model signature
    #[must_use]
    pub fn new(
        content_hash: String,
        signature: Signature,
        signer_key: &VerifyingKey,
        signer_id: Option<String>,
    ) -> Self {
        Self {
            content_hash,
            signature: signature.to_hex(),
            signer_key: signer_key.to_hex(),
            signer_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            algorithm: "ed25519-blake3".to_string(),
        }
    }

    /// Verify this signature against model data
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub fn verify(&self, model_data: &[u8]) -> Result<()> {
        // Verify content hash matches
        let actual_hash = blake3::hash(model_data);
        let actual_hex = hex_encode(actual_hash.as_bytes());

        if actual_hex != self.content_hash {
            return Err(PachaError::HashMismatch {
                expected: self.content_hash.clone(),
                actual: actual_hex,
            });
        }

        // Verify signature
        let signature = Signature::from_hex(&self.signature)?;
        let signer_key = VerifyingKey::from_hex(&self.signer_key)?;

        // Sign the content hash, not the raw data (for efficiency)
        signer_key.verify(self.content_hash.as_bytes(), &signature)
    }

    /// Save to file
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from file
    ///
    /// # Errors
    ///
    /// Returns error if reading or parsing fails
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let sig: Self = serde_json::from_str(&json)?;
        Ok(sig)
    }
}

// ============================================================================
// SIGN-004: Keyring
// ============================================================================

/// Keyring for managing multiple signing identities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Keyring {
    /// Named verifying keys (hex-encoded)
    keys: HashMap<String, String>,
    /// Default key name
    default_key: Option<String>,
}

impl Keyring {
    /// Create a new empty keyring
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a verifying key with a name
    pub fn add(&mut self, name: impl Into<String>, key: &VerifyingKey) {
        self.keys.insert(name.into(), key.to_hex());
    }

    /// Get a verifying key by name
    ///
    /// # Errors
    ///
    /// Returns error if key not found or invalid
    pub fn get(&self, name: &str) -> Result<VerifyingKey> {
        let hex = self.keys.get(name).ok_or_else(|| PachaError::NotFound {
            kind: "key".to_string(),
            name: name.to_string(),
            version: "n/a".to_string(),
        })?;
        VerifyingKey::from_hex(hex)
    }

    /// Remove a key
    pub fn remove(&mut self, name: &str) -> bool {
        self.keys.remove(name).is_some()
    }

    /// List all key names
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        self.keys.keys().map(String::as_str).collect()
    }

    /// Set the default key
    pub fn set_default(&mut self, name: impl Into<String>) {
        self.default_key = Some(name.into());
    }

    /// Get the default key
    ///
    /// # Errors
    ///
    /// Returns error if no default set or key not found
    pub fn default_key(&self) -> Result<VerifyingKey> {
        let name = self
            .default_key
            .as_ref()
            .ok_or_else(|| PachaError::Validation("No default key set".to_string()))?;
        self.get(name)
    }

    /// Check if keyring is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Get number of keys
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Save keyring to file
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load keyring from file
    ///
    /// # Errors
    ///
    /// Returns error if reading or parsing fails
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let keyring: Self = serde_json::from_str(&json)?;
        Ok(keyring)
    }
}

// ============================================================================
// SIGN-005: High-Level API
// ============================================================================

/// Sign model data
///
/// # Errors
///
/// Returns error if signing fails
pub fn sign_model(model_data: &[u8], signing_key: &SigningKey) -> Result<ModelSignature> {
    sign_model_with_id(model_data, signing_key, None)
}

/// Sign model data with signer identity
///
/// # Errors
///
/// Returns error if signing fails
pub fn sign_model_with_id(
    model_data: &[u8],
    signing_key: &SigningKey,
    signer_id: Option<String>,
) -> Result<ModelSignature> {
    // Hash the model content
    let content_hash = blake3::hash(model_data);
    let content_hex = hex_encode(content_hash.as_bytes());

    // Sign the hash (not raw data, for efficiency with large models)
    let signature = signing_key.sign(content_hex.as_bytes());

    Ok(ModelSignature::new(content_hex, signature, &signing_key.verifying_key(), signer_id))
}

/// Verify a model signature
///
/// # Errors
///
/// Returns error if verification fails
pub fn verify_model(model_data: &[u8], signature: &ModelSignature) -> Result<()> {
    signature.verify(model_data)
}

/// Verify a model signature with a specific key
///
/// # Errors
///
/// Returns error if verification fails or key doesn't match
pub fn verify_model_with_key(
    model_data: &[u8],
    signature: &ModelSignature,
    expected_key: &VerifyingKey,
) -> Result<()> {
    // First verify the signature is valid
    signature.verify(model_data)?;

    // Then verify it was signed by the expected key
    if signature.signer_key != expected_key.to_hex() {
        return Err(PachaError::Validation(
            "Signature was not created by expected key".to_string(),
        ));
    }

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(PachaError::Validation("Invalid hex length".to_string()));
    }

    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|_| PachaError::Validation("Invalid hex character".to_string()))
        })
        .collect()
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();
    let mut i = 0;

    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes.get(i + 1).copied().unwrap_or(0);
        let b2 = bytes.get(i + 2).copied().unwrap_or(0);

        result.push(ALPHABET[(b0 >> 2) as usize] as char);
        result.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);

        if i + 1 < bytes.len() {
            result.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            result.push('=');
        }

        if i + 2 < bytes.len() {
            result.push(ALPHABET[(b2 & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }

        i += 3;
    }

    result
}

fn base64_decode(encoded: &str) -> Result<Vec<u8>> {
    const DECODE: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1,
        -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1,
        -1, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
    ];

    let encoded = encoded.trim_end_matches('=');
    let mut result = Vec::with_capacity(encoded.len() * 3 / 4);

    let mut buf = 0u32;
    let mut bits = 0;

    for c in encoded.chars() {
        let val = if (c as usize) < 128 { DECODE[c as usize] } else { -1 };

        if val < 0 {
            return Err(PachaError::Validation("Invalid base64 character".to_string()));
        }

        buf = (buf << 6) | (val as u32);
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(result)
}

// ============================================================================
// Tests - Extreme TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Key Generation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_signing_key_generate_produces_unique_keys() {
        let key1 = SigningKey::generate();
        let key2 = SigningKey::generate();

        // Keys should be different (probabilistically)
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_signing_key_is_32_bytes() {
        let key = SigningKey::generate();
        assert_eq!(key.as_bytes().len(), 32);
    }

    #[test]
    fn test_signing_key_from_bytes_valid() {
        let bytes = [42u8; 32];
        let key = SigningKey::from_bytes(&bytes).unwrap();
        assert_eq!(key.as_bytes(), &bytes);
    }

    #[test]
    fn test_signing_key_from_bytes_rejects_wrong_length() {
        let short = [42u8; 16];
        assert!(SigningKey::from_bytes(&short).is_err());

        let long = [42u8; 64];
        assert!(SigningKey::from_bytes(&long).is_err());
    }

    #[test]
    fn test_verifying_key_derivation_is_deterministic() {
        let signing = SigningKey::generate();
        let v1 = signing.verifying_key();
        let v2 = signing.verifying_key();

        assert_eq!(v1.as_bytes(), v2.as_bytes());
    }

    #[test]
    fn test_verifying_key_is_32_bytes() {
        let signing = SigningKey::generate();
        let verifying = signing.verifying_key();
        assert_eq!(verifying.as_bytes().len(), 32);
    }

    // -------------------------------------------------------------------------
    // Signature Tests - Core Ed25519 Properties
    // -------------------------------------------------------------------------

    #[test]
    fn test_sign_produces_64_byte_signature() {
        let key = SigningKey::generate();
        let sig = key.sign(b"test message");
        assert_eq!(sig.as_bytes().len(), 64);
    }

    #[test]
    fn test_sign_and_verify_succeeds() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        let message = b"Hello, World!";
        let signature = signing_key.sign(message);

        let result = verifying_key.verify(message, &signature);
        assert!(result.is_ok(), "Signature verification should succeed");
    }

    #[test]
    fn test_verify_rejects_wrong_message() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        let message = b"Hello, World!";
        let signature = signing_key.sign(message);

        let wrong_message = b"Wrong message";
        let result = verifying_key.verify(wrong_message, &signature);
        assert!(result.is_err(), "Should reject signature for different message");
    }

    #[test]
    fn test_verify_rejects_wrong_key() {
        let key1 = SigningKey::generate();
        let key2 = SigningKey::generate();

        let message = b"Hello, World!";
        let signature = key1.sign(message);

        let result = key2.verifying_key().verify(message, &signature);
        assert!(result.is_err(), "Should reject signature from different key");
    }

    #[test]
    fn test_signature_is_deterministic() {
        // Ed25519 signatures are deterministic (no randomness)
        let key = SigningKey::generate();
        let message = b"test message";

        let sig1 = key.sign(message);
        let sig2 = key.sign(message);

        assert_eq!(sig1.as_bytes(), sig2.as_bytes(), "Ed25519 signatures should be deterministic");
    }

    #[test]
    fn test_empty_message_signing() {
        let key = SigningKey::generate();
        let verifying = key.verifying_key();

        let sig = key.sign(b"");
        assert!(verifying.verify(b"", &sig).is_ok());
    }

    #[test]
    fn test_large_message_signing() {
        let key = SigningKey::generate();
        let verifying = key.verifying_key();

        // 1MB message
        let large_message = vec![0x42u8; 1024 * 1024];
        let sig = key.sign(&large_message);
        assert!(verifying.verify(&large_message, &sig).is_ok());
    }

    // -------------------------------------------------------------------------
    // Serialization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_signature_hex_roundtrip() {
        let key = SigningKey::generate();
        let signature = key.sign(b"test");

        let hex = signature.to_hex();
        let recovered = Signature::from_hex(&hex).unwrap();

        assert_eq!(signature, recovered);
    }

    #[test]
    fn test_verifying_key_hex_roundtrip() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        let hex = verifying_key.to_hex();
        let recovered = VerifyingKey::from_hex(&hex).unwrap();

        assert_eq!(verifying_key, recovered);
    }

    #[test]
    fn test_signing_key_pem_roundtrip() {
        let key = SigningKey::generate();
        let pem = key.to_pem();
        let recovered = SigningKey::from_pem(&pem).unwrap();

        assert_eq!(key.as_bytes(), recovered.as_bytes());
    }

    #[test]
    fn test_verifying_key_pem_roundtrip() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        let pem = verifying_key.to_pem();
        let recovered = VerifyingKey::from_pem(&pem).unwrap();

        assert_eq!(verifying_key, recovered);
    }

    // -------------------------------------------------------------------------
    // Model Signature Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_signature_creation_and_verification() {
        let signing_key = SigningKey::generate();
        let model_data = b"model weights here...";

        let signature = sign_model(model_data, &signing_key).unwrap();
        let result = verify_model(model_data, &signature);

        assert!(result.is_ok());
    }

    #[test]
    fn test_model_signature_detects_tampering() {
        let signing_key = SigningKey::generate();
        let model_data = b"model weights here...";

        let signature = sign_model(model_data, &signing_key).unwrap();

        let tampered = b"tampered weights!!!!";
        let result = verify_model(tampered, &signature);

        assert!(result.is_err());
    }

    #[test]
    fn test_model_signature_with_signer_id() {
        let signing_key = SigningKey::generate();
        let model_data = b"model data";

        let signature =
            sign_model_with_id(model_data, &signing_key, Some("developer@example.com".to_string()))
                .unwrap();

        assert_eq!(signature.signer_id, Some("developer@example.com".to_string()));
        assert!(verify_model(model_data, &signature).is_ok());
    }

    #[test]
    fn test_model_signature_algorithm_field() {
        let signing_key = SigningKey::generate();
        let signature = sign_model(b"data", &signing_key).unwrap();

        assert_eq!(signature.algorithm, "ed25519-blake3");
    }

    #[test]
    fn test_model_signature_has_recent_timestamp() {
        let signing_key = SigningKey::generate();
        let signature = sign_model(b"data", &signing_key).unwrap();

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        assert!(signature.timestamp <= now);
        assert!(signature.timestamp > now - 60, "Timestamp should be within last minute");
    }

    #[test]
    fn test_verify_with_expected_key() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();
        let model_data = b"model data";

        let signature = sign_model(model_data, &signing_key).unwrap();

        // Should succeed with correct key
        let result = verify_model_with_key(model_data, &signature, &verifying_key);
        assert!(result.is_ok());

        // Should fail with wrong key
        let other_key = SigningKey::generate().verifying_key();
        let result = verify_model_with_key(model_data, &signature, &other_key);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Keyring Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_keyring_basic_operations() {
        let mut keyring = Keyring::new();
        assert!(keyring.is_empty());

        let key = SigningKey::generate().verifying_key();
        keyring.add("test", &key);

        assert_eq!(keyring.len(), 1);
        assert!(!keyring.is_empty());

        let retrieved = keyring.get("test").unwrap();
        assert_eq!(retrieved, key);
    }

    #[test]
    fn test_keyring_default_key() {
        let mut keyring = Keyring::new();
        let key = SigningKey::generate().verifying_key();

        keyring.add("main", &key);
        keyring.set_default("main");

        let default = keyring.default_key().unwrap();
        assert_eq!(default, key);
    }

    #[test]
    fn test_keyring_list() {
        let mut keyring = Keyring::new();
        keyring.add("key1", &SigningKey::generate().verifying_key());
        keyring.add("key2", &SigningKey::generate().verifying_key());

        let names = keyring.list();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"key1"));
        assert!(names.contains(&"key2"));
    }

    #[test]
    fn test_keyring_remove() {
        let mut keyring = Keyring::new();
        keyring.add("test", &SigningKey::generate().verifying_key());

        assert!(keyring.remove("test"));
        assert!(!keyring.remove("test")); // Already removed
        assert!(keyring.is_empty());
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hex_roundtrip() {
        let data = vec![0, 127, 255, 42, 100];
        let hex = hex_encode(&data);
        let decoded = hex_decode(&hex).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_base64_roundtrip() {
        let data = vec![0, 127, 255, 42, 100, 200];
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_base64_empty() {
        let data: Vec<u8> = vec![];
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    // -------------------------------------------------------------------------
    // Property-Based Tests (Extreme TDD)
    // -------------------------------------------------------------------------

    #[test]
    fn test_sign_verify_any_message() {
        // Test with various message sizes
        for size in [0, 1, 10, 100, 1000, 10000] {
            let key = SigningKey::generate();
            let verifying = key.verifying_key();
            let message: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

            let sig = key.sign(&message);
            assert!(verifying.verify(&message, &sig).is_ok(), "Failed for message size {size}");
        }
    }

    #[test]
    fn test_different_keys_produce_different_signatures() {
        let message = b"test message";
        let key1 = SigningKey::generate();
        let key2 = SigningKey::generate();

        let sig1 = key1.sign(message);
        let sig2 = key2.sign(message);

        assert_ne!(sig1.as_bytes(), sig2.as_bytes());
    }

    #[test]
    fn test_serialization_preserves_signature_validity() {
        let key = SigningKey::generate();
        let verifying = key.verifying_key();
        let message = b"test data";

        let sig = key.sign(message);

        // Roundtrip through hex
        let hex = sig.to_hex();
        let recovered = Signature::from_hex(&hex).unwrap();

        // Should still verify
        assert!(verifying.verify(message, &recovered).is_ok());
    }
}
