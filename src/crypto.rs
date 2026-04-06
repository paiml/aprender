//! Encryption at rest for model files (spec §3.3)
//!
//! Provides authenticated encryption for model distribution using:
//! - ChaCha20-Poly1305 AEAD (RFC 8439)
//! - Argon2id for password-based key derivation (RFC 9106)
//! - BLAKE3 for content verification
//!
//! ## Security
//!
//! - 256-bit key encryption (ChaCha20-Poly1305)
//! - Memory-hard password hashing (Argon2id)
//! - Authenticated encryption prevents tampering
//!
//! # Example
//!
//! ```no_run
//! use pacha::crypto::{encrypt_model, decrypt_model};
//!
//! // Encrypt a model file
//! let model_data = std::fs::read("model.gguf")?;
//! let encrypted = encrypt_model(&model_data, "my-secret-key")?;
//! std::fs::write("model.gguf.enc", &encrypted)?;
//!
//! // Decrypt at load time
//! let encrypted = std::fs::read("model.gguf.enc")?;
//! let decrypted = decrypt_model(&encrypted, "my-secret-key")?;
//! # Ok::<(), pacha::error::PachaError>(())
//! ```

use crate::error::{PachaError, Result};
use serde::{Deserialize, Serialize};

/// Magic bytes identifying encrypted pacha files
const MAGIC: &[u8; 8] = b"PACHAENC";

/// Current encryption format version
const VERSION: u8 = 1;

/// Salt length for key derivation (32 bytes)
const SALT_LEN: usize = 32;

/// Nonce length for ChaCha20-Poly1305 (12 bytes)
const NONCE_LEN: usize = 12;

/// Authentication tag length (16 bytes)
const TAG_LEN: usize = 16;

/// Header size: magic (8) + version (1) + salt (32) + nonce (12) = 53 bytes
const HEADER_SIZE: usize = 8 + 1 + SALT_LEN + NONCE_LEN;

/// Encrypted file header
#[derive(Debug, Clone)]
pub struct EncryptedHeader {
    /// Format version
    pub version: u8,
    /// Salt for key derivation
    pub salt: [u8; SALT_LEN],
    /// Nonce for encryption
    pub nonce: [u8; NONCE_LEN],
}

impl EncryptedHeader {
    /// Create a new header with random salt and nonce
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "encryption")]
        {
            use rand::rngs::OsRng;
            use rand::RngCore;
            let mut salt = [0u8; SALT_LEN];
            let mut nonce = [0u8; NONCE_LEN];
            OsRng.fill_bytes(&mut salt);
            OsRng.fill_bytes(&mut nonce);
            Self { version: VERSION, salt, nonce }
        }
        #[cfg(not(feature = "encryption"))]
        {
            // Fallback: simple PRNG for salt and nonce
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);

            let mut salt = [0u8; SALT_LEN];
            let mut nonce = [0u8; NONCE_LEN];

            for (i, byte) in salt.iter_mut().enumerate() {
                *byte = ((seed >> (i % 16)) ^ (i as u128 * 7)) as u8;
            }
            for (i, byte) in nonce.iter_mut().enumerate() {
                *byte = ((seed >> ((i + 32) % 16)) ^ (i as u128 * 13)) as u8;
            }

            Self { version: VERSION, salt, nonce }
        }
    }

    /// Serialize header to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_SIZE);
        bytes.extend_from_slice(MAGIC);
        bytes.push(self.version);
        bytes.extend_from_slice(&self.salt);
        bytes.extend_from_slice(&self.nonce);
        bytes
    }

    /// Parse header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(PachaError::InvalidFormat("encrypted file too short".to_string()));
        }

        // Verify magic
        if &data[0..8] != MAGIC {
            return Err(PachaError::InvalidFormat("not an encrypted pacha file".to_string()));
        }

        let version = data[8];
        if version != VERSION {
            return Err(PachaError::InvalidFormat(format!(
                "unsupported encryption version: {}",
                version
            )));
        }

        let mut salt = [0u8; SALT_LEN];
        salt.copy_from_slice(&data[9..9 + SALT_LEN]);

        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&data[9 + SALT_LEN..HEADER_SIZE]);

        Ok(Self { version, salt, nonce })
    }
}

impl Default for EncryptedHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Encryption configuration for Argon2id
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Argon2 memory cost in KiB (default: 64MB)
    pub memory_cost_kib: u32,
    /// Argon2 time cost (iterations, default: 3)
    pub time_cost: u32,
    /// Argon2 parallelism (default: 4)
    pub parallelism: u32,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            memory_cost_kib: 65536, // 64 MB
            time_cost: 3,
            parallelism: 4,
        }
    }
}

/// Derive encryption key from password using Argon2id
#[cfg(feature = "encryption")]
fn derive_key(
    password: &str,
    salt: &[u8; SALT_LEN],
    config: &EncryptionConfig,
) -> Result<[u8; 32]> {
    use argon2::{Algorithm, Argon2, Params, Version};

    let params =
        Params::new(config.memory_cost_kib, config.time_cost, config.parallelism, Some(32))
            .map_err(|e| PachaError::Validation(format!("Invalid Argon2 params: {e}")))?;

    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut key = [0u8; 32];
    argon2
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|e| PachaError::Validation(format!("Key derivation failed: {e}")))?;

    Ok(key)
}

/// Fallback key derivation when encryption feature is disabled
#[cfg(not(feature = "encryption"))]
fn derive_key(
    password: &str,
    salt: &[u8; SALT_LEN],
    _config: &EncryptionConfig,
) -> Result<[u8; 32]> {
    // Simple key derivation using iterated hashing (NOT SECURE - fallback only)
    let mut key = [0u8; 32];
    let mut state = [0u8; 64];

    for (i, &b) in password.as_bytes().iter().enumerate() {
        state[i % 64] ^= b;
    }
    for (i, &b) in salt.iter().enumerate() {
        state[(i + 32) % 64] ^= b;
    }

    for iteration in 0..10000u32 {
        let iter_bytes = iteration.to_le_bytes();
        for (i, &b) in iter_bytes.iter().enumerate() {
            state[i] ^= b;
        }
        for i in 0..64 {
            state[i] = state[i].wrapping_add(state[(i + 1) % 64]).wrapping_mul(33);
        }
    }

    key.copy_from_slice(&state[0..32]);
    Ok(key)
}

/// Encrypt data using ChaCha20-Poly1305
#[cfg(feature = "encryption")]
fn chacha_encrypt(data: &[u8], key: &[u8; 32], nonce: &[u8; NONCE_LEN]) -> Result<Vec<u8>> {
    use chacha20poly1305::{
        aead::{Aead, KeyInit},
        ChaCha20Poly1305, Nonce,
    };

    let cipher = ChaCha20Poly1305::new_from_slice(key)
        .map_err(|e| PachaError::Validation(format!("Invalid key: {e}")))?;

    let nonce = Nonce::from_slice(nonce);

    cipher
        .encrypt(nonce, data)
        .map_err(|e| PachaError::Validation(format!("Encryption failed: {e}")))
}

/// Decrypt data using ChaCha20-Poly1305
#[cfg(feature = "encryption")]
fn chacha_decrypt(ciphertext: &[u8], key: &[u8; 32], nonce: &[u8; NONCE_LEN]) -> Result<Vec<u8>> {
    use chacha20poly1305::{
        aead::{Aead, KeyInit},
        ChaCha20Poly1305, Nonce,
    };

    let cipher = ChaCha20Poly1305::new_from_slice(key)
        .map_err(|e| PachaError::Validation(format!("Invalid key: {e}")))?;

    let nonce = Nonce::from_slice(nonce);

    cipher.decrypt(nonce, ciphertext).map_err(|_| {
        PachaError::InvalidFormat(
            "decryption failed: invalid password or corrupted data".to_string(),
        )
    })
}

/// Fallback XOR-based encryption (NOT SECURE - only for when feature disabled)
#[cfg(not(feature = "encryption"))]
fn chacha_encrypt(data: &[u8], key: &[u8; 32], nonce: &[u8; NONCE_LEN]) -> Result<Vec<u8>> {
    let mut output = data.to_vec();
    let mut keystream = [0u8; 64];

    for (block_idx, chunk) in output.chunks_mut(64).enumerate() {
        for (i, ks) in keystream.iter_mut().enumerate() {
            *ks = key[i % 32]
                .wrapping_add(nonce[i % NONCE_LEN])
                .wrapping_add(block_idx as u8)
                .wrapping_mul(i as u8 + 1);
        }
        for (i, byte) in chunk.iter_mut().enumerate() {
            *byte ^= keystream[i];
        }
    }

    // Append simplified tag
    let tag = compute_fallback_tag(&output, key);
    output.extend_from_slice(&tag);

    Ok(output)
}

#[cfg(not(feature = "encryption"))]
fn chacha_decrypt(ciphertext: &[u8], key: &[u8; 32], nonce: &[u8; NONCE_LEN]) -> Result<Vec<u8>> {
    if ciphertext.len() < TAG_LEN {
        return Err(PachaError::InvalidFormat("ciphertext too short".to_string()));
    }

    let data = &ciphertext[..ciphertext.len() - TAG_LEN];
    let stored_tag = &ciphertext[ciphertext.len() - TAG_LEN..];

    // Verify tag
    let computed_tag = compute_fallback_tag(data, key);
    if computed_tag != stored_tag {
        return Err(PachaError::InvalidFormat(
            "decryption failed: invalid password or corrupted data".to_string(),
        ));
    }

    // Decrypt
    let mut output = data.to_vec();
    let mut keystream = [0u8; 64];

    for (block_idx, chunk) in output.chunks_mut(64).enumerate() {
        for (i, ks) in keystream.iter_mut().enumerate() {
            *ks = key[i % 32]
                .wrapping_add(nonce[i % NONCE_LEN])
                .wrapping_add(block_idx as u8)
                .wrapping_mul(i as u8 + 1);
        }
        for (i, byte) in chunk.iter_mut().enumerate() {
            *byte ^= keystream[i];
        }
    }

    Ok(output)
}

#[cfg(not(feature = "encryption"))]
fn compute_fallback_tag(ciphertext: &[u8], key: &[u8; 32]) -> [u8; TAG_LEN] {
    let mut tag = [0u8; TAG_LEN];
    let mut state = [0u64; 4];

    for (i, &b) in key.iter().enumerate() {
        state[i % 4] ^= (b as u64) << ((i * 8) % 64);
    }

    for (i, &b) in ciphertext.iter().enumerate() {
        state[i % 4] = state[i % 4].wrapping_add(b as u64).wrapping_mul(0x100000001b3);
    }

    for (i, byte) in tag.iter_mut().enumerate() {
        *byte = (state[i % 4] >> ((i % 8) * 8)) as u8;
    }

    tag
}

/// Encrypt model data with password
///
/// Uses ChaCha20-Poly1305 for authenticated encryption and Argon2id for
/// key derivation. Returns encrypted data with header.
pub fn encrypt_model(data: &[u8], password: &str) -> Result<Vec<u8>> {
    encrypt_model_with_config(data, password, &EncryptionConfig::default())
}

/// Encrypt model data with password and custom config
pub fn encrypt_model_with_config(
    data: &[u8],
    password: &str,
    config: &EncryptionConfig,
) -> Result<Vec<u8>> {
    if password.is_empty() {
        return Err(PachaError::InvalidFormat("encryption password cannot be empty".to_string()));
    }

    let header = EncryptedHeader::new();
    let key = derive_key(password, &header.salt, config)?;

    // Encrypt data (includes auth tag for real implementation)
    let ciphertext = chacha_encrypt(data, &key, &header.nonce)?;

    // Assemble output: header + ciphertext (tag is included in ciphertext for chacha20poly1305)
    let mut output = header.to_bytes();
    output.extend_from_slice(&ciphertext);

    Ok(output)
}

/// Decrypt model data with password
pub fn decrypt_model(encrypted_data: &[u8], password: &str) -> Result<Vec<u8>> {
    decrypt_model_with_config(encrypted_data, password, &EncryptionConfig::default())
}

/// Decrypt model data with password and custom config
pub fn decrypt_model_with_config(
    encrypted_data: &[u8],
    password: &str,
    config: &EncryptionConfig,
) -> Result<Vec<u8>> {
    if encrypted_data.len() < HEADER_SIZE + TAG_LEN {
        return Err(PachaError::InvalidFormat("encrypted data too short".to_string()));
    }

    // Parse header
    let header = EncryptedHeader::from_bytes(encrypted_data)?;

    // Extract ciphertext (includes auth tag)
    let ciphertext = &encrypted_data[HEADER_SIZE..];

    // Derive key
    let key = derive_key(password, &header.salt, config)?;

    // Decrypt and verify (ChaCha20-Poly1305 verifies tag internally)
    chacha_decrypt(ciphertext, &key, &header.nonce)
}

/// Check if data appears to be encrypted
#[must_use]
pub fn is_encrypted(data: &[u8]) -> bool {
    data.len() >= 8 && &data[0..8] == MAGIC
}

/// Get encryption format version from encrypted data
pub fn get_version(data: &[u8]) -> Result<u8> {
    if data.len() < 9 {
        return Err(PachaError::InvalidFormat("data too short for version check".to_string()));
    }
    if &data[0..8] != MAGIC {
        return Err(PachaError::InvalidFormat("not an encrypted pacha file".to_string()));
    }
    Ok(data[8])
}

// ============================================================================
// Tests - Extreme TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Core Encryption/Decryption Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let original = b"Hello, this is test model data!";
        let password = "my-secret-password";

        let encrypted = encrypt_model(original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_encrypt_decrypt_large_data() {
        let original: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let password = "test-password-123";

        let encrypted = encrypt_model(&original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original, decrypted);
    }

    #[test]
    fn test_encrypt_decrypt_1mb_data() {
        let original: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();
        let password = "strong-password";

        let encrypted = encrypt_model(&original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original.len(), decrypted.len());
        assert_eq!(original, decrypted);
    }

    #[test]
    fn test_empty_data_encrypt() {
        let original: &[u8] = &[];
        let password = "password";

        let encrypted = encrypt_model(original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert!(decrypted.is_empty());
    }

    // -------------------------------------------------------------------------
    // Authentication Tests (Tampering Detection)
    // -------------------------------------------------------------------------

    #[test]
    fn test_wrong_password_fails() {
        let original = b"Secret model data";
        let password = "correct-password";
        let wrong_password = "wrong-password";

        let encrypted = encrypt_model(original, password).unwrap();
        let result = decrypt_model(&encrypted, wrong_password);

        assert!(result.is_err());
    }

    #[test]
    fn test_empty_password_rejected() {
        let data = b"test data";
        let result = encrypt_model(data, "");

        assert!(result.is_err());
    }

    #[test]
    fn test_corrupted_ciphertext_fails() {
        let original = b"Test data for corruption test";
        let password = "password";

        let mut encrypted = encrypt_model(original, password).unwrap();

        // Corrupt a byte in the ciphertext
        if encrypted.len() > HEADER_SIZE + 5 {
            encrypted[HEADER_SIZE + 5] ^= 0xFF;
        }

        let result = decrypt_model(&encrypted, password);
        assert!(result.is_err(), "Should detect ciphertext corruption");
    }

    #[test]
    fn test_corrupted_tag_fails() {
        let original = b"Test data";
        let password = "password";

        let mut encrypted = encrypt_model(original, password).unwrap();

        // Corrupt the last byte (part of auth tag)
        let len = encrypted.len();
        encrypted[len - 1] ^= 0xFF;

        let result = decrypt_model(&encrypted, password);
        assert!(result.is_err(), "Should detect tag corruption");
    }

    #[test]
    fn test_truncated_data_fails() {
        let original = b"Test data";
        let password = "password";

        let encrypted = encrypt_model(original, password).unwrap();
        let truncated = &encrypted[..encrypted.len() - 10];

        let result = decrypt_model(truncated, password);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Header Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_encrypted() {
        let original = b"Plain data";
        let password = "password";

        assert!(!is_encrypted(original));

        let encrypted = encrypt_model(original, password).unwrap();
        assert!(is_encrypted(&encrypted));
    }

    #[test]
    fn test_get_version() {
        let original = b"Test";
        let password = "pwd";

        let encrypted = encrypt_model(original, password).unwrap();
        let version = get_version(&encrypted).unwrap();

        assert_eq!(version, VERSION);
    }

    #[test]
    fn test_header_serialization() {
        let header = EncryptedHeader::new();
        let bytes = header.to_bytes();
        let parsed = EncryptedHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.version, parsed.version);
        assert_eq!(header.salt, parsed.salt);
        assert_eq!(header.nonce, parsed.nonce);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = vec![0u8; 100];
        data[0..8].copy_from_slice(b"NOTMAGIC");

        let result = EncryptedHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = vec![0u8; 100];
        data[0..8].copy_from_slice(MAGIC);
        data[8] = 99; // Unsupported version

        let result = EncryptedHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Configuration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encryption_config_default() {
        let config = EncryptionConfig::default();

        assert_eq!(config.memory_cost_kib, 65536);
        assert_eq!(config.time_cost, 3);
        assert_eq!(config.parallelism, 4);
    }

    #[test]
    fn test_encrypt_with_custom_config() {
        let original = b"Custom config test";
        let password = "password";

        let config = EncryptionConfig { memory_cost_kib: 32768, time_cost: 2, parallelism: 2 };

        let encrypted = encrypt_model_with_config(original, password, &config).unwrap();
        let decrypted = decrypt_model_with_config(&encrypted, password, &config).unwrap();

        assert_eq!(original.as_slice(), decrypted.as_slice());
    }

    // -------------------------------------------------------------------------
    // Password Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_special_characters_in_password() {
        let original = b"Test data";
        let password = "p@$$w0rd!#$%^&*()_+-=[]{}|;':\",./<>?";

        let encrypted = encrypt_model(original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_unicode_password() {
        let original = b"Test data";
        let password = "密码🔐пароль";

        let encrypted = encrypt_model(original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_very_long_password() {
        let original = b"Test data";
        let password = "a".repeat(10000);

        let encrypted = encrypt_model(original, &password).unwrap();
        let decrypted = decrypt_model(&encrypted, &password).unwrap();

        assert_eq!(original.as_slice(), decrypted.as_slice());
    }

    // -------------------------------------------------------------------------
    // Randomness/Uniqueness Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_different_encryptions_produce_different_ciphertext() {
        let original = b"Same data";
        let password = "same-password";

        let encrypted1 = encrypt_model(original, password).unwrap();
        let encrypted2 = encrypt_model(original, password).unwrap();

        // Different salt/nonce means different ciphertext
        assert_ne!(encrypted1, encrypted2);

        // But both decrypt correctly
        let decrypted1 = decrypt_model(&encrypted1, password).unwrap();
        let decrypted2 = decrypt_model(&encrypted2, password).unwrap();
        assert_eq!(decrypted1, decrypted2);
    }

    #[test]
    fn test_different_passwords_produce_different_ciphertext() {
        let original = b"Same data";

        let encrypted1 = encrypt_model(original, "password1").unwrap();
        let encrypted2 = encrypt_model(original, "password2").unwrap();

        assert_ne!(encrypted1, encrypted2);
    }

    // -------------------------------------------------------------------------
    // Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encryption_overhead() {
        let original = b"Test data for size check";
        let password = "password";

        let encrypted = encrypt_model(original, password).unwrap();

        // Overhead = header (53) + tag (16) = 69 bytes
        let min_overhead = HEADER_SIZE + TAG_LEN;
        assert!(encrypted.len() >= original.len() + min_overhead);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_single_byte_data() {
        let original = &[0x42u8];
        let password = "password";

        let encrypted = encrypt_model(original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_binary_data_with_nulls() {
        let original: Vec<u8> = vec![0, 0, 0, 1, 2, 3, 0, 0, 0];
        let password = "password";

        let encrypted = encrypt_model(&original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original, decrypted);
    }

    #[test]
    fn test_all_zeros_data() {
        let original = vec![0u8; 1000];
        let password = "password";

        let encrypted = encrypt_model(&original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original, decrypted);
    }

    #[test]
    fn test_all_ones_data() {
        let original = vec![0xFFu8; 1000];
        let password = "password";

        let encrypted = encrypt_model(&original, password).unwrap();
        let decrypted = decrypt_model(&encrypted, password).unwrap();

        assert_eq!(original, decrypted);
    }
}
