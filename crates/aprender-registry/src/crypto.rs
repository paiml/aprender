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
//! ## On-disk format
//!
//! `PACHAENC` ‖ version:u8 ‖ salt[32] ‖ nonce[12] ‖ body
//!
//! | version | body | status |
//! |---------|------|--------|
//! | 1 | XOR keystream + FNV-shaped checksum, clock-derived salt | **not encryption**; written by pacha ≤ 0.63.x, refused here (#2590) |
//! | 2 | ChaCha20-Poly1305 over an Argon2id-derived key | current |
//!
//! Version 2 exists because the body changed and the header did not. #2590
//! replaced the primitive but not the identifier, which would have left two
//! mutually undecodable formats both claiming version 1 under the same magic.
//! The version byte is the discriminator, so an archive can be classified from
//! its own first nine bytes; see [`get_version`].
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

/// Current encryption format version: ChaCha20-Poly1305 body, Argon2id-derived key.
///
/// **Bumped 1 → 2 by #2590, and the bump is load-bearing.** The header layout
/// (`PACHAENC` + version + 32-byte salt + 12-byte nonce) is byte-identical
/// across the two, but the body is not: version 1 as *shipped* in 0.63.0 was
/// the `not(feature = "encryption")` fallback — a
/// `wrapping_mul(i as u8 + 1)` XOR keystream followed by an FNV-shaped 16-byte
/// checksum, keyed by a 10 000-round add/multiply KDF over a `SystemTime`
/// nanosecond salt. `encryption` was not a default feature and the real arm did
/// not compile, so version 1 on disk means the XOR format and nothing else.
///
/// Leaving this at 1 would have made two mutually undecodable formats claim the
/// same identifier. A v1 file fed to this code would fail the Poly1305 check and
/// be reported as "invalid password or corrupted data" — the wrong diagnosis for
/// a file that is neither. [`VERSION_LEGACY_XOR`] keeps the two distinguishable
/// from the file itself.
const VERSION: u8 = 2;

/// Version byte written by the pre-#2590 XOR fallback. Never produced or read here.
///
/// [`EncryptedHeader::from_bytes`] recognises it only to say what it is; there is
/// no decryption path, because reading it would mean re-implementing the homebrew
/// cipher #2590 deleted. See [`VERSION`] for what the two versions mean.
const VERSION_LEGACY_XOR: u8 = 1;

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
    /// Create a new header with a cryptographically random salt and nonce
    ///
    /// Only available with the `encryption` feature. The removed
    /// `not(encryption)` arm derived salt and nonce from a nanosecond
    /// `SystemTime` reading shifted by `i % 16`, so both were a 16-byte
    /// repetition of a low-entropy, attacker-predictable timestamp (#2590).
    #[must_use]
    #[cfg(feature = "encryption")]
    pub fn new() -> Self {
        // rand 0.9 removed the `RngCore` impl on `OsRng` (it is `TryRngCore` only),
        // which is why this arm had never compiled. `rand::rng()` is the same
        // CSPRNG-seeded-from-OS-entropy handle `SigningKey::generate` already uses.
        use rand::RngCore;
        let mut salt = [0u8; SALT_LEN];
        let mut nonce = [0u8; NONCE_LEN];
        rand::rng().fill_bytes(&mut salt);
        rand::rng().fill_bytes(&mut nonce);
        Self { version: VERSION, salt, nonce }
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
        if version == VERSION_LEGACY_XOR {
            // Do NOT let this fall through to the AEAD, which would fail the
            // Poly1305 check and report "invalid password or corrupted data" —
            // a wrong diagnosis that sends the operator hunting for a password
            // that never existed.
            return Err(PachaError::InvalidFormat(
                "this file is pacha encryption format v1, which was not encryption: \
                 pacha 0.63.0 shipped without the `encryption` feature and wrote an \
                 unauthenticated XOR keystream under this same PACHAENC magic (#2590). \
                 It cannot be read by a build that has only ChaCha20-Poly1305, and its \
                 contents were never confidentiality-protected. Recover the plaintext \
                 with pacha 0.63.x and re-encrypt to v2."
                    .to_string(),
            ));
        }
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

#[cfg(feature = "encryption")]
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

/// Encrypt model data with password
///
/// Uses ChaCha20-Poly1305 for authenticated encryption and Argon2id for
/// key derivation. Returns encrypted data with header.
pub fn encrypt_model(data: &[u8], password: &str) -> Result<Vec<u8>> {
    encrypt_model_with_config(data, password, &EncryptionConfig::default())
}

/// Encrypt model data with password and custom config
///
/// # Errors
///
/// Returns [`PachaError::UnsupportedOperation`] when the crate was built without the
/// `encryption` feature. It used to return `Ok` with an XOR keystream and a
/// homebrew checksum instead — see [`unavailable`] (#2590).
#[cfg(feature = "encryption")]
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
///
/// # Errors
///
/// Returns [`PachaError::UnsupportedOperation`] when the crate was built without the
/// `encryption` feature (#2590).
#[cfg(feature = "encryption")]
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

// ============================================================================
// #2590: fail closed when the crate is built WITHOUT the `encryption` feature.
//
// Before this, `not(feature = "encryption")` selected a homebrew replacement for
// every primitive — a `wrapping_mul(i + 1)` XOR keystream, an FNV-shaped
// "auth tag", a 10 000-round add/multiply KDF, and a `SystemTime` nanosecond
// salt. Those returned `Ok(..)` and the CLI printed "Model encrypted
// successfully", so the insecure build was indistinguishable from the secure
// one at every observable surface. `encryption` was not a default feature, so
// that insecure build was the one that shipped.
// ============================================================================

/// Error returned by every crypto entry point when built without `encryption`
#[cfg(not(feature = "encryption"))]
fn unavailable() -> PachaError {
    PachaError::UnsupportedOperation {
        operation: "model encryption".to_string(),
        reason: "pacha was built without the `encryption` feature \
                 (ChaCha20-Poly1305 + Argon2id); refusing to fall back to a \
                 non-authenticated cipher. Rebuild with `--features encryption`."
            .to_string(),
    }
}

/// Encrypt model data with password and custom config — unavailable build
///
/// # Errors
///
/// Always errors: this build has no authenticated cipher.
#[cfg(not(feature = "encryption"))]
pub fn encrypt_model_with_config(
    _data: &[u8],
    _password: &str,
    _config: &EncryptionConfig,
) -> Result<Vec<u8>> {
    Err(unavailable())
}

/// Decrypt model data with password and custom config — unavailable build
///
/// # Errors
///
/// Always errors: this build has no authenticated cipher.
#[cfg(not(feature = "encryption"))]
pub fn decrypt_model_with_config(
    _encrypted_data: &[u8],
    _password: &str,
    _config: &EncryptionConfig,
) -> Result<Vec<u8>> {
    Err(unavailable())
}

/// Check if data appears to be encrypted
#[must_use]
pub fn is_encrypted(data: &[u8]) -> bool {
    data.len() >= 8 && &data[0..8] == MAGIC
}

/// Get encryption format version from encrypted data
///
/// `1` means the pre-#2590 XOR format (not encryption, and not readable here);
/// `2` means ChaCha20-Poly1305 + Argon2id. This is the only way to tell the two
/// apart — the magic and header layout are identical. See the module docs.
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

// These exercise the AEAD itself, so they need it to exist. Before #2590 this
// module was `#[cfg(test)]` and `encryption` was NOT a default feature, so every
// one of these 26 tests ran against the XOR fallback and none had ever run
// against ChaCha20-Poly1305 on any machine.
#[cfg(all(test, feature = "encryption"))]
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

// ============================================================================
// #2590 — falsifiers and adversarial cases for the crypto surface
//
// Issue #2590: "an encrypt/decrypt round-trip that passes proves nothing about
// whether the ciphertext is actually secure". What the audit found was worse
// than an untested surface: `encryption` was not a default feature, the
// `not(feature = "encryption")` arm silently substituted a homebrew cipher, and
// `--features encryption` did not compile at all (rand 0.9 dropped `RngCore`
// for `OsRng`). Every one of the 26 round-trip tests above was green against a
// cipher no one intended to ship.
//
// Each test below must EXCLUDE an outcome. The three `falsify_*` cases are the
// discrimination checks; the `adversarial_*` cases are the negatives the issue
// asked for (wrong key, truncation, tampering, empty input).
// ============================================================================

#[cfg(test)]
mod falsify_2590 {
    #[allow(unused_imports)]
    use super::*;

    /// Argon2id parameters for the sweep tests ONLY.
    ///
    /// The sweeps below run hundreds of decryptions; at the production default
    /// (64 MiB, t=3) that is minutes of KDF. These tests measure whether the
    /// AEAD *rejects* bad input, which is independent of KDF hardness. Tests
    /// that assert anything about the default configuration call
    /// `encrypt_model`/`decrypt_model` directly.
    #[cfg(feature = "encryption")]
    fn cheap_kdf() -> EncryptionConfig {
        EncryptionConfig { memory_cost_kib: 64, time_cost: 1, parallelism: 1 }
    }

    // ------------------------------------------------------------------------
    // FALSIFIER A — RED on origin/main as it stands.
    // ------------------------------------------------------------------------

    /// A build with no authenticated cipher must REFUSE, never substitute one.
    ///
    /// On origin/main `default = ["compression", "cli", "signing"]`, so the
    /// shipped build took the `not(feature = "encryption")` arm and returned
    /// `Ok` from a `wrapping_mul` XOR keystream with an FNV-shaped checksum
    /// standing in for Poly1305. `apr pacha encrypt` then printed
    /// "Model encrypted successfully". This test is RED there.
    #[test]
    fn falsify_2590_a_build_without_aead_refuses_rather_than_substituting_xor() {
        let result = encrypt_model(b"model weights", "correct horse battery staple");

        #[cfg(not(feature = "encryption"))]
        {
            let err = result.expect_err(
                "a build without ChaCha20-Poly1305 must refuse to encrypt, not XOR the \
                 plaintext and report success",
            );
            let msg = err.to_string();
            assert!(
                msg.contains("encryption"),
                "the refusal must name the missing feature so the operator can act; got: {msg}"
            );
        }

        #[cfg(feature = "encryption")]
        {
            let ciphertext = result.expect("the default build must have a real AEAD available");
            assert!(is_encrypted(&ciphertext), "output must carry the PACHAENC magic");
        }
    }

    // ------------------------------------------------------------------------
    // FALSIFIER B — the cipher is not the deleted toy.
    // ------------------------------------------------------------------------

    /// Ciphertext must carry none of the deleted fallback's multiply structure.
    ///
    /// The removed keystream was
    ///   `ks[i] = (key[i % 32] + nonce[i % 12] + block_idx).wrapping_mul(i as u8 + 1)`
    /// so `ks[31]` was always a multiple of 32 and `ks[63]` always a multiple of
    /// 64, in every 64-byte block. Encrypting all-zero plaintext publishes the
    /// keystream verbatim, so those 16 probes are all-zero under the toy cipher
    /// and all match under ChaCha20 with probability 32^-8 * 64^-8 = 3e-27.
    #[cfg(feature = "encryption")]
    #[test]
    fn falsify_2590_b_ciphertext_has_no_toy_multiply_structure() {
        const BLOCKS: usize = 8;
        let plaintext = vec![0u8; 64 * BLOCKS];
        let ciphertext =
            encrypt_model_with_config(&plaintext, "falsify-2590-b", &cheap_kdf()).expect("encrypt");
        let body = &ciphertext[HEADER_SIZE..];

        let mut toy_shaped = 0usize;
        for block in 0..BLOCKS {
            if body[64 * block + 31] % 32 == 0 {
                toy_shaped += 1;
            }
            if body[64 * block + 63] % 64 == 0 {
                toy_shaped += 1;
            }
        }

        assert!(
            toy_shaped < 2 * BLOCKS,
            "all {} keystream probes matched the deleted toy cipher's multiply structure — \
             this ciphertext is not ChaCha20 output",
            2 * BLOCKS
        );
    }

    // ------------------------------------------------------------------------
    // FALSIFIER C — the salt is entropy, not a clock reading.
    // ------------------------------------------------------------------------

    /// Salt must not be a 16-byte periodic expansion of a timestamp.
    ///
    /// The removed fallback used
    ///   `salt[i] = ((nanos >> (i % 16)) ^ (i as u128 * 7)) as u8`
    /// so `salt[i] ^ 7i == salt[i+16] ^ 7(i+16)` held for every `i` no matter
    /// what the clock read. This is the issue's "a keygen producing low-entropy
    /// or deterministic keys looks identical to a correct one" made checkable:
    /// real entropy matches all 16 positions with probability 2^-128.
    #[cfg(feature = "encryption")]
    #[test]
    fn falsify_2590_c_salt_is_not_a_timestamp_expansion() {
        let header = EncryptedHeader::new();

        let mut periodic = 0usize;
        for i in 0..16usize {
            let lhs = header.salt[i] ^ (i.wrapping_mul(7) as u8);
            let rhs = header.salt[i + 16] ^ ((i + 16).wrapping_mul(7) as u8);
            if lhs == rhs {
                periodic += 1;
            }
        }

        assert!(
            periodic < 16,
            "salt repeats with period 16 — it is a shifted clock reading, not entropy"
        );
    }

    // ------------------------------------------------------------------------
    // Adversarial: key material
    // ------------------------------------------------------------------------

    /// Salt and nonce must be distinct across rapid successive invocations.
    ///
    /// The removed fallback seeded both from `SystemTime` nanoseconds, so a
    /// tight loop could reuse a (key, nonce) pair — the single failure that
    /// breaks a stream cipher outright.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_salt_and_nonce_never_repeat_in_a_tight_loop() {
        use std::collections::HashSet;

        const N: usize = 256;
        let mut salts = HashSet::new();
        let mut nonces = HashSet::new();
        for _ in 0..N {
            let header = EncryptedHeader::new();
            salts.insert(header.salt);
            nonces.insert(header.nonce);
        }

        assert_eq!(salts.len(), N, "salt collided within {N} back-to-back headers");
        assert_eq!(nonces.len(), N, "nonce collided within {N} back-to-back headers");
    }

    /// Two encryptions of different plaintexts must not share a keystream.
    ///
    /// Under keystream reuse `ct_a ^ ct_b == pt_a ^ pt_b`, which for these
    /// inputs is 0xFF in every position. That is the classic two-time-pad
    /// break, and it is invisible to any round-trip test.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_no_two_time_pad_across_encryptions() {
        let zeros = vec![0u8; 256];
        let ones = vec![0xFFu8; 256];

        let ct_a =
            encrypt_model_with_config(&zeros, "same-password", &cheap_kdf()).expect("encrypt a");
        let ct_b =
            encrypt_model_with_config(&ones, "same-password", &cheap_kdf()).expect("encrypt b");

        let body_a = &ct_a[HEADER_SIZE..HEADER_SIZE + 256];
        let body_b = &ct_b[HEADER_SIZE..HEADER_SIZE + 256];

        let leaked = body_a.iter().zip(body_b.iter()).filter(|(a, b)| (*a ^ *b) == 0xFF).count();

        assert!(
            leaked < 256,
            "ct_a ^ ct_b equals pt_a ^ pt_b in every byte — the keystream was reused"
        );
    }

    // ------------------------------------------------------------------------
    // Adversarial: wrong key
    // ------------------------------------------------------------------------

    /// A password differing by one character must not decrypt.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_off_by_one_password_is_rejected() {
        let plaintext = b"weights that matter";
        let ciphertext =
            encrypt_model_with_config(plaintext, "hunter2", &cheap_kdf()).expect("encrypt");

        for wrong in ["hunter3", "hunter", "hunter22", "Hunter2", " hunter2", "hunter2 "] {
            let result = decrypt_model_with_config(&ciphertext, wrong, &cheap_kdf());
            assert!(
                result.is_err(),
                "password {wrong:?} must not decrypt a ciphertext made for \"hunter2\""
            );
        }
    }

    /// Decrypting with the right password but the wrong KDF cost must fail.
    ///
    /// The Argon2id parameters are not stored in the header, so they are part of
    /// the key. A build that silently ignored them would decrypt anyway.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_wrong_kdf_parameters_are_rejected() {
        let plaintext = b"weights that matter";
        let ciphertext = encrypt_model_with_config(plaintext, "pw", &cheap_kdf()).expect("encrypt");

        let other = EncryptionConfig { memory_cost_kib: 128, time_cost: 2, parallelism: 1 };
        let result = decrypt_model_with_config(&ciphertext, "pw", &other);
        assert!(result.is_err(), "different Argon2id parameters must derive a different key");
    }

    // ------------------------------------------------------------------------
    // Adversarial: tampering
    // ------------------------------------------------------------------------

    /// EVERY single-bit flip anywhere in the ciphertext must be rejected.
    ///
    /// `test_corrupted_ciphertext_fails` flipped one byte at one offset. One
    /// position passing is an anecdote; this sweeps all of them, header
    /// included, so a cipher that authenticates only part of its output cannot
    /// pass.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_every_single_bit_flip_is_rejected() {
        let plaintext: Vec<u8> = (0..48u8).collect();
        let ciphertext =
            encrypt_model_with_config(&plaintext, "pw", &cheap_kdf()).expect("encrypt");

        for byte_index in 0..ciphertext.len() {
            for bit in [0u8, 3, 7] {
                let mut tampered = ciphertext.clone();
                tampered[byte_index] ^= 1 << bit;
                let result = decrypt_model_with_config(&tampered, "pw", &cheap_kdf());
                assert!(
                    result.is_err(),
                    "flipping bit {bit} of byte {byte_index} was accepted — the ciphertext is \
                     not authenticated end to end"
                );
            }
        }
    }

    /// EVERY truncation length must be rejected — none may return plaintext.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_every_truncation_is_rejected() {
        let plaintext: Vec<u8> = (0..48u8).collect();
        let ciphertext =
            encrypt_model_with_config(&plaintext, "pw", &cheap_kdf()).expect("encrypt");

        for len in 0..ciphertext.len() {
            let result = decrypt_model_with_config(&ciphertext[..len], "pw", &cheap_kdf());
            assert!(
                result.is_err(),
                "a {len}-byte prefix of a {}-byte ciphertext was accepted",
                ciphertext.len()
            );
        }
    }

    /// Appending trailing bytes must be rejected, not silently ignored.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_appended_bytes_are_rejected() {
        let plaintext = b"exactly what was encrypted";
        let ciphertext = encrypt_model_with_config(plaintext, "pw", &cheap_kdf()).expect("encrypt");

        for extra in [1usize, 16, 64] {
            let mut padded = ciphertext.clone();
            padded.extend(std::iter::repeat_n(0x41u8, extra));
            let result = decrypt_model_with_config(&padded, "pw", &cheap_kdf());
            assert!(result.is_err(), "{extra} appended bytes were silently ignored");
        }
    }

    /// Splicing the body of one ciphertext onto the header of another must fail.
    ///
    /// This is the attack a per-message tag alone does not stop if the header is
    /// unauthenticated and the key does not depend on it.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_spliced_header_and_body_is_rejected() {
        let a = encrypt_model_with_config(b"alpha payload", "pw", &cheap_kdf()).expect("encrypt a");
        let b = encrypt_model_with_config(b"bravo payload", "pw", &cheap_kdf()).expect("encrypt b");
        assert_eq!(a.len(), b.len(), "same-length plaintexts give same-length ciphertexts");

        let mut spliced = a[..HEADER_SIZE].to_vec();
        spliced.extend_from_slice(&b[HEADER_SIZE..]);

        let result = decrypt_model_with_config(&spliced, "pw", &cheap_kdf());
        assert!(result.is_err(), "a body spliced under a foreign header was accepted");
    }

    // ------------------------------------------------------------------------
    // Adversarial: empty and degenerate input
    // ------------------------------------------------------------------------

    /// Empty plaintext still produces an authenticated ciphertext.
    ///
    /// `test_empty_data_encrypt` only round-tripped it. An empty payload whose
    /// tag can be flipped without detection is a forgery oracle.
    #[cfg(feature = "encryption")]
    #[test]
    fn adversarial_2590_empty_plaintext_is_still_authenticated() {
        let ciphertext = encrypt_model_with_config(b"", "pw", &cheap_kdf()).expect("encrypt");
        assert_eq!(
            ciphertext.len(),
            HEADER_SIZE + TAG_LEN,
            "empty plaintext must still carry a full-size tag"
        );

        let round_tripped =
            decrypt_model_with_config(&ciphertext, "pw", &cheap_kdf()).expect("decrypt");
        assert!(round_tripped.is_empty());

        for index in HEADER_SIZE..ciphertext.len() {
            let mut tampered = ciphertext.clone();
            tampered[index] ^= 0x01;
            assert!(
                decrypt_model_with_config(&tampered, "pw", &cheap_kdf()).is_err(),
                "the tag of an empty payload was forgeable at byte {index}"
            );
        }
    }

    /// An all-zero buffer of plausible length must never look like a ciphertext.
    #[test]
    fn adversarial_2590_zero_buffer_is_not_mistaken_for_ciphertext() {
        let zeros = vec![0u8; HEADER_SIZE + TAG_LEN + 64];
        assert!(!is_encrypted(&zeros), "an all-zero buffer must not carry the magic");
        assert!(get_version(&zeros).is_err(), "an all-zero buffer has no version");
    }

    /// An empty password must be refused, in every build.
    #[test]
    fn adversarial_2590_empty_password_is_refused() {
        assert!(
            encrypt_model(b"data", "").is_err(),
            "an empty password must never be accepted for encryption"
        );
    }

    // ------------------------------------------------------------------------
    // FALSIFIER G — the format's IDENTIFIER changed with the format.
    // ------------------------------------------------------------------------

    /// Two incompatible bodies must not share one version byte.
    ///
    /// #2590 replaced the on-disk body — XOR keystream + FNV checksum becomes
    /// ChaCha20-Poly1305 — while `MAGIC` and `VERSION` stayed put. That leaves a
    /// reader holding a `PACHAENC`/`\x01` file with no way to know which of two
    /// mutually undecodable formats it has, and the AEAD's failure mode for a v1
    /// file is the *wrong* message: "invalid password or corrupted data" for a
    /// file that has neither problem.
    ///
    /// This test fails in both directions, and both were run. Reverting `VERSION`
    /// to 1 fails the first assertion (`left: 1, right: 1` — fresh output is
    /// stamped with the legacy identifier again). Removing the
    /// `VERSION_LEGACY_XOR` arm from `from_bytes` fails the last, because the
    /// refusal degrades to `unsupported encryption version: 1`, which tells the
    /// operator the file is unreadable but not that it was never encrypted.
    #[cfg(feature = "encryption")]
    #[test]
    fn falsify_2590_g_legacy_v1_and_aead_v2_are_distinguishable_from_the_file() {
        // 1. What we write now is NOT stamped with the legacy identifier.
        let ciphertext =
            encrypt_model_with_config(b"weights", "pw", &cheap_kdf()).expect("encrypt");
        let stamped = get_version(&ciphertext).expect("fresh output must carry a version");
        assert_ne!(
            stamped, VERSION_LEGACY_XOR,
            "the AEAD format is stamped with the same version byte the deleted XOR \
             fallback used — the two formats are indistinguishable on disk"
        );
        assert_eq!(stamped, VERSION, "fresh output must be stamped v{VERSION}");

        // 2. A legacy v1 file is still recognisably a pacha encrypted file...
        let mut legacy = ciphertext.clone();
        legacy[8] = VERSION_LEGACY_XOR;
        assert!(is_encrypted(&legacy), "the magic is shared, so `is_encrypted` still holds");
        assert_eq!(get_version(&legacy).expect("version readable"), VERSION_LEGACY_XOR);

        // 3. ...and is refused with a diagnosis that names the format, never a
        //    password. This is the assertion that excludes the silent-collision
        //    outcome: a generic AEAD failure here would pass every other check.
        let err = EncryptedHeader::from_bytes(&legacy)
            .expect_err("a v1 header must not parse as the v2 AEAD format");
        let msg = err.to_string();
        assert!(
            !msg.contains("invalid password"),
            "a v1 file is not a wrong-password case, and saying so sends the operator \
             after a password that never existed; got: {msg}"
        );
        assert!(
            msg.contains("not encryption"),
            "a bare 'unsupported version' is not enough: the operator must be told that \
             v1 never protected the contents, or they will assume the data was \
             confidential and is merely unreadable; got: {msg}"
        );

        // The whole-file entry point must reach the same verdict, not just the
        // header parser the test called directly.
        let err = decrypt_model_with_config(&legacy, "pw", &cheap_kdf())
            .expect_err("decrypt_model must refuse a v1 file");
        assert!(
            err.to_string().contains("not encryption"),
            "decrypt must give the same diagnosis as the header parser; got: {err}"
        );
    }
}
