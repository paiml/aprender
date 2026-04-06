//! Encryption support for .ald format (§5.1)
//!
//! Provides AES-256-GCM encryption with two key derivation modes:
//! - Password mode: Argon2id KDF for human-memorable passwords
//! - Recipient mode: X25519 key agreement for asymmetric encryption

use crate::error::{Error, Result};

/// Salt size for Argon2id (16 bytes)
pub const SALT_SIZE: usize = 16;

/// Nonce size for AES-GCM (12 bytes)
pub const NONCE_SIZE: usize = 12;

/// Result of password-based encryption: (mode_byte, salt, nonce, ciphertext)
pub type PasswordEncryptResult = (u8, [u8; 16], [u8; 12], Vec<u8>);

/// Result of recipient-based encryption: (mode_byte, ephemeral_pub_key, nonce,
/// ciphertext)
pub type RecipientEncryptResult = (u8, [u8; 32], [u8; 12], Vec<u8>);

/// AES-GCM authentication tag size (16 bytes)
pub const TAG_SIZE: usize = 16;

/// X25519 public key size (32 bytes)
pub const X25519_PUBLIC_KEY_SIZE: usize = 32;

/// Encryption mode byte values
pub mod mode {
    /// Password-based encryption using Argon2id
    pub const PASSWORD: u8 = 0x00;
    /// Recipient-based encryption using X25519
    pub const RECIPIENT: u8 = 0x01;
}

/// Encryption mode configuration
#[derive(Debug, Clone)]
pub enum EncryptionMode {
    /// Password-based encryption using Argon2id KDF
    Password(String),
    /// Recipient-based encryption using X25519 key agreement
    Recipient {
        /// Recipient's X25519 public key (32 bytes)
        recipient_public_key: [u8; 32],
    },
}

/// Encryption parameters for saving
#[derive(Debug, Clone)]
pub struct EncryptionParams {
    /// The encryption mode and key material
    pub mode: EncryptionMode,
}

impl EncryptionParams {
    /// Create password-based encryption parameters
    #[must_use]
    pub fn password(password: impl Into<String>) -> Self {
        Self {
            mode: EncryptionMode::Password(password.into()),
        }
    }

    /// Create recipient-based encryption parameters
    #[must_use]
    pub fn recipient(public_key: [u8; 32]) -> Self {
        Self {
            mode: EncryptionMode::Recipient {
                recipient_public_key: public_key,
            },
        }
    }
}

/// Decryption parameters for loading
#[derive(Debug, Clone)]
pub enum DecryptionParams {
    /// Password for password-based decryption
    Password(String),
    /// Private key for recipient-based decryption
    PrivateKey([u8; 32]),
}

impl DecryptionParams {
    /// Create password-based decryption parameters
    #[must_use]
    pub fn password(password: impl Into<String>) -> Self {
        Self::Password(password.into())
    }

    /// Create private-key-based decryption parameters
    #[must_use]
    pub fn private_key(key: [u8; 32]) -> Self {
        Self::PrivateKey(key)
    }
}

/// Encrypt data using password-based encryption (Argon2id + AES-256-GCM)
///
/// Returns: (mode_byte, salt, nonce, ciphertext_with_tag)
#[cfg(feature = "format-encryption")]
pub fn encrypt_password(plaintext: &[u8], password: &str) -> Result<PasswordEncryptResult> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    // Generate random salt and nonce
    let mut salt = [0u8; SALT_SIZE];
    let mut nonce = [0u8; NONCE_SIZE];
    getrandom::getrandom(&mut salt).map_err(|e| Error::Format(format!("RNG error: {e}")))?;
    getrandom::getrandom(&mut nonce).map_err(|e| Error::Format(format!("RNG error: {e}")))?;

    // Derive key using Argon2id
    let mut key = [0u8; 32];
    Argon2::default()
        .hash_password_into(password.as_bytes(), &salt, &mut key)
        .map_err(|e| Error::Format(format!("Argon2 error: {e}")))?;

    // Encrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| Error::Format(format!("AES init error: {e}")))?;
    let nonce_obj = Nonce::from_slice(&nonce);
    let ciphertext = cipher
        .encrypt(nonce_obj, plaintext)
        .map_err(|e| Error::Format(format!("Encryption error: {e}")))?;

    Ok((mode::PASSWORD, salt, nonce, ciphertext))
}

/// Decrypt data using password-based decryption
#[cfg(feature = "format-encryption")]
pub fn decrypt_password(
    ciphertext: &[u8],
    password: &str,
    salt: &[u8; 16],
    nonce: &[u8; 12],
) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    // Derive key using Argon2id
    let mut key = [0u8; 32];
    Argon2::default()
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|e| Error::Format(format!("Argon2 error: {e}")))?;

    // Decrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| Error::Format(format!("AES init error: {e}")))?;
    let nonce_obj = Nonce::from_slice(nonce);
    cipher.decrypt(nonce_obj, ciphertext).map_err(|_| {
        Error::Format("Decryption failed: wrong password or corrupted data".to_string())
    })
}

/// Encrypt data using recipient-based encryption (X25519 + HKDF + AES-256-GCM)
///
/// Returns: (mode_byte, ephemeral_public_key, nonce, ciphertext_with_tag)
#[cfg(feature = "format-encryption")]
pub fn encrypt_recipient(
    plaintext: &[u8],
    recipient_public_key: &[u8; 32],
) -> Result<RecipientEncryptResult> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;
    use x25519_dalek::{EphemeralSecret, PublicKey};

    // Generate ephemeral key pair
    let mut rng_bytes = [0u8; 32];
    getrandom::getrandom(&mut rng_bytes).map_err(|e| Error::Format(format!("RNG error: {e}")))?;
    let ephemeral_secret = EphemeralSecret::random_from_rng(RngWrapper(rng_bytes));
    let ephemeral_public = PublicKey::from(&ephemeral_secret);

    // Perform X25519 key agreement
    let recipient_pk = PublicKey::from(*recipient_public_key);
    let shared_secret = ephemeral_secret.diffie_hellman(&recipient_pk);

    // Derive encryption key using HKDF
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut key = [0u8; 32];
    hkdf.expand(b"ald-v1-encrypt", &mut key)
        .map_err(|e| Error::Format(format!("HKDF error: {e}")))?;

    // Generate random nonce
    let mut nonce = [0u8; NONCE_SIZE];
    getrandom::getrandom(&mut nonce).map_err(|e| Error::Format(format!("RNG error: {e}")))?;

    // Encrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| Error::Format(format!("AES init error: {e}")))?;
    let nonce_obj = Nonce::from_slice(&nonce);
    let ciphertext = cipher
        .encrypt(nonce_obj, plaintext)
        .map_err(|e| Error::Format(format!("Encryption error: {e}")))?;

    Ok((
        mode::RECIPIENT,
        ephemeral_public.to_bytes(),
        nonce,
        ciphertext,
    ))
}

/// Decrypt data using recipient-based decryption
#[cfg(feature = "format-encryption")]
pub fn decrypt_recipient(
    ciphertext: &[u8],
    recipient_private_key: &[u8; 32],
    ephemeral_public_key: &[u8; 32],
    nonce: &[u8; 12],
) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;
    use x25519_dalek::{PublicKey, StaticSecret};

    // Reconstruct the shared secret
    let recipient_secret = StaticSecret::from(*recipient_private_key);
    let ephemeral_pk = PublicKey::from(*ephemeral_public_key);
    let shared_secret = recipient_secret.diffie_hellman(&ephemeral_pk);

    // Derive encryption key using HKDF
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut key = [0u8; 32];
    hkdf.expand(b"ald-v1-encrypt", &mut key)
        .map_err(|e| Error::Format(format!("HKDF error: {e}")))?;

    // Decrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| Error::Format(format!("AES init error: {e}")))?;
    let nonce_obj = Nonce::from_slice(nonce);
    cipher
        .decrypt(nonce_obj, ciphertext)
        .map_err(|_| Error::Format("Decryption failed: wrong key or corrupted data".to_string()))
}

/// Wrapper to use getrandom bytes as RngCore
#[cfg(feature = "format-encryption")]
struct RngWrapper([u8; 32]);

#[cfg(feature = "format-encryption")]
impl rand_core::RngCore for RngWrapper {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.copy_from_slice(&self.0[..dest.len()]);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[cfg(feature = "format-encryption")]
impl rand_core::CryptoRng for RngWrapper {}

#[cfg(all(test, feature = "format-encryption"))]
mod tests {
    use super::*;

    #[test]
    fn test_password_encrypt_decrypt_roundtrip() {
        let plaintext = b"Hello, World! This is a test message for encryption.";
        let password = "my_secure_password_123";

        let (mode_byte, salt, nonce, ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        assert_eq!(mode_byte, mode::PASSWORD);
        assert_ne!(ciphertext, plaintext);

        let decrypted =
            decrypt_password(&ciphertext, password, &salt, &nonce).expect("decrypt failed");

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_password_wrong_password_fails() {
        let plaintext = b"Secret data";
        let password = "correct_password";
        let wrong_password = "wrong_password";

        let (_, salt, nonce, ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        let result = decrypt_password(&ciphertext, wrong_password, &salt, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_recipient_encrypt_decrypt_roundtrip() {
        use x25519_dalek::{PublicKey, StaticSecret};

        let plaintext = b"Hello, recipient! This is a secure message.";

        // Generate recipient key pair
        let mut key_bytes = [0u8; 32];
        getrandom::getrandom(&mut key_bytes).expect("rng failed");
        let recipient_secret = StaticSecret::from(key_bytes);
        let recipient_public = PublicKey::from(&recipient_secret);

        let (mode_byte, ephemeral_pub, nonce, ciphertext) =
            encrypt_recipient(plaintext, recipient_public.as_bytes()).expect("encrypt failed");

        assert_eq!(mode_byte, mode::RECIPIENT);
        assert_ne!(ciphertext, plaintext);

        let decrypted = decrypt_recipient(&ciphertext, &key_bytes, &ephemeral_pub, &nonce)
            .expect("decrypt failed");

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_recipient_wrong_key_fails() {
        use x25519_dalek::{PublicKey, StaticSecret};

        let plaintext = b"Secret for specific recipient";

        // Generate recipient key pair
        let mut key_bytes = [0u8; 32];
        getrandom::getrandom(&mut key_bytes).expect("rng failed");
        let recipient_secret = StaticSecret::from(key_bytes);
        let recipient_public = PublicKey::from(&recipient_secret);

        let (_, ephemeral_pub, nonce, ciphertext) =
            encrypt_recipient(plaintext, recipient_public.as_bytes()).expect("encrypt failed");

        // Generate wrong key pair
        let mut wrong_key_bytes = [0u8; 32];
        getrandom::getrandom(&mut wrong_key_bytes).expect("rng failed");

        let result = decrypt_recipient(&ciphertext, &wrong_key_bytes, &ephemeral_pub, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_produces_different_ciphertexts() {
        let plaintext = b"Same message";
        let password = "same_password";

        let (_, _, _, ct1) = encrypt_password(plaintext, password).expect("encrypt 1 failed");
        let (_, _, _, ct2) = encrypt_password(plaintext, password).expect("encrypt 2 failed");

        // Due to random salt/nonce, ciphertexts should differ
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_empty_plaintext_encryption() {
        let plaintext = b"";
        let password = "password";

        let (mode_byte, salt, nonce, ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        assert_eq!(mode_byte, mode::PASSWORD);
        // Even empty plaintext produces ciphertext (due to auth tag)
        assert!(!ciphertext.is_empty());

        let decrypted =
            decrypt_password(&ciphertext, password, &salt, &nonce).expect("decrypt failed");
        assert_eq!(decrypted.as_slice(), plaintext);
    }

    #[test]
    fn test_large_plaintext_encryption() {
        // Test with 1MB plaintext
        let plaintext: Vec<u8> = (0u32..1_000_000).map(|i| (i % 256) as u8).collect();
        let password = "large_data_password";

        let (mode_byte, salt, nonce, ciphertext) =
            encrypt_password(&plaintext, password).expect("encrypt failed");

        assert_eq!(mode_byte, mode::PASSWORD);
        assert!(ciphertext.len() >= plaintext.len());

        let decrypted =
            decrypt_password(&ciphertext, password, &salt, &nonce).expect("decrypt failed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_special_characters_in_password() {
        let plaintext = b"Test data";
        let password = "p@$$w0rd!#%^&*()_+-=[]{}|;':\",./<>?`~";

        let (_, salt, nonce, ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        let decrypted =
            decrypt_password(&ciphertext, password, &salt, &nonce).expect("decrypt failed");
        assert_eq!(decrypted.as_slice(), plaintext);
    }

    #[test]
    fn test_unicode_password() {
        let plaintext = b"Data with unicode password";
        let password = "密码🔐пароль";

        let (_, salt, nonce, ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        let decrypted =
            decrypt_password(&ciphertext, password, &salt, &nonce).expect("decrypt failed");
        assert_eq!(decrypted.as_slice(), plaintext);
    }

    #[test]
    fn test_corrupted_ciphertext_fails() {
        let plaintext = b"Original data";
        let password = "password";

        let (_, salt, nonce, mut ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        // Corrupt the ciphertext
        if !ciphertext.is_empty() {
            ciphertext[0] ^= 0xFF;
        }

        let result = decrypt_password(&ciphertext, password, &salt, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrupted_nonce_fails() {
        let plaintext = b"Original data";
        let password = "password";

        let (_, salt, mut nonce, ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        // Corrupt the nonce
        nonce[0] ^= 0xFF;

        let result = decrypt_password(&ciphertext, password, &salt, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrupted_salt_fails() {
        let plaintext = b"Original data";
        let password = "password";

        let (_, mut salt, nonce, ciphertext) =
            encrypt_password(plaintext, password).expect("encrypt failed");

        // Corrupt the salt
        salt[0] ^= 0xFF;

        let result = decrypt_password(&ciphertext, password, &salt, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_rng_wrapper() {
        use rand_core::RngCore;

        let mut rng = RngWrapper([0x42; 32]);

        // Test next_u32
        let val = rng.next_u32();
        assert_eq!(val, 0x42424242);

        // Test next_u64
        let val64 = rng.next_u64();
        assert_eq!(val64, 0x4242424242424242);

        // Test fill_bytes
        let mut buf = [0u8; 8];
        rng.fill_bytes(&mut buf);
        assert_eq!(buf, [0x42; 8]);

        // Test try_fill_bytes
        let mut buf2 = [0u8; 4];
        rng.try_fill_bytes(&mut buf2).expect("should succeed");
        assert_eq!(buf2, [0x42; 4]);
    }

    #[test]
    fn test_mode_constants() {
        // Verify mode constants are distinct
        assert_ne!(mode::PASSWORD, mode::RECIPIENT);
        assert_eq!(mode::PASSWORD, 0x00);
        assert_eq!(mode::RECIPIENT, 0x01);
    }
}
