//! APR format encryption (AES-256-GCM + Argon2id/X25519, spec §4.1)

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{de::DeserializeOwned, Serialize};

use super::core_io::{
    compress_payload, crc32, decompress_payload, parse_and_validate_header, read_file_content,
    verify_encrypted_flag, verify_file_checksum, verify_payload_boundary,
};
use super::{
    Header, ModelType, SaveOptions, HEADER_SIZE, HKDF_INFO, KEY_SIZE, NONCE_SIZE,
    RECIPIENT_HASH_SIZE, SALT_SIZE, X25519_PUBLIC_KEY_SIZE,
};
use super::{X25519PublicKey, X25519SecretKey};
use crate::error::{AprenderError, Result};

/// Save a model with password-based encryption (spec §4.1.2)
///
/// Encrypts the model payload using AES-256-GCM with a key derived from
/// the password using Argon2id. The salt and nonce are prepended to the
/// encrypted payload.
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `model_type` - Model type identifier
/// * `path` - Output file path
/// * `options` - Save options (compression, metadata)
/// * `password` - Password for encryption
///
/// # Errors
/// Returns error on I/O failure, serialization error, or encryption failure
#[cfg(feature = "format-encryption")]
#[allow(clippy::needless_pass_by_value)] // SaveOptions is small and passed by value for ergonomics
pub fn save_encrypted<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
    password: &str,
) -> Result<()> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    let path = path.as_ref();

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Generate random salt and nonce
    let mut salt = [0u8; SALT_SIZE];
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut salt);
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut nonce_bytes);

    // Derive key using Argon2id (spec §4.1.2)
    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), &salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    // Encrypt payload with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, payload_compressed.as_ref())
        .map_err(|e| AprenderError::Other(format!("Encryption failed: {e}")))?;

    // Serialize metadata as MessagePack with named fields (spec §2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header with ENCRYPTED flag
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    // Payload size now includes salt + nonce + ciphertext
    header.payload_size = (SALT_SIZE + NONCE_SIZE + ciphertext.len()) as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;
    header.flags = header.flags.with_encrypted();

    // Assemble file content
    let mut content = Vec::new();
    content.extend_from_slice(&header.to_bytes());
    content.extend_from_slice(&metadata_bytes);
    content.extend_from_slice(&salt);
    content.extend_from_slice(&nonce_bytes);
    content.extend_from_slice(&ciphertext);

    // Calculate and append checksum
    let checksum = crc32(&content);
    content.extend_from_slice(&checksum.to_le_bytes());

    // Write to file
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&content)?;
    writer.flush()?;

    Ok(())
}

/// Load a model with password-based decryption (spec §4.1.2)
///
/// Decrypts the model payload using AES-256-GCM with a key derived from
/// the password using Argon2id.
///
/// # Arguments
/// * `path` - Input file path
/// * `expected_type` - Expected model type (for type safety)
/// * `password` - Password for decryption
///
/// # Errors
/// Returns error on I/O failure, format error, type mismatch, or decryption failure
#[cfg(feature = "format-encryption")]
pub fn load_encrypted<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    password: &str,
) -> Result<M> {
    let path = path.as_ref();

    // Read and validate file
    let content = read_file_content(path)?;
    verify_password_encrypted_file_size(&content)?;
    verify_file_checksum(&content)?;

    // Parse and validate header
    let header = parse_and_validate_header(&content, expected_type)?;
    verify_encrypted_flag(&header)?;

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let salt_end = metadata_end + SALT_SIZE;
    let nonce_end = salt_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    verify_payload_boundary(payload_end, content.len())?;

    // Extract encryption components and decrypt
    let (salt, nonce_bytes, ciphertext) = extract_password_encryption_components(
        &content,
        metadata_end,
        salt_end,
        nonce_end,
        payload_end,
    )?;
    let payload_compressed = decrypt_password_payload(password, &salt, &nonce_bytes, ciphertext)?;

    // Decompress and deserialize
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Verify minimum file size for password-encrypted files.
#[cfg(feature = "format-encryption")]
fn verify_password_encrypted_file_size(content: &[u8]) -> Result<()> {
    if content.len() < HEADER_SIZE + SALT_SIZE + NONCE_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!(
                "File too small for encrypted model: {} bytes",
                content.len()
            ),
        });
    }
    Ok(())
}

/// Extract salt, nonce, and ciphertext from password-encrypted file.
#[cfg(feature = "format-encryption")]
fn extract_password_encryption_components(
    content: &[u8],
    metadata_end: usize,
    salt_end: usize,
    nonce_end: usize,
    payload_end: usize,
) -> Result<([u8; SALT_SIZE], [u8; NONCE_SIZE], &[u8])> {
    let salt: [u8; SALT_SIZE] =
        content[metadata_end..salt_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid salt size".to_string(),
            })?;
    let nonce_bytes: [u8; NONCE_SIZE] =
        content[salt_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &content[nonce_end..payload_end];
    Ok((salt, nonce_bytes, ciphertext))
}

/// Derive key from password and decrypt payload.
#[cfg(feature = "format-encryption")]
fn decrypt_password_payload(
    password: &str,
    salt: &[u8; SALT_SIZE],
    nonce_bytes: &[u8; NONCE_SIZE],
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    // Derive key using Argon2id
    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    // Decrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| AprenderError::DecryptionFailed {
            message: "Decryption failed (wrong password or corrupted data)".to_string(),
        })
}

/// Save a model encrypted for a specific recipient (spec §4.1.3)
///
/// Uses X25519 key agreement + AES-256-GCM. The sender generates an ephemeral
/// keypair, performs ECDH with the recipient's public key, and derives the
/// encryption key using HKDF-SHA256.
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `model_type` - Model type identifier
/// * `path` - Output file path
/// * `options` - Save options (compression, metadata)
/// * `recipient_public_key` - Recipient's X25519 public key
///
/// # Errors
/// Returns error on I/O failure, serialization error, or encryption failure
#[cfg(feature = "format-encryption")]
#[allow(clippy::needless_pass_by_value)] // SaveOptions is small and passed by value for ergonomics
pub fn save_for_recipient<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
    recipient_public_key: &X25519PublicKey,
) -> Result<()> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;

    let path = path.as_ref();

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Generate ephemeral keypair for this encryption
    let ephemeral_secret = X25519SecretKey::random_from_rng(rand::rngs::OsRng);
    let ephemeral_public = X25519PublicKey::from(&ephemeral_secret);

    // Perform X25519 key agreement
    let shared_secret = ephemeral_secret.diffie_hellman(recipient_public_key);

    // Derive encryption key using HKDF-SHA256 (spec §4.1.3)
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut key = [0u8; KEY_SIZE];
    hkdf.expand(HKDF_INFO, &mut key)
        .map_err(|_| AprenderError::Other("HKDF expansion failed".to_string()))?;

    // Generate random nonce
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut nonce_bytes);

    // Encrypt payload with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, payload_compressed.as_ref())
        .map_err(|e| AprenderError::Other(format!("Encryption failed: {e}")))?;

    // Create recipient hash (first 8 bytes of recipient public key for identification)
    let recipient_hash: [u8; RECIPIENT_HASH_SIZE] = recipient_public_key.as_bytes()
        [..RECIPIENT_HASH_SIZE]
        .try_into()
        .expect("recipient hash size is correct");

    // Serialize metadata as MessagePack with named fields (spec §2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header with ENCRYPTED flag
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    // Payload: ephemeral_pub (32) + recipient_hash (8) + nonce (12) + ciphertext
    header.payload_size =
        (X25519_PUBLIC_KEY_SIZE + RECIPIENT_HASH_SIZE + NONCE_SIZE + ciphertext.len()) as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;
    header.flags = header.flags.with_encrypted();

    // Assemble file content (spec §4.1.3 layout)
    let mut content = Vec::new();
    content.extend_from_slice(&header.to_bytes());
    content.extend_from_slice(&metadata_bytes);
    content.extend_from_slice(ephemeral_public.as_bytes()); // 32 bytes
    content.extend_from_slice(&recipient_hash); // 8 bytes
    content.extend_from_slice(&nonce_bytes); // 12 bytes
    content.extend_from_slice(&ciphertext);

    // Calculate and append checksum
    let checksum = crc32(&content);
    content.extend_from_slice(&checksum.to_le_bytes());

    // Write to file
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&content)?;
    writer.flush()?;

    Ok(())
}

/// Load a model encrypted for this recipient (spec §4.1.3)
///
/// Uses X25519 key agreement + AES-256-GCM. The recipient uses their secret key
/// to perform ECDH with the sender's ephemeral public key.
///
/// # Arguments
/// * `path` - Input file path
/// * `expected_type` - Expected model type (for type safety)
/// * `recipient_secret_key` - Recipient's X25519 secret key
///
/// # Errors
/// Returns error on I/O failure, format error, type mismatch, or decryption failure
#[cfg(feature = "format-encryption")]
pub fn load_as_recipient<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    recipient_secret_key: &X25519SecretKey,
) -> Result<M> {
    let path = path.as_ref();

    // Read and validate file
    let content = read_file_content(path)?;
    verify_x25519_encrypted_file_size(&content)?;
    verify_file_checksum(&content)?;

    // Parse and validate header
    let header = parse_and_validate_header(&content, expected_type)?;
    verify_encrypted_flag(&header)?;

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let ephemeral_pub_end = metadata_end + X25519_PUBLIC_KEY_SIZE;
    let recipient_hash_end = ephemeral_pub_end + RECIPIENT_HASH_SIZE;
    let nonce_end = recipient_hash_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    verify_payload_boundary(payload_end, content.len())?;

    // Extract and verify recipient components
    let (ephemeral_public, stored_recipient_hash) = extract_x25519_recipient_info(
        &content,
        metadata_end,
        ephemeral_pub_end,
        recipient_hash_end,
    )?;
    verify_recipient(recipient_secret_key, stored_recipient_hash)?;

    // Extract nonce and ciphertext, then decrypt
    let (nonce_bytes, ciphertext) =
        extract_nonce_and_ciphertext(&content, recipient_hash_end, nonce_end, payload_end)?;
    let payload_compressed = decrypt_x25519_payload(
        recipient_secret_key,
        &ephemeral_public,
        &nonce_bytes,
        ciphertext,
    )?;

    // Decompress and deserialize
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Verify minimum file size for X25519-encrypted files.
#[cfg(feature = "format-encryption")]
fn verify_x25519_encrypted_file_size(content: &[u8]) -> Result<()> {
    const MIN_PAYLOAD_SIZE: usize = X25519_PUBLIC_KEY_SIZE + RECIPIENT_HASH_SIZE + NONCE_SIZE;
    if content.len() < HEADER_SIZE + MIN_PAYLOAD_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!(
                "File too small for X25519 encrypted model: {} bytes",
                content.len()
            ),
        });
    }
    Ok(())
}

/// Extract ephemeral public key and recipient hash from X25519-encrypted file.
#[cfg(feature = "format-encryption")]
fn extract_x25519_recipient_info(
    content: &[u8],
    metadata_end: usize,
    ephemeral_pub_end: usize,
    recipient_hash_end: usize,
) -> Result<(X25519PublicKey, [u8; RECIPIENT_HASH_SIZE])> {
    let ephemeral_pub_bytes: [u8; X25519_PUBLIC_KEY_SIZE] = content
        [metadata_end..ephemeral_pub_end]
        .try_into()
        .map_err(|_| AprenderError::FormatError {
            message: "Invalid ephemeral public key size".to_string(),
        })?;
    let ephemeral_public = X25519PublicKey::from(ephemeral_pub_bytes);

    let stored_recipient_hash: [u8; RECIPIENT_HASH_SIZE] = content
        [ephemeral_pub_end..recipient_hash_end]
        .try_into()
        .map_err(|_| AprenderError::FormatError {
            message: "Invalid recipient hash size".to_string(),
        })?;

    Ok((ephemeral_public, stored_recipient_hash))
}

include!("encryption_part_02.rs");
