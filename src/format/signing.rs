//! APR format digital signatures (Ed25519, spec §4.2)

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{de::DeserializeOwned, Serialize};

use super::core_io::{
    compress_payload, crc32, decompress_and_deserialize, parse_and_validate_header,
    read_file_content, verify_file_checksum, verify_payload_boundary, verify_signed_flag,
};
use super::{Header, ModelType, SaveOptions, HEADER_SIZE, PUBLIC_KEY_SIZE, SIGNATURE_SIZE};
use crate::error::{AprenderError, Result};

/// Save a model with Ed25519 digital signature (spec §4.2)
///
/// Signs the model content (header + metadata + payload) for provenance verification.
/// The signature block (96 bytes) is appended before the checksum.
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `model_type` - Model type identifier
/// * `path` - Output file path
/// * `options` - Save options (compression, metadata)
/// * `signing_key` - Ed25519 signing key for creating signature
///
/// # Errors
/// Returns error on I/O failure, serialization error, or signing failure
#[allow(clippy::needless_pass_by_value)] // SaveOptions is small and passed by value for ergonomics
pub fn save_signed<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
    signing_key: &SigningKey,
) -> Result<()> {
    use ed25519_dalek::Signer;

    let path = path.as_ref();

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Serialize metadata as MessagePack with named fields (spec §2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header with SIGNED flag
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    header.payload_size = payload_compressed.len() as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;
    header.flags = header.flags.with_signed();

    // Assemble content to sign (header + metadata + payload)
    let mut signable_content = Vec::new();
    signable_content.extend_from_slice(&header.to_bytes());
    signable_content.extend_from_slice(&metadata_bytes);
    signable_content.extend_from_slice(&payload_compressed);

    // Sign the content
    let signature = signing_key.sign(&signable_content);
    let verifying_key = signing_key.verifying_key();

    // Assemble complete file content
    let mut content = signable_content;
    content.extend_from_slice(&signature.to_bytes()); // 64 bytes
    content.extend_from_slice(verifying_key.as_bytes()); // 32 bytes

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

/// Load a model with signature verification (spec §4.2, Jidoka)
///
/// Verifies the Ed25519 signature before deserializing the model.
/// If verification fails, loading halts immediately (Jidoka principle).
///
/// # Arguments
/// * `path` - Input file path
/// * `expected_type` - Expected model type (for type safety)
/// * `trusted_key` - Optional trusted public key for verification (if None, uses embedded key)
///
/// # Errors
/// Returns error on I/O failure, format error, type mismatch, or signature verification failure
pub fn load_verified<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    trusted_key: Option<&VerifyingKey>,
) -> Result<M> {
    let path = path.as_ref();

    // Read and validate file
    let content = read_file_content(path)?;
    verify_signed_file_size(&content)?;
    verify_file_checksum(&content)?;

    // Parse and validate header
    let header = parse_and_validate_header(&content, expected_type)?;
    verify_signed_flag(&header)?;

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let payload_end = metadata_end + header.payload_size as usize;
    let signature_start = payload_end;
    let pubkey_start = signature_start + SIGNATURE_SIZE;
    let pubkey_end = pubkey_start + PUBLIC_KEY_SIZE;

    verify_payload_boundary(pubkey_end, content.len())?;

    // Extract and verify signature
    let (signature, embedded_key) =
        extract_signature_and_key(&content, signature_start, pubkey_start, pubkey_end)?;
    let verifying_key = trusted_key.unwrap_or(&embedded_key);
    let signable_content = &content[..payload_end];
    verify_signature(verifying_key, signable_content, &signature)?;

    // Extract and deserialize payload
    decompress_and_deserialize(&content[metadata_end..payload_end], header.compression)
}

/// Verify minimum file size for signed files.
fn verify_signed_file_size(content: &[u8]) -> Result<()> {
    const SIGNATURE_BLOCK_SIZE: usize = SIGNATURE_SIZE + PUBLIC_KEY_SIZE;
    if content.len() < HEADER_SIZE + SIGNATURE_BLOCK_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("File too small for signed model: {} bytes", content.len()),
        });
    }
    Ok(())
}

/// Extract signature and public key from file content.
fn extract_signature_and_key(
    content: &[u8],
    signature_start: usize,
    pubkey_start: usize,
    pubkey_end: usize,
) -> Result<(ed25519_dalek::Signature, VerifyingKey)> {
    use ed25519_dalek::Signature;

    let signature_bytes: [u8; 64] =
        content[signature_start..pubkey_start]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid signature size".to_string(),
            })?;
    let signature = Signature::from_bytes(&signature_bytes);

    let pubkey_bytes: [u8; 32] =
        content[pubkey_start..pubkey_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid public key size".to_string(),
            })?;
    let embedded_key =
        VerifyingKey::from_bytes(&pubkey_bytes).map_err(|e| AprenderError::FormatError {
            message: format!("Invalid public key: {e}"),
        })?;

    Ok((signature, embedded_key))
}

/// Verify Ed25519 signature.
fn verify_signature(
    verifying_key: &VerifyingKey,
    signable_content: &[u8],
    signature: &ed25519_dalek::Signature,
) -> Result<()> {
    use ed25519_dalek::Verifier;

    verifying_key
        .verify(signable_content, signature)
        .map_err(|e| AprenderError::SignatureInvalid {
            reason: format!("Signature verification failed: {e}"),
        })
}
