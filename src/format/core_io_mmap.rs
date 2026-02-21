
/// Load a model using memory-mapped I/O (zero-copy where possible)
///
/// Toyota Way Principle: *Muda* (Waste Elimination) - Eliminates redundant
/// data copies by mapping the file directly into the process address space.
///
/// # Performance
///
/// - Cold load: ~4x faster than standard `load()` for large models
/// - Memory: Uses ~1x file size vs ~2x for standard load
/// - Syscalls: Reduces `brk` calls from ~970 to ~50
///
/// # Safety
///
/// Uses OS-level memory mapping. The file must not be modified while loaded.
/// See `bundle-mmap-spec.md` Section 4 for safety considerations.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{load_mmap, ModelType};
///
/// // Load large model efficiently
/// let model: RandomForest = load_mmap("large_model.apr", ModelType::RandomForest)?;
/// ```
///
/// # Feature Flag
///
/// When `format-mmap` is enabled, uses real OS mmap via `memmap2`.
/// Otherwise, falls back to standard file I/O (same API, heap-allocated).
///
/// # Errors
///
/// Returns error on file not found, format error, type mismatch, or checksum failure
pub fn load_mmap<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    use crate::bundle::MappedFile;

    let mapped = MappedFile::open(path.as_ref())?;

    load_from_bytes(mapped.as_slice(), expected_type)
}

/// Load a model with automatic strategy selection based on file size
///
/// Toyota Way Principle: *Heijunka* (Level Loading) - Chooses the optimal
/// loading strategy based on file size to balance memory and performance.
///
/// # Strategy
///
/// - Files <= 1MB: Standard `load()` (lower overhead for small files)
/// - Files > 1MB: Memory-mapped `load_mmap()` (better for large files)
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{load_auto, ModelType};
///
/// // Automatically chooses best loading strategy
/// let model: KMeans = load_auto("model.apr", ModelType::KMeans)?;
/// ```
///
/// # Errors
///
/// Returns error on file not found, format error, type mismatch, or checksum failure
pub fn load_auto<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    let metadata = std::fs::metadata(path.as_ref())?;

    if metadata.len() > MMAP_THRESHOLD {
        load_mmap(path, expected_type)
    } else {
        load(path, expected_type)
    }
}

// ============================================================================
// ENCRYPTION HELPER FUNCTIONS
// ============================================================================

/// Verify encrypted data has minimum required size.
#[cfg(feature = "format-encryption")]
pub(crate) fn verify_encrypted_data_size(data: &[u8]) -> Result<()> {
    if data.len() < HEADER_SIZE + SALT_SIZE + NONCE_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("Data too small for encrypted model: {} bytes", data.len()),
        });
    }
    Ok(())
}

/// Verify encrypted data checksum.
#[cfg(feature = "format-encryption")]
pub(crate) fn verify_encrypted_checksum(data: &[u8]) -> Result<()> {
    let stored_checksum = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_checksum = crc32(&data[..data.len() - 4]);
    if stored_checksum != computed_checksum {
        return Err(AprenderError::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }
    Ok(())
}

/// Verify header has ENCRYPTED flag and correct model type.
#[cfg(feature = "format-encryption")]
pub(crate) fn verify_encrypted_header(header: &Header, expected_type: ModelType) -> Result<()> {
    if !header.flags.is_encrypted() {
        return Err(AprenderError::FormatError {
            message: "Data is not encrypted (ENCRYPTED flag not set)".to_string(),
        });
    }
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: data contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }
    Ok(())
}

/// Extract salt, nonce, and ciphertext from encrypted data.
#[cfg(feature = "format-encryption")]
pub(crate) fn extract_encrypted_components<'a>(
    data: &'a [u8],
    header: &Header,
) -> Result<([u8; SALT_SIZE], [u8; NONCE_SIZE], &'a [u8])> {
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let salt_end = metadata_end + SALT_SIZE;
    let nonce_end = salt_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    if payload_end > data.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Encrypted payload extends beyond data boundary".to_string(),
        });
    }

    let salt: [u8; SALT_SIZE] =
        data[metadata_end..salt_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid salt size".to_string(),
            })?;
    let nonce: [u8; NONCE_SIZE] =
        data[salt_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &data[nonce_end..payload_end];

    Ok((salt, nonce, ciphertext))
}

/// Decrypt payload using password and extracted components.
#[cfg(feature = "format-encryption")]
pub(crate) fn decrypt_encrypted_payload(
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

    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| AprenderError::DecryptionFailed {
            message: "Decryption failed (wrong password or corrupted data)".to_string(),
        })
}

/// Load an encrypted model from a byte slice (spec 1.1 + 4.1.2)
///
/// Enables the `include_bytes!()` pattern for embedding encrypted models.
/// Combines single binary deployment with password-based encryption.
///
/// # Arguments
/// * `data` - Raw encrypted .apr file bytes
/// * `expected_type` - Expected model type
/// * `password` - Password for decryption
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{load_from_bytes_encrypted, ModelType};
///
/// // Embed encrypted model at compile time
/// const MODEL: &[u8] = include_bytes!("model.apr.enc");
///
/// fn main() -> Result<()> {
///     let model: NaiveBayes = load_from_bytes_encrypted(
///         MODEL,
///         ModelType::NaiveBayes,
///         &get_password_from_env(),
///     )?;
///     Ok(())
/// }
/// ```
///
/// # Errors
/// Returns error on format error, type mismatch, or decryption failure
#[cfg(feature = "format-encryption")]
pub fn load_from_bytes_encrypted<M: DeserializeOwned>(
    data: &[u8],
    expected_type: ModelType,
    password: &str,
) -> Result<M> {
    // Validate data integrity (Jidoka: stop the line on corruption)
    verify_encrypted_data_size(data)?;
    verify_encrypted_checksum(data)?;

    // Parse and verify header
    let header = Header::from_bytes(&data[..HEADER_SIZE])?;
    verify_encrypted_header(&header, expected_type)?;

    // Extract encryption components and decrypt
    let (salt, nonce, ciphertext) = extract_encrypted_components(data, &header)?;
    let payload_compressed = decrypt_encrypted_payload(password, &salt, &nonce, ciphertext)?;

    // Decompress and deserialize
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Inspect model data without loading the payload (spec 1.1)
///
/// Useful for validating embedded models or checking metadata
/// without deserializing the full model.
///
/// # Arguments
/// * `data` - Raw .apr file bytes
///
/// # Errors
/// Returns error on format error
pub fn inspect_bytes(data: &[u8]) -> Result<ModelInfo> {
    // Verify minimum size
    if data.len() < HEADER_SIZE {
        return Err(AprenderError::FormatError {
            message: format!("Data too small: {} bytes", data.len()),
        });
    }

    // Parse header
    let header = Header::from_bytes(&data[..HEADER_SIZE])?;

    // Extract metadata
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    if metadata_end > data.len() {
        return Err(AprenderError::FormatError {
            message: "Metadata extends beyond data boundary".to_string(),
        });
    }

    let metadata_bytes = &data[HEADER_SIZE..metadata_end];
    let metadata: Metadata = rmp_serde::from_slice(metadata_bytes)
        .map_err(|e| AprenderError::Serialization(format!("Failed to parse metadata: {e}")))?;

    Ok(ModelInfo {
        model_type: header.model_type,
        format_version: header.version,
        metadata,
        payload_size: header.payload_size as usize,
        uncompressed_size: header.uncompressed_size as usize,
        encrypted: header.flags.is_encrypted(),
        signed: header.flags.is_signed(),
        streaming: header.flags.is_streaming(),
        licensed: header.flags.is_licensed(),
        trueno_native: header.flags.is_trueno_native(),
        quantized: header.flags.is_quantized(),
        has_model_card: header.flags.has_model_card(),
    })
}

/// Inspect a model file without loading the payload
///
/// # Arguments
/// * `path` - Input file path
///
/// # Errors
/// Returns error on I/O failure or format error
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    let path = path.as_ref();

    // Read header + metadata only
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = Header::from_bytes(&header_bytes)?;

    // Read metadata (MessagePack per spec 2)
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader.read_exact(&mut metadata_bytes)?;
    let metadata: Metadata = rmp_serde::from_slice(&metadata_bytes)
        .map_err(|e| AprenderError::Serialization(format!("Failed to parse metadata: {e}")))?;

    Ok(ModelInfo {
        model_type: header.model_type,
        format_version: header.version,
        metadata,
        payload_size: header.payload_size as usize,
        uncompressed_size: header.uncompressed_size as usize,
        encrypted: header.flags.is_encrypted(),
        signed: header.flags.is_signed(),
        streaming: header.flags.is_streaming(),
        licensed: header.flags.is_licensed(),
        trueno_native: header.flags.is_trueno_native(),
        quantized: header.flags.is_quantized(),
        has_model_card: header.flags.has_model_card(),
    })
}
