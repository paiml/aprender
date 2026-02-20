
/// Verify this file was encrypted for the given recipient.
#[cfg(feature = "format-encryption")]
fn verify_recipient(
    recipient_secret_key: &X25519SecretKey,
    stored_recipient_hash: [u8; RECIPIENT_HASH_SIZE],
) -> Result<()> {
    let our_public = X25519PublicKey::from(recipient_secret_key);
    let our_hash: [u8; RECIPIENT_HASH_SIZE] = our_public.as_bytes()[..RECIPIENT_HASH_SIZE]
        .try_into()
        .expect("hash size is correct");

    if stored_recipient_hash != our_hash {
        return Err(AprenderError::DecryptionFailed {
            message: "This file was encrypted for a different recipient".to_string(),
        });
    }
    Ok(())
}

/// Extract nonce and ciphertext from encrypted content.
#[cfg(feature = "format-encryption")]
fn extract_nonce_and_ciphertext(
    content: &[u8],
    recipient_hash_end: usize,
    nonce_end: usize,
    payload_end: usize,
) -> Result<([u8; NONCE_SIZE], &[u8])> {
    let nonce_bytes: [u8; NONCE_SIZE] =
        content[recipient_hash_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &content[nonce_end..payload_end];
    Ok((nonce_bytes, ciphertext))
}

/// Perform X25519 key agreement and decrypt payload.
#[cfg(feature = "format-encryption")]
fn decrypt_x25519_payload(
    recipient_secret_key: &X25519SecretKey,
    ephemeral_public: &X25519PublicKey,
    nonce_bytes: &[u8; NONCE_SIZE],
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;

    // Perform X25519 key agreement
    let shared_secret = recipient_secret_key.diffie_hellman(ephemeral_public);

    // Derive encryption key using HKDF-SHA256
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut key = [0u8; KEY_SIZE];
    hkdf.expand(HKDF_INFO, &mut key)
        .map_err(|_| AprenderError::Other("HKDF expansion failed".to_string()))?;

    // Decrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| AprenderError::DecryptionFailed {
            message: "Decryption failed (wrong recipient key or corrupted data)".to_string(),
        })
}
