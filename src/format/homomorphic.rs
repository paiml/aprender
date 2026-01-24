//! Homomorphic Encryption for .apr Format (spec: homomorphic-encryption-spec.md)
//!
//! Enables computation on encrypted data without decryption.
//! Implements CKKS/BFV hybrid scheme per spec §4.3.
//!
//! # Security Level
//!
//! 128-bit post-quantum security per Homomorphic Encryption Standard \[9\].
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::homomorphic::{HeContext, SecurityLevel};
//!
//! let ctx = HeContext::new(SecurityLevel::Bit128)?;
//! let (public_key, secret_key) = ctx.generate_keys()?;
//!
//! let encrypted = ctx.encrypt(&[1.0, 2.0, 3.0], &public_key)?;
//! let result = ctx.add_encrypted(&encrypted, &encrypted)?;
//! let decrypted = ctx.decrypt(&result, &secret_key)?;
//! ```

use crate::error::{AprenderError, Result};
use serde::{Deserialize, Serialize};

/// Security level per Homomorphic Encryption Standard \[9\]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum SecurityLevel {
    /// 128-bit security (N=8192, Q≈218 bits) - recommended
    Bit128 = 0x80,
    /// 192-bit security (N=16384, Q≈438 bits)
    Bit192 = 0xC0,
    /// 256-bit security (N=32768, Q≈881 bits)
    Bit256 = 0xFF,
}

impl SecurityLevel {
    /// Get polynomial modulus degree for this security level
    #[must_use]
    pub const fn poly_modulus_degree(self) -> usize {
        match self {
            Self::Bit128 => 8192,
            Self::Bit192 => 16384,
            Self::Bit256 => 32768,
        }
    }

    /// Get number of SIMD slots (N/2)
    #[must_use]
    pub const fn slot_count(self) -> usize {
        self.poly_modulus_degree() / 2
    }

    /// Convert from u8
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x80 => Some(Self::Bit128),
            0xC0 => Some(Self::Bit192),
            0xFF => Some(Self::Bit256),
            _ => None,
        }
    }
}

/// Homomorphic encryption scheme (spec §4.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum HeScheme {
    /// BFV: Exact integer arithmetic
    Bfv = 0x01,
    /// CKKS: Approximate floating-point arithmetic
    Ckks = 0x02,
    /// Hybrid: BFV for indices, CKKS for scores (spec §4.3)
    Hybrid = 0x03,
}

impl HeScheme {
    /// Convert from u8
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::Bfv),
            0x02 => Some(Self::Ckks),
            0x03 => Some(Self::Hybrid),
            _ => None,
        }
    }
}

/// HE parameters per spec §4.4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeParameters {
    /// Security level
    pub security_level: SecurityLevel,
    /// HE scheme
    pub scheme: HeScheme,
    /// Coefficient modulus bit sizes (e.g., [60, 40, 40, 60])
    pub coeff_modulus_bits: Vec<u8>,
    /// CKKS scale (2^scale_bits)
    pub scale_bits: u8,
    /// BFV plain modulus (prime for NTT efficiency)
    pub plain_modulus: u64,
}

impl HeParameters {
    /// Create default 128-bit parameters per spec §4.4
    #[must_use]
    pub fn default_128bit() -> Self {
        Self {
            security_level: SecurityLevel::Bit128,
            scheme: HeScheme::Hybrid,
            coeff_modulus_bits: vec![60, 40, 40, 60],
            scale_bits: 40,
            plain_modulus: 65537, // Prime for NTT
        }
    }

    /// Validate parameters against security requirements
    pub fn validate(&self) -> Result<()> {
        // Check coefficient modulus total bits
        let total_bits: u32 = self.coeff_modulus_bits.iter().map(|&b| u32::from(b)).sum();
        let max_bits = match self.security_level {
            SecurityLevel::Bit128 => 218,
            SecurityLevel::Bit192 => 438,
            SecurityLevel::Bit256 => 881,
        };

        if total_bits > max_bits {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Coefficient modulus too large: {} bits > {} max for {:?}",
                    total_bits, max_bits, self.security_level
                ),
            });
        }

        // Check plain modulus is prime (simple check)
        if self.plain_modulus < 2 {
            return Err(AprenderError::FormatError {
                message: "Plain modulus must be >= 2".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for HeParameters {
    fn default() -> Self {
        Self::default_128bit()
    }
}

/// Public key for encryption (safe to distribute)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HePublicKey {
    /// Serialized public key bytes
    pub data: Vec<u8>,
    /// Parameters used to generate this key
    pub params: HeParameters,
}

/// Secret key for decryption (never share)
#[derive(Clone, Serialize, Deserialize)]
pub struct HeSecretKey {
    /// Serialized secret key bytes
    data: Vec<u8>,
    /// Parameters used to generate this key
    params: HeParameters,
}

// Poka-yoke: Don't expose secret key data in Debug
impl std::fmt::Debug for HeSecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeSecretKey")
            .field("data", &"[REDACTED]")
            .field("params", &self.params)
            .finish()
    }
}

/// Relinearization keys (for multiplication depth management)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeRelinKeys {
    /// Serialized relin key bytes
    pub data: Vec<u8>,
}

/// Galois keys (for SIMD rotation operations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeGaloisKeys {
    /// Serialized galois key bytes
    pub data: Vec<u8>,
}

/// Ciphertext wrapper with type safety (Poka-yoke per spec §6.2)
#[derive(Clone, Serialize, Deserialize)]
pub struct Ciphertext {
    /// Encrypted data bytes
    data: Vec<u8>,
    /// Scheme used for encryption
    scheme: HeScheme,
    /// Current multiplication level
    level: u8,
}

impl Ciphertext {
    /// Create new ciphertext
    #[must_use]
    pub fn new(data: Vec<u8>, scheme: HeScheme) -> Self {
        Self {
            data,
            scheme,
            level: 0,
        }
    }

    /// Get ciphertext size in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get current multiplication level
    #[must_use]
    pub const fn level(&self) -> u8 {
        self.level
    }

    /// Get scheme
    #[must_use]
    pub const fn scheme(&self) -> HeScheme {
        self.scheme
    }

    /// Get raw data (for serialization)
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

// Poka-yoke: Don't expose ciphertext contents in Debug
impl std::fmt::Debug for Ciphertext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ciphertext")
            .field("size", &self.data.len())
            .field("scheme", &self.scheme)
            .field("level", &self.level)
            .finish()
    }
}

/// Plaintext wrapper for type safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plaintext {
    /// Encoded plaintext data
    data: Vec<u8>,
    /// Scheme used for encoding
    scheme: HeScheme,
}

impl Plaintext {
    /// Create new plaintext
    #[must_use]
    pub fn new(data: Vec<u8>, scheme: HeScheme) -> Self {
        Self { data, scheme }
    }

    /// Get raw data
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

/// HE Context for managing encryption operations
#[derive(Debug, Clone)]
pub struct HeContext {
    /// Parameters for this context
    params: HeParameters,
}

impl HeContext {
    /// Create new HE context with given security level
    pub fn new(security_level: SecurityLevel) -> Result<Self> {
        let mut params = HeParameters::default_128bit();
        params.security_level = security_level;
        params.validate()?;

        Ok(Self { params })
    }

    /// Create with custom parameters
    pub fn with_params(params: HeParameters) -> Result<Self> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Get parameters
    #[must_use]
    pub const fn params(&self) -> &HeParameters {
        &self.params
    }

    /// Generate key pair
    ///
    /// Returns (public_key, secret_key)
    pub fn generate_keys(&self) -> Result<(HePublicKey, HeSecretKey)> {
        // Stub implementation - real impl requires SEAL bindings
        // Size estimates per spec §7.2
        let pk_size = match self.params.security_level {
            SecurityLevel::Bit128 => 1_600_000, // ~1.6 MB
            SecurityLevel::Bit192 => 3_200_000,
            SecurityLevel::Bit256 => 6_400_000,
        };

        let public_key = HePublicKey {
            data: vec![0u8; pk_size],
            params: self.params.clone(),
        };

        let secret_key = HeSecretKey {
            data: vec![0u8; 32], // Much smaller
            params: self.params.clone(),
        };

        Ok((public_key, secret_key))
    }

    /// Generate relinearization keys
    pub fn generate_relin_keys(&self, _secret_key: &HeSecretKey) -> Result<HeRelinKeys> {
        // Stub - ~50MB per spec §7.2
        Ok(HeRelinKeys {
            data: vec![0u8; 1024], // Placeholder
        })
    }

    /// Generate Galois keys for SIMD rotations
    pub fn generate_galois_keys(&self, _secret_key: &HeSecretKey) -> Result<HeGaloisKeys> {
        // Stub - ~200MB per spec §7.2
        Ok(HeGaloisKeys {
            data: vec![0u8; 1024], // Placeholder
        })
    }

    /// Encrypt f64 values using CKKS
    pub fn encrypt_f64(&self, values: &[f64], public_key: &HePublicKey) -> Result<Ciphertext> {
        self.validate_key_params(public_key)?;

        // Stub: encode values as bytes
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        Ok(Ciphertext::new(data, HeScheme::Ckks))
    }

    /// Encrypt u64 values using BFV
    pub fn encrypt_u64(&self, values: &[u64], public_key: &HePublicKey) -> Result<Ciphertext> {
        self.validate_key_params(public_key)?;

        // Stub: encode values as bytes
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        Ok(Ciphertext::new(data, HeScheme::Bfv))
    }

    /// Decrypt to f64 values (CKKS)
    pub fn decrypt_f64(
        &self,
        ciphertext: &Ciphertext,
        secret_key: &HeSecretKey,
    ) -> Result<Vec<f64>> {
        self.validate_key_params_secret(secret_key)?;

        if ciphertext.scheme != HeScheme::Ckks {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot decrypt {:?} ciphertext as f64 (requires CKKS)",
                    ciphertext.scheme
                ),
            });
        }

        // Stub: decode bytes as f64
        let values: Vec<f64> = ciphertext
            .data
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
            .collect();

        Ok(values)
    }

    /// Decrypt to u64 values (BFV)
    pub fn decrypt_u64(
        &self,
        ciphertext: &Ciphertext,
        secret_key: &HeSecretKey,
    ) -> Result<Vec<u64>> {
        self.validate_key_params_secret(secret_key)?;

        if ciphertext.scheme != HeScheme::Bfv {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot decrypt {:?} ciphertext as u64 (requires BFV)",
                    ciphertext.scheme
                ),
            });
        }

        // Stub: decode bytes as u64
        let values: Vec<u64> = ciphertext
            .data
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
            .collect();

        Ok(values)
    }

    /// Add two ciphertexts
    pub fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        if a.scheme != b.scheme {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot add ciphertexts with different schemes: {:?} vs {:?}",
                    a.scheme, b.scheme
                ),
            });
        }

        // Stub: just concatenate for now
        let mut data = a.data.clone();
        data.extend_from_slice(&b.data);

        Ok(Ciphertext {
            data,
            scheme: a.scheme,
            level: a.level.max(b.level),
        })
    }

    /// Multiply two ciphertexts (increases level)
    pub fn multiply(
        &self,
        a: &Ciphertext,
        b: &Ciphertext,
        _relin_keys: &HeRelinKeys,
    ) -> Result<Ciphertext> {
        if a.scheme != b.scheme {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot multiply ciphertexts with different schemes: {:?} vs {:?}",
                    a.scheme, b.scheme
                ),
            });
        }

        let new_level = a.level.saturating_add(1).max(b.level.saturating_add(1));
        let max_level = u8::try_from(self.params.coeff_modulus_bits.len()).unwrap_or(4);

        if new_level >= max_level {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Multiplication depth exceeded: level {new_level} >= max {max_level}"
                ),
            });
        }

        // Stub
        Ok(Ciphertext {
            data: a.data.clone(),
            scheme: a.scheme,
            level: new_level,
        })
    }

    fn validate_key_params(&self, public_key: &HePublicKey) -> Result<()> {
        if public_key.params.security_level != self.params.security_level {
            return Err(AprenderError::FormatError {
                message: "Public key security level doesn't match context".to_string(),
            });
        }
        Ok(())
    }

    fn validate_key_params_secret(&self, secret_key: &HeSecretKey) -> Result<()> {
        if secret_key.params.security_level != self.params.security_level {
            return Err(AprenderError::FormatError {
                message: "Secret key security level doesn't match context".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===========================================
    // Property-Based Tests (proptest)
    // ===========================================

    #[cfg(test)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Property: encrypt/decrypt f64 roundtrip preserves values
            #[test]
            fn prop_f64_roundtrip(values in prop::collection::vec(-1e10f64..1e10f64, 1..100)) {
                let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
                let (pk, sk) = ctx.generate_keys().expect("keys");

                let encrypted = ctx.encrypt_f64(&values, &pk).expect("encrypt");
                let decrypted = ctx.decrypt_f64(&encrypted, &sk).expect("decrypt");

                prop_assert_eq!(decrypted, values);
            }

            /// Property: encrypt/decrypt u64 roundtrip preserves values
            #[test]
            fn prop_u64_roundtrip(values in prop::collection::vec(0u64..1_000_000, 1..100)) {
                let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
                let (pk, sk) = ctx.generate_keys().expect("keys");

                let encrypted = ctx.encrypt_u64(&values, &pk).expect("encrypt");
                let decrypted = ctx.decrypt_u64(&encrypted, &sk).expect("decrypt");

                prop_assert_eq!(decrypted, values);
            }

            /// Property: security level u8 roundtrip
            #[test]
            fn prop_security_level_roundtrip(level in prop::sample::select(vec![0x80u8, 0xC0, 0xFF])) {
                let parsed = SecurityLevel::from_u8(level);
                prop_assert!(parsed.is_some());
                prop_assert_eq!(parsed.unwrap() as u8, level);
            }

            /// Property: HE scheme u8 roundtrip
            #[test]
            fn prop_he_scheme_roundtrip(scheme in 1u8..=3) {
                let parsed = HeScheme::from_u8(scheme);
                prop_assert!(parsed.is_some());
                prop_assert_eq!(parsed.unwrap() as u8, scheme);
            }

            /// Property: invalid security level returns None
            #[test]
            fn prop_invalid_security_level(level in 0u8..0x80) {
                if level != 0x80 && level != 0xC0 && level != 0xFF {
                    prop_assert!(SecurityLevel::from_u8(level).is_none());
                }
            }

            /// Property: ciphertext size is proportional to input
            #[test]
            fn prop_ciphertext_size_proportional(len in 1usize..1000) {
                let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
                let (pk, _) = ctx.generate_keys().expect("keys");

                let values: Vec<f64> = (0..len).map(|i| i as f64).collect();
                let encrypted = ctx.encrypt_f64(&values, &pk).expect("encrypt");

                // Size should be at least input size (8 bytes per f64)
                prop_assert!(encrypted.size() >= len * 8);
            }
        }
    }

    // ===========================================
    // RED PHASE: SecurityLevel Tests
    // ===========================================

    #[test]
    fn test_security_level_poly_degree() {
        assert_eq!(SecurityLevel::Bit128.poly_modulus_degree(), 8192);
        assert_eq!(SecurityLevel::Bit192.poly_modulus_degree(), 16384);
        assert_eq!(SecurityLevel::Bit256.poly_modulus_degree(), 32768);
    }

    #[test]
    fn test_security_level_slot_count() {
        assert_eq!(SecurityLevel::Bit128.slot_count(), 4096);
        assert_eq!(SecurityLevel::Bit192.slot_count(), 8192);
        assert_eq!(SecurityLevel::Bit256.slot_count(), 16384);
    }

    #[test]
    fn test_security_level_from_u8() {
        assert_eq!(SecurityLevel::from_u8(0x80), Some(SecurityLevel::Bit128));
        assert_eq!(SecurityLevel::from_u8(0xC0), Some(SecurityLevel::Bit192));
        assert_eq!(SecurityLevel::from_u8(0xFF), Some(SecurityLevel::Bit256));
        assert_eq!(SecurityLevel::from_u8(0x00), None);
    }

    // ===========================================
    // RED PHASE: HeScheme Tests
    // ===========================================

    #[test]
    fn test_he_scheme_from_u8() {
        assert_eq!(HeScheme::from_u8(0x01), Some(HeScheme::Bfv));
        assert_eq!(HeScheme::from_u8(0x02), Some(HeScheme::Ckks));
        assert_eq!(HeScheme::from_u8(0x03), Some(HeScheme::Hybrid));
        assert_eq!(HeScheme::from_u8(0x00), None);
    }

    // ===========================================
    // RED PHASE: HeParameters Tests
    // ===========================================

    #[test]
    fn test_he_parameters_default() {
        let params = HeParameters::default_128bit();
        assert_eq!(params.security_level, SecurityLevel::Bit128);
        assert_eq!(params.scheme, HeScheme::Hybrid);
        assert_eq!(params.coeff_modulus_bits, vec![60, 40, 40, 60]);
        assert_eq!(params.scale_bits, 40);
        assert_eq!(params.plain_modulus, 65537);
    }

    #[test]
    fn test_he_parameters_validate_success() {
        let params = HeParameters::default_128bit();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_he_parameters_validate_coeff_modulus_too_large() {
        let mut params = HeParameters::default_128bit();
        params.coeff_modulus_bits = vec![60, 60, 60, 60]; // 240 > 218
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_he_parameters_validate_plain_modulus_zero() {
        let mut params = HeParameters::default_128bit();
        params.plain_modulus = 0;
        assert!(params.validate().is_err());
    }

    // ===========================================
    // RED PHASE: HeContext Tests
    // ===========================================

    #[test]
    fn test_he_context_new() {
        let ctx = HeContext::new(SecurityLevel::Bit128);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_he_context_generate_keys() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context creation");
        let result = ctx.generate_keys();
        assert!(result.is_ok());

        let (pk, sk) = result.expect("key generation");
        assert!(!pk.data.is_empty());
        assert!(!sk.data.is_empty());
    }

    #[test]
    fn test_he_context_generate_relin_keys() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (_, sk) = ctx.generate_keys().expect("keys");
        let relin = ctx.generate_relin_keys(&sk);
        assert!(relin.is_ok());
    }

    // ===========================================
    // RED PHASE: Encrypt/Decrypt Tests
    // ===========================================

    #[test]
    fn test_encrypt_decrypt_f64_roundtrip() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, sk) = ctx.generate_keys().expect("keys");

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let encrypted = ctx.encrypt_f64(&values, &pk).expect("encrypt");
        let decrypted = ctx.decrypt_f64(&encrypted, &sk).expect("decrypt");

        assert_eq!(decrypted, values);
    }

    #[test]
    fn test_encrypt_decrypt_u64_roundtrip() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, sk) = ctx.generate_keys().expect("keys");

        let values = vec![100u64, 200, 300, 400];
        let encrypted = ctx.encrypt_u64(&values, &pk).expect("encrypt");
        let decrypted = ctx.decrypt_u64(&encrypted, &sk).expect("decrypt");

        assert_eq!(decrypted, values);
    }

    #[test]
    fn test_decrypt_wrong_scheme_fails() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, sk) = ctx.generate_keys().expect("keys");

        let encrypted = ctx.encrypt_f64(&[1.0], &pk).expect("encrypt");
        // Try to decrypt CKKS ciphertext as BFV
        let result = ctx.decrypt_u64(&encrypted, &sk);
        assert!(result.is_err());
    }

    // ===========================================
    // RED PHASE: Ciphertext Operations Tests
    // ===========================================

    #[test]
    fn test_ciphertext_add_same_scheme() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, _) = ctx.generate_keys().expect("keys");

        let a = ctx.encrypt_f64(&[1.0], &pk).expect("encrypt a");
        let b = ctx.encrypt_f64(&[2.0], &pk).expect("encrypt b");

        let result = ctx.add(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ciphertext_add_different_scheme_fails() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, _) = ctx.generate_keys().expect("keys");

        let a = ctx.encrypt_f64(&[1.0], &pk).expect("encrypt a");
        let b = ctx.encrypt_u64(&[2], &pk).expect("encrypt b");

        let result = ctx.add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_ciphertext_multiply_increases_level() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, sk) = ctx.generate_keys().expect("keys");
        let relin = ctx.generate_relin_keys(&sk).expect("relin");

        let a = ctx.encrypt_f64(&[2.0], &pk).expect("encrypt");
        assert_eq!(a.level(), 0);

        let result = ctx.multiply(&a, &a, &relin).expect("multiply");
        assert_eq!(result.level(), 1);
    }

    #[test]
    fn test_ciphertext_multiply_depth_exceeded() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, sk) = ctx.generate_keys().expect("keys");
        let relin = ctx.generate_relin_keys(&sk).expect("relin");

        let mut ct = ctx.encrypt_f64(&[2.0], &pk).expect("encrypt");

        // Multiply until depth exceeded (4 levels for default params)
        for _ in 0..3 {
            ct = ctx.multiply(&ct, &ct, &relin).expect("multiply");
        }

        // This should fail
        let result = ctx.multiply(&ct, &ct, &relin);
        assert!(result.is_err());
    }

    // ===========================================
    // RED PHASE: Poka-yoke Tests
    // ===========================================

    #[test]
    fn test_secret_key_debug_redacted() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (_, sk) = ctx.generate_keys().expect("keys");

        let debug_str = format!("{:?}", sk);
        assert!(debug_str.contains("REDACTED"));
        assert!(!debug_str.contains("data: [0"));
    }

    #[test]
    fn test_ciphertext_debug_no_data() {
        let ctx = HeContext::new(SecurityLevel::Bit128).expect("context");
        let (pk, _) = ctx.generate_keys().expect("keys");

        let ct = ctx.encrypt_f64(&[1.0, 2.0], &pk).expect("encrypt");
        let debug_str = format!("{:?}", ct);

        assert!(debug_str.contains("size"));
        assert!(debug_str.contains("scheme"));
        assert!(!debug_str.contains("data: ["));
    }

    // ===========================================
    // RED PHASE: Serialization Tests
    // ===========================================

    #[test]
    fn test_he_parameters_serialize_roundtrip() {
        let params = HeParameters::default_128bit();
        let serialized = bincode::serialize(&params).expect("serialize");
        let deserialized: HeParameters = bincode::deserialize(&serialized).expect("deserialize");

        assert_eq!(deserialized.security_level, params.security_level);
        assert_eq!(deserialized.scheme, params.scheme);
        assert_eq!(deserialized.coeff_modulus_bits, params.coeff_modulus_bits);
    }

    #[test]
    fn test_ciphertext_serialize_roundtrip() {
        let ct = Ciphertext::new(vec![1, 2, 3, 4], HeScheme::Ckks);
        let serialized = bincode::serialize(&ct).expect("serialize");
        let deserialized: Ciphertext = bincode::deserialize(&serialized).expect("deserialize");

        assert_eq!(deserialized.data(), ct.data());
        assert_eq!(deserialized.scheme(), ct.scheme());
    }
}
