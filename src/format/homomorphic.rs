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

include!("homomorphic_part_02.rs");
include!("homomorphic_part_03.rs");
