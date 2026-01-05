//! Cryptographic Agility (Post-Quantum Ready)
//!
//! Supports algorithm rotation without breaking file format compatibility.
//! Enables migration to post-quantum cryptography (NIST PQC standards).
//!
//! # References
//!
//! - [Barker 2020] NIST SP 800-57 Key Management Recommendations
//! - [NIST PQC] Post-Quantum Cryptography Standardization
//! - [Bernstein et al. 2012] Ed25519 specification
//! - [Cryptographic Agility RFC 7696]
//!
//! # Toyota Way Alignment
//!
//! - **Poka-yoke**: Type-safe suite selection prevents algorithm misuse
//! - **Kaizen**: Future-proof design allows gradual migration

use serde::{Deserialize, Serialize};

/// Supported cipher suites for cryptographic agility.
///
/// Allows rotation from classical to post-quantum algorithms without
/// breaking existing files.
///
/// # Design Rationale (Poka-yoke)
///
/// Hardcoding Ed25519 creates long-term risk. Post-quantum computers
/// could break classical signatures by 2030-2040. This enum allows
/// graceful migration without breaking existing files.
///
/// # Security Recommendations
///
/// - **Standard2025**: Use for current deployments (fast, small keys)
/// - **Hybrid2028**: Use for long-term sensitive data (transition period)
/// - **`PostQuantum2030`**: Use when NIST PQC is fully standardized
/// - **`GovHighAssurance`**: Use for government/defense applications
/// - **`LegacyRSA`**: DEPRECATED - load-only for migration
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CipherSuite {
    /// Current standard (2025): Fast, small keys
    /// - Sign: Ed25519 (ed25519-dalek)
    /// - Encrypt: XChaCha20-Poly1305
    /// - Hash: BLAKE3
    /// - KDF: Argon2id
    #[default]
    Standard2025 = 0x01,

    /// NIST PQC Standard (2030+): Post-Quantum resistant
    /// - Sign: ML-DSA-65 (Dilithium3)
    /// - KEM: ML-KEM-768 (Kyber768)
    /// - Hash: SHA3-256
    /// - KDF: HKDF-SHA3
    PostQuantum2030 = 0x02,

    /// Hybrid mode: Classical + Post-Quantum (transition period)
    /// - Sign: Ed25519 + ML-DSA-65 (both required)
    /// - KEM: X25519 + ML-KEM-768
    ///
    /// Provides security against both classical and quantum adversaries
    Hybrid2028 = 0x03,

    /// Government/High-Assurance (NSA Suite B compatible)
    /// - Sign: ECDSA P-384
    /// - Encrypt: AES-256-GCM
    /// - Hash: SHA-384
    ///
    /// Required for some government/defense applications
    GovHighAssurance = 0x04,

    /// Legacy support (deprecated, load-only)
    /// - Sign: RSA-2048
    /// - Encrypt: AES-128-CBC
    ///
    /// WARNING: Do not create new files with this suite
    LegacyRSA = 0xFF,
}

impl CipherSuite {
    /// Convert from u8 value with validation
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::Standard2025),
            0x02 => Some(Self::PostQuantum2030),
            0x03 => Some(Self::Hybrid2028),
            0x04 => Some(Self::GovHighAssurance),
            0xFF => Some(Self::LegacyRSA),
            _ => None,
        }
    }

    /// Convert to u8 value
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Check if this suite is considered secure (not deprecated)
    #[must_use]
    pub const fn is_secure(&self) -> bool {
        !matches!(self, Self::LegacyRSA)
    }

    /// Check if this suite provides post-quantum resistance
    #[must_use]
    pub const fn is_post_quantum(&self) -> bool {
        matches!(self, Self::PostQuantum2030 | Self::Hybrid2028)
    }

    /// Check if this suite is government-approved
    #[must_use]
    pub const fn is_government_approved(&self) -> bool {
        matches!(self, Self::GovHighAssurance)
    }

    /// Check if this suite is deprecated
    #[must_use]
    pub const fn is_deprecated(&self) -> bool {
        matches!(self, Self::LegacyRSA)
    }

    /// Get signature size in bytes
    #[must_use]
    pub const fn signature_size(&self) -> usize {
        match self {
            Self::Standard2025 => 64,      // Ed25519
            Self::PostQuantum2030 => 3293, // ML-DSA-65
            Self::Hybrid2028 => 64 + 3293, // Ed25519 + ML-DSA-65
            Self::GovHighAssurance => 96,  // ECDSA P-384
            Self::LegacyRSA => 256,        // RSA-2048
        }
    }

    /// Get public key size in bytes
    #[must_use]
    pub const fn public_key_size(&self) -> usize {
        match self {
            Self::Standard2025 => 32,      // Ed25519
            Self::PostQuantum2030 => 1952, // ML-DSA-65
            Self::Hybrid2028 => 32 + 1952, // Ed25519 + ML-DSA-65
            Self::GovHighAssurance => 97,  // ECDSA P-384 (uncompressed)
            Self::LegacyRSA => 256,        // RSA-2048 public key
        }
    }

    /// Get encryption key size in bytes
    #[must_use]
    pub const fn encryption_key_size(&self) -> usize {
        match self {
            // 32-byte keys for modern suites
            Self::Standard2025
            | Self::PostQuantum2030
            | Self::Hybrid2028
            | Self::GovHighAssurance => 32,
            Self::LegacyRSA => 16, // AES-128
        }
    }

    /// Get hash output size in bytes
    #[must_use]
    pub const fn hash_size(&self) -> usize {
        match self {
            Self::Standard2025 | Self::PostQuantum2030 | Self::Hybrid2028 | Self::LegacyRSA => 32,
            Self::GovHighAssurance => 48, // SHA-384
        }
    }

    /// Get human-readable name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Standard2025 => "Standard2025 (Ed25519 + XChaCha20)",
            Self::PostQuantum2030 => "PostQuantum2030 (ML-DSA-65 + ML-KEM-768)",
            Self::Hybrid2028 => "Hybrid2028 (Ed25519 + ML-DSA-65)",
            Self::GovHighAssurance => "GovHighAssurance (ECDSA P-384 + AES-256)",
            Self::LegacyRSA => "LegacyRSA (RSA-2048 + AES-128) [DEPRECATED]",
        }
    }

    /// Get signature algorithm name
    #[must_use]
    pub const fn signature_algorithm(&self) -> &'static str {
        match self {
            Self::Standard2025 => "Ed25519",
            Self::PostQuantum2030 => "ML-DSA-65 (Dilithium3)",
            Self::Hybrid2028 => "Ed25519 + ML-DSA-65",
            Self::GovHighAssurance => "ECDSA P-384",
            Self::LegacyRSA => "RSA-2048-PKCS1v15",
        }
    }

    /// Get encryption algorithm name
    #[must_use]
    pub const fn encryption_algorithm(&self) -> &'static str {
        match self {
            Self::Standard2025 => "XChaCha20-Poly1305",
            Self::PostQuantum2030 => "ML-KEM-768 + AES-256-GCM",
            Self::Hybrid2028 => "X25519 + ML-KEM-768 + AES-256-GCM",
            Self::GovHighAssurance => "AES-256-GCM",
            Self::LegacyRSA => "AES-128-CBC-HMAC-SHA256",
        }
    }

    /// Get hash algorithm name
    #[must_use]
    pub const fn hash_algorithm(&self) -> &'static str {
        match self {
            Self::Standard2025 => "BLAKE3",
            Self::PostQuantum2030 | Self::Hybrid2028 => "SHA3-256",
            Self::GovHighAssurance => "SHA-384",
            Self::LegacyRSA => "SHA-256",
        }
    }

    /// Get KDF algorithm name
    #[must_use]
    pub const fn kdf_algorithm(&self) -> &'static str {
        match self {
            Self::Standard2025 => "Argon2id",
            Self::PostQuantum2030 | Self::Hybrid2028 => "HKDF-SHA3",
            Self::GovHighAssurance => "HKDF-SHA384",
            Self::LegacyRSA => "PBKDF2-SHA256",
        }
    }

    /// Get recommended migration target
    #[must_use]
    pub const fn migration_target(&self) -> Option<Self> {
        match self {
            Self::LegacyRSA => Some(Self::Standard2025),
            Self::Standard2025 => Some(Self::Hybrid2028),
            Self::Hybrid2028 => Some(Self::PostQuantum2030),
            // GovHighAssurance stays on gov suite; PostQuantum2030 is already post-quantum
            Self::GovHighAssurance | Self::PostQuantum2030 => None,
        }
    }

    /// Get security level in bits (classical equivalent)
    #[must_use]
    pub const fn security_level_bits(&self) -> u32 {
        match self {
            Self::Standard2025 => 128,
            // 192-bit security for post-quantum/hybrid/gov suites
            Self::PostQuantum2030 | Self::Hybrid2028 | Self::GovHighAssurance => 192,
            Self::LegacyRSA => 112, // RSA-2048 equivalent
        }
    }

    /// Check if suite meets minimum security level
    #[must_use]
    pub const fn meets_security_level(&self, min_bits: u32) -> bool {
        self.security_level_bits() >= min_bits
    }

    /// Get all available suites
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Standard2025,
            Self::PostQuantum2030,
            Self::Hybrid2028,
            Self::GovHighAssurance,
            Self::LegacyRSA,
        ]
    }

    /// Get recommended suites for new deployments
    #[must_use]
    pub const fn recommended() -> &'static [Self] {
        &[
            Self::Standard2025,
            Self::Hybrid2028,
            Self::PostQuantum2030,
            Self::GovHighAssurance,
        ]
    }
}

/// Cipher suite capability flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Capabilities struct legitimately has independent booleans
pub struct CipherCapabilities {
    /// Suite identifier
    pub suite: CipherSuite,
    /// Can sign models
    pub can_sign: bool,
    /// Can encrypt models
    pub can_encrypt: bool,
    /// Can derive keys from passwords
    pub can_derive_keys: bool,
    /// Supports hardware acceleration
    pub has_hw_accel: bool,
}

impl CipherCapabilities {
    /// Get capabilities for a cipher suite
    #[must_use]
    pub const fn for_suite(suite: CipherSuite) -> Self {
        Self {
            suite,
            can_sign: true,
            can_encrypt: true,
            can_derive_keys: true,
            has_hw_accel: matches!(
                suite,
                CipherSuite::Standard2025 | CipherSuite::GovHighAssurance
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cipher_suite_from_u8() {
        assert_eq!(CipherSuite::from_u8(0x01), Some(CipherSuite::Standard2025));
        assert_eq!(
            CipherSuite::from_u8(0x02),
            Some(CipherSuite::PostQuantum2030)
        );
        assert_eq!(CipherSuite::from_u8(0x03), Some(CipherSuite::Hybrid2028));
        assert_eq!(
            CipherSuite::from_u8(0x04),
            Some(CipherSuite::GovHighAssurance)
        );
        assert_eq!(CipherSuite::from_u8(0xFF), Some(CipherSuite::LegacyRSA));
        assert_eq!(CipherSuite::from_u8(0x00), None);
        assert_eq!(CipherSuite::from_u8(0x10), None);
    }

    #[test]
    fn test_cipher_suite_to_u8_roundtrip() {
        for suite in CipherSuite::all() {
            let value = suite.to_u8();
            assert_eq!(CipherSuite::from_u8(value), Some(*suite));
        }
    }

    #[test]
    fn test_cipher_suite_security() {
        assert!(CipherSuite::Standard2025.is_secure());
        assert!(CipherSuite::PostQuantum2030.is_secure());
        assert!(CipherSuite::Hybrid2028.is_secure());
        assert!(CipherSuite::GovHighAssurance.is_secure());
        assert!(!CipherSuite::LegacyRSA.is_secure());
    }

    #[test]
    fn test_cipher_suite_post_quantum() {
        assert!(!CipherSuite::Standard2025.is_post_quantum());
        assert!(CipherSuite::PostQuantum2030.is_post_quantum());
        assert!(CipherSuite::Hybrid2028.is_post_quantum());
        assert!(!CipherSuite::GovHighAssurance.is_post_quantum());
        assert!(!CipherSuite::LegacyRSA.is_post_quantum());
    }

    #[test]
    fn test_cipher_suite_deprecated() {
        assert!(!CipherSuite::Standard2025.is_deprecated());
        assert!(CipherSuite::LegacyRSA.is_deprecated());
    }

    #[test]
    fn test_cipher_suite_signature_sizes() {
        assert_eq!(CipherSuite::Standard2025.signature_size(), 64);
        assert_eq!(CipherSuite::PostQuantum2030.signature_size(), 3293);
        assert_eq!(CipherSuite::Hybrid2028.signature_size(), 64 + 3293);
        assert_eq!(CipherSuite::GovHighAssurance.signature_size(), 96);
        assert_eq!(CipherSuite::LegacyRSA.signature_size(), 256);
    }

    #[test]
    fn test_cipher_suite_public_key_sizes() {
        assert_eq!(CipherSuite::Standard2025.public_key_size(), 32);
        assert!(CipherSuite::PostQuantum2030.public_key_size() > 1000);
        assert!(CipherSuite::Hybrid2028.public_key_size() > 1000);
    }

    #[test]
    fn test_cipher_suite_names() {
        assert!(!CipherSuite::Standard2025.name().is_empty());
        assert!(CipherSuite::LegacyRSA.name().contains("DEPRECATED"));
    }

    #[test]
    fn test_cipher_suite_algorithms() {
        assert_eq!(CipherSuite::Standard2025.signature_algorithm(), "Ed25519");
        assert!(CipherSuite::PostQuantum2030
            .signature_algorithm()
            .contains("ML-DSA"));
        assert_eq!(
            CipherSuite::Standard2025.encryption_algorithm(),
            "XChaCha20-Poly1305"
        );
        assert_eq!(CipherSuite::Standard2025.hash_algorithm(), "BLAKE3");
        assert_eq!(CipherSuite::Standard2025.kdf_algorithm(), "Argon2id");
    }

    #[test]
    fn test_cipher_suite_migration_target() {
        assert_eq!(
            CipherSuite::LegacyRSA.migration_target(),
            Some(CipherSuite::Standard2025)
        );
        assert_eq!(
            CipherSuite::Standard2025.migration_target(),
            Some(CipherSuite::Hybrid2028)
        );
        assert_eq!(
            CipherSuite::Hybrid2028.migration_target(),
            Some(CipherSuite::PostQuantum2030)
        );
        assert_eq!(CipherSuite::PostQuantum2030.migration_target(), None);
    }

    #[test]
    fn test_cipher_suite_security_level() {
        assert_eq!(CipherSuite::Standard2025.security_level_bits(), 128);
        assert_eq!(CipherSuite::PostQuantum2030.security_level_bits(), 192);
        assert_eq!(CipherSuite::LegacyRSA.security_level_bits(), 112);
    }

    #[test]
    fn test_cipher_suite_meets_security_level() {
        assert!(CipherSuite::Standard2025.meets_security_level(128));
        assert!(!CipherSuite::Standard2025.meets_security_level(192));
        assert!(CipherSuite::PostQuantum2030.meets_security_level(192));
    }

    #[test]
    fn test_cipher_suite_all() {
        let all = CipherSuite::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_cipher_suite_recommended() {
        let recommended = CipherSuite::recommended();
        assert!(!recommended.contains(&CipherSuite::LegacyRSA));
        assert!(recommended.contains(&CipherSuite::Standard2025));
    }

    #[test]
    fn test_cipher_capabilities() {
        let caps = CipherCapabilities::for_suite(CipherSuite::Standard2025);
        assert!(caps.can_sign);
        assert!(caps.can_encrypt);
        assert!(caps.has_hw_accel);
    }

    #[test]
    fn test_cipher_suite_default() {
        assert_eq!(CipherSuite::default(), CipherSuite::Standard2025);
    }

    #[test]
    fn test_cipher_suite_government_approved() {
        assert!(CipherSuite::GovHighAssurance.is_government_approved());
        assert!(!CipherSuite::Standard2025.is_government_approved());
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_cipher_suite_encryption_key_sizes() {
        assert_eq!(CipherSuite::Standard2025.encryption_key_size(), 32);
        assert_eq!(CipherSuite::PostQuantum2030.encryption_key_size(), 32);
        assert_eq!(CipherSuite::Hybrid2028.encryption_key_size(), 32);
        assert_eq!(CipherSuite::GovHighAssurance.encryption_key_size(), 32);
        assert_eq!(CipherSuite::LegacyRSA.encryption_key_size(), 16);
    }

    #[test]
    fn test_cipher_suite_hash_sizes() {
        assert_eq!(CipherSuite::Standard2025.hash_size(), 32);
        assert_eq!(CipherSuite::PostQuantum2030.hash_size(), 32);
        assert_eq!(CipherSuite::Hybrid2028.hash_size(), 32);
        assert_eq!(CipherSuite::GovHighAssurance.hash_size(), 48);
        assert_eq!(CipherSuite::LegacyRSA.hash_size(), 32);
    }

    #[test]
    fn test_cipher_suite_all_algorithms() {
        for suite in CipherSuite::all() {
            assert!(!suite.name().is_empty());
            assert!(!suite.signature_algorithm().is_empty());
            assert!(!suite.encryption_algorithm().is_empty());
            assert!(!suite.hash_algorithm().is_empty());
            assert!(!suite.kdf_algorithm().is_empty());
        }
    }

    #[test]
    fn test_cipher_suite_gov_high_assurance_algorithms() {
        let suite = CipherSuite::GovHighAssurance;
        assert!(suite.signature_algorithm().contains("ECDSA"));
        assert!(suite.encryption_algorithm().contains("AES-256"));
        assert!(suite.hash_algorithm().contains("SHA-384"));
        assert!(suite.kdf_algorithm().contains("HKDF"));
    }

    #[test]
    fn test_cipher_suite_legacy_rsa_algorithms() {
        let suite = CipherSuite::LegacyRSA;
        assert!(suite.signature_algorithm().contains("RSA"));
        assert!(suite.encryption_algorithm().contains("AES-128"));
        assert!(suite.hash_algorithm().contains("SHA-256"));
        assert!(suite.kdf_algorithm().contains("PBKDF2"));
    }

    #[test]
    fn test_cipher_suite_pq_and_hybrid_algorithms() {
        // PostQuantum2030
        assert!(CipherSuite::PostQuantum2030
            .hash_algorithm()
            .contains("SHA3"));
        assert!(CipherSuite::PostQuantum2030
            .kdf_algorithm()
            .contains("SHA3"));

        // Hybrid2028
        assert!(CipherSuite::Hybrid2028.hash_algorithm().contains("SHA3"));
        assert!(CipherSuite::Hybrid2028.kdf_algorithm().contains("SHA3"));
    }

    #[test]
    fn test_cipher_capabilities_all_suites() {
        for suite in CipherSuite::all() {
            let caps = CipherCapabilities::for_suite(*suite);
            assert_eq!(caps.suite, *suite);
            assert!(caps.can_sign);
            assert!(caps.can_encrypt);
            assert!(caps.can_derive_keys);
        }
    }

    #[test]
    fn test_cipher_capabilities_hw_accel() {
        let std_caps = CipherCapabilities::for_suite(CipherSuite::Standard2025);
        assert!(std_caps.has_hw_accel);

        let gov_caps = CipherCapabilities::for_suite(CipherSuite::GovHighAssurance);
        assert!(gov_caps.has_hw_accel);

        let pq_caps = CipherCapabilities::for_suite(CipherSuite::PostQuantum2030);
        assert!(!pq_caps.has_hw_accel);

        let hybrid_caps = CipherCapabilities::for_suite(CipherSuite::Hybrid2028);
        assert!(!hybrid_caps.has_hw_accel);

        let legacy_caps = CipherCapabilities::for_suite(CipherSuite::LegacyRSA);
        assert!(!legacy_caps.has_hw_accel);
    }

    #[test]
    fn test_cipher_suite_debug_clone_copy() {
        let suite = CipherSuite::Standard2025;
        let debug_str = format!("{:?}", suite);
        assert!(debug_str.contains("Standard2025"));

        let cloned = suite.clone();
        let copied = suite;
        assert_eq!(cloned, copied);
    }

    #[test]
    fn test_cipher_capabilities_debug_clone_copy() {
        let caps = CipherCapabilities::for_suite(CipherSuite::Standard2025);
        let debug_str = format!("{:?}", caps);
        assert!(debug_str.contains("CipherCapabilities"));

        let cloned = caps.clone();
        let copied = caps;
        assert_eq!(cloned, copied);
    }

    #[test]
    fn test_cipher_suite_gov_not_post_quantum() {
        // GovHighAssurance should not be marked as post-quantum
        // (it uses classical algorithms)
        assert!(!CipherSuite::GovHighAssurance.is_post_quantum());
        assert!(CipherSuite::GovHighAssurance.is_government_approved());
    }

    #[test]
    fn test_cipher_suite_gov_migration() {
        // GovHighAssurance has no migration target (stays on gov suite)
        assert_eq!(CipherSuite::GovHighAssurance.migration_target(), None);
    }
}
