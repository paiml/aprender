pub(crate) use super::*;

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
