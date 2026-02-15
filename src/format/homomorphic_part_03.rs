
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
