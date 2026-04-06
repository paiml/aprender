
    // ===== EXTREME TDD: Format Detection Tests =====

    #[test]
    fn test_detect_apr_format_legacy() {
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_apr_format_v1() {
        let data = b"APR1xxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_apr_format_v2() {
        let data = b"APR2xxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_gguf_format() {
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_safetensors_format() {
        // SafeTensors: first 8 bytes are header size (little-endian)
        // A reasonable header size like 1000 bytes
        let mut data = vec![0u8; 16];
        let header_size: u64 = 1000;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        assert_eq!(
            detect_format(&data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_too_short() {
        let data = b"APR"; // Only 3 bytes
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::TooShort { len: 3 })));
    }

    #[test]
    fn test_detect_format_empty() {
        let data: &[u8] = &[];
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::TooShort { len: 0 })));
    }

    #[test]
    fn test_detect_safetensors_header_too_large() {
        // Header size > 100MB should fail (DOS protection)
        let mut data = vec![0u8; 16];
        let header_size: u64 = 200_000_000; // 200MB
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        let result = detect_format(&data);
        assert!(matches!(
            result,
            Err(FormatError::HeaderTooLarge { size: 200_000_000 })
        ));
    }

    #[test]
    fn test_detect_unknown_format() {
        // Random bytes that don't match any format
        // Zero header size means not SafeTensors either
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x00xxxx";
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_apr() {
        let path = Path::new("model.apr");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_detect_format_from_path_gguf() {
        let path = Path::new("llama-7b-q4.gguf");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_format_from_path_safetensors() {
        let path = Path::new("model.safetensors");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_from_path_unknown() {
        let path = Path::new("model.bin");
        let result = detect_format_from_path(path);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_uppercase() {
        // Extension comparison should be case-insensitive
        let path = Path::new("MODEL.APR");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_detect_and_verify_format_match() {
        let path = Path::new("model.apr");
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_detect_and_verify_format_mismatch() {
        let path = Path::new("model.apr"); // Says APR
        let data = b"GGUFxxxxxxxxxxxxxxxx"; // But data is GGUF
        let result = detect_and_verify_format(path, data);
        assert!(matches!(
            result,
            Err(FormatError::ExtensionMismatch {
                detected: ModelFormat::Gguf,
                ..
            })
        ));
    }

    #[test]
    fn test_detect_and_verify_unknown_extension_ok() {
        // Unknown extension but valid magic should work
        let path = Path::new("model.bin");
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_model_format_display() {
        assert_eq!(format!("{}", ModelFormat::Apr), "APR");
        assert_eq!(format!("{}", ModelFormat::Gguf), "GGUF");
        assert_eq!(format!("{}", ModelFormat::SafeTensors), "SafeTensors");
    }

    #[test]
    fn test_format_error_display() {
        let err = FormatError::TooShort { len: 5 };
        assert!(err.to_string().contains("5 bytes"));

        let err = FormatError::UnknownFormat;
        assert!(err.to_string().contains("Unknown"));

        let err = FormatError::HeaderTooLarge { size: 999 };
        assert!(err.to_string().contains("999 bytes"));

        let err = FormatError::ExtensionMismatch {
            detected: ModelFormat::Gguf,
            extension: "apr".to_string(),
        };
        assert!(err.to_string().contains("GGUF"));
        assert!(err.to_string().contains(".apr"));
    }

    #[test]
    fn test_magic_constants() {
        // APR_MAGIC is now 3 bytes, version in 4th byte
        assert_eq!(APR_MAGIC, b"APR");
        assert_eq!(GGUF_MAGIC, b"GGUF");
        assert_eq!(MAX_SAFETENSORS_HEADER, 100_000_000);
    }

    // ===== Property-based edge cases =====

    #[test]
    fn test_exactly_8_bytes_safetensors() {
        // Exactly 8 bytes with valid header size
        let header_size: u64 = 500;
        let data = header_size.to_le_bytes();
        assert_eq!(
            detect_format(&data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_apr_with_trailing_data() {
        // APR magic followed by lots of other data
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&[0u8; 1000]);
        assert_eq!(detect_format(&data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_gguf_with_trailing_data() {
        // GGUF magic followed by lots of other data
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&[0u8; 1000]);
        assert_eq!(detect_format(&data).expect("test"), ModelFormat::Gguf);
    }

    #[test]
    fn test_safetensors_boundary_header_size() {
        // Just under the limit
        let mut data = vec![0u8; 16];
        let header_size: u64 = MAX_SAFETENSORS_HEADER - 1;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        assert_eq!(
            detect_format(&data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_safetensors_exactly_at_limit() {
        // Exactly at limit should fail
        let mut data = vec![0u8; 16];
        let header_size: u64 = MAX_SAFETENSORS_HEADER;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        let result = detect_format(&data);
        assert!(matches!(result, Err(FormatError::HeaderTooLarge { .. })));
    }

    // ===== T-COV-95 Phase 60: Extended format detection coverage =====

    // ----- APR version byte coverage -----

    #[test]
    fn test_detect_apr_format_version_n() {
        // APR_VERSIONS includes b'N'
        let data = b"APRNxxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_format(data).expect("APRN should be valid APR"),
            ModelFormat::Apr,
            "Version byte 'N' should be recognized as valid APR"
        );
    }

    #[test]
    fn test_detect_apr_invalid_version_byte() {
        // Version byte 'X' is NOT in APR_VERSIONS [b'N', b'1', b'2', 0]
        let data = b"APRXxxxxxxxxxxxxxxxx".to_vec();
        // This has "APR" prefix but invalid version, should NOT be detected as APR
        // It will fall through to GGUF check (fails), then SafeTensors check
        let result = detect_format(&data);
        assert_ne!(
            result.as_ref().ok().copied(),
            Some(ModelFormat::Apr),
            "Invalid version byte 'X' should not be recognized as APR"
        );
    }

    #[test]
    fn test_detect_apr_version_3_invalid() {
        // b'3' is NOT in APR_VERSIONS
        let data = b"APR3xxxxxxxxxxxxxxxx";
        let result = detect_format(data);
        assert_ne!(
            result.as_ref().ok().copied(),
            Some(ModelFormat::Apr),
            "Version byte '3' should not be recognized as APR"
        );
    }

    // ----- try_detect_apr edge cases -----

    #[test]
    fn test_try_detect_apr_exactly_4_bytes() {
        let data = b"APR\0";
        assert_eq!(try_detect_apr(data), Some(ModelFormat::Apr));
    }

    #[test]
    fn test_try_detect_apr_3_bytes_too_short() {
        let data = b"APR";
        assert_eq!(
            try_detect_apr(data),
            None,
            "3 bytes is not enough for APR detection"
        );
    }

    #[test]
    fn test_try_detect_apr_2_bytes() {
        let data = b"AP";
        assert_eq!(try_detect_apr(data), None);
    }

    #[test]
    fn test_try_detect_apr_empty() {
        let data: &[u8] = &[];
        assert_eq!(try_detect_apr(data), None);
    }

    #[test]
    fn test_try_detect_apr_wrong_magic() {
        let data = b"XPR\0xxxxxxxxx";
        assert_eq!(try_detect_apr(data), None);
    }

    // ----- try_detect_gguf edge cases -----

    #[test]
    fn test_try_detect_gguf_exactly_4_bytes() {
        let data = b"GGUF";
        assert_eq!(try_detect_gguf(data), Some(ModelFormat::Gguf));
    }

    #[test]
    fn test_try_detect_gguf_3_bytes_too_short() {
        let data = b"GGU";
        assert_eq!(
            try_detect_gguf(data),
            None,
            "3 bytes is not enough for GGUF detection"
        );
    }

    #[test]
    fn test_try_detect_gguf_empty() {
        let data: &[u8] = &[];
        assert_eq!(try_detect_gguf(data), None);
    }

    #[test]
    fn test_try_detect_gguf_wrong_magic() {
        let data = b"GGXFxxxxxxxx";
        assert_eq!(try_detect_gguf(data), None);
    }

    #[test]
    fn test_try_detect_gguf_lowercase_rejected() {
        let data = b"ggufxxxxxxxx";
        assert_eq!(
            try_detect_gguf(data),
            None,
            "lowercase gguf should not match"
        );
    }

    // ----- try_detect_safetensors edge cases -----

    #[test]
    fn test_try_detect_safetensors_header_size_zero() {
        let header_size: u64 = 0;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data).expect("should not error");
        assert_eq!(
            result, None,
            "header_size=0 should not be detected as SafeTensors"
        );
    }

    #[test]
    fn test_try_detect_safetensors_header_size_one() {
        let header_size: u64 = 1;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data).expect("should not error");
        assert_eq!(result, Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn test_try_detect_safetensors_header_just_below_max() {
        let header_size: u64 = MAX_SAFETENSORS_HEADER - 1;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data).expect("should not error");
        assert_eq!(result, Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn test_try_detect_safetensors_header_at_max() {
        let header_size: u64 = MAX_SAFETENSORS_HEADER;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data);
        assert!(
            result.is_err(),
            "header_size at MAX should return HeaderTooLarge error"
        );
        assert!(matches!(
            result.unwrap_err(),
            FormatError::HeaderTooLarge { size } if size == MAX_SAFETENSORS_HEADER
        ));
    }

    #[test]
    fn test_try_detect_safetensors_header_above_max() {
        let header_size: u64 = MAX_SAFETENSORS_HEADER + 1;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_detect_safetensors_u64_max() {
        let header_size: u64 = u64::MAX;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data);
        assert!(result.is_err(), "u64::MAX header should be rejected");
    }

    // ----- detect_format priority order -----

    #[test]
    fn test_detect_format_apr_takes_priority_over_safetensors() {
        // APR\0 could also be interpreted as a small header_size in little-endian
        // but APR detection should take priority
        let data = b"APR\0\x00\x00\x00\x00";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_gguf_takes_priority_over_safetensors() {
        // GGUF bytes could be a valid header_size, but GGUF should win
        let data = b"GGUFxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_7_bytes_too_short() {
        let data = b"APR\0xxx";
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::TooShort { len: 7 })));
    }

    #[test]
    fn test_detect_format_exactly_8_bytes_unknown() {
        // 8 zero bytes: header_size=0, not APR, not GGUF -> UnknownFormat
        let data = [0u8; 8];
        let result = detect_format(&data);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }
