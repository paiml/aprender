
#[cfg(test)]
mod tests {
    use super::*;

    // ----- detect_format_from_path edge cases -----

    #[test]
    fn test_detect_format_from_path_no_extension() {
        let path = Path::new("model");
        let result = detect_format_from_path(path);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_dot_only() {
        let path = Path::new("model.");
        let result = detect_format_from_path(path);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_double_extension() {
        // Only the last extension is checked
        let path = Path::new("model.tar.gguf");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_format_from_path_uppercase_gguf() {
        let path = Path::new("MODEL.GGUF");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_format_from_path_uppercase_safetensors() {
        let path = Path::new("MODEL.SAFETENSORS");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_from_path_mixed_case_apr() {
        let path = Path::new("model.ApR");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Apr
        );
    }

    // ----- detect_and_verify_format edge cases -----

    #[test]
    fn test_detect_and_verify_gguf_match() {
        let path = Path::new("model.gguf");
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_and_verify_safetensors_match() {
        let path = Path::new("model.safetensors");
        let mut data = vec![0u8; 16];
        let header_size: u64 = 500;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        assert_eq!(
            detect_and_verify_format(path, &data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_and_verify_gguf_as_safetensors_mismatch() {
        let path = Path::new("model.safetensors");
        let data = b"GGUFxxxxxxxxxxxxxxxx";
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
    fn test_detect_and_verify_apr_as_gguf_mismatch() {
        let path = Path::new("model.gguf");
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        let result = detect_and_verify_format(path, data);
        assert!(matches!(
            result,
            Err(FormatError::ExtensionMismatch {
                detected: ModelFormat::Apr,
                ..
            })
        ));
    }

    #[test]
    fn test_detect_and_verify_no_extension_data_valid() {
        // No recognized extension, but data is valid GGUF
        let path = Path::new("model.xyz");
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Gguf,
            "Unrecognized extension should not prevent data-based detection"
        );
    }

    #[test]
    fn test_detect_and_verify_data_too_short() {
        let path = Path::new("model.gguf");
        let data = b"GGU"; // Only 3 bytes
        let result = detect_and_verify_format(path, data);
        assert!(matches!(result, Err(FormatError::TooShort { .. })));
    }

    // ----- ModelFormat trait impls -----

    #[test]
    fn test_model_format_copy() {
        let fmt = ModelFormat::Apr;
        let copied = fmt; // Copy
        assert_eq!(fmt, copied);
    }

    #[test]
    fn test_model_format_clone() {
        let fmt = ModelFormat::SafeTensors;
        let cloned = fmt.clone();
        assert_eq!(fmt, cloned);
    }

    #[test]
    fn test_model_format_debug() {
        let debug = format!("{:?}", ModelFormat::Gguf);
        assert_eq!(debug, "Gguf");
    }

    #[test]
    fn test_model_format_eq_all_variants() {
        assert_eq!(ModelFormat::Apr, ModelFormat::Apr);
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_eq!(ModelFormat::SafeTensors, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::Apr, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Apr, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
    }

    // ----- FormatError trait impls -----

    #[test]
    fn test_format_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(FormatError::UnknownFormat);
        assert!(err.to_string().contains("Unknown"));
    }

    #[test]
    fn test_format_error_debug_all_variants() {
        let too_short = format!("{:?}", FormatError::TooShort { len: 3 });
        assert!(too_short.contains("TooShort"));
        assert!(too_short.contains("3"));

        let unknown = format!("{:?}", FormatError::UnknownFormat);
        assert!(unknown.contains("UnknownFormat"));

        let too_large = format!("{:?}", FormatError::HeaderTooLarge { size: 999 });
        assert!(too_large.contains("HeaderTooLarge"));
        assert!(too_large.contains("999"));

        let mismatch = format!(
            "{:?}",
            FormatError::ExtensionMismatch {
                detected: ModelFormat::Apr,
                extension: "gguf".to_string(),
            }
        );
        assert!(mismatch.contains("ExtensionMismatch"));
        assert!(mismatch.contains("Apr"));
        assert!(mismatch.contains("gguf"));
    }

    #[test]
    fn test_format_error_clone_all_variants() {
        let errors: Vec<FormatError> = vec![
            FormatError::TooShort { len: 5 },
            FormatError::UnknownFormat,
            FormatError::HeaderTooLarge { size: 42 },
            FormatError::ExtensionMismatch {
                detected: ModelFormat::Gguf,
                extension: "apr".to_string(),
            },
        ];
        for err in &errors {
            let cloned = err.clone();
            assert_eq!(*err, cloned);
        }
    }

    #[test]
    fn test_format_error_eq() {
        assert_eq!(FormatError::UnknownFormat, FormatError::UnknownFormat);
        assert_eq!(
            FormatError::TooShort { len: 5 },
            FormatError::TooShort { len: 5 }
        );
        assert_ne!(
            FormatError::TooShort { len: 5 },
            FormatError::TooShort { len: 6 }
        );
        assert_ne!(FormatError::UnknownFormat, FormatError::TooShort { len: 0 });
    }

    // ----- FormatError Display messages are informative -----

    #[test]
    fn test_format_error_too_short_display_message() {
        let err = FormatError::TooShort { len: 0 };
        let msg = err.to_string();
        assert!(
            msg.contains("0 bytes"),
            "should show actual byte count: {}",
            msg
        );
        assert!(
            msg.contains("need 8"),
            "should show required minimum: {}",
            msg
        );
    }

    #[test]
    fn test_format_error_header_too_large_display_message() {
        let err = FormatError::HeaderTooLarge { size: 500_000_000 };
        let msg = err.to_string();
        assert!(
            msg.contains("500000000"),
            "should show actual size: {}",
            msg
        );
        assert!(msg.contains("100MB"), "should mention the limit: {}", msg);
    }

    #[test]
    fn test_format_error_extension_mismatch_display_message() {
        let err = FormatError::ExtensionMismatch {
            detected: ModelFormat::SafeTensors,
            extension: "gguf".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("SafeTensors"),
            "should show detected format: {}",
            msg
        );
        assert!(msg.contains(".gguf"), "should show file extension: {}", msg);
    }

    // ----- APR_VERSIONS coverage -----

    #[test]
    fn test_apr_versions_all_valid() {
        // Test all valid version bytes
        for &version in &APR_VERSIONS {
            let mut data = b"APR\0xxxxxxxxxxxxxxxx".to_vec();
            data[3] = version;
            assert_eq!(
                detect_format(&data).expect("valid APR version should be detected"),
                ModelFormat::Apr,
                "Version byte {} should be valid APR",
                version
            );
        }
    }

    #[test]
    fn test_apr_versions_invalid_adjacent() {
        // Test version bytes adjacent to valid ones that should NOT match
        let invalid_versions: &[u8] = &[b'0', b'3', b'4', b'M', b'O', 1, 255];
        for &version in invalid_versions {
            if APR_VERSIONS.contains(&version) {
                continue;
            }
            let mut data = b"APR\0xxxxxxxxxxxxxxxx".to_vec();
            data[3] = version;
            let result = detect_format(&data);
            if let Ok(fmt) = result {
                assert_ne!(
                    fmt,
                    ModelFormat::Apr,
                    "Invalid version byte {} should NOT be recognized as APR",
                    version
                );
            }
        }
    }

    // ----- SafeTensors header size boundary values -----

    #[test]
    fn test_safetensors_header_size_small_values() {
        for size in 1u64..=10 {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert_eq!(
                result.expect("small header sizes should be valid SafeTensors"),
                ModelFormat::SafeTensors,
                "Header size {} should be valid SafeTensors",
                size
            );
        }
    }

    #[test]
    fn test_safetensors_header_size_typical_values() {
        let typical_sizes: &[u64] = &[256, 1024, 4096, 65536, 1_000_000, 50_000_000];
        for &size in typical_sizes {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert_eq!(
                result.expect("typical header sizes should be valid SafeTensors"),
                ModelFormat::SafeTensors,
                "Header size {} should be valid SafeTensors",
                size
            );
        }
    }
include!("format_detect_apr.rs");
}
