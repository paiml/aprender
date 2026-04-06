
#[cfg(test)]
mod tests {
    use super::*;

    // ===== EXTREME TDD: LoadError Tests =====

    #[test]
    fn test_load_error_format_error() {
        let err = LoadError::FormatError(FormatError::UnknownFormat);
        assert!(err.to_string().contains("Format detection error"));
        assert!(err.to_string().contains("Unknown"));
    }

    #[test]
    fn test_load_error_io_error() {
        let err = LoadError::IoError("file not found".to_string());
        assert!(err.to_string().contains("I/O error"));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_load_error_parse_error() {
        let err = LoadError::ParseError("invalid header".to_string());
        assert!(err.to_string().contains("Parse error"));
        assert!(err.to_string().contains("invalid header"));
    }

    #[test]
    fn test_load_error_unsupported_type() {
        let err = LoadError::UnsupportedType("UnknownModel".to_string());
        assert!(err.to_string().contains("Unsupported model type"));
        assert!(err.to_string().contains("UnknownModel"));
    }

    #[test]
    fn test_load_error_integrity_error() {
        let err = LoadError::IntegrityError("CRC32 mismatch".to_string());
        assert!(err.to_string().contains("Integrity check failed"));
        assert!(err.to_string().contains("CRC32"));
    }

    #[test]
    fn test_load_error_type_mismatch() {
        let err = LoadError::TypeMismatch {
            expected: "LogisticRegression".to_string(),
            actual: "DecisionTree".to_string(),
        };
        assert!(err.to_string().contains("type mismatch"));
        assert!(err.to_string().contains("LogisticRegression"));
        assert!(err.to_string().contains("DecisionTree"));
    }

    #[test]
    fn test_load_error_from_format_error() {
        let format_err = FormatError::TooShort { len: 3 };
        let load_err: LoadError = format_err.into();
        assert!(matches!(load_err, LoadError::FormatError(_)));
    }

    // ===== EXTREME TDD: ModelMetadata Tests =====

    #[test]
    fn test_model_metadata_new() {
        let meta = ModelMetadata::new(ModelFormat::Apr);
        assert_eq!(meta.format, ModelFormat::Apr);
        assert!(meta.model_type.is_none());
        assert!(meta.version.is_none());
        assert!(meta.input_dim.is_none());
        assert!(meta.output_dim.is_none());
        assert_eq!(meta.file_size, 0);
    }

    #[test]
    fn test_model_metadata_with_model_type() {
        let meta = ModelMetadata::new(ModelFormat::Apr).with_model_type("LogisticRegression");
        assert_eq!(meta.model_type, Some("LogisticRegression".to_string()));
    }

    #[test]
    fn test_model_metadata_with_version() {
        let meta = ModelMetadata::new(ModelFormat::Gguf).with_version("v1.0.0");
        assert_eq!(meta.version, Some("v1.0.0".to_string()));
    }

    #[test]
    fn test_model_metadata_with_input_dim() {
        let meta = ModelMetadata::new(ModelFormat::SafeTensors).with_input_dim(784);
        assert_eq!(meta.input_dim, Some(784));
    }

    #[test]
    fn test_model_metadata_with_output_dim() {
        let meta = ModelMetadata::new(ModelFormat::Apr).with_output_dim(10);
        assert_eq!(meta.output_dim, Some(10));
    }

    #[test]
    fn test_model_metadata_with_file_size() {
        let meta = ModelMetadata::new(ModelFormat::Gguf).with_file_size(1_000_000);
        assert_eq!(meta.file_size, 1_000_000);
    }

    #[test]
    fn test_model_metadata_chained_builders() {
        let meta = ModelMetadata::new(ModelFormat::Apr)
            .with_model_type("RandomForest")
            .with_version("v2.1")
            .with_input_dim(128)
            .with_output_dim(3)
            .with_file_size(50_000);

        assert_eq!(meta.format, ModelFormat::Apr);
        assert_eq!(meta.model_type, Some("RandomForest".to_string()));
        assert_eq!(meta.version, Some("v2.1".to_string()));
        assert_eq!(meta.input_dim, Some(128));
        assert_eq!(meta.output_dim, Some(3));
        assert_eq!(meta.file_size, 50_000);
    }

    // ===== EXTREME TDD: detect_model_from_bytes Tests =====

    #[test]
    fn test_detect_model_from_bytes_apr() {
        // F-COV-95: APR v1 uses "APRN" magic, v2 uses "APR\0"
        let mut data = b"APRN".to_vec();
        data.extend_from_slice(&[0x02, 0x00, 0x01, 0x00]); // LogisticRegression type
        data.extend_from_slice(&[0u8; 100]); // Padding

        let meta = detect_model_from_bytes(&data).expect("Should detect APR");
        assert_eq!(meta.format, ModelFormat::Apr);
        assert_eq!(meta.file_size, 108);
    }

    #[test]
    fn test_detect_model_from_bytes_gguf() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&[0u8; 100]); // Padding

        let meta = detect_model_from_bytes(&data).expect("Should detect GGUF");
        assert_eq!(meta.format, ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_model_from_bytes_safetensors() {
        let header_size: u64 = 100;
        let mut data = header_size.to_le_bytes().to_vec();
        data.extend_from_slice(&[0u8; 200]);

        let meta = detect_model_from_bytes(&data).expect("Should detect SafeTensors");
        assert_eq!(meta.format, ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_model_from_bytes_too_small() {
        let data = b"APR";
        let result = detect_model_from_bytes(data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LoadError::ParseError(_)));
    }

    // ===== EXTREME TDD: read_apr_model_type Tests =====

    #[test]
    fn test_read_apr_model_type_linear_regression() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0001u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("LinearRegression".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_logistic_regression() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0002u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("LogisticRegression".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_decision_tree() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0003u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("DecisionTree".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_random_forest() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0004u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("RandomForest".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_gradient_boosting() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0005u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("GradientBoosting".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_kmeans() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0006u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("KMeans".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_pca() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0007u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("PCA".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_naive_bayes() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0008u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("NaiveBayes".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_knn() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0009u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("KNN".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_svm() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x000Au16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("SVM".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_ngram_lm() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0010u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("NgramLM".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_tfidf() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0011u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("TFIDF".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_count_vectorizer() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0012u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("CountVectorizer".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_neural_sequential() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0020u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("NeuralSequential".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_neural_custom() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0021u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("NeuralCustom".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_content_recommender() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0030u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("ContentRecommender".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_mixture_of_experts() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0040u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("MixtureOfExperts".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_custom() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x00FFu16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("Custom".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_unknown() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // Unknown type
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), None);
    }

    #[test]
    fn test_read_apr_model_type_too_short() {
        let data = b"APR\0"; // Only 4 bytes
        assert_eq!(read_apr_model_type(data), None);
    }

    // ===== EXTREME TDD: validate_model_type Tests =====

    #[test]
    fn test_validate_model_type_match() {
        let result = validate_model_type("LogisticRegression", "LogisticRegression");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_type_mismatch() {
        let result = validate_model_type("LogisticRegression", "DecisionTree");
        assert!(result.is_err());

        if let Err(LoadError::TypeMismatch { expected, actual }) = result {
            assert_eq!(expected, "LogisticRegression");
            assert_eq!(actual, "DecisionTree");
        } else {
            panic!("Expected TypeMismatch error");
        }
    }

    #[test]
    fn test_validate_model_type_case_sensitive() {
        // Type names are case-sensitive
        let result = validate_model_type("logisticregression", "LogisticRegression");
        assert!(result.is_err());
    }

    // ===== EXTREME TDD: Integration Tests =====

    #[test]
    fn test_detect_and_extract_apr_type() {
        // Simulate APR file with LogisticRegression type
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
        data.extend_from_slice(&[0, 0]); // version placeholder
        data.extend_from_slice(&[0u8; 100]); // Padding

        let meta = detect_model_from_bytes(&data).expect("Detection should succeed");
        assert_eq!(meta.format, ModelFormat::Apr);

        let model_type = read_apr_model_type(&data).expect("Should extract model type");
        assert_eq!(model_type, "LogisticRegression");
    }

    #[test]
    fn test_full_metadata_extraction() {
        let mut data = b"APRN".to_vec(); // F-COV-95: APR v1 uses APRN magic
        data.extend_from_slice(&0x0004u16.to_le_bytes()); // RandomForest
        data.extend_from_slice(&[0, 0]);
        data.extend_from_slice(&[0u8; 500]);

        let meta = detect_model_from_bytes(&data)
            .expect("Detection should succeed")
            .with_model_type(read_apr_model_type(&data).unwrap_or_default())
            .with_version("v1.0")
            .with_input_dim(128);

        assert_eq!(meta.format, ModelFormat::Apr);
        assert_eq!(meta.model_type, Some("RandomForest".to_string()));
        assert_eq!(meta.version, Some("v1.0".to_string()));
        assert_eq!(meta.input_dim, Some(128));
        assert_eq!(meta.file_size, 508);
    }

    // ===== EXTREME TDD: Debug/Error Trait Tests =====

    #[test]
    fn test_load_error_debug() {
        let err = LoadError::IoError("test".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("IoError"));
    }

    #[test]
    fn test_model_metadata_debug() {
        let meta = ModelMetadata::new(ModelFormat::Apr);
        let debug_str = format!("{meta:?}");
        assert!(debug_str.contains("Apr"));
    }

    #[test]
    fn test_model_metadata_clone() {
        let meta = ModelMetadata::new(ModelFormat::Gguf)
            .with_model_type("LLM")
            .with_file_size(1000);
        let cloned = meta.clone();

        assert_eq!(cloned.format, ModelFormat::Gguf);
        assert_eq!(cloned.model_type, Some("LLM".to_string()));
        assert_eq!(cloned.file_size, 1000);
    }

    #[test]
    fn test_load_error_clone() {
        let err = LoadError::ParseError("test".to_string());
        let cloned = err.clone();
        assert!(matches!(cloned, LoadError::ParseError(_)));
    }
}
