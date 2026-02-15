
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_test_data_creation() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));

        assert_eq!(data.n_samples(), 3);
        assert_eq!(data.n_features(), 2);
        assert_eq!(data.size_bytes(), 24); // 6 floats * 4 bytes
    }

    #[test]
    fn test_embedded_test_data_with_targets() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
            .with_targets(vec![0.0, 1.0, 0.0]);

        assert_eq!(data.y_data, Some(vec![0.0, 1.0, 0.0]));
        assert_eq!(data.size_bytes(), 36); // 6 + 3 floats * 4 bytes
    }

    #[test]
    fn test_embedded_test_data_with_feature_names() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
            .with_feature_names(vec!["a".into(), "b".into()]);

        assert_eq!(data.feature_names, Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_embedded_test_data_get_row() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));

        assert_eq!(data.get_row(0), Some(&[1.0, 2.0][..]));
        assert_eq!(data.get_row(1), Some(&[3.0, 4.0][..]));
        assert_eq!(data.get_row(2), Some(&[5.0, 6.0][..]));
        assert_eq!(data.get_row(3), None);
    }

    #[test]
    fn test_embedded_test_data_get_target() {
        let data =
            EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).with_targets(vec![0.0, 1.0]);

        assert_eq!(data.get_target(0), Some(0.0));
        assert_eq!(data.get_target(1), Some(1.0));
        assert_eq!(data.get_target(2), None);
    }

    #[test]
    fn test_embedded_test_data_validate() {
        // Valid data
        let valid = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        assert!(valid.validate().is_ok());

        // Invalid: contains NaN
        let mut invalid_nan = EmbeddedTestData::new(vec![1.0, f32::NAN, 3.0, 4.0], (2, 2));
        invalid_nan.x_data[1] = f32::NAN;
        let err = invalid_nan.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));

        // Invalid: contains Inf
        let mut invalid_inf = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        invalid_inf.x_data[0] = f32::INFINITY;
        let err = invalid_inf.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));
    }

    #[test]
    fn test_data_provenance() {
        let provenance = DataProvenance::new("UCI Iris")
            .with_subset("first 50 samples")
            .with_preprocessing("normalize")
            .with_preprocessing("pca")
            .with_license("CC0")
            .with_version("1.0")
            .with_metadata("author", "Fisher");

        assert_eq!(provenance.source, "UCI Iris");
        assert_eq!(provenance.subset_criteria, Some("first 50 samples".into()));
        assert_eq!(provenance.preprocessing, vec!["normalize", "pca"]);
        assert_eq!(provenance.license, Some("CC0".into()));
        assert_eq!(provenance.version, Some("1.0".into()));
        assert_eq!(provenance.metadata.get("author"), Some(&"Fisher".into()));
        assert!(provenance.is_complete());
    }

    #[test]
    fn test_data_provenance_incomplete() {
        let incomplete = DataProvenance::new("test");
        assert!(!incomplete.is_complete()); // missing license

        let complete = DataProvenance::new("test").with_license("MIT");
        assert!(complete.is_complete());
    }

    #[test]
    fn test_data_compression_none() {
        let comp = DataCompression::None;
        assert_eq!(comp.name(), "none");
        assert!((comp.estimated_ratio() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_data_compression_zstd() {
        let comp = DataCompression::zstd();
        assert_eq!(comp.name(), "zstd");
        assert!(comp.estimated_ratio() > 1.0);

        let comp_high = DataCompression::zstd_level(15);
        assert!(comp_high.estimated_ratio() > comp.estimated_ratio());
    }

    #[test]
    fn test_data_compression_delta_zstd() {
        let comp = DataCompression::delta_zstd();
        assert_eq!(comp.name(), "delta-zstd");
        assert!(comp.estimated_ratio() > DataCompression::zstd().estimated_ratio());
    }

    #[test]
    fn test_data_compression_quantized() {
        let comp_4bit = DataCompression::quantized(4);
        let comp_8bit = DataCompression::quantized(8);

        assert_eq!(comp_4bit.name(), "quantized-entropy");
        assert!(comp_4bit.estimated_ratio() > comp_8bit.estimated_ratio());
    }

    #[test]
    fn test_data_compression_sparse() {
        let comp = DataCompression::sparse(0.001);
        assert_eq!(comp.name(), "sparse");
        assert!(comp.estimated_ratio() > 1.0);
    }

    #[test]
    fn test_embed_error_display() {
        let err = EmbedError::ShapeMismatch {
            expected: 100,
            actual: 50,
        };
        let msg = format!("{err}");
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));

        let err = EmbedError::InvalidValue {
            index: 5,
            value: f32::NAN,
        };
        let msg = format!("{err}");
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_embedded_test_data_with_provenance() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2))
            .with_provenance(DataProvenance::new("test").with_license("MIT"));

        assert!(data.provenance.is_some());
        assert!(data.provenance.as_ref().unwrap().is_complete());
    }

    #[test]
    fn test_embedded_test_data_with_sample_ids() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2))
            .with_sample_ids(vec!["sample_1".into(), "sample_2".into()]);

        assert_eq!(
            data.sample_ids,
            Some(vec!["sample_1".into(), "sample_2".into()])
        );
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_embedded_test_data_shape_mismatch_panics() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0], (2, 2)); // 3 != 4
    }

    #[test]
    #[should_panic(expected = "Target length")]
    fn test_embedded_test_data_target_mismatch_panics() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2))
            .with_targets(vec![0.0, 1.0, 2.0]); // 3 != 2
    }

    #[test]
    fn test_embedded_test_data_default() {
        let data = EmbeddedTestData::default();
        assert_eq!(data.n_samples(), 0);
        assert_eq!(data.n_features(), 0);
        assert!(data.x_data.is_empty());
    }

    #[test]
    fn test_data_compression_default() {
        let comp = DataCompression::default();
        assert_eq!(comp, DataCompression::None);
    }

    #[test]
    fn test_embedded_test_data_with_compression() {
        let data =
            EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_compression(DataCompression::zstd());

        assert_eq!(data.compression, DataCompression::Zstd { level: 3 });
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_data_provenance_default() {
        let prov = DataProvenance::default();
        assert_eq!(prov.source, "unknown");
        assert!(!prov.is_complete()); // no license
    }

    #[test]
    fn test_data_provenance_with_preprocessing_steps() {
        let prov = DataProvenance::new("test")
            .with_preprocessing_steps(vec!["step1".into(), "step2".into()]);
        assert_eq!(prov.preprocessing.len(), 2);
    }

    #[test]
    fn test_data_compression_delta_zstd_high_level() {
        let comp = DataCompression::DeltaZstd { level: 10 };
        assert_eq!(comp.name(), "delta-zstd");
        assert!((comp.estimated_ratio() - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_data_compression_zstd_high_level() {
        let comp = DataCompression::zstd_level(12);
        assert!((comp.estimated_ratio() - 6.0).abs() < 0.1);

        let comp_medium = DataCompression::zstd_level(7);
        assert!((comp_medium.estimated_ratio() - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_data_compression_quantized_16bit() {
        let comp = DataCompression::quantized(16);
        assert!((comp.estimated_ratio() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_embed_error_compression_failed_display() {
        let err = EmbedError::CompressionFailed {
            strategy: "zstd",
            message: "out of memory".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("zstd"));
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn test_embed_error_decompression_failed_display() {
        let err = EmbedError::DecompressionFailed {
            strategy: "delta-zstd",
            message: "corrupt data".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("delta-zstd"));
        assert!(msg.contains("corrupt data"));
    }

    #[test]
    fn test_embed_error_target_mismatch_display() {
        let err = EmbedError::TargetMismatch {
            expected: 10,
            actual: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_embedded_test_data_validate_target_nan() {
        let mut data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_targets(vec![0.0]);
        data.y_data = Some(vec![f32::NAN]);

        let err = data.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));
    }

    #[test]
    fn test_embedded_test_data_validate_target_inf() {
        let mut data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_targets(vec![0.0]);
        data.y_data = Some(vec![f32::INFINITY]);

        let err = data.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));
    }

    #[test]
    fn test_embedded_test_data_validate_target_mismatch() {
        let mut data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        // Manually set mismatched targets
        data.y_data = Some(vec![0.0, 1.0, 2.0]); // 3 targets for 2 samples

        let err = data.validate();
        assert!(matches!(err, Err(EmbedError::TargetMismatch { .. })));
    }

    #[test]
    fn test_embedded_test_data_clone() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_targets(vec![1.0]);
        let cloned = data.clone();
        assert_eq!(cloned.x_data, data.x_data);
        assert_eq!(cloned.y_data, data.y_data);
    }

    #[test]
    fn test_data_provenance_clone() {
        let prov = DataProvenance::new("test").with_license("MIT");
        let cloned = prov.clone();
        assert_eq!(cloned.source, prov.source);
        assert_eq!(cloned.license, prov.license);
    }

    #[test]
    fn test_data_compression_copy() {
        let comp = DataCompression::zstd();
        let copied = comp;
        assert_eq!(copied.name(), "zstd");
    }

    #[test]
    fn test_embed_error_clone() {
        let err = EmbedError::ShapeMismatch {
            expected: 10,
            actual: 5,
        };
        let cloned = err.clone();
        let msg = format!("{}", cloned);
        assert!(msg.contains("10"));
    }

    #[test]
    #[should_panic(expected = "Feature names length")]
    fn test_embedded_test_data_feature_names_mismatch() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).with_feature_names(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]); // 3 != 2
    }

    #[test]
    #[should_panic(expected = "Sample IDs length")]
    fn test_embedded_test_data_sample_ids_mismatch() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).with_sample_ids(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]); // 3 != 2
    }

    #[test]
    fn test_embedded_test_data_partial_eq() {
        let data1 = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2));
        let data2 = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2));
        let data3 = EmbeddedTestData::new(vec![1.0, 3.0], (1, 2));

        assert_eq!(data1, data2);
        assert_ne!(data1, data3);
    }

    #[test]
    fn test_data_provenance_partial_eq() {
        let prov1 = DataProvenance::new("test");
        let prov2 = DataProvenance::new("test");
        let prov3 = DataProvenance::new("other");

        assert_eq!(prov1.source, prov2.source);
        assert_ne!(prov1.source, prov3.source);
    }
}
