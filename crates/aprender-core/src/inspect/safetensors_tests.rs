pub(crate) use super::*;

#[test]
fn test_tensor_data() {
    let data = TensorData {
        name: "test".to_string(),
        shape: vec![2, 3],
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        dtype: "F32".to_string(),
    };

    assert_eq!(data.numel(), 6);
    assert!(!data.is_empty());
    assert!((data.mean() - 3.5).abs() < 0.001);
}

#[test]
fn test_tensor_comparison() {
    let a = TensorData {
        name: "test".to_string(),
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
        dtype: "F32".to_string(),
    };
    let b = vec![1.0, 2.0, 3.0];

    let comp = TensorComparison::compare("test", &a, &b, 1e-5);
    assert!(comp.shape_match);
    assert!(comp.passes_threshold);
    assert!(comp.is_close(1e-5));
}

#[test]
fn test_tensor_comparison_mismatch() {
    let a = TensorData {
        name: "test".to_string(),
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
        dtype: "F32".to_string(),
    };
    let b = vec![1.0, 2.0, 4.0]; // Different!

    let comp = TensorComparison::compare("test", &a, &b, 1e-5);
    assert!(comp.shape_match);
    assert!(!comp.passes_threshold); // Fails threshold
}

#[test]
fn test_batch_comparison() {
    let comparisons = vec![
        TensorComparison {
            name: "a".to_string(),
            shape_match: true,
            shape_a: vec![3],
            shape_b: vec![3],
            weight_diff: Some(WeightDiff::empty()),
            passes_threshold: true,
        },
        TensorComparison {
            name: "b".to_string(),
            shape_match: true,
            shape_a: vec![3],
            shape_b: vec![3],
            weight_diff: Some(WeightDiff::empty()),
            passes_threshold: true,
        },
    ];

    let batch = BatchComparison::from_comparisons(comparisons);
    assert_eq!(batch.total_compared, 2);
    assert_eq!(batch.total_passed, 2);
    assert!(batch.all_passed());
}

#[test]
fn test_safetensors_error_display() {
    let err = SafetensorsError::TensorNotFound("foo".to_string());
    assert!(err.to_string().contains("foo"));
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_safetensors_error_file_not_found() {
    let err = SafetensorsError::FileNotFound("/path/to/file".to_string());
    let msg = err.to_string();
    assert!(msg.contains("File not found"));
    assert!(msg.contains("/path/to/file"));
}

#[test]
fn test_safetensors_error_parse_error() {
    let err = SafetensorsError::ParseError("invalid header".to_string());
    let msg = err.to_string();
    assert!(msg.contains("Parse error"));
    assert!(msg.contains("invalid header"));
}

#[test]
fn test_safetensors_error_download_error() {
    let err = SafetensorsError::DownloadError("network timeout".to_string());
    let msg = err.to_string();
    assert!(msg.contains("Download error"));
    assert!(msg.contains("network timeout"));
}

#[test]
fn test_safetensors_error_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let err = SafetensorsError::IoError(io_err);
    let msg = err.to_string();
    assert!(msg.contains("IO error"));
}

#[test]
fn test_safetensors_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let err: SafetensorsError = io_err.into();
    assert!(matches!(err, SafetensorsError::IoError(_)));
}

#[test]
fn test_safetensors_error_debug() {
    let err = SafetensorsError::TensorNotFound("weight".to_string());
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("TensorNotFound"));
}

#[test]
fn test_tensor_data_empty() {
    let data = TensorData {
        name: "empty".to_string(),
        shape: vec![0],
        data: vec![],
        dtype: "F32".to_string(),
    };

    assert_eq!(data.numel(), 0);
    assert!(data.is_empty());
    assert_eq!(data.mean(), 0.0);
    assert_eq!(data.l2_norm(), 0.0);
}

#[test]
fn test_tensor_data_l2_norm() {
    let data = TensorData {
        name: "test".to_string(),
        shape: vec![3],
        data: vec![3.0, 4.0, 0.0], // 3^2 + 4^2 = 25, sqrt = 5
        dtype: "F32".to_string(),
    };

    assert!((data.l2_norm() - 5.0).abs() < 0.001);
}

#[test]
fn test_tensor_data_clone() {
    let data = TensorData {
        name: "test".to_string(),
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
        dtype: "F32".to_string(),
    };

    let cloned = data.clone();
    assert_eq!(cloned.name, "test");
    assert_eq!(cloned.shape, vec![2, 2]);
    assert_eq!(cloned.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_data_debug() {
    let data = TensorData {
        name: "test".to_string(),
        shape: vec![2],
        data: vec![1.0, 2.0],
        dtype: "F32".to_string(),
    };

    let debug_str = format!("{:?}", data);
    assert!(debug_str.contains("TensorData"));
    assert!(debug_str.contains("test"));
}

#[test]
fn test_tensor_comparison_shape_mismatch() {
    let a = TensorData {
        name: "test".to_string(),
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
        dtype: "F32".to_string(),
    };
    let b = vec![1.0, 2.0]; // Different size!

    let comp = TensorComparison::compare("test", &a, &b, 1e-5);
    assert!(!comp.shape_match);
    assert!(comp.weight_diff.is_none());
    assert!(!comp.passes_threshold);
}

#[test]
fn test_tensor_comparison_is_close_shape_mismatch() {
    let comp = TensorComparison {
        name: "test".to_string(),
        shape_match: false,
        shape_a: vec![3],
        shape_b: vec![2],
        weight_diff: None,
        passes_threshold: false,
    };

    assert!(!comp.is_close(1.0)); // Should be false even with high threshold
}

#[test]
fn test_tensor_comparison_clone() {
    let a = TensorData {
        name: "test".to_string(),
        shape: vec![2],
        data: vec![1.0, 2.0],
        dtype: "F32".to_string(),
    };
    let b = vec![1.0, 2.0];

    let comp = TensorComparison::compare("test", &a, &b, 1e-5);
    let cloned = comp.clone();
    assert_eq!(cloned.name, "test");
    assert!(cloned.shape_match);
}

#[test]
fn test_tensor_comparison_debug() {
    let comp = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        shape_a: vec![3],
        shape_b: vec![3],
        weight_diff: None,
        passes_threshold: true,
    };

    let debug_str = format!("{:?}", comp);
    assert!(debug_str.contains("TensorComparison"));
}

#[test]
fn test_batch_comparison_with_failures() {
    let comparisons = vec![
        TensorComparison {
            name: "good".to_string(),
            shape_match: true,
            shape_a: vec![3],
            shape_b: vec![3],
            weight_diff: Some(WeightDiff::empty()),
            passes_threshold: true,
        },
        TensorComparison {
            name: "bad".to_string(),
            shape_match: true,
            shape_a: vec![3],
            shape_b: vec![3],
            weight_diff: Some(WeightDiff {
                changed_count: 3,
                max_diff: 0.5,
                mean_diff: 0.1,
                l2_distance: 0.2,
                cosine_similarity: 0.9,
            }),
            passes_threshold: false,
        },
    ];

    let batch = BatchComparison::from_comparisons(comparisons);
    assert_eq!(batch.total_compared, 2);
    assert_eq!(batch.total_passed, 1);
    assert!(!batch.all_passed());
    assert_eq!(batch.worst_tensor, Some("bad".to_string()));
    assert!((batch.worst_diff - 0.5).abs() < 1e-5);
}

#[test]
fn test_batch_comparison_with_shape_mismatches() {
    let comparisons = vec![TensorComparison {
        name: "mismatch".to_string(),
        shape_match: false,
        shape_a: vec![3],
        shape_b: vec![2],
        weight_diff: None,
        passes_threshold: false,
    }];

    let batch = BatchComparison::from_comparisons(comparisons);
    assert_eq!(batch.shape_mismatches, 1);
    assert!(!batch.all_passed());
}

#[test]
fn test_batch_comparison_empty() {
    let batch = BatchComparison::from_comparisons(vec![]);
    assert_eq!(batch.total_compared, 0);
    assert_eq!(batch.total_passed, 0);
    assert_eq!(batch.shape_mismatches, 0);
    assert!(batch.all_passed()); // Empty is all passed
    assert!(batch.worst_tensor.is_none());
    assert_eq!(batch.worst_diff, 0.0);
}

#[test]
fn test_batch_comparison_summary() {
    let comparisons = vec![TensorComparison {
        name: "tensor1".to_string(),
        shape_match: true,
        shape_a: vec![3],
        shape_b: vec![3],
        weight_diff: Some(WeightDiff {
            changed_count: 0,
            max_diff: 0.001,
            mean_diff: 0.0005,
            l2_distance: 0.002,
            cosine_similarity: 0.999,
        }),
        passes_threshold: true,
    }];

    let batch = BatchComparison::from_comparisons(comparisons);
    let summary = batch.summary();
    assert!(summary.contains("1 tensors"));
    assert!(summary.contains("1 passed"));
    assert!(summary.contains("0 shape mismatches"));
    assert!(summary.contains("tensor1"));
}

#[test]
fn test_batch_comparison_summary_no_worst() {
    let comparisons = vec![TensorComparison {
        name: "no_diff".to_string(),
        shape_match: false,
        shape_a: vec![3],
        shape_b: vec![2],
        weight_diff: None,
        passes_threshold: false,
    }];

    let batch = BatchComparison::from_comparisons(comparisons);
    let summary = batch.summary();
    assert!(summary.contains("none"));
}

#[test]
fn test_batch_comparison_debug() {
    let batch = BatchComparison::from_comparisons(vec![]);
    let debug_str = format!("{:?}", batch);
    assert!(debug_str.contains("BatchComparison"));
}
