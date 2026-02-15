use super::*;
use crate::format::test_factory::{
    build_pygmy_apr, build_pygmy_apr_f16, build_pygmy_apr_q4, build_pygmy_apr_q8,
    build_pygmy_apr_with_config, build_pygmy_safetensors, build_pygmy_safetensors_with_config,
    PygmyConfig,
};

// ========================================================================
// Format Detection Tests
// ========================================================================

#[test]
fn test_detect_format_v2() {
    assert_eq!(detect_format(&MAGIC_APR2), Some("v2"));
    assert_eq!(detect_format(&MAGIC_APR0), Some("v2"));
}

#[test]
fn test_detect_format_v1() {
    assert_eq!(detect_format(&MAGIC_APRN), Some("v1"));
    assert_eq!(detect_format(&MAGIC_APR1), Some("v1"));
}

#[test]
fn test_detect_format_invalid() {
    assert_eq!(detect_format(&[0x00, 0x00, 0x00, 0x00]), None);
    assert_eq!(detect_format(&[0xFF, 0xFF, 0xFF, 0xFF]), None);
    assert_eq!(detect_format(b"GGUF"), None);
}

#[test]
fn test_is_valid_apr_magic() {
    assert!(is_valid_apr_magic(&MAGIC_APR2));
    assert!(is_valid_apr_magic(&MAGIC_APR0));
    assert!(is_valid_apr_magic(&MAGIC_APRN));
    assert!(!is_valid_apr_magic(b"GGUF"));
    assert!(!is_valid_apr_magic(&[0x00; 4]));
}

// ========================================================================
// TensorListOptions Tests
// ========================================================================

#[test]
fn test_options_default() {
    let opts = TensorListOptions::default();
    assert!(!opts.compute_stats);
    assert!(opts.filter.is_none());
    // Default limit is usize::MAX (effectively unlimited)
    assert_eq!(opts.limit, usize::MAX);
}

#[test]
fn test_options_builder() {
    let opts = TensorListOptions::new()
        .with_stats()
        .with_filter("weight")
        .with_limit(10);

    assert!(opts.compute_stats);
    assert_eq!(opts.filter, Some("weight".to_string()));
    assert_eq!(opts.limit, 10);
}

// ========================================================================
// Pygmy APR v2 Tests (TOOL-APR-001 Fix)
// ========================================================================

#[test]
fn test_list_tensors_pygmy_apr_default() {
    let apr_bytes = build_pygmy_apr();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    assert_eq!(result.format_version, "v2");
    assert!(result.tensor_count > 0, "Expected at least one tensor");
    assert!(result.total_size_bytes > 0);

    // Check we got tensor names from the index (could be various naming conventions)
    let names: Vec<_> = result.tensors.iter().map(|t| t.name.as_str()).collect();

    // Pygmy uses "model." prefix
    let has_model_tensors = names.iter().any(|n| n.starts_with("model."));
    let has_lm_head = names.iter().any(|n| n.contains("lm_head"));

    assert!(
        has_model_tensors || has_lm_head,
        "Expected model tensors, got: {:?}",
        names
    );
}

#[test]
fn test_list_tensors_pygmy_apr_with_filter() {
    let apr_bytes = build_pygmy_apr();
    let opts = TensorListOptions::new().with_filter("self_attn");
    let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

    // All returned tensors should match filter
    for tensor in &result.tensors {
        assert!(
            tensor.name.contains("self_attn"),
            "Expected tensor {} to contain 'self_attn'",
            tensor.name
        );
    }
}

#[test]
fn test_list_tensors_pygmy_apr_with_limit() {
    // Use llama_style which has many tensors (>10)
    let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::llama_style());

    // First, get total count without limit
    let full_result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("full list");
    let total_tensors = full_result.tensor_count;

    // GH-195 FIX: Verify model has more than our test limit
    assert!(
        total_tensors > 3,
        "Test requires model with >3 tensors, got {total_tensors}"
    );

    // Now apply limit
    let opts = TensorListOptions::new().with_limit(3);
    let limited_result = list_tensors_from_bytes(&apr_bytes, opts).expect("limited list");

    // P2 FIX: Use exact assertions, not tautological >= checks
    assert_eq!(
        limited_result.tensors.len(),
        3,
        "tensors.len() should equal the limit when total > limit"
    );
    assert_eq!(
        limited_result.tensor_count, total_tensors,
        "tensor_count must reflect TRUE total ({total_tensors}), not truncated length"
    );
}

#[test]
fn test_list_tensors_pygmy_apr_with_stats() {
    let apr_bytes = build_pygmy_apr();
    let opts = TensorListOptions::new().with_stats().with_limit(5);
    let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

    // Check at least one tensor has stats
    let has_stats = result
        .tensors
        .iter()
        .any(|t| t.mean.is_some() && t.std.is_some() && t.nan_count.is_some());
    assert!(has_stats, "Expected at least one tensor to have stats");
}

#[test]
fn test_list_tensors_pygmy_apr_f16() {
    let apr_bytes = build_pygmy_apr_f16();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // F16 tensors should be detected
    let f16_tensors: Vec<_> = result.tensors.iter().filter(|t| t.dtype == "f16").collect();
    assert!(!f16_tensors.is_empty(), "Expected F16 tensors");
}

#[test]
fn test_list_tensors_pygmy_apr_q8() {
    let apr_bytes = build_pygmy_apr_q8();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // Should have at least some tensors (mix of Q8 and F32)
    assert!(!result.tensors.is_empty(), "Expected tensors in Q8 model");

    // Q8 model contains both F32 (embedding) and Q8 (attention) tensors
    let dtypes: Vec<_> = result.tensors.iter().map(|t| t.dtype.as_str()).collect();
    assert!(
        dtypes.iter().any(|d| *d == "q8" || *d == "f32"),
        "Expected Q8 or F32 tensors, got: {:?}",
        dtypes
    );
}

#[test]
fn test_list_tensors_pygmy_apr_q4() {
    let apr_bytes = build_pygmy_apr_q4();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // Should have at least some tensors (mix of Q4 and F32)
    assert!(!result.tensors.is_empty(), "Expected tensors in Q4 model");

    // Q4 model contains both F32 (embedding) and Q4 (attention) tensors
    let dtypes: Vec<_> = result.tensors.iter().map(|t| t.dtype.as_str()).collect();
    assert!(
        dtypes.iter().any(|d| *d == "q4" || *d == "f32"),
        "Expected Q4 or F32 tensors, got: {:?}",
        dtypes
    );
}

#[test]
fn test_list_tensors_pygmy_apr_minimal() {
    let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::minimal());
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // Minimal config has embedding only
    assert!(result.tensor_count >= 1);
}

#[test]
fn test_list_tensors_pygmy_apr_llama_style() {
    let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::llama_style());
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // LLaMA style has many tensors (at least some)
    assert!(
        result.tensor_count >= 1,
        "Expected at least 1 tensor, got {}",
        result.tensor_count
    );

    // Should have tensors with various components
    let has_any_tensor = !result.tensors.is_empty();
    assert!(has_any_tensor);
}

// ========================================================================
// Error Handling Tests
// ========================================================================

#[test]
fn test_list_tensors_empty_data() {
    let result = list_tensors_from_bytes(&[], TensorListOptions::default());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("too small"));
}

#[test]
fn test_list_tensors_invalid_magic() {
    let data = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
    let result = list_tensors_from_bytes(&data, TensorListOptions::default());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Unknown model format"));
}

#[test]
fn test_list_tensors_truncated_v2() {
    // Valid v2 magic but truncated file
    let mut data = vec![0x41, 0x50, 0x52, 0x00]; // "APR\0"
    data.extend_from_slice(&[0u8; 10]); // Not enough for header

    let result = list_tensors_from_bytes(&data, TensorListOptions::default());
    assert!(result.is_err());
}

// ========================================================================
// Statistics Tests
// ========================================================================

#[test]
fn test_compute_stats_normal_values() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![4],
        dtype: "f32".to_string(),
        size_bytes: 16,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    let data = vec![1.0, 2.0, 3.0, 4.0];
    compute_tensor_stats(&mut info, &data);

    assert_eq!(info.mean, Some(2.5));
    assert!(info.std.expect("test value") > 1.0 && info.std.expect("test value") < 1.2);
    assert_eq!(info.min, Some(1.0));
    assert_eq!(info.max, Some(4.0));
    assert_eq!(info.nan_count, Some(0));
    assert_eq!(info.inf_count, Some(0));
}

#[test]
fn test_compute_stats_with_nan() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![3],
        dtype: "f32".to_string(),
        size_bytes: 12,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    let data = vec![1.0, f32::NAN, 2.0];
    compute_tensor_stats(&mut info, &data);

    assert_eq!(info.nan_count, Some(1));
    assert_eq!(info.inf_count, Some(0));
    // Mean should be computed from valid values only
    assert_eq!(info.mean, Some(1.5));
}

#[test]
fn test_compute_stats_with_inf() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![3],
        dtype: "f32".to_string(),
        size_bytes: 12,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    let data = vec![1.0, f32::INFINITY, 2.0];
    compute_tensor_stats(&mut info, &data);

    assert_eq!(info.nan_count, Some(0));
    assert_eq!(info.inf_count, Some(1));
}

#[test]
fn test_compute_stats_empty() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![0],
        dtype: "f32".to_string(),
        size_bytes: 0,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    compute_tensor_stats(&mut info, &[]);

    assert!(info.mean.is_none());
    assert!(info.std.is_none());
}

// ========================================================================
// Format Size Tests
// ========================================================================

#[test]
fn test_format_size_bytes() {
    assert_eq!(format_size(500), "500 B");
    assert_eq!(format_size(0), "0 B");
}

#[test]
fn test_format_size_kb() {
    assert_eq!(format_size(1024), "1.00 KB");
    assert_eq!(format_size(2048), "2.00 KB");
}

#[test]
fn test_format_size_mb() {
    assert_eq!(format_size(1024 * 1024), "1.00 MB");
    assert_eq!(format_size(100 * 1024 * 1024), "100.00 MB");
}

#[test]
fn test_format_size_gb() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
    assert_eq!(format_size(5 * 1024 * 1024 * 1024), "5.00 GB");
}

// ========================================================================
// TensorInfo Tests
// ========================================================================

#[test]
fn test_tensor_info_clone() {
    let info = TensorInfo {
        name: "test".to_string(),
        shape: vec![10, 20],
        dtype: "f32".to_string(),
        size_bytes: 800,
        mean: Some(0.5),
        std: Some(0.1),
        min: Some(0.0),
        max: Some(1.0),
        nan_count: Some(0),
        inf_count: Some(0),
    };

    let cloned = info.clone();
    assert_eq!(cloned.name, info.name);
    assert_eq!(cloned.shape, info.shape);
    assert_eq!(cloned.dtype, info.dtype);
}

#[test]
fn test_tensor_list_result_clone() {
    let result = TensorListResult {
        file: "test.apr".to_string(),
        format_version: "v2".to_string(),
        tensor_count: 5,
        total_size_bytes: 1000,
        tensors: vec![],
    };

    let cloned = result.clone();
    assert_eq!(cloned.file, result.file);
    assert_eq!(cloned.format_version, result.format_version);
}

include!("tensors_tests_part_02.rs");
include!("tensors_tests_part_03.rs");
include!("tensors_tests_part_04.rs");
