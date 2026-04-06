//! T-COV-95 Coverage Bridge: safetensors/validation.rs Part 02
//!
//! Targets uncovered lines: ValidatedWeight accessors (data, into_inner, out_dim, in_dim, name, stats),
//! ValidatedVector accessors (data, into_inner, name, stats),
//! ValidatedEmbedding accessors (data, into_inner, vocab_size, hidden_dim, stats),
//! ContractValidationError Display/Error/From impls,
//! validate_weight edge cases, validate_vector edge cases,
//! enforce_embedding_validation, enforce_weight_validation,
//! ValidatedAprTransformer with optional biases,
//! TensorStats edge cases (zero_pct, empty data).

use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::error::RealizarError;
use crate::safetensors::validation::*;

// ============================================================================
// TensorStats edge cases
// ============================================================================

#[test]
fn test_tensor_stats_empty_data() {
    let stats = TensorStats::compute(&[]);
    assert_eq!(stats.len, 0);
    assert_eq!(stats.zero_count, 0);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.l2_norm, 0.0);
}

#[test]
fn test_tensor_stats_zero_pct_empty() {
    let stats = TensorStats::compute(&[]);
    assert_eq!(stats.zero_pct(), 0.0);
}

#[test]
fn test_tensor_stats_zero_pct_all_zeros() {
    let data = vec![0.0f32; 100];
    let stats = TensorStats::compute(&data);
    assert!((stats.zero_pct() - 100.0).abs() < 0.01);
}

#[test]
fn test_tensor_stats_zero_pct_half_zeros() {
    let mut data = vec![0.0f32; 50];
    data.extend(vec![1.0f32; 50]);
    let stats = TensorStats::compute(&data);
    assert!((stats.zero_pct() - 50.0).abs() < 0.01);
}

#[test]
fn test_tensor_stats_zero_pct_no_zeros() {
    let data = vec![1.0f32; 100];
    let stats = TensorStats::compute(&data);
    assert!((stats.zero_pct() - 0.0).abs() < 0.01);
}

#[test]
fn test_tensor_stats_with_nan() {
    let data = vec![f32::NAN, 1.0, 2.0, f32::NAN];
    let stats = TensorStats::compute(&data);
    assert_eq!(stats.nan_count, 2);
    assert_eq!(stats.len, 4);
}

#[test]
fn test_tensor_stats_with_inf() {
    let data = vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 2.0];
    let stats = TensorStats::compute(&data);
    assert_eq!(stats.inf_count, 2);
}

#[test]
fn test_tensor_stats_all_nan() {
    let data = vec![f32::NAN; 10];
    let stats = TensorStats::compute(&data);
    assert_eq!(stats.nan_count, 10);
    // min/max should be 0.0 since all values were NaN
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
}

#[test]
fn test_tensor_stats_single_element() {
    let stats = TensorStats::compute(&[42.0]);
    assert_eq!(stats.len, 1);
    assert_eq!(stats.zero_count, 0);
    assert!((stats.min - 42.0).abs() < 0.001);
    assert!((stats.max - 42.0).abs() < 0.001);
    assert!((stats.mean - 42.0).abs() < 0.001);
}

#[test]
fn test_tensor_stats_negative_values() {
    let data = vec![-1.0, -2.0, -3.0, 1.0, 2.0, 3.0];
    let stats = TensorStats::compute(&data);
    assert!((stats.min - (-3.0)).abs() < 0.001);
    assert!((stats.max - 3.0).abs() < 0.001);
    assert!(stats.mean.abs() < 0.001); // mean should be ~0
}

#[test]
fn test_tensor_stats_debug() {
    let stats = TensorStats::compute(&[1.0, 2.0, 3.0]);
    let debug = format!("{:?}", stats);
    assert!(debug.contains("TensorStats"));
}

#[test]
fn test_tensor_stats_clone() {
    let stats = TensorStats::compute(&[1.0, 2.0, 3.0]);
    let cloned = stats.clone();
    assert_eq!(cloned.len, stats.len);
    assert!((cloned.mean - stats.mean).abs() < 0.001);
}

// ============================================================================
// ValidatedEmbedding accessor tests
// ============================================================================

#[test]
fn test_validated_embedding_data_accessor() {
    let vocab_size = 10;
    let hidden_dim = 8;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let ve =
        ValidatedEmbedding::new(data.clone(), vocab_size, hidden_dim).expect("should validate");

    assert_eq!(ve.data().len(), vocab_size * hidden_dim);
    assert!((ve.data()[0] - data[0]).abs() < 1e-6);
}

#[test]
fn test_validated_embedding_vocab_size() {
    let vocab_size = 20;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    assert_eq!(ve.vocab_size(), 20);
}

#[test]
fn test_validated_embedding_hidden_dim() {
    let vocab_size = 10;
    let hidden_dim = 16;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    assert_eq!(ve.hidden_dim(), 16);
}

#[test]
fn test_validated_embedding_stats() {
    let vocab_size = 10;
    let hidden_dim = 8;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let stats = ve.stats();
    assert_eq!(stats.len, vocab_size * hidden_dim);
    assert_eq!(stats.nan_count, 0);
}

#[test]
fn test_validated_embedding_into_inner() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let original_len = data.len();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let inner = ve.into_inner();
    assert_eq!(inner.len(), original_len);
}

#[test]
fn test_validated_embedding_debug() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let debug = format!("{:?}", ve);
    assert!(debug.contains("ValidatedEmbedding"));
}

#[test]
fn test_validated_embedding_clone() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let cloned = ve.clone();
    assert_eq!(cloned.vocab_size(), ve.vocab_size());
    assert_eq!(cloned.hidden_dim(), ve.hidden_dim());
}

// ============================================================================
// ValidatedEmbedding error paths
// ============================================================================

#[test]
fn test_validated_embedding_rejects_inf() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    data[3] = f32::INFINITY;
    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_embedding_rejects_zero_l2() {
    // All values extremely close to zero (below 1e-10 threshold)
    let vocab_size = 5;
    let hidden_dim = 4;
    let data = vec![1e-11f32; vocab_size * hidden_dim];
    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(result.is_err());
}

#[test]
fn test_validated_embedding_rejects_constant() {
    // All identical non-zero values
    let vocab_size = 5;
    let hidden_dim = 4;
    let data = vec![0.5f32; vocab_size * hidden_dim];
    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.rule_id, "F-DATA-QUALITY-003");
}

// ============================================================================
// ValidatedWeight accessor tests
// ============================================================================

#[test]
fn test_validated_weight_data_accessor() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data.clone(), 10, 10, "test_weight").expect("should validate");
    assert_eq!(vw.data().len(), 100);
    assert!((vw.data()[0] - data[0]).abs() < 1e-6);
}

#[test]
fn test_validated_weight_into_inner() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "test_weight").expect("should validate");
    let inner = vw.into_inner();
    assert_eq!(inner.len(), 100);
}

#[test]
fn test_validated_weight_out_dim() {
    let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 20, 10, "test").expect("should validate");
    assert_eq!(vw.out_dim(), 20);
}

#[test]
fn test_validated_weight_in_dim() {
    let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 20, 10, "test").expect("should validate");
    assert_eq!(vw.in_dim(), 10);
}

#[test]
fn test_validated_weight_name() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "my_weight").expect("should validate");
    assert_eq!(vw.name(), "my_weight");
}

#[test]
fn test_validated_weight_stats() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "test").expect("should validate");
    let stats = vw.stats();
    assert_eq!(stats.len, 100);
    assert_eq!(stats.nan_count, 0);
    assert!(stats.l2_norm > 0.0);
}

#[test]
fn test_validated_weight_debug() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "dbg").expect("should validate");
    let debug = format!("{:?}", vw);
    assert!(debug.contains("ValidatedWeight"));
}

#[test]
fn test_validated_weight_clone() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "clone_test").expect("should validate");
    let cloned = vw.clone();
    assert_eq!(cloned.name(), "clone_test");
    assert_eq!(cloned.out_dim(), 10);
}

// ============================================================================
// ValidatedWeight error paths
// ============================================================================

#[test]
fn test_validated_weight_rejects_nan() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::NAN;
    let result = ValidatedWeight::new(data, 10, 10, "nan_weight");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_weight_rejects_inf() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::INFINITY;
    let result = ValidatedWeight::new(data, 10, 10, "inf_weight");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_weight_rejects_zero_l2() {
    let data = vec![1e-11f32; 100]; // All near zero
    let result = ValidatedWeight::new(data, 10, 10, "zero_l2");
    assert!(result.is_err());
}

#[test]
fn test_validated_weight_shape_mismatch() {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let result = ValidatedWeight::new(data, 5, 5, "wrong_shape"); // expects 25
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-LAYOUT-CONTRACT-001");
}

// ============================================================================
// ValidatedVector accessor tests
// ============================================================================

#[test]
fn test_validated_vector_data_accessor() {
    let data = vec![1.0f32; 64];
    let vv = ValidatedVector::new(data, 64, "test_vec").expect("should validate");
    assert_eq!(vv.data().len(), 64);
    assert!((vv.data()[0] - 1.0).abs() < 0.001);
}

#[test]
fn test_validated_vector_into_inner() {
    let data = vec![2.0f32; 32];
    let vv = ValidatedVector::new(data, 32, "test_vec").expect("should validate");
    let inner = vv.into_inner();
    assert_eq!(inner.len(), 32);
    assert!((inner[0] - 2.0).abs() < 0.001);
}

#[test]
fn test_validated_vector_name() {
    let data = vec![1.0f32; 16];
    let vv = ValidatedVector::new(data, 16, "my_bias").expect("should validate");
    assert_eq!(vv.name(), "my_bias");
}

#[test]
fn test_validated_vector_stats() {
    let data = vec![1.0f32; 64];
    let vv = ValidatedVector::new(data, 64, "norm_weight").expect("should validate");
    let stats = vv.stats();
    assert_eq!(stats.len, 64);
    assert_eq!(stats.nan_count, 0);
}

#[test]
fn test_validated_vector_debug() {
    let data = vec![1.0f32; 8];
    let vv = ValidatedVector::new(data, 8, "dbg").expect("should validate");
    let debug = format!("{:?}", vv);
    assert!(debug.contains("ValidatedVector"));
}

#[test]
fn test_validated_vector_clone() {
    let data = vec![1.0f32; 16];
    let vv = ValidatedVector::new(data, 16, "clone_vec").expect("should validate");
    let cloned = vv.clone();
    assert_eq!(cloned.name(), "clone_vec");
    assert_eq!(cloned.data().len(), 16);
}

// ============================================================================
// ValidatedVector error paths
// ============================================================================

#[test]
fn test_validated_vector_rejects_nan() {
    let mut data = vec![1.0f32; 16];
    data[8] = f32::NAN;
    let result = ValidatedVector::new(data, 16, "nan_vec");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_vector_rejects_inf() {
    let mut data = vec![1.0f32; 16];
    data[8] = f32::NEG_INFINITY;
    let result = ValidatedVector::new(data, 16, "inf_vec");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_vector_length_mismatch() {
    let data = vec![1.0f32; 16];
    let result = ValidatedVector::new(data, 32, "wrong_len");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-LAYOUT-CONTRACT-003");
}

include!("validation_tests_contract.rs");
include!("validation_tests.rs");
