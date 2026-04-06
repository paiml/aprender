//! Part 05: Inference configuration, model execution paths, and error handling
//!
//! Coverage for src/inference/mod.rs including:
//! - Q4KWeight construction and validation
//! - Q4KWeight matvec execution paths
//! - Error propagation and recovery
//! - Memory statistics and compression ratios

use crate::error::RealizarError;
use crate::inference::Q4KWeight;

// ============================================================================
// Q4KWeight Construction Tests
// ============================================================================

#[test]
fn test_q4k_weight_new_minimal_valid() {
    let in_dim = 256;
    let out_dim = 1;
    let bytes_per_row = 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim);
    assert!(weight.is_ok());

    let w = weight.expect("w");
    assert_eq!(w.in_dim, 256);
    assert_eq!(w.out_dim, 1);
}

#[test]
fn test_q4k_weight_new_multiple_rows() {
    let in_dim = 256;
    let out_dim = 8;
    let bytes_per_row = 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim);
    assert!(weight.is_ok());

    let w = weight.expect("w");
    assert_eq!(w.out_dim, 8);
    assert_eq!(w.data.len(), 8 * 144);
}

#[test]
fn test_q4k_weight_new_multiple_blocks_per_row() {
    let in_dim: usize = 512;
    let out_dim: usize = 2;
    let blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = blocks_per_row * 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim);
    assert!(weight.is_ok());

    let w = weight.expect("w");
    assert_eq!(w.in_dim, 512);
    assert_eq!(w.memory_bytes(), 2 * 288);
}

#[test]
fn test_q4k_weight_new_non_multiple_of_256() {
    let in_dim: usize = 300;
    let out_dim: usize = 1;
    let blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = blocks_per_row * 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim);
    assert!(weight.is_ok());
}

// ============================================================================
// Q4KWeight Invalid Construction Tests
// ============================================================================

#[test]
fn test_q4k_weight_new_data_too_small() {
    let in_dim = 256;
    let out_dim = 1;
    let data = vec![0u8; 100];

    let result = Q4KWeight::new(data, in_dim, out_dim);
    assert!(result.is_err());

    if let Err(err) = result {
        match err {
            RealizarError::InvalidShape { reason } => {
                assert!(reason.contains("doesn't match"));
                assert!(reason.contains("100"));
                assert!(reason.contains("144"));
            },
            _ => panic!("Expected InvalidShape error"),
        }
    }
}

#[test]
fn test_q4k_weight_new_data_too_large() {
    let data = vec![0u8; 200];
    let result = Q4KWeight::new(data, 256, 1);
    assert!(result.is_err());
}

#[test]
fn test_q4k_weight_new_empty_data() {
    let data: Vec<u8> = vec![];
    let result = Q4KWeight::new(data, 256, 1);
    assert!(result.is_err());
}

#[test]
fn test_q4k_weight_new_zero_output_dim() {
    let data: Vec<u8> = vec![];
    let result = Q4KWeight::new(data, 256, 0);
    assert!(result.is_ok());

    let w = result.expect("w");
    assert_eq!(w.out_dim, 0);
    assert_eq!(w.memory_bytes(), 0);
}

#[test]
fn test_q4k_weight_new_data_slightly_off() {
    let data = vec![0u8; 143];
    let result = Q4KWeight::new(data, 256, 1);
    assert!(result.is_err());
}

// ============================================================================
// Q4KWeight Memory Statistics Tests
// ============================================================================

#[test]
fn test_q4k_weight_memory_bytes() {
    let data = vec![0u8; 144];
    let weight = Q4KWeight::new(data, 256, 1).expect("weight");
    assert_eq!(weight.memory_bytes(), 144);

    let data = vec![0u8; 10 * 144];
    let weight = Q4KWeight::new(data, 256, 10).expect("weight");
    assert_eq!(weight.memory_bytes(), 10 * 144);
}

#[test]
fn test_q4k_weight_f32_equivalent_bytes() {
    let data = vec![0u8; 4 * 144];
    let weight = Q4KWeight::new(data, 256, 4).expect("weight");
    assert_eq!(weight.f32_equivalent_bytes(), 256 * 4 * 4);
}

#[test]
fn test_q4k_weight_f32_equivalent_large() {
    let in_dim: usize = 4096;
    let out_dim: usize = 4096;
    let blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = blocks_per_row * 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("weight");
    assert_eq!(weight.f32_equivalent_bytes(), 4096 * 4096 * 4);
}

#[test]
fn test_q4k_weight_compression_ratio() {
    let data = vec![0u8; 144];
    let weight = Q4KWeight::new(data, 256, 1).expect("weight");

    // F32 bytes: 256 * 1 * 4 = 1024, Q4_K bytes: 144, Ratio: ~7.11
    let ratio = weight.compression_ratio();
    assert!(ratio > 7.0, "Expected ratio > 7.0, got {}", ratio);
    assert!(ratio < 8.0, "Expected ratio < 8.0, got {}", ratio);
}

#[test]
fn test_q4k_weight_compression_ratio_consistency() {
    let sizes: [(usize, usize); 4] = [(256, 1), (256, 10), (512, 4), (1024, 2)];

    for (in_dim, out_dim) in sizes {
        let blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = blocks_per_row * 144;
        let data = vec![0u8; out_dim * bytes_per_row];

        let weight = Q4KWeight::new(data, in_dim, out_dim).expect("weight");
        let ratio = weight.compression_ratio();

        assert!(
            ratio > 7.0,
            "Ratio for {}x{} was {}",
            in_dim,
            out_dim,
            ratio
        );
    }
}

// ============================================================================
// Q4KWeight Matvec Error Handling Tests
// ============================================================================

#[test]
fn test_q4k_weight_matvec_wrong_input_length() {
    let data = vec![0u8; 144];
    let weight = Q4KWeight::new(data, 256, 1).expect("weight");

    // Too short
    let result = weight.matvec(&vec![1.0f32; 128]);
    assert!(result.is_err());
    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("128"));
        assert!(reason.contains("256"));
    } else {
        panic!("Expected InvalidShape error");
    }

    // Too long
    let result = weight.matvec(&vec![1.0f32; 512]);
    assert!(result.is_err());
}

#[test]
fn test_q4k_weight_matvec_empty_input() {
    let data = vec![0u8; 144];
    let weight = Q4KWeight::new(data, 256, 1).expect("weight");
    let result = weight.matvec(&[]);
    assert!(result.is_err());
}

#[test]
fn test_q4k_weight_matvec_off_by_one() {
    let data = vec![0u8; 144];
    let weight = Q4KWeight::new(data, 256, 1).expect("weight");

    assert!(weight.matvec(&vec![1.0f32; 255]).is_err());
    assert!(weight.matvec(&vec![1.0f32; 257]).is_err());
}

// ============================================================================
// Q4KWeight Clone and Field Access Tests
// ============================================================================

#[test]
fn test_q4k_weight_clone_preserves_data() {
    let in_dim = 256;
    let out_dim = 2;
    let data: Vec<u8> = (0..out_dim * 144).map(|i| (i % 256) as u8).collect();

    let weight = Q4KWeight::new(data.clone(), in_dim, out_dim).expect("weight");
    let cloned = weight.clone();

    assert_eq!(weight.in_dim, cloned.in_dim);
    assert_eq!(weight.out_dim, cloned.out_dim);
    assert_eq!(weight.data, cloned.data);
    assert_eq!(weight.memory_bytes(), cloned.memory_bytes());
}

#[test]
fn test_q4k_weight_public_fields() {
    let in_dim: usize = 512;
    let out_dim: usize = 4;
    let blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = blocks_per_row * 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("weight");

    assert_eq!(weight.in_dim, 512);
    assert_eq!(weight.out_dim, 4);
    assert_eq!(weight.data.len(), out_dim * bytes_per_row);
}

#[test]
fn test_q4k_weight_data_content() {
    let data: Vec<u8> = (0..144).map(|i| i as u8).collect();
    let weight = Q4KWeight::new(data.clone(), 256, 1).expect("weight");
    assert_eq!(weight.data, data);
}

// ============================================================================
// Integration: Error Message Quality Tests
// ============================================================================

#[test]
fn test_q4k_weight_error_message_includes_dimensions() {
    let data = vec![0u8; 100];
    let result = Q4KWeight::new(data, 256, 1);
    assert!(result.is_err());

    if let Err(err) = result {
        let msg = err.to_string();
        assert!(
            msg.contains("100") || msg.contains("144"),
            "Error should mention sizes: {}",
            msg
        );
    }
}

#[test]
fn test_q4k_weight_matvec_error_message_quality() {
    let data = vec![0u8; 144];
    let weight = Q4KWeight::new(data, 256, 1).expect("weight");

    let result = weight.matvec(&vec![1.0f32; 128]);
    assert!(result.is_err());

    if let Err(err) = result {
        let msg = err.to_string();
        assert!(msg.contains("128") || msg.contains("256"), "Error: {}", msg);
    }
}

// ============================================================================
// Edge Cases: Boundary Conditions
// ============================================================================

#[test]
fn test_q4k_weight_multiple_of_256() {
    for multiplier in 1..=4 {
        let in_dim = 256 * multiplier;
        let blocks_per_row = multiplier;
        let bytes_per_row = blocks_per_row * 144;
        let data = vec![0u8; bytes_per_row];

        let weight = Q4KWeight::new(data, in_dim, 1);
        assert!(weight.is_ok(), "Failed for in_dim={}", in_dim);
        assert_eq!(weight.expect("expected value").in_dim, in_dim);
    }
}

#[test]
fn test_q4k_weight_ceil_division_boundary() {
    // 257 elements requires 2 blocks
    let data = vec![0u8; 2 * 144];
    let weight = Q4KWeight::new(data, 257, 1);
    assert!(weight.is_ok());

    // 255 elements requires 1 block
    let data = vec![0u8; 144];
    let weight = Q4KWeight::new(data, 255, 1);
    assert!(weight.is_ok());
}

// ============================================================================
// Stress Tests: Large Configurations
// ============================================================================

#[test]
fn test_q4k_weight_large_output_dimension() {
    let data = vec![0u8; 1000 * 144];
    let weight = Q4KWeight::new(data, 256, 1000).expect("weight");

    assert_eq!(weight.out_dim, 1000);
    assert_eq!(weight.memory_bytes(), 1000 * 144);
    assert!(weight.compression_ratio() > 7.0);
}

#[test]
fn test_q4k_weight_statistics_consistency() {
    let in_dim: usize = 1024;
    let out_dim: usize = 512;
    let blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = blocks_per_row * 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("weight");

    let memory = weight.memory_bytes();
    let f32_equiv = weight.f32_equivalent_bytes();
    let ratio = weight.compression_ratio();

    let calculated_ratio = f32_equiv as f32 / memory as f32;
    assert!(
        (ratio - calculated_ratio).abs() < 0.001,
        "Ratio mismatch: {} vs calculated {}",
        ratio,
        calculated_ratio
    );
}
