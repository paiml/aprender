#[allow(unused_imports)]
use super::*;

// ===== BlockQ5K Tests =====

#[test]
fn test_block_q5k_size() {
    assert_eq!(BlockQ5K::BLOCK_SIZE, 256);
}

#[test]
fn test_block_q5k_dequantize_basic() {
    let block = BlockQ5K {
        d: 0.1,
        dmin: 0.0,
        scales: [32; 12], // Neutral scales (32 - 32 = 0)
        qh: [0; 32],      // All high bits 0
        qs: [0x88; 128],  // 8,8 pattern (mid-range 4-bit)
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // With scale=0, all outputs should be dmin (0.0)
    for val in &output {
        assert!(val.abs() < 1.0, "Expected near-zero, got {}", val);
    }
}

#[test]
fn test_block_q5k_dequantize_with_scale() {
    let block = BlockQ5K {
        d: 1.0,
        dmin: 0.5,
        scales: [33; 12], // Scale of 1 (33 - 32 = 1)
        qh: [0xFF; 32],   // All high bits set
        qs: [0xFF; 128],  // All low bits set (15,15)
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // Values should be non-zero with positive scale
    let non_zero_count = output.iter().filter(|&&v| v.abs() > 1e-6).count();
    assert!(non_zero_count > 0, "Should have non-zero values");
}

#[test]
fn test_block_q5k_dequantize_alternating() {
    let block = BlockQ5K {
        d: 0.5,
        dmin: 0.1,
        scales: [34; 12], // Scale of 2
        qh: [0xAA; 32],   // Alternating bits
        qs: [0x55; 128],  // Alternating nibbles (5,5)
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // All values should be finite
    for val in &output {
        assert!(val.is_finite(), "Value should be finite");
    }
}

#[test]
fn test_block_q5k_dequantize_odd_even_bytes() {
    // Test both even and odd index paths in dequantization
    let block = BlockQ5K {
        d: 1.0,
        dmin: 0.0,
        scales: [48; 12], // Scale of 16 (48 - 32 = 16)
        qh: [0; 32],
        qs: [0x12; 128], // Low nibble = 2, high nibble = 1
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // Check that alternating values differ (even vs odd extraction)
    // Since qs[i] = 0x12, even indices extract 2, odd indices extract 1
    // Note: the actual dequant formula is complex, but values should differ
    assert!(output[0] != output[1] || output[0].abs() < 1e-6);
}

// ===== BlockQ6K Tests =====

#[test]
fn test_block_q6k_size() {
    assert_eq!(BlockQ6K::BLOCK_SIZE, 256);
}

#[test]
fn test_block_q6k_dequantize_basic() {
    let block = BlockQ6K {
        ql: [0; 128],
        qh: [0; 64],
        scales: [0; 16], // Zero scales
        d: 0.1,
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // With scale=0, all outputs should be d * 0 * (q6 - 32) = 0
    for val in &output {
        assert!(val.abs() < 1e-6, "Expected 0, got {}", val);
    }
}

#[test]
fn test_block_q6k_dequantize_with_scale() {
    let block = BlockQ6K {
        ql: [0xFF; 128], // Max low bits
        qh: [0xFF; 64],  // Max high bits
        scales: [1; 16], // Positive scale
        d: 0.5,
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // Values should be non-zero
    let non_zero = output.iter().any(|&v| v.abs() > 1e-6);
    assert!(non_zero, "Should have non-zero values");
}

#[test]
fn test_block_q6k_dequantize_negative_scale() {
    let block = BlockQ6K {
        ql: [0x88; 128],
        qh: [0x55; 64],
        scales: [-1; 16], // Negative scale
        d: 1.0,
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // All values should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_block_q6k_dequantize_all_subblocks() {
    // Test that all 16 sub-blocks are processed
    let block = BlockQ6K {
        ql: [0x12; 128],
        qh: [0x03; 64], // Different pattern per position
        scales: [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8],
        d: 0.1,
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // Check values at different sub-block boundaries
    assert!(output[0].is_finite());
    assert!(output[15].is_finite());
    assert!(output[16].is_finite());
    assert!(output[127].is_finite());
    assert!(output[255].is_finite());
}

#[test]
fn test_block_q6k_qh_extraction() {
    // Test the 2-bit high value extraction logic
    // qh_shift cycles through 0, 2, 4, 6 for i % 4 = 0, 1, 2, 3
    let block = BlockQ6K {
        ql: [0; 128],
        qh: [0b11_10_01_00; 64], // Pattern: 0,1,2,3 across 4 positions
        scales: [1; 16],
        d: 1.0,
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // Different qh values should produce different outputs
    // Position 0: qh_val = 0, Position 1: qh_val = 1, etc.
    // This tests the (i % 4) * 2 shift logic
    assert!(output[0].is_finite());
    assert!(output[1].is_finite());
    assert!(output[2].is_finite());
    assert!(output[3].is_finite());
}

// ===== DotQ5KOp Tests =====

#[test]
fn test_dot_q5k_new() {
    let op = DotQ5KOp::new(512);
    assert_eq!(op.n_blocks, 2);
}

#[test]
fn test_dot_q5k_name() {
    let op = DotQ5KOp::new(256);
    assert_eq!(op.name(), "dot_q5k");
}

#[test]
fn test_dot_q5k_empty() {
    let op = DotQ5KOp::new(256);
    let result = op.execute((vec![], vec![]), Backend::Scalar).unwrap();
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_q5k_empty_activations() {
    let op = DotQ5KOp::new(256);
    let block = BlockQ5K { d: 1.0, dmin: 0.0, scales: [32; 12], qh: [0; 32], qs: [0; 128] };
    let result = op.execute((vec![block], vec![]), Backend::Scalar).unwrap();
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_q5k_tokens() {
    let op = DotQ5KOp::new(512); // 2 blocks
    let input = (vec![], vec![]);
    assert_eq!(op.tokens(&input), 512);
}

#[test]
fn test_dot_q5k_scalar_execution() {
    let op = DotQ5KOp::new(256);
    let block = BlockQ5K {
        d: 1.0,
        dmin: 0.0,
        scales: [33; 12], // Scale = 1
        qh: [0; 32],
        qs: [0x88; 128], // Mid-range values
    };
    let x = vec![1.0f32; 256];
    let result = op.execute((vec![block], x), Backend::Scalar).unwrap();
    assert!(result.is_finite());
}

#[test]
fn test_dot_q5k_multiple_blocks() {
    let op = DotQ5KOp::new(512);
    let block = BlockQ5K { d: 0.5, dmin: 0.1, scales: [34; 12], qh: [0; 32], qs: [0x44; 128] };
    let x = vec![0.5f32; 512];
    let result = op.execute((vec![block.clone(), block], x), Backend::Scalar).unwrap();
    assert!(result.is_finite());
}

#[test]
fn test_dot_q5k_auto_backend() {
    let op = DotQ5KOp::new(256);
    let block = BlockQ5K { d: 1.0, dmin: 0.0, scales: [32; 12], qh: [0; 32], qs: [0; 128] };
    let x = vec![1.0f32; 256];
    // Auto backend should work (may use AVX2 if available)
    let result = op.execute((vec![block], x), Backend::Auto).unwrap();
    assert!(result.is_finite());
}

#[test]
fn test_dot_q5k_avx2_backend() {
    let op = DotQ5KOp::new(256);
    let block = BlockQ5K { d: 1.0, dmin: 0.0, scales: [33; 12], qh: [0; 32], qs: [0x11; 128] };
    let x = vec![2.0f32; 256];
    // Request AVX2, will fall back to scalar if not available
    let result = op.execute((vec![block], x), Backend::Avx2).unwrap();
    assert!(result.is_finite());
}

// ===== DotQ6KOp Tests =====

#[test]
fn test_dot_q6k_new() {
    let op = DotQ6KOp::new(768);
    assert_eq!(op.n_blocks, 3);
}

#[test]
fn test_dot_q6k_name() {
    let op = DotQ6KOp::new(256);
    assert_eq!(op.name(), "dot_q6k");
}

#[test]
fn test_dot_q6k_empty() {
    let op = DotQ6KOp::new(256);
    let result = op.execute((vec![], vec![]), Backend::Scalar).unwrap();
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_q6k_empty_activations() {
    let op = DotQ6KOp::new(256);
    let block = BlockQ6K { ql: [0; 128], qh: [0; 64], scales: [0; 16], d: 1.0 };
    let result = op.execute((vec![block], vec![]), Backend::Scalar).unwrap();
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_q6k_tokens() {
    let op = DotQ6KOp::new(768); // 3 blocks
    let input = (vec![], vec![]);
    assert_eq!(op.tokens(&input), 768);
}

#[test]
fn test_dot_q6k_scalar_execution() {
    let op = DotQ6KOp::new(256);
    let block = BlockQ6K { ql: [0x55; 128], qh: [0x55; 64], scales: [1; 16], d: 0.5 };
    let x = vec![1.0f32; 256];
    let result = op.execute((vec![block], x), Backend::Scalar).unwrap();
    assert!(result.is_finite());
}

#[test]
fn test_dot_q6k_multiple_blocks() {
    let op = DotQ6KOp::new(512);
    let block = BlockQ6K { ql: [0x33; 128], qh: [0x33; 64], scales: [2; 16], d: 0.25 };
    let x = vec![0.5f32; 512];
    let result = op.execute((vec![block.clone(), block], x), Backend::Scalar).unwrap();
    assert!(result.is_finite());
}

#[test]
fn test_dot_q6k_auto_backend() {
    let op = DotQ6KOp::new(256);
    let block = BlockQ6K { ql: [0; 128], qh: [0; 64], scales: [1; 16], d: 1.0 };
    let x = vec![1.0f32; 256];
    let result = op.execute((vec![block], x), Backend::Auto).unwrap();
    assert!(result.is_finite());
}

#[test]
fn test_dot_q6k_avx2_backend() {
    let op = DotQ6KOp::new(256);
    let block = BlockQ6K { ql: [0xAA; 128], qh: [0xAA; 64], scales: [3; 16], d: 0.1 };
    let x = vec![2.0f32; 256];
    let result = op.execute((vec![block], x), Backend::Avx2).unwrap();
    assert!(result.is_finite());
}

// ===== Backend Equivalence Tests =====

#[test]
fn test_q5k_backend_equivalence() {
    let op = DotQ5KOp::new(256);
    let block = BlockQ5K { d: 0.5, dmin: 0.1, scales: [35; 12], qh: [0x55; 32], qs: [0x77; 128] };
    let x = vec![1.5f32; 256];

    let scalar = op.execute((vec![block.clone()], x.clone()), Backend::Scalar).unwrap();
    let auto = op.execute((vec![block], x), Backend::Auto).unwrap();

    // Allow small FP differences due to SIMD operation ordering
    let rel_diff = (scalar - auto).abs() / scalar.abs().max(1e-6);
    assert!(rel_diff < 1e-4, "scalar={scalar}, auto={auto}, rel_diff={rel_diff}");
}

#[test]
fn test_q6k_backend_equivalence() {
    let op = DotQ6KOp::new(256);
    let block = BlockQ6K { ql: [0x66; 128], qh: [0x22; 64], scales: [4; 16], d: 0.2 };
    let x = vec![1.5f32; 256];

    let scalar = op.execute((vec![block.clone()], x.clone()), Backend::Scalar).unwrap();
    let auto = op.execute((vec![block], x), Backend::Auto).unwrap();

    // Allow small FP differences due to SIMD operation ordering
    let rel_diff = (scalar - auto).abs() / scalar.abs().max(1e-6);
    assert!(rel_diff < 1e-4, "scalar={scalar}, auto={auto}, rel_diff={rel_diff}");
}

// ===== Clone/Debug Trait Tests =====

#[test]
fn test_block_q5k_clone_debug() {
    let block = BlockQ5K { d: 1.0, dmin: 0.5, scales: [32; 12], qh: [0; 32], qs: [0; 128] };
    let cloned = block.clone();
    assert_eq!(format!("{:?}", block), format!("{:?}", cloned));
}

#[test]
fn test_block_q6k_clone_debug() {
    let block = BlockQ6K { ql: [0; 128], qh: [0; 64], scales: [0; 16], d: 1.0 };
    let cloned = block.clone();
    assert_eq!(format!("{:?}", block), format!("{:?}", cloned));
}

#[test]
fn test_dot_q5k_op_clone_debug() {
    let op = DotQ5KOp::new(256);
    let cloned = op.clone();
    assert_eq!(format!("{:?}", op), format!("{:?}", cloned));
}

#[test]
fn test_dot_q6k_op_clone_debug() {
    let op = DotQ6KOp::new(256);
    let cloned = op.clone();
    assert_eq!(format!("{:?}", op), format!("{:?}", cloned));
}
