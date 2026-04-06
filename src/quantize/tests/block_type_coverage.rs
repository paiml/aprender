
// ============================================================================
// Block Type Coverage
// ============================================================================

/// Test Q4_0Block Debug and Clone
#[test]
fn test_q4_0_block_traits() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0xABu8; 16],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q4_0Block"));
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

/// Test Q4_KBlock Debug and Clone
#[test]
fn test_q4_k_block_traits() {
    let block = Q4_KBlock {
        d: 1.5,
        dmin: 0.25,
        scales: [10u8; 12],
        qs: [0x55u8; 128],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    assert!((block.dmin - cloned.dmin).abs() < 1e-6);
    assert_eq!(block.scales, cloned.scales);
    assert_eq!(block.qs, cloned.qs);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q4_KBlock"));
}

/// Test Q5_KBlock Debug and Clone
#[test]
fn test_q5_k_block_traits() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 0.5,
        scales: [5u8; 12],
        qh: [0xFFu8; 32],
        qs: [0x33u8; 128],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    assert!((block.dmin - cloned.dmin).abs() < 1e-6);
    assert_eq!(block.scales, cloned.scales);
    assert_eq!(block.qh, cloned.qh);
    assert_eq!(block.qs, cloned.qs);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q5_KBlock"));
}

/// Test Q6_KBlock Debug and Clone
#[test]
fn test_q6_k_block_traits() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [1i8; 16],
        qh: [0x11u8; 64],
        qs: [0x22u8; 128],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    assert_eq!(block.scales, cloned.scales);
    assert_eq!(block.qh, cloned.qh);
    assert_eq!(block.qs, cloned.qs);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q6_KBlock"));
}

/// Test Q8KSuperBlock Debug and Clone
#[test]
fn test_q8k_superblock_traits() {
    let block = Q8KSuperBlock {
        scale: 0.5,
        quants: [10i8; 256],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q8KSuperBlock"));
}

// ============================================================================
// Constants Coverage
// ============================================================================

/// Test BLOCK_SIZE constant value
#[test]
fn test_block_size_value() {
    assert_eq!(BLOCK_SIZE, 32);
}

/// Test QK_K constant value
#[test]
fn test_qk_k_value() {
    assert_eq!(QK_K, 256);
}

/// Test relationship between constants
#[test]
fn test_constants_relationship() {
    // QK_K should be exactly 8 blocks worth
    assert_eq!(QK_K, BLOCK_SIZE * 8);
}

// ============================================================================
// Edge Cases for Quantization Boundaries
// ============================================================================

/// Test Q8_0Block with value exactly at -128 boundary
#[test]
fn test_q8_0_block_boundary_minus_128() {
    // Scale = 1.0, so -128.0 should map to -128 (clamped)
    let mut values = [0.0f32; 32];
    values[0] = 128.0; // This sets the scale
    values[1] = -128.0; // This should map to -128

    let block = Q8_0Block::quantize(&values);
    // Scale should be 128/127 ≈ 1.0079
    // -128 / 1.0079 ≈ -127.0 (within valid range)
    assert!(
        block.quants[1] <= 0,
        "Negative value should quantize to negative"
    );
}

/// Test Q8_0Block with value exactly at 127 boundary
#[test]
fn test_q8_0_block_boundary_127() {
    let values = [1.0f32; 32]; // All values at max
    let block = Q8_0Block::quantize(&values);

    // All should be 127
    assert!(
        block.quants.iter().all(|&q| q == 127),
        "Max values should quantize to 127"
    );
}

/// Test Q8KSuperBlock with value at clamping boundary
#[test]
fn test_q8k_superblock_clamping() {
    let mut values = [0.0f32; 256];
    values[0] = 1.0; // Reference max
    values[1] = 1.0; // Same max
    for i in 2..256 {
        values[i] = 1.0;
    }

    let block = Q8KSuperBlock::quantize(&values);

    // All should be 127
    assert!(
        block.quants.iter().all(|&q| q == 127),
        "All max values should be 127"
    );
}

// ============================================================================
// Additional Coverage for types.rs InterleavedQ4K fields access
// ============================================================================

/// Test that InterleavedQ4K fields are accessible and have correct values
#[test]
fn test_interleaved_q4k_field_access() {
    // Create data with specific values
    let mut data = vec![0u8; 144];

    // d = 1.5 (f16: 0x3E00)
    data[0] = 0x00;
    data[1] = 0x3E;

    // dmin = 0.25 (f16: 0x3400)
    data[2] = 0x00;
    data[3] = 0x34;

    // Set some scale values
    for i in 4..16 {
        data[i] = (i - 4) as u8;
    }

    // Set some qs values
    for i in 16..144 {
        data[i] = ((i - 16) % 256) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");

    // Check d value is close to 1.5
    assert!(
        (interleaved.d[0] - 1.5).abs() < 0.1,
        "d should be approximately 1.5, got {}",
        interleaved.d[0]
    );

    // Check dmin value is close to 0.25
    assert!(
        (interleaved.dmin[0] - 0.25).abs() < 0.1,
        "dmin should be approximately 0.25, got {}",
        interleaved.dmin[0]
    );

    // Check scales were copied correctly
    for i in 0..12 {
        assert_eq!(interleaved.scales[i], i as u8, "scales[{i}] should be {i}");
    }

    // Check qs were copied correctly
    for i in 0..128 {
        assert_eq!(
            interleaved.qs[i],
            (i % 256) as u8,
            "qs[{i}] should be {}",
            i % 256
        );
    }
}

/// Test InterleavedQ4K with actual non-zero quantized values
#[test]
fn test_interleaved_q4k_with_real_data() {
    let mut data = vec![0u8; 144 * 2]; // 2 super-blocks

    // First super-block: d=2.0, dmin=1.0
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes()); // d=2.0
    data[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin=1.0

    // Set scales to non-zero
    for i in 4..16 {
        data[i] = 32; // Scale value of 32
    }

    // Set qs to alternating pattern
    for i in 16..144 {
        data[i] = 0x12; // low=2, high=1
    }

    // Second super-block: d=0.5, dmin=0.25
    data[144..146].copy_from_slice(&0x3800u16.to_le_bytes()); // d=0.5
    data[146..148].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin=0.25

    for i in 148..160 {
        data[i] = 16;
    }

    for i in 160..288 {
        data[i] = 0x34;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");

    assert_eq!(interleaved.num_super_blocks, 2);

    // Verify d values
    assert!((interleaved.d[0] - 2.0).abs() < 0.1);
    assert!((interleaved.d[1] - 0.5).abs() < 0.1);

    // Verify dmin values
    assert!((interleaved.dmin[0] - 1.0).abs() < 0.1);
    assert!((interleaved.dmin[1] - 0.25).abs() < 0.1);
}
