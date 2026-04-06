
// ============================================================================
// T-COV-95: ADDITIONAL COVERAGE FOR fused_k.rs
// ============================================================================

// --- fused_q4k_dot: known-value computation verification ---

#[test]
fn test_fused_q4k_dot_known_value_single_block_uniform() {
    // Build a Q4_K super-block with known properties:
    // d = 1.0, dmin = 0, all scales = 1, all mins = 0, all qs = 0x33 (low=3, high=3)
    let mut data = vec![0u8; 144];

    // d = 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // dmin = 0.0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    // scales[0..3] = 1 (first 4 blocks use simple layout: scale = scales[j] & 63)
    data[4] = 1;
    data[5] = 1;
    data[6] = 1;
    data[7] = 1;
    // scales[4..7] = 0 (mins for first 4 blocks: scales[j+4] & 63)
    // scales[8..11] = 0 (packed upper bits)

    // Set all qs to 0x33: low nibble = 3, high nibble = 3
    for i in 0..128 {
        data[16 + i] = 0x33;
    }

    // Activations: all 1.0
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Manually compute:
    // 4 chunks of 64 values, each chunk has 32 low nibbles + 32 high nibbles
    // For blocks 0-3 (is=0..7, but first 4 have scale=1):
    //   Chunk 0 (j=0): is=0 -> sc1=scales[0]&63=1, m1=scales[4]&63=0
    //                   is=1 -> sc2=scales[1]&63=1, m2=scales[5]&63=0
    //     d1 = 1.0 * 1 = 1.0, dm1 = 0
    //     d2 = 1.0 * 1 = 1.0, dm2 = 0
    //     Low nibbles (32 values): q_val=3, value=1.0*3-0=3.0, sum=32*3.0=96.0
    //     High nibbles (32 values): q_val=3, value=1.0*3-0=3.0, sum=32*3.0=96.0
    //   Chunk 1 (j=64): is=2 -> sc1=scales[2]&63=1, sc2=scales[3]&63=1
    //     Same pattern: 96.0 + 96.0 = 192.0
    //   Chunk 2 (j=128): is=4 -> packed layout, scales byte at data[12]=0, data[0]>>6=0
    //     sc1 = (scales[8]&0x0F)|((scales[0]>>6)<<4) = 0
    //     So all values are 0 for blocks 4-7
    //   Chunk 3 (j=192): is=6 -> same, packed = 0
    //
    // Total = 96 + 96 + 96 + 96 + 0 + 0 + 0 + 0 = 384.0
    assert!(
        (result - 384.0).abs() < 1.0,
        "Expected about 384.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_dot_known_value_with_dmin() {
    // Test that dmin is correctly subtracted
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // scales[0] = 2 (scale for block 0), scales[4] = 3 (min for block 0)
    data[4] = 2;
    data[8] = 3;
    // All other scales/mins = 0

    // qs all zeros: low nibble = 0, high nibble = 0
    // (all qs are already 0)

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Chunk 0 (j=0): is=0 -> sc1=scales[0]&63=2, m1=scales[4]&63=3
    //                 is=1 -> sc2=scales[1]&63=0, m2=scales[5]&63=0
    //   d1=1.0*2=2.0, dm1=0.5*3=1.5
    //   Low nibbles (32 values): val = 2.0*0 - 1.5 = -1.5, sum = 32 * (-1.5) = -48.0
    //   d2=1.0*0=0, dm2=0.5*0=0
    //   High nibbles (32 values): val = 0*0 - 0 = 0, sum = 0
    // Chunks 1-3: all scales=0, so vals = 0 - 0 = 0
    //
    // Total = -48.0
    assert!(
        (result - (-48.0)).abs() < 0.1,
        "Expected about -48.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_dot_varied_nibble_pattern() {
    // Verify correct nibble extraction: qs byte 0xAB = low=0xB=11, high=0xA=10
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // scale for block 0 = 1
    data[4] = 1;

    // Set first qs byte = 0xAB, rest = 0
    data[16] = 0xAB;

    // Activations: all 1.0
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Chunk 0, block 0 (sc1=1, m1=0):
    //   Low nibble byte[0] = 0xB = 11, val = 1.0*1*11 - 0 = 11.0
    //   Low nibble bytes[1..31] = 0, val = 0
    //   -> Low nibble sum = 11.0
    //
    // Chunk 0, block 1 (sc2=scales[1]&63=0):
    //   High nibble byte[0] = 0xA = 10, val = 1.0*0*10 - 0 = 0
    //   -> High nibble sum = 0
    //
    // All other chunks: scales = 0, so 0
    //
    // Total = 11.0
    assert!(
        (result - 11.0).abs() < 0.01,
        "Expected 11.0, got {}",
        result
    );
}

// --- fused_q4k_q8k_dot: known-value computation verification ---

#[test]
fn test_fused_q4k_q8k_dot_known_value_simple() {
    // Build a Q4_K block with known properties and Q8_K activations
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    // scale for block 0 = 1
    data[4] = 1;

    // All qs = 0x11 (low=1, high=1)
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    // Q8_K: scale = 1.0, all quants = 2
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![2i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Chunk 0 (j=0): sc1=1, m1=0 -> d_sc1_q8=1.0*1*1.0=1.0, dm1_q8=0
    //   sc2=0 (scales[1]=0) -> d_sc2_q8=0, dm2_q8=0
    //   32 bytes: q4_lo = 1, q8_lo = 2 -> sum_lo = 32 * (1*2) = 64
    //   q8_sum_lo = 32 * 2 = 64
    //   result += 1.0 * 64 - 0 * 64 = 64.0
    //   High nibble: q4_hi = 1, but sc2=0 so no contribution
    //   result += 0 * ... - 0 * ... = 0
    // Chunks 1-3: scales=0 so result += 0 * ... = 0
    // Total = 64.0
    assert!(
        (result - 64.0).abs() < 1.0,
        "Expected about 64.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_known_value_with_dmin() {
    let mut data = vec![0u8; 144];

    // d = 2.0, dmin = 1.0
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());

    // scale[0] = 3, min[0] = 2
    data[4] = 3; // scales[0] & 63 = 3
    data[8] = 2; // scales[4] & 63 = 2

    // All qs = 0 (low=0, high=0)

    // Q8_K: scale = 0.5, quants all = 10
    let q8k_scales = vec![0.5f32];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Chunk 0 (j=0): sc1=3, m1=2
    //   d_sc1_q8 = 2.0 * 3 * 0.5 = 3.0
    //   dm1_q8 = 1.0 * 2 * 0.5 = 1.0
    //   Low nibbles: q4_lo=0 for all 32 -> sum_lo=0
    //   q8_sum_lo = 32 * 10 = 320
    //   contrib1 = 3.0 * 0 - 1.0 * 320 = -320.0
    //
    //   sc2=scales[1]&63=0, m2=scales[5]&63=0
    //   contrib2 = 0 - 0 = 0
    //
    // Other chunks: all zero scales/mins
    // Total = -320.0
    assert!(
        (result - (-320.0)).abs() < 1.0,
        "Expected about -320.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_two_super_blocks_known() {
    // 2 super-blocks: first with d=1.0, second with d=0.5
    let mut data = vec![0u8; 288];

    // Block 0: d=1.0, dmin=0, scale[0]=1, qs all = 0x22 (lo=2, hi=2)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 1;
    for i in 0..128 {
        data[16 + i] = 0x22;
    }

    // Block 1: d=0.5, dmin=0, scale[0]=1, qs all = 0x33 (lo=3, hi=3)
    data[144..146].copy_from_slice(&0x3800u16.to_le_bytes());
    data[148] = 1;
    for i in 0..128 {
        data[160 + i] = 0x33;
    }

    let q8k_scales = vec![1.0f32; 2];
    let q8k_quants = vec![1i8; 512];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Block 0, Chunk 0: sc1=1, m1=0
    //   sum_lo = 32 * (2*1) = 64
    //   contrib = 1.0*1*1.0 * 64 = 64.0
    //   High nibble: sc2=0 -> 0
    // Block 0 total = 64.0
    //
    // Block 1, Chunk 0: sc1=1, m1=0
    //   sum_lo = 32 * (3*1) = 96
    //   contrib = 0.5*1*1.0 * 96 = 48.0
    // Block 1 total = 48.0
    //
    // Grand total = 64.0 + 48.0 = 112.0
    assert!(
        (result - 112.0).abs() < 1.0,
        "Expected about 112.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_negative_q8_quants() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[4] = 1; // scale=1

    // All qs = 0x11 (lo=1, hi=1)
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-5i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Chunk 0: sc1=1, m1=0
    //   sum_lo = 32 * (1 * (-5)) = -160
    //   q8_sum_lo = 32 * (-5) = -160
    //   contrib = 1.0 * (-160) - 0 = -160.0
    // Total = -160.0
    assert!(
        (result - (-160.0)).abs() < 1.0,
        "Expected about -160.0, got {}",
        result
    );
}

// --- SIMD vs scalar parity with known values ---

#[test]
fn test_fused_q4k_dot_simd_vs_scalar_varied_activations() {
    // Use varied activations and qs patterns to stress-test parity
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin=0.5

    // Set varied scales
    data[4] = 5;
    data[5] = 10;
    data[6] = 15;
    data[7] = 20;
    data[8] = 3;
    data[9] = 7;
    data[10] = 11;
    data[11] = 14;

    // Varied qs pattern
    for i in 0..128 {
        data[16 + i] = ((i * 13 + 7) % 256) as u8;
    }

    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

    let scalar_result = fused_q4k_dot(&data, &activations).expect("scalar should succeed");
    let simd_result = fused_q4k_dot_simd(&data, &activations).expect("simd should succeed");

    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.01,
        "SIMD/scalar parity failed: scalar={}, simd={}, rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_simd_vs_scalar_varied() {
    // Varied data to exercise all code paths
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin=0.5

    // Set all scales to small known values
    for i in 0..12 {
        data[4 + i] = ((i + 1) * 3) as u8;
    }

    // Varied qs
    for i in 0..128 {
        data[16 + i] = ((i * 7 + 11) % 256) as u8;
    }

    let q8k_scales = vec![0.3f32];
    let q8k_quants: Vec<i8> = (0..256).map(|i| (i % 100) as i8 - 50).collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let rel_err = if scalar.abs() > 1e-6 {
        (simd - scalar).abs() / scalar.abs()
    } else {
        (simd - scalar).abs()
    };
    assert!(
        rel_err < 0.02,
        "Q4K×Q8K SIMD/scalar parity: scalar={}, simd={}, rel_err={}",
        scalar,
        simd,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_simd_multiple_super_blocks_parity() {
    // 4 super-blocks to exercise prefetch paths and multi-block accumulation
    let mut data = vec![0u8; 576]; // 4 * 144

    for sb in 0..4 {
        let offset = sb * 144;
        // d = 0.5
        data[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
        // dmin = 0.25
        data[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
        // Set varied scales
        for i in 0..12 {
            data[offset + 4 + i] = ((sb * 5 + i * 3 + 1) % 63) as u8;
        }
        // Set varied qs
        for i in 0..128 {
            data[offset + 16 + i] = ((sb * 17 + i * 11 + 3) % 256) as u8;
        }
    }

    let q8k_scales = vec![0.2f32; 4];
    let q8k_quants: Vec<i8> = (0..1024)
        .map(|i| (((i * 3 + 7) % 200) as i32 - 100) as i8)
        .collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let rel_err = if scalar.abs() > 1e-6 {
        (simd - scalar).abs() / scalar.abs()
    } else {
        (simd - scalar).abs()
    };
    assert!(
        rel_err < 0.02,
        "4-superblock parity: scalar={}, simd={}, rel_err={}",
        scalar,
        simd,
        rel_err
    );
}

// --- fused_q4k_dot: additional error path tests ---

#[test]
fn test_fused_q4k_dot_error_message_contains_details() {
    let data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];
    let err = fused_q4k_dot(&data, &activations).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("100"),
        "Error should contain data length: {}",
        msg
    );
    assert!(
        msg.contains("144"),
        "Error should contain block size: {}",
        msg
    );
}

#[test]
fn test_fused_q4k_dot_activation_error_message() {
    let data = vec![0u8; 144];
    let activations = vec![0.0f32; 100]; // Wrong
    let err = fused_q4k_dot(&data, &activations).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("100"),
        "Error should contain act length: {}",
        msg
    );
    assert!(
        msg.contains("256"),
        "Error should contain expected count: {}",
        msg
    );
}
