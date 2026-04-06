
/// Integration test: Q4_1 matmul produces correct output
/// This tests the entire Q4_1 dequantize + trueno Matrix path
#[test]
fn test_q4_1_matmul_integration() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Create a small 2x2 matmul (2 output rows, each with 32 elements = 1 block)
    let in_dim = 32;
    let out_dim = 2;

    // Create Q4_1 data for 2 rows (2 blocks × 20 bytes = 40 bytes)
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes quants = 20 bytes
    let mut data = vec![0u8; 40];

    // Row 0: scale=1.0, min=0.0, byte[0]=0x10 -> pos 0=0, pos 16=1
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data[4] = 0x10; // low=0 (pos 0), high=1 (pos 16)
                    // Rest of quants are 0

    // Row 1: scale=1.0, min=0.0, all quants give 1.0 for all positions
    data[20..22].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[22..24].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // Fill row 1 quants with 0x11 -> all positions get value 1
    for i in 24..40 {
        data[i] = 0x11;
    }

    // Dequantize
    let weights_f32 = dequantize_q4_1(&data).expect("dequantization failed");
    assert_eq!(weights_f32.len(), out_dim * in_dim);

    // Debug: Print first few values of each row
    eprintln!("Row 0 (first 20): {:?}", &weights_f32[0..20]);
    eprintln!("Row 0 positions 16-20: {:?}", &weights_f32[16..20]);
    eprintln!("Row 1 (first 20): {:?}", &weights_f32[32..52]);

    // Create Matrix [out_dim, in_dim] = [2, 32]
    let weight_matrix = TruenoMatrix::from_vec(out_dim, in_dim, weights_f32.clone())
        .expect("matrix creation failed");

    // Create activation vector: all 1.0
    let activations = vec![1.0f32; in_dim];
    let x_vec = TruenoVector::from_slice(&activations);

    // Compute matmul
    let result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    eprintln!("Matmul result: {:?}", result.as_slice());

    // Compute expected by manual sum
    let expected_row0: f32 = weights_f32[0..32].iter().sum();
    let expected_row1: f32 = weights_f32[32..64].iter().sum();
    eprintln!("Expected row 0 sum (manual): {}", expected_row0);
    eprintln!("Expected row 1 sum (manual): {}", expected_row1);

    // Row 0: only position 16 has value 1.0, rest are 0
    // Sum should be 1.0
    let row0_sum = result.as_slice()[0];

    // Row 1: all positions have value 1.0
    // Sum of [1, 1, ..., 1] = 32.0
    let row1_sum = result.as_slice()[1];

    assert!(
        (row0_sum - expected_row0).abs() < 0.1,
        "Row 0 sum should be {}, got {}",
        expected_row0,
        row0_sum
    );
    assert!(
        (row1_sum - expected_row1).abs() < 0.1,
        "Row 1 sum should be {}, got {}",
        expected_row1,
        row1_sum
    );
}

/// Integration test: Q4_1 matmul with realistic dimensions (896x896)
#[test]
fn test_q4_1_matmul_large_dimensions() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Qwen2-0.5B dimensions: hidden_dim=896
    let in_dim = 896;
    let out_dim = 896;

    // blocks_per_row = 896 / 32 = 28
    // bytes_per_row = 28 * 20 = 560
    let blocks_per_row = in_dim / 32;
    let bytes_per_row = blocks_per_row * 20;
    let total_bytes = out_dim * bytes_per_row;

    // Create Q4_1 data: all zeros except row 0 has scale=1.0
    let mut data = vec![0u8; total_bytes];

    // Row 0: scale=1.0, min=0.0, quants all 0x11 (value 1 for all positions)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    for i in 4..bytes_per_row {
        if i >= 4 && (i - 4) % 20 < 16 {
            // Only set quant bytes, not scale/min of subsequent blocks
            data[i] = 0x11;
        }
    }

    // Dequantize
    let weights_f32 = dequantize_q4_1(&data).expect("dequantization failed");

    // Verify dimensions
    assert_eq!(
        weights_f32.len(),
        out_dim * in_dim,
        "Dequantized size mismatch: {} vs {}",
        weights_f32.len(),
        out_dim * in_dim
    );

    // Create Matrix
    let weight_matrix =
        TruenoMatrix::from_vec(out_dim, in_dim, weights_f32).expect("matrix creation failed");

    // Create activation: all 1.0
    let activations = vec![1.0f32; in_dim];
    let x_vec = TruenoVector::from_slice(&activations);

    // Compute matmul
    let result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    assert_eq!(result.len(), out_dim);

    // Row 0 should have some non-zero sum (from the first block's values)
    // Other rows should be ~0 (scale=0 from zero-initialized data)
    let row0_sum = result.as_slice()[0];
    assert!(
        row0_sum.abs() > 0.1,
        "Row 0 should have non-zero output, got {}",
        row0_sum
    );

    // Verify no NaN or Inf
    for (i, &v) in result.as_slice().iter().enumerate() {
        assert!(v.is_finite(), "Output at {} is not finite: {}", i, v);
    }
}

/// Falsification test: Q4_1 should NOT use interleaved layout
/// If this test fails, the code is using wrong (interleaved) layout
#[test]
fn test_q4_1_not_interleaved_layout() {
    let mut block = vec![0u8; 20];

    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    // byte[0] = 0x10 -> low=0, high=1
    // In WRONG interleaved layout: pos 0 = 0, pos 1 = 1
    // In CORRECT candle layout: pos 0 = 0, pos 16 = 1
    block[4] = 0x10;

    let result = dequantize_q4_1(&block).expect("dequantization failed");

    // If interleaved (WRONG): result[1] would be 1.0
    // If candle (CORRECT): result[1] should be 0.0 (from byte 1 which is 0)
    assert!(
        (result[1] - 0.0).abs() < 1e-5,
        "INTERLEAVED LAYOUT DETECTED! pos 1 should be 0.0 (candle), got {} (interleaved would give 1.0)",
        result[1]
    );

    // Verify high nibble goes to position 16
    assert!(
        (result[16] - 1.0).abs() < 1e-5,
        "CANDLE LAYOUT BROKEN! pos 16 should be 1.0, got {}",
        result[16]
    );
}

// =========================================================================
// BUG-GGUF-001: Q4_0 Falsification Tests
// Compare fused Q4_0 × Q8_0 kernel against reference (dequantize + naive)
// =========================================================================

/// Falsification test: fused Q4_0 matmul vs dequantize + TruenoMatrix path
/// If these produce different results, one of the paths has a bug.
#[test]
fn test_q4_0_fused_vs_dequantize_matmul() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Small test: 2 output rows, 32 input dims (1 block per row)
    let in_dim = 32;
    let out_dim = 2;

    // Q4_0 block: 2 bytes scale + 16 bytes quants = 18 bytes per 32 elements
    let bytes_per_row = 18;
    let total_bytes = out_dim * bytes_per_row;
    let mut data = vec![0u8; total_bytes];

    // Row 0: scale=1.0, quants pattern [0,1,2,...,15] for low nibbles
    //        high nibbles will be [0,0,...,0]
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Set quant bytes: 0x10, 0x32, 0x54, ... for positions 0-15 = 0,2,4,...
    // Q4_0 uses signed: value = (nibble - 8) * scale
    for i in 0..16 {
        // Put increasing value in low nibble (0,1,2,...,15)
        // Put 8 in high nibble so high_quant = 8 - 8 = 0
        data[2 + i] = (8 << 4) | (i as u8);
    }

    // Row 1: scale=2.0, all quants = 0x99 (low=9, high=9)
    // Signed: 9 - 8 = 1, so value = 2.0 * 1 = 2.0 for all positions
    data[18..20].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    for i in 0..16 {
        data[20 + i] = 0x99;
    }

    // Path 1: Fused Q4_0 × Q8_0 matmul
    let activations = vec![1.0f32; in_dim];
    let fused_result = fused_q4_0_q8_0_parallel_matvec(&data, &activations, in_dim, out_dim)
        .expect("fused matmul failed");

    // Path 2: Dequantize + TruenoMatrix matmul (reference)
    let weights_f32 = dequantize_q4_0(&data).expect("dequantize failed");
    let weight_matrix = TruenoMatrix::from_vec(out_dim, in_dim, weights_f32.clone())
        .expect("matrix creation failed");
    let x_vec = TruenoVector::from_slice(&activations);
    let reference_result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    eprintln!("Q4_0 Fused result: {:?}", fused_result);
    eprintln!("Q4_0 Reference result: {:?}", reference_result.as_slice());
    eprintln!("Q4_0 Row 0 weights (first 20): {:?}", &weights_f32[0..20]);
    eprintln!("Q4_0 Row 0 weights (pos 16-20): {:?}", &weights_f32[16..20]);

    // Compare results
    for i in 0..out_dim {
        let fused_val = fused_result[i];
        let ref_val = reference_result.as_slice()[i];
        let diff = (fused_val - ref_val).abs();

        // Allow small tolerance for quantization noise
        assert!(
            diff < 1.0,
            "Q4_0 MISMATCH at row {}: fused={}, reference={}, diff={}",
            i,
            fused_val,
            ref_val,
            diff
        );
    }
}

/// Falsification test: Q4_0 fused matmul with Qwen2-0.5B dimensions
#[test]
fn test_q4_0_fused_matmul_qwen_dimensions() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Qwen2-0.5B: hidden_dim=896
    let in_dim: usize = 896;
    let out_dim: usize = 128; // Smaller for faster test

    // Q4_0: 18 bytes per 32 elements
    let blocks_per_row = in_dim.div_ceil(32);
    let bytes_per_row = blocks_per_row * 18;
    let total_bytes = out_dim * bytes_per_row;

    // Create deterministic test data
    let mut data = vec![0u8; total_bytes];
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        for block in 0..blocks_per_row {
            let block_start = row_start + block * 18;

            // Scale varies by row
            let scale = 0.1 + (row as f32) * 0.01;
            data[block_start..block_start + 2]
                .copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());

            // Fill quants with deterministic pattern
            for i in 0..16 {
                data[block_start + 2 + i] =
                    (((row + block + i) % 16) << 4 | ((row + i) % 16)) as u8;
            }
        }
    }

    // Create random-ish activations (deterministic)
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Path 1: Fused matmul
    let fused_result = fused_q4_0_q8_0_parallel_matvec(&data, &activations, in_dim, out_dim)
        .expect("fused matmul failed");

    // Path 2: Reference path
    let weights_f32 = dequantize_q4_0(&data).expect("dequantize failed");
    let weight_matrix =
        TruenoMatrix::from_vec(out_dim, in_dim, weights_f32).expect("matrix creation failed");
    let x_vec = TruenoVector::from_slice(&activations);
    let reference_result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    // Compare results
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..out_dim {
        let diff = (fused_result[i] - reference_result.as_slice()[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    eprintln!(
        "Q4_0 Qwen dims: max diff = {} at row {}",
        max_diff, max_diff_idx
    );
    eprintln!("Fused[{}] = {}", max_diff_idx, fused_result[max_diff_idx]);
    eprintln!(
        "Reference[{}] = {}",
        max_diff_idx,
        reference_result.as_slice()[max_diff_idx]
    );

    // Q8_0 quantization introduces error, so allow larger tolerance
    // But systematic bugs would cause HUGE differences
    assert!(
        max_diff < 50.0,
        "Q4_0 fused vs reference max diff {} is too large (indicates bug)",
        max_diff
    );
}
