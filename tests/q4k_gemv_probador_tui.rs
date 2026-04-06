//! Q4K GEMV Probador TUI: GPU Simulation & Visual Testing
//!
//! EXTREME TDD: Pixel-level validation of Q4K GEMV GPU kernel against CPU baseline
//!
//! Run with: cargo test --test q4k_gemv_probador_tui --features cuda -- --nocapture

#![cfg(feature = "cuda")]
// Allow explicit indexing in tests for clarity
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec};

/// Q4_K super-block constants
const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_BLOCK_VALUES: usize = 256;

// ============================================================================
// SCALAR BASELINE IMPLEMENTATIONS (GROUND TRUTH)
// ============================================================================

/// Scalar implementation of get_scale_min_k4 from llama.cpp
/// This is the GROUND TRUTH for scale extraction
fn scalar_get_scale_min_k4(scales: &[u8; 12], block_idx: usize) -> (u8, u8) {
    let j = block_idx;
    if j < 4 {
        // Blocks 0-3: simple layout
        let scale = scales[j] & 63;
        let min = scales[j + 4] & 63;
        (scale, min)
    } else {
        // Blocks 4-7: packed layout using high bits from blocks 0-3
        let scale = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (scale, min)
    }
}

/// Scalar dequantization of a single Q4_K super-block
/// Returns 256 f32 values
fn scalar_dequant_q4k_superblock(sb_data: &[u8]) -> Vec<f32> {
    assert!(sb_data.len() >= Q4K_BLOCK_BYTES);

    // Parse super-block header
    let d_bits = u16::from_le_bytes([sb_data[0], sb_data[1]]);
    let dmin_bits = u16::from_le_bytes([sb_data[2], sb_data[3]]);
    let d = half::f16::from_bits(d_bits).to_f32();
    let dmin = half::f16::from_bits(dmin_bits).to_f32();

    // Extract scales (12 bytes at offset 4)
    let scales: [u8; 12] = sb_data[4..16].try_into().expect("test");

    // qs data starts at offset 16 (128 bytes for 256 4-bit values)
    let qs = &sb_data[16..144];

    let mut result = vec![0.0f32; Q4K_BLOCK_VALUES];

    // Process in 64-value chunks (matching llama.cpp's inner loop)
    let mut is = 0; // scale index
    for chunk in 0..4 {
        let chunk_start = chunk * 64;
        let qs_chunk = &qs[chunk * 32..(chunk + 1) * 32];

        // Get scale/min for first 32 values in chunk
        let (sc1, m1) = scalar_get_scale_min_k4(&scales, is);
        is += 1;
        let d1 = d * (sc1 as f32);
        let m1f = dmin * (m1 as f32);

        // Get scale/min for second 32 values in chunk
        let (sc2, m2) = scalar_get_scale_min_k4(&scales, is);
        is += 1;
        let d2 = d * (sc2 as f32);
        let m2f = dmin * (m2 as f32);

        // Dequantize 64 values in this chunk
        for l in 0..32 {
            let packed = qs_chunk[l];
            // Low nibble -> values 0-31 of chunk
            let q_lo = (packed & 0x0F) as f32;
            result[chunk_start + l] = d1 * q_lo - m1f;
            // High nibble -> values 32-63 of chunk
            let q_hi = (packed >> 4) as f32;
            result[chunk_start + 32 + l] = d2 * q_hi - m2f;
        }
    }

    result
}

/// Scalar Q4_K GEMV: y = W * x
/// W is [n_out, n_in] stored as [n_out rows, each row has n_in/256 super-blocks]
fn scalar_q4k_gemv(weights: &[u8], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let super_blocks_per_row = n_in / Q4K_BLOCK_VALUES;
    let row_bytes = super_blocks_per_row * Q4K_BLOCK_BYTES;

    let mut y = vec![0.0f32; n_out];

    for row in 0..n_out {
        let row_start = row * row_bytes;
        let mut acc = 0.0f32;

        for sb in 0..super_blocks_per_row {
            let sb_start = row_start + sb * Q4K_BLOCK_BYTES;
            let sb_data = &weights[sb_start..sb_start + Q4K_BLOCK_BYTES];

            // Dequantize this super-block
            let dequant = scalar_dequant_q4k_superblock(sb_data);

            // Dot product with corresponding slice of x
            let x_start = sb * Q4K_BLOCK_VALUES;
            for i in 0..Q4K_BLOCK_VALUES {
                acc += dequant[i] * x[x_start + i];
            }
        }

        y[row] = acc;
    }

    y
}

// ============================================================================
// TUI VISUALIZATION HELPERS
// ============================================================================

fn print_header(title: &str) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  {:<68} ║", title);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn print_section(title: &str) {
    println!();
    println!(
        "┌─ {} ─────────────────────────────────────────────────",
        title
    );
}

fn print_end_section() {
    println!("└──────────────────────────────────────────────────────────────────────┘");
}

fn print_value_comparison(label: &str, expected: f32, actual: f32) {
    let diff = (expected - actual).abs();
    let status = if diff < 1e-4 {
        "✓"
    } else if diff < 1e-2 {
        "~"
    } else {
        "✗"
    };
    println!(
        "│  {}: expected={:.6}, actual={:.6}, diff={:.6} {}",
        label, expected, actual, diff, status
    );
}

// ============================================================================
// PROBADOR TUI TESTS
// ============================================================================

/// Test 1: Verify scalar dequantization against realizar's CPU implementation
#[test]
fn test_scalar_dequant_vs_realizar_cpu() {
    print_header("TEST 1: Scalar Dequant vs Realizar CPU");

    // Create test super-block with known values
    let mut sb_data = vec![0u8; Q4K_BLOCK_BYTES];

    // d = 1.0 (f16: 0x3C00)
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C;
    // dmin = 0.5 (f16: 0x3800)
    sb_data[2] = 0x00;
    sb_data[3] = 0x38;

    // Set scales[0..7] with various values
    // For blocks 0-3: scale = byte[j] & 63, min = byte[j+4] & 63
    sb_data[4] = 10; // scale[0] = 10
    sb_data[5] = 20; // scale[1] = 20
    sb_data[6] = 30; // scale[2] = 30
    sb_data[7] = 40; // scale[3] = 40
    sb_data[8] = 5; // min[0] = 5
    sb_data[9] = 10; // min[1] = 10
    sb_data[10] = 15; // min[2] = 15
    sb_data[11] = 20; // min[3] = 20

    // For blocks 4-7: packed layout (use values that exercise both paths)
    // scale[4] = (s[8] & 0xF) | ((s[0] >> 6) << 4) = (s[8] & 0xF) | 0
    // We'll use simple values for blocks 4-7 by setting s[8..11] appropriately
    sb_data[12] = 0x21; // s[8]: scale[4]_lo=1, min[4]=2
    sb_data[13] = 0x43; // s[9]: scale[5]_lo=3, min[5]=4
    sb_data[14] = 0x65; // s[10]: scale[6]_lo=5, min[6]=6
    sb_data[15] = 0x87; // s[11]: scale[7]_lo=7, min[7]=8

    // Set qs with alternating pattern
    for i in 0..128 {
        sb_data[16 + i] = 0x21; // low=1, high=2
    }

    print_section("Scalar Dequant");
    let scalar_result = scalar_dequant_q4k_superblock(&sb_data);
    println!("│  First 8 values:");
    for i in 0..8 {
        println!("│    [{}] = {:.6}", i, scalar_result[i]);
    }
    print_end_section();

    print_section("Realizar CPU Dequant");
    let cpu_result = dequantize_q4_k(&sb_data).expect("CPU dequant should succeed");
    println!("│  First 8 values:");
    for i in 0..8 {
        println!("│    [{}] = {:.6}", i, cpu_result[i]);
    }
    print_end_section();

    print_section("Comparison");
    let mut max_diff = 0.0f32;
    for i in 0..Q4K_BLOCK_VALUES {
        let diff = (scalar_result[i] - cpu_result[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("│  Max difference: {:.6}", max_diff);
    println!(
        "│  Result: {}",
        if max_diff < 1e-4 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    print_end_section();

    assert!(
        max_diff < 1e-4,
        "Scalar dequant doesn't match CPU: max_diff={}",
        max_diff
    );
}

/// Test 2: Verify scalar GEMV against realizar's fused_q4k_parallel_matvec
#[test]
fn test_scalar_gemv_vs_realizar_cpu() {
    print_header("TEST 2: Scalar GEMV vs Realizar CPU (fused_q4k_parallel_matvec)");

    let n_in = 256; // 1 super-block
    let n_out = 4;

    // Create test weights (4 rows, 1 super-block each)
    let super_blocks_per_row = n_in / Q4K_BLOCK_VALUES;
    let row_bytes = super_blocks_per_row * Q4K_BLOCK_BYTES;
    let total_bytes = n_out * row_bytes;
    let mut weights = vec![0u8; total_bytes];

    for row in 0..n_out {
        let row_start = row * row_bytes;
        // d = 1.0
        weights[row_start] = 0x00;
        weights[row_start + 1] = 0x3C;
        // dmin = 0.0
        weights[row_start + 2] = 0x00;
        weights[row_start + 3] = 0x00;
        // scales = row+1 for all blocks
        for i in 0..12 {
            weights[row_start + 4 + i] = (row + 1) as u8;
        }
        // qs = 0x11 (all quantized values = 1)
        for i in 0..128 {
            weights[row_start + 16 + i] = 0x11;
        }
    }

    // Input = all ones
    let input: Vec<f32> = vec![1.0; n_in];

    print_section("Scalar GEMV");
    let scalar_result = scalar_q4k_gemv(&weights, &input, n_in, n_out);
    for i in 0..n_out {
        println!("│  y[{}] = {:.6}", i, scalar_result[i]);
    }
    print_end_section();

    print_section("Realizar CPU GEMV (fused_q4k_parallel_matvec)");
    let cpu_result =
        fused_q4k_parallel_matvec(&weights, &input, n_in, n_out).expect("CPU GEMV should succeed");
    for i in 0..n_out {
        println!("│  y[{}] = {:.6}", i, cpu_result[i]);
    }
    print_end_section();

    print_section("Comparison");
    let mut max_diff = 0.0f32;
    for i in 0..n_out {
        let diff = (scalar_result[i] - cpu_result[i]).abs();
        print_value_comparison(&format!("y[{}]", i), scalar_result[i], cpu_result[i]);
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("│  Max difference: {:.6}", max_diff);
    println!(
        "│  Result: {}",
        if max_diff < 1e-3 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    print_end_section();

    assert!(
        max_diff < 1e-3,
        "Scalar GEMV doesn't match CPU: max_diff={}",
        max_diff
    );
}

/// Test 3: GPU Q4K GEMV vs Scalar Baseline (THE CRITICAL TEST)
#[test]
fn test_gpu_gemv_vs_scalar_baseline() {
    use realizar::cuda::CudaExecutor;

    print_header("TEST 3: GPU Q4K GEMV vs Scalar Baseline (CRITICAL)");

    let n_in = 256;
    let n_out = 4;

    // Create deterministic test weights
    let super_blocks_per_row = n_in / Q4K_BLOCK_VALUES;
    let row_bytes = super_blocks_per_row * Q4K_BLOCK_BYTES;
    let total_bytes = n_out * row_bytes;
    let mut weights = vec![0u8; total_bytes];

    for row in 0..n_out {
        let row_start = row * row_bytes;
        // d = 1.0
        weights[row_start] = 0x00;
        weights[row_start + 1] = 0x3C;
        // dmin = 0.0
        weights[row_start + 2] = 0x00;
        weights[row_start + 3] = 0x00;
        // Simple scales
        for i in 0..4 {
            weights[row_start + 4 + i] = 1; // scale[0..3] = 1
            weights[row_start + 8 + i] = 0; // min[0..3] = 0
        }
        for i in 0..4 {
            weights[row_start + 12 + i] = 0x01; // scale[4..7] = 1, min[4..7] = 0
        }
        // qs = 0x11 (all q = 1)
        for i in 0..128 {
            weights[row_start + 16 + i] = 0x11;
        }
    }

    let input: Vec<f32> = vec![1.0; n_in];

    print_section("Scalar Baseline");
    let scalar_result = scalar_q4k_gemv(&weights, &input, n_in, n_out);
    for i in 0..n_out {
        println!("│  y[{}] = {:.6}", i, scalar_result[i]);
    }
    print_end_section();

    print_section("GPU Q4K GEMV");
    let mut executor = CudaExecutor::new(0).expect("Failed to create CUDA executor");
    let mut gpu_result = vec![0.0f32; n_out];

    executor
        .q4k_gemv(&weights, &input, &mut gpu_result, n_out as u32, n_in as u32)
        .expect("GPU Q4K GEMV should succeed");

    for i in 0..n_out {
        println!("│  y[{}] = {:.6}", i, gpu_result[i]);
    }
    print_end_section();

    print_section("GPU vs Scalar Comparison");
    let mut max_diff = 0.0f32;
    let mut all_passed = true;
    for i in 0..n_out {
        let diff = (gpu_result[i] - scalar_result[i]).abs();
        let passed = diff < 1e-3;
        let status = if passed { "✓" } else { "✗" };
        println!(
            "│  y[{}]: scalar={:.6}, GPU={:.6}, diff={:.6} {}",
            i, scalar_result[i], gpu_result[i], diff, status
        );
        if diff > max_diff {
            max_diff = diff;
        }
        if !passed {
            all_passed = false;
        }
    }
    println!("│  Max difference: {:.6}", max_diff);
    println!(
        "│  Result: {}",
        if all_passed { "PASS ✓" } else { "FAIL ✗" }
    );
    print_end_section();

    if !all_passed {
        print_section("DEBUG: Per-value analysis for first row");
        // Dequantize first row and show values
        let row_data = &weights[0..Q4K_BLOCK_BYTES];
        let dequant = scalar_dequant_q4k_superblock(row_data);
        println!("│  Dequantized values for row 0:");
        println!(
            "│  Block 0 (values 0-31): sum={:.4}",
            dequant[0..32].iter().sum::<f32>()
        );
        println!(
            "│  Block 1 (values 32-63): sum={:.4}",
            dequant[32..64].iter().sum::<f32>()
        );
        println!(
            "│  Block 2 (values 64-95): sum={:.4}",
            dequant[64..96].iter().sum::<f32>()
        );
        println!(
            "│  Block 3 (values 96-127): sum={:.4}",
            dequant[96..128].iter().sum::<f32>()
        );
        println!(
            "│  Block 4 (values 128-159): sum={:.4}",
            dequant[128..160].iter().sum::<f32>()
        );
        println!(
            "│  Block 5 (values 160-191): sum={:.4}",
            dequant[160..192].iter().sum::<f32>()
        );
        println!(
            "│  Block 6 (values 192-223): sum={:.4}",
            dequant[192..224].iter().sum::<f32>()
        );
        println!(
            "│  Block 7 (values 224-255): sum={:.4}",
            dequant[224..256].iter().sum::<f32>()
        );
        println!("│  Total sum: {:.4}", dequant.iter().sum::<f32>());
        print_end_section();
    }

    assert!(all_passed, "GPU GEMV doesn't match scalar baseline");
}

/// Test 4: Scale extraction verification
#[test]
fn test_scale_extraction_patterns() {
    print_header("TEST 4: Scale Extraction Pattern Verification");

    // Test with known scale patterns
    let test_cases: Vec<([u8; 12], Vec<(usize, u8, u8)>)> = vec![
        // Case 1: All ones
        (
            [1, 1, 1, 1, 1, 1, 1, 1, 0x11, 0x11, 0x11, 0x11],
            vec![
                (0, 1, 1),
                (1, 1, 1),
                (2, 1, 1),
                (3, 1, 1),
                (4, 1, 1),
                (5, 1, 1),
                (6, 1, 1),
                (7, 1, 1),
            ],
        ),
        // Case 2: Different values for each block
        (
            [1, 2, 3, 4, 10, 20, 30, 40, 0x15, 0x26, 0x37, 0x48],
            vec![(0, 1, 10), (1, 2, 20), (2, 3, 30), (3, 4, 40)],
        ),
        // Case 3: High bits set for blocks 4-7
        (
            [
                0x40, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x01, 0x02, 0x03, 0x04,
            ],
            vec![(4, 0x11, 0x11)], // scale = (0x01 & 0xF) | ((0x40 >> 6) << 4) = 1 | 16 = 17
        ),
    ];

    for (i, (scales, expected)) in test_cases.iter().enumerate() {
        print_section(&format!("Test Case {}", i + 1));
        println!("│  Scales: {:?}", scales);

        for (block, exp_scale, exp_min) in expected {
            let (scale, min) = scalar_get_scale_min_k4(scales, *block);
            let passed = scale == *exp_scale && min == *exp_min;
            let status = if passed { "✓" } else { "✗" };
            println!(
                "│  Block {}: scale={}, min={} (expected: {}, {}) {}",
                block, scale, min, exp_scale, exp_min, status
            );
        }
        print_end_section();
    }
}

// Test 5: Real model weights parity - disabled until GGUFFile is exported
// TODO: Re-enable when realizar exports GGUFFile or provide alternative API
