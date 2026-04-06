//! CORRECTNESS-002: Debug single Q6K row computation step by step
//!
//! Compares CPU Q6K dot with hand-traced computation.
//!
//! Run with: cargo run --release --features cuda --example debug_q6k_single_row

use realizar::quantize::fused_q6k_dot;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Q6K single row debug\n");

    // Create one super-block (256 values, 210 bytes) with known pattern
    let hidden_dim = 256;
    let bytes_per_row = 210;

    // Q6_K layout for one super-block:
    // - ql[128]: bytes 0-127 (low 4 bits, 2 values per byte)
    // - qh[64]: bytes 128-191 (high 2 bits, 4 values per byte)
    // - scales[16]: bytes 192-207 (signed i8 per 16-element sub-block)
    // - d: bytes 208-209 (f16 global scale)

    let mut q6k_data = vec![0u8; bytes_per_row];

    // Set simple known values:
    // - All ql bytes = 0x11 (low nibble = 1, high nibble = 1)
    // - All qh bytes = 0x00 (no high bits)
    // - All scales = 1
    // - d = 1.0

    // ql[0..128] = 0x11 (value 1 for both nibbles)
    for byte in q6k_data.iter_mut().take(128) {
        *byte = 0x11; // low nibble = 1, high nibble = 1
    }

    // qh[128..192] = 0x00 (no high bits)
    for byte in q6k_data.iter_mut().take(192).skip(128) {
        *byte = 0x00;
    }

    // scales[192..208] = 1 (positive scale)
    for byte in q6k_data.iter_mut().take(208).skip(192) {
        *byte = 1;
    }

    // d = 1.0 in f16
    let d_f16 = half::f16::from_f32(1.0);
    q6k_data[208..210].copy_from_slice(&d_f16.to_bits().to_le_bytes());

    // Input: all ones
    let input: Vec<f32> = vec![1.0; hidden_dim];

    // Hand-compute expected result:
    // For each of 256 values:
    //   - ql value: 1 (from either nibble)
    //   - qh value: 0
    //   - quant = (ql | (qh << 4)) - 32 = (1 | 0) - 32 = -31
    //   - dequant = d * scale * quant = 1.0 * 1 * (-31) = -31.0
    //   - contribution = dequant * input = -31.0 * 1.0 = -31.0
    // Total = 256 * (-31.0) = -7936.0
    let expected = 256.0 * (-31.0);
    eprintln!("Expected (hand computed): {:.2}", expected);

    // CPU reference
    let cpu_result = fused_q6k_dot(&q6k_data, &input)?;
    eprintln!("CPU fused_q6k_dot result: {:.2}", cpu_result);

    if (cpu_result - expected).abs() < 0.01 {
        eprintln!("[OK] CPU matches expected");
    } else {
        eprintln!("[FAIL] CPU diverges from expected!");

        // Debug: trace through the CPU algorithm manually
        eprintln!("\nDebugging CPU algorithm:");

        let d = half::f16::from_bits(u16::from_le_bytes([q6k_data[208], q6k_data[209]])).to_f32();
        eprintln!("d = {:.4}", d);

        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = q6k_data[192 + i] as i8;
        }
        eprintln!("scales = {:?}", scales);

        // Trace first few values
        let ql = &q6k_data[0..128];
        let qh = &q6k_data[128..192];

        eprintln!("\nTracing first 128 values (n=0):");
        for l in 0..4 {
            let is = l / 16;

            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

            eprintln!(
                "l={}: ql[l]=0x{:02x}, ql[l+32]=0x{:02x}, qh[l]=0x{:02x}",
                l,
                ql[l],
                ql[l + 32],
                qh[l]
            );
            eprintln!("      q1={}, q2={}, q3={}, q4={}", q1, q2, q3, q4);
            eprintln!(
                "      is={}, scales: sc[is]={}, sc[is+2]={}, sc[is+4]={}, sc[is+6]={}",
                is,
                scales[is],
                scales[is + 2],
                scales[is + 4],
                scales[is + 6]
            );

            let v1 = d * (scales[is] as f32) * (q1 as f32);
            let v2 = d * (scales[is + 2] as f32) * (q2 as f32);
            let v3 = d * (scales[is + 4] as f32) * (q3 as f32);
            let v4 = d * (scales[is + 6] as f32) * (q4 as f32);
            eprintln!(
                "      v1={:.2}, v2={:.2}, v3={:.2}, v4={:.2}",
                v1, v2, v3, v4
            );
        }
    }

    // Now test with actual model data
    eprintln!("\n=== Testing with actual LM head row ===");

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    if std::path::Path::new(model_path).exists() {
        use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

        let mapped = MappedGGUFModel::from_path(model_path)?;
        let model = OwnedQuantizedModel::from_mapped(&mapped)?;

        let hidden_dim = model.config().hidden_dim;
        let vocab_size = model.config().vocab_size;
        let sb_per_row = hidden_dim.div_ceil(256);
        let bytes_per_row = sb_per_row * 210;

        eprintln!(
            "hidden_dim={}, vocab_size={}, sb_per_row={}, bytes_per_row={}",
            hidden_dim, vocab_size, sb_per_row, bytes_per_row
        );

        // Test with simple input
        let test_input: Vec<f32> = vec![1.0; hidden_dim];

        // Get first few rows of LM head
        let lm_data = &model.lm_head_weight().data;
        eprintln!("LM head data length: {}", lm_data.len());

        for row in [0, 1, 2] {
            let row_start = row * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &lm_data[row_start..row_end];

            let cpu_result = fused_q6k_dot(row_data, &test_input)?;
            eprintln!("Row {}: CPU dot with all-ones = {:.4}", row, cpu_result);

            // Print row's d value and first few scales
            let d =
                half::f16::from_bits(u16::from_le_bytes([row_data[208], row_data[209]])).to_f32();
            eprintln!(
                "  d = {:.6}, scales[0..8] = {:?}",
                d,
                &row_data[192..200]
                    .iter()
                    .map(|&x| x as i8)
                    .collect::<Vec<_>>()
            );
        }
    }

    Ok(())
}
