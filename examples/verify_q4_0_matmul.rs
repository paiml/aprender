//! Verify Q4_0 matmul produces correct results
//!
//! This test checks if our Q4_0 dequantize + matmul gives same results as
//! manual computation with dequantized weights.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Q4_0 Matmul Verification ===\n");

    // Get the first layer's Q weight
    let layer = &model.layers()[0];
    let realizar::gguf::OwnedQKVWeights::Separate { q: q_weight, .. } = &layer.qkv_weight else {
        panic!("Expected separate Q/K/V weights")
    };

    println!("Q weight tensor:");
    println!("  in_dim: {}", q_weight.in_dim);
    println!("  out_dim: {}", q_weight.out_dim);
    println!("  qtype: {}", q_weight.qtype);
    println!("  data len: {} bytes", q_weight.data.len());

    // Create a simple test input (all 1s should give us sum of each row)
    let hidden_dim = model.config().hidden_dim;
    let input = vec![1.0f32; hidden_dim];

    // Use the model's QKV matmul
    let qkv_output = model.qkv_matmul(&input, &layer.qkv_weight)?;

    let q_dim = q_weight.out_dim;
    let q_output = &qkv_output[0..q_dim];

    println!("\nQ matmul output (input = all 1.0):");
    println!("  Q output len: {}", q_output.len());
    println!("  Q output first 8: {:?}", &q_output[..8]);
    println!("  Q output sum: {:.4}", q_output.iter().sum::<f32>());
    println!(
        "  Q output norm: {:.4}",
        q_output.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Now manually dequantize and verify
    // Q4_0: 18 bytes per 32 elements (2 byte f16 scale + 16 bytes of packed 4-bit values)
    const Q4_0_BLOCK_SIZE: usize = 32;
    const Q4_0_BLOCK_BYTES: usize = 18;

    let blocks_per_row = hidden_dim / Q4_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;

    println!("\nManual verification:");
    println!("  blocks_per_row: {}", blocks_per_row);
    println!("  bytes_per_row: {}", bytes_per_row);
    println!(
        "  expected total bytes: {}",
        q_weight.out_dim * bytes_per_row
    );

    // Dequantize first row and compute dot product manually
    let first_row_bytes = &q_weight.data[0..bytes_per_row];

    // Dequantize the first row
    let mut first_row_dequant = vec![0.0f32; hidden_dim];
    for block_idx in 0..blocks_per_row {
        let block_start = block_idx * Q4_0_BLOCK_BYTES;
        let block = &first_row_bytes[block_start..block_start + Q4_0_BLOCK_BYTES];

        // First 2 bytes are f16 scale
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        // Next 16 bytes are 32 4-bit values (16 bytes * 2 nibbles = 32 values)
        for i in 0..16 {
            let byte = block[2 + i];
            // Lower nibble first, then upper nibble
            let val0 = (byte & 0x0F) as i8 - 8; // Q4_0 values are offset by 8
            let val1 = ((byte >> 4) & 0x0F) as i8 - 8;

            let idx0 = block_idx * Q4_0_BLOCK_SIZE + i * 2;
            let idx1 = block_idx * Q4_0_BLOCK_SIZE + i * 2 + 1;

            if idx0 < hidden_dim {
                first_row_dequant[idx0] = val0 as f32 * scale;
            }
            if idx1 < hidden_dim {
                first_row_dequant[idx1] = val1 as f32 * scale;
            }
        }
    }

    // Compute dot product with input (all 1.0) = sum of weights
    let manual_sum: f32 = first_row_dequant.iter().sum();
    println!("\n  First row dequantized:");
    println!("    first 8 weights: {:?}", &first_row_dequant[..8]);
    println!("    manual row sum: {:.4}", manual_sum);
    println!("    model Q output[0]: {:.4}", q_output[0]);
    println!("    match: {}", (manual_sum - q_output[0]).abs() < 1.0);

    // If they don't match, the weight layout might be transposed
    // Let's check if treating it as column-major gives correct result
    if (manual_sum - q_output[0]).abs() > 1.0 {
        println!("\n  WARNING: Manual computation doesn't match model output!");
        println!("  This suggests a weight layout issue (row-major vs column-major)");

        // Try column-major interpretation
        // In column-major, first row would be elements [0, out_dim, 2*out_dim, ...]
        let mut col_major_row = vec![0.0f32; hidden_dim];
        #[allow(clippy::needless_range_loop)] // col used in complex indexing calculations
        for col in 0..hidden_dim {
            let row_in_col_major = 0; // We want first row
            let linear_idx = col * q_weight.out_dim + row_in_col_major;
            let block_idx = linear_idx / Q4_0_BLOCK_SIZE;
            let block_start = block_idx * Q4_0_BLOCK_BYTES;

            if block_start + Q4_0_BLOCK_BYTES > q_weight.data.len() {
                continue;
            }

            let block = &q_weight.data[block_start..block_start + Q4_0_BLOCK_BYTES];
            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();

            let idx_in_block = linear_idx % Q4_0_BLOCK_SIZE;
            let byte_idx = idx_in_block / 2;
            let byte = block[2 + byte_idx];

            let val = if idx_in_block.is_multiple_of(2) {
                (byte & 0x0F) as i8 - 8
            } else {
                ((byte >> 4) & 0x0F) as i8 - 8
            };

            col_major_row[col] = val as f32 * scale;
        }

        let col_major_sum: f32 = col_major_row.iter().sum();
        println!("  Column-major interpretation:");
        println!("    first 8 weights: {:?}", &col_major_row[..8]);
        println!("    col-major row sum: {:.4}", col_major_sum);
    }

    Ok(())
}
