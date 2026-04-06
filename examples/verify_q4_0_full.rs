//! Full verification of Q4_0 matmul for Qwen2
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Q4_0 Matmul Verification ===\n");

    let hidden_dim = model.config().hidden_dim;
    let layer0 = &model.layers()[0];

    // Get Q weight tensor
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &layer0.qkv_weight else {
        panic!("Expected separate QKV")
    };

    println!(
        "Q weight: in_dim={}, out_dim={}, qtype={}",
        q_weight.in_dim, q_weight.out_dim, q_weight.qtype
    );
    println!("Q weight data len: {}", q_weight.data.len());

    // Create a test input (normalized embedding for token 17)
    let token_17_emb = &model.token_embedding()[17 * hidden_dim..(17 + 1) * hidden_dim];

    // RMSNorm
    let eps = model.config().eps;
    let ss: f32 = token_17_emb.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    let input: Vec<f32> = token_17_emb
        .iter()
        .zip(layer0.attn_norm_weight.iter())
        .map(|(x, w)| x * scale * w)
        .collect();

    let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nInput (after RMSNorm) norm: {:.4}", input_norm);

    // Use the model's qkv_matmul and extract Q
    let qkv_output = model.qkv_matmul(&input, &layer0.qkv_weight)?;
    let q_dim = q_weight.out_dim;
    let q_output = &qkv_output[0..q_dim];
    println!(
        "Q output (model): len={}, first 4: {:?}",
        q_output.len(),
        &q_output[..4]
    );

    // Now manually dequantize Q weight and do matmul
    // Q4_0 format: blocks of 32 values, each block has:
    // - 1 f16 scale (2 bytes)
    // - 16 bytes of packed 4-bit values (32 values, 4 bits each)
    // Total: 18 bytes per block

    let block_size = 32;
    let bytes_per_block = 18; // Q4_0: 2 (scale) + 16 (data)
    let num_blocks = q_weight.in_dim.div_ceil(block_size);

    println!("\nQ4_0 format:");
    println!("  block_size: {}", block_size);
    println!("  bytes_per_block: {}", bytes_per_block);
    println!("  num_blocks per row: {}", num_blocks);
    println!("  out_dim (rows): {}", q_weight.out_dim);
    println!(
        "  expected data len: {}",
        num_blocks * bytes_per_block * q_weight.out_dim
    );

    // Manually compute first few output values
    let mut manual_output = vec![0.0f32; q_weight.out_dim];

    #[allow(clippy::needless_range_loop)] // row used in complex offset calculations
    for row in 0..q_weight.out_dim.min(4) {
        let mut dot = 0.0f32;
        for block_idx in 0..num_blocks {
            let block_offset = (row * num_blocks + block_idx) * bytes_per_block;

            // Read f16 scale
            let scale_bytes = [q_weight.data[block_offset], q_weight.data[block_offset + 1]];
            let scale_f16 = half::f16::from_le_bytes(scale_bytes);
            let scale_f32 = scale_f16.to_f32();

            // Read and dequantize 32 values
            for i in 0..32 {
                if block_idx * 32 + i >= q_weight.in_dim {
                    break;
                }

                let byte_idx = block_offset + 2 + i / 2;
                let byte = q_weight.data[byte_idx];
                let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };

                // Q4_0 dequant: value = (nibble - 8) * scale
                let dequant = (nibble as i8 - 8) as f32 * scale_f32;

                let input_idx = block_idx * 32 + i;
                dot += dequant * input[input_idx];
            }
        }
        manual_output[row] = dot;
    }

    println!("\nManual Q output (first 4): {:?}", &manual_output[..4]);
    println!("Model Q output (first 4):  {:?}", &q_output[..4]);

    // Compare
    let max_diff = manual_output
        .iter()
        .take(4)
        .zip(q_output.iter().take(4))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\nMax difference (first 4): {:.6}", max_diff);

    if max_diff > 0.001 {
        println!("⚠️ WARNING: Significant difference detected!");

        // Debug the first block
        println!("\nDebug first block:");
        let block_offset = 0;
        let scale_bytes = [q_weight.data[block_offset], q_weight.data[block_offset + 1]];
        let scale_f16 = half::f16::from_le_bytes(scale_bytes);
        println!("  Scale f16: {:?}", scale_f16);
        println!("  Scale f32: {:.6}", scale_f16.to_f32());

        // Show first few nibbles
        for (i, &inp) in input.iter().enumerate().take(8) {
            let byte_idx = block_offset + 2 + i / 2;
            let byte = q_weight.data[byte_idx];
            let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let dequant = (nibble as i8 - 8) as f32 * scale_f16.to_f32();
            println!(
                "  nibble[{}]: {} -> dequant: {:.4}, input: {:.4}, product: {:.4}",
                i,
                nibble,
                dequant,
                inp,
                dequant * inp
            );
        }
    } else {
        println!("✓ Q4_0 matmul is correct!");
    }

    Ok(())
}
