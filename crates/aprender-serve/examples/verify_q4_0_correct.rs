//! Verify Q4_0 matmul with correct layout
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Q4_0 Matmul Verification (Correct Layout) ===\n");

    let hidden_dim = model.config().hidden_dim;
    let layer0 = &model.layers()[0];

    // Get Q weight tensor
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &layer0.qkv_weight else {
        panic!("Expected separate QKV")
    };

    // Create a test input (normalized embedding for token 17)
    let token_17_emb = &model.token_embedding()[17 * hidden_dim..(17 + 1) * hidden_dim];
    let eps = model.config().eps;
    let ss: f32 = token_17_emb.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    let input: Vec<f32> = token_17_emb
        .iter()
        .zip(layer0.attn_norm_weight.iter())
        .map(|(x, w)| x * scale * w)
        .collect();

    // Get model's Q output
    let qkv_output = model.qkv_matmul(&input, &layer0.qkv_weight)?;
    let q_dim = q_weight.out_dim;
    let q_output = &qkv_output[0..q_dim];

    // Manually compute using CORRECT Q4_0 layout
    // Q4_0 format per block (18 bytes):
    // - 2 bytes: f16 scale
    // - 16 bytes: 32 4-bit values
    //   - low nibble of byte[j] → position j (0..16)
    //   - high nibble of byte[j] → position j+16 (16..32)

    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 18;

    let num_blocks = q_weight.in_dim.div_ceil(BLOCK_SIZE);
    let mut manual_output = vec![0.0f32; q_weight.out_dim];

    #[allow(clippy::needless_range_loop)] // row used in complex offset calculations
    for row in 0..q_weight.out_dim.min(10) {
        let mut total_sum = 0.0f32;

        for block_idx in 0..num_blocks {
            let block_offset = (row * num_blocks + block_idx) * BYTES_PER_BLOCK;

            // Read f16 scale
            let scale_bytes = [q_weight.data[block_offset], q_weight.data[block_offset + 1]];
            let scale_f32 = half::f16::from_le_bytes(scale_bytes).to_f32();

            let act_start = block_idx * BLOCK_SIZE;
            let act_end = (act_start + BLOCK_SIZE).min(q_weight.in_dim);

            let mut block_sum = 0.0f32;

            // Each byte contains 2 nibbles
            for j in 0..16 {
                let byte = q_weight.data[block_offset + 2 + j];

                // Low nibble → position j
                let low_idx = act_start + j;
                let low_quant = (byte & 0x0F) as i8 - 8;
                if low_idx < act_end {
                    block_sum += (low_quant as f32) * input[low_idx];
                }

                // High nibble → position j + 16
                let high_idx = act_start + j + 16;
                let high_quant = (byte >> 4) as i8 - 8;
                if high_idx < act_end {
                    block_sum += (high_quant as f32) * input[high_idx];
                }
            }

            total_sum += scale_f32 * block_sum;
        }
        manual_output[row] = total_sum;
    }

    println!("Manual Q output (first 10): {:?}", &manual_output[..10]);
    println!("Model Q output (first 10):  {:?}", &q_output[..10]);

    // Compare
    let max_diff = manual_output
        .iter()
        .take(10)
        .zip(q_output.iter().take(10))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\nMax difference (first 10): {:.6}", max_diff);

    if max_diff < 0.001 {
        println!("✓ Q4_0 matmul is correct!");
    } else {
        println!("⚠️ Still different - bug elsewhere?");
    }

    Ok(())
}
