//! Check FFN down weight column 5475 - does it compensate for large gate output?
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q6_k;

const Q6_K_BLOCK_SIZE: usize = 210;

#[allow(dead_code)]
fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("FFN down weight analysis (Q6_K, row-major [hidden_dim, intermediate_dim]):");
    println!("  in_dim=5632 (intermediate), out_dim=2048 (hidden)");

    // FFN down: [intermediate_dim] -> [hidden_dim]
    // Weight shape: [out_dim, in_dim] = [2048, 5632] row-major
    // Column 5475 spans all 2048 rows

    // For Q6_K row-major: each row has 5632/256 = 22 superblocks
    // Column 5475 is in superblock 5475/256 = 21 (position 99 in that superblock)

    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];
        let down = &layer.ffn_down_weight;

        // Check column 5475 by sampling a few rows
        let bytes_per_row = (down.in_dim / 256) * Q6_K_BLOCK_SIZE;
        let superblock_idx = 5475 / 256; // = 21
        let pos_in_block = 5475 % 256; // = 99

        // Get a few values from column 5475
        let mut col_vals = Vec::new();
        for row in [0, 100, 500, 1000, 1500, 2000] {
            if row >= down.out_dim {
                break;
            }
            let row_start = row * bytes_per_row;
            let block_start = row_start + superblock_idx * Q6_K_BLOCK_SIZE;
            let block_data = &down.data[block_start..block_start + Q6_K_BLOCK_SIZE];
            let dequant = dequantize_q6_k(block_data).expect("test");
            col_vals.push(dequant[pos_in_block]);
        }

        // Also check a typical column (e.g., 100)
        let mut col100_vals = Vec::new();
        for row in [0, 100, 500, 1000, 1500, 2000] {
            if row >= down.out_dim {
                break;
            }
            let row_start = row * bytes_per_row;
            let block_start = row_start; // superblock 0
            let block_data = &down.data[block_start..block_start + Q6_K_BLOCK_SIZE];
            let dequant = dequantize_q6_k(block_data).expect("test");
            col100_vals.push(dequant[100]);
        }

        println!("  Layer {}: col_5475={:?}", layer_idx, col_vals);
        println!("           col_100 ={:?}", col100_vals);
    }
}
