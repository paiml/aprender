//! Check Q4_K scale values at row 5475
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

const Q4_K_BLOCK_SIZE: usize = 144;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("=== Q4_K scale values at row 5475 (first superblock) ===\n");

    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];
        if let Some(ref gw) = layer.ffn_gate_weight {
            let bytes_per_row = (gw.in_dim / 256) * Q4_K_BLOCK_SIZE;
            let row_start = 5475 * bytes_per_row;
            let block = &gw.data[row_start..row_start + Q4_K_BLOCK_SIZE];

            // Q4_K block structure: d(2) + dmin(2) + scales(12) + qs(128)
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

            println!(
                "Layer {} gate row 5475: d={:.6}, dmin={:.6}",
                layer_idx, d, dmin
            );
        }
    }

    println!("\nFor comparison, row 100:");
    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];
        if let Some(ref gw) = layer.ffn_gate_weight {
            let bytes_per_row = (gw.in_dim / 256) * Q4_K_BLOCK_SIZE;
            let row_start = 100 * bytes_per_row;
            let block = &gw.data[row_start..row_start + Q4_K_BLOCK_SIZE];

            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

            println!(
                "Layer {} gate row 100: d={:.6}, dmin={:.6}",
                layer_idx, d, dmin
            );
        }
    }
}

use half::f16;
