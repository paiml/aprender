//! Check FFN down weight

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q6_k;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let down_weight = &model.layers()[2].ffn_down_weight;

    println!("FFN down weight (layer 2):");
    println!("  in_dim: {}", down_weight.in_dim);
    println!("  out_dim: {}", down_weight.out_dim);
    println!("  qtype: {} (Q6_K=14)", down_weight.qtype);
    println!("  data len: {} bytes", down_weight.data.len());

    // Dequantize first row
    // For Q6_K row-major [out_dim, in_dim] = [2048, 5632]
    // Each row has ceil(5632/256) = 22 superblocks = 22 * 210 = 4620 bytes
    let superblocks_per_row = down_weight.in_dim.div_ceil(256);
    let bytes_per_row = superblocks_per_row * 210;
    println!("  superblocks_per_row: {}", superblocks_per_row);
    println!("  bytes_per_row: {}", bytes_per_row);

    // Dequantize first superblock of first row
    let first_sb = dequantize_q6_k(&down_weight.data[..210]).expect("test");
    println!("\nFirst superblock (256 values):");
    println!("  L2: {:.4}", l2_norm(&first_sb));
    println!("  first 10: {:?}", &first_sb[..10]);
}
