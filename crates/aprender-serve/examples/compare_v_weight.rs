//! Compare V weight values with HuggingFace

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::dequantize_q6_k;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[0];
    let OwnedQKVWeights::Separate {
        q: _,
        k: _,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate")
    };

    println!("V weight info:");
    println!("  in_dim: {}", v_weight.in_dim);
    println!("  out_dim: {}", v_weight.out_dim);
    println!("  qtype: {} (Q6_K=14)", v_weight.qtype);
    println!("  data len: {} bytes", v_weight.data.len());

    // Q6_K: 210 bytes per 256 elements
    println!(
        "  Expected bytes for [256, 2048]: {}",
        (256 * 2048 / 256) * 210
    );

    // Dequantize first 256 values (first row or column?)
    let first_block = dequantize_q6_k(&v_weight.data[..210]).expect("test");
    println!("\nFirst 256 dequantized values:");
    println!("  First 5: {:?}", &first_block[..5]);
    println!("  L2: {:.6}", l2_norm(&first_block));

    // HuggingFace first row first 5: [0.028076171875, 0.00592041015625, -0.000347137451171875, -0.005615234375, 0.00750732421875]
    // HuggingFace first col first 5: [0.028076171875, 0.017578125, 0.035888671875, 0.0164794921875, -0.022216796875]

    println!("\nHuggingFace reference:");
    println!("  First row first 5: [0.0281, 0.0059, -0.0003, -0.0056, 0.0075]");
    println!("  First col first 5: [0.0281, 0.0176, 0.0359, 0.0165, -0.0222]");

    // Dequantize entire weight
    let total_elements = v_weight.in_dim * v_weight.out_dim;
    let num_blocks = total_elements / 256;
    let mut full_weight = Vec::new();
    for i in 0..num_blocks {
        let block_data = &v_weight.data[i * 210..(i + 1) * 210];
        let dequant = dequantize_q6_k(block_data).expect("test");
        full_weight.extend(dequant);
    }
    println!("\nFull weight L2: {:.6}", l2_norm(&full_weight));
    println!("HuggingFace V weight L2: 7.976477");

    // Storage layout check
    // If stored row-major [256, 2048]: first 2048 values are first row
    // If stored col-major [2048, 256]: first 256 values are first column
    println!("\nCheck if column-major storage (first 256 = first column of [2048, 256]):");
    println!("  full_weight[0]: {:.6}", full_weight[0]);
    println!("  full_weight[256]: {:.6}", full_weight[256]); // Second column if col-major
    println!("  full_weight[1]: {:.6}", full_weight[1]); // Second element of first column
}
