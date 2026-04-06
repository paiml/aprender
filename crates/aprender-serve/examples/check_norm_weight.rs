//! Check norm weights

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("Output norm weight:");
    println!("  L2: {:.6}", l2_norm(model.output_norm_weight()));
    println!("  First 5: {:?}", &model.output_norm_weight()[..5]);
    println!("  Length: {}", model.output_norm_weight().len());

    println!("\nLayer 0 attn norm weight:");
    println!("  L2: {:.6}", l2_norm(&model.layers()[0].attn_norm_weight));
    println!("  First 5: {:?}", &model.layers()[0].attn_norm_weight[..5]);
}
