//! Verify embedding matches between CPU and GPU paths
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");

    let token_id = 791u32;
    let hidden_dim = model.config().hidden_dim;

    // Method 1: Direct token_embedding access
    let start = token_id as usize * hidden_dim;
    let direct_embed: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    // Method 2: Using embed() function
    let embed_result = model.embed(&[token_id]);

    println!("Token {} embedding verification:", token_id);
    println!("  hidden_dim: {}", hidden_dim);
    println!("  Direct access first 5: {:?}", &direct_embed[..5]);
    println!("  embed() first 5: {:?}", &embed_result[..5]);
    println!("  Direct sum: {:.6}", direct_embed.iter().sum::<f32>());
    println!("  embed() sum: {:.6}", embed_result.iter().sum::<f32>());

    // Check if they match
    let match_count = direct_embed
        .iter()
        .zip(embed_result.iter())
        .filter(|(a, b)| (*a - *b).abs() < 1e-6)
        .count();
    println!("  Match count: {}/{}", match_count, hidden_dim);

    // Now verify against CPU forward path
    let cpu_logits = model.forward(&[token_id]).expect("forward");
    let argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("\nCPU forward result:");
    println!("  argmax: {} (logit: {:.4})", argmax.0, argmax.1);
}
