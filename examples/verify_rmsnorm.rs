//! Verify RMSNorm implementation
fn rms_norm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    // Reference implementation from LLaMA paper
    let n = input.len();
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| (x / rms) * w)
        .collect()
}

fn main() {
    // Test case from a known working implementation
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = rms_norm_reference(&input, &weight, eps);

    // Manual calculation:
    // sum_sq = 1 + 4 + 9 + 16 = 30
    // rms = sqrt(30/4 + 1e-5) = sqrt(7.50001) = 2.7386...
    // output = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386] = [0.365, 0.730, 1.095, 1.461]

    println!("Input: {:?}", input);
    println!("RMSNorm output: {:?}", output);
    println!("Expected: [0.365, 0.730, 1.095, 1.461] (approx)");

    // Also test L2 norm of output
    let l2: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Output L2: {:.4}", l2);
    // For unit weights, RMSNorm should give L2 = sqrt(n)
    println!("Expected L2 (sqrt(n)): {:.4}", (4.0f32).sqrt());

    // Now test with actual model values
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = 2048;
    let token_id = 450u32;
    let eps = model.config().eps;

    let start = token_id as usize * hidden_dim;
    let embedding = &model.token_embedding()[start..start + hidden_dim];
    let layer = &model.layers()[0];

    let normed = rms_norm_reference(embedding, &layer.attn_norm_weight, eps);

    println!("\n=== Layer 0 RMSNorm ===");
    println!(
        "Embedding L2: {:.4}",
        embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!("Embedding first 5: {:?}", &embedding[..5]);
    println!("Norm weight first 5: {:?}", &layer.attn_norm_weight[..5]);
    println!(
        "Normed L2: {:.4}",
        normed.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!("Normed first 5: {:?}", &normed[..5]);

    // Check norm weight statistics
    let norm_l2: f32 = layer
        .attn_norm_weight
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    let norm_mean: f32 =
        layer.attn_norm_weight.iter().sum::<f32>() / layer.attn_norm_weight.len() as f32;
    println!("\nNorm weight L2: {:.4}, mean: {:.4}", norm_l2, norm_mean);
}
