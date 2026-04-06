//! CORRECTNESS-011: Trace CPU hidden state BEFORE output_norm
//!
//! Compare with GPU: sum=466.2486, rms=39.4793
//!
//! Run with: cargo run --example cpu_hidden_state_trace --release

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    println!("CORRECTNESS-011: CPU Hidden State Before Output Norm");
    println!("=====================================================");
    println!("Model: {}", path);

    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let token_id = 791u32;
    println!("\nToken ID: {}", token_id);
    println!("Hidden dim: {}", model.config().hidden_dim);

    // Use the model's forward_with_hidden_state if available,
    // or manually trace through layers

    // Get embedding
    let hidden = model.embed(&[token_id]);
    let embed_sum: f32 = hidden.iter().sum();
    let embed_rms: f32 = (hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32).sqrt();
    println!("\n=== Initial Embedding ===");
    println!("first 5: {:?}", &hidden[..5.min(hidden.len())]);
    println!("sum={:.4}, rms={:.4}", embed_sum, embed_rms);

    // Since we can't easily instrument the model's forward() function,
    // let's use forward_with_kv_cache which might give us intermediate access
    // Or we can use generate_with_cache to see the internal state

    // For now, let's run forward and check the final logits
    let logits = model.forward(&[token_id])?;

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));

    println!("\n=== Final Logits ===");
    println!("Argmax: {:?}", argmax);
    println!("first 5: {:?}", &logits[..5.min(logits.len())]);

    // Compare with GPU values
    println!("\n=== GPU Hidden State (from debug output) ===");
    println!("Hidden before output_norm:");
    println!("  first 5: [1.2728, 7.7476, -18.4799, 22.1341, -23.2289]");
    println!("  sum=466.2486, rms=39.4793");
    println!("Normed hidden:");
    println!("  first 5: [0.1421, 0.9015, -1.5506, 2.5930, -2.6661]");
    println!("  sum=107.5945, rms=4.6616");

    // To get CPU hidden state, we need to modify the model or create a custom forward
    // For now, let's estimate based on the output_norm transformation

    // GPU normed_hidden = hidden * scale where scale = 1/sqrt(mean(hidden^2) + eps)
    // GPU shows rms = 39.48 before norm, rms = 4.66 after norm
    // scale = 4.66 / 39.48 ≈ 0.118
    // This corresponds to rms_inv = 1/sqrt(mean_sq + eps) where mean_sq = sum_sq/n
    // rms = sqrt(mean_sq) = 39.48
    // mean_sq = 39.48^2 = 1559
    // For n=1536: sum_sq = mean_sq * n = 1559 * 1536 = 2,394,624

    println!("\n=== Analysis ===");
    println!(
        "GPU hidden RMS = 39.48 suggests mean_sq = {:.2}",
        39.48_f32.powi(2)
    );
    println!(
        "For hidden_dim=1536, sum_sq ≈ {:.2}",
        39.48_f32.powi(2) * 1536.0
    );

    // Check if rms_norm is applied correctly
    // rms_inv = rsqrt(mean_sq + eps) = rsqrt(1559 + 1e-5) ≈ 0.0253
    let rms_inv = 1.0 / (39.48_f32.powi(2) + 1e-5).sqrt();
    println!(
        "Expected rms_inv = 1/sqrt({:.2} + 1e-5) = {:.6}",
        39.48_f32.powi(2),
        rms_inv
    );

    // The normed values should be: normed = hidden * rms_inv * weight
    // If we assume weight ≈ 1 (on average), then:
    // normed_rms ≈ hidden_rms * rms_inv = 39.48 * 0.0253 = 1.0
    // But GPU shows normed_rms = 4.66, which is ~4.66x larger
    // This suggests the output_norm weights have average value ≈ 4.66

    println!(
        "\nGPU normed_rms = 4.66, expected if weight=1: {:.4}",
        39.48 * rms_inv
    );
    println!(
        "This implies output_norm weights have mean ≈ {:.2}",
        4.66 / (39.48 * rms_inv)
    );

    // The key question is whether CPU hidden state matches GPU hidden state
    // If they match, the bug is in the GPU output_norm or LM head
    // If they differ, the bug is in the transformer layers

    println!("\n=== CONCLUSION ===");
    println!("To determine root cause, need to compare:");
    println!("1. CPU hidden state sum/rms BEFORE output_norm");
    println!("2. GPU hidden state sum/rms = 466.25/39.48");
    println!("\nIf they match → bug is in GPU output_norm or LM head");
    println!("If they differ → bug is in GPU transformer layers (per spec: RoPE/Cache)");

    Ok(())
}
