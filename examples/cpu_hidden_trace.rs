//! CORRECTNESS-011: Trace CPU hidden state for comparison with GPU
//!
//! Prints CPU hidden state sum/rms before output_norm to compare with:
//! [CORRECTNESS-001] GPU shows: sum = 466.2486, rms = 39.4793
//!
//! Run with: cargo run --example cpu_hidden_trace --release

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    println!("CORRECTNESS-011: CPU Hidden State Trace");
    println!("=======================================");
    println!("Model: {}", path);

    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let token_id = 791u32;
    println!("\nToken ID: {}", token_id);
    println!("Hidden dim: {}", model.config().hidden_dim);
    println!("Num layers: {}", model.config().num_layers);

    // Get embedding
    let embedding = model.embed(&[token_id]);
    let embed_sum: f32 = embedding.iter().sum();
    let embed_rms: f32 =
        (embedding.iter().map(|x| x * x).sum::<f32>() / embedding.len() as f32).sqrt();
    println!(
        "\nEmbedding: first 5 = {:?}",
        &embedding[..5.min(embedding.len())]
    );
    println!("Embedding: sum={:.6}, rms={:.6}", embed_sum, embed_rms);

    // We need to run forward and capture hidden state before output_norm
    // Since forward() is not easily instrumentable, let's manually trace

    // Run forward to get logits for comparison
    let logits = model.forward(&[token_id])?;
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));

    println!("\n=== CPU Forward Results ===");
    println!("Argmax: {:?}", argmax);

    // Print logits at key positions
    println!("\nLogits at key positions:");
    for pos in [0, 13, 14, 15, 16, 17, 18, 19, 74403usize] {
        if pos < logits.len() {
            println!("  logits[{}] = {:.6}", pos, logits[pos]);
        }
    }

    // Manual forward trace using internal structures
    // This requires accessing model internals which we can't easily do
    // Instead, let's compute expected values

    println!("\n=== GPU Comparison Values (from debug output) ===");
    println!("GPU hidden before output_norm: sum=466.2486, rms=39.4793");
    println!("GPU normed hidden: sum=107.5945, rms=4.6616");
    println!("GPU logits[0..20]: [0.39, -1.77, -2.05, -2.57, -1.96, -1.06, 0.03, 1.09, 0.57, -2.89, 1.11, 0.82, 0.61, 0.01, 2.30, -5.00, 0.87, -0.49, -0.80, -1.76]");

    println!("\n=== CPU logits[0..20] ===");
    println!("{:?}", &logits[..20.min(logits.len())]);

    // Compute correlation with GPU logits (from debug output)
    let gpu_first_20: [f32; 20] = [
        0.39, -1.77, -2.05, -2.57, -1.96, -1.06, 0.03, 1.09, 0.57, -2.89, 1.11, 0.82, 0.61, 0.01,
        2.30, -5.00, 0.87, -0.49, -0.80, -1.76,
    ];

    println!("\n=== First 20 position comparison ===");
    println!("{:<5} {:>10} {:>10} {:>10}", "pos", "CPU", "GPU", "diff");
    for i in 0..20 {
        let cpu_val = logits[i];
        let gpu_val = gpu_first_20[i];
        let diff = cpu_val - gpu_val;
        println!(
            "{:<5} {:>10.4} {:>10.4} {:>10.4}",
            i, cpu_val, gpu_val, diff
        );
    }

    // Key positions analysis
    println!("\n=== Key Position Analysis ===");
    println!(
        "pos=16 (CPU argmax): CPU={:.4}, GPU=0.87, diff={:.4}",
        logits[16],
        logits[16] - 0.87
    );
    println!(
        "pos=13: CPU={:.4}, GPU=0.01, diff={:.4}",
        logits[13],
        logits[13] - 0.01
    );
    println!(
        "pos=15: CPU={:.4}, GPU=-5.00, diff={:.4}",
        logits[15],
        logits[15] - (-5.00)
    );
    if logits.len() > 74403 {
        println!(
            "pos=74403 (GPU argmax): CPU={:.4}, GPU=10.53, diff={:.4}",
            logits[74403],
            logits[74403] - 10.53
        );
    }

    println!("\n=== Diagnosis ===");
    println!("If CPU values >> GPU values at positions 13-19, the GPU has a systematic error");
    println!("in the first rows of the LM head projection or in hidden state computation.");

    Ok(())
}
