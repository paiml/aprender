//! Benchmark forward pass timing
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::time::Instant;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("GGUF_MODEL").ok())
        .unwrap_or_else(|| {
            eprintln!("Usage: bench_forward <model.gguf>");
            eprintln!("   or: GGUF_MODEL=path cargo run --example bench_forward");
            std::process::exit(1);
        });

    println!("Loading model: {}", model_path);
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(&model_path).expect("test");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    let config = model.config();
    println!(
        "Config: hidden_dim={}, vocab_size={}, num_layers={}",
        config.hidden_dim, config.vocab_size, config.num_layers
    );

    // Warmup
    println!("\nWarming up (3 iterations)...");
    for _ in 0..3 {
        let _ = model.forward(&[1]).expect("test");
    }

    // Benchmark single-token forward passes
    println!("\nBenchmarking single-token forward (10 iterations)...");
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward(&[1]).expect("test");
    }
    let total_time = start.elapsed();
    let avg_time = total_time / iterations;
    let tok_per_sec = 1.0 / avg_time.as_secs_f64();

    println!(
        "Total time for {} iterations: {:.3}s",
        iterations,
        total_time.as_secs_f32()
    );
    println!(
        "Average forward pass: {:.1}ms",
        avg_time.as_secs_f64() * 1000.0
    );
    println!("Throughput: {:.1} tok/s", tok_per_sec);

    // Test with single token (for validation)
    let logits = model.forward(&[1]).expect("test");
    println!(
        "\nLogits len: {}, first 5: {:?}",
        logits.len(),
        &logits[..5]
    );

    // Find top prediction
    let (best_idx, best_logit) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("test");

    let vocab = mapped.model.vocabulary().expect("test");
    let tok = if best_idx < vocab.len() {
        &vocab[best_idx]
    } else {
        "?"
    };
    println!("Top prediction: {} '{}' = {:.4}", best_idx, tok, best_logit);
}
