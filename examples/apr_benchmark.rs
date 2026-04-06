//! Quick APR throughput benchmark
use realizar::apr_transformer::{AprBenchmarkRunner, AprTransformer, AprTransformerConfig};

fn main() {
    println!("=== APR Transformer Benchmark ===\n");

    // Small model (fast)
    let config_small = AprTransformerConfig {
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 1024,
        context_length: 512,
        ..Default::default()
    };

    let transformer = AprTransformer::new(config_small);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(3);
    runner.set_measure_iterations(10);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let result = runner.benchmark_decode(&prompt, 32).expect("test");

    println!("Small Model (256 hidden, 4 layers):");
    println!("  Tokens: {}", result.tokens_generated);
    println!("  Time: {:.2}ms", result.total_time_ms);
    println!("  Throughput: {:.1} tok/s", result.tokens_per_second);
    println!("  Target: >= 50 tok/s (CPU spec)");
    println!();

    if result.tokens_per_second >= 50.0 {
        println!("✓ PASS: {:.1} tok/s >= 50 tok/s", result.tokens_per_second);
    } else {
        println!("✗ FAIL: {:.1} tok/s < 50 tok/s", result.tokens_per_second);
    }
}
