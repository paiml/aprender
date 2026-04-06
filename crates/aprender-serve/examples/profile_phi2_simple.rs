//! Simple Phi-2 profiling - measure full forward pass breakdown
use realizar::apr_transformer::QuantizedAprTransformerQ4;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::time::Instant;

fn main() {
    let model_path = "/mnt/ssd/models/phi-2.Q4_0.gguf";

    println!("=== Phi-2 Bottleneck Analysis ===\n");

    // Load model
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path).expect("Failed to load");
    let load_time = start.elapsed();

    let start = Instant::now();
    let gguf = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");
    let parse_time = start.elapsed();

    let start = Instant::now();
    let apr = QuantizedAprTransformerQ4::from_gguf(&gguf);
    let convert_time = start.elapsed();

    println!("Model loading:");
    println!("  mmap:    {:>6}ms", load_time.as_millis());
    println!("  parse:   {:>6}ms", parse_time.as_millis());
    println!("  convert: {:>6}ms", convert_time.as_millis());

    let config = gguf.config();
    println!("\nModel config:");
    println!("  hidden_dim:      {}", config.hidden_dim);
    println!("  num_layers:      {}", config.num_layers);
    println!("  num_heads:       {}", config.num_heads);
    println!("  num_kv_heads:    {}", config.num_kv_heads);
    println!("  intermediate:    {}", config.intermediate_dim);
    println!("  vocab_size:      {}", config.vocab_size);

    // Warmup
    println!("\nWarming up...");
    for _ in 0..3 {
        let _ = apr.forward(&[1]);
    }

    // Profile forward passes
    println!("\nProfiling forward passes:");
    let mut times = Vec::new();
    for i in 0..20 {
        let start = Instant::now();
        let _ = apr.forward(&[1]);
        let elapsed = start.elapsed();
        times.push(elapsed.as_micros());

        if i < 5 {
            println!(
                "  Pass {}: {:>6}µs ({:.1}ms)",
                i + 1,
                elapsed.as_micros(),
                elapsed.as_micros() as f64 / 1000.0
            );
        }
    }

    times.sort_unstable();
    let min = times[0];
    let max = times[times.len() - 1];
    let median = times[times.len() / 2];
    let avg = times.iter().sum::<u128>() / times.len() as u128;
    let p95 = times[(times.len() as f64 * 0.95) as usize];

    println!("\nLatency Statistics (µs):");
    println!("  min:    {:>8}", min);
    println!("  median: {:>8}", median);
    println!("  avg:    {:>8}", avg);
    println!("  p95:    {:>8}", p95);
    println!("  max:    {:>8}", max);

    println!(
        "\nThroughput: {:.2} tok/s (median)",
        1_000_000.0 / median as f64
    );

    // Calculate theoretical limits
    let hidden = config.hidden_dim as f64;
    let layers = config.num_layers as f64;
    let vocab = config.vocab_size as f64;
    let inter = config.intermediate_dim as f64;

    // Weight sizes (Q4_0 = 0.5 bytes per weight)
    let bytes_per_weight = 0.5;
    let qkv_size = 3.0 * hidden * hidden * bytes_per_weight;
    let out_size = hidden * hidden * bytes_per_weight;
    let ffn_size = 3.0 * hidden * inter * bytes_per_weight; // up, gate, down
    let layer_size = qkv_size + out_size + ffn_size;
    let total_layer_size = layer_size * layers;
    let lm_head_size = hidden * vocab * bytes_per_weight;
    let total_size = total_layer_size + lm_head_size;

    println!("\n=== Memory Bandwidth Analysis ===");
    println!("Per-layer weights: {:.1} MB", layer_size / 1e6);
    println!("  - QKV:  {:.1} MB", qkv_size / 1e6);
    println!("  - Out:  {:.1} MB", out_size / 1e6);
    println!("  - FFN:  {:.1} MB", ffn_size / 1e6);
    println!(
        "All layers ({:.0}): {:.1} MB",
        layers,
        total_layer_size / 1e6
    );
    println!(
        "LM head: {:.1} MB ({:.1}% of total)",
        lm_head_size / 1e6,
        100.0 * lm_head_size / total_size
    );
    println!("Total: {:.1} MB", total_size / 1e6);

    let bandwidth_gbps = 30.0;
    let theoretical_min_ms = (total_size / 1e9) / bandwidth_gbps * 1000.0;
    let actual_ms = median as f64 / 1000.0;
    let efficiency = theoretical_min_ms / actual_ms * 100.0;

    println!("\nRoofline Analysis (@ {:.0} GB/s):", bandwidth_gbps);
    println!("  Theoretical minimum: {:.1}ms", theoretical_min_ms);
    println!("  Actual (median):     {:.1}ms", actual_ms);
    println!("  Efficiency:          {:.1}%", efficiency);

    // Estimate operation breakdown
    println!("\n=== Estimated Time Breakdown ===");
    let qkv_time = (qkv_size / 1e9) / bandwidth_gbps * 1000.0 * layers;
    let out_time = (out_size / 1e9) / bandwidth_gbps * 1000.0 * layers;
    let ffn_time = (ffn_size / 1e9) / bandwidth_gbps * 1000.0 * layers;
    let lm_time = (lm_head_size / 1e9) / bandwidth_gbps * 1000.0;

    println!("If memory-bound @ {:.0} GB/s:", bandwidth_gbps);
    println!(
        "  QKV projections:  {:.1}ms ({:.1}%)",
        qkv_time,
        100.0 * qkv_time / theoretical_min_ms
    );
    println!(
        "  Output proj:      {:.1}ms ({:.1}%)",
        out_time,
        100.0 * out_time / theoretical_min_ms
    );
    println!(
        "  FFN layers:       {:.1}ms ({:.1}%)",
        ffn_time,
        100.0 * ffn_time / theoretical_min_ms
    );
    println!(
        "  LM head:          {:.1}ms ({:.1}%)",
        lm_time,
        100.0 * lm_time / theoretical_min_ms
    );

    let overhead_ms = actual_ms - theoretical_min_ms;
    println!("\nOverhead analysis:");
    println!("  Theoretical:  {:.1}ms", theoretical_min_ms);
    println!("  Actual:       {:.1}ms", actual_ms);
    println!(
        "  Overhead:     {:.1}ms ({:.1}%)",
        overhead_ms,
        100.0 * overhead_ms / actual_ms
    );

    if efficiency < 50.0 {
        println!("\nBottleneck likely:");
        println!("  - Thread synchronization in Rayon parallel matmul");
        println!("  - Cache thrashing on 1.6GB model weights");
        println!("  - Memory prefetch not keeping up with compute");
    }
}
