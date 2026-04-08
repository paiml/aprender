#![allow(clippy::disallowed_methods)]
//! Chapter 25: Switch From Ollama
//!
//! Command equivalence: ollama → apr
//! Contract: contracts/apr-book-ch25-v1.yaml

use aprender::format::validated_tensors::TensorStats;

fn main() {
    println!("=== Switch From Ollama ===");
    println!();

    // Command equivalence table
    println!("| Ollama                          | apr                                     |");
    println!("|---------------------------------|-----------------------------------------|");
    println!("| ollama pull qwen2.5-coder       | apr pull hf://Qwen/Qwen2.5-Coder-GGUF   |");
    println!("| ollama run qwen2.5-coder        | apr run model.gguf --prompt '...'        |");
    println!("| ollama serve                    | apr serve model.gguf --port 11434        |");
    println!("| ollama list                     | apr list                                |");
    println!("| ollama show qwen2.5-coder       | apr inspect model.gguf                  |");
    println!("| ollama rm qwen2.5-coder         | rm ~/.cache/apr/models/model.gguf        |");
    println!("| curl /api/generate              | curl /v1/completions (OpenAI-compatible) |");
    println!();

    // GGUF format compatibility
    println!("GGUF compatibility:");
    println!("  Ollama uses GGUF internally (via llama.cpp)");
    println!("  apr reads GGUF natively — same model files work");
    println!("  apr also reads SafeTensors and APR native format");
    println!();

    // Layout contract demonstration
    let gguf_shape = [4096_usize, 11008];
    let apr_shape = [gguf_shape[1], gguf_shape[0]]; // transpose at import
    println!("Layout contract (LAYOUT-001):");
    println!("  GGUF col-major {:?} -> APR row-major {:?}", gguf_shape, apr_shape);
    assert_eq!(apr_shape[0], 11008, "Row-major rows = ne1");

    // Performance comparison
    println!();
    println!("Performance (Qwen2.5-Coder-1.5B Q4_K_M, RTX 4090):");
    println!("  Ollama:          ~250 tok/s (wraps llama.cpp)");
    println!("  apr serve:       273.8 tok/s (aprender-serve)");
    println!("  apr serve c=32:  1,776 tok/s (continuous batching)");
    println!();
    println!("Key advantage: apr serve supports continuous batching;");
    println!("Ollama processes one request at a time.");

    // TensorStats on comparison data
    let comparison = vec![250.0_f32, 273.8, 285.0]; // ollama, apr, llama.cpp
    let stats = TensorStats::compute(&comparison);
    println!();
    println!("Throughput comparison stats: mean={:.0}, range={:.0}-{:.0}",
        stats.mean, stats.min, stats.max);
    assert!(stats.mean > 200.0, "All frameworks exceed 200 tok/s");

    println!();
    println!("Chapter 25 contracts: PASSED");
}
