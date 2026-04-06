//! KAIZEN-047: Profile CUDA transformer trainer step breakdown.
//!
//! Runs 20 training steps on a tiny model (2 layers, 128 hidden, seq_len=32)
//! with the StepProfiler enabled, then prints the timing breakdown.
//!
//! Run with:
//!   cargo run --example profile_cuda_trainer --release --features cuda
//!
//! Output: per-phase wall-clock timing table showing where time is spent.

#[cfg(feature = "cuda")]
fn main() {
    use entrenar::train::{CudaTransformerTrainer, LMBatch, TransformerTrainConfig};
    use entrenar::transformer::TransformerConfig;

    println!("=== KAIZEN-047: CUDA Trainer Step Profiler ===\n");

    // Parse args: --large for production-scale model
    let args: Vec<String> = std::env::args().collect();
    let large = args.iter().any(|a| a == "--large");

    let (model_config, max_seq, num_steps, vocab) = if large {
        // 350M-class: 24 layers, 1024 hidden, 128 seq_len
        println!("Mode: LARGE (24L, 1024H, seq=128, vocab=32000, 20 steps)");
        (
            TransformerConfig {
                hidden_size: 1024,
                intermediate_size: 4096,
                num_hidden_layers: 24,
                num_attention_heads: 16,
                num_kv_heads: 16,
                vocab_size: 32000,
                max_position_embeddings: 512,
                ..TransformerConfig::tiny()
            },
            128_usize,
            20_usize,
            32000_usize,
        )
    } else {
        // Tiny model for quick smoke test
        println!("Mode: TINY (2L, 128H, seq=32, vocab=1000, 20 steps)");
        println!("  Use --large for production-scale profiling");
        (
            TransformerConfig {
                hidden_size: 128,
                intermediate_size: 512,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_kv_heads: 4,
                vocab_size: 1000,
                max_position_embeddings: 64,
                ..TransformerConfig::tiny()
            },
            32_usize,
            20_usize,
            1000_usize,
        )
    };

    let mut train_config = TransformerTrainConfig::new(model_config);
    train_config.max_seq_len = max_seq;
    train_config.lr = 1e-4;
    train_config.profile_interval = 0; // We'll print manually

    let mut trainer = match CudaTransformerTrainer::new(train_config) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("CUDA init failed: {e}");
            std::process::exit(1);
        }
    };

    trainer.enable_profiler(0); // Enable, no auto-report

    println!("\nRunning {num_steps} training steps...\n");

    // Create synthetic batches
    let sequences: Vec<Vec<u32>> = (0..num_steps)
        .map(|i| (0..=max_seq).map(|j| ((i * (max_seq + 1) + j) % vocab) as u32).collect())
        .collect();

    for seq in &sequences {
        let batch = LMBatch::from_sequences(&[seq.clone()], (vocab - 1) as u32, max_seq as u32);
        let loss = trainer.train_batch(&batch);
        print!(".");
        let _ = loss;
    }
    println!("\n");

    trainer.print_profiler_report();

    println!("\n=== Profile complete. Use this data to justify optimizations. ===");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires --features cuda");
    std::process::exit(1);
}
