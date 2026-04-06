//! PMAT-235: Validated Tensors - Compile-Time Contract Enforcement
//!
//! Demonstrates the Poka-Yoke (mistake-proofing) pattern for tensor validation
//! in the realizar inference engine.
//!
//! This mirrors the aprender implementation to ensure cross-crate parity.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example validated_tensors
//! ```

use realizar::safetensors::{ContractValidationError, ValidatedEmbedding, ValidatedWeight};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PMAT-235: Validated Tensors (realizar) - Compile-Time Enforcement");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Demo 1: Valid embedding
    demo_valid_embedding();

    // Demo 2: Density rejection (PMAT-234 bug)
    demo_density_rejection();

    // Demo 3: ValidatedWeight
    demo_validated_weight();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Cross-crate parity: realizar validation matches aprender");
    println!("═══════════════════════════════════════════════════════════════════");
}

fn demo_valid_embedding() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 1: Valid Embedding                                         │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let vocab_size = 100;
    let hidden_dim = 64;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    match ValidatedEmbedding::new(data, vocab_size, hidden_dim) {
        Ok(embedding) => {
            println!("  ✅ ValidatedEmbedding created!");
            println!("     vocab_size: {}", embedding.vocab_size());
            println!("     hidden_dim: {}", embedding.hidden_dim());
            let stats = embedding.stats();
            println!("     zero_pct: {:.1}%", stats.zero_pct());
        },
        Err(e) => println!("  ❌ Error: {e}"),
    }
    println!();
}

fn demo_density_rejection() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 2: Density Rejection (Catches PMAT-234 Bug)                │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let vocab_size = 1000;
    let hidden_dim = 64;
    let mut data = vec![0.0f32; vocab_size * hidden_dim];
    for item in &mut data[(945 * hidden_dim)..(vocab_size * hidden_dim)] {
        *item = 0.1;
    }

    println!("  Creating embedding with 94.5% zeros...");
    match ValidatedEmbedding::new(data, vocab_size, hidden_dim) {
        Ok(_) => println!("  ❌ Should have been rejected!"),
        Err(e) => {
            println!("  ✅ Correctly rejected!");
            print_error(&e);
        },
    }
    println!();
}

fn demo_validated_weight() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 3: ValidatedWeight                                         │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Valid weight
    let data: Vec<f32> = (0..100 * 64).map(|i| i as f32 * 0.001).collect();
    match ValidatedWeight::new(data, 100, 64, "q_proj.weight") {
        Ok(w) => println!("  ✅ ValidatedWeight created: {}", w.name()),
        Err(e) => println!("  ❌ Error: {e}"),
    }

    // Invalid weight (all zeros)
    let bad_data = vec![0.0f32; 100 * 64];
    println!("\n  Creating all-zero weight...");
    match ValidatedWeight::new(bad_data, 100, 64, "broken.weight") {
        Ok(_) => println!("  ❌ Should have been rejected!"),
        Err(e) => {
            println!("  ✅ Correctly rejected!");
            print_error(&e);
        },
    }
    println!();
}

fn print_error(e: &ContractValidationError) {
    println!("     Rule: {}", e.rule_id);
    println!("     Tensor: {}", e.tensor_name);
    println!("     Error: {}", e.message);
}
