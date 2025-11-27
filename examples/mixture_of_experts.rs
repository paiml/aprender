//! Mixture of Experts (MoE) Ensemble Example
//!
//! Demonstrates specialized ensemble learning with learnable gating network
//! that routes inputs to the most appropriate expert(s).

use aprender::ensemble::{GatingNetwork, MixtureOfExperts, MoeConfig, SoftmaxGating};
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;
use aprender::Result;
use serde::{Deserialize, Serialize};

fn main() {
    println!("Mixture of Experts (MoE) - Ensemble Learning Example");
    println!("=====================================================\n");

    println!("Architecture:");
    println!("  Input --> Gating Network --> Expert Weights");
    println!("                    |");
    println!("             +------+------+");
    println!("             v      v      v");
    println!("          Expert0 Expert1 Expert2");
    println!("             v      v      v");
    println!("             +------+------+");
    println!("                    v");
    println!("           Weighted Output\n");

    // Example 1: Basic MoE with SimpleExperts
    println!("Example 1: Basic MoE with 3 Experts");
    println!("-----------------------------------");
    basic_moe_example();

    // Example 2: Sparse MoE (top-k routing)
    println!("\nExample 2: Sparse MoE (top-k = 1)");
    println!("---------------------------------");
    sparse_moe_example();

    // Example 3: Temperature control
    println!("\nExample 3: Gating Temperature Control");
    println!("--------------------------------------");
    temperature_example();

    // Example 4: Save/Load persistence
    println!("\nExample 4: Model Persistence");
    println!("----------------------------");
    persistence_example();

    // Example 5: APR format
    println!("\nExample 5: APR Format (Bundled)");
    println!("-------------------------------");
    apr_format_example();

    println!("\n=== MoE Examples Complete! ===");
    println!("\nKey Benefits:");
    println!("  - Specialization: Each expert focuses on subset of problem");
    println!("  - Conditional Compute: Only top-k experts execute per input");
    println!("  - Scalability: Add experts without retraining others");
    println!("  - Bundled Persistence: Single .apr file for deployment");
}

fn basic_moe_example() {
    // Create gating network: 4 input features, 3 experts
    let gating = SoftmaxGating::new(4, 3);

    // Build MoE with 3 specialized experts
    let moe = MixtureOfExperts::<SimpleExpert, _>::builder()
        .gating(gating)
        .expert(SimpleExpert::new(10.0)) // Expert 0: low values
        .expert(SimpleExpert::new(50.0)) // Expert 1: medium values
        .expert(SimpleExpert::new(90.0)) // Expert 2: high values
        .build()
        .expect("MoE build should succeed");

    println!("  Experts: 3 (SimpleExpert)");
    println!("  Input Features: 4");
    println!("  Config: top_k=1 (default)\n");

    // Test predictions
    let input = [1.0, 2.0, 3.0, 4.0];
    let output = moe.predict(&input);

    println!("  Input: {:?}", input);
    println!("  Output: {:.2}", output);
    println!("  (Weighted combination of expert outputs)");
}

fn sparse_moe_example() {
    let gating = SoftmaxGating::new(4, 3);
    let config = MoeConfig::default().with_top_k(1); // Only 1 expert per input

    let moe = MixtureOfExperts::<SimpleExpert, _>::builder()
        .gating(gating)
        .expert(SimpleExpert::new(10.0))
        .expert(SimpleExpert::new(50.0))
        .expert(SimpleExpert::new(90.0))
        .config(config)
        .build()
        .expect("MoE build should succeed");

    println!("  Config: top_k=1 (sparse routing)");
    println!("  Only highest-weighted expert executes\n");

    let input = [1.0, 2.0, 3.0, 4.0];
    let output = moe.predict(&input);

    println!("  Input: {:?}", input);
    println!("  Output: {:.2}", output);
    println!("  (Single expert output, no averaging)");

    // With sparse routing, output should be exactly one expert's value
    let is_exact = (output - 10.0).abs() < 1e-6
        || (output - 50.0).abs() < 1e-6
        || (output - 90.0).abs() < 1e-6;
    println!(
        "  Exact expert output: {}",
        if is_exact { "Yes" } else { "No" }
    );
}

fn temperature_example() {
    let input = [1.0, 2.0, 3.0, 4.0];

    // Low temperature = peaked distribution (confident routing)
    let sharp_gating = SoftmaxGating::new(4, 3).with_temperature(0.1);
    let sharp_weights = sharp_gating.forward(&input);

    // High temperature = uniform distribution (uncertain routing)
    let uniform_gating = SoftmaxGating::new(4, 3).with_temperature(10.0);
    let uniform_weights = uniform_gating.forward(&input);

    println!("  Temperature controls routing confidence:\n");

    println!("  Low temp (0.1) - Peaked distribution:");
    println!(
        "    Weights: [{:.3}, {:.3}, {:.3}]",
        sharp_weights[0], sharp_weights[1], sharp_weights[2]
    );
    let max_sharp = sharp_weights.iter().cloned().fold(0.0f32, f32::max);
    println!("    Max weight: {:.3} (confident)", max_sharp);

    println!("\n  High temp (10.0) - Uniform distribution:");
    println!(
        "    Weights: [{:.3}, {:.3}, {:.3}]",
        uniform_weights[0], uniform_weights[1], uniform_weights[2]
    );
    let max_uniform = uniform_weights.iter().cloned().fold(0.0f32, f32::max);
    println!("    Max weight: {:.3} (uncertain)", max_uniform);

    // Verify weights sum to 1.0
    let sum: f32 = sharp_weights.iter().sum();
    println!("\n  Weights sum to: {:.3} (normalized)", sum);
}

fn persistence_example() {
    let gating = SoftmaxGating::new(4, 2);
    let moe = MixtureOfExperts::<SimpleExpert, _>::builder()
        .gating(gating)
        .expert(SimpleExpert::new(25.0))
        .expert(SimpleExpert::new(75.0))
        .build()
        .expect("MoE build should succeed");

    let input = [1.0, 2.0, 3.0, 4.0];
    let original_output = moe.predict(&input);

    // Save to binary format
    let tmp_path = std::env::temp_dir().join("moe_example.bin");
    moe.save(&tmp_path).expect("Save should succeed");

    let file_size = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);

    // Load and verify
    let loaded = MixtureOfExperts::<SimpleExpert, SoftmaxGating>::load(&tmp_path)
        .expect("Load should succeed");
    let loaded_output = loaded.predict(&input);

    println!("  Binary format (bincode):");
    println!("    File: {}", tmp_path.display());
    println!("    Size: {} bytes", file_size);
    println!("    Original output: {:.4}", original_output);
    println!("    Loaded output:   {:.4}", loaded_output);
    println!(
        "    Match: {}",
        if (original_output - loaded_output).abs() < 1e-6 {
            "Yes"
        } else {
            "No"
        }
    );

    // Cleanup
    let _ = std::fs::remove_file(&tmp_path);
}

fn apr_format_example() {
    let gating = SoftmaxGating::new(4, 2);
    let moe = MixtureOfExperts::<SimpleExpert, _>::builder()
        .gating(gating)
        .expert(SimpleExpert::new(30.0))
        .expert(SimpleExpert::new(70.0))
        .build()
        .expect("MoE build should succeed");

    // Save to APR format with header
    let tmp_path = std::env::temp_dir().join("moe_example.apr");
    moe.save_apr(&tmp_path).expect("Save APR should succeed");

    // Verify APR header
    let bytes = std::fs::read(&tmp_path).expect("Read should succeed");
    let magic = std::str::from_utf8(&bytes[0..4]).unwrap_or("????");
    let file_size = bytes.len();

    println!("  APR format (with header):");
    println!("    File: {}", tmp_path.display());
    println!("    Size: {} bytes", file_size);
    println!("    Magic: {} (APRN = valid)", magic);

    println!("\n  Bundled Architecture:");
    println!("    model.apr");
    println!("    +-- Header (ModelType::MixtureOfExperts = 0x0040)");
    println!("    +-- Metadata (MoeConfig)");
    println!("    +-- Payload");
    println!("        +-- Gating Network");
    println!("        +-- Experts[0..n]");

    println!("\n  Benefits:");
    println!("    - Atomic save/load (no partial states)");
    println!("    - Single file deployment");
    println!("    - Checksummed integrity");

    // Cleanup
    let _ = std::fs::remove_file(&tmp_path);
}

/// Simple expert for demonstration purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimpleExpert {
    output_value: f32,
}

impl SimpleExpert {
    fn new(value: f32) -> Self {
        Self {
            output_value: value,
        }
    }
}

impl Estimator for SimpleExpert {
    fn fit(&mut self, _x: &Matrix<f32>, _y: &Vector<f32>) -> Result<()> {
        Ok(())
    }

    fn predict(&self, _x: &Matrix<f32>) -> Vector<f32> {
        Vector::from_slice(&[self.output_value])
    }

    fn score(&self, _x: &Matrix<f32>, _y: &Vector<f32>) -> f32 {
        1.0
    }
}
