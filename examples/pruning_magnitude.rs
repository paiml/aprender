//! Magnitude Pruning Example
//!
//! Demonstrates neural network pruning using magnitude-based importance:
//! - L1 magnitude (absolute value)
//! - L2 magnitude (squared value)
//! - Unstructured and N:M sparsity patterns
//!
//! # References
//! - Han et al. (2015) - Learning both Weights and Connections
//!
//! Run with: cargo run --example pruning_magnitude

use aprender::nn::Linear;
use aprender::pruning::{
    generate_nm_mask, generate_unstructured_mask, Importance, MagnitudeImportance,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Magnitude Pruning with Aprender                      ║");
    println!("║         Prune neural networks by weight magnitude            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. Create a simple linear layer
    // =========================================================================
    println!("📊 Creating Linear Layer (16 → 8)");
    let layer = Linear::new(16, 8);
    let weights = layer.weight();
    let total_params = weights.data().len();
    println!("   Weight shape: {:?}", weights.shape());
    println!("   Total parameters: {}\n", total_params);

    // =========================================================================
    // 2. Compute L1 Magnitude Importance
    // =========================================================================
    println!("🔬 Computing L1 Magnitude Importance");
    let l1_importance = MagnitudeImportance::l1();
    let l1_scores = l1_importance.compute(&layer, None).expect("L1 importance");

    println!("   Method: {}", l1_scores.method);
    println!("   Stats:");
    println!("     - Min:  {:.6}", l1_scores.stats.min);
    println!("     - Max:  {:.6}", l1_scores.stats.max);
    println!("     - Mean: {:.6}", l1_scores.stats.mean);
    println!("     - Std:  {:.6}\n", l1_scores.stats.std);

    // =========================================================================
    // 3. Compute L2 Magnitude Importance
    // =========================================================================
    println!("🔬 Computing L2 Magnitude Importance");
    let l2_importance = MagnitudeImportance::l2();
    let l2_scores = l2_importance.compute(&layer, None).expect("L2 importance");

    println!("   Method: {}", l2_scores.method);
    println!("   Stats:");
    println!("     - Min:  {:.6}", l2_scores.stats.min);
    println!("     - Max:  {:.6}", l2_scores.stats.max);
    println!("     - Mean: {:.6}", l2_scores.stats.mean);
    println!("     - Std:  {:.6}\n", l2_scores.stats.std);

    // =========================================================================
    // 4. Generate Unstructured Sparsity Mask (50%)
    // =========================================================================
    println!("✂️  Generating Unstructured Mask (50% sparsity)");
    let mask = generate_unstructured_mask(&l1_scores.values, 0.5).expect("Unstructured mask");

    let sparsity = mask.sparsity();
    let nonzeros = mask.nnz();
    let zeros = mask.num_zeros();
    println!("   Achieved sparsity: {:.1}%", sparsity * 100.0);
    println!("   Non-zero weights: {}", nonzeros);
    println!("   Pruned weights: {}\n", zeros);

    // =========================================================================
    // 5. Generate N:M Sparsity Mask (2:4)
    // =========================================================================
    println!("✂️  Generating 2:4 N:M Mask (50% structured sparsity)");

    // For N:M, we need the weight tensor to have compatible dimensions
    let nm_layer = Linear::new(8, 8); // 64 elements, divisible by 4
    let nm_importance = MagnitudeImportance::l1();

    match nm_importance.compute(&nm_layer, None) {
        Ok(nm_scores) => match generate_nm_mask(&nm_scores.values, 2, 4) {
            Ok(nm_mask) => {
                println!("   Pattern: 2:4 (2 non-zeros per 4 elements)");
                println!("   Achieved sparsity: {:.1}%", nm_mask.sparsity() * 100.0);

                // Verify structure
                let mask_data = nm_mask.tensor().data();
                let mut valid_groups = 0;
                let mut total_groups = 0;
                for chunk in mask_data.chunks(4) {
                    if chunk.len() == 4 {
                        let chunk_nonzeros: usize =
                            chunk.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).sum();
                        if chunk_nonzeros == 2 {
                            valid_groups += 1;
                        }
                        total_groups += 1;
                    }
                }
                println!("   Valid 2:4 groups: {}/{}", valid_groups, total_groups);
            }
            Err(e) => println!("   N:M mask error: {}", e),
        },
        Err(e) => println!("   Importance error: {}", e),
    }

    // =========================================================================
    // 6. Apply mask to weights
    // =========================================================================
    println!("\n📉 Applying Mask to Weights");
    let mut pruned_weights = weights.clone();
    mask.apply(&mut pruned_weights).expect("Apply mask");

    // Count zeros after pruning
    let zeros_after: usize = pruned_weights
        .data()
        .iter()
        .filter(|&&v| v.abs() < 1e-10)
        .count();
    let pruned_len = pruned_weights.data().len();
    println!(
        "   Zeros after pruning: {} ({:.1}%)",
        zeros_after,
        zeros_after as f32 / pruned_len as f32 * 100.0
    );

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Pruning Summary                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Original parameters:   {:>6}                               ║",
        total_params
    );
    println!(
        "║  Pruned parameters:     {:>6} ({:.0}% reduction)              ║",
        zeros_after,
        sparsity * 100.0
    );
    println!(
        "║  Remaining parameters:  {:>6}                               ║",
        total_params - zeros_after
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}
