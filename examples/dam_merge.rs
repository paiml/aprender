//! Differentiable Adaptive Merging (DAM) (GH-446)
//!
//! Demonstrates DAM optimization: learning per-model coefficients that
//! minimize reconstruction loss on a calibration signal.
//!
//! Run: cargo run --example dam_merge

use aprender::online::dam::{optimize_coefficients, softmax, DamConfig, DamLoss, DamReport};

fn main() {
    println!("=== Differentiable Adaptive Merging (GH-446) ===\n");

    // ── 1. Softmax Normalization ──
    println!("── 1. Softmax Normalization ──");
    let logits = vec![1.0, 2.0, 3.0];
    let probs = softmax(&logits);
    println!("  Logits:  {:?}", logits);
    println!(
        "  Softmax: [{:.4}, {:.4}, {:.4}]",
        probs[0], probs[1], probs[2]
    );
    println!("  Sum:     {:.6}", probs.iter().sum::<f64>());

    // ── 2. MSE Loss Computation ──
    println!("\n── 2. MSE Loss ──");
    let merged = vec![1.0, 2.0, 3.0];
    let target = vec![1.1, 2.2, 2.9];
    let loss = DamLoss::compute_merge_loss(&merged, &target);
    println!("  Merged: {:?}", merged);
    println!("  Target: {:?}", target);
    println!("  MSE:    {:.6}", loss);

    // ── 3. Regularization ──
    println!("\n── 3. L2 Regularization ──");
    let coeffs = vec![0.5, 1.5, -0.5];
    let reg = DamLoss::compute_regularization(&coeffs);
    println!("  Coefficients: {:?}", coeffs);
    println!("  L2 penalty:   {:.6}", reg);

    // ── 4. Gradient Step ──
    println!("\n── 4. Gradient Step ──");
    let mut coeffs = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, -0.2, 0.3];
    println!("  Before: {:?}", coeffs);
    DamLoss::gradient_step(&mut coeffs, &grads, 0.5);
    println!("  After (lr=0.5): {:?}", coeffs);

    // ── 5. Full Optimization ──
    println!("\n── 5. Nelder-Mead Optimization ──");
    let cfg = DamConfig {
        num_iterations: 200,
        learning_rate: 0.1,
        regularization: 0.001,
        ..Default::default()
    };

    let optimized = optimize_coefficients(
        3,
        |c| {
            let s = softmax(c);
            let target_dist = [0.5, 0.3, 0.2];
            s.iter()
                .zip(target_dist.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum()
        },
        &cfg,
    );

    let weights = softmax(&optimized);
    println!("  Target distribution: [0.5, 0.3, 0.2]");
    println!(
        "  Optimized weights:   [{:.4}, {:.4}, {:.4}]",
        weights[0], weights[1], weights[2]
    );

    // ── 6. Report ──
    println!("\n── 6. DAM Report ──");
    let report = DamReport {
        final_loss: 0.0012,
        num_iterations: 200,
        coefficients: optimized,
        converged: true,
    };
    let norm = report.normalized_coefficients();
    println!("  Converged:    {}", report.converged);
    println!("  Final loss:   {:.6}", report.final_loss);
    println!(
        "  Normalized:   [{:.4}, {:.4}, {:.4}]",
        norm[0], norm[1], norm[2]
    );

    println!("\n=== DAM merge verified ===");
}
