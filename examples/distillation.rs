//! Example: Knowledge Distillation
//!
//! This example demonstrates three types of knowledge distillation:
//! 1. Standard temperature-scaled KL divergence
//! 2. Multi-teacher ensemble distillation
//! 3. Progressive layer-wise distillation

use entrenar::distill::{DistillationLoss, EnsembleDistiller, ProgressiveDistiller};
use ndarray::array;

fn main() {
    println!("=== Knowledge Distillation Examples ===\n");

    // Example 1: Standard Distillation
    println!("1. STANDARD DISTILLATION\n");
    standard_distillation_example();

    // Example 2: Multi-Teacher Ensemble
    println!("\n2. MULTI-TEACHER ENSEMBLE DISTILLATION\n");
    ensemble_distillation_example();

    // Example 3: Progressive Layer-wise Distillation
    println!("\n3. PROGRESSIVE LAYER-WISE DISTILLATION\n");
    progressive_distillation_example();

    println!("\n=== Examples Complete ===\n");
}

fn standard_distillation_example() {
    println!("Standard temperature-scaled distillation with KL divergence\n");

    // Create distillation loss function
    let temperature = 3.0;
    let alpha = 0.7; // 70% soft targets, 30% hard targets
    let loss_fn = DistillationLoss::new(temperature, alpha);

    println!("Configuration:");
    println!("  Temperature: {temperature}");
    println!("  Alpha (soft weight): {alpha}");
    println!("  Hard weight: {}", 1.0 - alpha);

    // Teacher model outputs (confident predictions)
    let teacher_logits = array![
        [10.0, 2.0, 1.0], // Strong on class 0
        [1.0, 12.0, 2.0], // Strong on class 1
        [2.0, 1.0, 11.0]  // Strong on class 2
    ];

    // Student model outputs (less confident)
    let student_logits = array![[7.0, 3.0, 2.0], [2.0, 8.0, 3.0], [3.0, 2.0, 7.0]];

    let labels = vec![0, 1, 2];

    let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);

    println!("\nResults:");
    println!("  Teacher logits: [[10, 2, 1], [1, 12, 2], [2, 1, 11]]");
    println!("  Student logits: [[7, 3, 2], [2, 8, 3], [3, 2, 7]]");
    println!("  Ground truth labels: [0, 1, 2]");
    println!("  Distillation loss: {loss:.4}");

    // Show effect of temperature
    println!("\nTemperature effect:");
    for t in [1.0, 2.0, 5.0, 10.0] {
        let loss_fn_temp = DistillationLoss::new(t, alpha);
        let loss_temp = loss_fn_temp.forward(&student_logits, &teacher_logits, &labels);
        println!("  T={t:4.1} → loss={loss_temp:.4}");
    }

    // Show effect of alpha
    println!("\nAlpha (soft weight) effect:");
    for a in [0.0, 0.3, 0.5, 0.7, 1.0] {
        let loss_fn_alpha = DistillationLoss::new(temperature, a);
        let loss_alpha = loss_fn_alpha.forward(&student_logits, &teacher_logits, &labels);
        println!("  α={a:.1} → loss={loss_alpha:.4}");
    }
}

fn ensemble_distillation_example() {
    println!("Distilling knowledge from multiple teachers\n");

    // Three specialized teachers
    let teacher1 = array![
        [10.0, 3.0, 2.0], // Strong on class 0
        [8.0, 4.0, 3.0],
        [7.0, 5.0, 4.0]
    ];

    let teacher2 = array![
        [3.0, 10.0, 2.0], // Strong on class 1
        [4.0, 9.0, 3.0],
        [5.0, 8.0, 4.0]
    ];

    let teacher3 = array![
        [2.0, 3.0, 10.0], // Strong on class 2
        [3.0, 4.0, 9.0],
        [4.0, 5.0, 8.0]
    ];

    println!("Three specialized teachers:");
    println!("  Teacher 1: Strong on class 0");
    println!("  Teacher 2: Strong on class 1");
    println!("  Teacher 3: Strong on class 2");

    // Uniform ensemble (equal weights)
    println!("\n--- Uniform Ensemble ---");
    let uniform_distiller = EnsembleDistiller::uniform(3, 2.0);
    let teachers = vec![teacher1.clone(), teacher2.clone(), teacher3.clone()];
    let uniform_ensemble = uniform_distiller.combine_teachers(&teachers);

    println!("Weights: [0.33, 0.33, 0.33]");
    println!("Ensemble logits (first sample):");
    println!(
        "  [{:.2}, {:.2}, {:.2}]",
        uniform_ensemble[[0, 0]],
        uniform_ensemble[[0, 1]],
        uniform_ensemble[[0, 2]]
    );

    // Weighted ensemble (prefer teacher 1)
    println!("\n--- Weighted Ensemble ---");
    let weighted_distiller = EnsembleDistiller::new(vec![3.0, 1.0, 1.0], 2.0);
    let weighted_ensemble = weighted_distiller.combine_teachers(&teachers);

    println!("Weights: [0.60, 0.20, 0.20]");
    println!("Ensemble logits (first sample):");
    println!(
        "  [{:.2}, {:.2}, {:.2}]",
        weighted_ensemble[[0, 0]],
        weighted_ensemble[[0, 1]],
        weighted_ensemble[[0, 2]]
    );
    println!("  → Biased toward class 0 due to higher teacher 1 weight");

    // Compute distillation loss
    let student = array![[6.0, 5.0, 4.0], [5.0, 6.0, 4.0], [4.0, 5.0, 6.0]];
    let labels = vec![0, 1, 2];

    let loss = weighted_distiller.distillation_loss(&student, &teachers, &labels, 0.7);
    println!("\nStudent distillation loss: {loss:.4}");
}

fn progressive_distillation_example() {
    println!("Layer-wise distillation of intermediate representations\n");

    // Simulate 3 layers of hidden states
    let student_hiddens = vec![
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 1
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], // Layer 2
        array![[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], // Layer 3
    ];

    let teacher_hiddens = vec![
        array![[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]], // Layer 1
        array![[2.2, 3.2, 4.2], [5.2, 6.2, 7.2]], // Layer 2
        array![[3.5, 4.5, 5.5], [6.5, 7.5, 8.5]], // Layer 3 (larger diff)
    ];

    println!("Configuration:");
    println!("  Number of layers: 3");
    println!("  Hidden dimension: 3");
    println!("  Batch size: 2");

    // Uniform weighting across layers
    println!("\n--- Uniform Layer Weights ---");
    let uniform_distiller = ProgressiveDistiller::uniform(3, 2.0);
    println!("Weights: [0.33, 0.33, 0.33]");

    let mse_loss = uniform_distiller.layer_wise_mse_loss(&student_hiddens, &teacher_hiddens);
    let cosine_loss = uniform_distiller.layer_wise_cosine_loss(&student_hiddens, &teacher_hiddens);

    println!("  MSE loss: {mse_loss:.4}");
    println!("  Cosine loss: {cosine_loss:.4}");

    // Progressive weighting (higher weight on later layers)
    println!("\n--- Progressive Layer Weights ---");
    let progressive_distiller = ProgressiveDistiller::new(vec![0.5, 1.0, 2.0], 2.0);
    println!("Weights: [0.14, 0.29, 0.57] (later layers weighted more)");

    let mse_loss_prog =
        progressive_distiller.layer_wise_mse_loss(&student_hiddens, &teacher_hiddens);
    let cosine_loss_prog =
        progressive_distiller.layer_wise_cosine_loss(&student_hiddens, &teacher_hiddens);

    println!("  MSE loss: {mse_loss_prog:.4}");
    println!("  Cosine loss: {cosine_loss_prog:.4}");
    println!("  → Higher loss due to emphasis on layer 3 (which has larger error)");

    // Combined distillation (logits + hidden states)
    println!("\n--- Combined Distillation ---");

    let student_logits = array![[5.0, 4.0, 3.0], [4.0, 5.0, 3.0]];
    let teacher_logits = array![[6.0, 4.5, 3.0], [4.5, 6.0, 3.0]];
    let labels = vec![0, 1];

    let alpha = 0.7; // Soft target weight
    let beta = 0.3; // Hidden state weight

    let combined_loss = progressive_distiller.combined_loss(
        &student_logits,
        &teacher_logits,
        &student_hiddens,
        &teacher_hiddens,
        &labels,
        alpha,
        beta,
    );

    println!("Alpha (soft target weight): {alpha}");
    println!("Beta (hidden state weight): {beta}");
    println!("Combined loss: {combined_loss:.4}");
    println!(
        "  = {:.0}% logit distillation + {:.0}% hidden state matching",
        (1.0 - beta) * 100.0,
        beta * 100.0
    );
}
