//! Neural Network Classification Example
//!
//! Demonstrates training a classifier using aprender's nn module:
//! - Building a multi-layer perceptron for classification
//! - Training with CrossEntropyLoss
//! - Using Softmax for probability outputs
//!
//! This example trains a network to classify 2D points into 4 quadrants.
//!
//! Run with: cargo run --example classification_training

use aprender::autograd::Tensor;
use aprender::nn::{
    loss::CrossEntropyLoss, optim::Adam, Linear, Module, Optimizer, ReLU, Sequential, Softmax,
};

#[allow(clippy::too_many_lines)]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Neural Network Classification with Aprender            ║");
    println!("║       Classifying 2D Points into 4 Quadrants                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. Prepare the quadrant classification dataset
    // =========================================================================
    println!("📊 Dataset: Quadrant Classification");
    println!("   Class 0: Q1 (+x, +y)  |  Class 1: Q2 (-x, +y)");
    println!("   Class 2: Q3 (-x, -y)  |  Class 3: Q4 (+x, -y)\n");

    // Training data: 8 points, 2 per quadrant
    let x_data = vec![
        // Q1 (class 0): positive x, positive y
        1.0, 1.0, 0.5, 0.8, // Q2 (class 1): negative x, positive y
        -1.0, 1.0, -0.7, 0.6, // Q3 (class 2): negative x, negative y
        -1.0, -1.0, -0.8, -0.5, // Q4 (class 3): positive x, negative y
        1.0, -1.0, 0.6, -0.9,
    ];
    let x = Tensor::new(&x_data, &[8, 2]);

    // Class labels (as indices)
    let y_data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let y = Tensor::new(&y_data, &[8]);

    // =========================================================================
    // 2. Build the classification network
    // =========================================================================
    println!("🏗️  Building Model: MLP Classifier");

    let mut model = Sequential::new()
        .add(Linear::with_seed(2, 16, Some(42)))   // Input → Hidden 1
        .add(ReLU::new())
        .add(Linear::with_seed(16, 16, Some(43)))  // Hidden 1 → Hidden 2
        .add(ReLU::new())
        .add(Linear::with_seed(16, 4, Some(44))); // Hidden 2 → Output (4 classes)

    // Note: CrossEntropyLoss includes softmax internally, so we don't add Softmax to model
    // We'll add it for inference only

    println!("   Architecture: 2 → 16 → 16 → 4");
    println!("   Activation: ReLU (hidden)");
    println!("   Output: 4 classes (quadrants)\n");

    // =========================================================================
    // 3. Set up training components
    // =========================================================================
    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters_mut(), 0.05);

    println!("⚙️  Training Configuration:");
    println!("   Loss: CrossEntropyLoss");
    println!("   Optimizer: Adam (lr=0.05)");
    println!("   Epochs: 300\n");

    // =========================================================================
    // 4. Training loop
    // =========================================================================
    println!("🚀 Training...\n");
    println!("   Epoch    Loss       Accuracy");
    println!("   ─────────────────────────────");

    let epochs = 300;

    for epoch in 0..epochs {
        // Forward pass (logits)
        let logits = model.forward(&x);

        // Compute loss
        let loss = loss_fn.forward(&logits, &y);
        let loss_val = loss.data()[0];

        // Backward pass
        loss.backward();

        // Update weights
        {
            let mut params = model.parameters_mut();
            optimizer.step_with_params(&mut params);
        }

        // Zero gradients
        optimizer.zero_grad();

        // Compute accuracy
        let accuracy = compute_accuracy(&logits, &y);

        // Print progress
        if epoch % 30 == 0 || epoch == epochs - 1 {
            println!(
                "   {:>5}    {:.4}     {:.0}%",
                epoch,
                loss_val,
                accuracy * 100.0
            );
        }
    }

    // =========================================================================
    // 5. Evaluate the model
    // =========================================================================
    println!("\n📈 Training Complete!");

    model.eval();
    let final_logits = model.forward(&x);

    // Apply softmax for probabilities
    let softmax = Softmax::new(-1);
    let probs = softmax.forward(&final_logits);

    println!("\n🔍 Predictions:");
    println!("   Point        Target  Predicted  Confidence");
    println!("   ─────────────────────────────────────────────");

    let points = [
        (1.0, 1.0),
        (0.5, 0.8), // Q1
        (-1.0, 1.0),
        (-0.7, 0.6), // Q2
        (-1.0, -1.0),
        (-0.8, -0.5), // Q3
        (1.0, -1.0),
        (0.6, -0.9), // Q4
    ];
    let targets = [0, 0, 1, 1, 2, 2, 3, 3];
    let quadrant_names = ["Q1", "Q2", "Q3", "Q4"];

    let mut correct = 0;
    for (i, ((px, py), &target)) in points.iter().zip(targets.iter()).enumerate() {
        let row_start = i * 4;
        let prob_slice = &probs.data()[row_start..row_start + 4];

        // Find predicted class
        let (pred_class, &max_prob) = prob_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("probability slice should not be empty");

        let check = if pred_class == target {
            correct += 1;
            "✓"
        } else {
            "✗"
        };

        println!(
            "   ({:>4.1}, {:>4.1})   {}       {}         {:.1}%  {}",
            px,
            py,
            quadrant_names[target],
            quadrant_names[pred_class],
            max_prob * 100.0,
            check
        );
    }

    println!(
        "\n   Final Accuracy: {}/8 ({:.0}%)",
        correct,
        (correct as f32 / 8.0) * 100.0
    );

    // =========================================================================
    // 6. Test on new points
    // =========================================================================
    println!("\n🧪 Testing on New Points:");

    let test_points = vec![
        2.0, 2.0, // Q1
        -3.0, 0.5, // Q2
        -0.1, -0.1, // Q3 (close to origin)
        0.5, -2.0, // Q4
    ];
    let test_x = Tensor::new(&test_points, &[4, 2]);
    let test_targets = [0, 1, 2, 3];

    let test_logits = model.forward(&test_x);
    let test_probs = softmax.forward(&test_logits);

    println!("   Point        Expected  Predicted  Confidence");
    println!("   ───────────────────────────────────────────────");

    let test_coords = [(2.0, 2.0), (-3.0, 0.5), (-0.1, -0.1), (0.5, -2.0)];
    for (i, ((px, py), &expected)) in test_coords.iter().zip(test_targets.iter()).enumerate() {
        let row_start = i * 4;
        let prob_slice = &test_probs.data()[row_start..row_start + 4];

        let (pred_class, &max_prob) = prob_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("probability slice should not be empty");

        let check = if pred_class == expected { "✓" } else { "✗" };

        println!(
            "   ({:>4.1}, {:>4.1})   {}        {}         {:.1}%  {}",
            px,
            py,
            quadrant_names[expected],
            quadrant_names[pred_class],
            max_prob * 100.0,
            check
        );
    }

    // =========================================================================
    // 7. Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        Summary                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  ✓ Built MLP classifier with 4 output classes                ║");
    println!("║  ✓ Trained with CrossEntropyLoss                             ║");
    println!("║  ✓ Used Softmax for probability outputs                      ║");
    println!("║  ✓ Achieved classification of 2D quadrants                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

/// Compute classification accuracy
fn compute_accuracy(logits: &Tensor, targets: &Tensor) -> f32 {
    let batch_size = logits.shape()[0];
    let num_classes = logits.shape()[1];

    let mut correct = 0;
    for i in 0..batch_size {
        let row_start = i * num_classes;
        let logit_slice = &logits.data()[row_start..row_start + num_classes];

        // Find predicted class (argmax)
        let pred_class = logit_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .expect("logit slice should not be empty");

        let target_class = targets.data()[i] as usize;

        if pred_class == target_class {
            correct += 1;
        }
    }

    correct as f32 / batch_size as f32
}
