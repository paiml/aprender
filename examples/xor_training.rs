//! XOR Neural Network Training Example
//!
//! Demonstrates aprender's PyTorch-compatible deep learning by solving the classic
//! XOR problem - the "Hello World" of neural networks.
//!
//! XOR is not linearly separable, proving the network learns non-linear patterns:
//!   (0,0) → 0
//!   (0,1) → 1
//!   (1,0) → 1
//!   (1,1) → 0
//!
//! Run with: cargo run --example xor_training

use aprender::autograd::{clear_graph, Tensor};
use aprender::nn::{
    loss::MSELoss, optim::SGD, Linear, Module, Optimizer, ReLU, Sequential, Sigmoid,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║           XOR Neural Network Training Example              ║");
    println!("║      Proving Non-Linear Learning with Backpropagation      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. Define the XOR dataset
    // =========================================================================
    println!("📊 Dataset: XOR Truth Table");
    println!("   ┌─────────┬─────────┬──────────┐");
    println!("   │   X1    │   X2    │  Target  │");
    println!("   ├─────────┼─────────┼──────────┤");
    println!("   │    0    │    0    │    0     │");
    println!("   │    0    │    1    │    1     │");
    println!("   │    1    │    0    │    1     │");
    println!("   │    1    │    1    │    0     │");
    println!("   └─────────┴─────────┴──────────┘\n");

    // Input: 4 samples, 2 features each
    let x = Tensor::new(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);

    // Target: 4 samples, 1 output each
    let y = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);

    // =========================================================================
    // 2. Build the neural network
    // =========================================================================
    println!("🧠 Network Architecture:");
    println!("   Input(2) → Linear(2→8) → ReLU → Linear(8→1) → Sigmoid");
    println!("   Total parameters: 2×8 + 8 + 8×1 + 1 = 33\n");

    let mut model = Sequential::new()
        .add(Linear::with_seed(2, 8, Some(42))) // Hidden layer
        .add(ReLU::new())
        .add(Linear::with_seed(8, 1, Some(43))) // Output layer
        .add(Sigmoid::new());

    // =========================================================================
    // 3. Setup optimizer and loss function
    // =========================================================================
    let learning_rate = 0.5;
    let mut optimizer = SGD::new(model.parameters_mut(), learning_rate);
    let loss_fn = MSELoss::new();

    println!("⚙️  Training Configuration:");
    println!("   Optimizer: SGD (lr={})", learning_rate);
    println!("   Loss: Mean Squared Error");
    println!("   Epochs: 1000\n");

    // =========================================================================
    // 4. Training loop
    // =========================================================================
    println!("🚀 Training Progress:");
    println!("   ┌─────────┬──────────────┬──────────┐");
    println!("   │  Epoch  │     Loss     │ Accuracy │");
    println!("   ├─────────┼──────────────┼──────────┤");

    let epochs = 1000;
    #[allow(unused_assignments)]
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        // Clear computation graph from previous iteration
        clear_graph();

        // Forward pass
        let x_grad = x.clone().requires_grad();
        let output = model.forward(&x_grad);

        // Compute loss
        let loss = loss_fn.forward(&output, &y);
        final_loss = loss.item();

        // Backward pass
        loss.backward();

        // Update weights - must pass mutable params
        let mut params = model.parameters_mut();
        optimizer.step_with_params(&mut params);
        optimizer.zero_grad();

        // Print progress every 100 epochs
        if epoch % 100 == 0 || epoch == epochs - 1 {
            // Calculate accuracy
            let predictions: Vec<f32> = output
                .data()
                .iter()
                .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
                .collect();
            let targets = y.data();
            let correct = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(&p, &t)| (p - t).abs() < 0.01)
                .count();
            let accuracy = (correct as f32 / 4.0) * 100.0;

            println!(
                "   │  {:>5}  │    {:.6}  │   {:>3.0}%   │",
                epoch, final_loss, accuracy
            );
        }
    }

    println!("   └─────────┴──────────────┴──────────┘\n");

    // =========================================================================
    // 5. Final evaluation
    // =========================================================================
    println!("📈 Final Results:");
    clear_graph();
    let final_output = model.forward(&x);

    println!("   ┌─────────┬─────────┬──────────┬────────────┬────────┐");
    println!("   │   X1    │   X2    │  Target  │ Prediction │ Status │");
    println!("   ├─────────┼─────────┼──────────┼────────────┼────────┤");

    let inputs: [(f32, f32); 4] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
    let targets: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
    let mut all_correct = true;

    for (i, ((x1, x2), target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let pred = final_output.data()[i];
        let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
        let correct = (pred_class - *target).abs() < 0.01;
        all_correct = all_correct && correct;
        let status = if correct { "  ✓  " } else { "  ✗  " };

        println!(
            "   │   {:>3}   │   {:>3}   │    {}     │    {:.3}    │{}│",
            *x1 as i32, *x2 as i32, *target as i32, pred, status
        );
    }

    println!("   └─────────┴─────────┴──────────┴────────────┴────────┘\n");

    // =========================================================================
    // 6. Summary
    // =========================================================================
    if all_correct {
        println!("✅ SUCCESS: Network learned XOR perfectly!");
        println!("   The network discovered the non-linear decision boundary.\n");
    } else {
        println!("⚠️  Network is still learning. Try more epochs or adjust learning rate.\n");
    }

    println!("📚 Key Takeaways:");
    println!("   • XOR requires hidden layers (not linearly separable)");
    println!("   • Backpropagation computes gradients automatically");
    println!("   • ReLU activation enables non-linear transformations");
    println!("   • Sigmoid squashes output to [0,1] for binary classification");
}
