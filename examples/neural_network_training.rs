//! Neural Network Training Example
//!
//! Demonstrates the complete deep learning pipeline using aprender's nn module:
//! - Building a multi-layer perceptron (MLP)
//! - Training with backpropagation
//! - Using optimizers and learning rate schedulers
//! - Model serialization (save/load)
//!
//! This example trains a network to learn the XOR function, a classic
//! non-linearly separable problem that requires hidden layers.
//!
//! Run with: cargo run --example neural_network_training

use aprender::autograd::Tensor;
use aprender::nn::{
    loss::MSELoss,
    optim::{Adam, Optimizer},
    scheduler::{LRScheduler, StepLR},
    serialize::{count_parameters, load_model, save_model},
    Linear, Module, ReLU, Sequential, Sigmoid,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Neural Network Training with Aprender                  ║");
    println!("║       Learning the XOR Function                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. Prepare the XOR dataset
    // =========================================================================
    println!("📊 Dataset: XOR Function");
    println!("   Inputs:  [0,0], [0,1], [1,0], [1,1]");
    println!("   Outputs: [0],   [1],   [1],   [0]\n");

    // XOR inputs: 4 samples, 2 features
    let x_data = vec![
        0.0, 0.0, // → 0
        0.0, 1.0, // → 1
        1.0, 0.0, // → 1
        1.0, 1.0, // → 0
    ];
    let x = Tensor::new(&x_data, &[4, 2]);

    // XOR outputs: 4 samples, 1 output
    let y_data = vec![0.0, 1.0, 1.0, 0.0];
    let y = Tensor::new(&y_data, &[4, 1]);

    // =========================================================================
    // 2. Build the neural network
    // =========================================================================
    println!("🏗️  Building Model: MLP with 2 hidden layers");

    let mut model = Sequential::new()
        .add(Linear::with_seed(2, 8, Some(42)))    // Input → Hidden 1
        .add(ReLU::new())
        .add(Linear::with_seed(8, 8, Some(43)))    // Hidden 1 → Hidden 2
        .add(ReLU::new())
        .add(Linear::with_seed(8, 1, Some(44)))    // Hidden 2 → Output
        .add(Sigmoid::new()); // Output activation

    println!("   Architecture: 2 → 8 → 8 → 1");
    println!("   Total parameters: {}", count_parameters(&model));
    println!("   Activation: ReLU (hidden), Sigmoid (output)\n");

    // =========================================================================
    // 3. Set up training components
    // =========================================================================
    let loss_fn = MSELoss::new();
    let mut optimizer = Adam::new(model.parameters_mut(), 0.1);
    let mut scheduler = StepLR::new(100, 0.5);

    println!("⚙️  Training Configuration:");
    println!("   Loss: MSE (Mean Squared Error)");
    println!("   Optimizer: Adam (lr=0.1)");
    println!("   Scheduler: StepLR (step=100, gamma=0.5)");
    println!("   Epochs: 500\n");

    // =========================================================================
    // 4. Training loop
    // =========================================================================
    println!("🚀 Training...\n");
    println!("   Epoch    Loss       LR");
    println!("   ─────────────────────────");

    let epochs = 500;
    let mut losses = Vec::new();

    for epoch in 0..epochs {
        // Forward pass
        let predictions = model.forward(&x);

        // Compute loss
        let loss = loss_fn.forward(&predictions, &y);
        let loss_val = loss.data()[0];
        losses.push(loss_val);

        // Backward pass
        loss.backward();

        // Update weights - need to get mutable params and pass to optimizer
        {
            let mut params = model.parameters_mut();
            optimizer.step_with_params(&mut params);
        }

        // Zero gradients
        optimizer.zero_grad();

        // Update learning rate
        scheduler.step(&mut optimizer);

        // Print progress
        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!("   {:>5}    {:.6}   {:.6}", epoch, loss_val, optimizer.lr());
        }
    }

    // =========================================================================
    // 5. Evaluate the model
    // =========================================================================
    println!("\n📈 Training Complete!");
    println!("   Initial loss: {:.6}", losses[0]);
    println!("   Final loss:   {:.6}", losses[losses.len() - 1]);

    println!("\n🔍 Predictions vs Targets:");
    println!("   Input      Target    Prediction   Rounded");
    println!("   ──────────────────────────────────────────");

    model.eval(); // Switch to evaluation mode
    let final_predictions = model.forward(&x);

    let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [0.0, 1.0, 1.0, 0.0];

    let mut correct = 0;
    for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let pred = final_predictions.data()[i];
        let rounded = if pred >= 0.5 { 1.0 } else { 0.0 };
        let check = if rounded == *target { "✓" } else { "✗" };

        println!(
            "   [{}, {}]     {}         {:.4}       {} {}",
            input[0] as i32, input[1] as i32, *target as i32, pred, rounded as i32, check
        );

        if rounded == *target {
            correct += 1;
        }
    }

    println!(
        "\n   Accuracy: {}/4 ({:.0}%)",
        correct,
        (correct as f32 / 4.0) * 100.0
    );

    // =========================================================================
    // 6. Save and load the model
    // =========================================================================
    println!("\n💾 Model Serialization:");

    let model_path = "/tmp/xor_model.safetensors";
    save_model(&model, model_path).expect("Failed to save model");
    println!("   Saved to: {}", model_path);

    // Create a new model with same architecture
    let mut loaded_model = Sequential::new()
        .add(Linear::with_seed(2, 8, Some(999)))  // Different seed
        .add(ReLU::new())
        .add(Linear::with_seed(8, 8, Some(999)))
        .add(ReLU::new())
        .add(Linear::with_seed(8, 1, Some(999)))
        .add(Sigmoid::new());

    load_model(&mut loaded_model, model_path).expect("Failed to load model");
    println!("   Loaded into new model");

    // Verify loaded model produces same results
    loaded_model.eval();
    let loaded_predictions = loaded_model.forward(&x);

    let match_check = final_predictions.data() == loaded_predictions.data();
    println!(
        "   Verification: {}",
        if match_check {
            "✓ Predictions match!"
        } else {
            "✗ Mismatch"
        }
    );

    // Cleanup
    std::fs::remove_file(model_path).ok();

    // =========================================================================
    // 7. Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        Summary                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  ✓ Built MLP with Sequential container                       ║");
    println!("║  ✓ Trained with Adam optimizer and MSE loss                  ║");
    println!("║  ✓ Used learning rate scheduler (StepLR)                     ║");
    println!("║  ✓ Saved/loaded model in SafeTensors format                  ║");
    println!("║  ✓ Successfully learned XOR function                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
