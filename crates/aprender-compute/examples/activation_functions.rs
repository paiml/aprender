#![allow(clippy::disallowed_methods)]
//! Activation Functions Example
//!
//! Demonstrates all 11 activation functions available in Trueno for neural network
//! architectures. Each activation function is shown with its typical use case and
//! characteristics.
//!
//! Run with: cargo run --example activation_functions

use trueno::Vector;

fn main() {
    println!("🧠 Trueno Activation Functions Demo");
    println!("=====================================\n");

    // Sample input spanning negative to positive range
    let input = Vector::from_slice(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    println!("Input vector: {:?}\n", input.as_slice());

    // ========================================================================
    // Classic Activations
    // ========================================================================

    println!("📊 Classic Activation Functions");
    println!("--------------------------------\n");

    // ReLU: max(0, x) - Most common activation
    let relu_output = input.relu().expect("Example should not fail");
    println!("✓ ReLU (max(0, x))");
    println!("  Use case: Hidden layers in most neural networks");
    println!("  Output: {:?}", relu_output.as_slice());
    println!("  Characteristic: Zero for negative, identity for positive\n");

    // Leaky ReLU: Prevents dying ReLU problem
    let leaky_relu_output = input.leaky_relu(0.01).expect("Example should not fail");
    println!("✓ Leaky ReLU (x if x > 0, else 0.01*x)");
    println!("  Use case: When dying ReLU is a problem");
    println!("  Output: {:?}", leaky_relu_output.as_slice());
    println!("  Characteristic: Small negative slope prevents neuron death\n");

    // ELU: Exponential Linear Unit
    let elu_output = input.elu(1.0).expect("Example should not fail");
    println!("✓ ELU (x if x > 0, else α(e^x - 1))");
    println!("  Use case: When smooth gradients are needed");
    println!("  Output: {:?}", elu_output.as_slice());
    println!("  Characteristic: Smooth, bounded below by -α\n");

    // Sigmoid: Logistic function
    let sigmoid_output = input.sigmoid().expect("Example should not fail");
    println!("✓ Sigmoid (1 / (1 + e^(-x)))");
    println!("  Use case: Binary classification, output layer");
    println!("  Output: {:?}", sigmoid_output.as_slice());
    println!("  Characteristic: Bounded [0, 1], probabilistic interpretation\n");

    // ========================================================================
    // Modern Activations
    // ========================================================================

    println!("🚀 Modern Activation Functions");
    println!("--------------------------------\n");

    // GELU: Gaussian Error Linear Unit (used in transformers)
    let gelu_output = input.gelu().expect("Example should not fail");
    println!("✓ GELU (Gaussian Error Linear Unit)");
    println!("  Use case: BERT, GPT, T5, modern transformers");
    println!("  Output: {:?}", gelu_output.as_slice());
    println!("  Characteristic: Smooth, non-monotonic, better gradient flow\n");

    // Swish/SiLU: Self-gated activation
    let swish_output = input.swish().expect("Example should not fail");
    println!("✓ Swish/SiLU (x * sigmoid(x))");
    println!("  Use case: EfficientNet, MobileNet v3");
    println!("  Output: {:?}", swish_output.as_slice());
    println!("  Characteristic: Self-gated, smooth, unbounded above\n");

    // Hardswish: Efficient approximation for mobile
    let hardswish_output = input.hardswish().expect("Example should not fail");
    println!("✓ Hardswish (x * clip(x+3, 0, 6) / 6)");
    println!("  Use case: MobileNetV3, efficient on-device inference");
    println!("  Output: {:?}", hardswish_output.as_slice());
    println!("  Characteristic: Fast computation, no exp(), mobile-optimized\n");

    // Mish: Self-regularized non-monotonic
    let mish_output = input.mish().expect("Example should not fail");
    println!("✓ Mish (x * tanh(softplus(x)))");
    println!("  Use case: YOLOv4, modern object detection");
    println!("  Output: {:?}", mish_output.as_slice());
    println!("  Characteristic: Smooth, self-regularized, non-monotonic\n");

    // SELU: Self-normalizing
    let selu_output = input.selu().expect("Example should not fail");
    println!("✓ SELU (λ * elu(x, α))");
    println!("  Use case: Self-normalizing neural networks");
    println!("  Output: {:?}", selu_output.as_slice());
    println!("  Characteristic: Self-normalizing, reduces need for batch norm\n");

    // ========================================================================
    // Probabilistic Activations
    // ========================================================================

    println!("📈 Probabilistic Activation Functions");
    println!("--------------------------------------\n");

    // Softmax: Convert logits to probabilities
    let logits = Vector::from_slice(&[2.0, 1.0, 0.1]);
    let softmax_output = logits.softmax().expect("Example should not fail");
    println!("✓ Softmax (multi-class output layer)");
    println!("  Input logits: {:?}", logits.as_slice());
    println!("  Output probabilities: {:?}", softmax_output.as_slice());
    println!("  Sum: {:.6}", softmax_output.as_slice().iter().sum::<f32>());
    println!("  Characteristic: Outputs sum to 1.0, probabilistic\n");

    // Log-Softmax: Numerically stable for cross-entropy loss
    let log_softmax_output = logits.log_softmax().expect("Example should not fail");
    println!("✓ Log-Softmax (numerically stable cross-entropy)");
    println!("  Output: {:?}", log_softmax_output.as_slice());
    println!("  Characteristic: More stable than log(softmax(x))\n");

    // ========================================================================
    // Comparison: Different behaviors
    // ========================================================================

    println!("🔍 Behavior Comparison at Key Points");
    println!("-------------------------------------\n");

    let test_points = Vector::from_slice(&[-5.0, -1.0, 0.0, 1.0, 5.0]);
    println!("Test points: {:?}\n", test_points.as_slice());

    println!("Activation   | -5.0     | -1.0     | 0.0      | 1.0      | 5.0");
    println!("-------------|----------|----------|----------|----------|----------");

    print_activation_row("ReLU        ", &test_points.relu().expect("Example should not fail"));
    print_activation_row(
        "Leaky ReLU  ",
        &test_points.leaky_relu(0.01).expect("Example should not fail"),
    );
    print_activation_row("ELU         ", &test_points.elu(1.0).expect("Example should not fail"));
    print_activation_row("Sigmoid     ", &test_points.sigmoid().expect("Example should not fail"));
    print_activation_row("GELU        ", &test_points.gelu().expect("Example should not fail"));
    print_activation_row("Swish       ", &test_points.swish().expect("Example should not fail"));
    print_activation_row(
        "Hardswish   ",
        &test_points.hardswish().expect("Example should not fail"),
    );
    print_activation_row("Mish        ", &test_points.mish().expect("Example should not fail"));
    print_activation_row("SELU        ", &test_points.selu().expect("Example should not fail"));

    println!("\n");

    // ========================================================================
    // Practical Example: Simple Neural Network Layer
    // ========================================================================

    println!("🎯 Practical Example: Neural Network Layer");
    println!("-------------------------------------------\n");

    // Simulate a batch of pre-activations from a linear layer
    let layer_output = Vector::from_slice(&[0.5, -0.3, 1.2, -0.8, 2.1]);
    println!("Layer output (pre-activation): {:?}", layer_output.as_slice());

    // Different activation choices for different architectures
    println!("\nIf using CNN (e.g., ResNet):");
    let cnn_activation = layer_output.relu().expect("Example should not fail");
    println!("  → ReLU: {:?}", cnn_activation.as_slice());

    println!("\nIf using Transformer (e.g., BERT):");
    let transformer_activation = layer_output.gelu().expect("Example should not fail");
    println!("  → GELU: {:?}", transformer_activation.as_slice());

    println!("\nIf using EfficientNet:");
    let efficientnet_activation = layer_output.swish().expect("Example should not fail");
    println!("  → Swish: {:?}", efficientnet_activation.as_slice());

    println!("\nIf using MobileNetV3 (on-device):");
    let mobilenet_activation = layer_output.hardswish().expect("Example should not fail");
    println!("  → Hardswish: {:?}", mobilenet_activation.as_slice());

    println!("\nIf using YOLOv4 (object detection):");
    let yolo_activation = layer_output.mish().expect("Example should not fail");
    println!("  → Mish: {:?}", yolo_activation.as_slice());

    println!("\nIf using self-normalizing network:");
    let snn_activation = layer_output.selu().expect("Example should not fail");
    println!("  → SELU: {:?}", snn_activation.as_slice());

    println!("\n✨ All 11 activation functions computed successfully!");
    println!("   Note: Activations use scalar ops; SIMD optimization planned for v0.3.0");
}

fn print_activation_row(name: &str, output: &Vector<f32>) {
    print!("{}", name);
    for &val in output.as_slice() {
        print!("| {:8.4} ", val);
    }
    println!("|");
}
