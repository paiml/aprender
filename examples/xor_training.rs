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
    print_header();
    let (x, y) = create_dataset();
    let mut model = build_model();
    let final_loss = train_model(&mut model, &x, &y);
    let all_correct = evaluate_model(&model, &x, final_loss);
    print_summary(all_correct);
}

fn print_header() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║           XOR Neural Network Training Example              ║");
    println!("║      Proving Non-Linear Learning with Backpropagation      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

fn create_dataset() -> (Tensor, Tensor) {
    println!("📊 Dataset: XOR Truth Table");
    println!("   ┌─────────┬─────────┬──────────┐");
    println!("   │   X1    │   X2    │  Target  │");
    println!("   ├─────────┼─────────┼──────────┤");
    println!("   │    0    │    0    │    0     │");
    println!("   │    0    │    1    │    1     │");
    println!("   │    1    │    0    │    1     │");
    println!("   │    1    │    1    │    0     │");
    println!("   └─────────┴─────────┴──────────┘\n");

    let x = Tensor::new(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let y = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);
    (x, y)
}

fn build_model() -> Sequential {
    println!("🧠 Network Architecture:");
    println!("   Input(2) → Linear(2→8) → ReLU → Linear(8→1) → Sigmoid");
    println!("   Total parameters: 2×8 + 8 + 8×1 + 1 = 33\n");

    Sequential::new()
        .add(Linear::with_seed(2, 8, Some(42)))
        .add(ReLU::new())
        .add(Linear::with_seed(8, 1, Some(43)))
        .add(Sigmoid::new())
}

fn train_model(model: &mut Sequential, x: &Tensor, y: &Tensor) -> f32 {
    let learning_rate = 0.5;
    let mut optimizer = SGD::new(model.parameters_mut(), learning_rate);
    let loss_fn = MSELoss::new();
    let epochs = 1000;

    print_training_config(learning_rate);
    print_training_header();

    let mut final_loss = 0.0;
    for epoch in 0..epochs {
        final_loss = train_epoch(model, &mut optimizer, &loss_fn, x, y);
        print_epoch_progress(epoch, epochs, final_loss, model, x, y);
    }

    println!("   └─────────┴──────────────┴──────────┘\n");
    final_loss
}

fn print_training_config(learning_rate: f32) {
    println!("⚙️  Training Configuration:");
    println!("   Optimizer: SGD (lr={learning_rate})");
    println!("   Loss: Mean Squared Error");
    println!("   Epochs: 1000\n");
}

fn print_training_header() {
    println!("🚀 Training Progress:");
    println!("   ┌─────────┬──────────────┬──────────┐");
    println!("   │  Epoch  │     Loss     │ Accuracy │");
    println!("   ├─────────┼──────────────┼──────────┤");
}

fn train_epoch(
    model: &mut Sequential,
    optimizer: &mut SGD,
    loss_fn: &MSELoss,
    x: &Tensor,
    y: &Tensor,
) -> f32 {
    clear_graph();
    let x_grad = x.clone().requires_grad();
    let output = model.forward(&x_grad);
    let loss = loss_fn.forward(&output, y);
    let loss_value = loss.item();

    loss.backward();
    let mut params = model.parameters_mut();
    optimizer.step_with_params(&mut params);
    optimizer.zero_grad();

    loss_value
}

fn print_epoch_progress(
    epoch: usize,
    epochs: usize,
    loss: f32,
    model: &mut Sequential,
    x: &Tensor,
    y: &Tensor,
) {
    if epoch % 100 != 0 && epoch != epochs - 1 {
        return;
    }

    clear_graph();
    let output = model.forward(x);
    let accuracy = compute_accuracy(&output, y);
    println!("   │  {epoch:>5}  │    {loss:.6}  │   {accuracy:>3.0}%   │");
}

fn compute_accuracy(output: &Tensor, y: &Tensor) -> f32 {
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
    (correct as f32 / 4.0) * 100.0
}

fn evaluate_model(model: &Sequential, x: &Tensor, _final_loss: f32) -> bool {
    println!("📈 Final Results:");
    clear_graph();
    let final_output = model.forward(x);

    print_results_header();
    let all_correct = print_predictions(&final_output);
    println!("   └─────────┴─────────┴──────────┴────────────┴────────┘\n");

    all_correct
}

fn print_results_header() {
    println!("   ┌─────────┬─────────┬──────────┬────────────┬────────┐");
    println!("   │   X1    │   X2    │  Target  │ Prediction │ Status │");
    println!("   ├─────────┼─────────┼──────────┼────────────┼────────┤");
}

fn print_predictions(output: &Tensor) -> bool {
    let inputs: [(f32, f32); 4] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
    let targets: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
    let mut all_correct = true;

    for (i, ((x1, x2), target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let pred = output.data()[i];
        let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
        let correct = (pred_class - *target).abs() < 0.01;
        all_correct = all_correct && correct;
        let status = if correct { "  ✓  " } else { "  ✗  " };

        println!(
            "   │   {:>3}   │   {:>3}   │    {}     │    {:.3}    │{}│",
            *x1 as i32, *x2 as i32, *target as i32, pred, status
        );
    }

    all_correct
}

fn print_summary(all_correct: bool) {
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
