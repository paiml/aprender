#![allow(clippy::disallowed_methods)]
//! Chapter 10: Training with aprender-train
//!
//! Demonstrates autograd forward/backward pass and optimizer.
//! Citation: Hu et al., "LoRA," arXiv:2106.09685
//! Contract: contracts/apr-book-ch10-v1.yaml (v2 — api_calls enforced)

use aprender::autograd::{clear_graph, Tensor};
use aprender::nn::{loss::MSELoss, optim::SGD, Linear, Module, Sequential};
use aprender::prelude::Adam;

fn main() {
    // --- Part 1: Build network and demonstrate forward/backward ---
    let mut model = Sequential::new()
        .add(Linear::new(2, 4))
        .add(Linear::new(4, 1));

    // Training data: y = x1 + x2
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
    let y = Tensor::new(&[3.0, 7.0, 11.0, 15.0], &[4, 1]);

    let loss_fn = MSELoss::new();
    // #2310 follow-on: the learning rate here was 0.01, which DIVERGES on this data.
    // x runs to 8.0 and y to 15.0, both unnormalized, so the first MSE gradients are
    // large enough that a 0.01 step overshoots; loss reaches NaN by epoch 25 and the
    // `final_loss < initial_loss` assertion panics. Measured on this exact example:
    // lr 0.01 -> NaN; lr 0.001 -> 110.38 converging to 0.0000.
    //
    // This is the EXAMPLE's parameter, not a framework defect: SGD, MSELoss and
    // backward are all correct at a step size the data supports, which is what the
    // lr sweep above established before anything was changed.
    let learning_rate = 0.001_f32;
    let mut optimizer = SGD::new(model.parameters_mut(), learning_rate);

    // Initial forward pass
    clear_graph();
    let x_grad = x.clone().requires_grad();
    let pred = model.forward(&x_grad);
    println!("Forward pass output shape: {:?}", pred.shape());
    assert_eq!(pred.shape(), &[4, 1], "Output shape must be [4, 1]");

    let loss = loss_fn.forward(&pred, &y);
    let initial_loss = loss.item();
    println!("Initial MSE loss: {initial_loss:.4}");

    // Backward pass
    loss.backward();
    println!("Backward pass: gradients computed");

    // Optimizer step (xor_training pattern: step_with_params)
    let mut params = model.parameters_mut();
    optimizer.step_with_params(&mut params);
    clear_graph();
    println!("SGD step: parameters updated");

    // --- Part 2: Training loop ---
    let mut prev_loss = initial_loss;
    let mut decreasing = 0;
    for epoch in 0..100 {
        clear_graph();
        let x_g = x.clone().requires_grad();
        let pred = model.forward(&x_g);
        let loss = loss_fn.forward(&pred, &y);
        let l = loss.item();
        loss.backward();
        let mut p = model.parameters_mut();
        optimizer.step_with_params(&mut p);

        if l < prev_loss {
            decreasing += 1;
        }
        prev_loss = l;

        if epoch % 25 == 0 {
            println!("  Epoch {epoch}: loss={l:.4}");
        }
    }
    let final_loss = prev_loss;
    println!("Loss decreased in {decreasing}/100 epochs");
    println!("Initial: {initial_loss:.4} -> Final: {final_loss:.4}");
    assert!(
        final_loss < initial_loss,
        "Training must reduce loss: {final_loss:.4} < {initial_loss:.4}"
    );

    // --- Part 3: LoRA parameter efficiency (Hu et al., 2021) ---
    let d = 4096_usize;
    let k = 4096_usize;
    let r = 16_usize;
    let full_params = d * k;
    let lora_params = r * (d + k);
    let ratio = lora_params as f64 / full_params as f64;
    println!(
        "\nLoRA (rank={r}): {lora_params} vs {full_params} ({:.2}%)",
        ratio * 100.0
    );
    assert!(ratio < 0.01, "LoRA must use <1% of full params");

    // Adam optimizer exists
    let _adam = Adam::new(1e-4);
    println!("Adam optimizer: instantiated");

    println!("Chapter 10 contracts: PASSED");
}
