//! Optimizer Demo Example
//!
//! Demonstrates SGD and Adam optimizers for gradient-based optimization:
//! - SGD with and without momentum
//! - Adam optimizer with adaptive learning rates
//! - Comparison on quadratic optimization problem
//! - Loss function visualization
//!
//! Run with: `cargo run --example optimizer_demo`

use aprender::prelude::*;

fn main() {
    println!("=== Optimizer Demo ===\n");

    // Simple quadratic optimization problem: minimize f(x) = (x - 5)^2
    // Gradient: f'(x) = 2(x - 5)
    // Optimal solution: x* = 5.0

    let target = 5.0;
    let learning_rate = 0.1;
    let max_iterations = 50;

    println!("Optimization problem: minimize f(x) = (x - {})^2", target);
    println!("Optimal solution: x* = {}\n", target);

    // 1. SGD without momentum
    println!("--- SGD (no momentum) ---");
    let mut sgd = SGD::new(learning_rate);
    let mut params = Vector::from_slice(&[0.0]); // Start at x = 0

    println!(
        "Initial: x = {:.4}, loss = {:.4}",
        params[0],
        loss_fn(params[0], target)
    );

    for i in 0..max_iterations {
        let grad = gradient_fn(params[0], target);
        let grad_vec = Vector::from_slice(&[grad]);
        sgd.step(&mut params, &grad_vec);

        if i % 10 == 9 || i == 0 {
            println!(
                "Iter {}: x = {:.4}, loss = {:.4}",
                i + 1,
                params[0],
                loss_fn(params[0], target)
            );
        }
    }
    println!("Final: x = {:.4}\n", params[0]);

    // 2. SGD with momentum
    println!("--- SGD (with momentum = 0.9) ---");
    let mut sgd_momentum = SGD::new(learning_rate).with_momentum(0.9);
    let mut params = Vector::from_slice(&[0.0]);

    println!(
        "Initial: x = {:.4}, loss = {:.4}",
        params[0],
        loss_fn(params[0], target)
    );

    for i in 0..max_iterations {
        let grad = gradient_fn(params[0], target);
        let grad_vec = Vector::from_slice(&[grad]);
        sgd_momentum.step(&mut params, &grad_vec);

        if i % 10 == 9 || i == 0 {
            println!(
                "Iter {}: x = {:.4}, loss = {:.4}",
                i + 1,
                params[0],
                loss_fn(params[0], target)
            );
        }
    }
    println!("Final: x = {:.4}\n", params[0]);

    // 3. Adam optimizer
    println!("--- Adam Optimizer ---");
    let mut adam = Adam::new(learning_rate);
    let mut params = Vector::from_slice(&[0.0]);

    println!(
        "Initial: x = {:.4}, loss = {:.4}",
        params[0],
        loss_fn(params[0], target)
    );

    for i in 0..max_iterations {
        let grad = gradient_fn(params[0], target);
        let grad_vec = Vector::from_slice(&[grad]);
        adam.step(&mut params, &grad_vec);

        if i % 10 == 9 || i == 0 {
            println!(
                "Iter {}: x = {:.4}, loss = {:.4}",
                i + 1,
                params[0],
                loss_fn(params[0], target)
            );
        }
    }
    println!("Final: x = {:.4}\n", params[0]);

    // 4. Loss function comparison
    println!("=== Loss Functions Demo ===\n");

    // Create some predictions and targets
    let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_true = Vector::from_slice(&[1.5, 2.3, 2.8, 4.2, 4.9]);

    println!("Predictions: {:?}", y_pred.as_slice());
    println!("True values: {:?}", y_true.as_slice());
    println!();

    // MSE Loss
    let mse = mse_loss(&y_pred, &y_true);
    let mse_loss_obj = MSELoss;
    println!("MSE Loss:");
    println!("  Functional API: {:.4}", mse);
    println!("  OOP API: {:.4}", mse_loss_obj.compute(&y_pred, &y_true));
    println!();

    // MAE Loss
    let mae = mae_loss(&y_pred, &y_true);
    let mae_loss_obj = MAELoss;
    println!("MAE Loss:");
    println!("  Functional API: {:.4}", mae);
    println!("  OOP API: {:.4}", mae_loss_obj.compute(&y_pred, &y_true));
    println!();

    // Huber Loss
    let delta = 1.0;
    let huber = huber_loss(&y_pred, &y_true, delta);
    let huber_loss_obj = HuberLoss::new(delta);
    println!("Huber Loss (delta = {}):", delta);
    println!("  Functional API: {:.4}", huber);
    println!("  OOP API: {:.4}", huber_loss_obj.compute(&y_pred, &y_true));
    println!();

    println!("=== Key Insights ===");
    println!("- SGD: Simple gradient descent, may oscillate");
    println!("- SGD + Momentum: Accelerates convergence, smoother trajectory");
    println!("- Adam: Adaptive learning rates, often fastest convergence");
    println!("- MSE: Sensitive to outliers (quadratic)");
    println!("- MAE: Robust to outliers (linear)");
    println!("- Huber: Combines MSE and MAE benefits");
}

/// Loss function: (x - target)^2
fn loss_fn(x: f32, target: f32) -> f32 {
    (x - target).powi(2)
}

/// Gradient function: 2(x - target)
fn gradient_fn(x: f32, target: f32) -> f32 {
    2.0 * (x - target)
}
