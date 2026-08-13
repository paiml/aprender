#![allow(clippy::disallowed_methods)]
//! Chapter 24: Switch From PyTorch
//!
//! API equivalence: torch → aprender
//! Citation: Paszke et al., "PyTorch," arXiv:1912.01703
//! Contract: contracts/apr-book-ch24-v1.yaml

use aprender::autograd::{clear_graph, Tensor};
use aprender::nn::{loss::MSELoss, optim::SGD, Linear, Module, Sequential};

fn main() {
    println!("=== Switch From PyTorch ===");
    println!();

    // API equivalence table
    println!("| PyTorch                    | aprender                           |");
    println!("|----------------------------|------------------------------------|");
    println!("| torch.tensor([...])        | Tensor::new(&[...], &shape)        |");
    println!("| torch.nn.Linear(in, out)   | Linear::new(in, out)               |");
    println!("| torch.nn.Sequential(...)   | Sequential::new().add(...)         |");
    println!("| loss.backward()            | loss.backward()                    |");
    println!("| optimizer.step()           | optimizer.step_with_params(&mut p) |");
    println!("| optimizer.zero_grad()      | clear_graph()                      |");
    println!("| torch.nn.MSELoss()         | MSELoss::new()                     |");
    println!();

    // Demonstrate: same training loop as PyTorch
    // PyTorch: model = nn.Sequential(nn.Linear(2,4), nn.Linear(4,1))
    let mut model = Sequential::new()
        .add(Linear::new(2, 4))
        .add(Linear::new(4, 1));

    // PyTorch: x = torch.tensor([[1,2],[3,4],[5,6],[7,8]], dtype=torch.float32)
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
    let mut optimizer = SGD::new(model.parameters_mut(), 0.001);

    // PyTorch training loop equivalent
    let mut initial_loss = 0.0_f32;
    let mut final_loss = 0.0_f32;
    for epoch in 0..100 {
        clear_graph(); // PyTorch: optimizer.zero_grad()
        let x_g = x.clone().requires_grad();
        let pred = model.forward(&x_g); // PyTorch: pred = model(x)
        let loss = loss_fn.forward(&pred, &y); // PyTorch: loss = criterion(pred, y)
        let l = loss.item();
        if epoch == 0 {
            initial_loss = l;
        }
        loss.backward(); // PyTorch: loss.backward()
        let mut p = model.parameters_mut();
        optimizer.step_with_params(&mut p); // PyTorch: optimizer.step()
        final_loss = l;
    }

    println!("Training loop (100 epochs):");
    println!("  Initial loss: {initial_loss:.4}");
    println!("  Final loss:   {final_loss:.4}");
    assert!(final_loss < initial_loss, "Training must reduce loss");

    // Key differences
    println!();
    println!("Key differences:");
    println!("  1. Ownership: aprender uses Rust ownership (no GC)");
    println!("  2. clear_graph() replaces zero_grad() (tape-based autograd)");
    println!("  3. step_with_params() takes &mut — Rust borrow checker enforced");
    println!("  4. No CUDA/Python interop overhead — pure Rust + SIMD");

    println!();
    println!("Chapter 24 contracts: PASSED");
}
