//! BEAT-PYTORCH-AUTOGRAD — Pillar-2 (PyTorch) correctness beat (PMAT-746).
//!
//! apr concedes raw MLP training THROUGHPUT to PyTorch (~11× slower: MKL + fused
//! autograd vs apr's per-step graph rebuild — see docs/BEATS.md Pillar-2 CONCEDED).
//! Its defensible Pillar-2 win is the same wedge as P3/P4: provable CORRECTNESS.
//! This gate proves apr's reverse-mode autograd computes gradients NUMERICALLY
//! EQUIVALENT to PyTorch's on a fixed 2-layer MLP — i.e. apr's training math is a
//! faithful, contract-gated replacement, not an approximation. It also hard-guards
//! the #2000 Linear weight-gradient-path fix against silent regression: a broken
//! backward would diverge from these pinned PyTorch values.
//!
//! Reference PINNED from PyTorch (`uv run --with torch`) on the fixed network
//! relu(x @ W1^T + b1) @ W2^T + b2 with MSELoss (mean reduction), measured
//! 2026-06-13. apr must match every parameter gradient element-wise.
//! Contract: contracts/apr-pytorch-autograd-equivalence-beat-v1.yaml.

use aprender::autograd::{clear_graph, get_grad, Tensor};
use aprender::nn::{Linear, MSELoss, Module, ReLU};

// Fixed deterministic network + data (identical to the pinned PyTorch run).
const X: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, 0.3]; // [2,4]
const Y: [f32; 4] = [0.5, -0.2, 0.1, 0.4]; // [2,2]
const W1: [f32; 12] = [
    0.1, -0.2, 0.3, 0.0, 0.2, 0.1, -0.1, 0.4, -0.3, 0.2, 0.1, -0.2,
]; // [3,4]
const B1: [f32; 3] = [0.05, -0.05, 0.1];
const W2: [f32; 6] = [0.3, -0.1, 0.2, 0.1, 0.25, -0.15]; // [2,3]
const B2: [f32; 2] = [0.0, 0.1];

// PyTorch reference (torch, MSE mean), pinned 2026-06-13.
const PT_LOSS: f32 = 0.100079;
const PT_DW1: [f32; 12] = [
    -0.019070, -0.007945, -0.010545, -0.029615, -0.006577, 0.015583, 0.024680, 0.018103, -0.007160,
    -0.014320, -0.021480, -0.028640,
];
const PT_DB1: [f32; 3] = [-0.080900, 0.038725, -0.071600];
const PT_DW2: [f32; 6] = [
    -0.028685, -0.037020, -0.014010, 0.010790, -0.002490, 0.009960,
];
const PT_DB2: [f32; 2] = [-0.283500, 0.041500];

const TOL: f32 = 1e-4;

fn max_abs_diff(apr: &[f32], pt: &[f32], name: &str) -> f32 {
    assert_eq!(
        apr.len(),
        pt.len(),
        "{name}: apr grad len {} != pytorch {}",
        apr.len(),
        pt.len()
    );
    apr.iter()
        .zip(pt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn beat_apr_autograd_matches_pytorch_gradients() {
    // from_vec defaults to requires_grad=false; Linear::new uses .requires_grad(),
    // so re-enable grad tracking on the pinned weights or backward populates nothing.
    let mut l1 = Linear::new(4, 3);
    l1.set_weight(Tensor::from_vec(W1.to_vec(), &[3, 4]).requires_grad());
    l1.set_bias(Tensor::from_vec(B1.to_vec(), &[3]).requires_grad());
    let mut l2 = Linear::new(3, 2);
    l2.set_weight(Tensor::from_vec(W2.to_vec(), &[2, 3]).requires_grad());
    l2.set_bias(Tensor::from_vec(B2.to_vec(), &[2]).requires_grad());

    clear_graph();
    let x = Tensor::from_vec(X.to_vec(), &[2, 4]);
    let y = Tensor::from_vec(Y.to_vec(), &[2, 2]);

    // relu(x @ W1^T + b1) @ W2^T + b2
    let h = ReLU.forward(&l1.forward(&x));
    let out = l2.forward(&h);
    let loss = MSELoss::new().forward(&out, &y);

    // (0) Forward parity: same loss as PyTorch.
    assert!(
        (loss.item() - PT_LOSS).abs() < TOL,
        "forward MSE {} != PyTorch {PT_LOSS} — forward diverged before backward",
        loss.item()
    );

    loss.backward();

    let dw1 = get_grad(l1.weight().id())
        .expect("dW1 (Linear weight grad) — backward path broken")
        .data()
        .to_vec();
    let db1 = get_grad(l1.bias().unwrap().id())
        .expect("db1")
        .data()
        .to_vec();
    let dw2 = get_grad(l2.weight().id()).expect("dW2").data().to_vec();
    let db2 = get_grad(l2.bias().unwrap().id())
        .expect("db2")
        .data()
        .to_vec();

    let d_w1 = max_abs_diff(&dw1, &PT_DW1, "dW1");
    let d_b1 = max_abs_diff(&db1, &PT_DB1, "db1");
    let d_w2 = max_abs_diff(&dw2, &PT_DW2, "dW2");
    let d_b2 = max_abs_diff(&db2, &PT_DB2, "db2");
    let worst = d_w1.max(d_b1).max(d_w2).max(d_b2);

    assert!(
        worst < TOL,
        "apr autograd NOT equivalent to PyTorch: max|Δgrad|={worst:.6} (dW1={d_w1:.2e} db1={d_b1:.2e} dW2={d_w2:.2e} db2={d_b2:.2e}). \
         apr dW1={dw1:?}",
    );
    println!(
        "BEAT-PYTORCH-AUTOGRAD: apr gradients ≡ PyTorch — max|Δ|={worst:.2e} (dW1={d_w1:.1e} db1={d_b1:.1e} dW2={d_w2:.1e} db2={d_b2:.1e})"
    );
}
