//! Falsifier: Flatten / MaxPool1d / MaxPool2d / AvgPool2d / GlobalAvgPool2d
//! backward MUST flow gradient to their input (severed-graph guard + gradcheck).
//!
//! Obligations:
//! - OBLIG-FLATTEN-BACKWARD-GRAD-FLOW
//! - OBLIG-MAXPOOL1D-BACKWARD-GRAD-FLOW
//! - OBLIG-MAXPOOL2D-BACKWARD-GRAD-FLOW
//! - OBLIG-AVGPOOL2D-BACKWARD-GRAD-FLOW
//! - OBLIG-GLOBALAVGPOOL2D-BACKWARD-GRAD-FLOW
//!
//! BUG (PMAT-913): these forwards built their output via `Tensor::new`, which
//! severs the autograd graph. After `loss.backward()`, `input.grad` was `None`,
//! so a pooling/flatten layer in the middle of a network blocked gradient from
//! reaching the upstream conv/linear weights. Each test below asserts the input
//! grad is non-None (severed-graph guard) AND matches a self-contained central
//! finite-difference gradcheck (no torch dep) — not a tautological `is_some`.
//!
//! The MaxPool inputs use DISTINCT values per window so the argmax is
//! unambiguous (the subgradient of `max` is well defined and the finite
//! difference is exact away from ties).

use super::{AvgPool2d, Flatten, GlobalAvgPool2d, MaxPool1d, MaxPool2d};
use crate::autograd::{self, Tensor};
use crate::nn::Module;

const FD_EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

/// Detached, non-uniform upstream coefficient vector.
fn coeff(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.37 + 0.13 * (i as f32)).collect()
}

/// Scalar loss = sum_i c_i * y_i, with c detached (no graph edge through it).
fn scalar_loss(output: &Tensor, c: &[f32]) -> Tensor {
    let ct = Tensor::new(c, output.shape());
    output.mul(&ct).sum()
}

/// Run forward + loss WITHOUT a graph, with the input value at `flat_idx`
/// perturbed by `delta`. Used for the finite-difference reference.
fn perturbed_loss<F>(
    x_data: &[f32],
    x_shape: &[usize],
    flat_idx: usize,
    delta: f32,
    fwd: F,
    c: &[f32],
) -> f32
where
    F: Fn(&Tensor) -> Tensor,
{
    autograd::no_grad(|| {
        let mut xd = x_data.to_vec();
        xd[flat_idx] += delta;
        let x = Tensor::new(&xd, x_shape);
        let y = fwd(&x);
        scalar_loss(&y, c).item()
    })
}

fn assert_close(analytic: f32, numeric: f32, what: &str) {
    let denom = analytic.abs().max(numeric.abs()).max(1.0);
    let rel = (analytic - numeric).abs() / denom;
    assert!(
        rel < TOL,
        "{what}: analytic grad {analytic} != finite-diff {numeric} (rel err {rel})"
    );
}

/// Generic gradcheck driver: forward `fwd(x)`, build scalar loss, backward,
/// assert input grad present + finite + matches central finite difference at
/// every input element.
fn gradcheck_input<F>(name: &str, x_data: &[f32], x_shape: &[usize], fwd: F)
where
    F: Fn(&Tensor) -> Tensor,
{
    autograd::clear_graph();
    let x = Tensor::new(x_data, x_shape).requires_grad();
    let xid = x.id();
    let y = fwd(&x);
    let c = coeff(y.numel());
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let grad = autograd::get_grad(xid)
        .unwrap_or_else(|| panic!("{name}: input received NO gradient — autograd graph severed"));
    assert_eq!(grad.shape(), x_shape, "{name}: grad shape mismatch");
    assert!(
        grad.data().iter().all(|v| v.is_finite()),
        "{name}: non-finite grad"
    );
    assert!(
        grad.data().iter().any(|&v| v.abs() > 1e-9),
        "{name}: all-zero grad"
    );

    for i in 0..x_data.len() {
        let num = (perturbed_loss(x_data, x_shape, i, FD_EPS, &fwd, &c)
            - perturbed_loss(x_data, x_shape, i, -FD_EPS, &fwd, &c))
            / (2.0 * FD_EPS);
        assert_close(grad.data()[i], num, &format!("{name} dL/dx[{i}]"));
    }
}

#[test]
fn flatten_backward_flows_grad_to_input() {
    // [2,3,4] -> [2,12]; pure view, grad is identity reshaped.
    let x: Vec<f32> = (0..24)
        .map(|i| 0.2 + 0.05 * (i as f32) - 0.01 * ((i * i) as f32))
        .collect();
    gradcheck_input("Flatten", &x, &[2, 3, 4], |t| Flatten::new().forward(t));
}

#[test]
fn maxpool1d_backward_routes_grad_to_argmax() {
    // [1,2,6], kernel=2 stride=2 -> [1,2,3]. Distinct values => unambiguous argmax.
    let x: Vec<f32> = vec![
        0.1, 0.9, 0.3, 0.8, 0.2, 0.7, // channel 0
        1.5, 0.4, 1.1, 0.6, 1.9, 0.5, // channel 1
    ];
    gradcheck_input("MaxPool1d", &x, &[1, 2, 6], |t| {
        MaxPool1d::new(2).forward(t)
    });
}

#[test]
fn maxpool2d_backward_routes_grad_to_argmax() {
    // [1,1,4,4], kernel=2 stride=2 -> [1,1,2,2]. Distinct per-window maxima.
    let x: Vec<f32> = vec![
        0.1, 0.2, 0.5, 0.4, //
        0.9, 0.3, 0.6, 0.45, //
        0.15, 0.25, 0.8, 0.35, //
        0.7, 0.2, 0.55, 0.95, //
    ];
    gradcheck_input("MaxPool2d", &x, &[1, 1, 4, 4], |t| {
        MaxPool2d::new(2).forward(t)
    });
}

#[test]
fn avgpool2d_backward_distributes_grad() {
    let x: Vec<f32> = (0..16).map(|i| 0.3 + 0.07 * (i as f32)).collect();
    gradcheck_input("AvgPool2d", &x, &[1, 1, 4, 4], |t| {
        AvgPool2d::new(2).forward(t)
    });
}

#[test]
fn globalavgpool2d_backward_distributes_grad() {
    let x: Vec<f32> = (0..16).map(|i| 0.4 + 0.09 * (i as f32)).collect();
    gradcheck_input("GlobalAvgPool2d", &x, &[1, 1, 4, 4], |t| {
        GlobalAvgPool2d::new().forward(t)
    });
}

#[test]
fn maxpool2d_two_channel_independent_argmax() {
    // Distinct argmax per channel/window; confirms grad lands per-channel.
    let x: Vec<f32> = vec![
        // channel 0
        2.0, 0.1, 0.2, 0.3, //
        0.4, 0.5, 0.6, 0.7, //
        0.8, 0.9, 3.0, 1.0, //
        1.1, 1.2, 1.3, 1.4, //
        // channel 1
        0.05, 5.0, 0.15, 0.25, //
        0.35, 0.45, 0.55, 0.65, //
        0.75, 0.85, 0.95, 6.0, //
        1.05, 1.15, 1.25, 1.35, //
    ];
    gradcheck_input("MaxPool2d-2ch", &x, &[1, 2, 4, 4], |t| {
        MaxPool2d::new(2).forward(t)
    });
}
