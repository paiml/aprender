//! Falsifier: LayerNorm + RMSNorm backward MUST flow gradient to affine params.
//!
//! Obligations:
//! - OBLIG-LAYERNORM-BACKWARD-GRAD-FLOW
//! - OBLIG-RMSNORM-BACKWARD-GRAD-FLOW
//!
//! BUG (PMAT-907): the canonical `nn::functional::layer_norm` / `rms_norm`
//! built their output via `Tensor::from_vec`, which severs the autograd graph.
//! After `loss.backward()`, `weight.grad()` / `bias.grad()` were `None` — the
//! affine scale (gamma) and shift (beta) never received gradient, so every
//! transformer using LayerNorm/RMSNorm was NON-FINE-TUNABLE (the norm params
//! could never update).
//!
//! These tests are a self-contained finite-difference gradcheck (no torch dep):
//! perturb each gamma[i] / beta[i] by ±eps, recompute the scalar loss, and
//! compare the central difference (L(+eps) - L(-eps)) / 2eps against the
//! analytic `.grad`. They also assert grad is non-None (the severed-graph
//! guard). The finite-diff comparison genuinely catches a wrong/missing
//! gradient — it is not a tautological `is_some` on a hardcoded value.

use crate::autograd::{self, Tensor};
use crate::nn::normalization::{LayerNorm, RMSNorm};
use crate::nn::Module;

/// Fixed, non-uniform upstream coefficient. Multiplying the norm output by a
/// per-feature constant and summing makes BOTH dL/dgamma and dL/dbeta nonzero
/// (a bare `sum(output)` leaves dL/dgamma ~ sum(x_hat) ~ 0 for LayerNorm, which
/// would not exercise the gamma gradient path).
fn coeff(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.3 + 0.17 * (i as f32)).collect()
}

/// Scalar loss = sum_b sum_i c_i * y[b, i], where y is the norm output.
/// `c` is detached (no grad) so the only graph edges are through gamma/beta/x.
fn scalar_loss(output: &Tensor, c: &[f32]) -> Tensor {
    let shape = output.shape();
    let feat = c.len();
    let batch = output.numel() / feat;
    let mut cvec = Vec::with_capacity(output.numel());
    for _ in 0..batch {
        cvec.extend_from_slice(c);
    }
    let c_tensor = Tensor::new(&cvec, shape);
    output.mul(&c_tensor).sum()
}

/// Re-run forward + loss WITHOUT building a graph, with the affine params
/// (gamma, optional beta) overridden by the supplied perturbed values.
/// Used for the finite-difference reference — pure function of the params.
fn forward_loss_layernorm(x: &Tensor, gamma: &[f32], beta: &[f32], eps: f32, c: &[f32]) -> f32 {
    autograd::no_grad(|| {
        let g = Tensor::new(gamma, &[gamma.len()]);
        let b = Tensor::new(beta, &[beta.len()]);
        let y = crate::nn::functional::layer_norm(x, &g, &b, eps);
        scalar_loss(&y, c).item()
    })
}

fn forward_loss_rmsnorm(x: &Tensor, gamma: &[f32], eps: f32, c: &[f32]) -> f32 {
    autograd::no_grad(|| {
        let g = Tensor::new(gamma, &[gamma.len()]);
        let y = crate::nn::functional::rms_norm(x, &g, eps);
        scalar_loss(&y, c).item()
    })
}

const FD_EPS: f32 = 1e-3;
const TOL: f32 = 2e-2; // f32 central-difference on this composite is ~1e-2 accurate

fn assert_close(analytic: f32, numeric: f32, what: &str) {
    let denom = analytic.abs().max(numeric.abs()).max(1.0);
    let rel = (analytic - numeric).abs() / denom;
    assert!(
        rel < TOL,
        "{what}: analytic grad {analytic} != finite-diff {numeric} (rel err {rel})"
    );
}

#[test]
fn layernorm_backward_flows_grad_to_gamma_and_beta() {
    autograd::clear_graph();

    let feat = 5usize;
    let batch = 3usize;
    let eps = 1e-5f32;
    // Deterministic, non-degenerate input (varying per feature & per row).
    let x_data: Vec<f32> = (0..batch * feat)
        .map(|k| {
            let r = (k / feat) as f32;
            let f = (k % feat) as f32;
            0.5 + 0.4 * f - 0.3 * r + 0.05 * (f * r)
        })
        .collect();
    let x = Tensor::new(&x_data, &[batch, feat]);
    let c = coeff(feat);

    // Non-trivial affine params (not all-ones / all-zeros) so a severed gamma
    // OR beta edge is observable.
    let gamma0: Vec<f32> = (0..feat).map(|i| 1.0 + 0.2 * (i as f32)).collect();
    let beta0: Vec<f32> = (0..feat).map(|i| -0.1 + 0.15 * (i as f32)).collect();

    let mut ln = LayerNorm::with_eps(&[feat], eps);
    ln.set_weight(Tensor::new(&gamma0, &[feat]).requires_grad());
    ln.set_bias(Tensor::new(&beta0, &[feat]).requires_grad());

    let gamma_id = ln.weight().id();
    let beta_id = ln.bias().id();

    let y = ln.forward(&x);
    let loss = scalar_loss(&y, &c);
    loss.backward();

    // Severed-graph guard: grads MUST exist.
    let gamma_grad = autograd::get_grad(gamma_id)
        .expect("LayerNorm gamma (weight) received NO gradient — autograd graph severed");
    let beta_grad = autograd::get_grad(beta_id)
        .expect("LayerNorm beta (bias) received NO gradient — autograd graph severed");

    // Finite-difference gradcheck for every gamma[i] and beta[i].
    for i in 0..feat {
        let mut gp = gamma0.clone();
        gp[i] += FD_EPS;
        let mut gm = gamma0.clone();
        gm[i] -= FD_EPS;
        let num = (forward_loss_layernorm(&x, &gp, &beta0, eps, &c)
            - forward_loss_layernorm(&x, &gm, &beta0, eps, &c))
            / (2.0 * FD_EPS);
        assert_close(
            gamma_grad.data()[i],
            num,
            &format!("LayerNorm dL/dgamma[{i}]"),
        );

        let mut bp = beta0.clone();
        bp[i] += FD_EPS;
        let mut bm = beta0.clone();
        bm[i] -= FD_EPS;
        let num_b = (forward_loss_layernorm(&x, &gamma0, &bp, eps, &c)
            - forward_loss_layernorm(&x, &gamma0, &bm, eps, &c))
            / (2.0 * FD_EPS);
        assert_close(
            beta_grad.data()[i],
            num_b,
            &format!("LayerNorm dL/dbeta[{i}]"),
        );
    }

    // Grads must be finite and at least one component nonzero.
    assert!(gamma_grad.data().iter().all(|g| g.is_finite()));
    assert!(beta_grad.data().iter().all(|g| g.is_finite()));
    assert!(gamma_grad.data().iter().any(|&g| g.abs() > 1e-6));
    assert!(beta_grad.data().iter().any(|&g| g.abs() > 1e-6));
}

#[test]
fn rmsnorm_backward_flows_grad_to_gamma() {
    autograd::clear_graph();

    let feat = 5usize;
    let batch = 3usize;
    let eps = 1e-6f32;
    let x_data: Vec<f32> = (0..batch * feat)
        .map(|k| {
            let r = (k / feat) as f32;
            let f = (k % feat) as f32;
            0.6 + 0.35 * f - 0.25 * r + 0.04 * (f * r)
        })
        .collect();
    let x = Tensor::new(&x_data, &[batch, feat]);
    let c = coeff(feat);

    let gamma0: Vec<f32> = (0..feat).map(|i| 0.9 + 0.18 * (i as f32)).collect();

    let mut rn = RMSNorm::with_eps(&[feat], eps);
    rn.set_weight(Tensor::new(&gamma0, &[feat]).requires_grad());

    let gamma_id = rn.weight().id();

    let y = rn.forward(&x);
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let gamma_grad = autograd::get_grad(gamma_id)
        .expect("RMSNorm gamma (weight) received NO gradient — autograd graph severed");

    for i in 0..feat {
        let mut gp = gamma0.clone();
        gp[i] += FD_EPS;
        let mut gm = gamma0.clone();
        gm[i] -= FD_EPS;
        let num = (forward_loss_rmsnorm(&x, &gp, eps, &c) - forward_loss_rmsnorm(&x, &gm, eps, &c))
            / (2.0 * FD_EPS);
        assert_close(
            gamma_grad.data()[i],
            num,
            &format!("RMSNorm dL/dgamma[{i}]"),
        );
    }

    assert!(gamma_grad.data().iter().all(|g| g.is_finite()));
    assert!(gamma_grad.data().iter().any(|&g| g.abs() > 1e-6));
}

/// Gradient must also flow to the INPUT x (so stacked norm layers train), not
/// just the affine params. Finite-difference check on each x entry.
#[test]
fn layernorm_backward_flows_grad_to_input() {
    autograd::clear_graph();

    let feat = 4usize;
    let eps = 1e-5f32;
    let x_data: Vec<f32> = vec![0.5, -0.2, 1.1, 0.3];
    let gamma0: Vec<f32> = vec![1.0, 1.3, 0.7, 1.1];
    let beta0: Vec<f32> = vec![0.0, 0.2, -0.1, 0.05];
    let c = coeff(feat);

    let x = Tensor::new(&x_data, &[1, feat]).requires_grad();
    let x_id = x.id();

    let g = Tensor::new(&gamma0, &[feat]).requires_grad();
    let b = Tensor::new(&beta0, &[feat]).requires_grad();
    let y = crate::nn::functional::layer_norm(&x, &g, &b, eps);
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let x_grad = autograd::get_grad(x_id).expect("LayerNorm input x received NO gradient");

    let recompute = |xd: &[f32]| -> f32 {
        autograd::no_grad(|| {
            let xx = Tensor::new(xd, &[1, feat]);
            let gg = Tensor::new(&gamma0, &[feat]);
            let bb = Tensor::new(&beta0, &[feat]);
            let yy = crate::nn::functional::layer_norm(&xx, &gg, &bb, eps);
            scalar_loss(&yy, &c).item()
        })
    };

    for i in 0..feat {
        let mut xp = x_data.clone();
        xp[i] += FD_EPS;
        let mut xm = x_data.clone();
        xm[i] -= FD_EPS;
        let num = (recompute(&xp) - recompute(&xm)) / (2.0 * FD_EPS);
        assert_close(x_grad.data()[i], num, &format!("LayerNorm dL/dx[{i}]"));
    }
}

#[test]
fn rmsnorm_backward_flows_grad_to_input() {
    autograd::clear_graph();

    let feat = 4usize;
    let eps = 1e-6f32;
    let x_data: Vec<f32> = vec![0.7, -0.3, 1.2, 0.4];
    let gamma0: Vec<f32> = vec![0.9, 1.2, 0.8, 1.05];
    let c = coeff(feat);

    let x = Tensor::new(&x_data, &[1, feat]).requires_grad();
    let x_id = x.id();

    let g = Tensor::new(&gamma0, &[feat]).requires_grad();
    let y = crate::nn::functional::rms_norm(&x, &g, eps);
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let x_grad = autograd::get_grad(x_id).expect("RMSNorm input x received NO gradient");

    let recompute = |xd: &[f32]| -> f32 {
        autograd::no_grad(|| {
            let xx = Tensor::new(xd, &[1, feat]);
            let gg = Tensor::new(&gamma0, &[feat]);
            let yy = crate::nn::functional::rms_norm(&xx, &gg, eps);
            scalar_loss(&yy, &c).item()
        })
    };

    for i in 0..feat {
        let mut xp = x_data.clone();
        xp[i] += FD_EPS;
        let mut xm = x_data.clone();
        xm[i] -= FD_EPS;
        let num = (recompute(&xp) - recompute(&xm)) / (2.0 * FD_EPS);
        assert_close(x_grad.data()[i], num, &format!("RMSNorm dL/dx[{i}]"));
    }
}
