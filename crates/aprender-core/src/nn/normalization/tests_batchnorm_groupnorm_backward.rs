//! Falsifier: BatchNorm1d (train mode) + GroupNorm backward MUST flow gradient
//! to their affine params (and input x).
//!
//! Obligations:
//! - OBLIG-BATCHNORM1D-BACKWARD-GRAD-FLOW
//! - OBLIG-GROUPNORM-BACKWARD-GRAD-FLOW
//!
//! BUG (PMAT-911, sibling of PMAT-907): `BatchNorm1d::forward` and
//! `GroupNorm::forward` built their output via `Tensor::new`, which severs the
//! autograd graph. After `loss.backward()`, `weight.grad()` / `bias.grad()`
//! were `None` — the affine scale (gamma) and shift (beta) never received
//! gradient, so every BatchNorm/GroupNorm layer was NON-FINE-TUNABLE.
//!
//! These tests are a self-contained central finite-difference gradcheck (no
//! torch dep): perturb each gamma[i] / beta[i] by ±eps, recompute the scalar
//! loss, and compare the central difference against the analytic `.grad`. They
//! also assert grad is non-None (the severed-graph guard). The finite-diff
//! comparison genuinely catches a wrong/missing gradient — it is NOT a
//! tautological `is_some` on a hardcoded value.
//!
//! BatchNorm1d is tested in TRAIN mode: the backward differentiates through the
//! BATCH statistics (the standard, trickiest batchnorm-backward), matching the
//! biased (÷N) variance the forward uses for normalization.

use crate::autograd::{self, Tensor};
use crate::nn::normalization::{BatchNorm1d, GroupNorm};
use crate::nn::Module;

/// Fixed, non-uniform upstream coefficient so BOTH dL/dgamma and dL/dbeta are
/// nonzero (a bare `sum(output)` leaves dL/dgamma ~ sum(x_hat) ~ 0).
fn coeff(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.3 + 0.17 * (i as f32)).collect()
}

const FD_EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

fn assert_close(analytic: f32, numeric: f32, what: &str) {
    let denom = analytic.abs().max(numeric.abs()).max(1.0);
    let rel = (analytic - numeric).abs() / denom;
    assert!(
        rel < TOL,
        "{what}: analytic grad {analytic} != finite-diff {numeric} (rel err {rel})"
    );
}

// ============================================================================
// BatchNorm1d (TRAIN mode)
// ============================================================================

/// Per-ELEMENT (per [b, j]) detached loss coefficient. CRITICAL: the
/// coefficient must vary across the BATCH reduction (rows), not only across
/// features. dL/dgamma_j = sum_b c[b,j] * x_hat[b,j]; with a feature-constant
/// c, sum_b x_hat[b,j] == 0 (batch-normalized) ⟹ dgamma == 0 and the gamma
/// gradcheck would be vacuous (a scale-by-1.5 mutation would survive). A
/// row-varying coefficient makes dgamma genuinely nonzero.
fn batch_coeff(batch: usize, feat: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(batch * feat);
    for b in 0..batch {
        for j in 0..feat {
            v.push(0.3 + 0.17 * (j as f32) + 0.23 * (b as f32) - 0.07 * ((b * j) as f32));
        }
    }
    v
}

/// Scalar loss = sum_{b,j} cvec[b,j] * y[b, j] over a 2D output `[N, C]`.
/// `cvec` is detached so the only graph edges are through gamma/beta/x.
fn scalar_loss_2d(output: &Tensor, cvec: &[f32]) -> Tensor {
    let c_tensor = Tensor::new(cvec, output.shape());
    output.mul(&c_tensor).sum()
}

/// Re-run BatchNorm1d (train mode) forward + loss WITHOUT a graph, with the
/// affine params overridden. Pure function of (gamma, beta) — the finite-diff
/// reference. A FRESH layer each call keeps running-stat buffers irrelevant
/// (train-mode normalization uses batch stats, not running stats).
fn forward_loss_batchnorm(x: &Tensor, gamma: &[f32], beta: &[f32], eps: f32, c: &[f32]) -> f32 {
    autograd::no_grad(|| {
        let mut bn = BatchNorm1d::new(gamma.len()).with_eps(eps);
        bn.set_weight(Tensor::new(gamma, &[gamma.len()]));
        bn.set_bias(Tensor::new(beta, &[beta.len()]));
        let y = bn.forward(x);
        scalar_loss_2d(&y, c).item()
    })
}

#[test]
fn batchnorm1d_train_backward_flows_grad_to_gamma_and_beta() {
    autograd::clear_graph();

    let feat = 4usize;
    let batch = 5usize; // N>1 so batch variance is well-defined
    let eps = 1e-5f32;
    let x_data: Vec<f32> = (0..batch * feat)
        .map(|k| {
            let r = (k / feat) as f32;
            let f = (k % feat) as f32;
            0.5 + 0.4 * f - 0.3 * r + 0.05 * (f * r)
        })
        .collect();
    let x = Tensor::new(&x_data, &[batch, feat]);
    let c = batch_coeff(batch, feat);

    let gamma0: Vec<f32> = (0..feat).map(|i| 1.0 + 0.2 * (i as f32)).collect();
    let beta0: Vec<f32> = (0..feat).map(|i| -0.1 + 0.15 * (i as f32)).collect();

    let mut bn = BatchNorm1d::new(feat).with_eps(eps);
    bn.set_weight(Tensor::new(&gamma0, &[feat]).requires_grad());
    bn.set_bias(Tensor::new(&beta0, &[feat]).requires_grad());
    assert!(bn.training(), "must be train mode for batch-stat backward");

    let gamma_id = bn.weight().id();
    let beta_id = bn.bias().id();

    let y = bn.forward(&x);
    let loss = scalar_loss_2d(&y, &c);
    loss.backward();

    let gamma_grad = autograd::get_grad(gamma_id)
        .expect("BatchNorm1d gamma (weight) received NO gradient — autograd graph severed");
    let beta_grad = autograd::get_grad(beta_id)
        .expect("BatchNorm1d beta (bias) received NO gradient — autograd graph severed");

    for i in 0..feat {
        let mut gp = gamma0.clone();
        gp[i] += FD_EPS;
        let mut gm = gamma0.clone();
        gm[i] -= FD_EPS;
        let num = (forward_loss_batchnorm(&x, &gp, &beta0, eps, &c)
            - forward_loss_batchnorm(&x, &gm, &beta0, eps, &c))
            / (2.0 * FD_EPS);
        assert_close(
            gamma_grad.data()[i],
            num,
            &format!("BatchNorm1d dL/dgamma[{i}]"),
        );

        let mut bp = beta0.clone();
        bp[i] += FD_EPS;
        let mut bm = beta0.clone();
        bm[i] -= FD_EPS;
        let num_b = (forward_loss_batchnorm(&x, &gamma0, &bp, eps, &c)
            - forward_loss_batchnorm(&x, &gamma0, &bm, eps, &c))
            / (2.0 * FD_EPS);
        assert_close(
            beta_grad.data()[i],
            num_b,
            &format!("BatchNorm1d dL/dbeta[{i}]"),
        );
    }

    assert!(gamma_grad.data().iter().all(|g| g.is_finite()));
    assert!(beta_grad.data().iter().all(|g| g.is_finite()));
    assert!(gamma_grad.data().iter().any(|&g| g.abs() > 1e-6));
    assert!(beta_grad.data().iter().any(|&g| g.abs() > 1e-6));
}

#[test]
fn batchnorm1d_train_backward_flows_grad_to_input() {
    autograd::clear_graph();

    let feat = 3usize;
    let batch = 4usize;
    let eps = 1e-5f32;
    let x_data: Vec<f32> = vec![
        0.5, -0.2, 1.1, 0.3, 0.7, -0.4, -0.1, 0.9, 0.2, 0.6, -0.3, 0.4,
    ];
    let gamma0: Vec<f32> = vec![1.0, 1.3, 0.7];
    let beta0: Vec<f32> = vec![0.0, 0.2, -0.1];
    let c = batch_coeff(batch, feat);

    let x = Tensor::new(&x_data, &[batch, feat]).requires_grad();
    let x_id = x.id();

    let mut bn = BatchNorm1d::new(feat).with_eps(eps);
    bn.set_weight(Tensor::new(&gamma0, &[feat]).requires_grad());
    bn.set_bias(Tensor::new(&beta0, &[feat]).requires_grad());

    let y = bn.forward(&x);
    let loss = scalar_loss_2d(&y, &c);
    loss.backward();

    let x_grad = autograd::get_grad(x_id).expect("BatchNorm1d input x received NO gradient");

    let recompute = |xd: &[f32]| -> f32 {
        autograd::no_grad(|| {
            let xx = Tensor::new(xd, &[batch, feat]);
            forward_loss_batchnorm(&xx, &gamma0, &beta0, eps, &c)
        })
    };

    for i in 0..(batch * feat) {
        let mut xp = x_data.clone();
        xp[i] += FD_EPS;
        let mut xm = x_data.clone();
        xm[i] -= FD_EPS;
        let num = (recompute(&xp) - recompute(&xm)) / (2.0 * FD_EPS);
        assert_close(x_grad.data()[i], num, &format!("BatchNorm1d dL/dx[{i}]"));
    }
}

// ============================================================================
// GroupNorm
// ============================================================================

/// Scalar loss for a GroupNorm output `[N, C, *]`. Coefficient is per-channel
/// (broadcast across spatial), detached.
fn scalar_loss_groupnorm(output: &Tensor, c_per_channel: &[f32], spatial: usize) -> Tensor {
    let shape = output.shape();
    let batch = shape[0];
    let channels = shape[1];
    let mut cvec = Vec::with_capacity(output.numel());
    for _ in 0..batch {
        for &cc in c_per_channel.iter().take(channels) {
            for _ in 0..spatial {
                cvec.push(cc);
            }
        }
    }
    let c_tensor = Tensor::new(&cvec, shape);
    output.mul(&c_tensor).sum()
}

fn forward_loss_groupnorm(
    x: &Tensor,
    num_groups: usize,
    channels: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    c: &[f32],
    spatial: usize,
) -> f32 {
    autograd::no_grad(|| {
        let mut gn = GroupNorm::with_eps(num_groups, channels, eps);
        gn.set_weight(Tensor::new(gamma, &[channels]));
        gn.set_bias(Tensor::new(beta, &[channels]));
        let y = gn.forward(x);
        scalar_loss_groupnorm(&y, c, spatial).item()
    })
}

#[test]
fn groupnorm_backward_flows_grad_to_gamma_and_beta() {
    autograd::clear_graph();

    let num_groups = 2usize;
    let channels = 4usize;
    let spatial = 3usize; // [N, C, L]
    let batch = 2usize;
    let eps = 1e-5f32;
    let total = batch * channels * spatial;
    let x_data: Vec<f32> = (0..total)
        .map(|k| 0.4 + 0.31 * (k as f32) - 0.05 * ((k * k % 7) as f32))
        .collect();
    let x = Tensor::new(&x_data, &[batch, channels, spatial]);
    let c = coeff(channels);

    let gamma0: Vec<f32> = (0..channels).map(|i| 1.0 + 0.2 * (i as f32)).collect();
    let beta0: Vec<f32> = (0..channels).map(|i| -0.1 + 0.15 * (i as f32)).collect();

    let mut gn = GroupNorm::with_eps(num_groups, channels, eps);
    gn.set_weight(Tensor::new(&gamma0, &[channels]).requires_grad());
    gn.set_bias(Tensor::new(&beta0, &[channels]).requires_grad());

    let gamma_id = gn.weight().id();
    let beta_id = gn.bias().id();

    let y = gn.forward(&x);
    let loss = scalar_loss_groupnorm(&y, &c, spatial);
    loss.backward();

    let gamma_grad = autograd::get_grad(gamma_id)
        .expect("GroupNorm gamma (weight) received NO gradient — autograd graph severed");
    let beta_grad = autograd::get_grad(beta_id)
        .expect("GroupNorm beta (bias) received NO gradient — autograd graph severed");

    for i in 0..channels {
        let mut gp = gamma0.clone();
        gp[i] += FD_EPS;
        let mut gm = gamma0.clone();
        gm[i] -= FD_EPS;
        let num = (forward_loss_groupnorm(&x, num_groups, channels, &gp, &beta0, eps, &c, spatial)
            - forward_loss_groupnorm(&x, num_groups, channels, &gm, &beta0, eps, &c, spatial))
            / (2.0 * FD_EPS);
        assert_close(
            gamma_grad.data()[i],
            num,
            &format!("GroupNorm dL/dgamma[{i}]"),
        );

        let mut bp = beta0.clone();
        bp[i] += FD_EPS;
        let mut bm = beta0.clone();
        bm[i] -= FD_EPS;
        let num_b =
            (forward_loss_groupnorm(&x, num_groups, channels, &gamma0, &bp, eps, &c, spatial)
                - forward_loss_groupnorm(&x, num_groups, channels, &gamma0, &bm, eps, &c, spatial))
                / (2.0 * FD_EPS);
        assert_close(
            beta_grad.data()[i],
            num_b,
            &format!("GroupNorm dL/dbeta[{i}]"),
        );
    }

    assert!(gamma_grad.data().iter().all(|g| g.is_finite()));
    assert!(beta_grad.data().iter().all(|g| g.is_finite()));
    assert!(gamma_grad.data().iter().any(|&g| g.abs() > 1e-6));
    assert!(beta_grad.data().iter().any(|&g| g.abs() > 1e-6));
}

#[test]
fn groupnorm_backward_flows_grad_to_input() {
    autograd::clear_graph();

    let num_groups = 2usize;
    let channels = 4usize;
    let spatial = 2usize;
    let batch = 1usize;
    let eps = 1e-5f32;
    let total = batch * channels * spatial;
    let x_data: Vec<f32> = (0..total)
        .map(|k| 0.3 + 0.27 * (k as f32) - 0.04 * ((k * 3 % 5) as f32))
        .collect();
    let gamma0: Vec<f32> = vec![1.0, 1.3, 0.7, 1.1];
    let beta0: Vec<f32> = vec![0.0, 0.2, -0.1, 0.05];
    let c = coeff(channels);

    let x = Tensor::new(&x_data, &[batch, channels, spatial]).requires_grad();
    let x_id = x.id();

    let mut gn = GroupNorm::with_eps(num_groups, channels, eps);
    gn.set_weight(Tensor::new(&gamma0, &[channels]).requires_grad());
    gn.set_bias(Tensor::new(&beta0, &[channels]).requires_grad());

    let y = gn.forward(&x);
    let loss = scalar_loss_groupnorm(&y, &c, spatial);
    loss.backward();

    let x_grad = autograd::get_grad(x_id).expect("GroupNorm input x received NO gradient");

    let recompute = |xd: &[f32]| -> f32 {
        autograd::no_grad(|| {
            let xx = Tensor::new(xd, &[batch, channels, spatial]);
            forward_loss_groupnorm(&xx, num_groups, channels, &gamma0, &beta0, eps, &c, spatial)
        })
    };

    for i in 0..total {
        let mut xp = x_data.clone();
        xp[i] += FD_EPS;
        let mut xm = x_data.clone();
        xm[i] -= FD_EPS;
        let num = (recompute(&xp) - recompute(&xm)) / (2.0 * FD_EPS);
        assert_close(x_grad.data()[i], num, &format!("GroupNorm dL/dx[{i}]"));
    }
}
