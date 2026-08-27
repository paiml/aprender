//! Falsifier: `cosine_similarity_rows` MUST reach BOTH inputs with the correct
//! per-input clamp branch, and `mse_loss` MUST return a graph-connected `[1]`
//! tensor rather than an `f32`.
//!
//! Obligations:
//! - OBLIG-ENC-06 pair loss (`setfit-encoder-conformance-v1`, equations
//!   `cosine_similarity_rows` and `mse_loss`)
//! - OBLIG-DETACH-REJECTION (`setfit-encoder-conformance-v1`)
//!
//! BUG CLASS 1 (PF-001, the phase's reason to exist): `nn/loss.rs` and
//! `nn/self_supervised.rs` compute cosine/MSE losses that return `f32`. An
//! `f32` carries no `grad_fn`, so a training loop built on them reports a
//! falling loss while every encoder weight stays frozen. `mse_loss` here
//! returns `Tensor[1]` with a recorded edge, and
//! `mse_loss_returns_a_graph_connected_tensor_not_a_scalar` pins that.
//!
//! BUG CLASS 2 (one-sided edge): a siamese objective differentiates through
//! BOTH branches. A backward that returns a gradient for `a` and drops `b`
//! silently freezes half the encoder and still trains — badly — so every
//! gradcheck here checks both inputs, and the failure message says so.
//!
//! BUG CLASS 3 (one derivative for four branch combinations): the denominator
//! is `max(n_a, eps) * max(n_b, eps)`, with each factor clamped INDEPENDENTLY.
//! Whenever a factor is clamped it becomes a constant and its projection term
//! vanishes — for that input only. Tests below drive the a-clamped, b-clamped
//! and both-clamped combinations and pin each against the closed form, because
//! a finite-difference check that only ever sees well-scaled embeddings visits
//! exactly one of the four.
//!
//! Tolerances here are OP-LEVEL finite-difference tolerances only. Fixture
//! comparison epsilons live in the contract YAML (D-14).

use super::{cosine_similarity_rows, mse_loss, OpError};
use crate::autograd::{self, Tensor};

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

fn assert_close(analytic: f32, numeric: f32, what: &str) {
    let denom = analytic.abs().max(numeric.abs()).max(1.0);
    let rel = (analytic - numeric).abs() / denom;
    assert!(
        rel < TOL,
        "{what}: analytic grad {analytic} != finite-diff {numeric} (rel err {rel})"
    );
}

/// Gradcheck driver for a TWO-input op. Asserts both inputs receive a
/// gradient, that both are finite, and that both match central finite
/// differences at every element. Returns `(grad_a, grad_b)` so a caller can
/// additionally pin them against a closed form — FD proves forward/backward
/// consistency, not that the intended branch was taken.
fn gradcheck_pair<F>(
    name: &str,
    a_data: &[f32],
    b_data: &[f32],
    shape: &[usize],
    fwd: F,
) -> (Vec<f32>, Vec<f32>)
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    autograd::clear_graph();
    let a = Tensor::new(a_data, shape).requires_grad();
    let b = Tensor::new(b_data, shape).requires_grad();
    let (aid, bid) = (a.id(), b.id());
    let y = fwd(&a, &b);
    let c = coeff(y.numel());
    scalar_loss(&y, &c).backward();

    let ga = autograd::get_grad(aid)
        .unwrap_or_else(|| panic!("{name}: input `a` received NO gradient — graph severed"));
    let gb = autograd::get_grad(bid).unwrap_or_else(|| {
        panic!(
            "{name}: input `b` received NO gradient — a one-sided edge silently \
             freezes half of a siamese encoder while still appearing to train"
        )
    });

    for (label, g) in [("a", &ga), ("b", &gb)] {
        assert_eq!(g.shape(), shape, "{name}: grad_{label} shape mismatch");
        assert!(
            g.data().iter().all(|v| v.is_finite()),
            "{name}: non-finite grad_{label}"
        );
        assert!(
            g.data().iter().any(|&v| v.abs() > 1e-9),
            "{name}: all-zero grad_{label}"
        );
    }

    // Central differences over `a`, then over `b`.
    let fd = |perturb_a: bool, idx: usize, delta: f32| -> f32 {
        autograd::no_grad(|| {
            let mut ad = a_data.to_vec();
            let mut bd = b_data.to_vec();
            if perturb_a {
                ad[idx] += delta;
            } else {
                bd[idx] += delta;
            }
            let at = Tensor::new(&ad, shape);
            let bt = Tensor::new(&bd, shape);
            let y = fwd(&at, &bt);
            scalar_loss(&y, &c).item()
        })
    };

    for i in 0..a_data.len() {
        let num = (fd(true, i, FD_EPS) - fd(true, i, -FD_EPS)) / (2.0 * FD_EPS);
        assert_close(ga.data()[i], num, &format!("{name} dL/da[{i}]"));
    }
    for i in 0..b_data.len() {
        let num = (fd(false, i, FD_EPS) - fd(false, i, -FD_EPS)) / (2.0 * FD_EPS);
        assert_close(gb.data()[i], num, &format!("{name} dL/db[{i}]"));
    }

    (ga.data().to_vec(), gb.data().to_vec())
}

// ===========================================================================
// cosine_similarity_rows — forward
// ===========================================================================

#[test]
fn cosine_similarity_rows_of_identical_rows_is_one() {
    let a = Tensor::new(&[3.0, 4.0, 1.0, 0.0], &[2, 2]);
    let s = cosine_similarity_rows(&a, &a, 1e-12).expect("cosine must succeed");

    assert_eq!(s.shape(), &[2], "one similarity per row pair");
    for (i, &v) in s.data().iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-6, "row {i}: expected 1.0, got {v}");
    }
}

#[test]
fn cosine_similarity_rows_of_orthogonal_and_antiparallel_rows() {
    // Row 0: orthogonal -> 0. Row 1: anti-parallel -> -1.
    let a = Tensor::new(&[1.0, 0.0, 0.6, 0.8], &[2, 2]);
    let b = Tensor::new(&[0.0, 1.0, -3.0, -4.0], &[2, 2]);
    let s = cosine_similarity_rows(&a, &b, 1e-12).expect("cosine must succeed");

    assert!(
        s.data()[0].abs() < 1e-6,
        "orthogonal rows: got {}",
        s.data()[0]
    );
    assert!(
        (s.data()[1] + 1.0).abs() < 1e-6,
        "anti-parallel rows: got {}",
        s.data()[1]
    );
}

#[test]
fn cosine_similarity_rows_is_scale_invariant_above_the_clamp() {
    // The defining property of a cosine: magnitude must not matter, only angle.
    let a = Tensor::new(&[0.3, -0.9, 1.7], &[1, 3]);
    let b = Tensor::new(&[1.1, 0.4, -0.25], &[1, 3]);
    let scaled = Tensor::new(&[11.0, 4.0, -2.5], &[1, 3]);

    let s1 = cosine_similarity_rows(&a, &b, 1e-12).expect("cosine must succeed");
    let s2 = cosine_similarity_rows(&a, &scaled, 1e-12).expect("cosine must succeed");
    assert!(
        (s1.data()[0] - s2.data()[0]).abs() < 1e-5,
        "scaling one input by 10 must not change the cosine: {} vs {}",
        s1.data()[0],
        s2.data()[0]
    );
}

#[test]
fn cosine_similarity_rows_stays_within_unit_bounds_in_every_branch() {
    // |<a,b>| <= n_a * n_b <= max(n_a,eps) * max(n_b,eps), so the bound survives
    // BOTH clamp branches. eps = 0.5 clamps row 0 (norm 0.2) and leaves row 1
    // (norm 1.0) alone.
    let a = Tensor::new(&[0.12, 0.16, 0.6, 0.8], &[2, 2]);
    let b = Tensor::new(&[0.12, 0.16, 0.6, 0.8], &[2, 2]);
    let s = cosine_similarity_rows(&a, &b, 0.5).expect("cosine must succeed");

    for (i, &v) in s.data().iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "row {i}: cosine escaped [-1, 1] with value {v}"
        );
    }
    // Row 0 is clamped on BOTH sides: s = 0.04 / (0.5*0.5) = 0.16, not 1.0.
    assert!(
        (s.data()[0] - 0.16).abs() < 1e-5,
        "a doubly-clamped row divides by eps*eps, expected 0.16, got {}",
        s.data()[0]
    );
    assert!((s.data()[1] - 1.0).abs() < 1e-5, "row 1 is unclamped");
}

#[test]
fn cosine_similarity_rows_zero_row_yields_zero_not_nan() {
    let a = Tensor::new(&[0.0, 0.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let s = cosine_similarity_rows(&a, &b, 1e-6).expect("cosine must succeed");

    assert!(
        s.data().iter().all(|v| v.is_finite()),
        "a zero row must not produce NaN, got {:?}",
        s.data()
    );
    assert_eq!(s.data()[0], 0.0, "a zero row has zero similarity");
}

// ===========================================================================
// cosine_similarity_rows — fail-closed rejections
// ===========================================================================

#[test]
fn cosine_similarity_rows_rejects_wrong_rank() {
    let ok = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let rank1 = Tensor::new(&[1.0, 2.0], &[2]);

    assert_eq!(
        cosine_similarity_rows(&rank1, &ok, 1e-6).expect_err("a must be 2-D"),
        OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: vec![2],
        }
    );
    assert_eq!(
        cosine_similarity_rows(&ok, &rank1, 1e-6).expect_err("b must be 2-D"),
        OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: vec![2],
        }
    );
}

#[test]
fn cosine_similarity_rows_rejects_shape_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    assert_eq!(
        cosine_similarity_rows(&a, &b, 1e-6)
            .expect_err("row-wise similarity needs matching shapes"),
        OpError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![1, 3],
        }
    );
}

#[test]
fn cosine_similarity_rows_rejects_zero_dimensions() {
    let zero_batch = Tensor::new(&[], &[0, 4]);
    assert_eq!(
        cosine_similarity_rows(&zero_batch, &zero_batch, 1e-6)
            .expect_err("batch 0 must be rejected"),
        OpError::ZeroDimension { which: "batch" }
    );

    let zero_hidden = Tensor::new(&[], &[3, 0]);
    assert_eq!(
        cosine_similarity_rows(&zero_hidden, &zero_hidden, 1e-6)
            .expect_err("hidden 0 must be rejected"),
        OpError::ZeroDimension { which: "hidden" }
    );
}

#[test]
fn cosine_similarity_rows_rejects_invalid_epsilon() {
    let a = Tensor::new(&[3.0, 4.0], &[1, 2]);

    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            cosine_similarity_rows(&a, &a, bad).expect_err("epsilon must be finite and > 0"),
            OpError::invalid_epsilon(bad),
            "eps = {bad}"
        );
    }
}

#[test]
fn cosine_similarity_rows_rejects_non_finite_input() {
    let good = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let nan = Tensor::new(&[1.0, f32::NAN, 3.0, 4.0], &[2, 2]);
    let inf = Tensor::new(&[1.0, 2.0, f32::NEG_INFINITY, 4.0], &[2, 2]);

    assert_eq!(
        cosine_similarity_rows(&nan, &good, 1e-6).expect_err("NaN in a must be rejected"),
        OpError::NonFiniteInput { position: 1 }
    );
    assert_eq!(
        cosine_similarity_rows(&good, &inf, 1e-6).expect_err("Inf in b must be rejected"),
        OpError::NonFiniteInput { position: 2 }
    );
}

// ===========================================================================
// cosine_similarity_rows — graph connectivity and gradients
// ===========================================================================

#[test]
fn cosine_similarity_rows_is_not_grad_connected_without_requires_grad() {
    autograd::clear_graph();
    let a = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let s = cosine_similarity_rows(&a, &a, 1e-6).expect("cosine must succeed");
    assert!(
        !s.requires_grad_enabled(),
        "comparing two frozen tensors must not fabricate a graph edge"
    );
}

#[test]
fn cosine_similarity_rows_records_the_named_backward_edge() {
    autograd::clear_graph();
    let a = Tensor::new(&[3.0, 4.0], &[1, 2]).requires_grad();
    let b = Tensor::new(&[1.0, 0.0], &[1, 2]);
    let s = cosine_similarity_rows(&a, &b, 1e-6).expect("cosine must succeed");

    assert!(
        s.requires_grad_enabled(),
        "one differentiable input is enough to require grad"
    );
    assert_eq!(
        s.grad_fn().map(|f| f.name()),
        Some("CosineSimilarityBackward")
    );
}

#[test]
fn cosine_similarity_rows_backward_matches_central_finite_differences_for_both_inputs() {
    // Non-degenerate [2,4], every row norm ~O(1) against eps 1e-8: firmly on the
    // projected side for both inputs and both rows.
    let a: Vec<f32> = (0..8)
        .map(|i| 0.31 + 0.23 * (i as f32) - 0.04 * ((i * i) as f32))
        .collect();
    let b: Vec<f32> = (0..8)
        .map(|i| -0.17 + 0.41 * (i as f32) - 0.05 * ((i * i) as f32))
        .collect();
    gradcheck_pair(
        "cosine_similarity_rows above clamp",
        &a,
        &b,
        &[2, 4],
        |x, y| cosine_similarity_rows(x, y, 1e-8).expect("cosine must succeed"),
    );
}

#[test]
fn cosine_similarity_rows_backward_reaches_both_inputs_through_a_shared_graph() {
    // The siamese guard, stated separately from the gradcheck so its failure
    // message is unambiguous.
    autograd::clear_graph();
    let a = Tensor::new(&[0.6, 0.8, 0.1], &[1, 3]).requires_grad();
    let b = Tensor::new(&[0.2, -0.4, 0.9], &[1, 3]).requires_grad();
    let (aid, bid) = (a.id(), b.id());

    let s = cosine_similarity_rows(&a, &b, 1e-8).expect("cosine must succeed");
    s.sum().backward();

    let ga = autograd::get_grad(aid).expect("`a` must receive gradient");
    let gb = autograd::get_grad(bid).expect("`b` must receive gradient");
    assert!(
        ga.data().iter().any(|v| v.abs() > 1e-9),
        "grad_a is entirely zero"
    );
    assert!(
        gb.data().iter().any(|v| v.abs() > 1e-9),
        "grad_b is entirely zero — half the siamese encoder would never train"
    );
}

#[test]
fn cosine_similarity_rows_backward_below_clamp_on_a_drops_only_the_a_projection() {
    // ||a|| = 0.2 (clamped by eps = 0.5), ||b|| = 1.0 (not clamped).
    // Expected: ds/da = b / (eps * n_b), a plain linear map with NO projection.
    //           ds/db keeps ITS projection term, using d_a = eps as the constant.
    let eps = 0.5f32;
    let ad = [0.12f32, 0.16, 0.0];
    let bd = [0.0f32, 0.6, 0.8];

    let (ga, gb) = gradcheck_pair("cosine a-clamped", &ad, &bd, &[1, 3], |x, y| {
        cosine_similarity_rows(x, y, eps).expect("cosine must succeed")
    });

    let c0 = coeff(1)[0];
    let n_b = 1.0f32;
    let s = (ad[0] * bd[0] + ad[1] * bd[1] + ad[2] * bd[2]) / (eps * n_b);

    for i in 0..3 {
        let want = c0 * bd[i] / (eps * n_b);
        assert!(
            (ga[i] - want).abs() < 1e-5,
            "grad_a[{i}]: the clamped side must be the plain linear b/(eps*n_b) = {want}, got {}",
            ga[i]
        );
        // The b side keeps its projection, with d_a pinned to the constant eps.
        let want_b = c0 * (ad[i] / eps - s * bd[i] / n_b) / n_b;
        assert!(
            (gb[i] - want_b).abs() < 1e-5,
            "grad_b[{i}]: the UNclamped side keeps its projection, expected {want_b}, got {}",
            gb[i]
        );
    }

    // What a projected-everywhere implementation would have produced for `a`.
    let n_a = (ad.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let wrong: Vec<f32> = (0..3)
        .map(|i| c0 * (bd[i] / n_b - s * ad[i] / n_a) / n_a)
        .collect();
    assert!(
        (0..3).any(|i| (wrong[i] - ga[i]).abs() > 1e-3),
        "the clamped and projected forms must be DIFFERENT answers here, otherwise \
         this test proves nothing about which branch was taken"
    );
}

#[test]
fn cosine_similarity_rows_backward_below_clamp_on_b_drops_only_the_b_projection() {
    // Mirror image of the previous test: ||a|| = 1.0, ||b|| = 0.2, eps = 0.5.
    // Proves the two branch decisions are genuinely independent rather than one
    // shared flag.
    let eps = 0.5f32;
    let ad = [0.0f32, 0.6, 0.8];
    let bd = [0.12f32, 0.16, 0.0];

    let (ga, gb) = gradcheck_pair("cosine b-clamped", &ad, &bd, &[1, 3], |x, y| {
        cosine_similarity_rows(x, y, eps).expect("cosine must succeed")
    });

    let c0 = coeff(1)[0];
    let n_a = 1.0f32;
    let s = (ad[0] * bd[0] + ad[1] * bd[1] + ad[2] * bd[2]) / (n_a * eps);

    for i in 0..3 {
        let want_b = c0 * ad[i] / (eps * n_a);
        assert!(
            (gb[i] - want_b).abs() < 1e-5,
            "grad_b[{i}]: the clamped side must be the plain linear a/(eps*n_a) = {want_b}, got {}",
            gb[i]
        );
        let want_a = c0 * (bd[i] / eps - s * ad[i] / n_a) / n_a;
        assert!(
            (ga[i] - want_a).abs() < 1e-5,
            "grad_a[{i}]: the UNclamped side keeps its projection, expected {want_a}, got {}",
            ga[i]
        );
    }
}

#[test]
fn cosine_similarity_rows_backward_with_both_inputs_clamped_is_bilinear() {
    // Both norms below eps: the denominator is the constant eps*eps and the map
    // degenerates to the plain bilinear form <a,b>/eps^2. Neither side keeps a
    // projection term, so ds/da = b/eps^2 and ds/db = a/eps^2 exactly.
    let eps = 0.5f32;
    let ad = [0.12f32, 0.16, 0.0];
    let bd = [0.0f32, 0.09, 0.12];

    let (ga, gb) = gradcheck_pair("cosine both-clamped", &ad, &bd, &[1, 3], |x, y| {
        cosine_similarity_rows(x, y, eps).expect("cosine must succeed")
    });

    let c0 = coeff(1)[0];
    let inv = 1.0 / (eps * eps);
    for i in 0..3 {
        assert!(
            (ga[i] - c0 * bd[i] * inv).abs() < 1e-5,
            "grad_a[{i}]: expected b/eps^2 scaled, got {}",
            ga[i]
        );
        assert!(
            (gb[i] - c0 * ad[i] * inv).abs() < 1e-5,
            "grad_b[{i}]: expected a/eps^2 scaled, got {}",
            gb[i]
        );
    }
}

// ===========================================================================
// mse_loss
// ===========================================================================

#[test]
fn mse_loss_matches_hand_computed_mean_square() {
    let pred = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let loss = mse_loss(&pred, &[1.5, 2.0, 1.0]).expect("mse must succeed");

    assert_eq!(loss.shape(), &[1], "the reduction produces a [1] tensor");
    // ((-0.5)^2 + 0^2 + 2^2) / 3 = (0.25 + 0 + 4) / 3
    let want = 4.25f32 / 3.0;
    assert!(
        (loss.item() - want).abs() < 1e-6,
        "expected {want}, got {}",
        loss.item()
    );
}

#[test]
fn mse_loss_is_zero_exactly_when_prediction_equals_target() {
    let pred = Tensor::new(&[0.25, -1.5, 7.0], &[3]);
    let loss = mse_loss(&pred, &[0.25, -1.5, 7.0]).expect("mse must succeed");
    assert_eq!(loss.item(), 0.0, "MSE = 0 iff pred == target elementwise");

    let off = mse_loss(&pred, &[0.25, -1.5, 7.001]).expect("mse must succeed");
    assert!(off.item() > 0.0, "any mismatch must make the loss positive");
}

#[test]
fn mse_loss_returns_a_graph_connected_tensor_not_a_scalar() {
    // The PF-001 guard. `nn/loss.rs` returns f32; an f32 cannot carry a grad_fn,
    // so a loss built on it decreases while the encoder never moves.
    autograd::clear_graph();
    let pred = Tensor::new(&[1.0, 2.0], &[2]).requires_grad();
    let loss = mse_loss(&pred, &[0.0, 0.0]).expect("mse must succeed");

    assert_eq!(loss.shape(), &[1]);
    assert!(loss.requires_grad_enabled(), "the loss must track gradient");
    assert_eq!(loss.grad_fn().map(|f| f.name()), Some("MseBackward"));
}

#[test]
fn mse_loss_is_not_grad_connected_without_requires_grad() {
    autograd::clear_graph();
    let pred = Tensor::new(&[1.0, 2.0], &[2]);
    let loss = mse_loss(&pred, &[0.0, 0.0]).expect("mse must succeed");
    assert!(
        !loss.requires_grad_enabled(),
        "a frozen prediction must not fabricate a graph edge"
    );
}

#[test]
fn mse_loss_backward_matches_the_closed_form_two_over_n() {
    autograd::clear_graph();
    let pd = [1.0f32, 2.0, 3.0];
    let td = [1.5f32, 2.0, 1.0];
    let pred = Tensor::new(&pd, &[3]).requires_grad();
    let pid = pred.id();

    mse_loss(&pred, &td).expect("mse must succeed").backward();
    let g = autograd::get_grad(pid).expect("prediction must receive gradient");

    for i in 0..3 {
        let want = 2.0 * (pd[i] - td[i]) / 3.0;
        assert!(
            (g.data()[i] - want).abs() < 1e-6,
            "dL/dpred[{i}]: expected 2*(p-t)/n = {want}, got {}",
            g.data()[i]
        );
    }
}

#[test]
fn mse_loss_backward_matches_central_finite_differences() {
    let pd: Vec<f32> = (0..5).map(|i| 0.4 + 0.31 * (i as f32)).collect();
    let td: Vec<f32> = (0..5).map(|i| 1.1 - 0.17 * (i as f32)).collect();

    autograd::clear_graph();
    let pred = Tensor::new(&pd, &[5]).requires_grad();
    let pid = pred.id();
    mse_loss(&pred, &td).expect("mse must succeed").backward();
    let g = autograd::get_grad(pid).expect("prediction must receive gradient");

    for i in 0..5 {
        let f = |delta: f32| -> f32 {
            autograd::no_grad(|| {
                let mut p = pd.clone();
                p[i] += delta;
                mse_loss(&Tensor::new(&p, &[5]), &td)
                    .expect("mse must succeed")
                    .item()
            })
        };
        let num = (f(FD_EPS) - f(-FD_EPS)) / (2.0 * FD_EPS);
        assert_close(g.data()[i], num, &format!("mse_loss dL/dpred[{i}]"));
    }
}

#[test]
fn mse_loss_rejects_length_mismatch() {
    let pred = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(
        mse_loss(&pred, &[1.0, 2.0]).expect_err("target length must match"),
        OpError::LengthMismatch { ids: 3, mask: 2 }
    );
}

#[test]
fn mse_loss_rejects_wrong_rank_and_zero_length() {
    let rank2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        mse_loss(&rank2, &[0.0; 4]).expect_err("pred must be 1-D [B]"),
        OpError::ShapeMismatch {
            expected: vec![0],
            got: vec![2, 2],
        }
    );

    let empty = Tensor::new(&[], &[0]);
    assert_eq!(
        mse_loss(&empty, &[]).expect_err("an empty mean has no denominator"),
        OpError::ZeroDimension { which: "batch" }
    );
}

#[test]
fn mse_loss_rejects_non_finite_target() {
    let pred = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(
        mse_loss(&pred, &[0.0, f32::NAN, 1.0]).expect_err("labels are untrusted input"),
        OpError::NonFiniteInput { position: 1 }
    );
    assert_eq!(
        mse_loss(&pred, &[0.0, 1.0, f32::INFINITY]).expect_err("labels are untrusted input"),
        OpError::NonFiniteInput { position: 2 }
    );
}

// ===========================================================================
// Composition — the shape plan 01-07 will build on
// ===========================================================================

#[test]
fn mse_loss_of_cosine_similarity_rows_reaches_both_embedding_inputs() {
    // The full ENC-06 objective in miniature: two [B,H] embedding matrices ->
    // per-row cosine -> MSE against binary labels -> one scalar. Gradient must
    // arrive at BOTH matrices, finite and non-zero.
    autograd::clear_graph();
    let u = Tensor::new(&[0.6, 0.8, 0.1, -0.3, 0.9, 0.2], &[2, 3]).requires_grad();
    let v = Tensor::new(&[0.2, -0.4, 0.9, 0.7, 0.15, -0.6], &[2, 3]).requires_grad();
    let (uid, vid) = (u.id(), v.id());

    let sim = cosine_similarity_rows(&u, &v, 1e-8).expect("cosine must succeed");
    let loss = mse_loss(&sim, &[1.0, 0.0]).expect("mse must succeed");
    assert_eq!(loss.shape(), &[1]);
    assert!(loss.item().is_finite() && loss.item() >= 0.0);
    loss.backward();

    for (label, id) in [("u", uid), ("v", vid)] {
        let g = autograd::get_grad(id).unwrap_or_else(|| {
            panic!("{label} received NO gradient — the composed graph is severed")
        });
        assert_eq!(g.shape(), &[2, 3]);
        assert!(
            g.data().iter().all(|x| x.is_finite()),
            "{label}: non-finite gradient {:?}",
            g.data()
        );
        assert!(
            g.data().iter().any(|x| x.abs() > 1e-9),
            "{label}: gradient is entirely zero through the composed graph"
        );
    }
}

#[test]
fn mse_loss_of_cosine_similarity_rows_matches_central_finite_differences() {
    // The composition, gradchecked end to end — the two ops could each be
    // self-consistent and still be wired together wrongly.
    let ud = [0.6f32, 0.8, 0.1, -0.3, 0.9, 0.2];
    let vd = [0.2f32, -0.4, 0.9, 0.7, 0.15, -0.6];
    let labels = [1.0f32, 0.0];

    autograd::clear_graph();
    let u = Tensor::new(&ud, &[2, 3]).requires_grad();
    let v = Tensor::new(&vd, &[2, 3]).requires_grad();
    let (uid, vid) = (u.id(), v.id());
    let sim = cosine_similarity_rows(&u, &v, 1e-8).expect("cosine must succeed");
    mse_loss(&sim, &labels)
        .expect("mse must succeed")
        .backward();
    let gu = autograd::get_grad(uid).expect("u must receive gradient");
    let gv = autograd::get_grad(vid).expect("v must receive gradient");

    let eval = |uu: &[f32], vv: &[f32]| -> f32 {
        autograd::no_grad(|| {
            let s =
                cosine_similarity_rows(&Tensor::new(uu, &[2, 3]), &Tensor::new(vv, &[2, 3]), 1e-8)
                    .expect("cosine must succeed");
            mse_loss(&s, &labels).expect("mse must succeed").item()
        })
    };

    for i in 0..6 {
        let mut up = ud;
        up[i] += FD_EPS;
        let mut um = ud;
        um[i] -= FD_EPS;
        let num = (eval(&up, &vd) - eval(&um, &vd)) / (2.0 * FD_EPS);
        assert_close(gu.data()[i], num, &format!("composed dL/du[{i}]"));

        let mut vp = vd;
        vp[i] += FD_EPS;
        let mut vm = vd;
        vm[i] -= FD_EPS;
        let num = (eval(&ud, &vp) - eval(&ud, &vm)) / (2.0 * FD_EPS);
        assert_close(gv.data()[i], num, &format!("composed dL/dv[{i}]"));
    }
}
