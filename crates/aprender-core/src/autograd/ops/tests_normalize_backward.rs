//! Falsifier: `l2_normalize_rows` MUST be graph-connected, MUST fail closed on
//! hostile input, and MUST take the CORRECT side of the epsilon clamp in the
//! backward as well as the forward.
//!
//! Obligations:
//! - OBLIG-ENC-03 L2 normalization (`setfit-encoder-conformance-v1`,
//!   equation `l2_normalize_rows`)
//! - OBLIG-DETACH-REJECTION (`setfit-encoder-conformance-v1`)
//!
//! BUG CLASS 1 (PMAT-913 severed graph): a normalization forward built with a
//! bare `Tensor::from_vec` and no adjacent `set_grad_fn` blocks gradient from
//! reaching every encoder weight upstream, while shapes and loss values stay
//! entirely plausible. `l2_normalize_rows` sits between the pooler and the loss
//! on the ENC-03 path, so severing it freezes the WHOLE encoder.
//!
//! BUG CLASS 2 (one derivative for two functions) — the reason this file exists.
//! `y = x / max(||x||, eps)` is piecewise:
//!
//! ```text
//! n >  eps :  dy/dx = (I - y yᵀ) / n     # d = n depends on x
//! n <= eps :  dy/dx = I / eps            # d is CONSTANT — no projection term
//! ```
//!
//! Applying the projected form below the clamp is not an approximation; it is
//! the derivative of a function that is not being evaluated. It is also
//! **invisible to a finite-difference test that never visits the clamped
//! branch** — which is every FD test written against well-scaled embeddings.
//! Hence a dedicated below-clamp gradcheck, a dedicated boundary test, and a
//! mixed-branch batch where the two rows must take DIFFERENT branches inside a
//! single call.
//!
//! Tolerances here are OP-LEVEL finite-difference tolerances only. Fixture
//! comparison epsilons live in the contract YAML (D-14).

use super::{l2_normalize_rows, OpError};
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

/// Run forward + loss WITHOUT a graph, with the input value at `flat_idx`
/// perturbed by `delta`. Used for the finite-difference reference.
fn perturbed_loss<F>(
    x_data: &[f32],
    x_shape: &[usize],
    flat_idx: usize,
    delta: f32,
    fwd: &F,
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

/// Gradcheck driver: forward `fwd(x)`, build a scalar loss, backward, then
/// assert the input grad is present, finite, non-zero, and equal to a central
/// finite difference at EVERY element.
///
/// Returns the analytic gradient so a caller can additionally pin it against a
/// closed form — FD alone proves consistency between forward and backward, not
/// that either takes the intended branch.
fn gradcheck_input<F>(name: &str, x_data: &[f32], x_shape: &[usize], fwd: F) -> Vec<f32>
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

    grad.data().to_vec()
}

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

#[test]
fn l2_normalize_rows_matches_hand_computed_unit_rows() {
    // Both rows are the SAME direction at different magnitudes, so a correct
    // row-wise normalization collapses them onto the identical unit vector.
    let x = Tensor::new(&[3.0, 4.0, 0.6, 0.8], &[2, 2]);
    let y = l2_normalize_rows(&x, 1e-12).expect("normalize must succeed");

    assert_eq!(y.shape(), &[2, 2], "normalization is shape-preserving");
    let want = [0.6f32, 0.8, 0.6, 0.8];
    for (i, &w) in want.iter().enumerate() {
        assert!(
            (y.data()[i] - w).abs() < 1e-6,
            "element {i}: got {}, want {w}",
            y.data()[i]
        );
    }
}

#[test]
fn l2_normalize_rows_output_rows_have_unit_norm_above_the_clamp() {
    // Non-degenerate [3,4]; every row norm is far above eps, so every output row
    // must be exactly unit length up to f32 rounding.
    let x: Vec<f32> = (0..12)
        .map(|i| 0.41 + 0.27 * (i as f32) - 0.031 * ((i * i) as f32))
        .collect();
    let y = l2_normalize_rows(&Tensor::new(&x, &[3, 4]), 1e-12).expect("normalize must succeed");

    for b in 0..3 {
        let n: f32 = y.data()[b * 4..b * 4 + 4]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (n - 1.0).abs() < 1e-6,
            "row {b} must have unit L2 norm, got {n}"
        );
    }
}

#[test]
fn l2_normalize_rows_below_the_clamp_divides_by_eps_not_by_the_norm() {
    // ||x_row|| = 0.2 with eps = 0.5, so the denominator is the CONSTANT eps.
    // Dividing by the norm instead would return a unit row — the single most
    // likely way to get this wrong, and the assertion below separates them.
    let x = Tensor::new(&[0.12, 0.16, 0.0, 0.0], &[1, 4]);
    let y = l2_normalize_rows(&x, 0.5).expect("normalize must succeed");

    let want = [0.24f32, 0.32, 0.0, 0.0]; // x / 0.5
    for (i, &w) in want.iter().enumerate() {
        assert!(
            (y.data()[i] - w).abs() < 1e-6,
            "element {i}: below the clamp the divisor is eps, got {} want {w}",
            y.data()[i]
        );
    }

    let n: f32 = y.data().iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (n - 0.4).abs() < 1e-6,
        "a clamped row is NOT renormalized to unit length; norm should be ||x||/eps = 0.4, got {n}"
    );
}

#[test]
fn l2_normalize_rows_zero_row_yields_a_zero_row_not_nan() {
    // The reason the epsilon floor exists at all.
    let x = Tensor::new(&[0.0, 0.0, 0.0, 3.0, 4.0, 0.0], &[2, 3]);
    let y = l2_normalize_rows(&x, 1e-6).expect("normalize must succeed");

    assert!(
        y.data().iter().all(|v| v.is_finite()),
        "an all-zero row must not produce NaN, got {:?}",
        y.data()
    );
    for i in 0..3 {
        assert_eq!(
            y.data()[i],
            0.0,
            "zero row stays exactly zero at element {i}"
        );
    }
    // The other row is unaffected by its neighbour.
    assert!((y.data()[3] - 0.6).abs() < 1e-6);
    assert!((y.data()[4] - 0.8).abs() < 1e-6);
}

#[test]
fn l2_normalize_rows_stays_finite_at_extreme_underflow_below_the_clamp() {
    // Row norm 2e-20 against eps 1e-6. Two things are being pinned:
    //   * the sum of squares must not underflow (1e-40 is subnormal in f32),
    //   * the BACKWARD must use I/eps. Applying the projected form here would
    //     divide by 2e-20 and produce ~1e19-scale gradients that are still
    //     FINITE in f32 — so a finiteness assertion alone would not catch it.
    //     The closed-form comparison below does.
    autograd::clear_graph();
    let x = Tensor::new(&[1e-20, 1e-20, 1e-20, 1e-20], &[1, 4]).requires_grad();
    let xid = x.id();
    let eps = 1e-6f32;

    let y = l2_normalize_rows(&x, eps).expect("normalize must succeed");
    assert!(
        y.data().iter().all(|v| v.is_finite()),
        "underflowing row must stay finite, got {:?}",
        y.data()
    );
    for (i, &v) in y.data().iter().enumerate() {
        assert!(
            (v - 1e-14).abs() < 1e-18,
            "element {i}: expected x/eps = 1e-14, got {v}"
        );
    }

    let c = coeff(4);
    scalar_loss(&y, &c).backward();
    let g = autograd::get_grad(xid).expect("input must receive gradient");
    for (i, &want) in c.iter().enumerate() {
        let expect = want / eps;
        let rel = (g.data()[i] - expect).abs() / expect.abs();
        assert!(
            rel < 1e-5,
            "element {i}: clamped branch must give c/eps = {expect}, got {} \
             (the projected form would give ~{:e} here)",
            g.data()[i],
            want / 2e-20
        );
    }
}

// ---------------------------------------------------------------------------
// Fail-closed rejections
// ---------------------------------------------------------------------------

#[test]
fn l2_normalize_rows_rejects_wrong_rank() {
    let rank1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(
        l2_normalize_rows(&rank1, 1e-6).expect_err("x must be 2-D [B,H]"),
        OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: vec![3],
        }
    );

    let rank3 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    assert_eq!(
        l2_normalize_rows(&rank3, 1e-6).expect_err("x must be 2-D [B,H]"),
        OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: vec![1, 2, 2],
        }
    );
}

#[test]
fn l2_normalize_rows_rejects_zero_dimensions() {
    let zero_batch = Tensor::new(&[], &[0, 4]);
    assert_eq!(
        l2_normalize_rows(&zero_batch, 1e-6).expect_err("batch 0 must be rejected"),
        OpError::ZeroDimension { which: "batch" }
    );

    let zero_hidden = Tensor::new(&[], &[3, 0]);
    assert_eq!(
        l2_normalize_rows(&zero_hidden, 1e-6).expect_err("hidden 0 must be rejected"),
        OpError::ZeroDimension { which: "hidden" }
    );
}

#[test]
fn l2_normalize_rows_rejects_non_positive_epsilon() {
    let x = Tensor::new(&[3.0, 4.0], &[1, 2]);

    for bad in [0.0f32, -1e-6, -1.0] {
        let err = l2_normalize_rows(&x, bad)
            .expect_err("a non-positive epsilon removes the divide-by-zero guard");
        assert_eq!(err, OpError::invalid_epsilon(bad), "eps = {bad}");
        assert_eq!(
            err.epsilon(),
            Some(bad),
            "the offending value is recoverable"
        );
    }
}

#[test]
fn l2_normalize_rows_rejects_non_finite_epsilon() {
    let x = Tensor::new(&[3.0, 4.0], &[1, 2]);

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err =
            l2_normalize_rows(&x, bad).expect_err("a non-finite epsilon is not a usable floor");
        // Compared by BITS: `NaN != NaN` under PartialEq, so an f32-carrying
        // variant would make this assertion unsatisfiable for the NaN case.
        assert_eq!(err, OpError::invalid_epsilon(bad), "eps = {bad}");
    }
}

#[test]
fn l2_normalize_rows_rejects_non_finite_input() {
    let nan = Tensor::new(&[1.0, f32::NAN, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        l2_normalize_rows(&nan, 1e-6).expect_err("NaN input must be rejected at the boundary"),
        OpError::NonFiniteInput { position: 1 }
    );

    let inf = Tensor::new(&[1.0, 2.0, f32::INFINITY, 4.0], &[2, 2]);
    assert_eq!(
        l2_normalize_rows(&inf, 1e-6).expect_err("Inf input must be rejected at the boundary"),
        OpError::NonFiniteInput { position: 2 }
    );
}

// ---------------------------------------------------------------------------
// Graph connectivity
// ---------------------------------------------------------------------------

#[test]
fn l2_normalize_rows_is_not_grad_connected_without_requires_grad() {
    autograd::clear_graph();
    let x = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let y = l2_normalize_rows(&x, 1e-6).expect("normalize must succeed");
    assert!(
        !y.requires_grad_enabled(),
        "normalizing a frozen tensor must not fabricate a graph edge"
    );
}

#[test]
fn l2_normalize_rows_records_the_named_backward_edge() {
    autograd::clear_graph();
    let x = Tensor::new(&[3.0, 4.0], &[1, 2]).requires_grad();
    let y = l2_normalize_rows(&x, 1e-6).expect("normalize must succeed");

    assert!(y.requires_grad_enabled(), "output must track gradient");
    assert_eq!(
        y.grad_fn().map(|f| f.name()),
        Some("L2NormalizeRowsBackward"),
        "the edge must be the named backward, not some borrowed neighbour"
    );
}

// ---------------------------------------------------------------------------
// Gradient — ABOVE the clamp (the projected form)
// ---------------------------------------------------------------------------

#[test]
fn l2_normalize_rows_backward_matches_central_finite_differences_above_the_clamp() {
    // Non-degenerate [3,5], every row norm ~O(1) and eps 1e-8, so every row is
    // firmly on the projected side of the clamp and stays there under a 1e-3
    // perturbation.
    let x: Vec<f32> = (0..15)
        .map(|i| 0.29 + 0.17 * (i as f32) - 0.021 * ((i * i) as f32))
        .collect();
    gradcheck_input("l2_normalize_rows above clamp", &x, &[3, 5], |t| {
        l2_normalize_rows(t, 1e-8).expect("normalize must succeed")
    });
}

#[test]
fn l2_normalize_rows_backward_is_orthogonal_to_the_row_above_the_clamp() {
    // A structural consequence of the projected form that the clamped form does
    // NOT have: (I - y yᵀ) annihilates the row direction, so <dL/dx, x> == 0
    // for any upstream gradient. Below the clamp the derivative is I/eps and
    // this inner product is generally non-zero — so this is a second, entirely
    // independent, discriminator between the two branches.
    autograd::clear_graph();
    let xd = [0.7f32, -1.3, 0.45, 2.1];
    let x = Tensor::new(&xd, &[1, 4]).requires_grad();
    let xid = x.id();

    let y = l2_normalize_rows(&x, 1e-8).expect("normalize must succeed");
    scalar_loss(&y, &coeff(4)).backward();
    let g = autograd::get_grad(xid).expect("input must receive gradient");

    let dot: f32 = g.data().iter().zip(xd.iter()).map(|(a, b)| a * b).sum();
    assert!(
        dot.abs() < 1e-4,
        "above the clamp the gradient must be orthogonal to the row, got <g,x> = {dot}"
    );
}

// ---------------------------------------------------------------------------
// Gradient — BELOW the clamp (I / eps, NOT the projected form)
// ---------------------------------------------------------------------------

#[test]
fn l2_normalize_rows_backward_below_epsilon_clamp_matches_central_finite_differences() {
    // eps = 0.5 with row norms ~0.2: a 1e-3 FD perturbation cannot push any row
    // across the boundary, so the finite differences measure the clamped branch
    // itself rather than an average of the two.
    let x: Vec<f32> = (0..8).map(|i| 0.03 + 0.019 * (i as f32)).collect();
    let grad = gradcheck_input("l2_normalize_rows below clamp", &x, &[2, 4], |t| {
        l2_normalize_rows(t, 0.5).expect("normalize must succeed")
    });

    // FD proves forward/backward consistency. This pins the branch: below the
    // clamp the map is the plain scaling x -> x/eps, so dL/dx == c/eps exactly.
    let c = coeff(8);
    for (i, &ci) in c.iter().enumerate() {
        let want = ci / 0.5;
        assert!(
            (grad[i] - want).abs() < 1e-5,
            "element {i}: clamped branch must give c/eps = {want}, got {}",
            grad[i]
        );
    }
}

#[test]
fn l2_normalize_rows_backward_below_epsilon_clamp_is_identity_over_eps_not_the_projected_form() {
    // The direct falsifier for "one derivative reused for two functions".
    // Row norm 0.2, eps 0.5 -> clamped. The projected form is computed here in
    // the test and asserted to be a DIFFERENT answer, so this cannot pass by
    // coincidence.
    autograd::clear_graph();
    let xd = [0.12f32, 0.16, 0.0];
    let x = Tensor::new(&xd, &[1, 3]).requires_grad();
    let xid = x.id();
    let eps = 0.5f32;

    let y = l2_normalize_rows(&x, eps).expect("normalize must succeed");
    let c = coeff(3);
    scalar_loss(&y, &c).backward();
    let g = autograd::get_grad(xid).expect("input must receive gradient");

    // Analytic clamped form.
    for (i, &ci) in c.iter().enumerate() {
        let want = ci / eps;
        assert!(
            (g.data()[i] - want).abs() < 1e-5,
            "element {i}: expected c/eps = {want}, got {}",
            g.data()[i]
        );
    }

    // What the WRONG (projected-everywhere) implementation would have produced.
    let n: f32 = xd.iter().map(|v| v * v).sum::<f32>().sqrt();
    let yv: Vec<f32> = y.data().to_vec();
    let dot: f32 = c.iter().zip(yv.iter()).map(|(a, b)| a * b).sum();
    let mut differs = false;
    for i in 0..3 {
        let projected = (c[i] - yv[i] * dot) / n;
        if (projected - g.data()[i]).abs() > 1e-3 {
            differs = true;
        }
    }
    assert!(
        differs,
        "the projected form and the clamped form must be DIFFERENT answers here, \
         otherwise this test proves nothing about which branch was taken"
    );
}

#[test]
fn l2_normalize_rows_backward_takes_each_row_branch_independently() {
    // Row 0 is ABOVE the clamp (norm 1.0 > 0.5), row 1 is BELOW it (norm 0.1).
    // A single global branch decision — the natural way to get this wrong once
    // the clamp is hoisted out of the row loop — fails one row or the other.
    let x = [0.6f32, 0.8, 0.0, 0.06, 0.08, 0.0];
    let grad = gradcheck_input("l2_normalize_rows mixed branches", &x, &[2, 3], |t| {
        l2_normalize_rows(t, 0.5).expect("normalize must succeed")
    });

    let c = coeff(6);

    // Row 0 (projected): grad must be orthogonal to the row.
    let dot0: f32 = grad[0] * x[0] + grad[1] * x[1] + grad[2] * x[2];
    assert!(
        dot0.abs() < 1e-4,
        "row 0 is above the clamp; its gradient must be orthogonal to the row, got {dot0}"
    );

    // Row 1 (clamped): grad must be exactly c/eps.
    for j in 0..3 {
        let want = c[3 + j] / 0.5;
        assert!(
            (grad[3 + j] - want).abs() < 1e-5,
            "row 1 element {j} is below the clamp; expected c/eps = {want}, got {}",
            grad[3 + j]
        );
    }
}

#[test]
fn l2_normalize_rows_backward_switches_branch_at_the_documented_boundary() {
    // The boundary is `n > eps` for the projected form, so `n == eps` is
    // CLAMPED. eps = 0.25 is a power of two, so a one-hot row of exactly eps
    // has n == eps exactly in f32 — the comparison is not at the mercy of
    // rounding.
    //
    // The two sides give visibly different answers for a one-hot row:
    //   clamped   -> dL/dx[0] = c0 / eps          (large, non-zero)
    //   projected -> dL/dx[0] = (c0 - 1*c0) / n = 0  (the projection removes it)
    let eps = 0.25f32;
    let c = coeff(3);

    // AT the boundary: n == eps exactly -> clamped branch.
    autograd::clear_graph();
    let at = Tensor::new(&[eps, 0.0, 0.0], &[1, 3]).requires_grad();
    let at_id = at.id();
    let y_at = l2_normalize_rows(&at, eps).expect("normalize must succeed");
    scalar_loss(&y_at, &c).backward();
    let g_at = autograd::get_grad(at_id).expect("input must receive gradient");
    assert!(
        (g_at.data()[0] - c[0] / eps).abs() < 1e-5,
        "n == eps is assigned to the CLAMPED branch; expected {}, got {}",
        c[0] / eps,
        g_at.data()[0]
    );

    // JUST ABOVE the boundary -> projected branch, where the same element's
    // gradient collapses to ~0.
    autograd::clear_graph();
    let above = Tensor::new(&[eps * 1.01, 0.0, 0.0], &[1, 3]).requires_grad();
    let above_id = above.id();
    let y_above = l2_normalize_rows(&above, eps).expect("normalize must succeed");
    scalar_loss(&y_above, &c).backward();
    let g_above = autograd::get_grad(above_id).expect("input must receive gradient");
    assert!(
        g_above.data()[0].abs() < 1e-4,
        "just above the clamp the projected form annihilates the row direction; \
         expected ~0, got {}",
        g_above.data()[0]
    );

    assert!(
        g_at.data().iter().all(|v| v.is_finite()) && g_above.data().iter().all(|v| v.is_finite()),
        "neither side of the boundary may produce a non-finite gradient"
    );
}
