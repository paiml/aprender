//! Falsifier: `masked_mean_pool` MUST flow gradient to its input with a PER-ROW
//! denominator and exactly zero into padding (severed-graph guard + per-element
//! central-difference gradcheck), and MUST reject a zero denominator.
//!
//! Obligations:
//! - OBLIG-ENC-03-PADDING-INVARIANCE (`setfit-encoder-conformance-v1`)
//! - OBLIG-DETACH-REJECTION (`setfit-encoder-conformance-v1`)
//! - OBLIG-POOL-BACKWARD-GRAD-FLOW (`pool-flatten-embedding-backward-gradflow-v1`)
//!
//! BUG CLASS 1 (PMAT-913 severed graph): a pooling forward built with
//! `Tensor::new` and no adjacent `set_grad_fn` blocks gradient from reaching
//! every encoder weight upstream of it, while shapes and loss values stay
//! plausible.
//!
//! BUG CLASS 2 (uniform denominator): dividing every row by the same count is
//! INVISIBLE on a uniform-length batch and wrong on every mixed-length one. The
//! mixed-length cases below (row counts 2 and 3) are the falsifier.
//!
//! BUG CLASS 3 (gradient leaking into padding): padded positions must receive
//! EXACTLY `0.0`, otherwise the encoder is trained on positions carrying no
//! input.
//!
//! Tolerances here are OP-LEVEL finite-difference tolerances only. Fixture
//! comparison epsilons live in the contract YAML (D-14).

use super::{masked_mean_pool, OpError};
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

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

#[test]
fn masked_mean_pool_matches_hand_computed_2x3x4_case() {
    // [B=2, S=3, H=4] row-major. Row 0 keeps positions 0,1 (count 2);
    // row 1 keeps all three (count 3).
    let x: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let hidden = Tensor::new(&x, &[2, 3, 4]);
    let mask = [1u8, 1, 0, 1, 1, 1];

    let out = masked_mean_pool(&hidden, &mask).expect("pool must succeed");
    assert_eq!(out.shape(), &[2, 4], "pooling reduces [B,S,H] to [B,H]");

    // Row 0: mean of positions 0 (0..4) and 1 (4..8) -> (0+4)/2, (1+5)/2, ...
    for h in 0..4 {
        let want = (x[h] + x[4 + h]) / 2.0;
        assert!(
            (out.data()[h] - want).abs() < 1e-6,
            "row 0 h{h}: got {} want {want}",
            out.data()[h]
        );
    }
    // Row 1: mean of positions 0,1,2 of the second block (12..24).
    for h in 0..4 {
        let want = (x[12 + h] + x[16 + h] + x[20 + h]) / 3.0;
        assert!(
            (out.data()[4 + h] - want).abs() < 1e-6,
            "row 1 h{h}: got {} want {want}",
            out.data()[4 + h]
        );
    }
}

#[test]
fn masked_mean_pool_uses_a_per_row_denominator() {
    // Both rows hold the SAME constant value but different valid counts. With a
    // correct per-row divisor both pooled rows equal that constant. With one
    // shared denominator they differ — this is the uniform-denominator
    // falsifier, and it is invisible on a uniform-length batch.
    let x = vec![5.0f32; 2 * 3 * 2];
    let hidden = Tensor::new(&x, &[2, 3, 2]);
    let mask = [1u8, 0, 0, 1, 1, 1]; // counts 1 and 3

    let out = masked_mean_pool(&hidden, &mask).expect("pool must succeed");
    for (i, &v) in out.data().iter().enumerate() {
        assert!(
            (v - 5.0).abs() < 1e-6,
            "element {i}: a per-row mean of a constant row must be that constant, got {v}"
        );
    }
}

#[test]
fn masked_mean_pool_ignores_padded_values_entirely() {
    // Poison the padded positions: a correct pool never reads them.
    let mut x = vec![1.0f32; 1 * 3 * 2];
    x[4] = 1.0e9; // position 2, hidden 0 — padded
    x[5] = -1.0e9; // position 2, hidden 1 — padded
    let hidden = Tensor::new(&x, &[1, 3, 2]);

    let out = masked_mean_pool(&hidden, &[1u8, 1, 0]).expect("pool must succeed");
    assert!(
        out.data().iter().all(|&v| (v - 1.0).abs() < 1e-6),
        "padded values must not reach the mean, got {:?}",
        out.data()
    );
}

#[test]
fn masked_mean_pool_is_not_grad_connected_without_requires_grad() {
    autograd::clear_graph();
    let hidden = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let out = masked_mean_pool(&hidden, &[1u8, 1]).expect("pool must succeed");
    assert!(
        !out.requires_grad_enabled(),
        "pooling a frozen tensor must not fabricate a graph edge"
    );
}

// ---------------------------------------------------------------------------
// Gradient flow
// ---------------------------------------------------------------------------

#[test]
fn masked_mean_pool_backward_matches_central_finite_differences() {
    // Non-degenerate MIXED-LENGTH case: [B=2, S=3, H=2], row counts 2 and 3.
    let x: Vec<f32> = (0..12)
        .map(|i| 0.31 + 0.19 * (i as f32) - 0.011 * ((i * i) as f32))
        .collect();
    let mask = [1u8, 1, 0, 1, 1, 1];
    gradcheck_input("masked_mean_pool", &x, &[2, 3, 2], move |t| {
        masked_mean_pool(t, &mask).expect("pool must succeed")
    });
}

#[test]
fn masked_mean_pool_backward_routes_zero_to_padded_positions() {
    // [B=1, S=3, H=2], only positions 0 and 1 valid. dL/dx must be 1/2 at every
    // valid element and EXACTLY 0.0 at every padded one.
    autograd::clear_graph();
    let x: Vec<f32> = (0..6).map(|i| 0.5 + (i as f32)).collect();
    let hidden = Tensor::new(&x, &[1, 3, 2]).requires_grad();
    let hid = hidden.id();

    masked_mean_pool(&hidden, &[1u8, 1, 0])
        .expect("pool must succeed")
        .sum()
        .backward();
    let grad = autograd::get_grad(hid).expect("input must receive gradient");
    assert_eq!(grad.shape(), &[1, 3, 2]);

    for i in 0..4 {
        assert!(
            (grad.data()[i] - 0.5).abs() < 1e-6,
            "valid element {i} must receive 1/n_b = 0.5, got {}",
            grad.data()[i]
        );
    }
    assert_eq!(
        grad.data()[4],
        0.0,
        "padded position must receive EXACTLY zero gradient"
    );
    assert_eq!(
        grad.data()[5],
        0.0,
        "padded position must receive EXACTLY zero gradient"
    );
}

#[test]
fn masked_mean_pool_backward_denominator_differs_per_row() {
    // Row 0 has 2 valid positions, row 1 has 3. The gradient magnitudes must be
    // 1/2 and 1/3 respectively — a shared denominator makes them equal.
    autograd::clear_graph();
    let x = vec![1.0f32; 2 * 3 * 1];
    let hidden = Tensor::new(&x, &[2, 3, 1]).requires_grad();
    let hid = hidden.id();

    masked_mean_pool(&hidden, &[1u8, 1, 0, 1, 1, 1])
        .expect("pool must succeed")
        .sum()
        .backward();
    let g = autograd::get_grad(hid).expect("input must receive gradient");

    assert!((g.data()[0] - 0.5).abs() < 1e-6, "row 0 must divide by 2");
    assert!((g.data()[1] - 0.5).abs() < 1e-6, "row 0 must divide by 2");
    assert_eq!(g.data()[2], 0.0, "row 0 padding gets nothing");
    for i in 3..6 {
        assert!(
            (g.data()[i] - 1.0 / 3.0).abs() < 1e-6,
            "row 1 must divide by 3, got {} at {i}",
            g.data()[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Fail-closed rejections
// ---------------------------------------------------------------------------

#[test]
fn masked_mean_pool_rejects_all_padding_row() {
    // The checked denominator (D-03). Row 1 has no valid position.
    let hidden = Tensor::new(&vec![1.0f32; 12], &[2, 3, 2]);
    assert_eq!(
        masked_mean_pool(&hidden, &[1u8, 1, 1, 0, 0, 0])
            .expect_err("a zero denominator must be rejected, never divided by"),
        OpError::AllPaddingRow { row: 1 }
    );
}

#[test]
fn masked_mean_pool_rejects_length_mismatch() {
    let hidden = Tensor::new(&vec![1.0f32; 12], &[2, 3, 2]);
    assert_eq!(
        masked_mean_pool(&hidden, &[1u8, 1, 1]).expect_err("mask.len() must be batch * seq"),
        OpError::LengthMismatch { ids: 6, mask: 3 }
    );
}

#[test]
fn masked_mean_pool_rejects_non_binary_mask_value() {
    let hidden = Tensor::new(&vec![1.0f32; 12], &[2, 3, 2]);
    assert_eq!(
        masked_mean_pool(&hidden, &[1u8, 3, 1, 1, 1, 1]).expect_err("only 0 and 1 are valid"),
        OpError::NonBinaryMaskValue {
            value: 3,
            position: 1,
        }
    );
}

#[test]
fn masked_mean_pool_rejects_wrong_rank_and_zero_dimensions() {
    let rank2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        masked_mean_pool(&rank2, &[1u8, 1]).expect_err("hidden must be 3-D [B,S,H]"),
        OpError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: vec![2, 2],
        }
    );

    let zero_batch = Tensor::new(&[], &[0, 3, 2]);
    assert_eq!(
        masked_mean_pool(&zero_batch, &[]).expect_err("batch 0 must be rejected"),
        OpError::ZeroDimension { which: "batch" }
    );

    let zero_seq = Tensor::new(&[], &[2, 0, 2]);
    assert_eq!(
        masked_mean_pool(&zero_seq, &[]).expect_err("seq 0 must be rejected"),
        OpError::ZeroDimension { which: "seq" }
    );

    let zero_hidden = Tensor::new(&[], &[2, 3, 0]);
    assert_eq!(
        masked_mean_pool(&zero_hidden, &[1u8, 1, 1, 1, 1, 1])
            .expect_err("hidden 0 must be rejected"),
        OpError::ZeroDimension { which: "hidden" }
    );
}

#[test]
fn masked_mean_pool_rejects_before_computing_anything() {
    // A poisoned tensor plus an invalid mask: the typed error must come from the
    // MASK check, proving validation runs before any pooling arithmetic could
    // turn the poison into NaN.
    let hidden = Tensor::new(&[f32::MAX, f32::MAX, f32::MAX, f32::MAX], &[1, 2, 2]);
    assert_eq!(
        masked_mean_pool(&hidden, &[0u8, 0]).expect_err("all-padding row must be rejected"),
        OpError::AllPaddingRow { row: 0 }
    );
}
