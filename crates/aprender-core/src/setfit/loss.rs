//! The SetFit pair objective (ENC-06).
//!
//! Contract: `setfit-encoder-conformance-v1`, equation `pair_cosine_mse`.
//!
//! One function, [`pair_cosine_mse`], composed from exactly two 01-03
//! primitives: `cosine_similarity_rows` and `mse_loss`. It computes
//!
//! ```text
//! L = (1/B) * Σ_b ( cos(za[b], zb[b]) - labels[b] )²
//! ```
//!
//! and returns a **graph-connected `[1]` tensor**, not an `f32`.
//!
//! # Why this is its own equation, and why it is not in `nn/`
//!
//! The `nn/loss.rs` and `nn/self_supervised.rs` helpers return `f32`. An `f32`
//! carries no `grad_fn`, so a training loop built on one reports a falling loss
//! while every encoder weight stays exactly where it started — PF-001, the trap
//! this phase exists to close. Nothing here imports from, calls into, or
//! re-exports either module, and a source assertion in `loss_tests.rs` holds
//! that line.
//!
//! The contract annotation names `pair_cosine_mse`, **not** `mse_loss`. They are
//! different obligations: `mse_loss` relates a prediction vector to a target
//! vector, while this relates two `[B,H]` embedding matrices to a vector of
//! binary pair labels. Annotating this wrapper as raw MSE would misdescribe its
//! inputs and quietly drop the two-embedding-input obligation ENC-06 gates.

use crate::autograd::{cosine_similarity_rows, mse_loss, OpError, Tensor};

use super::error::SetFitError;

/// Epsilon floor on each cosine norm.
///
/// The same explicit constant the encoder's trailing L2 normalization uses
/// (`encoder::L2_EPS`), so the pair objective and the embeddings it consumes
/// agree about what "degenerate" means. `pair_loss_epsilon_agrees_with_the_
/// encoder_normalize_path` asserts the equality rather than trusting two
/// literals to stay in step.
pub(crate) const PAIR_COSINE_EPS: f32 = 1e-12;

/// Mean-squared error between row-wise cosine similarity and binary pair
/// labels, as a graph-connected `[1]` tensor.
///
/// `za` and `zb` are the two siamese branches' `[B, H]` sentence embeddings;
/// `labels[b]` is `1.0` when the pair is a positive and `0.0` when it is a
/// negative. The backward reaches BOTH inputs, so one backward pass updates the
/// shared encoder body through both branches.
///
/// # Validation order
///
/// Shapes first (so nothing is computed on mismatched inputs), then labels:
/// **length**, then **finiteness**, then **binary membership**. The finiteness
/// check is explicit and comes first on purpose. `NaN != 0.0 && NaN != 1.0` is
/// true, so the membership test happens to reject `NaN` today — but only
/// incidentally, and it would report "not in {0,1}" for a value whose real
/// problem is that it is not a number.
///
/// # Errors
///
/// * [`SetFitError::Op`] wrapping [`OpError::ShapeMismatch`] — either input is
///   not rank 2, or the two shapes differ.
/// * [`SetFitError::BatchInvalid`] — `labels.len()` disagrees with the batch, a
///   label is non-finite, or a finite label is outside `{0.0, 1.0}`.
/// * [`SetFitError::Op`] — anything the two composed primitives reject
///   (zero dimension, non-finite embedding).
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "pair_cosine_mse"
)]
pub fn pair_cosine_mse(za: &Tensor, zb: &Tensor, labels: &[f32]) -> Result<Tensor, SetFitError> {
    // ---- 1. Shapes, before anything is computed --------------------------
    // `cosine_similarity_rows` would reject these too, but only AFTER this
    // function has already consulted `za.shape()[0]` to size the label checks.
    // Rejecting here keeps "no compute on mismatched inputs" a reachable
    // property rather than a claim about an unreachable branch.
    if za.shape().len() != 2 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: za.shape().to_vec(),
        }
        .into());
    }
    if zb.shape().len() != 2 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: zb.shape().to_vec(),
        }
        .into());
    }
    if za.shape() != zb.shape() {
        return Err(OpError::ShapeMismatch {
            expected: za.shape().to_vec(),
            got: zb.shape().to_vec(),
        }
        .into());
    }
    let batch = za.shape()[0];

    // ---- 2. Labels: length, then finiteness, then binary membership ------
    if labels.len() != batch {
        return Err(SetFitError::BatchInvalid {
            reason: format!(
                "labels has {} entries but the pair batch is {batch}",
                labels.len()
            ),
        });
    }
    // EXPLICIT and FIRST. `NaN != 0.0 && NaN != 1.0` is true, so the membership
    // test below happens to reject a NaN — incidentally, and with a diagnosis
    // ("not in {0,1}") that describes the wrong problem.
    if let Some(position) = labels.iter().position(|v| !v.is_finite()) {
        return Err(SetFitError::BatchInvalid {
            reason: format!(
                "labels[{position}] is non-finite ({}); SetFit pair labels must be finite",
                labels[position]
            ),
        });
    }
    if let Some(position) = labels.iter().position(|v| *v != 0.0 && *v != 1.0) {
        return Err(SetFitError::BatchInvalid {
            reason: format!(
                "labels[{position}] is {}; SetFit pair labels are binary and must be \
                 exactly 0.0 (negative pair) or 1.0 (positive pair)",
                labels[position]
            ),
        });
    }

    // Domain established by the guards above.
    contract_pre_pair_cosine_mse!(labels);

    // ---- 3. Compose, never reimplement ----------------------------------
    // Two calls, no arithmetic of our own: the epsilon-clamp branch structure
    // and both backward edges are 01-03's, already gradchecked on all four
    // clamp combinations. Inlining the math here would be a second copy of a
    // derivative that took a dedicated plan to get right.
    let similarity = cosine_similarity_rows(za, zb, PAIR_COSINE_EPS)?;
    let loss = mse_loss(&similarity, labels)?;

    contract_post_pair_cosine_mse!(loss.data());
    Ok(loss)
}

#[cfg(all(test, feature = "setfit"))]
#[path = "loss_tests.rs"]
mod loss_tests;
