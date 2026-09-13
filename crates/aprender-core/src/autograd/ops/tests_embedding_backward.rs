//! Falsifier: `embedding_gather` MUST flow gradient into the weight table
//! (severed-graph guard + per-element central-difference gradcheck) and MUST
//! fail closed on hostile input.
//!
//! Obligations:
//! - OBLIG-EMBEDDING-BACKWARD-GRAD-FLOW (`pool-flatten-embedding-backward-gradflow-v1`)
//! - OBLIG-DETACH-REJECTION (`setfit-encoder-conformance-v1`)
//!
//! BUG CLASS (PMAT-913): a forward that builds its output with `Tensor::new` /
//! `Tensor::from_vec` and no adjacent `set_grad_fn` SEVERS the autograd graph.
//! The embedding table then receives no gradient at all while every shape
//! assertion stays green and the loss still decreases — the encoder is silently
//! frozen (PF-001). These tests assert the weight grad is present (severed-graph
//! guard) AND matches a self-contained central finite difference at EVERY
//! element — never a tautological `is_some`.
//!
//! SECOND BUG CLASS (fail-open OOV): `aprender-train/src/transformer/embedding.rs`
//! zero-fills out-of-vocabulary ids and `models/qwen2/mod.rs:100-108` warns and
//! emits zeros. A zero row is indistinguishable from a legitimately zero
//! embedding, so the corruption surfaces only as accuracy loss. The rejection
//! tests below pin the fail-CLOSED behaviour.
//!
//! Tolerances here are OP-LEVEL finite-difference tolerances only. Fixture
//! comparison epsilons live in the contract YAML (D-14) and must never appear
//! in a test file.

use super::{embedding_gather, OpError};
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

/// Run forward + loss WITHOUT a graph, with the weight value at `flat_idx`
/// perturbed by `delta`. Used for the finite-difference reference.
fn perturbed_loss<F>(
    w_data: &[f32],
    w_shape: &[usize],
    flat_idx: usize,
    delta: f32,
    fwd: &F,
    c: &[f32],
) -> f32
where
    F: Fn(&Tensor) -> Tensor,
{
    autograd::no_grad(|| {
        let mut wd = w_data.to_vec();
        wd[flat_idx] += delta;
        let w = Tensor::new(&wd, w_shape);
        let y = fwd(&w);
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

/// Gradcheck driver over the WEIGHT table (the only differentiable input to a
/// gather): forward, scalar loss, backward, then assert the weight grad is
/// present, finite, non-zero, and equal to a central finite difference at every
/// element.
fn gradcheck_weight<F>(name: &str, w_data: &[f32], w_shape: &[usize], fwd: F)
where
    F: Fn(&Tensor) -> Tensor,
{
    autograd::clear_graph();
    let w = Tensor::new(w_data, w_shape).requires_grad();
    let wid = w.id();
    let y = fwd(&w);
    let c = coeff(y.numel());
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let grad = autograd::get_grad(wid)
        .unwrap_or_else(|| panic!("{name}: weight received NO gradient — autograd graph severed"));
    assert_eq!(grad.shape(), w_shape, "{name}: grad shape mismatch");
    assert!(
        grad.data().iter().all(|v| v.is_finite()),
        "{name}: non-finite grad"
    );
    assert!(
        grad.data().iter().any(|&v| v.abs() > 1e-9),
        "{name}: all-zero grad"
    );

    for i in 0..w_data.len() {
        let num = (perturbed_loss(w_data, w_shape, i, FD_EPS, &fwd, &c)
            - perturbed_loss(w_data, w_shape, i, -FD_EPS, &fwd, &c))
            / (2.0 * FD_EPS);
        assert_close(grad.data()[i], num, &format!("{name} dL/dW[{i}]"));
    }
}

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

#[test]
fn embedding_gather_forward_matches_hand_computed_rows() {
    // V=4, H=3 table; B=2, S=3 batch that reuses ids 0 and 2.
    let w: Vec<f32> = (0..12).map(|i| 0.5 + (i as f32)).collect();
    let weight = Tensor::new(&w, &[4, 3]);
    let ids = [0u32, 2, 1, 0, 3, 2];

    let out = embedding_gather(&weight, &ids, 2, 3).expect("gather must succeed");
    assert_eq!(out.shape(), &[2, 3, 3], "row-major [B,S,H] expected");

    // Row-major: out[b][s][h] == w[ids[b*3+s] * 3 + h]
    for (i, &id) in ids.iter().enumerate() {
        for h in 0..3 {
            assert_eq!(
                out.data()[i * 3 + h],
                w[(id as usize) * 3 + h],
                "out[{i}][{h}] must be weight row {id}"
            );
        }
    }
}

#[test]
fn embedding_gather_forward_is_not_grad_connected_without_requires_grad() {
    // A gather from a non-trainable table must NOT fabricate a graph edge.
    autograd::clear_graph();
    let weight = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let out = embedding_gather(&weight, &[0, 1], 1, 2).expect("gather must succeed");
    assert!(
        !out.requires_grad_enabled(),
        "gather from a frozen table must not require grad"
    );
}

// ---------------------------------------------------------------------------
// Gradient flow
// ---------------------------------------------------------------------------

#[test]
fn embedding_gather_backward_matches_central_finite_differences() {
    // V=4, H=3; ids reuse rows 0 and 2 so the accumulation path is exercised
    // inside the gradcheck itself.
    let w: Vec<f32> = (0..12)
        .map(|i| 0.21 + 0.17 * (i as f32) - 0.013 * ((i * i) as f32))
        .collect();
    let ids = [0u32, 2, 1, 0, 3, 2];
    gradcheck_weight("embedding_gather", &w, &[4, 3], move |t| {
        embedding_gather(t, &ids, 2, 3).expect("gather must succeed")
    });
}

#[test]
fn embedding_gather_backward_accumulates_repeated_ids() {
    // A token appearing twice must accumulate TWICE the single-occurrence
    // gradient. An overwriting backward silently drops gradient for every
    // repeated token and is invisible on a batch with distinct ids.
    let w = [1.0f32, 2.0, 3.0, 4.0];

    autograd::clear_graph();
    let w1 = Tensor::new(&w, &[2, 2]).requires_grad();
    let id1 = w1.id();
    embedding_gather(&w1, &[0], 1, 1)
        .expect("gather must succeed")
        .sum()
        .backward();
    let g1 = autograd::get_grad(id1).expect("single-occurrence gradient must exist");

    autograd::clear_graph();
    let w2 = Tensor::new(&w, &[2, 2]).requires_grad();
    let id2 = w2.id();
    embedding_gather(&w2, &[0, 0], 1, 2)
        .expect("gather must succeed")
        .sum()
        .backward();
    let g2 = autograd::get_grad(id2).expect("repeated-occurrence gradient must exist");

    for i in 0..4 {
        assert!(
            (g2.data()[i] - 2.0 * g1.data()[i]).abs() < 1e-6,
            "repeated id must accumulate: g2[{i}]={} != 2 * g1[{i}]={}",
            g2.data()[i],
            g1.data()[i]
        );
    }
    // Row 1 is never referenced, so it must stay exactly zero.
    assert_eq!(
        g2.data()[2],
        0.0,
        "unreferenced row must receive no gradient"
    );
    assert_eq!(
        g2.data()[3],
        0.0,
        "unreferenced row must receive no gradient"
    );
}

// ---------------------------------------------------------------------------
// Fail-closed rejections
// ---------------------------------------------------------------------------

#[test]
fn embedding_gather_rejects_out_of_vocabulary_id() {
    let weight = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = embedding_gather(&weight, &[0, 7], 1, 2)
        .expect_err("an id at or beyond vocab_size must be rejected, never zero-filled");
    assert_eq!(
        err,
        OpError::OutOfVocabulary {
            id: 7,
            vocab_size: 2,
            position: 1,
        },
        "the error must name the offending id AND its flat position"
    );
}

#[test]
fn embedding_gather_rejects_id_exactly_at_vocab_size() {
    // Off-by-one boundary: `id == vocab_size` is out of range.
    let weight = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = embedding_gather(&weight, &[2], 1, 1).expect_err("id == vocab_size is out of range");
    assert!(matches!(err, OpError::OutOfVocabulary { id: 2, .. }));
}

#[test]
fn embedding_gather_rejects_zero_dimension() {
    let weight = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        embedding_gather(&weight, &[], 0, 2).expect_err("batch 0 must be rejected"),
        OpError::ZeroDimension { which: "batch" }
    );
    assert_eq!(
        embedding_gather(&weight, &[], 2, 0).expect_err("seq 0 must be rejected"),
        OpError::ZeroDimension { which: "seq" }
    );

    let zero_hidden = Tensor::new(&[], &[3, 0]);
    assert_eq!(
        embedding_gather(&zero_hidden, &[0], 1, 1).expect_err("hidden 0 must be rejected"),
        OpError::ZeroDimension { which: "hidden" }
    );

    let zero_vocab = Tensor::new(&[], &[0, 3]);
    assert_eq!(
        embedding_gather(&zero_vocab, &[0], 1, 1).expect_err("vocab_size 0 must be rejected"),
        OpError::ZeroDimension {
            which: "vocab_size"
        }
    );
}

#[test]
fn embedding_gather_rejects_shape_overflow() {
    let weight = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // batch * seq overflows on its own.
    let err = embedding_gather(&weight, &[0], usize::MAX, 2)
        .expect_err("batch * seq overflow must be rejected before allocating");
    assert_eq!(
        err,
        OpError::ShapeOverflow {
            dims: vec![usize::MAX, 2, 3],
        }
    );

    // batch * seq fits, but (batch * seq) * hidden does not.
    let err = embedding_gather(&weight, &[0], usize::MAX / 2, 1)
        .expect_err("positions * hidden overflow must be rejected before allocating");
    assert_eq!(
        err,
        OpError::ShapeOverflow {
            dims: vec![usize::MAX / 2, 1, 3],
        }
    );
}

#[test]
fn embedding_gather_rejects_non_finite_weight() {
    let nan = Tensor::new(&[1.0, f32::NAN, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        embedding_gather(&nan, &[0], 1, 1).expect_err("NaN weight must be rejected"),
        OpError::NonFiniteInput { position: 1 }
    );

    let inf = Tensor::new(&[1.0, 2.0, f32::INFINITY, 4.0], &[2, 2]);
    assert_eq!(
        embedding_gather(&inf, &[0], 1, 1).expect_err("Inf weight must be rejected"),
        OpError::NonFiniteInput { position: 2 }
    );
}

#[test]
fn embedding_gather_rejects_rank_and_length_mismatch() {
    let rank3 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    assert_eq!(
        embedding_gather(&rank3, &[0], 1, 1).expect_err("weight must be 2-D"),
        OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: vec![1, 2, 2],
        }
    );

    let weight = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        embedding_gather(&weight, &[0, 1, 0], 2, 3).expect_err("ids.len() must equal batch * seq"),
        OpError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3],
        }
    );
}

#[test]
fn embedding_gather_error_display_names_the_condition() {
    let weight = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = embedding_gather(&weight, &[9], 1, 1).expect_err("OOV expected");
    let rendered = err.to_string();
    assert!(
        rendered.contains("OutOfVocabulary") && rendered.contains('9'),
        "Display must name the condition and the offending id, got {rendered}"
    );
}
