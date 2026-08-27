//! Falsifier: `additive_attention_mask` MUST build a correctly shaped, correctly
//! valued CONSTANT and MUST reject every malformed mask.
//!
//! Obligations:
//! - OBLIG-MASK-BROADCAST-GRAPH-PRESERVATION (`setfit-encoder-conformance-v1`)
//! - OBLIG-DETACH-REJECTION (`setfit-encoder-conformance-v1`)
//!
//! Why the rank matters: a `[B,1,1,S]` mask broadcasts over `[B,heads,T,S]`
//! attention scores. A `[B,S]` mask does not, and the pre-existing `add_mask` in
//! `nn/transformer/positional_encoding.rs` "handled" that with a truncating
//! `.zip()` that silently stops at the shorter iterator — wrong at every
//! `B > 1` / `heads > 1` and invisible at `B == 1`. Plan 01-09 repairs the
//! APPLICATION; this file pins the BUILDER.
//!
//! Why this op has no `grad_fn`: it takes no differentiable input, so it is a
//! constant. `contracts/setfit-encoder-conformance-v1.yaml` carves it out of the
//! general graph-connectivity invariant explicitly, and puts that invariant on
//! `apply_additive_mask` instead. `additive_attention_mask_is_a_constant` below
//! pins the carve-out so a future "fix" cannot quietly attach a bogus edge here
//! and call the masking path graph-connected.

use super::{additive_attention_mask, OpError, NEG_MASK};
use crate::autograd::{self, Tensor};

// ---------------------------------------------------------------------------
// Shape and values
// ---------------------------------------------------------------------------

#[test]
fn additive_attention_mask_builds_broadcastable_rank4_shape() {
    // B=2, S=5, mixed lengths (3 valid then 2 pad; 5 valid).
    let mask = [1u8, 1, 1, 0, 0, 1, 1, 1, 1, 1];
    let m = additive_attention_mask(&mask, 2, 5).expect("mask must build");

    assert_eq!(
        m.shape(),
        &[2, 1, 1, 5],
        "rank-4 [B,1,1,S] is what broadcasts over [B,heads,T,S] scores"
    );
    assert_eq!(m.numel(), 10);

    for (i, &keep) in mask.iter().enumerate() {
        let want = if keep == 1 { 0.0 } else { NEG_MASK };
        assert_eq!(
            m.data()[i],
            want,
            "position {i}: keep={keep} must map to {want}"
        );
    }
}

#[test]
fn additive_attention_mask_uses_a_finite_negative_constant() {
    // NEG_MASK must be large-negative but FINITE: -inf and f32::MIN both invite
    // NaN once a softmax subtracts a row max.
    assert!(NEG_MASK.is_finite(), "NEG_MASK must be finite");
    assert!(NEG_MASK < -1.0e8, "NEG_MASK must dominate any real logit");
    assert!(
        NEG_MASK != f32::MIN && NEG_MASK != f32::NEG_INFINITY,
        "NEG_MASK must be neither f32::MIN nor -inf"
    );

    // The documented underflow property: exp(NEG_MASK - max) is exactly 0 in f32
    // for any plausible logit maximum.
    let softmax_term = (NEG_MASK - 20.0f32).exp();
    assert_eq!(
        softmax_term, 0.0,
        "exp(NEG_MASK - max) must underflow to exactly 0.0"
    );
}

#[test]
fn additive_attention_mask_is_a_constant_not_graph_connected() {
    autograd::clear_graph();
    let m = additive_attention_mask(&[1u8, 1, 0], 1, 3).expect("mask must build");
    assert!(
        !m.requires_grad_enabled(),
        "the mask BUILDER is a constant by contract; the graph-connectivity \
         obligation lives on apply_additive_mask (plan 01-09), not here"
    );
    assert!(
        m.is_leaf(),
        "a constant mask must be a leaf with no recorded backward op"
    );
}

#[test]
fn additive_attention_mask_single_row_batch_is_all_zeros_when_fully_valid() {
    let m = additive_attention_mask(&[1u8, 1, 1, 1], 1, 4).expect("mask must build");
    assert_eq!(m.shape(), &[1, 1, 1, 4]);
    assert!(
        m.data().iter().all(|&v| v == 0.0),
        "a fully valid row must add nothing to the scores"
    );
}

// ---------------------------------------------------------------------------
// Fail-closed rejections
// ---------------------------------------------------------------------------

#[test]
fn additive_attention_mask_rejects_all_padding_row() {
    // Row 0 valid, row 1 entirely padding.
    let mask = [1u8, 1, 0, 0, 0, 0];
    assert_eq!(
        additive_attention_mask(&mask, 2, 3)
            .expect_err("an all-padding row would make the whole softmax row -1e9"),
        OpError::AllPaddingRow { row: 1 }
    );
}

#[test]
fn additive_attention_mask_rejects_length_mismatch() {
    assert_eq!(
        additive_attention_mask(&[1u8, 1, 1], 2, 3).expect_err("mask.len() must be batch * seq"),
        OpError::LengthMismatch { ids: 6, mask: 3 }
    );
}

#[test]
fn additive_attention_mask_rejects_non_binary_value() {
    // A `2` must NOT be silently treated as "keep".
    assert_eq!(
        additive_attention_mask(&[1u8, 2, 1], 1, 3).expect_err("only 0 and 1 are valid"),
        OpError::NonBinaryMaskValue {
            value: 2,
            position: 1,
        }
    );
}

#[test]
fn additive_attention_mask_reports_non_binary_before_all_padding() {
    // A row of all 2s is a malformed mask, not an all-padding row. Reporting
    // AllPaddingRow here would send a caller looking for the wrong defect.
    assert_eq!(
        additive_attention_mask(&[2u8, 2, 2], 1, 3).expect_err("malformed values come first"),
        OpError::NonBinaryMaskValue {
            value: 2,
            position: 0,
        }
    );
}

#[test]
fn additive_attention_mask_rejects_zero_dimension() {
    assert_eq!(
        additive_attention_mask(&[], 0, 3).expect_err("batch 0 must be rejected"),
        OpError::ZeroDimension { which: "batch" }
    );
    assert_eq!(
        additive_attention_mask(&[], 2, 0).expect_err("seq 0 must be rejected"),
        OpError::ZeroDimension { which: "seq" }
    );
}

#[test]
fn additive_attention_mask_rejects_shape_overflow() {
    assert_eq!(
        additive_attention_mask(&[1u8], usize::MAX, 2)
            .expect_err("batch * seq overflow must be rejected before allocating"),
        OpError::ShapeOverflow {
            dims: vec![usize::MAX, 2],
        }
    );
}

#[test]
fn additive_attention_mask_error_display_names_the_condition() {
    let err = additive_attention_mask(&[1u8, 5], 1, 2).expect_err("non-binary expected");
    let rendered = err.to_string();
    assert!(
        rendered.contains("NonBinaryMaskValue") && rendered.contains('5'),
        "Display must name the condition and the offending value, got {rendered}"
    );
}

// ---------------------------------------------------------------------------
// Constant-tensor plumbing: the mask must survive an autograd-aware add.
// ---------------------------------------------------------------------------

#[test]
fn additive_attention_mask_added_to_scores_keeps_the_score_graph_alive() {
    // This is the PMAT-922 pattern: the mask is a non-grad CONSTANT combined
    // with the scores through an autograd-aware op, so the edge back to the
    // scores survives. Broadcasting itself is plan 01-09's job; here the shapes
    // already match so `add` is exercised directly.
    autograd::clear_graph();
    let scores = Tensor::new(&[0.5, 0.25, -0.75, 1.5], &[1, 1, 1, 4]).requires_grad();
    let sid = scores.id();
    let m = additive_attention_mask(&[1u8, 1, 0, 0], 1, 4).expect("mask must build");

    let masked = scores.add(&m);
    assert!(
        masked.requires_grad_enabled(),
        "adding a constant mask must NOT sever the score graph"
    );

    masked.sum().backward();
    let grad = autograd::get_grad(sid).expect("scores must receive gradient through the mask add");
    assert_eq!(grad.shape(), &[1, 1, 1, 4]);
    assert!(
        grad.data().iter().all(|&v| (v - 1.0).abs() < 1e-6),
        "d(sum(scores + const))/dscores must be all ones, got {:?}",
        grad.data()
    );
}
