//! `pair_cosine_mse` tests (plan 01-07, ENC-06).
//!
//! Every test name starts `pair_loss_` so the plan's positional filter selects
//! exactly this file. `grep -c` over the crate before choosing the prefix
//! returned zero pre-existing matches, which is the check D13/D30 asked for.
//!
//! The numbers here are HAND-COMPUTED, not fixture-derived: fixture parity for
//! the pair loss is 01-08's gate, and re-deriving the same values from the same
//! JSON one wave early would prove only that two readers of one file agree.

use std::path::PathBuf;

use super::*;

use crate::autograd::{self};

// ---------------------------------------------------------------------------
// Source assertions — true in RED as well as GREEN, on purpose: they describe
// the SHAPE of the implementation, not its behaviour.
// ---------------------------------------------------------------------------

fn loss_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/setfit/loss.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn pair_loss_never_reaches_the_f32_loss_utilities() {
    let src = loss_source();
    // PF-001: those helpers return f32, which cannot carry a grad_fn. The doc
    // comment above deliberately spells them with slashes (`nn/loss.rs`) so the
    // Rust path form below has no legitimate occurrence anywhere in the file.
    assert!(
        !src.contains("nn::loss"),
        "loss.rs reaches into the f32 nn::loss utilities (PF-001)"
    );
    assert!(
        !src.contains("nn::self_supervised"),
        "loss.rs reaches into the f32 nn::self_supervised utilities (PF-001)"
    );
}

#[test]
fn pair_loss_annotates_its_own_contract_equation_not_raw_mse() {
    let src = loss_source();
    assert!(
        src.contains("equation = \"pair_cosine_mse\""),
        "the contract annotation must name the dedicated pair_cosine_mse equation"
    );
    assert!(
        !src.contains("equation = \"mse_loss\""),
        "annotating this wrapper as raw mse_loss misdescribes its inputs (two [B,H] \
         embedding matrices plus binary labels, not pred/target vectors)"
    );
}

#[test]
fn pair_loss_epsilon_agrees_with_the_encoder_normalize_path() {
    // Two literals that must not drift: the encoder L2-normalizes its output
    // with this epsilon, and this objective clamps the cosine norms with it.
    assert_eq!(
        PAIR_COSINE_EPS,
        crate::setfit::encoder::L2_EPS,
        "the pair objective's epsilon must equal the encoder's normalize epsilon"
    );
}

// ---------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------

/// The hand-computed 3-pair case used by several tests below.
///
/// | pair | za     | zb     | cos                       | label | sq. err |
/// |------|--------|--------|---------------------------|-------|---------|
/// | 0    | (3, 4) | (3, 4) | 25 / (5*5)      = 1       | 1.0   | 0       |
/// | 1    | (1, 0) | (0, 1) | 0 / (1*1)       = 0       | 0.0   | 0       |
/// | 2    | (1, 0) | (1, 1) | 1 / (1*sqrt(2)) = 0.70711 | 0.0   | 1/2     |
///
/// mean = (0 + 0 + 1/2) / 3 = **1/6**.
fn hand_case() -> (Tensor, Tensor, Vec<f32>) {
    let za = Tensor::new(&[3.0, 4.0, 1.0, 0.0, 1.0, 0.0], &[3, 2]);
    let zb = Tensor::new(&[3.0, 4.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    (za, zb, vec![1.0, 0.0, 0.0])
}

#[test]
fn pair_loss_equals_the_hand_computed_three_pair_value() {
    let (za, zb, labels) = hand_case();
    let loss = pair_cosine_mse(&za, &zb, &labels).expect("hand case must evaluate");
    assert_eq!(loss.shape(), &[1], "the objective reduces to a [1] tensor");
    let expected = 1.0f32 / 6.0;
    assert!(
        (loss.item() - expected).abs() < 1e-6,
        "expected {expected}, got {}",
        loss.item()
    );
}

#[test]
fn pair_loss_equals_the_composition_of_the_two_primitives_bitwise() {
    let (za, zb, labels) = hand_case();
    let direct = pair_cosine_mse(&za, &zb, &labels).expect("pair loss");

    let composed = mse_loss(
        &cosine_similarity_rows(&za, &zb, PAIR_COSINE_EPS).expect("cosine"),
        &labels,
    )
    .expect("mse");

    // to_bits, not a tolerance: this objective must BE the composition, not
    // merely agree with it to some epsilon. A hand-rolled reimplementation that
    // happens to land within 1e-6 would still be a second copy of the math.
    assert_eq!(
        direct.item().to_bits(),
        composed.item().to_bits(),
        "pair_cosine_mse must be exactly mse_loss(cosine_similarity_rows(..), labels)"
    );
}

#[test]
fn pair_loss_is_zero_for_identical_embeddings_labelled_positive() {
    let za = Tensor::new(&[0.3, -0.9, 0.4, 1.2, 0.1, -0.2], &[2, 3]);
    let zb = za.clone();
    let loss = pair_cosine_mse(&za, &zb, &[1.0, 1.0]).expect("identical pair");
    assert!(
        loss.item().abs() < 1e-6,
        "cos == 1 against label 1 must give ~0 loss, got {}",
        loss.item()
    );
}

#[test]
fn pair_loss_is_one_for_orthogonal_embeddings_labelled_positive() {
    let za = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let zb = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[2, 2]);
    let loss = pair_cosine_mse(&za, &zb, &[1.0, 1.0]).expect("orthogonal pair");
    assert!(
        (loss.item() - 1.0).abs() < 1e-6,
        "cos == 0 against label 1 must give ~1 loss, got {}",
        loss.item()
    );
}

#[test]
fn pair_loss_is_one_for_antiparallel_embeddings_labelled_negative() {
    // The other end of the range: cos == -1 against label 0 is also 1.0, so the
    // "~1" tests above cannot both be satisfied by a constant.
    let za = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let zb = Tensor::new(&[-1.0, -2.0], &[1, 2]);
    let loss = pair_cosine_mse(&za, &zb, &[0.0]).expect("antiparallel pair");
    assert!(
        (loss.item() - 1.0).abs() < 1e-6,
        "cos == -1 against label 0 must give ~1 loss, got {}",
        loss.item()
    );
}

// ---------------------------------------------------------------------------
// Graph connectivity — the ENC-06 property that an f32 loss cannot have
// ---------------------------------------------------------------------------

#[test]
fn pair_loss_requires_grad_when_either_input_does() {
    autograd::clear_graph();
    let (za, zb, labels) = hand_case();
    let za = za.requires_grad();
    let loss = pair_cosine_mse(&za, &zb, &labels).expect("loss");
    assert!(
        loss.requires_grad_enabled(),
        "a graph-connected input must produce a graph-connected loss"
    );
}

#[test]
fn pair_loss_does_not_require_grad_when_neither_input_does() {
    autograd::clear_graph();
    let (za, zb, labels) = hand_case();
    let loss = pair_cosine_mse(&za, &zb, &labels).expect("loss");
    // The two-sided half: without this, a `requires_grad_(true)` hardcoded on
    // the result would satisfy the test above.
    assert!(!loss.requires_grad_enabled());
}

#[test]
fn pair_loss_backward_reaches_both_embedding_matrices() {
    autograd::clear_graph();
    // Deliberately NOT the hand case: pair 0 there is an exact self-pair whose
    // cosine gradient is analytically zero, so a one-sided severed edge could
    // hide in it.
    let za = Tensor::new(&[0.7, -0.2, 0.4, 0.9, 1.1, 0.3], &[3, 2]).requires_grad();
    let zb = Tensor::new(&[0.1, 0.8, -0.5, 0.2, 0.6, -0.4], &[3, 2]).requires_grad();
    let loss = pair_cosine_mse(&za, &zb, &[1.0, 0.0, 1.0]).expect("loss");
    assert!(loss.item().is_finite(), "loss is {}", loss.item());
    loss.backward();

    for (name, t) in [("za", &za), ("zb", &zb)] {
        let g = autograd::get_grad(t.id())
            .unwrap_or_else(|| panic!("`{name}` received NO gradient — the edge is severed"));
        assert_eq!(g.numel(), t.numel(), "`{name}` gradient arity");
        let l2: f64 = g.data().iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        assert!(
            l2.sqrt() > 1e-6,
            "`{name}` gradient L2 is {:e}: the backward does not reach this branch",
            l2.sqrt()
        );
        for (i, v) in g.data().iter().enumerate() {
            assert!(v.is_finite(), "`{name}`[{i}] gradient is {v}");
        }
    }
}

#[test]
fn pair_loss_gradients_are_bitwise_those_of_the_explicit_composition() {
    // Guards against a hand-rolled derivative replacing the composed one: the
    // forward could still agree to 1e-6 while the backward diverged.
    fn grads_via<F>(build: F) -> (Vec<f32>, Vec<f32>)
    where
        F: Fn(&Tensor, &Tensor, &[f32]) -> Tensor,
    {
        autograd::clear_graph();
        let za = Tensor::new(&[0.7, -0.2, 0.4, 0.9, 1.1, 0.3], &[3, 2]).requires_grad();
        let zb = Tensor::new(&[0.1, 0.8, -0.5, 0.2, 0.6, -0.4], &[3, 2]).requires_grad();
        let labels = [1.0, 0.0, 1.0];
        let loss = build(&za, &zb, &labels);
        loss.backward();
        (
            autograd::get_grad(za.id())
                .expect("za grad")
                .data()
                .to_vec(),
            autograd::get_grad(zb.id())
                .expect("zb grad")
                .data()
                .to_vec(),
        )
    }

    let (da, db) = grads_via(|a, b, l| pair_cosine_mse(a, b, l).expect("pair loss"));
    let (ca, cb) = grads_via(|a, b, l| {
        mse_loss(
            &cosine_similarity_rows(a, b, PAIR_COSINE_EPS).expect("cosine"),
            l,
        )
        .expect("mse")
    });

    for (i, (x, y)) in da.iter().zip(ca.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "za grad element {i}: {x} vs {y}");
    }
    for (i, (x, y)) in db.iter().zip(cb.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "zb grad element {i}: {x} vs {y}");
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn batch_invalid_reason(e: &SetFitError) -> String {
    match e {
        SetFitError::BatchInvalid { reason } => reason.clone(),
        other => panic!("expected SetFitError::BatchInvalid, got {other}"),
    }
}

#[test]
fn pair_loss_rejects_a_label_count_that_disagrees_with_the_batch() {
    let (za, zb, _) = hand_case();
    let err = pair_cosine_mse(&za, &zb, &[1.0, 0.0]).expect_err("2 labels for 3 pairs");
    let reason = batch_invalid_reason(&err);
    assert!(reason.contains('2') && reason.contains('3'), "got {reason}");
}

#[test]
fn pair_loss_rejects_a_nan_label_naming_non_finiteness_not_membership() {
    let (za, zb, _) = hand_case();
    let err = pair_cosine_mse(&za, &zb, &[1.0, f32::NAN, 0.0]).expect_err("NaN label");
    let reason = batch_invalid_reason(&err);
    // `NaN != 0.0 && NaN != 1.0` is true, so a membership-only implementation
    // ALSO rejects this input — with the wrong diagnosis. The explicit
    // finiteness check must run first and must say so.
    assert!(
        reason.contains("non-finite"),
        "the NaN rejection must name non-finiteness, got {reason}"
    );
    assert!(
        reason.contains('1'),
        "the position must be named, got {reason}"
    );
}

#[test]
fn pair_loss_rejects_an_infinite_label_naming_non_finiteness() {
    let (za, zb, _) = hand_case();
    for (i, bad) in [f32::INFINITY, f32::NEG_INFINITY].into_iter().enumerate() {
        let err = pair_cosine_mse(&za, &zb, &[bad, 0.0, 0.0]).expect_err("infinite label");
        let reason = batch_invalid_reason(&err);
        assert!(
            reason.contains("non-finite"),
            "case {i}: expected a non-finiteness diagnosis, got {reason}"
        );
    }
}

#[test]
fn pair_loss_rejects_a_finite_label_outside_the_binary_set() {
    let (za, zb, _) = hand_case();
    let err = pair_cosine_mse(&za, &zb, &[1.0, 0.5, 0.0]).expect_err("0.5 is not a pair label");
    let reason = batch_invalid_reason(&err);
    assert!(
        !reason.contains("non-finite"),
        "0.5 is finite; the diagnosis must not claim otherwise: {reason}"
    );
    assert!(
        reason.contains("0.5"),
        "the offending value must be named, got {reason}"
    );
}

#[test]
fn pair_loss_distinguishes_the_two_label_rejections() {
    let (za, zb, _) = hand_case();
    let non_finite =
        batch_invalid_reason(&pair_cosine_mse(&za, &zb, &[f32::NAN, 0.0, 0.0]).expect_err("NaN"));
    let non_binary =
        batch_invalid_reason(&pair_cosine_mse(&za, &zb, &[0.5, 0.0, 0.0]).expect_err("0.5"));
    assert_ne!(
        non_finite, non_binary,
        "the two label failures must be distinguishable from the error alone"
    );
}

#[test]
fn pair_loss_accepts_both_binary_label_values() {
    // The other side of the membership gate: a validator that rejected
    // everything would satisfy every rejection test above.
    let (za, zb, _) = hand_case();
    for labels in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]] {
        pair_cosine_mse(&za, &zb, &labels).unwrap_or_else(|e| panic!("{labels:?} rejected: {e}"));
    }
}

#[test]
fn pair_loss_rejects_a_shape_mismatch_between_the_two_branches() {
    let za = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let zb = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let err = pair_cosine_mse(&za, &zb, &[1.0, 0.0]).expect_err("shapes differ");
    match err {
        SetFitError::Op(OpError::ShapeMismatch { expected, got }) => {
            assert_eq!(expected, vec![2, 2]);
            assert_eq!(got, vec![2, 3]);
        }
        other => panic!("expected a ShapeMismatch, got {other}"),
    }
}

#[test]
fn pair_loss_rejects_a_shape_mismatch_before_validating_labels() {
    // Ordering assertion: with BOTH a shape mismatch and a bad label present,
    // the shape must win — otherwise "no compute on mismatched inputs" is a
    // claim about an unreachable branch.
    let za = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let zb = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let err = pair_cosine_mse(&za, &zb, &[f32::NAN, 0.5]).expect_err("shape wins");
    assert!(
        matches!(err, SetFitError::Op(OpError::ShapeMismatch { .. })),
        "expected the shape rejection to precede label validation, got {err}"
    );
}

#[test]
fn pair_loss_rejects_a_non_rank_two_embedding_matrix() {
    let za = Tensor::new(&[1.0, 2.0], &[2]);
    let zb = Tensor::new(&[1.0, 2.0], &[2]);
    let err = pair_cosine_mse(&za, &zb, &[1.0, 0.0]).expect_err("rank 1 is not [B,H]");
    assert!(
        matches!(err, SetFitError::Op(OpError::ShapeMismatch { .. })),
        "got {err}"
    );
}

#[test]
fn pair_loss_rejects_an_empty_label_slice_against_an_empty_batch() {
    // `labels.len() > 0` is the contract's own precondition. A [0,H] tensor
    // cannot be built by `Tensor::new` without tripping its own guard, so the
    // reachable form of this is a zero-length label slice against a real batch.
    let (za, zb, _) = hand_case();
    let err = pair_cosine_mse(&za, &zb, &[]).expect_err("no labels");
    let reason = batch_invalid_reason(&err);
    assert!(reason.contains('0') && reason.contains('3'), "got {reason}");
}
