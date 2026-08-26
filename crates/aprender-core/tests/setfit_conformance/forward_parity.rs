//! Slice forward-path parity against the frozen torch reference (plan 01-08).
//!
//! Obligations: `OBLIG-ENC-03-PER-LAYER-FORWARD-PARITY`,
//! `OBLIG-ENC-03-POOLED-EMBEDDING-PARITY`, `OBLIG-ENC-03-ACTIVATION-PARITY`,
//! `OBLIG-ENC-03-PADDING-INVARIANCE`, `OBLIG-ENC-06-LOSS-FORWARD-PARITY`.
//!
//! Every batch here is built by `super::batch_from_case` from the fixture's
//! `texts`, so the vocabulary remap runs inside the encoder. Every tolerance
//! comes from `super::tolerances_generated`, which is derived from the contract.
//!
//! # Why activation parity is a separate suite
//!
//! `gelu_exact` is compared on its own numeric grid, taking no batch at all.
//! Folded into the per-layer comparison a GELU regression would present as a
//! whole-layer mismatch with no indication of the cause; separated, it localizes
//! immediately — the same D-15 argument that makes the per-layer intermediates
//! worth capturing.

use aprender::autograd::{masked_mean_pool, Tensor};
use aprender::setfit::pair_cosine_mse;

use super::{
    all_within, assert_close, batch_from_case, encode, read_fixture, slice_model, tol,
    ActivationFixture, ForwardFixture, InvarianceFixture, LossFixture, PoolingFixture,
};

// ---------------------------------------------------------------------------
// Activation (ENC-03) — an independent localization point
// ---------------------------------------------------------------------------

#[test]
fn conformance_activation_matches_the_exact_erf_reference() {
    let f: ActivationFixture = read_fixture("activation_reference.json");
    assert_eq!(f.op, "gelu_exact");
    assert_eq!(f.approximate, "none");

    let x = Tensor::from_vec(f.x.clone(), &[f.x.len()]);
    let y = x.gelu_exact();
    assert_close(
        y.data(),
        &f.y,
        tol::ACTIVATION,
        "gelu_exact over the frozen grid",
    );
}

#[test]
fn conformance_activation_gate_rejects_the_tanh_approximation() {
    // The other half of the obligation, and the proof that the gate above can
    // turn red: a tanh-form GELU must FAIL it. 01-04 measured the exact-vs-tanh
    // separation at 4.734993e-04 (not the >1e-3 the plan predicted) — still
    // ~106x the frozen tolerance, so the two functions are separated rather
    // than absorbed. Both facts are asserted here rather than described.
    let f: ActivationFixture = read_fixture("activation_reference.json");
    let x = Tensor::from_vec(f.x.clone(), &[f.x.len()]);

    assert!(
        f.tanh_vs_exact_max_delta > 10.0 * tol::ACTIVATION,
        "the recorded exact-vs-tanh separation {} is not comfortably above the frozen \
         tolerance {} — this obligation would be absorbing the difference instead of \
         separating the two functions",
        f.tanh_vs_exact_max_delta,
        tol::ACTIVATION
    );
    let tanh_form = x.gelu();
    assert!(
        !all_within(tanh_form.data(), &f.y, tol::ACTIVATION),
        "the tanh-approximation GELU passes the exact-erf activation gate; the gate \
         cannot distinguish the two forms and proves nothing"
    );
}

// ---------------------------------------------------------------------------
// Per-layer forward (ENC-03, D-15 localization)
// ---------------------------------------------------------------------------

#[test]
fn conformance_per_layer_forward_matches_the_frozen_reference() {
    let model = slice_model();
    let f: ForwardFixture = read_fixture("forward_per_layer.json");
    assert!(!f.cases.is_empty(), "forward_per_layer.json has no cases");

    for case in &f.cases {
        let batch = batch_from_case(&model, &case.texts).expect("tokenize");
        assert_eq!(batch.batch(), case.shape.batch, "`{}`: batch", case.case_id);
        assert_eq!(batch.seq(), case.shape.seq, "`{}`: seq", case.case_id);
        let mask: Vec<u8> = case.attention_mask.iter().flatten().copied().collect();
        assert_eq!(
            batch.attention_mask(),
            mask.as_slice(),
            "`{}`: attention mask",
            case.case_id
        );

        // 01-06 owns this method; this harness adds no forward path of its own.
        let (embeddings, layers) = model
            .encoder()
            .forward_tokens_per_layer(&batch)
            .expect("per-layer forward");

        assert_close(
            embeddings.data(),
            &case.embeddings_out,
            tol::FORWARD_PER_LAYER,
            &format!("case `{}` / embeddings_out", case.case_id),
        );
        assert_eq!(
            layers.len(),
            case.layer_outputs.len(),
            "`{}`: encoder produced {} layer outputs, fixture records {}",
            case.case_id,
            layers.len(),
            case.layer_outputs.len()
        );
        for (i, (rust, fixture)) in layers.iter().zip(case.layer_outputs.iter()).enumerate() {
            assert_close(
                rust.data(),
                fixture,
                tol::FORWARD_PER_LAYER,
                &format!("case `{}` / layer {i}", case.case_id),
            );
        }
        let last = layers.last().expect("at least one layer");
        assert_close(
            last.data(),
            &case.final_tokens,
            tol::FORWARD_PER_LAYER,
            &format!("case `{}` / final_tokens", case.case_id),
        );
    }
}

// ---------------------------------------------------------------------------
// Pooling and normalization (ENC-03)
// ---------------------------------------------------------------------------

#[test]
fn conformance_pooled_and_normalized_embeddings_match_the_frozen_reference() {
    let model = slice_model();
    let f: PoolingFixture = read_fixture("pooling_normalize.json");
    assert!(!f.cases.is_empty(), "pooling_normalize.json has no cases");

    for case in &f.cases {
        let batch = batch_from_case(&model, &case.texts).expect("tokenize");
        let tokens = model.encoder().forward_tokens(&batch).expect("forward");

        // BOTH stages are compared, per the obligation: normalization is a
        // contraction, so a wrong pooling denominator can be partly masked by
        // it and a normalized-only gate lets a uniform-denominator bug survive
        // on a mixed-length batch.
        let pooled = masked_mean_pool(&tokens, batch.attention_mask()).expect("pool");
        assert_close(
            pooled.data(),
            &case.pooled,
            tol::POOLING_NORMALIZE,
            &format!("case `{}` / pooled", case.case_id),
        );

        // The normalized side goes through the production `encode` path, so the
        // clamp epsilon is the encoder's own constant rather than a second copy.
        let normalized = encode(&model, &batch);
        assert_eq!(normalized.shape(), &[case.shape.batch, case.shape.hidden]);
        assert_close(
            normalized.data(),
            &case.normalized,
            tol::POOLING_NORMALIZE,
            &format!("case `{}` / normalized", case.case_id),
        );
    }
}

// ---------------------------------------------------------------------------
// Padding invariance (ENC-03, PF-014)
// ---------------------------------------------------------------------------

#[test]
fn conformance_padding_does_not_change_a_sentence_embedding() {
    let model = slice_model();
    let f: InvarianceFixture = read_fixture("batch_invariance.json");

    let single_batch = batch_from_case(&model, &f.single.texts).expect("tokenize single");
    let padded_batch = batch_from_case(&model, &f.padded_batch.texts).expect("tokenize padded");
    assert!(
        padded_batch.seq() > single_batch.seq(),
        "the padded batch is not actually wider than the single one ({} vs {}), so this \
         gate would hold trivially",
        padded_batch.seq(),
        single_batch.seq()
    );

    let single = encode(&model, &single_batch);
    let padded = encode(&model, &padded_batch);

    let hidden = single.shape()[1];
    let row = f.padded_batch.target_row;
    let slice = &padded.data()[row * hidden..(row + 1) * hidden];

    assert_close(
        single.data(),
        &f.single.embedding,
        tol::BATCH_INVARIANCE,
        "batch-of-one embedding vs fixture",
    );
    assert_close(
        slice,
        &f.padded_batch.embeddings[row * hidden..(row + 1) * hidden],
        tol::BATCH_INVARIANCE,
        "padded-batch target row vs fixture",
    );
    assert_close(
        single.data(),
        slice,
        tol::BATCH_INVARIANCE,
        "batch-of-one vs the same sentence inside a padded batch",
    );
}

// ---------------------------------------------------------------------------
// Pair loss forward (ENC-06)
// ---------------------------------------------------------------------------

#[test]
fn conformance_pair_loss_forward_matches_the_frozen_reference() {
    let model = slice_model();
    let f: LossFixture = read_fixture("loss_pair.json");

    let a = batch_from_case(&model, &f.pair.a_texts).expect("tokenize a");
    let b = batch_from_case(&model, &f.pair.b_texts).expect("tokenize b");
    let za = encode(&model, &a);
    let zb = encode(&model, &b);

    // BOTH stages, per the obligation: comparing only the scalar mean would let
    // compensating per-row sign or ordering errors average out into a matching
    // number, which is exactly what a single-scalar gate cannot see.
    //
    // `PAIR_COSINE_EPS` is `pub(crate)` (01-07) and this is an out-of-crate
    // test, so the per-row stage is taken with `f32::MIN_POSITIVE`. That is not
    // a hand-written tolerance and it is not a different computation: `encode`
    // returns unit-norm rows, so BOTH clamps are inactive under either epsilon
    // and the two calls take the identical branch. The assertion below states
    // that rather than assuming it, and the scalar stage goes through
    // `pair_cosine_mse`, which uses the crate's own constant.
    for row in 0..za.shape()[0] {
        let h = za.shape()[1];
        let n = super::l2(&za.data()[row * h..(row + 1) * h]);
        assert!(
            n > 0.5,
            "row {row} of the encoded batch has norm {n}, close enough to the clamp that \
             the epsilon used here would stop being immaterial"
        );
    }
    let cosine =
        aprender::autograd::cosine_similarity_rows(&za, &zb, f32::MIN_POSITIVE).expect("cosine");
    assert_close(
        cosine.data(),
        &f.cosine,
        tol::LOSS_PAIR,
        "per-row cosine similarity",
    );

    let loss = pair_cosine_mse(&za, &zb, &f.pair.labels).expect("pair objective");
    assert_eq!(
        loss.shape(),
        &[1],
        "the pair objective must be a [1] tensor"
    );
    assert_close(&[loss.item()], &[f.mse], tol::LOSS_PAIR, "pair MSE");
}
