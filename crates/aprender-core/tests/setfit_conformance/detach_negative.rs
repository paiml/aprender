//! D-24 — the gradient gate is not theater.
//!
//! Obligation: `OBLIG-DETACH-REJECTION`.
//!
//! A gate that has only ever been observed passing is not evidence. This file
//! builds a deliberately DETACHED encoder variant — identical arithmetic, the
//! pooled output rebuilt with `Tensor::from_vec` so the graph is severed — and
//! requires the SAME helper that accepts the real encoder to REJECT it. That
//! reproduces the PMAT-913/914/922/931 class exactly: never missing math, always
//! a silently detached graph around existing math.
//!
//! # Why the rebuilt leaf explicitly requires grad
//!
//! Without `requires_grad_(true)` the loss would be non-differentiable outright
//! and the gate would fail for the wrong reason — "nothing is differentiable
//! anywhere" rather than "the graph is severed upstream of the encoder". The
//! test asserts BOTH: the encoder parameters receive nothing, AND the rebuilt
//! leaf receives a real gradient. Only the pair distinguishes the two failures.
//!
//! # Scope (user acceptance, 2026-08-07)
//!
//! This gate runs in every feature-enabled test pass
//! (`--features setfit,conformance-fixtures`) and via `make tier2`. It is NOT in
//! a default-feature `cargo test`, because the whole harness is feature-gated.
//! Adding the CI matrix legs that enable those features is an approval-gated
//! follow-up (CI workflow edits need prior approval per CLAUDE.md).

use aprender::autograd::{self, get_grad, Tensor};
use aprender::setfit::pair_cosine_mse;

use super::{
    encode, l2, pair_batch, read_fixture, slice_model, tol, trainable_grads, GateInput,
    GradientsFixture,
};

/// Sever the graph the way PMAT-913 severed it: read the values out and rebuild
/// a fresh leaf, then explicitly ask for gradient on that leaf.
fn detached_leaf(t: &Tensor) -> Tensor {
    let mut leaf = Tensor::from_vec(t.data().to_vec(), t.shape());
    leaf.requires_grad_(true);
    leaf
}

#[test]
fn detach_negative_the_gate_rejects_a_detached_encoder() {
    autograd::clear_graph();
    let mut model = slice_model();
    let g: GradientsFixture = read_fixture("gradients.json");
    let pair = pair_batch(&model, &g.source);
    let layers = model.num_layers();

    let za = detached_leaf(&encode(&model, &pair.a));
    let zb = detached_leaf(&encode(&model, &pair.b));
    let loss = pair_cosine_mse(&za, &zb, &pair.labels).expect("pair objective");
    assert!(loss.item().is_finite());
    loss.backward();

    // The leaf DID receive gradient: the loss is differentiable, the severance
    // is upstream. Without this the assertion below would also pass against a
    // loss that is not differentiable at all.
    let leaf_grad = get_grad(za.id()).expect("the rebuilt leaf must receive a gradient");
    assert!(
        l2(leaf_grad.data()) > 0.0,
        "the rebuilt leaf's gradient is zero, so this variant proves nothing about \
         DETACHMENT — it would be indistinguishable from a dead objective"
    );

    let grads = trainable_grads(&mut model);
    let result = super::assert_encoder_updates(&GateInput {
        grads: &grads,
        deltas: None,
        step_lr: None,
        exemptions: &g.exempt_names(),
        floor: tol::ZERO_GRAD_FLOOR,
        layers,
    });

    let report = result.expect_err(
        "the ENC-04 gate ACCEPTED a detached encoder. The gate cannot distinguish a \
         connected graph from a severed one and every positive result in this harness is \
         worthless.",
    );
    assert!(
        report.contains("received NO gradient"),
        "the failure does not report missing gradients: {report}"
    );
    // The message must NAME the parameters, because that is what makes a real
    // failure diagnosable rather than merely red.
    for probe in [
        "embeddings.word_embeddings.weight",
        "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.1.output.dense.weight",
    ] {
        assert!(
            report.contains(probe),
            "the failure does not name `{probe}`: {report}"
        );
    }
}

#[test]
fn detach_negative_uses_the_same_gate_helper_as_the_positive_suites() {
    // The identity is the whole point of D-24: a second implementation could be
    // wrong in exactly the way that lets both the positive and the negative
    // pass. Needles are assembled at runtime so this scan cannot trip itself.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/setfit_conformance");
    let call = format!("{}{}", "assert_encoder_", "updates(");
    let mut callers: Vec<String> = Vec::new();
    for name in ["gradient_gate.rs", "frozen_gate.rs", "detach_negative.rs"] {
        let text = std::fs::read_to_string(dir.join(name)).expect("read gate file");
        let hits = text
            .lines()
            .filter(|l| l.split("//").next().unwrap_or("").contains(&call))
            .count();
        assert!(
            hits >= 1,
            "{name} does not call the shared ENC-04 gate helper"
        );
        callers.push(format!("{name}:{hits}"));
    }
    assert_eq!(
        callers.len(),
        3,
        "expected all three gate files: {callers:?}"
    );
}

#[test]
fn detach_negative_the_connected_encoder_passes_the_same_call() {
    // The mirror. Without it, "the gate rejects the detached variant" would also
    // be satisfied by a gate that rejects everything.
    autograd::clear_graph();
    let mut model = slice_model();
    let g: GradientsFixture = read_fixture("gradients.json");
    let pair = pair_batch(&model, &g.source);
    let layers = model.num_layers();

    let za = encode(&model, &pair.a);
    let zb = encode(&model, &pair.b);
    let loss = pair_cosine_mse(&za, &zb, &pair.labels).expect("pair objective");
    loss.backward();

    let grads = trainable_grads(&mut model);
    super::assert_encoder_updates(&GateInput {
        grads: &grads,
        deltas: None,
        step_lr: None,
        exemptions: &g.exempt_names(),
        floor: tol::ZERO_GRAD_FLOOR,
        layers,
    })
    .expect("the identical call must ACCEPT the connected encoder");
}
