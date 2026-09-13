//! MODEL B — the FROZEN-POLICY proof, on its own clean model (ENC-04).
//!
//! # Why there is NO Python comparison in this file
//!
//! `optimizer_step.json` records a step in which every parameter was trainable
//! (`all_trainable: true`, D-20). It therefore CANNOT validate a Rust model
//! configured with frozen groups: the two runs update different parameter sets
//! and would disagree everywhere by construction. Conflating the two was the
//! T-1-24 failure this split exists to prevent, so nothing here reads
//! `optimizer_step.json.post_step`.
//!
//! What is proven instead is purely internal, and stronger for being bitwise:
//! the trainable and frozen partitions are disjoint and complete, a frozen
//! tensor is absent from the optimizer's parameter set, and after a real
//! backward and a real AdamW step every frozen tensor is BIT-IDENTICAL to its
//! snapshot. The step is also asserted to have moved something, so the proof
//! cannot hold vacuously by doing nothing — that is 01-06's mutation-F standard
//! applied here: a policy nothing consults satisfies every configuration
//! round-trip, and only observed movement can tell the difference.
//!
//! D-21 pins `LayerAttention(1)` deliberately: it is the group whose boundary
//! with `LayerNorm(1)` is easiest to get wrong, and 01-07's mutation G showed a
//! widened prefix silently swallowing `attention.output.LayerNorm`.

use aprender::autograd::{self, Tensor};
use aprender::nn::{AdamW, Module};
use aprender::setfit::{pair_cosine_mse, FreezeGroup, SetFitMiniLm};

use super::{
    encode, pair_batch, read_fixture, slice_model, snapshot, tol, trainable_grads, GateInput,
    GradientsFixture, OptimizerStepFixture,
};

/// The pinned policy under test.
const POLICY: [FreezeGroup; 1] = [FreezeGroup::LayerAttention(1)];

/// Model B: a SEPARATE, freshly constructed model.
///
/// Never Model A's instance. A model that has already taken a step is not a
/// clean subject for a byte-identity proof — its "before" snapshot would
/// already carry the previous update.
fn model_b() -> SetFitMiniLm {
    let mut m = slice_model();
    m.apply_freeze(&POLICY)
        .expect("the pinned policy must apply");
    m
}

fn names_of(pairs: Vec<(String, &Tensor)>) -> Vec<String> {
    pairs.into_iter().map(|(n, _)| n).collect()
}

#[test]
fn frozen_gate_partitions_are_disjoint_and_complete() {
    let mut m = model_b();
    let all: Vec<String> = names_of(m.encoder().named_parameters());
    let frozen: Vec<String> = names_of(m.frozen_parameters());
    let trainable: Vec<String> = m
        .trainable_parameters_mut()
        .into_iter()
        .map(|(n, _)| n)
        .collect();

    assert!(!frozen.is_empty(), "the policy froze nothing");
    assert!(!trainable.is_empty(), "the policy froze everything");
    for n in &frozen {
        assert!(
            !trainable.contains(n),
            "`{n}` is in BOTH partitions — the optimizer would update a frozen tensor"
        );
    }
    let mut union: Vec<String> = frozen.iter().chain(trainable.iter()).cloned().collect();
    union.sort();
    let mut expected = all;
    expected.sort();
    assert_eq!(
        union, expected,
        "the two partitions do not cover the named parameter set exactly"
    );
}

#[test]
fn frozen_gate_every_frozen_tensor_is_excluded_and_flagged() {
    let mut m = model_b();
    let trainable: Vec<String> = m
        .trainable_parameters_mut()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    for (name, t) in m.frozen_parameters() {
        assert!(
            !t.requires_grad_enabled(),
            "frozen `{name}` still has requires_grad set"
        );
        assert!(
            !trainable.contains(&name),
            "frozen `{name}` is still in the optimizer's parameter set — and EXCLUSION is \
             the load-bearing half, not the flag (D42)"
        );
    }
}

#[test]
fn frozen_gate_frozen_tensors_are_bitwise_unchanged_across_an_adamw_step() {
    autograd::clear_graph();
    let mut m = model_b();
    let g: GradientsFixture = read_fixture("gradients.json");
    let o: OptimizerStepFixture = read_fixture("optimizer_step.json");
    let pair = pair_batch(&m, &g.source);
    let layers = m.num_layers();

    let before = snapshot(&m);
    let za = encode(&m, &pair.a);
    let zb = encode(&m, &pair.b);
    let loss = pair_cosine_mse(&za, &zb, &pair.labels).expect("pair objective");
    assert!(loss.item().is_finite(), "loss is {}", loss.item());
    loss.backward();

    // The step runs over the TRAINABLE partition only, with the fixture's
    // hyperparameters so it is the same shape of step Model A takes.
    let (lr, b1, b2, eps, wd) = (
        o.adamw.lr,
        o.adamw.betas[0],
        o.adamw.betas[1],
        o.adamw.eps,
        o.adamw.weight_decay,
    );
    let mut opt = {
        let params: Vec<&mut Tensor> = m
            .trainable_parameters_mut()
            .into_iter()
            .map(|(_, t)| t)
            .collect();
        AdamW::new(params, lr)
            .betas(b1, b2)
            .eps(eps)
            .weight_decay(wd)
    };
    {
        let mut named = m.trainable_parameters_mut();
        let mut refs: Vec<&mut Tensor> = named.iter_mut().map(|(_, t)| &mut **t).collect();
        opt.step_with_params(&mut refs);
    }
    let after = snapshot(&m);

    let frozen: Vec<String> = names_of(m.frozen_parameters());
    let mut moved: Vec<String> = Vec::new();
    for (name, b) in &before {
        let a = &after
            .iter()
            .find(|(n, _)| n == name)
            .expect("name present in both snapshots")
            .1;
        if b != a {
            moved.push(name.clone());
        }
        if frozen.contains(name) {
            // to_bits, not an epsilon: a frozen tensor has no numerical excuse
            // for any drift at all, and an approximate comparison would pass a
            // decay-sized update.
            assert_eq!(
                b, a,
                "`{name}` is frozen but its BITS moved across the optimizer step"
            );
        }
    }

    assert!(
        !moved.is_empty(),
        "nothing moved at all — the step is inert, so the frozen assertion above proves \
         nothing (01-06 mutation F: a site that is never called satisfies every gate that \
         only inspects configuration)"
    );
    for n in &moved {
        assert!(
            !frozen.contains(n),
            "`{n}` moved and is frozen — reported after the fact by the movement scan"
        );
    }

    // The trainable partition still satisfies ENC-04, through the SAME helper
    // Model A and the D-24 negative use. Frozen components simply do not appear
    // in the partition, so only unfrozen components are aggregated.
    let grads = trainable_grads(&mut m);
    super::assert_encoder_updates(&GateInput {
        grads: &grads,
        deltas: None,
        step_lr: None,
        exemptions: &g.exempt_names(),
        floor: tol::ZERO_GRAD_FLOOR,
        layers,
    })
    .expect("the trainable partition must still satisfy the ENC-04 gate under a freeze");
}

#[test]
fn frozen_gate_never_compares_against_the_all_trainable_optimizer_fixture() {
    // A structural guard on this file's own discipline: the moment someone
    // reaches for the all-trainable fixture's parameter values here, the T-1-24
    // conflation is back. The needle is assembled at runtime AND this test's own
    // name avoids it, because a guard whose identifiers contain its own needle
    // trips itself — measured here on the first run, exactly as D40 records for
    // 01-07.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/setfit_conformance/frozen_gate.rs");
    let text = std::fs::read_to_string(&path).expect("read frozen_gate.rs");
    let needle = format!("{}{}", "post_", "step");
    for line in text.lines() {
        let code = line.split("//").next().unwrap_or("");
        assert!(
            !code.contains(&needle),
            "frozen_gate.rs reads the all-trainable post-step fixture: {}",
            line.trim()
        );
    }
}
