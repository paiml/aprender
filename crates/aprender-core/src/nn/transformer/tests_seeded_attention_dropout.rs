//! Seeded attention-probs dropout (plan 01-06 amendment A5; MIGRATED by 03-02).
//!
//! `nn::functional::dropout(x, p, training)` takes no seed, so the dropout
//! inside `scaled_dot_product_attention` was not reproducible. This file covers
//! the hook that fixes that, and — just as importantly — the claim that callers
//! who do NOT opt in are unaffected.
//!
//! # What 03-02 changed here, and why it could not be left alone
//!
//! 01-06's hook took a `u64` SEED at construction and mixed a per-call counter
//! into it. Plan 03-02 replaces that with an [`AttentionDropoutMasks`] source,
//! because a construction-time seed cannot carry D-15's forward-call ordinal —
//! the SetFit pair objective runs two encoder forwards per training step, and
//! keying on "how many calls have happened" is not a coordinate any caller can
//! name or replay. The three call sites that used the `u64` API therefore
//! MIGRATE; they are not preserved verbatim, because the API they called is the
//! thing being replaced.
//!
//! What the tests still assert is unchanged in substance: seeded determinism,
//! replay equality, stream separation, and that an un-hooked caller's numerics
//! did not move. The mask VALUES legitimately differ from 01-06's.
//!
//! These tests are ungated: the hook lives in `nn/`, not behind `setfit`. The
//! mask source below is therefore a local one — `setfit::dropout_rng` is behind a
//! feature these tests must not require, and depending on it would also make a
//! failure here ambiguous between the hook and the derivation.

use super::*;

use std::sync::Arc;

/// A deterministic, index-pure mask source for these ungated tests.
///
/// `SplitMix64` over `(seed, block, index)`, spelled out here so the file needs
/// no RNG dependency and so a failure localizes to the HOOK rather than to
/// whatever `setfit::dropout_rng` happens to derive. `block` stands in for the
/// forward ordinal the SetFit encoder supplies.
#[derive(Debug)]
struct ProbeMasks {
    seed: u64,
    block: u64,
    p: f32,
}

impl ProbeMasks {
    fn new(seed: u64, block: u64, p: f32) -> Arc<Self> {
        Arc::new(Self { seed, block, p })
    }

    fn bits(&self, i: u64) -> u64 {
        let mut z = self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ self.block.wrapping_mul(0xbf58_476d_1ce4_e5b9)
            ^ i.wrapping_mul(0x94d0_49bb_1331_11eb);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl AttentionDropoutMasks for ProbeMasks {
    fn attention_dropout_mask(&self, len: usize) -> Vec<f32> {
        let threshold = (f64::from(self.p) * 18_446_744_073_709_551_616.0_f64) as u128;
        let scale = 1.0 / (1.0 - self.p);
        (0..len)
            .map(|i| {
                if u128::from(self.bits(i as u64)) >= threshold {
                    scale
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Deterministic inputs, so a failure is about the hook and not about which
/// random tensor happened to be drawn.
fn qkv(batch: usize, seq: usize, embed: usize) -> Tensor {
    let n = batch * seq * embed;
    #[allow(clippy::cast_precision_loss)]
    let data: Vec<f32> = (0..n)
        .map(|i| ((i % 17) as f32).mul_add(0.031, -0.25))
        .collect();
    Tensor::new(&data, &[batch, seq, embed])
}

/// A `MultiHeadAttention` with DETERMINISTIC weights.
///
/// `MultiHeadAttention::new` builds four `Linear::new` projections, which draw
/// random weights. Two freshly constructed modules therefore differ *before*
/// any dropout runs — the first draft of these tests compared two such modules
/// and measured the weight initialiser, not the dropout hook. Installing fixed
/// weights makes the seed the only thing that varies.
fn deterministic_mha(dropout_p: f32, seed: Option<u64>) -> MultiHeadAttention {
    deterministic_mha_at(dropout_p, seed, 0)
}

/// [`deterministic_mha`] at an explicit forward-ordinal `block` (D-15).
fn deterministic_mha_at(dropout_p: f32, seed: Option<u64>, block: u64) -> MultiHeadAttention {
    const EMBED: usize = 16;
    let mut mha = MultiHeadAttention::new(EMBED, 2).with_dropout(dropout_p);
    if let Some(seed) = seed {
        mha = mha.with_attention_dropout_masks(ProbeMasks::new(seed, block, dropout_p));
    }
    #[allow(clippy::cast_precision_loss)]
    fn weights(salt: usize) -> Tensor {
        let w: Vec<f32> = (0..EMBED * EMBED)
            .map(|i| (((i + salt) % 23) as f32).mul_add(0.017, -0.19))
            .collect();
        Tensor::new(&w, &[EMBED, EMBED])
    }
    #[allow(clippy::cast_precision_loss)]
    fn bias(salt: usize) -> Tensor {
        let b: Vec<f32> = (0..EMBED)
            .map(|i| (((i + salt) % 7) as f32).mul_add(0.011, -0.03))
            .collect();
        Tensor::new(&b, &[EMBED])
    }
    mha.q_proj_mut().set_weight(weights(0));
    mha.q_proj_mut().set_bias(bias(0));
    mha.k_proj_mut().set_weight(weights(3));
    mha.k_proj_mut().set_bias(bias(1));
    mha.v_proj_mut().set_weight(weights(7));
    mha.v_proj_mut().set_bias(bias(2));
    mha.out_proj_mut().set_weight(weights(11));
    mha.out_proj_mut().set_bias(bias(3));
    mha
}

#[test]
fn mha_seeded_dropout_defaults_to_none() {
    let mha = MultiHeadAttention::new(16, 2);
    assert!(
        !mha.has_attention_dropout_masks(),
        "the hook must be opt-in; a default mask source would change every existing caller"
    );
    assert_eq!(mha.dropout_p(), 0.0, "MultiHeadAttention::new default");
}

#[test]
fn mha_seeded_dropout_builder_installs_the_seed() {
    let mha = MultiHeadAttention::new(16, 2)
        .with_attention_dropout_masks(ProbeMasks::new(0xabcd, 0, 0.3));
    assert!(mha.has_attention_dropout_masks());
}

#[test]
fn mha_seeded_dropout_same_seed_gives_bitwise_identical_output() {
    let x = qkv(2, 5, 16);
    let run = || {
        let mha = deterministic_mha(0.3, Some(0x5eed));
        assert!(
            mha.training(),
            "MultiHeadAttention::new starts in train mode"
        );
        let (out, _) = mha.forward_self(&x, None);
        out.data().to_vec()
    };
    let a = run();
    let b = run();
    for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            p.to_bits(),
            q.to_bits(),
            "element {i}: two identically seeded modules disagree ({p} vs {q}) — the \
             attention-probs dropout is still drawing from the ambient RNG"
        );
    }
}

#[test]
fn mha_seeded_dropout_different_seeds_give_different_output() {
    // Without this, "same seed gives the same answer" is also satisfied by a
    // module whose dropout never fires.
    let x = qkv(2, 5, 16);
    let run = |seed: u64| {
        let mha = deterministic_mha(0.3, Some(seed));
        let (out, _) = mha.forward_self(&x, None);
        out.data().to_vec()
    };
    let a = run(0x5eed);
    let b = run(0x0bad_5eed);
    assert!(
        a.iter()
            .zip(b.iter())
            .any(|(p, q)| p.to_bits() != q.to_bits()),
        "changing the seed changed nothing — the seed is not reaching the dropout"
    );
}

#[test]
fn mha_seeded_dropout_stream_advances_with_the_forward_ordinal() {
    // MIGRATED (03-02). 01-06 asserted that two consecutive forwards on ONE
    // module differ, because the module advanced an internal per-call counter.
    // Under D-15 the coordinate is the caller's forward ordinal, so the honest
    // form of "the stream advances" is: the SAME module at a DIFFERENT ordinal
    // draws a different mask. A site that ignored the ordinal would replay one
    // fixed mask every step — reproducible, and no longer dropout.
    let x = qkv(2, 5, 16);
    let at_zero = deterministic_mha_at(0.3, Some(0x5eed), 0);
    let at_one = deterministic_mha_at(0.3, Some(0x5eed), 1);
    let (first, _) = at_zero.forward_self(&x, None);
    let (second, _) = at_one.forward_self(&x, None);
    assert!(
        first
            .data()
            .iter()
            .zip(second.data().iter())
            .any(|(p, q)| p.to_bits() != q.to_bits()),
        "two forward ordinals gave identical output — the ordinal is not reaching \
         the mask source"
    );

    // And the other half, which 01-06 could not state at all: at the SAME
    // ordinal the module is now REPLAY-EXACT across calls. This is the property
    // TRN-06's bitwise two-clean-runs guarantee is built on, and an internal
    // counter made it structurally impossible.
    let (again, _) = at_zero.forward_self(&x, None);
    for (i, (p, q)) in first.data().iter().zip(again.data().iter()).enumerate() {
        assert_eq!(
            p.to_bits(),
            q.to_bits(),
            "element {i}: a second forward at the same ordinal moved — the mask \
             source is carrying hidden state"
        );
    }
}

#[test]
fn mha_seeded_dropout_absent_seed_leaves_the_existing_path_untouched() {
    // The regression guard for every pre-01-06 caller.
    //
    // At `dropout_p == 0.0` — which is `MultiHeadAttention::new`'s default and
    // what GroupedQueryAttention and the attention contract tests use — the
    // dropout branch is not entered at all, so the presence or absence of a seed
    // cannot change a single BIT. Asserted rather than argued, on identical
    // weights so the comparison is about the hook and not the initialiser.
    let x = qkv(2, 5, 16);
    let plain = deterministic_mha(0.0, None);
    let seeded = deterministic_mha(0.0, Some(0x5eed));
    let (a, _) = plain.forward_self(&x, None);
    let (b, _) = seeded.forward_self(&x, None);
    assert_eq!(a.shape(), b.shape());
    for (i, (p, q)) in a.data().iter().zip(b.data().iter()).enumerate() {
        assert_eq!(
            p.to_bits(),
            q.to_bits(),
            "element {i}: installing a seed changed the p == 0 path, so every \
             pre-01-06 caller's numerics moved"
        );
    }

    // And an UNSEEDED module with dropout on still uses the ambient RNG, i.e.
    // two forwards differ. That is the behaviour that existed before this plan
    // and it must survive it.
    let ambient = deterministic_mha(0.3, None);
    assert!(!ambient.has_attention_dropout_masks());
    let (u, _) = ambient.forward_self(&x, None);
    let (v, _) = ambient.forward_self(&x, None);
    assert!(
        u.data()
            .iter()
            .zip(v.data().iter())
            .any(|(p, q)| p.to_bits() != q.to_bits()),
        "an unseeded module became deterministic — the None branch no longer \
         delegates to the ambient-RNG path"
    );
}

#[test]
fn mha_seeded_dropout_is_inert_in_eval_mode() {
    let x = qkv(2, 5, 16);
    let mut mha = deterministic_mha(0.3, Some(0x5eed));
    mha.set_training(false);
    let (a, _) = mha.forward_self(&x, None);
    let (b, _) = mha.forward_self(&x, None);
    for (i, (p, q)) in a.data().iter().zip(b.data().iter()).enumerate() {
        assert_eq!(
            p.to_bits(),
            q.to_bits(),
            "element {i}: eval-mode attention is not deterministic"
        );
    }
}

#[test]
fn mha_seeded_dropout_seed_is_not_a_registered_parameter() {
    // Pitfall 7: mask sources and RNG state are module state. Naming them would
    // put non-learnable values into optimizer and freeze partitions and break the
    // ENC-05 mode-flip byte-identity proof.
    let mha = MultiHeadAttention::new(16, 2)
        .with_attention_dropout_masks(ProbeMasks::new(0x5eed, 0, 0.3));
    let names: Vec<String> = mha.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        vec![
            "q_proj.weight",
            "q_proj.bias",
            "k_proj.weight",
            "k_proj.bias",
            "v_proj.weight",
            "v_proj.bias",
            "out_proj.weight",
            "out_proj.bias",
        ],
        "the seeded-dropout field changed the registered parameter list"
    );
}

#[test]
fn mha_seeded_dropout_none_delegates_to_the_unseeded_helper() {
    // The delegation is what makes "existing callers are unchanged" a
    // structural claim rather than a hope. Exercised directly on the helper.
    let x = Tensor::from_vec(vec![1.0f32; 4096], &[4096]);
    let a = apply_dropout_seeded(&x, 0.0, None);
    let b = apply_dropout(&x, 0.0);
    assert_eq!(a.data(), b.data(), "p == 0 must be a no-op on both paths");

    // With p > 0 and no seed the helper must be non-deterministic, exactly like
    // apply_dropout: two calls differ.
    let u = apply_dropout_seeded(&x, 0.5, None);
    let v = apply_dropout_seeded(&x, 0.5, None);
    assert!(
        u.data()
            .iter()
            .zip(v.data().iter())
            .any(|(p, q)| p.to_bits() != q.to_bits()),
        "the None branch became deterministic"
    );

    // With a seed, two calls agree.
    let u = apply_dropout_seeded(&x, 0.5, Some(7));
    let v = apply_dropout_seeded(&x, 0.5, Some(7));
    for (i, (p, q)) in u.data().iter().zip(v.data().iter()).enumerate() {
        assert_eq!(p.to_bits(), q.to_bits(), "element {i}: seeded call differs");
    }
}

#[test]
fn mha_seeded_dropout_keeps_the_autograd_edge() {
    // The seeded path routes through Dropout::with_seed, which applies the mask
    // with the autograd-aware `mul` (PMAT-922). A severed edge here would freeze
    // every parameter upstream of attention in training mode.
    let x = Tensor::from_vec(vec![0.7f32; 64], &[64]).requires_grad();
    let y = apply_dropout_seeded(&x, 0.5, Some(11));
    assert!(
        y.requires_grad_enabled(),
        "seeded dropout severed the graph — the PMAT-922 failure mode"
    );
}
