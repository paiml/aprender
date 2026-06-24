//! END-TO-END capability proof: a tiny transformer built from apr's own `nn`
//! modules trains a real (deterministic) task to a DECREASING loss, AND every
//! trainable parameter group actually updates.
//!
//! Obligation: OBLIG-TRANSFORMER-END-TO-END-TRAINABLE
//!
//! WHY THIS BEAT EXISTS (PMAT-921): the autograd severed-graph sweep
//! (PMAT-907/911/913/914) un-severed norms, embedding, pools and attention,
//! but every fix was verified by a PER-LAYER finite-difference gradcheck — never
//! by training a real model to a loss target. A composition of individually
//! correct layers can still fail end-to-end (e.g. a residual-add or a reshape on
//! the *integration* path silently detaches one parameter), and a per-layer
//! gradcheck would not catch it. This test closes that gap.
//!
//! THE TASK: memorize a single fixed (input -> next-token) sequence (a degenerate
//! language-modeling / "copy-the-answer" task). Loss MUST collapse toward ~0; if
//! ANY parameter on the live path were still severed, the model could not fit the
//! sequence and that parameter would be frozen.
//!
//! THE FALSIFIER (two independent guards, both must hold):
//!   (a) final loss << initial loss (drops by a large factor toward ~0), AND
//!   (b) for EVERY trainable param group — embedding weight, the attention
//!       Q/K/V/out projection weights (+biases), both LayerNorm gamma AND beta,
//!       the FFN linear1/linear2 weights (+biases), and lm_head weight — the
//!       parameter genuinely CHANGED from its init (||p_final - p_init|| > eps)
//!       AND received a finite, non-zero gradient on at least one step.
//!
//! A frozen param (severed edge) plateaus the loss and leaves ||Δp|| == 0, so
//! either guard catches a severed graph that per-layer gradchecks miss in
//! composition.
//!
//! RED-confirmation (manual, see PR notes): detaching the attention output edge
//! (or zeroing a norm grad) plateaus the loss and freezes the corresponding
//! param — both guards fire. This proves the test is a real end-to-end guard,
//! not a tautology. Everything is seeded, so it is deterministic and CI-stable;
//! the model is tiny (vocab=32, hidden=32, heads=2, seq=8) and the step budget is
//! bounded, so it runs as a fast per-PR test, not a slow bench.

use crate::autograd::{self, Tensor};
use crate::nn::optim::Adam;
use crate::nn::transformer::TransformerEncoderLayer;
use crate::nn::{CrossEntropyLoss, Linear, Module, Reduction};

const VOCAB: usize = 32;
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const FFN: usize = 64; // 2 * hidden — small FFN width
const SEQ: usize = 8;
const STEPS: usize = 200;
const LR: f32 = 5e-3;
const SEED: u64 = 0x5EED;

/// A tiny deterministic LCG so weight init is fully reproducible WITHOUT relying
/// on the nn modules' entropy-seeded default init (which would make the loss
/// trajectory nondeterministic and the test flaky in CI).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    /// Uniform in (-scale, scale).
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((self.0 >> 33) as f32) / ((1u64 << 31) as f32); // [0, 1)
        (u * 2.0 - 1.0) * scale
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(scale)).collect()
    }
}

/// The fixed training example: a deterministic prompt of `SEQ` tokens whose
/// per-position next-token target is also fixed. The model is trained to map
/// `input[t]`-in-context to `target[t]`. Both are derived from the seed so the
/// test is self-contained and reproducible.
fn fixed_example() -> (Vec<f32>, Vec<f32>) {
    // A non-trivial, non-monotone token pattern (so the task isn't degenerately
    // "predict a constant") within [0, VOCAB).
    let input: Vec<f32> = (0..SEQ).map(|t| ((t * 7 + 3) % VOCAB) as f32).collect();
    // Next-token target: a fixed permutation of the input pattern.
    let target: Vec<f32> = (0..SEQ).map(|t| ((t * 5 + 11) % VOCAB) as f32).collect();
    (input, target)
}

/// Build a `[1, SEQ, VOCAB]` one-hot encoding of the input token ids. Detached
/// (no grad): the only live edge into the embedding is through the learnable
/// embedding WEIGHT, so a frozen embedding weight => no gradient => caught.
fn one_hot(input: &[f32]) -> Tensor {
    let mut data = vec![0.0f32; SEQ * VOCAB];
    for (t, &tok) in input.iter().enumerate() {
        data[t * VOCAB + tok as usize] = 1.0;
    }
    Tensor::new(&data, &[1, SEQ, VOCAB])
}

/// A named handle to one trainable parameter so the falsifier can report exactly
/// which group (if any) was frozen / received no gradient.
struct Param {
    name: &'static str,
    init: Vec<f32>,
    saw_grad: bool,
}

/// The exhaustive, fixed-order names of a `TransformerEncoderLayer`'s trainable
/// params, matching `TransformerEncoderLayer::parameters()` ordering: attn
/// q/k/v/out (weight,bias ×4), linear1(w,b), linear2(w,b), norm1(gamma,beta),
/// norm2(gamma,beta).
const LAYER_NAMES: [&str; 16] = [
    "attn.q.weight",
    "attn.q.bias",
    "attn.k.weight",
    "attn.k.bias",
    "attn.v.weight",
    "attn.v.bias",
    "attn.out.weight",
    "attn.out.bias",
    "ffn.linear1.weight",
    "ffn.linear1.bias",
    "ffn.linear2.weight",
    "ffn.linear2.bias",
    "norm1.gamma",
    "norm1.beta",
    "norm2.gamma",
    "norm2.beta",
];

/// Collect every trainable param of the tiny model into a flat, named,
/// snapshot-able list. Order is fixed and exhaustive; if a future change adds a
/// trainable tensor on the live path it must be added here too.
fn snapshot_params(
    embed_w: &Tensor,
    layer: &TransformerEncoderLayer,
    lm_head: &Linear,
) -> Vec<Param> {
    let mut out = Vec::new();
    let mut push = |name: &'static str, t: &Tensor| {
        out.push(Param {
            name,
            init: t.data().to_vec(),
            saw_grad: false,
        });
    };
    push("embedding.weight", embed_w);
    let params = layer.parameters();
    assert_eq!(
        params.len(),
        LAYER_NAMES.len(),
        "TransformerEncoderLayer param count changed; update LAYER_NAMES so no \
         trainable group escapes the frozen-param guard"
    );
    for (name, t) in LAYER_NAMES.iter().zip(params.iter()) {
        push(name, t);
    }
    push("lm_head.weight", lm_head.weight());
    if let Some(b) = lm_head.bias() {
        push("lm_head.bias", b);
    }
    out
}

/// Final-state read of every param in the SAME order as `snapshot_params`.
fn read_finals(
    embed_w: &Tensor,
    layer: &TransformerEncoderLayer,
    lm_head: &Linear,
) -> Vec<(&'static str, Vec<f32>)> {
    let mut finals: Vec<(&'static str, Vec<f32>)> = Vec::new();
    finals.push(("embedding.weight", embed_w.data().to_vec()));
    for (name, t) in LAYER_NAMES.iter().zip(layer.parameters().iter()) {
        finals.push((name, t.data().to_vec()));
    }
    finals.push(("lm_head.weight", lm_head.weight().data().to_vec()));
    if let Some(b) = lm_head.bias() {
        finals.push(("lm_head.bias", b.data().to_vec()));
    }
    finals
}

#[test]
fn tiny_transformer_trains_to_decreasing_loss_all_params_update() {
    autograd::clear_graph();
    let mut rng = Lcg::new(SEED);

    // ---- Build the tiny model from apr's own nn modules ----
    // Embedding as a learnable [VOCAB, HIDDEN] weight (one-hot @ W lookup).
    let mut embed_w = Tensor::new(&rng.vec(VOCAB * HIDDEN, 0.1), &[VOCAB, HIDDEN]).requires_grad();

    // One transformer block: LayerNorm + MHA + LayerNorm + FFN (pre-norm).
    // dropout=0.0 => deterministic identity, no stochasticity in CI.
    let mut layer = TransformerEncoderLayer::new(HIDDEN, HEADS, FFN).with_dropout(0.0);

    // lm_head: HIDDEN -> VOCAB.
    let mut lm_head = Linear::new(HIDDEN, VOCAB);

    // Deterministically (re)seed every weight matrix on the live path so the
    // loss trajectory is reproducible regardless of the module's default init.
    {
        let mut p = layer.parameters_mut();
        for t in &mut p {
            let n = t.numel();
            if n > HIDDEN {
                // a weight matrix — give it a small deterministic spread
                let scale = (1.0 / HIDDEN as f32).sqrt();
                **t = Tensor::new(&rng.vec(n, scale), t.shape()).requires_grad();
            }
        }
    }
    {
        let (out_f, in_f) = (lm_head.out_features(), lm_head.in_features());
        let scale = (1.0 / in_f as f32).sqrt();
        lm_head
            .set_weight(Tensor::new(&rng.vec(out_f * in_f, scale), &[out_f, in_f]).requires_grad());
        lm_head.set_bias(Tensor::new(&vec![0.0f32; out_f], &[out_f]).requires_grad());
    }

    // ---- The fixed deterministic task ----
    let (input, target) = fixed_example();
    let x = one_hot(&input); // [1, SEQ, VOCAB], detached
    let targets = Tensor::new(&target, &[SEQ]); // class indices, [SEQ]
    let loss_fn = CrossEntropyLoss::with_reduction(Reduction::Mean);

    // ---- Snapshot init params for the frozen/grad guards ----
    let mut tracked = snapshot_params(&embed_w, &layer, &lm_head);

    // Adam over EVERY trainable param; params are passed per-step via
    // step_with_params so updates land in the same tensors we forward through.
    let mut adam = Adam::new(vec![], LR);

    let mut initial_loss = f32::NAN;
    let mut final_loss = f32::NAN;

    for step in 0..STEPS {
        autograd::clear_graph();

        // Forward: embed -> transformer block -> lm_head.
        let x2 = x.view(&[SEQ, VOCAB]);
        let embedded = x2.matmul(&embed_w).view(&[1, SEQ, HIDDEN]); // [1,SEQ,HIDDEN]
        let hidden = layer.forward(&embedded).view(&[SEQ, HIDDEN]); // [SEQ,HIDDEN]
        let logits = lm_head.forward(&hidden); // [SEQ, VOCAB]

        let loss = loss_fn.forward(&logits, &targets);
        let loss_val = loss.item();
        if step == 0 {
            initial_loss = loss_val;
        }
        final_loss = loss_val;

        loss.backward();

        // Record which params received a finite, non-zero gradient THIS step
        // (before the optimizer clears it). IDs gathered in the SAME order as
        // `tracked` so indices line up.
        {
            let mut ids: Vec<crate::autograd::TensorId> = Vec::with_capacity(tracked.len());
            ids.push(embed_w.id());
            for t in layer.parameters() {
                ids.push(t.id());
            }
            ids.push(lm_head.weight().id());
            if let Some(b) = lm_head.bias() {
                ids.push(b.id());
            }
            assert_eq!(ids.len(), tracked.len());
            for (i, id) in ids.iter().enumerate() {
                if let Some(g) = autograd::get_grad(*id) {
                    let gd = g.data();
                    if gd.iter().all(|v| v.is_finite()) && gd.iter().any(|&v| v.abs() > 1e-12) {
                        tracked[i].saw_grad = true;
                    }
                }
            }
        }

        // Optimizer step over EVERY param group (embedding, layer, lm_head).
        {
            let mut params: Vec<&mut Tensor> = Vec::new();
            params.push(&mut embed_w);
            params.extend(layer.parameters_mut());
            params.extend(lm_head.parameters_mut());
            assert_eq!(params.len(), tracked.len());
            adam.step_with_params(&mut params);
        }
    }

    // ---- GUARD (a): loss genuinely decreased toward ~0 ----
    assert!(
        initial_loss.is_finite() && final_loss.is_finite(),
        "loss became non-finite (init {initial_loss}, final {final_loss}) — \
         a training instability, not a converged graph"
    );
    // ln(VOCAB) ≈ 3.47 is the uniform-prediction loss; init should be near that,
    // final must collapse well below it. Require a >5× drop AND an absolute floor.
    assert!(
        final_loss < initial_loss * 0.2,
        "loss did NOT decrease enough: init {initial_loss} -> final {final_loss} \
         (need final < 0.2*init). A plateau here means a severed edge froze part \
         of the model so it cannot fit the fixed sequence."
    );
    assert!(
        final_loss < 0.5,
        "loss plateaued above 0.5 (init {initial_loss} -> final {final_loss}); the tiny \
         memorize task should be fit to near-zero loss when the full graph is live."
    );

    // ---- GUARD (b): EVERY trainable param updated AND saw a gradient ----
    let finals = read_finals(&embed_w, &layer, &lm_head);
    assert_eq!(finals.len(), tracked.len());

    let mut frozen = Vec::new();
    let mut no_grad = Vec::new();
    for (p, (fname, fdata)) in tracked.iter().zip(finals.iter()) {
        assert_eq!(p.name, *fname, "param ordering drift between snapshots");
        // L2 norm of the change from init.
        let delta: f32 = p
            .init
            .iter()
            .zip(fdata.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt();
        if delta <= 1e-6 {
            frozen.push((p.name, delta));
        }
        if !p.saw_grad {
            no_grad.push(p.name);
        }
    }

    assert!(
        no_grad.is_empty(),
        "these trainable param groups NEVER received a finite non-zero gradient — \
         their autograd edge is SEVERED in composition: {no_grad:?}"
    );
    assert!(
        frozen.is_empty(),
        "these trainable param groups did NOT change from init (||Δp|| ~ 0) after \
         {STEPS} Adam steps — they are FROZEN, a severed-graph regression: {frozen:?}"
    );
}
