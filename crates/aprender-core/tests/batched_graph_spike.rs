//! # Batched mixed-length gradient-flow spike (plan 01-03, Task 3)
//!
//! **RESEARCH Open Question 1 / Pitfall 2.** Until this file existed, nothing in
//! the repository had ever run a `batch > 1` forward through a **masked**
//! attention graph and taken a backward pass off the far end. Plan 01-09
//! established why: `add_mask` applied the mask with a truncating `.zip()`, and
//! `Tensor::from_vec` asserts on length, so any non-matching broadcast shape
//! **panicked outright**. The masked path was unreachable, not merely wrong, and
//! `scaled_dot_product_attention` is its only caller — no existing test passed
//! it a mask. 01-09 repaired the broadcast; this file is the end-to-end
//! regression test for that repair, and the first evidence that batched
//! gradient flow works at all.
//!
//! ## What is proven here, and what is NOT
//!
//! Weights are seeded-deterministic synthetic values, not real MiniLM weights.
//! This is a **graph-flow** proof — that gradient reaches every parameter,
//! finite, with the right structure — and deliberately not a numerical-parity
//! proof. Real-weight conformance is D-09's job and lands in 01-06/01-08.
//!
//! ## Chain under test
//!
//! ```text
//! embedding_gather(tok) + embedding_gather(pos)  -> autograd add
//!   -> LayerNorm
//!   -> x2 { MultiHeadAttention::forward_self(x, Some(additive_attention_mask))
//!           -> residual add -> LayerNorm
//!           -> Linear -> gelu_exact -> Linear -> residual add -> LayerNorm }
//!   -> masked_mean_pool -> l2_normalize_rows
//!   -> cosine_similarity_rows(other batch) -> mse_loss -> backward()
//! ```
//!
//! `hidden = 16` with `2` heads of `8`, so the `[B,1,1,S]` mask must broadcast
//! across a head axis of extent 2 — a wrong-axis broadcast cannot survive. The
//! two batch rows carry DIFFERENT valid lengths (5 and 9 of 9), so a mask that
//! is broadcast over the batch axis instead of applied per-row also fails.
//!
//! ## A5 (SDPA dropout seeding) — CONFIRMED, recorded for plan 01-06
//!
//! The attention-probs dropout inside `scaled_dot_product_attention`
//! (`nn/transformer/mod.rs:66-70`) calls `apply_dropout`
//! (`nn/transformer/positional_encoding.rs:515`), which calls
//! `crate::nn::functional::dropout(x, p, true)`. That function's signature is
//!
//! ```text
//! crates/aprender-core/src/nn/functional.rs:333
//! pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor
//! ```
//!
//! It takes **no seed parameter**, so the internal attention-probs dropout is
//! NOT seedable today. Plan 01-06 must implement the seeded hook as its primary
//! path, not as a contingency. This spike runs with `dropout_p == 0.0`, where
//! `functional::dropout` returns `x.clone()` before touching any RNG
//! (`functional.rs:334`), so nothing here depends on that decision.

use aprender::autograd::{
    self, additive_attention_mask, cosine_similarity_rows, embedding_gather, l2_normalize_rows,
    masked_mean_pool, mse_loss, Tensor,
};
use aprender::nn::{LayerNorm, Linear, Module, MultiHeadAttention};

const VOCAB: usize = 32;
const HIDDEN: usize = 16;
const HEADS: usize = 2;
const FFN: usize = 32;
const MAX_POS: usize = 16;
const BATCH: usize = 2;
const SEQ: usize = 9;

// ---------------------------------------------------------------------------
// Deterministic weights
// ---------------------------------------------------------------------------

/// xorshift64* — a deterministic, dependency-free stream.
///
/// Determinism is not cosmetic here. This file is a PERMANENT regression guard,
/// and a guard that samples fresh random weights on every run is a guard that
/// can fail once a month for reasons nobody can reproduce.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Uniform in `[-0.5, 0.5)`.
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32) / 16_777_216.0 - 0.5
    }

    fn tensor(&mut self, shape: &[usize], scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.next_f32() * scale).collect();
        Tensor::new(&data, shape).requires_grad()
    }
}

// ---------------------------------------------------------------------------
// Miniature encoder
// ---------------------------------------------------------------------------

struct EncoderLayer {
    attn: MultiHeadAttention,
    norm_attn: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    norm_ffn: LayerNorm,
}

struct MiniEncoder {
    tok: Tensor,
    pos: Tensor,
    norm_embed: LayerNorm,
    layers: Vec<EncoderLayer>,
}

fn install_linear(linear: &mut Linear, rng: &mut Rng, out_f: usize, in_f: usize, scale: f32) {
    linear.set_weight(rng.tensor(&[out_f, in_f], scale));
    linear.set_bias(rng.tensor(&[out_f], scale));
}

fn install_norm(norm: &mut LayerNorm, rng: &mut Rng) {
    // gamma centred on 1.0 rather than on 0.0: an all-zero gamma would collapse
    // the whole layer output to beta and make the test vacuous.
    let mut g = rng.tensor(&[HIDDEN], 0.2);
    for v in g.data_mut() {
        *v += 1.0;
    }
    norm.set_weight(g);
    norm.set_bias(rng.tensor(&[HIDDEN], 0.1));
}

impl MiniEncoder {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);

        let tok = rng.tensor(&[VOCAB, HIDDEN], 0.8);
        let pos = rng.tensor(&[MAX_POS, HIDDEN], 0.3);

        let mut norm_embed = LayerNorm::new(&[HIDDEN]);
        install_norm(&mut norm_embed, &mut rng);

        let mut layers = Vec::with_capacity(2);
        for _ in 0..2 {
            // dropout_p defaults to 0.0 (`MultiHeadAttention::new`), so the
            // unseedable attention-probs dropout never runs — see the A5 note.
            let mut attn = MultiHeadAttention::new(HIDDEN, HEADS);
            install_linear(attn.q_proj_mut(), &mut rng, HIDDEN, HIDDEN, 0.5);
            install_linear(attn.k_proj_mut(), &mut rng, HIDDEN, HIDDEN, 0.5);
            install_linear(attn.v_proj_mut(), &mut rng, HIDDEN, HIDDEN, 0.5);
            install_linear(attn.out_proj_mut(), &mut rng, HIDDEN, HIDDEN, 0.5);

            let mut norm_attn = LayerNorm::new(&[HIDDEN]);
            install_norm(&mut norm_attn, &mut rng);

            let mut ff_in = Linear::new(HIDDEN, FFN);
            install_linear(&mut ff_in, &mut rng, FFN, HIDDEN, 0.4);
            let mut ff_out = Linear::new(FFN, HIDDEN);
            install_linear(&mut ff_out, &mut rng, HIDDEN, FFN, 0.4);

            let mut norm_ffn = LayerNorm::new(&[HIDDEN]);
            install_norm(&mut norm_ffn, &mut rng);

            layers.push(EncoderLayer {
                attn,
                norm_attn,
                ff_in,
                ff_out,
                norm_ffn,
            });
        }

        Self {
            tok,
            pos,
            norm_embed,
            layers,
        }
    }

    /// `[B, S] token ids -> [B, S, H]` contextual states.
    fn forward_tokens(&self, ids: &[u32], mask: &[u8], batch: usize, seq: usize) -> Tensor {
        let tok = embedding_gather(&self.tok, ids, batch, seq).expect("token gather must succeed");
        let pos_ids: Vec<u32> = (0..batch)
            .flat_map(|_| (0..seq).map(|s| s as u32))
            .collect();
        let pos = embedding_gather(&self.pos, &pos_ids, batch, seq)
            .expect("position gather must succeed");

        let mut x = self.norm_embed.forward(&tok.add(&pos));
        let attn_mask =
            additive_attention_mask(mask, batch, seq).expect("additive mask must build");

        for layer in &self.layers {
            // THE path under test: a [B,1,1,S] mask against [B,HEADS,S,S] scores.
            let (attended, _) = layer.attn.forward_self(&x, Some(&attn_mask));
            x = layer.norm_attn.forward(&x.add(&attended));

            // BERT-correct activation: gelu_exact (01-09), never the tanh gelu.
            let ffn = layer.ff_out.forward(&layer.ff_in.forward(&x).gelu_exact());
            x = layer.norm_ffn.forward(&x.add(&ffn));
        }
        x
    }

    /// `[B, S] token ids -> [B, H]` unit-norm sentence embeddings.
    fn embed(&self, ids: &[u32], mask: &[u8], batch: usize, seq: usize) -> Tensor {
        let hidden = self.forward_tokens(ids, mask, batch, seq);
        let pooled = masked_mean_pool(&hidden, mask).expect("pooling must succeed");
        l2_normalize_rows(&pooled, 1e-12).expect("normalization must succeed")
    }

    /// Every trainable tensor as `(ENC-04 component, HF-style name, tensor)`.
    ///
    /// Names follow the HF dotted convention so the assertion messages point at
    /// the tensor a reader would look for in a checkpoint.
    fn named_params(&self) -> Vec<(String, String, &Tensor)> {
        let mut out: Vec<(String, String, &Tensor)> = vec![
            (
                "embeddings".to_string(),
                "embeddings.word_embeddings.weight".to_string(),
                &self.tok,
            ),
            (
                "embeddings".to_string(),
                "embeddings.position_embeddings.weight".to_string(),
                &self.pos,
            ),
        ];
        for (n, t) in self.norm_embed.named_parameters() {
            out.push((
                "embeddings".to_string(),
                format!("embeddings.LayerNorm.{n}"),
                t,
            ));
        }

        for (i, layer) in self.layers.iter().enumerate() {
            for (n, t) in layer.attn.named_parameters() {
                out.push((
                    format!("layer{i}.attention"),
                    format!("encoder.layer.{i}.attention.{n}"),
                    t,
                ));
            }
            for (n, t) in layer.norm_attn.named_parameters() {
                out.push((
                    format!("layer{i}.norm"),
                    format!("encoder.layer.{i}.attention.output.LayerNorm.{n}"),
                    t,
                ));
            }
            for (n, t) in layer.ff_in.named_parameters() {
                out.push((
                    format!("layer{i}.ffn"),
                    format!("encoder.layer.{i}.intermediate.dense.{n}"),
                    t,
                ));
            }
            for (n, t) in layer.ff_out.named_parameters() {
                out.push((
                    format!("layer{i}.ffn"),
                    format!("encoder.layer.{i}.output.dense.{n}"),
                    t,
                ));
            }
            for (n, t) in layer.norm_ffn.named_parameters() {
                out.push((
                    format!("layer{i}.norm"),
                    format!("encoder.layer.{i}.output.LayerNorm.{n}"),
                    t,
                ));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Fixtures — mixed lengths, so a wrong-axis broadcast cannot pass
// ---------------------------------------------------------------------------

/// Batch A: row 0 has 5 valid tokens of 9, row 1 has all 9.
fn batch_a() -> (Vec<u32>, Vec<u8>) {
    let ids = vec![
        3, 7, 11, 2, 19, 0, 0, 0, 0, // 5 valid
        5, 8, 13, 21, 4, 17, 6, 29, 9, // 9 valid
    ];
    let mask = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    (ids, mask)
}

/// Batch B: row 0 has 7 valid, row 1 has 4 — a different length profile again.
fn batch_b() -> (Vec<u32>, Vec<u8>) {
    let ids = vec![
        12, 4, 30, 1, 16, 22, 8, 0, 0, // 7 valid
        25, 14, 3, 27, 0, 0, 0, 0, 0, // 4 valid
    ];
    let mask = vec![1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    (ids, mask)
}

/// `sum_j c_j * emb[row][j]` — a scalar loss reading ONE batch row.
fn single_row_loss(emb: &Tensor, row: usize) -> Tensor {
    let mut sel = vec![0.0f32; BATCH * HIDDEN];
    for j in 0..HIDDEN {
        sel[row * HIDDEN + j] = 0.37 + 0.13 * (j as f32);
    }
    emb.mul(&Tensor::new(&sel, &[BATCH, HIDDEN])).sum()
}

fn l2(v: &[f32]) -> f32 {
    v.iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt() as f32
}

/// Run the full siamese objective and return `(encoder, grads by name)`.
fn run_full_backward() -> (MiniEncoder, Vec<(String, String, Vec<f32>)>) {
    autograd::clear_graph();
    let enc = MiniEncoder::new(0x5E7F_1701);
    let (ids_a, mask_a) = batch_a();
    let (ids_b, mask_b) = batch_b();

    let e1 = enc.embed(&ids_a, &mask_a, BATCH, SEQ);
    let e2 = enc.embed(&ids_b, &mask_b, BATCH, SEQ);
    let sim = cosine_similarity_rows(&e1, &e2, 1e-12).expect("cosine must succeed");
    let loss = mse_loss(&sim, &[1.0, 0.0]).expect("mse must succeed");

    assert!(
        loss.item().is_finite(),
        "the loss itself is non-finite ({}) — nothing downstream would mean anything",
        loss.item()
    );
    loss.backward();

    let grads = enc
        .named_params()
        .into_iter()
        .map(|(component, name, t)| {
            let g = autograd::get_grad(t.id()).unwrap_or_else(|| {
                panic!(
                    "`{name}` received NO gradient at batch {BATCH} — the graph is severed \
                     somewhere upstream of it"
                )
            });
            assert_eq!(
                g.numel(),
                t.numel(),
                "`{name}`: gradient has {} elements, parameter has {}",
                g.numel(),
                t.numel()
            );
            (component, name, g.data().to_vec())
        })
        .collect();

    (enc, grads)
}

// ===========================================================================
// (a) Finite gradient on every NAMED parameter tensor
// ===========================================================================

#[test]
fn batched_graph_every_named_parameter_receives_a_finite_gradient() {
    let (_enc, grads) = run_full_backward();

    assert_eq!(
        grads.len(),
        2 + 2 + 2 * (8 + 2 + 2 + 2 + 2),
        "parameter roster changed — update the expected count deliberately"
    );

    for (_component, name, g) in &grads {
        if let Some(pos) = g.iter().position(|v| !v.is_finite()) {
            panic!(
                "`{name}`: non-finite gradient at element {pos} (value {})",
                g[pos]
            );
        }
    }
}

// ===========================================================================
// (b) Non-zero AGGREGATE per ENC-04 component, with the key-bias exemption
//     asserted from BOTH sides
// ===========================================================================

#[test]
fn batched_graph_every_enc04_component_has_a_non_zero_aggregate_gradient() {
    let (_enc, grads) = run_full_backward();

    // Deliberately NOT "every tensor has a non-zero gradient". That wording is
    // unsatisfiable against a CORRECT implementation: see the k_proj.bias test
    // below. Aggregating per component keeps the gate meaningful without
    // failing the phase on correct code.
    let mut components: Vec<String> = grads.iter().map(|(c, _, _)| c.clone()).collect();
    components.sort();
    components.dedup();
    assert_eq!(
        components.len(),
        1 + 2 * 3,
        "expected embeddings + {{attention, ffn, norm}} x 2 layers, got {components:?}"
    );

    for component in &components {
        let mut acc = 0.0f64;
        for (c, _, g) in &grads {
            if c == component {
                acc += g.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
            }
        }
        let norm = acc.sqrt();
        assert!(
            norm > 1e-9,
            "component `{component}` has an aggregate gradient L2 norm of {norm:e} — \
             gradient is not reaching it at batch {BATCH}"
        );
    }
}

#[test]
fn batched_graph_key_projection_bias_gradient_is_near_zero_by_softmax_shift_invariance() {
    let (_enc, grads) = run_full_backward();

    // PROOF this is not a hole. The key bias adds the SAME vector b_k to every
    // key, so for a fixed query the term q_i·b_k / sqrt(d) is identical across
    // all keys j in that row. Softmax is invariant under adding a constant to
    // every logit of a row, therefore dL/db_k == 0 in exact arithmetic. The
    // masked keys do not disturb this: they receive -1e9, which underflows to
    // exactly 0 after exp regardless of any constant shift.
    //
    // Asserting NON-zero here would fail the phase on correct code; skipping it
    // would leave a hole. Asserting NEAR-zero makes the exemption two-sided —
    // an unexpectedly LARGE gradient here is a failure, because it would mean
    // the mask, the softmax, or the head split is not doing what it claims.
    // MEASURED on this implementation: |g| tops out at 9.09e-10 (layer 0) and
    // 1.05e-9 (layer 1) — pure f32 roundoff. The 1e-6 gate therefore carries
    // ~3 orders of headroom and is not tuned to the current numbers.
    let mut checked = 0;
    for (_c, name, g) in &grads {
        if !name.ends_with("attention.k_proj.bias") {
            continue;
        }
        checked += 1;
        for (i, &v) in g.iter().enumerate() {
            assert!(
                v.abs() <= 1e-6,
                "`{name}`[{i}] = {v:e}: the key bias must be analytically zero \
                 (softmax shift invariance). A value this large means the constant \
                 shift is NOT cancelling — suspect the mask, the softmax, or the \
                 head-axis broadcast."
            );
        }
    }
    assert_eq!(checked, 2, "expected one k_proj.bias per layer");

    // SECOND SIDE. "Near zero" is only evidence if the other biases are NOT.
    // Without this, a backward that returned zeros for every bias would sail
    // through the assertion above. MEASURED separation: k_proj.bias L2 is
    // ~1.3e-9 while q_proj.bias L2 is ~3.0e-2 — seven orders.
    let l2_of = |suffix: &str| -> f32 {
        let mut acc = 0.0f32;
        for (_c, n, g) in &grads {
            if n.ends_with(suffix) {
                acc += l2(g);
            }
        }
        acc
    };
    let k_bias = l2_of("attention.k_proj.bias");
    let q_bias = l2_of("attention.q_proj.bias");
    let v_bias = l2_of("attention.v_proj.bias");
    assert!(
        q_bias > 1e-4 && v_bias > 1e-4,
        "the query bias ({q_bias:e}) and value bias ({v_bias:e}) must carry REAL \
         gradient — if every bias were zero the k_proj.bias check above would be \
         vacuous rather than a proof"
    );
    assert!(
        q_bias > k_bias * 1e4,
        "the key bias ({k_bias:e}) must be orders below the query bias ({q_bias:e}); \
         they are structurally different (softmax is shift-invariant in the KEY \
         direction only), and a comparable magnitude means that structure is gone"
    );
}

// ===========================================================================
// (c) Per-row contributions differ — measured with SEPARATE backward passes
// ===========================================================================

#[test]
fn batched_graph_separate_per_row_losses_produce_different_embedding_gradients() {
    // A single combined backward sums every row's contribution into one tensor
    // and cannot expose them separately, so "gradients differ across batch rows"
    // is only measurable by running the two losses independently.
    let (ids, mask) = batch_a();

    let mut per_row: Vec<Vec<f32>> = Vec::new();
    for row in 0..BATCH {
        autograd::clear_graph();
        let enc = MiniEncoder::new(0x5E7F_1701);
        let emb = enc.embed(&ids, &mask, BATCH, SEQ);
        single_row_loss(&emb, row).backward();

        let g = autograd::get_grad(enc.tok.id())
            .expect("the token embedding table must receive gradient");
        assert!(
            g.data().iter().all(|v| v.is_finite()),
            "row {row}: non-finite embedding gradient"
        );
        assert!(
            l2(g.data()) > 1e-9,
            "row {row}: embedding gradient is entirely zero"
        );
        per_row.push(g.data().to_vec());
    }

    let diff: Vec<f32> = per_row[0]
        .iter()
        .zip(per_row[1].iter())
        .map(|(a, b)| a - b)
        .collect();
    assert!(
        l2(&diff) > 1e-6,
        "the two batch rows produced IDENTICAL embedding gradients (L2 difference {:e}). \
         They use different token ids and different valid lengths, so identical \
         gradients mean one row's contribution is being dropped or duplicated.",
        l2(&diff)
    );

    // Sharper still: row 0's sentence never uses id 29, row 1's does. A row-0
    // loss must leave that embedding row untouched.
    let row_29 = &per_row[0][29 * HIDDEN..30 * HIDDEN];
    assert!(
        l2(row_29) < 1e-9,
        "a loss reading only batch row 0 gave gradient to token id 29, which appears \
         only in batch row 1 — rows are leaking into each other"
    );
    let row_29_b = &per_row[1][29 * HIDDEN..30 * HIDDEN];
    assert!(
        l2(row_29_b) > 1e-9,
        "batch row 1 uses token id 29 but gave it no gradient"
    );
}

// ===========================================================================
// (d) Padding invariance
// ===========================================================================

#[test]
fn batched_graph_a_sentence_encodes_identically_alone_and_inside_a_padded_batch() {
    let (ids, mask) = batch_a();
    let enc = MiniEncoder::new(0x5E7F_1701);

    let padded = autograd::no_grad(|| enc.embed(&ids, &mask, BATCH, SEQ));

    // The same sentence as a batch of one at its true length (5), no padding.
    let solo_ids: Vec<u32> = ids[0..5].to_vec();
    let solo_mask = vec![1u8; 5];
    let solo = autograd::no_grad(|| enc.embed(&solo_ids, &solo_mask, 1, 5));

    assert_eq!(solo.shape(), &[1, HIDDEN]);
    // MEASURED: the max delta on this platform is EXACTLY 0.0 — padding is
    // bit-exact, not merely close. The gate stays at the plan-mandated 1e-5
    // rather than being tightened to 0, because the reduction order inside
    // trueno's SIMD kernels is architecture-dependent and a bit-exactness gate
    // would be asserting a property of the host rather than of the mask.
    for j in 0..HIDDEN {
        let a = padded.data()[j];
        let b = solo.data()[j];
        assert!(
            (a - b).abs() < 1e-5,
            "dim {j}: padded-batch row 0 gives {a}, the same sentence alone gives {b} \
             (delta {:e}). Padding is changing the answer, so masked positions are \
             still contributing.",
            (a - b).abs()
        );
    }
}

// ===========================================================================
// (e) Mask-repair regression: gradient must reach Q and K of BOTH layers
// ===========================================================================

#[test]
fn batched_graph_gradient_reaches_query_and_key_projections_of_every_layer() {
    // If 01-09's `add_mask` repair were reverted, the broadcast fallback would
    // either panic (`Tensor::from_vec` length assert) or rebuild the scores with
    // a bare `Tensor::from_vec` that records no grad_fn — severing everything
    // upstream of the mask, which is precisely Q and K. This test is the alarm.
    let (_enc, grads) = run_full_backward();

    for layer in 0..2 {
        for proj in ["q_proj", "k_proj"] {
            let want = format!("encoder.layer.{layer}.attention.{proj}.weight");
            let (_, _, g) = grads
                .iter()
                .find(|(_, n, _)| *n == want)
                .unwrap_or_else(|| panic!("`{want}` is missing from the parameter roster"));
            let norm = l2(g);
            assert!(
                norm > 1e-9,
                "`{want}` has gradient L2 norm {norm:e}. Q/K sit UPSTREAM of the masked \
                 scores, so a severed mask application cuts exactly here."
            );
        }
    }
}

// ===========================================================================
// Sanity: the mask is actually doing something
// ===========================================================================

#[test]
fn batched_graph_masking_changes_the_result_so_the_mask_is_not_a_no_op() {
    // Every assertion above would also pass if the mask were silently dropped.
    // This one fails in that case: replacing the true mask with an all-valid one
    // must change row 0's contextual states, because 4 padded keys would then be
    // attended to.
    let (ids, mask) = batch_a();
    let enc = MiniEncoder::new(0x5E7F_1701);

    let masked = autograd::no_grad(|| enc.forward_tokens(&ids, &mask, BATCH, SEQ));
    let unmasked =
        autograd::no_grad(|| enc.forward_tokens(&ids, &vec![1u8; BATCH * SEQ], BATCH, SEQ));

    // Compare only VALID positions of row 0 (the row that actually has padding).
    let mut max_delta = 0.0f32;
    for s in 0..5 {
        for j in 0..HIDDEN {
            let idx = s * HIDDEN + j;
            max_delta = max_delta.max((masked.data()[idx] - unmasked.data()[idx]).abs());
        }
    }
    assert!(
        max_delta > 1e-4,
        "masking changed nothing (max delta {max_delta:e}) — the additive mask is \
         not reaching the attention scores, and every other assertion in this file \
         would still pass"
    );

    // Row 1 has no padding at all, so it must be UNAFFECTED by the mask change.
    let base = SEQ * HIDDEN;
    for i in 0..SEQ * HIDDEN {
        let d = (masked.data()[base + i] - unmasked.data()[base + i]).abs();
        assert!(
            d < 1e-5,
            "row 1 is fully valid, yet the mask change moved element {i} by {d:e} — \
             the mask is being applied across the wrong axis"
        );
    }
}
