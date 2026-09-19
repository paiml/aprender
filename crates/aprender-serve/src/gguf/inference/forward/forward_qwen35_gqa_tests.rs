//! PMAT-3477 / aprender#3346, #3510: the Gated `DeltaNet` recurrence when
//! `num_v_heads > num_k_heads` (grouped-query attention *inside* the recurrence).
//!
//! Qwen3.5-0.8B and -2B have `num_k_heads == num_v_heads`, so the whole hybrid
//! stack was written as if q, k and v always had the same head count. 4B and 9B
//! carry `linear_num_value_heads = 32` against `linear_num_key_heads = 16`, and
//! 27B carries 48 against 16, so `q`/`k` are half (or a third) as wide as `v` and
//! the recurrence panicked on its own `assert_eq!(q.len(), num_v_heads *
//! head_v_dim)` before a single token was produced.
//!
//! The reference semantics (HF `modeling_qwen3_next.py::GatedDeltaNet`, llama.cpp
//! `delta-net-base.cpp`) are a `repeat_interleave` on the head axis: value head
//! `h` reads key/query head `h / (num_v_heads / num_k_heads)`. The first test
//! below pins exactly that, against a host reference that literally expands q and
//! k and then calls the old `num_k_heads == num_v_heads` code path — so the
//! mapping is fixed independently of any model file, and independently of the
//! generalised implementation's own arithmetic.

use super::{delta_rule_recurrence, delta_rule_recurrence_gqa};

/// A deterministic, dependency-free value stream: an LCG mapped into [-1, 1).
/// Fixtures must be reproducible byte for byte — two runs of the same test have
/// to compare the same numbers, and the GPU worker mirroring this mapping needs
/// to be able to regenerate them.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32;
        (f32::from(u16::try_from(bits & 0xFFFF).unwrap_or(0)) / 32768.0) - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// `repeat_interleave(x, ratio)` on the head axis: head `h` of the output is head
/// `h / ratio` of the input. This is the host reference for the mapping — it is
/// the definition, not a re-derivation of the implementation.
fn repeat_interleave_heads(x: &[f32], head_dim: usize, ratio: usize) -> Vec<f32> {
    let num_heads = x.len() / head_dim;
    let mut out = Vec::with_capacity(num_heads * ratio * head_dim);
    for h in 0..num_heads * ratio {
        let src = h / ratio;
        out.extend_from_slice(&x[src * head_dim..(src + 1) * head_dim]);
    }
    out
}

/// The same expansion under the WRONG (strided / `h % num_k_heads`) mapping. Used
/// only to prove the fixture discriminates: if the two expansions produced the
/// same recurrence output, the test above would pass for a broken implementation.
fn strided_heads(x: &[f32], head_dim: usize, num_k_heads: usize, num_v_heads: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_v_heads * head_dim);
    for h in 0..num_v_heads {
        let src = h % num_k_heads;
        out.extend_from_slice(&x[src * head_dim..(src + 1) * head_dim]);
    }
    out
}

/// GQA in the recurrence: with `num_k_heads = 2`, `num_v_heads = 4`, `D = 8`, the
/// generalised recurrence must equal the old `num_k_heads == num_v_heads` code
/// path fed `repeat_interleave`d q and k — on the OUTPUT and on the recurrent
/// STATE the token leaves behind, which is what the next token reads.
///
/// Held at exact equality: the generalised loop performs the same additions in
/// the same order on the same values, so any difference is a semantic one.
#[test]
fn qwen35_gqa_recurrence_is_repeat_interleave_of_q_and_k() {
    let (num_k_heads, num_v_heads, d) = (2usize, 4usize, 8usize);
    let ratio = num_v_heads / num_k_heads;

    let mut rng = Lcg::new(0x3477);
    let q = rng.vec(num_k_heads * d);
    let k = rng.vec(num_k_heads * d);
    let v = rng.vec(num_v_heads * d);
    let beta = rng.vec(num_v_heads);
    let gate: Vec<f32> = rng.vec(num_v_heads).iter().map(|g| g * 0.1).collect();
    let state0 = rng.vec(num_v_heads * d * d);

    let mut state_gqa = state0.clone();
    let mut out_gqa = vec![0.0f32; num_v_heads * d];
    delta_rule_recurrence_gqa(
        &q,
        &k,
        &v,
        &beta,
        &gate,
        &mut state_gqa,
        &mut out_gqa,
        num_k_heads,
        d,
        num_v_heads,
        d,
    );

    let q_exp = repeat_interleave_heads(&q, d, ratio);
    let k_exp = repeat_interleave_heads(&k, d, ratio);
    let mut state_ref = state0.clone();
    let mut out_ref = vec![0.0f32; num_v_heads * d];
    delta_rule_recurrence(
        &q_exp,
        &k_exp,
        &v,
        &beta,
        &gate,
        &mut state_ref,
        &mut out_ref,
        num_v_heads,
        d,
    );

    assert_eq!(
        out_gqa, out_ref,
        "the GQA recurrence output is not the repeat_interleave reference"
    );
    assert_eq!(
        state_gqa, state_ref,
        "the GQA recurrence left a different recurrent state than the repeat_interleave reference"
    );

    // Anti-vacuity: the strided mapping (h % num_k_heads), the other obvious
    // reading of "share the key heads", must give a DIFFERENT answer — otherwise
    // the assertions above would hold for an implementation that got the mapping
    // backwards.
    let q_strided = strided_heads(&q, d, num_k_heads, num_v_heads);
    let k_strided = strided_heads(&k, d, num_k_heads, num_v_heads);
    let mut state_strided = state0;
    let mut out_strided = vec![0.0f32; num_v_heads * d];
    delta_rule_recurrence(
        &q_strided,
        &k_strided,
        &v,
        &beta,
        &gate,
        &mut state_strided,
        &mut out_strided,
        num_v_heads,
        d,
    );
    assert_ne!(
        out_strided, out_ref,
        "the fixture does not discriminate: the strided head mapping gives the same output as \
         repeat_interleave, so this test would pass for a wrong mapping"
    );
}

/// `ratio == 1` (0.8B and 2B: `num_k_heads == num_v_heads == 16`) must be BYTE
/// IDENTICAL to the pre-GQA recurrence. The GPU parity tests on the real 0.8B
/// file are the end-to-end regression guard; this is the unit-level one.
#[test]
fn qwen35_gqa_ratio_one_is_byte_identical_to_the_legacy_recurrence() {
    let (num_heads, d) = (3usize, 4usize);
    let mut rng = Lcg::new(0x0808);
    let q = rng.vec(num_heads * d);
    let k = rng.vec(num_heads * d);
    let v = rng.vec(num_heads * d);
    let beta = rng.vec(num_heads);
    let gate: Vec<f32> = rng.vec(num_heads).iter().map(|g| g * 0.1).collect();
    let state0 = rng.vec(num_heads * d * d);

    let mut state_new = state0.clone();
    let mut out_new = vec![0.0f32; num_heads * d];
    delta_rule_recurrence_gqa(
        &q,
        &k,
        &v,
        &beta,
        &gate,
        &mut state_new,
        &mut out_new,
        num_heads,
        d,
        num_heads,
        d,
    );

    let mut state_old = state0;
    let mut out_old = vec![0.0f32; num_heads * d];
    delta_rule_recurrence(
        &q,
        &k,
        &v,
        &beta,
        &gate,
        &mut state_old,
        &mut out_old,
        num_heads,
        d,
    );

    assert_eq!(out_new, out_old, "ratio 1 changed the output bit pattern");
    assert_eq!(state_new, state_old, "ratio 1 changed the recurrent state");
}

/// A rectangular state (`head_k_dim != head_v_dim`) must be indexed as
/// `S[i][j] = state[h * head_v_dim * head_k_dim + j * head_k_dim + i]`, `i` over
/// the key dim and `j` over the value dim. Every Qwen3.5 size ships
/// `head_k_dim == head_v_dim == 128`, so nothing in the fleet exercises the two
/// dims being distinct — which is exactly why it is pinned here rather than left
/// to a future file to discover. The oracle is arithmetic done by hand on a
/// one-head, `k_dim = 2`, `v_dim = 3` case with an identity-shaped state.
#[test]
fn qwen35_gqa_state_is_rectangular_k_dim_by_v_dim() {
    let (head_k_dim, head_v_dim) = (2usize, 3usize);
    let q = [1.0f32, 2.0];
    let k = [0.5f32, 0.5];
    let v = [1.0f32, -1.0, 2.0];
    let beta = [0.5f32];
    let gate = [0.0f32]; // exp(0) = 1
                         // S is [2 x 3] (i over k_dim, j over v_dim); memory row j is column j of S.
    let mut state = [
        1.0f32, 0.0, // j = 0: S[0][0], S[1][0]
        0.0, 1.0, // j = 1
        1.0, 1.0, // j = 2
    ];
    let mut output = [0.0f32; 3];
    delta_rule_recurrence_gqa(
        &q,
        &k,
        &v,
        &beta,
        &gate,
        &mut state,
        &mut output,
        1,
        head_k_dim,
        1,
        head_v_dim,
    );

    // S^T k = [0.5, 0.5, 1.0]; delta = (v - S^T k) * 0.5 = [0.25, -0.75, 0.5]
    // S += k delta^T  ->  row j gets k * delta[j]:
    //   j=0: [1.125, 0.125]   j=1: [-0.375, 0.625]   j=2: [1.25, 1.25]
    // out[j] = dot(row j, q) / sqrt(head_k_dim)
    let s2 = 2.0f32.sqrt();
    let want = [
        (1.125 + 2.0 * 0.125) / s2,
        (-0.375 + 2.0 * 0.625) / s2,
        (1.25 + 2.0 * 1.25) / s2,
    ];
    for (j, w) in want.iter().enumerate() {
        assert!(
            (output[j] - w).abs() < 1e-6,
            "output[{j}] = {}, want {w} (state: {state:?})",
            output[j]
        );
    }
    assert!((state[0] - 1.125).abs() < 1e-6, "{state:?}");
    assert!((state[1] - 0.125).abs() < 1e-6, "{state:?}");
    assert!((state[4] - 1.25).abs() < 1e-6, "{state:?}");
    assert!((state[5] - 1.25).abs() < 1e-6, "{state:?}");
}

/// The real 4B file (`linear_num_value_heads = 32`, `linear_num_key_heads = 16`):
/// load it through the production path and run three tokens of a fixed prompt
/// through `forward_single_qwen35`. Before this ticket this panicked in the
/// recurrence's shape assertion at the first `DeltaNet` layer of the first token.
///
/// Asserts the loader's own head shape (a `num_v_heads == num_k_heads` file would
/// make the test vacuous), that every logit is finite, and that the argmax is a
/// real vocabulary entry. The argmax is PRINTED, not asserted against a golden:
/// pinning a golden token belongs with a tokenizer-level e2e test, and the
/// acceptance command (`apr run --no-gpu`) is what reads the text.
#[test]
fn qwen35_gqa_4b_real_file_runs_three_tokens() {
    const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-4B-Q4_K_M.gguf";
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return;
    }
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the 4B GGUF");
    let base =
        super::Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("4B base model");
    let qwen = super::Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())
        .expect("4B hybrid layers");

    assert!(
        qwen.num_v_heads > qwen.num_k_heads,
        "this file is not a GQA DeltaNet file (num_v_heads {} vs num_k_heads {}), so it cannot \
         falsify the head mapping",
        qwen.num_v_heads,
        qwen.num_k_heads
    );
    assert_eq!(
        qwen.num_v_heads % qwen.num_k_heads,
        0,
        "num_v_heads {} is not a multiple of num_k_heads {}",
        qwen.num_v_heads,
        qwen.num_k_heads
    );

    // The per-head value width the gated RMSNorm weight actually carries must be
    // the head_v_dim the recurrence and the state are sized with: a mismatch here
    // is a silently mis-shaped forward, not a crash.
    if let Some(super::Qwen35OwnedLayer::DeltaNet(d)) = qwen
        .layers
        .iter()
        .find(|l| matches!(l, super::Qwen35OwnedLayer::DeltaNet(_)))
    {
        assert_eq!(
            d.ssm_norm_weight.len(),
            qwen.head_v_dim,
            "ssm_norm_weight is {} wide but head_v_dim is {}",
            d.ssm_norm_weight.len(),
            qwen.head_v_dim
        );
        assert_eq!(
            d.ssm_a.len(),
            qwen.num_v_heads,
            "ssm_a (A) is per value head"
        );
    }

    // "The capital of France is" under the Qwen BPE vocabulary. No golden is
    // asserted on the output, so a drifted id costs nothing but a less
    // interesting print.
    let prompt = [785u32, 6722, 315, 9625, 374];
    let mut state = qwen.new_state(prompt.len() + 4);
    let mut logits = Vec::new();
    for (pos, &token) in prompt.iter().take(3).enumerate() {
        logits = qwen
            .forward_single_qwen35(token, &mut state, pos)
            .expect("4B forward");
        assert_eq!(logits.len(), qwen.base.config.vocab_size);
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "position {pos}: the 4B forward produced a non-finite logit"
        );
    }
    let argmax = crate::gguf::ops::argmax(&logits);
    assert!(
        (argmax as usize) < qwen.base.config.vocab_size,
        "argmax {argmax} is outside the vocabulary"
    );
    println!(
        "qwen35 4B (num_k_heads {}, num_v_heads {}, head_k_dim {}, head_v_dim {}): argmax after \
         3 tokens = {argmax}",
        qwen.num_k_heads, qwen.num_v_heads, qwen.head_k_dim, qwen.head_v_dim
    );
}
