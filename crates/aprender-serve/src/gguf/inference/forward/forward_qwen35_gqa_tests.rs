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
//! The mapping a **GGUF** file carries is `ggml`'s TILED broadcast: value head
//! `h` reads key/query head `h % num_k_heads`. Source, in the order it settles
//! the question:
//!
//! * `llama.cpp/conversion/qwen.py:455-464` — `_LinearAttentionVReorderBase`,
//!   the class `Qwen3_5TextModel` (line 639) is built from: *"reorders V heads
//!   from grouped to tiled order for ggml broadcast … The HF weights store V
//!   heads grouped by K head: `[G0_v0..v{r-1}, G1_v0..v{r-1}, …]`. ggml binary
//!   ops use tiled broadcast … We reorder V heads to tiled order"*. Every
//!   v-head-indexed tensor is permuted together (the v rows of `in_proj_qkv`,
//!   `in_proj_z`, `in_proj_a`/`_b`, `A_log`/`dt_bias`, the v channels of
//!   `conv1d`, the v columns of `out_proj` — `modify_tensors`, lines 568-613).
//! * `llama.cpp/src/models/qwen35.cpp:436-441` — when `num_k_heads !=
//!   num_v_heads` the graph expands q and k with `ggml_repeat_4d`, which tiles
//!   (`h % num_k_heads`), not `repeat_interleave`.
//! * `llama.cpp/ggml/src/ggml-cpu/ops.cpp:10976-10977` — the fused kernel that
//!   skips that repeat indexes `iq1 = iv1 % neq1; ik1 = iv1 % nek1;`.
//!
//! So HF's `repeat_interleave` (`h / ratio`) is the right reading of the HF
//! *checkpoint*, and the wrong reading of the *file we load*: the conversion has
//! already permuted the value heads into tiled order. The first test below pins
//! the tiled mapping against a host reference that literally expands q and k and
//! then calls the `num_k_heads == num_v_heads` code path — so the mapping is
//! fixed independently of any model file, and independently of the generalised
//! implementation's own arithmetic.

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

/// `ggml_repeat` on the head axis: head `h` of the output is head
/// `h % num_k_heads` of the input. This is the host reference for the mapping —
/// it is the definition (`conversion/qwen.py` writes the value heads in exactly
/// this order so that `ggml_repeat_4d` is correct), not a re-derivation of the
/// implementation.
fn tiled_heads(x: &[f32], head_dim: usize, num_k_heads: usize, num_v_heads: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_v_heads * head_dim);
    for h in 0..num_v_heads {
        let src = h % num_k_heads;
        out.extend_from_slice(&x[src * head_dim..(src + 1) * head_dim]);
    }
    out
}

/// The same expansion under the HF-checkpoint (`repeat_interleave` / `h / ratio`)
/// mapping, which a GGUF file does NOT carry. Used only to prove the fixture
/// discriminates: if the two expansions produced the same recurrence output, the
/// test above would pass for a broken implementation.
fn repeat_interleave_heads(x: &[f32], head_dim: usize, ratio: usize) -> Vec<f32> {
    let num_heads = x.len() / head_dim;
    let mut out = Vec::with_capacity(num_heads * ratio * head_dim);
    for h in 0..num_heads * ratio {
        let src = h / ratio;
        out.extend_from_slice(&x[src * head_dim..(src + 1) * head_dim]);
    }
    out
}

/// GQA in the recurrence: with `num_k_heads = 2`, `num_v_heads = 4`, `D = 8`, the
/// generalised recurrence must equal the `num_k_heads == num_v_heads` code path
/// fed TILED q and k (`ggml_repeat`, `h % num_k_heads` — see the module header
/// for the three source lines) — on the OUTPUT and on the recurrent STATE the
/// token leaves behind, which is what the next token reads.
///
/// Held at exact equality: the generalised loop performs the same additions in
/// the same order on the same values, so any difference is a semantic one.
#[test]
fn qwen35_gqa_recurrence_tiles_q_and_k_over_the_value_heads() {
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

    let q_exp = tiled_heads(&q, d, num_k_heads, num_v_heads);
    let k_exp = tiled_heads(&k, d, num_k_heads, num_v_heads);
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
        "the GQA recurrence output is not the tiled (ggml_repeat) reference"
    );
    assert_eq!(
        state_gqa, state_ref,
        "the GQA recurrence left a different recurrent state than the tiled reference"
    );

    // Anti-vacuity: the HF-checkpoint mapping (repeat_interleave, h / ratio) —
    // the other obvious reading of "share the key heads", and the one the
    // conversion permutes AWAY — must give a DIFFERENT answer, otherwise the
    // assertions above would hold for an implementation that got the mapping
    // backwards.
    let q_interleaved = repeat_interleave_heads(&q, d, ratio);
    let k_interleaved = repeat_interleave_heads(&k, d, ratio);
    let mut state_interleaved = state0;
    let mut out_interleaved = vec![0.0f32; num_v_heads * d];
    delta_rule_recurrence(
        &q_interleaved,
        &k_interleaved,
        &v,
        &beta,
        &gate,
        &mut state_interleaved,
        &mut out_interleaved,
        num_v_heads,
        d,
    );
    assert_ne!(
        out_interleaved, out_ref,
        "the fixture does not discriminate: repeat_interleave gives the same output as the tiled \
         mapping, so this test would pass for a wrong mapping"
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
/// load it through the production path and take FOUR greedy tokens of a fixed
/// prompt through `forward_single_qwen35`, against the tokens llama.cpp produces
/// for the same raw prompt. Before this ticket this panicked in the recurrence's
/// shape assertion at the first `DeltaNet` layer of the first token.
///
/// The reference (llama.cpp master `60b06ab9a`, CPU build, this box):
///
/// ```text
/// llama-completion -m /home/noah/models/Qwen3.5-4B-Q4_K_M.gguf \
///     -p "The capital of France is" -n 8 --temp 0 -ngl 0 -no-cnv --seed 0 -t 16
/// The capital of France is Paris.
/// A. True
/// B
/// ```
///
/// `-no-cnv` is load-bearing: without it llama.cpp applies the model's chat
/// template, and the comparison is then against a different prompt. (So does
/// `apr run`, which is why a CLI-level A/B against this reference compares two
/// different prompts and cannot settle anything.) The eight continuation ids
/// under this file's own vocabulary (248320 entries — NOT the Qwen2/Qwen3
/// vocabulary; `The` is 760 here and 785 there) are `[11751 " Paris", 13 ".",
/// 198 "\n", 32 "A", 13 ".", 2912 " True", 198 "\n", 33 "B"]`, and all eight are
/// asserted.
#[test]
fn qwen35_gqa_4b_matches_llama_cpp_greedy_tokens() {
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

    // The loader must take the depth from the file, not from a default: 4B is 32 blocks,
    // and a silently-24-deep stack is exactly what produced finite logits and drifting text.
    assert_eq!(
        qwen.layers.len(),
        mapped
            .model
            .num_layers()
            .expect("4B GGUF carries qwen35.block_count"),
        "the hybrid loader built {} layers for a {:?}-block file",
        qwen.layers.len(),
        mapped.model.num_layers()
    );

    // "The capital of France is" under THIS file's vocabulary.
    let prompt = [760u32, 6511, 314, 9338, 369];
    const WANT: [u32; 8] = [11751, 13, 198, 32, 13, 2912, 198, 33];

    let mut state = qwen.new_state(prompt.len() + WANT.len());
    let mut logits = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        logits = qwen
            .forward_single_qwen35(token, &mut state, pos)
            .expect("4B forward");
        assert_eq!(logits.len(), qwen.base.config.vocab_size);
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "position {pos}: the 4B forward produced a non-finite logit"
        );
    }

    let mut got = Vec::with_capacity(WANT.len());
    for step in 0..WANT.len() {
        let next = crate::gguf::ops::argmax(&logits);
        got.push(next);
        logits = qwen
            .forward_single_qwen35(next, &mut state, prompt.len() + step)
            .expect("4B forward");
    }

    assert_eq!(
        got,
        WANT.to_vec(),
        "the 4B CPU forward does not follow llama.cpp greedily (head mapping / Gated DeltaNet \
         arithmetic): got {got:?}, want {:?} — num_k_heads {}, num_v_heads {}, head_k_dim {}, \
         head_v_dim {}",
        WANT,
        qwen.num_k_heads,
        qwen.num_v_heads,
        qwen.head_k_dim,
        qwen.head_v_dim
    );
}

/// CONTROL: the same comparison on 0.8B (`num_k_heads == num_v_heads == 16`), a
/// file the CPU forward already served before this ticket. It shares every line
/// of the hybrid stack with 4B except the head mapping, so a RED here would mean
/// the 4B failure is not a GQA defect at all.
///
/// Reference (llama.cpp master `60b06ab9a`, CPU build, this box):
///
/// ```text
/// llama-completion -m /home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf \
///     -p "The capital of France is" -n 8 --temp 0 -ngl 0 -no-cnv --seed 0 -t 16
/// The capital of France is the capital of the country.
/// The
/// ```
///
/// Only the first FOUR ids are asserted, and deliberately so: this prompt is a
/// near-tie on 0.8B and the reference itself moved. A January llama.cpp build on
/// another box answers `[7172 " located", 303 " in", 279 " the", 9514 " south"]`
/// for the same file and the same prompt, where master answers `[279 " the",
/// 6511 " capital", 314 " of", 279 " the"]`. A RED here is therefore only
/// evidence of a defect when it is reproduced against the llama.cpp build named
/// above; the load-bearing 0.8B guard is the CUDA `forward_single` parity test,
/// which compares apr to apr and cannot drift with an upstream build.
#[test]
fn qwen35_0_8b_matches_llama_cpp_greedy_tokens() {
    const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return;
    }
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the 0.8B GGUF");
    let base = super::Qwen35Model::create_base_model(&mapped.model, mapped.data())
        .expect("0.8B base model");
    let qwen = super::Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())
        .expect("0.8B hybrid layers");
    assert_eq!(
        qwen.num_v_heads, qwen.num_k_heads,
        "0.8B is supposed to be the ratio-1 control"
    );

    let prompt = [760u32, 6511, 314, 9338, 369];
    const WANT: [u32; 4] = [279, 6511, 314, 279]; // " the" " capital" " of" " the"

    let mut state = qwen.new_state(prompt.len() + WANT.len());
    let mut logits = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        logits = qwen
            .forward_single_qwen35(token, &mut state, pos)
            .expect("0.8B forward");
    }
    let mut got = Vec::with_capacity(WANT.len());
    for step in 0..WANT.len() {
        let next = crate::gguf::ops::argmax(&logits);
        got.push(next);
        logits = qwen
            .forward_single_qwen35(next, &mut state, prompt.len() + step)
            .expect("0.8B forward");
    }
    assert_eq!(
        got,
        WANT.to_vec(),
        "the 0.8B CPU forward does not follow llama.cpp greedily: got {got:?}"
    );
}
