//! PMAT-3477 / aprender#3090: CPU parity for [`Qwen35CudaModel`] on the real
//! Qwen3.5-0.8B file, at three scopes that must not be confused.
//!
//! 1. [`qwen35_cuda_gdn_ops_match_cpu_given_the_same_inputs`] — the six Gated
//!    `DeltaNet` kernels, each fed the GPU's OWN input and compared against the
//!    CPU reference function on that same input, at **1e-3 relative**. This is
//!    the kernel contract, and it is what the beta-sign mutant falsifies.
//! 2. [`qwen35_cuda_deltanet_layers_match_cpu_on_the_real_file`] — the whole
//!    wired layer, every `DeltaNet` layer, three tokens, teacher-forced. This
//!    one also runs the **projection GEMVs**, and those are a separate,
//!    pre-existing CPU/GPU approximation: measured op by op on layer 0 (the
//!    diagnostic that produced these numbers is the op-by-op test above),
//!
//!    | stage | qtype | relative L∞ |
//!    |-------|-------|-------------|
//!    | `rms_norm` -> `normed` | f32 | 9.1e-8 |
//!    | `attn_qkv` GEMV | Q5_K | 6.6e-7 |
//!    | conv+SiLU+per-head L2 | — | 6.0e-7 |
//!    | `ssm_alpha` GEMV | Q8_0 | **5.3e-3** |
//!    | `ssm_beta` GEMV | Q8_0 | **2.4e-3** |
//!    | `attn_gate` GEMV | Q4_K | **4.8e-3** |
//!
//!    so the layer output cannot be inside 1e-3 and the GEMV kernels be what
//!    they are today. [`LAYER_TOL`] is the measured budget, not a wish; the
//!    kernels this ticket adds are held to [`TOL`].
//!
//! 3. the two WHOLE-FORWARD comparisons against the CPU's production forward
//!    ([`qwen35_cuda_attention_layers_match_cpu_on_the_real_file`] and
//!    [`qwen35_cuda_forward_single_matches_cpu_logits_end_to_end`]) — these
//!    cannot be held to a tight L∞ at all, because the reference is the noisier
//!    side (it quantizes its activation to Q8_K). They assert identical argmax
//!    + cosine >= [`COSINE_FLOOR`] + an L∞ BUDGET. Read [`COSINE_FLOOR`] before
//!    touching any number in this file: it carries the measurements, and the
//!    difference between a bar and a budget is the whole point.
//!
//! Every comparison is **scale-relative** L∞ — `max|gpu - cpu| <= tol *
//! max|cpu|` — and asserts `max|cpu| > 0` first. An absolute 1e-3 on these
//! tensors is vacuous: the delta-rule outputs are ~5e-3 (q and k are unit-norm
//! per head and the recurrence scales by `D^-0.5`), so the whole signal sits
//! inside the tolerance and even a 1 % mutation stays GREEN.

use super::Qwen35CudaModel;
use crate::gguf::forward_qwen35::{
    causal_conv1d, delta_rule_recurrence, gated_rmsnorm, l2_norm_per_head, silu, softplus,
    Qwen35Model, Qwen35OwnedDeltaNetLayer, Qwen35OwnedLayer,
};
use trueno_gpu::driver::GpuBuffer;

/// The real hybrid file this phase is specified against.
const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Tolerance for the Gated `DeltaNet` kernels themselves. They route exp/ln
/// through `ex2.approx`/`lg2.approx` (~2 ulp), so 1e-4 is not available; 1e-3
/// relative is the contract the kernel module states.
const TOL: f32 = 1e-3;

/// Tolerance for the whole wired layer, which also runs the Q4_K/Q8_0
/// projection GEMVs. See the module docs: those contribute ~5e-3 relative on
/// their own, before the layer's own arithmetic. Measured max over the three
/// tokens and every `DeltaNet` layer: 9.8e-3 with the float GEMV pinned.
const LAYER_TOL: f32 = 2e-2;

/// Tolerance for the K and V rows an attention layer appends, against the
/// EXACT-arithmetic reference ([`exact_gemv`] + the CPU's own norm and rope,
/// which are unquantized ops). One GEMV plus a per-head RMSNorm plus the
/// partial rope away from the layer input, with no quantized activation on
/// either side, so this is the kernel contract [`TOL`], not a budget.
const KV_TOL: f32 = TOL;

/// THE PARITY CONTRACT FOR A WHOLE-FORWARD COMPARISON (PMAT-3477 / #3090).
///
/// The per-op tests above are held to [`TOL`] against an EXACT reference
/// ([`exact_gemv`]) wherever the CPU production op quantizes its activation.
/// The two comparisons that run the CPU's own production forward —
/// [`qwen35_cuda_attention_layers_match_cpu_on_the_real_file`] and
/// [`qwen35_cuda_forward_single_matches_cpu_logits_end_to_end`] — cannot be, and
/// `qwen35_cuda_the_parity_floor_is_the_cpu_references_own_activation_quantization`
/// says why with numbers: on the same projection the GPU float GEMV is 9.96e-8
/// from exact arithmetic and the CPU `fused_matmul_into` is 2.95e-3, because it
/// quantizes the ACTIVATION to Q8_K. The CPU is the noisier side, so a tight
/// L∞ against it is not a correctness bar — it is a demand that the GPU
/// reproduce the reference's own rounding.
///
/// So those two tests assert three things instead, all of which can fail:
///
/// 1. **identical argmax at every position** — exact, and the only property a
///    greedy decode actually consumes;
/// 2. **cosine >= [`COSINE_FLOOR`]** — direction, which an error that is pure
///    GEMV rounding barely moves and a wrong kernel destroys;
/// 3. **relative L∞ <= [`LOGITS_BUDGET`] / [`LAYER_BUDGET`]** — a budget, held
///    as a regression tripwire. It is a reading, never a claim of exactness.
///
/// MEASURED, on the real 0.8B file with the float GEMV pinned — the tests print
/// every one of these lines, so the constants below are a reading:
///
/// | e2e position | 0 | 1 | 2 | 3 | 4 | 5 |
/// |---|---|---|---|---|---|---|
/// | cosine | **0.998348** | 0.999487 | 0.998936 | 0.999510 | 0.999309 | 0.999636 |
/// | relative L∞ | **6.721e-2** | 2.123e-2 | 3.490e-2 | 2.338e-2 | 3.065e-2 | 1.972e-2 |
/// | argmax (both sides) | 13 | 198 | 11 | 198 | 0 | 908 |
///
/// and over the whole-attention-layer comparison (18 comparisons: every
/// attention layer × three positions) the worst is layer 15 at position 0 —
/// cosine **0.997419**, relative L∞ **4.867e-2** — with every other comparison
/// at cosine ≥ 0.999407.
///
/// `COSINE_FLOOR` is the minimum over BOTH tests (0.997419) rounded DOWN to
/// three decimals — 0.997 — minus 0.001: **0.996**. That leaves ~1.4e-3 of
/// cosine margin under the worst comparison, and is still nowhere near what a
/// broken kernel produces: the DP4A falsifier below reads 0.466 down to
/// **-0.246**, so the assertion discriminates rather than decorates.
const COSINE_FLOOR: f32 = 0.996;

/// The measured end-to-end logit L∞ — 6.721e-2 relative, at position 0 —
/// rounded UP to one significant figure. A budget on the accumulated
/// CPU-reference activation quantization through 24 layers and the `lm_head`,
/// not a tolerance anyone should read as accuracy: see [`COSINE_FLOOR`].
const LOGITS_BUDGET: f32 = 7e-2;

/// The same, for one whole attention layer's output hidden state: measured
/// 4.867e-2 relative (layer 15, position 0), rounded up to one significant
/// figure.
const LAYER_BUDGET: f32 = 5e-2;

/// A fixed three-token prompt. Any ids work for parity — what matters is that
/// both sides see the same ones.
const PROMPT: [u32; 3] = [9707, 11, 1879];

/// A fixed six-token prompt for the end-to-end comparison.
const LONG_PROMPT: [u32; 6] = [9707, 11, 1879, 0, 2610, 525];

/// `max|want|`, the scale the tolerance is relative to.
fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// `max|got - want| / max|want|` — the scale-relative L∞ every assertion here
/// is written in.
fn rel_linf(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "rel_linf: length mismatch");
    let scale = max_abs(want);
    assert!(scale > 0.0, "rel_linf: the reference is all zeros");
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

/// Assert `max|got - want| <= tol * max|want|`, refusing a vacuous comparison.
fn assert_rel_linf(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let scale = max_abs(want);
    assert!(
        scale > 0.0,
        "{what}: the reference is all zeros — the comparison would pass on anything"
    );
    let (mut worst, mut at) = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol * scale,
        "{what}: relative L-inf {:.3e} (abs {:.3e} at [{at}], scale {:.3e}) exceeds {tol:.0e}; \
         cpu={:.6} gpu={:.6}",
        worst / scale,
        worst,
        scale,
        want[at],
        got[at],
    );
}

/// Cosine similarity, accumulated in f64 so the metric itself is not the thing
/// being measured.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch");
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    assert!(
        na > 0.0 && nb > 0.0,
        "cosine: a zero vector — the comparison would be undefined"
    );
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// The EXACT reference for one quantized projection: the weight dequantized to
/// f32, the row dots accumulated in **f64**.
///
/// This is the third reference the parity contract rests on. The CPU
/// production op (`fused_matmul_into`) is NOT it: it quantizes the activation
/// to Q8_K and takes int8 dots, which costs ~3e-3 relative all by itself
/// (measured in the parity-floor test). Any comparison whose reference side
/// runs a quantized GEMV must come here instead, or it measures the reference.
///
/// Panics on a qtype it has no dequantizer for, rather than silently comparing
/// against something else.
fn exact_gemv(tensor: &crate::gguf::quantized::OwnedQuantizedTensor, x: &[f32]) -> Vec<f32> {
    use crate::gguf::types::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0};
    let w = match tensor.qtype {
        GGUF_TYPE_Q4_K => crate::quantize::dequantize_q4_k(&tensor.data),
        GGUF_TYPE_Q5_K => crate::quantize::dequantize_q5_k(&tensor.data),
        GGUF_TYPE_Q6_K => crate::quantize::dequantize_q6_k(&tensor.data),
        GGUF_TYPE_Q8_0 => crate::quantize::dequantize_q8_0(&tensor.data),
        other => panic!("exact_gemv: no dequantizer for GGML type {other}"),
    }
    .expect("dequantize the projection");
    assert_eq!(
        x.len(),
        tensor.in_dim,
        "exact_gemv: the input is not in_dim wide"
    );
    assert!(
        w.len() >= tensor.out_dim * tensor.in_dim,
        "exact_gemv: the dequantized weight is short of out_dim x in_dim"
    );
    (0..tensor.out_dim)
        .map(|r| {
            let row = &w[r * tensor.in_dim..(r + 1) * tensor.in_dim];
            row.iter()
                .zip(x)
                .fold(0.0f64, |s, (wv, xv)| s + f64::from(*wv) * f64::from(*xv)) as f32
        })
        .collect()
}

/// The whole-forward parity contract of [`COSINE_FLOOR`], applied at one
/// position: exact argmax equality, cosine above the floor, L∞ inside the
/// budget. Returns `(cosine, relative L∞)` so the caller can print the reading
/// the constants were set from.
fn assert_forward_parity(got: &[f32], want: &[f32], budget: f32, what: &str) -> (f32, f32) {
    let cos = cosine(got, want);
    let linf = rel_linf(got, want);
    assert_eq!(
        crate::gguf::ops::argmax(got),
        crate::gguf::ops::argmax(want),
        "{what}: the GPU and the CPU must pick the same index (gpu {} vs cpu {}, cosine \
         {cos:.6}, relative L-inf {linf:.3e})",
        crate::gguf::ops::argmax(got),
        crate::gguf::ops::argmax(want),
    );
    assert!(
        cos >= COSINE_FLOOR,
        "{what}: cosine {cos:.6} is below the measured floor {COSINE_FLOOR} — the two sides \
         point in different directions, which GEMV rounding does not do"
    );
    assert!(
        linf <= budget,
        "{what}: relative L-inf {linf:.3e} exceeds the measured budget {budget:.0e}"
    );
    (cos, linf)
}

/// Skip unless both the device and the model file are here.
macro_rules! qwen35_cuda_fixture_or_skip {
    () => {{
        if !std::path::Path::new(MODEL_PATH).exists() {
            eprintln!("SKIP: {MODEL_PATH} is absent");
            return;
        }
        crate::cuda_executor_or_skip!(0)
    }};
}

/// Map the file and build the CPU model.
fn load_cpu_model(mapped: &crate::gguf::MappedGGUFModel) -> crate::gguf::OwnedQuantizedModel {
    Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model")
}

/// Per-layer parity of the whole wired Gated `DeltaNet` GPU forward against the
/// CPU reference, over the first three tokens of a fixed prompt, on EVERY
/// `DeltaNet` layer, with teacher forcing — on the layer output hidden state AND
/// on the causal-conv window and the recurrent state the token leaves behind.
#[test]
#[serial_test::serial]
fn qwen35_cuda_deltanet_layers_match_cpu_on_the_real_file() {
    let executor = qwen35_cuda_fixture_or_skip!();

    // Load exactly as run_qwen35_generate does.
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");

    let deltanet_layers = qwen
        .layers
        .iter()
        .filter(|l| matches!(l, Qwen35OwnedLayer::DeltaNet(_)))
        .count();
    assert!(
        deltanet_layers > 0,
        "the fixture must carry Gated DeltaNet layers, else this test proves nothing"
    );

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    // The float (non-DP4A) GEMV is the one that reproduces the CPU's float
    // dequant; DP4A quantizes the activation to int8 first and costs another
    // ~1e-3 on top (1.1e-2 vs 9.8e-3 measured end to end).
    gpu.pin_reference_gemv();

    let hidden_dim = qwen.base.config.hidden_dim;
    let mut cpu_state = qwen.new_state(PROMPT.len() + 1);
    let mut normed = vec![0.0f32; hidden_dim];
    let mut post_normed = vec![0.0f32; hidden_dim];
    let mut compared = 0usize;

    for (pos, &token) in PROMPT.iter().enumerate() {
        let mut hidden = qwen.base.token_embedding()
            [(token as usize) * hidden_dim..(token as usize + 1) * hidden_dim]
            .to_vec();

        for (il, layer) in qwen.layers.iter().enumerate() {
            match layer {
                Qwen35OwnedLayer::DeltaNet(d) => {
                    // Teacher forcing: the GPU starts this layer from the CPU's
                    // conv window and recurrent state, and from the same hidden.
                    gpu.upload_layer(il, &cpu_state.conv_states[il], &cpu_state.ssm_states[il])
                        .expect("seed the device state");
                    let dev = GpuBuffer::from_host(gpu.executor_mut().context(), &hidden)
                        .expect("upload hidden");

                    qwen.forward_deltanet(
                        d,
                        &mut hidden,
                        &mut cpu_state,
                        il,
                        pos,
                        &mut normed,
                        &mut post_normed,
                    )
                    .expect("cpu deltanet");

                    gpu.forward_deltanet_layer(il, &dev).expect("gpu deltanet");
                    gpu.executor_mut().sync_stream().expect("sync");
                    let mut got = vec![0.0f32; hidden_dim];
                    dev.copy_to_host(&mut got).expect("download hidden");

                    assert_rel_linf(
                        &got,
                        &hidden,
                        LAYER_TOL,
                        &format!("pos {pos} layer {il} hidden"),
                    );

                    let (conv, ssm) = gpu.download_layer(il).expect("download state");
                    // The conv window is downstream of the Q5_K attn_qkv GEMV
                    // only, which is exact to 1e-7 — hold it to the kernel
                    // tolerance, not the layer budget.
                    assert_rel_linf(
                        &conv,
                        &cpu_state.conv_states[il],
                        TOL,
                        &format!("pos {pos} layer {il} conv window"),
                    );
                    assert_rel_linf(
                        &ssm,
                        &cpu_state.ssm_states[il],
                        LAYER_TOL,
                        &format!("pos {pos} layer {il} ssm state"),
                    );
                    compared += 1;
                },
                Qwen35OwnedLayer::Attention(a) => {
                    // Not on the GPU in this phase (#3090 phase 2) — the CPU
                    // advances the hidden state and the KV cache so the next
                    // DeltaNet layer sees the real input.
                    qwen.forward_attention(
                        a,
                        &mut hidden,
                        &mut cpu_state,
                        il,
                        pos,
                        &mut normed,
                        &mut post_normed,
                    )
                    .expect("cpu attention");
                },
            }
        }
        cpu_state.kv_cache.advance();
    }

    assert_eq!(
        compared,
        deltanet_layers * PROMPT.len(),
        "every DeltaNet layer must be compared at every position"
    );
}

/// The six Gated `DeltaNet` kernels, each against the CPU reference function on
/// the GPU's OWN input, at 1e-3 relative.
///
/// Feeding the CPU the GPU's inputs is what makes this a kernel test: the
/// projection GEMVs are upstream of every one of these and carry ~5e-3 of their
/// own (module docs), so comparing "the CPU's out_h" against "the GPU's out_h"
/// would measure the GEMVs and call it the delta rule.
#[test]
#[serial_test::serial]
#[allow(clippy::too_many_lines)]
fn qwen35_cuda_gdn_ops_match_cpu_given_the_same_inputs() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let il = qwen
        .layers
        .iter()
        .position(|l| matches!(l, Qwen35OwnedLayer::DeltaNet(_)))
        .expect("a DeltaNet layer");
    let Qwen35OwnedLayer::DeltaNet(d) = &qwen.layers[il] else {
        unreachable!("just matched")
    };

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    gpu.pin_reference_gemv();

    let hidden_dim = qwen.base.config.hidden_dim;
    let eps = qwen.base.config.eps;
    let k_dim = qwen.head_k_dim * qwen.num_k_heads;
    let v_dim = qwen.head_v_dim * qwen.num_v_heads;
    let conv_dim = k_dim * 2 + v_dim;
    let mut cpu_state = qwen.new_state(PROMPT.len() + 1);
    let mut normed = vec![0.0f32; hidden_dim];
    let mut post_normed = vec![0.0f32; hidden_dim];

    for (pos, &token) in PROMPT.iter().enumerate() {
        // Positions 1 and 2 run against a NON-ZERO conv window and recurrent
        // state — a delta rule that ignored its state would pass at pos 0.
        let conv0 = cpu_state.conv_states[il].clone();
        let ssm0 = cpu_state.ssm_states[il].clone();
        assert_eq!(
            pos > 0,
            max_abs(&ssm0) > 0.0,
            "pos {pos}: the recurrent state must be non-zero after the first token"
        );

        let mut hidden = qwen.base.token_embedding()
            [(token as usize) * hidden_dim..(token as usize + 1) * hidden_dim]
            .to_vec();
        gpu.upload_layer(il, &conv0, &ssm0).expect("seed");
        let dev =
            GpuBuffer::from_host(gpu.executor_mut().context(), &hidden).expect("upload hidden");
        gpu.forward_deltanet_layer(il, &dev).expect("gpu deltanet");

        let conv_in_g = gpu.dump_stage("conv_in");
        let conv_out_g = gpu.dump_stage("conv_out");
        let alpha_g = gpu.dump_stage("alpha_raw");
        let beta_raw_g = gpu.dump_stage("beta_raw");
        let dt_g = gpu.dump_stage("dt");
        let beta_g = gpu.dump_stage("beta");
        let gate_g = gpu.dump_stage("gate");
        let out_h_g = gpu.dump_stage("out_h");
        let ssm_out_in_g = gpu.dump_stage("ssm_out_in");
        let (conv_state_g, ssm_state_g) = gpu.download_layer(il).expect("download state");

        // --- causal_conv1d + SiLU + per-head L2, from the GPU's own conv input.
        let mut conv_state_c = conv0.clone();
        let mut conv_out_c = vec![0.0f32; conv_dim];
        causal_conv1d(
            &conv_in_g,
            &mut conv_state_c[..],
            &d.ssm_conv1d_weight,
            qwen.conv_kernel,
            conv_dim,
            &mut conv_out_c,
        );
        for x in &mut conv_out_c {
            *x = silu(*x);
        }
        l2_norm_per_head(&mut conv_out_c[0..k_dim], qwen.head_k_dim, eps);
        l2_norm_per_head(&mut conv_out_c[k_dim..k_dim * 2], qwen.head_k_dim, eps);
        assert_rel_linf(
            &conv_out_g,
            &conv_out_c,
            TOL,
            &format!("pos {pos}: conv1d+SiLU+per-head L2"),
        );
        assert_rel_linf(
            &conv_state_g,
            &conv_state_c,
            TOL,
            &format!("pos {pos}: conv window after the shift"),
        );

        // --- the dt / beta gates, from the GPU's own alpha and beta_raw.
        let dt_c: Vec<f32> = alpha_g
            .iter()
            .enumerate()
            .map(|(i, a)| softplus(a + d.ssm_dt_bias[i]) * d.ssm_a[i])
            .collect();
        let beta_c: Vec<f32> = beta_raw_g
            .iter()
            .map(|b| 1.0 / (1.0 + (-b).exp()))
            .collect();
        assert_rel_linf(&dt_g, &dt_c, TOL, &format!("pos {pos}: dt gate"));
        assert_rel_linf(&beta_g, &beta_c, TOL, &format!("pos {pos}: beta gate"));

        // --- the delta rule, from the GPU's own q/k/v, beta, dt and state.
        let mut ssm_state_c = ssm0.clone();
        let mut out_h_c = vec![0.0f32; v_dim];
        delta_rule_recurrence(
            &conv_out_g[0..k_dim],
            &conv_out_g[k_dim..k_dim * 2],
            &conv_out_g[k_dim * 2..conv_dim],
            &beta_g,
            &dt_g,
            &mut ssm_state_c[..],
            &mut out_h_c,
            qwen.num_v_heads,
            qwen.head_v_dim,
        );
        assert_rel_linf(
            &out_h_g,
            &out_h_c,
            TOL,
            &format!("pos {pos}: delta-rule output"),
        );
        assert_rel_linf(
            &ssm_state_g,
            &ssm_state_c,
            TOL,
            &format!("pos {pos}: recurrent state after the delta rule"),
        );

        // --- gated RMSNorm, from the GPU's own out_h and gate.
        let mut ssm_out_in_c = vec![0.0f32; v_dim];
        gated_rmsnorm(
            &out_h_g,
            &gate_g,
            &d.ssm_norm_weight,
            eps,
            qwen.head_v_dim,
            &mut ssm_out_in_c,
        );
        assert_rel_linf(
            &ssm_out_in_g,
            &ssm_out_in_c,
            TOL,
            &format!("pos {pos}: gated RMSNorm"),
        );

        // Advance the CPU reference so the next position starts from a real state.
        qwen.forward_deltanet(
            d,
            &mut hidden,
            &mut cpu_state,
            il,
            pos,
            &mut normed,
            &mut post_normed,
        )
        .expect("cpu deltanet");
        cpu_state.kv_cache.advance();
    }
}

/// The sigmoid-gate kernel (the full-attention output gate, phase 2's) against
/// `apply_sigmoid_gate` — wired here so it is not dark until phase 2 lands.
#[test]
#[serial_test::serial]
fn qwen35_cuda_sigmoid_gate_matches_cpu() {
    let mut executor = qwen35_cuda_fixture_or_skip!();
    let n = 2048usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.001, -1.0)).collect();
    let g: Vec<f32> = (0..n).map(|i| 2.0 - (i as f32) * 0.002).collect();
    let mut want = x.clone();
    crate::gguf::forward_qwen35::apply_sigmoid_gate(&mut want, &g);

    let xd = GpuBuffer::from_host(executor.context(), &x).expect("upload x");
    let gd = GpuBuffer::from_host(executor.context(), &g).expect("upload gate");
    executor
        .gdn_sigmoid_gate_into(&xd, &gd, u32::try_from(n).expect("n fits"))
        .expect("launch");
    executor.sync_stream().expect("sync");
    let mut got = vec![0.0f32; n];
    xd.copy_to_host(&mut got).expect("download");
    assert_rel_linf(&got, &want, TOL, "sigmoid gate");
}

/// Each layer path refuses the other kind, loudly and by name, instead of
/// silently running a `DeltaNet` layer's code on an attention layer or the
/// reverse.
#[test]
#[serial_test::serial]
fn qwen35_cuda_layer_kind_mismatch_refuses_by_name() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let attention_il = qwen
        .layers
        .iter()
        .position(|l| matches!(l, Qwen35OwnedLayer::Attention(_)))
        .expect("the hybrid file carries full-attention layers");
    let deltanet_il = qwen
        .layers
        .iter()
        .position(|l| matches!(l, Qwen35OwnedLayer::DeltaNet(_)))
        .expect("the hybrid file carries Gated DeltaNet layers");

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    let dev = GpuBuffer::<f32>::from_host(
        gpu.executor_mut().context(),
        &vec![0.0f32; qwen.base.config.hidden_dim],
    )
    .expect("upload hidden");

    let err = gpu
        .forward_deltanet_layer(attention_il, &dev)
        .expect_err("an attention layer must not run the DeltaNet path");
    assert!(
        format!("{err}").contains("qwen35_cuda_attention"),
        "the refusal must name the seam: {err}"
    );
    let err = gpu
        .forward_attention_layer(deltanet_il, &dev, 0)
        .expect_err("a DeltaNet layer must not run the attention path");
    assert!(format!("{err}").contains("qwen35_cuda_deltanet"), "{err}");
}

/// Per-layer parity of the wired full-attention GPU forward against the CPU
/// reference, over the first three tokens of a fixed prompt, on EVERY attention
/// layer, with teacher forcing — on the layer output hidden state AND on the K
/// and V rows the token appends at its position.
#[test]
#[serial_test::serial]
#[allow(clippy::too_many_lines)]
fn qwen35_cuda_attention_layers_match_cpu_on_the_real_file() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");

    let attention_layers = qwen
        .layers
        .iter()
        .filter(|l| matches!(l, Qwen35OwnedLayer::Attention(_)))
        .count();
    assert!(
        attention_layers > 0,
        "the fixture must carry full-attention layers, else this test proves nothing"
    );

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    gpu.pin_reference_gemv();

    let hidden_dim = qwen.base.config.hidden_dim;
    let head_dim = qwen.head_dim;
    let num_kv_heads = qwen.base.config.num_kv_heads;
    let kv_row = num_kv_heads * head_dim;
    let eps = qwen.base.config.eps;
    let n_rot = 2 * qwen.rope_sections.iter().sum::<usize>();
    let freq_base = qwen.base.config.rope_theta;
    let mut cpu_state = qwen.new_state(PROMPT.len() + 1);
    let mut normed = vec![0.0f32; hidden_dim];
    let mut post_normed = vec![0.0f32; hidden_dim];
    let mut compared = 0usize;
    let mut worst_cos = 1.0f32;
    let mut worst_linf = 0.0f32;

    for (pos, &token) in PROMPT.iter().enumerate() {
        let mut hidden = qwen.base.token_embedding()
            [(token as usize) * hidden_dim..(token as usize + 1) * hidden_dim]
            .to_vec();

        for (il, layer) in qwen.layers.iter().enumerate() {
            match layer {
                Qwen35OwnedLayer::DeltaNet(d) => {
                    // Already proven above; here it only advances the CPU
                    // reference so the next attention layer sees a real input.
                    qwen.forward_deltanet(
                        d,
                        &mut hidden,
                        &mut cpu_state,
                        il,
                        pos,
                        &mut normed,
                        &mut post_normed,
                    )
                    .expect("cpu deltanet");
                },
                Qwen35OwnedLayer::Attention(a) => {
                    // Teacher forcing: the GPU starts from the CPU's KV rows for
                    // positions 0..pos, and from the same hidden state.
                    let k_before = cpu_state.kv_cache.get_k(il).to_vec();
                    let v_before = cpu_state.kv_cache.get_v(il).to_vec();
                    let hidden_in = hidden.clone();
                    assert_eq!(
                        k_before.len(),
                        pos * kv_row,
                        "pos {pos} layer {il}: the CPU cache must hold exactly {pos} rows"
                    );
                    gpu.upload_attention_kv(il, &k_before, &v_before)
                        .expect("seed the device KV cache");
                    let dev = GpuBuffer::from_host(gpu.executor_mut().context(), &hidden)
                        .expect("upload hidden");

                    qwen.forward_attention(
                        a,
                        &mut hidden,
                        &mut cpu_state,
                        il,
                        pos,
                        &mut normed,
                        &mut post_normed,
                    )
                    .expect("cpu attention");

                    gpu.forward_attention_layer(il, &dev, pos)
                        .expect("gpu attention");
                    gpu.executor_mut().sync_stream().expect("sync");
                    let mut got = vec![0.0f32; hidden_dim];
                    dev.copy_to_host(&mut got).expect("download hidden");

                    // The CPU side here IS the production forward, which
                    // quantizes every projection's activation to Q8_K — so the
                    // contract is argmax + cosine + a budget, not a tight L∞.
                    // See COSINE_FLOOR.
                    let (cos, linf) = assert_forward_parity(
                        &got,
                        &hidden,
                        LAYER_BUDGET,
                        &format!("pos {pos} layer {il} hidden"),
                    );
                    eprintln!(
                        "[attn layer] pos {pos} layer {il}: cosine {cos:.6} relative L-inf \
                         {linf:.3e}"
                    );
                    worst_cos = worst_cos.min(cos);
                    worst_linf = worst_linf.max(linf);

                    // The appended row is the only one the GPU computed — the
                    // earlier rows were seeded from the CPU and would compare
                    // equal to themselves.
                    //
                    // The reference for it is EXACT arithmetic, not the CPU's
                    // own cache row: K and V are one quantized GEMV off the
                    // layer input, and the CPU's GEMV quantizes its activation
                    // to Q8_K (parity-floor test). Comparing against the CPU row
                    // measured that quantization and nothing else — it read
                    // 7.6e-3 relative on a K row whose every non-GEMV op is
                    // inside 1e-3. Built from the same layer input with
                    // dequantized weights and f64 dots, the same row is a
                    // kernel-contract comparison and holds at TOL.
                    let (k_after, v_after) = gpu.download_layer(il).expect("download KV");
                    assert_eq!(k_after.len(), (pos + 1) * kv_row, "K rows written");
                    let mut normed_ref = vec![0.0f32; hidden_dim];
                    crate::gguf::ops::rms_norm_into(&hidden_in, &a.attn_norm, eps, &mut normed_ref);
                    let mut k_ref = exact_gemv(&a.attn_k, &normed_ref);
                    crate::gguf::ops::apply_per_head_rms_norm(
                        &mut k_ref,
                        &a.attn_k_norm,
                        num_kv_heads,
                        eps,
                    );
                    crate::gguf::forward_qwen35::apply_partial_neox_rope(
                        &mut k_ref,
                        num_kv_heads,
                        head_dim,
                        n_rot,
                        pos,
                        freq_base,
                    );
                    let v_ref = exact_gemv(&a.attn_v, &normed_ref);
                    eprintln!(
                        "[attn layer] pos {pos} layer {il}: K row vs exact {:.3e}  V row vs \
                         exact {:.3e}",
                        rel_linf(&k_after[pos * kv_row..], &k_ref),
                        rel_linf(&v_after[pos * kv_row..], &v_ref),
                    );
                    assert_rel_linf(
                        &k_after[pos * kv_row..],
                        &k_ref,
                        KV_TOL,
                        &format!("pos {pos} layer {il} K row"),
                    );
                    assert_rel_linf(
                        &v_after[pos * kv_row..],
                        &v_ref,
                        KV_TOL,
                        &format!("pos {pos} layer {il} V row"),
                    );
                    compared += 1;
                },
            }
        }
        cpu_state.kv_cache.advance();
    }

    assert_eq!(
        compared,
        attention_layers * PROMPT.len(),
        "every attention layer must be compared at every position"
    );
    eprintln!(
        "[attn layer] worst over {compared} comparisons: cosine {worst_cos:.6} relative L-inf \
         {worst_linf:.3e}"
    );
}

/// The three attention-side kernels (split, partial rope, decode attention) and
/// the per-head RMSNorm, each against the CPU reference function on the GPU's
/// OWN input, at 1e-3 relative.
///
/// Feeding the CPU the GPU's inputs is what makes this a kernel test: the
/// projection GEMVs are upstream of every one of them and carry ~5e-3 of their
/// own, so comparing "the CPU's attention output" against "the GPU's" would
/// measure the GEMVs and call it the attention block.
#[test]
#[serial_test::serial]
#[allow(clippy::too_many_lines)]
fn qwen35_cuda_attention_ops_match_cpu_given_the_same_inputs() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let il = qwen
        .layers
        .iter()
        .position(|l| matches!(l, Qwen35OwnedLayer::Attention(_)))
        .expect("an attention layer");
    let Qwen35OwnedLayer::Attention(a) = &qwen.layers[il] else {
        unreachable!("just matched")
    };

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    gpu.pin_reference_gemv();

    let hidden_dim = qwen.base.config.hidden_dim;
    let eps = qwen.base.config.eps;
    let num_heads = qwen.base.config.num_heads;
    let num_kv_heads = qwen.base.config.num_kv_heads;
    let head_dim = a.attn_q_norm.len();
    let kv_row = num_kv_heads * head_dim;
    let n_rot = 2 * qwen.rope_sections.iter().sum::<usize>();
    let freq_base = qwen.base.config.rope_theta;
    let mut cpu_state = qwen.new_state(PROMPT.len() + 1);
    let mut normed = vec![0.0f32; hidden_dim];
    let mut post_normed = vec![0.0f32; hidden_dim];

    for (pos, &token) in PROMPT.iter().enumerate() {
        let k_before = cpu_state.kv_cache.get_k(il).to_vec();
        let v_before = cpu_state.kv_cache.get_v(il).to_vec();
        let mut hidden = qwen.base.token_embedding()
            [(token as usize) * hidden_dim..(token as usize + 1) * hidden_dim]
            .to_vec();
        gpu.upload_attention_kv(il, &k_before, &v_before)
            .expect("seed");
        let dev =
            GpuBuffer::from_host(gpu.executor_mut().context(), &hidden).expect("upload hidden");
        gpu.forward_attention_layer(il, &dev, pos)
            .expect("gpu attention");

        let normed_g = gpu.dump_stage("normed");
        let q_full_g = gpu.dump_stage("q_full");
        let q_g = gpu.dump_stage("q");
        let q_rot_g = gpu.dump_stage("q_normed");
        let gate_g = gpu.dump_stage("attn_gate");
        let k_raw_g = gpu.dump_stage("k_raw");
        let attn_out_in_g = gpu.dump_stage("attn_out_in");
        let (k_all_g, v_all_g) = gpu.download_layer(il).expect("download KV");

        // --- the q|gate split, from the GPU's own joint projection. Pure data
        // movement: this one must be EXACT.
        let mut q_c = vec![0.0f32; num_heads * head_dim];
        let mut gate_c = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let src = h * head_dim * 2;
            let dst = h * head_dim;
            q_c[dst..dst + head_dim].copy_from_slice(&q_full_g[src..src + head_dim]);
            gate_c[dst..dst + head_dim]
                .copy_from_slice(&q_full_g[src + head_dim..src + head_dim * 2]);
        }
        assert_eq!(q_g, q_c, "pos {pos}: the q half of the split is not exact");
        assert_eq!(
            gate_g, gate_c,
            "pos {pos}: the gate half of the split is not exact"
        );

        // --- per-head RMSNorm + partial rope on q, from the GPU's own q.
        let mut q_rot_c = q_g.clone();
        crate::gguf::ops::apply_per_head_rms_norm(&mut q_rot_c, &a.attn_q_norm, num_heads, eps);
        crate::gguf::forward_qwen35::apply_partial_neox_rope(
            &mut q_rot_c,
            num_heads,
            head_dim,
            n_rot,
            pos,
            freq_base,
        );
        assert_rel_linf(&q_rot_g, &q_rot_c, TOL, &format!("pos {pos}: q norm+rope"));

        // --- the same on k, whose result IS the appended cache row.
        let mut k_rot_c = k_raw_g.clone();
        crate::gguf::ops::apply_per_head_rms_norm(&mut k_rot_c, &a.attn_k_norm, num_kv_heads, eps);
        crate::gguf::forward_qwen35::apply_partial_neox_rope(
            &mut k_rot_c,
            num_kv_heads,
            head_dim,
            n_rot,
            pos,
            freq_base,
        );
        assert_rel_linf(
            &k_all_g[pos * kv_row..],
            &k_rot_c,
            TOL,
            &format!("pos {pos}: k norm+rope (the appended row)"),
        );

        // --- the attention block + output gate, from the GPU's own q, KV cache
        // and gate.
        let mut attn_c = vec![0.0f32; num_heads * head_dim];
        let group_size = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_h = h / group_size;
            let q_h = &q_rot_g[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; pos + 1];
            for (p, score) in scores.iter_mut().enumerate() {
                let base = p * kv_row + kv_h * head_dim;
                let mut dot = 0.0;
                for i in 0..head_dim {
                    dot += q_h[i] * k_all_g[base + i];
                }
                *score = dot / (head_dim as f32).sqrt();
            }
            crate::gguf::ops::softmax(&mut scores);
            let out_h = &mut attn_c[h * head_dim..(h + 1) * head_dim];
            for (p, &w) in scores.iter().enumerate() {
                let base = p * kv_row + kv_h * head_dim;
                for i in 0..head_dim {
                    out_h[i] += w * v_all_g[base + i];
                }
            }
        }
        crate::gguf::forward_qwen35::apply_sigmoid_gate(&mut attn_c, &gate_g);
        assert_rel_linf(
            &attn_out_in_g,
            &attn_c,
            TOL,
            &format!("pos {pos}: decode attention + output gate"),
        );

        // --- the projection GEMVs, from the GPU's own normed input, against
        // EXACT arithmetic (dequantized weight, f64 dots) and NOT against the
        // CPU production op: `fused_matmul_into` quantizes its activation to
        // Q8_K and is itself ~3e-3 from exact, so it cannot be the reference a
        // 1e-3 bar is written against. See COSINE_FLOOR and the parity-floor
        // test. These are not this ticket's kernels, but the float GEMV is what
        // the constructor pins, so this is where a regression in that choice
        // shows up first.
        let q_full_e = exact_gemv(&a.attn_q, &normed_g);
        let k_e = exact_gemv(&a.attn_k, &normed_g);
        let out_e = exact_gemv(&a.attn_output, &attn_out_in_g);
        let attn_out_g = gpu.dump_stage("attn_out");
        eprintln!(
            "[GEMV vs exact pos {pos}] attn_q(t{}) {:.3e}  attn_k(t{}) {:.3e}  attn_output(t{}) \
             {:.3e}",
            a.attn_q.qtype,
            rel_linf(&q_full_g, &q_full_e),
            a.attn_k.qtype,
            rel_linf(&k_raw_g, &k_e),
            a.attn_output.qtype,
            rel_linf(&attn_out_g, &out_e),
        );
        assert_rel_linf(
            &q_full_g,
            &q_full_e,
            TOL,
            &format!("pos {pos}: attn_q GEMV"),
        );
        assert_rel_linf(&k_raw_g, &k_e, TOL, &format!("pos {pos}: attn_k GEMV"));
        assert_rel_linf(
            &attn_out_g,
            &out_e,
            TOL,
            &format!("pos {pos}: attn_output GEMV"),
        );

        // --- rms_norm itself, on the same hidden state.
        let mut normed_c = vec![0.0f32; hidden_dim];
        crate::gguf::ops::rms_norm_into(&hidden, &a.attn_norm, eps, &mut normed_c);
        assert_rel_linf(&normed_g, &normed_c, TOL, &format!("pos {pos}: rms_norm"));

        qwen.forward_attention(
            a,
            &mut hidden,
            &mut cpu_state,
            il,
            pos,
            &mut normed,
            &mut post_normed,
        )
        .expect("cpu attention");
        cpu_state.kv_cache.advance();
    }
}

/// END TO END: the whole per-token forward — both layer kinds, the output norm
/// and the `lm_head` — against `forward_single_qwen35`, from fresh states, token
/// by token over a fixed six-token prompt.
///
/// No teacher forcing: after the first token the two sides diverge or they do
/// not, and the argmax equality is what a decode actually depends on.
///
/// The contract is [`COSINE_FLOOR`]'s three assertions, and the restatement did
/// NOT cost the test its teeth: skipping `gdn_sigmoid_gate_into` in
/// `deltanet_layer_inner` (one line, reverted) turns this RED at position 0 on
/// all three at once — argmax **220 vs 13**, cosine **0.765610** (floor 0.996),
/// relative L∞ **1.427e0** (budget 7e-2). A gate the forward does not apply is
/// a different token, not a rounding difference.
#[test]
#[serial_test::serial]
fn qwen35_cuda_forward_single_matches_cpu_logits_end_to_end() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    gpu.pin_reference_gemv();
    let mut gpu_state = gpu.new_state().expect("device state");
    let mut cpu_state = qwen.new_state(LONG_PROMPT.len() + 1);
    let mut worst_cos = 1.0f32;
    let mut worst_linf = 0.0f32;

    for (pos, &token) in LONG_PROMPT.iter().enumerate() {
        let want = qwen
            .forward_single_qwen35(token, &mut cpu_state, pos)
            .expect("cpu forward");
        // Wall time of ONE device token, printed and never asserted on: a
        // timing assertion in a correctness test is a flake, but the reading is
        // what says whether a sync change cost or bought anything (#3090).
        let t0 = std::time::Instant::now();
        let got = gpu
            .forward_single(token, &mut gpu_state, pos)
            .expect("gpu forward");
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (cos, linf) =
            assert_forward_parity(&got, &want, LOGITS_BUDGET, &format!("pos {pos} logits"));
        eprintln!(
            "[e2e] pos {pos}: argmax {} cosine {cos:.6} relative L-inf {linf:.3e} \
             forward_single {gpu_ms:.3} ms",
            crate::gguf::ops::argmax(&want),
        );
        worst_cos = worst_cos.min(cos);
        worst_linf = worst_linf.max(linf);
        assert_eq!(
            gpu_state.kv_len(),
            pos + 1,
            "pos {pos}: the device KV cache must have advanced"
        );
    }
    eprintln!(
        "[e2e] worst over {} positions: cosine {worst_cos:.6} relative L-inf {worst_linf:.3e}",
        LONG_PROMPT.len(),
    );
}

/// The device state is sized from the config, never from a constant.
#[test]
#[serial_test::serial]
fn qwen35_cuda_state_is_sized_from_the_config() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let conv_dim = qwen.head_k_dim * qwen.num_k_heads * 2 + qwen.head_v_dim * qwen.num_v_heads;

    let gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    assert_eq!(
        gpu.state().conv_len(),
        conv_dim * (qwen.conv_kernel - 1),
        "conv window is conv_dim * (conv_kernel - 1)"
    );
    assert_eq!(
        gpu.state().ssm_len(),
        qwen.num_v_heads * qwen.head_v_dim * qwen.head_v_dim,
        "recurrent state is num_v_heads * head_v_dim^2"
    );
}

/// `forward_hidden_deltanet_only` refuses a hidden state of the wrong width
/// instead of reading past the buffer.
#[test]
#[serial_test::serial]
fn qwen35_cuda_forward_hidden_refuses_a_wrong_width() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    let err = gpu
        .forward_hidden_deltanet_only(&[0.0f32; 7])
        .expect_err("7 is not the hidden width");
    assert!(format!("{err}").contains("hidden state is 7 wide"), "{err}");
}

/// WHERE THE PARITY FLOOR COMES FROM (PMAT-3477 / #3090).
///
/// The wired-layer and end-to-end comparisons above cannot be driven below the
/// projection GEMVs, and a GPU-vs-CPU number on its own never says WHICH side
/// moved. This puts both sides against a third reference that is neither: the
/// Q4_K weight dequantized to f32 and the row dots accumulated in **f64**.
///
/// Measured on `blk.<first attention>.attn_q` of the real 0.8B file:
///
/// | side | relative L∞ vs the exact f64 dot |
/// |------|----------------------------------|
/// | GPU, float (`Mwv`) GEMV | **9.96e-8** |
/// | CPU `fused_matmul_into` | **2.95e-3** |
///
/// The CPU is the noisier side by four and a half orders of magnitude, because
/// `fused_q4k_parallel_matvec_into` quantizes the **activation** to Q8_K and
/// takes int8 dots; the GPU float path dequantizes the weight and accumulates
/// in f32. So ~3e-3 per projection is a property of the *reference*, not a GPU
/// defect, and it is the floor under [`LAYER_TOL`], [`LAYER_BUDGET`] and
/// [`LOGITS_BUDGET`], and the reason the two whole-forward tests assert argmax
/// and [`COSINE_FLOOR`] instead of a tight L∞: no
/// correct GPU implementation can be closer to this CPU than the CPU is to
/// arithmetic.
///
/// Both bounds bite. If the GPU float GEMV ever regresses it fails on the first
/// assertion; if the CPU path ever stops quantizing its activation the second
/// assertion fails and says so — at which point the end-to-end budget can come
/// down and this test is the thing that tells you.
#[test]
#[serial_test::serial]
fn qwen35_cuda_the_parity_floor_is_the_cpu_references_own_activation_quantization() {
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let il = qwen
        .layers
        .iter()
        .position(|l| matches!(l, Qwen35OwnedLayer::Attention(_)))
        .expect("the hybrid file carries full-attention layers");
    let Qwen35OwnedLayer::Attention(a) = &qwen.layers[il] else {
        unreachable!("just matched")
    };
    assert_eq!(
        a.attn_q.qtype,
        crate::gguf::types::GGUF_TYPE_Q4_K,
        "this test reads the Q4_K dequantizer; the projection is no longer Q4_K"
    );

    let hidden_dim = qwen.base.config.hidden_dim;
    let x: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.37).sin() * 1.3)
        .collect();

    // the exact reference: dequantized weight, f64 accumulation — the same
    // helper every GEMV-fed assertion in this file compares against.
    let exact = exact_gemv(&a.attn_q, &x);

    let mut cpu = vec![0.0f32; a.attn_q.out_dim];
    qwen.base
        .fused_matmul_into(&x, &a.attn_q, &mut cpu)
        .expect("cpu attn_q");

    // NOT `pin_reference_gemv()`: the constructor pins the float variants for
    // every model of this architecture, and a test that re-pins measures its
    // own call instead of production behaviour (#3090 review). The pin itself
    // is asserted by `qwen35_cuda_a_fresh_model_pins_the_float_gemv_variants`;
    // if it ever stops holding, the first assertion below turns red here too.
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    let got = gpu
        .attn_q_gemv_of_host_input(il, &x)
        .expect("gpu attn_q GEMV");

    let gpu_err = rel_linf(&got, &exact);
    let cpu_err = rel_linf(&cpu, &exact);
    eprintln!("[parity floor] gpu-vs-exact {gpu_err:.3e}  cpu-vs-exact {cpu_err:.3e}");
    assert!(
        gpu_err <= 1e-5,
        "the GPU float GEMV must reproduce the exact dot: {gpu_err:.3e} > 1e-5"
    );
    assert!(
        cpu_err > 1e-4,
        "the CPU reference no longer quantizes its activation (cpu-vs-exact {cpu_err:.3e} <= \
         1e-4) — the end-to-end parity budget can now come down; re-measure LOGITS_BUDGET and \
         consider a tight L-inf bar again"
    );
    assert!(
        cpu_err > gpu_err * 100.0,
        "the CPU, not the GPU, must be the side that is far from arithmetic: cpu {cpu_err:.3e} \
         vs gpu {gpu_err:.3e}"
    );
}

/// A model built the ordinary way already runs the FLOAT GEMV kernels — the pin
/// is production behaviour, not something a test remembers to do.
///
/// `GpuProfile::detect` picks the DP4A variants for this device; the
/// constructor overrides them for this architecture because DP4A is
/// catastrophic through the recurrence (the falsifier below). If that override
/// is ever dropped, decode silently returns garbage tokens and only this
/// assertion says so before the parity tests do.
#[test]
#[serial_test::serial]
fn qwen35_cuda_a_fresh_model_pins_the_float_gemv_variants() {
    use crate::cuda::gpu_profile::{Q4kVariant, Q6kVariant};
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");

    let gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    assert_eq!(
        gpu.gemv_variants(),
        (Q4kVariant::Mwv, Q6kVariant::Mwv),
        "Qwen35CudaModel::new must pin the float GEMV variants for this architecture"
    );
}

/// WHY THE PIN EXISTS (PMAT-3477 / #3090): with the DP4A GEMV kernels armed,
/// the same forward that matches the CPU token for token produces a DIFFERENT
/// token, or a direction nowhere near the CPU's.
///
/// This is a falsifier, not a bug reproduction: it asserts the failure, so if a
/// future DP4A activation-quantization change makes the path correct, this test
/// goes RED and the pin (and the DP4A-through-recurrence ticket, 0.69.0) can be
/// reconsidered on evidence.
///
/// Measured on the real 0.8B file at position 0: see the printed reading. The
/// DeltaNet-only path is 1.656 relative away from the CPU under `HwDp4a` versus
/// 0.000 float-vs-float; end to end the argmax is simply wrong.
#[test]
#[serial_test::serial]
fn qwen35_cuda_dp4a_gemv_is_catastrophic_through_the_recurrence() {
    use crate::cuda::gpu_profile::{Q4kVariant, Q6kVariant};
    let executor = qwen35_cuda_fixture_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    // The variant IS selectable from a test: the executor's profile is what
    // every `gemv_dispatch` reads, and `gemv_variants()` reports it back.
    gpu.executor_mut().gpu_profile.q4k = Q4kVariant::HwDp4a;
    gpu.executor_mut().gpu_profile.q6k = Q6kVariant::HwDp4a;
    assert_eq!(
        gpu.gemv_variants(),
        (Q4kVariant::HwDp4a, Q6kVariant::HwDp4a),
        "the DP4A variants must actually be armed, else this test proves nothing"
    );

    let mut gpu_state = gpu.new_state().expect("device state");
    let mut cpu_state = qwen.new_state(LONG_PROMPT.len() + 1);
    let mut broken = 0usize;

    for (pos, &token) in LONG_PROMPT.iter().enumerate() {
        let want = qwen
            .forward_single_qwen35(token, &mut cpu_state, pos)
            .expect("cpu forward");
        let got = gpu
            .forward_single(token, &mut gpu_state, pos)
            .expect("gpu forward");
        let cos = cosine(&got, &want);
        let (gpu_arg, cpu_arg) = (
            crate::gguf::ops::argmax(&got),
            crate::gguf::ops::argmax(&want),
        );
        eprintln!(
            "[dp4a] pos {pos}: gpu argmax {gpu_arg} cpu argmax {cpu_arg} cosine {cos:.6} \
             relative L-inf {:.3e}",
            rel_linf(&got, &want),
        );
        if gpu_arg != cpu_arg || cos < COSINE_FLOOR {
            broken += 1;
        }
    }

    assert!(
        broken > 0,
        "the DP4A GEMV path passed the parity contract at every position — it is no longer \
         catastrophic through the recurrence, so re-measure and revisit the pin in \
         Qwen35CudaModel::with_max_seq_len (DP4A-through-recurrence ticket, 0.69.0)"
    );
}

/// Keep the layer type in the compiled surface: the parity tests bind it, and a
/// rename of the CPU struct must break here, not silently at phase 2.
const _: Option<&Qwen35OwnedDeltaNetLayer> = None;
