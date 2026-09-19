//! PMAT-3477 / aprender#3090: CPU parity for [`Qwen35CudaModel`] on the real
//! Qwen3.5-0.8B file, at two scopes that must not be confused.
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

/// Tolerance for the K and V rows an attention layer appends. They are one
/// Q4_K/Q6_K GEMV plus a per-head RMSNorm and the partial rope away from the
/// layer input, so they carry the GEMV budget but not the attention block's.
const KV_TOL: f32 = 5e-3;

/// Tolerance for the end-to-end logits of a whole token. Every layer's GEMV
/// error compounds through 24 layers and the `lm_head`; this is the measured
/// budget, and it is held together with an EXACT argmax equality, which is what
/// a decode actually depends on.
const LOGITS_TOL: f32 = 1e-3;

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
    let kv_row = qwen.base.config.num_kv_heads * qwen.head_dim;
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

                    assert_rel_linf(
                        &got,
                        &hidden,
                        LAYER_TOL,
                        &format!("pos {pos} layer {il} hidden"),
                    );

                    // The appended row is the only one the GPU computed — the
                    // earlier rows were seeded from the CPU and would compare
                    // equal to themselves.
                    let (k_after, v_after) = gpu.download_layer(il).expect("download KV");
                    assert_eq!(k_after.len(), (pos + 1) * kv_row, "K rows written");
                    let cpu_k = cpu_state.kv_cache.get_k(il);
                    let cpu_v = cpu_state.kv_cache.get_v(il);
                    assert_rel_linf(
                        &k_after[pos * kv_row..],
                        &cpu_k[pos * kv_row..(pos + 1) * kv_row],
                        KV_TOL,
                        &format!("pos {pos} layer {il} K row"),
                    );
                    assert_rel_linf(
                        &v_after[pos * kv_row..],
                        &cpu_v[pos * kv_row..(pos + 1) * kv_row],
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

        // --- the projection GEMVs, from the GPU's own normed input. These are
        // NOT this ticket's kernels: they are the pre-existing CPU/GPU
        // quantization gap, measured here so the layer budget below is a
        // reading and not a wish.
        let mut q_full_c = vec![0.0f32; a.attn_q.out_dim];
        qwen.base
            .fused_matmul_into(&normed_g, &a.attn_q, &mut q_full_c)
            .expect("cpu attn_q");
        let mut k_c = vec![0.0f32; a.attn_k.out_dim];
        qwen.base
            .fused_matmul_into(&normed_g, &a.attn_k, &mut k_c)
            .expect("cpu attn_k");
        let mut out_c = vec![0.0f32; a.attn_output.out_dim];
        qwen.base
            .fused_matmul_into(&attn_out_in_g, &a.attn_output, &mut out_c)
            .expect("cpu attn_output");
        eprintln!(
            "[GEMV pos {pos}] attn_q(t{}) {:.3e}  attn_k(t{}) {:.3e}  attn_output(t{}) {:.3e}",
            a.attn_q.qtype,
            rel_linf(&q_full_g, &q_full_c),
            a.attn_k.qtype,
            rel_linf(&k_raw_g, &k_c),
            a.attn_output.qtype,
            rel_linf(&gpu.dump_stage("attn_out"), &out_c),
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

    for (pos, &token) in LONG_PROMPT.iter().enumerate() {
        let want = qwen
            .forward_single_qwen35(token, &mut cpu_state, pos)
            .expect("cpu forward");
        let got = gpu
            .forward_single(token, &mut gpu_state, pos)
            .expect("gpu forward");
        assert_rel_linf(&got, &want, LOGITS_TOL, &format!("pos {pos} logits"));
        assert_eq!(
            crate::gguf::ops::argmax(&got),
            crate::gguf::ops::argmax(&want),
            "pos {pos}: the GPU and CPU must choose the same token"
        );
        assert_eq!(
            gpu_state.kv_len(),
            pos + 1,
            "pos {pos}: the device KV cache must have advanced"
        );
    }
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
/// defect, and it is the floor under [`LAYER_TOL`] and [`LOGITS_TOL`]: no
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

    // the exact reference: dequantized weight, f64 accumulation
    let w = crate::quantize::dequantize_q4_k(&a.attn_q.data).expect("dequantize the projection");
    assert_eq!(
        w.len(),
        a.attn_q.out_dim * a.attn_q.in_dim,
        "the dequantized weight must be out_dim x in_dim, row-major"
    );
    let exact: Vec<f32> = (0..a.attn_q.out_dim)
        .map(|r| {
            let row = &w[r * a.attn_q.in_dim..(r + 1) * a.attn_q.in_dim];
            row.iter()
                .zip(&x)
                .fold(0.0f64, |s, (wv, xv)| s + f64::from(*wv) * f64::from(*xv)) as f32
        })
        .collect();

    let mut cpu = vec![0.0f32; a.attn_q.out_dim];
    qwen.base
        .fused_matmul_into(&x, &a.attn_q, &mut cpu)
        .expect("cpu attn_q");

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    gpu.pin_reference_gemv();
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
         1e-4) — the end-to-end parity budget can now come down; re-measure LOGITS_TOL"
    );
    assert!(
        cpu_err > gpu_err * 100.0,
        "the CPU, not the GPU, must be the side that is far from arithmetic: cpu {cpu_err:.3e} \
         vs gpu {gpu_err:.3e}"
    );
}

/// Keep the layer type in the compiled surface: the parity tests bind it, and a
/// rename of the CPU struct must break here, not silently at phase 2.
const _: Option<&Qwen35OwnedDeltaNetLayer> = None;
