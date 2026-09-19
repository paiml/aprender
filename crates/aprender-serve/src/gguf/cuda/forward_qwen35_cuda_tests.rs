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

/// A fixed three-token prompt. Any ids work for parity — what matters is that
/// both sides see the same ones.
const PROMPT: [u32; 3] = [9707, 11, 1879];

/// `max|want|`, the scale the tolerance is relative to.
fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
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

/// The attention seam refuses, loudly and by name, instead of silently running a
/// `DeltaNet` layer's code on an attention layer.
#[test]
#[serial_test::serial]
fn qwen35_cuda_attention_layer_is_an_unimplemented_seam() {
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
        .forward_attention_layer(attention_il, &dev)
        .expect_err("attention is phase 2");
    assert!(format!("{err}").contains("qwen35_cuda_attention"), "{err}");
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

/// Keep the layer type in the compiled surface: the parity tests bind it, and a
/// rename of the CPU struct must break here, not silently at phase 2.
const _: Option<&Qwen35OwnedDeltaNetLayer> = None;
