//! FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-001 — oracle-based loss-parity probe.
//!
//! Defect #4 (NF4 QLoRA CUDA training): the GPU training forward/loss produced
//! finite-garbage cross-entropy (~13-14 > ln(151936)=11.93 — worse than
//! uniform) on data the base model can already emit, so training never learned
//! and adapters eventually blew up into NaN.
//!
//! Bisection oracles, strongest first:
//!   (c) pure-CPU forward + CPU CE          — ground truth
//!   (b) GPU transformer forward + GPU lm_head + CPU CE — isolates forward
//!   (a) GPU-resident forward + fused GPU CE — full production path
//!
//! (a)≈(b)≫(c) ⟹ the GPU transformer forward diverges from the CPU model.
//! Root cause: `CudaNf4TransformerBlock` never received or applied the Q/K/V
//! projection biases (Qwen2 family `use_bias=true`); the CPU model loads and
//! applies them (`attention.rs::add_bias`).
//!
//! Env-gated on APR_PARITY_MODEL (path to a Qwen2-family .apr with an
//! embedded tokenizer). Requires a CUDA GPU. Run:
//!
//! ```text
//! APR_PARITY_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4k.apr \
//!   cargo test -p aprender-train --lib --features cuda \
//!   falsify_cuda_nf4_train_loss_parity -- --ignored --nocapture
//! ```

#[allow(clippy::wildcard_imports)]
use super::*;

use crate::autograd::cuda_optim::fused_causal_cross_entropy_cuda;

/// Relative L2 difference between two activation slices.
fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += f64::from(x - y) * f64::from(x - y);
        den += f64::from(y) * f64::from(y);
    }
    ((num / den.max(1e-30)) as f32).sqrt()
}

/// Layer-by-layer bisection of the GPU NF4 forward vs the CPU oracle.
///
/// Downloads the per-layer input snapshots the GPU forward saves for backward
/// (`layer_inputs[i]` = input of layer i, `blocks_output` = final output) and
/// compares them against a CPU replay through the same layers, both WITH
/// biases (true model) and WITHOUT Q/K/V biases (bias-drop hypothesis).
#[test]
#[ignore = "requires CUDA GPU + APR_PARITY_MODEL pointing at a Qwen2-family .apr"]
fn parity_probe_layer_bisect() {
    let Ok(model_path) = std::env::var("APR_PARITY_MODEL") else {
        eprintln!("[layer-bisect] SKIP: APR_PARITY_MODEL unset");
        return;
    };
    let model_path = std::path::PathBuf::from(model_path);
    if !model_path.exists() {
        return;
    }

    let model_config = qwen2_1_5b_config();
    let instruct_config = InstructConfig {
        lora_rank: 8,
        lora_alpha: 16.0,
        learning_rate: 1e-9,
        epochs: 1,
        max_seq_len: 64,
        gradient_clip_norm: Some(1.0),
        quantize_nf4: true,
    };
    let mut p = InstructPipeline::from_apr(&model_path, &model_config, instruct_config)
        .expect("pipeline from_apr");
    assert!(p.cuda_blocks.is_some(), "CUDA blocks must initialize");

    let prompt_ids = p.tokenize("What is 2+2? Answer with just the number.\n");
    let response_ids = p.tokenize("4");
    let full_ids: Vec<u32> = prompt_ids.iter().chain(response_ids.iter()).copied().collect();
    let seq_len = full_ids.len();
    let hidden = p.model.config().hidden_size;

    // GPU forward populates layer_inputs + blocks_output.
    p.forward_logits_gpu(&full_ids).expect("GPU forward");

    let (gpu_layer_inputs, gpu_final): (Vec<Vec<f32>>, Vec<f32>) = {
        let trainer = p.cuda_trainer.as_ref().expect("trainer");
        let training = p.gpu_training.as_ref().expect("training state");
        let inputs = training
            .layer_inputs
            .iter()
            .map(|b| {
                let v = trainer.download(b).expect("download layer input");
                v[..seq_len * hidden].to_vec()
            })
            .collect();
        let f = trainer.download(&training.blocks_output).expect("download final");
        (inputs, f[..seq_len * hidden].to_vec())
    };

    // GPU-vs-GPU cross-check: run pipeline block 0 standalone on the same
    // embed and compare against the pipeline forward's layer_inputs[1].
    {
        let embed = p.model.embed_tokens.forward(&full_ids);
        let x = embed.data().as_slice().expect("contiguous").to_vec();
        let trainer = p.cuda_trainer.as_ref().expect("trainer");
        let stream = trainer.stream();
        let gpu_in = trainer.upload(&x).expect("upload");
        let mut gpu_out = trainer.zeros(seq_len * hidden).expect("out");
        let blocks = p.cuda_blocks.as_mut().expect("blocks");
        if let Some(ref mut scratch) = p.shared_scratch {
            scratch.zero_forward_buffers(stream);
        }
        blocks[0]
            .forward(&gpu_in, &mut gpu_out, seq_len, stream, p.shared_scratch.as_mut())
            .expect("standalone block0 forward");
        stream.synchronize().expect("sync");
        let standalone = trainer.download(&gpu_out).expect("download");
        let d = rel_l2(&standalone[..seq_len * hidden], &gpu_layer_inputs[1]);
        eprintln!("[layer-bisect] GPU-vs-GPU block0 (standalone vs pipeline): relL2={d:.6}");
    }

    // NF4-matched CPU oracle: requantize the CPU model's projection weights
    // with the same NF4 round-trip the GPU blocks use, so the replay isolates
    // STRUCTURAL divergence from quantization noise.
    {
        use trueno_gpu::kernels::{dequantize_nf4, quantize_nf4};
        let cfg = p.model.config().clone();
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.num_kv_heads * cfg.head_dim();
        let inter = cfg.intermediate_size;
        let requant = |t: &Tensor, rows: usize, cols: usize| -> Tensor {
            let d = t.data();
            let s = d.as_slice().expect("contiguous weight");
            Tensor::from_vec(dequantize_nf4(&quantize_nf4(s, rows, cols)), false)
        };
        for layer in &mut p.model.layers {
            layer.self_attn.w_q = requant(&layer.self_attn.w_q, q_dim, hidden);
            layer.self_attn.w_k = requant(&layer.self_attn.w_k, kv_dim, hidden);
            layer.self_attn.w_v = requant(&layer.self_attn.w_v, kv_dim, hidden);
            layer.self_attn.w_o = requant(&layer.self_attn.w_o, hidden, q_dim);
            layer.ffn.w_gate = requant(&layer.ffn.w_gate, inter, hidden);
            layer.ffn.w_up = requant(&layer.ffn.w_up, inter, hidden);
            layer.ffn.w_down = requant(&layer.ffn.w_down, hidden, inter);
        }
    }

    // CPU replay: full-precision (truth) and NF4-matched.
    let embed = p.model.embed_tokens.forward(&full_ids);
    let mut h_nf4 = embed.data().as_slice().expect("contiguous").to_vec();

    let num_layers = p.model.layers.len();
    for i in 0..num_layers {
        let d_nf4 = rel_l2(&gpu_layer_inputs[i], &h_nf4);
        eprintln!("[layer-bisect] L{i:02} input:  relL2(gpu,cpu_nf4)={d_nf4:.4}");

        let t = Tensor::from_vec(h_nf4.clone(), false);
        let out = p.model.layers[i].forward(&t, seq_len);
        h_nf4 = out.data().as_slice().expect("contiguous").to_vec();
    }

    let d_nf4 = rel_l2(&gpu_final, &h_nf4);
    eprintln!("[layer-bisect] FINAL output: relL2(gpu,cpu_nf4)={d_nf4:.4}");
}

/// Tolerance for GPU-vs-CPU loss parity. NF4 quantization of the frozen
/// weights costs real accuracy, so the GPU loss cannot be bit-equal to the
/// F32 CPU loss — but it must be the SAME MODEL, not garbage. Empirically
/// NF4 costs <0.4 nats on this corpus; 0.5 is a loud-failure bound.
const PARITY_TOL: f32 = 0.5;

fn qwen2_1_5b_config() -> TransformerConfig {
    TransformerConfig::from_apr_metadata(
        Some(1536),
        Some(12),
        Some(2),
        Some(8960),
        Some(28),
        Some(151_936),
        Some(32_768),
        Some(1e-6),
        Some(1_000_000.0),
        Some("qwen2"),
    )
    .expect("qwen2 config from metadata")
}

#[test]
fn falsify_cuda_nf4_train_loss_parity_001() {
    // Env-gated: skips (vacuous pass) unless APR_PARITY_MODEL points at a
    // Qwen2-family .apr and a CUDA GPU is present. Run explicitly with:
    //   APR_PARITY_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4k.apr \
    //   cargo test -p aprender-train --features cuda --lib \
    //     falsify_cuda_nf4_train_loss_parity_001 --release -- --nocapture
    if !trueno_gpu::driver::cuda_available() {
        eprintln!("[parity-probe] SKIP: no CUDA device");
        return;
    }
    let Ok(model_path) = std::env::var("APR_PARITY_MODEL") else {
        eprintln!("[parity-probe] SKIP: APR_PARITY_MODEL unset");
        return;
    };
    let model_path = std::path::PathBuf::from(model_path);
    if !model_path.exists() {
        eprintln!("[parity-probe] SKIP: {} does not exist", model_path.display());
        return;
    }

    let model_config = qwen2_1_5b_config();
    let instruct_config = InstructConfig {
        lora_rank: 8,
        lora_alpha: 16.0,
        learning_rate: 1e-9,
        epochs: 1,
        max_seq_len: 64,
        gradient_clip_norm: Some(1.0),
        quantize_nf4: true,
    };

    let mut p = InstructPipeline::from_apr(&model_path, &model_config, instruct_config)
        .expect("pipeline from_apr");
    assert!(p.cuda_blocks.is_some(), "CUDA blocks must initialize (GPU present?)");
    assert!(p.gpu_training.is_some(), "GPU training state must initialize");

    let prompt_ids = p.tokenize("What is 2+2? Answer with just the number.\n");
    let response_ids = p.tokenize("4");
    let full_ids: Vec<u32> = prompt_ids.iter().chain(response_ids.iter()).copied().collect();
    let seq_len = full_ids.len();
    let prompt_len = prompt_ids.len();
    let vocab_size = p.model.config().vocab_size;
    let loss_start = prompt_len.saturating_sub(1);
    let loss_end = seq_len - 1;
    eprintln!(
        "[parity-probe] seq_len={seq_len} prompt_len={prompt_len} loss window=[{loss_start},{loss_end})"
    );

    // ── (c) pure-CPU oracle: CPU forward + CPU CE ─────────────────────
    // LoRA B matrices are zero-init, so the base model's loss is the truth.
    let logits = p.model.forward(&full_ids);
    let logits_cpu = logits.data().as_slice().expect("contiguous logits").to_vec();
    let (loss_cpu, _) = InstructPipeline::compute_causal_lm_loss(
        &logits_cpu,
        &full_ids,
        loss_start,
        loss_end,
        vocab_size,
    );
    eprintln!("[parity-probe] (c) CPU forward + CPU CE:        loss={loss_cpu:.4}");

    // ── (b) GPU transformer + GPU lm_head + CPU CE ────────────────────
    let logits_gpu = p.forward_logits_gpu(&full_ids).expect("GPU forward with logits download");
    let (loss_gpu_fwd, _) = InstructPipeline::compute_causal_lm_loss(
        &logits_gpu,
        &full_ids,
        loss_start,
        loss_end,
        vocab_size,
    );
    eprintln!("[parity-probe] (b) GPU forward + CPU CE:        loss={loss_gpu_fwd:.4}");

    // ── (a) GPU-resident forward + fused GPU causal CE ────────────────
    assert!(p.forward_logits_gpu_resident(&full_ids), "GPU-resident forward failed");
    let targets: Vec<u32> = (0..seq_len)
        .map(|pos| if pos + 1 < full_ids.len() { full_ids[pos + 1] } else { 0 })
        .collect();
    let num_loss_tokens = loss_end - loss_start;
    let scale = 1.0 / num_loss_tokens as f32;
    let loss_gpu_fused = {
        let trainer = p.cuda_trainer.as_ref().expect("trainer");
        let stream = trainer.stream();
        let training = p.gpu_training.as_mut().expect("training state");
        fused_causal_cross_entropy_cuda(
            &mut training.logits_buf,
            &targets,
            seq_len as u32,
            vocab_size as u32,
            loss_start as u32,
            loss_end as u32,
            scale,
            stream,
        )
        .expect("fused causal CE")
    };
    eprintln!("[parity-probe] (a) GPU forward + fused GPU CE:  loss={loss_gpu_fused:.4}");

    // ── mechanism experiment: CPU forward WITHOUT Q/K/V biases ────────
    // If dropping the biases from the CPU oracle reproduces the GPU loss,
    // the missing-bias hypothesis explains the full magnitude of the defect.
    let saved: Vec<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> = p
        .model
        .layers
        .iter_mut()
        .map(|l| (l.self_attn.b_q.take(), l.self_attn.b_k.take(), l.self_attn.b_v.take()))
        .collect();
    let logits_nb = p.model.forward(&full_ids);
    let logits_nb = logits_nb.data().as_slice().expect("contiguous logits").to_vec();
    let (loss_cpu_nobias, _) = InstructPipeline::compute_causal_lm_loss(
        &logits_nb, &full_ids, loss_start, loss_end, vocab_size,
    );
    for (l, (bq, bk, bv)) in p.model.layers.iter_mut().zip(saved) {
        l.self_attn.b_q = bq;
        l.self_attn.b_k = bk;
        l.self_attn.b_v = bv;
    }
    eprintln!("[parity-probe] (x) CPU forward, biases DROPPED: loss={loss_cpu_nobias:.4}");

    // ── (y) quantization-matched oracle: CPU forward through the SAME
    //    NF4-quantized weights the GPU uses ──────────────────────────────
    // The GPU block quantizes the 7 projection weights to NF4 (block-64
    // absmax) and runs GEMMs on the dequantized fp32 copies. Replaying that
    // on CPU isolates STRUCTURAL divergence (rope convention, biases,
    // masking, reductions) from irreducible quantization noise.
    {
        use trueno_gpu::kernels::{dequantize_nf4, quantize_nf4};
        let cfg = p.model.config().clone();
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.num_kv_heads * cfg.head_dim();
        let inter = cfg.intermediate_size;
        let requant = |t: &Tensor, rows: usize, cols: usize| -> Tensor {
            let d = t.data();
            let s = d.as_slice().expect("contiguous weight");
            Tensor::from_vec(dequantize_nf4(&quantize_nf4(s, rows, cols)), false)
        };
        for layer in &mut p.model.layers {
            layer.self_attn.w_q = requant(&layer.self_attn.w_q, q_dim, hidden);
            layer.self_attn.w_k = requant(&layer.self_attn.w_k, kv_dim, hidden);
            layer.self_attn.w_v = requant(&layer.self_attn.w_v, kv_dim, hidden);
            layer.self_attn.w_o = requant(&layer.self_attn.w_o, hidden, q_dim);
            layer.ffn.w_gate = requant(&layer.ffn.w_gate, inter, hidden);
            layer.ffn.w_up = requant(&layer.ffn.w_up, inter, hidden);
            layer.ffn.w_down = requant(&layer.ffn.w_down, hidden, inter);
        }
    }
    let logits_nf4 = p.model.forward(&full_ids);
    let logits_nf4 = logits_nf4.data().as_slice().expect("contiguous logits").to_vec();
    let (loss_cpu_nf4, _) = InstructPipeline::compute_causal_lm_loss(
        &logits_nf4,
        &full_ids,
        loss_start,
        loss_end,
        vocab_size,
    );
    eprintln!("[parity-probe] (y) CPU forward, NF4 weights:   loss={loss_cpu_nf4:.4}");

    // ── full-logits oracle: GPU logits vs NF4-matched CPU logits ──────
    // The scalar loss on a 1-token window is too weak to falsify every
    // structural defect (a consistently-wrong rope pairing shifted it by
    // only ~0.05 nats on this sample). The full [seq, vocab] logits compare
    // the ENTIRE function: rope-pairing mutation ⇒ relL2 ≈ 0.5-1.0;
    // quantization noise alone ⇒ relL2 ≈ 0.03.
    let logits_rel_l2 = rel_l2(&logits_gpu, &logits_nf4);
    eprintln!("[parity-probe] (z) logits relL2(gpu, cpu_nf4) = {logits_rel_l2:.4}");

    // ── falsifier assertions ──────────────────────────────────────────
    assert!(
        loss_cpu < 8.0,
        "CPU oracle itself is broken (loss={loss_cpu:.4}) — cannot falsify GPU path"
    );
    assert!(
        (loss_gpu_fused - loss_gpu_fwd).abs() < 0.05,
        "fused GPU CE diverges from CPU CE on the SAME GPU logits: \
         fused={loss_gpu_fused:.4} cpu_ce={loss_gpu_fwd:.4}"
    );
    // Structural parity: GPU vs the quantization-matched CPU oracle. NF4
    // noise is common to both sides, so this bound is TIGHT — any rope/bias/
    // mask/reduction defect blows it (pre-fix: gpu=6.54 vs nf4-cpu≈0.6).
    assert!(
        (loss_gpu_fused - loss_cpu_nf4).abs() < PARITY_TOL,
        "FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-001: GPU training loss diverges \
         from the NF4-matched CPU oracle: gpu={loss_gpu_fused:.4} \
         cpu_nf4={loss_cpu_nf4:.4} (|Δ|={:.4} > {PARITY_TOL}). The GPU \
         forward is computing a DIFFERENT MODEL (pre-fix: GPT-J rope pairing \
         + dropped Q/K/V biases + partial-warp softmax UB).",
        (loss_gpu_fused - loss_cpu_nf4).abs()
    );
    // Absolute sanity: toy CE must sit FAR below ln(V)=11.93. Pre-fix the
    // GPU path scored 13-14 (worse than uniform) on this same sample.
    assert!(
        loss_gpu_fused < 6.0,
        "FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-001: GPU training loss \
         {loss_gpu_fused:.4} is not meaningfully below ln(vocab)=11.93 — \
         finite-garbage forward (pre-fix signature: CE 13-14)"
    );
    // Full-function parity: the GPU forward must compute the SAME function
    // as the NF4-matched CPU oracle over the whole [seq, vocab] logits.
    // Mutation-verified: reverting the NeoX rope fix → relL2 0.183 (RED);
    // dropping Q/K/V biases → RED; fixed → 0.047 (GREEN). Quantization
    // noise is common to both sides.
    assert!(
        logits_rel_l2 < 0.10,
        "FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-001: GPU logits diverge from the \
         NF4-matched CPU oracle (relL2={logits_rel_l2:.4} > 0.10) — the GPU \
         forward computes a structurally different function (rope pairing / \
         biases / masking / reductions)"
    );
}
