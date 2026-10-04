//! FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-003 on a device: the instruct pipeline hands each NF4
//! block its LoRA adapter as `(Aᵀ, s·Bᵀ)` and takes it back as `(A, B)`.
//!
//! Defect. `LoRALayer` holds A `[rank, d_in]` and B `[d_out, rank]`, and the NF4 block computes
//! `(x·A)·B` with row-major `gemm_forward`, so it needs `Aᵀ` and `Bᵀ`. `init_cuda` and
//! `sync_lora_to_cpu` copied the buffers raw, which reshapes them. A raw round trip is the
//! identity, so CUDA training and CUDA resume looked right, while the CPU forward, checkpoints,
//! merge and PEFT export read a different adapter from the one the device trained.
//!
//! Falsifier, on `TransformerConfig::tiny()` at rank 4 and alpha 8 (s = 2), dense A and B:
//!   (c1) after `init_cuda`, every block holds exactly `(Aᵀ, 2·Bᵀ)`;
//!   (c2) `sync_lora_to_cpu` right after `init_cuda` gives back A and B bit for bit;
//!   (a)  an adapter whose product B·A is exactly zero (rows 2..4 of A, columns 0..2 of B
//!        zero), uploaded as `(Aᵀ, 2·Bᵀ)`, leaves the logits as they are with B = 0, and
//!        uploaded raw it moves them, so the device GEMMs run `(x·A)·B` on what they are given;
//!   (b)  the dense adapter's logit delta on the device is at least 2× closer to the CPU
//!        `forward_with_lora` delta than to the delta of the raw-copied adapter.
//!
//! Run it on a CUDA device, holding the GPU lock around the test binary only:
//!
//! ```text
//! cargo test -p aprender-train --features cuda --lib --no-run
//! <that test binary> falsify_cuda_nf4_train_loss_parity_003 --ignored --nocapture
//! ```

#[allow(clippy::wildcard_imports)]
use super::*;

#[test]
#[ignore = "requires a CUDA GPU"]
fn falsify_cuda_nf4_train_loss_parity_003_instruct_lora_layout() {
    if !trueno_gpu::driver::cuda_available() {
        eprintln!("[lora-layout] SKIP: no CUDA device");
        return;
    }
    // Closures, not helper fns, so cargo-mutants generates no mutant for code only reachable
    // from this `#[ignore]` GPU test.
    let dist = |x: &[f32], y: &[f32]| -> f32 {
        x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt()
    };
    let norm = |x: &[f32]| -> f32 { x.iter().map(|v| v * v).sum::<f32>().sqrt() };
    let val = |seed: usize, i: usize| -> f32 {
        let h = (i * 2_654_435_761 + seed * 40_503) % 1_000_003;
        (h as f32 / 1_000_003.0 - 0.5) * 0.1
    };
    let scaled = |(a, b): (Vec<f32>, Vec<f32>), s: f32| -> (Vec<f32>, Vec<f32>) {
        (a, b.iter().map(|v| v * s).collect())
    };
    // One (A, B) pair per slot, B scaled by s as `download_lora_weights` returns it.
    let upload = |p: &mut InstructPipeline, pairs: &[(Vec<f32>, Vec<f32>)]| {
        let blocks = p.cuda_blocks.as_mut().expect("cuda blocks");
        for (i, block) in blocks.iter_mut().enumerate() {
            let ((a_q, b_q), (a_v, b_v)) = (&pairs[2 * i], &pairs[2 * i + 1]);
            block.upload_lora_weights(a_q, b_q, a_v, b_v).expect("upload LoRA weights");
        }
    };

    let model_config = TransformerConfig::tiny();
    let (rank, alpha) = (4, 8.0);
    let s = alpha / rank as f32;
    let instruct_config = InstructConfig {
        lora_rank: rank,
        lora_alpha: alpha,
        learning_rate: 1e-4,
        epochs: 1,
        max_seq_len: 16,
        gradient_clip_norm: None,
        quantize_nf4: false,
        ..InstructConfig::default()
    };
    // Built without CUDA, so the adapters can be set before `init_cuda` uploads them.
    let mut p = InstructPipeline::new(&model_config, instruct_config);
    for (idx, layer) in p.lora_layers.iter_mut().enumerate() {
        let a = (0..layer.rank() * layer.d_in()).map(|i| val(2 * idx + 1, i)).collect();
        let b = (0..layer.d_out() * layer.rank()).map(|i| val(2 * idx + 2, i)).collect();
        *layer.lora_a_mut() = crate::Tensor::from_vec(a, true);
        *layer.lora_b_mut() = crate::Tensor::from_vec(b, true);
    }
    let cpu_layers = p.lora_layers.clone();
    assert_eq!(cpu_layers.len(), 2 * model_config.num_hidden_layers);

    p.config.quantize_nf4 = true;
    p.init_cuda(&model_config);
    assert!(
        p.cuda_blocks.is_some() && p.gpu_training.is_some(),
        "NF4 CUDA init must succeed (VRAM guard, kernels, training state)"
    );

    // (c1) The device holds (Aᵀ, s·Bᵀ).
    for (i, block) in p.cuda_blocks.as_ref().expect("cuda blocks").iter().enumerate() {
        let (a_q, b_q, a_v, b_v) = block.download_lora_weights().expect("download LoRA");
        for (slot, a, b) in [(2 * i, a_q, b_q), (2 * i + 1, a_v, b_v)] {
            let (a_t, b_t) = scaled(cpu_layers[slot].device_layout(), s);
            assert!(a == a_t, "(c1) slot {slot}: the device A is not Aᵀ");
            assert!(b == b_t, "(c1) slot {slot}: the device B is not s·Bᵀ");
        }
    }

    // (c2) A CPU -> GPU -> CPU round trip is exact.
    p.sync_lora_to_cpu();
    for (slot, (got, want)) in p.lora_layers.iter().zip(&cpu_layers).enumerate() {
        assert!(got.lora_a().data() == want.lora_a().data(), "(c2) slot {slot}: A changed");
        assert!(got.lora_b().data() == want.lora_b().data(), "(c2) slot {slot}: B changed");
    }

    let ids: [u32; 8] = [1, 5, 9, 42, 100, 7, 3, 999];
    let dense = p.forward_logits_gpu(&ids).expect("GPU forward, dense adapter");
    let zero_b: Vec<_> = cpu_layers
        .iter()
        .map(|l| {
            let (a_t, b_t) = l.device_layout();
            (a_t, vec![0.0; b_t.len()])
        })
        .collect();
    upload(&mut p, &zero_b);
    let base = p.forward_logits_gpu(&ids).expect("GPU forward, B = 0");
    let base_norm = norm(&base);

    // (a) A zero-product adapter is a no-op only in the device layout.
    let null_layers: Vec<LoRALayer> = cpu_layers
        .iter()
        .map(|l| {
            let (r, d_in) = (l.rank(), l.d_in());
            let mut z = l.clone();
            let a = l.lora_a().data().iter().enumerate();
            let a = a.map(|(i, &v)| if i / d_in >= r / 2 { 0.0 } else { v }).collect();
            let b = l.lora_b().data().iter().enumerate();
            let b = b.map(|(i, &v)| if i % r < r / 2 { 0.0 } else { v }).collect();
            *z.lora_a_mut() = crate::Tensor::from_vec(a, true);
            *z.lora_b_mut() = crate::Tensor::from_vec(b, true);
            z
        })
        .collect();
    let null_device: Vec<_> = null_layers.iter().map(|l| scaled(l.device_layout(), s)).collect();
    let null_raw: Vec<_> = null_layers
        .iter()
        .map(|l| scaled((l.lora_a().data().to_vec(), l.lora_b().data().to_vec()), s))
        .collect();
    upload(&mut p, &null_device);
    let null_logits = p.forward_logits_gpu(&ids).expect("GPU forward, B·A = 0 in device layout");
    upload(&mut p, &null_raw);
    let raw_logits = p.forward_logits_gpu(&ids).expect("GPU forward, B·A = 0 copied raw");
    let (null_d, raw_d) = (dist(&null_logits, &base), dist(&raw_logits, &base));
    eprintln!("[lora-layout] (a) |base| {base_norm:.4e} device {null_d:.3e} raw {raw_d:.3e}");
    assert!(null_d <= 1e-6 * base_norm, "(a) B·A = 0 in device layout moved the logits");
    assert!(raw_d > 1e-3 * base_norm, "(a) power: B·A = 0 copied raw must move the logits");

    // (b) The device computes the CPU adapter, not the raw-copied one.
    let delta = |with: &[f32], without: &[f32]| -> Vec<f32> {
        with.iter().zip(without).map(|(w, o)| w - o).collect()
    };
    let gpu_delta = delta(&dense, &base);
    let cpu_base = p.model.forward(&ids).data().to_vec();
    let cpu_with = p.model.forward_with_lora(&ids, &cpu_layers).data().to_vec();
    let cpu_delta = delta(&cpu_with, &cpu_base);
    let raw_layers: Vec<LoRALayer> = cpu_layers
        .iter()
        .map(|l| {
            let mut raw = l.clone();
            raw.set_from_device_layout(&l.lora_a().data().to_vec(), &l.lora_b().data().to_vec());
            raw
        })
        .collect();
    let raw_with = p.model.forward_with_lora(&ids, &raw_layers).data().to_vec();
    let raw_delta = delta(&raw_with, &cpu_base);
    let (gpu_n, cpu_n) = (norm(&gpu_delta), norm(&cpu_delta));
    let (to_cpu, to_raw) = (dist(&gpu_delta, &cpu_delta), dist(&gpu_delta, &raw_delta));
    let cpu_to_raw = dist(&cpu_delta, &raw_delta);
    eprintln!(
        "[lora-layout] (b) |gpu Δ| {gpu_n:.4e} |cpu Δ| {cpu_n:.4e} gpu→cpu {to_cpu:.4e} \
         gpu→raw {to_raw:.4e} cpu→raw {cpu_to_raw:.4e}"
    );
    assert!(gpu_n > 1e-3 * base_norm, "(b) the dense adapter must move the GPU logits");
    assert!(cpu_to_raw > 0.5 * cpu_n, "(b) power: the raw copy must be another adapter");
    assert!(
        2.0 * to_cpu < to_raw,
        "(b) the device adapter is not the CPU adapter: distance {to_cpu:.4e} to it, \
         {to_raw:.4e} to the raw copy"
    );
}
