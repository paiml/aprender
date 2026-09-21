//! #3714: backend selection for `qwen3_moe` generation.
//!
//! Before #3714 `apr run` sent every qwen3moe file to the CPU chain and
//! reported `used_gpu = false` without ever trying CUDA, so `--gpu` exited 14
//! with a message about a "runtime attempt" that never happened. This module
//! is the qwen3moe sibling of `run_qwen35_generate_dispatch`: the CUDA forward
//! (`Qwen3MoeCudaModel`) serves unless `--no-gpu` or a build without `cuda`,
//! it proves itself against the CPU forward on the real prompt first (the F2
//! rule every GPU path is held to), and a GPU that cannot serve says why —
//! unconditionally, not only under `--verbose` — before the CPU runs.

use super::qwen3_moe_generate::run_qwen3_moe_generate;
use crate::error::Result;
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

/// The line a GPU fallback starts with.
pub const QWEN3MOE_GPU_FALLBACK_PREFIX: &str = "qwen3moe: the CUDA forward did not serve this run";

/// Generate for a `qwen3_moe` file on the backend that serves it; the `bool`
/// is whether that was the GPU.
///
/// # Errors
/// Only the CPU forward's own failure: a GPU failure is a printed fallback.
pub fn run_qwen3_moe_generate_dispatch(
    mapped: &MappedGGUFModel,
    model: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &QuantizedGenerateConfig,
    no_gpu: bool,
) -> Result<(Vec<u32>, bool)> {
    #[cfg(feature = "cuda")]
    if !no_gpu {
        match gpu::run_qwen3_moe_generate_gpu(mapped, model, input_tokens, gen_config) {
            Ok(tokens) => return Ok((tokens, true)),
            Err(reason) => {
                eprintln!("{QWEN3MOE_GPU_FALLBACK_PREFIX}, falling back to CPU: {reason}");
            },
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = no_gpu;
    let tokens = run_qwen3_moe_generate(mapped, model, input_tokens, gen_config)?;
    Ok((tokens, false))
}

/// The MoE shape the GGUF metadata declares, or the key that is missing.
///
/// # Errors
/// A missing `{arch}.expert_count`, `.expert_used_count` or
/// `.expert_feed_forward_length`.
#[cfg(feature = "cuda")]
pub fn qwen3_moe_shape(
    mapped: &MappedGGUFModel,
) -> std::result::Result<crate::gguf::Qwen3MoeShape, String> {
    let m = &mapped.model;
    let arch = m.architecture().unwrap_or("qwen3moe");
    Ok(crate::gguf::Qwen3MoeShape {
        num_experts: m
            .expert_count()
            .ok_or_else(|| format!("'{arch}.expert_count' is missing from the GGUF metadata"))?,
        num_experts_per_tok: m.expert_used_count().ok_or_else(|| {
            format!("'{arch}.expert_used_count' is missing from the GGUF metadata")
        })?,
        expert_dim: m.expert_feed_forward_length().ok_or_else(|| {
            format!("'{arch}.expert_feed_forward_length' is missing from the GGUF metadata")
        })?,
    })
}

#[cfg(feature = "cuda")]
pub(crate) mod gpu {
    use super::qwen3_moe_shape;
    use crate::gguf::qwen3_moe_load::{load_qwen3_moe_layer, Qwen3MoeQuantizedLayer};
    use crate::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
        Qwen3MoeCudaModel, Qwen3MoeShape,
    };
    use crate::infer::qwen3_moe_generate::sample_from_logits;

    /// Positions the F2 guard forwards on both backends before the GPU may
    /// serve — the cap the dense and hybrid guards use.
    pub(crate) const QWEN3MOE_F2_PROBE_MAX: usize = 64;

    /// Every per-layer descriptor, or the first layer that would not load.
    pub(crate) fn load_moe_layers(
        mapped: &MappedGGUFModel,
        num_layers: usize,
    ) -> std::result::Result<Vec<Qwen3MoeQuantizedLayer>, String> {
        (0..num_layers)
            .map(|il| {
                load_qwen3_moe_layer(&mapped.model, mapped.data(), il)
                    .map_err(|e| format!("layer {il}'s MoE tensors would not load: {e}"))
            })
            .collect()
    }

    /// The GPU twin of `run_qwen3_moe_generate`: build the model on CUDA,
    /// prove it against the CPU forward, then decode with the CPU path's own
    /// sampler. `Err` is a fallback reason, never a user-visible failure.
    pub(crate) fn run_qwen3_moe_generate_gpu(
        mapped: &MappedGGUFModel,
        model: &OwnedQuantizedModel,
        input_tokens: &[u32],
        gen_config: &QuantizedGenerateConfig,
    ) -> std::result::Result<Vec<u32>, String> {
        if input_tokens.is_empty() {
            return Err("the prompt is empty".to_string());
        }
        let shape = qwen3_moe_shape(mapped)?;
        let moe_layers = load_moe_layers(mapped, model.config().num_layers)?;
        let build_start = std::time::Instant::now();
        let executor = crate::cuda::CudaExecutor::new(0)
            .map_err(|e| format!("CUDA initialization failed: {e}"))?;
        let max_seq_len = input_tokens.len() + gen_config.max_tokens + 1;
        let mut gpu = Qwen3MoeCudaModel::with_max_seq_len(
            model,
            &moe_layers,
            shape,
            mapped.data(),
            executor,
            max_seq_len,
        )
        .map_err(|e| format!("the CUDA model would not build: {e}"))?;

        let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
        let (device, vram_mb) = gpu.device_summary();
        // Unconditional, like every backend-selection line: a GPU run must be
        // distinguishable from a CPU one without --verbose.
        eprintln!(
            "Backend: GPU (CUDA, {device}, {vram_mb} MB VRAM) [qwen3moe routed-expert forward, \
             #3714; weights resident in {build_ms:.0} ms]"
        );

        f2_validate_qwen3_moe(
            &mut gpu,
            model,
            &moe_layers,
            shape,
            mapped.data(),
            input_tokens,
        )?;
        let decode_start = std::time::Instant::now();
        let tokens = decode(&mut gpu, input_tokens, gen_config)?;
        let generated = tokens.len().saturating_sub(input_tokens.len());
        let decode_s = decode_start.elapsed().as_secs_f64();
        eprintln!(
            "qwen3moe CUDA: {} prompt + {generated} generated tokens in {:.0} ms ({:.1} tok/s \
             including the token-by-token prefill)",
            input_tokens.len(),
            decode_s * 1000.0,
            (input_tokens.len() + generated) as f64 / decode_s.max(1e-9)
        );
        Ok(tokens)
    }

    /// Prefill + decode on the GPU with the CPU path's sampler and stop rule.
    fn decode(
        gpu: &mut Qwen3MoeCudaModel<'_>,
        input_tokens: &[u32],
        gen_config: &QuantizedGenerateConfig,
    ) -> std::result::Result<Vec<u32>, String> {
        use rand::SeedableRng;
        let max_seq_len = input_tokens.len() + gen_config.max_tokens + 1;
        let mut state = gpu
            .new_state()
            .map_err(|e| format!("the decode state would not allocate: {e}"))?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(gen_config.seed);

        let mut logits = Vec::new();
        for (pos, &token) in input_tokens.iter().enumerate() {
            logits = gpu
                .forward_single(token, &mut state, pos)
                .map_err(|e| format!("the GPU forward failed at prompt position {pos}: {e}"))?;
        }
        let mut tokens = input_tokens.to_vec();
        for _ in 0..gen_config.max_tokens {
            let next = sample_from_logits(&logits, gen_config, &mut rng, &tokens)
                .map_err(|e| format!("sampling failed: {e}"))?;
            tokens.push(next);
            if gen_config.stop_tokens.contains(&next) || tokens.len() >= max_seq_len {
                break;
            }
            let pos = tokens.len() - 1;
            logits = gpu
                .forward_single(next, &mut state, pos)
                .map_err(|e| format!("the GPU forward failed at decode position {pos}: {e}"))?;
        }
        Ok(tokens)
    }

    /// Per-position logits of the CPU forward over `probe`, then one greedy
    /// decode step — the reference the F2 guard judges the GPU against.
    ///
    /// Computed with FP32 activations (`with_fp32_activations`), not the
    /// production CPU path's Q8_K ones: this GPU forward runs float GEMVs, so
    /// the exact-activation CPU forward is the one it must reproduce. Measured
    /// on Qwen3-Coder-30B-A3B after a `<|im_start|>`: cosine 1.000000 at every
    /// one of 65 positions against this reference, 0.985 against the Q8_K one
    /// — judged against Q8_K the guard would be measuring the reference's
    /// quantization error, not the GPU (#3714).
    pub(crate) fn cpu_reference(
        model: &OwnedQuantizedModel,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        shape: Qwen3MoeShape,
        data: &[u8],
        probe: &[u32],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        crate::quantize::with_fp32_activations(|| {
            cpu_reference_q8k_or_fp32(model, moe_layers, shape, data, probe)
        })
    }

    /// The body of [`cpu_reference`], on whatever activation path the calling
    /// thread's scope selects.
    fn cpu_reference_q8k_or_fp32(
        model: &OwnedQuantizedModel,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        shape: Qwen3MoeShape,
        data: &[u8],
        probe: &[u32],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let mut cache = OwnedQuantizedKVCache::from_config(model.config(), probe.len() + 2);
        let mut per_pos = Vec::with_capacity(probe.len() + 1);
        let forward = |cache: &mut OwnedQuantizedKVCache, token: u32, pos: usize| {
            model
                .forward_single_qwen3_moe_with_cache(
                    token,
                    cache,
                    pos,
                    moe_layers,
                    shape.num_experts,
                    shape.num_experts_per_tok,
                    shape.expert_dim,
                    data,
                )
                .map_err(|e| format!("the CPU reference failed at position {pos}: {e}"))
        };
        for (pos, &token) in probe.iter().enumerate() {
            per_pos.push(forward(&mut cache, token, pos)?);
        }
        let next = per_pos.last().map_or(0, |l| crate::infer::argmax_u32(l));
        per_pos.push(forward(&mut cache, next, probe.len())?);
        Ok(per_pos)
    }

    /// The same positions and the same decode token, on the GPU, from a
    /// throwaway state.
    pub(crate) fn gpu_logits(
        gpu: &mut Qwen3MoeCudaModel<'_>,
        probe: &[u32],
        decode_token: u32,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let mut state = gpu
            .new_state()
            .map_err(|e| format!("the probe state would not allocate: {e}"))?;
        let mut per_pos = Vec::with_capacity(probe.len() + 1);
        for (pos, &token) in probe
            .iter()
            .chain(std::iter::once(&decode_token))
            .enumerate()
        {
            per_pos.push(
                gpu.forward_single(token, &mut state, pos)
                    .map_err(|e| format!("the GPU probe failed at position {pos}: {e}"))?,
            );
        }
        Ok(per_pos)
    }

    /// The F2 runtime guard: forward the real prompt (its last
    /// [`QWEN3MOE_F2_PROBE_MAX`] tokens) plus one greedy decode step on BOTH
    /// backends and let the GPU serve only if `f2_multi_position_report`
    /// accepts — the dense and hybrid paths' own rule and floors.
    fn f2_validate_qwen3_moe(
        gpu: &mut Qwen3MoeCudaModel<'_>,
        model: &OwnedQuantizedModel,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        shape: Qwen3MoeShape,
        data: &[u8],
        prompt: &[u32],
    ) -> std::result::Result<(), String> {
        if std::env::var("SKIP_PARITY_GATE").is_ok_and(|v| v == "1") {
            eprintln!("F2 guard: SKIP_PARITY_GATE=1 — nothing was compared on this run");
            return Ok(());
        }
        let probe = &prompt[prompt.len().saturating_sub(QWEN3MOE_F2_PROBE_MAX)..];
        if probe.len() < 2 {
            eprintln!(
                "F2 guard: a {}-token prompt has no real position to compare; nothing was judged",
                probe.len()
            );
            return Ok(());
        }
        let start = std::time::Instant::now();
        let cpu = cpu_reference(model, moe_layers, shape, data, probe)?;
        let decode_token = cpu
            .get(probe.len() - 1)
            .map_or(0, |l| crate::infer::argmax_u32(l));
        let gpu_per_pos = gpu_logits(gpu, probe, decode_token)?;
        let report = crate::infer::f2_multi_position_report(&cpu, &gpu_per_pos);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if report.accepted {
            eprintln!(
                "F2 guard: GPU matches the CPU forward on {} positions (min cosine {:.4}) in {ms:.0} ms",
                cpu.len(),
                report.min_cosine_real
            );
            Ok(())
        } else {
            Err(crate::infer::f2_divergence_msg(
                &report,
                crate::infer::F2ProbePath::Serial,
            ))
        }
    }
}
