// ═══════════════════════════════════════════════════════════════════════════════
// #3714 R2: THE MoE ARM — Qwen3-MoE CPU vs CUDA
// ═══════════════════════════════════════════════════════════════════════════════
//
// `apr parity` refused qwen3moe from the header (PMAT-1098, #3367): the dense
// loop would read the MoE placeholder, and no GPU forward existed. #3714 R1
// built one (`Qwen3MoeCudaModel`), so the architecture the runtime dispatches
// to the routed-expert forward has BOTH forwards and parity measures them.
//
// The pair is the one the runtime F2 guard judges (`qwen3_moe_dispatch`):
// the CPU `forward_single_qwen3_moe_with_cache` with FP32 activations against
// `Qwen3MoeCudaModel::forward_single`. The GPU runs float GEMVs; judged against
// the default Q8_K-activation CPU it lost 0.13 cosine on Qwen3-30B-A3B-
// Instruct-2507's first tokens to the REFERENCE's quantization (#3714, #3751),
// so parity measuring any other pair would report a divergence the runtime
// never sees — or miss one it rejects.

/// What one MoE measurement produced: the per-position metrics, plus the
/// attention geometry `auto_diagnose` prints.
#[cfg(feature = "cuda")]
struct MoeParity {
    metrics: Vec<SpcMetrics>,
    hidden_dim: usize,
    num_heads: usize,
    kv_heads: usize,
}

/// Measure the Qwen3-MoE routed-expert forward: CPU (FP32 activations)
/// against its CUDA twin, position by position over `tokens`, into the same
/// `SpcMetrics` the dense and hybrid loops produce.
///
/// Both backends see the SAME token at the SAME position from their own fresh
/// state, as the F2 guard does, so a divergence this reports is one that would
/// reject the GPU at inference time.
///
/// # Errors
/// MoE tensors that will not load, a CUDA device that will not initialize, a
/// device that cannot hold the model (the capacity plan's arithmetic, exit 14 —
/// a busy or small GPU is not a parity result), or either forward failing.
#[cfg(feature = "cuda")]
fn measure_moe(
    mapped: &realizar::gguf::MappedGGUFModel,
    tokens: &[u32],
    verbose: bool,
) -> Result<MoeParity> {
    use realizar::gguf::{OwnedQuantizedKVCache, OwnedQuantizedModel, Qwen3MoeCudaModel};

    let model = OwnedQuantizedModel::from_mapped(mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load the MoE model: {e}")))?;
    let shape = realizar::infer::qwen3_moe_dispatch::qwen3_moe_shape(mapped)
        .map_err(CliError::ValidationFailed)?;
    let config = model.config();
    let layers = (0..config.num_layers)
        .map(|il| {
            realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer(&mapped.model, mapped.data(), il)
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load the MoE layers: {e}")))?;

    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let (head_dim, _kv_dim, gqa_ratio) = head_geometry(hidden_dim, num_heads, kv_heads);
    eprintln!(
        "  {} hidden={} heads={} kv_heads={} head_dim={} GQA={} layers={} vocab={}",
        "Arch:".white().bold(),
        hidden_dim,
        num_heads,
        kv_heads,
        head_dim,
        gqa_ratio,
        config.num_layers,
        config.vocab_size,
    );
    eprintln!(
        "  {}",
        format!(
            "MoE: {} experts, {} routed per token, expert FFN {} (#3367 CPU / #3714 CUDA); \
             CPU reference with FP32 activations",
            shape.num_experts, shape.num_experts_per_tok, shape.expert_dim
        )
        .dimmed()
    );

    let executor = realizar::cuda::CudaExecutor::new(0)
        .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;
    let max_seq = tokens.len() + 1;
    let mut gpu = Qwen3MoeCudaModel::with_max_seq_len(
        &model,
        &layers,
        shape,
        mapped.data(),
        executor,
        max_seq,
    )
    .map_err(|e| match e {
        realizar::error::RealizarError::CapacityRefused(_) => {
            CliError::BackendUnavailable(format!("{e}"))
        },
        other => CliError::ValidationFailed(format!("CUDA MoE model build failed: {other}")),
    })?;
    let (device_name, vram_mb) = gpu.device_summary();
    eprintln!(
        "  {} {} ({} MB VRAM)",
        "GPU:".white().bold(),
        device_name.green(),
        vram_mb,
    );

    let cpu_logits: Vec<Vec<f32>> = realizar::quantize::with_fp32_activations(|| {
        let mut cache = OwnedQuantizedKVCache::from_config(model.config(), max_seq);
        tokens
            .iter()
            .enumerate()
            .map(|(pos, &token_id)| {
                model
                    .forward_single_qwen3_moe_with_cache(
                        token_id,
                        &mut cache,
                        pos,
                        &layers,
                        shape.num_experts,
                        shape.num_experts_per_tok,
                        shape.expert_dim,
                        mapped.data(),
                    )
                    .map_err(|e| format!("CPU forward failed at pos {pos}: {e}"))
            })
            .collect::<std::result::Result<Vec<_>, String>>()
    })
    .map_err(CliError::InferenceFailed)?;

    let mut gpu_state = gpu.new_state().map_err(|e| {
        CliError::ValidationFailed(format!("CUDA decode state allocation failed: {e}"))
    })?;

    eprintln!();
    print_header();

    let mut metrics = Vec::with_capacity(tokens.len());
    for (pos, (&token_id, cpu)) in tokens.iter().zip(&cpu_logits).enumerate() {
        let gpu_logits = gpu.forward_single(token_id, &mut gpu_state, pos).map_err(|e| {
            CliError::InferenceFailed(format!("GPU forward failed at pos {pos}: {e}"))
        })?;
        let m = compute_metrics(cpu, &gpu_logits, pos, token_id);
        print_row(&m);
        if verbose && m.verdict().is_fail() {
            print_verbose_failure(&m);
        }
        metrics.push(m);
    }

    Ok(MoeParity {
        metrics,
        hidden_dim,
        num_heads,
        kv_heads,
    })
}

/// The MoE arm of `run`: measure, then emit the shared report.
#[cfg(feature = "cuda")]
fn run_moe(
    file: &Path,
    mapped: &realizar::gguf::MappedGGUFModel,
    tokens: &[u32],
    verbose: bool,
    json: bool,
) -> Result<()> {
    let measured = measure_moe(mapped, tokens, verbose)?;
    emit_parity_outcome(
        file,
        &measured.metrics,
        measured.hidden_dim,
        measured.num_heads,
        measured.kv_heads,
        json,
    )
}

/// End-to-end on the real file (#3714 done_when 2): ≥ 64 positions, every
/// position PASSing parity's own verdict. Skips without the file, a device, or
/// room on it.
#[cfg(all(test, feature = "cuda"))]
mod parity_moe_tests {
    use super::measure_moe;

    const MODEL: &str = "/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf";

    const PROMPT: &str = "<|im_start|>user\nThe history of the printing press begins in the \
        fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink \
        and a wooden screw press into a system that could reproduce books quickly and cheaply. \
        Within fifty years, presses operated in more than two hundred cities across Europe. \
        Printed pamphlets carried arguments about religion, science and politics to readers who \
        had never owned a manuscript, and the price of a book fell by an order of magnitude.";

    #[test]
    fn parity_moe_holds_at_64_positions_on_the_real_file() {
        let path = std::env::var("APR_QWEN3MOE_GGUF").unwrap_or_else(|_| MODEL.to_string());
        if !std::path::Path::new(&path).exists() {
            eprintln!("SKIP: {path} is absent");
            return;
        }
        if realizar::cuda::CudaExecutor::num_devices() == 0 {
            eprintln!("SKIP: no CUDA device");
            return;
        }
        let mapped = realizar::gguf::MappedGGUFModel::from_path(&path).expect("map the GGUF");
        let tokens = mapped.model.encode(PROMPT).expect("tokenize");
        assert!(tokens.len() >= 64, "the prompt must reach 64 positions, got {}", tokens.len());
        let measured = match measure_moe(&mapped, &tokens, false) {
            Ok(m) => m,
            Err(crate::error::CliError::BackendUnavailable(why)) => {
                eprintln!("SKIP: {why}");
                return;
            },
            Err(e) => panic!("measure: {e}"),
        };
        assert_eq!(measured.metrics.len(), tokens.len());
        for m in &measured.metrics {
            assert!(
                !m.verdict().is_fail(),
                "pos {}: parity FAIL (cpu argmax {} gpu argmax {}, max |diff| {})",
                m.position,
                m.cpu_argmax,
                m.gpu_argmax,
                m.max_abs_diff
            );
        }
    }
}
