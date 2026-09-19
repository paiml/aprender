

// ═══════════════════════════════════════════════════════════════════════════════
// #3477: THE HYBRID ARM — Qwen3.5 (Gated DeltaNet) CPU vs CUDA
// ═══════════════════════════════════════════════════════════════════════════════
//
// `apr parity` had ONE loop: the dense `forward_single_with_cache` against
// `forward_gpu_resident`. Everything it could not route was refused from the
// header (PMAT-1098). #3517 stopped refusing `qwen35` — the hybrid now has a GPU
// forward (`Qwen35CudaModel`, #3090) as well as a CPU one (`Qwen35Model`, #3091)
// — but the dense loop still asked `OwnedQuantizedModel::from_mapped` for the
// layers, and the dense loader refuses the hybrid. So a REFUSAL (exit 12,
// UNMEASURED-TOOL in the C14 row) turned into a plain FAIL naming the MODEL.
// The missing piece was never a refusal: it is this second pair of forwards.

/// Which pair of forwards measures an architecture (#3477).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParityArm {
    /// `Qwen35Model::forward_single_qwen35` vs `Qwen35CudaModel::forward_single`.
    Hybrid,
    /// `OwnedQuantizedModel::forward_single_with_cache` vs `forward_gpu_resident`.
    Dense,
}

/// Pure: the arm `run` dispatches `architecture` to.
///
/// Split out of `run` so the choice is falsifiable with no GPU and no model
/// file. The predicate is realizar's own dispatch predicate — the same one
/// `apr run`, `apr chat`, `apr qa`, `apr ptx-map` and the parity refusal ask —
/// so which loop parity measures cannot drift from which forward actually
/// serves the tokens.
#[cfg(feature = "inference")]
#[allow(dead_code)] // the only non-test caller is `run`, which is `cuda`-gated
pub(crate) fn parity_arm(architecture: &str) -> ParityArm {
    if realizar::gguf::hybrid_forward_handles(architecture) {
        ParityArm::Hybrid
    } else {
        ParityArm::Dense
    }
}

/// The refusal for a hybrid GGUF whose quantization the GPU weight upload has
/// no kernel for, or `None`.
///
/// The hybrid arm exists because BOTH forwards exist — but the CPU forward
/// decodes quantizations the GPU upload cannot take (the 0.8B file on this box
/// is IQ4_XS). Discovering that INSIDE `Qwen35CudaModel::with_max_seq_len`
/// produces `FAIL … Operation 'qwen35_cuda_upload' not supported`, a red C14 row
/// naming the MODEL for what is a missing KERNEL. Discovering it from the header
/// produces the refusal (exit 12) that `scripts/check_model_parity.sh` reads as
/// UNMEASURED-TOOL — the same verdict the architecture refusal earns, for the
/// same reason: no run of this build can produce that number.
///
/// The predicate is realizar's own header-level SSOT — `hybrid_gpu_quant_refusal`
/// delegating to `gpu_unsupported_quant_qtype` (PMAT-781/783/785), the whitelist
/// the upload itself enforces — so parity cannot refuse a file the GPU would
/// have taken, nor admit one it would have decoded as Q4_K.
#[cfg(feature = "inference")]
#[allow(dead_code)] // the only non-test caller is `run`, which is `cuda`-gated
pub(crate) fn hybrid_quant_refusal<'n>(
    architecture: &str,
    tensors: impl IntoIterator<Item = (&'n str, u32)>,
) -> Option<ParityRefusal> {
    let detail = realizar::gguf::hybrid_gpu_quant_refusal(architecture, tensors)?;
    // realizar's message opens by naming the architecture, which the refusal
    // line has already printed as `architecture=…`; drop the duplicate rather
    // than say it twice.
    let prefix = format!("Architecture '{architecture}': ");
    let reason = detail.strip_prefix(&prefix).unwrap_or(&detail).to_string();
    Some(ParityRefusal {
        architecture: architecture.to_string(),
        reason,
        issue: "PMAT-785",
    })
}

/// The per-position detail a `--verbose` run owes a failing position.
///
/// Shared by both arms: a diagnostic that exists on one loop and not the other
/// is how the two drift apart.
#[cfg(feature = "cuda")]
fn print_verbose_failure(m: &SpcMetrics) {
    eprintln!(
        "{}     {} mean_diff={:.6} rmse={:.6} oos={}/{} {}",
        "│".dimmed(),
        "".dimmed(),
        m.mean_abs_diff,
        m.rmse,
        m.out_of_spec_count,
        m.vocab_size,
        "│".dimmed(),
    );
}

/// The report every arm ends with: the JSON document (GH-636) or the SPC
/// footer + summary + auto-diagnosis, and the exit status the display promises
/// (GH-615).
///
/// ONE function, not one per arm: the JSON shape and the exit rule must be
/// identical whichever pair of forwards produced the metrics, or the C14 row
/// starts depending on the architecture it measured.
#[cfg(feature = "cuda")]
fn emit_parity_outcome(
    file: &Path,
    all_metrics: &[SpcMetrics],
    hidden_dim: usize,
    num_heads: usize,
    kv_heads: usize,
    json: bool,
) -> Result<()> {
    let has_failures = all_metrics.iter().any(|m| m.verdict().is_fail());

    if json {
        let metrics_json: Vec<serde_json::Value> = all_metrics
            .iter()
            .map(|m| {
                serde_json::json!({
                    "position": m.position,
                    "token_id": m.token_id,
                    "cpu_argmax": m.cpu_argmax,
                    "gpu_argmax": m.gpu_argmax,
                    "max_abs_diff": m.max_abs_diff,
                    "mean_abs_diff": m.mean_abs_diff,
                    "cosine_similarity": m.cosine_similarity,
                    "kl_divergence": m.kl_divergence,
                    "sigma_level": m.sigma_level,
                    "cpk": m.cpk(),
                    "verdict": format!("{:?}", m.verdict()),
                })
            })
            .collect();
        let summary = serde_json::json!({
            "model": file.display().to_string(),
            "tokens": all_metrics.len(),
            "passed": all_metrics.iter().filter(|m| !m.verdict().is_fail()).count(),
            "failed": all_metrics.iter().filter(|m| m.verdict().is_fail()).count(),
            "parity": !has_failures,
            "metrics": metrics_json,
        });
        println!("{}", serde_json::to_string_pretty(&summary).unwrap_or_default());
        return if has_failures {
            Err(CliError::ValidationFailed(
                "PARITY DISPROVEN: GPU/CPU divergence exceeds tolerance".to_string(),
            ))
        } else {
            Ok(())
        };
    }

    print_footer();
    print_summary(all_metrics);
    auto_diagnose(all_metrics, hidden_dim, num_heads, kv_heads);

    if has_failures {
        Err(CliError::ValidationFailed(
            "PARITY DISPROVEN: GPU/CPU divergence exceeds tolerance".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// What one hybrid measurement produced: the per-position metrics, plus the
/// attention geometry `auto_diagnose` prints.
#[cfg(feature = "cuda")]
struct HybridParity {
    metrics: Vec<SpcMetrics>,
    hidden_dim: usize,
    num_heads: usize,
    kv_heads: usize,
}

/// Measure the Qwen3.5 hybrid: its CPU forward (#3091) against its CUDA twin
/// (#3090), position by position over `tokens`, into the same `SpcMetrics` the
/// dense loop produces.
///
/// Both backends see the SAME token at the SAME position from their own fresh
/// state, exactly as the F2 runtime guard does
/// (`forward_qwen35::f2_validate_qwen35`) — so a divergence this reports is the
/// one that would reject the GPU at inference time, not a different experiment.
///
/// Returned rather than printed-and-dropped so the end-to-end contract
/// (≥64 positions, cosine and argmax) is assertable from a test.
///
/// # Errors
/// A hybrid file whose layers will not load, a CUDA device that will not
/// initialize, or either forward failing at any position.
#[cfg(feature = "cuda")]
fn measure_hybrid(
    mapped: &realizar::gguf::MappedGGUFModel,
    tokens: &[u32],
    verbose: bool,
) -> Result<HybridParity> {
    use realizar::gguf::forward_qwen35::Qwen35Model;
    use realizar::gguf::Qwen35CudaModel;

    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to load the Qwen3.5 base model: {e}"))
    })?;
    let cpu = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).map_err(
        |e| CliError::ValidationFailed(format!("Failed to load the Qwen3.5 hybrid layers: {e}")),
    )?;

    // The geometry of the FULL-ATTENTION layers, which is what `head_geometry`
    // and `auto_diagnose`'s GQA analysis describe. The interleaved Gated
    // DeltaNet layers have their own shape (num_k_heads / head_k_dim /
    // num_v_heads / conv_kernel) that these three numbers do not describe at
    // all — a DeltaNet divergence is NOT a GQA-ratio story, and the diagnosis
    // must not invent one.
    let config = base.config();
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
        "Hybrid: Gated DeltaNet interleaved with gated full attention (#3090/#3091)".dimmed()
    );

    let executor = realizar::cuda::CudaExecutor::new(0)
        .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;
    let device_name = executor
        .device_name()
        .unwrap_or_else(|_| "Unknown GPU".to_string());
    let vram_mb = executor.memory_info().unwrap_or((0, 0)).1 / (1024 * 1024);

    let max_seq = tokens.len() + 1;
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&cpu, executor, max_seq).map_err(|e| {
        CliError::ValidationFailed(format!("CUDA hybrid model build failed: {e}"))
    })?;

    eprintln!(
        "  {} {} ({} MB VRAM)",
        "GPU:".white().bold(),
        device_name.green(),
        vram_mb,
    );

    let mut cpu_state = cpu.new_state(max_seq);
    let mut gpu_state = gpu.new_state().map_err(|e| {
        CliError::ValidationFailed(format!("CUDA decode state allocation failed: {e}"))
    })?;

    eprintln!();
    print_header();

    let mut metrics = Vec::with_capacity(tokens.len());
    for (pos, &token_id) in tokens.iter().enumerate() {
        let cpu_logits = cpu
            .forward_single_qwen35(token_id, &mut cpu_state, pos)
            .map_err(|e| {
                CliError::InferenceFailed(format!("CPU forward failed at pos {pos}: {e}"))
            })?;
        let gpu_logits = gpu.forward_single(token_id, &mut gpu_state, pos).map_err(|e| {
            CliError::InferenceFailed(format!("GPU forward failed at pos {pos}: {e}"))
        })?;

        let m = compute_metrics(&cpu_logits, &gpu_logits, pos, token_id);
        print_row(&m);
        if verbose && m.verdict().is_fail() {
            print_verbose_failure(&m);
        }
        metrics.push(m);
    }

    Ok(HybridParity {
        metrics,
        hidden_dim,
        num_heads,
        kv_heads,
    })
}

/// The hybrid arm of `run`: measure, then emit the shared report.
#[cfg(feature = "cuda")]
fn run_hybrid(
    file: &Path,
    mapped: &realizar::gguf::MappedGGUFModel,
    tokens: &[u32],
    verbose: bool,
    json: bool,
) -> Result<()> {
    let measured = measure_hybrid(mapped, tokens, verbose)?;
    emit_parity_outcome(
        file,
        &measured.metrics,
        measured.hidden_dim,
        measured.num_heads,
        measured.kv_heads,
        json,
    )
}
