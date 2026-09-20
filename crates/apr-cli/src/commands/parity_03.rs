

/// The pre-load refusal (PMAT-1098): an architecture the dense CPU-vs-GPU loop
/// cannot route is refused from the GGUF header alone — before any weight is
/// materialized — so the C14 row reports a TOOL limitation, never a MODEL FAIL.
#[cfg(feature = "cuda")]
fn refuse_unroutable(mapped: &realizar::gguf::MappedGGUFModel, json: bool) -> Result<()> {
    let arch = mapped.model.architecture().unwrap_or_default().to_string();
    let names: Vec<&str> = mapped.model.tensors.iter().map(|t| t.name.as_str()).collect();
    // #3477: a hybrid this build CAN route may still carry a quantization its
    // GPU upload has no kernel for. Both refusals are header-only, so both
    // happen here, before a weight is materialized.
    let refusal = parity_refusal_for(&arch, names).or_else(|| {
        hybrid_quant_refusal(
            &arch,
            mapped
                .model
                .tensors
                .iter()
                .map(|t| (t.name.as_str(), t.qtype)),
        )
    });
    let Some(refusal) = refusal else {
        return Ok(());
    };
    eprintln!();
    eprintln!("{}", refusal.line());
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&refusal.json()).unwrap_or_default()
        );
    }
    Err(refusal.into_error())
}

/// `(head_dim, kv_dim, gqa_ratio)` from the attention geometry; zero heads
/// yield zeros rather than a division by zero.
fn head_geometry(hidden_dim: usize, num_heads: usize, kv_heads: usize) -> (usize, usize, usize) {
    let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 0 };
    let gqa_ratio = if kv_heads > 0 { num_heads / kv_heads } else { 0 };
    (head_dim, kv_heads * head_dim, gqa_ratio)
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(feature = "cuda")]
#[provable_contracts_macros::contract("apr-cli-command-safety-v1", equation = "read_only_no_side_effects")]
pub fn run(file: &Path, prompt: &str, _assert: bool, verbose: bool, json: bool) -> Result<()> {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };
    contract_pre_gpu_cpu_parity!();

    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    // ── Header ──────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!(
        "{}",
        "══════════════════════════════════════════════════════════════════════"
            .cyan()
            .bold()
    );
    eprintln!(
        "  {}  {}",
        "apr parity".cyan().bold(),
        "GPU/CPU Statistical Process Control".white()
    );
    eprintln!(
        "  {}",
        "Scope: compares inference OUTPUT (logits/tokens) between GPU and CPU".dimmed()
    );
    eprintln!(
        "  {}",
        "(See also: apr ptx-map for kernel dispatch verification)".dimmed()
    );
    eprintln!(
        "  {}",
        "Exit: 0 parity held · 5 parity disproven · 12 REFUSED (this build cannot measure"
            .dimmed()
    );
    eprintln!(
        "  {}",
        "        that architecture — the line says which, and the issue that lifts it)".dimmed()
    );
    eprintln!(
        "{}",
        "══════════════════════════════════════════════════════════════════════"
            .cyan()
            .bold()
    );

    // ── Load model ──────────────────────────────────────────────────────────
    let mapped = MappedGGUFModel::from_path(file)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to map model: {e}")))?;

    // PMAT-1098: REFUSE, before a single weight is materialized, an architecture
    // this dense CPU-vs-GPU loop cannot route. `MappedGGUFModel::from_path`
    // above is a mmap + header parse — no tensor bytes are read — so an 18 GB
    // MoE is refused in header time, not after a minute of loading. Falling
    // through here is what made the 0.68.0 T-2 dogfood report a TOOL limitation
    // as three FAIL rows naming the MODEL.
    refuse_unroutable(&mapped, json)?;

    let tokens = mapped.model.encode(prompt).unwrap_or_else(|| vec![1u32]);

    eprintln!();
    eprintln!("  {} {}", "Model:".white().bold(), file.display());
    eprintln!("  {} {:?}", "Prompt:".white().bold(), prompt);
    eprintln!(
        "  {} {} tokens: {:?}",
        "Tokens:".white().bold(),
        tokens.len(),
        &tokens[..tokens.len().min(20)],
    );

    // #3477: the Qwen3.5 hybrid is no longer refused above (#3517) because it
    // HAS both forwards — but it is not the dense pair. `OwnedQuantizedModel::
    // from_mapped` below would hand it straight to `dense_loader_refusal`,
    // which is the FAIL this dispatch replaces with a measurement.
    if parity_arm(mapped.model.architecture().unwrap_or_default()) == ParityArm::Hybrid {
        return run_hybrid(file, &mapped, &tokens, verbose, json);
    }

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let config = model.config();
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let (head_dim, kv_dim, gqa_ratio) = head_geometry(hidden_dim, num_heads, kv_heads);
    let num_layers = config.num_layers;

    eprintln!(
        "  {} hidden={} heads={} kv_heads={} head_dim={} GQA={} layers={} vocab={}",
        "Arch:".white().bold(),
        hidden_dim,
        num_heads,
        kv_heads,
        head_dim,
        gqa_ratio,
        num_layers,
        config.vocab_size,
    );

    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;

    eprintln!(
        "  {} {} ({} MB VRAM)",
        "GPU:".white().bold(),
        cuda_model.device_name().green(),
        cuda_model.vram_mb(),
    );

    let max_seq = tokens.len() + 1;

    // ── Run parity check ────────────────────────────────────────────────────
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    cuda_model.executor_mut().reset_kv_cache_gpu();

    eprintln!();
    print_header();

    let mut all_metrics = Vec::new();

    for (pos, &token_id) in tokens.iter().enumerate() {
        let cpu_logits = cuda_model
            .model()
            .forward_single_with_cache(token_id, &mut cpu_cache, pos)
            .map_err(|e| {
                CliError::InferenceFailed(format!("CPU forward failed at pos {pos}: {e}"))
            })?;

        let gpu_logits = cuda_model
            .forward_gpu_resident(token_id, &mut gpu_cache, pos)
            .map_err(|e| {
                CliError::InferenceFailed(format!("GPU forward failed at pos {pos}: {e}"))
            })?;

        let m = compute_metrics(&cpu_logits, &gpu_logits, pos, token_id);
        print_row(&m);

        if verbose && m.verdict().is_fail() {
            print_verbose_failure(&m);
        }

        all_metrics.push(m);
    }

    // ── Report ──────────────────────────────────────────────────────────────
    // GH-636 (JSON shape), GH-615 (exit code matches the display) and the
    // summary/auto-diagnosis, shared verbatim with the #3477 hybrid arm
    // (`parity_hybrid.rs`) so the C14 row cannot depend on which pair of
    // forwards produced the metrics.
    emit_parity_outcome(file, &all_metrics, hidden_dim, num_heads, kv_heads, json)
}

#[cfg(not(feature = "cuda"))]
pub fn run(_file: &Path, _prompt: &str, _assert: bool, _verbose: bool, _json: bool) -> Result<()> {
    Err(CliError::FeatureDisabled(
        "cuda feature required for parity check".to_string(),
    ))
}

#[cfg(test)]
mod head_geometry_tests {
    use super::head_geometry;

    #[test]
    fn head_geometry_divides_when_heads_are_present() {
        assert_eq!(head_geometry(4096, 32, 8), (128, 1024, 4));
    }

    #[test]
    fn head_geometry_yields_zeros_instead_of_dividing_by_zero() {
        assert_eq!(head_geometry(4096, 0, 0), (0, 0, 0));
        assert_eq!(head_geometry(4096, 32, 0), (128, 0, 0));
    }
}
