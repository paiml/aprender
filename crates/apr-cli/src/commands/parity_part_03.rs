
// ═══════════════════════════════════════════════════════════════════════════════
// MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
pub fn run(file: &Path, prompt: &str, assert: bool, verbose: bool) -> Result<()> {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };

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
        "{}",
        "══════════════════════════════════════════════════════════════════════"
            .cyan()
            .bold()
    );

    // ── Load model ──────────────────────────────────────────────────────────
    let mapped = MappedGGUFModel::from_path(file)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to map model: {e}")))?;

    let tokens = mapped.model.encode(prompt).unwrap_or_else(|| vec![1u32]);

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let config = &model.config;
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = if num_heads > 0 {
        hidden_dim / num_heads
    } else {
        0
    };
    let kv_dim = kv_heads * head_dim;
    let num_layers = config.num_layers;
    let gqa_ratio = if kv_heads > 0 {
        num_heads / kv_heads
    } else {
        0
    };

    eprintln!();
    eprintln!("  {} {}", "Model:".white().bold(), file.display());
    eprintln!("  {} {:?}", "Prompt:".white().bold(), prompt);
    eprintln!(
        "  {} {} tokens: {:?}",
        "Tokens:".white().bold(),
        tokens.len(),
        &tokens[..tokens.len().min(20)],
    );
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

        all_metrics.push(m);
    }

    print_footer();

    // ── Summary statistics ──────────────────────────────────────────────────
    print_summary(&all_metrics);

    // ── Auto-diagnosis ──────────────────────────────────────────────────────
    auto_diagnose(&all_metrics, hidden_dim, num_heads, kv_heads);

    // ── Exit code ───────────────────────────────────────────────────────────
    let has_failures = all_metrics.iter().any(|m| m.verdict().is_fail());
    if has_failures && assert {
        Err(CliError::ValidationFailed(
            "PARITY DISPROVEN: GPU/CPU divergence exceeds tolerance".to_string(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
pub fn run(_file: &Path, _prompt: &str, _assert: bool, _verbose: bool) -> Result<()> {
    Err(CliError::FeatureDisabled(
        "cuda feature required for parity check".to_string(),
    ))
}
