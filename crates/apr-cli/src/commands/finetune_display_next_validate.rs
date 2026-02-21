/// Display next steps when no training data is provided.
fn display_next_steps(json_output: bool) {
    if !json_output {
        println!();
        println!("{}", "NEXT STEPS".white().bold());
        println!("{}", "─".repeat(50));
        println!("  Provide --data <train.jsonl> to start training.");
        println!("  Example: apr finetune model.apr --method lora --data train.jsonl -o adapter/");
    }
}

/// Validate and resolve merge input paths.
fn validate_merge_paths<'a>(
    model_path: Option<&'a Path>,
    adapter_path: Option<&'a Path>,
) -> Result<(&'a Path, &'a Path)> {
    let model = model_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Model path required for merge. Usage: apr finetune merge model.apr --adapter adapter/"
                .to_string(),
        )
    })?;
    let adapter = adapter_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Adapter path required for merge. Use --adapter <path>".to_string(),
        )
    })?;
    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }
    if !adapter.exists() {
        return Err(CliError::FileNotFound(adapter.to_path_buf()));
    }
    Ok((model, adapter))
}

/// Read LoRA rank/alpha from adapter metadata.
fn read_adapter_lora_params(adapter: &Path) -> Result<(u32, f32)> {
    let reader = aprender::serialization::apr::AprReader::open(adapter)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read adapter: {e}")))?;
    let rank = reader
        .get_metadata("lora_rank")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(16) as u32;
    let alpha = reader
        .get_metadata("lora_alpha")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(16.0) as f32;
    Ok((rank, alpha))
}

/// Display merge result.
#[allow(clippy::disallowed_methods)]
fn display_merge_result(
    model: &Path,
    adapter: &Path,
    output_path: &Path,
    output_size: u64,
    merged_count: u64,
    total_layers: usize,
    lora_rank: u32,
    lora_alpha: f32,
    json_output: bool,
) {
    if json_output {
        let json = serde_json::json!({
            "status": "merged",
            "base_model": model.display().to_string(),
            "adapter": adapter.display().to_string(),
            "output": output_path.display().to_string(),
            "output_size": output_size,
            "merged_layers": merged_count,
            "total_layers": total_layers,
            "rank": lora_rank,
            "alpha": lora_alpha,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::pipeline_stage("Merging", output::StageStatus::Done);
        println!();
        output::subheader("Merge Complete");
        println!(
            "{}",
            output::kv_table(&[
                ("Layers merged", format!("{merged_count} / {total_layers}")),
                (
                    "Output size",
                    humansize::format_size(output_size, humansize::BINARY)
                ),
                ("Output", output_path.display().to_string()),
            ])
        );
    }
}

/// Run adapter merge (finetune merge)
#[allow(clippy::disallowed_methods)]
fn run_merge(
    model_path: Option<&Path>,
    adapter_path: Option<&Path>,
    output_path: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let (model, adapter) = validate_merge_paths(model_path, adapter_path)?;
    let out = output_path.unwrap_or(Path::new("merged.apr"));

    if !json_output {
        output::header("APR Finetune — Merge Adapter");
        println!(
            "{}",
            output::kv_table(&[
                ("Base model", model.display().to_string()),
                ("Adapter", adapter.display().to_string()),
                ("Output", out.display().to_string()),
            ])
        );
        println!();
        output::pipeline_stage("Merging", output::StageStatus::Running);
    }

    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let base_report = rosetta
        .inspect(model)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect base model: {e}")))?;
    let adapter_report = rosetta
        .inspect(adapter)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect adapter: {e}")))?;

    let (lora_rank, lora_alpha) = read_adapter_lora_params(adapter)?;

    let adapter_names: std::collections::HashSet<String> = adapter_report
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .collect();

    let engine = MergeEngine::new();
    let mut writer = aprender::serialization::apr::AprWriter::new();
    let mut merged_count = 0u64;

    writer.set_metadata(
        "merge_source",
        serde_json::json!(model.display().to_string()),
    );
    writer.set_metadata(
        "merge_adapter",
        serde_json::json!(adapter.display().to_string()),
    );
    writer.set_metadata("lora_rank", serde_json::json!(lora_rank));
    writer.set_metadata("lora_alpha", serde_json::json!(lora_alpha));

    for ti in &base_report.tensors {
        let base_data = rosetta
            .load_tensor_f32(model, &ti.name)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load {}: {e}", ti.name)))?;

        let a_name = format!("{}.lora_a", ti.name);
        let b_name = format!("{}.lora_b", ti.name);

        let merged = if adapter_names.contains(&a_name) && adapter_names.contains(&b_name) {
            let lora_a = rosetta
                .load_tensor_f32(adapter, &a_name)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to load {a_name}: {e}")))?;
            let lora_b = rosetta
                .load_tensor_f32(adapter, &b_name)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to load {b_name}: {e}")))?;

            merged_count += 1;
            engine.merge(&base_data, &lora_a, &lora_b, lora_alpha, lora_rank)
        } else {
            base_data
        };

        writer.add_tensor_f32(&ti.name, ti.shape.clone(), &merged);
    }

    let bytes = writer.to_bytes().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to serialize merged model: {e}"))
    })?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    display_merge_result(
        model,
        adapter,
        out,
        bytes.len() as u64,
        merged_count,
        base_report.tensors.len(),
        lora_rank,
        lora_alpha,
        json_output,
    );
    Ok(())
}

/// Check if a tensor name is eligible for LoRA adaptation.
fn is_lora_eligible(name: &str) -> bool {
    // Target attention and MLP projection layers
    let targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "self_attn",
        "mlp",
    ];
    // Must be a weight tensor (not bias, norm, embedding)
    let is_weight = name.ends_with(".weight") || name.ends_with("weight");
    let is_excluded = name.contains("embed")
        || name.contains("norm")
        || name.contains("bias")
        || name.contains("lm_head")
        || name.contains("token_embd")
        || name.contains("wte")
        || name.contains("wpe");

    is_weight && !is_excluded && targets.iter().any(|t| name.contains(t))
}

/// Deterministic pseudo-random seed from tensor name + index.
fn hash_seed(name: &str, idx: usize) -> u64 {
    // Simple FNV-1a inspired hash for deterministic initialization
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash ^= idx as u64;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    hash
}

/// Parse model size string (e.g., "7B", "1.5B", "70B")
fn parse_model_size(size: &str) -> Result<u64> {
    let size = size.to_uppercase();
    let (num_str, multiplier) = if size.ends_with('B') {
        (&size[..size.len() - 1], 1_000_000_000u64)
    } else if size.ends_with('M') {
        (&size[..size.len() - 1], 1_000_000u64)
    } else {
        return Err(CliError::ValidationFailed(format!(
            "Invalid model size format: {size}. Use: 7B, 1.5B, 70B, etc."
        )));
    };

    let num: f64 = num_str.parse().map_err(|_| {
        CliError::ValidationFailed(format!("Invalid number in model size: {num_str}"))
    })?;

    Ok((num * multiplier as f64) as u64)
}

/// Estimate parameters from model file size
fn estimate_params_from_file(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model file: {e}")))?;
    // Conservative estimate: assume Q4 (~0.5 bytes/param)
    Ok(metadata.len() * 2)
}

/// Format parameter count for display
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{params}")
    }
}

