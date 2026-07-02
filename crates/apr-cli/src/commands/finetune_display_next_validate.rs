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

/// Read LoRA rank/alpha from the trainer's sidecar `metadata.json`
/// (written next to safetensors checkpoints by `apr finetune`; the
/// safetensors `__metadata__` block may carry only epoch/val_loss).
fn read_sidecar_lora_params(adapter: &Path) -> (Option<u32>, Option<f32>) {
    let Some(sidecar) = adapter.parent().map(|d| d.join("metadata.json")) else {
        return (None, None);
    };
    let Ok(text) = std::fs::read_to_string(&sidecar) else {
        return (None, None);
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return (None, None);
    };
    let rank = json
        .get("lora_rank")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok());
    let alpha = json
        .get("lora_alpha")
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32);
    (rank, alpha)
}

/// Read LoRA rank/alpha from adapter metadata.
fn read_adapter_lora_params(adapter: &Path) -> Result<(u32, f32)> {
    // Handle safetensors adapter (from wgpu training pipeline)
    // Parse safetensors header to extract lora_rank/lora_alpha from metadata
    if adapter.extension().map(|e| e == "safetensors").unwrap_or(false) {
        let file = std::fs::File::open(adapter)
            .map_err(|e| CliError::ValidationFailed(format!("Read adapter: {e}")))?;
        let mut buf = [0u8; 8];
        std::io::Read::read_exact(&mut &file, &mut buf)
            .map_err(|e| CliError::ValidationFailed(format!("Read header: {e}")))?;
        let header_len = u64::from_le_bytes(buf) as usize;
        let mut header = vec![0u8; header_len];
        std::io::Read::read_exact(&mut (&file), &mut header)
            .map_err(|e| CliError::ValidationFailed(format!("Read header: {e}")))?;
        let header_str = String::from_utf8_lossy(&header);
        // Parse lora_rank and lora_alpha from JSON metadata
        let header_rank = header_str.split("\"lora_rank\"")
            .nth(1).and_then(|s| s.split('"').nth(1))
            .and_then(|v| v.parse::<u32>().ok());
        let header_alpha = header_str.split("\"lora_alpha\"")
            .nth(1).and_then(|s| s.split('"').nth(1))
            .and_then(|v| v.parse::<f32>().ok());
        // C-APR-MERGE-RUNNABLE: fall back to the trainer's sidecar
        // metadata.json before defaulting — a defaulted alpha silently
        // mis-scales the merged delta (scale = alpha/rank).
        let (sidecar_rank, sidecar_alpha) = read_sidecar_lora_params(adapter);
        let rank = header_rank.or(sidecar_rank).unwrap_or(64);
        let alpha = header_alpha.or(sidecar_alpha).unwrap_or(16.0);
        return Ok((rank, alpha));
    }
    // APR format adapter
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

/// C-APR-MERGE-RUNNABLE: merge writes an APR v2 container — refuse
/// format-in-disguise outputs (an APR container named `.safetensors` is
/// rejected by every loader AND by `safetensors` itself).
fn validate_merge_output_extension(out: &Path) -> Result<()> {
    match out.extension().and_then(|e| e.to_str()) {
        Some("apr") => Ok(()),
        other => Err(CliError::ValidationFailed(format!(
            "apr finetune --merge writes an APR v2 container; -o must end in .apr \
             (got: {}). Use `apr export --format {}` on the merged .apr for other formats.",
            out.display(),
            other.unwrap_or("safetensors"),
        ))),
    }
}

/// GH-376-style preset: (num_heads, num_kv_heads, intermediate_size) by
/// (architecture, hidden_size) for imports that lost head counts.
fn arch_preset_dims(arch: &str, hidden: usize) -> Option<(usize, usize, usize)> {
    if !arch.starts_with("qwen2") {
        return None;
    }
    match hidden {
        896 => Some((14, 2, 4864)),   // Qwen2.5-0.5B
        1536 => Some((12, 2, 8960)),  // Qwen2.5-1.5B
        3584 => Some((28, 4, 18944)), // Qwen2.5-7B
        _ => None,
    }
}

/// Parse a transformer block index from an HF (`model.layers.N.`) or
/// GGUF (`blk.N.`) style tensor name.
fn parse_layer_index(name: &str) -> Option<usize> {
    let rest = name
        .strip_prefix("model.layers.")
        .or_else(|| name.strip_prefix("blk."))?;
    rest.split('.').next()?.parse().ok()
}

/// Candidate adapter `(lora_a, lora_b)` tensor names for a base tensor.
///
/// Two conventions produced by this toolchain (C-APR-MERGE-RUNNABLE):
/// 1. Legacy full-name: `{base}.lora_a` / `{base}.lora_b`
/// 2. Entrenar trainer checkpoints (`cuda_trainer.rs` / `wgpu_checkpoint.rs`):
///    `lora.{layer}.{proj}.lora_a` for base
///    `model.layers.{layer}.self_attn.{proj}.weight` (and mlp projections).
fn adapter_pair_names(base_name: &str) -> Vec<(String, String)> {
    let mut candidates = vec![(
        format!("{base_name}.lora_a"),
        format!("{base_name}.lora_b"),
    )];
    if let (Some(layer), Some(proj)) = (
        parse_layer_index(base_name),
        base_name
            .strip_suffix(".weight")
            .and_then(|s| s.rsplit('.').next()),
    ) {
        candidates.push((
            format!("lora.{layer}.{proj}.lora_a"),
            format!("lora.{layer}.{proj}.lora_b"),
        ));
    }
    candidates
}

/// Backfill architecture + C-03 dims from tensor shapes/names when the base
/// metadata (post-canonicalization) still lacks them (C-APR-MERGE-RUNNABLE).
fn backfill_arch_dims(
    md: &mut aprender::format::v2::AprV2Metadata,
    tensors: &[aprender::format::rosetta::TensorInfo],
) {
    // vocab/hidden from embedding shape [vocab, hidden]
    if let Some(t) = tensors.iter().find(|t| {
        matches!(
            t.name.as_str(),
            "model.embed_tokens.weight" | "token_embd.weight" | "tok_embeddings.weight"
        )
    }) {
        if t.shape.len() == 2 {
            md.vocab_size = md.vocab_size.or(Some(t.shape[0]));
            md.hidden_size = md.hidden_size.or(Some(t.shape[1]));
        }
    }
    // num_layers = max block index + 1
    if md.num_layers.is_none() {
        md.num_layers = tensors
            .iter()
            .filter_map(|t| parse_layer_index(&t.name))
            .max()
            .map(|m| m + 1);
    }
    // intermediate_size from FFN gate/up shape [intermediate, hidden]
    if md.intermediate_size.is_none() {
        md.intermediate_size = tensors
            .iter()
            .find(|t| {
                t.shape.len() == 2
                    && (t.name.contains("mlp.gate_proj.weight")
                        || t.name.contains("ffn_gate.weight")
                        || t.name.contains("mlp.up_proj.weight")
                        || t.name.contains("ffn_up.weight"))
            })
            .map(|t| t.shape[0]);
    }
    // architecture from tensor-name pattern: Q/K/V projection *biases* are the
    // qwen2 signature (llama-family has none).
    if md.architecture.is_none() {
        let has_attn_bias = tensors.iter().any(|t| {
            t.name.ends_with("self_attn.q_proj.bias") || t.name.ends_with("attn_q.bias")
        });
        if has_attn_bias {
            md.architecture = Some("qwen2".to_string());
        }
    }
    // head counts via preset (not derivable from shapes without head_dim)
    if md.num_heads.is_none() || md.num_kv_heads.is_none() {
        if let (Some(arch), Some(hidden)) = (md.architecture.clone(), md.hidden_size) {
            if let Some((heads, kv_heads, intermediate)) = arch_preset_dims(&arch, hidden) {
                md.num_heads = md.num_heads.or(Some(heads));
                md.num_kv_heads = md.num_kv_heads.or(Some(kv_heads));
                md.intermediate_size = md.intermediate_size.or(Some(intermediate));
            }
        }
    }
}

/// Assert exactly one spelling of a metadata dimension is present and it is a
/// positive integer. More than one spelling (even a `null`-valued canonical
/// field next to an HF alias) makes realizar's serde-aliased deserializer fail
/// with "duplicate field" — which its loader swallows into EMPTY metadata.
fn gate_one_numeric(
    map: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
    what: &str,
) -> Result<()> {
    let present: Vec<&&str> = keys.iter().filter(|k| map.contains_key(**k)).collect();
    if present.len() > 1 {
        return Err(CliError::ValidationFailed(format!(
            "merged metadata has {n} spellings of {what} ({present:?}) — realizar's \
             alias-aware parser rejects duplicates and silently drops ALL metadata",
            n = present.len(),
        )));
    }
    let key = present.first().ok_or_else(|| {
        CliError::ValidationFailed(format!("merged metadata missing {what} (C-03)"))
    })?;
    let ok = map
        .get(**key)
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|v| v > 0);
    if ok {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "merged metadata {what} ({key}) is not a positive integer"
        )))
    }
}

/// Structural half of the post-merge gate: re-open the written container and
/// assert the RAW metadata JSON satisfies realizar's load requirements
/// (C-01 architecture, C-03 dims — exactly one spelling each) and carries a
/// loadable embedded tokenizer (vocabulary + merges|scores, PMAT-171/172).
fn gate_metadata_structural(bytes: &[u8]) -> Result<()> {
    let header =
        aprender::format::v2::AprV2Header::from_bytes(bytes).map_err(|e| {
            CliError::ValidationFailed(format!("post-merge gate: header re-parse failed: {e}"))
        })?;
    let start = usize::try_from(header.metadata_offset).unwrap_or(usize::MAX);
    let end = start.saturating_add(header.metadata_size as usize);
    let raw = bytes.get(start..end).ok_or_else(|| {
        CliError::ValidationFailed("post-merge gate: metadata section out of bounds".to_string())
    })?;
    let value: serde_json::Value = serde_json::from_slice(raw).map_err(|e| {
        CliError::ValidationFailed(format!("post-merge gate: metadata is not valid JSON: {e}"))
    })?;
    let map = value.as_object().ok_or_else(|| {
        CliError::ValidationFailed("post-merge gate: metadata is not a JSON object".to_string())
    })?;

    // C-01: architecture (realizar cannot infer model type without it)
    let arch_ok = map
        .get("architecture")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|a| !a.is_empty());
    if !arch_ok {
        return Err(CliError::ValidationFailed(
            "merged metadata missing 'architecture' (C-01)".to_string(),
        ));
    }
    // C-03 dims, one spelling each (realizar serde-alias groups)
    gate_one_numeric(map, &["hidden_size", "hidden_dim", "d_model", "n_embd"], "hidden_size")?;
    gate_one_numeric(
        map,
        &["num_layers", "n_layers", "num_hidden_layers", "n_layer"],
        "num_layers",
    )?;
    gate_one_numeric(
        map,
        &["num_heads", "n_heads", "num_attention_heads", "n_head"],
        "num_heads",
    )?;
    gate_one_numeric(
        map,
        &["intermediate_size", "ffn_dim", "intermediate_dim", "n_inner"],
        "intermediate_size",
    )?;
    // num_kv_heads is optional (defaults to num_heads) but must not be duplicated
    let kv_present = ["num_kv_heads", "n_kv_heads", "num_key_value_heads"]
        .iter()
        .filter(|k| map.contains_key(**k))
        .count();
    if kv_present > 1 {
        return Err(CliError::ValidationFailed(
            "merged metadata has multiple spellings of num_kv_heads".to_string(),
        ));
    }

    // Embedded tokenizer (PMAT-172: APR must be self-contained; PMAT-171 BPE
    // loader needs vocabulary + merges; GH-366 SentencePiece needs scores).
    let vocab_len = map
        .get("tokenizer.vocabulary")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if vocab_len == 0 {
        return Err(CliError::ValidationFailed(
            "merged model has no embedded tokenizer (tokenizer.vocabulary missing/empty). \
             Re-convert the base with `apr convert` so it carries an embedded tokenizer \
             (PMAT-172: APR files must be self-contained)."
                .to_string(),
        ));
    }
    let merges_len = map
        .get("tokenizer.merges")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    let scores_len = map
        .get("tokenizer.scores")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if merges_len == 0 && scores_len == 0 {
        return Err(CliError::ValidationFailed(
            "merged model tokenizer is not loadable: needs tokenizer.merges (BPE, PMAT-171) \
             or tokenizer.scores (SentencePiece, GH-366) alongside tokenizer.vocabulary"
                .to_string(),
        ));
    }
    Ok(())
}

/// Oracle half of the post-merge gate: load the output through REALIZAR's own
/// deserializer + config extraction — the exact code path `apr run` uses.
#[cfg(feature = "inference")]
fn gate_realizar_oracle(out: &Path) -> Result<()> {
    let mapped = realizar::apr::MappedAprModel::from_path(out).map_err(|e| {
        CliError::ValidationFailed(format!("post-merge gate: realizar mmap load failed: {e}"))
    })?;
    let vocab_size = mapped.metadata.vocab_size.unwrap_or(0);
    realizar::gguf::GGUFConfig::from_apr(&mapped, vocab_size).map_err(|e| {
        CliError::ValidationFailed(format!(
            "post-merge gate: realizar config extraction failed (C-01/C-03): {e}"
        ))
    })?;
    let model = realizar::apr::AprV2Model::load(out).map_err(|e| {
        CliError::ValidationFailed(format!("post-merge gate: realizar APR load failed: {e}"))
    })?;
    let tokenizer_loads = model.load_embedded_bpe_tokenizer().is_some()
        || model.load_embedded_sentencepiece_tokenizer().is_some();
    if tokenizer_loads {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(
            "post-merge gate: realizar cannot load an embedded tokenizer from the merged model"
                .to_string(),
        ))
    }
}

/// Fail-closed post-write gate (C-APR-MERGE-RUNNABLE): re-open the output and
/// assert it is directly runnable. On failure the output is DELETED — merge
/// must never leave an unrunnable artifact on disk.
fn verify_merged_runnable(out: &Path) -> Result<()> {
    let bytes = std::fs::read(out).map_err(|e| {
        CliError::ValidationFailed(format!("post-merge gate: cannot re-read output: {e}"))
    })?;
    aprender::format::v2::AprV2Reader::from_bytes(&bytes).map_err(|e| {
        CliError::ValidationFailed(format!("post-merge gate: output is not valid APR v2: {e}"))
    })?;
    gate_metadata_structural(&bytes)?;
    #[cfg(feature = "inference")]
    gate_realizar_oracle(out)?;
    Ok(())
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
    validate_merge_output_extension(out)?;

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
    let mut merged_count = 0u64;

    // Read base model as V2 to preserve metadata INCLUDING embedded tokenizer.
    // Previous code used AprWriter (v1) which lost the tokenizer section.
    let base_bytes = std::fs::read(model)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read base model: {e}")))?;
    let base_v2 = aprender::format::v2::AprV2Reader::from_bytes(&base_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse base model as V2: {e}")))?;
    let mut metadata = base_v2.metadata().clone();

    // Add merge provenance to metadata
    metadata.custom.insert(
        "merge_source".to_string(),
        serde_json::json!(model.display().to_string()),
    );
    metadata.custom.insert(
        "merge_adapter".to_string(),
        serde_json::json!(adapter.display().to_string()),
    );
    metadata.custom.insert("lora_rank".to_string(), serde_json::json!(lora_rank));
    metadata.custom.insert("lora_alpha".to_string(), serde_json::json!(lora_alpha));

    // C-APR-MERGE-RUNNABLE: import-produced bases carry HF-alias dimension
    // keys (num_hidden_layers, …) in `custom`. Re-serializing the cloned
    // metadata used to emit BOTH `"num_layers": null` (typed field) AND the
    // alias — poisoning realizar's alias-aware parser with "duplicate field",
    // which its loader swallowed into EMPTY metadata (C-01 + lost tokenizer).
    // Canonicalize aliases into typed fields, then backfill anything still
    // missing from tensor shapes / GH-376-style presets.
    metadata.canonicalize_hf_aliases();
    backfill_arch_dims(&mut metadata, &base_report.tensors);
    // Merged tensors are written as F32 — drop stale quantization markers
    // cloned from a quantized base.
    metadata.quantization = None;
    if metadata.model_type.starts_with("transformer_lm") {
        metadata.model_type = "transformer_lm".to_string();
    }

    let mut writer = aprender::format::v2::AprV2Writer::new(metadata);

    for ti in &base_report.tensors {
        let base_data = rosetta
            .load_tensor_f32(model, &ti.name)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load {}: {e}", ti.name)))?;

        let pair = adapter_pair_names(&ti.name)
            .into_iter()
            .find(|(a, b)| adapter_names.contains(a) && adapter_names.contains(b));

        let merged = if let Some((a_name, b_name)) = pair {
            let lora_a = rosetta
                .load_tensor_f32(adapter, &a_name)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to load {a_name}: {e}")))?;
            let lora_b = rosetta
                .load_tensor_f32(adapter, &b_name)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to load {b_name}: {e}")))?;

            // C-APR-MERGE-RUNNABLE: rank from the adapter's own [rank, d_in]
            // lora_a shape — a stale/defaulted global rank makes MergeEngine
            // mis-derive dims and silently return the base unchanged.
            let rank = adapter_report
                .tensors
                .iter()
                .find(|t| t.name == a_name)
                .filter(|t| t.shape.len() == 2)
                .and_then(|t| u32::try_from(t.shape[0]).ok())
                .unwrap_or(lora_rank);

            merged_count += 1;
            engine.merge(&base_data, &lora_a, &lora_b, lora_alpha, rank)
        } else {
            base_data
        };

        writer.add_f32_tensor(&ti.name, ti.shape.clone(), &merged);
    }

    // FALSIFY-APR-MERGE-RUNNABLE-005: a merge that merges NOTHING while the
    // adapter carries LoRA tensors is a silent no-op (wrong naming scheme) —
    // fail loudly instead of shipping a byte-identical copy of the base.
    let adapter_lora_count = adapter_names.iter().filter(|n| n.ends_with(".lora_a")).count();
    if merged_count == 0 && adapter_lora_count > 0 {
        let example = adapter_names
            .iter()
            .find(|n| n.ends_with(".lora_a"))
            .cloned()
            .unwrap_or_default();
        return Err(CliError::ValidationFailed(format!(
            "Adapter contains {adapter_lora_count} LoRA tensor pairs but NONE matched any base \
             tensor (example adapter tensor: {example}). Supported namings: \
             '{{base_tensor}}.lora_a' or 'lora.{{layer}}.{{proj}}.lora_a'. \
             Refusing to write a no-op merge."
        )));
    }

    let bytes = writer.write().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to serialize merged model: {e}"))
    })?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    // FALSIFY-APR-MERGE-RUNNABLE-001: fail-closed post-write gate. The merged
    // file must be directly runnable by `apr run` (C-01/C-03 metadata +
    // loadable embedded tokenizer). On failure: delete the output, error loud.
    if let Err(e) = verify_merged_runnable(out) {
        let _ = std::fs::remove_file(out);
        return Err(CliError::ValidationFailed(format!(
            "Post-merge runnability gate FAILED — output deleted ({}): {e}",
            out.display()
        )));
    }

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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Estimate parameters from GGUF metadata or file size fallback.
/// GH-625: file_size * 2 overestimates (Q4_K_M has mixed dtypes + F32 biases).
fn estimate_params_from_file(path: &Path) -> Result<u64> {
    // Try RosettaStone inspection for accurate param count
    if let Ok(report) = aprender::format::rosetta::RosettaStone::new().inspect(path) {
        if report.total_params > 0 {
            return Ok(report.total_params as u64);
        }
    }
    // Fallback: file size heuristic (Q4 ≈ 0.5 bytes/param)
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model file: {e}")))?;
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

