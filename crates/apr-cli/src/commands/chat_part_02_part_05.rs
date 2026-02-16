/// Load tokenizers based on model format.
/// F-MODEL-COMPLETE-001: GGUF tokenizer failure is fatal.
fn load_tokenizers(
    format: ModelFormat,
    model_bytes: &[u8],
    path: &Path,
) -> Result<(Option<LlamaTokenizer>, Option<Qwen2BpeTokenizer>), CliError> {
    match format {
        ModelFormat::Gguf => {
            let tok = LlamaTokenizer::from_gguf_bytes(model_bytes).map_err(|e| {
                CliError::InvalidFormat(format!(
                    "Model is incomplete: Failed to load GGUF tokenizer: {}. \
                    This usually indicates a corrupted or improperly converted model.",
                    e
                ))
            })?;
            println!(
                "{} tokenizer with {} tokens",
                "Loaded".green(),
                tok.vocab_size()
            );
            Ok((Some(tok), None))
        }
        ModelFormat::SafeTensors | ModelFormat::Apr => {
            let tok = find_qwen_tokenizer(path)?;
            Ok((None, tok))
        }
        ModelFormat::Demo => Ok((None, None)),
    }
}

/// Print SafeTensors config info for user feedback.
fn print_safetensors_config(path: &Path) {
    let Some(parent) = path.parent() else { return; };
    let Ok(json) = std::fs::read_to_string(parent.join("config.json")) else { return; };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) else { return; };
    println!(
        "{} config: {} layers, {} hidden, {} heads",
        "Loaded".green(),
        v["num_hidden_layers"].as_u64().unwrap_or(0),
        v["hidden_size"].as_u64().unwrap_or(0),
        v["num_attention_heads"].as_u64().unwrap_or(0),
    );
}

/// Detect model architecture from format-specific metadata.
/// GH-222: APR v2 metadata, GGUF parsed metadata, SafeTensors config.json.
fn detect_model_architecture(format: ModelFormat, model_bytes: &[u8], path: &Path) -> String {
    match format {
        ModelFormat::Gguf => detect_arch_from_gguf(model_bytes, path),
        ModelFormat::Apr => detect_arch_from_apr(model_bytes, path),
        ModelFormat::SafeTensors => detect_arch_from_config(path),
        ModelFormat::Demo => "demo".to_string(),
    }
}

fn detect_arch_from_gguf(model_bytes: &[u8], path: &Path) -> String {
    use realizar::gguf::GGUFModel;
    match GGUFModel::from_bytes(model_bytes) {
        Ok(gguf) => gguf.architecture().unwrap_or("unknown").to_string(),
        Err(_) => path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
    }
}

/// GH-222: Read architecture from APR v2 metadata, then config.json, then dir name.
fn detect_arch_from_apr(model_bytes: &[u8], path: &Path) -> String {
    if let Ok(reader) = aprender::format::v2::AprV2Reader::from_bytes(model_bytes) {
        if let Some(apr_arch) = &reader.metadata().architecture {
            if !apr_arch.is_empty() {
                return apr_arch.clone();
            }
        }
    }
    let arch = read_model_type_from_config(path);
    if arch != "unknown" { return arch; }
    dir_name_fallback(path)
}

/// PMAT-120: Read architecture from config.json, then dir name.
fn detect_arch_from_config(path: &Path) -> String {
    let arch = read_model_type_from_config(path);
    if arch != "unknown" { return arch; }
    dir_name_fallback(path)
}

/// Read model_type or architectures[0] from sibling config.json.
fn read_model_type_from_config(path: &Path) -> String {
    let Some(parent) = path.parent() else { return "unknown".to_string(); };
    let Ok(json) = std::fs::read_to_string(parent.join("config.json")) else { return "unknown".to_string(); };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) else { return "unknown".to_string(); };
    if let Some(model_type) = v["model_type"].as_str() {
        return model_type.to_lowercase();
    }
    if let Some(archs) = v["architectures"].as_array() {
        if let Some(first) = archs.first().and_then(|a| a.as_str()) {
            return first
                .trim_end_matches("ForCausalLM")
                .trim_end_matches("LMHeadModel")
                .to_lowercase();
        }
    }
    "unknown".to_string()
}

fn dir_name_fallback(path: &Path) -> String {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn template_format_name(tf: TemplateFormat) -> &'static str {
    match tf {
        TemplateFormat::ChatML => "ChatML",
        TemplateFormat::Llama2 => "LLaMA2",
        TemplateFormat::Mistral => "Mistral",
        TemplateFormat::Phi => "Phi",
        TemplateFormat::Alpaca => "Alpaca",
        TemplateFormat::Custom => "Custom",
        TemplateFormat::Raw => "Raw",
    }
}

/// GH-224: Try to initialize GGUF CUDA model from a mapped model.
/// Returns (cuda_model, init_failed).
#[cfg(feature = "cuda")]
fn try_init_gguf_cuda(
    mapped: &realizar::gguf::MappedGGUFModel,
) -> (Option<realizar::gguf::OwnedQuantizedModelCuda>, bool) {
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
    if !OwnedQuantizedModelCuda::is_available() {
        return (None, false);
    }
    let owned = match OwnedQuantizedModel::from_mapped(mapped) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[GGUF model parse failed: {}, will use CPU]", e);
            return (None, true);
        }
    };
    match OwnedQuantizedModelCuda::new(owned, 0) {
        Ok(cuda_model) => {
            println!(
                "{}",
                format!(
                    "[GGUF CUDA: {} ({} MB VRAM) — pre-cached]",
                    cuda_model.device_name(),
                    cuda_model.vram_mb()
                )
                .bright_green()
            );
            (Some(cuda_model), false)
        }
        Err(e) => {
            println!(
                "{}",
                format!("[GGUF CUDA init failed: {}, will use CPU]", e).yellow()
            );
            (None, true)
        }
    }
}

/// GH-224: Try to initialize APR CUDA model.
/// GH-272: Warns about F32 performance when VRAM > 2GB.
/// Returns (cuda_model, init_failed).
#[cfg(feature = "cuda")]
fn try_init_apr_cuda(
    model_bytes: &[u8],
    path: &Path,
) -> (Option<realizar::apr::AprV2ModelCuda>, bool) {
    use realizar::apr::{AprV2Model, AprV2ModelCuda};
    if !AprV2ModelCuda::is_available() {
        return (None, false);
    }
    let apr_model = match AprV2Model::from_bytes(model_bytes.to_vec()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[APR model parse failed: {}, will use CPU]", e);
            return (None, true);
        }
    };
    match AprV2ModelCuda::new(apr_model, 0) {
        Ok(cuda_model) => {
            let vram_mb = cuda_model.vram_mb();
            println!(
                "{}",
                format!(
                    "[APR CUDA: {} ({} MB VRAM) — pre-cached]",
                    cuda_model.device_name(),
                    vram_mb
                )
                .bright_green()
            );
            // GH-272: Warn about F32 performance when model VRAM > 2GB
            if vram_mb > 2048 {
                print_apr_f32_perf_tip(vram_mb, path);
            }
            (Some(cuda_model), false)
        }
        Err(e) => {
            eprintln!("[APR CUDA init failed: {}, will use CPU]", e);
            (None, true)
        }
    }
}

/// GH-272: Print F32 performance tip suggesting APR-native Q4K quantization.
#[cfg(feature = "cuda")]
fn print_apr_f32_perf_tip(vram_mb: u64, path: &Path) {
    println!(
        "{}",
        format!(
            "  Performance tip: This APR model uses {} MB VRAM (F32 tensors).",
            vram_mb
        )
        .yellow()
    );
    println!(
        "{}",
        "  For ~4x faster inference, quantize to Q4K:".yellow()
    );
    println!(
        "{}",
        format!(
            "    apr convert {} --quantize q4k -o model-q4k.apr",
            path.display()
        )
        .yellow()
    );
    println!("{}", "    apr chat model-q4k.apr".yellow());
}

/// GH-224: Try to initialize SafeTensors CUDA model.
/// Returns (cuda_model, init_failed).
#[cfg(feature = "cuda")]
fn try_init_safetensors_cuda(
    model_path: &Path,
) -> (Option<realizar::safetensors_cuda::SafeTensorsCudaModel>, bool) {
    use realizar::safetensors_cuda::SafeTensorsCudaModel;
    match SafeTensorsCudaModel::load(model_path, 0) {
        Ok(cuda_model) => {
            println!(
                "{}",
                format!(
                    "[SafeTensors CUDA: {} ({} MB VRAM) — pre-cached]",
                    cuda_model.device_name(),
                    cuda_model.vram_mb()
                )
                .bright_green()
            );
            (Some(cuda_model), false)
        }
        Err(e) => {
            let err_msg = format!("{e}");
            if err_msg.contains("VRAM") {
                eprintln!(
                    "  {} {}",
                    "[BUG-214]".yellow(),
                    "SafeTensors F32 exceeds GPU VRAM. Will use CPU.".yellow()
                );
            } else {
                eprintln!("[SafeTensors CUDA init failed: {}, will use CPU]", e);
            }
            (None, true)
        }
    }
}
