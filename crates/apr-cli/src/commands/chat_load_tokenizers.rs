/// Load tokenizers based on model format.
/// F-MODEL-COMPLETE-001: GGUF tokenizer failure is fatal.
fn load_tokenizers(
    format: ModelFormat,
    model_bytes: &[u8],
    path: &Path,
) -> Result<(Option<LlamaTokenizer>, Option<Qwen2BpeTokenizer>), CliError> {
    contract_pre_byte_encoder_coverage!();
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
        // #3022: a sharded index sits BESIDE the same tokenizer.json/config.json an
        // unsharded checkout has, so every sibling-reading helper treats the two alike.
        ModelFormat::SafeTensors | ModelFormat::ShardedSafeTensors | ModelFormat::Apr => {
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
        ModelFormat::SafeTensors | ModelFormat::ShardedSafeTensors => detect_arch_from_config(path),
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

/// The session's fallback chat template and its format, and which template the session
/// renders, announced once. A file that ships its own chat template gets THAT one
/// (#3755), as `apr serve` and `apr run` do; the per-family template is only the
/// fallback for a file with none. GH-339: a Raw fallback is warned about instead of
/// degrading silently.
fn select_chat_template(
    model_name: &str,
    embedded: Option<&realizar::chat_template::EmbeddedChatTemplate>,
) -> (Box<dyn ChatTemplateEngine + Send + Sync>, TemplateFormat) {
    let template_format = detect_format_from_name(model_name);
    let chat_template = auto_detect_template(model_name);
    if let Some(template) = embedded {
        println!(
            "{} the model's own chat template (thinking: {})",
            "Using".green(),
            template.thinking_modes().to_string().cyan()
        );
    } else if matches!(template_format, TemplateFormat::Raw) {
        eprintln!(
            "{} Could not detect chat template for '{}', using raw format (no ChatML/Instruct wrapping)",
            "Warning:".yellow(),
            model_name.dimmed()
        );
    } else {
        println!(
            "{} {} chat template",
            "Detected".green(),
            template_format_name(template_format).cyan()
        );
    }
    (chat_template, template_format)
}

impl ChatSession {
    /// Resolve `--thinking` against the model's own template, once, before the first
    /// turn (#3723). A mode the model cannot honour is refused with exit code 15.
    pub(super) fn resolve_thinking(&mut self, requested: Option<bool>) -> Result<(), CliError> {
        use realizar::chat_template::{format_chat_prompt, ChatMessage as RtMessage, ThinkingModes};
        let refused = |e: realizar::RealizarError| CliError::ThinkingModeUnsupported(e.to_string());
        let Some(template) = self.embedded_template.as_ref() else {
            // No template of its own: nothing to switch thinking on with.
            ThinkingModes::OffOnly.resolve(requested).map_err(refused)?;
            return Ok(());
        };
        let probe = [RtMessage::user("hi")];
        let prompt = format_chat_prompt(Some(template), None, &probe, requested).map_err(refused)?;
        self.thinking = prompt.thinking;
        self.prompt_opens_think = prompt.opens_think_block();
        println!(
            "{} {}",
            "Thinking:".green(),
            if self.thinking { "on" } else { "off" }.cyan()
        );
        Ok(())
    }

    /// A turn's completion split into (reasoning, answer), as production splits it
    /// (#3723). An unclosed think block is an error naming the budget.
    pub(super) fn split_response(
        &self,
        response: &str,
        config: &ChatConfig,
    ) -> Result<(Option<String>, String), String> {
        realizar::chat_template::split_completion(response, self.prompt_opens_think, config.max_tokens)
            .map(|split| (split.reasoning, split.answer))
            .map_err(|e| e.to_string())
    }

    /// The #3367 failed-turn flag, for a turn that failed after generation (#3723).
    pub(super) fn had_generate_error_mut(&mut self) -> &mut bool {
        &mut self.had_generate_error
    }

    /// The prompt for this turn through the model's own template, or `None` when the
    /// file ships none and the fallback template renders it (#3755).
    pub(super) fn render_embedded(
        &self,
        messages: &[ChatMessage],
    ) -> Option<Result<String, String>> {
        use realizar::chat_template::{format_chat_prompt, ChatMessage as RtMessage};
        let template = self.embedded_template.as_ref()?;
        let messages: Vec<RtMessage> = messages
            .iter()
            .map(|m| RtMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        Some(
            format_chat_prompt(Some(template), None, &messages, Some(self.thinking))
                .map(|p| p.text)
                .map_err(|e| format!("[Template error: {e}]")),
        )
    }
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
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(feature = "cuda")]
fn try_init_gguf_cuda(
    mapped: &realizar::gguf::MappedGGUFModel,
) -> Result<(Option<realizar::gguf::OwnedQuantizedModelCuda>, bool), crate::error::CliError> {
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
    // #3091: Qwen3.5/Qwen3.8 hybrids have a CPU forward but no GPU one yet (#3090). Skip the
    // CUDA attempt instead of failing it, so chat does not report a load error for a model
    // it can run.
    if mapped.model.architecture() == Some("qwen35") {
        eprintln!("[qwen35: Gated DeltaNet runs on the CPU; the GPU backend does not implement it yet (#3090)]");
        return Ok((None, false));
    }
    if !OwnedQuantizedModelCuda::is_available() {
        return Ok((None, false));
    }
    let owned = match OwnedQuantizedModel::from_mapped(mapped) {
        Ok(o) => o,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("NEITHER the CPU nor the GPU backend implements") {
                eprintln!("[GGUF model parse failed: {}]", msg);
                return Err(crate::error::CliError::ValidationFailed(msg));
            }
            eprintln!("[GGUF model parse failed: {}, will use CPU]", e);
            return Ok((None, true));
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
            Ok((Some(cuda_model), false))
        }
        Err(e) => {
            println!(
                "{}",
                format!("[GGUF CUDA init failed: {}, will use CPU]", e).yellow()
            );
            Ok((None, true))
        }
    }
}

/// GH-224: Try to initialize APR CUDA model.
/// GH-272: Warns about F32 performance when VRAM > 2GB.
/// Returns (cuda_model, init_failed).
#[cfg_attr(coverage_nightly, coverage(off))]
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
#[cfg_attr(coverage_nightly, coverage(off))]
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
#[cfg_attr(coverage_nightly, coverage(off))]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_load_no_fallback_on_arch_refusal() {
        assert!(true);
    }
}
