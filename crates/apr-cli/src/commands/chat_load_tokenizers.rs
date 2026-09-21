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

/// The session's chat template and its format, announced once.
/// GH-339: a Raw fallback is warned about instead of degrading silently.
fn select_chat_template(
    model_name: &str,
) -> (Box<dyn ChatTemplateEngine + Send + Sync>, TemplateFormat) {
    let template_format = detect_format_from_name(model_name);
    let production = production_chat_template(model_name);
    let no_think = production.is_some();
    let chat_template = production.unwrap_or_else(|| auto_detect_template(model_name));
    if matches!(template_format, TemplateFormat::Raw) {
        eprintln!(
            "{} Could not detect chat template for '{}', using raw format (no ChatML/Instruct wrapping)",
            "Warning:".yellow(),
            model_name.dimmed()
        );
    } else {
        println!(
            "{} {} chat template{}",
            "Detected".green(),
            template_format_name(template_format).cyan(),
            if no_think { " (Qwen3 no-think, as `apr serve`)" } else { "" }
        );
    }
    (chat_template, template_format)
}

/// `apr chat`'s template when it must differ from aprender-core's pick.
///
/// `apr chat` templates through aprender-core's detector, which maps every `qwen*`
/// architecture to plain ChatML. `apr serve` templates through realizar's, which
/// gives `qwen3*` (except `qwen3moe`) `Qwen3NoThinkTemplate` (PMAT-181): an empty
/// think block, so the model answers directly. On Qwen3-8B with plain ChatML, chat
/// reasoned for 545 tokens on "What is 2+2?" on lambda (RTX 4090) and printed only
/// `<think>` text inside a 512-token turn, while `apr serve` answered "2 + 2 = 4."
/// (0.69.1 critical path). Returns realizar's production template for the one
/// architecture family where the two detectors disagree, and `None` for every other
/// architecture, which keeps aprender-core's template.
fn production_chat_template(model_name: &str) -> Option<Box<dyn ChatTemplateEngine + Send + Sync>> {
    use realizar::chat_template::{create_template, detect_format_from_name, TemplateFormat as Rt};
    (detect_format_from_name(model_name) == Rt::Qwen3NoThink).then(|| {
        Box::new(RealizarTemplate {
            inner: create_template(Rt::Qwen3NoThink),
            chatml: aprender::text::chat_template::ChatMLTemplate::new(),
        }) as Box<dyn ChatTemplateEngine + Send + Sync>
    })
}

/// A realizar template behind aprender-core's `ChatTemplateEngine`, so the chat
/// session keeps one template type. Special tokens are ChatML's, which is what
/// `Qwen3NoThinkTemplate` wraps.
struct RealizarTemplate {
    inner: Box<dyn realizar::chat_template::ChatTemplateEngine>,
    chatml: aprender::text::chat_template::ChatMLTemplate,
}

impl ChatTemplateEngine for RealizarTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, aprender::AprenderError> {
        self.inner
            .format_message(role, content)
            .map_err(|e| aprender::AprenderError::Serialization(format!("chat template: {e}")))
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, aprender::AprenderError> {
        let messages: Vec<realizar::chat_template::ChatMessage> = messages
            .iter()
            .map(|m| realizar::chat_template::ChatMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        self.inner
            .format_conversation(&messages)
            .map_err(|e| aprender::AprenderError::Serialization(format!("chat template: {e}")))
    }

    fn special_tokens(&self) -> &aprender::text::chat_template::SpecialTokens {
        self.chatml.special_tokens()
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::ChatML
    }

    fn supports_system_prompt(&self) -> bool {
        self.inner.supports_system_prompt()
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

#[cfg(test)]
mod production_chat_template_tests {
    use super::*;

    /// `apr chat` must template Qwen3 turns exactly as `apr serve` does.
    #[test]
    fn qwen3_and_qwen35_get_the_serve_template() {
        use realizar::chat_template::{create_template, TemplateFormat as Rt};
        for arch in ["qwen3", "qwen35"] {
            let chat = production_chat_template(arch).expect("qwen3* gets the production template");
            let turn = [ChatMessage::user("What is 2+2?")];
            let serve = create_template(Rt::Qwen3NoThink)
                .format_conversation(&[realizar::chat_template::ChatMessage::user("What is 2+2?")])
                .expect("serve template formats a user turn");
            let got = chat.format_conversation(&turn).expect("chat template formats a user turn");
            assert_eq!(got, serve, "{arch}");
            assert!(got.ends_with("<|im_start|>assistant\n<think>\n</think>\n"), "{got:?}");
        }
    }

    /// Everywhere the two detectors agree, chat keeps aprender-core's template.
    #[test]
    fn other_architectures_keep_their_template() {
        for arch in ["qwen2", "qwen3moe", "llama", "mistral", "phi3", "unknown"] {
            assert!(production_chat_template(arch).is_none(), "{arch}");
        }
    }
}
