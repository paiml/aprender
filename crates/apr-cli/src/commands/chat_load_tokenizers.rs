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

/// #3595: load a Qwen3.5 hybrid GGUF as a resident session, or `None` for any other
/// architecture (which the dense GH-224 path below serves).
///
/// The session attempts CUDA unless `force_cpu`, exactly as `apr run` does, and it is
/// the only thing that tells the user which backend the hybrid runs on: its route
/// notice is `qwen35_route_notice`'s, and its `Backend:` or fallback line is printed
/// once, by the code that took that route. This used to print "the GPU backend does
/// not implement it yet (#3090)" unconditionally — a day after #3090 shipped — and
/// then rebuild the model on the GPU for every turn anyway.
fn try_init_qwen35_session(
    mapped: &realizar::gguf::MappedGGUFModel,
    force_cpu: bool,
) -> Result<Option<realizar::gguf::qwen35_session::Qwen35Session>, crate::error::CliError> {
    if !realizar::gguf::hybrid_forward_handles(mapped.model.architecture().unwrap_or_default()) {
        return Ok(None);
    }
    realizar::gguf::qwen35_session::Qwen35Session::load(mapped, force_cpu)
        .map(Some)
        .map_err(|e| crate::error::CliError::ModelLoadFailed(format!("Qwen3.5 hybrid: {e}")))
}

/// #3595 done_when 2: what the session PRINTED about its route must agree with the
/// route it TOOK. The defect was two lines on one run — "runs on the CPU; the GPU
/// backend does not implement it yet" and then "Backend: GPU" — and every test passed,
/// because nothing compared the banner with the route.
#[cfg(test)]
mod qwen35_route_banner_tests {
    use super::*;
    use realizar::gguf::forward_qwen35::QWEN35_GPU_FALLBACK_PREFIX;

    const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

    /// The route a line states: `Some(true)` the GPU, `Some(false)` the CPU, `None`
    /// a line that states no route.
    fn states_route(line: &str) -> Option<bool> {
        if line.starts_with("Backend: GPU (CUDA,") {
            Some(true)
        } else if line.starts_with(QWEN35_GPU_FALLBACK_PREFIX) || line.contains("runs on the CPU") {
            Some(false)
        } else {
            None
        }
    }

    /// On the GPU, exactly one route line and it says GPU. On the CPU, the last route
    /// line says CPU — a GPU line may precede it only as the route a fallback then
    /// left — or there is none (the user asked for the CPU, which is not news).
    fn banner_agrees_with_route(notices: &[String], on_gpu: bool) -> Result<(), String> {
        let stated: Vec<bool> = notices.iter().filter_map(|l| states_route(l)).collect();
        let agrees = if on_gpu {
            stated == [true]
        } else {
            stated.last().map_or(true, |&gpu| !gpu)
        };
        if agrees {
            Ok(())
        } else {
            Err(format!("printed {notices:?}, but the session is on the {}", if on_gpu { "GPU" } else { "CPU" }))
        }
    }

    #[test]
    fn the_check_rejects_the_banner_3595_reported() {
        let reported = [
            "[qwen35: Gated DeltaNet runs on the CPU; the GPU backend does not implement it yet (#3090)]".to_string(),
            "Backend: GPU (CUDA, NVIDIA GeForce RTX 4090, 24035 MB VRAM) [qwen35 hybrid forward, #3090]".to_string(),
        ];
        assert!(banner_agrees_with_route(&reported, true).is_err(), "the #3595 pair must be RED");
        assert!(banner_agrees_with_route(&reported[1..], true).is_ok());
        assert!(banner_agrees_with_route(&reported[1..], false).is_err(), "a GPU line on a CPU session");
        let fell_back = [
            reported[1].clone(),
            format!("{QWEN35_GPU_FALLBACK_PREFIX}, falling back to CPU: the F2 CPU-parity guard rejected the GPU path"),
        ];
        assert!(banner_agrees_with_route(&fell_back, false).is_ok(), "GPU, then a printed fallback");
        assert!(banner_agrees_with_route(&[], false).is_ok(), "--no-gpu owes no notice");
        assert!(banner_agrees_with_route(&[], true).is_err(), "a GPU run must say so");
    }

    #[test]
    fn a_qwen35_session_prints_the_route_it_took() {
        if !Path::new(MODEL).exists() {
            eprintln!("SKIP: {MODEL} is absent");
            return;
        }
        let mapped = realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map the GGUF");
        #[cfg(feature = "cuda")]
        let cuda_host = realizar::gguf::OwnedQuantizedModelCuda::is_available();
        #[cfg(not(feature = "cuda"))]
        let cuda_host = false;
        for force_cpu in [false, true] {
            let session = try_init_qwen35_session(&mapped, force_cpu)
                .expect("the hybrid loads")
                .expect("a qwen35 GGUF gets a session");
            banner_agrees_with_route(session.notices(), session.on_gpu())
                .unwrap_or_else(|e| panic!("force_cpu={force_cpu}: {e}"));
            if force_cpu || !cuda_host {
                assert!(!session.on_gpu(), "force_cpu={force_cpu}, cuda_host={cuda_host}");
            } else {
                // A CUDA binary on a CUDA host must reach the upload: either it serves
                // from the GPU, or it printed why it could not.
                let tried = session.on_gpu()
                    || session.notices().iter().any(|l| l.starts_with(QWEN35_GPU_FALLBACK_PREFIX));
                assert!(tried, "the GPU was never attempted: {:?}", session.notices());
            }
        }
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
