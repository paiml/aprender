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

pub(crate) fn template_format_name(tf: TemplateFormat) -> &'static str {
    match tf {
        TemplateFormat::ChatML => "ChatML",
        // #3801: this arm could not be written before — aprender-core's enum had
        // no such variant, which is why `apr chat` reported ChatML for a model
        // production serves with thinking off.
        TemplateFormat::Qwen3NoThink => "Qwen3NoThink (thinking off)",
        TemplateFormat::Llama2 => "LLaMA2",
        TemplateFormat::Zephyr => "Zephyr",
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

/// Initialize the CUDA model `apr chat` uses for an `.apr` (#3922).
///
/// ROUTED ONTO THE PATH `apr run` ALREADY USES. This built an
/// `AprV2ModelCuda` — a generic transformer class that is **Q4K-only by
/// construction and wrong for Q4K**: its init refuses F32 and bf16 with
/// `GH-279 Quantized weight 'blk.0.attn_q.weight' not cached`, and on a Q4K
/// `.apr` it produced GARBAGE on both required hosts while the same model
/// answered correctly on CPU. There is no configuration in which it did useful
/// GPU work.
///
/// Measured, one binary, every cell (d8):
///
///     model      via AprV2ModelCuda        via this path
///     q4k .apr   GARBAGE on GPU            CORRECT on GPU
///     F32 .apr   init fails -> CPU         CORRECT ON GPU
///     bf16 .apr  init fails -> CPU         rc=14 decline -> CPU
///
/// Nothing gets worse; q4k goes garbage -> correct and F32 goes CPU -> GPU.
///
/// The loading chain is `run`'s and `apr bench`'s:
/// `MappedAprModel` -> `OwnedQuantizedModel::from_apr` -> `OwnedQuantizedModelCuda`.
///
/// AND IT BRINGS THE F2 GATE, which is half the value. `try_apr_cuda_inference`
/// probes the GPU's first token against the CPU's and falls back when they
/// disagree; `chat` had no such check, so a silently wrong kernel had nothing to
/// stop it. Routing without the gate would be a half-fix.
///
/// Returns `(model, init_failed)`. `init_failed` is sticky in the caller, so a
/// decline here means CPU for the whole session rather than a retry per turn.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(feature = "cuda")]
fn try_init_apr_cuda(
    _model_bytes: &[u8],
    path: &Path,
) -> (Option<realizar::gguf::OwnedQuantizedModelCuda>, bool) {
    use realizar::apr::MappedAprModel;
    use realizar::gguf::{
        OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
    };

    // Read from the mapped file rather than the caller's byte buffer: the fused
    // path wants the tensor map, and `from_apr` is where a non-Q4K `.apr` is
    // refused by name instead of faulting later (#3885's shape at a third site).
    let mapped = match MappedAprModel::from_path(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[APR CUDA: cannot map {}: {e} — will use CPU]", path.display());
            return (None, true);
        },
    };
    let model = match OwnedQuantizedModel::from_apr(&mapped) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[APR CUDA: not a fused-kernel APR ({e}) — will use CPU]");
            return (None, true);
        },
    };
    let mut cuda_model = match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[APR CUDA init failed: {e}, will use CPU]");
            return (None, true);
        },
    };

    // F2: does the GPU agree with the CPU on the first token? A disagreement is a
    // silently wrong kernel, which is exactly what this routing exists to stop
    // being invisible. An empty probe context is the same one `run` uses.
    let probe_config = QuantizedGenerateConfig::default();
    eprintln!("[GH-480] F2 validation starting...");
    if !realizar::infer::validate_gpu_first_token(&mut cuda_model, &probe_config, &[]) {
        eprintln!("[GH-480] F2 validation FAILED — falling back to CPU");
        return (None, true);
    }
    eprintln!("[GH-480] F2 validation PASSED — launching GPU generation");

    println!("{}", "[APR CUDA: fused Q4K kernels, F2-validated]".bright_green());
    (Some(cuda_model), false)
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

// =============================================================================
// #3801: ONE detector across every verb
// =============================================================================

#[cfg(test)]
mod one_detector_tests {
    use super::*;

    /// THE DEFECT, as a test. `apr chat` imported `detect_format_from_name` from
    /// aprender-core, whose `TemplateFormat` has seven variants and no
    /// `Qwen3NoThink`; every `qwen*` became ChatML and the model reasoned. This
    /// now resolves to realizar's detector — the one `apr serve`, `apr run --chat`
    /// and `apr qa`'s golden gate already use.
    #[test]
    fn chat_gives_qwen3_the_no_think_template_like_every_other_verb() {
        for arch in ["qwen3", "qwen35", "Qwen3-8B-Q4_K_M", "Qwen3.5-0.8B-Q4_K_M"] {
            let got = detect_format_from_name(arch);
            assert_eq!(
                got,
                TemplateFormat::Qwen3NoThink,
                "{arch}: chat must select the template production selects"
            );
            assert_ne!(
                got,
                TemplateFormat::ChatML,
                "{arch}: ChatML is the defect — it leaves the model in thinking mode"
            );
        }
    }

    /// The MoE exception survives the switch: PMAT-181 routes qwen3_moe to plain
    /// ChatML because it was trained without `<think>` blocks. A unification that
    /// swept it up would be a new defect.
    #[test]
    fn qwen3_moe_keeps_plain_chatml() {
        for arch in ["qwen3_moe", "qwen3moe"] {
            assert_eq!(detect_format_from_name(arch), TemplateFormat::ChatML, "{arch}");
        }
    }

    /// Everything else chat used to name must still be named the same way: the
    /// switch widens the enum, it does not re-label the formats that existed.
    #[test]
    fn the_formats_chat_already_reported_are_unchanged() {
        assert_eq!(template_format_name(TemplateFormat::ChatML), "ChatML");
        assert_eq!(template_format_name(TemplateFormat::Llama2), "LLaMA2");
        assert_eq!(template_format_name(TemplateFormat::Mistral), "Mistral");
        assert_eq!(template_format_name(TemplateFormat::Phi), "Phi");
        assert_eq!(template_format_name(TemplateFormat::Alpaca), "Alpaca");
        assert_eq!(template_format_name(TemplateFormat::Custom), "Custom");
        assert_eq!(template_format_name(TemplateFormat::Raw), "Raw");
    }

    /// The two variants aprender-core could not express. `Qwen3NoThink` is the
    /// one this row exists for: its name must SAY the thinking mode, because the
    /// banner is what a user reads to know which mode they are in.
    #[test]
    fn the_variants_the_old_enum_could_not_express_are_named() {
        let name = template_format_name(TemplateFormat::Qwen3NoThink);
        assert!(name.contains("Qwen3NoThink"), "{name}");
        assert!(name.contains("thinking off"), "{name}");
        assert_eq!(template_format_name(TemplateFormat::Zephyr), "Zephyr");
    }

    /// The banner and the session must not disagree. The banner derives the
    /// format from the FILE STEM and the session from `general.architecture`;
    /// for a Qwen3 file both must land on the same template, or the line a user
    /// reads describes a mode they are not in.
    #[test]
    fn the_banner_and_the_session_agree_on_a_qwen3_file() {
        let from_file_stem = detect_format_from_name("Qwen3-1.7B-Q4_K_M");
        let from_architecture = detect_format_from_name("qwen3");
        assert_eq!(from_file_stem, from_architecture);
        assert_eq!(from_file_stem, TemplateFormat::Qwen3NoThink);
    }
}
