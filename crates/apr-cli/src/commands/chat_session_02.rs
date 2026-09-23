/// The "Loaded <FORMAT> format in Ns (N MB)" banner.
///
/// Extracted from `ChatSession::new` to bring its cognitive complexity back under the
/// ratchet's ceiling of 25 (#3844): a five-arm match and a branch, both purely about
/// reporting, inside a function whose job is construction.
fn report_loaded_format(format: ModelFormat, model_bytes: &[u8], elapsed: std::time::Duration) {
    let format_name = match format {
        ModelFormat::Apr => "APR",
        ModelFormat::Gguf => "GGUF",
        ModelFormat::SafeTensors => "SafeTensors",
        ModelFormat::ShardedSafeTensors => "Sharded SafeTensors",
        ModelFormat::Demo => "Demo",
    };
    // For a sharded index the bytes just read are the ~20 KB manifest, not the
    // model: printing their length would report "0.0 MB" for a 7B model — the
    // same lie under a different name. The manifest states the real total.
    let reported_bytes = if format == ModelFormat::ShardedSafeTensors {
        sharded_total_size(model_bytes).unwrap_or(0)
    } else {
        model_bytes.len() as u64
    };
    println!(
        "{} {} format in {:.2}s ({:.1} MB)",
        "Loaded".green(),
        format_name,
        elapsed.as_secs_f32(),
        reported_bytes as f32 / 1_000_000.0
    );
}

/// GH-339: warn on a Raw fallback instead of degrading silently.
fn report_template_detection(template_format: TemplateFormat, model_name: &str) {
    if matches!(template_format, TemplateFormat::Raw) {
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
}

/// What a GGUF preload decided, so `ChatSession::new` can read it instead of nesting it.
struct GgufPreload {
    qwen35: Option<realizar::gguf::qwen35_session::Qwen35Session>,
    #[cfg(feature = "cuda")]
    cuda: Option<realizar::gguf::OwnedQuantizedModelCuda>,
    #[cfg(feature = "cuda")]
    cuda_failed: bool,
}

/// #3791 unit (2): the hybrid routes to its resident session BEFORE any dense CUDA
/// preload. #3794's `cuda_preload_allowed` gate is KEPT on the dense branch — taking
/// #3791's side whole would drop it and `apr chat --no-gpu` would upload the weights
/// again (measured 4070 MiB).
///
/// Extracted from `ChatSession::new` (#3844): this arm was six levels deep
/// (`if` → `match` → arm → `if let` → `else` → `cfg` `if` → `if`), which is what put
/// `new` at cognitive 35 against the ratchet's ceiling of 25. The branching is
/// unchanged; only its home is.
fn preload_gguf(
    mapped: &realizar::gguf::MappedGGUFModel,
    force_cpu: bool,
    format: ModelFormat,
) -> Result<GgufPreload, CliError> {
    if let Some(session) = try_init_qwen35_session(mapped, force_cpu)? {
        return Ok(GgufPreload {
            qwen35: Some(session),
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(feature = "cuda")]
            cuda_failed: false,
        });
    }
    // #3987: qwen3moe has no DENSE FFN (only per-expert tensors), so the dense
    // `OwnedQuantizedModelCuda` preloaded below dereferences a null `ffn_gate` on its first
    // forward -- measured on gx10: "ffn_gate_ptr is null (0)", then rc 8. It is served per
    // turn by the ONE dispatch `apr run` uses (see `generate_gguf_with_prompt`), so no dense
    // model is preloaded for it.
    if is_qwen3_moe_gguf(mapped) {
        return Ok(GgufPreload {
            qwen35: None,
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(feature = "cuda")]
            cuda_failed: false,
        });
    }
    #[cfg(feature = "cuda")]
    if super::cuda_preload_allowed(force_cpu, format) {
        let (cuda, cuda_failed) = try_init_gguf_cuda(mapped)?;
        return Ok(GgufPreload {
            qwen35: None,
            cuda,
            cuda_failed,
        });
    }
    let _ = format;
    Ok(GgufPreload {
        qwen35: None,
        #[cfg(feature = "cuda")]
        cuda: None,
        #[cfg(feature = "cuda")]
        cuda_failed: false,
    })
}

/// #3987: a plain Qwen3 MoE GGUF, served through `run_qwen3_moe_generate_dispatch`.
///
/// ONE predicate for both the preload and the turn generator, so they cannot disagree.
/// Qwen3.5-MoE spellings are excluded: the normaliser folds them into `qwen3_moe`, but they
/// carry SSM layers the qwen3moe forward does not run, and the capability refusal names them.
fn is_qwen3_moe_gguf(mapped: &realizar::gguf::MappedGGUFModel) -> bool {
    let arch = mapped.model.architecture().unwrap_or_default();
    realizar::gguf::moe_forward_handles(arch)
        && realizar::capability::no_cuda_forward_reason(arch).is_none()
}

impl ChatSession {
        /// #3794: `force_cpu` gates CUDA INITIALISATION, not just generation.
        ///
        /// Generation already honoured it (`chat_generate_session_02.rs`), so the
        /// answer came from the CPU — but the session had already built an
        /// `OwnedQuantizedModelCuda`, uploaded the weights and printed
        /// `[GGUF CUDA: … — pre-cached]`, holding VRAM for its whole lifetime. On
        /// the shared GPU hosts that is memory taken OUTSIDE `/tmp/apr-gpu.lock`
        /// by a run that asked for none, so it is invisible to `gpu-q` and can
        /// starve the very serialization rationing the card. A `--no-gpu` flag
        /// that takes the GPU also fails its own claim on its face.
        pub(super) fn new(path: &Path, force_cpu: bool) -> Result<Self, CliError> {
            contract_pre_session_persistence!();
            println!("{}", "Loading model...".cyan());
            let start = Instant::now();

            // #3022: the format is decided ONCE, by name-then-magic-bytes, and the same
            // answer drives the banner, the tokenizer and the generator. Reading the file
            // and classifying its first eight bytes used to be the session's own
            // independent decision, so the banner could say "SafeTensors" while the
            // session loaded `Demo` — which is precisely how a 7B model reported a
            // successful empty answer. `resolve_chat_format` refuses instead of
            // substituting, so `Demo` is not reachable from a path the user named.
            let format = resolve_chat_format(path)?;

            // Read file bytes
            let mut file = File::open(path).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to open model file: {e}"))
            })?;
            let mut model_bytes = Vec::new();
            file.read_to_end(&mut model_bytes).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to read model file: {e}"))
            })?;

            report_loaded_format(format, &model_bytes, start.elapsed());

            let (llama_tokenizer, qwen_tokenizer) = load_tokenizers(format, &model_bytes, path)?;

            if matches!(
                format,
                ModelFormat::SafeTensors | ModelFormat::ShardedSafeTensors
            ) {
                print_safetensors_config(path);
            }

            // Detect chat template from model architecture
            // GH-339: Warn on Raw fallback instead of silent degradation
            let model_name = detect_model_architecture(format, &model_bytes, path);
            let template_format = detect_format_from_name(&model_name);
            let chat_template = auto_detect_template(&model_name);

            report_template_detection(template_format, &model_name);

            // GH-224: Eagerly initialize GPU models during "Loading model..." phase
            let model_path_buf = path.to_path_buf();

            let mut cached_gguf_mapped = None;
            let mut qwen35_session = None;
            #[cfg(feature = "cuda")]
            let mut cached_gguf_cuda = None;
            #[cfg(feature = "cuda")]
            let mut cuda_init_failed = false;

            if format == ModelFormat::Gguf {
                match realizar::gguf::MappedGGUFModel::from_path(&model_path_buf) {
                    Ok(mapped) => {
                        let pre = preload_gguf(&mapped, force_cpu, format)?;
                        qwen35_session = pre.qwen35;
                        #[cfg(feature = "cuda")]
                        {
                            cached_gguf_cuda = pre.cuda;
                            if pre.cuda_failed {
                                cuda_init_failed = true;
                            }
                        }
                        cached_gguf_mapped = Some(mapped);
                    }
                    Err(e) => {
                        eprintln!("[GGUF mmap failed: {}, will mmap per message]", e);
                    }
                }
            }

            #[cfg(feature = "cuda")]
            let mut cached_apr_cuda = None;
            #[cfg(feature = "cuda")]
            if super::cuda_preload_allowed(force_cpu, format) && format == ModelFormat::Apr {
                let (cuda, failed) = try_init_apr_cuda(&model_bytes, path);
                cached_apr_cuda = cuda;
                if failed { cuda_init_failed = true; }
            }

            #[cfg(feature = "cuda")]
            let mut cached_safetensors_cuda = None;
            #[cfg(feature = "cuda")]
            if super::cuda_preload_allowed(force_cpu, format) && format == ModelFormat::SafeTensors {
                let (cuda, failed) = try_init_safetensors_cuda(&model_path_buf);
                cached_safetensors_cuda = cuda;
                if failed { cuda_init_failed = true; }
            }

            let session = Self {
                model_bytes,
                model_path: model_path_buf,
                format,
                history: Vec::new(),
                chat_template,
                template_format,
                llama_tokenizer,
                qwen_tokenizer,
                cached_gguf_mapped,
                qwen35_session,
                #[cfg(feature = "cuda")]
                cached_gguf_cuda,
                #[cfg(feature = "cuda")]
                cached_apr_cuda,
                #[cfg(feature = "cuda")]
                cached_safetensors_cuda,
                #[cfg(feature = "cuda")]
                cuda_init_failed,
                had_generate_error: false,
                generated_on_gpu: false,
            };
            contract_post_session_persistence!(&());
            Ok(session)
        }
}
