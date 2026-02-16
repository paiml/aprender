impl ChatSession {
        pub(super) fn new(path: &Path) -> Result<Self, CliError> {
            println!("{}", "Loading model...".cyan());
            let start = Instant::now();

            // Read file bytes
            let mut file = File::open(path).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to open model file: {e}"))
            })?;
            let mut model_bytes = Vec::new();
            file.read_to_end(&mut model_bytes).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to read model file: {e}"))
            })?;

            // Detect format from magic bytes (Y14)
            let format = detect_format_from_bytes(&model_bytes);

            let elapsed = start.elapsed();
            let format_name = match format {
                ModelFormat::Apr => "APR",
                ModelFormat::Gguf => "GGUF",
                ModelFormat::SafeTensors => "SafeTensors",
                ModelFormat::Demo => "Demo",
            };
            println!(
                "{} {} format in {:.2}s ({:.1} MB)",
                "Loaded".green(),
                format_name,
                elapsed.as_secs_f32(),
                model_bytes.len() as f32 / 1_000_000.0
            );

            // Load tokenizer based on format
            let (llama_tokenizer, qwen_tokenizer) = match format {
                ModelFormat::Gguf => {
                    // Load LLaMA tokenizer from GGUF
                    let tok = match LlamaTokenizer::from_gguf_bytes(&model_bytes) {
                        Ok(tok) => {
                            println!(
                                "{} tokenizer with {} tokens",
                                "Loaded".green(),
                                tok.vocab_size()
                            );
                            Some(tok)
                        }
                        Err(e) => {
                            // F-MODEL-COMPLETE-001: Failed tokenizer is a fatal error
                            return Err(CliError::InvalidFormat(format!(
                                "Model is incomplete: Failed to load GGUF tokenizer: {}. \
                                This usually indicates a corrupted or improperly converted model.",
                                e
                            )));
                        }
                    };
                    (tok, None)
                }
                ModelFormat::SafeTensors | ModelFormat::Apr => {
                    // PMAT-109: Search multiple standard locations for Qwen tokenizer
                    // Priority: 1) model dir, 2) HuggingFace cache, 3) ~/.apr/tokenizers
                    let tok = find_qwen_tokenizer(path)?;
                    (None, tok)
                }
                ModelFormat::Demo => (None, None),
            };

            // PMAT-108: SafeTensors inference uses realizar's SafetensorsToAprConverter
            // No need to pre-load weights here - done lazily in generate_safetensors()
            // This avoids loading the model twice and removes aprender::models dependency
            if format == ModelFormat::SafeTensors {
                // Just print config info for user feedback (via realizar's SafetensorsConfig)
                if let Some(parent) = path.parent() {
                    let config_path = parent.join("config.json");
                    if config_path.exists() {
                        if let Ok(json) = std::fs::read_to_string(&config_path) {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                                let num_layers = v["num_hidden_layers"].as_u64().unwrap_or(0);
                                let hidden_size = v["hidden_size"].as_u64().unwrap_or(0);
                                let num_heads = v["num_attention_heads"].as_u64().unwrap_or(0);
                                println!(
                                    "{} config: {} layers, {} hidden, {} heads",
                                    "Loaded".green(),
                                    num_layers,
                                    hidden_size,
                                    num_heads
                                );
                            }
                        }
                    }
                }
            }

            // Detect chat template from model architecture (Toyota Way: Jidoka - auto-detect)
            // F-TEMPLATE-001: Use GGUF metadata for architecture detection, not filename
            // PMAT-120 FIX: SafeTensors/APR use config.json for architecture detection
            // Hash-based filenames (e.g., d4c4d9763127153c.gguf) don't contain model name
            let model_name = match format {
                ModelFormat::Gguf => {
                    // Parse GGUF metadata to get architecture (e.g., "qwen2", "llama")
                    use realizar::gguf::GGUFModel;
                    match GGUFModel::from_bytes(&model_bytes) {
                        Ok(gguf) => {
                            let arch = gguf.architecture().unwrap_or("unknown").to_string();
                            // Map GGUF architecture to template-detectable name
                            // Architecture names: qwen2, llama, phi3, mistral, etc.
                            arch
                        }
                        Err(_) => path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string(),
                    }
                }
                ModelFormat::Apr => {
                    // GH-222: Read architecture from APR v2 metadata (stored during import)
                    // Standalone APR files have no sibling config.json, so directory-name
                    // fallback produces garbage. APR metadata stores architecture directly.
                    let mut arch = String::from("unknown");
                    if let Ok(reader) = aprender::format::v2::AprV2Reader::from_bytes(&model_bytes)
                    {
                        if let Some(apr_arch) = &reader.metadata().architecture {
                            if !apr_arch.is_empty() {
                                arch.clone_from(apr_arch);
                            }
                        }
                    }
                    // Fallback: config.json in parent directory
                    if arch == "unknown" {
                        if let Some(parent) = path.parent() {
                            let config_path = parent.join("config.json");
                            if config_path.exists() {
                                if let Ok(json) = std::fs::read_to_string(&config_path) {
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json)
                                    {
                                        if let Some(model_type) = v["model_type"].as_str() {
                                            arch = model_type.to_lowercase();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Fallback: parent directory name
                    if arch == "unknown" {
                        arch = path
                            .parent()
                            .and_then(|p| p.file_name())
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string();
                    }
                    arch
                }
                ModelFormat::SafeTensors => {
                    // PMAT-120: Read config.json for architecture detection
                    // Five-Whys: "model.safetensors" filename doesn't indicate architecture
                    // Root cause: detect_format_from_name("model") returns Raw template
                    // Fix: Extract model_type or architectures from config.json
                    let mut arch = String::from("unknown");
                    if let Some(parent) = path.parent() {
                        let config_path = parent.join("config.json");
                        if config_path.exists() {
                            if let Ok(json) = std::fs::read_to_string(&config_path) {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                                    // Try model_type first (e.g., "qwen2")
                                    if let Some(model_type) = v["model_type"].as_str() {
                                        arch = model_type.to_lowercase();
                                    }
                                    // Fallback to architectures array (e.g., ["Qwen2ForCausalLM"])
                                    else if let Some(archs) = v["architectures"].as_array() {
                                        if let Some(first) = archs.first().and_then(|a| a.as_str())
                                        {
                                            // Extract base name: "Qwen2ForCausalLM" -> "qwen2"
                                            arch = first
                                                .trim_end_matches("ForCausalLM")
                                                .trim_end_matches("LMHeadModel")
                                                .to_lowercase();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Fallback to parent directory name (often contains model name)
                    if arch == "unknown" {
                        arch = path
                            .parent()
                            .and_then(|p| p.file_name())
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string();
                    }
                    arch
                }
                ModelFormat::Demo => "demo".to_string(),
            };
            let template_format = detect_format_from_name(&model_name);
            let chat_template = auto_detect_template(&model_name);

            let template_name = match template_format {
                TemplateFormat::ChatML => "ChatML",
                TemplateFormat::Llama2 => "LLaMA2",
                TemplateFormat::Mistral => "Mistral",
                TemplateFormat::Phi => "Phi",
                TemplateFormat::Alpaca => "Alpaca",
                TemplateFormat::Custom => "Custom",
                TemplateFormat::Raw => "Raw",
            };
            println!(
                "{} {} chat template",
                "Detected".green(),
                template_name.cyan()
            );

            // GH-224: Eagerly initialize GPU models during "Loading model..." phase
            // This moves the ~8s VRAM upload from first generate() to session init.
            let model_path_buf = path.to_path_buf();

            // GGUF: cache MappedGGUFModel (for tokenizer) + OwnedQuantizedModelCuda
            let mut cached_gguf_mapped = None;
            #[cfg(feature = "cuda")]
            let mut cached_gguf_cuda = None;
            #[cfg(feature = "cuda")]
            let mut cuda_init_failed = false;

            if format == ModelFormat::Gguf {
                match realizar::gguf::MappedGGUFModel::from_path(&model_path_buf) {
                    Ok(mapped) => {
                        #[cfg(feature = "cuda")]
                        {
                            use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
                            if OwnedQuantizedModelCuda::is_available() {
                                match OwnedQuantizedModel::from_mapped(&mapped) {
                                    Ok(owned) => match OwnedQuantizedModelCuda::new(owned, 0) {
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
                                            cached_gguf_cuda = Some(cuda_model);
                                        }
                                        Err(e) => {
                                            println!(
                                                "{}",
                                                format!(
                                                    "[GGUF CUDA init failed: {}, will use CPU]",
                                                    e
                                                )
                                                .yellow()
                                            );
                                            cuda_init_failed = true;
                                        }
                                    },
                                    Err(e) => {
                                        eprintln!("[GGUF model parse failed: {}, will use CPU]", e);
                                        cuda_init_failed = true;
                                    }
                                }
                            }
                        }
                        cached_gguf_mapped = Some(mapped);
                    }
                    Err(e) => {
                        eprintln!("[GGUF mmap failed: {}, will mmap per message]", e);
                    }
                }
            }

            // APR: cache AprV2ModelCuda
            #[cfg(feature = "cuda")]
            let mut cached_apr_cuda = None;
            #[cfg(feature = "cuda")]
            if format == ModelFormat::Apr {
                use realizar::apr::{AprV2Model, AprV2ModelCuda};
                if AprV2ModelCuda::is_available() {
                    match AprV2Model::from_bytes(model_bytes.clone()) {
                        Ok(apr_model) => match AprV2ModelCuda::new(apr_model, 0) {
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
                                // GH-262: Warn about F32 performance when model VRAM > 2GB
                                // F32 models use ~4x more memory bandwidth than Q4K, resulting
                                // in ~4x slower inference. A 1.5B F32 model uses ~5.9 GB vs ~0.9 GB Q4K.
                                if vram_mb > 2048 {
                                    println!(
                                        "{}",
                                        format!(
                                            "  Performance tip: This APR model uses {} MB VRAM (F32 tensors).",
                                            vram_mb
                                        ).yellow()
                                    );
                                    println!(
                                        "{}",
                                        "  For ~4x faster inference, convert to GGUF Q4K:".yellow()
                                    );
                                    println!(
                                        "{}",
                                        format!(
                                            "    apr export {} --format gguf -o model.gguf",
                                            path.display()
                                        ).yellow()
                                    );
                                    println!(
                                        "{}",
                                        "    apr chat model.gguf".yellow()
                                    );
                                }
                                cached_apr_cuda = Some(cuda_model);
                            }
                            Err(e) => {
                                eprintln!("[APR CUDA init failed: {}, will use CPU]", e);
                                cuda_init_failed = true;
                            }
                        },
                        Err(e) => {
                            eprintln!("[APR model parse failed: {}, will use CPU]", e);
                            cuda_init_failed = true;
                        }
                    }
                }
            }

            // SafeTensors: cache SafeTensorsCudaModel
            #[cfg(feature = "cuda")]
            let mut cached_safetensors_cuda = None;
            #[cfg(feature = "cuda")]
            if format == ModelFormat::SafeTensors {
                use realizar::safetensors_cuda::SafeTensorsCudaModel;
                match SafeTensorsCudaModel::load(&model_path_buf, 0) {
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
                        cached_safetensors_cuda = Some(cuda_model);
                    }
                    Err(e) => {
                        let err_msg = format!("{e}");
                        if err_msg.contains("VRAM") {
                            eprintln!(
                                "  {} {}",
                                "[BUG-214]".yellow(),
                                "SafeTensors F32 exceeds GPU VRAM. Will use CPU.".yellow()
                            );
                            cuda_init_failed = true;
                        } else {
                            eprintln!("[SafeTensors CUDA init failed: {}, will use CPU]", e);
                            cuda_init_failed = true;
                        }
                    }
                }
            }

            Ok(Self {
                model_bytes,
                model_path: model_path_buf,
                format,
                history: Vec::new(),
                chat_template,
                template_format,
                llama_tokenizer,
                qwen_tokenizer,
                cached_gguf_mapped,
                #[cfg(feature = "cuda")]
                cached_gguf_cuda,
                #[cfg(feature = "cuda")]
                cached_apr_cuda,
                #[cfg(feature = "cuda")]
                cached_safetensors_cuda,
                #[cfg(feature = "cuda")]
                cuda_init_failed,
            })
        }
}
