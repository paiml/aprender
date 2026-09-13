impl ChatSession {
        pub(super) fn new(path: &Path) -> Result<Self, CliError> {
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

            let elapsed = start.elapsed();
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
                sharded_total_size(&model_bytes).unwrap_or(0)
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

            // GH-224: Eagerly initialize GPU models during "Loading model..." phase
            let model_path_buf = path.to_path_buf();

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
                            let (cuda, failed) = try_init_gguf_cuda(&mapped)?;
                            cached_gguf_cuda = cuda;
                            if failed { cuda_init_failed = true; }
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
            if format == ModelFormat::Apr {
                let (cuda, failed) = try_init_apr_cuda(&model_bytes, path);
                cached_apr_cuda = cuda;
                if failed { cuda_init_failed = true; }
            }

            #[cfg(feature = "cuda")]
            let mut cached_safetensors_cuda = None;
            #[cfg(feature = "cuda")]
            if format == ModelFormat::SafeTensors {
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
                #[cfg(feature = "cuda")]
                cached_gguf_cuda,
                #[cfg(feature = "cuda")]
                cached_apr_cuda,
                #[cfg(feature = "cuda")]
                cached_safetensors_cuda,
                #[cfg(feature = "cuda")]
                cuda_init_failed,
            };
            contract_post_session_persistence!(&());
            Ok(session)
        }
}
