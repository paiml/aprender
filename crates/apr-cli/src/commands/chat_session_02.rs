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

            let (llama_tokenizer, qwen_tokenizer) = load_tokenizers(format, &model_bytes, path)?;

            if format == ModelFormat::SafeTensors {
                print_safetensors_config(path);
            }

            // Detect chat template from model architecture
            let model_name = detect_model_architecture(format, &model_bytes, path);
            let template_format = detect_format_from_name(&model_name);
            let chat_template = auto_detect_template(&model_name);

            println!(
                "{} {} chat template",
                "Detected".green(),
                template_format_name(template_format).cyan()
            );

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
                            let (cuda, failed) = try_init_gguf_cuda(&mapped);
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
