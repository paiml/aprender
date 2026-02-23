impl ChatSession {

        pub(super) fn generate(&mut self, user_input: &str, config: &ChatConfig) -> String {
            let start = Instant::now();

            let formatted_prompt = match self.build_formatted_prompt(user_input, config) {
                Ok(prompt) => prompt,
                Err(e) => return e,
            };

            // For GGUF, use embedded tokenizer directly (correct special token IDs)
            if self.format == ModelFormat::Gguf {
                return self.generate_gguf_response(&formatted_prompt, config, start);
            }

            let prompt_tokens = self.tokenize_prompt(&formatted_prompt, config);

            let result = match self.format {
                ModelFormat::Apr => self.generate_apr(&prompt_tokens, config),
                ModelFormat::SafeTensors => self.generate_safetensors(&prompt_tokens, config),
                ModelFormat::Demo => Ok(vec![]),
                ModelFormat::Gguf => unreachable!(), // handled above
            };

            let gen_time = start.elapsed();

            match result {
                Ok(output_tokens) => {
                    let new_tokens = Self::strip_prompt_tokens(&output_tokens, &prompt_tokens);
                    self.print_token_stats(new_tokens, gen_time);
                    self.debug_inspect_tokens(new_tokens, config);
                    let raw_response = self.decode_tokens(new_tokens);
                    clean_chat_response(&raw_response)
                }
                Err(e) => format!("[Error: {}]", e),
            }
        }

        /// Build the formatted prompt from conversation history and user input.
        ///
        /// Returns Ok(prompt) or Err(error_string) for template failures.
        fn build_formatted_prompt(
            &self,
            user_input: &str,
            config: &ChatConfig,
        ) -> Result<String, String> {
            let mut messages: Vec<ChatMessage> = Vec::new();

            if let Some(ref system) = config.system {
                messages.push(ChatMessage::system(system));
            }
            messages.extend(self.history.iter().cloned());
            messages.push(ChatMessage::user(user_input));

            let formatted_prompt = self
                .chat_template
                .format_conversation(&messages)
                .map_err(|e| format!("[Template error: {}]", e))?;

            if config.trace {
                eprintln!(
                    "[APR-TRACE] Formatted prompt ({} chars):",
                    formatted_prompt.len()
                );
                eprintln!(
                    "[APR-TRACE] {:?}",
                    &formatted_prompt[..formatted_prompt.len().min(500)]
                );
            }

            Ok(formatted_prompt)
        }

        /// Handle the GGUF generation path, returning a cleaned response string.
        fn generate_gguf_response(
            &mut self,
            formatted_prompt: &str,
            config: &ChatConfig,
            start: Instant,
        ) -> String {
            match self.generate_gguf_with_prompt(formatted_prompt, config) {
                Ok(response) => {
                    let gen_time = start.elapsed();
                    let approx_tokens = response.split_whitespace().count().max(1) * 4 / 3;
                    let tps = approx_tokens as f32 / gen_time.as_secs_f32();
                    println!(
                        "{}",
                        format!("[{:.1}s, ~{:.0} tok/s]", gen_time.as_secs_f32(), tps).dimmed()
                    );
                    clean_chat_response(&response)
                }
                Err(e) => format!("[Error: {}]", e),
            }
        }

        /// Tokenize the formatted prompt using the available tokenizer.
        fn tokenize_prompt(&self, formatted_prompt: &str, config: &ChatConfig) -> Vec<u32> {
            let prompt_tokens: Vec<u32> = if let Some(ref tokenizer) = self.llama_tokenizer {
                tokenizer.encode_with_bos(formatted_prompt)
            } else if let Some(ref tokenizer) = self.qwen_tokenizer {
                tokenizer.encode(formatted_prompt)
            } else {
                formatted_prompt.chars().map(|c| c as u32).collect()
            };

            if config.trace {
                eprintln!(
                    "[APR-TRACE] Prompt tokens ({} tokens): {:?}",
                    prompt_tokens.len(),
                    &prompt_tokens[..prompt_tokens.len().min(50)]
                );
            }

            prompt_tokens
        }

        /// Strip prompt tokens from output, returning only newly generated tokens.
        fn strip_prompt_tokens<'a>(output_tokens: &'a [u32], prompt_tokens: &[u32]) -> &'a [u32] {
            if output_tokens.len() > prompt_tokens.len() {
                &output_tokens[prompt_tokens.len()..]
            } else {
                output_tokens
            }
        }

        /// Print token generation performance stats (GH-262).
        #[allow(clippy::unused_self)]
        fn print_token_stats(&self, new_tokens: &[u32], gen_time: std::time::Duration) {
            let tps = new_tokens.len() as f32 / gen_time.as_secs_f32();
            println!(
                "{}",
                format!(
                    "[{} tokens in {:.1}s = {:.1} tok/s]",
                    new_tokens.len(),
                    gen_time.as_secs_f32(),
                    tps,
                )
                .dimmed()
            );
        }

        /// Print debug token inspection output (only with --inspect flag).
        fn debug_inspect_tokens(&self, new_tokens: &[u32], config: &ChatConfig) {
            if config.inspect {
                if let Some(ref tok) = self.llama_tokenizer {
                    println!(
                        "[DEBUG: first 10 new tokens: {:?}]",
                        &new_tokens[..new_tokens.len().min(10)]
                    );
                    for &id in new_tokens.iter().take(10) {
                        println!("[DEBUG: {} -> {:?}]", id, tok.id_to_token(id));
                    }
                }
            }
        }

        /// Decode token IDs to a string using the available tokenizer.
        fn decode_tokens(&self, tokens: &[u32]) -> String {
            if let Some(ref tokenizer) = self.llama_tokenizer {
                tokenizer.decode(tokens)
            } else if let Some(ref tokenizer) = self.qwen_tokenizer {
                tokenizer.decode(tokens)
            } else {
                tokens
                    .iter()
                    .filter_map(|&t| char::from_u32(t))
                    .collect()
            }
        }

        /// Generate response for GGUF models using the embedded tokenizer
        ///
        /// GGUF models have their own tokenizer with correct special token IDs.
        /// Using LlamaTokenizer/Qwen2BpeTokenizer causes wrong token IDs for:
        /// - `<|im_start|>` (source of truth: `SpecialTokens::qwen2().im_start_id`)
        /// - `<|im_end|>` (source of truth: `SpecialTokens::qwen2().im_end_id`)
        /// - `<|endoftext|>` (source of truth: `SpecialTokens::qwen2().bos_id`)
        ///
        /// GH-224: Uses cached MappedGGUFModel and OwnedQuantizedModelCuda to avoid
        /// re-mmapping and re-uploading weights to VRAM on every message.
        fn generate_gguf_with_prompt(
            &mut self,
            prompt: &str,
            config: &ChatConfig,
        ) -> Result<String, String> {
            use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

            // GH-224: Use cached MappedGGUFModel for tokenizer, fall back to fresh mmap
            let fresh_mapped;
            let mapped = if let Some(ref m) = self.cached_gguf_mapped {
                m
            } else {
                fresh_mapped = MappedGGUFModel::from_path(&self.model_path)
                    .map_err(|e| format!("Failed to mmap GGUF: {e}"))?;
                &fresh_mapped
            };

            // Encode prompt using GGUF's embedded tokenizer (correct special token IDs)
            let prompt_tokens = mapped
                .model
                .encode(prompt)
                .ok_or_else(|| "Failed to encode prompt with GGUF tokenizer".to_string())?;
            let prompt_len = prompt_tokens.len();

            // APR-TRACE-001: Debug token IDs
            if config.trace {
                eprintln!(
                    "[APR-TRACE] Prompt tokens ({} tokens): {:?}",
                    prompt_len,
                    &prompt_tokens[..prompt_len.min(50)]
                );
                let decoded = mapped.model.decode(&prompt_tokens);
                eprintln!(
                    "[APR-TRACE] Decoded: {:?}",
                    &decoded[..decoded.len().min(200)]
                );
            }

            // C-06 (Meyer DbC): EOS from GGUF metadata, not hardcoded.
            let stop_tokens = mapped
                .model
                .eos_token_id()
                .map(|id| vec![id])
                .unwrap_or_default();

            let gen_config = QuantizedGenerateConfig {
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_k: 40,
                stop_tokens,
                trace: config.trace,
            };

            // GH-224: Try cached CUDA model first (no re-upload)
            #[cfg(feature = "cuda")]
            if !config.force_cpu && !self.cuda_init_failed {
                if let Some(ref mut cuda_model) = self.cached_gguf_cuda {
                    let output_tokens = cuda_model
                        .generate_gpu_resident(&prompt_tokens, &gen_config)
                        .map_err(|e| format!("CUDA generate failed: {e}"))?;

                    let new_tokens = if output_tokens.len() > prompt_len {
                        &output_tokens[prompt_len..]
                    } else {
                        &output_tokens[..]
                    };

                    if config.trace {
                        eprintln!(
                            "[APR-TRACE] Generated {} new tokens: {:?}",
                            new_tokens.len(),
                            &new_tokens[..new_tokens.len().min(50)]
                        );
                        for (i, &tok) in new_tokens.iter().take(20).enumerate() {
                            let decoded = mapped.model.decode(&[tok]);
                            eprintln!("[APR-TRACE] Token {}: {} -> {:?}", i, tok, decoded);
                        }
                    }

                    return Ok(mapped.model.decode(new_tokens));
                }
            }

            // CPU path — create fresh OwnedQuantizedModel from cached or fresh mapped
            let model = OwnedQuantizedModel::from_mapped(mapped)
                .map_err(|e| format!("Failed to create GGUF model: {e}"))?;

            // APR-TRACE-001: Use traced generation when --trace is enabled
            let output_tokens = if config.trace {
                use realizar::{InferenceTracer, ModelInfo, TraceConfig};

                let trace_config = TraceConfig {
                    enabled: true,
                    verbose: false,
                    output: config.trace_output.clone(),
                    ..Default::default()
                };

                let mut tracer = InferenceTracer::new(trace_config);
                tracer.set_model_info(ModelInfo {
                    name: "GGUF Model (CPU)".to_string(),
                    num_layers: model.config().num_layers,
                    hidden_dim: model.config().hidden_dim,
                    vocab_size: model.config().vocab_size,
                    num_heads: model.config().num_heads,
                    quant_type: None,
                });

                eprintln!("Warning: CPU traced generation not implemented, using non-traced path");

                let result = model
                    .generate_with_cache(&prompt_tokens, &gen_config)
                    .map_err(|e| format!("GGUF generate failed: {e}"))?;

                if let Err(e) = tracer.write_output() {
                    eprintln!("Warning: Failed to write trace output: {e}");
                }

                result
            } else {
                model
                    .generate_with_cache(&prompt_tokens, &gen_config)
                    .map_err(|e| format!("GGUF generate failed: {e}"))?
            };

            let new_tokens = if output_tokens.len() > prompt_len {
                &output_tokens[prompt_len..]
            } else {
                &output_tokens[..]
            };

            let decoded = mapped.model.decode(new_tokens);
            Ok(decoded)
        }

        fn generate_apr(
            &mut self,
            prompt: &[u32],
            config: &ChatConfig,
        ) -> Result<Vec<u32>, String> {
            // C-04 (Meyer DbC): EOS from model metadata — no hardcoded fallback.
            // If model has no EOS metadata, use 0 (no token matches → rely on max_tokens).
            let eos_token_id = self.extract_apr_eos_token().unwrap_or(0);

            // GH-224: Try cached CUDA model first (no re-upload per message)
            #[cfg(feature = "cuda")]
            if !config.force_cpu && !self.cuda_init_failed {
                if let Some(ref mut cuda_model) = self.cached_apr_cuda {
                    let max_tokens = config.max_tokens;
                    return cuda_model
                        .generate_cuda_with_cache(prompt, max_tokens, eos_token_id)
                        .map_err(|e| format!("APR CUDA generate failed: {e}"));
                }
            }

            // CPU path using AprTransformer (has temperature/top_p sampling + KV cache)
            use realizar::apr_transformer::{AprTransformer, GenerateConfig};

            let transformer = AprTransformer::from_apr_bytes(&self.model_bytes)
                .map_err(|e| format!("Failed to load APR transformer: {e}"))?;

            let gen_config = GenerateConfig {
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: 0,
                repetition_penalty: 1.0,
                trace: config.trace,
                stop_tokens: vec![],
            };

            transformer
                .generate_with_cache(prompt, &gen_config)
                .map_err(|e| format!("APR generate failed: {e}"))
        }

        /// PMAT-181: Extract EOS token ID from APR metadata (fixes GH-170)
        ///
        /// Five-Whys Root Cause: Different Qwen2 model sizes may have different EOS tokens
        /// in their metadata. The 1.5B model's EOS token was being mismatched with hardcoded
        /// 151645, causing generation to terminate immediately or hang.
        ///
        /// Toyota Way: Genchi Genbutsu - Go and see the actual model metadata
        fn extract_apr_eos_token(&self) -> Option<u32> {
            // Parse APR metadata from model bytes
            let reader = AprReader::from_bytes(self.model_bytes.clone()).ok()?;

            // Try common metadata keys for EOS token (in order of specificity)
            // 1. tokenizer.eos_token_id (GGUF-style)
            // 2. eos_token_id (direct)
            // 3. tokenizer_config.eos_token_id (nested)
            let keys = [
                "tokenizer.eos_token_id",
                "eos_token_id",
                "tokenizer.ggml.eos_token_id",
            ];

            for key in keys {
                if let Some(value) = reader.get_metadata(key) {
                    if let Some(id) = value.as_u64() {
                        return Some(id as u32);
                    }
                }
            }

            // Try nested tokenizer_config
            if let Some(config) = reader.get_metadata("tokenizer_config") {
                if let Some(obj) = config.as_object() {
                    if let Some(value) = obj.get("eos_token_id") {
                        if let Some(id) = value.as_u64() {
                            return Some(id as u32);
                        }
                    }
                }
            }

            None
        }

        #[allow(dead_code)]
        fn generate_gguf(&self, prompt: &[u32], config: &ChatConfig) -> Result<Vec<u32>, String> {
            use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

            // Use MappedGGUFModel -> OwnedQuantizedModel for proper attention
            // This has RoPE position encoding, causal mask, and GQA support
            let mapped = MappedGGUFModel::from_path(&self.model_path)
                .map_err(|e| format!("Failed to mmap GGUF: {e}"))?;

            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| format!("Failed to create GGUF model: {e}"))?;

            // With KV cache, we can generate more tokens efficiently
            let practical_max = config.max_tokens;

            let is_gqa = model.config().num_kv_heads < model.config().num_heads;
            let gqa_note = if is_gqa {
                format!(" (GQA: {} kv_heads)", model.config().num_kv_heads)
            } else {
                String::new()
            };

            // C-06 (Meyer DbC): EOS from model config, not hardcoded.
            let stop_tokens = model
                .config()
                .eos_token_id
                .map(|id| vec![id])
                .unwrap_or_default();

            let gen_config = QuantizedGenerateConfig {
                max_tokens: practical_max,
                temperature: config.temperature,
                top_k: 40,
                stop_tokens,
                trace: config.trace,
            };

            // Try CUDA GPU path first (200+ tok/s target)
            // Uses generate_gpu_resident which is the tested/working GPU path
            #[cfg(feature = "cuda")]
            if !config.force_cpu {
                use realizar::gguf::OwnedQuantizedModelCuda;
                if OwnedQuantizedModelCuda::is_available() {
                    // Print model info before attempting CUDA (model consumed on success)
                    let num_layers = model.config().num_layers;

                    match OwnedQuantizedModelCuda::new(model, 0) {
                        Ok(mut cuda_model) => {
                            let gpu_name = cuda_model.device_name().to_string();
                            let vram_mb = cuda_model.vram_mb();
                            println!(
                                "{}",
                                format!(
                                    "[GGUF CUDA: {} ({} MB VRAM), {} layers, {} tokens{}]",
                                    gpu_name, vram_mb, num_layers, practical_max, gqa_note
                                )
                                .bright_green()
                            );
                            // Use generate_gpu_resident (tested working path) not generate_full_cuda_with_cache
                            return cuda_model
                                .generate_gpu_resident(prompt, &gen_config)
                                .map_err(|e| format!("CUDA generate failed: {e}"));
                        }
                        Err(e) => {
                            println!(
                                "{}",
                                format!("[CUDA init failed: {}, falling back to CPU]", e).yellow()
                            );
                            // Re-create model for CPU fallback (model was consumed)
                            let model = OwnedQuantizedModel::from_mapped(&mapped)
                                .map_err(|e| format!("Failed to recreate model: {e}"))?;

                            return model
                                .generate_with_cache(prompt, &gen_config)
                                .map_err(|e| format!("GGUF generate failed: {e}"));
                        }
                    }
                }
            }

            // CPU path with KV cache (12+ tok/s) - used when CUDA feature disabled or unavailable
            // Use KV cache path for O(n) instead of O(n²)
            model
                .generate_with_cache(prompt, &gen_config)
                .map_err(|e| format!("GGUF generate failed: {e}"))
        }
}
