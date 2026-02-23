impl ChatSession {

        fn generate_safetensors(
            &mut self,
            prompt: &[u32],
            config: &ChatConfig,
        ) -> Result<Vec<u32>, String> {
            // GH-224: Try cached CUDA model first (no re-loading per message)
            #[cfg(feature = "cuda")]
            if !config.force_cpu && !self.cuda_init_failed {
                if let Some(ref mut cuda_model) = self.cached_safetensors_cuda {
                    // C-05 (Meyer DbC): EOS from model config, not hardcoded.
                    let eos_id = cuda_model.config().eos_token_id.unwrap_or(0);

                    let tokens = cuda_model
                        .generate(prompt, config.max_tokens, eos_id)
                        .map_err(|e| format!("SafeTensors CUDA generate failed: {e}"))?;

                    if config.trace {
                        let new_tokens = &tokens[prompt.len()..];
                        eprintln!(
                            "[APR-TRACE] SafeTensors GPU generated {} tokens: {:?}",
                            new_tokens.len(),
                            &new_tokens[..new_tokens.len().min(20)]
                        );
                    }

                    return Ok(tokens[prompt.len()..].to_vec());
                }
            }

            // CPU path: Use realizar's SafeTensors inference via AprTransformer
            use realizar::apr_transformer::GenerateConfig;
            use realizar::safetensors_infer::SafetensorsToAprConverter;

            let transformer = SafetensorsToAprConverter::convert(&self.model_path)
                .map_err(|e| format!("SafeTensors conversion failed: {e}"))?;

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
                .map_err(|e| format!("SafeTensors generate failed: {e}"))
        }

        #[allow(dead_code)]
        pub(super) fn history(&self) -> &[ChatMessage] {
            &self.history
        }

        pub(super) fn history_len(&self) -> usize {
            self.history.len()
        }

        pub(super) fn add_to_history(&mut self, role: &str, content: &str) {
            self.history.push(ChatMessage::new(role, content));
        }

        pub(super) fn clear_history(&mut self) {
            self.history.clear();
        }

        #[allow(dead_code)]
        pub(super) fn format(&self) -> ModelFormat {
            self.format
        }

        #[allow(dead_code)]
        pub(super) fn template_format(&self) -> TemplateFormat {
            self.template_format
        }
}
