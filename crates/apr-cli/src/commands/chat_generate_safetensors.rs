impl ChatSession {

        fn generate_safetensors(
            &mut self,
            prompt: &[u32],
            config: &ChatConfig,
        ) -> Result<Vec<u32>, String> {
            // GH-224: Try cached CUDA model first (no re-loading per message)
            // #4269: the ST CUDA path is untouched by the one-engine port —
            // only the CPU branch below now drives `crate::session::Session`.
            #[cfg(feature = "cuda")]
            if !config.force_cpu && !self.cuda_init_failed {
                if let Some(ref mut cuda_model) = self.cached_safetensors_cuda {
                    // C-05 (Meyer DbC): EOS from model config, not hardcoded.
                    let eos_id = cuda_model.config().eos_token_id.unwrap_or(0);

                    let tokens = cuda_model
                        .generate(prompt, config.max_tokens, eos_id)
                        .map_err(|e| format!("SafeTensors CUDA generate failed: {e}"))?;
                    // #3794: record the backend that actually answered.
                    self.generated_on_gpu = true;

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

            // CPU path (#4269, workstream M of #4263): drive generation through
            // the one engine, `realizar::session::Session<StCpuForward>`,
            // instead of `AprTransformer::generate_with_cache`'s own loop.
            use realizar::gguf::QuantizedGenerateConfig;
            use realizar::safetensors_infer::{SafetensorsToAprConverter, StCpuForward};
            use realizar::session::Session;

            // #3022: a sharded checkout reaches the SAME transformer, through the same
            // three calls `apr run` already makes (infer/mod_log_transformer_eos.rs:116).
            // `convert` takes a path to tensor bytes and cannot be pointed at a manifest,
            // so the index gets its own two lines rather than a second code path.
            let transformer = if self.format == ModelFormat::ShardedSafeTensors {
                use realizar::safetensors::{SafetensorsConfig, ShardedSafeTensorsModel};
                let sharded = ShardedSafeTensorsModel::load_from_index(&self.model_path)
                    .map_err(|e| format!("Sharded SafeTensors index load failed: {e}"))?;
                let st_config = SafetensorsConfig::load_from_sibling(&self.model_path)
                    .ok_or_else(|| {
                        format!(
                            "config.json not found beside {} (required for SafeTensors inference)",
                            self.model_path.display()
                        )
                    })?;
                SafetensorsToAprConverter::convert_sharded(&sharded, &st_config)
                    .map_err(|e| format!("Sharded SafeTensors conversion failed: {e}"))?
            } else {
                SafetensorsToAprConverter::convert(&self.model_path)
                    .map_err(|e| format!("SafeTensors conversion failed: {e}"))?
            };
            let model = transformer.into_inner();

            let gen_config = QuantizedGenerateConfig {
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_k: 0,
                top_p: config.top_p,
                // #3760: the sampler draws now; no seed is plumbed from this caller.
                seed: realizar::apr_transformer::DEFAULT_SEED,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                // apr_transformer::generation::is_eos_token (GH-330) treated
                // token 0 as EOS unconditionally; Session has no such builtin,
                // so it is carried here as an explicit stop token to keep this
                // port's stopping behavior identical.
                stop_tokens: vec![0],
                trace: config.trace,
                logprobs: false,
                cancel: realizar::generate::CancelToken::never(),
            };

            let mut session = Session::new(StCpuForward::new(&model));
            let turn = session
                .generate(prompt, &gen_config, &mut |_tok| true)
                .map_err(|e| format!("SafeTensors generate failed: {e}"))?;
            Ok(turn.tokens)
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

        /// #3367: did any turn in this session fail to generate? Read once, by
        /// `run_repl`, to decide the command's exit code.
        pub(super) fn had_generate_error(&self) -> bool {
            self.had_generate_error
        }

        /// #3794: did an accelerator actually answer any turn this session?
        pub(super) fn generated_on_gpu(&self) -> bool {
            self.generated_on_gpu
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
