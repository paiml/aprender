//! InstructPipeline constructors: `new`, `from_pretrained`, `from_apr`,
//! `build_lora_layers`, `inject_adapter_weights`.

#[allow(clippy::wildcard_imports)]
use super::*;
use provable_contracts_macros::{ensures, requires};

impl InstructPipeline {
    /// Create a new pipeline with random weights.
    pub fn new(model_config: &TransformerConfig, instruct_config: InstructConfig) -> Self {
        let model = Transformer::new(model_config);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &instruct_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        #[allow(unused_mut)]
        let mut pipeline = Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer: None,
            model_dir: None,
            profiler: StepProfiler::disabled(),
            #[cfg(feature = "cuda")]
            cuda_trainer: None,
            #[cfg(feature = "cuda")]
            cuda_blocks: None,
            #[cfg(feature = "cuda")]
            shared_scratch: None,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training: None,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace: None,
            #[cfg(feature = "cuda")]
            lora_fused_clip: None,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states: None,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "cuda")]
            vram_guard: None,
            #[cfg(feature = "gpu")]
            wgpu_training: None,
        };

        #[cfg(feature = "cuda")]
        if pipeline.config.quantize_nf4 {
            pipeline.init_cuda(model_config);
        }

        // Initialize wgpu training if CUDA is not available
        #[cfg(feature = "gpu")]
        if pipeline.wgpu_training.is_none() {
            #[cfg(feature = "cuda")]
            let cuda_active = pipeline.cuda_blocks.is_some();
            #[cfg(not(feature = "cuda"))]
            let cuda_active = false;

            if !cuda_active {
                pipeline.try_init_wgpu(model_config);
            }
        }

        pipeline
    }

    /// Create pipeline from pretrained model weights.
    ///
    /// Loads transformer from SafeTensors and optionally a BPE tokenizer.
    ///
    /// # Errors
    /// Returns error if model files cannot be loaded.
    pub fn from_pretrained(
        model_dir: &Path,
        model_config: &TransformerConfig,
        instruct_config: InstructConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_safetensors(model_dir, model_config)?;
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &instruct_config);

        // ENT-269: Auto-load trained LoRA adapter if present in model directory.
        let adapter_path = model_dir.join("adapter_model.safetensors");
        if adapter_path.exists() {
            match crate::lora::load_adapter_peft(model_dir) {
                Ok((_config, weights)) => {
                    Self::inject_adapter_weights(
                        &mut lora_layers,
                        &weights,
                        model_config.num_hidden_layers,
                    );
                    eprintln!(
                        "[adapter] Loaded trained LoRA adapter ({} tensors) from {}",
                        weights.len(),
                        model_dir.display()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[adapter] Warning: adapter_model.safetensors found but failed to load: {e}"
                    );
                }
            }
        }

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        // CONTRACT: Training requires a BPE tokenizer — byte-fallback is not acceptable.
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Some(HfTokenizer::from_file(&tokenizer_path).map_err(|e| {
                crate::Error::ConfigError(format!(
                    "Failed to load tokenizer from '{}': {e}. \
                     Training requires a BPE tokenizer.",
                    tokenizer_path.display(),
                ))
            })?)
        } else {
            return Err(crate::Error::ConfigError(format!(
                "No tokenizer.json found in '{}'. Training requires a BPE tokenizer.",
                model_dir.display(),
            )));
        };

        #[allow(unused_mut)]
        let mut pipeline = Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer,
            model_dir: Some(model_dir.to_path_buf()),
            profiler: StepProfiler::disabled(),
            #[cfg(feature = "cuda")]
            cuda_trainer: None,
            #[cfg(feature = "cuda")]
            cuda_blocks: None,
            #[cfg(feature = "cuda")]
            shared_scratch: None,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training: None,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace: None,
            #[cfg(feature = "cuda")]
            lora_fused_clip: None,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states: None,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "cuda")]
            vram_guard: None,
            #[cfg(feature = "gpu")]
            wgpu_training: None,
        };

        #[cfg(feature = "cuda")]
        if pipeline.config.quantize_nf4 {
            pipeline.init_cuda(model_config);
        }

        Ok(pipeline)
    }

    /// Create pipeline from APR model file (.apr format).
    ///
    /// Loads transformer weights from the APR binary, dequantizing from any
    /// stored dtype (F16, Q4K, etc.) to F32. Loads sibling tokenizer if present
    /// (e.g., `model.tokenizer.json` next to `model.apr`).
    ///
    /// # Errors
    /// Returns error if APR file cannot be loaded or weights are invalid.
    /// CONTRACT L5: apr_tokenizer_embedding (model-format-conversion-v1.yaml)
    /// APR files are self-contained — tokenizer is extracted from embedded metadata.
    /// Sibling .tokenizer.json is a legacy fallback only.
    #[requires(apr_path.exists())]
    pub fn from_apr(
        apr_path: &Path,
        model_config: &TransformerConfig,
        instruct_config: InstructConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_apr(apr_path, model_config)?;
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &instruct_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        // Tokenizer resolution: APR is an embedded format — extract from metadata first.
        // Fallback 1: Sibling {stem}.tokenizer.json next to the .apr file
        // Fallback 2: Error — training requires a BPE tokenizer.
        let tokenizer = {
            // PRIMARY: Extract embedded tokenizer from APR metadata
            let embedded = Self::extract_embedded_tokenizer(apr_path);

            if let Some(tok) = embedded {
                eprintln!(
                    "[tokenizer] Loaded embedded BPE tokenizer from APR metadata (vocab_size={})",
                    tok.vocab_size(),
                );
                Some(tok)
            } else {
                // FALLBACK: Sibling tokenizer.json file
                let sibling = apr_path.file_stem().and_then(|stem| {
                    apr_path
                        .parent()
                        .map(|p| p.join(format!("{}.tokenizer.json", stem.to_str().unwrap_or(""))))
                });

                match sibling {
                    Some(ref path) if path.exists() => {
                        let tok = HfTokenizer::from_file(path).map_err(|e| {
                            crate::Error::ConfigError(format!(
                                "Failed to load tokenizer from '{}': {e}. \
                                 Training requires a BPE tokenizer.",
                                path.display(),
                            ))
                        })?;
                        eprintln!(
                            "[tokenizer] Loaded BPE tokenizer from sibling {} (vocab_size={})",
                            path.display(),
                            tok.vocab_size(),
                        );
                        Some(tok)
                    }
                    _ => {
                        return Err(crate::Error::ConfigError(format!(
                            "No tokenizer found for '{}'. APR metadata has no embedded \
                             tokenizer, and no sibling '{}.tokenizer.json' found. \
                             Re-import with `apr import` to embed the tokenizer, or \
                             place a tokenizer.json file next to the .apr file.",
                            apr_path.display(),
                            apr_path.file_stem().unwrap_or_default().to_str().unwrap_or(""),
                        )));
                    }
                }
            }
        };

        #[allow(unused_mut)]
        let mut pipeline = Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer,
            model_dir: Some(apr_path.to_path_buf()),
            profiler: StepProfiler::disabled(),
            #[cfg(feature = "cuda")]
            cuda_trainer: None,
            #[cfg(feature = "cuda")]
            cuda_blocks: None,
            #[cfg(feature = "cuda")]
            shared_scratch: None,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training: None,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace: None,
            #[cfg(feature = "cuda")]
            lora_fused_clip: None,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states: None,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "cuda")]
            vram_guard: None,
            #[cfg(feature = "gpu")]
            wgpu_training: None,
        };

        #[cfg(feature = "cuda")]
        if pipeline.config.quantize_nf4 {
            pipeline.init_cuda(model_config);
        }

        Ok(pipeline)
    }

    /// Extract BPE tokenizer from APR file's embedded metadata.
    ///
    /// CONTRACT: apr_tokenizer_embedding (model-format-conversion-v1.yaml, PMAT-154)
    /// APR files store tokenizer vocabulary and merges in the metadata section.
    /// This reconstructs a HuggingFace-compatible tokenizer.json from those fields.
    ///
    /// Returns None if the APR file lacks embedded tokenizer data (pre-PMAT-154 files).
    // CONTRACT L5: If tokenizer is extracted, it must have non-zero vocab
    #[ensures(ret.as_ref().is_none_or(|t| t.vocab_size() > 0))]
    fn extract_embedded_tokenizer(apr_path: &Path) -> Option<HfTokenizer> {
        use aprender::serialization::apr::AprReader;

        let reader = AprReader::open(apr_path).ok()?;

        // Extract vocabulary: tokenizer.vocabulary is an array of token strings
        let vocab_array = reader.metadata.get("tokenizer.vocabulary")?;
        let vocab: Vec<&str> = vocab_array.as_array()?.iter().filter_map(|v| v.as_str()).collect();

        if vocab.is_empty() {
            return None;
        }

        // Extract merges: tokenizer.merges is an array of "token1 token2" strings
        let merges: Vec<&str> = reader
            .metadata
            .get("tokenizer.merges")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // Reconstruct HuggingFace tokenizer.json format
        // Format: {"model": {"type": "BPE", "vocab": {"token": id, ...}, "merges": [...]}, "added_tokens": []}
        let mut vocab_map = serde_json::Map::new();
        for (id, token) in vocab.iter().enumerate() {
            vocab_map.insert(
                (*token).to_string(),
                serde_json::Value::Number(serde_json::Number::from(id)),
            );
        }

        let merges_json: Vec<serde_json::Value> =
            merges.iter().map(|m| serde_json::Value::String((*m).to_string())).collect();

        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": vocab_map,
                "merges": merges_json,
            },
            "added_tokens": [],
        });

        let json_str = serde_json::to_string(&tokenizer_json).ok()?;
        HfTokenizer::from_json(&json_str).ok()
    }

    /// Build LoRA layers for Q and V projections (same pattern as ClassifyPipeline).
    /// Build LoRA layers for Q and V projections of each transformer layer.
    pub fn build_lora_layers(
        model: &Transformer,
        model_config: &TransformerConfig,
        config: &InstructConfig,
    ) -> Vec<LoRALayer> {
        // rank=0 means no LoRA — return empty (no trainable adapters)
        if config.lora_rank == 0 {
            return Vec::new();
        }

        let hidden = model_config.hidden_size;
        let head_dim =
            model_config.head_dim_override.unwrap_or(hidden / model_config.num_attention_heads);

        let mut lora_layers = Vec::new();

        for layer in &model.layers {
            let attn = &layer.self_attn;

            // Q projection LoRA
            let q_dim = model_config.num_attention_heads * head_dim;
            let q_weight = Tensor::from_vec(
                attn.w_q.data().as_slice().expect("contiguous w_q").to_vec(),
                false,
            );
            lora_layers.push(LoRALayer::new(
                q_weight,
                q_dim,
                hidden,
                config.lora_rank,
                config.lora_alpha,
            ));

            // V projection LoRA
            let v_dim = model_config.num_kv_heads * head_dim;
            let v_weight = Tensor::from_vec(
                attn.w_v.data().as_slice().expect("contiguous w_v").to_vec(),
                false,
            );
            lora_layers.push(LoRALayer::new(
                v_weight,
                v_dim,
                hidden,
                config.lora_rank,
                config.lora_alpha,
            ));
        }

        lora_layers
    }

    /// Inject trained adapter weights from PEFT format into LoRA layers (ENT-269).
    ///
    /// Maps PEFT tensor names (e.g., `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`)
    /// to the corresponding LoRA layer index. Layers are ordered as [Q(0), V(0), Q(1), V(1), ...].
    fn inject_adapter_weights(
        lora_layers: &mut [LoRALayer],
        weights: &[(String, Vec<f32>)],
        num_layers: usize,
    ) {
        let mut loaded = 0usize;
        for (name, data) in weights {
            // Parse layer index from "layers.{idx}" in the tensor name
            let parts: Vec<&str> = name.split('.').collect();
            let layer_idx = parts
                .iter()
                .position(|&p| p == "layers")
                .and_then(|i| parts.get(i + 1))
                .and_then(|s| s.parse::<usize>().ok());

            let is_q = name.contains("q_proj");
            let is_a = name.contains("lora_A");

            if let Some(idx) = layer_idx {
                if idx >= num_layers {
                    continue;
                }
                let lora_idx = idx * 2 + usize::from(!is_q);
                if lora_idx >= lora_layers.len() {
                    continue;
                }

                let tensor = Tensor::from_vec(data.clone(), true);
                if is_a {
                    *lora_layers[lora_idx].lora_a_mut() = tensor;
                } else {
                    *lora_layers[lora_idx].lora_b_mut() = tensor;
                }
                loaded += 1;
            }
        }
        eprintln!("[adapter] Injected {loaded}/{} weight tensors", weights.len());
    }
}
