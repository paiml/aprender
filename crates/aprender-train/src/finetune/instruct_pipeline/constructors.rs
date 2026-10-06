//! InstructPipeline constructors: `new`, `from_pretrained`, `from_apr`,
//! `build_lora_layers`, `inject_adapter_weights`.

#[allow(clippy::wildcard_imports)]
use super::*;
use crate::lora::LoraTarget;
use provable_contracts_macros::{ensures, requires};

impl InstructPipeline {
    /// Create a new pipeline with random weights.
    ///
    /// # Panics
    /// If `instruct_config.lora_targets` is not `q_proj`, `v_proj`
    /// ([`InstructPipeline::check_lora_targets`]).
    pub fn new(model_config: &TransformerConfig, instruct_config: InstructConfig) -> Self {
        if let Err(e) = Self::check_lora_targets(&instruct_config) {
            panic!("{e}");
        }
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
    /// Returns error if `instruct_config.lora_targets` is not `q_proj`, `v_proj`
    /// ([`InstructPipeline::check_lora_targets`]), before anything is loaded,
    /// or if model files cannot be loaded.
    pub fn from_pretrained(
        model_dir: &Path,
        model_config: &TransformerConfig,
        instruct_config: InstructConfig,
    ) -> crate::Result<Self> {
        Self::check_lora_targets(&instruct_config)?;
        let model = Transformer::from_safetensors(model_dir, model_config)?;
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &instruct_config);

        // ENT-269: Auto-load trained LoRA adapter if present in model directory.
        if let Some(tensors) =
            Self::load_trained_adapter(model_dir, &mut lora_layers, &instruct_config.lora_targets)?
        {
            eprintln!(
                "[adapter] Loaded trained LoRA adapter ({tensors} tensors) from {}",
                model_dir.display()
            );
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
    /// Returns error if `instruct_config.lora_targets` is not `q_proj`, `v_proj`
    /// ([`InstructPipeline::check_lora_targets`]), before anything is loaded,
    /// or if APR file cannot be loaded or weights are invalid.
    /// CONTRACT L5: apr_tokenizer_embedding (model-format-conversion-v1.yaml)
    /// APR files are self-contained — tokenizer is extracted from embedded metadata.
    /// Sibling .tokenizer.json is a legacy fallback only.
    #[requires(apr_path.exists())]
    pub fn from_apr(
        apr_path: &Path,
        model_config: &TransformerConfig,
        instruct_config: InstructConfig,
    ) -> crate::Result<Self> {
        Self::check_lora_targets(&instruct_config)?;
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

    /// Build one LoRA layer per transformer layer and target in
    /// `config.lora_targets`, in slot order: slot `|T|·layer + position`
    /// ([`LoraTargets::slot`], `FALSIFY-LORA_TARGET_SELECTION_V1_004`).
    /// Each adapter wraps a copy of its projection's base weight, stored
    /// `[d_out, d_in]`.
    pub fn build_lora_layers(
        model: &Transformer,
        model_config: &TransformerConfig,
        config: &InstructConfig,
    ) -> Vec<LoRALayer> {
        // rank=0 means no LoRA — return empty (no trainable adapters)
        if config.lora_rank == 0 {
            return Vec::new();
        }

        let mut lora_layers = Vec::new();
        for layer in &model.layers {
            let (attn, ffn) = (&layer.self_attn, &layer.ffn);
            for &target in config.lora_targets.as_slice() {
                let base = match target {
                    LoraTarget::Q => &attn.w_q,
                    LoraTarget::K => &attn.w_k,
                    LoraTarget::V => &attn.w_v,
                    LoraTarget::O => &attn.w_o,
                    LoraTarget::Gate => &ffn.w_gate,
                    LoraTarget::Up => &ffn.w_up,
                    LoraTarget::Down => &ffn.w_down,
                };
                let (d_out, d_in) = target.dims(model_config);
                let weight = Tensor::from_vec(
                    base.data().as_slice().expect("contiguous base weight").to_vec(),
                    false,
                );
                lora_layers.push(LoRALayer::new(
                    weight,
                    d_out,
                    d_in,
                    config.lora_rank,
                    config.lora_alpha,
                ));
            }
        }

        lora_layers
    }

    /// Refuse a target set the instruct pipeline does not train
    /// (`FALSIFY-LORA_TARGET_SELECTION_V1_004`).
    ///
    /// `build_lora_layers` and `inject_adapter_weights` place any target set,
    /// and the CUDA blocks and their sync back to the CPU move every target
    /// (`FALSIFY-LORA_TARGET_SELECTION_V1_007`), but the CPU forward
    /// (`forward_hidden_with_lora`) reads slot `2·layer` as `q_proj` and
    /// `2·layer + 1` as `v_proj`, and the checkpoint names follow the same
    /// rule. Under any other set they would apply an adapter to the wrong
    /// projection or leave one untrained, so every constructor calls this
    /// before it loads a weight.
    ///
    /// # Errors
    /// `Error::ConfigError` naming every selected target that would not be
    /// trained and every one of `q_proj`, `v_proj` that is missing.
    pub fn check_lora_targets(config: &InstructConfig) -> crate::Result<()> {
        let trained = LoraTargets::default();
        let targets = &config.lora_targets;
        if *targets == trained {
            return Ok(());
        }
        // The targets of `from` that `other` lacks, by name.
        let outside = |from: &LoraTargets, other: &LoraTargets| -> String {
            let names: Vec<&str> = from
                .as_slice()
                .iter()
                .filter(|&&t| !other.contains(t))
                .map(|t| t.module_name())
                .collect();
            names.join(", ")
        };
        let mut parts =
            vec![format!("LoRA targets {targets}: the instruct pipeline trains exactly {trained}")];
        let untrained = outside(targets, &trained);
        if !untrained.is_empty() {
            parts.push(format!("would not train: {untrained}"));
        }
        let missing = outside(&trained, targets);
        if !missing.is_empty() {
            parts.push(format!("missing: {missing}"));
        }
        Err(crate::Error::ConfigError(parts.join("; ")))
    }

    /// Load the trained adapter in `model_dir`, if there is one, into `lora_layers`.
    ///
    /// Returns `Ok(None)` when `adapter_model.safetensors` is absent; the layers keep
    /// their init. An adapter that is present must load: one that cannot be read, or
    /// whose `adapter_config.json` declares targets other than `targets`, refuses the
    /// load instead of serving fresh LoRA weights, and no layer changes
    /// (`FALSIFY-LORA_TARGET_SELECTION_V1_015`). Returns the number of tensors loaded.
    ///
    /// # Errors
    /// Returns `Error::ConfigError` naming the path and cause of a failed read, both
    /// target sets on a mismatch, or the tensors [`Self::inject_adapter_weights`] refuses.
    pub(crate) fn load_trained_adapter(
        model_dir: &Path,
        lora_layers: &mut [LoRALayer],
        targets: &LoraTargets,
    ) -> crate::Result<Option<usize>> {
        let adapter_path = model_dir.join("adapter_model.safetensors");
        if !adapter_path.exists() {
            return Ok(None);
        }
        let (peft, weights) = crate::lora::load_adapter_peft(model_dir).map_err(|e| {
            crate::Error::ConfigError(format!(
                "adapter: {} found but failed to load: {e}",
                adapter_path.display()
            ))
        })?;
        check_adapter_targets(&peft.target_modules, targets)?;
        Self::inject_adapter_weights(lora_layers, &weights, targets)?;
        Ok(Some(weights.len()))
    }

    /// Inject trained adapter weights from PEFT format into LoRA layers (ENT-269).
    ///
    /// Maps PEFT tensor names (e.g., `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`)
    /// to the LoRA layer of their layer and projection, at slot `|T|·layer + position`
    /// ([`LoraTargets::slot`]), the order `build_lora_layers` builds them in. For the
    /// default targets that is [Q(0), V(0), Q(1), V(1), ...].
    ///
    /// Every tensor is placed before any is written (`FALSIFY-LORA_TARGET_SELECTION_V1_003`).
    /// A tensor that targets a projection outside `targets` or a layer the model does not
    /// have, is neither `lora_A` nor `lora_B`, repeats a slot, or differs in length from
    /// its slot (a rank mismatch, or bf16/f16 bytes read as f32) refuses the whole load,
    /// and no LoRA layer changes.
    ///
    /// The adapter must also fill every place, the A and the B of every slot; a place it
    /// leaves refuses the load too, so no slot keeps its fresh init
    /// (`FALSIFY-LORA_TARGET_SELECTION_V1_014`).
    ///
    /// # Errors
    /// Returns `Error::ConfigError` naming every tensor that could not be placed and every place left unfilled.
    fn inject_adapter_weights(
        lora_layers: &mut [LoRALayer],
        weights: &[(String, Vec<f32>)],
        targets: &LoraTargets,
    ) -> crate::Result<()> {
        let mut placed: Vec<(usize, bool, &[f32])> = Vec::with_capacity(weights.len());
        let mut unplaced: Vec<&str> = Vec::new();
        for (name, data) in weights {
            let slot = adapter_slot(name, targets).filter(|&(idx, is_a)| {
                slot_len(lora_layers, idx, is_a) == Some(data.len())
                    && !placed.iter().any(|&(i, a, _)| i == idx && a == is_a)
            });
            match slot {
                Some((idx, is_a)) => placed.push((idx, is_a, data)),
                None => unplaced.push(name),
            }
        }
        unplaced.sort_unstable();
        let unfilled = unfilled_places(lora_layers.len(), &placed, targets);
        let mut parts = Vec::new();
        if !unplaced.is_empty() {
            parts.push(format!(
                "{} of {} tensors fit no LoRA layer of this model (targets {targets}; \
                 wrong projection, layer, rank or dtype, or a repeated slot): {}",
                unplaced.len(),
                weights.len(),
                unplaced.join(", ")
            ));
        }
        if !unfilled.is_empty() {
            parts.push(format!(
                "{} of {} LoRA places have no tensor, so they would keep their fresh init: {}",
                unfilled.len(),
                2 * lora_layers.len(),
                unfilled.join(", ")
            ));
        }
        if !parts.is_empty() {
            return Err(crate::Error::ConfigError(format!("adapter: {}", parts.join("; "))));
        }
        for &(idx, is_a, data) in &placed {
            let tensor = Tensor::from_vec(data.to_vec(), true);
            if is_a {
                *lora_layers[idx].lora_a_mut() = tensor;
            } else {
                *lora_layers[idx].lora_b_mut() = tensor;
            }
        }
        eprintln!("[adapter] Injected {}/{} weight tensors", placed.len(), weights.len());
        Ok(())
    }
}

/// The LoRA slot a PEFT tensor name targets, as `(index into lora_layers, is lora_A)`,
/// or `None` when the name is not a `lora_A`/`lora_B` tensor of a projection in
/// `targets` of a numbered layer. Components are matched whole, so `qkv_proj` or
/// `lora_a` targets nothing. A layer the model does not have yields a slot past the
/// end of `lora_layers`, which [`slot_len`] rejects.
fn adapter_slot(name: &str, targets: &LoraTargets) -> Option<(usize, bool)> {
    let parts: Vec<&str> = name.split('.').collect();
    let layer = parts
        .iter()
        .position(|&p| p == "layers")
        .and_then(|i| parts.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())?;
    let target = parts.iter().find_map(|p| LoraTarget::from_module_name(p))?;
    let is_a = match (parts.contains(&"lora_A"), parts.contains(&"lora_B")) {
        (true, false) => true,
        (false, true) => false,
        _ => return None,
    };
    Some((targets.slot(layer, target)?, is_a))
}

/// Every LoRA place (slot, A or B) of `num_slots` slots that `placed` leaves without a
/// tensor, as `layers.{l}.{target}.lora_{A|B}` with l = slot / |T| and target =
/// T[slot mod |T|] (R15a C8, FALSIFY-LORA_TARGET_SELECTION_V1_014).
fn unfilled_places(
    num_slots: usize,
    placed: &[(usize, bool, &[f32])],
    targets: &LoraTargets,
) -> Vec<String> {
    let per_layer = targets.per_layer().max(1);
    (0..num_slots)
        .flat_map(|idx| [(idx, true), (idx, false)])
        .filter(|&(idx, is_a)| !placed.iter().any(|&(i, a, _)| i == idx && a == is_a))
        .map(|(idx, is_a)| {
            let module = targets
                .as_slice()
                .get(idx % per_layer)
                .map_or("?", |t| LoraTarget::module_name(*t));
            let kind = if is_a { "lora_A" } else { "lora_B" };
            format!("layers.{}.{module}.{kind}", idx / per_layer)
        })
        .collect()
}

/// Length of the A or B tensor of LoRA slot `idx`, or `None` past the last slot.
fn slot_len(lora_layers: &[LoRALayer], idx: usize, is_a: bool) -> Option<usize> {
    let layer = lora_layers.get(idx)?;
    Some(if is_a { layer.lora_a().len() } else { layer.lora_b().len() })
}

/// Refuse an adapter whose `adapter_config.json` declares targets other than `targets`
/// (`FALSIFY-LORA_TARGET_SELECTION_V1_015`). An empty declaration names no targets, so
/// tensor placement alone decides (rows 003 and 014).
fn check_adapter_targets(declared: &[String], targets: &LoraTargets) -> crate::Result<()> {
    if declared.is_empty() {
        return Ok(());
    }
    let declared = LoraTargets::parse(declared).map_err(|e| {
        crate::Error::ConfigError(format!("adapter: adapter_config.json target_modules: {e}"))
    })?;
    if declared != *targets {
        return Err(crate::Error::ConfigError(format!(
            "adapter: trained on targets {declared}, but this pipeline selects {targets}; \
             load it with lora_targets {declared}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "constructors_adapter_tests.rs"]
mod adapter_tests;

#[cfg(test)]
#[path = "constructors_target_tests.rs"]
mod target_tests;
