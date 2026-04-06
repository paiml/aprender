
fn log_transformer_cpu_info(
    config: &crate::apr_transformer::AprTransformerConfig,
    load_ms: f64,
) {
    let thread_count = rayon::current_num_threads();
    eprintln!(
        "Architecture: {} ({} layers, vocab_size={})",
        config.architecture, config.num_layers, config.vocab_size
    );
    eprintln!(
        "Config: hidden_size={}, context_length={}, quant=F32, threads={}",
        config.hidden_dim, config.context_length, thread_count
    );
    eprintln!("Model loaded in {:.1}ms", load_ms);
    eprintln!("Backend: CPU (SIMD-accelerated)");
}

/// GH-330: Check EOS using config-provided stop tokens (Design by Contract).
fn is_eos_token(token: u32, stop_tokens: &[u32]) -> bool {
    token == 0 || stop_tokens.contains(&token)
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

fn greedy_decode_with_transformer(
    transformer: &crate::safetensors::ValidatedAprTransformer,
    input_tokens: &[u32],
    max_tokens: usize,
) -> Result<Vec<u32>> {
    use crate::apr_transformer::AprKVCache;

    let mut cache = AprKVCache::new(&transformer.config);
    let mut all_tokens = input_tokens.to_vec();

    let mut logits = Vec::new();
    for (pos, &token) in input_tokens.iter().enumerate() {
        logits = transformer.forward_with_cache(token, &mut cache, pos)?;
    }

    for _ in 0..max_tokens {
        let next_token = greedy_argmax(&logits);
        // GH-330: Use model config EOS
        let stop_tokens: Vec<u32> = transformer.config.eos_token_id.into_iter().collect();
        if is_eos_token(next_token, &stop_tokens) {
            break;
        }
        all_tokens.push(next_token);
        let pos = all_tokens.len() - 1;
        logits = transformer.forward_with_cache(next_token, &mut cache, pos)?;
    }

    Ok(all_tokens)
}

fn decode_safetensors_output(model_path: &std::path::Path, generated_tokens: &[u32]) -> String {
    use crate::apr::AprV2Model;

    if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        clean_model_output(&tokenizer.decode(generated_tokens))
    } else {
        format!(
            "[{} tokens generated, tokenizer not found]",
            generated_tokens.len()
        )
    }
}

/// Run SafeTensors inference on CPU via AprTransformer conversion (PMAT-103)
fn run_safetensors_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
) -> Result<InferenceResult> {
    use crate::safetensors_infer::SafetensorsToAprConverter;

    let load_start = Instant::now();
    let transformer = SafetensorsToAprConverter::convert(&config.model_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        log_transformer_cpu_info(&transformer.config, load_ms);
    }

    let infer_start = Instant::now();
    let all_tokens = greedy_decode_with_transformer(&transformer, input_tokens, config.max_tokens)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let generated_tokens = &all_tokens[input_token_count..];
    let text = decode_safetensors_output(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "SafeTensors".to_string(),
        used_gpu: false,
    })
}

/// Run sharded SafeTensors inference (GH-213)
///
/// Loads a sharded model from its index.json, converts to AprTransformer,
/// and runs the same CPU inference loop as single-file SafeTensors.
fn run_sharded_safetensors_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    use crate::safetensors::{SafetensorsConfig, ShardedSafeTensorsModel};
    use crate::safetensors_infer::SafetensorsToAprConverter;

    if config.verbose {
        eprintln!(
            "Loading sharded SafeTensors model: {}",
            config.model_path.display()
        );
    }

    let load_start = Instant::now();

    let sharded = ShardedSafeTensorsModel::load_from_index(&config.model_path)?;

    if config.verbose {
        eprintln!(
            "Loaded {} shards, {} tensors",
            sharded.shard_count(),
            sharded.tensor_count()
        );
    }

    let st_config = SafetensorsConfig::load_from_sibling(&config.model_path).ok_or_else(|| {
        RealizarError::UnsupportedOperation {
            operation: "sharded_safetensors_convert".to_string(),
            reason: "config.json not found (required for SafeTensors inference)".to_string(),
        }
    })?;

    let transformer = SafetensorsToAprConverter::convert_sharded(&sharded, &st_config)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        log_transformer_cpu_info(&transformer.config, load_ms);
    }

    let input_tokens = prepared.tokens();
    let input_token_count = prepared.input_count();

    let infer_start = Instant::now();
    let all_tokens = greedy_decode_with_transformer(&transformer, input_tokens, config.max_tokens)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let generated_tokens = &all_tokens[input_token_count..];
    let text = decode_safetensors_output(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "SafeTensors".to_string(),
        used_gpu: false,
    })
}

/// Pre-fault mmap pages to avoid page faults during inference
fn prefault_mmap(data: &[u8]) {
    let page_size = 4096;
    let mut checksum: u8 = 0;
    for i in (0..data.len()).step_by(page_size) {
        checksum = checksum.wrapping_add(data[i]);
    }
    std::hint::black_box(checksum);
}

/// Find a fallback tokenizer for APR models (GH-156)
///
/// This function tries to load the embedded tokenizer from the APR model.
/// APR files can contain the vocabulary in metadata, so we don't need
/// a sibling tokenizer.json file.
///
/// # Arguments
/// * `model_path` - Path to the APR model file
///
/// # Returns
/// * `Some(BpeTokenizer)` - If embedded tokenizer found and converted
/// * `None` - If no embedded tokenizer available
fn find_fallback_tokenizer(model_path: &std::path::Path) -> Option<crate::apr::BpeTokenizer> {
    use crate::apr::AprV2Model;

    // F-REGR-232: Only search if the model can be loaded
    let model = AprV2Model::load(model_path).ok()?;

    // 1. Embedded BPE tokenizer (preferred — has merges)
    if let Some(bpe_tokenizer) = model.load_embedded_bpe_tokenizer() {
        return Some(bpe_tokenizer);
    }

    // 2. GH-366: SentencePiece tokenizer converted to BPE for decode-only use
    if let Some(tok) = convert_sentencepiece_to_bpe(&model) {
        return Some(tok);
    }

    // 3. SimpleTokenizer converted to BPE (decode-only, no merges)
    if let Some(tok) = convert_simple_tokenizer_to_bpe(&model) {
        return Some(tok);
    }

    // 3. Search HuggingFace cache and APR tokenizer cache
    search_external_tokenizer_caches()
}

/// GH-366: Convert embedded SentencePiece vocab to BPE for decode-only inference
fn convert_sentencepiece_to_bpe(
    model: &crate::apr::AprV2Model,
) -> Option<crate::apr::BpeTokenizer> {
    let sp = model.load_embedded_sentencepiece_tokenizer()?;
    let id_to_token: Vec<String> = (0..sp.vocab_size() as u32)
        .map(|id| sp.decode(&[id]).unwrap_or_default())
        .collect();
    let token_to_id: std::collections::HashMap<String, u32> = id_to_token
        .iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), id as u32))
        .collect();
    let special_tokens = crate::apr::extract_special_tokens_from_vocab(&token_to_id);
    eprintln!(
        "[GH-366] Converted SentencePiece to BPE-compatible tokenizer for inference: {} tokens",
        id_to_token.len()
    );
    Some(crate::apr::BpeTokenizer {
        token_to_id,
        id_to_token,
        merge_rules: Vec::new(),
        bos_id: None,
        eos_id: None,
        special_tokens,
    })
}

/// Convert embedded SimpleTokenizer to BpeTokenizer (GH-189)
fn convert_simple_tokenizer_to_bpe(
    model: &crate::apr::AprV2Model,
) -> Option<crate::apr::BpeTokenizer> {
    let simple_tokenizer = model.load_embedded_tokenizer()?;
    let token_to_id: std::collections::HashMap<String, u32> = simple_tokenizer
        .id_to_token
        .iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), id as u32))
        .collect();
    let special_tokens = crate::apr::extract_special_tokens_from_vocab(&token_to_id);
    Some(crate::apr::BpeTokenizer {
        token_to_id,
        id_to_token: simple_tokenizer.id_to_token,
        merge_rules: Vec::new(),
        bos_id: simple_tokenizer.bos_token_id,
        eos_id: simple_tokenizer.eos_token_id,
        special_tokens,
    })
}

/// Search HuggingFace and APR caches for Qwen tokenizer
fn search_external_tokenizer_caches() -> Option<crate::apr::BpeTokenizer> {
    use crate::apr::AprV2Model;

    let home = std::env::var("HOME").ok().map(std::path::PathBuf::from)?;

    // Search HuggingFace cache (PMAT-SHOWCASE-TOKENIZER-001)
    let hf_cache = home.join(".cache/huggingface/hub");
    if let Some(tok) = search_hf_cache_for_tokenizer(&hf_cache) {
        return Some(tok);
    }

    // Check APR tokenizer cache
    AprV2Model::load_tokenizer_from_path(&home.join(".apr/tokenizers/qwen2/tokenizer.json"))
}

/// Search HuggingFace model cache for Qwen tokenizer.json
fn search_hf_cache_for_tokenizer(hf_cache: &std::path::Path) -> Option<crate::apr::BpeTokenizer> {
    use crate::apr::AprV2Model;

    let entries = std::fs::read_dir(hf_cache).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("models--Qwen") {
            continue;
        }
        let snapshots_dir = entry.path().join("snapshots");
        let snapshots = std::fs::read_dir(&snapshots_dir).ok()?;
        for snapshot in snapshots.flatten() {
            let tokenizer_path = snapshot.path().join("tokenizer.json");
            if let Some(tok) = AprV2Model::load_tokenizer_from_path(&tokenizer_path) {
                return Some(tok);
            }
        }
    }
    None
}

/// Clean model output by stripping ChatML markers
fn clean_model_output(raw: &str) -> String {
    let mut cleaned = raw.to_string();
    let markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ];
    for marker in markers {
        cleaned = cleaned.replace(marker, "");
    }
    cleaned.trim().to_string()
}

// ============================================================================
// MOCK BACKEND (PMAT-COV-95: Testing without disk I/O)
// ============================================================================

/// Run mock inference for testing (PMAT-COV-95)
///
/// This function returns deterministic results without reading disk or
/// performing actual model inference. It exercises the full InferenceResult
/// construction, token counting, timing calculation, and formatting logic.
///
/// # Mock Behavior
///
/// - Input tokens: parsed from prompt or used directly from config
/// - Generated tokens: deterministic sequence [100, 101, 102, ...]
/// - Text output: "mock response for: <prompt>"
/// - Timing: simulated 10ms load, 50ms inference
/// - Format: "Mock"
pub fn run_mock_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    // Simulate model loading time
    let load_ms = 10.0;

    // Parse input tokens
    let input_tokens = if let Some(ref tokens) = config.input_tokens {
        tokens.clone()
    } else if let Some(ref prompt) = config.prompt {
        // Mock tokenization: each word becomes a token ID
        prompt
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| (i + 1) as u32)
            .collect()
    } else {
        vec![1u32] // BOS token
    };

    let input_token_count = input_tokens.len();

    // Generate deterministic output tokens
    let num_to_generate = config.max_tokens.min(32);
    let generated_tokens: Vec<u32> = (0..num_to_generate).map(|i| 100 + i as u32).collect();

    // Combine input and generated tokens
    let mut all_tokens = input_tokens;
    all_tokens.extend(&generated_tokens);

    // Mock text output
    let prompt_text = config.prompt.as_deref().unwrap_or("(no prompt)");
    let text = format!("mock response for: {}", prompt_text);

    // Simulate inference timing
    let inference_ms = 50.0 + (num_to_generate as f64 * 2.0);
    let generated_token_count = generated_tokens.len();
    let tok_per_sec = if inference_ms > 0.0 {
        generated_token_count as f64 / (inference_ms / 1000.0)
    } else {
        0.0
    };

    // Validate configuration constraints
    if config.temperature < 0.0 {
        return Err(RealizarError::InvalidConfiguration(
            "temperature cannot be negative".to_string(),
        ));
    }

    if config.max_tokens == 0 {
        return Err(RealizarError::InvalidConfiguration(
            "max_tokens must be > 0".to_string(),
        ));
    }

    // Write trace output if requested
    if let Some(ref trace_path) = config.trace_output {
        let trace_json = format!(
            r#"{{
  "version": "1.0",
  "mock": true,
  "input_tokens": {},
  "generated_tokens": {},
  "load_ms": {:.2},
  "inference_ms": {:.2}
}}
"#,
            input_token_count, generated_token_count, load_ms, inference_ms
        );
        std::fs::write(trace_path, trace_json).map_err(|e| RealizarError::IoError {
            message: format!("Failed to write trace: {}", e),
        })?;
    }

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec,
        load_ms,
        format: "Mock".to_string(),
        used_gpu: false,
    })
}

/// Create a mock inference config for testing
#[must_use]
pub fn mock_config(prompt: &str) -> InferenceConfig {
    InferenceConfig::new("/dev/null")
        .with_prompt(prompt)
        .with_max_tokens(16)
        .with_mock_backend()
}

impl InferenceConfig {
    /// Enable mock backend for testing (PMAT-COV-95)
    #[must_use]
    pub fn with_mock_backend(mut self) -> Self {
        self.use_mock_backend = true;
        self
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod infer_tests;

// Additional coverage tests (tests_part_02.rs)
#[cfg(test)]
#[path = "tests_max_tokens.rs"]
mod infer_tests_part_02;
