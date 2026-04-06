/// Batch inference: load model once, process multiple prompts sequentially.
///
/// Eliminates per-invocation model load + CUDA JIT overhead (~80s on gx10)
/// by keeping the model resident across all prompts.
/// Supports both GGUF and APR model formats.

/// Configuration for batch inference
#[derive(Debug, Clone)]
pub struct BatchInferenceConfig {
    /// Path to model file
    pub model_path: PathBuf,
    /// Maximum tokens to generate per prompt
    pub max_tokens: usize,
    /// Sampling temperature
    pub temperature: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Disable GPU
    pub no_gpu: bool,
    /// Verbose output
    pub verbose: bool,
    /// Additional stop tokens
    pub stop_tokens: Vec<u32>,
}

/// Single prompt in a batch
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BatchPrompt {
    /// The prompt text
    pub prompt: String,
    /// Optional task ID (passed through to output)
    #[serde(default)]
    pub task_id: Option<String>,
    /// Optional per-prompt max tokens override
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// Single result from batch inference
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchResult {
    /// Task ID (passed through from input)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Tokens per second
    pub tok_per_sec: f64,
    /// Inference time in milliseconds (generation only, excludes model load)
    pub inference_ms: f64,
    /// Whether GPU was used
    pub used_gpu: bool,
    /// Error message if this prompt failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Aggregate stats for the entire batch
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total number of prompts processed
    pub total_prompts: usize,
    /// Number of successful generations
    pub successful: usize,
    /// Number of failed generations
    pub failed: usize,
    /// Total tokens generated across all prompts
    pub total_tokens_generated: usize,
    /// Total inference time in milliseconds (excludes model load)
    pub total_inference_ms: f64,
    /// Model load time in milliseconds
    pub model_load_ms: f64,
}

/// Loaded batch model: holds GPU and/or CPU model handles.
struct BatchModel {
    #[cfg(feature = "cuda")]
    gpu: Option<crate::gguf::OwnedQuantizedModelCuda>,
    /// GH-560: wgpu (Vulkan) fallback for batch inference
    #[cfg(feature = "gpu")]
    wgpu: Option<WgpuBatchState>,
    cpu: Option<crate::gguf::OwnedQuantizedModel>,
}

// GH-560: wgpu batch state extracted to batch_wgpu.rs (< 500 line gate)
#[cfg(feature = "gpu")]
include!("batch_wgpu.rs");

impl BatchModel {
    fn generate(
        &mut self,
        input_tokens: &[u32],
        config: &crate::gguf::QuantizedGenerateConfig,
    ) -> std::result::Result<(Vec<u32>, bool), RealizarError> {
        #[cfg(feature = "cuda")]
        if let Some(ref mut gpu) = self.gpu {
            return gpu
                .generate_gpu_resident(input_tokens, config)
                .map(|tokens| (tokens, true))
                .map_err(|e| {
                    RealizarError::InferenceError(format!("GPU generation failed: {}", e))
                });
        }

        // GH-560: wgpu fallback before CPU
        #[cfg(feature = "gpu")]
        if let Some(ref mut wgpu) = self.wgpu {
            return wgpu.generate(input_tokens, config)
                .map(|tokens| (tokens, true));
        }

        if let Some(ref cpu) = self.cpu {
            return cpu
                .generate_with_cache(input_tokens, config)
                .map(|tokens| (tokens, false))
                .map_err(|e| {
                    RealizarError::InferenceError(format!("CPU generation failed: {}", e))
                });
        }

        Err(RealizarError::InferenceError("No model available".to_string()))
    }
}

/// Run batch inference, auto-detecting model format (GGUF or APR).
///
/// Loads the model once, then processes each prompt from the reader.
/// Results are written as JSONL to the writer, one line per prompt.
pub fn run_batch_inference<R, W>(
    config: &BatchInferenceConfig,
    reader: R,
    writer: W,
) -> Result<BatchStats>
where
    R: std::io::BufRead,
    W: std::io::Write,
{
    validate_model_path(&config.model_path)?;

    let format = {
        let mut file = std::fs::File::open(&config.model_path).map_err(|e| {
            RealizarError::IoError { message: format!("Failed to open model: {}", e) }
        })?;
        let mut magic = [0u8; 8];
        std::io::Read::read_exact(&mut file, &mut magic).map_err(|e| {
            RealizarError::IoError { message: format!("Failed to read magic: {}", e) }
        })?;
        crate::format::detect_format(&magic).map_err(|e| RealizarError::FormatError {
            reason: format!("Format detection failed: {}", e),
        })?
    };

    match format {
        ModelFormat::Gguf => run_batch_gguf(config, reader, writer),
        ModelFormat::Apr => run_batch_apr(config, reader, writer),
        other => Err(RealizarError::FormatError {
            reason: format!("Batch inference not supported for {:?} format", other),
        }),
    }
}

/// Batch inference for GGUF models.
fn run_batch_gguf<R, W>(
    config: &BatchInferenceConfig,
    reader: R,
    mut writer: W,
) -> Result<BatchStats>
where
    R: std::io::BufRead,
    W: std::io::Write,
{
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    let load_start = Instant::now();
    if config.verbose {
        eprintln!("[batch] Loading GGUF model: {}", config.model_path.display());
    }

    let mapped = MappedGGUFModel::from_path(&config.model_path)?;
    prefault_mmap(mapped.data());
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let mut stop_tokens: Vec<u32> = model.config.eos_token_id.into_iter().collect();
    for &t in &config.stop_tokens {
        if !stop_tokens.contains(&t) {
            stop_tokens.push(t);
        }
    }

    let num_layers = model.config.num_layers;
    let vocab_size = model.config.vocab_size;
    let batch_model = init_batch_model(model, &stop_tokens, config)?;

    // If no backend has the model, reload for CPU
    let has_gpu_gguf = {
        let mut has = false;
        #[cfg(feature = "cuda")]
        { has = has || batch_model.gpu.is_some(); }
        #[cfg(feature = "gpu")]
        { has = has || batch_model.wgpu.is_some(); }
        has
    };
    let batch_model = if batch_model.cpu.is_none() && !has_gpu_gguf {
        let m = OwnedQuantizedModel::from_mapped(&mapped)?;
        BatchModel {
            #[cfg(feature = "cuda")] gpu: None,
            #[cfg(feature = "gpu")] wgpu: None,
            cpu: Some(m),
        }
    } else {
        batch_model
    };

    let model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    if config.verbose {
        let gguf_arch = mapped.model.architecture().unwrap_or("transformer");
        eprintln!(
            "[batch] Model loaded in {:.1}ms (arch={}, layers={}, vocab={})",
            model_load_ms, gguf_arch, num_layers, vocab_size,
        );
    }

    let encode = |text: &str| -> Option<Vec<u32>> { mapped.model.encode(text) };
    let decode = |tokens: &[u32]| -> String { mapped.model.decode(tokens) };

    run_batch_loop(config, reader, &mut writer, &stop_tokens, model_load_ms, batch_model, &encode, &decode)
}

/// Batch inference for APR models.
fn run_batch_apr<R, W>(
    config: &BatchInferenceConfig,
    reader: R,
    mut writer: W,
) -> Result<BatchStats>
where
    R: std::io::BufRead,
    W: std::io::Write,
{
    use crate::apr::MappedAprModel;
    use crate::gguf::OwnedQuantizedModel;

    let load_start = Instant::now();
    if config.verbose {
        eprintln!("[batch] Loading APR model: {}", config.model_path.display());
    }

    let mapped_apr = MappedAprModel::from_path(&config.model_path)?;
    let model = OwnedQuantizedModel::from_apr(&mapped_apr)?;

    let stop_tokens = resolve_apr_stop_tokens(
        model.config.eos_token_id,
        &config.stop_tokens,
        &config.model_path,
    );

    let num_layers = model.config.num_layers;
    let vocab_size = model.config.vocab_size;
    let arch = model.config.architecture.clone();
    let batch_model = init_batch_model(model, &stop_tokens, config)?;

    // If no backend has the model, reload for CPU
    let has_gpu_backend = {
        let mut has = false;
        #[cfg(feature = "cuda")]
        { has = has || batch_model.gpu.is_some(); }
        #[cfg(feature = "gpu")]
        { has = has || batch_model.wgpu.is_some(); }
        has
    };
    let batch_model = if batch_model.cpu.is_none() && !has_gpu_backend {
        let m = OwnedQuantizedModel::from_apr(&mapped_apr)?;
        BatchModel {
            #[cfg(feature = "cuda")] gpu: None,
            #[cfg(feature = "gpu")] wgpu: None,
            cpu: Some(m),
        }
    } else {
        batch_model
    };

    let model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    if config.verbose {
        eprintln!(
            "[batch] Model loaded in {:.1}ms (arch={}, layers={}, vocab={})",
            model_load_ms, arch, num_layers, vocab_size,
        );
    }

    let model_path = config.model_path.clone();
    let encode = |text: &str| -> Option<Vec<u32>> {
        crate::apr::AprV2Model::encode_text(&model_path, text)
    };
    let model_path2 = config.model_path.clone();
    let decode = |tokens: &[u32]| -> String { decode_apr_tokens(&model_path2, tokens) };

    run_batch_loop(config, reader, &mut writer, &stop_tokens, model_load_ms, batch_model, &encode, &decode)
}


/// Initialize batch model with GPU/CPU fallback.
fn init_batch_model(
    model: crate::gguf::OwnedQuantizedModel,
    stop_tokens: &[u32],
    config: &BatchInferenceConfig,
) -> Result<BatchModel> {
    // GH-560 FIXED (2026-03-28): wgpu batch works after FFN buffer + KV cache fix.
    // Try wgpu before CUDA to avoid CUDA JIT overhead + parity gate on sm_121.
    // Disable with WGPU_BATCH=0 to skip wgpu and go straight to CUDA.
    #[cfg(feature = "gpu")]
    if !config.no_gpu && !model_has_legacy_quant(&model)
        && std::env::var("WGPU_BATCH").map(|v| v != "0").unwrap_or(true)
    {
        if let Some(wgpu_state) = try_init_wgpu_batch(&model, config) {
            return Ok(BatchModel {
                #[cfg(feature = "cuda")]
                gpu: None,
                wgpu: Some(wgpu_state),
                cpu: None,
            });
        }
    }

    #[cfg(feature = "cuda")]
    {
        if !config.no_gpu && !model_has_legacy_quant(&model) {
            use crate::gguf::{OwnedQuantizedModelCuda, QuantizedGenerateConfig};
            match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048) {
                Ok(mut cuda_model) => {
                    if config.verbose {
                        eprintln!(
                            "[batch] GPU: {} ({} MB VRAM)",
                            cuda_model.device_name(), cuda_model.vram_mb()
                        );
                    }
                    let probe_config = QuantizedGenerateConfig {
                        max_tokens: 1, temperature: 0.0, top_k: 1,
                        stop_tokens: stop_tokens.to_vec(), trace: false,
            ..Default::default()
                    };
                    if validate_gpu_first_token(&mut cuda_model, &probe_config) {
                        return Ok(BatchModel { gpu: Some(cuda_model), #[cfg(feature = "gpu")] wgpu: None, cpu: None });
                    }
                    eprintln!("[batch] CUDA validation failed, trying wgpu...");
                    let model = cuda_model.into_model();
                    // GH-560 FIXED: wgpu batch works. Try wgpu before CPU fallback.
                    #[cfg(feature = "gpu")]
                    if let Some(wgpu_state) = try_init_wgpu_batch(&model, config) {
                        return Ok(BatchModel { gpu: None, wgpu: Some(wgpu_state), cpu: None });
                    }
                    return Ok(BatchModel { gpu: None, #[cfg(feature = "gpu")] wgpu: None, cpu: Some(model) });
                }
                Err(e) => {
                    if config.verbose {
                        eprintln!("[batch] CUDA unavailable: {}, trying wgpu...", e);
                    }
                    let model = e.into_model();
                    // GH-560 FIXED: wgpu batch works. Try wgpu before CPU fallback.
                    #[cfg(feature = "gpu")]
                    if let Some(wgpu_state) = try_init_wgpu_batch(&model, config) {
                        return Ok(BatchModel { gpu: None, wgpu: Some(wgpu_state), cpu: None });
                    }
                    return Ok(BatchModel { gpu: None, #[cfg(feature = "gpu")] wgpu: None, cpu: Some(model) });
                }
            }
        }

        // No GPU attempted — use CPU directly
        // GH-560: Try wgpu before CPU
        #[cfg(feature = "gpu")]
        if !config.no_gpu {
            if let Some(wgpu_state) = try_init_wgpu_batch(&model, config) {
                return Ok(BatchModel {
                    #[cfg(feature = "cuda")]
                    gpu: None,
                    #[cfg(feature = "gpu")]
                    wgpu: Some(wgpu_state),
                    cpu: None,
                });
            }
        }

        Ok(BatchModel {
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "gpu")]
            wgpu: None,
            cpu: Some(model),
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stop_tokens, config);
        Ok(BatchModel { #[cfg(feature = "gpu")] wgpu: None, cpu: Some(model) })
    }
}

/// Core batch processing loop.
#[allow(clippy::too_many_arguments)]
fn run_batch_loop<R, W, E, D>(
    config: &BatchInferenceConfig,
    reader: R,
    writer: &mut W,
    stop_tokens: &[u32],
    model_load_ms: f64,
    mut batch_model: BatchModel,
    encode: &E,
    decode: &D,
) -> Result<BatchStats>
where
    R: std::io::BufRead,
    W: std::io::Write,
    E: Fn(&str) -> Option<Vec<u32>>,
    D: Fn(&[u32]) -> String,
{
    use crate::gguf::QuantizedGenerateConfig;
    use std::io::Write as _;

    let mut stats = BatchStats {
        total_prompts: 0, successful: 0, failed: 0,
        total_tokens_generated: 0, total_inference_ms: 0.0, model_load_ms,
    };

    for line in reader.lines() {
        let line = match line {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(e) => { eprintln!("[batch] Error reading input: {}", e); break; }
        };

        stats.total_prompts += 1;
        let prompt_idx = stats.total_prompts;

        let batch_prompt: BatchPrompt = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(e) => {
                let result = BatchResult {
                    task_id: None, text: String::new(), tokens_generated: 0,
                    tok_per_sec: 0.0, inference_ms: 0.0, used_gpu: false,
                    error: Some(format!("JSON parse error: {}", e)),
                };
                stats.failed += 1;
                writeln!(writer, "{}", serde_json::to_string(&result).unwrap_or_default())
                    .map_err(|e| RealizarError::IoError { message: format!("Write error: {}", e) })?;
                continue;
            }
        };

        let max_tokens = batch_prompt.max_tokens.unwrap_or(config.max_tokens);

        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            batch_prompt.prompt
        );
        let input_tokens = match encode(&chat_prompt) {
            Some(tokens) => tokens,
            None => {
                let result = BatchResult {
                    task_id: batch_prompt.task_id, text: String::new(), tokens_generated: 0,
                    tok_per_sec: 0.0, inference_ms: 0.0, used_gpu: false,
                    error: Some("Tokenizer encode failed".to_string()),
                };
                stats.failed += 1;
                writeln!(writer, "{}", serde_json::to_string(&result).unwrap_or_default())
                    .map_err(|e| RealizarError::IoError { message: format!("Write error: {}", e) })?;
                continue;
            }
        };
        let input_token_count = input_tokens.len();

        let gen_config = QuantizedGenerateConfig {
            max_tokens, temperature: config.temperature, top_k: config.top_k,
            stop_tokens: stop_tokens.to_vec(), trace: false,
            ..Default::default()
        };

        let infer_start = Instant::now();
        let gen_result = batch_model.generate(&input_tokens, &gen_config);
        let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

        let result = match gen_result {
            Ok((tokens, used_gpu)) => {
                let generated = &tokens[input_token_count..];
                let raw_text = decode(generated);
                let text = clean_model_output(&raw_text);
                let generated_count = generated.len();
                let tps = tok_per_sec(generated_count, inference_ms);

                stats.successful += 1;
                stats.total_tokens_generated += generated_count;
                stats.total_inference_ms += inference_ms;

                BatchResult {
                    task_id: batch_prompt.task_id, text,
                    tokens_generated: generated_count,
                    tok_per_sec: (tps * 10.0).round() / 10.0,
                    inference_ms: (inference_ms * 100.0).round() / 100.0,
                    used_gpu, error: None,
                }
            }
            Err(e) => {
                stats.failed += 1;
                BatchResult {
                    task_id: batch_prompt.task_id, text: String::new(),
                    tokens_generated: 0, tok_per_sec: 0.0,
                    inference_ms: (inference_ms * 100.0).round() / 100.0,
                    used_gpu: false, error: Some(format!("{}", e)),
                }
            }
        };

        writeln!(writer, "{}", serde_json::to_string(&result).unwrap_or_default())
            .map_err(|e| RealizarError::IoError { message: format!("Write error: {}", e) })?;
        writer.flush()
            .map_err(|e| RealizarError::IoError { message: format!("Flush error: {}", e) })?;

        if prompt_idx % 10 == 0 {
            eprintln!("[batch] {prompt_idx}/? processed ({} ok, {} failed)",
                stats.successful, stats.failed);
        }
    }

    eprintln!(
        "[batch] Complete: {} prompts, {} ok, {} failed, {:.1}s total inference, {:.1}s model load",
        stats.total_prompts, stats.successful, stats.failed,
        stats.total_inference_ms / 1000.0, stats.model_load_ms / 1000.0
    );

    Ok(stats)
}
