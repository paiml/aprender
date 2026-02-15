
/// Run golden output test for SafeTensors format models. Returns None if tokenizer missing.
#[cfg(feature = "inference")]
fn golden_output_safetensors(
    path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> Result<Option<(Vec<u32>, String)>> {
    use aprender::text::bpe::{load_from_json, BpeTokenizer};
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    let tokenizer_path = realizar::safetensors::find_sibling_file(path, "tokenizer.json");
    let tokenizer: Option<BpeTokenizer> = tokenizer_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|json| load_from_json(&json).ok());

    let Some(tokenizer) = tokenizer else {
        return Ok(None);
    };

    let transformer = SafetensorsToAprConverter::convert(path)
        .map_err(|e| CliError::ValidationFailed(format!("SafeTensors convert failed: {e}")))?;

    let prompt_tokens = tokenizer.encode(prompt);
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    let tokens = transformer
        .generate_with_cache(&prompt_tokens, &gen_config)
        .map_err(|e| CliError::ValidationFailed(format!("Generation failed: {e}")))?;
    let text = tokenizer.decode(&tokens);
    Ok(Some((tokens, text)))
}

/// Run golden output CPU generation for GGUF format.
#[cfg(feature = "inference")]
fn golden_output_gguf_cpu(
    mapped: &realizar::gguf::MappedGGUFModel,
    gguf: &realizar::gguf::GGUFModel,
    prompt: &str,
    max_tokens: usize,
) -> Result<(Vec<u32>, String)> {
    use realizar::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};

    let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![151643, 9707]);
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let model = OwnedQuantizedModel::from_mapped(mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
    let tokens = model
        .generate_with_cache(&prompt_tokens, &gen_config)
        .map_err(|e| CliError::ValidationFailed(format!("CPU generation failed: {e}")))?;
    let text = gguf.decode(&tokens);
    Ok((tokens, text))
}

/// Gate 1: Golden Output Test
///
/// Runs the model with a known prompt and verifies the output contains expected patterns.
/// Uses verify_output() for structured validation (PMAT-QA-PROTOCOL-001 §7.4).
/// Golden test cases: ChatML prompt + expected output patterns.
fn golden_test_cases() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
            vec!["4"],
        ),
        (
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            vec!["Hello", "Hi", "hey", "hello", "!"],
        ),
    ]
}

/// Generate output for a single test case based on model format.
/// GH-239: GGUF objects are Optional — only provided when format is GGUF.
#[cfg(feature = "inference")]
fn generate_golden_for_format(
    path: &Path,
    prompt: &str,
    max_tokens: usize,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
) -> Result<Option<(Vec<u32>, String)>> {
    use realizar::format::ModelFormat;

    match format {
        ModelFormat::Gguf => {
            let mapped = mapped.ok_or_else(|| {
                CliError::ValidationFailed("GGUF mapped model required".to_string())
            })?;
            let gguf_model = gguf_model
                .ok_or_else(|| CliError::ValidationFailed("GGUF model required".to_string()))?;
            Ok(Some(golden_output_gguf_cpu(
                mapped, gguf_model, prompt, max_tokens,
            )?))
        }
        ModelFormat::Apr => Ok(Some(golden_output_apr(path, prompt, max_tokens)?)),
        ModelFormat::SafeTensors => golden_output_safetensors(path, prompt, max_tokens),
    }
}

/// Validate a single golden test case: generate output, check GPU parity, verify patterns.
///
/// Returns `Ok(None)` on success, `Ok(Some(GateResult))` on failure/skip.
#[cfg(feature = "inference")]
fn validate_golden_test_case(
    path: &Path,
    prompt: &str,
    expected_patterns: &[&str],
    config: &QaConfig,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
    cuda_available: bool,
    start: Instant,
) -> Result<Option<GateResult>> {
    use realizar::format::ModelFormat;

    let Some((_, output_text)) =
        generate_golden_for_format(path, prompt, config.max_tokens, format, mapped, gguf_model)?
    else {
        return Ok(Some(GateResult::skipped(
            "golden_output",
            "SafeTensors: tokenizer.json not found",
        )));
    };

    #[cfg(feature = "cuda")]
    if cuda_available && format == ModelFormat::Gguf {
        use realizar::gguf::QuantizedGenerateConfig;
        // Safe: format==Gguf guarantees these are Some
        let gguf_ref = gguf_model.expect("GGUF model required for GPU golden output");
        let mapped_ref = mapped.expect("GGUF mapped model required for GPU golden output");
        let prompt_tokens = gguf_ref
            .encode(prompt)
            .unwrap_or_else(|| vec![151643, 9707]);
        let gen_config = QuantizedGenerateConfig {
            max_tokens: config.max_tokens,
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        };
        if let Some(failure) = validate_gpu_golden_output(
            mapped_ref,
            &prompt_tokens,
            &gen_config,
            gguf_ref,
            expected_patterns,
            config,
        )? {
            return Ok(Some(GateResult::failed(
                "golden_output",
                &failure,
                None,
                None,
                start.elapsed(),
            )));
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = cuda_available;

    if let OutputVerification::Fail { reason } =
        verify_output(&output_text, "golden_output", expected_patterns)
    {
        return Ok(Some(GateResult::failed(
            "golden_output",
            &reason,
            None,
            None,
            start.elapsed(),
        )));
    }

    Ok(None)
}

fn run_golden_output_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running golden output test...".yellow());
    }

    let test_cases = golden_test_cases();

    #[cfg(feature = "inference")]
    {
        use realizar::cuda::CudaExecutor;
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{GGUFModel, MappedGGUFModel};

        let cuda_available = CudaExecutor::is_available() && CudaExecutor::num_devices() > 0;
        let model_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
        let format = detect_format(&model_bytes[..8.min(model_bytes.len())])
            .map_err(|e| CliError::ValidationFailed(format!("Failed to detect format: {e}")))?;

        // GH-239: Only create GGUF objects when format is actually GGUF
        let (mapped, gguf_model) = if format == ModelFormat::Gguf {
            let m = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
            let g = GGUFModel::from_bytes(&model_bytes)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;
            (Some(m), Some(g))
        } else {
            (None, None)
        };

        for (prompt, expected_patterns) in &test_cases {
            if let Some(result) = validate_golden_test_case(
                path,
                prompt,
                expected_patterns,
                config,
                format,
                mapped.as_ref(),
                gguf_model.as_ref(),
                cuda_available,
                start,
            )? {
                return Ok(result);
            }
        }

        Ok(GateResult::passed(
            "golden_output",
            &format!("{} golden test cases passed", test_cases.len()),
            Some(test_cases.len() as f64),
            Some(test_cases.len() as f64),
            start.elapsed(),
        ))
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config, test_cases);
        Ok(GateResult::skipped(
            "golden_output",
            "Requires 'inference' feature",
        ))
    }
}

/// Run warmup+measure loop for throughput benchmarking.
///
/// Calls `generate_fn` for `warmup` iterations (discarding results), then
/// measures `iterations` runs to compute tokens per second.
#[cfg(feature = "inference")]
fn measure_generate_throughput(
    warmup: usize,
    iterations: usize,
    prompt_len: usize,
    overall_start: Instant,
    mut generate_fn: impl FnMut() -> Vec<u32>,
) -> (f64, Duration) {
    for _ in 0..warmup {
        let _ = generate_fn();
    }

    let mut total_tokens = 0usize;
    let measure_start = Instant::now();
    for _ in 0..iterations {
        let output = generate_fn();
        total_tokens += output.len().saturating_sub(prompt_len);
    }
    let measure_time = measure_start.elapsed();
    (
        total_tokens as f64 / measure_time.as_secs_f64(),
        overall_start.elapsed(),
    )
}

/// Measure throughput for a GGUF model (GPU or CPU path).
#[cfg(feature = "inference")]
fn throughput_gguf(
    path: &Path,
    model_bytes: &[u8],
    config: &QaConfig,
    cuda_available: bool,
    start: Instant,
    prompt: &str,
) -> Result<(f64, Duration)> {
    use realizar::gguf::{
        GGUFModel, MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
        QuantizedGenerateConfig,
    };

    let gguf = GGUFModel::from_bytes(model_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;
    let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![151643, 9707]);
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;

    if cuda_available {
        let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
            .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;
        Ok(measure_generate_throughput(
            config.warmup,
            config.iterations,
            prompt_tokens.len(),
            start,
            || {
                cuda_model
                    .generate_gpu_resident(&prompt_tokens, &gen_config)
                    .unwrap_or_default()
            },
        ))
    } else {
        Ok(measure_generate_throughput(
            config.warmup,
            config.iterations,
            prompt_tokens.len(),
            start,
            || {
                model
                    .generate_with_cache(&prompt_tokens, &gen_config)
                    .unwrap_or_default()
            },
        ))
    }
}

/// Measure throughput for an APR model.
#[cfg(feature = "inference")]
fn throughput_apr(
    path: &Path,
    config: &QaConfig,
    start: Instant,
    prompt: &str,
) -> Result<(f64, Duration)> {
    use realizar::apr::AprV2Model;
    use realizar::apr_transformer::{AprTransformer, GenerateConfig};

    let apr_model = AprV2Model::load(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;
    let tokenizer = apr_model
        .load_embedded_bpe_tokenizer()
        .ok_or_else(|| CliError::ValidationFailed("APR missing embedded tokenizer".to_string()))?;
    let transformer = AprTransformer::from_apr_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR transformer: {e}")))?;

    let prompt_tokens = tokenizer.encode(prompt);
    let gen_config = GenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    Ok(measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        start,
        || {
            transformer
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    ))
}

/// Measure throughput for a SafeTensors model.
#[cfg(feature = "inference")]
fn throughput_safetensors(
    path: &Path,
    config: &QaConfig,
    start: Instant,
    prompt: &str,
) -> Result<Option<(f64, Duration)>> {
    use aprender::text::bpe::{load_from_json, BpeTokenizer};
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    let tokenizer_path = realizar::safetensors::find_sibling_file(path, "tokenizer.json");
    let tokenizer: Option<BpeTokenizer> = tokenizer_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|json| load_from_json(&json).ok());

    let Some(tokenizer) = tokenizer else {
        return Ok(None);
    };

    let transformer = SafetensorsToAprConverter::convert(path)
        .map_err(|e| CliError::ValidationFailed(format!("SafeTensors convert failed: {e}")))?;

    let prompt_tokens = tokenizer.encode(prompt);
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    Ok(Some(measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        start,
        || {
            transformer
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    )))
}

/// Dispatch throughput measurement to the correct format handler.
#[cfg(feature = "inference")]
fn throughput_for_format(
    path: &Path,
    model_bytes: &[u8],
    format: realizar::format::ModelFormat,
    prompt: &str,
    config: &QaConfig,
    cuda_available: bool,
    start: Instant,
) -> Result<Option<(f64, Duration)>> {
    use realizar::format::ModelFormat;

    match format {
        ModelFormat::Gguf => {
            throughput_gguf(path, model_bytes, config, cuda_available, start, prompt).map(Some)
        }
        ModelFormat::Apr => throughput_apr(path, config, start, prompt).map(Some),
        ModelFormat::SafeTensors => throughput_safetensors(path, config, start, prompt),
    }
}
