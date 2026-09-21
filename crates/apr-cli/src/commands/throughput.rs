/// Measure throughput for a SafeTensors model.
#[cfg(feature = "inference")]
fn throughput_safetensors(
    path: &Path,
    config: &QaConfig,
    tracer: &TracerImpl,
    prompt: &str,
) -> Result<Option<(f64, Duration)>> {
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    // #3742: the tokenizer.json's own pre-tokenizer + ranked merges (realizar's canonical
    // byte-level BPE); aprender-core's whitespace-splitting BPE refuses these vocabularies.
    let tokenizer_path = realizar::safetensors::find_sibling_file(path, "tokenizer.json");
    let tokenizer = tokenizer_path
        .as_ref()
        .and_then(|p| crate::commands::hf_tokenizer::HfTokenizer::from_file(p).ok());

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
    let budget_us = config.max_tokens as u64 * config.iterations as u64 * 100_000;

    Ok(Some(measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        tracer,
        "qa_throughput_safetensors",
        budget_us,
        config.verbose,
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
    tracer: &TracerImpl,
) -> Result<Option<(f64, Duration)>> {
    use realizar::format::ModelFormat;

    match format {
        ModelFormat::Gguf => {
            throughput_gguf(path, model_bytes, config, cuda_available, tracer, prompt).map(Some)
        }
        ModelFormat::Apr => throughput_apr(path, config, tracer, prompt).map(Some),
        ModelFormat::SafeTensors => throughput_safetensors(path, config, tracer, prompt),
    }
}
