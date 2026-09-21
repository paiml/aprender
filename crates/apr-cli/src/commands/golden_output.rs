
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

    let specials = aprender::demo::SpecialTokens::qwen2();
    let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![specials.bos_id, 9707]);
    // #1864: without stop_tokens, the model runs for the full max_tokens
    // budget and starts emitting in-distribution chat-template tokens like
    // `<|im_start|>` from accumulated drift — which `verify_output` then
    // (correctly) flags as gibberish. The actual underlying inference path
    // (`apr serve` /v1/chat/completions) sets stop_tokens to EOS — so user
    // traffic was never affected. This aligns the gate's gen_config with the
    // production path. See `cuda_chat_backend.rs:113` for the symmetric setup.
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![specials.eos_id],
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
///
/// SELECTION RULE (#2350): a golden prompt must have a WIDE argmax margin, so its
/// greedy continuation is the same on every backend. These assert on exact
/// generated content under `temperature 0.0 / top_k 1`, which turns any near-tie
/// into a coin flip — and CPU and CUDA kernels legitimately differ by small
/// numerics well inside tolerance (F2 measures full prefill parity at cosine
/// 0.9937 with zero argmax mismatches on this model).
///
/// The removed case was `"Hello"` alone, expecting a greeting back. Measured on
/// one binary, one model, `max_tokens 512`, greedy:
///
///   prompt                                   CPU                        CUDA
///   ---------------------------------------  -------------------------  -------------------------
///   "Hello"                                  "Hello! How can I ..."     "I'm sorry, but I'm not
///                                                                        sure what you're asking"
///   "Hi"                                     "I'm sorry, but I'm not    "I'm here to help! ..."
///                                             sure what you're asking"
///   "What is 2+2?"                           "2+2 equals 4."            "2 + 2 equals 4."
///   "Hello there, how are you doing today    "Hello! I'm doing well,    same
///    my friend?"                              thank you."
///   "What is the capital of France?"         "The capital of France     same
///                                             is Paris."
///
/// The decisive row is the second: the SAME evasive completion appears on **CPU**,
/// just for `"Hi"` rather than `"Hello"`. Both backends answer bare one-word
/// greetings evasively; they only disagree about which one tips over. So the old
/// case was not detecting a GPU defect — it was sampling a knife-edge, and it
/// flipped when #2323 made the CUDA path reachable on sm_89.
///
/// The three cases below were all verified to produce identical continuations on
/// CPU and CUDA. Keep it that way: if you add a case, run it on both backends
/// first (`crates/apr-cli/tests/golden_prompt_tokenization.rs` is the harness).
fn golden_test_cases() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
            vec!["4"],
        ),
        // Replaces the bare "Hello". Still exercises a conversational turn, but
        // with enough context that the first generated token is not a near-tie.
        (
            "<|im_start|>user\nHello there, how are you doing today my friend?<|im_end|>\n<|im_start|>assistant\n",
            vec!["Hello", "Hi", "hey", "hello", "well", "!"],
        ),
        // Factual recall: a wide-margin argmax and a check that the model is
        // actually reasoning over its weights rather than emitting boilerplate.
        (
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
            vec!["Paris"],
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

/// Is there a CUDA device this build can use? Both golden rungs ask (#3711).
#[cfg(feature = "cuda")]
fn cuda_device_present() -> bool {
    use realizar::cuda::CudaExecutor;
    CudaExecutor::is_available() && CudaExecutor::num_devices() > 0
}

/// Without the cuda feature there is no device this build can use.
#[cfg(not(feature = "cuda"))]
fn cuda_device_present() -> bool {
    false
}

/// One golden case, judged (#3711).
#[cfg(feature = "inference")]
enum GoldenCaseOutcome {
    /// The CPU answer is right, and so is the GPU's when its leg ran; the leg says which.
    Passed(GpuGoldenLeg),
    /// The gate's verdict for this case: a failure, or the tokenizer skip.
    Verdict(GateResult),
}

/// The GPU half of one golden case (#3711): run and judged when a cuda build has a device
/// and the model is a GGUF, else `NotRun` naming which of those it lacks.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn gpu_golden_leg(
    prompt: &str,
    expected_patterns: &[&str],
    config: &QaConfig,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
    cuda_available: bool,
    golden_max_tokens: usize,
) -> Result<GpuGoldenLeg> {
    use realizar::format::ModelFormat;
    use realizar::gguf::QuantizedGenerateConfig;

    if let Some(why) = gpu_golden_not_run(true, cuda_available, format == ModelFormat::Gguf) {
        return Ok(GpuGoldenLeg::NotRun(why));
    }
    // Safe: format==Gguf guarantees these are Some
    let gguf_ref = gguf_model.expect("GGUF model required for GPU golden output");
    let mapped_ref = mapped.expect("GGUF mapped model required for GPU golden output");
    let specials = aprender::demo::SpecialTokens::qwen2();
    let prompt_tokens = gguf_ref
        .encode(prompt)
        .unwrap_or_else(|| vec![specials.bos_id, 9707]);
    // #1864: GPU path mirrors the CPU gate's fix above — set stop_tokens
    // to EOS so generation terminates at end-of-turn rather than running
    // the full 512-token budget and drifting into `<|im_start|>` repeats.
    let gen_config = QuantizedGenerateConfig {
        max_tokens: golden_max_tokens, // GH-279-4: match CPU budget
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![specials.eos_id],
        ..Default::default()
    };
    validate_gpu_golden_output(
        mapped_ref,
        &prompt_tokens,
        &gen_config,
        gguf_ref,
        expected_patterns,
        config,
    )
}

/// Without the cuda feature the GPU leg never starts, and says so.
#[cfg(all(feature = "inference", not(feature = "cuda")))]
fn gpu_golden_leg(
    prompt: &str,
    expected_patterns: &[&str],
    config: &QaConfig,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
    cuda_available: bool,
    golden_max_tokens: usize,
) -> Result<GpuGoldenLeg> {
    let _ = (prompt, expected_patterns, config, mapped, gguf_model, golden_max_tokens);
    let gguf = format == realizar::format::ModelFormat::Gguf;
    Ok(GpuGoldenLeg::NotRun(
        gpu_golden_not_run(false, cuda_available, gguf).unwrap_or("this build has no cuda feature"),
    ))
}

/// Validate a single golden test case: generate output, check GPU parity, verify patterns.
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
) -> Result<GoldenCaseOutcome> {
    // GH-279-4: Thinking models (Qwen3) need extra tokens for <think>...</think>
    // chain-of-thought before the answer. 32 tokens is not enough — the model
    // exhausts the budget on reasoning and never emits the answer. Qwen3's
    // thinking can be verbose (~100-200 tokens for simple math), so 512 gives
    // ample room for reasoning + answer.
    let golden_max_tokens = config.max_tokens.max(512);

    let Some((_, output_text)) =
        generate_golden_for_format(path, prompt, golden_max_tokens, format, mapped, gguf_model)?
    else {
        return Ok(GoldenCaseOutcome::Verdict(GateResult::skipped(
            "golden_output",
            "SafeTensors: tokenizer.json not found",
        )));
    };

    // #3711: a GPU leg that errored is a FAIL naming the error, never a skip.
    let gpu_leg = gpu_golden_leg(
        prompt,
        expected_patterns,
        config,
        format,
        mapped,
        gguf_model,
        cuda_available,
        golden_max_tokens,
    )?;
    if let Some(failure) = gpu_leg.failure() {
        return Ok(GoldenCaseOutcome::Verdict(GateResult::failed(
            "golden_output",
            &failure,
            None,
            None,
            start.elapsed(),
        )));
    }

    // GH-279-4: generate_with_cache returns prompt + generated tokens.
    // Strip the prompt echo so we verify only the model's generated output.
    let generated_text = output_text
        .strip_prefix(prompt)
        .unwrap_or(&output_text);
    let answer_text = strip_thinking_blocks(generated_text); // GH-279-4
    if let OutputVerification::Fail { reason } =
        verify_output(&answer_text, "golden_output", expected_patterns)
    {
        return Ok(GoldenCaseOutcome::Verdict(GateResult::failed(
            "golden_output",
            &reason,
            None,
            None,
            start.elapsed(),
        )));
    }

    Ok(GoldenCaseOutcome::Passed(gpu_leg))
}

fn run_golden_output_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running golden output test...".yellow());
    }

    let test_cases = golden_test_cases();

    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{GGUFModel, MappedGGUFModel};

        let cuda_available = cuda_device_present();
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

        // #3711: the pass message names the GPU leg, so a skipped leg says which skip it was
        let mut gpu_leg = GpuGoldenLeg::NotRun("no golden case ran");
        for (prompt, expected_patterns) in &test_cases {
            match validate_golden_test_case(
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
                GoldenCaseOutcome::Verdict(result) => return Ok(result),
                GoldenCaseOutcome::Passed(leg) => gpu_leg = leg,
            }
        }

        Ok(GateResult::passed(
            "golden_output",
            &format!(
                "{} golden test cases passed ({})",
                test_cases.len(),
                gpu_leg.describe()
            ),
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
/// measures `iterations` runs using `BrickTracer` for syscall-level diagnostics.
/// Returns (tokens_per_second, measurement_duration).
#[cfg(feature = "inference")]
fn measure_generate_throughput(
    warmup: usize,
    iterations: usize,
    prompt_len: usize,
    tracer: &TracerImpl,
    brick_name: &str,
    budget_us: u64,
    verbose: bool,
    mut generate_fn: impl FnMut() -> Vec<u32>,
) -> (f64, Duration) {
    // Warmup (untraced)
    for _ in 0..warmup {
        let _ = generate_fn();
    }

    // Measurement (traced via BrickTracer for syscall breakdown)
    let traced = tracer.trace(brick_name, budget_us, || {
        let mut tokens = 0usize;
        for _ in 0..iterations {
            let output = generate_fn();
            tokens += output.len().saturating_sub(prompt_len);
        }
        tokens
    });
    let total_tokens = traced.result;
    let measure_secs = traced.duration_us as f64 / 1_000_000.0;
    let tps = if measure_secs > 0.0 {
        total_tokens as f64 / measure_secs
    } else {
        0.0
    };

    if verbose {
        let bd = &traced.syscall_breakdown;
        eprintln!(
            "  BrickTracer [{brick_name}]: {:.1} tok/s, {}us total",
            tps, traced.duration_us
        );
        eprintln!(
            "    compute: {}us  mmap: {}us  futex: {}us  ioctl: {}us",
            bd.compute_us, bd.mmap_us, bd.futex_us, bd.ioctl_us
        );
        eprintln!(
            "    overhead: {:.1}%  dominant: {}",
            bd.syscall_overhead_percent(),
            bd.dominant_syscall()
        );
        if let Some(ref meta) = traced.metadata {
            eprintln!(
                "    budget: {}us  actual: {}us  efficiency: {:.1}%",
                meta.budget_us,
                meta.actual_us,
                meta.efficiency * 100.0
            );
        }
    }

    let duration = Duration::from_micros(traced.duration_us);
    (tps, duration)
}

/// Measure throughput for a GGUF model (GPU or CPU path).
#[cfg(feature = "inference")]
fn throughput_gguf(
    path: &Path,
    model_bytes: &[u8],
    config: &QaConfig,
    cuda_available: bool,
    tracer: &TracerImpl,
    prompt: &str,
) -> Result<(f64, Duration)> {
    use realizar::gguf::{
        GGUFModel, MappedGGUFModel, OwnedQuantizedModel,
        QuantizedGenerateConfig,
    };

    let gguf = GGUFModel::from_bytes(model_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;
    let bos = aprender::demo::SpecialTokens::qwen2().bos_id;
    let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![bos, 9707]);
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let budget_us = config.max_tokens as u64 * config.iterations as u64 * 100_000;

    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;

    // GH-284: Try CUDA, fall back to CPU on capability mismatch (e.g. missing QkNorm kernel)
    #[cfg(feature = "cuda")]
    if cuda_available {
        use realizar::gguf::OwnedQuantizedModelCuda;
        match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048) {
            Ok(mut cuda_model) => {
                return Ok(measure_generate_throughput(
                    config.warmup,
                    config.iterations,
                    prompt_tokens.len(),
                    tracer,
                    "qa_throughput_gguf_gpu",
                    budget_us,
                    config.verbose,
                    || {
                        cuda_model
                            .generate_gpu_resident(&prompt_tokens, &gen_config)
                            .unwrap_or_default()
                    },
                ));
            }
            Err(e) => {
                let model = e.into_model();
                return Ok(measure_generate_throughput(
                    config.warmup,
                    config.iterations,
                    prompt_tokens.len(),
                    tracer,
                    "qa_throughput_gguf_cpu_fallback",
                    budget_us,
                    config.verbose,
                    || {
                        model
                            .generate_with_cache(&prompt_tokens, &gen_config)
                            .unwrap_or_default()
                    },
                ));
            }
        }
    }
    Ok(measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        tracer,
        "qa_throughput_gguf_cpu",
        budget_us,
        config.verbose,
        || {
            model
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    ))
}

/// Measure throughput for an APR model.
#[cfg(feature = "inference")]
fn throughput_apr(
    path: &Path,
    config: &QaConfig,
    tracer: &TracerImpl,
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
    let budget_us = config.max_tokens as u64 * config.iterations as u64 * 100_000;

    Ok(measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        tracer,
        "qa_throughput_apr",
        budget_us,
        config.verbose,
        || {
            transformer
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    ))
}

// =============================================================================
// PMAT-125 B4: golden_test_cases fixture coverage
// =============================================================================

#[cfg(test)]
mod golden_output_tests {
    use super::*;

    #[test]
    fn test_golden_test_cases_nonempty_and_structured() {
        let cases = golden_test_cases();
        assert!(!cases.is_empty());
        for (prompt, patterns) in &cases {
            assert!(prompt.contains("<|im_start|>assistant"));
            assert!(prompt.contains("<|im_start|>user"));
            assert!(!patterns.is_empty());
        }
    }

    #[test]
    fn test_golden_test_cases_arithmetic_case_present() {
        let cases = golden_test_cases();
        let arith = cases
            .iter()
            .find(|(p, _)| p.contains("2+2"))
            .expect("arithmetic golden case must exist");
        assert!(arith.1.contains(&"4"));
    }

    #[test]
    fn test_golden_test_cases_deterministic() {
        let a = golden_test_cases();
        let b = golden_test_cases();
        assert_eq!(a.len(), b.len());
        for ((pa, _), (pb, _)) in a.iter().zip(b.iter()) {
            assert_eq!(pa, pb);
        }
    }

    /// Poka-yoke for #2350: no golden prompt may be a bare one-or-two-word user
    /// message.
    ///
    /// These cases assert on exact greedy-decoded content, so a prompt whose
    /// first generated token is a near-tie flips between backends. The removed
    /// case was a single word ("Hello") and produced
    /// "Hello! How can I assist you today?" on CPU but
    /// "I'm sorry, but I'm not sure what you're asking" on CUDA — while "Hi"
    /// produced the evasive answer on CPU instead. The model answers bare
    /// greetings on a knife edge; the backends merely disagree about which side.
    ///
    /// Word count is a crude proxy for "wide argmax margin", but it is the one
    /// that is checkable without a GPU in CI, and it blocks the specific shape
    /// that actually cost 24 days of a red nightly.
    #[test]
    fn golden_prompts_are_not_bare_one_word_messages() {
        for (prompt, _) in golden_test_cases() {
            let user_msg = prompt
                .split("<|im_start|>user\n")
                .nth(1)
                .and_then(|s| s.split("<|im_end|>").next())
                .unwrap_or("");
            let words = user_msg.split_whitespace().count();
            assert!(
                words >= 3,
                "golden prompt user message {user_msg:?} has {words} word(s). \
                 Bare greetings sit on a near-tie under greedy decoding and flip \
                 between CPU and CUDA (#2350) — use a prompt with a wide argmax \
                 margin and verify it on BOTH backends before adding it."
            );
        }
    }

    // =========================================================================
    // #3711: the GPU leg's verdicts. An ERROR is a FAIL naming the error, never
    // a skip; a skip is only a leg that never started, and it names which.
    // =========================================================================

    const TWO_PLUS_TWO: &[&str] = &["4"];

    #[test]
    fn gpu_generation_error_fails_the_gate_naming_the_error() {
        let leg = GpuGoldenLeg::judge(
            Err("GPU generation: CUDA_ERROR_ILLEGAL_ADDRESS".to_string()),
            TWO_PLUS_TWO,
        );
        assert_eq!(
            leg,
            GpuGoldenLeg::Errored("GPU generation: CUDA_ERROR_ILLEGAL_ADDRESS".to_string())
        );
        let failure = leg.failure().expect("a GPU generation error must FAIL the gate");
        assert!(failure.contains("CUDA_ERROR_ILLEGAL_ADDRESS"), "{failure}");
        assert!(failure.contains("not a skip"), "{failure}");
    }

    #[test]
    fn cuda_init_error_on_a_host_with_a_device_fails_the_gate() {
        let leg = GpuGoldenLeg::judge(
            Err("CUDA init on device 0: CUDA_ERROR_OUT_OF_MEMORY".to_string()),
            TWO_PLUS_TWO,
        );
        let failure = leg.failure().expect("a CUDA init error must FAIL the gate");
        assert!(failure.contains("CUDA_ERROR_OUT_OF_MEMORY"), "{failure}");
    }

    #[test]
    fn gpu_wrong_answer_fails_the_gate() {
        let leg = GpuGoldenLeg::judge(Ok("2 + 2 = 5".to_string()), TWO_PLUS_TWO);
        assert!(matches!(leg, GpuGoldenLeg::WrongAnswer(_)), "{leg:?}");
        let failure = leg.failure().expect("a wrong GPU answer must FAIL the gate");
        assert!(failure.starts_with("GPU output failed (CPU passed)"), "{failure}");
    }

    #[test]
    fn gpu_right_answer_passes_and_says_it_was_judged() {
        for text in ["2 + 2 = 4", "<think>two and two</think>The answer is 4."] {
            let leg = GpuGoldenLeg::judge(Ok(text.to_string()), TWO_PLUS_TWO);
            assert_eq!(leg, GpuGoldenLeg::Passed, "{text:?}");
            assert_eq!(leg.failure(), None);
            assert_eq!(leg.describe(), "GPU leg judged on CUDA device 0");
        }
    }

    #[test]
    fn no_device_is_the_skip_and_names_it() {
        let why = gpu_golden_not_run(true, false, true).expect("no device: the leg never starts");
        assert_eq!(why, "no CUDA device on this host");
        let leg = GpuGoldenLeg::NotRun(why);
        assert_eq!(leg.failure(), None, "a leg that never started is not a failure");
        assert_eq!(leg.describe(), "GPU leg SKIPPED: no CUDA device on this host");
    }

    #[test]
    fn each_skip_names_which_and_a_cuda_host_runs_the_leg() {
        assert_eq!(
            gpu_golden_not_run(false, true, true),
            Some("this build has no cuda feature")
        );
        assert_eq!(
            gpu_golden_not_run(false, false, false),
            Some("this build has no cuda feature")
        );
        assert_eq!(
            gpu_golden_not_run(true, true, false),
            Some("the dense GPU leg judges GGUF only")
        );
        // a cuda build with a device and a GGUF: the leg STARTS, so no skip can be reported
        assert_eq!(gpu_golden_not_run(true, true, true), None);
    }

    // The runtime rung (the hybrid, and architectures the GPU declines) names the
    // backend the dispatch REPORTED. It never names the one the build could have used.

    #[test]
    fn runtime_cuda_build_with_no_device_is_labelled_cpu_never_gpu() {
        // the old message said "GPU hybrid forward" on every cuda build
        let not_run = gpu_golden_not_run(true, false, true);
        let label = runtime_golden_backend(false, not_run).expect("CPU was expected: a pass");
        assert!(label.starts_with("CPU: the dispatch reported used_gpu=false"), "{label}");
        assert!(label.contains("no CUDA device on this host"), "{label}");
        assert!(!label.contains("GPU:"), "{label}");
    }

    #[test]
    fn runtime_gpu_fallback_on_a_cuda_host_fails_the_gate() {
        // a cuda build, a device, an architecture the GPU runs, and the dispatch reported CPU
        let failure = runtime_golden_backend(false, gpu_golden_not_run(true, true, true))
            .expect_err("a GPU that fell back to the CPU must FAIL the gate");
        assert!(failure.contains("fell back"), "{failure}");
        assert!(failure.contains("not a pass"), "{failure}");
    }

    #[test]
    fn runtime_gpu_that_served_is_labelled_gpu_from_the_dispatch() {
        let label = runtime_golden_backend(true, gpu_golden_not_run(true, true, true))
            .expect("the GPU served: a pass");
        assert_eq!(label, "GPU: the dispatch reported used_gpu=true");
    }

    #[test]
    fn runtime_architecture_the_gpu_declines_is_a_cpu_pass_that_says_why() {
        let label = runtime_golden_backend(false, Some("the GPU backend declines this architecture"))
            .expect("CPU was expected: a pass");
        assert!(label.contains("the GPU backend declines this architecture"), "{label}");
    }
}

include!("throughput.rs");
