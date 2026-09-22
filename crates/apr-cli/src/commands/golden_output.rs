
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
/// The golden QUESTIONS — what is asked, with no opinion about how it is wrapped.
///
/// #3724 split this from the rendering. The gate used to carry three ChatML
/// strings, so every model was asked in ChatML whatever production sends it: on
/// `qwen3` that left the model in thinking mode, a mode no production path uses,
/// and on lambda its greedy reasoning for "2+2" overran the 512-token budget and
/// the gate reported "Empty output" for a model that answers "2 + 2 = 4." through
/// `apr serve`.
fn golden_questions() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("What is 2+2?", vec!["4"]),
        // Replaces the bare "Hello". Still exercises a conversational turn, but
        // with enough context that the first generated token is not a near-tie.
        (
            "Hello there, how are you doing today my friend?",
            vec!["Hello", "Hi", "hey", "hello", "well", "!"],
        ),
        // Factual recall: a wide-margin argmax and a check that the model is
        // actually reasoning over its weights rather than emitting boilerplate.
        ("What is the capital of France?", vec!["Paris"]),
    ]
}

/// Render one golden question the way PRODUCTION renders it for this architecture
/// (#3724 done_when 2): the same detector and the same template `apr serve` and
/// `apr run` use, keyed on the GGUF's `general.architecture`.
///
/// `None` — a format that declares no architecture — keeps the ChatML the gate has
/// always sent. For every architecture the detector maps to `ChatML` this is
/// byte-identical to the old hardcoded strings, because `ChatMLTemplate`'s
/// `format_conversation` of a single user message is exactly
/// `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n`; that identity is
/// pinned by `chatml_architectures_render_the_legacy_prompt_byte_for_byte`.
/// `qwen3`/`qwen35` map to `Qwen3NoThink`, which appends `<think>\n</think>\n` —
/// the production prompt, and the fix.
#[cfg(feature = "inference")]
fn golden_prompt_for(architecture: Option<&str>, question: &str) -> String {
    use realizar::chat_template::{format_messages, ChatMessage};

    let legacy = || format!("<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n");
    let Some(architecture) = architecture.map(str::trim).filter(|a| !a.is_empty()) else {
        return legacy();
    };
    // A rendering failure must not silently change what is asked: fall back to the
    // prompt the gate has always sent rather than to an empty or partial one.
    format_messages(&[ChatMessage::user(question)], Some(architecture)).unwrap_or_else(|_| legacy())
}

/// The budget the THINKING-ON leg gets (#3724 done_when 3: "a budget that
/// overdoes it").
///
/// Measured on lambda, qwen3-8b-q4km, greedy: at 2048 the three golden questions
/// produce think blocks of 545 / 114 / 126 tokens and every block closes. 512 —
/// the production leg's budget — is NOT enough for the first one on x86, which is
/// the whole defect. This number belongs to the ON leg only: the production leg's
/// budget is deliberately untouched, because the fix is the prompt, never the
/// budget.
#[cfg(feature = "inference")]
const THINKING_ON_BUDGET: usize = 2048;

/// The same conversation with production's THINKING SUPPRESSION removed, or
/// `None` when production does not suppress thinking for this architecture.
///
/// Derived from production's own rendered prompt rather than from a template
/// name: a prompt that ends in an EMPTY `<think></think>` block is one that
/// pre-closes the model's reasoning so it answers directly. Removing that block
/// is the same conversation in the other mode. Reading it off the rendering means
/// a replacement template (#3755) is followed automatically instead of silently
/// bypassed.
#[cfg(feature = "inference")]
fn without_thinking_prefill(prompt: &str) -> Option<String> {
    let start = prompt.rfind("<think>")?;
    let tail = &prompt[start + "<think>".len()..];
    let close = tail.find("</think>")?;
    // Only an EMPTY block is a suppression prefill; a block with reasoning in it
    // is content, and cutting it would change the conversation.
    if !tail[..close].trim().is_empty() {
        return None;
    }
    if !tail[close + "</think>".len()..].trim().is_empty() {
        return None;
    }
    Some(prompt[..start].to_string())
}

/// #3724 done_when 3: judge a thinking-capable model in the OTHER mode too.
///
/// OFF is the production prompt, already judged by the normal cases. ON is the
/// same conversation with the suppression removed and a budget that overdoes it:
/// the think block must CLOSE and the answer after it must be correct. An
/// unclosed block is reported by name with its budget — never stripped to an
/// empty answer, which is what turned this defect into "Empty output".
///
/// One question, not three: the ON leg exists to prove the model still answers
/// correctly when it reasons, and a 2048-token CPU generation per case would
/// quadruple the gate for every Qwen3 model to prove the same thing three times.
#[cfg(feature = "inference")]
fn thinking_on_case(architecture: Option<&str>) -> Option<(String, Vec<&'static str>)> {
    let (question, patterns) = golden_questions().into_iter().next()?;
    let production = golden_prompt_for(architecture, question);
    let on_prompt = without_thinking_prefill(&production)?;
    Some((on_prompt, patterns))
}

/// Judge one generated ON-mode output: the block closed, and the answer is right.
///
/// `generated` is the model's continuation with the prompt echo already removed.
#[cfg(feature = "inference")]
fn judge_thinking_on_output(generated: &str, patterns: &[&str], budget: usize) -> Option<String> {
    match split_thinking_blocks(generated) {
        ThinkingSplit::Unclosed => Some(unclosed_think_reason(
            "golden_output_thinking_on",
            budget,
            generated.len(),
        )),
        ThinkingSplit::Answer(answer) => {
            if !generated.contains("<think>") {
                // Nothing was suppressed and nothing was thought: the ON leg
                // proved nothing, and saying so is better than a green.
                return Some(format!(
                    "golden_output_thinking_on: no <think> block was produced within {budget} tokens \
                     — the thinking mode this leg exists to judge was never entered"
                ));
            }
            match verify_output(&answer, "golden_output_thinking_on", patterns) {
                OutputVerification::Fail { reason } => Some(reason),
                OutputVerification::Pass => None,
            }
        }
    }
}

/// The golden cases as (prompt, expected patterns) for one architecture.
#[cfg(feature = "inference")]
fn golden_test_cases_for(architecture: Option<&str>) -> Vec<(String, Vec<&'static str>)> {
    golden_questions()
        .into_iter()
        .map(|(question, patterns)| (golden_prompt_for(architecture, question), patterns))
        .collect()
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

    // GH-279-4: Thinking models (Qwen3) need extra tokens for <think>...</think>
    // chain-of-thought before the answer. 32 tokens is not enough — the model
    // exhausts the budget on reasoning and never emits the answer. Qwen3's
    // thinking can be verbose (~100-200 tokens for simple math), so 512 gives
    // ample room for reasoning + answer.
    let golden_max_tokens = config.max_tokens.max(512);

    let Some((_, output_text)) =
        generate_golden_for_format(path, prompt, golden_max_tokens, format, mapped, gguf_model)?
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

    // GH-279-4: generate_with_cache returns prompt + generated tokens.
    // Strip the prompt echo so we verify only the model's generated output.
    let generated_text = output_text
        .strip_prefix(prompt)
        .unwrap_or(&output_text);
    // GH-279-4 / #3724: a block that never closed is REPORTED with its budget. It
    // used to be truncated away, which made "still reasoning at the cut" and "the
    // model answered nothing" the same report ("Empty output").
    let answer_text = match split_thinking_blocks(generated_text) {
        ThinkingSplit::Answer(answer) => answer,
        ThinkingSplit::Unclosed => {
            return Ok(Some(GateResult::failed(
                "golden_output",
                &unclosed_think_reason("golden_output", golden_max_tokens, generated_text.len()),
                None,
                None,
                start.elapsed(),
            )));
        }
    };
    if let OutputVerification::Fail { reason } =
        verify_output(&answer_text, "golden_output", expected_patterns)
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

    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{GGUFModel, MappedGGUFModel};

        #[cfg(feature = "cuda")]
        let cuda_available = {
            use realizar::cuda::CudaExecutor;
            CudaExecutor::is_available() && CudaExecutor::num_devices() > 0
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_available = false;
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

        // #3724: ask the model the way production asks it. `general.architecture`
        // is the key the detector takes; a format that declares none keeps ChatML.
        let architecture = mapped
            .as_ref()
            .and_then(|m| m.model.architecture())
            .map(String::from);
        let test_cases = golden_test_cases_for(architecture.as_deref());

        for (prompt, expected_patterns) in &test_cases {
            if let Some(result) = validate_golden_test_case(
                path,
                prompt.as_str(),
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

        // #3724 done_when 3: a thinking-capable model is judged in BOTH modes.
        if let Some((on_prompt, on_patterns)) = thinking_on_case(architecture.as_deref()) {
            if let Some((_, on_text)) = generate_golden_for_format(
                path,
                &on_prompt,
                THINKING_ON_BUDGET,
                format,
                mapped.as_ref(),
                gguf_model.as_ref(),
            )? {
                let generated = on_text.strip_prefix(on_prompt.as_str()).unwrap_or(&on_text);
                if let Some(reason) =
                    judge_thinking_on_output(generated, &on_patterns, THINKING_ON_BUDGET)
                {
                    return Ok(GateResult::failed(
                        "golden_output",
                        &reason,
                        None,
                        None,
                        start.elapsed(),
                    ));
                }
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
        let _ = (path, config);
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
        let cases = golden_test_cases_for(None);
        assert!(!cases.is_empty());
        for (prompt, patterns) in &cases {
            assert!(prompt.contains("<|im_start|>assistant"));
            assert!(prompt.contains("<|im_start|>user"));
            assert!(!patterns.is_empty());
        }
    }

    #[test]
    fn test_golden_test_cases_arithmetic_case_present() {
        let cases = golden_test_cases_for(None);
        let arith = cases
            .iter()
            .find(|(p, _)| p.contains("2+2"))
            .expect("arithmetic golden case must exist");
        assert!(arith.1.contains(&"4"));
    }

    #[test]
    fn test_golden_test_cases_deterministic() {
        let a = golden_test_cases_for(None);
        let b = golden_test_cases_for(None);
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
        for (prompt, _) in golden_test_cases_for(None) {
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
    // #3724: the gate asks the model the way PRODUCTION asks it
    // =========================================================================

    /// The legacy prompt, spelled out once so the pin is on a literal and not on
    /// the function under test.
    fn legacy_chatml(question: &str) -> String {
        format!("<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
    }

    /// done_when 2, second half: "Every other architecture keeps byte-identical
    /// ChatML, pinned by a test."
    ///
    /// The CLAIM here is derived, not hand-written: the detector decides which
    /// architectures are ChatML, and each one must render exactly what the gate
    /// hardcoded before #3724. The architecture names below are a SAMPLE drawn
    /// from `contracts/arch-constraints-v1.yaml` — they are not a registry, and
    /// nothing may be concluded from one being absent. Both partitions are
    /// asserted non-empty so the test cannot pass vacuously if the detector
    /// changes under it.
    #[test]
    fn chatml_architectures_render_the_legacy_prompt_byte_for_byte() {
        use realizar::chat_template::{detect_format_from_name, TemplateFormat};
        let sample = [
            "gpt2", "llama", "qwen2", "qwen3", "qwen3_moe", "qwen35", "mistral", "phi2", "phi",
            "gemma", "deepseek", "mamba", "stablelm", "yi", "internlm2", "smollm", "olmo",
            "granite", "starcoder", "falcon_7b",
        ];
        let (mut chatml, mut other) = (0usize, 0usize);
        for arch in sample {
            let is_chatml = detect_format_from_name(arch) == TemplateFormat::ChatML;
            for (question, _) in golden_questions() {
                let rendered = golden_prompt_for(Some(arch), question);
                if is_chatml {
                    assert_eq!(
                        rendered,
                        legacy_chatml(question),
                        "{arch} maps to ChatML, so #3724 must not have changed its prompt"
                    );
                } else {
                    // Not a failure — it is the point. Recorded so the blast
                    // radius of #3724 is visible rather than silent.
                    assert_ne!(
                        rendered,
                        legacy_chatml(question),
                        "{arch} maps to {:?}, so the gate must now send THAT, not ChatML",
                        detect_format_from_name(arch)
                    );
                }
            }
            if is_chatml {
                chatml += 1;
            } else {
                other += 1;
            }
        }
        assert!(chatml > 0, "no ChatML architecture in the sample — the byte-identity half of done_when 2 went untested");
        assert!(other > 0, "no non-ChatML architecture in the sample — the detector half went untested");
        assert_eq!(
            detect_format_from_name("qwen2"),
            TemplateFormat::ChatML,
            "qwen2 is a ladder rung and its prompt must be the unchanged one"
        );
    }

    /// done_when 2, first half, and the whole defect: `qwen3` must be asked with
    /// production's no-think template, not ChatML. Without the `<think>\n</think>`
    /// prefill the model reasons for the full budget and the gate reports "Empty
    /// output" (measured on lambda, 527 tokens for "What is 2+2?").
    #[test]
    fn qwen3_is_asked_with_the_production_no_think_prompt() {
        for arch in ["qwen3", "qwen35"] {
            let prompt = golden_prompt_for(Some(arch), "What is 2+2?");
            assert_eq!(
                prompt,
                "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n",
                "{arch} must get production's no-think prompt"
            );
            assert_ne!(
                prompt,
                legacy_chatml("What is 2+2?"),
                "{arch} must NOT get the gate's old ChatML — that is the defect"
            );
        }
    }

    /// The gate and production must not drift apart by construction: whatever the
    /// detector says for an architecture is what the gate sends. This fails if a
    /// later change re-hardcodes a template in the gate.
    #[test]
    fn the_gate_renders_through_the_same_detector_production_uses() {
        use realizar::chat_template::{format_messages, ChatMessage};
        for arch in ["qwen2", "qwen3", "qwen35", "llama", "mistral", "phi2"] {
            for (question, _) in golden_questions() {
                let production = format_messages(&[ChatMessage::user(question)], Some(arch))
                    .expect("production renders this architecture");
                assert_eq!(
                    golden_prompt_for(Some(arch), question),
                    production,
                    "{arch}: the gate's prompt must BE production's prompt"
                );
            }
        }
    }

    /// A format that declares no architecture (APR, SafeTensors) is unchanged.
    #[test]
    fn an_unknown_architecture_keeps_the_legacy_chatml() {
        for arch in [None, Some(""), Some("   ")] {
            for (question, _) in golden_questions() {
                assert_eq!(golden_prompt_for(arch, question), legacy_chatml(question));
            }
        }
    }

    // ---- done_when 3: both modes -------------------------------------------

    /// The ON prompt is production's own prompt with the suppression removed —
    /// derived from the rendering, so a replacement template is followed rather
    /// than bypassed.
    #[test]
    fn the_thinking_on_prompt_is_production_minus_the_suppression() {
        let production = golden_prompt_for(Some("qwen3"), "What is 2+2?");
        assert!(production.ends_with("<think>\n</think>\n"), "{production}");
        let (on_prompt, patterns) = thinking_on_case(Some("qwen3")).expect("qwen3 is judged in both modes");
        assert_eq!(on_prompt, legacy_chatml("What is 2+2?"));
        assert_eq!(patterns, golden_questions()[0].1);
        assert!(!on_prompt.contains("<think>"), "the suppression is gone: {on_prompt}");
    }

    /// An architecture production does NOT suppress has no second mode to judge,
    /// and the gate must not invent one.
    #[test]
    fn an_architecture_without_suppression_has_no_thinking_on_leg() {
        assert!(thinking_on_case(Some("qwen2")).is_none());
        assert!(thinking_on_case(Some("llama")).is_none());
        assert!(thinking_on_case(None).is_none());
    }

    /// Only an EMPTY prefilled block is suppression. A block with reasoning in it
    /// is content, and cutting it would change the conversation.
    #[test]
    fn a_think_block_with_content_is_not_a_suppression_prefill() {
        assert_eq!(
            without_thinking_prefill("<|im_start|>assistant\n<think>\n</think>\n").as_deref(),
            Some("<|im_start|>assistant\n")
        );
        assert_eq!(
            without_thinking_prefill("<|im_start|>assistant\n<think>hmm</think>\n"),
            None
        );
        assert_eq!(without_thinking_prefill("<|im_start|>assistant\n"), None);
        // Something after the block is not a prefill either.
        assert_eq!(
            without_thinking_prefill("<think>\n</think>\nalready answering"),
            None
        );
    }

    /// done_when 3: ON passes only when the block CLOSES and the answer is right.
    #[test]
    fn the_thinking_on_leg_judges_closure_and_the_answer() {
        let patterns = vec!["4"];
        // closed + correct → pass
        assert_eq!(
            judge_thinking_on_output("<think>2 plus 2</think>2 + 2 = 4.", &patterns, 2048),
            None
        );
        // closed + wrong answer → the answer's failure, not a think failure
        let wrong = judge_thinking_on_output("<think>2 plus 2</think>It is five.", &patterns, 2048)
            .expect("a wrong answer fails");
        assert!(wrong.contains("golden_output_thinking_on"), "{wrong}");
        assert!(!wrong.contains("unclosed"), "{wrong}");
        // unclosed → reported by name WITH the budget, never as an empty answer
        let unclosed = judge_thinking_on_output("<think>let me work through this", &patterns, 2048)
            .expect("an unclosed block fails");
        assert!(
            unclosed.contains("think block unclosed within 2048 tokens"),
            "{unclosed}"
        );
        assert!(!unclosed.contains("Empty output"), "{unclosed}");
        // never entered thinking at all → the leg proved nothing, and says so
        let absent = judge_thinking_on_output("2 + 2 = 4.", &patterns, 2048)
            .expect("no think block means the leg judged nothing");
        assert!(absent.contains("never entered"), "{absent}");
    }

    /// The ON budget is the ON leg's alone. #3724's ruling: the fix is the prompt,
    /// never the budget — the production leg keeps the budget it had.
    #[test]
    fn the_thinking_on_budget_does_not_touch_the_production_leg() {
        assert_eq!(THINKING_ON_BUDGET, 2048);
        let config = QaConfig::default();
        assert_eq!(
            config.max_tokens.max(512),
            512,
            "the production leg's budget is still 512 by default"
        );
    }

    /// Whatever the architecture, the QUESTIONS and the expected patterns are the
    /// same ones — #3724 changed how the gate asks, never what it checks, and
    /// "no check is relaxed" is part of done_when 2.
    #[test]
    fn changing_the_architecture_changes_the_wrapper_and_never_the_question() {
        let chatml = golden_test_cases_for(Some("qwen2"));
        let nothink = golden_test_cases_for(Some("qwen3"));
        assert_eq!(chatml.len(), golden_questions().len());
        assert_eq!(chatml.len(), nothink.len());
        for (((c_prompt, c_pat), (n_prompt, n_pat)), (question, patterns)) in
            chatml.iter().zip(nothink.iter()).zip(golden_questions())
        {
            assert_eq!(c_pat, &patterns, "patterns are architecture-independent");
            assert_eq!(n_pat, &patterns, "patterns are architecture-independent");
            assert!(c_prompt.contains(question), "the question survives the wrapper");
            assert!(n_prompt.contains(question), "the question survives the wrapper");
        }
    }
}

include!("throughput.rs");
