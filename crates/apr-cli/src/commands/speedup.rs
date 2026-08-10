
/// The throughput floor the gate applies, in tok/s.
///
/// An explicit `--assert-tps N` IS the threshold, verbatim. The format-aware
/// numbers are a DEFAULT, and a default may only apply when the user asserted
/// nothing.
///
/// This used to read `Gguf => 10.0.max(config.min_tps / 10.0)` with
/// `Apr | SafeTensors => 1.0` — the user's assertion divided by ten for GGUF
/// and discarded outright for the other two formats. `apr qa --assert-tps
/// 100000` therefore passed at 4.1 tok/s and exited 0: the project's own
/// release gate could not fail. It is a free function so the decision can be
/// tested without a model on disk.
#[cfg(feature = "inference")]
fn throughput_threshold(asserted: Option<f64>, format: realizar::format::ModelFormat) -> f64 {
    use realizar::format::ModelFormat;
    asserted.unwrap_or(match format {
        // Quantized GGUF on GPU is ~100x faster than F32 CPU, so an unasserted
        // run holds each format to what that format can actually do.
        ModelFormat::Gguf => 10.0,
        ModelFormat::Apr | ModelFormat::SafeTensors => 1.0, // F32 CPU: 1 tok/s minimum
    })
}

/// Gate 2: Throughput Falsification
///
/// Runs a benchmark and asserts minimum tokens per second.
/// This is falsifiable - if throughput < threshold, test fails.
fn run_throughput_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running throughput benchmark...".yellow());
    }

    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};

        let tracer = TracerImpl::new_local();
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

        let prompt = "Write a hello world program in Python:";
        let Some((tps, _measurement_duration)) = throughput_for_format(
            path,
            &model_bytes,
            format,
            prompt,
            config,
            cuda_available,
            &tracer,
        )?
        else {
            return Ok(GateResult::skipped(
                "throughput",
                "SafeTensors: tokenizer.json not found in model directory",
            ));
        };

        let duration = start.elapsed();

        let threshold = throughput_threshold(config.min_tps, format);

        if tps >= threshold {
            Ok(GateResult::passed(
                "throughput",
                &format!("{:.1} tok/s >= {:.0} tok/s threshold", tps, threshold),
                Some(tps),
                Some(threshold),
                duration,
            ))
        } else {
            Ok(GateResult::failed(
                "throughput",
                &format!("{:.1} tok/s < {:.0} tok/s threshold", tps, threshold),
                Some(tps),
                Some(threshold),
                duration,
            ))
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "throughput",
            "Requires 'inference' feature",
        ))
    }
}

/// Compute Ollama parity letter grade from speedup ratio.
///
/// Grading system (from showcase spec §Executive Summary):
/// F (<50% Ollama) → D (50-75%) → C (75-100% = parity) → B (100-150%) → A (150-200%) → A+ (200%+)
#[cfg(feature = "inference")]
fn ollama_parity_grade(ratio: f64) -> &'static str {
    if ratio >= 2.0 {
        "A+"
    } else if ratio >= 1.5 {
        "A"
    } else if ratio >= 1.0 {
        "B"
    } else if ratio >= 0.75 {
        "C"
    } else if ratio >= 0.5 {
        "D"
    } else {
        "F"
    }
}

/// Gate 3: Ollama Parity Test
///
/// Compares performance against Ollama baseline (if available).
/// This is falsifiable - if speedup < target, test fails.
/// Measure our GGUF throughput for Ollama parity comparison.
///
/// Uses 128-token minimum to amortize prefill overhead — Ollama reports
/// decode-only throughput (eval_count/eval_duration), so short runs
/// unfairly penalize our measurement.
#[cfg(feature = "inference")]
fn measure_our_gguf_tps(
    path: &Path,
    config: &QaConfig,
    tracer: &TracerImpl,
) -> Result<f64> {
    use realizar::gguf::{
        GGUFModel, MappedGGUFModel, OwnedQuantizedModel,
        QuantizedGenerateConfig,
    };

    let model_bytes = std::fs::read(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
    let gguf = GGUFModel::from_bytes(&model_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

    let prompt = "Write a function to check if a number is prime:";
    let bos = aprender::demo::SpecialTokens::qwen2().bos_id;
    let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![bos]);
    let parity_max_tokens = config.max_tokens.max(128);
    let gen_config = QuantizedGenerateConfig {
        max_tokens: parity_max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let budget_us = parity_max_tokens as u64 * config.iterations as u64 * 100_000;

    #[cfg(feature = "cuda")]
    let cuda_available = {
        use realizar::cuda::CudaExecutor;
        CudaExecutor::is_available() && CudaExecutor::num_devices() > 0
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_available = false;

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
                let (tps, _) = measure_generate_throughput(
                    config.warmup,
                    config.iterations,
                    prompt_tokens.len(),
                    tracer,
                    "qa_ollama_parity_gpu",
                    budget_us,
                    config.verbose,
                    || {
                        cuda_model
                            .generate_gpu_resident(&prompt_tokens, &gen_config)
                            .unwrap_or_default()
                    },
                );
                return Ok(tps);
            }
            Err(e) => {
                // Recover the model for CPU fallback (CudaInitError preserves the model)
                let model = e.into_model();
                let (tps, _) = measure_generate_throughput(
                    config.warmup,
                    config.iterations,
                    prompt_tokens.len(),
                    tracer,
                    "qa_ollama_parity_cpu_fallback",
                    budget_us,
                    config.verbose,
                    || {
                        model
                            .generate_with_cache(&prompt_tokens, &gen_config)
                            .unwrap_or_default()
                    },
                );
                return Ok(tps);
            }
        }
    }
    let (tps, _) = measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        tracer,
        "qa_ollama_parity_cpu",
        budget_us,
        config.verbose,
        || {
            model
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    );
    Ok(tps)
}

fn run_ollama_parity_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running Ollama parity test...".yellow());
    }

    if !check_ollama_available() {
        return Ok(GateResult::skipped(
            "ollama_parity",
            "Ollama not available (start with: ollama serve)",
        ));
    }

    #[cfg(feature = "inference")]
    {
        let tracer = TracerImpl::new_local();
        let ollama_tps = measure_ollama_throughput(path, config)?;

        if ollama_tps <= 0.0 {
            return Ok(GateResult::skipped(
                "ollama_parity",
                "Could not measure Ollama throughput",
            ));
        }

        let our_tps = measure_our_gguf_tps(path, config, &tracer)?;
        let speedup = our_tps / ollama_tps;
        let grade = ollama_parity_grade(speedup);
        let duration = start.elapsed();

        if speedup >= config.min_speedup {
            Ok(GateResult::passed(
                "ollama_parity",
                &format!(
                    "{:.1}x Ollama ({:.0} vs {:.0} tok/s) Grade {grade} >= {:.1}x threshold",
                    speedup, our_tps, ollama_tps, config.min_speedup
                ),
                Some(speedup),
                Some(config.min_speedup),
                duration,
            ))
        } else {
            Ok(GateResult::failed(
                "ollama_parity",
                &format!(
                    "{:.2}x Ollama ({:.0} vs {:.0} tok/s) Grade {grade} < {:.1}x threshold",
                    speedup, our_tps, ollama_tps, config.min_speedup
                ),
                Some(speedup),
                Some(config.min_speedup),
                duration,
            ))
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "ollama_parity",
            "Requires 'inference' feature",
        ))
    }
}

/// Gate 4: GPU vs CPU Speedup Test (F-PERF-042)
///
/// Measures throughput on both GPU and CPU, verifies GPU >= 2x CPU.
/// This is falsifiable - if GPU speedup < threshold, test fails.
///
/// Toyota Way: Genchi Genbutsu - Go and see for yourself. Measure real performance.
/// Measure GPU and CPU throughput for a GGUF model, returning (cpu_tps, gpu_tps).
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn measure_gpu_cpu_tps(
    path: &Path,
    config: &QaConfig,
    tracer: &TracerImpl,
) -> Result<(f64, f64)> {
    use realizar::gguf::{
        GGUFModel, MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
        QuantizedGenerateConfig,
    };

    let model_bytes = std::fs::read(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
    let gguf = GGUFModel::from_bytes(&model_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

    let prompt = "Write a function to calculate factorial:";
    let bos = aprender::demo::SpecialTokens::qwen2().bos_id;
    let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![bos]);
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let budget_us = config.max_tokens as u64 * config.iterations as u64 * 100_000;

    // CPU throughput
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
    let (cpu_tps, _) = measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        tracer,
        "qa_gpu_speedup_cpu",
        budget_us,
        config.verbose,
        || {
            model
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    );

    // GPU throughput — GH-284: fall back to 0.0 on capability mismatch
    let mapped2 = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
    let model2 = OwnedQuantizedModel::from_mapped(&mapped2)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
    let gpu_tps = match OwnedQuantizedModelCuda::with_max_seq_len(model2, 0, 2048) {
        Ok(mut cuda_model) => {
            let (tps, _) = measure_generate_throughput(
                config.warmup,
                config.iterations,
                prompt_tokens.len(),
                tracer,
                "qa_gpu_speedup_gpu",
                budget_us,
                config.verbose,
                || {
                    cuda_model
                        .generate_gpu_resident(&prompt_tokens, &gen_config)
                        .unwrap_or_default()
                },
            );
            tps
        }
        Err(_) => 0.0, // GPU unavailable for this architecture (e.g. missing QkNorm)
    };

    Ok((cpu_tps, gpu_tps))
}

fn run_gpu_speedup_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running GPU vs CPU speedup test...".yellow());
    }

    #[cfg(all(feature = "inference", feature = "cuda"))]
    {
        use realizar::cuda::CudaExecutor;
        use realizar::format::{detect_format, ModelFormat};

        let cuda_available = CudaExecutor::is_available() && CudaExecutor::num_devices() > 0;
        if !cuda_available {
            return Ok(GateResult::skipped(
                "gpu_speedup",
                "CUDA not available - cannot compare GPU vs CPU",
            ));
        }

        let model_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
        let format = detect_format(&model_bytes[..8.min(model_bytes.len())])
            .map_err(|e| CliError::ValidationFailed(format!("Failed to detect format: {e}")))?;
        if format != ModelFormat::Gguf {
            return Ok(GateResult::skipped(
                "gpu_speedup",
                "Only GGUF format supported",
            ));
        }

        let tracer = TracerImpl::new_local();
        let (cpu_tps, gpu_tps) = measure_gpu_cpu_tps(path, config, &tracer)?;
        let duration = start.elapsed();

        if cpu_tps <= 0.0 {
            return Ok(GateResult::failed(
                "gpu_speedup",
                "CPU throughput was zero - cannot calculate speedup",
                None,
                None,
                duration,
            ));
        }

        let speedup = gpu_tps / cpu_tps;

        if speedup >= config.min_gpu_speedup {
            Ok(GateResult::passed(
                "gpu_speedup",
                &format!(
                    "GPU {:.1}x faster than CPU ({:.0} vs {:.0} tok/s) >= {:.1}x threshold",
                    speedup, gpu_tps, cpu_tps, config.min_gpu_speedup
                ),
                Some(speedup),
                Some(config.min_gpu_speedup),
                duration,
            ))
        } else {
            Ok(GateResult::failed(
                "gpu_speedup",
                &format!(
                    "GPU {:.2}x faster than CPU ({:.0} vs {:.0} tok/s) < {:.1}x threshold",
                    speedup, gpu_tps, cpu_tps, config.min_gpu_speedup
                ),
                Some(speedup),
                Some(config.min_gpu_speedup),
                duration,
            ))
        }
    }

    #[cfg(not(all(feature = "inference", feature = "cuda")))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "gpu_speedup",
            "Requires 'inference' and 'cuda' features",
        ))
    }
}

// ============================================================================
// Throughput-threshold falsifiers (dogfood 0.63.0)
// ============================================================================
//
// `apr qa --assert-tps 100000` measured 4.1 tok/s, reported
// "4.1 tok/s >= 1 tok/s threshold", passed the gate and exited 0. On GGUF the
// assertion was divided by ten (floored at 10); on APR and SafeTensors it was
// discarded and replaced with 1.0. `apr qa` is the project's own release gate
// and its throughput assertion could not fail.
#[cfg(all(test, feature = "inference"))]
mod assert_tps_tests {
    use super::throughput_threshold;
    use realizar::format::ModelFormat;

    const ALL_FORMATS: [ModelFormat; 3] = [
        ModelFormat::Gguf,
        ModelFormat::Apr,
        ModelFormat::SafeTensors,
    ];

    #[test]
    fn an_asserted_threshold_is_used_verbatim_for_every_format() {
        for format in ALL_FORMATS {
            for asserted in [1.0_f64, 50.0, 100.0, 100_000.0] {
                let got = throughput_threshold(Some(asserted), format);
                assert!(
                    (got - asserted).abs() < f64::EPSILON,
                    "--assert-tps {asserted} on {format:?} became {got}; the user's \
                     assertion must be the threshold, not a suggestion"
                );
            }
        }
    }

    #[test]
    fn the_shipped_0_63_0_repro_now_fails_the_gate() {
        // The exact numbers from the dogfood run: 4.1 tok/s measured against
        // --assert-tps 100000 on an APR model, which reported PASS at a
        // substituted threshold of 1.0.
        let measured = 4.050_120_237_944_564_f64;
        let threshold = throughput_threshold(Some(100_000.0), ModelFormat::Apr);
        assert!(
            measured < threshold,
            "4.1 tok/s must not satisfy --assert-tps 100000 (threshold was {threshold})"
        );

        // And the GGUF case: 15.9 tok/s measured against --assert-tps 50,
        // which passed at a substituted threshold of 10.
        let measured_gguf = 15.881_003_639_727_522_f64;
        let threshold_gguf = throughput_threshold(Some(50.0), ModelFormat::Gguf);
        assert!(
            measured_gguf < threshold_gguf,
            "15.9 tok/s must not satisfy --assert-tps 50 (threshold was {threshold_gguf})"
        );
    }

    #[test]
    fn no_assertion_keeps_the_format_aware_defaults() {
        // Unasserted behaviour is unchanged, so this fix introduces no new
        // failures for runs that never asked for a floor.
        assert!((throughput_threshold(None, ModelFormat::Gguf) - 10.0).abs() < f64::EPSILON);
        assert!((throughput_threshold(None, ModelFormat::Apr) - 1.0).abs() < f64::EPSILON);
        assert!((throughput_threshold(None, ModelFormat::SafeTensors) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn an_assertion_below_the_default_still_binds() {
        // The old code floored GGUF at 10, so a deliberately lax
        // `--assert-tps 2` was silently tightened to 10. An assertion is an
        // assertion in both directions.
        assert!((throughput_threshold(Some(2.0), ModelFormat::Gguf) - 2.0).abs() < f64::EPSILON);
    }
}
