
/// Detect model format by reading magic bytes (first 8 bytes only).
///
/// Avoids loading multi-GB files just to check format. Used by PTX parity
/// and GPU state isolation gates.
#[cfg(feature = "inference")]
fn detect_model_format(path: &Path) -> Option<realizar::format::ModelFormat> {
    let magic = std::fs::File::open(path).ok().and_then(|mut f| {
        use std::io::Read;
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf).ok()?;
        Some(buf.to_vec())
    })?;
    realizar::format::detect_format(&magic).ok()
}

/// Gate 6: PTX Parity Validation (GH-219, F-PTX-001)
///
/// Validates that all 6 batched GPU kernels maintain structural parity with their
/// single-vector references. This catches compile-time PTX generation bugs like:
/// - Missing batch dispatch mechanism (no ctaid.y or m_dim)
/// - u64 shared memory addressing (should use u32 for portability)
/// - Wrong dispatch strategy for kernel type
///
/// Toyota Way: Poka-Yoke - error-proof PTX at generation time, not at runtime.
fn run_ptx_parity_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running PTX parity validation...".yellow());
    }

    // Extract model dimensions from GGUF metadata
    #[cfg(feature = "inference")]
    {
        use realizar::format::ModelFormat;
        use realizar::ptx_parity::{validate_all_kernel_pairs, KernelDimensions};

        if detect_model_format(path) != Some(ModelFormat::Gguf) {
            return Ok(GateResult::skipped(
                "ptx_parity",
                "Non-GGUF format (PTX kernels only apply to quantized inference)",
            ));
        }

        // Load model config to get dimensions
        let mapped = realizar::gguf::MappedGGUFModel::from_path(path.to_str().unwrap_or_default())
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

        let model_config = realizar::gguf::GGUFConfig::from_gguf(&mapped.model)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read config: {e}")))?;

        let dims = KernelDimensions {
            hidden_dim: model_config.hidden_dim as u32,
            intermediate_dim: model_config.intermediate_dim as u32,
            num_heads: model_config.num_heads as u32,
            head_dim: (model_config.hidden_dim / model_config.num_heads) as u32,
            rope_theta: model_config.rope_theta,
            epsilon: model_config.eps,
        };

        let report = validate_all_kernel_pairs(&dims);
        let duration = start.elapsed();

        if report.all_passed() {
            Ok(GateResult::passed(
                "ptx_parity",
                &report.summary(),
                Some(report.passed as f64),
                Some(report.total as f64),
                duration,
            ))
        } else {
            // Show violations in verbose mode
            if !config.json && config.verbose {
                for result in &report.results {
                    if !result.passed {
                        println!(
                            "  {} {} ({}): {}",
                            "FAIL".red(),
                            result.name,
                            result.dispatch_strategy,
                            result.violations.join("; ")
                        );
                    }
                }
            }
            Ok(GateResult::failed(
                "ptx_parity",
                &report.summary(),
                Some(report.passed as f64),
                Some(report.total as f64),
                duration,
            ))
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config, start);
        Ok(GateResult::skipped(
            "ptx_parity",
            "Requires inference feature",
        ))
    }
}

/// Gate 8: GPU State Isolation Test
///
/// Verifies that GPU state (KV cache, CUDA graphs, position buffers) is properly
/// isolated between generations. Catches PMAT-PREFILL-FIX class bugs where stale
/// state from a previous generation leaks into the next.
///
/// Protocol:
/// 1. Generate with prompt A → output_A
/// 2. Reset KV cache
/// 3. Generate with prompt B → output_B
/// 4. Reset KV cache
/// 5. Generate with prompt A again → output_A2
/// 6. Assert: output_A == output_A2 (state isolation)
/// 7. Assert: output_A != output_B (model is functional)
fn run_gpu_state_isolation_gate(path: &Path, _config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    #[cfg(all(feature = "inference", feature = "cuda"))]
    {
        use realizar::cuda::CudaExecutor;
        use realizar::format::ModelFormat;
        use realizar::gguf::{
            GGUFModel, MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
            QuantizedGenerateConfig,
        };

        if !CudaExecutor::is_available() || CudaExecutor::num_devices() == 0 {
            return Ok(GateResult::skipped(
                "gpu_state_isolation",
                "CUDA not available",
            ));
        }

        if detect_model_format(path) != Some(ModelFormat::Gguf) {
            return Ok(GateResult::skipped(
                "gpu_state_isolation",
                "Only GGUF format supported for GPU state isolation",
            ));
        }

        let model_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
        let gguf = GGUFModel::from_bytes(&model_bytes)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

        let prompt_a = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
        let prompt_b =
            "<|im_start|>user\nWrite hello world in Python<|im_end|>\n<|im_start|>assistant\n";

        let tokens_a = gguf.encode(prompt_a).unwrap_or_else(|| vec![151643, 9707]);
        let tokens_b = gguf.encode(prompt_b).unwrap_or_else(|| vec![151643, 1234]);

        let gen_config = QuantizedGenerateConfig {
            max_tokens: 16,
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        };

        let mapped = MappedGGUFModel::from_path(path)
            .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
        let model = OwnedQuantizedModel::from_mapped(&mapped)
            .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
        let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
            .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;

        let output_a = cuda_model
            .generate_gpu_resident(&tokens_a, &gen_config)
            .map_err(|e| CliError::ValidationFailed(format!("Gen 1 failed: {e}")))?;
        let output_b = cuda_model
            .generate_gpu_resident(&tokens_b, &gen_config)
            .map_err(|e| CliError::ValidationFailed(format!("Gen 2 failed: {e}")))?;
        let output_a2 = cuda_model
            .generate_gpu_resident(&tokens_a, &gen_config)
            .map_err(|e| CliError::ValidationFailed(format!("Gen 3 failed: {e}")))?;

        let duration = start.elapsed();

        if output_a != output_a2 {
            let text_a = gguf.decode(&output_a);
            let text_a2 = gguf.decode(&output_a2);
            return Ok(GateResult::failed(
                "gpu_state_isolation",
                &format!(
                    "State leak: prompt A produced different output on retry. \
                     First: '{}', Retry: '{}'",
                    text_a.chars().take(50).collect::<String>(),
                    text_a2.chars().take(50).collect::<String>()
                ),
                None,
                None,
                duration,
            ));
        }

        if output_a == output_b {
            return Ok(GateResult::failed(
                "gpu_state_isolation",
                "Model stuck: same output for different prompts (GPU state not functional)",
                None,
                None,
                duration,
            ));
        }

        Ok(GateResult::passed(
            "gpu_state_isolation",
            "GPU state properly isolated: 3 generations, deterministic replay confirmed",
            Some(3.0),
            Some(3.0),
            duration,
        ))
    }

    #[cfg(not(all(feature = "inference", feature = "cuda")))]
    {
        let _ = (path, _config);
        Ok(GateResult::skipped(
            "gpu_state_isolation",
            "Requires inference+cuda features",
        ))
    }
}

/// Gate 9: Performance Regression Detection
///
/// Compares current gate results against a previous QA report to detect
/// performance regressions. Catches Bug 206 class issues where metrics
/// silently degrade between rounds.
fn run_performance_regression_gate(
    current_gates: &[GateResult],
    config: &QaConfig,
) -> Result<GateResult> {
    let start = Instant::now();

    let Some(prev_path) = &config.previous_report else {
        return Ok(GateResult::skipped(
            "performance_regression",
            "No previous report provided",
        ));
    };

    let prev_json = std::fs::read_to_string(prev_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read previous report: {e}")))?;

    let prev_report: QaReport = serde_json::from_str(&prev_json)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse previous report: {e}")))?;

    let threshold = config.regression_threshold;
    let mut regressions = Vec::new();

    // Compare metrics for gates that have numeric values in both reports
    let comparable_gates = ["throughput", "ollama_parity", "gpu_speedup"];
    for gate_name in &comparable_gates {
        let prev_gate = prev_report.gates.iter().find(|g| g.name == *gate_name);
        let curr_gate = current_gates.iter().find(|g| g.name == *gate_name);

        if let Some(msg) = detect_regression(prev_gate, curr_gate, gate_name, threshold) {
            regressions.push(msg);
        }
    }

    let duration = start.elapsed();

    if regressions.is_empty() {
        Ok(GateResult::passed(
            "performance_regression",
            &format!(
                "No regressions >{:.0}% vs {}",
                threshold * 100.0,
                prev_path.display()
            ),
            Some(0.0),
            Some(threshold),
            duration,
        ))
    } else {
        Ok(GateResult::failed(
            "performance_regression",
            &format!("Regressions detected: {}", regressions.join("; ")),
            Some(regressions.len() as f64),
            Some(0.0),
            duration,
        ))
    }
}

/// Compare a single gate's value between previous and current reports for regression.
fn detect_regression(
    prev: Option<&GateResult>,
    curr: Option<&GateResult>,
    name: &str,
    threshold: f64,
) -> Option<String> {
    let (prev, curr) = (prev?, curr?);
    let (prev_val, curr_val) = (prev.value?, curr.value?);
    if prev_val <= 0.0 || prev.skipped || curr.skipped {
        return None;
    }
    let regression = (prev_val - curr_val) / prev_val;
    if regression <= threshold {
        return None;
    }
    Some(format!(
        "{name}: {prev_val:.1} -> {curr_val:.1} ({:.0}% regression)",
        regression * 100.0
    ))
}
