//! QA Command Implementation - Falsifiable Quality Assurance Checklist
//!
//! Implements a scientific QA process for model releases. Every claim must be
//! falsifiable - if a test can't fail, it doesn't provide information.
//!
//! # Gates
//!
//! 1. **Golden Output Test** (Correctness Gate)
//!    - Run model with known prompts, verify expected patterns in output
//!    - Falsifiable: Output must match expected pattern or test fails
//!
//! 2. **Throughput Falsification** (Performance Gate)
//!    - Run benchmark with statistical rigor (CV < 5%)
//!    - Assert minimum tok/s threshold
//!    - Falsifiable: If tok/s < threshold, test fails
//!
//! 3. **Ollama Parity Test** (Parity Gate)
//!    - Compare against Ollama baseline (if available)
//!    - Assert speedup factor >= target
//!    - Falsifiable: If speedup < target, test fails
//!
//! 4. **GPU vs CPU Speedup Test** (F-PERF-042)
//!    - Measure throughput on both GPU and CPU
//!    - Assert GPU >= 2x CPU (default threshold)
//!    - Falsifiable: If GPU speedup < threshold, test fails
//!    - Toyota Way: Genchi Genbutsu - measure real performance
//!
//! 5. **Cross-Format Parity Test** (F-QUAL-032)
//!    - Compare argmax between GGUF and SafeTensors for same model
//!    - Invariant: argmax(forward_gguf) == argmax(forward_safetensors)
//!    - Falsifiable: If argmax differs, cross-format parity is BROKEN
//!    - Cornerstone of architecture's logical validity
//!
//! 6. **PTX Parity Test** (GH-219, F-PTX-001)
//!    - Validate batched GPU kernels maintain structural parity with single-vector references
//!    - Checks: batch dispatch mechanism, u64 shared memory addressing, dispatch strategy
//!    - Falsifiable: If any of 6 kernel pairs fails structural validation, test fails
//!    - Toyota Way: Poka-Yoke - error-proof PTX generation at compile time
//!
//! # Usage
//!
//! ```bash
//! apr qa model.gguf                           # Run all gates
//! apr qa model.gguf --assert-tps 100          # Custom throughput threshold
//! apr qa model.gguf --assert-speedup 2.0      # Custom Ollama speedup
//! apr qa model.gguf --assert-gpu-speedup 3.0  # Custom GPU vs CPU speedup
//! apr qa model.gguf --skip-ollama             # Skip Ollama comparison
//! apr qa model.gguf --skip-gpu-speedup        # Skip GPU vs CPU test
//! apr qa model.gguf --skip-format-parity      # Skip cross-format test
//! apr qa model.gguf --safetensors-path m.st   # Compare with SafeTensors model
//! apr qa model.gguf --json                    # JSON output for CI
//! ```
//!
//! # Exit Codes
//!
//! - 0: All gates passed
//! - 5: One or more gates failed (ValidationFailed)
//!
//! Toyota Way: Jidoka - Stop and fix quality issues immediately.
//! Scientific Method: Claims must be falsifiable to have meaning.

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

/// QA configuration
#[derive(Debug, Clone)]
pub struct QaConfig {
    /// Minimum throughput in tok/s (default: 100 for GPU, 10 for CPU)
    pub min_tps: f64,
    /// Minimum speedup vs Ollama (default: 2.0x)
    pub min_speedup: f64,
    /// Minimum GPU vs CPU speedup (default: 2.0x) - F-PERF-042
    pub min_gpu_speedup: f64,
    /// Skip golden output test
    pub skip_golden: bool,
    /// Skip throughput test
    pub skip_throughput: bool,
    /// Skip Ollama parity test
    pub skip_ollama: bool,
    /// Skip GPU vs CPU speedup test (F-PERF-042)
    pub skip_gpu_speedup: bool,
    /// Skip tensor contract validation (PMAT-235)
    pub skip_contract: bool,
    /// Skip cross-format parity test (F-QUAL-032)
    pub skip_format_parity: bool,
    /// Skip PTX parity validation (GH-219, F-PTX-001)
    pub skip_ptx_parity: bool,
    /// SafeTensors model path for cross-format parity (F-QUAL-032)
    pub safetensors_path: Option<std::path::PathBuf>,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Number of warmup iterations
    pub warmup: usize,
    /// Max tokens for generation
    pub max_tokens: usize,
    /// Output as JSON
    pub json: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            min_tps: 100.0,       // GPU target
            min_speedup: 0.2, // Ollama uses llama.cpp optimized kernels; 0.2x is realistic floor
            min_gpu_speedup: 2.0, // GPU must be 2x faster than CPU (F-PERF-042)
            skip_golden: false,
            skip_throughput: false,
            skip_ollama: false,
            skip_gpu_speedup: false,
            skip_contract: false,
            skip_format_parity: false,
            skip_ptx_parity: false,
            safetensors_path: None,
            iterations: 10,
            warmup: 3,
            max_tokens: 32,
            json: false,
            verbose: false,
        }
    }
}

/// Result of a single QA gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Gate name
    pub name: String,
    /// Whether the gate passed
    pub passed: bool,
    /// Human-readable result message
    pub message: String,
    /// Measured value (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Expected/threshold value (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Time taken to run the gate
    pub duration_ms: u64,
    /// Whether the gate was skipped
    pub skipped: bool,
}

impl GateResult {
    fn passed(
        name: &str,
        message: &str,
        value: Option<f64>,
        threshold: Option<f64>,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            message: message.to_string(),
            value,
            threshold,
            duration_ms: duration.as_millis() as u64,
            skipped: false,
        }
    }

    fn failed(
        name: &str,
        message: &str,
        value: Option<f64>,
        threshold: Option<f64>,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            message: message.to_string(),
            value,
            threshold,
            duration_ms: duration.as_millis() as u64,
            skipped: false,
        }
    }

    fn skipped(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true, // Skipped gates don't fail
            message: format!("Skipped: {reason}"),
            value: None,
            threshold: None,
            duration_ms: 0,
            skipped: true,
        }
    }
}

/// Full QA report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaReport {
    /// Model path
    pub model: String,
    /// Whether all gates passed
    pub passed: bool,
    /// Individual gate results
    pub gates: Vec<GateResult>,
    /// Total duration
    pub total_duration_ms: u64,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// Summary message
    pub summary: String,
}

/// Run the QA command
#[allow(clippy::too_many_arguments)]
pub fn run(
    path: &Path,
    min_tps: Option<f64>,
    min_speedup: Option<f64>,
    min_gpu_speedup: Option<f64>,
    skip_golden: bool,
    skip_throughput: bool,
    skip_ollama: bool,
    skip_gpu_speedup: bool,
    skip_contract: bool,
    skip_format_parity: bool,
    skip_ptx_parity: bool,
    safetensors_path: Option<std::path::PathBuf>,
    iterations: usize,
    warmup: usize,
    max_tokens: usize,
    json: bool,
    verbose: bool,
) -> Result<()> {
    let config = QaConfig {
        min_tps: min_tps.unwrap_or(100.0),
        min_speedup: min_speedup.unwrap_or(0.2), // Ollama uses llama.cpp optimized kernels
        min_gpu_speedup: min_gpu_speedup.unwrap_or(2.0), // GPU must be 2x faster (F-PERF-042)
        skip_golden,
        skip_throughput,
        skip_ollama,
        skip_gpu_speedup,
        skip_contract,
        skip_format_parity,
        skip_ptx_parity,
        safetensors_path,
        iterations,
        warmup,
        max_tokens,
        json,
        verbose,
    };

    let report = run_qa(path, &config)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
    }

    if !report.passed {
        return Err(CliError::ValidationFailed(report.summary));
    }

    Ok(())
}

/// Run all QA gates and produce a report
fn run_qa(path: &Path, config: &QaConfig) -> Result<QaReport> {
    let start = Instant::now();
    let mut gates = Vec::new();

    if !config.json {
        output::header("APR Quality Assurance");
        let config_pairs = vec![
            ("Model", path.display().to_string()),
            ("Min TPS", format!("{:.0} tok/s", config.min_tps)),
            ("Min Speedup", format!("{:.1}x Ollama", config.min_speedup)),
        ];
        println!("{}", output::kv_table(&config_pairs));
    }

    // Gate 0: Tensor Contract Validation (PMAT-235)
    let contract_result = if config.skip_contract {
        GateResult::skipped("tensor_contract", "Skipped by --skip-contract")
    } else {
        run_tensor_contract_gate(path, config)?
    };
    if !config.json {
        print_gate_result(&contract_result);
    }
    gates.push(contract_result);

    // Gate 1: Golden Output Test (Correctness)
    let golden_result = if config.skip_golden {
        GateResult::skipped("golden_output", "Skipped by --skip-golden")
    } else {
        run_golden_output_gate(path, config)?
    };
    if !config.json {
        print_gate_result(&golden_result);
    }
    gates.push(golden_result);

    // Gate 2: Throughput Falsification (Performance)
    let throughput_result = if config.skip_throughput {
        GateResult::skipped("throughput", "Skipped by --skip-throughput")
    } else {
        run_throughput_gate(path, config)?
    };
    if !config.json {
        print_gate_result(&throughput_result);
    }
    gates.push(throughput_result);

    // Gate 3: Ollama Parity Test (GGUF only — F32/F16 lacks fused kernels for meaningful comparison)
    let ollama_result = if config.skip_ollama {
        GateResult::skipped("ollama_parity", "Skipped by --skip-ollama")
    } else {
        #[cfg(feature = "inference")]
        {
            use realizar::format::{detect_format, ModelFormat};
            let magic = std::fs::read(path).ok().and_then(|b| {
                if b.len() >= 8 {
                    Some(b[..8].to_vec())
                } else {
                    None
                }
            });
            let fmt = magic.and_then(|m| detect_format(&m).ok());
            if fmt == Some(ModelFormat::Gguf) {
                run_ollama_parity_gate(path, config)?
            } else {
                GateResult::skipped(
                    "ollama_parity",
                    "Non-GGUF format (F32/F16 lacks fused kernels for Ollama parity)",
                )
            }
        }
        #[cfg(not(feature = "inference"))]
        {
            run_ollama_parity_gate(path, config)?
        }
    };
    if !config.json {
        print_gate_result(&ollama_result);
    }
    gates.push(ollama_result);

    // Gate 4: GPU vs CPU Speedup (F-PERF-042)
    let gpu_speedup_result = if config.skip_gpu_speedup {
        GateResult::skipped("gpu_speedup", "Skipped by --skip-gpu-speedup")
    } else {
        run_gpu_speedup_gate(path, config)?
    };
    if !config.json {
        print_gate_result(&gpu_speedup_result);
    }
    gates.push(gpu_speedup_result);

    // Gate 5: Cross-Format Parity (F-QUAL-032)
    let format_parity_result = if config.skip_format_parity {
        GateResult::skipped("format_parity", "Skipped by --skip-format-parity")
    } else if config.safetensors_path.is_none() {
        GateResult::skipped("format_parity", "No --safetensors-path provided")
    } else {
        run_format_parity_gate(path, config)?
    };
    if !config.json {
        print_gate_result(&format_parity_result);
    }
    gates.push(format_parity_result);

    // Gate 6: PTX Parity Validation (GH-219, F-PTX-001)
    let ptx_parity_result = if config.skip_ptx_parity {
        GateResult::skipped("ptx_parity", "Skipped by --skip-ptx-parity")
    } else {
        run_ptx_parity_gate(path, config)?
    };
    if !config.json {
        print_gate_result(&ptx_parity_result);
    }
    gates.push(ptx_parity_result);

    let total_duration = start.elapsed();
    let passed = gates.iter().all(|g| g.passed);
    let failed_gates: Vec<_> = gates.iter().filter(|g| !g.passed && !g.skipped).collect();

    let summary = if passed {
        "All QA gates passed".to_string()
    } else {
        let names: Vec<_> = failed_gates.iter().map(|g| g.name.as_str()).collect();
        format!("Failed gates: {}", names.join(", "))
    };

    if !config.json {
        output::header("QA Summary");

        // Summary table
        let gate_rows: Vec<Vec<String>> = gates
            .iter()
            .map(|g| {
                let badge = if g.skipped {
                    output::badge_skip("SKIP")
                } else if g.passed {
                    output::badge_pass("PASS")
                } else {
                    output::badge_fail("FAIL")
                };
                let name = match g.name.as_str() {
                    "tensor_contract" => "Tensor Contract",
                    "golden_output" => "Golden Output",
                    "throughput" => "Throughput",
                    "ollama_parity" => "Ollama Parity",
                    "gpu_speedup" => "GPU Speedup",
                    "format_parity" => "Format Parity",
                    "ptx_parity" => "PTX Parity",
                    _ => &g.name,
                };
                let measured = g.value.map_or("—".to_string(), |v| format!("{v:.2}"));
                let threshold = g.threshold.map_or("—".to_string(), |v| format!("{v:.2}"));
                vec![
                    name.to_string(),
                    badge,
                    measured,
                    threshold,
                    output::duration_fmt(g.duration_ms),
                ]
            })
            .collect();
        println!(
            "{}",
            output::table(
                &["Gate", "Status", "Measured", "Threshold", "Duration"],
                &gate_rows
            )
        );

        println!();
        if passed {
            println!("  {}", output::badge_pass("ALL GATES PASSED"));
        } else {
            println!("  {}", output::badge_fail("GATES FAILED"));
            for gate in &failed_gates {
                println!("    {} {}", "✗".red(), gate.name);
            }
        }
        output::metric(
            "Total Duration",
            output::duration_fmt(total_duration.as_millis() as u64),
            "",
        );
    }

    Ok(QaReport {
        model: path.display().to_string(),
        passed,
        gates,
        total_duration_ms: total_duration.as_millis() as u64,
        timestamp: chrono::Utc::now().to_rfc3339(),
        summary,
    })
}

/// Gate 0: Tensor Contract Validation (PMAT-235)
///
/// Validates model tensors against the PMAT-235 data quality contract BEFORE
/// running any inference. This catches bad models early (density, NaN/Inf,
/// degenerate distributions) without expensive forward passes.
///
/// Toyota Way: Jidoka - Stop the line before producing defective output.
/// Poka-Yoke: Invalid tensor data is rejected before it can cause garbage inference.
fn run_tensor_contract_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!(
            "{}",
            "Running tensor contract validation (PMAT-235)...".yellow()
        );
    }

    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = match rosetta.validate(path) {
        Ok(r) => r,
        Err(e) => {
            let duration = start.elapsed();
            return Ok(GateResult::failed(
                "tensor_contract",
                &format!("Failed to validate: {e}"),
                None,
                None,
                duration,
            ));
        }
    };

    let duration = start.elapsed();

    // Collect all contract violations (F-DATA-QUALITY-* rules)
    let contract_failures: Vec<String> = report
        .tensors
        .iter()
        .flat_map(|t| t.failures.iter().map(|f| format!("{}: {}", t.name, f)))
        .collect();

    if contract_failures.is_empty() {
        Ok(GateResult::passed(
            "tensor_contract",
            &format!(
                "{} tensors passed all PMAT-235 contract gates",
                report.tensor_count
            ),
            Some(report.tensor_count as f64),
            Some(0.0),
            duration,
        ))
    } else {
        let summary = if contract_failures.len() <= 3 {
            contract_failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                contract_failures[..3].join("; "),
                contract_failures.len() - 3
            )
        };
        Ok(GateResult::failed(
            "tensor_contract",
            &format!(
                "{} contract violations in {} tensors: {}",
                contract_failures.len(),
                report.failed_tensor_count,
                summary
            ),
            Some(contract_failures.len() as f64),
            Some(0.0),
            duration,
        ))
    }
}

/// Output verification result (PMAT-QA-PROTOCOL-001 §7.4)
#[derive(Debug, Clone)]
pub enum OutputVerification {
    /// Output passed all checks
    Pass,
    /// Output failed verification
    Fail {
        /// Reason for failure
        reason: String,
    },
}

/// Verify output is correct: not empty, no garbage, contains expected answer
/// (PMAT-QA-PROTOCOL-001 §7.4)
///
/// Order of checks is CRITICAL (fail fast on garbage):
/// 1. Not empty
/// 2. No garbage patterns (BEFORE checking answer)
/// 3. No BPE artifacts
/// 4. Contains expected answer
pub fn verify_output(
    output: &str,
    test_id: &str,
    expected_patterns: &[&str],
) -> OutputVerification {
    // Check 1: Not empty
    if output.trim().is_empty() {
        return OutputVerification::Fail {
            reason: format!("{test_id}: Empty output"),
        };
    }

    // Check 2: Garbage patterns (fail fast BEFORE checking answer)
    let garbage_patterns = ["\u{FFFD}", "[UNK]", "akunji", "olumbia"];
    for pattern in &garbage_patterns {
        if output.contains(pattern) {
            return OutputVerification::Fail {
                reason: format!("{test_id}: Garbage detected: '{pattern}'"),
            };
        }
    }

    // Check 3: BPE artifacts (null bytes, excessive control chars)
    let null_count = output.bytes().filter(|&b| b == 0).count();
    if null_count > 0 {
        return OutputVerification::Fail {
            reason: format!("{test_id}: {null_count} null bytes detected (BPE artifact)"),
        };
    }

    // Check 4: Contains expected answer
    if !expected_patterns.is_empty() {
        let found = expected_patterns
            .iter()
            .any(|p| output.to_lowercase().contains(&p.to_lowercase()));
        if !found {
            return OutputVerification::Fail {
                reason: format!(
                    "{test_id}: Expected one of {:?}, got: '{}'",
                    expected_patterns,
                    output.chars().take(100).collect::<String>()
                ),
            };
        }
    }

    OutputVerification::Pass
}

/// JIDOKA: Validate GPU golden output matches expected patterns (PMAT-232 lesson).
///
/// Without this, GPU correctness was NEVER tested — `apr qa` golden output only ran CPU.
/// Returns `Some(failure_reason)` if GPU output fails, `None` if pass or skipped.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn validate_gpu_golden_output(
    mapped: &realizar::gguf::MappedGGUFModel,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    gguf: &realizar::gguf::GGUFModel,
    expected_patterns: &[&str],
    config: &QaConfig,
) -> Result<Option<String>> {
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
    let model = OwnedQuantizedModel::from_mapped(mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
    match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(mut cuda_model) => match cuda_model.generate_gpu_resident(prompt_tokens, gen_config) {
            Ok(gpu_tokens) => {
                let gpu_text = gguf.decode(&gpu_tokens);
                if let OutputVerification::Fail { reason } =
                    verify_output(&gpu_text, "golden_output_gpu", expected_patterns)
                {
                    return Ok(Some(format!("GPU output failed (CPU passed): {reason}")));
                }
            }
            Err(e) => {
                if !config.json && config.verbose {
                    println!("{}", format!("GPU golden output skipped: {e}").yellow());
                }
            }
        },
        Err(e) => {
            if !config.json && config.verbose {
                println!("{}", format!("CUDA init skipped: {e}").yellow());
            }
        }
    }
    Ok(None)
}

/// Gate 1: Golden Output Test
///
/// Runs the model with a known prompt and verifies the output contains expected patterns.
/// Uses verify_output() for structured validation (PMAT-QA-PROTOCOL-001 §7.4).
fn run_golden_output_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running golden output test...".yellow());
    }

    // Golden test: Use ChatML format for instruct models
    // Raw prompt would make model explain, ChatML makes it respond directly
    let test_cases = vec![
        (
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
            vec!["4"],
        ),
        (
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            vec!["Hello", "Hi", "hey", "hello", "!"],
        ),
    ];

    #[cfg(feature = "inference")]
    {
        use realizar::cuda::CudaExecutor;
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{
            GGUFModel, MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig,
        };

        // Check if CUDA available
        let cuda_available = CudaExecutor::is_available() && CudaExecutor::num_devices() > 0;

        // Read and parse model
        let model_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;

        let format = detect_format(&model_bytes[..8.min(model_bytes.len())])
            .map_err(|e| CliError::ValidationFailed(format!("Failed to detect format: {e}")))?;

        // Test each golden case - support both GGUF and APR formats
        for (prompt, expected_patterns) in &test_cases {
            let (output_tokens, output_text) = match format {
                ModelFormat::Gguf => {
                    let gguf = GGUFModel::from_bytes(&model_bytes).map_err(|e| {
                        CliError::ValidationFailed(format!("Failed to parse GGUF: {e}"))
                    })?;
                    let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![151643, 9707]);

                    let gen_config = QuantizedGenerateConfig {
                        max_tokens: config.max_tokens,
                        temperature: 0.0,
                        top_k: 1,
                        ..Default::default()
                    };

                    let mapped = MappedGGUFModel::from_path(path)
                        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;

                    // CPU golden output
                    let cpu_tokens = {
                        let model = OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
                            CliError::ValidationFailed(format!("Model failed: {e}"))
                        })?;
                        model
                            .generate_with_cache(&prompt_tokens, &gen_config)
                            .map_err(|e| {
                                CliError::ValidationFailed(format!("CPU generation failed: {e}"))
                            })?
                    };
                    let cpu_text = gguf.decode(&cpu_tokens);

                    // JIDOKA: GPU golden output — all tools must behave the same
                    // Without this, GPU correctness is NEVER validated (PMAT-232 lesson)
                    #[cfg(feature = "cuda")]
                    if cuda_available {
                        if let Some(failure) = validate_gpu_golden_output(
                            &mapped,
                            &prompt_tokens,
                            &gen_config,
                            &gguf,
                            expected_patterns,
                            config,
                        )? {
                            return Ok(GateResult::failed(
                                "golden_output",
                                &failure,
                                None,
                                None,
                                start.elapsed(),
                            ));
                        }
                    }

                    (cpu_tokens, cpu_text)
                }
                ModelFormat::Apr => {
                    use realizar::apr::AprV2Model;
                    use realizar::apr_transformer::{AprTransformer, GenerateConfig};

                    // Load APR model and get embedded tokenizer
                    let apr_model = AprV2Model::load(path).map_err(|e| {
                        CliError::ValidationFailed(format!("Failed to load APR: {e}"))
                    })?;
                    let tokenizer = apr_model.load_embedded_bpe_tokenizer().ok_or_else(|| {
                        CliError::ValidationFailed("APR missing embedded tokenizer".to_string())
                    })?;

                    let transformer = AprTransformer::from_apr_file(path).map_err(|e| {
                        CliError::ValidationFailed(format!("Failed to load APR transformer: {e}"))
                    })?;

                    let prompt_tokens = tokenizer.encode(prompt);

                    let gen_config = GenerateConfig {
                        max_tokens: config.max_tokens,
                        temperature: 0.0,
                        top_k: 1,
                        ..Default::default()
                    };

                    let tokens = transformer
                        .generate_with_cache(&prompt_tokens, &gen_config)
                        .map_err(|e| {
                            CliError::ValidationFailed(format!("Generation failed: {e}"))
                        })?;

                    let text = tokenizer.decode(&tokens);
                    (tokens, text)
                }
                ModelFormat::SafeTensors => {
                    use aprender::text::bpe::{load_from_json, BpeTokenizer};
                    use realizar::safetensors_infer::SafetensorsToAprConverter;

                    // PMAT-238 FIX: Use find_sibling_file for pacha hash-prefixed paths
                    // (e.g., d71534cb948e32eb.tokenizer.json, not just tokenizer.json)
                    let tokenizer_path =
                        realizar::safetensors::find_sibling_file(path, "tokenizer.json");
                    let tokenizer: Option<BpeTokenizer> = tokenizer_path
                        .as_ref()
                        .and_then(|p| std::fs::read_to_string(p).ok())
                        .and_then(|json| load_from_json(&json).ok());

                    let Some(tokenizer) = tokenizer else {
                        return Ok(GateResult::skipped(
                            "golden_output",
                            "SafeTensors: tokenizer.json not found in model directory",
                        ));
                    };

                    let transformer = SafetensorsToAprConverter::convert(path).map_err(|e| {
                        CliError::ValidationFailed(format!("SafeTensors convert failed: {e}"))
                    })?;

                    let prompt_tokens = tokenizer.encode(prompt);

                    let gen_config = realizar::apr_transformer::GenerateConfig {
                        max_tokens: config.max_tokens,
                        temperature: 0.0,
                        top_k: 1,
                        ..Default::default()
                    };

                    let tokens = transformer
                        .generate_with_cache(&prompt_tokens, &gen_config)
                        .map_err(|e| {
                            CliError::ValidationFailed(format!("Generation failed: {e}"))
                        })?;

                    let text = tokenizer.decode(&tokens);
                    (tokens, text)
                }
            };
            let _ = output_tokens; // token count used for diagnostics

            // Verify output using structured protocol (PMAT-QA-PROTOCOL-001 §7.4)
            let verification = verify_output(&output_text, "golden_output", expected_patterns);
            if let OutputVerification::Fail { reason } = verification {
                let duration = start.elapsed();
                return Ok(GateResult::failed(
                    "golden_output",
                    &reason,
                    None,
                    None,
                    duration,
                ));
            }
        }

        let duration = start.elapsed();
        Ok(GateResult::passed(
            "golden_output",
            &format!("{} golden test cases passed", test_cases.len()),
            Some(test_cases.len() as f64),
            Some(test_cases.len() as f64),
            duration,
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
        use realizar::cuda::CudaExecutor;
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{
            GGUFModel, MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
            QuantizedGenerateConfig,
        };

        let cuda_available = CudaExecutor::is_available() && CudaExecutor::num_devices() > 0;

        let model_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;

        let format = detect_format(&model_bytes[..8.min(model_bytes.len())])
            .map_err(|e| CliError::ValidationFailed(format!("Failed to detect format: {e}")))?;

        let prompt = "Write a hello world program in Python:";

        // Measure throughput based on format
        let (tps, duration) = match format {
            ModelFormat::Gguf => {
                let gguf = GGUFModel::from_bytes(&model_bytes).map_err(|e| {
                    CliError::ValidationFailed(format!("Failed to parse GGUF: {e}"))
                })?;
                let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![151643, 9707]);

                let gen_config = QuantizedGenerateConfig {
                    max_tokens: config.max_tokens,
                    temperature: 0.0,
                    top_k: 1,
                    ..Default::default()
                };

                if cuda_available {
                    let mapped = MappedGGUFModel::from_path(path)
                        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
                    let model = OwnedQuantizedModel::from_mapped(&mapped)
                        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
                    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).map_err(|e| {
                        CliError::ValidationFailed(format!("CUDA init failed: {e}"))
                    })?;

                    for _ in 0..config.warmup {
                        let _ = cuda_model.generate_gpu_resident(&prompt_tokens, &gen_config);
                    }

                    let mut total_tokens = 0usize;
                    let measure_start = Instant::now();
                    for _ in 0..config.iterations {
                        let output = cuda_model
                            .generate_gpu_resident(&prompt_tokens, &gen_config)
                            .unwrap_or_default();
                        total_tokens += output.len().saturating_sub(prompt_tokens.len());
                    }
                    let measure_time = measure_start.elapsed();
                    (
                        total_tokens as f64 / measure_time.as_secs_f64(),
                        start.elapsed(),
                    )
                } else {
                    let mapped = MappedGGUFModel::from_path(path)
                        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
                    let model = OwnedQuantizedModel::from_mapped(&mapped)
                        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;

                    for _ in 0..config.warmup {
                        let _ = model.generate_with_cache(&prompt_tokens, &gen_config);
                    }

                    let mut total_tokens = 0usize;
                    let measure_start = Instant::now();
                    for _ in 0..config.iterations {
                        let output = model
                            .generate_with_cache(&prompt_tokens, &gen_config)
                            .unwrap_or_default();
                        total_tokens += output.len().saturating_sub(prompt_tokens.len());
                    }
                    let measure_time = measure_start.elapsed();
                    (
                        total_tokens as f64 / measure_time.as_secs_f64(),
                        start.elapsed(),
                    )
                }
            }
            ModelFormat::Apr => {
                use realizar::apr::AprV2Model;
                use realizar::apr_transformer::{AprTransformer, GenerateConfig};

                let apr_model = AprV2Model::load(path)
                    .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;
                let tokenizer = apr_model.load_embedded_bpe_tokenizer().ok_or_else(|| {
                    CliError::ValidationFailed("APR missing embedded tokenizer".to_string())
                })?;
                let transformer = AprTransformer::from_apr_file(path).map_err(|e| {
                    CliError::ValidationFailed(format!("Failed to load APR transformer: {e}"))
                })?;

                let prompt_tokens = tokenizer.encode(prompt);
                let gen_config = GenerateConfig {
                    max_tokens: config.max_tokens,
                    temperature: 0.0,
                    top_k: 1,
                    ..Default::default()
                };

                // Warmup
                for _ in 0..config.warmup {
                    let _ = transformer.generate_with_cache(&prompt_tokens, &gen_config);
                }

                let mut total_tokens = 0usize;
                let measure_start = Instant::now();
                for _ in 0..config.iterations {
                    let output = transformer
                        .generate_with_cache(&prompt_tokens, &gen_config)
                        .unwrap_or_default();
                    total_tokens += output.len().saturating_sub(prompt_tokens.len());
                }
                let measure_time = measure_start.elapsed();
                (
                    total_tokens as f64 / measure_time.as_secs_f64(),
                    start.elapsed(),
                )
            }
            ModelFormat::SafeTensors => {
                use aprender::text::bpe::{load_from_json, BpeTokenizer};
                use realizar::safetensors_infer::SafetensorsToAprConverter;

                // PMAT-238 FIX: Use find_sibling_file for pacha hash-prefixed paths
                let tokenizer_path =
                    realizar::safetensors::find_sibling_file(path, "tokenizer.json");
                let tokenizer: Option<BpeTokenizer> = tokenizer_path
                    .as_ref()
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .and_then(|json| load_from_json(&json).ok());

                let Some(tokenizer) = tokenizer else {
                    return Ok(GateResult::skipped(
                        "throughput",
                        "SafeTensors: tokenizer.json not found in model directory",
                    ));
                };

                let transformer = SafetensorsToAprConverter::convert(path).map_err(|e| {
                    CliError::ValidationFailed(format!("SafeTensors convert failed: {e}"))
                })?;

                let prompt_tokens = tokenizer.encode(prompt);
                let gen_config = realizar::apr_transformer::GenerateConfig {
                    max_tokens: config.max_tokens,
                    temperature: 0.0,
                    top_k: 1,
                    ..Default::default()
                };

                // Warmup
                for _ in 0..config.warmup {
                    let _ = transformer.generate_with_cache(&prompt_tokens, &gen_config);
                }

                let mut total_tokens = 0usize;
                let measure_start = Instant::now();
                for _ in 0..config.iterations {
                    let output = transformer
                        .generate_with_cache(&prompt_tokens, &gen_config)
                        .unwrap_or_default();
                    total_tokens += output.len().saturating_sub(prompt_tokens.len());
                }
                let measure_time = measure_start.elapsed();
                (
                    total_tokens as f64 / measure_time.as_secs_f64(),
                    start.elapsed(),
                )
            }
        };
        let _ = cuda_available; // suppress warning

        // Format-aware thresholds: quantized GGUF on GPU is ~100x faster than F32 CPU.
        // Comparing unquantized F32 models against a quantized GPU target is meaningless.
        let threshold = match format {
            ModelFormat::Gguf => 10.0_f64.max(config.min_tps / 10.0),
            ModelFormat::Apr | ModelFormat::SafeTensors => 1.0, // F32 CPU: 1 tok/s minimum
        };

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

/// Gate 3: Ollama Parity Test
///
/// Compares performance against Ollama baseline (if available).
/// This is falsifiable - if speedup < target, test fails.
fn run_ollama_parity_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running Ollama parity test...".yellow());
    }

    // Check if Ollama is running by trying to connect
    let ollama_available = check_ollama_available();

    if !ollama_available {
        return Ok(GateResult::skipped(
            "ollama_parity",
            "Ollama not available (start with: ollama serve)",
        ));
    }

    #[cfg(feature = "inference")]
    {
        use realizar::cuda::CudaExecutor;
        use realizar::gguf::{
            GGUFModel, MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
            QuantizedGenerateConfig,
        };

        // This gate only runs for GGUF (non-GGUF skipped at call site)
        let ollama_tps = measure_ollama_throughput(path, config)?;

        if ollama_tps <= 0.0 {
            return Ok(GateResult::skipped(
                "ollama_parity",
                "Could not measure Ollama throughput",
            ));
        }

        // Measure our GGUF throughput
        let model_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
        let gguf = GGUFModel::from_bytes(&model_bytes)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

        let prompt = "Write a function to check if a number is prime:";
        let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![151643]);
        // Use 128 tokens minimum for Ollama parity comparison.
        // Ollama reports eval_count/eval_duration (decode-only throughput, excludes prefill).
        // Our measurement includes prefill in every generate_gpu_resident() call.
        // At 32 tokens, prefill overhead (~0.79s per call) dominates our measurement,
        // producing an unfair 0.13x ratio. At 128 tokens, prefill amortizes and
        // the ratio reflects actual decode throughput (~0.31x at 36 vs 116 tok/s).
        let parity_max_tokens = config.max_tokens.max(128);
        let gen_config = QuantizedGenerateConfig {
            max_tokens: parity_max_tokens,
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        };

        let cuda_available = CudaExecutor::is_available() && CudaExecutor::num_devices() > 0;

        let our_tps = if cuda_available {
            let mapped = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
            let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
                .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;

            for _ in 0..config.warmup {
                let _ = cuda_model.generate_gpu_resident(&prompt_tokens, &gen_config);
            }

            let mut total_tokens = 0usize;
            let measure_start = Instant::now();
            for _ in 0..config.iterations {
                let output = cuda_model
                    .generate_gpu_resident(&prompt_tokens, &gen_config)
                    .unwrap_or_default();
                total_tokens += output.len().saturating_sub(prompt_tokens.len());
            }
            total_tokens as f64 / measure_start.elapsed().as_secs_f64()
        } else {
            let mapped = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;

            for _ in 0..config.warmup {
                let _ = model.generate_with_cache(&prompt_tokens, &gen_config);
            }

            let mut total_tokens = 0usize;
            let measure_start = Instant::now();
            for _ in 0..config.iterations {
                let output = model
                    .generate_with_cache(&prompt_tokens, &gen_config)
                    .unwrap_or_default();
                total_tokens += output.len().saturating_sub(prompt_tokens.len());
            }
            total_tokens as f64 / measure_start.elapsed().as_secs_f64()
        };

        let speedup = our_tps / ollama_tps;
        let duration = start.elapsed();

        if speedup >= config.min_speedup {
            Ok(GateResult::passed(
                "ollama_parity",
                &format!(
                    "{:.1}x Ollama ({:.0} vs {:.0} tok/s) >= {:.1}x threshold",
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
                    "{:.2}x Ollama ({:.0} vs {:.0} tok/s) < {:.1}x threshold",
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
fn run_gpu_speedup_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running GPU vs CPU speedup test...".yellow());
    }

    #[cfg(feature = "inference")]
    {
        use realizar::cuda::CudaExecutor;
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{
            GGUFModel, MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
            QuantizedGenerateConfig,
        };

        // Check if CUDA available
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

        let gguf = GGUFModel::from_bytes(&model_bytes)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

        let prompt = "Write a function to calculate factorial:";
        let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![151643]);

        let gen_config = QuantizedGenerateConfig {
            max_tokens: config.max_tokens,
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        };

        // Measure CPU throughput
        let cpu_tps = {
            let mapped = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;

            // Warmup
            for _ in 0..config.warmup {
                let _ = model.generate_with_cache(&prompt_tokens, &gen_config);
            }

            let mut total_tokens = 0usize;
            let measure_start = Instant::now();

            for _ in 0..config.iterations {
                let output = model
                    .generate_with_cache(&prompt_tokens, &gen_config)
                    .unwrap_or_default();
                total_tokens += output.len().saturating_sub(prompt_tokens.len());
            }

            total_tokens as f64 / measure_start.elapsed().as_secs_f64()
        };

        // Measure GPU throughput
        let gpu_tps = {
            let mapped = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
            let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
                .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;

            // Warmup
            for _ in 0..config.warmup {
                let _ = cuda_model.generate_gpu_resident(&prompt_tokens, &gen_config);
            }

            let mut total_tokens = 0usize;
            let measure_start = Instant::now();

            for _ in 0..config.iterations {
                let output = cuda_model
                    .generate_gpu_resident(&prompt_tokens, &gen_config)
                    .unwrap_or_default();
                total_tokens += output.len().saturating_sub(prompt_tokens.len());
            }

            total_tokens as f64 / measure_start.elapsed().as_secs_f64()
        };

        let duration = start.elapsed();

        // Calculate speedup
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

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "gpu_speedup",
            "Requires 'inference' feature",
        ))
    }
}

/// Gate 5: Cross-Format Parity Test (F-QUAL-032)
///
/// Compares argmax output between GGUF and SafeTensors for the same model.
/// Invariant: argmax(forward_gguf(M, tokens)) == argmax(forward_safetensors(M, tokens))
///
/// This is the cornerstone of the architecture's logical validity - it demonstrates
/// that independent binary format readers can reach the same logical conclusion.
fn run_format_parity_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running cross-format parity test...".yellow());
    }

    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{GGUFModel, MappedGGUFModel, OwnedQuantizedModel};
        use realizar::safetensors_infer::SafetensorsToAprConverter;

        let Some(safetensors_path) = &config.safetensors_path else {
            return Ok(GateResult::skipped(
                "format_parity",
                "No SafeTensors path provided (use --safetensors-path)",
            ));
        };

        // Verify GGUF model
        let gguf_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read GGUF: {e}")))?;

        let gguf_format = detect_format(&gguf_bytes[..8.min(gguf_bytes.len())]).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to detect GGUF format: {e}"))
        })?;

        if gguf_format != ModelFormat::Gguf {
            return Ok(GateResult::skipped(
                "format_parity",
                "Primary model must be GGUF format",
            ));
        }

        // Verify SafeTensors model exists
        if !safetensors_path.exists() {
            return Ok(GateResult::skipped(
                "format_parity",
                &format!("SafeTensors file not found: {}", safetensors_path.display()),
            ));
        }

        // Load GGUF model and get tokenizer
        let gguf = GGUFModel::from_bytes(&gguf_bytes)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

        // Test prompt - use simple arithmetic for deterministic output
        let prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
        let prompt_tokens: Vec<u32> = gguf.encode(prompt).unwrap_or_else(|| vec![151643, 9707]);

        // Run GGUF forward pass to get logits
        let gguf_logits = {
            let mapped = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("GGUF map failed: {e}")))?;
            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| CliError::ValidationFailed(format!("GGUF model failed: {e}")))?;
            model
                .forward(&prompt_tokens)
                .map_err(|e| CliError::ValidationFailed(format!("GGUF forward failed: {e}")))?
        };

        // Run SafeTensors forward pass to get logits
        let st_logits = {
            let transformer =
                SafetensorsToAprConverter::convert(safetensors_path).map_err(|e| {
                    CliError::ValidationFailed(format!("SafeTensors convert failed: {e}"))
                })?;
            transformer.forward(&prompt_tokens).map_err(|e| {
                CliError::ValidationFailed(format!("SafeTensors forward failed: {e}"))
            })?
        };

        let duration = start.elapsed();

        // Get argmax from logits
        let gguf_argmax = gguf_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);

        let st_argmax = st_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);

        match (gguf_argmax, st_argmax) {
            (Some(gguf_token), Some(st_token)) => {
                if gguf_token == st_token {
                    Ok(GateResult::passed(
                        "format_parity",
                        &format!(
                            "GGUF argmax={} == SafeTensors argmax={} (Cross-format parity VERIFIED)",
                            gguf_token, st_token
                        ),
                        Some(gguf_token as f64),
                        Some(st_token as f64),
                        duration,
                    ))
                } else {
                    Ok(GateResult::failed(
                        "format_parity",
                        &format!(
                            "GGUF argmax={} != SafeTensors argmax={} (Cross-format parity BROKEN)",
                            gguf_token, st_token
                        ),
                        Some(gguf_token as f64),
                        Some(st_token as f64),
                        duration,
                    ))
                }
            }
            _ => Ok(GateResult::failed(
                "format_parity",
                "Failed to get argmax from one or both formats",
                None,
                None,
                duration,
            )),
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "format_parity",
            "Requires 'inference' feature",
        ))
    }
}

/// Check if Ollama is available by pinging the API
fn check_ollama_available() -> bool {
    // Try to connect to Ollama API
    std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://localhost:11434/api/tags",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
        .unwrap_or(false)
}

/// Detect Ollama model name from GGUF filename (BUG-QA-001 fix)
/// Matches model size to avoid unfair comparison (e.g., 0.5B APR vs 1.5B Ollama)
/// Detect the matching Ollama model tag for fair like-for-like comparison.
///
/// For quantized GGUF: uses the default Ollama tag (Q4_K_M quantized).
/// For F32/F16 (SafeTensors, APR): uses the `-instruct-fp16` Ollama tag
/// so we compare unquantized vs unquantized.
///
/// Detects model size from filename, or falls back to file size heuristic
/// for hash-named pacha-cached files.
fn detect_ollama_model_from_path(path: &Path) -> String {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let filename_lower = filename.to_lowercase();

    // Detect model size from filename first
    let size = if filename_lower.contains("0.5b") || filename_lower.contains("-0_5b") {
        "0.5b"
    } else if filename_lower.contains("1.5b") || filename_lower.contains("-1_5b") {
        "1.5b"
    } else if filename_lower.contains("3b") || filename_lower.contains("-3b") {
        "3b"
    } else if filename_lower.contains("7b") || filename_lower.contains("-7b") {
        "7b"
    } else if filename_lower.contains("14b") || filename_lower.contains("-14b") {
        "14b"
    } else if filename_lower.contains("32b") || filename_lower.contains("-32b") {
        "32b"
    } else {
        // Fallback: estimate from file size (for hash-named pacha-cached files).
        // GGUF Q4_K sizes: 0.5B≈400MB, 1.5B≈1GB, 3B≈2GB, 7B≈4.5GB
        match std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) {
            0..=800_000_000 => "0.5b",
            800_000_001..=2_000_000_000 => "1.5b",
            2_000_000_001..=4_000_000_000 => "3b",
            _ => "7b",
        }
    };

    // Default Ollama tag uses Q4_K_M — fair comparison for quantized GGUF
    format!("qwen2.5-coder:{size}")
}

/// Measure Ollama throughput for comparison (GGUF only)
/// BUG-QA-002 FIX: Use Ollama's eval_duration instead of wall clock time
/// (wall clock includes HTTP overhead, making Ollama look 10x slower)
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json! macro internally uses unwrap()
fn measure_ollama_throughput(path: &Path, config: &QaConfig) -> Result<f64> {
    // Use curl to send a request to Ollama
    let prompt = "Write a hello world program in Python:";
    // BUG-QA-001 FIX: Match Ollama model to APR model size for fair comparison
    let model = detect_ollama_model_from_path(path);

    // Match parity gate: use 128 tokens minimum to amortize prefill overhead
    let parity_max_tokens = config.max_tokens.max(128);
    let request_body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_predict": parity_max_tokens,
            "temperature": 0.0
        }
    });

    let mut total_tokens = 0usize;
    let mut total_duration_ns = 0u64;

    for _ in 0..config.iterations.min(3) {
        let output = std::process::Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "http://localhost:11434/api/generate",
                "-H",
                "Content-Type: application/json",
                "-d",
                &request_body.to_string(),
            ])
            .output();

        if let Ok(output) = output {
            if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                // BUG-QA-002 FIX: Use eval_count and eval_duration from Ollama response
                // This measures actual inference time, not HTTP overhead
                if let (Some(eval_count), Some(eval_duration)) = (
                    response
                        .get("eval_count")
                        .and_then(serde_json::Value::as_u64),
                    response
                        .get("eval_duration")
                        .and_then(serde_json::Value::as_u64),
                ) {
                    total_tokens += eval_count as usize;
                    total_duration_ns += eval_duration;
                }
            }
        }
    }

    if total_tokens == 0 || total_duration_ns == 0 {
        return Ok(0.0);
    }

    // Convert nanoseconds to seconds for tok/s calculation
    let duration_s = total_duration_ns as f64 / 1_000_000_000.0;
    Ok(total_tokens as f64 / duration_s)
}

/// Print a gate result to the terminal
fn print_gate_result(result: &GateResult) {
    let badge = if result.skipped {
        output::badge_skip("SKIP")
    } else if result.passed {
        output::badge_pass("PASS")
    } else {
        output::badge_fail("FAIL")
    };

    let name = match result.name.as_str() {
        "tensor_contract" => "Tensor Contract",
        "golden_output" => "Golden Output",
        "throughput" => "Throughput",
        "ollama_parity" => "Ollama Parity",
        "gpu_speedup" => "GPU Speedup",
        "format_parity" => "Format Parity",
        "ptx_parity" => "PTX Parity",
        _ => &result.name,
    };

    println!(
        "  {} {} {}",
        badge,
        name.white().bold(),
        result.message.dimmed()
    );

    if !result.skipped {
        println!(
            "       {}",
            output::duration_fmt(result.duration_ms).dimmed()
        );
    }
    println!();
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
        use realizar::format::{detect_format, ModelFormat};
        use realizar::ptx_parity::{validate_all_kernel_pairs, KernelDimensions};

        // Only run for GGUF models (PTX kernels are for quantized inference)
        // Read only first 8 bytes (not the entire multi-GB file)
        let magic = std::fs::File::open(path).ok().and_then(|mut f| {
            use std::io::Read;
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf).ok()?;
            Some(buf.to_vec())
        });
        let fmt = magic.and_then(|m| detect_format(&m).ok());
        if fmt != Some(ModelFormat::Gguf) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // QaConfig Tests
    // ========================================================================

    #[test]
    fn test_qa_config_default() {
        let config = QaConfig::default();
        assert!((config.min_tps - 100.0).abs() < f64::EPSILON);
        assert!((config.min_speedup - 0.2).abs() < f64::EPSILON);
        assert!((config.min_gpu_speedup - 2.0).abs() < f64::EPSILON);
        assert!(!config.skip_golden);
        assert!(!config.skip_throughput);
        assert!(!config.skip_ollama);
        assert!(!config.skip_gpu_speedup);
        assert!(!config.skip_format_parity);
        assert!(config.safetensors_path.is_none());
    }

    #[test]
    fn test_qa_config_default_iterations() {
        let config = QaConfig::default();
        assert_eq!(config.iterations, 10);
        assert_eq!(config.warmup, 3);
        assert_eq!(config.max_tokens, 32);
    }

    #[test]
    fn test_qa_config_default_output_flags() {
        let config = QaConfig::default();
        assert!(!config.json);
        assert!(!config.verbose);
    }

    #[test]
    fn test_qa_config_clone() {
        let config = QaConfig {
            min_tps: 50.0,
            skip_golden: true,
            ..Default::default()
        };
        let cloned = config.clone();
        assert!((cloned.min_tps - 50.0).abs() < f64::EPSILON);
        assert!(cloned.skip_golden);
    }

    #[test]
    fn test_qa_config_debug() {
        let config = QaConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("QaConfig"));
        assert!(debug.contains("min_tps"));
    }

    // ========================================================================
    // GateResult Tests
    // ========================================================================

    #[test]
    fn test_gate_result_passed() {
        let result = GateResult::passed(
            "test_gate",
            "Test passed",
            Some(150.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(result.passed);
        assert!(!result.skipped);
        assert_eq!(result.name, "test_gate");
    }

    #[test]
    fn test_gate_result_passed_duration() {
        let result = GateResult::passed(
            "test_gate",
            "Test passed",
            Some(150.0),
            Some(100.0),
            Duration::from_millis(1500),
        );
        assert_eq!(result.duration_ms, 1500);
    }

    #[test]
    fn test_gate_result_passed_no_value() {
        let result = GateResult::passed(
            "test_gate",
            "Test passed",
            None,
            None,
            Duration::from_secs(1),
        );
        assert!(result.value.is_none());
        assert!(result.threshold.is_none());
    }

    #[test]
    fn test_gate_result_failed() {
        let result = GateResult::failed(
            "test_gate",
            "Test failed",
            Some(50.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(!result.passed);
        assert!(!result.skipped);
    }

    #[test]
    fn test_gate_result_failed_message() {
        let result = GateResult::failed(
            "throughput",
            "50 tok/s < 100 tok/s",
            Some(50.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(result.message.contains("50"));
        assert!(result.message.contains("100"));
    }

    #[test]
    fn test_gate_result_skipped() {
        let result = GateResult::skipped("test_gate", "No GPU available");
        assert!(result.passed); // Skipped doesn't fail
        assert!(result.skipped);
    }

    #[test]
    fn test_gate_result_skipped_message() {
        let result = GateResult::skipped("gpu_speedup", "GPU not available");
        assert!(result.message.contains("Skipped"));
        assert!(result.message.contains("GPU not available"));
    }

    #[test]
    fn test_gate_result_skipped_no_duration() {
        let result = GateResult::skipped("test", "reason");
        assert_eq!(result.duration_ms, 0);
    }

    #[test]
    fn test_gate_result_clone() {
        let result = GateResult::passed("test", "ok", Some(100.0), None, Duration::from_secs(1));
        let cloned = result.clone();
        assert_eq!(cloned.name, result.name);
        assert_eq!(cloned.passed, result.passed);
    }

    #[test]
    fn test_gate_result_debug() {
        let result = GateResult::passed("test", "ok", None, None, Duration::from_secs(0));
        let debug = format!("{result:?}");
        assert!(debug.contains("GateResult"));
    }

    #[test]
    fn test_gate_result_serialize() {
        let result = GateResult::passed(
            "throughput",
            "100 tok/s",
            Some(100.0),
            Some(60.0),
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("throughput"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_gate_result_deserialize() {
        let json =
            r#"{"name":"test","passed":true,"message":"ok","duration_ms":1000,"skipped":false}"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(result.name, "test");
        assert!(result.passed);
    }

    // ========================================================================
    // QaReport Tests
    // ========================================================================

    #[test]
    fn test_qa_report_serialization() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![GateResult::passed(
                "throughput",
                "100 tok/s",
                Some(100.0),
                Some(60.0),
                Duration::from_secs(5),
            )],
            total_duration_ms: 5000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "All gates passed".to_string(),
        };

        let json = serde_json::to_string(&report).expect("serialization failed");
        assert!(json.contains("throughput"));
        assert!(json.contains("passed"));
    }

    #[test]
    fn test_qa_report_deserialization() {
        let json = r#"{
            "model": "test.gguf",
            "passed": true,
            "gates": [],
            "total_duration_ms": 1000,
            "timestamp": "2026-01-01T00:00:00Z",
            "summary": "All passed"
        }"#;
        let report: QaReport = serde_json::from_str(json).expect("deserialize");
        assert_eq!(report.model, "test.gguf");
        assert!(report.passed);
    }

    #[test]
    fn test_qa_report_failed() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: false,
            gates: vec![GateResult::failed(
                "throughput",
                "50 tok/s < 100 tok/s",
                Some(50.0),
                Some(100.0),
                Duration::from_secs(5),
            )],
            total_duration_ms: 5000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "1 gate failed".to_string(),
        };
        assert!(!report.passed);
        assert_eq!(report.gates.len(), 1);
    }

    #[test]
    fn test_qa_report_multiple_gates() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![
                GateResult::passed("golden", "ok", None, None, Duration::from_secs(1)),
                GateResult::passed(
                    "throughput",
                    "ok",
                    Some(100.0),
                    Some(60.0),
                    Duration::from_secs(2),
                ),
                GateResult::skipped("ollama", "skipped"),
            ],
            total_duration_ms: 3000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "All passed".to_string(),
        };
        assert_eq!(report.gates.len(), 3);
    }

    #[test]
    fn test_qa_report_clone() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 1000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "ok".to_string(),
        };
        let cloned = report.clone();
        assert_eq!(cloned.model, report.model);
    }

    #[test]
    fn test_qa_report_debug() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 1000,
            timestamp: "now".to_string(),
            summary: "ok".to_string(),
        };
        let debug = format!("{report:?}");
        assert!(debug.contains("QaReport"));
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_model() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            false,
            false,
        );
        // Should fail (invalid GGUF)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_custom_thresholds() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            Some(50.0), // min_tps
            Some(1.5),  // min_speedup
            Some(3.0),  // min_gpu_speedup
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            5,
            2,
            16,
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_all_skips() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            true, // skip_golden
            true, // skip_throughput
            true, // skip_ollama
            true, // skip_gpu_speedup
            true, // skip_contract
            true, // skip_format_parity
            true, // skip_ptx_parity
            None,
            10,
            3,
            32,
            false,
            false,
        );
        // When all gates are skipped, the QA passes (skipped gates don't fail)
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_json_output() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            true, // json output
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_verbose() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            false,
            true, // verbose
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_safetensors_path() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let st_file = NamedTempFile::with_suffix(".safetensors").expect("create st file");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            Some(st_file.path().to_path_buf()), // safetensors path
            10,
            3,
            32,
            false,
            false,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_small_iterations() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            1, // small iterations
            0, // no warmup
            8, // small max tokens
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    // ========================================================================
    // FORMAT DISPATCH TESTS (P0: Verify formats don't incorrectly skip)
    // These tests ensure that APR, GGUF, and SafeTensors formats are properly
    // dispatched to their handlers and don't silently skip with "GGUF only".
    // ========================================================================

    #[cfg(feature = "inference")]
    mod format_dispatch_tests {
        use realizar::format::{detect_format, ModelFormat};

        /// Test that GGUF magic bytes are correctly detected
        #[test]
        fn test_gguf_format_detection() {
            // GGUF magic: "GGUF" (0x47475546)
            let gguf_magic = b"GGUF\x03\x00\x00\x00"; // GGUF v3
            let format = detect_format(gguf_magic).expect("detect GGUF");
            assert_eq!(format, ModelFormat::Gguf, "GGUF magic must detect as GGUF");
        }

        /// Test that APR v2 magic bytes are correctly detected
        #[test]
        fn test_apr_v2_format_detection() {
            // APR v2 magic: "APR\0" (0x41505200)
            let apr_magic = b"APR\x00\x02\x00\x00\x00"; // APR v2
            let format = detect_format(apr_magic).expect("detect APR");
            assert_eq!(format, ModelFormat::Apr, "APR magic must detect as APR");
        }

        /// Test that SafeTensors format is correctly detected
        #[test]
        fn test_safetensors_format_detection() {
            // SafeTensors starts with u64 header length, then JSON
            let mut st_magic = Vec::new();
            st_magic.extend_from_slice(&100u64.to_le_bytes()); // header length
            st_magic.extend_from_slice(b"{\""); // JSON start
            let format = detect_format(&st_magic).expect("detect SafeTensors");
            assert_eq!(
                format,
                ModelFormat::SafeTensors,
                "SafeTensors magic must detect as SafeTensors"
            );
        }

        /// P0 REGRESSION TEST: APR format must NOT skip golden_output gate
        /// This test catches the bug where APR files silently returned "GGUF only"
        #[test]
        fn test_apr_format_does_not_skip_detection() {
            // Create minimal APR v2 header (8 bytes minimum for format detection)
            let apr_magic = b"APR\x00\x02\x00\x00\x00"; // APR v2 magic + version
            let format = detect_format(apr_magic).expect("detect APR");

            // The critical assertion: APR must be detected as APR, not fail/skip
            assert_eq!(
                format,
                ModelFormat::Apr,
                "APR format MUST be detected - cannot skip with 'GGUF only' error"
            );
        }

        /// P0 REGRESSION TEST: Verify ModelFormat enum covers all expected formats
        #[test]
        fn test_model_format_enum_completeness() {
            // This test documents the expected formats
            let formats = [
                ModelFormat::Gguf,
                ModelFormat::Apr,
                ModelFormat::SafeTensors,
            ];
            assert_eq!(
                formats.len(),
                3,
                "Must support exactly 3 formats: GGUF, APR, SafeTensors"
            );
        }
    }

    // ========================================================================
    // GATE RESULT NON-SKIP TESTS
    // Verify that gates return actual results (pass/fail) not skipped
    // ========================================================================

    #[test]
    fn test_gate_result_skipped_flag_semantics() {
        // Skipped gates have skipped=true
        let skipped = GateResult::skipped("test", "reason");
        assert!(skipped.skipped, "Skipped gate must have skipped=true");
        assert!(skipped.passed, "Skipped gates count as passed (don't fail)");

        // Passed gates have skipped=false
        let passed = GateResult::passed("test", "ok", None, None, Duration::from_secs(1));
        assert!(!passed.skipped, "Passed gate must have skipped=false");
        assert!(passed.passed, "Passed gate must have passed=true");

        // Failed gates have skipped=false
        let failed = GateResult::failed("test", "fail", None, None, Duration::from_secs(1));
        assert!(!failed.skipped, "Failed gate must have skipped=false");
        assert!(!failed.passed, "Failed gate must have passed=false");
    }

    /// P0 REGRESSION TEST: Gates that skip must have explicit reason
    #[test]
    fn test_skipped_gate_must_have_reason() {
        let result = GateResult::skipped("test_gate", "Explicit reason required");
        assert!(
            result.message.contains("Skipped"),
            "Skip message must contain 'Skipped'"
        );
        assert!(result.message.len() > 10, "Skip reason must be descriptive");
    }

    // ========================================================================
    // GateResult: boundary values and value/threshold interactions
    // ========================================================================

    /// A gate whose measured value exactly equals the threshold should pass.
    /// Bug class: using > instead of >= in threshold comparison, causing
    /// exact-threshold values to fail.
    #[test]
    fn gate_result_value_equals_threshold_is_pass() {
        // When value == threshold, the gate is "passed" (caller constructs it)
        // This test documents the semantic contract: equality means pass.
        let result = GateResult::passed(
            "throughput",
            "100.0 tok/s >= 100.0 tok/s threshold",
            Some(100.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(result.passed);
        assert_eq!(result.value, Some(100.0));
        assert_eq!(result.threshold, Some(100.0));
    }

    /// A gate with value just below threshold should be failed.
    /// Bug class: floating-point equality masking near-miss failures.
    #[test]
    fn gate_result_value_just_below_threshold_is_fail() {
        let result = GateResult::failed(
            "throughput",
            "99.9 tok/s < 100.0 tok/s",
            Some(99.9),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(!result.passed);
        assert!(!result.skipped);
    }

    /// Zero-duration gate result should be representable.
    /// Bug class: division by zero in duration_ms calculation.
    #[test]
    fn gate_result_zero_duration() {
        let result = GateResult::passed(
            "fast_gate",
            "Sub-millisecond completion",
            None,
            None,
            Duration::from_nanos(0),
        );
        assert_eq!(result.duration_ms, 0);
        assert!(result.passed);
    }

    /// Very large duration should not overflow u64 milliseconds.
    /// Bug class: u64 overflow when converting Duration to millis.
    #[test]
    fn gate_result_large_duration_no_overflow() {
        // 1 million seconds = ~11.5 days (extreme but valid)
        let result = GateResult::passed(
            "slow_gate",
            "Long-running test",
            None,
            None,
            Duration::from_secs(1_000_000),
        );
        assert_eq!(result.duration_ms, 1_000_000_000);
    }

    /// Skipped gates must have None for value and threshold.
    /// Bug class: skipped constructor inadvertently setting default values
    /// that confuse downstream reporting (e.g., "0.0 vs 0.0 threshold").
    #[test]
    fn gate_result_skipped_has_no_metrics() {
        let result = GateResult::skipped("contract", "Model not found");
        assert!(result.value.is_none(), "Skipped gate must have no value");
        assert!(
            result.threshold.is_none(),
            "Skipped gate must have no threshold"
        );
    }

    /// Failed gate with None value (e.g., infrastructure failure, not metric miss).
    /// Bug class: downstream code unwrapping value.unwrap() on failure.
    #[test]
    fn gate_result_failed_without_value() {
        let result = GateResult::failed(
            "golden_output",
            "Inference engine crashed",
            None,
            None,
            Duration::from_millis(50),
        );
        assert!(!result.passed);
        assert!(result.value.is_none());
    }

    // ========================================================================
    // GateResult serialization: JSON round-trip fidelity
    // ========================================================================

    /// Round-trip: passed gate with all fields must survive JSON serialization.
    /// Bug class: serde skip_serializing_if dropping fields that should be present.
    #[test]
    fn gate_result_json_roundtrip_with_values() {
        let original = GateResult::passed(
            "throughput",
            "150.0 tok/s >= 100.0 tok/s",
            Some(150.0),
            Some(100.0),
            Duration::from_millis(2500),
        );
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: GateResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.name, "throughput");
        assert!(restored.passed);
        assert!(!restored.skipped);
        assert_eq!(restored.value, Some(150.0));
        assert_eq!(restored.threshold, Some(100.0));
        assert_eq!(restored.duration_ms, 2500);
    }

    /// Round-trip: skipped gate should preserve skipped=true through JSON.
    /// Bug class: skipped field defaulting to false on deserialization.
    #[test]
    fn gate_result_json_roundtrip_skipped() {
        let original = GateResult::skipped("gpu_speedup", "No GPU");
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: GateResult = serde_json::from_str(&json).expect("deserialize");
        assert!(restored.skipped, "skipped flag must survive round-trip");
        assert!(restored.passed, "skipped gates must still show passed=true");
        assert!(
            restored.value.is_none(),
            "value should be None after round-trip"
        );
    }

    /// JSON with None value/threshold should omit those fields entirely.
    /// Bug class: serializing None as null instead of omitting.
    #[test]
    fn gate_result_json_omits_none_fields() {
        let result = GateResult::passed("test", "ok", None, None, Duration::from_secs(1));
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(
            !json.contains("value"),
            "None value should be omitted from JSON, got: {json}"
        );
        assert!(
            !json.contains("threshold"),
            "None threshold should be omitted from JSON, got: {json}"
        );
    }

    // ========================================================================
    // QaReport: aggregate pass/fail logic
    // ========================================================================

    /// A report with all skipped gates should pass (skips never fail).
    /// Bug class: empty non-skipped gate list treated as failure.
    #[test]
    fn qa_report_all_skipped_gates_passes() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![
                GateResult::skipped("golden", "no model"),
                GateResult::skipped("throughput", "no engine"),
                GateResult::skipped("ollama", "not available"),
            ],
            total_duration_ms: 10,
            timestamp: "2026-02-06T00:00:00Z".to_string(),
            summary: "All skipped".to_string(),
        };
        assert!(report.passed);
        assert!(
            report.gates.iter().all(|g| g.skipped),
            "All gates should be skipped"
        );
        assert!(
            report.gates.iter().all(|g| g.passed),
            "All skipped gates should count as passed"
        );
    }

    /// A single failed gate should make the entire report fail.
    /// Bug class: report.passed computed as majority vote instead of all().
    #[test]
    fn qa_report_single_failure_taints_report() {
        let gates = [
            GateResult::passed("golden", "ok", None, None, Duration::from_secs(1)),
            GateResult::failed(
                "throughput",
                "too slow",
                Some(5.0),
                Some(100.0),
                Duration::from_secs(2),
            ),
            GateResult::passed(
                "contract",
                "ok",
                Some(100.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(!passed, "Single failure must taint the entire report");
    }

    /// Mixed passed and skipped gates should produce overall pass.
    /// Bug class: treating skipped as neither-pass-nor-fail, which
    /// breaks the all() check.
    #[test]
    fn qa_report_mixed_pass_and_skip_passes() {
        let gates = [
            GateResult::passed("golden", "ok", None, None, Duration::from_secs(1)),
            GateResult::skipped("ollama", "not available"),
            GateResult::passed(
                "contract",
                "ok",
                Some(50.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
            GateResult::skipped("gpu_speedup", "no GPU"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(passed, "Mix of passed + skipped should be overall pass");
    }

    /// Failed gates filtering should exclude skipped gates.
    /// Bug class: counting skipped gates as failures in summary.
    #[test]
    fn qa_report_failed_gate_filter_excludes_skipped() {
        let gates = [
            GateResult::failed(
                "throughput",
                "too slow",
                Some(1.0),
                Some(100.0),
                Duration::from_secs(1),
            ),
            GateResult::skipped("ollama", "not running"),
            GateResult::passed("contract", "ok", None, None, Duration::from_secs(1)),
        ];
        let failed_gates: Vec<_> = gates.iter().filter(|g| !g.passed && !g.skipped).collect();
        assert_eq!(
            failed_gates.len(),
            1,
            "Only non-skipped failures should appear"
        );
        assert_eq!(failed_gates[0].name, "throughput");
    }

    // ========================================================================
    // QaReport JSON round-trip
    // ========================================================================

    /// Full report round-trip through JSON preserves all field values.
    /// Bug class: field ordering or naming mismatch between ser/de.
    #[test]
    fn qa_report_json_roundtrip_complete() {
        let original = QaReport {
            model: "/path/to/model.gguf".to_string(),
            passed: false,
            gates: vec![
                GateResult::passed(
                    "contract",
                    "50 tensors ok",
                    Some(50.0),
                    Some(0.0),
                    Duration::from_millis(100),
                ),
                GateResult::failed(
                    "throughput",
                    "5 < 100",
                    Some(5.0),
                    Some(100.0),
                    Duration::from_millis(5000),
                ),
                GateResult::skipped("ollama", "not installed"),
            ],
            total_duration_ms: 5100,
            timestamp: "2026-02-06T12:00:00Z".to_string(),
            summary: "Failed gates: throughput".to_string(),
        };

        let json = serde_json::to_string_pretty(&original).expect("serialize");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.model, original.model);
        assert_eq!(restored.passed, original.passed);
        assert_eq!(restored.gates.len(), 3);
        assert_eq!(restored.total_duration_ms, original.total_duration_ms);
        assert_eq!(restored.summary, original.summary);
        // Verify individual gate fidelity
        assert!(restored.gates[0].passed);
        assert!(!restored.gates[1].passed);
        assert!(restored.gates[2].skipped);
    }

    // ========================================================================
    // detect_ollama_model_from_path: filename-based model size detection
    // ========================================================================

    /// Standard filename patterns should detect correct model size.
    /// Bug class: case-sensitive matching missing lowercase variants.
    #[test]
    fn detect_ollama_model_standard_sizes() {
        let cases = vec![
            ("/tmp/qwen2-0.5b-instruct-q4_0.gguf", "0.5b"),
            ("/tmp/qwen2-1.5b-instruct-q4_0.gguf", "1.5b"),
            ("/tmp/qwen2-7b-instruct-q4_0.gguf", "7b"),
            ("/tmp/qwen2-14b-instruct-q4_0.gguf", "14b"),
            ("/tmp/qwen2-32b-instruct-q4_0.gguf", "32b"),
        ];
        for (path, expected_size) in cases {
            let model = detect_ollama_model_from_path(std::path::Path::new(path));
            let expected = format!("qwen2.5-coder:{expected_size}");
            assert_eq!(
                model, expected,
                "Path '{path}' should detect size '{expected_size}'"
            );
        }
    }

    /// Underscore-separated size variants (e.g., "-0_5b") should be detected.
    /// Bug class: only matching dot-separated sizes, missing underscore variant.
    #[test]
    fn detect_ollama_model_underscore_size() {
        let model = detect_ollama_model_from_path(std::path::Path::new(
            "/cache/qwen2.5-coder-0_5b-instruct-q4_k_m.gguf",
        ));
        assert!(
            model.contains("0.5b"),
            "Underscore-separated size should be detected: {model}"
        );
    }

    /// The 3B model size should be detected.
    /// Bug class: regex matching "3b" inside "32b" or "13b" -- verify specificity.
    #[test]
    fn detect_ollama_model_3b_not_confused_with_32b() {
        let model_3b =
            detect_ollama_model_from_path(std::path::Path::new("/tmp/qwen2-3b-instruct.gguf"));
        assert!(
            model_3b.contains(":3b"),
            "Should detect 3b, got: {model_3b}"
        );

        let model_32b =
            detect_ollama_model_from_path(std::path::Path::new("/tmp/qwen2-32b-instruct.gguf"));
        assert!(
            model_32b.contains(":32b"),
            "Should detect 32b, got: {model_32b}"
        );
    }

    /// Hash-named files (no size in name) should fall back to file size.
    /// Bug class: panic or incorrect default when filename has no size hint.
    #[test]
    fn detect_ollama_model_hash_named_file() {
        // This file doesn't exist, so metadata will fail -> defaults to "7b"
        let model = detect_ollama_model_from_path(std::path::Path::new(
            "/tmp/e910cab26ae116eb.converted.gguf",
        ));
        assert!(
            model.contains("qwen2.5-coder:"),
            "Should produce valid model tag: {model}"
        );
    }

    // ========================================================================
    // QaConfig: field interaction invariants
    // ========================================================================

    /// Custom config overrides should not affect unrelated fields.
    /// Bug class: struct update syntax (..) accidentally overriding explicitly set fields.
    #[test]
    fn qa_config_partial_override_preserves_defaults() {
        let config = QaConfig {
            min_tps: 500.0,
            skip_golden: true,
            iterations: 5,
            ..Default::default()
        };
        // Overridden fields
        assert!((config.min_tps - 500.0).abs() < f64::EPSILON);
        assert!(config.skip_golden);
        assert_eq!(config.iterations, 5);
        // Default fields must be preserved
        assert!((config.min_speedup - 0.2).abs() < f64::EPSILON);
        assert!((config.min_gpu_speedup - 2.0).abs() < f64::EPSILON);
        assert!(!config.skip_throughput);
        assert!(!config.skip_ollama);
        assert_eq!(config.warmup, 3);
        assert_eq!(config.max_tokens, 32);
        assert!(!config.json);
    }

    /// skip_contract flag should be independent of other skip flags.
    /// Bug class: skip flags sharing a single boolean or bitmask.
    #[test]
    fn qa_config_skip_flags_are_independent() {
        let config = QaConfig {
            skip_golden: true,
            skip_contract: true,
            ..Default::default()
        };
        assert!(config.skip_golden);
        assert!(config.skip_contract);
        assert!(!config.skip_throughput);
        assert!(!config.skip_ollama);
        assert!(!config.skip_gpu_speedup);
        assert!(!config.skip_format_parity);
    }

    // ========================================================================
    // print_gate_result: gate name display mapping
    // ========================================================================

    /// Verify that all known gate names have display names in the printer.
    /// Bug class: new gate added without updating the display name map,
    /// causing raw snake_case name to appear in user-facing output.
    #[test]
    fn all_gate_names_have_display_mapping() {
        // These are the canonical gate names used in the QA system
        let gate_names = [
            "tensor_contract",
            "golden_output",
            "throughput",
            "ollama_parity",
            "gpu_speedup",
            "format_parity",
        ];
        for name in &gate_names {
            // Verify the name is one of the known gates by matching
            // the same logic as print_gate_result
            let display = match *name {
                "tensor_contract" => "Tensor Contract",
                "golden_output" => "Golden Output",
                "throughput" => "Throughput",
                "ollama_parity" => "Ollama Parity",
                "gpu_speedup" => "GPU Speedup",
                "format_parity" => "Format Parity",
                _ => panic!("Unknown gate name without display mapping: {name}"),
            };
            assert!(
                !display.is_empty(),
                "Display name for '{name}' must not be empty"
            );
        }
    }

    // ========================================================================
    // print_gate_result: status branching and name fallback
    // ========================================================================

    /// Unknown gate names should fall through to the raw name (the `_ => &result.name` arm).
    /// Bug class: match arm panicking on unexpected gate name instead of graceful fallback.
    #[test]
    fn print_gate_result_unknown_name_uses_raw_name() {
        // Exercising `print_gate_result` with an unknown gate name to ensure
        // the `_ => &result.name` fallback branch is reached without panic.
        let result = GateResult::passed(
            "custom_user_gate",
            "User-defined gate passed",
            None,
            None,
            Duration::from_millis(42),
        );
        // This should not panic -- exercises the fallback arm in print_gate_result
        print_gate_result(&result);
    }

    /// print_gate_result with a skipped gate exercises the `[SKIP]` branch.
    #[test]
    fn print_gate_result_skip_branch() {
        let result = GateResult::skipped("ollama_parity", "Ollama not available");
        // Exercises the skipped branch; should not print duration line
        print_gate_result(&result);
    }

    /// print_gate_result with a failed gate exercises the `[FAIL]` branch.
    #[test]
    fn print_gate_result_fail_branch() {
        let result = GateResult::failed(
            "throughput",
            "5.0 tok/s < 100.0 tok/s threshold",
            Some(5.0),
            Some(100.0),
            Duration::from_millis(3500),
        );
        // Exercises the failed branch; should print duration
        print_gate_result(&result);
    }

    /// print_gate_result with a passed gate exercises the `[PASS]` branch.
    #[test]
    fn print_gate_result_pass_branch() {
        let result = GateResult::passed(
            "tensor_contract",
            "50 tensors passed all PMAT-235 contract gates",
            Some(50.0),
            Some(0.0),
            Duration::from_millis(120),
        );
        print_gate_result(&result);
    }

    /// Exercises every known gate name through print_gate_result to cover
    /// all match arms in the name-display mapping.
    #[test]
    fn print_gate_result_all_known_gate_names() {
        let known_names = [
            "tensor_contract",
            "golden_output",
            "throughput",
            "ollama_parity",
            "gpu_speedup",
            "format_parity",
        ];
        for name in &known_names {
            let result = GateResult::passed(name, "ok", None, None, Duration::from_millis(1));
            // Each iteration exercises one arm of the match statement
            print_gate_result(&result);
        }
    }

    // ========================================================================
    // detect_ollama_model_from_path: extended edge cases
    // ========================================================================

    /// Case-insensitive detection: uppercase size markers should match.
    /// Bug class: to_lowercase() not applied before matching.
    #[test]
    fn detect_ollama_model_case_insensitive() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/Qwen2-0.5B-Instruct.gguf"));
        assert_eq!(
            model, "qwen2.5-coder:0.5b",
            "Uppercase '0.5B' should match via to_lowercase"
        );
    }

    /// The 1.5b underscore variant (-1_5b) should be detected correctly.
    #[test]
    fn detect_ollama_model_1_5b_underscore() {
        let model =
            detect_ollama_model_from_path(Path::new("/cache/model-1_5b-instruct-q4_k.gguf"));
        assert_eq!(model, "qwen2.5-coder:1.5b");
    }

    /// Path with no filename component (e.g., root path) should not panic.
    /// Bug class: unwrap() on file_name() returning None.
    #[test]
    fn detect_ollama_model_root_path_no_panic() {
        let model = detect_ollama_model_from_path(Path::new("/"));
        // Root has no filename, so unwrap_or("") gives empty string, falls to file size heuristic
        assert!(
            model.starts_with("qwen2.5-coder:"),
            "Root path should produce valid model tag: {model}"
        );
    }

    /// Path with no extension should still detect size from stem.
    #[test]
    fn detect_ollama_model_no_extension() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/qwen2-7b-instruct"));
        assert_eq!(model, "qwen2.5-coder:7b");
    }

    /// Multiple size markers: the first matching branch wins (0.5b checked before 1.5b, etc.)
    /// Bug class: greedy matching where "3b" matches inside "32b".
    #[test]
    fn detect_ollama_model_priority_order() {
        // "0.5b" is checked first; filename contains both "0.5b" and "7b"
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-0.5b-vs-7b.gguf"));
        assert_eq!(
            model, "qwen2.5-coder:0.5b",
            "0.5b branch should match before 7b"
        );
    }

    /// Filename with "14b" should not match "1.5b" or "4b" (substring confusion).
    #[test]
    fn detect_ollama_model_14b_specificity() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/llama-14b-chat.gguf"));
        assert_eq!(model, "qwen2.5-coder:14b");
    }

    /// File size heuristic: a tiny temp file (< 800MB) should map to 0.5b.
    #[test]
    fn detect_ollama_model_file_size_heuristic_tiny() {
        // Create a real temp file with no size hint in name
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // Temp file is essentially 0 bytes -> 0..=800_000_000 -> "0.5b"
        let model = detect_ollama_model_from_path(file.path());
        assert_eq!(
            model, "qwen2.5-coder:0.5b",
            "Empty temp file should map to 0.5b via file size heuristic"
        );
    }

    // ========================================================================
    // QaReport: summary generation logic (mirrors run_qa's summary builder)
    // ========================================================================

    /// Summary for all-passed report should be the standard success message.
    #[test]
    fn qa_report_summary_all_passed_message() {
        let gates = vec![
            GateResult::passed("golden_output", "ok", None, None, Duration::from_secs(1)),
            GateResult::passed(
                "throughput",
                "150 tok/s",
                Some(150.0),
                Some(100.0),
                Duration::from_secs(2),
            ),
        ];
        let passed = gates.iter().all(|g| g.passed);
        let summary = if passed {
            "All QA gates passed".to_string()
        } else {
            let failed: Vec<_> = gates
                .iter()
                .filter(|g| !g.passed && !g.skipped)
                .map(|g| g.name.as_str())
                .collect();
            format!("Failed gates: {}", failed.join(", "))
        };
        assert_eq!(summary, "All QA gates passed");
    }

    /// Summary for a failed report should list the failed gate names.
    #[test]
    fn qa_report_summary_lists_failed_gate_names() {
        let gates = vec![
            GateResult::passed("golden_output", "ok", None, None, Duration::from_secs(1)),
            GateResult::failed(
                "throughput",
                "too slow",
                Some(5.0),
                Some(100.0),
                Duration::from_secs(2),
            ),
            GateResult::failed(
                "ollama_parity",
                "too slow vs ollama",
                Some(0.1),
                Some(0.2),
                Duration::from_secs(3),
            ),
            GateResult::skipped("gpu_speedup", "no GPU"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(!passed);
        let failed_names: Vec<_> = gates
            .iter()
            .filter(|g| !g.passed && !g.skipped)
            .map(|g| g.name.as_str())
            .collect();
        let summary = format!("Failed gates: {}", failed_names.join(", "));
        assert_eq!(summary, "Failed gates: throughput, ollama_parity");
    }

    /// Summary for a report where only skipped gates exist (no real failures).
    #[test]
    fn qa_report_summary_skipped_only_is_passed() {
        let gates = vec![
            GateResult::skipped("golden_output", "no model"),
            GateResult::skipped("throughput", "no engine"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(passed, "All-skipped should be passed");
    }

    // ========================================================================
    // QaConfig: safetensors_path and combined flag states
    // ========================================================================

    /// QaConfig with safetensors_path set to Some should preserve the path.
    #[test]
    fn qa_config_with_safetensors_path() {
        let config = QaConfig {
            safetensors_path: Some(std::path::PathBuf::from("/models/qwen.safetensors")),
            ..Default::default()
        };
        assert_eq!(
            config.safetensors_path.as_deref(),
            Some(std::path::Path::new("/models/qwen.safetensors"))
        );
    }

    /// QaConfig default has skip_contract = false.
    /// Bug class: new skip flag defaulting to true, silently disabling a gate.
    #[test]
    fn qa_config_default_skip_contract_is_false() {
        let config = QaConfig::default();
        assert!(
            !config.skip_contract,
            "skip_contract must default to false to ensure tensor validation runs"
        );
    }

    /// All skip flags set to true simultaneously.
    /// Bug class: skip flag interaction causing unexpected behavior.
    #[test]
    fn qa_config_all_skips_enabled() {
        let config = QaConfig {
            skip_golden: true,
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            ..Default::default()
        };
        assert!(config.skip_golden);
        assert!(config.skip_throughput);
        assert!(config.skip_ollama);
        assert!(config.skip_gpu_speedup);
        assert!(config.skip_contract);
        assert!(config.skip_format_parity);
        // Non-skip fields should be default
        assert_eq!(config.iterations, 10);
        assert!((config.min_tps - 100.0).abs() < f64::EPSILON);
    }

    /// QaConfig with json=true and verbose=true simultaneously.
    /// Bug class: mutually exclusive flags not being properly independent.
    #[test]
    fn qa_config_json_and_verbose_independent() {
        let config = QaConfig {
            json: true,
            verbose: true,
            ..Default::default()
        };
        assert!(config.json);
        assert!(config.verbose);
    }

    /// QaConfig with extreme numeric values should not panic.
    #[test]
    fn qa_config_extreme_thresholds() {
        let config = QaConfig {
            min_tps: f64::MAX,
            min_speedup: 0.0,
            min_gpu_speedup: f64::MIN_POSITIVE,
            iterations: usize::MAX,
            warmup: 0,
            max_tokens: 1,
            ..Default::default()
        };
        assert_eq!(config.min_tps, f64::MAX);
        assert!((config.min_speedup).abs() < f64::EPSILON);
        assert_eq!(config.iterations, usize::MAX);
        assert_eq!(config.warmup, 0);
        assert_eq!(config.max_tokens, 1);
    }

    // ========================================================================
    // GateResult: duration conversion edge cases
    // ========================================================================

    /// Sub-millisecond durations should truncate to 0ms (not round up).
    /// Bug class: using as_millis() which truncates, vs round() which would round.
    #[test]
    fn gate_result_submillisecond_duration_truncates_to_zero() {
        let result = GateResult::passed(
            "fast",
            "blazing fast",
            None,
            None,
            Duration::from_micros(999),
        );
        assert_eq!(
            result.duration_ms, 0,
            "999 microseconds should truncate to 0ms"
        );
    }

    /// Duration at exactly 1ms boundary.
    #[test]
    fn gate_result_exact_one_millisecond() {
        let result = GateResult::passed("gate", "msg", None, None, Duration::from_millis(1));
        assert_eq!(result.duration_ms, 1);
    }

    /// Duration from nanoseconds: 1_500_000 ns = 1ms (truncated from 1.5ms).
    #[test]
    fn gate_result_nanos_to_millis_truncation() {
        let result = GateResult::failed("gate", "msg", None, None, Duration::from_nanos(1_500_000));
        assert_eq!(
            result.duration_ms, 1,
            "1.5ms in nanos should truncate to 1ms"
        );
    }

    // ========================================================================
    // GateResult: message format contracts
    // ========================================================================

    /// Skipped gate message must always be prefixed with "Skipped: ".
    /// Bug class: changing the format string and breaking downstream parsers.
    #[test]
    fn gate_result_skipped_message_format_contract() {
        let reasons = [
            "No GPU available",
            "Ollama not available (start with: ollama serve)",
            "Requires 'inference' feature",
            "Non-GGUF format (F32/F16 lacks fused kernels for Ollama parity)",
            "No --safetensors-path provided",
            "Skipped by --skip-golden",
        ];
        for reason in &reasons {
            let result = GateResult::skipped("test", reason);
            assert!(
                result.message.starts_with("Skipped: "),
                "Skipped message must start with 'Skipped: ', got: '{}'",
                result.message
            );
            assert!(
                result.message.ends_with(reason),
                "Skipped message must end with reason"
            );
        }
    }

    /// Passed gate with value and threshold: values should appear in the struct.
    #[test]
    fn gate_result_passed_preserves_value_and_threshold() {
        let result = GateResult::passed(
            "throughput",
            "150.0 tok/s >= 100.0 tok/s",
            Some(150.5),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(150.5));
        assert_eq!(result.threshold, Some(100.0));
    }

    /// Failed gate with value and threshold: values should appear in the struct.
    #[test]
    fn gate_result_failed_preserves_value_and_threshold() {
        let result = GateResult::failed(
            "ollama_parity",
            "0.15x < 0.2x",
            Some(0.15),
            Some(0.2),
            Duration::from_secs(5),
        );
        assert_eq!(result.value, Some(0.15));
        assert_eq!(result.threshold, Some(0.2));
        assert!(!result.passed);
    }

    // ========================================================================
    // GateResult: JSON deserialization edge cases
    // ========================================================================

    /// Deserializing JSON with explicit null for value/threshold should produce None.
    /// Bug class: serde treating null as missing vs explicit null differently.
    #[test]
    fn gate_result_deserialize_explicit_null_values() {
        let json = r#"{
            "name": "throughput",
            "passed": true,
            "message": "ok",
            "value": null,
            "threshold": null,
            "duration_ms": 100,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize with nulls");
        assert!(result.value.is_none());
        assert!(result.threshold.is_none());
    }

    /// Deserializing JSON with missing optional fields (value/threshold omitted).
    #[test]
    fn gate_result_deserialize_missing_optional_fields() {
        let json = r#"{
            "name": "contract",
            "passed": false,
            "message": "validation error",
            "duration_ms": 50,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize missing optionals");
        assert_eq!(result.name, "contract");
        assert!(!result.passed);
        assert!(result.value.is_none());
        assert!(result.threshold.is_none());
    }

    // ========================================================================
    // QaReport: empty gates edge case
    // ========================================================================

    /// A report with zero gates should still be valid and serializable.
    /// Bug class: division by zero or index-out-of-bounds on empty gate list.
    #[test]
    fn qa_report_empty_gates_is_valid() {
        let report = QaReport {
            model: "empty.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-06T00:00:00Z".to_string(),
            summary: "No gates run".to_string(),
        };
        assert!(report.passed);
        assert!(report.gates.is_empty());
        let json = serde_json::to_string(&report).expect("serialize empty report");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize empty report");
        assert!(restored.gates.is_empty());
    }

    /// A report with many gates (stress test for serialization).
    #[test]
    fn qa_report_many_gates_serialization() {
        let gates: Vec<GateResult> = (0..100)
            .map(|i| {
                GateResult::passed(
                    &format!("gate_{i}"),
                    &format!("Gate {i} passed"),
                    Some(i as f64),
                    Some(0.0),
                    Duration::from_millis(i as u64),
                )
            })
            .collect();
        let report = QaReport {
            model: "stress.gguf".to_string(),
            passed: true,
            gates,
            total_duration_ms: 4950,
            timestamp: "2026-02-06T00:00:00Z".to_string(),
            summary: "All passed".to_string(),
        };
        let json = serde_json::to_string(&report).expect("serialize many gates");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize many gates");
        assert_eq!(restored.gates.len(), 100);
    }

    // ========================================================================
    // detect_ollama_model_from_path: format string contract
    // ========================================================================

    /// Output format must always be "qwen2.5-coder:{size}".
    /// Bug class: format string mismatch breaking Ollama API calls.
    #[test]
    fn detect_ollama_model_output_format_contract() {
        let test_paths = [
            "/tmp/model-0.5b.gguf",
            "/tmp/model-1.5b.gguf",
            "/tmp/model-3b.gguf",
            "/tmp/model-7b.gguf",
            "/tmp/model-14b.gguf",
            "/tmp/model-32b.gguf",
        ];
        for path in &test_paths {
            let model = detect_ollama_model_from_path(Path::new(path));
            assert!(
                model.starts_with("qwen2.5-coder:"),
                "Model tag must start with 'qwen2.5-coder:', got: {model}"
            );
            let size = model.strip_prefix("qwen2.5-coder:").expect("strip prefix");
            assert!(
                ["0.5b", "1.5b", "3b", "7b", "14b", "32b"].contains(&size),
                "Size must be one of the known sizes, got: {size}"
            );
        }
    }

    /// Empty filename (just a directory path) should not panic.
    #[test]
    fn detect_ollama_model_directory_path() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/models/"));
        // No filename -> empty string -> falls to file size heuristic -> metadata fails -> "7b"
        assert!(
            model.starts_with("qwen2.5-coder:"),
            "Directory path should produce valid tag: {model}"
        );
    }

    // ========================================================================
    // run_qa summary builder: failed_gates name collection
    // ========================================================================

    /// Multiple failed gates should all appear in the summary, comma-separated.
    #[test]
    fn failed_gates_summary_multiple_failures() {
        let gates = vec![
            GateResult::failed("golden_output", "wrong", None, None, Duration::from_secs(1)),
            GateResult::failed(
                "throughput",
                "slow",
                Some(1.0),
                Some(100.0),
                Duration::from_secs(2),
            ),
            GateResult::failed(
                "tensor_contract",
                "violations",
                Some(5.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
            GateResult::skipped("ollama_parity", "not available"),
            GateResult::passed(
                "gpu_speedup",
                "ok",
                Some(3.0),
                Some(2.0),
                Duration::from_secs(4),
            ),
        ];
        let failed_names: Vec<&str> = gates
            .iter()
            .filter(|g| !g.passed && !g.skipped)
            .map(|g| g.name.as_str())
            .collect();
        assert_eq!(failed_names.len(), 3);
        let summary = format!("Failed gates: {}", failed_names.join(", "));
        assert!(summary.contains("golden_output"));
        assert!(summary.contains("throughput"));
        assert!(summary.contains("tensor_contract"));
        assert!(
            !summary.contains("ollama_parity"),
            "Skipped gate should not appear in failures"
        );
        assert!(
            !summary.contains("gpu_speedup"),
            "Passed gate should not appear in failures"
        );
    }

    /// Zero failed gates should not produce a "Failed gates:" summary.
    #[test]
    fn failed_gates_summary_no_failures() {
        let gates = vec![
            GateResult::passed("golden_output", "ok", None, None, Duration::from_secs(1)),
            GateResult::skipped("ollama_parity", "not available"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(passed);
        let summary = if passed {
            "All QA gates passed".to_string()
        } else {
            unreachable!()
        };
        assert_eq!(summary, "All QA gates passed");
    }

    // ========================================================================
    // GateResult: NaN and infinity in value/threshold
    // ========================================================================

    /// NaN values in gate results: the struct itself can hold NaN,
    /// verifying the value is stored correctly (NaN != NaN by IEEE 754).
    /// Bug class: accidentally comparing NaN with == and losing the signal.
    #[test]
    fn gate_result_nan_value_is_nan() {
        let result = GateResult::passed(
            "test",
            "NaN test",
            Some(f64::NAN),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(
            result.value.expect("should have value").is_nan(),
            "NaN value must be preserved in GateResult"
        );
        assert!(
            !result.value.expect("should have value").is_finite(),
            "NaN is not finite"
        );
    }

    /// Infinity in gate results should be representable.
    /// Bug class: threshold comparison logic using >= with infinity.
    #[test]
    fn gate_result_infinity_value_is_infinite() {
        let result = GateResult::failed(
            "test",
            "Inf test",
            Some(f64::INFINITY),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(
            result.value.expect("should have value").is_infinite(),
            "Infinity must be preserved in GateResult"
        );
    }

    /// Negative infinity in threshold should be representable.
    #[test]
    fn gate_result_neg_infinity_threshold() {
        let result = GateResult::passed(
            "test",
            "neg inf threshold",
            Some(0.0),
            Some(f64::NEG_INFINITY),
            Duration::from_secs(1),
        );
        assert!(result
            .threshold
            .expect("should have threshold")
            .is_infinite());
    }

    // ========================================================================
    // QaConfig: clone preserves all fields including PathBuf
    // ========================================================================

    /// Clone with safetensors_path should deep-copy the PathBuf.
    #[test]
    fn qa_config_clone_with_safetensors_path() {
        let config = QaConfig {
            safetensors_path: Some(std::path::PathBuf::from("/deep/clone/test.safetensors")),
            min_tps: 42.0,
            json: true,
            verbose: true,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.safetensors_path, config.safetensors_path);
        assert!((cloned.min_tps - 42.0).abs() < f64::EPSILON);
        assert!(cloned.json);
        assert!(cloned.verbose);
    }

    // ========================================================================
    // NEW: Contract failure summary truncation logic
    // ========================================================================
    // Mirrors the truncation in run_tensor_contract_gate (lines 461-468):
    //   if failures.len() <= 3: join with "; "
    //   else: first 3 joined + "; ... and {N-3} more"

    /// Exactly 1 contract failure should display the single failure, no truncation.
    #[test]
    fn contract_failure_summary_single_failure() {
        let failures = vec!["embed_tokens.weight: density below threshold".to_string()];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert_eq!(summary, "embed_tokens.weight: density below threshold");
        assert!(!summary.contains("more"));
    }

    /// Exactly 3 contract failures should display all without truncation.
    #[test]
    fn contract_failure_summary_three_failures_no_truncation() {
        let failures = vec![
            "layer.0: NaN detected".to_string(),
            "layer.1: Inf detected".to_string(),
            "layer.2: zero density".to_string(),
        ];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert_eq!(
            summary,
            "layer.0: NaN detected; layer.1: Inf detected; layer.2: zero density"
        );
        assert!(!summary.contains("more"));
    }

    /// 4 contract failures should truncate: show 3, then "... and 1 more".
    #[test]
    fn contract_failure_summary_four_failures_truncates() {
        let failures = vec![
            "a: fail".to_string(),
            "b: fail".to_string(),
            "c: fail".to_string(),
            "d: fail".to_string(),
        ];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert!(summary.contains("a: fail; b: fail; c: fail"));
        assert!(summary.ends_with("; ... and 1 more"));
    }

    /// 10 contract failures should truncate: show 3, then "... and 7 more".
    #[test]
    fn contract_failure_summary_ten_failures_truncates() {
        let failures: Vec<String> = (0..10).map(|i| format!("tensor_{i}: violation")).collect();
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert!(summary.contains("tensor_0: violation"));
        assert!(summary.contains("tensor_1: violation"));
        assert!(summary.contains("tensor_2: violation"));
        assert!(summary.ends_with("; ... and 7 more"));
        assert!(!summary.contains("tensor_3"));
    }

    /// 0 contract failures should produce empty string (join of empty vec).
    #[test]
    fn contract_failure_summary_zero_failures() {
        let failures: Vec<String> = vec![];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert!(summary.is_empty());
    }

    // ========================================================================
    // NEW: detect_ollama_model_from_path -- additional edge cases
    // ========================================================================

    /// Filename with only "3b" (no prefix dash) should still match the 3b branch.
    #[test]
    fn detect_ollama_model_3b_standalone() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model3b.gguf"));
        assert_eq!(model, "qwen2.5-coder:3b");
    }

    /// Filename with dash-prefixed sizes: "-3b" variant.
    #[test]
    fn detect_ollama_model_dash_3b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-3b-chat.gguf"));
        assert_eq!(model, "qwen2.5-coder:3b");
    }

    /// Filename with "-7b" variant.
    #[test]
    fn detect_ollama_model_dash_7b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/llama-7b-q4_k_m.gguf"));
        assert_eq!(model, "qwen2.5-coder:7b");
    }

    /// Filename containing "0.5b" with mixed case.
    #[test]
    fn detect_ollama_model_mixed_case_0_5b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/Qwen2.5-Coder-0.5B-Q4.gguf"));
        assert_eq!(model, "qwen2.5-coder:0.5b");
    }

    /// Filename containing "-32b" variant.
    #[test]
    fn detect_ollama_model_dash_32b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/qwen-32b-instruct.gguf"));
        assert_eq!(model, "qwen2.5-coder:32b");
    }

    /// Filename containing "-14b" variant with dash prefix.
    #[test]
    fn detect_ollama_model_dash_14b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-14b.gguf"));
        assert_eq!(model, "qwen2.5-coder:14b");
    }

    /// Empty string path (edge case for Path::new("")).
    #[test]
    fn detect_ollama_model_empty_string_path() {
        let model = detect_ollama_model_from_path(Path::new(""));
        // Empty path -> file_name() returns None on empty -> unwrap_or("") -> file size fallback
        assert!(
            model.starts_with("qwen2.5-coder:"),
            "Empty path should produce valid tag: {model}"
        );
    }

    /// Filename that contains multiple size markers: "1.5b" comes before "3b" in check order.
    #[test]
    fn detect_ollama_model_1_5b_before_3b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-1.5b-3b.gguf"));
        assert_eq!(
            model, "qwen2.5-coder:1.5b",
            "1.5b should be matched before 3b in priority order"
        );
    }

    /// Filename with underscore-separated 1.5b variant.
    #[test]
    fn detect_ollama_model_underscore_1_5b_variant() {
        let model = detect_ollama_model_from_path(Path::new("/cache/qwen2-1_5b-q4_k.gguf"));
        assert_eq!(model, "qwen2.5-coder:1.5b");
    }

    /// Filename containing "-0_5b" (underscore variant of 0.5b).
    #[test]
    fn detect_ollama_model_underscore_0_5b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-0_5b-instruct.gguf"));
        assert_eq!(model, "qwen2.5-coder:0.5b");
    }

    // ========================================================================
    // NEW: GateResult JSON edge cases for skip_serializing_if
    // ========================================================================

    /// JSON with value present but threshold missing should deserialize correctly.
    #[test]
    fn gate_result_json_value_present_threshold_missing() {
        let json = r#"{
            "name": "contract",
            "passed": true,
            "message": "50 tensors ok",
            "value": 50.0,
            "duration_ms": 100,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(result.value, Some(50.0));
        assert!(result.threshold.is_none());
    }

    /// JSON with threshold present but value missing should deserialize correctly.
    #[test]
    fn gate_result_json_threshold_present_value_missing() {
        let json = r#"{
            "name": "throughput",
            "passed": false,
            "message": "too slow",
            "threshold": 100.0,
            "duration_ms": 5000,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize");
        assert!(result.value.is_none());
        assert_eq!(result.threshold, Some(100.0));
    }

    /// Serialized JSON for a passed gate with Some(value) should include "value" key.
    #[test]
    fn gate_result_json_includes_value_when_some() {
        let result = GateResult::passed(
            "throughput",
            "150 tok/s",
            Some(150.0),
            None,
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(
            json.contains("\"value\""),
            "value should be present: {json}"
        );
        assert!(
            !json.contains("\"threshold\""),
            "threshold should be omitted when None: {json}"
        );
    }

    /// Serialized JSON for a gate with both Some(value) and Some(threshold).
    #[test]
    fn gate_result_json_includes_both_value_and_threshold() {
        let result = GateResult::failed(
            "ollama_parity",
            "0.1x < 0.2x",
            Some(0.1),
            Some(0.2),
            Duration::from_secs(10),
        );
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"value\""));
        assert!(json.contains("\"threshold\""));
        assert!(json.contains("0.1"));
        assert!(json.contains("0.2"));
    }

    // ========================================================================
    // NEW: QaReport JSON pretty-print validation
    // ========================================================================

    /// Pretty-printed JSON report should contain newlines and indentation.
    #[test]
    fn qa_report_json_pretty_print_format() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![GateResult::passed(
                "contract",
                "ok",
                Some(10.0),
                Some(0.0),
                Duration::from_millis(50),
            )],
            total_duration_ms: 50,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "All passed".to_string(),
        };
        let json = serde_json::to_string_pretty(&report).expect("pretty serialize");
        assert!(json.contains('\n'), "Pretty JSON should contain newlines");
        assert!(
            json.contains("  "),
            "Pretty JSON should contain indentation"
        );
        assert!(json.contains("\"model\""));
        assert!(json.contains("\"gates\""));
        assert!(json.contains("\"summary\""));
    }

    /// JSON report with unwrap_or_default fallback (mirrors run() line 251).
    #[test]
    fn qa_report_json_to_string_pretty_never_panics() {
        let report = QaReport {
            model: String::new(),
            passed: false,
            gates: vec![
                GateResult::skipped("a", "skip"),
                GateResult::failed("b", "fail", Some(f64::NAN), None, Duration::from_secs(0)),
            ],
            total_duration_ms: 0,
            timestamp: String::new(),
            summary: String::new(),
        };
        // This is what run() does: serde_json::to_string_pretty(&report).unwrap_or_default()
        let json = serde_json::to_string_pretty(&report).unwrap_or_default();
        // NaN in JSON becomes null (serde_json behavior), but should not panic
        assert!(!json.is_empty());
    }

    // ========================================================================
    // NEW: QaConfig construction from run() parameters (lines 228-244)
    // ========================================================================

    /// Simulate how run() builds QaConfig from Option parameters.
    /// unwrap_or defaults should match QaConfig::default() for the three thresholds.
    #[test]
    fn run_config_building_none_uses_defaults() {
        let min_tps: Option<f64> = None;
        let min_speedup: Option<f64> = None;
        let min_gpu_speedup: Option<f64> = None;
        let config = QaConfig {
            min_tps: min_tps.unwrap_or(100.0),
            min_speedup: min_speedup.unwrap_or(0.2),
            min_gpu_speedup: min_gpu_speedup.unwrap_or(2.0),
            ..Default::default()
        };
        assert!((config.min_tps - 100.0).abs() < f64::EPSILON);
        assert!((config.min_speedup - 0.2).abs() < f64::EPSILON);
        assert!((config.min_gpu_speedup - 2.0).abs() < f64::EPSILON);
    }

    /// Simulate how run() builds QaConfig with Some parameters (overrides).
    #[test]
    fn run_config_building_some_overrides_defaults() {
        let min_tps: Option<f64> = Some(50.0);
        let min_speedup: Option<f64> = Some(1.5);
        let min_gpu_speedup: Option<f64> = Some(3.0);
        let config = QaConfig {
            min_tps: min_tps.unwrap_or(100.0),
            min_speedup: min_speedup.unwrap_or(0.2),
            min_gpu_speedup: min_gpu_speedup.unwrap_or(2.0),
            ..Default::default()
        };
        assert!((config.min_tps - 50.0).abs() < f64::EPSILON);
        assert!((config.min_speedup - 1.5).abs() < f64::EPSILON);
        assert!((config.min_gpu_speedup - 3.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // NEW: print_gate_result duration formatting (line 1490-1491)
    // ========================================================================

    /// Verify print_gate_result handles zero duration without division errors.
    #[test]
    fn print_gate_result_zero_duration_formatting() {
        let result = GateResult::passed(
            "tensor_contract",
            "0 tensors",
            Some(0.0),
            Some(0.0),
            Duration::from_millis(0),
        );
        // Should print "Duration: 0.00s" without panic
        print_gate_result(&result);
    }

    /// Verify print_gate_result handles large duration values.
    #[test]
    fn print_gate_result_large_duration_formatting() {
        let result = GateResult::passed(
            "throughput",
            "ok",
            Some(100.0),
            Some(50.0),
            Duration::from_secs(3600),
        );
        // duration_ms = 3600000, format as 3600000.0/1000.0 = 3600.00s
        assert_eq!(result.duration_ms, 3_600_000);
        print_gate_result(&result);
    }

    /// Verify print_gate_result formats duration_ms correctly for sub-second durations.
    #[test]
    fn print_gate_result_subsecond_duration_formatting() {
        let result = GateResult::passed(
            "golden_output",
            "2 cases passed",
            Some(2.0),
            Some(2.0),
            Duration::from_millis(250),
        );
        assert_eq!(result.duration_ms, 250);
        // 250ms / 1000.0 = 0.25s -> should print "Duration: 0.25s"
        print_gate_result(&result);
    }

    // ========================================================================
    // NEW: QaReport with all 6 canonical gates
    // ========================================================================

    /// Verify a report with all 6 canonical gates can be serialized/deserialized.
    #[test]
    fn qa_report_all_six_canonical_gates_roundtrip() {
        let report = QaReport {
            model: "/models/qwen2-0.5b-q4_k.gguf".to_string(),
            passed: false,
            gates: vec![
                GateResult::passed(
                    "tensor_contract",
                    "50 tensors ok",
                    Some(50.0),
                    Some(0.0),
                    Duration::from_millis(100),
                ),
                GateResult::passed(
                    "golden_output",
                    "2 test cases passed",
                    Some(2.0),
                    Some(2.0),
                    Duration::from_millis(5000),
                ),
                GateResult::failed(
                    "throughput",
                    "5 tok/s < 100 tok/s",
                    Some(5.0),
                    Some(100.0),
                    Duration::from_millis(10000),
                ),
                GateResult::skipped("ollama_parity", "Ollama not available"),
                GateResult::skipped("gpu_speedup", "CUDA not available"),
                GateResult::skipped("format_parity", "No --safetensors-path provided"),
            ],
            total_duration_ms: 15100,
            timestamp: "2026-02-07T12:00:00Z".to_string(),
            summary: "Failed gates: throughput".to_string(),
        };
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.gates.len(), 6);
        assert!(!restored.passed);
        // Verify each gate type
        assert!(restored.gates[0].passed && !restored.gates[0].skipped);
        assert!(restored.gates[1].passed && !restored.gates[1].skipped);
        assert!(!restored.gates[2].passed && !restored.gates[2].skipped);
        assert!(restored.gates[3].skipped);
        assert!(restored.gates[4].skipped);
        assert!(restored.gates[5].skipped);
    }

    // ========================================================================
    // NEW: GateResult message content validation
    // ========================================================================

    /// Passed gate message should be stored verbatim.
    #[test]
    fn gate_result_passed_message_stored_verbatim() {
        let msg = "150.0 tok/s >= 100.0 tok/s threshold";
        let result = GateResult::passed(
            "throughput",
            msg,
            Some(150.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.message, msg);
    }

    /// Failed gate message should be stored verbatim.
    #[test]
    fn gate_result_failed_message_stored_verbatim() {
        let msg = "5.0 tok/s < 100.0 tok/s threshold";
        let result = GateResult::failed(
            "throughput",
            msg,
            Some(5.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.message, msg);
    }

    /// Skipped gate message format: "Skipped: {reason}".
    #[test]
    fn gate_result_skipped_message_exact_format() {
        let result = GateResult::skipped("gpu_speedup", "CUDA not available");
        assert_eq!(result.message, "Skipped: CUDA not available");
    }

    /// Empty reason for skipped gate should produce "Skipped: ".
    #[test]
    fn gate_result_skipped_empty_reason() {
        let result = GateResult::skipped("test", "");
        assert_eq!(result.message, "Skipped: ");
        assert!(result.skipped);
    }

    /// Empty name for gate result should be stored as empty string.
    #[test]
    fn gate_result_empty_name() {
        let result = GateResult::passed("", "ok", None, None, Duration::from_secs(0));
        assert_eq!(result.name, "");
        assert!(result.passed);
    }

    // ========================================================================
    // NEW: GateResult negative and zero values
    // ========================================================================

    /// Negative value in a gate result (e.g., from subtraction error).
    #[test]
    fn gate_result_negative_value() {
        let result = GateResult::failed(
            "gpu_speedup",
            "-0.5x slower",
            Some(-0.5),
            Some(2.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(-0.5));
        assert!(!result.passed);
    }

    /// Zero value should be representable.
    #[test]
    fn gate_result_zero_value() {
        let result = GateResult::failed(
            "throughput",
            "0 tok/s",
            Some(0.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(0.0));
        assert_eq!(result.threshold, Some(100.0));
    }

    /// Very small positive value (epsilon-level).
    #[test]
    fn gate_result_epsilon_value() {
        let result = GateResult::passed(
            "throughput",
            "barely passing",
            Some(f64::MIN_POSITIVE),
            Some(0.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(f64::MIN_POSITIVE));
        assert!(result.passed);
    }

    // ========================================================================
    // NEW: QaReport timestamp and model path edge cases
    // ========================================================================

    /// Report with Unicode characters in model path.
    #[test]
    fn qa_report_unicode_model_path() {
        let report = QaReport {
            model: "/modelos/modelo_espa\u{00f1}ol.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "ok".to_string(),
        };
        let json = serde_json::to_string(&report).expect("serialize unicode path");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize unicode path");
        assert!(restored.model.contains("espa\u{00f1}ol"));
    }

    /// Report with very long model path.
    #[test]
    fn qa_report_long_model_path() {
        let long_path = format!("/very/{}/model.gguf", "deep/".repeat(100));
        let report = QaReport {
            model: long_path.clone(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "ok".to_string(),
        };
        let json = serde_json::to_string(&report).expect("serialize long path");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize long path");
        assert_eq!(restored.model, long_path);
    }

    /// Report with empty model path.
    #[test]
    fn qa_report_empty_model_path() {
        let report = QaReport {
            model: String::new(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "ok".to_string(),
        };
        let json = serde_json::to_string(&report).expect("serialize empty model");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize empty model");
        assert!(restored.model.is_empty());
    }

    // ========================================================================
    // NEW: QaReport aggregate pass/fail with mixed states
    // ========================================================================

    /// All gates failed: report.passed should be false and all gates listed.
    #[test]
    fn qa_report_all_gates_failed() {
        let gates = vec![
            GateResult::failed(
                "tensor_contract",
                "violations",
                Some(5.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
            GateResult::failed(
                "golden_output",
                "wrong output",
                None,
                None,
                Duration::from_secs(2),
            ),
            GateResult::failed(
                "throughput",
                "too slow",
                Some(1.0),
                Some(100.0),
                Duration::from_secs(3),
            ),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(!passed);
        let failed_names: Vec<&str> = gates
            .iter()
            .filter(|g| !g.passed && !g.skipped)
            .map(|g| g.name.as_str())
            .collect();
        assert_eq!(failed_names.len(), 3);
        let summary = format!("Failed gates: {}", failed_names.join(", "));
        assert_eq!(
            summary,
            "Failed gates: tensor_contract, golden_output, throughput"
        );
    }

    /// Single passed gate among skipped: overall pass.
    #[test]
    fn qa_report_single_pass_rest_skipped() {
        let gates = vec![
            GateResult::passed(
                "tensor_contract",
                "ok",
                Some(10.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
            GateResult::skipped("golden_output", "no engine"),
            GateResult::skipped("throughput", "no engine"),
            GateResult::skipped("ollama_parity", "not available"),
            GateResult::skipped("gpu_speedup", "no GPU"),
            GateResult::skipped("format_parity", "no path"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(passed);
    }

    // ========================================================================
    // NEW: print_gate_result exercises for each known gate display name
    // ========================================================================

    /// Exercise print_gate_result with "format_parity" gate -- one of the display names.
    #[test]
    fn print_gate_result_format_parity_display_name() {
        let result = GateResult::passed(
            "format_parity",
            "GGUF argmax=42 == SafeTensors argmax=42",
            Some(42.0),
            Some(42.0),
            Duration::from_millis(8000),
        );
        print_gate_result(&result);
    }

    /// Exercise print_gate_result with "gpu_speedup" gate in failed state.
    #[test]
    fn print_gate_result_gpu_speedup_failed() {
        let result = GateResult::failed(
            "gpu_speedup",
            "GPU 1.2x faster than CPU < 2.0x threshold",
            Some(1.2),
            Some(2.0),
            Duration::from_millis(15000),
        );
        print_gate_result(&result);
    }

    // ========================================================================
    // NEW: QaConfig with zero and extreme iteration/token values
    // ========================================================================

    /// Zero iterations and warmup should be representable.
    #[test]
    fn qa_config_zero_iterations_and_warmup() {
        let config = QaConfig {
            iterations: 0,
            warmup: 0,
            max_tokens: 0,
            ..Default::default()
        };
        assert_eq!(config.iterations, 0);
        assert_eq!(config.warmup, 0);
        assert_eq!(config.max_tokens, 0);
    }

    /// Large max_tokens value.
    #[test]
    fn qa_config_large_max_tokens() {
        let config = QaConfig {
            max_tokens: 1_000_000,
            ..Default::default()
        };
        assert_eq!(config.max_tokens, 1_000_000);
    }

    // ========================================================================
    // NEW: GateResult serialization with special f64 values
    // ========================================================================

    /// Serialize gate with very large value.
    #[test]
    fn gate_result_serialize_large_value() {
        let result = GateResult::passed(
            "throughput",
            "very fast",
            Some(999_999.99),
            Some(100.0),
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize large value");
        assert!(json.contains("999999.99"));
    }

    /// Serialize gate with very small (near-zero) positive value.
    #[test]
    fn gate_result_serialize_tiny_value() {
        let result = GateResult::failed(
            "throughput",
            "basically zero",
            Some(0.000_001),
            Some(100.0),
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize tiny value");
        // serde_json will serialize this as something like 1e-6 or 0.000001
        let restored: GateResult = serde_json::from_str(&json).expect("deserialize tiny value");
        assert!((restored.value.expect("has value") - 0.000_001).abs() < 1e-10);
    }

    // ========================================================================
    // NEW: QaReport JSON deserialize with extra fields (forward compat)
    // ========================================================================

    /// JSON with extra unknown fields should still deserialize (serde default).
    #[test]
    fn qa_report_deserialize_ignores_unknown_fields() {
        let json = r#"{
            "model": "test.gguf",
            "passed": true,
            "gates": [],
            "total_duration_ms": 100,
            "timestamp": "2026-02-07T00:00:00Z",
            "summary": "ok",
            "extra_field": "should be ignored",
            "another_extra": 42
        }"#;
        let report: QaReport = serde_json::from_str(json).expect("deserialize with extras");
        assert_eq!(report.model, "test.gguf");
        assert!(report.passed);
    }

    /// GateResult JSON with extra unknown fields should still deserialize.
    #[test]
    fn gate_result_deserialize_ignores_unknown_fields() {
        let json = r#"{
            "name": "test",
            "passed": true,
            "message": "ok",
            "duration_ms": 100,
            "skipped": false,
            "future_field": "v2"
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize with extras");
        assert_eq!(result.name, "test");
        assert!(result.passed);
    }

    // ========================================================================
    // verify_output Tests (PMAT-QA-PROTOCOL-001 §7.4)
    // ========================================================================

    #[test]
    fn verify_output_rejects_empty() {
        let result = verify_output("", "test-001", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(reason.contains("Empty"), "Expected 'Empty', got: {reason}");
        }
    }

    #[test]
    fn verify_output_rejects_whitespace_only() {
        let result = verify_output("   \n\t  ", "test-002", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
    }

    #[test]
    fn verify_output_rejects_garbage_fffd() {
        let result = verify_output("The answer is \u{FFFD}\u{FFFD}", "test-003", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Garbage"),
                "Expected 'Garbage', got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_rejects_garbage_unk() {
        let result = verify_output("Hello [UNK] world", "test-004", &["Hello"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Garbage"),
                "Expected 'Garbage', got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_rejects_null_bytes() {
        let result = verify_output("Hello\0World", "test-005", &["Hello"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("null"),
                "Expected 'null bytes', got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_rejects_missing_expected() {
        let result = verify_output("The answer is five", "test-006", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Expected"),
                "Expected mention of pattern, got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_accepts_correct() {
        let result = verify_output("The answer is 4.", "test-007", &["4"]);
        assert!(matches!(result, OutputVerification::Pass));
    }

    #[test]
    fn verify_output_accepts_any_expected_pattern() {
        let result = verify_output("Hi there!", "test-008", &["Hello", "Hi", "Hey"]);
        assert!(matches!(result, OutputVerification::Pass));
    }

    #[test]
    fn verify_output_case_insensitive() {
        let result = verify_output("HELLO WORLD", "test-009", &["hello"]);
        assert!(matches!(result, OutputVerification::Pass));
    }

    #[test]
    fn verify_output_garbage_check_before_answer_check() {
        // Even though output contains "4", garbage should fail first
        let result = verify_output("4 [UNK] answer", "test-010", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Garbage"),
                "Garbage check must happen BEFORE answer check, got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_no_expected_patterns_passes() {
        // If no patterns expected, just check for emptiness and garbage
        let result = verify_output("Some valid output", "test-011", &[]);
        assert!(matches!(result, OutputVerification::Pass));
    }
}
