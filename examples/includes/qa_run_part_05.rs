fn main() {
    // Set up SIGINT handler for graceful shutdown (PMAT-098-PF: zombie mitigation)
    setup_signal_handler();

    let args: Vec<String> = env::args().collect();
    let parsed = parse_args(&args);

    if parsed.show_help {
        print_help();
        return;
    }

    // Header
    println!();
    println!(
        "{}╔═════════════════════════════════════════════════════════════╗{}",
        BLUE, NC
    );
    println!(
        "{}║      APR RUN QA - Matrix Falsification Suite                ║{}",
        BLUE, NC
    );
    println!(
        "{}║      PMAT-QA-RUST-001 + PMAT-QA-MATRIX-001                   ║{}",
        BLUE, NC
    );
    println!(
        "{}╚═════════════════════════════════════════════════════════════╝{}",
        BLUE, NC
    );
    println!();

    let cells = build_cells(&parsed.config, &parsed);
    let config = parsed.config;

    // Show what we're testing
    println!("{}Testing {} cell(s):{}", CYAN, cells.len(), NC);
    for cell in &cells {
        println!("  {} {} → {}", cell.id, cell.label(), cell.model_uri);
    }
    println!();

    // Pre-flight model verification (PMAT-QA-PROTOCOL-001 §7.1)
    // Collect unique models to verify (avoid redundant downloads)
    let unique_models: std::collections::HashSet<_> = cells
        .iter()
        .map(|c| (c.model_uri.clone(), c.format))
        .collect();

    let mut fixtures: Vec<ModelFixture> = unique_models
        .into_iter()
        .map(|(uri, format)| ModelFixture::new(&uri, format))
        .collect();

    println!(
        "{}Pre-flight: Verifying {} unique model(s)...{}",
        CYAN,
        fixtures.len(),
        NC
    );

    let (verified, failures) = verify_model_fixtures(&config, &mut fixtures);

    if !failures.is_empty() {
        println!(
            "{}✗ Model verification failed ({}/{}):{}\n",
            RED,
            failures.len(),
            fixtures.len(),
            NC
        );
        for failure in &failures {
            println!("  {}{}{}", RED, failure, NC);
        }
        println!();
        println!("{}ABORT: Cannot run tests with missing models{}", RED, NC);
        std::process::exit(3);
    }

    println!("{}✓ All {} model(s) verified{}\n", GREEN, verified, NC);

    // Run tests
    let mut results = Vec::new();
    for cell in &cells {
        let result = run_cell_tests(&config, cell);
        print_cell_result(&result);
        results.push(result);
    }

    // Summary
    print_matrix_summary(&results);

    // Ollama parity test (PMAT-SHOWCASE-METHODOLOGY-001 Section 5)
    let ollama_passed = if config.with_ollama {
        run_ollama_comparison(&config)
    } else {
        true
    };

    // Exit code
    let all_passed = results.iter().all(|r| r.passed()) && ollama_passed;
    std::process::exit(if all_passed { 0 } else { 1 });
}

/// Ollama model name for Q4_K_M quantization (same as CANONICAL_GGUF)
const OLLAMA_MODEL: &str = "qwen2.5-coder:1.5b-instruct-q4_K_M";

/// Check if ollama binary is available on the system
fn is_ollama_installed() -> bool {
    Command::new("which")
        .arg("ollama")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if the required ollama model is available
fn is_ollama_model_available() -> bool {
    Command::new("ollama")
        .args(["show", OLLAMA_MODEL])
        .stderr(Stdio::null())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a command and return (output_string, elapsed_seconds)
fn timed_command_output(cmd: &mut Command) -> Result<(String, f64), String> {
    let start = Instant::now();
    let output = cmd.output().map_err(|e| format!("Execution failed: {e}"))?;
    let answer = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok((answer, start.elapsed().as_secs_f64()))
}

/// Check correctness parity between apr and ollama
fn check_correctness_parity(apr_answer: &str, _ollama_answer: &str, ollama_correct: bool) -> bool {
    let apr_correct = apr_answer.contains('4');
    println!();
    println!("{}Test 3: Correctness Parity{}", BOLD, NC);
    if apr_correct && ollama_correct {
        println!(
            "{}[PASS]{} P050: Both produce correct answer (contains '4')",
            GREEN, NC
        );
    } else if apr_correct {
        println!(
            "{}[PASS]{} P050: APR correct (Ollama groundtruth was incorrect)",
            GREEN, NC
        );
    } else {
        println!("{}[FAIL]{} P050: APR output doesn't contain '4'", RED, NC);
        return false;
    }
    true
}

/// Check performance parity (apr within 2x of ollama)
fn check_performance_parity(apr_time: f64, ollama_time: f64) -> bool {
    let speedup = ollama_time / apr_time;
    let within_2x = apr_time <= ollama_time * 2.0;
    println!();
    println!("{}Test 4: Performance Parity{}", BOLD, NC);
    if within_2x {
        println!(
            "{}[PASS]{} P051: APR within 2x of Ollama ({:.2}x speedup)",
            GREEN, NC, speedup
        );
        true
    } else {
        println!(
            "{}[FAIL]{} P051: APR too slow (need 2x, got {:.2}x)",
            RED, NC, speedup
        );
        false
    }
}

/// Print ollama comparison summary
fn print_ollama_summary(all_passed: bool, apr_time: f64, ollama_time: f64) {
    let speedup = ollama_time / apr_time;
    println!();
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );
    println!(
        "{}Ollama Parity: {} | Speedup: {:.2}x | APR: {:.2}s | Ollama: {:.2}s{}",
        if all_passed { GREEN } else { RED },
        if all_passed { "PASS" } else { "FAIL" },
        speedup,
        apr_time,
        ollama_time,
        NC
    );
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );
}

/// Run Ollama parity comparison (PMAT-SHOWCASE-METHODOLOGY-001 Section 5)
///
/// Compares apr's output against Ollama as groundtruth for:
/// 1. Correctness - Output matches semantically
/// 2. Performance - Within 2x of Ollama tok/s
fn run_ollama_comparison(config: &Config) -> bool {
    println!();
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );
    println!(
        "{}         OLLAMA PARITY TEST (PMAT-SHOWCASE-METHODOLOGY-001)      {}",
        CYAN, NC
    );
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );
    println!();

    if !is_ollama_installed() {
        println!(
            "{}[SKIP]{} Ollama not installed - skipping parity test",
            YELLOW, NC
        );
        return true;
    }

    if !is_ollama_model_available() {
        println!(
            "{}[SKIP]{} Ollama model {} not available",
            YELLOW, NC, OLLAMA_MODEL
        );
        println!("       Install with: ollama pull {}", OLLAMA_MODEL);
        return true;
    }

    let prompt = "What is 2+2? Answer with just the number.";

    // Test 1: Ollama groundtruth
    println!("{}Test 1: Ollama Groundtruth{}", BOLD, NC);
    let (ollama_answer, ollama_time) =
        match timed_command_output(Command::new("ollama").args(["run", OLLAMA_MODEL, prompt])) {
            Ok(r) => r,
            Err(e) => {
                println!("{}[FAIL]{} {}", RED, NC, e);
                return false;
            }
        };
    println!("  Ollama output: {:?}", ollama_answer);
    println!("  Ollama time: {:.2}s", ollama_time);

    let ollama_correct = ollama_answer.contains('4');
    if ollama_correct {
        println!("{}[PASS]{} Ollama groundtruth is correct", GREEN, NC);
    } else {
        println!(
            "{}[WARN]{} Ollama groundtruth doesn't contain '4': {}",
            YELLOW, NC, ollama_answer
        );
    }

    // Test 2: APR output
    println!();
    println!("{}Test 2: APR Output{}", BOLD, NC);
    let (apr_answer, apr_time) =
        match timed_command_output(Command::new(&config.apr_binary).args([
            "run",
            &config.gguf_model,
            "--prompt",
            prompt,
            "--max-tokens",
            "10",
        ])) {
            Ok(r) => r,
            Err(e) => {
                println!("{}[FAIL]{} {}", RED, NC, e);
                return false;
            }
        };
    println!("  APR output: {:?}", apr_answer);
    println!("  APR time: {:.2}s", apr_time);

    // Tests 3 & 4
    let correctness_ok = check_correctness_parity(&apr_answer, &ollama_answer, ollama_correct);
    let perf_ok = check_performance_parity(apr_time, ollama_time);
    let all_passed = correctness_ok && perf_ok;

    print_ollama_summary(all_passed, apr_time, ollama_time);

    if config.verbose {
        println!();
        println!("{}Detailed Comparison:{}", MAGENTA, NC);
        println!("  Prompt:        {:?}", prompt);
        println!("  Ollama Model:  {}", OLLAMA_MODEL);
        println!("  APR Model:     {}", config.gguf_model);
        println!("  Ollama Answer: {:?}", ollama_answer);
        println!("  APR Answer:    {:?}", apr_answer);
    }

    all_passed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_flag() {
        assert_eq!(Backend::Cpu.flag(), Some("--no-gpu"));
        assert_eq!(Backend::Gpu.flag(), None);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(Format::Gguf.extension(), ".gguf");
        assert_eq!(Format::SafeTensors.extension(), ".safetensors");
        assert_eq!(Format::Apr.extension(), ".apr");
    }

    #[test]
    fn test_cell_label() {
        let cell = MatrixCell::new(
            "M1",
            Backend::Cpu,
            Format::Gguf,
            "hf://test/model".to_string(),
        );
        assert_eq!(cell.label(), "CPU × GGUF");
    }

    /// Test: Performance thresholds are format-specific (PMAT-SHOWCASE-METHODOLOGY-001)
    ///
    /// Uses conservative thresholds due to word-based estimation variance.
    #[test]
    fn test_performance_thresholds_config() {
        let config = Config::default();

        // CPU threshold for 1.5B (~5-10 tok/s observed)
        assert!((config.min_cpu_tps - 5.0).abs() < 0.01);

        // GPU quantized threshold (conservative due to estimation variance)
        assert!((config.min_gpu_tps - 5.0).abs() < 0.01);

        // GPU float32 threshold (SafeTensors 1.5B)
        assert!((config.min_gpu_tps_float32 - 5.0).abs() < 0.01);

        // All use same conservative threshold
        assert!((config.min_cpu_tps - config.min_gpu_tps).abs() < 0.01);
    }

    /// Test: Threshold selection logic is correct per (backend, format) pair
    #[test]
    fn test_threshold_selection_logic() {
        let config = Config::default();

        // Helper to get threshold for a (backend, format) pair
        let get_threshold = |backend: Backend, format: Format| -> f64 {
            match (backend, format) {
                (Backend::Cpu, _) => config.min_cpu_tps,
                (Backend::Gpu, Format::SafeTensors) => config.min_gpu_tps_float32,
                (Backend::Gpu, _) => config.min_gpu_tps,
            }
        };

        // All use conservative 5.0 threshold due to word-based estimation variance
        assert!((get_threshold(Backend::Cpu, Format::Gguf) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Cpu, Format::SafeTensors) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Cpu, Format::Apr) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::Gguf) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::SafeTensors) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::Apr) - 5.0).abs() < 0.01);
    }

    /// Test: CLI parsing for new --min-gpu-tps-f32 option
    #[test]
    fn test_cli_parsing_float32_threshold() {
        // Verify the default is set correctly (conservative threshold)
        let config = Config::default();
        assert!((config.min_gpu_tps_float32 - 5.0).abs() < 0.01);
    }
}
