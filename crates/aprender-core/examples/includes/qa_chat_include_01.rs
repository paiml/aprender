fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = QaConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                config.model_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--format-parity" => {
                config.format_parity = true;
                i += 1;
            }
            "--min-cpu-tps" if i + 1 < args.len() => {
                config.min_cpu_tps = args[i + 1].parse().unwrap_or(30.0);
                i += 2;
            }
            "--verbose" | "-v" => {
                config.verbose = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --example qa_chat [OPTIONS]");
                println!("  --model PATH       Path to model file");
                println!("  --min-cpu-tps N    Minimum CPU tok/s (default: 30.0)");
                println!("  --verbose          Verbose output");
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    print_header();

    let model = config.model_path.clone().or_else(find_default_model);
    let model = match model {
        Some(m) => m,
        None => {
            println!("{}ERROR: No model found.{}", RED, NC);
            std::process::exit(2);
        }
    };

    println!("{}Model:{} {}", CYAN, NC, model.display());
    println!();

    let mut results = Vec::new();
    println!(
        "{}=== Section B: qa_chat.rs Tests (20 Points) ==={}",
        YELLOW, NC
    );
    println!();

    results.push(test_model_exists(&model));
    results
        .last()
        .expect("results should not be empty after push")
        .print();

    results.push(test_correct_answer(&config, &model));
    results
        .last()
        .expect("results should not be empty after push")
        .print();

    results.push(test_no_garbage(&config, &model));
    results
        .last()
        .expect("results should not be empty after push")
        .print();

    results.push(test_no_bpe_artifacts(&config, &model));
    results
        .last()
        .expect("results should not be empty after push")
        .print();

    results.push(test_performance_cpu(&config, &model));
    results
        .last()
        .expect("results should not be empty after push")
        .print();

    results.push(test_performance_gpu(&config, &model));
    results
        .last()
        .expect("results should not be empty after push")
        .print();

    print_summary(&results);

    let failed = results.iter().filter(|r| !r.passed).count();
    std::process::exit(if failed == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QaConfig::default();
        // Conservative defaults for 1.5B Q4_K model (PMAT-SHOWCASE-METHODOLOGY-001)
        assert!((config.min_cpu_tps - 3.0).abs() < 0.01);
        assert!((config.min_gpu_tps - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_result_types() {
        let pass = TestResult::pass("P011", "Test", 5);
        assert!(pass.passed);

        let fail = TestResult::fail("P011", "Test", 5, "err".to_string());
        assert!(!fail.passed);

        let skip = TestResult::skip("P011", "Test", 5, "reason".to_string());
        assert!(skip.passed);
        assert_eq!(skip.points, 0);
    }
}
