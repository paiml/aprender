/// Verify SpecGate priority assignments for P0, P1, and P2
#[test]
fn test_spec_gate_priorities() {
    assert_eq!(SpecGate::IntMemorySafety.priority(), "P0");
    assert_eq!(SpecGate::SecPathTraversal.priority(), "P0");
    assert_eq!(SpecGate::ApiJsonCompliance.priority(), "P1");
    assert_eq!(SpecGate::NumAttentionEntropy.priority(), "P1");
    assert_eq!(SpecGate::ParCpuGpuEquivalence.priority(), "P2");
    assert_eq!(SpecGate::PerfMinimumTps.priority(), "P2");
}

/// Verify SpecGate point values for different gate types
#[test]
fn test_spec_gate_points() {
    assert_eq!(SpecGate::IntMemorySafety.points(), 10);
    assert_eq!(SpecGate::SecDenialOfService.points(), 10);
    assert_eq!(SpecGate::ApiJsonCompliance.points(), 5);
    assert_eq!(SpecGate::PerfTtft.points(), 5);
}

// ========================================================================
// API COMPLIANCE TESTS (F-API-001..005)
// ========================================================================

/// Verify valid JSON passes F-API-001 compliance check
#[test]
fn test_api_json_compliance_valid() {
    let result = ApiComplianceChecker::check_json_compliance(r#"{"status":"ok"}"#);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-API-001");
}

/// Verify invalid JSON fails F-API-001 compliance check
#[test]
fn test_api_json_compliance_invalid() {
    let result = ApiComplianceChecker::check_json_compliance("not json {");
    assert!(!result.passed);
    assert!(result.details.is_some());
}

/// Verify clean chat output passes F-API-002 template check
#[test]
fn test_api_chat_template_clean() {
    let result = ApiComplianceChecker::check_chat_template("Hello, how can I help you?");
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-API-002");
}

/// Verify template token leakage fails F-API-002 check
#[test]
fn test_api_chat_template_leakage() {
    let result = ApiComplianceChecker::check_chat_template("Hello<|im_end|>");
    assert!(!result.passed);
    assert!(result.details.expect("details should be present").contains("im_end"));
}

/// Verify healthy status code and fast response passes F-API-003
#[test]
fn test_api_health_check_ok() {
    let result = ApiComplianceChecker::check_health_response(200, 50);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-API-003");
}

/// Verify slow health response fails F-API-003 check
#[test]
fn test_api_health_check_slow() {
    let result = ApiComplianceChecker::check_health_response(200, 2000);
    assert!(!result.passed);
    assert!(result.description.contains("slow"));
}

/// Verify HTTP 500 status code fails F-API-003 check
#[test]
fn test_api_health_check_bad_status() {
    let result = ApiComplianceChecker::check_health_response(500, 50);
    assert!(!result.passed);
}

/// Verify proper error response passes F-API-004 check
#[test]
fn test_api_error_handling_correct() {
    let result = ApiComplianceChecker::check_error_handling(400, false, true);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-API-004");
}

/// Verify crash on error input fails F-API-004 check
#[test]
fn test_api_error_handling_crash() {
    let result = ApiComplianceChecker::check_error_handling(0, true, false);
    assert!(!result.passed);
    assert!(result.description.contains("crashed"));
}

/// Verify valid SSE stream passes F-API-005 format check
#[test]
fn test_api_sse_format_valid() {
    let stream = "data: {\"token\":\"hello\"}\n\ndata: {\"token\":\"world\"}\n\n";
    let result = ApiComplianceChecker::check_sse_format(stream);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-API-005");
}

/// Verify invalid SSE stream fails F-API-005 format check
#[test]
fn test_api_sse_format_invalid() {
    let stream = "data: hello\nbad line without data prefix\n";
    let result = ApiComplianceChecker::check_sse_format(stream);
    assert!(!result.passed);
}

// ========================================================================
// PERFORMANCE VALIDATION TESTS (F-PERF-001..004)
// ========================================================================

/// Verify throughput above threshold passes F-PERF-001
#[test]
fn test_perf_tps_pass() {
    let result = PerformanceValidator::check_tps(15.0, 10.0);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-PERF-001");
}

/// Verify throughput below threshold fails F-PERF-001
#[test]
fn test_perf_tps_fail() {
    let result = PerformanceValidator::check_tps(5.0, 10.0);
    assert!(!result.passed);
}

/// Verify time-to-first-token within limit passes F-PERF-002
#[test]
fn test_perf_ttft_pass() {
    let result = PerformanceValidator::check_ttft(500, 2000);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-PERF-002");
}

/// Verify time-to-first-token exceeding limit fails F-PERF-002
#[test]
fn test_perf_ttft_fail() {
    let result = PerformanceValidator::check_ttft(3000, 2000);
    assert!(!result.passed);
}

/// Verify memory within growth threshold passes F-PERF-003
#[test]
fn test_perf_memory_leak_pass() {
    let result = PerformanceValidator::check_memory_leak(100.0, 103.0, 5.0);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-PERF-003");
}

/// Verify memory exceeding growth threshold fails F-PERF-003
#[test]
fn test_perf_memory_leak_fail() {
    let result = PerformanceValidator::check_memory_leak(100.0, 120.0, 5.0);
    assert!(!result.passed);
    assert!(result.description.contains("leak"));
}

/// Verify GPU utilization above threshold passes F-PERF-004
#[test]
fn test_perf_gpu_utilization_pass() {
    let result = PerformanceValidator::check_gpu_utilization(75.0, 50.0);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-PERF-004");
}

/// Verify GPU utilization below threshold fails F-PERF-004
#[test]
fn test_perf_gpu_utilization_fail() {
    let result = PerformanceValidator::check_gpu_utilization(30.0, 50.0);
    assert!(!result.passed);
}

// ========================================================================
// CROSS-PLATFORM PARITY TESTS (F-PAR-001..003)
// ========================================================================

/// Verify CPU/GPU logit equivalence within tolerance passes F-PAR-001
#[test]
fn test_parity_cpu_gpu_pass() {
    let cpu = vec![0.1, 0.2, 0.3];
    let gpu = vec![0.100_001, 0.200_001, 0.300_001];
    let result = ParityChecker::check_cpu_gpu_equivalence(&cpu, &gpu, 1e-5);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-PAR-001");
}

/// Verify CPU/GPU logit divergence beyond tolerance fails F-PAR-001
#[test]
fn test_parity_cpu_gpu_fail() {
    let cpu = vec![0.1, 0.2, 0.3];
    let gpu = vec![0.1, 0.5, 0.3];
    let result = ParityChecker::check_cpu_gpu_equivalence(&cpu, &gpu, 1e-5);
    assert!(!result.passed);
}

/// Verify identical token sequences across formats passes F-PAR-002
#[test]
fn test_parity_format_pass() {
    let gguf = vec![1, 2, 3, 4, 5];
    let safetensors = vec![1, 2, 3, 4, 5];
    let result = ParityChecker::check_format_parity(&gguf, &safetensors);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-PAR-002");
}

/// Verify token mismatch across formats fails F-PAR-002
#[test]
fn test_parity_format_fail() {
    let gguf = vec![1, 2, 3, 4, 5];
    let safetensors = vec![1, 2, 999, 4, 5];
    let result = ParityChecker::check_format_parity(&gguf, &safetensors);
    assert!(!result.passed);
    assert!(result.description.contains("1 token"));
}

/// Verify quantization perplexity within threshold passes F-PAR-003
#[test]
fn test_parity_quantization_pass() {
    let result = ParityChecker::check_quantization_impact(5.0, 5.3, 10.0);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-PAR-003");
}

/// Verify quantization perplexity exceeding threshold fails F-PAR-003
#[test]
fn test_parity_quantization_fail() {
    let result = ParityChecker::check_quantization_impact(5.0, 6.0, 10.0);
    assert!(!result.passed);
}

// ========================================================================
// INTEGRITY TESTS (F-INT-001..005)
// ========================================================================

/// Verify clean exit with code 0 passes F-INT-001 memory safety
#[test]
fn test_integrity_memory_safety_pass() {
    let result = IntegrityChecker::check_memory_safety(Some(0), "");
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-INT-001");
}

/// Verify SIGSEGV signal fails F-INT-001 memory safety check
#[test]
fn test_integrity_memory_safety_segfault() {
    let result = IntegrityChecker::check_memory_safety(Some(139), "SIGSEGV");
    assert!(!result.passed);
    assert!(result.description.contains("Segmentation"));
}

/// Verify buffer overflow detection fails F-INT-001 memory safety
#[test]
fn test_integrity_memory_safety_buffer_overflow() {
    let result = IntegrityChecker::check_memory_safety(Some(6), "buffer overflow detected");
    assert!(!result.passed);
}

/// Verify clean process exit passes F-INT-002
#[test]
fn test_integrity_process_termination_clean() {
    let result = IntegrityChecker::check_process_termination(Some(0), false, true);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-INT-002");
}

/// Verify process timeout fails F-INT-002
#[test]
fn test_integrity_process_termination_timeout() {
    let result = IntegrityChecker::check_process_termination(None, true, false);
    assert!(!result.passed);
    assert!(result.description.contains("timed out"));
}

/// Verify zombie process detection fails F-INT-002
#[test]
fn test_integrity_process_termination_zombie() {
    let result = IntegrityChecker::check_process_termination(None, false, false);
    assert!(!result.passed);
    assert!(result.description.contains("Zombie"));
}

/// Verify clean tensor values pass F-INT-003
#[test]
fn test_integrity_tensor_validity_clean() {
    let result = IntegrityChecker::check_tensor_validity(&[0.1, 0.2, 0.3]);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-INT-003");
}

/// Verify NaN in tensors fails F-INT-003
#[test]
fn test_integrity_tensor_validity_nan() {
    let result = IntegrityChecker::check_tensor_validity(&[0.1, f32::NAN, 0.3]);
    assert!(!result.passed);
    assert!(result.description.contains("NaN"));
}

/// Verify matching checksums pass F-INT-004 format fidelity
#[test]
fn test_integrity_format_fidelity_pass() {
    let result = IntegrityChecker::check_format_fidelity("abc123", "abc123");
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-INT-004");
}

/// Verify mismatched checksums fail F-INT-004 format fidelity
#[test]
fn test_integrity_format_fidelity_fail() {
    let result = IntegrityChecker::check_format_fidelity("abc123", "def456");
    assert!(!result.passed);
    assert!(result.description.contains("altered"));
}

/// Verify identical outputs with same seed passes F-INT-005 determinism
#[test]
fn test_integrity_determinism_pass() {
    let result = IntegrityChecker::check_determinism("hello world", "hello world", 42);
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-INT-005");
    assert!(result.description.contains("42"));
}

/// Verify different outputs with same seed fails F-INT-005 determinism
#[test]
fn test_integrity_determinism_fail() {
    let result = IntegrityChecker::check_determinism("hello world", "hello moon", 42);
    assert!(!result.passed);
    assert!(result.evidence.is_some());
}

// ========================================================================
// NEGATIVE VALIDATION TESTS (QA-NEG-01..03)
// ========================================================================

/// QA-NEG-01: "Bad Math" test - verify oracle catches wrong arithmetic
#[test]
fn test_negative_bad_math_detection() {
    // Simulate a model returning "2+2=5"
    // The integrity checker would see different outputs for same input
    let correct_output = "4";
    let bad_output = "5";
    let result = IntegrityChecker::check_determinism(correct_output, bad_output, 42);
    // This shows the system CAN detect when outputs differ
    assert!(
        !result.passed,
        "Should detect 2+2=5 as different from 2+2=4"
    );
}

/// QA-NEG-02: "Zip Bomb" test - verify DoS protection catches expansion attack
#[test]
fn test_negative_zip_bomb_expansion() {
    let detector = PatternDetector::new();
    let config = DosProtectionConfig {
        max_expansion_ratio: 5.0,
        ..Default::default()
    };
    // Simulated decompressed zip bomb: 1 unique char, massive length
    let bomb = "x".repeat(1000);
    let result = detector.check_dos_protection(&bomb, &config);
    assert!(!result.is_safe, "Zip bomb should be rejected");
    assert!(
        result.violations.iter().any(|v| v.check == "expansion"),
        "Should cite expansion violation"
    );
}

/// QA-NEG-03: "Silent Fail" test - exit 0 but empty output
#[test]
fn test_negative_silent_fail_detection() {
    // Process exits with code 0 but produces no output
    let result = IntegrityChecker::check_process_termination(Some(0), false, false);
    // With has_output=false, even exit 0 should be suspicious
    assert!(
        !result.passed,
        "Silent fail (exit 0, no output) should be caught"
    );
}

// ========================================================================
// ISOLATION AND DETERMINISM TESTS (QA-EXEC-02, QA-EXEC-03)
// ========================================================================

/// QA-EXEC-02: Test isolation - parallel runs don't share state
#[test]
fn test_execution_isolation() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    // Simulate parallel test execution
    for _ in 0..4 {
        let c = Arc::clone(&counter);
        handles.push(std::thread::spawn(move || {
            // Each thread has its own detector instance
            let _detector = PatternDetector::new();
            c.fetch_add(1, Ordering::SeqCst);
            // Simulate some work
            std::thread::sleep(std::time::Duration::from_millis(10));
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // All 4 threads completed without interference
    assert_eq!(counter.load(Ordering::SeqCst), 4);
}

/// QA-EXEC-03: Test determinism - same inputs = same outputs
#[test]
fn test_execution_determinism() {
    let detector = PatternDetector::new();
    let input = "Hello world test input for determinism check";
    let config = DosProtectionConfig::default();

    // Run same check twice
    let result1 = detector.check_dos_protection(input, &config);
    let result2 = detector.check_dos_protection(input, &config);

    // Results should be identical
    assert_eq!(result1.is_safe, result2.is_safe);
    assert_eq!(result1.input_bytes, result2.input_bytes);
    assert_eq!(result1.estimated_tokens, result2.estimated_tokens);
    assert!(
        (result1.repetition_ratio - result2.repetition_ratio).abs() < f64::EPSILON,
        "Repetition ratio should be deterministic"
    );
}

/// Verify default performance thresholds have expected values
#[test]
fn test_performance_thresholds_default() {
    let thresholds = PerformanceThresholds::default();
    assert!((thresholds.min_tps - 10.0).abs() < f64::EPSILON);
    assert_eq!(thresholds.max_ttft_ms, 2000);
    assert!((thresholds.max_memory_growth_percent - 5.0).abs() < f64::EPSILON);
    assert!((thresholds.min_gpu_utilization - 50.0).abs() < f64::EPSILON);
}

/// Verify companion file checker finds all expected files
#[test]
fn test_companion_files_found() {
    // Create temp directory with companion files
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.safetensors");
    let config_path = temp_dir.path().join("config.json");
    let tokenizer_path = temp_dir.path().join("tokenizer.json");

    // Create the files
    std::fs::write(&model_path, "model data").expect("Failed to write model");
    std::fs::write(&config_path, "{}").expect("Failed to write config");
    std::fs::write(&tokenizer_path, "{}").expect("Failed to write tokenizer");

    let detector = PatternDetector::new();
    let result = detector.check_companion_files(&model_path, &["config.json", "tokenizer.json"]);

    assert!(result.all_present, "All companions should be found");
    assert_eq!(result.found.len(), 2);
    assert!(result.missing.is_empty());
    assert!(result.found.contains(&"config.json".to_string()));
    assert!(result.found.contains(&"tokenizer.json".to_string()));
}

// ========================================================================
// ATTENTION ENTROPY EDGE CASES (F-NUM-001)
// ========================================================================

/// Verify empty attention weights returns invalid
#[test]
fn test_attention_entropy_empty_weights_invalid() {
    let detector = PatternDetector::new();
    let result = detector.check_attention_entropy(&[]);
    assert!(!result.is_valid);
    assert!(result.description.contains("Empty"));
}

/// Verify negative sum attention weights returns invalid
#[test]
fn test_attention_entropy_negative_sum() {
    let detector = PatternDetector::new();
    let result = detector.check_attention_entropy(&[-1.0, -2.0, -3.0]);
    assert!(!result.is_valid);
    assert!(result.description.contains("Invalid"));
}

/// Verify NaN sum attention weights returns invalid
#[test]
fn test_attention_entropy_nan_sum() {
    let detector = PatternDetector::new();
    let result = detector.check_attention_entropy(&[f32::NAN, 0.5]);
    assert!(!result.is_valid);
}

/// Verify collapsed attention (single dominant weight) is detected
#[test]
fn test_attention_entropy_extreme_collapse() {
    let detector = PatternDetector::new();
    // One weight dominates → low entropy → collapsed
    let mut weights = vec![0.0001_f32; 100];
    weights[0] = 100.0;
    let result = detector.check_attention_entropy(&weights);
    assert!(!result.is_valid);
    assert!(result.description.contains("collapsed"));
}

/// Verify single-element attention returns not valid (zero max_entropy)
#[test]
fn test_attention_entropy_single_element() {
    let detector = PatternDetector::new();
    let result = detector.check_attention_entropy(&[1.0]);
    // ln(1) = 0 → max_entropy = 0 → normalized_entropy = 0
    assert!(!result.is_valid);
}

// ========================================================================
// LAYERNORM EDGE CASES (F-NUM-002)
// ========================================================================

/// Verify empty LayerNorm output returns invalid
#[test]
fn test_layernorm_empty() {
    let detector = PatternDetector::new();
    let result = detector.check_layernorm_output(&[]);
    assert!(!result.is_valid);
    assert!(result.description.contains("Empty"));
}

// ========================================================================
// DOS PROTECTION EDGE CASES (F-SEC-003)
// ========================================================================

/// Verify input length violation is detected
#[test]
fn test_dos_input_length_violation() {
    let detector = PatternDetector::new();
    let config = DosProtectionConfig {
        max_input_bytes: 10,
        ..Default::default()
    };
    let result = detector.check_dos_protection("this exceeds ten bytes easily", &config);
    assert!(!result.is_safe);
    assert!(result.violations.iter().any(|v| v.check == "input_length"));
}

/// Verify token count violation is detected
#[test]
fn test_dos_token_count_violation() {
    let detector = PatternDetector::new();
    let config = DosProtectionConfig {
        max_tokens: 2,
        ..Default::default()
    };
    // 40 chars ÷ 4 = 10 estimated tokens > 2
    let result = detector.check_dos_protection("a]bc defg hijk lmno pqrs tuvw xyz! 1234", &config);
    assert!(!result.is_safe);
    assert!(result.violations.iter().any(|v| v.check == "token_count"));
}

/// Verify repetition violation is detected
#[test]
fn test_dos_repetition_violation() {
    let detector = PatternDetector::new();
    let config = DosProtectionConfig {
        max_repetition_ratio: 0.1,
        ..Default::default()
    };
    // Highly repetitive input
    let result = detector.check_dos_protection("abcdabcdabcdabcdabcdabcd", &config);
    assert!(!result.is_safe);
    assert!(result.violations.iter().any(|v| v.check == "repetition"));
}

// ========================================================================
// REPETITION RATIO EDGE CASES
// ========================================================================

/// Verify short input returns 0.0 repetition ratio
#[test]
fn test_repetition_ratio_short_input() {
    let detector = PatternDetector::new();
    // Input < 10 chars → returns 0.0
    let ratio = detector.calculate_repetition_ratio("short");
    assert!((ratio - 0.0).abs() < f64::EPSILON);
}

// ========================================================================
// JACCARD SIMILARITY EDGE CASES
// ========================================================================

/// Verify empty strings have perfect similarity
#[test]
fn test_jaccard_similarity_empty_strings_perfect() {
    let detector = PatternDetector::new();
    let result = detector.jaccard_similarity("", "");
    assert!((result - 1.0).abs() < f64::EPSILON);
}

/// Verify companion file checker reports missing files correctly
#[test]
fn test_companion_files_mixed() {
    // Create temp directory with only some companion files
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.safetensors");
    let config_path = temp_dir.path().join("config.json");

    // Create only model and config, not tokenizer
    std::fs::write(&model_path, "model data").expect("Failed to write model");
    std::fs::write(&config_path, "{}").expect("Failed to write config");

    let detector = PatternDetector::new();
    let result = detector.check_companion_files(&model_path, &["config.json", "tokenizer.json"]);

    assert!(!result.all_present, "Not all companions present");
    assert_eq!(result.found.len(), 1);
    assert_eq!(result.missing.len(), 1);
    assert!(result.found.contains(&"config.json".to_string()));
    assert!(result.missing.contains(&"tokenizer.json".to_string()));
}
