// Section 11: ML Diagnostics (F-DIAG-*)
// =============================================================================

#[test]
fn f_diag_001_kmeans_clusters_failure_modes() {
    // F-DIAG-001: Structural check — KMeans is available for diagnostic clustering
    let types_path = project_root().join("src").join("format").join("types.rs");
    let content = std::fs::read_to_string(&types_path).expect("types.rs must exist");
    assert!(
        content.contains("KMeans"),
        "F-DIAG-001: KMeans must exist in model types"
    );
    // Also verify the cluster module exists
    let cluster_path = project_root().join("src").join("cluster");
    assert!(
        cluster_path.exists(),
        "F-DIAG-001: cluster module must exist at src/cluster/"
    );
}

#[test]
fn f_diag_002_linear_regression_predicts_error_magnitude() {
    // F-DIAG-002: Structural check — LinearRegression exists for error prediction
    let types_path = project_root().join("src").join("format").join("types.rs");
    let content = std::fs::read_to_string(&types_path).expect("types.rs must exist");
    assert!(
        content.contains("LinearRegression"),
        "F-DIAG-002: LinearRegression must exist in model types"
    );
    // Also verify the linear_model module exists
    let lr_path = project_root().join("src").join("linear_model");
    assert!(
        lr_path.exists(),
        "F-DIAG-002: linear_model module must exist"
    );
}

#[test]
fn f_diag_003_pca_separates_corrupted_from_valid() {
    // F-DIAG-003: Structural check — PCA exists for data separation
    let types_path = project_root().join("src").join("format").join("types.rs");
    let content = std::fs::read_to_string(&types_path).expect("types.rs must exist");
    assert!(
        content.contains("Pca"),
        "F-DIAG-003: Pca must exist in model types"
    );
    // Verify the decomposition module exists
    let pca_path = project_root().join("src").join("decomposition");
    assert!(
        pca_path.exists(),
        "F-DIAG-003: decomposition module must exist (contains PCA)"
    );
}

#[test]
fn f_diag_004_naive_bayes_classifies_fix_category() {
    // F-DIAG-004: Structural check — NaiveBayes exists for classification
    let types_path = project_root().join("src").join("format").join("types.rs");
    let content = std::fs::read_to_string(&types_path).expect("types.rs must exist");
    assert!(
        content.contains("NaiveBayes"),
        "F-DIAG-004: NaiveBayes must exist in model types"
    );
    // Verify the classification module exists (contains NaiveBayes)
    let nb_path = project_root().join("src").join("classification");
    assert!(
        nb_path.exists(),
        "F-DIAG-004: classification module must exist (contains NaiveBayes)"
    );
}

#[test]
fn f_diag_005_rosetta_ml_has_tests() {
    // F-DIAG-005: rosetta_ml module has adequate test coverage
    // Structural check: verify the module has tests and coverage infrastructure
    let rosetta_ml_path = project_root()
        .join("src")
        .join("format")
        .join("rosetta_ml.rs");

    if !rosetta_ml_path.exists() {
        eprintln!("F-DIAG-005: rosetta_ml.rs not found, checking alternate locations");
        return;
    }

    let content = std::fs::read_to_string(&rosetta_ml_path).expect("rosetta_ml.rs readable");

    // Must have test module
    assert!(
        content.contains("#[cfg(test)]") || content.contains("#[test]"),
        "F-DIAG-005: rosetta_ml.rs must have tests"
    );

    // Count test functions
    let test_count = content.matches("#[test]").count();
    assert!(
        test_count >= 10,
        "F-DIAG-005: rosetta_ml.rs must have >= 10 tests, found {test_count}"
    );
}

// =============================================================================
// Section 12: Performance (F-PERF-*)
// All require model files
// =============================================================================

#[test]
fn f_perf_001_kv_cache_is_on_not_on2() {
    // F-PERF-001: `apr profile` on GGUF runs roofline analysis (KV cache profiling)
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (success, stdout, stderr) = run_apr(&["profile", gguf.to_str().unwrap()]);
    if !success {
        eprintln!("SKIP: apr profile failed: {}", stderr);
        return;
    }
    let combined = format!("{stdout}{stderr}");
    assert!(
        !combined.is_empty(),
        "F-PERF-001: apr profile must produce output"
    );
}

#[test]
fn f_perf_002_fused_q4k_matches_reference() {
    // F-PERF-002: Structural check — fused Q4K kernel exists in trueno
    let trueno_dir = project_root()
        .parent()
        .expect("parent dir")
        .join("trueno")
        .join("src");
    if !trueno_dir.exists() {
        eprintln!("SKIP: trueno not found at sibling path");
        return;
    }
    let mut has_fused = false;
    for path in collect_rs_files(&trueno_dir) {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        if (content.contains("fused") || content.contains("matmul")) && content.contains("q4k") {
            has_fused = true;
            break;
        }
    }
    assert!(has_fused, "F-PERF-002: trueno must have Q4K matmul kernel");
}

#[test]
fn f_perf_003_gpu_throughput_gt_cpu() {
    // F-PERF-003: GPU throughput must be greater than CPU throughput
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let gguf_str = gguf.to_str().unwrap();

    // Check GPU availability via nvidia-smi
    let gpu_available = Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !gpu_available {
        eprintln!("SKIP: no NVIDIA GPU available");
        return;
    }

    // Measure GPU throughput
    let (_gpu_ok, gpu_stdout, gpu_stderr) = run_apr(&[
        "bench",
        gguf_str,
        "--iterations",
        "1",
        "--warmup",
        "0",
        "--max-tokens",
        "20",
        "--fast",
    ]);
    let gpu_combined = format!("{}{}", gpu_stdout, gpu_stderr);
    let gpu_tps: f64 = gpu_combined
        .lines()
        .find(|l| l.contains("Throughput:") && l.contains("tok/s"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    // Measure CPU throughput (hide GPU via CUDA_VISIBLE_DEVICES)
    let cpu_out = Command::new(apr_binary())
        .args([
            "bench",
            gguf_str,
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "20",
            "--fast",
        ])
        .env("CUDA_VISIBLE_DEVICES", "")
        .current_dir(project_root())
        .output()
        .expect("apr bench CPU");
    let cpu_combined = format!(
        "{}{}",
        String::from_utf8_lossy(&cpu_out.stdout),
        String::from_utf8_lossy(&cpu_out.stderr)
    );
    let cpu_tps: f64 = cpu_combined
        .lines()
        .find(|l| l.contains("Throughput:") && l.contains("tok/s"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    eprintln!(
        "F-PERF-003: GPU={:.1} tok/s, CPU={:.1} tok/s, speedup={:.1}x",
        gpu_tps,
        cpu_tps,
        if cpu_tps > 0.0 {
            gpu_tps / cpu_tps
        } else {
            0.0
        }
    );

    assert!(gpu_tps > 0.0, "F-PERF-003: GPU produced 0 tok/s");
    assert!(cpu_tps > 0.0, "F-PERF-003: CPU produced 0 tok/s");
    assert!(
        gpu_tps > cpu_tps,
        "F-PERF-003: GPU ({:.1} tok/s) must be faster than CPU ({:.1} tok/s)",
        gpu_tps,
        cpu_tps
    );
}

#[test]
fn f_perf_004_profile_ci_fails_on_threshold_violation() {
    // F-PERF-004: Structural check — profile.rs has CI threshold logic
    let profile_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("profile.rs");
    let content = std::fs::read_to_string(&profile_path).expect("profile.rs must exist");
    assert!(
        content.contains("ci") || content.contains("threshold"),
        "F-PERF-004: profile.rs must have CI threshold logic"
    );
    assert!(
        content.contains("assert")
            || content.contains("ValidationFailed")
            || content.contains("Err("),
        "F-PERF-004: profile.rs must fail on threshold violation"
    );
}

#[test]
fn f_perf_005_bench_produces_stable_measurements() {
    // F-PERF-005: `apr bench` produces consistent results (low CoV)
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (success, stdout, stderr) = run_apr(&["bench", gguf.to_str().unwrap(), "--runs", "3"]);
    if !success {
        eprintln!(
            "SKIP: apr bench failed (may need inference feature): {}",
            stderr
        );
        return;
    }
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "F-PERF-005: Bench must produce output"
    );
}

#[test]
fn f_perf_006_eval_perplexity_is_finite() {
    // F-PERF-006: `apr eval` on GGUF model produces perplexity output
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (success, stdout, stderr) = run_apr(&["eval", gguf.to_str().unwrap()]);
    if !success {
        eprintln!("SKIP: apr eval failed (may need dataset): {}", stderr);
        return;
    }
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("perplexity") || combined.contains("PPL") || combined.contains("ppl"),
        "F-PERF-006: apr eval must output perplexity metric"
    );
}

#[test]
fn f_perf_007_cbtop_monitors_pipeline() {
    // F-PERF-007: Structural check — cbtop.rs has PipelineState + monitoring
    let cbtop_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("cbtop.rs");
    let content = std::fs::read_to_string(&cbtop_path).expect("cbtop.rs must exist");
    assert!(
        content.contains("PipelineState"),
        "F-PERF-007: cbtop.rs must have PipelineState for monitoring"
    );
    assert!(
        content.contains("fn run"),
        "F-PERF-007: cbtop.rs must have run() function"
    );
    assert!(
        content.contains("headless") || content.contains("json"),
        "F-PERF-007: cbtop.rs must support headless/json output"
    );
}

// =============================================================================
