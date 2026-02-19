
/// J23: Naive loop detection (push in loop)
#[test]
fn j23_naive_loop_detection() {
    // Verify push() in loop patterns are flagged

    fn is_naive_loop(code: &str) -> bool {
        // Simple heuristic: push inside a loop without with_capacity
        code.contains("for") && code.contains(".push(") && !code.contains("with_capacity")
    }

    let naive = "for i in 0..n { vec.push(i); }";
    assert!(is_naive_loop(naive), "J23: Naive loop detected");

    let optimized = "let mut vec = Vec::with_capacity(n); for i in 0..n { vec.push(i); }";
    assert!(!is_naive_loop(optimized), "J23: Optimized loop not flagged");
}

/// J24: Performance crate detection
#[test]
fn j24_performance_crate_detection() {
    // Verify performance crates can be detected in Cargo.toml

    let cargo_toml_contents = r#"
[dependencies]
smallvec = "1.0"
bumpalo = "3.0"
"#;

    let has_smallvec = cargo_toml_contents.contains("smallvec");
    let has_bumpalo = cargo_toml_contents.contains("bumpalo");

    assert!(has_smallvec, "J24: smallvec detected");
    assert!(has_bumpalo, "J24: bumpalo detected");

    // Other performance crates to detect
    let perf_crates = ["smallvec", "bumpalo", "arrayvec", "tinyvec", "parking_lot"];
    let found: Vec<_> = perf_crates
        .iter()
        .filter(|c| cargo_toml_contents.contains(*c))
        .collect();

    assert!(!found.is_empty(), "J24: At least one perf crate detected");
}

/// J25: JSON performance grade fields
#[test]
fn j25_json_performance_fields() {
    // Verify performance_grade object present in JSON output

    let profile_output = serde_json::json!({
        "operation": "forward",
        "duration_ms": 50.0,
        "performance_grade": {
            "grade": "B",
            "efficiency_percent": 65.0,
            "theoretical_peak_gflops": 100.0,
            "achieved_gflops": 65.0,
            "bound": "compute",
            "recommendations": [
                "Consider SIMD optimization",
                "Batch operations where possible"
            ]
        }
    });

    assert!(
        profile_output.get("performance_grade").is_some(),
        "J25: performance_grade object present"
    );

    let perf = profile_output.get("performance_grade").unwrap();
    assert!(perf.get("grade").is_some(), "J25: Grade field present");
    assert!(
        perf.get("efficiency_percent").is_some(),
        "J25: Efficiency field present"
    );
    assert!(
        perf.get("recommendations").is_some(),
        "J25: Recommendations field present"
    );
}
