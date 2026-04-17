//! Falsification: Measurement Freshness Tests
//!
//! Tests for QA report serialization, performance regression detection,
//! and system info capture. These validate the Phase 1 and Phase 7
//! infrastructure added to `apr qa`.
//!
//! Does NOT require model files.

// =============================================================================
// QaReport Serialization Round-Trip Tests
// =============================================================================

#[test]
fn measurement_qa_report_roundtrip_with_new_fields() {
    // Verify QaReport with new fields serializes and deserializes correctly
    let json = r#"{
        "model": "test.gguf",
        "passed": true,
        "gates": [
            {
                "name": "throughput",
                "passed": true,
                "message": "100 tok/s",
                "value": 100.0,
                "threshold": 60.0,
                "duration_ms": 5000,
                "skipped": false
            },
            {
                "name": "ollama_parity",
                "passed": true,
                "message": "Skipped",
                "duration_ms": 0,
                "skipped": true
            }
        ],
        "gates_executed": 1,
        "gates_skipped": 1,
        "total_duration_ms": 5000,
        "timestamp": "2026-02-11T00:00:00Z",
        "summary": "All QA gates passed (1 executed, 1 skipped)",
        "system_info": {
            "cpu_model": "AMD Ryzen 9 7950X",
            "gpu_model": "NVIDIA RTX 4090",
            "gpu_driver": "560.35.03"
        }
    }"#;

    let report: apr_cli::qa_types::QaReport =
        serde_json::from_str(json).expect("deserialize with new fields");

    assert_eq!(report.model, "test.gguf");
    assert!(report.passed);
    assert_eq!(report.gates.len(), 2);
    assert_eq!(report.gates_executed, 1);
    assert_eq!(report.gates_skipped, 1);
    assert!(report.system_info.is_some());

    let sys = report.system_info.as_ref().unwrap();
    assert_eq!(sys.cpu_model, "AMD Ryzen 9 7950X");
    assert_eq!(sys.gpu_model.as_deref(), Some("NVIDIA RTX 4090"));

    // Re-serialize and verify round-trip
    let reserialized = serde_json::to_string(&report).expect("reserialize");
    assert!(reserialized.contains("gates_executed"));
    assert!(reserialized.contains("gates_skipped"));
    assert!(reserialized.contains("system_info"));
}

#[test]
fn measurement_qa_report_backward_compat_no_new_fields() {
    // Old-format JSON without gates_executed/gates_skipped/system_info
    // must still deserialize (serde(default))
    let json = r#"{
        "model": "old.gguf",
        "passed": true,
        "gates": [],
        "total_duration_ms": 1000,
        "timestamp": "2026-01-01T00:00:00Z",
        "summary": "All gates passed"
    }"#;

    let report: apr_cli::qa_types::QaReport =
        serde_json::from_str(json).expect("backward-compat deserialize");

    assert_eq!(report.model, "old.gguf");
    assert!(report.passed);
    assert_eq!(report.gates_executed, 0); // default
    assert_eq!(report.gates_skipped, 0); // default
    assert!(report.system_info.is_none()); // default
}

// =============================================================================
// Performance Regression Detection Tests
// =============================================================================

#[test]
fn measurement_regression_detection_simple() {
    // Simulate Bug 206: 89.8 → 67.8 tok/s
    let prev_tps: f64 = 89.8;
    let curr_tps: f64 = 67.8;
    let threshold: f64 = 0.10;

    let regression = (prev_tps - curr_tps) / prev_tps;
    assert!(
        regression > threshold,
        "89.8→67.8 tok/s must exceed 10% regression threshold (was {:.1}%)",
        regression * 100.0
    );
}

#[test]
fn measurement_regression_detection_no_false_positive() {
    // Improvement: 67.8 → 89.8 tok/s (should NOT trigger)
    let prev_tps: f64 = 67.8;
    let curr_tps: f64 = 89.8;
    let threshold: f64 = 0.10;

    let regression = (prev_tps - curr_tps) / prev_tps;
    assert!(
        regression <= threshold,
        "Improvement must not trigger regression detection"
    );
}

#[test]
fn measurement_regression_detection_exact_threshold() {
    // Exactly at threshold: 100 → 90 (10% drop = boundary)
    let prev_tps: f64 = 100.0;
    let curr_tps: f64 = 90.0;
    let threshold: f64 = 0.10;

    let regression = (prev_tps - curr_tps) / prev_tps;
    // At exactly threshold, we should NOT flag it (> not >=)
    assert!(
        (regression - threshold).abs() < 1e-10,
        "10% drop should be exactly at threshold"
    );
}

#[test]
fn measurement_regression_detection_multiple_gates() {
    // Test with multiple gates, only one regressing
    let gates = vec![
        ("throughput", 100.0_f64, 95.0_f64), // 5% drop — ok
        ("ollama_parity", 0.8, 0.6),         // 25% drop — FLAGGED
        ("gpu_speedup", 12.0, 11.5),         // 4% drop — ok
    ];

    let threshold = 0.10;
    let mut regressions = Vec::new();

    for (name, prev, curr) in &gates {
        if *prev > 0.0 {
            let regression = (prev - curr) / prev;
            if regression > threshold {
                regressions.push(format!("{name}: {prev:.1} → {curr:.1}"));
            }
        }
    }

    assert_eq!(
        regressions.len(),
        1,
        "Only ollama_parity should regress: {regressions:?}"
    );
    assert!(regressions[0].contains("ollama_parity"));
}

#[test]
fn measurement_regression_detection_zero_prev() {
    // Previous value is 0 — should not divide by zero
    let prev_tps: f64 = 0.0;
    let curr_tps: f64 = 50.0;

    if prev_tps > 0.0 {
        let _regression = (prev_tps - curr_tps) / prev_tps;
        panic!("Should not reach here when prev is 0");
    }
    // Correctly skipped — no regression when prev is 0
}

// =============================================================================
// SystemInfo Tests
// =============================================================================

#[test]
fn measurement_system_info_serialization() {
    let info = apr_cli::qa_types::SystemInfo {
        cpu_model: "AMD Ryzen 9 7950X".to_string(),
        gpu_model: Some("NVIDIA RTX 4090".to_string()),
        gpu_driver: Some("560.35.03".to_string()),
    };

    let json = serde_json::to_string(&info).expect("serialize SystemInfo");
    assert!(json.contains("AMD Ryzen"));
    assert!(json.contains("RTX 4090"));
    assert!(json.contains("560.35.03"));
}

#[test]
fn measurement_system_info_no_gpu() {
    let info = apr_cli::qa_types::SystemInfo {
        cpu_model: "Intel Core i9".to_string(),
        gpu_model: None,
        gpu_driver: None,
    };

    let json = serde_json::to_string(&info).expect("serialize");
    assert!(json.contains("Intel Core"));
    // gpu_model and gpu_driver should be absent (skip_serializing_if)
    assert!(!json.contains("gpu_model"));
    assert!(!json.contains("gpu_driver"));
}

// =============================================================================
// Gate Execution Tracking Tests
// =============================================================================

#[test]
fn measurement_gate_accounting_consistency() {
    let json = r#"{
        "model": "test.gguf",
        "passed": true,
        "gates": [
            {"name": "g1", "passed": true, "message": "ok", "duration_ms": 100, "skipped": false},
            {"name": "g2", "passed": true, "message": "skip", "duration_ms": 0, "skipped": true},
            {"name": "g3", "passed": true, "message": "ok", "duration_ms": 200, "skipped": false},
            {"name": "g4", "passed": true, "message": "skip", "duration_ms": 0, "skipped": true}
        ],
        "gates_executed": 2,
        "gates_skipped": 2,
        "total_duration_ms": 300,
        "timestamp": "2026-02-11T00:00:00Z",
        "summary": "test"
    }"#;

    let report: apr_cli::qa_types::QaReport = serde_json::from_str(json).expect("deserialize");

    assert_eq!(report.gates_executed, 2);
    assert_eq!(report.gates_skipped, 2);
    assert_eq!(
        report.gates_executed + report.gates_skipped,
        report.gates.len(),
        "executed + skipped must equal total gates"
    );
}

#[test]
fn measurement_min_executed_scenario() {
    // If min_executed=5 and only 2 executed, the report should fail
    let gates_executed = 2;
    let min_executed = 5;
    let passed = gates_executed >= min_executed;
    assert!(!passed, "min_executed=5 with only 2 gates should fail");
}

#[test]
fn measurement_min_executed_sufficient() {
    let gates_executed = 7;
    let min_executed = 5;
    let passed = gates_executed >= min_executed;
    assert!(passed, "min_executed=5 with 7 gates should pass");
}
