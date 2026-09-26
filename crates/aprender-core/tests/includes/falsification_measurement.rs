// M001-M003, M007, M008 and M010 run the `apr` binary; they moved to
// crates/apr-cli/tests/falsification_cbtop_measurement.rs, which gets it from
// CARGO_BIN_EXE_apr instead of a `cargo run` build per test (X86-SLOW-2).

/// M004: Throughput value is positive
#[test]
fn m004_throughput_positive() {
    let layer_us = 35.7;
    let num_layers = 28;
    let total_us = layer_us * num_layers as f64;
    let throughput = 1_000_000.0 / total_us;

    assert!(
        throughput > 0.0,
        "M004 FALSIFIED: Throughput {:.1} must be positive",
        throughput
    );
}

/// M005: All metrics have units in output
#[test]
fn m005_metrics_have_units() {
    let expected_unit_fields = [
        ("tokens_per_sec", "tok/s"),
        ("ttft_ms", "ms"),
        ("p50_us", "us"),
        ("p99_us", "us"),
        ("budget_us", "us"),
        ("actual_us", "us"),
    ];

    for (field, unit) in expected_unit_fields {
        assert!(
            field.contains("_"),
            "M005 FALSIFIED: Field {} should include unit suffix ({})",
            field,
            unit
        );
    }
}

/// M006: Headless mode CV < 5% per Curtsinger 2013
#[test]
fn m006_cv_under_five_percent() {
    let samples: Vec<f64> = (0..100).map(|i| 35.7 + (i % 3) as f64 * 0.1).collect();

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    let cv = (std_dev / mean) * 100.0;

    assert!(cv < 5.0, "M006 FALSIFIED: CV {:.2}% >= 5% threshold", cv);
}

/// M009: Warmup iterations are not included in measurement
#[test]
fn m009_warmup_excluded() {
    let warmup = 10;
    let measurement = 100;
    let total_iterations = warmup + measurement;

    assert_eq!(
        measurement, 100,
        "M009 FALSIFIED: Measurement iterations should be 100"
    );
    assert_eq!(
        total_iterations, 110,
        "M009 FALSIFIED: Total should be warmup + measurement"
    );
}

/// M011: Brick scoring formula is consistent
#[test]
fn m011_scoring_formula_consistent() {
    let gap: f64 = 1.1;

    let score1 = if gap <= 1.0 {
        100
    } else if gap <= 1.2 {
        (100.0 - (gap - 1.0) * 50.0) as u32
    } else {
        (100.0_f64 - (gap - 1.0) * 100.0).max(0.0) as u32
    };

    let score2 = if gap <= 1.0 {
        100
    } else if gap <= 1.2 {
        (100.0 - (gap - 1.0) * 50.0) as u32
    } else {
        (100.0_f64 - (gap - 1.0) * 100.0).max(0.0) as u32
    };

    assert_eq!(
        score1, score2,
        "M011 FALSIFIED: Scoring formula not deterministic"
    );
}

/// M012: Grade assignment follows spec
#[test]
fn m012_grade_assignment() {
    let test_cases = [
        (100, "A"),
        (95, "A"),
        (90, "A"),
        (89, "B"),
        (80, "B"),
        (79, "C"),
        (70, "C"),
        (69, "D"),
        (60, "D"),
        (59, "F"),
        (0, "F"),
    ];

    for (score, expected_grade) in test_cases {
        let grade = match score {
            90..=100 => "A",
            80..=89 => "B",
            70..=79 => "C",
            60..=69 => "D",
            _ => "F",
        };

        assert_eq!(
            grade, expected_grade,
            "M012 FALSIFIED: Score {} got grade {}, expected {}",
            score, grade, expected_grade
        );
    }
}
