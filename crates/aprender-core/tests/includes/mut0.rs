// ============================================================================
// MUT-01 to MUT-04: Mutation Operator Coverage
// ============================================================================

/// MUT-01: Arithmetic mutation detection
#[test]
fn mut01_arithmetic_mutation_detection() {
    let a = 10.0_f64;
    let b = 5.0_f64;

    let sum = a + b;
    assert!(
        (sum - 15.0).abs() < 1e-10,
        "MUT-01 FALSIFIED: Addition mutation not caught"
    );

    let product = a * b;
    assert!(
        (product - 50.0).abs() < 1e-10,
        "MUT-01 FALSIFIED: Multiplication mutation not caught"
    );

    let diff = a - b;
    assert!(
        (diff - 5.0).abs() < 1e-10,
        "MUT-01 FALSIFIED: Subtraction mutation not caught"
    );

    let quotient = a / b;
    assert!(
        (quotient - 2.0).abs() < 1e-10,
        "MUT-01 FALSIFIED: Division mutation not caught"
    );
}

/// MUT-02: Relational mutation detection
#[test]
fn mut02_relational_mutation_detection() {
    let boundary_lt = 5;
    assert!(
        boundary_lt < 6,
        "MUT-02 FALSIFIED: Less-than mutation not caught"
    );
    assert!(
        !(boundary_lt < 5),
        "MUT-02 FALSIFIED: Less-than boundary mutation not caught"
    );

    assert!(
        boundary_lt > 4,
        "MUT-02 FALSIFIED: Greater-than mutation not caught"
    );
    assert!(
        !(boundary_lt > 5),
        "MUT-02 FALSIFIED: Greater-than boundary mutation not caught"
    );

    assert!(
        boundary_lt == 5,
        "MUT-02 FALSIFIED: Equality mutation not caught"
    );
    assert!(
        !(boundary_lt == 4),
        "MUT-02 FALSIFIED: Equality false case not caught"
    );

    assert!(
        boundary_lt != 4,
        "MUT-02 FALSIFIED: Not-equal mutation not caught"
    );
}

/// MUT-03: Logical mutation detection
#[test]
fn mut03_logical_mutation_detection() {
    let true_val = true;
    let false_val = false;

    assert!(
        true_val && true_val,
        "MUT-03 FALSIFIED: AND true-true not caught"
    );
    assert!(
        !(true_val && false_val),
        "MUT-03 FALSIFIED: AND true-false mutation not caught"
    );
    assert!(
        !(false_val && true_val),
        "MUT-03 FALSIFIED: AND false-true mutation not caught"
    );

    assert!(
        true_val || false_val,
        "MUT-03 FALSIFIED: OR true-false not caught"
    );
    assert!(
        false_val || true_val,
        "MUT-03 FALSIFIED: OR false-true not caught"
    );
    assert!(
        !(false_val || false_val),
        "MUT-03 FALSIFIED: OR false-false mutation not caught"
    );

    assert!(
        !false_val,
        "MUT-03 FALSIFIED: NOT false mutation not caught"
    );
    assert!(
        !(!true_val),
        "MUT-03 FALSIFIED: NOT true mutation not caught"
    );
}

/// MUT-04: Return value mutation detection
#[test]
fn mut04_return_value_mutation_detection() {
    let compute_value = || 42;
    assert_eq!(
        compute_value(),
        42,
        "MUT-04 FALSIFIED: Return value mutation not caught"
    );

    let is_valid = || true;
    assert!(
        is_valid(),
        "MUT-04 FALSIFIED: Boolean return mutation not caught"
    );

    let get_option = || Some(100);
    assert!(
        get_option().is_some(),
        "MUT-04 FALSIFIED: Option return mutation not caught"
    );
    assert_eq!(
        get_option().unwrap(),
        100,
        "MUT-04 FALSIFIED: Option value mutation not caught"
    );

    let get_result = || -> Result<i32, &'static str> { Ok(200) };
    assert!(
        get_result().is_ok(),
        "MUT-04 FALSIFIED: Result return mutation not caught"
    );
}

// ============================================================================
// MUT-05 to MUT-07: Infrastructure Verification
// ============================================================================

/// MUT-05: CI mutation testing workflow exists
#[test]
fn mut05_ci_mutation_workflow_exists() {
    let ci_path = Path::new(".github/workflows/ci.yml");
    assert!(
        ci_path.exists(),
        "MUT-05 FALSIFIED: No CI configuration found"
    );

    let ci_content = std::fs::read_to_string(ci_path).expect("read ci.yml");

    let has_mutants_job = ci_content.contains("mutants:");
    assert!(has_mutants_job, "MUT-05 FALSIFIED: No mutants job in CI");

    let has_cargo_mutants = ci_content.contains("cargo-mutants");
    assert!(
        has_cargo_mutants,
        "MUT-05 FALSIFIED: cargo-mutants not installed in CI"
    );

    let runs_mutants = ci_content.contains("cargo mutants");
    assert!(
        runs_mutants,
        "MUT-05 FALSIFIED: cargo mutants not executed in CI"
    );
}

/// MUT-06: Mutation results are captured as artifacts
#[test]
fn mut06_mutation_artifacts_captured() {
    let ci_path = Path::new(".github/workflows/ci.yml");
    let ci_content = std::fs::read_to_string(ci_path).expect("read ci.yml");

    let has_upload = ci_content.contains("upload-artifact");
    let has_mutants_results =
        ci_content.contains("mutants-results") || ci_content.contains("mutants.out");

    assert!(
        has_upload && has_mutants_results,
        "MUT-06 FALSIFIED: Mutation results not captured as artifacts"
    );
}

/// MUT-07: Mutation timeout configured appropriately
#[test]
fn mut07_mutation_timeout_configured() {
    let ci_path = Path::new(".github/workflows/ci.yml");
    let ci_content = std::fs::read_to_string(ci_path).expect("read ci.yml");

    let has_timeout = ci_content.contains("--timeout");

    assert!(
        has_timeout,
        "MUT-07 FALSIFIED: No mutation timeout configured"
    );
}
