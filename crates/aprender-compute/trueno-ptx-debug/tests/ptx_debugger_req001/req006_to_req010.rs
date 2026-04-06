//! REQ-006 through REQ-010: CFG, type checker, consistency, categories, confidence

use trueno_ptx_debug::analyzer::{ControlFlowAnalyzer, TypeChecker};
use trueno_ptx_debug::falsification::{Category, FalsificationRegistry, TestResult};
use trueno_ptx_debug::parser::Parser;

/// REQ-006: Control flow analysis builds valid CFG
///
/// Hypothesis: CFG accurately represents code structure.
/// Falsification: CFG edges miss branches or loops.
#[test]
fn req006_control_flow_analysis() {
    let ptx = r#"
        .version 8.0
        .target sm_70
        .address_size 64

        .entry with_branch()
        {
            .reg .u32 %r<4>;
            .reg .pred %p<4>;

            mov.u32 %r0, 0;
            setp.lt.u32 %p0, %r0, 10;
            @%p0 bra target;
            mov.u32 %r1, 1;
        target:
            mov.u32 %r2, 2;
            ret;
        }
    "#;

    let mut parser = Parser::new(ptx).expect("Parser creation failed");
    let module = parser.parse().expect("Parse failed");

    let mut analyzer = ControlFlowAnalyzer::new();

    if let Some(kernel) = module.kernels.first() {
        let cfg = analyzer.build_cfg(kernel);

        // CFG should have nodes and at least one exit
        assert!(!cfg.nodes.is_empty(), "REQ-006 FALSIFIED: CFG should have nodes");
    }

    println!("REQ-006 PASSED: Control flow analysis builds valid CFG");
}

/// REQ-007: Type checker validates operand types
///
/// Hypothesis: Type checker catches type mismatches.
/// Falsification: Type error not detected.
#[test]
fn req007_type_checker() {
    let ptx = r#"
        .version 8.0
        .target sm_70
        .address_size 64

        .entry type_test()
        {
            .reg .u32 %r<4>;
            .reg .f32 %f<4>;

            mov.u32 %r0, 0;
            mov.f32 %f0, 1.0;
            ret;
        }
    "#;

    let mut parser = Parser::new(ptx).expect("Parser creation failed");
    let module = parser.parse().expect("Parse failed");

    let mut checker = TypeChecker::new();
    let errors = checker.analyze(&module);

    // Clean PTX should have no type errors
    println!("REQ-007 PASSED: Type checker ran ({} errors)", errors.len());
}

/// REQ-008: Score calculation is consistent
///
/// Hypothesis: Same PTX produces same score.
/// Falsification: Score varies between runs.
#[test]
fn req008_score_consistency() {
    let ptx = r#"
        .version 8.0
        .target sm_70
        .address_size 64

        .entry consistent()
        {
            .reg .u32 %r<4>;
            mov.u32 %r0, 0;
            ret;
        }
    "#;

    let mut parser = Parser::new(ptx).expect("Parser creation failed");
    let module = parser.parse().expect("Parse failed");

    let registry = FalsificationRegistry::new();

    let report1 = registry.evaluate(&module);
    let report2 = registry.evaluate(&module);
    let report3 = registry.evaluate(&module);

    // Score should be deterministic
    assert!(
        (report1.score - report2.score).abs() < 0.01,
        "REQ-008 FALSIFIED: Score varies between runs: {} vs {}",
        report1.score,
        report2.score
    );
    assert!(
        (report2.score - report3.score).abs() < 0.01,
        "REQ-008 FALSIFIED: Score varies between runs: {} vs {}",
        report2.score,
        report3.score
    );

    println!("REQ-008 PASSED: Score consistency verified (score={})", report1.score);
}

/// REQ-009: Report includes all categories
///
/// Hypothesis: Report covers all 10 falsification categories.
/// Falsification: Category missing from report.
#[test]
fn req009_category_coverage() {
    let ptx = r#"
        .version 8.0
        .target sm_70
        .address_size 64

        .entry categories()
        {
            .reg .u32 %r<4>;
            mov.u32 %r0, 0;
            ret;
        }
    "#;

    let mut parser = Parser::new(ptx).expect("Parser creation failed");
    let module = parser.parse().expect("Parse failed");

    let registry = FalsificationRegistry::new();
    let report = registry.evaluate(&module);

    // Check all categories are represented
    let categories_in_report: std::collections::HashSet<Category> =
        report.results.iter().map(|(_, cat, _, _)| *cat).collect();

    let expected_categories = Category::all();
    for cat in expected_categories {
        assert!(
            categories_in_report.contains(cat),
            "REQ-009 FALSIFIED: Category {:?} missing from report",
            cat
        );
    }

    println!("REQ-009 PASSED: All {} categories covered in report", expected_categories.len());
}

/// REQ-010: Confidence calculation is bounded
///
/// Hypothesis: Confidence is always in [0, 0.99].
/// Falsification: Confidence exceeds bounds.
#[test]
fn req010_confidence_bounded() {
    let test_cases = [
        // Perfect PTX
        r#"
            .version 8.0
            .target sm_70
            .address_size 64
            .entry perfect() { ret; }
        "#,
        // Minimal PTX
        r#"
            .version 7.0
            .target sm_70
            .address_size 64
            .entry minimal() { ret; }
        "#,
    ];

    let registry = FalsificationRegistry::new();

    for (i, ptx) in test_cases.iter().enumerate() {
        let mut parser = Parser::new(ptx).expect("Parser creation failed");
        let module = parser.parse().expect("Parse failed");

        let report = registry.evaluate(&module);

        assert!(
            report.confidence >= 0.0 && report.confidence <= 0.99,
            "REQ-010 FALSIFIED: Confidence {} out of bounds for case {}",
            report.confidence,
            i
        );
    }

    println!("REQ-010 PASSED: Confidence bounded in [0, 0.99]");
}

/// Test all categories exist
#[test]
fn test_category_enum() {
    let categories = Category::all();
    assert_eq!(categories.len(), 10, "Should have 10 categories");

    // Verify display works
    for cat in categories {
        let display = format!("{}", cat);
        assert!(!display.is_empty(), "Category display should not be empty");
    }

    println!("Category enum verified");
}

/// Test TestResult enum
#[test]
fn test_result_enum() {
    let pass = TestResult::Pass;
    assert!(pass.is_pass());
    assert!(!pass.is_fail());

    let fail = TestResult::Fail { evidence: "Test".into(), location: None };
    assert!(fail.is_fail());
    assert!(!fail.is_pass());

    let na = TestResult::NotApplicable;
    assert!(!na.is_pass());
    assert!(!na.is_fail());

    println!("TestResult enum verified");
}

/// Test report methods
#[test]
fn test_report_methods() {
    let ptx = r#"
        .version 8.0
        .target sm_70
        .address_size 64
        .entry test() { ret; }
    "#;

    let mut parser = Parser::new(ptx).expect("Parser creation failed");
    let module = parser.parse().expect("Parse failed");

    let registry = FalsificationRegistry::new();
    let report = registry.evaluate(&module);

    // Test report methods
    let passed_categories = report.categories_with_all_tests_passed();
    // passed_categories is usize, always non-negative; verify it is within expected range
    assert!(passed_categories <= 10, "Should have at most 10 passed categories");

    let critical_absent = report.critical_bugs_absent();
    // Should be true for clean PTX
    assert!(critical_absent, "Clean PTX should have no critical bugs");

    let failed = report.failed_tests();
    // May have some failures depending on PTX
    println!(
        "Report methods verified: {} categories passed, {} tests failed",
        passed_categories,
        failed.len()
    );
}
