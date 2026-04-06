use super::registry::calculate_confidence;
use super::*;
use crate::parser::Parser;

#[test]
fn test_registry_creation() {
    let registry = FalsificationRegistry::new();
    assert!(!registry.tests().is_empty());
    // Should have 100 tests
    assert!(
        registry.tests().len() >= 90,
        "Expected at least 90 tests, got {}",
        registry.tests().len()
    );
}

#[test]
fn test_valid_ptx_passes() {
    let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                mov.u32 %r0, 0;
                ret;
            }
        "#;
    let mut parser = Parser::new(ptx).expect("parser creation should succeed");
    let module = parser.parse().expect("parsing should succeed");

    let registry = FalsificationRegistry::new();
    let report = registry.evaluate(&module);

    // Should pass basic syntax tests
    assert!(report.score >= 80.0, "Score too low: {}", report.score);
    assert!(report.confidence > 0.7, "Confidence too low: {}", report.confidence);
}

#[test]
fn test_missing_version_fails() {
    let ptx = r#"
            .target sm_70
            .address_size 64

            .entry test()
            {
                ret;
            }
        "#;
    let mut parser = Parser::new(ptx).expect("parser creation should succeed");
    let module = parser.parse().expect("parsing should succeed");

    let registry = FalsificationRegistry::new();
    let report = registry.evaluate(&module);

    // F001 should fail
    let f001_result =
        report.results.iter().find(|(id, _, _, _)| id == "F001").map(|(_, _, _, r)| r);
    assert!(f001_result.is_some());
    assert!(f001_result.expect("F001 test should exist").is_fail());
}

#[test]
fn test_confidence_calculation() {
    // Full pass should have high confidence
    let conf = calculate_confidence(100, 100, &[]);
    assert!(conf > 0.9);

    // Partial pass should have lower confidence
    let conf = calculate_confidence(50, 100, &[]);
    assert!(conf < 0.8);
}
