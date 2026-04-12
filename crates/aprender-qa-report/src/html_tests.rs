use super::*;

use crate::mqs::{CategoryScores, GatewayResult};
use crate::popperian::FalsificationDetail;

/// Create a default MQS score for HTML report test helpers
fn test_mqs() -> MqsScore {
    MqsScore {
        model_id: "test/model".to_string(),
        raw_score: 850,
        normalized_score: 92.5,
        grade: "A-".to_string(),
        gateways: vec![
            GatewayResult::passed("G1", "Model loads"),
            GatewayResult::passed("G2", "Inference works"),
            GatewayResult::passed("G3", "No crashes"),
            GatewayResult::passed("G4", "Output valid"),
        ],
        gateways_passed: true,
        categories: CategoryScores {
            qual: 180,
            perf: 130,
            stab: 170,
            comp: 130,
            edge: 120,
            regr: 120,
        },
        total_tests: 100,
        tests_passed: 95,
        tests_failed: 5,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    }
}

/// Create a default Popperian score for HTML report test helpers
fn test_popperian() -> PopperianScore {
    PopperianScore {
        model_id: "test/model".to_string(),
        hypotheses_tested: 100,
        corroborated: 95,
        falsified: 5,
        inconclusive: 0,
        corroboration_ratio: 0.95,
        severity_weighted_score: 0.93,
        confidence_level: 0.92,
        reproducibility_index: 0.85,
        black_swan_count: 0,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-EDGE-001".to_string(),
            hypothesis: "Handles empty input".to_string(),
            evidence: "Returned garbage".to_string(),
            severity: 3,
            is_black_swan: false,
            occurrence_count: 1,
        }],
    }
}

/// Verify HTML generation includes doctype, title, model ID, and score
#[test]
fn test_html_generation() {
    let dashboard = HtmlDashboard::new("Test Dashboard");
    let collector = EvidenceCollector::new();

    let html = dashboard
        .generate(&test_mqs(), &test_popperian(), &collector)
        .expect("Failed to generate");

    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("Test Dashboard"));
    assert!(html.contains("test/model"));
    assert!(html.contains("A-"));
    assert!(html.contains("92.5"));
}

/// Verify grade_color maps A+, B, C, F to correct hex colors
#[test]
fn test_grade_colors() {
    assert_eq!(HtmlDashboard::grade_color("A+"), "#00d26a");
    assert_eq!(HtmlDashboard::grade_color("B"), "#7bed9f");
    assert_eq!(HtmlDashboard::grade_color("C"), "#ffc107");
    assert_eq!(HtmlDashboard::grade_color("F"), "#ff4757");
}

/// Verify gateway rendering includes gateway ID and pass class
#[test]
fn test_gateway_rendering() {
    let dashboard = HtmlDashboard::new("Test");
    let mqs = test_mqs();

    let html = dashboard.render_gateways(&mqs);

    assert!(html.contains("G1"));
    assert!(html.contains("gateway-pass"));
    assert!(html.contains("Model loads"));
}

/// Verify category rendering includes category names and score fractions
#[test]
fn test_category_rendering() {
    let dashboard = HtmlDashboard::new("Test");
    let mqs = test_mqs();

    let html = dashboard.render_categories(&mqs);

    assert!(html.contains("QUAL"));
    assert!(html.contains("PERF"));
    assert!(html.contains("180/200"));
}

/// Verify falsification rendering includes gate ID and description
#[test]
fn test_falsification_rendering() {
    let dashboard = HtmlDashboard::new("Test");
    let popperian = test_popperian();

    let html = dashboard.render_falsifications(&popperian);

    assert!(html.contains("F-EDGE-001"));
    assert!(html.contains("empty input"));
}

/// Verify HTML escaping converts angle brackets to entities
#[test]
fn test_html_escaping() {
    assert_eq!(HtmlDashboard::escape_html("<script>"), "&lt;script&gt;");
}

/// Verify HTML escaping converts ampersand to entity
#[test]
fn test_html_escaping_ampersand() {
    assert_eq!(HtmlDashboard::escape_html("a & b"), "a &amp; b");
}

/// Verify HTML escaping converts double quotes to entities
#[test]
fn test_html_escaping_quotes() {
    assert_eq!(
        HtmlDashboard::escape_html("say \"hi\""),
        "say &quot;hi&quot;"
    );
}

/// Verify D and D+ grade colors map to orange
#[test]
fn test_grade_color_d() {
    assert_eq!(HtmlDashboard::grade_color("D"), "#ff9f43");
    assert_eq!(HtmlDashboard::grade_color("D+"), "#ff9f43");
}

/// Verify B+ and B- grade colors map to green
#[test]
fn test_grade_color_b_variants() {
    assert_eq!(HtmlDashboard::grade_color("B+"), "#7bed9f");
    assert_eq!(HtmlDashboard::grade_color("B-"), "#7bed9f");
}

/// Verify C+ and C- grade colors map to yellow
#[test]
fn test_grade_color_c_variants() {
    assert_eq!(HtmlDashboard::grade_color("C+"), "#ffc107");
    assert_eq!(HtmlDashboard::grade_color("C-"), "#ffc107");
}

/// Verify failed gateway renders with gateway-fail CSS class
#[test]
fn test_gateway_failed_rendering() {
    let dashboard = HtmlDashboard::new("Test");
    let mut mqs = test_mqs();
    mqs.gateways = vec![GatewayResult::failed("G1", "Model loads", "OOM")];
    mqs.gateways_passed = false;

    let html = dashboard.render_gateways(&mqs);
    assert!(html.contains("gateway-fail"));
}

/// Verify HTML dashboard includes custom title in output
#[test]
fn test_html_dashboard_default_title() {
    let dashboard = HtmlDashboard::new("MQS Report");
    let collector = EvidenceCollector::new();

    let html = dashboard
        .generate(&test_mqs(), &test_popperian(), &collector)
        .expect("Failed");

    assert!(html.contains("MQS Report"));
}

/// Verify black swan falsification renders gate ID in HTML
#[test]
fn test_popperian_with_black_swan_rendering() {
    let dashboard = HtmlDashboard::new("Test");
    let popperian = PopperianScore {
        model_id: "test".to_string(),
        hypotheses_tested: 100,
        corroborated: 99,
        falsified: 1,
        inconclusive: 0,
        corroboration_ratio: 0.99,
        severity_weighted_score: 0.99,
        confidence_level: 0.95,
        reproducibility_index: 1.0,
        black_swan_count: 1,
        falsifications: vec![FalsificationDetail {
            gate_id: "G1-CRASH".to_string(),
            hypothesis: "No crash".to_string(),
            evidence: "SIGSEGV".to_string(),
            severity: 5,
            is_black_swan: true,
            occurrence_count: 1,
        }],
    };

    let html = dashboard.render_falsifications(&popperian);
    assert!(html.contains("G1-CRASH"));
}

/// Verify without_charts disables chart rendering
#[test]
fn test_html_dashboard_without_charts() {
    let dashboard = HtmlDashboard::new("Test").without_charts();
    assert!(!dashboard.include_charts);
}

/// Verify default HtmlDashboard has empty title
#[test]
fn test_html_dashboard_default() {
    let dashboard = HtmlDashboard::default();
    assert!(dashboard.title.is_empty());
}

/// Verify HtmlDashboard Debug format contains struct name
#[test]
fn test_html_dashboard_debug() {
    let dashboard = HtmlDashboard::new("Test");
    let debug_str = format!("{dashboard:?}");
    assert!(debug_str.contains("HtmlDashboard"));
}

/// Verify HTML generation handles zero tests gracefully
#[test]
fn test_html_zero_tests() {
    let mut mqs = test_mqs();
    mqs.total_tests = 0;
    mqs.tests_passed = 0;
    mqs.tests_failed = 0;

    let dashboard = HtmlDashboard::new("Test");
    let collector = EvidenceCollector::new();

    let html = dashboard
        .generate(&mqs, &test_popperian(), &collector)
        .expect("Failed to generate");

    // Should handle zero tests gracefully
    assert!(html.contains("<!DOCTYPE html>"));
}

/// Verify unknown grade falls through to F color
#[test]
fn test_grade_color_unknown() {
    // Test fallback for unknown grade
    let color = HtmlDashboard::grade_color("Z");
    assert_eq!(color, "#ff4757"); // Falls through to F case
}

/// Verify HTML escaping handles adjacent angle brackets
#[test]
fn test_html_escaping_special_chars() {
    assert_eq!(HtmlDashboard::escape_html("test<>test"), "test&lt;&gt;test");
}

/// Verify 80% pass rate renders with warning CSS class
#[test]
fn test_pass_rate_warning_class() {
    // Test pass rate between 70 and 90 for "warning" class
    let mut mqs = test_mqs();
    mqs.tests_passed = 80;
    mqs.tests_failed = 20;
    mqs.total_tests = 100;

    let dashboard = HtmlDashboard::new("Test");
    let collector = EvidenceCollector::new();

    let html = dashboard
        .generate(&mqs, &test_popperian(), &collector)
        .expect("Failed to generate");

    // 80% pass rate should get "warning" class
    assert!(html.contains("warning"));
}

/// Verify empty gateways list renders appropriate message
#[test]
fn test_empty_gateways() {
    let dashboard = HtmlDashboard::new("Test");
    let mut mqs = test_mqs();
    mqs.gateways = vec![]; // Empty gateways

    let html = dashboard.render_gateways(&mqs);
    assert!(html.contains("No gateway checks recorded"));
}

/// Verify more than 10 falsifications shows overflow count
#[test]
fn test_more_than_ten_falsifications() {
    let dashboard = HtmlDashboard::new("Test");

    // Create more than 10 falsifications
    let mut falsifications = Vec::new();
    for i in 0..15 {
        falsifications.push(FalsificationDetail {
            gate_id: format!("F-EDGE-{:03}", i),
            hypothesis: format!("Test hypothesis {}", i),
            evidence: format!("Evidence {}", i),
            severity: 3,
            is_black_swan: false,
            occurrence_count: 1,
        });
    }

    let popperian = PopperianScore {
        model_id: "test".to_string(),
        hypotheses_tested: 100,
        corroborated: 85,
        falsified: 15,
        inconclusive: 0,
        corroboration_ratio: 0.85,
        severity_weighted_score: 0.80,
        confidence_level: 0.75,
        reproducibility_index: 0.70,
        black_swan_count: 0,
        falsifications,
    };

    let html = dashboard.render_falsifications(&popperian);
    assert!(html.contains("and 5 more falsifications"));
}
