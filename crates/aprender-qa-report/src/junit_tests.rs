use super::*;

use crate::mqs::{CategoryScores, GatewayResult};
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

fn test_score() -> MqsScore {
    MqsScore {
        model_id: "test/model".to_string(),
        raw_score: 850,
        normalized_score: 92.5,
        grade: "A-".to_string(),
        gateways: vec![GatewayResult::passed("G1", "Model loads")],
        gateways_passed: true,
        categories: CategoryScores::default(),
        total_tests: 100,
        tests_passed: 95,
        tests_failed: 5,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    }
}

#[test]
fn test_junit_basic() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated(
        "F-QUAL-001",
        test_scenario(),
        "4",
        100,
    ));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("<?xml"));
    assert!(xml.contains("testsuite"));
    assert!(xml.contains("TestSuite"));
    assert!(xml.contains("tests=\"1\""));
}

#[test]
fn test_junit_with_failure() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::falsified(
        "F-QUAL-001",
        test_scenario(),
        "Wrong answer",
        "5",
        100,
    ));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("failures=\"1\""));
    assert!(xml.contains("<failure"));
    assert!(xml.contains("Wrong answer"));
}

#[test]
fn test_junit_with_error() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::crashed(
        "F-QUAL-001",
        test_scenario(),
        "SIGSEGV",
        -11,
        0,
    ));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("errors=\"1\""));
    assert!(xml.contains("<error"));
    assert!(xml.contains("CrashError"));
}

#[test]
fn test_junit_with_skip() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();

    // Create skipped evidence manually
    let mut evidence = Evidence::corroborated("F-QUAL-001", test_scenario(), "", 0);
    evidence.outcome = Outcome::Skipped;
    collector.add(evidence);

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("skipped=\"1\""));
    assert!(xml.contains("<skipped"));
}

#[test]
fn test_junit_mqs_properties() {
    let report = JunitReport::new("TestSuite");
    let collector = EvidenceCollector::new();
    let score = test_score();

    let xml = report
        .generate(&collector, &score)
        .expect("Failed to generate");

    assert!(xml.contains("mqs.raw_score"));
    assert!(xml.contains("mqs.normalized_score"));
    assert!(xml.contains("mqs.grade"));
    assert!(xml.contains("\"A-\""));
}

#[test]
fn test_xml_escaping() {
    assert_eq!(JunitReport::escape_xml("<test>"), "&lt;test&gt;");
    assert_eq!(JunitReport::escape_xml("a & b"), "a &amp; b");
    assert_eq!(JunitReport::escape_xml("\"quoted\""), "&quot;quoted&quot;");
}

#[test]
fn test_junit_custom_class_name() {
    let report = JunitReport::new("Suite").with_class_name("CustomClass");
    let collector = EvidenceCollector::new();

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    // Class name should appear (but no testcases since empty collector)
    assert!(xml.contains("Suite"));
}

#[test]
fn test_junit_with_timeout() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::timeout("F-PERF-001", test_scenario(), 30000));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    // Timeout renders as <error> element, not <failure> — so failures=0, errors=1
    assert!(xml.contains("failures=\"0\""));
    assert!(xml.contains("errors=\"1\""));
    assert!(xml.contains("TimeoutError"));
}

#[test]
fn test_xml_escape_apos() {
    assert_eq!(JunitReport::escape_xml("it's"), "it&apos;s");
}

#[test]
fn test_junit_multiple_cases() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));
    collector.add(Evidence::corroborated("F-002", test_scenario(), "ok", 100));
    collector.add(Evidence::falsified(
        "F-003",
        test_scenario(),
        "bad",
        "err",
        100,
    ));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("tests=\"3\""));
    assert!(xml.contains("failures=\"1\""));
}

#[test]
fn test_junit_testcase_name() {
    let report = JunitReport::new("Suite");
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated(
        "F-QUAL-001",
        test_scenario(),
        "ok",
        100,
    ));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("F-QUAL-001"));
}

#[test]
fn test_junit_time_attribute() {
    let report = JunitReport::new("Suite");
    let mut collector = EvidenceCollector::new();
    let mut evidence = Evidence::corroborated("F-001", test_scenario(), "ok", 1500);
    evidence.metrics.duration_ms = 1500;
    collector.add(evidence);

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("time="));
}

#[test]
fn test_junit_report_default() {
    let report = JunitReport::default();
    let collector = EvidenceCollector::new();

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("apr-qa-report"));
}

#[test]
fn test_junit_report_debug() {
    let report = JunitReport::new("Test");
    let debug_str = format!("{report:?}");
    assert!(debug_str.contains("JunitReport"));
}

#[test]
fn test_junit_with_stderr() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();
    let mut evidence = Evidence::crashed("F-QUAL-001", test_scenario(), "SIGSEGV", -11, 0);
    evidence.stderr = Some("Error details here".to_string());
    collector.add(evidence);

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("Error details"));
}

#[test]
fn test_junit_escape_special_chars() {
    let report = JunitReport::new("Test<Suite>");
    let collector = EvidenceCollector::new();

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("&lt;"));
    assert!(xml.contains("&gt;"));
}

#[test]
fn test_junit_failure_output() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::falsified(
        "F-QUAL-001",
        test_scenario(),
        "Test failed",
        "bad output",
        100,
    ));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    assert!(xml.contains("bad output"));
    assert!(xml.contains("Gate:"));
}

#[test]
fn test_junit_combined_errors_and_failures() {
    let report = JunitReport::new("TestSuite");
    let mut collector = EvidenceCollector::new();

    // Add a crash (error)
    collector.add(Evidence::crashed(
        "F-QUAL-001",
        test_scenario(),
        "Crash",
        -11,
        0,
    ));

    // Add a failure
    collector.add(Evidence::falsified(
        "F-QUAL-002",
        test_scenario(),
        "Wrong",
        "bad",
        100,
    ));

    let xml = report
        .generate(&collector, &test_score())
        .expect("Failed to generate");

    // Should have both errors and failures counted correctly
    assert!(xml.contains("errors=\"1\""));
    assert!(xml.contains("failures=\"1\""));
}
