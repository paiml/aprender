//! #5695: the mock `Page::take_coverage` / `stop_coverage` contract.
//!
//! These run against the mock `Page` (the `browser` feature off), which is the
//! build every default-feature consumer gets.
#![cfg(not(feature = "browser"))]

use jugar_probar::{FunctionCoverage, Page};

fn func(name: &str) -> FunctionCoverage {
    FunctionCoverage {
        function_name: name.to_string(),
        ranges: Vec::new(),
        is_block_coverage: false,
    }
}

#[test]
fn take_coverage_before_start_is_invalid_state() {
    let page = Page::new(800, 600);
    assert!(!page.is_coverage_enabled());
    let err = page.take_coverage().expect_err("coverage is not enabled");
    assert!(
        err.to_string().contains("Coverage not enabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn take_coverage_reports_page_url_and_added_functions() {
    let mut page = Page::new(800, 600);
    page.start_coverage().expect("mock start never fails");
    assert!(page.is_coverage_enabled());
    page.add_mock_coverage(func("alpha"));
    page.add_mock_coverage(func("beta"));

    let report = page.take_coverage().expect("enabled");
    assert_eq!(report.scripts.len(), 1);
    let script = &report.scripts[0];
    assert_eq!(script.script_id, "mock-script-1");
    assert_eq!(script.url, "about:blank");
    let names: Vec<&str> = script
        .functions
        .iter()
        .map(|f| f.function_name.as_str())
        .collect();
    assert_eq!(names, ["alpha", "beta"]);
    assert!(report.timestamp_ms > 0);
    // take_coverage is a snapshot: it must not disable collection.
    assert!(page.is_coverage_enabled());
}

#[test]
fn stop_coverage_returns_final_report_then_disables() {
    let mut page = Page::new(800, 600);
    page.start_coverage().expect("start");
    page.add_mock_coverage(func("gamma"));

    let report = page.stop_coverage().expect("enabled");
    assert_eq!(report.scripts[0].functions.len(), 1);
    assert!(!page.is_coverage_enabled());
    assert!(page.take_coverage().is_err());
    assert!(page.stop_coverage().is_err());
}
