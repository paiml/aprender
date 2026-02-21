use super::*;

#[test]
fn test_p7_heavy_check() {
    let result = p7_test_heavy_exists();
    assert_eq!(result.id, "P7");
    assert!(!result.details.is_empty());
}

#[test]
fn test_p8_nextest_check() {
    let result = p8_nextest_supported();
    assert_eq!(result.id, "P8");
    assert!(!result.details.is_empty());
}

#[test]
fn test_p9_ci_content_check() {
    let result = p9_ci_fast_first();
    assert_eq!(result.id, "P9");
    assert!(!result.details.is_empty());
}

#[test]
fn test_velocity_result_pass_creates_none_duration() {
    let result = VelocityResult::pass("id", "name", "details");
    assert!(result.duration.is_none());
}

#[test]
fn test_velocity_result_fail_creates_none_duration() {
    let result = VelocityResult::fail("id", "name", "details");
    assert!(result.duration.is_none());
}

#[test]
fn test_all_velocity_ids_are_sequential() {
    let results = run_all_velocity_tests();
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.id, format!("P{}", i + 1));
    }
}

#[test]
fn test_velocity_score_range() {
    let (passed, total) = velocity_score();
    assert!(passed <= total);
    assert!(total == 10);
}

#[test]
fn test_all_velocity_tests_have_names() {
    let results = run_all_velocity_tests();
    for result in results {
        assert!(!result.name.is_empty(), "Test {} has empty name", result.id);
    }
}

#[test]
fn test_all_velocity_tests_have_details() {
    let results = run_all_velocity_tests();
    for result in results {
        assert!(
            !result.details.is_empty(),
            "Test {} has empty details",
            result.id
        );
    }
}

#[test]
fn test_p3_coverage_value_check() {
    let result = p3_test_fast_coverage();
    // Verify the details contain expected format
    assert!(result.details.contains('%') || result.details.contains("Coverage"));
}

#[test]
fn test_p4_network_policy_description() {
    let result = p4_no_network_calls();
    assert!(
        result.details.contains("network")
            || result.details.contains("HF")
            || result.details.contains("integration")
    );
}

#[test]
fn test_p5_disk_policy_description() {
    let result = p5_no_disk_writes();
    assert!(
        result.details.contains("tmp")
            || result.details.contains("tempfile")
            || result.details.contains("disk")
    );
}

#[test]
fn test_p6_compile_policy_description() {
    let result = p6_compile_under_5s();
    assert!(
        result.details.contains("ncremental")
            || result.details.contains("compile")
            || result.details.contains("fast")
    );
}

#[test]
fn test_p10_sleep_policy_description() {
    let result = p10_no_sleep_in_fast();
    assert!(
        result.details.contains("ignore")
            || result.details.contains("sleep")
            || result.details.contains("excluded")
    );
}

#[test]
fn test_velocity_result_with_duration_preserves_other_fields() {
    let result = VelocityResult::fail("FAIL", "Fail Name", "Fail Details")
        .with_duration(Duration::from_secs(5));
    assert_eq!(result.id, "FAIL");
    assert_eq!(result.name, "Fail Name");
    assert!(!result.passed);
    assert_eq!(result.details, "Fail Details");
    assert_eq!(result.duration, Some(Duration::from_secs(5)));
}

#[test]
fn test_velocity_score_invokes_all_tests() {
    // This ensures velocity_score() actually runs all tests
    let (passed, total) = velocity_score();
    assert_eq!(total, 10);
    // The number passed depends on actual project state
    assert!(passed <= total);
}
