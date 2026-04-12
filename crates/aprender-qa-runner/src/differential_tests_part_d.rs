#[test]
fn test_six_column_profile_all_assertions_passed_empty() {
    let profile = SixColumnProfile::default();
    // Popperian: no throughput measured → untested ≠ passed
    assert!(!profile.all_assertions_passed());
}

#[test]
fn test_six_column_profile_all_assertions_passed_with_failures() {
    let mut profile = SixColumnProfile::default();
    profile.failed_assertions.push(ProfileAssertion {
        format: "gguf".to_string(),
        backend: "cpu".to_string(),
        actual_tps: 5.0,
        min_threshold: 10.0,
        passed: false,
    });
    assert!(!profile.all_assertions_passed());
}

#[test]
fn test_six_column_profile_check_assertions_all_pass() {
    let mut profile = SixColumnProfile {
        tps_gguf_cpu: Some(20.0),
        tps_gguf_gpu: Some(50.0),
        tps_apr_cpu: Some(18.0),
        tps_apr_gpu: Some(45.0),
        ..Default::default()
    };
    profile.check_assertions(10.0, 30.0);
    assert!(profile.all_assertions_passed());
}

#[test]
fn test_six_column_profile_check_assertions_gguf_cpu_fail() {
    let mut profile = SixColumnProfile {
        tps_gguf_cpu: Some(5.0), // Below threshold
        tps_gguf_gpu: Some(50.0),
        ..Default::default()
    };
    profile.check_assertions(10.0, 30.0);
    assert!(!profile.all_assertions_passed());
    assert_eq!(profile.failed_assertions.len(), 1);
    assert_eq!(profile.failed_assertions[0].format, "gguf");
    assert_eq!(profile.failed_assertions[0].backend, "cpu");
}

#[test]
fn test_six_column_profile_check_assertions_gguf_gpu_fail() {
    let mut profile = SixColumnProfile {
        tps_gguf_cpu: Some(20.0),
        tps_gguf_gpu: Some(25.0), // Below threshold
        ..Default::default()
    };
    profile.check_assertions(10.0, 30.0);
    assert!(!profile.all_assertions_passed());
    assert_eq!(profile.failed_assertions.len(), 1);
    assert_eq!(profile.failed_assertions[0].format, "gguf");
    assert_eq!(profile.failed_assertions[0].backend, "gpu");
}

#[test]
fn test_six_column_profile_check_assertions_apr_cpu_fail() {
    let mut profile = SixColumnProfile {
        tps_apr_cpu: Some(5.0), // Below threshold
        ..Default::default()
    };
    profile.check_assertions(10.0, 30.0);
    assert!(!profile.all_assertions_passed());
    assert_eq!(profile.failed_assertions.len(), 1);
    assert_eq!(profile.failed_assertions[0].format, "apr");
    assert_eq!(profile.failed_assertions[0].backend, "cpu");
}

#[test]
fn test_six_column_profile_check_assertions_apr_gpu_fail() {
    let mut profile = SixColumnProfile {
        tps_apr_gpu: Some(20.0), // Below threshold
        ..Default::default()
    };
    profile.check_assertions(10.0, 30.0);
    assert!(!profile.all_assertions_passed());
    assert_eq!(profile.failed_assertions.len(), 1);
    assert_eq!(profile.failed_assertions[0].format, "apr");
    assert_eq!(profile.failed_assertions[0].backend, "gpu");
}

#[test]
fn test_six_column_profile_check_assertions_multiple_failures() {
    let mut profile = SixColumnProfile {
        tps_gguf_cpu: Some(5.0),
        tps_gguf_gpu: Some(20.0),
        tps_apr_cpu: Some(8.0),
        tps_apr_gpu: Some(25.0),
        ..Default::default()
    };
    profile.check_assertions(10.0, 30.0);
    // All 4 should fail
    assert_eq!(profile.failed_assertions.len(), 4);
}

#[test]
fn test_six_column_profile_check_assertions_none_values() {
    let mut profile = SixColumnProfile::default();
    profile.check_assertions(10.0, 30.0);
    // Popperian: no throughput measured → untested ≠ passed
    assert!(!profile.all_assertions_passed());
}

#[test]
fn test_profile_assertion_fields() {
    let assertion = ProfileAssertion {
        format: "safetensors".to_string(),
        backend: "gpu".to_string(),
        actual_tps: 25.5,
        min_threshold: 30.0,
        passed: false,
    };
    assert_eq!(assertion.format, "safetensors");
    assert_eq!(assertion.backend, "gpu");
    assert!((assertion.actual_tps - 25.5).abs() < f64::EPSILON);
    assert!((assertion.min_threshold - 30.0).abs() < f64::EPSILON);
    assert!(!assertion.passed);
}

#[test]
fn test_profile_assertion_clone() {
    let assertion = ProfileAssertion {
        format: "gguf".to_string(),
        backend: "cpu".to_string(),
        actual_tps: 15.0,
        min_threshold: 10.0,
        passed: true,
    };
    let cloned = assertion.clone();
    assert_eq!(cloned.format, assertion.format);
    assert_eq!(cloned.backend, assertion.backend);
}

#[test]
fn test_profile_assertion_debug() {
    let assertion = ProfileAssertion {
        format: "apr".to_string(),
        backend: "cpu".to_string(),
        actual_tps: 12.0,
        min_threshold: 10.0,
        passed: true,
    };
    let debug_str = format!("{assertion:?}");
    assert!(debug_str.contains("ProfileAssertion"));
}

#[test]
fn test_profile_assertion_serialization() {
    let assertion = ProfileAssertion {
        format: "gguf".to_string(),
        backend: "gpu".to_string(),
        actual_tps: 45.5,
        min_threshold: 40.0,
        passed: true,
    };
    let json = serde_json::to_string(&assertion).unwrap();
    let parsed: ProfileAssertion = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.format, "gguf");
    assert!(parsed.passed);
}

#[test]
fn test_six_column_profile_clone() {
    let profile = SixColumnProfile {
        tps_gguf_cpu: Some(15.0),
        total_duration_ms: 5000,
        ..Default::default()
    };
    let cloned = profile.clone();
    assert_eq!(cloned.tps_gguf_cpu, Some(15.0));
    assert_eq!(cloned.total_duration_ms, 5000);
}

#[test]
fn test_six_column_profile_debug() {
    let profile = SixColumnProfile::default();
    let debug_str = format!("{profile:?}");
    assert!(debug_str.contains("SixColumnProfile"));
}
