use super::*;

#[test]
fn test_is_fma_intrinsic() {
    assert!(is_fma_intrinsic("_mm256_fmadd_ps"));
    assert!(is_fma_intrinsic("_mm_fmsub_ps"));
    assert!(is_fma_intrinsic("_mm256_fnmadd_ps"));
    assert!(is_fma_intrinsic("_mm256_fnmsub_ps"));
    assert!(is_fma_intrinsic("_mm_fnmadd_ps"));
    assert!(is_fma_intrinsic("_mm_fnmsub_ps"));
    assert!(!is_fma_intrinsic("_mm256_add_ps"));
    assert!(!is_fma_intrinsic("_mm_mul_ps"));
}

#[test]
fn test_get_intrinsic_patterns() {
    assert!(get_intrinsic_patterns("sse2").is_some());
    assert!(get_intrinsic_patterns("avx2").is_some());
    assert!(get_intrinsic_patterns("avx512").is_some());
    assert!(get_intrinsic_patterns("neon").is_some());
    assert!(get_intrinsic_patterns("unknown").is_none());
    assert!(get_intrinsic_patterns("").is_none());
}

#[test]
fn test_intrinsic_pattern_fields() {
    let pattern = get_intrinsic_patterns("sse2").unwrap();
    assert_eq!(pattern.required_feature, "sse2");
    assert!(pattern.pattern.is_match("_mm_add_ps"));
    assert!(!pattern.pattern.is_match("_mm256_add_ps"));

    let pattern = get_intrinsic_patterns("avx2").unwrap();
    assert_eq!(pattern.required_feature, "avx2");
    assert!(pattern.pattern.is_match("_mm256_add_ps"));

    let pattern = get_intrinsic_patterns("avx512").unwrap();
    assert_eq!(pattern.required_feature, "avx512f");
    assert!(pattern.pattern.is_match("_mm512_add_ps"));

    let pattern = get_intrinsic_patterns("neon").unwrap();
    assert_eq!(pattern.required_feature, "neon");
    assert!(pattern.pattern.is_match("vaddq_f32"));
}

#[test]
fn test_check_attribute_mismatch() {
    let mut intrinsics = HashSet::new();
    intrinsics.insert("_mm512_add_ps".to_string());

    assert!(check_attribute_mismatch("avx2", &intrinsics).is_some());
    assert!(check_attribute_mismatch("avx512f", &intrinsics).is_none());

    // Test AVX2 with SSE2 attribute
    let mut intrinsics = HashSet::new();
    intrinsics.insert("_mm256_add_ps".to_string());
    assert!(check_attribute_mismatch("sse2", &intrinsics).is_some());

    // Test no mismatch for SSE2
    let mut intrinsics = HashSet::new();
    intrinsics.insert("_mm_add_ps".to_string());
    assert!(check_attribute_mismatch("sse2", &intrinsics).is_none());

    // Test AVX-512 attribute with no AVX-512 intrinsics
    let mut intrinsics = HashSet::new();
    intrinsics.insert("_mm256_add_ps".to_string());
    assert!(check_attribute_mismatch("avx512f", &intrinsics).is_some());
}

#[test]
fn test_check_fma_feature() {
    let mut intrinsics = HashSet::new();
    intrinsics.insert("_mm256_fmadd_ps".to_string());

    assert!(check_fma_feature("avx2", &intrinsics).is_some());
    assert!(check_fma_feature("avx2,fma", &intrinsics).is_none());
    assert!(check_fma_feature("fma", &intrinsics).is_none());

    // Test with non-FMA intrinsics
    let mut intrinsics = HashSet::new();
    intrinsics.insert("_mm256_add_ps".to_string());
    assert!(check_fma_feature("avx2", &intrinsics).is_none());
}

#[test]
fn test_check_target_feature_attribute() {
    let lines = vec![
        String::new(),
        "#[target_feature(enable = \"avx2\")]".to_string(),
        "unsafe fn foo() {".to_string(),
    ];
    assert_eq!(check_target_feature_attribute(&lines, 2), Some("avx2".to_string()));

    // Test with no attribute
    let lines = vec![String::new(), "unsafe fn foo() {".to_string()];
    assert_eq!(check_target_feature_attribute(&lines, 1), None);

    // Test with attribute far away (>15 lines)
    let mut lines = vec!["#[target_feature(enable = \"sse2\")]".to_string()];
    for _ in 0..20 {
        lines.push(String::new());
    }
    lines.push("unsafe fn foo() {".to_string());
    assert_eq!(check_target_feature_attribute(&lines, 21), None);
}

#[test]
fn test_has_safety_comment() {
    let lines = vec!["// SAFETY: This is safe".to_string(), "unsafe fn foo() {".to_string()];
    assert!(has_safety_comment(&lines, 1));

    let lines = vec!["unsafe fn foo() {".to_string()];
    assert!(!has_safety_comment(&lines, 0));

    // Test with comment far away (>10 lines)
    let mut lines = vec!["// SAFETY: Safe".to_string()];
    for _ in 0..15 {
        lines.push(String::new());
    }
    lines.push("unsafe fn foo() {".to_string());
    assert!(!has_safety_comment(&lines, 16));
}

#[test]
fn test_has_inline_attribute() {
    let lines = vec!["#[inline]".to_string(), "unsafe fn foo() {".to_string()];
    assert!(has_inline_attribute(&lines, 1));

    let lines = vec!["#[inline(always)]".to_string(), "unsafe fn foo() {".to_string()];
    assert!(has_inline_attribute(&lines, 1));

    let lines = vec!["unsafe fn foo() {".to_string()];
    assert!(!has_inline_attribute(&lines, 0));
}

#[test]
fn test_find_intrinsics_in_function() {
    let pattern = Regex::new(r"_mm256_\w+").unwrap();
    let lines = vec![
        "unsafe fn foo() {".to_string(),
        "    let a = _mm256_add_ps(x, y);".to_string(),
        "    let b = _mm256_mul_ps(a, z);".to_string(),
        "}".to_string(),
    ];

    let (end, intrinsics) = find_intrinsics_in_function(&lines, 0, &pattern);
    assert_eq!(end, 3);
    assert_eq!(intrinsics.len(), 2);
    assert!(intrinsics.contains("_mm256_add_ps"));
    assert!(intrinsics.contains("_mm256_mul_ps"));

    // Test nested braces
    let lines = vec![
        "unsafe fn foo() {".to_string(),
        "    if true {".to_string(),
        "        let a = _mm256_add_ps(x, y);".to_string(),
        "    }".to_string(),
        "}".to_string(),
    ];
    let (end, intrinsics) = find_intrinsics_in_function(&lines, 0, &pattern);
    assert_eq!(end, 4);
    assert_eq!(intrinsics.len(), 1);
}

#[test]
fn test_violation_level() {
    assert_eq!(ViolationLevel::Critical, ViolationLevel::Critical);
    assert_ne!(ViolationLevel::Critical, ViolationLevel::Error);
    assert_ne!(ViolationLevel::Error, ViolationLevel::Warning);
}

#[test]
fn test_check_file_with_missing_target_feature() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r"
unsafe fn test_func() {{
let a = _mm256_add_ps(x, y);
}}
"
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "avx2").unwrap();
    assert!(!violations.is_empty());
    assert!(violations.iter().any(|v| v.level == ViolationLevel::Critical));
}

#[test]
fn test_check_file_with_proper_attributes() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r#"
// SAFETY: This is safe
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn test_func() {{
let a = _mm256_add_ps(x, y);
}}
"#
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "avx2").unwrap();
    // Should have no critical or error violations
    assert!(!violations.iter().any(|v| v.level == ViolationLevel::Critical));
    assert!(!violations.iter().any(|v| v.level == ViolationLevel::Error));
}

#[test]
fn test_check_file_with_attribute_mismatch() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r#"
#[target_feature(enable = "sse2")]
unsafe fn test_func() {{
let a = _mm256_add_ps(x, y);
}}
"#
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "avx2").unwrap();
    assert!(violations.iter().any(|v| v.level == ViolationLevel::Error));
}

#[test]
fn test_check_file_with_fma_missing() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r#"
#[target_feature(enable = "avx2")]
unsafe fn test_func() {{
let a = _mm256_fmadd_ps(x, y, z);
}}
"#
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "avx2").unwrap();
    assert!(violations
        .iter()
        .any(|v| v.level == ViolationLevel::Error && v.message.contains("FMA")));
}

#[test]
fn test_check_file_missing_safety_comment() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r#"
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn test_func() {{
let a = _mm256_add_ps(x, y);
}}
"#
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "avx2").unwrap();
    assert!(violations
        .iter()
        .any(|v| v.level == ViolationLevel::Warning && v.message.contains("SAFETY")));
}

#[test]
fn test_check_file_missing_inline() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r#"
// SAFETY: Safe
#[target_feature(enable = "avx2")]
unsafe fn test_func() {{
let a = _mm256_add_ps(x, y);
}}
"#
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "avx2").unwrap();
    assert!(violations
        .iter()
        .any(|v| v.level == ViolationLevel::Warning && v.message.contains("inline")));
}

#[test]
fn test_check_file_no_intrinsics() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r"
unsafe fn test_func() {{
let a = 1 + 2;
}}
"
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "avx2").unwrap();
    // No violations because no intrinsics used
    assert!(violations.is_empty());
}

#[test]
fn test_check_file_unknown_backend() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(
        temp_file,
        r"
unsafe fn test_func() {{
let a = _mm256_add_ps(x, y);
}}
"
    )
    .unwrap();

    let violations = check_file(temp_file.path(), "unknown").unwrap();
    // Should return empty for unknown backend
    assert!(violations.is_empty());
}

#[test]
fn test_violation_fields() {
    let v = Violation {
        level: ViolationLevel::Critical,
        filepath: PathBuf::from("test.rs"),
        line_num: 42,
        function_name: "test_func".to_string(),
        message: "Test message".to_string(),
        fix_suggestion: "Test fix".to_string(),
    };

    assert_eq!(v.level, ViolationLevel::Critical);
    assert_eq!(v.filepath, PathBuf::from("test.rs"));
    assert_eq!(v.line_num, 42);
    assert_eq!(v.function_name, "test_func");
    assert_eq!(v.message, "Test message");
    assert_eq!(v.fix_suggestion, "Test fix");
}
