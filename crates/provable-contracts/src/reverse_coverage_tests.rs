#![allow(clippy::all)]
use super::*;

// ---------------------------------------------------------------------------
// extract_feature_gate
// ---------------------------------------------------------------------------

#[test]
fn test_extract_feature_gate_simple() {
    let gate = extract_feature_gate(r#"#[cfg(feature = "gpu")]"#);
    assert_eq!(gate.as_deref(), Some("gpu"));
}

#[test]
fn test_extract_feature_gate_all_test() {
    let gate = extract_feature_gate(r#"#[cfg(all(test, feature = "gpu"))]"#);
    assert_eq!(gate.as_deref(), Some("gpu"));
}

#[test]
fn test_extract_feature_gate_not() {
    let gate = extract_feature_gate(r#"#[cfg(not(feature = "x"))]"#);
    assert_eq!(gate.as_deref(), Some("x"));
}

#[test]
fn test_extract_feature_gate_no_feature() {
    let gate = extract_feature_gate(r"#[cfg(test)]");
    assert_eq!(gate, None);
}

#[test]
fn test_extract_feature_gate_target_os() {
    let gate = extract_feature_gate(r#"#[cfg(target_os = "linux")]"#);
    assert_eq!(gate, None);
}

#[test]
fn test_extract_feature_gate_complex_all() {
    let gate =
        extract_feature_gate(r#"#[cfg(all(not(miri), feature = "simd", target_arch = "x86"))]"#);
    assert_eq!(gate.as_deref(), Some("simd"));
}

// ---------------------------------------------------------------------------
// extract_bound_functions
// ---------------------------------------------------------------------------

#[test]
fn test_extract_bound_functions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binding.yaml");
    std::fs::write(
        &path,
        "bindings:\n  - function: \"Foo::bar\"\n    status: implemented\n  - function: baz\n    status: implemented\n",
    )
    .unwrap();
    let names = extract_bound_functions(&path);
    assert!(names.contains("bar"), "Expected 'bar' in {names:?}");
    assert!(names.contains("baz"), "Expected 'baz' in {names:?}");
}

#[test]
fn test_extract_bound_functions_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nonexistent.yaml");
    let names = extract_bound_functions(&path);
    assert!(names.is_empty());
}

#[test]
fn test_extract_bound_functions_no_function_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binding.yaml");
    std::fs::write(
        &path,
        "bindings:\n  - contract: foo-v1\n    status: implemented\n",
    )
    .unwrap();
    let names = extract_bound_functions(&path);
    assert!(names.is_empty());
}

#[test]
fn test_extract_bound_functions_quoted_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binding.yaml");
    std::fs::write(
        &path,
        "bindings:\n  - function: 'Module::compute_attention'\n  - function: \"train_step\"\n",
    )
    .unwrap();
    let names = extract_bound_functions(&path);
    assert!(
        names.contains("compute_attention"),
        "Expected 'compute_attention' in {names:?}"
    );
    assert!(
        names.contains("train_step"),
        "Expected 'train_step' in {names:?}"
    );
}

#[test]
fn test_extract_bound_functions_lowercased() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binding.yaml");
    std::fs::write(&path, "  - function: \"Module::FooBar\"\n").unwrap();
    let names = extract_bound_functions(&path);
    assert!(
        names.contains("foobar"),
        "Expected lowercased 'foobar' in {names:?}"
    );
}

// ---------------------------------------------------------------------------
// scan_file
// ---------------------------------------------------------------------------

#[test]
fn test_scan_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        "pub fn hello() {}\n#[contract(\"test\", equation = \"eq\")]\npub fn world() {}\nfn private() {}\n",
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 2);
    assert!(!results[0].has_contract_macro);
    assert!(results[1].has_contract_macro);
}

#[test]
fn test_scan_file_skips_main_and_new() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        "pub fn main() {}\npub fn new() {}\npub fn real_fn() {}\n",
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "real_fn");
}

#[test]
fn test_scan_file_pub_async_fn() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(tmp.path(), "pub async fn serve() {}\n").unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "serve");
}

#[test]
fn test_scan_file_generic_fn() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(tmp.path(), "pub fn process<T: Clone>(x: T) -> T {}\n").unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "process");
}

#[test]
fn test_scan_file_cfg_feature_gate_on_function() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        "#[cfg(feature = \"gpu\")]\npub fn gpu_dispatch() {}\npub fn cpu_dispatch() {}\n",
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].path, "gpu_dispatch");
    assert_eq!(results[0].feature_gate.as_deref(), Some("gpu"));
    assert_eq!(results[1].path, "cpu_dispatch");
    assert!(results[1].feature_gate.is_none());
}

#[test]
fn test_scan_file_cfg_feature_gate_on_module() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        r#"#[cfg(feature = "simd")]
mod simd_impl {
    pub fn simd_add() {}
}
pub fn normal_fn() {}
"#,
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].path, "simd_add");
    assert_eq!(results[0].feature_gate.as_deref(), Some("simd"));
    assert_eq!(results[1].path, "normal_fn");
    assert!(results[1].feature_gate.is_none());
}

#[test]
fn test_scan_file_records_line_number() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        "// comment\n// another\npub fn third_line() {}\n",
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].line, 3);
}

#[test]
fn test_scan_file_empty_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(tmp.path(), "").unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert!(results.is_empty());
}

#[test]
fn test_scan_file_private_fns_only() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(tmp.path(), "fn private_a() {}\nfn private_b() {}\n").unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert!(results.is_empty());
}

#[test]
fn test_scan_file_contract_annotation_not_sticky() {
    // #[contract] only applies to the immediately following pub fn, not later ones
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        "#[contract(\"test\", equation = \"eq\")]\npub fn annotated() {}\npub fn not_annotated() {}\n",
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 2);
    assert!(results[0].has_contract_macro);
    assert!(!results[1].has_contract_macro);
}

#[test]
fn test_scan_file_contract_reset_by_non_attr_line() {
    // A non-comment, non-attribute line between #[contract] and pub fn resets the flag
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        "#[contract(\"test\", equation = \"eq\")]\nlet x = 1;\npub fn after_reset() {}\n",
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 1);
    assert!(!results[0].has_contract_macro);
}

// ---------------------------------------------------------------------------
// scan_pub_fns
// ---------------------------------------------------------------------------

#[test]
fn test_scan_pub_fns_src_dir() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn alpha() {}\npub fn beta() {}\n").unwrap();

    let results = scan_pub_fns(dir.path());
    assert_eq!(results.len(), 2);
}

#[test]
fn test_scan_pub_fns_skips_target_and_tests_dirs() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn visible() {}\n").unwrap();

    // These should be skipped
    let target = src.join("target");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("hidden.rs"), "pub fn hidden() {}\n").unwrap();

    let tests = src.join("tests");
    std::fs::create_dir_all(&tests).unwrap();
    std::fs::write(tests.join("test.rs"), "pub fn test_hidden() {}\n").unwrap();

    let results = scan_pub_fns(dir.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "visible");
}

#[test]
fn test_scan_pub_fns_no_src_dir() {
    let dir = tempfile::tempdir().unwrap();
    let results = scan_pub_fns(dir.path());
    assert!(results.is_empty());
}

#[test]
fn test_scan_pub_fns_nested_modules() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("src").join("sub").join("deep");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("mod.rs"), "pub fn deep_fn() {}\n").unwrap();

    let results = scan_pub_fns(dir.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "deep_fn");
}

// ---------------------------------------------------------------------------
// reverse_coverage (integration)
// ---------------------------------------------------------------------------

#[test]
fn test_reverse_coverage_all_bound() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn alpha() {}\npub fn beta() {}\n").unwrap();

    let binding = dir.path().join("binding.yaml");
    std::fs::write(
        &binding,
        "bindings:\n  - function: alpha\n  - function: beta\n",
    )
    .unwrap();

    let report = reverse_coverage(dir.path(), &binding);
    assert_eq!(report.total_pub_fns, 2);
    assert_eq!(report.bound_fns, 2);
    assert!(report.unbound.is_empty());
    assert!((report.coverage_pct - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_reverse_coverage_some_unbound() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    // Use a name that won't be auto-exempt: must be >4 chars, no underscore, not in exempt list
    std::fs::write(
        src.join("lib.rs"),
        "pub fn nonexempt_function_alpha_zz() {}\npub fn nonexempt_function_beta_zz() {}\n",
    )
    .unwrap();

    let binding = dir.path().join("binding.yaml");
    std::fs::write(
        &binding,
        "bindings:\n  - function: nonexempt_function_alpha_zz\n",
    )
    .unwrap();

    let report = reverse_coverage(dir.path(), &binding);
    assert_eq!(report.total_pub_fns, 2);
    assert_eq!(report.bound_fns, 1);
    assert_eq!(report.unbound.len(), 1);
    assert_eq!(report.unbound[0].path, "nonexempt_function_beta_zz");
}

#[test]
fn test_reverse_coverage_annotated_fn() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        "#[contract(\"x\", equation = \"eq\")]\npub fn annotated_thing() {}\n",
    )
    .unwrap();

    let binding = dir.path().join("binding.yaml");
    std::fs::write(&binding, "bindings: []\n").unwrap();

    let report = reverse_coverage(dir.path(), &binding);
    assert_eq!(report.total_pub_fns, 1);
    assert_eq!(report.annotated_fns, 1);
    assert_eq!(report.bound_fns, 1);
    assert!(report.unbound.is_empty());
}

#[test]
fn test_reverse_coverage_auto_exempt() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    // "default" is in the auto-exempt list as an exact match
    std::fs::write(src.join("lib.rs"), "pub fn default() {}\n").unwrap();

    let binding = dir.path().join("binding.yaml");
    std::fs::write(&binding, "bindings: []\n").unwrap();

    let report = reverse_coverage(dir.path(), &binding);
    assert_eq!(report.total_pub_fns, 1);
    assert_eq!(report.exempt_fns, 1);
    assert!(report.unbound.is_empty());
}

#[test]
fn test_reverse_coverage_empty_crate() {
    let dir = tempfile::tempdir().unwrap();
    let binding = dir.path().join("binding.yaml");
    std::fs::write(&binding, "bindings: []\n").unwrap();

    let report = reverse_coverage(dir.path(), &binding);
    assert_eq!(report.total_pub_fns, 0);
    assert!((report.coverage_pct - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_reverse_coverage_case_insensitive_matching() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn MyFunction() {}\n").unwrap();

    let binding = dir.path().join("binding.yaml");
    std::fs::write(&binding, "  - function: \"Mod::myfunction\"\n").unwrap();

    let report = reverse_coverage(dir.path(), &binding);
    assert_eq!(report.total_pub_fns, 1);
    assert_eq!(report.bound_fns, 1);
}

// ---------------------------------------------------------------------------
// scan_dir edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_scan_dir_skips_git_dir() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();

    let git = src.join(".git");
    std::fs::create_dir_all(&git).unwrap();
    std::fs::write(git.join("file.rs"), "pub fn hidden() {}\n").unwrap();

    std::fs::write(src.join("lib.rs"), "pub fn visible() {}\n").unwrap();

    let results = scan_pub_fns(dir.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "visible");
}

#[test]
fn test_scan_dir_handles_non_rs_files() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn real() {}\n").unwrap();
    std::fs::write(src.join("notes.txt"), "pub fn fake() {}\n").unwrap();
    std::fs::write(src.join("data.json"), "{}").unwrap();

    let results = scan_pub_fns(dir.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "real");
}

// ---------------------------------------------------------------------------
// scan_file: feature gate cleared after module exit
// ---------------------------------------------------------------------------

#[test]
fn test_scan_file_feature_gate_cleared_after_module_close() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        r#"#[cfg(feature = "gpu")]
mod gpu_impl {
    pub fn gpu_only() {}
}
pub fn after_module() {}
"#,
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 2);
    // gpu_only should have the gate
    assert_eq!(results[0].feature_gate.as_deref(), Some("gpu"));
    // after_module should NOT have the gate
    assert!(
        results[1].feature_gate.is_none(),
        "feature gate should be cleared after module closes"
    );
}

// ---------------------------------------------------------------------------
// scan_file: pending_feature_gate cleared by non-attr/non-comment line
// ---------------------------------------------------------------------------

#[test]
fn test_scan_file_pending_gate_cleared_by_code() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        r#"#[cfg(feature = "gpu")]
let x = 42;
pub fn no_gate_here() {}
"#,
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].feature_gate.is_none(),
        "pending gate should be cleared by intervening code line"
    );
}
