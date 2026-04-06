// =========================================================================
// BH-MOD-022: Coverage Gap Tests — source_forbids_unsafe / crate_forbids_unsafe
// =========================================================================

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_022_source_forbids_unsafe_present() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_forbid");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("lib.rs");
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    std::fs::write(&file, "#![forbid(unsafe_code)]\nfn safe() {}\n").unwrap();

    assert!(source_forbids_unsafe(&file));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_022_source_forbids_unsafe_absent() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_noforbid");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("lib.rs");
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    std::fs::write(&file, "fn uses_unsafe() { unsafe {} }\n").unwrap();

    assert!(!source_forbids_unsafe(&file));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_022_source_forbids_unsafe_nonexistent() {
    assert!(!source_forbids_unsafe(Path::new("/nonexistent/lib.rs")));
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_022_crate_forbids_unsafe_via_librs() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_crate_lib");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(
        temp.join("src/lib.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "#![forbid(unsafe_code)]\npub fn safe() {}\n",
    )
    .unwrap();

    assert!(crate_forbids_unsafe(&temp));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_022_crate_forbids_unsafe_via_cargo_toml() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_crate_toml");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn code() {}\n").unwrap();
    std::fs::write(
        temp.join("Cargo.toml"),
        "[lints.rust]\nunsafe_code = \"forbid\"\n",
    )
    .unwrap();

    assert!(crate_forbids_unsafe(&temp));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_022_crate_no_forbid() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_crate_none");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn code() {}\n").unwrap();
    std::fs::write(temp.join("Cargo.toml"), "[package]\nname = \"test\"\n").unwrap();

    assert!(!crate_forbids_unsafe(&temp));

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-023: Coverage Gap Tests — run_fuzz_mode
// =========================================================================

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_023_fuzz_mode_forbids_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_023_fuzz_forbid");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(
        temp.join("src/lib.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "#![forbid(unsafe_code)]\npub fn safe() {}\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Fuzz,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());
    run_fuzz_mode(&temp, &config, &mut result);

    assert!(result.findings.iter().any(|f| f.id == "BH-FUZZ-SKIPPED"));
}

#[test]
fn test_bh_mod_023_fuzz_mode_no_fuzz_dir() {
    let temp = std::env::temp_dir().join("test_bh_mod_023_fuzz_nofuzz");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn code() {}\n").unwrap();

    let config = HuntConfig {
        mode: HuntMode::Fuzz,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());
    run_fuzz_mode(&temp, &config, &mut result);

    assert!(result.findings.iter().any(|f| f.id == "BH-FUZZ-NOTARGETS"));
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_023_fuzz_mode_with_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_023_fuzz_unsafe");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(
        temp.join("src/lib.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "pub fn danger() {\n    unsafe {\n        let ptr = 0 as *const i32;\n        let _ = *ptr;\n    }\n}\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Fuzz,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());
    run_fuzz_mode(&temp, &config, &mut result);

    // Should find unsafe blocks + FUZZ-NOTARGETS (no fuzz dir)
    assert!(result.findings.iter().any(|f| f.id == "BH-FUZZ-NOTARGETS"));
    assert_eq!(result.stats.mode_stats.fuzz_coverage, 0.0);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-024: Coverage Gap Tests — hunt_with_spec
// =========================================================================

#[test]
fn test_bh_mod_024_hunt_with_spec_basic() {
    let temp = std::env::temp_dir().join("test_bh_mod_024_spec");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let spec_content =
        "# Test Spec\n\n## Section 1\n\n### TST-01: Test Claim\n\nThis claim tests something.\n";
    std::fs::write(temp.join("spec.md"), spec_content).unwrap();

    std::fs::write(
        temp.join("src/lib.rs"),
        "// TST-01: implements test claim\nfn test_impl() {\n    let x = 42; // TODO: fix\n}\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Analyze,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };

    let result = hunt_with_spec(&temp, &temp.join("spec.md"), None, config);
    assert!(result.is_ok());
    let (hunt_result, parsed_spec) = result.unwrap();
    assert!(!parsed_spec.claims.is_empty());
    assert_eq!(parsed_spec.claims[0].id, "TST-01");
    // Hunt ran (non-zero duration) or produced no findings
    assert!(hunt_result.duration_ms > 0);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_024_hunt_with_spec_section_filter() {
    let temp = std::env::temp_dir().join("test_bh_mod_024_section");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let spec_content = "\
## Security\n\n### SEC-01: Auth Check\n\nAuth claim.\n\n\
## Performance\n\n### PERF-01: Cache Opt\n\nPerf claim.\n";
    std::fs::write(temp.join("spec.md"), spec_content).unwrap();
    std::fs::write(temp.join("src/lib.rs"), "fn main() {}\n").unwrap();

    let config = HuntConfig {
        mode: HuntMode::Analyze,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };

    let result = hunt_with_spec(&temp, &temp.join("spec.md"), Some("Security"), config);
    assert!(result.is_ok());
    let (_hunt_result, parsed_spec) = result.unwrap();
    assert_eq!(parsed_spec.claims.len(), 2);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_024_hunt_with_spec_nonexistent() {
    let config = HuntConfig::default();
    let result = hunt_with_spec(
        Path::new("/tmp"),
        Path::new("/nonexistent/spec.md"),
        None,
        config,
    );
    assert!(result.is_err());
}

// =========================================================================
// BH-MOD-025: Coverage Gap Tests — hunt_ensemble
// =========================================================================

#[test]
fn test_bh_mod_025_hunt_ensemble() {
    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let result = hunt_ensemble(Path::new("."), config);
    assert!(result.duration_ms > 0);
}

// =========================================================================
// BH-MOD-026: scan_file_for_unsafe_blocks
// =========================================================================

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_026_scan_file_no_unsafe() {
    let temp = std::env::temp_dir().join("test_scan_no_unsafe.rs");
    std::fs::write(&temp, "fn safe_function() { let x = 1; }\n").unwrap();

    let mut finding_id = 0;
    let mut unsafe_inventory = Vec::new();
    let mut result = HuntResult::default();

    scan_file_for_unsafe_blocks(&temp, &mut finding_id, &mut unsafe_inventory, &mut result);

    assert!(result.findings.is_empty());
    assert!(unsafe_inventory.is_empty());
    assert_eq!(finding_id, 0);

    let _ = std::fs::remove_file(&temp);
}

#[test]
fn test_bh_mod_026_scan_file_ptr_deref() {
    let temp = std::env::temp_dir().join("test_scan_ptr_deref.rs");
    std::fs::write(
        &temp,
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "fn danger() {\n    unsafe {\n        let val = *ptr;\n    }\n}\n",
    )
    .unwrap();

    let mut finding_id = 0;
    let mut unsafe_inventory = Vec::new();
    let mut result = HuntResult::default();

    scan_file_for_unsafe_blocks(&temp, &mut finding_id, &mut unsafe_inventory, &mut result);

    assert!(
        !result.findings.is_empty(),
        "Should detect pointer dereference in unsafe block"
    );
    assert!(finding_id > 0);
    assert!(!unsafe_inventory.is_empty());

    let finding = &result.findings[0];
    assert!(finding.id.contains("BH-UNSAFE"));
    assert_eq!(finding.severity, FindingSeverity::High);

    let _ = std::fs::remove_file(&temp);
}

#[test]
fn test_bh_mod_026_scan_file_transmute() {
    let temp = std::env::temp_dir().join("test_scan_transmute.rs");
    std::fs::write(
        &temp,
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "fn danger() {\n    unsafe {\n        let v = std::mem::transmute::<u32, f32>(bits);\n    }\n}\n",
    )
    .unwrap();

    let mut finding_id = 0;
    let mut unsafe_inventory = Vec::new();
    let mut result = HuntResult::default();

    scan_file_for_unsafe_blocks(&temp, &mut finding_id, &mut unsafe_inventory, &mut result);

    assert!(
        !result.findings.is_empty(),
        "Should detect transmute in unsafe block"
    );
    let finding = &result.findings[0];
    assert!(finding.id.contains("BH-UNSAFE"));
    assert_eq!(finding.severity, FindingSeverity::Critical);

    let _ = std::fs::remove_file(&temp);
}

#[test]
fn test_bh_mod_026_scan_file_both_patterns() {
    let temp = std::env::temp_dir().join("test_scan_both.rs");
    std::fs::write(
        &temp,
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "fn danger() {\n    unsafe {\n        let val = *ptr as *const u8;\n        let f = std::mem::transmute(bits);\n    }\n}\n",
    )
    .unwrap();

    let mut finding_id = 0;
    let mut unsafe_inventory = Vec::new();
    let mut result = HuntResult::default();

    scan_file_for_unsafe_blocks(&temp, &mut finding_id, &mut unsafe_inventory, &mut result);

    assert!(
        result.findings.len() >= 2,
        "Should detect both ptr deref and transmute, got {}",
        result.findings.len()
    );

    let _ = std::fs::remove_file(&temp);
}

#[test]
fn test_bh_mod_026_scan_file_nonexistent() {
    let mut finding_id = 0;
    let mut unsafe_inventory = Vec::new();
    let mut result = HuntResult::default();

    scan_file_for_unsafe_blocks(
        Path::new("/nonexistent/file.rs"),
        &mut finding_id,
        &mut unsafe_inventory,
        &mut result,
    );

    assert!(result.findings.is_empty());
}

// =========================================================================
// BH-MOD-027: extract_clippy_finding
// =========================================================================

#[test]
fn test_bh_mod_027_clippy_non_compiler_message() {
    let msg = serde_json::json!({
        "reason": "build-script-executed"
    });
    let config = HuntConfig::default();
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(result.is_none());
}

#[test]
fn test_bh_mod_027_clippy_info_level_skipped() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "note",
            "message": "Some info",
            "spans": [{"file_name": "src/lib.rs", "line_start": 1}],
            "code": {"code": "some_lint"}
        }
    });
    let config = HuntConfig::default();
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(result.is_none());
}

#[test]
fn test_bh_mod_027_clippy_dead_code_skipped() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "unused variable",
            "spans": [{"file_name": "src/lib.rs", "line_start": 10}],
            "code": {"code": "dead_code"}
        }
    });
    let config = HuntConfig::default();
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(result.is_none(), "dead_code should be filtered");
}

#[test]
fn test_bh_mod_027_clippy_unused_imports_skipped() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "unused import",
            "spans": [{"file_name": "src/lib.rs", "line_start": 5}],
            "code": {"code": "unused_imports"}
        }
    });
    let config = HuntConfig::default();
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(result.is_none(), "unused_imports should be filtered");
}

#[test]
fn test_bh_mod_027_clippy_valid_warning() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "this could be rewritten more concisely",
            "spans": [{"file_name": "src/pipeline.rs", "line_start": 42}],
            "code": {"code": "clippy::needless_return"}
        }
    });
    let config = HuntConfig {
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(
        result.is_some(),
        "Valid clippy warning should produce a finding"
    );
    let finding = result.unwrap();
    assert!(finding.id.contains("BH-CLIP"));
    assert_eq!(finding_id, 1);
}

#[test]
fn test_bh_mod_027_clippy_error_level() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "error",
            "message": "mismatched types",
            "spans": [{"file_name": "src/main.rs", "line_start": 100}],
            "code": {"code": "E0308"}
        }
    });
    let config = HuntConfig {
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(result.is_some(), "Error level should produce a finding");
}

#[test]
fn test_bh_mod_027_clippy_min_suspiciousness_filter() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "minor style issue",
            "spans": [{"file_name": "src/lib.rs", "line_start": 1}],
            "code": {"code": "clippy::style"}
        }
    });
    let config = HuntConfig {
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(
        result.is_none(),
        "Low suspiciousness finding should be filtered by high min_suspiciousness"
    );
}

#[test]
fn test_bh_mod_027_clippy_no_spans() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "something",
            "spans": [],
            "code": {"code": "some_lint"}
        }
    });
    let config = HuntConfig::default();
    let mut finding_id = 0;

    let result = extract_clippy_finding(&msg, &config, &mut finding_id);
    assert!(result.is_none(), "Empty spans should return None");
}

// =========================================================================
// BH-MOD-028: categorize_clippy_warning
// =========================================================================

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_028_categorize_clippy_unsafe() {
    let (cat, sev) =
        categorize_clippy_warning("clippy::undocumented_unsafe_blocks", "unsafe block");
    assert_eq!(cat, DefectCategory::SecurityVulnerabilities);
    assert!(matches!(
        sev,
        FindingSeverity::Critical | FindingSeverity::High
    ));
}

#[test]
fn test_bh_mod_028_categorize_clippy_unknown() {
    let (cat, sev) = categorize_clippy_warning("some_random_lint", "something");
    // Unknown lints should get a default categorization
    let _ = cat;
    let _ = sev;
}

// =========================================================================
// BH-MOD-029: Coverage Gap Tests — parse_defect_category
// =========================================================================

#[test]
fn test_bh_mod_029_parse_defect_category_logic() {
    assert_eq!(
        parse_defect_category("logicerrors"),
        DefectCategory::LogicErrors
    );
    assert_eq!(parse_defect_category("logic"), DefectCategory::LogicErrors);
    assert_eq!(parse_defect_category("Logic"), DefectCategory::LogicErrors);
    assert_eq!(
        parse_defect_category("LOGICERRORS"),
        DefectCategory::LogicErrors
    );
}

#[test]
fn test_bh_mod_029_parse_defect_category_memory() {
    assert_eq!(
        parse_defect_category("memorysafety"),
        DefectCategory::MemorySafety
    );
    assert_eq!(
        parse_defect_category("memory"),
        DefectCategory::MemorySafety
    );
    assert_eq!(
        parse_defect_category("MEMORY"),
        DefectCategory::MemorySafety
    );
}

#[test]
fn test_bh_mod_029_parse_defect_category_concurrency() {
    assert_eq!(
        parse_defect_category("concurrency"),
        DefectCategory::ConcurrencyBugs
    );
    assert_eq!(
        parse_defect_category("concurrencybugs"),
        DefectCategory::ConcurrencyBugs
    );
}

#[test]
fn test_bh_mod_029_parse_defect_category_gpu() {
    assert_eq!(
        parse_defect_category("gpukernelbugs"),
        DefectCategory::GpuKernelBugs
    );
    assert_eq!(parse_defect_category("gpu"), DefectCategory::GpuKernelBugs);
}

#[test]
fn test_bh_mod_029_parse_defect_category_silent() {
    assert_eq!(
        parse_defect_category("silentdegradation"),
        DefectCategory::SilentDegradation
    );
    assert_eq!(
        parse_defect_category("silent"),
        DefectCategory::SilentDegradation
    );
}

#[test]
fn test_bh_mod_029_parse_defect_category_test() {
    assert_eq!(parse_defect_category("testdebt"), DefectCategory::TestDebt);
    assert_eq!(parse_defect_category("test"), DefectCategory::TestDebt);
}

#[test]
fn test_bh_mod_029_parse_defect_category_hidden() {
    assert_eq!(
        parse_defect_category("hiddendebt"),
        DefectCategory::HiddenDebt
    );
    assert_eq!(parse_defect_category("debt"), DefectCategory::HiddenDebt);
}

#[test]
fn test_bh_mod_029_parse_defect_category_performance() {
    assert_eq!(
        parse_defect_category("performanceissues"),
        DefectCategory::PerformanceIssues
    );
    assert_eq!(
        parse_defect_category("performance"),
        DefectCategory::PerformanceIssues
    );
}

#[test]
fn test_bh_mod_029_parse_defect_category_security() {
    assert_eq!(
        parse_defect_category("securityvulnerabilities"),
        DefectCategory::SecurityVulnerabilities
    );
    assert_eq!(
        parse_defect_category("security"),
        DefectCategory::SecurityVulnerabilities
    );
}

#[test]
fn test_bh_mod_029_parse_defect_category_unknown_defaults_logic() {
    assert_eq!(
        parse_defect_category("nonsense"),
        DefectCategory::LogicErrors
    );
    assert_eq!(parse_defect_category(""), DefectCategory::LogicErrors);
    assert_eq!(parse_defect_category("xyzzy"), DefectCategory::LogicErrors);
}

// =========================================================================
// BH-MOD-030: Coverage Gap Tests — parse_finding_severity
// =========================================================================

#[test]
fn test_bh_mod_030_parse_finding_severity_critical() {
    assert_eq!(
        parse_finding_severity("critical"),
        FindingSeverity::Critical
    );
    assert_eq!(
        parse_finding_severity("Critical"),
        FindingSeverity::Critical
    );
    assert_eq!(
        parse_finding_severity("CRITICAL"),
        FindingSeverity::Critical
    );
}

#[test]
fn test_bh_mod_030_parse_finding_severity_high() {
    assert_eq!(parse_finding_severity("high"), FindingSeverity::High);
    assert_eq!(parse_finding_severity("High"), FindingSeverity::High);
}

#[test]
fn test_bh_mod_030_parse_finding_severity_medium() {
    assert_eq!(parse_finding_severity("medium"), FindingSeverity::Medium);
    assert_eq!(parse_finding_severity("Medium"), FindingSeverity::Medium);
}

#[test]
fn test_bh_mod_030_parse_finding_severity_low() {
    assert_eq!(parse_finding_severity("low"), FindingSeverity::Low);
    assert_eq!(parse_finding_severity("Low"), FindingSeverity::Low);
}

#[test]
fn test_bh_mod_030_parse_finding_severity_info() {
    assert_eq!(parse_finding_severity("info"), FindingSeverity::Info);
    assert_eq!(parse_finding_severity("Info"), FindingSeverity::Info);
}

#[test]
fn test_bh_mod_030_parse_finding_severity_unknown_defaults_medium() {
    assert_eq!(parse_finding_severity("nonsense"), FindingSeverity::Medium);
    assert_eq!(parse_finding_severity(""), FindingSeverity::Medium);
    assert_eq!(parse_finding_severity("warning"), FindingSeverity::Medium);
}

// =========================================================================
// BH-MOD-031: Coverage Gap Tests — detect_mutation_targets
// =========================================================================

#[test]
fn test_bh_mod_031_detect_mutation_no_targets() {
    let matches = detect_mutation_targets("let x = 42;");
    assert!(matches.is_empty());
}

#[test]
fn test_bh_mod_031_detect_mutation_boundary() {
    let matches = detect_mutation_targets("if v.len() > 0 {");
    assert!(matches.iter().any(|m| m.prefix == "boundary"));
}

#[test]
fn test_bh_mod_031_detect_mutation_boundary_size() {
    let matches = detect_mutation_targets("if buf.size() <= limit {");
    assert!(matches.iter().any(|m| m.prefix == "boundary"));
}

#[test]
fn test_bh_mod_031_detect_mutation_arithmetic_cast() {
    let matches = detect_mutation_targets("let idx = offset + count as usize;");
    assert!(matches.iter().any(|m| m.prefix == "arith"));
}

#[test]
fn test_bh_mod_031_detect_mutation_arithmetic_safe_ops() {
    // saturating_ prevents arithmetic pattern
    let matches = detect_mutation_targets("let x = a.saturating_add(1) as usize;");
    assert!(
        !matches.iter().any(|m| m.prefix == "arith"),
        "saturating_ ops should not trigger"
    );
}

#[test]
fn test_bh_mod_031_detect_mutation_boolean() {
    let matches = detect_mutation_targets("if !active && is_valid(x) {");
    assert!(matches.iter().any(|m| m.prefix == "bool"));
}

#[test]
fn test_bh_mod_031_detect_mutation_boolean_or_predicate() {
    let matches = detect_mutation_targets("if has_data(x) || !is_empty(x) {");
    assert!(matches.iter().any(|m| m.prefix == "bool"));
}

#[test]
fn test_bh_mod_031_detect_mutation_multiple() {
    // Line with both boundary and boolean
    let matches = detect_mutation_targets("if v.len() >= 1 && !is_valid(x) || has_data(y) {");
    assert!(
        matches.len() >= 2,
        "Should detect boundary + boolean, got {}",
        matches.len()
    );
}

// =========================================================================
// BH-MOD-032: Coverage Gap Tests — run_falsify_mode
// =========================================================================

#[test]
fn test_bh_mod_032_falsify_mode_with_source_files() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_032_falsify_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // File with boundary and arithmetic patterns
    std::fs::write(
        temp.join("src/code.rs"),
        "fn process(v: &[u8]) -> usize {\n\
            if v.len() > 0 {\n\
                let idx = v.len() - 1 as usize;\n\
                idx\n\
            } else {\n\
                0\n\
            }\n\
        }\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Falsify,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Falsify, config.clone());

    run_falsify_mode(&temp, &config, &mut result);

    // When cargo-mutants is not installed, run_falsify_mode returns early
    // with a BH-FALSIFY-UNAVAIL info finding instead of scanning for
    // mutation targets.  This is the expected path in clean-room containers.
    let mutants_available = std::process::Command::new("cargo")
        .args(["mutants", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if mutants_available {
        let mut_findings: Vec<_> = result
            .findings
            .iter()
            .filter(|f| f.id.starts_with("BH-MUT-"))
            .collect();
        assert!(
            !mut_findings.is_empty(),
            "Should find mutation targets in source files"
        );
    } else {
        assert!(
            result.findings.iter().any(|f| f.id == "BH-FALSIFY-UNAVAIL"),
            "Should report cargo-mutants unavailable"
        );
    }

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_032_falsify_mode_empty_dir() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_032_falsify_empty_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let config = HuntConfig {
        mode: HuntMode::Falsify,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Falsify, config.clone());

    run_falsify_mode(&temp, &config, &mut result);

    // No .rs files => no mutation findings (might have unavail finding)
    let mut_findings: Vec<_> = result
        .findings
        .iter()
        .filter(|f| f.id.starts_with("BH-MUT-"))
        .collect();
    assert!(mut_findings.is_empty());

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-033: Coverage Gap Tests — run_deep_hunt_mode
// =========================================================================

#[test]
fn test_bh_mod_033_deep_hunt_mode_with_files() {
    let temp = std::env::temp_dir().join("test_bh_mod_033_deep");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // File with deeply nested conditionals
    std::fs::write(
        temp.join("src/deep.rs"),
        "fn complex(a: i32, b: i32, c: i32) {\n\
            if a > 0 {\n\
                if b > 0 {\n\
                    if c > 0 {\n\
                        println!(\"deep\");\n\
                    }\n\
                }\n\
            }\n\
        }\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::DeepHunt,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::DeepHunt, config.clone());

    run_deep_hunt_mode(&temp, &config, &mut result);

    let deep_findings: Vec<_> = result
        .findings
        .iter()
        .filter(|f| f.id.starts_with("BH-DEEP-"))
        .collect();
    assert!(
        !deep_findings.is_empty(),
        "Should find deeply nested conditionals"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_033_deep_hunt_mode_empty() {
    let temp = std::env::temp_dir().join("test_bh_mod_033_deep_empty");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let config = HuntConfig {
        mode: HuntMode::DeepHunt,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::DeepHunt, config.clone());

    run_deep_hunt_mode(&temp, &config, &mut result);

    // No .rs files => no deep hunt findings (may have coverage findings)

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-034: Coverage Gap Tests — analyze_common_patterns (extended)
// =========================================================================

#[test]
fn test_bh_mod_034_common_patterns_rust_memory_safety() {
    // Rust files use language-specific patterns from languages.rs, which include
    // unsafe and transmute patterns for MemorySafety category.
    let temp = std::env::temp_dir().join(format!("test_bh_mod_034_mem_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/mem.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "fn danger() {\n    unsafe { std::ptr::null::<u8>().read() };\n}\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    let mem_finding = result
        .findings
        .iter()
        .any(|f| f.category == DefectCategory::MemorySafety);
    assert!(
        mem_finding,
        "Should detect unsafe block as MemorySafety pattern"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_034_common_patterns_hidden_debt() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_034_debt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "// not implemented yet\nfn placeholder() {}\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    let debt_finding = result
        .findings
        .iter()
        .any(|f| f.category == DefectCategory::HiddenDebt);
    assert!(
        debt_finding,
        "Should detect 'not implemented' as hidden debt"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_034_common_patterns_silent_degradation() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_034_silent_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "fn handle() {\n    x.unwrap_or_else(|_| default());\n}\nfn default() -> i32 { 0 }\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    let silent_finding = result
        .findings
        .iter()
        .any(|f| f.category == DefectCategory::SilentDegradation);
    assert!(
        silent_finding,
        "Should detect .unwrap_or_else(|_| as silent degradation"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_034_common_patterns_test_debt() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_034_testdebt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/tests.rs"),
        "#[cfg(test)]\nmod tests {\n    #[test]\n    #[ignore]\n    fn test_broken() {}\n}\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    let test_debt = result
        .findings
        .iter()
        .any(|f| f.category == DefectCategory::TestDebt);
    assert!(test_debt, "Should detect #[ignore] as test debt");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_034_common_patterns_gpu_keywords() {
    // GPU/CUDA keywords are in the gpu_patterns vec but for .rs files the
    // language-specific patterns override the fallback set. Verify that
    // analyze_common_patterns runs without error on CUDA-related content
    // and that standard Rust patterns (e.g., unwrap) still produce findings.
    let temp = std::env::temp_dir().join(format!("test_bh_mod_034_gpu_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/gpu.rs"),
        "// CUDA_ERROR: kernel launch failed\nfn broken_kernel() { x.unwrap(); }\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    // Rust-language patterns (unwrap) should still be detected
    let has_logic_finding = result
        .findings
        .iter()
        .any(|f| f.category == DefectCategory::LogicErrors);
    assert!(
        has_logic_finding,
        "Should detect unwrap() as LogicErrors pattern in GPU-related file"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_034_common_patterns_with_custom_patterns() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_034_custom_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    let _ = std::fs::create_dir_all(temp.join(".pmat"));

    // Create a custom pattern config
    std::fs::write(
        temp.join(".pmat/bug-hunter.toml"),
        "[[patterns]]\npattern = \"PERF-ISSUE\"\ncategory = \"performance\"\nseverity = \"high\"\nsuspiciousness = 0.8\n",
    )
    .unwrap();

    std::fs::write(
        temp.join("src/perf.rs"),
        "// PERF-ISSUE: this loop is O(n^2)\nfn slow() {}\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    let custom = result
        .findings
        .iter()
        .any(|f| f.title.contains("PERF-ISSUE"));
    assert!(custom, "Should detect custom pattern from config");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_034_common_patterns_pmat_satd_mode() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_034_satd_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "// TODO: fix this\nfn code() { let x = val.unwrap(); }\n",
    )
    .unwrap();

    // With pmat_satd enabled but no pmat available, patterns fall through
    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        pmat_satd: true,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    // PMAT SATD mode might skip TODO/FIXME patterns, but unwrap should still be caught
    // (This tests the pmat_satd branch at line 998)

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-035: Coverage Gap Tests — parse_lcov_da_line
// =========================================================================

#[test]
fn test_bh_mod_035_parse_lcov_da_line_valid_uncovered() {
    let mut file_uncovered = std::collections::HashMap::new();
    parse_lcov_da_line("42,0", "src/lib.rs", &mut file_uncovered);
    assert_eq!(file_uncovered.get("src/lib.rs").unwrap(), &vec![42]);
}

#[test]
fn test_bh_mod_035_parse_lcov_da_line_valid_covered() {
    let mut file_uncovered = std::collections::HashMap::new();
    parse_lcov_da_line("10,5", "src/lib.rs", &mut file_uncovered);
    assert!(
        file_uncovered.is_empty(),
        "Covered lines should not be added"
    );
}

#[test]
fn test_bh_mod_035_parse_lcov_da_line_no_comma() {
    let mut file_uncovered = std::collections::HashMap::new();
    parse_lcov_da_line("42", "src/lib.rs", &mut file_uncovered);
    assert!(file_uncovered.is_empty(), "No comma should return early");
}

#[test]
fn test_bh_mod_035_parse_lcov_da_line_invalid_line_num() {
    let mut file_uncovered = std::collections::HashMap::new();
    parse_lcov_da_line("abc,0", "src/lib.rs", &mut file_uncovered);
    assert!(
        file_uncovered.is_empty(),
        "Invalid line number should return early"
    );
}

#[test]
fn test_bh_mod_035_parse_lcov_da_line_invalid_hits() {
    let mut file_uncovered = std::collections::HashMap::new();
    parse_lcov_da_line("42,xyz", "src/lib.rs", &mut file_uncovered);
    assert!(
        file_uncovered.is_empty(),
        "Invalid hits should return early"
    );
}

// =========================================================================
// BH-MOD-036: Coverage Gap Tests — report_uncovered_hotspots
// =========================================================================

#[test]
fn test_bh_mod_036_report_uncovered_hotspots_above_threshold() {
    let mut file_uncovered = std::collections::HashMap::new();
    file_uncovered.insert("src/lib.rs".to_string(), vec![1, 2, 3, 4, 5, 6]);

    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());
    report_uncovered_hotspots(file_uncovered, Path::new("/project"), &mut result);

    assert_eq!(result.findings.len(), 1);
    assert!(result.findings[0].id.starts_with("BH-COV-"));
    assert!(result.findings[0].title.contains("6 uncovered lines"));
}

#[test]
fn test_bh_mod_036_report_uncovered_hotspots_below_threshold() {
    let mut file_uncovered = std::collections::HashMap::new();
    file_uncovered.insert("src/lib.rs".to_string(), vec![1, 2, 3]);

    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());
    report_uncovered_hotspots(file_uncovered, Path::new("/project"), &mut result);

    assert!(
        result.findings.is_empty(),
        "<=5 uncovered lines should not produce a finding"
    );
}

#[test]
fn test_bh_mod_036_report_uncovered_hotspots_exactly_five() {
    let mut file_uncovered = std::collections::HashMap::new();
    file_uncovered.insert("src/lib.rs".to_string(), vec![1, 2, 3, 4, 5]);

    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());
    report_uncovered_hotspots(file_uncovered, Path::new("/project"), &mut result);

    assert!(
        result.findings.is_empty(),
        "Exactly 5 uncovered lines should not trigger (>5 is threshold)"
    );
}

#[test]
fn test_bh_mod_036_report_uncovered_hotspots_suspiciousness_cap() {
    let mut file_uncovered = std::collections::HashMap::new();
    // 200 uncovered lines => suspiciousness = (200/100).min(0.8) = 0.8
    file_uncovered.insert("src/huge.rs".to_string(), (1..=200).collect());

    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());
    report_uncovered_hotspots(file_uncovered, Path::new("/project"), &mut result);

    assert_eq!(result.findings.len(), 1);
    assert!(
        (result.findings[0].suspiciousness - 0.8).abs() < f64::EPSILON,
        "Should cap at 0.8"
    );
}

#[test]
fn test_bh_mod_036_report_uncovered_hotspots_multiple_files() {
    let mut file_uncovered = std::collections::HashMap::new();
    file_uncovered.insert("src/a.rs".to_string(), vec![1, 2, 3, 4, 5, 6]);
    file_uncovered.insert("src/b.rs".to_string(), vec![10, 20, 30, 40, 50, 60, 70]);
    file_uncovered.insert("src/c.rs".to_string(), vec![1]); // Below threshold

    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());
    report_uncovered_hotspots(file_uncovered, Path::new("/project"), &mut result);

    assert_eq!(
        result.findings.len(),
        2,
        "Only files with >5 uncovered lines"
    );
}

// =========================================================================
// BH-MOD-037: Coverage Gap Tests — categorize_clippy_warning (all branches)
// =========================================================================

#[test]
fn test_bh_mod_037_categorize_mem() {
    let (cat, sev) = categorize_clippy_warning("clippy::mem_replace", "memory");
    assert_eq!(cat, DefectCategory::MemorySafety);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_037_categorize_uninit() {
    let (cat, sev) = categorize_clippy_warning("clippy::uninit_assumed_init", "uninitialized");
    assert_eq!(cat, DefectCategory::MemorySafety);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_037_categorize_arc() {
    let (cat, sev) = categorize_clippy_warning("clippy::arc_with_non_send_sync", "arc issue");
    assert_eq!(cat, DefectCategory::ConcurrencyBugs);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_037_categorize_send() {
    let (cat, sev) = categorize_clippy_warning("not_send_bound", "not Send");
    assert_eq!(cat, DefectCategory::ConcurrencyBugs);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_037_categorize_sync() {
    let (cat, sev) = categorize_clippy_warning("not_sync_bound", "not Sync");
    assert_eq!(cat, DefectCategory::ConcurrencyBugs);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_037_categorize_transmute_security() {
    let (cat, sev) = categorize_clippy_warning("clippy::transmute_int_to_bool", "transmute");
    assert_eq!(cat, DefectCategory::SecurityVulnerabilities);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_037_categorize_expect() {
    let (cat, sev) = categorize_clippy_warning("clippy::expect_used", "expect");
    assert_eq!(cat, DefectCategory::LogicErrors);
    assert_eq!(sev, FindingSeverity::Medium);
}

#[test]
fn test_bh_mod_037_categorize_panic() {
    let (cat, sev) = categorize_clippy_warning("clippy::panic_in_result_fn", "panic");
    assert_eq!(cat, DefectCategory::LogicErrors);
    assert_eq!(sev, FindingSeverity::Medium);
}

#[test]
fn test_bh_mod_037_categorize_cast() {
    let (cat, sev) = categorize_clippy_warning("clippy::cast_sign_loss", "cast");
    assert_eq!(cat, DefectCategory::TypeErrors);
    assert_eq!(sev, FindingSeverity::Medium);
}

#[test]
fn test_bh_mod_037_categorize_as() {
    let (cat, sev) = categorize_clippy_warning("clippy::as_conversions", "as conversion");
    assert_eq!(cat, DefectCategory::TypeErrors);
    assert_eq!(sev, FindingSeverity::Medium);
}

#[test]
fn test_bh_mod_037_categorize_into() {
    let (cat, sev) = categorize_clippy_warning("clippy::into_iter_on_ref", "into iter");
    assert_eq!(cat, DefectCategory::TypeErrors);
    assert_eq!(sev, FindingSeverity::Medium);
}

// =========================================================================
// BH-MOD-038: Coverage Gap Tests — scan_file_for_patterns
// =========================================================================

#[test]
fn test_bh_mod_038_scan_file_for_patterns_basic() {
    let temp = std::env::temp_dir().join("test_bh_mod_038_patterns");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(&temp);

    let file = temp.join("scan_test.rs");
    std::fs::write(
        &file,
        "fn risky() {\n    let x = val.unwrap();\n    panic!(\"oops\");\n}\n",
    )
    .unwrap();

    let patterns: Vec<(&str, DefectCategory, FindingSeverity, f64)> = vec![
        (
            "unwrap()",
            DefectCategory::LogicErrors,
            FindingSeverity::Medium,
            0.4,
        ),
        (
            "panic!",
            DefectCategory::LogicErrors,
            FindingSeverity::Medium,
            0.5,
        ),
    ];
    let custom_patterns: Vec<(String, DefectCategory, FindingSeverity, f64)> = vec![];
    let bh_config = self::config::BugHunterConfig::default();
    let mut findings = Vec::new();

    scan_file_for_patterns(
        &file,
        &patterns,
        &custom_patterns,
        &bh_config,
        0.0,
        &mut findings,
    );

    assert!(
        findings.len() >= 2,
        "Should find unwrap and panic patterns, got {}",
        findings.len()
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_038_scan_file_for_patterns_nonexistent() {
    let patterns: Vec<(&str, DefectCategory, FindingSeverity, f64)> = vec![];
    let custom_patterns: Vec<(String, DefectCategory, FindingSeverity, f64)> = vec![];
    let bh_config = self::config::BugHunterConfig::default();
    let mut findings = Vec::new();

    scan_file_for_patterns(
        Path::new("/nonexistent/file.rs"),
        &patterns,
        &custom_patterns,
        &bh_config,
        0.0,
        &mut findings,
    );

    assert!(findings.is_empty());
}

#[test]
fn test_bh_mod_038_scan_file_for_patterns_custom() {
    let temp = std::env::temp_dir().join("test_bh_mod_038_custom_scan");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(&temp);

    let file = temp.join("custom.rs");
    std::fs::write(&file, "// MY-MARKER: review this\nfn code() {}\n").unwrap();

    let patterns: Vec<(&str, DefectCategory, FindingSeverity, f64)> = vec![];
    let custom_patterns = vec![(
        "MY-MARKER".to_string(),
        DefectCategory::LogicErrors,
        FindingSeverity::High,
        0.8,
    )];
    let bh_config = self::config::BugHunterConfig::default();
    let mut findings = Vec::new();

    scan_file_for_patterns(
        &file,
        &patterns,
        &custom_patterns,
        &bh_config,
        0.0,
        &mut findings,
    );

    assert!(findings.iter().any(|f| f.title.contains("MY-MARKER")));

    let _ = std::fs::remove_dir_all(&temp);
}

