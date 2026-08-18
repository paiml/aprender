
// =========================================================================
// Shared fixture for hunt()-level tests
// =========================================================================

/// Build a throwaway project fixture for `hunt` / `hunt_ensemble` tests.
///
/// These tests used to run against `Path::new(".")` — the real crate — so each
/// one shelled out to `cargo clippy --all-targets`, `pmat query` and `git blame`
/// over the whole source tree (measured 40s-172s apiece). The fixture is
/// deliberately *not* a cargo package: with no `Cargo.toml`, analyze mode's
/// `cargo clippy` exits immediately instead of compiling a crate, so nothing
/// here needs nextest's `serial-build` group.
///
/// The planted source carries one deterministic trigger per hunt mode:
/// `unwrap()` for Analyze, a `len()` comparison plus a cast-arithmetic line for
/// Falsify, three nested `if`s for DeepHunt. `lcov.info` gives Hunt mode
/// coverage to chew on; the absent `fuzz/` dir makes Fuzz mode report missing
/// fuzz targets.
fn hunt_fixture(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("test_bh_fixture_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).expect("create fixture src dir");
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub fn probe(v: &[u8]) -> usize {
    let first = v.first().copied().unwrap();
    if v.len() > 0 {
        let idx = v.len() - 1 as usize;
        if idx > 2 {
            if idx > first as usize {
                return idx;
            }
        }
    }
    0
}
",
    )
    .expect("write fixture source");
    // Six uncovered lines in one file clears report_uncovered_hotspots' threshold.
    std::fs::write(
        dir.join("lcov.info"),
        "SF:src/lib.rs\nDA:1,0\nDA:2,0\nDA:3,0\nDA:4,0\nDA:5,0\nDA:6,0\nend_of_record\n",
    )
    .expect("write fixture lcov");
    dir
}

/// Hermetic hunt config for `hunt_fixture`: scan only `src`, keep every finding,
/// and leave the pmat SATD subprocess out of it. BH-23's pmat integration is
/// covered by BH-MOD-053; shelling out to pmat here bought no coverage and cost
/// tens of seconds per test.
fn hunt_fixture_config(mode: HuntMode) -> HuntConfig {
    HuntConfig {
        mode,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        pmat_satd: false,
        ..Default::default()
    }
}

// =========================================================================
// BH-MOD-001: Hunt Function
// =========================================================================

#[test]
fn test_bh_mod_001_hunt_returns_result() {
    let fixture = hunt_fixture("mod_001_returns");

    let result = hunt(&fixture, hunt_fixture_config(HuntMode::Analyze));

    assert_eq!(result.mode, HuntMode::Analyze);
    // Analyze mode must reach the pattern scan, not merely echo the mode back.
    assert!(
        result.findings.iter().any(|f| f.title.contains("unwrap()")),
        "Analyze mode missed the planted unwrap(): {:?}",
        result.findings.iter().map(|f| &f.title).collect::<Vec<_>>()
    );
    assert_eq!(
        result.stats.total_findings,
        result.findings.len(),
        "finalize() must count every finding"
    );

    let _ = std::fs::remove_dir_all(&fixture);
}

#[test]
fn test_bh_mod_001_hunt_all_modes() {
    let fixture = hunt_fixture("mod_001_all_modes");

    for mode in [
        HuntMode::Falsify,
        HuntMode::Hunt,
        HuntMode::Analyze,
        HuntMode::Fuzz,
        HuntMode::DeepHunt,
    ] {
        let result = hunt(&fixture, hunt_fixture_config(mode));
        assert_eq!(result.mode, mode);
        // Dispatch must land in the mode's own handler. On this fixture every
        // mode tags at least one finding with itself: Falsify emits mutation
        // targets (or the cargo-mutants-unavailable notice), Hunt the lcov
        // hotspot, Analyze the unwrap() pattern, Fuzz the missing fuzz/ dir,
        // DeepHunt the nested conditionals.
        assert!(
            result.findings.iter().any(|f| f.discovered_by == mode),
            "{} mode produced no finding of its own: {:?}",
            mode,
            result.findings.iter().map(|f| (&f.id, f.discovered_by)).collect::<Vec<_>>()
        );
    }

    let _ = std::fs::remove_dir_all(&fixture);
}

// =========================================================================
// BH-MOD-002: Ensemble Hunt
// =========================================================================

#[test]
fn test_bh_mod_002_hunt_ensemble() {
    let fixture = hunt_fixture("mod_002_ensemble");

    let result = hunt_ensemble(&fixture, hunt_fixture_config(HuntMode::Analyze));

    // The ensemble runs Analyze + Hunt + Falsify and merges the three result
    // sets, so all three must be represented. (The previous assertion was
    // `duration_ms > 0` — a wall-clock check that could not fail.)
    for mode in [HuntMode::Analyze, HuntMode::Hunt, HuntMode::Falsify] {
        assert!(
            result.findings.iter().any(|f| f.discovered_by == mode),
            "ensemble dropped every {} finding: {:?}",
            mode,
            result.findings.iter().map(|f| (&f.id, f.discovered_by)).collect::<Vec<_>>()
        );
    }

    let _ = std::fs::remove_dir_all(&fixture);
}

// =========================================================================
// BH-MOD-003: Category Classification
// =========================================================================

#[test]
fn test_bh_mod_003_categorize_clippy_memory() {
    let (cat, sev) = categorize_clippy_warning("ptr_null", "test");
    assert_eq!(cat, DefectCategory::MemorySafety);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_003_categorize_clippy_concurrency() {
    let (cat, sev) = categorize_clippy_warning("mutex_atomic", "test");
    assert_eq!(cat, DefectCategory::ConcurrencyBugs);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_003_categorize_clippy_unknown() {
    let (cat, sev) = categorize_clippy_warning("some_other_lint", "test");
    assert_eq!(cat, DefectCategory::Unknown);
    assert_eq!(sev, FindingSeverity::Low);
}

// =========================================================================
// BH-MOD-004: Result Finalization
// =========================================================================

#[test]
fn test_bh_mod_004_result_finalize() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new(".", HuntMode::Analyze, config);
    result.add_finding(
        Finding::new("F-001", "test.rs", 1, "Test")
            .with_severity(FindingSeverity::High)
            .with_suspiciousness(0.8),
    );
    result.finalize();

    assert_eq!(result.stats.total_findings, 1);
    assert_eq!(
        result.stats.by_severity.get(&FindingSeverity::High),
        Some(&1)
    );
}

// =========================================================================
// BH-MOD-005: Test Code Detection
// =========================================================================

#[test]
fn test_bh_mod_005_compute_test_lines_cfg_test() {
    let content = r#"fn production_code() {
    panic!("this should be caught");
}

#[cfg(test)]
mod tests {
    #[test]
    fn my_test() {
        panic!("this should be ignored");
    }
}"#;
    let test_lines = compute_test_lines(content);
    assert!(
        test_lines.contains(&5),
        "Line 5 (#[cfg(test)]) should be test code"
    );
    assert!(
        test_lines.contains(&6),
        "Line 6 (mod tests) should be test code"
    );
    assert!(
        test_lines.contains(&7),
        "Line 7 (#[test]) should be test code"
    );
    assert!(
        test_lines.contains(&9),
        "Line 9 (panic) should be test code"
    );
    assert!(
        !test_lines.contains(&2),
        "Line 2 (production panic) should NOT be test code"
    );
}

#[test]
fn test_bh_mod_005_compute_test_lines_individual_test() {
    let content = r#"fn production_code() {
    println!("production");
}

#[test]
fn standalone_test() {
    panic!("test assertion");
}"#;
    let test_lines = compute_test_lines(content);
    assert!(
        test_lines.contains(&5),
        "Line 5 (#[test]) should be test code"
    );
    assert!(
        test_lines.contains(&6),
        "Line 6 (fn standalone_test) should be test code"
    );
    assert!(
        test_lines.contains(&7),
        "Line 7 (panic) should be test code"
    );
    assert!(
        !test_lines.contains(&1),
        "Line 1 (fn production_code) should NOT be test code"
    );
    assert!(
        !test_lines.contains(&2),
        "Line 2 (println) should NOT be test code"
    );
}

#[test]
fn test_bh_mod_005_compute_test_lines_empty() {
    let content = "fn main() {}\n";
    let test_lines = compute_test_lines(content);
    assert!(test_lines.is_empty());
}

// =========================================================================
// BH-MOD-006: Real Pattern Detection
// =========================================================================

#[test]
fn test_bh_mod_006_real_pattern_todo_in_comment() {
    assert!(is_real_pattern("// TODO: fix this", "TODO"));
    assert!(is_real_pattern("    // TODO fix later", "TODO"));
}

#[test]
fn test_bh_mod_006_real_pattern_todo_in_string() {
    assert!(!is_real_pattern(r#"let msg = "TODO: implement";"#, "TODO"));
    assert!(!is_real_pattern(
        r#"println!("TODO/FIXME markers");"#,
        "TODO"
    ));
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_006_real_pattern_unsafe_in_code() {
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    assert!(is_real_pattern("unsafe { ptr::read(p) }", "unsafe {"));
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_006_real_pattern_unsafe_in_comment() {
    assert!(!is_real_pattern(
        "// unsafe blocks need safety comments",
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "unsafe {"
    ));
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_006_real_pattern_unsafe_in_variable() {
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    assert!(!is_real_pattern("if in_unsafe {", "unsafe {"));
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    assert!(!is_real_pattern("let foo_unsafe = true;", "unsafe {"));
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    assert!(is_real_pattern("    unsafe { foo() }", "unsafe {"));
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    assert!(is_real_pattern("return unsafe { bar };", "unsafe {"));
}

#[test]
fn test_bh_mod_006_real_pattern_unwrap_in_code() {
    assert!(is_real_pattern("let x = opt.unwrap();", "unwrap()"));
}

#[test]
fn test_bh_mod_006_real_pattern_unwrap_in_doc() {
    assert!(!is_real_pattern(
        "/// Use unwrap() for testing only",
        "unwrap()"
    ));
}

// =========================================================================
// BH-MOD-007: GH-18 lcov.info path detection tests
// =========================================================================

#[test]
fn test_bh_mod_007_coverage_path_config() {
    let mut config = HuntConfig::default();
    assert!(config.coverage_path.is_none());

    config.coverage_path = Some(std::path::PathBuf::from("/custom/lcov.info"));
    assert_eq!(
        config.coverage_path.as_ref().unwrap().to_str().unwrap(),
        "/custom/lcov.info"
    );
}

#[test]
fn test_bh_mod_007_analyze_coverage_hotspots_no_file() {
    let temp = std::env::temp_dir().join("test_bh_mod_007");
    let _ = std::fs::create_dir_all(&temp);
    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    analyze_coverage_hotspots(&temp, &config, &mut result);

    let nocov = result.findings.iter().any(|f| f.id == "BH-HUNT-NOCOV");
    assert!(
        nocov,
        "Should report BH-HUNT-NOCOV when no coverage file exists"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-008: GH-19 forbid(unsafe_code) detection tests
// =========================================================================

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_008_crate_forbids_unsafe_lib_rs() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_lib");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "#![forbid(unsafe_code)]\n\npub fn safe_fn() {}\n",
    )
    .unwrap();

    assert!(
        crate_forbids_unsafe(&temp),
        "Should detect #![forbid(unsafe_code)] in lib.rs"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_008_crate_forbids_unsafe_main_rs() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_main");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/main.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "#![forbid(unsafe_code)]\n\nfn main() {}\n",
    )
    .unwrap();

    assert!(
        crate_forbids_unsafe(&temp),
        "Should detect #![forbid(unsafe_code)] in main.rs"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_008_crate_forbids_unsafe_cargo_toml() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_cargo");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(temp.join("src/lib.rs"), "pub fn safe_fn() {}\n").unwrap();

    std::fs::write(
        temp.join("Cargo.toml"),
        r#"[package]
name = "test"
version = "0.1.0"

[lints.rust]
unsafe_code = "forbid"
"#,
    )
    .unwrap();

    assert!(
        crate_forbids_unsafe(&temp),
        "Should detect unsafe_code = \"forbid\" in Cargo.toml"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_008_crate_allows_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_allows");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "pub fn maybe_unsafe() { /* could have unsafe later */ }\n",
    )
    .unwrap();

    assert!(
        !crate_forbids_unsafe(&temp),
        "Should return false when unsafe_code is not forbidden"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_008_fuzz_mode_skips_forbid_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_fuzz");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
        "#![forbid(unsafe_code)]\n\npub fn safe_fn() {}\n",
    )
    .unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());

    run_fuzz_mode(&temp, &config, &mut result);

    let skipped = result.findings.iter().any(|f| f.id == "BH-FUZZ-SKIPPED");
    let notargets = result.findings.iter().any(|f| f.id == "BH-FUZZ-NOTARGETS");

    assert!(
        skipped,
        "Should report BH-FUZZ-SKIPPED for forbid(unsafe_code) crates"
    );
    assert!(
        !notargets,
        "Should NOT report BH-FUZZ-NOTARGETS for forbid(unsafe_code) crates"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

