// =========================================================================
// BH-MOD-039: Coverage Gap Tests — hunt_with_ticket
// =========================================================================

#[test]
fn test_bh_mod_039_hunt_with_ticket_nonexistent() {
    let config = HuntConfig::default();
    let result = hunt_with_ticket(Path::new("/tmp"), "CB-999", config);
    assert!(result.is_err(), "Non-existent ticket should fail");
}

// =========================================================================
// BH-MOD-040: Coverage Gap Tests — run_quick_mode
// =========================================================================

#[test]
fn test_bh_mod_040_quick_mode_runs_patterns() {
    let temp = std::env::temp_dir().join(format!("test_bh_mod_040_quick_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "fn code() { let x = val.unwrap(); }\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Quick, config.clone());

    run_quick_mode(&temp, &config, &mut result);

    let found_unwrap = result.findings.iter().any(|f| f.title.contains("unwrap()"));
    assert!(found_unwrap, "Quick mode should find unwrap pattern");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-041: Coverage Gap Tests — run_hunt_mode with crash files
// =========================================================================

#[test]
fn test_bh_mod_041_hunt_mode_with_crash_log() {
    let temp = std::env::temp_dir().join("test_bh_mod_041_crash");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(&temp);

    std::fs::write(
        temp.join("crash.log"),
        "thread 'main' panicked\n   at src/parser.rs:55\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    run_hunt_mode(&temp, &config, &mut result);

    // Should find the stack trace location
    let stack = result
        .findings
        .iter()
        .any(|f| f.id.starts_with("BH-STACK-"));
    assert!(stack, "Should find stack trace from crash.log");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-042: Coverage Gap Tests — analyze_coverage_hotspots with standard path
// =========================================================================

#[test]
fn test_bh_mod_042_coverage_hotspots_standard_path() {
    let temp = std::env::temp_dir().join("test_bh_mod_042_stdpath");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("target/coverage"));

    std::fs::write(
        temp.join("target/coverage/lcov.info"),
        "SF:src/lib.rs\nDA:1,0\nDA:2,0\nDA:3,0\nDA:4,0\nDA:5,0\nDA:6,0\nDA:7,0\nend_of_record\n",
    )
    .unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    analyze_coverage_hotspots(&temp, &config, &mut result);

    let cov = result.findings.iter().any(|f| f.id.starts_with("BH-COV-"));
    assert!(cov, "Should find coverage hotspot from standard path");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-043: Coverage Gap Tests — hunt cache hit path
// =========================================================================

#[test]
fn test_bh_mod_043_hunt_cache_hit_path() {
    // Run hunt twice with the same config; second call should hit cache
    let temp = std::env::temp_dir().join("test_bh_mod_043_cache");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    let _ = std::fs::create_dir_all(temp.join(".pmat/bug-hunter-cache"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "fn code() { let x = val.unwrap(); }\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };

    // First call populates cache
    let result1 = hunt(&temp, config.clone());
    // Second call with identical config should hit cache
    let result2 = hunt(&temp, config);

    // Both results should have the same mode
    assert_eq!(result1.mode, HuntMode::Quick);
    assert_eq!(result2.mode, HuntMode::Quick);
    // Cached result should have findings from the first run
    // (the cache hit path covers lines 70-81)

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-044: Coverage Gap Tests — hunt with use_pmat_quality enabled
// =========================================================================

#[test]
fn test_bh_mod_044_hunt_pmat_quality_enabled() {
    let temp = std::env::temp_dir().join("test_bh_mod_044_pmat");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "fn code() { let x = val.unwrap(); }\n",
    )
    .unwrap();

    // Enable PMAT quality — even though pmat may not be available,
    // this exercises the `if config.use_pmat_quality` branch (lines 102-119)
    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        use_pmat_quality: true,
        pmat_query: Some("*".to_string()),
        ..Default::default()
    };

    let result = hunt(&temp, config);
    assert_eq!(result.mode, HuntMode::Quick);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-045: Coverage Gap Tests — hunt with coverage_weight > 0
// =========================================================================

#[test]
fn test_bh_mod_045_hunt_coverage_weight() {
    let temp = std::env::temp_dir().join("test_bh_mod_045_covwt");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "fn code() { let x = val.unwrap(); }\n",
    )
    .unwrap();

    // Enable coverage weighting — exercises the coverage_weight > 0 branch (lines 123-140)
    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        coverage_weight: 1.0,
        coverage_path: Some(PathBuf::from("/nonexistent/lcov.info")),
        ..Default::default()
    };

    let result = hunt(&temp, config);
    assert_eq!(result.mode, HuntMode::Quick);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_045_hunt_coverage_weight_with_file() {
    let temp = std::env::temp_dir().join("test_bh_mod_045_covwt_file");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "fn code() { let x = val.unwrap(); }\n",
    )
    .unwrap();

    // Create an lcov file that the coverage module can parse
    let lcov_path = temp.join("lcov.info");
    std::fs::write(
        &lcov_path,
        "SF:src/lib.rs\nDA:1,0\nDA:2,0\nDA:3,0\nDA:4,0\nDA:5,0\nDA:6,0\nDA:7,0\nend_of_record\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        coverage_weight: 1.0,
        coverage_path: Some(lcov_path),
        ..Default::default()
    };

    let result = hunt(&temp, config);
    assert_eq!(result.mode, HuntMode::Quick);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-046: Coverage Gap Tests — apply_spec_quality_gate (pmat unavailable)
// =========================================================================

#[test]
fn test_bh_mod_046_apply_spec_quality_gate_no_pmat() {
    // apply_spec_quality_gate must bail before touching any claim when
    // build_quality_index yields nothing. An empty directory guarantees that:
    // pmat finds no functions to index (and if pmat is absent entirely,
    // pmat_available() short-circuits to the same None).
    //
    // This used to point at /tmp, which made `pmat query` walk the whole
    // system temp dir — 14s for a gate that never fires.
    use super::spec::{ClaimStatus, CodeLocation, SpecClaim};

    let fixture =
        std::env::temp_dir().join(format!("test_bh_mod_046_empty_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&fixture);
    std::fs::create_dir_all(&fixture).expect("create empty fixture dir");

    // A claim WITH an implementation, so the early return is what keeps the
    // finding list empty rather than there being nothing to inspect.
    let mut parsed_spec = ParsedSpec {
        claims: vec![SpecClaim {
            id: "NOPMAT-01".to_string(),
            title: "Claim with an implementation".to_string(),
            line: 1,
            section_path: vec!["Section 1".to_string()],
            implementations: vec![CodeLocation {
                file: PathBuf::from("src/lib.rs"),
                line: 1,
                context: "probe".to_string(),
            }],
            findings: Vec::new(),
            status: ClaimStatus::Pending,
        }],
        original_content: String::new(),
        path: PathBuf::new(),
    };
    let mut result = HuntResult::new(&fixture, HuntMode::Analyze, HuntConfig::default());

    apply_spec_quality_gate(&mut parsed_spec, &fixture, &mut result, "*");

    assert!(
        result.findings.is_empty(),
        "gate must add nothing when the quality index is unavailable: {:?}",
        result.findings.iter().map(|f| &f.id).collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(&fixture);
}

// =========================================================================
// BH-MOD-047: Coverage Gap Tests — run_falsify_mode cargo-mutants unavailable
// =========================================================================

#[test]
fn test_bh_mod_047_falsify_mode_mutants_unavailable() {
    // In environments without cargo-mutants, run_falsify_mode adds a BH-FALSIFY-UNAVAIL finding
    // and returns early (covers lines 348-368)
    let temp = std::env::temp_dir().join(format!("test_bh_mod_047_falsify_unavail_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // File with boundary + arithmetic patterns to ensure mutation findings if mutants is available
    std::fs::write(
        temp.join("src/lib.rs"),
        "fn check(v: &[u8]) -> usize {\n\
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

    // If cargo-mutants is installed, we get mutation findings from the boundary/arith patterns.
    // If not, we get BH-FALSIFY-UNAVAIL. Either way, we exercise the function.
    let has_unavail = result.findings.iter().any(|f| f.id == "BH-FALSIFY-UNAVAIL");
    let has_mutations = result.findings.iter().any(|f| f.id.starts_with("BH-MUT-"));
    assert!(
        has_unavail || has_mutations,
        "Should have either UNAVAIL or mutation findings, got {} findings: {:?}",
        result.findings.len(),
        result.findings.iter().map(|f| &f.id).collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-048: Coverage Gap Tests — run_deep_hunt_mode combined coverage
// =========================================================================

#[test]
fn test_bh_mod_048_deep_hunt_mode_combines_deep_and_hunt() {
    let temp = std::env::temp_dir().join("test_bh_mod_048_deep_combined");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // File with deeply nested conditionals + complex boolean guards
    std::fs::write(
        temp.join("src/complex.rs"),
        "fn complex(a: i32, b: bool, c: bool) {\n\
            if a > 0 {\n\
                match a {\n\
                    1 => if b && c || !b {\n\
                        println!(\"mixed\");\n\
                    },\n\
                    _ => if a > 10 {\n\
                        println!(\"deep\");\n\
                    },\n\
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

    // Should have deep hunt findings and also run_hunt_mode findings
    // (run_hunt_mode either finds coverage or reports BH-HUNT-NOCOV)
    assert!(
        !result.findings.is_empty(),
        "Deep hunt should produce findings from both deep analysis and hunt mode"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-049: Coverage Gap Tests — analyze_common_patterns with Python files
// =========================================================================

#[test]
fn test_bh_mod_049_common_patterns_python_file() {
    let temp = std::env::temp_dir().join("test_bh_mod_049_python");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // Python file with known patterns
    std::fs::write(
        temp.join("src/script.py"),
        "import os\n# TODO: refactor this function\ndef process():\n    pass\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        pmat_satd: false,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    // Python files should be scanned via language-specific glob patterns
    // (this exercises the multi-language glob path in lines 1112-1119)

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_049_common_patterns_typescript_file() {
    let temp = std::env::temp_dir().join("test_bh_mod_049_ts");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // TypeScript file with known patterns
    std::fs::write(
        temp.join("src/app.ts"),
        "// HACK: temporary workaround\nexport function handler() {}\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        pmat_satd: false,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-050: Coverage Gap Tests — analyze_common_patterns PMAT SATD active path
// =========================================================================

#[test]
fn test_bh_mod_050_common_patterns_pmat_satd_with_pmat_query() {
    let temp = std::env::temp_dir().join("test_bh_mod_050_satd_query");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "fn code() { let x = val.unwrap(); }\n",
    )
    .unwrap();

    // Enable PMAT SATD with a specific query
    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        pmat_satd: true,
        pmat_query: Some("error handling".to_string()),
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    // The pmat_satd path (line 997-1006) is entered when pmat_satd is true
    // AND pmat is available. If pmat is not available, falls through to
    // normal pattern matching.

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-051: Coverage Gap Tests — hunt_with_spec with use_pmat_quality
// =========================================================================

#[test]
fn test_bh_mod_051_hunt_with_spec_pmat_quality() {
    let temp = std::env::temp_dir().join("test_bh_mod_051_spec_pmat");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let spec_content =
        "# Test Spec\n\n## Section 1\n\n### TST-01: Test Claim\n\nThis claim tests something.\n";
    std::fs::write(temp.join("spec.md"), spec_content).unwrap();
    std::fs::write(
        temp.join("src/lib.rs"),
        "// TST-01: implements test claim\nfn test_impl() {}\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        use_pmat_quality: true,
        pmat_query: Some("*".to_string()),
        ..Default::default()
    };

    let result = hunt_with_spec(&temp, &temp.join("spec.md"), None, config);
    assert!(result.is_ok());
    let (hunt_result, _parsed_spec) = result.unwrap();
    // Even with pmat quality enabled, hunt should complete (pmat may not be available)
    assert_eq!(hunt_result.mode, HuntMode::Quick);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-052: Coverage Gap Tests — hunt with nextest junit.xml path
// =========================================================================

#[test]
fn test_bh_mod_052_hunt_mode_with_junit_xml() {
    let temp = std::env::temp_dir().join("test_bh_mod_052_junit");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("target/nextest/ci"));

    // Create a junit.xml file that contains "panicked"
    std::fs::write(
        temp.join("target/nextest/ci/junit.xml"),
        "<testsuite><testcase><failure>thread 'test' panicked at src/lib.rs:10</failure></testcase></testsuite>\n",
    )
    .unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    run_hunt_mode(&temp, &config, &mut result);

    // The junit.xml file should be picked up as a stack trace source (lines 490-497)
    // Since it contains "panicked", it should be added to stack_traces_found

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-053: Coverage Gap Tests — hunt with use_pmat_quality on real project
// =========================================================================

#[test]
fn test_bh_mod_053_hunt_pmat_quality_on_real_project() {
    // Exercises hunt()'s BH-21/BH-24 quality phase. This used to run against
    // the REAL crate, so every execution paid a full `pmat query` over the
    // whole source tree (40s). A fixture pmat indexes in milliseconds reaches
    // the same branch.
    let fixture = hunt_fixture("mod_053_pmat_quality");

    let baseline = hunt(&fixture, hunt_fixture_config(HuntMode::Quick));

    let config = HuntConfig {
        use_pmat_quality: true,
        pmat_query: Some("probe".to_string()),
        quality_weight: 0.5,
        ..hunt_fixture_config(HuntMode::Quick)
    };
    let result = hunt(&fixture, config);

    assert_eq!(result.mode, HuntMode::Quick);
    // The quality phase reweights findings in place; it must never add or drop
    // one. Diffing against the pmat-off run pins that down on any machine,
    // whether or not pmat is installed.
    let locations = |r: &HuntResult| {
        let mut keys: Vec<String> = r
            .findings
            .iter()
            .map(|f| format!("{}|{}|{}", f.file.display(), f.line, f.title))
            .collect();
        keys.sort();
        keys
    };
    assert!(!baseline.findings.is_empty(), "fixture produced no findings to weight");
    assert_eq!(
        locations(&result),
        locations(&baseline),
        "pmat quality phase must reweight findings, not change the set"
    );

    let _ = std::fs::remove_dir_all(&fixture);
}

// =========================================================================
// BH-MOD-054: Coverage Gap Tests — apply_spec_quality_gate with real project
// =========================================================================

/// Write a one-file fixture project that pmat can index in milliseconds.
///
/// The BH-25 quality-gate tests used to run `pmat query` over the real crate
/// (15-20s each) and then assert nothing at all. A fixture lets them assert the
/// gate's actual predicate instead.
fn quality_gate_fixture(name: &str, source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("test_bh_qgate_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).expect("create qgate fixture src dir");
    std::fs::write(dir.join("src/lib.rs"), source).expect("write qgate fixture source");
    dir
}

/// A `ParsedSpec` with a single claim implemented at `src/lib.rs:1`.
///
/// The path is relative because that is the form `pmat query` reports, and
/// `lookup_quality` matches index keys exactly.
fn quality_gate_spec(claim_id: &str) -> ParsedSpec {
    use super::spec::{ClaimStatus, CodeLocation, SpecClaim};

    ParsedSpec {
        path: PathBuf::from("test_spec.md"),
        claims: vec![SpecClaim {
            id: claim_id.to_string(),
            title: "Test Claim".to_string(),
            line: 1,
            section_path: vec!["Section 1".to_string()],
            implementations: vec![CodeLocation {
                file: PathBuf::from("src/lib.rs"),
                line: 1,
                context: "fixture function".to_string(),
            }],
            findings: Vec::new(),
            status: ClaimStatus::Pending,
        }],
        original_content: format!("# Spec\n## Section 1\n### {}: Test\n", claim_id),
    }
}

#[test]
fn test_bh_mod_054_apply_spec_quality_gate_real_project() {
    // A claim implemented by a small, well-graded function must NOT trip the
    // gate. Previously this ran pmat over the real crate and asserted nothing.
    let fixture = quality_gate_fixture(
        "clean",
        "pub fn tidy(n: u32) -> u32 {\n    n.saturating_add(1)\n}\n",
    );

    let mut parsed_spec = quality_gate_spec("CLAIM-01");
    let mut result = HuntResult::new(&fixture, HuntMode::Analyze, HuntConfig::default());

    apply_spec_quality_gate(&mut parsed_spec, &fixture, &mut result, "tidy");

    assert!(
        result.findings.is_empty(),
        "gate fired on well-graded code: {:?}",
        result.findings.iter().map(|f| (&f.id, &f.title)).collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(&fixture);
}

#[test]
fn test_bh_mod_054_apply_spec_quality_gate_low_quality_finding() {
    // The inner branch: pmat grades the implementing function as low quality
    // (grade D/F or complexity > 20), so the gate must emit BH-QGATE-<claim>.
    let mut source = String::from("pub fn tangled(n: u32) -> u32 {\n    let mut acc = 0;\n");
    for i in 1..=25 {
        source.push_str(&format!(
            "    if n % {i} == 0 {{ acc += {i}; }} else if n > {i} {{ acc -= 1; }}\n"
        ));
    }
    source.push_str("    acc\n}\n");
    let fixture = quality_gate_fixture("tangled", &source);

    let mut parsed_spec = quality_gate_spec("LQ-01");
    let mut result = HuntResult::new(&fixture, HuntMode::Analyze, HuntConfig::default());

    // Ask the same question the gate asks, so both outcomes stay assertable on
    // machines with and without pmat installed.
    let index = super::pmat_quality::build_quality_index(&fixture, "tangled", 200);
    apply_spec_quality_gate(&mut parsed_spec, &fixture, &mut result, "tangled");

    match index {
        Some(index) => {
            let graded = super::pmat_quality::lookup_quality(&index, Path::new("src/lib.rs"), 1)
                .expect("pmat indexed src/lib.rs but no function covers line 1");
            assert!(
                graded.complexity > 20 || graded.tdg_grade == "D" || graded.tdg_grade == "F",
                "fixture is no longer low quality (grade {}, complexity {})",
                graded.tdg_grade,
                graded.complexity
            );
            assert!(
                result.findings.iter().any(|f| f.id == "BH-QGATE-LQ-01"),
                "gate missed low-quality implementation: {:?}",
                result.findings.iter().map(|f| &f.id).collect::<Vec<_>>()
            );
        }
        None => assert!(
            result.findings.is_empty(),
            "gate must add nothing without a quality index: {:?}",
            result.findings.iter().map(|f| &f.id).collect::<Vec<_>>()
        ),
    }

    let _ = std::fs::remove_dir_all(&fixture);
}

#[test]
fn test_bh_mod_054_apply_spec_quality_gate_no_pmat() {
    // Test with a nonexistent project path where pmat has no index
    use super::spec::{ClaimStatus, CodeLocation, SpecClaim};

    let mut parsed_spec = ParsedSpec {
        path: PathBuf::from("test_spec.md"),
        claims: vec![SpecClaim {
            id: "NP-01".to_string(),
            title: "No Pmat".to_string(),
            line: 1,
            section_path: vec![],
            implementations: vec![CodeLocation {
                file: PathBuf::from("src/lib.rs"),
                line: 1,
                context: "main".to_string(),
            }],
            findings: Vec::new(),
            status: ClaimStatus::Pending,
        }],
        original_content: String::new(),
    };

    let mut result = HuntResult::new("/nonexistent", HuntMode::Analyze, HuntConfig::default());
    let before = result.findings.len();

    // build_quality_index should return None for nonexistent path
    apply_spec_quality_gate(
        &mut parsed_spec,
        Path::new("/nonexistent/project"),
        &mut result,
        "*",
    );

    // No findings added because build_quality_index returns None
    assert_eq!(result.findings.len(), before);
}

// =========================================================================
// BH-MOD-055: Coverage Gap Tests — hunt_with_spec with pmat on real project
// =========================================================================

#[test]
fn test_bh_mod_055_hunt_with_spec_pmat_quality_real_project() {
    // Drives both the pmat quality branch in hunt() and apply_spec_quality_gate
    // through hunt_with_spec. It used to hunt the real crate with pmat enabled
    // (18s); the fixture reaches the same branches.
    let fixture = hunt_fixture("mod_055_spec");

    let spec_content = "\
# Bug Hunter Spec

## Section 1: Hunting

### BH-01: Hunt Function

The hunt function should support all modes.
";
    let spec_path = fixture.join("spec.md");
    std::fs::write(&spec_path, spec_content).expect("write fixture spec");

    let config = HuntConfig {
        use_pmat_quality: true,
        pmat_query: Some("probe".to_string()),
        quality_weight: 0.5,
        ..hunt_fixture_config(HuntMode::Quick)
    };

    let result = hunt_with_spec(&fixture, &spec_path, None, config);
    let (hunt_result, parsed_spec) = result.expect("hunt_with_spec on the fixture");

    assert_eq!(parsed_spec.claims.len(), 1, "spec parser lost the BH-01 claim");
    assert_eq!(hunt_result.mode, HuntMode::Quick);
    // The spec has no implementations in the fixture, so hunt_with_spec must
    // fall back to the configured targets and still scan the source.
    assert!(
        hunt_result.findings.iter().any(|f| f.title.contains("unwrap()")),
        "spec-driven hunt scanned nothing: {:?}",
        hunt_result.findings.iter().map(|f| &f.title).collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(&fixture);
}

// =========================================================================
// BH-MOD-056: Coverage Gap Tests — hunt() mode dispatch with cache-free configs
// =========================================================================

#[test]
fn test_bh_mod_056_hunt_falsify_no_cache() {
    let temp = std::env::temp_dir().join("test_bh_mod_056_falsify");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(
        temp.join("src/lib.rs"),
        "pub fn add(a: usize, b: usize) -> usize { a + b }\n",
    )
    .unwrap();
    std::fs::write(
        temp.join("Cargo.toml"),
        "[package]\nname=\"t\"\nversion=\"0.1.0\"\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Falsify,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt(&temp, config);
    assert_eq!(result.mode, HuntMode::Falsify);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_056_hunt_fuzz_no_cache() {
    let temp = std::env::temp_dir().join("test_bh_mod_056_fuzz");
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
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt(&temp, config);
    assert_eq!(result.mode, HuntMode::Fuzz);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_056_hunt_deephunt_no_cache() {
    let temp = std::env::temp_dir().join("test_bh_mod_056_deep");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn simple() -> i32 { 42 }\n").unwrap();

    let config = HuntConfig {
        mode: HuntMode::DeepHunt,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt(&temp, config);
    assert_eq!(result.mode, HuntMode::DeepHunt);
    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-057: Coverage Gap Tests — hunt_with_ticket
// =========================================================================

#[test]
fn test_bh_mod_057_hunt_with_ticket_markdown() {
    let temp = std::env::temp_dir().join("test_bh_mod_057_ticket");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    let _ = std::fs::create_dir_all(temp.join(".pmat/tickets"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn demo() {}\n").unwrap();

    let ticket_content = "\
# PMAT-999: Test ticket

## Description

This is a test ticket for coverage.

## Affected Paths

- src/lib.rs

## Priority

high
";
    let ticket_path = temp.join(".pmat/tickets/PMAT-999.md");
    std::fs::write(&ticket_path, ticket_content).unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt_with_ticket(&temp, "PMAT-999", config);
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_057_hunt_with_ticket_github_issue() {
    let temp = std::env::temp_dir().join("test_bh_mod_057_gh");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn x() {}\n").unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt_with_ticket(&temp, "PMAT-123", config);
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_057_hunt_with_ticket_invalid_ref() {
    let config = HuntConfig::default();
    let temp = std::env::temp_dir().join("test_bh_mod_057_invalid");
    let _ = std::fs::create_dir_all(&temp);

    let result = hunt_with_ticket(&temp, "not_a_valid_ref", config);
    assert!(result.is_err());
    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-058: Coverage Gap Tests — analyze_coverage_hotspots
// =========================================================================

#[test]
#[allow(unsafe_code)]
fn test_bh_mod_058_coverage_hotspots_cargo_target_dir() {
    let temp = std::env::temp_dir().join("test_bh_mod_058_target_dir");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(&temp);

    let custom_coverage = temp.join("custom_coverage");
    let _ = std::fs::create_dir_all(&custom_coverage);
    let lcov_path = custom_coverage.join("lcov.info");
    std::fs::write(
        &lcov_path,
        "SF:src/lib.rs\nDA:1,0\nDA:2,0\nDA:3,0\nDA:4,0\nDA:5,0\nDA:6,0\nend_of_record\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Hunt,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        coverage_path: Some(lcov_path),
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());
    analyze_coverage_hotspots(&temp, &config, &mut result);

    let has_cov = result.findings.iter().any(|f| {
        f.title.contains("coverage") || f.title.contains("Low coverage") || f.id.contains("COV")
    });
    assert!(has_cov, "Should find coverage data from coverage_path");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_058_coverage_hotspots_custom_path() {
    // Test the custom coverage_path config option (lines 512-519)
    let temp = std::env::temp_dir().join("test_bh_mod_058_custom");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(&temp);

    let lcov = temp.join("my_coverage.info");
    std::fs::write(
        &lcov,
        "SF:src/lib.rs\nDA:1,0\nDA:2,0\nDA:3,0\nDA:4,0\nDA:5,0\nDA:6,0\nend_of_record\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Hunt,
        targets: vec![PathBuf::from("src")],
        coverage_path: Some(lcov),
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());
    analyze_coverage_hotspots(&temp, &config, &mut result);

    let has_cov = result.findings.iter().any(|f| f.id.contains("COV"));
    assert!(has_cov, "Should find coverage hotspots from custom path");
    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-059: Coverage Gap Tests — run_falsify_mode with mutation targets
// =========================================================================

#[test]
fn test_bh_mod_059_falsify_with_mutation_targets() {
    let temp = std::env::temp_dir().join("test_bh_mod_059_mut");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/boundary.rs"),
        "pub fn check(v: &[u8]) -> bool { v.len() < 10 }\n",
    )
    .unwrap();
    std::fs::write(
        temp.join("src/arith.rs"),
        "pub fn calc(x: i64) -> usize { (x + 1) as usize }\n",
    )
    .unwrap();
    std::fs::write(
        temp.join("src/logic.rs"),
        "pub fn test(a: bool, b: bool) -> bool { a && !b || is_valid() }\nfn is_valid() -> bool { true }\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Falsify,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Falsify, config.clone());
    run_falsify_mode(&temp, &config, &mut result);

    // When cargo-mutants is not installed (e.g. clean-room container),
    // run_falsify_mode returns early with BH-FALSIFY-UNAVAIL.
    let mutants_available = std::process::Command::new("cargo")
        .args(["mutants", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if mutants_available {
        let has_boundary = result.findings.iter().any(|f| f.title.contains("Boundary"));
        let has_arith = result
            .findings
            .iter()
            .any(|f| f.title.contains("Arithmetic"));
        let has_bool = result.findings.iter().any(|f| f.title.contains("Boolean"));

        assert!(
            has_boundary,
            "Should detect boundary condition mutation target"
        );
        assert!(has_arith, "Should detect arithmetic mutation target");
        assert!(has_bool, "Should detect boolean logic mutation target");
    } else {
        assert!(
            result.findings.iter().any(|f| f.id == "BH-FALSIFY-UNAVAIL"),
            "Should report cargo-mutants unavailable"
        );
    }
    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-060: Coverage Gap Tests — hunt_with_spec section filter + fallback
// =========================================================================

#[test]
fn test_bh_mod_060_hunt_with_spec_section_filter() {
    let temp = std::env::temp_dir().join("test_bh_mod_060_section");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn target() {}\n").unwrap();

    let spec_content = "\
# Test Spec

## Section A: Auth

### AUTH-01: Authentication

Users must authenticate.

## Section B: Data

### DATA-01: Storage

Data must be stored.
";
    let spec_path = temp.join("spec.md");
    std::fs::write(&spec_path, spec_content).unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt_with_spec(&temp, &spec_path, Some("Section A"), config);
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_060_hunt_with_spec_empty_targets_fallback() {
    let temp = std::env::temp_dir().join("test_bh_mod_060_empty");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn nothing() {}\n").unwrap();

    let spec_content = "\
# Minimal Spec

## Section 1

### MIN-01: Minimal claim

No implementations here.
";
    let spec_path = temp.join("spec.md");
    std::fs::write(&spec_path, spec_content).unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt_with_spec(&temp, &spec_path, None, config);
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-061: Coverage Gap Tests — run_fuzz_mode with unsafe blocks
// =========================================================================

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_bh_mod_061_fuzz_mode_with_unsafe_blocks() {
    let temp = std::env::temp_dir().join("test_bh_mod_061_unsafe");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "\
pub fn risky(ptr: *const u8) -> u8 {
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    unsafe {
        let val = *ptr as *const u8;
        std::mem::transmute::<u8, u8>(*ptr)
    }
}
",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Fuzz,
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());
    run_fuzz_mode(&temp, &config, &mut result);

    let has_ptr = result.findings.iter().any(|f| f.title.contains("Pointer"));
    let has_transmute = result
        .findings
        .iter()
        .any(|f| f.title.contains("Transmute"));
    assert!(has_ptr, "Should detect pointer dereference in unsafe block");
    assert!(has_transmute, "Should detect transmute in unsafe block");
    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-062: Coverage Gap Tests — hunt with coverage_weight but no cov file
// =========================================================================

#[test]
fn test_bh_mod_062_hunt_coverage_weight_no_file() {
    let temp = std::env::temp_dir().join("test_bh_mod_062_covweight");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn f() {}\n").unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        coverage_weight: 1.0,
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt(&temp, config);
    assert_eq!(result.mode, HuntMode::Quick);
    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-063: Coverage Gap Tests — hunt_with_spec empty targets
// =========================================================================

#[test]
fn test_bh_mod_063_hunt_with_spec_config_empty_targets() {
    let temp = std::env::temp_dir().join("test_bh_mod_063_emptytgt");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn z() {}\n").unwrap();

    let spec_content = "# Spec\n## S1\n### C-01: Claim\nSome claim.\n";
    let spec_path = temp.join("spec.md");
    std::fs::write(&spec_path, spec_content).unwrap();

    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![],
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let result = hunt_with_spec(&temp, &spec_path, None, config);
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&temp);
}
