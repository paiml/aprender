use super::*;

// =========================================================================
// BH-TYP-001: HuntMode Display
// =========================================================================

#[test]
fn test_bh_typ_001_hunt_mode_display() {
    assert_eq!(format!("{}", HuntMode::Falsify), "Falsify");
    assert_eq!(format!("{}", HuntMode::Hunt), "Hunt");
    assert_eq!(format!("{}", HuntMode::Analyze), "Analyze");
    assert_eq!(format!("{}", HuntMode::Fuzz), "Fuzz");
    assert_eq!(format!("{}", HuntMode::DeepHunt), "DeepHunt");
}

// =========================================================================
// BH-TYP-002: FindingSeverity Ordering
// =========================================================================

#[test]
fn test_bh_typ_002_finding_severity_ordering() {
    assert!(FindingSeverity::Critical > FindingSeverity::High);
    assert!(FindingSeverity::High > FindingSeverity::Medium);
    assert!(FindingSeverity::Medium > FindingSeverity::Low);
    assert!(FindingSeverity::Low > FindingSeverity::Info);
}

#[test]
fn test_bh_typ_002_finding_severity_display() {
    assert_eq!(format!("{}", FindingSeverity::Critical), "CRITICAL");
    assert_eq!(format!("{}", FindingSeverity::High), "HIGH");
    assert_eq!(format!("{}", FindingSeverity::Medium), "MEDIUM");
    assert_eq!(format!("{}", FindingSeverity::Low), "LOW");
    assert_eq!(format!("{}", FindingSeverity::Info), "INFO");
}

// =========================================================================
// BH-TYP-003: Finding Builder
// =========================================================================

#[test]
fn test_bh_typ_003_finding_builder() {
    let finding = Finding::new("BH-001", "src/lib.rs", 42, "Potential null dereference")
        .with_column(10)
        .with_description("This line may dereference a null pointer")
        .with_severity(FindingSeverity::High)
        .with_category(DefectCategory::MemorySafety)
        .with_suspiciousness(0.95);

    assert_eq!(finding.id, "BH-001");
    assert_eq!(finding.line, 42);
    assert_eq!(finding.column, Some(10));
    assert_eq!(finding.severity, FindingSeverity::High);
    assert_eq!(finding.category, DefectCategory::MemorySafety);
    assert!((finding.suspiciousness - 0.95).abs() < 0.001);
}

#[test]
fn test_bh_typ_003_finding_location() {
    let finding = Finding::new("BH-001", "src/lib.rs", 42, "Test");
    assert_eq!(finding.location(), "src/lib.rs:42");

    let finding_with_col = finding.with_column(10);
    assert_eq!(finding_with_col.location(), "src/lib.rs:42:10");
}

// =========================================================================
// BH-TYP-004: Evidence Types
// =========================================================================

#[test]
fn test_bh_typ_004_evidence_mutation() {
    let evidence = FindingEvidence::mutation("mut_001", true);
    assert_eq!(evidence.evidence_type, EvidenceKind::MutationSurvival);
    assert!(evidence.description.contains("SURVIVED"));
}

#[test]
fn test_bh_typ_004_evidence_sbfl() {
    let evidence = FindingEvidence::sbfl("Ochiai", 0.875);
    assert_eq!(evidence.evidence_type, EvidenceKind::SbflScore);
    assert!(evidence.description.contains("0.875"));
}

#[test]
fn test_bh_typ_004_evidence_static() {
    let evidence = FindingEvidence::static_analysis("clippy", "unused variable");
    assert_eq!(evidence.evidence_type, EvidenceKind::StaticAnalysis);
    assert!(evidence.description.contains("[clippy]"));
}

// =========================================================================
// BH-TYP-005: HuntConfig Defaults
// =========================================================================

#[test]
fn test_bh_typ_005_hunt_config_defaults() {
    let config = HuntConfig::default();
    assert_eq!(config.mode, HuntMode::Analyze);
    assert_eq!(config.max_findings, 50);
    assert!((config.min_suspiciousness - 0.5).abs() < 0.001);
    assert_eq!(config.sbfl_formula, SbflFormula::Ochiai);
    assert!(!config.llm_filter);
}

// =========================================================================
// BH-TYP-006: HuntResult Operations
// =========================================================================

#[test]
fn test_bh_typ_006_hunt_result_new() {
    let config = HuntConfig::default();
    let result = HuntResult::new(".", HuntMode::Analyze, config);

    assert_eq!(result.mode, HuntMode::Analyze);
    assert!(result.findings.is_empty());
}

#[test]
fn test_bh_typ_006_hunt_result_add_finding() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new(".", HuntMode::Analyze, config);

    result.add_finding(Finding::new("F-001", "test.rs", 1, "Test finding"));
    result.add_finding(Finding::new("F-002", "test.rs", 2, "Another finding"));

    assert_eq!(result.findings.len(), 2);
}

#[test]
fn test_bh_typ_006_hunt_result_top_findings() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new(".", HuntMode::Analyze, config);

    result.add_finding(Finding::new("F-001", "test.rs", 1, "Low").with_suspiciousness(0.3));
    result.add_finding(Finding::new("F-002", "test.rs", 2, "High").with_suspiciousness(0.9));
    result.add_finding(Finding::new("F-003", "test.rs", 3, "Medium").with_suspiciousness(0.6));

    let top = result.top_findings(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].id, "F-002"); // Highest suspiciousness
    assert_eq!(top[1].id, "F-003");
}

// =========================================================================
// BH-TYP-007: HuntStats Computation
// =========================================================================

#[test]
fn test_bh_typ_007_hunt_stats_from_findings() {
    let findings = vec![
        Finding::new("F-001", "test.rs", 1, "Critical")
            .with_severity(FindingSeverity::Critical)
            .with_category(DefectCategory::MemorySafety)
            .with_suspiciousness(0.9),
        Finding::new("F-002", "test.rs", 2, "High")
            .with_severity(FindingSeverity::High)
            .with_category(DefectCategory::ConcurrencyBugs)
            .with_suspiciousness(0.7),
        Finding::new("F-003", "test.rs", 3, "Medium")
            .with_severity(FindingSeverity::Medium)
            .with_category(DefectCategory::MemorySafety)
            .with_suspiciousness(0.5),
    ];

    let stats = HuntStats::from_findings(&findings);

    assert_eq!(stats.total_findings, 3);
    assert_eq!(stats.by_severity.get(&FindingSeverity::Critical), Some(&1));
    assert_eq!(stats.by_severity.get(&FindingSeverity::High), Some(&1));
    assert_eq!(stats.by_severity.get(&FindingSeverity::Medium), Some(&1));
    assert_eq!(stats.by_category.get(&DefectCategory::MemorySafety), Some(&2));
    assert!((stats.avg_suspiciousness - 0.7).abs() < 0.001);
    assert!((stats.max_suspiciousness - 0.9).abs() < 0.001);
}

#[test]
fn test_bh_typ_007_hunt_stats_empty() {
    let stats = HuntStats::from_findings(&[]);
    assert_eq!(stats.total_findings, 0);
    assert_eq!(stats.avg_suspiciousness, 0.0);
}

// =========================================================================
// BH-TYP-008: SbflFormula Display
// =========================================================================

#[test]
fn test_bh_typ_008_sbfl_formula_display() {
    assert_eq!(format!("{}", SbflFormula::Tarantula), "Tarantula");
    assert_eq!(format!("{}", SbflFormula::Ochiai), "Ochiai");
    assert_eq!(format!("{}", SbflFormula::DStar2), "DStar2");
    assert_eq!(format!("{}", SbflFormula::DStar3), "DStar3");
}

// =========================================================================
// BH-TYP-009: DefectCategory Display
// =========================================================================

#[test]
fn test_bh_typ_009_defect_category_display() {
    assert_eq!(format!("{}", DefectCategory::MemorySafety), "MemorySafety");
    assert_eq!(format!("{}", DefectCategory::ConcurrencyBugs), "ConcurrencyBugs");
    assert_eq!(format!("{}", DefectCategory::SecurityVulnerabilities), "SecurityVulnerabilities");
}

// =========================================================================
// BH-TYP-010: Suspiciousness Clamping
// =========================================================================

#[test]
fn test_bh_typ_010_suspiciousness_clamping() {
    let finding = Finding::new("F-001", "test.rs", 1, "Test").with_suspiciousness(1.5); // Above 1.0
    assert!((finding.suspiciousness - 1.0).abs() < 0.001);

    let finding = Finding::new("F-002", "test.rs", 1, "Test").with_suspiciousness(-0.5); // Below 0.0
    assert!((finding.suspiciousness - 0.0).abs() < 0.001);
}

// =========================================================================
// BH-TYP-011: HuntMode::Quick Display (missing from BH-TYP-001)
// =========================================================================

#[test]
fn test_bh_typ_011_hunt_mode_quick_display() {
    assert_eq!(format!("{}", HuntMode::Quick), "Quick");
}

// =========================================================================
// BH-TYP-012: DefectCategory Display - ALL variants
// =========================================================================

#[test]
fn test_bh_typ_012_defect_category_display_all_variants() {
    assert_eq!(format!("{}", DefectCategory::TraitBounds), "TraitBounds");
    assert_eq!(format!("{}", DefectCategory::AstTransform), "ASTTransform");
    assert_eq!(format!("{}", DefectCategory::OwnershipBorrow), "OwnershipBorrow");
    assert_eq!(format!("{}", DefectCategory::ConfigurationErrors), "ConfigurationErrors");
    assert_eq!(format!("{}", DefectCategory::TypeErrors), "TypeErrors");
    assert_eq!(format!("{}", DefectCategory::LogicErrors), "LogicErrors");
    assert_eq!(format!("{}", DefectCategory::PerformanceIssues), "PerformanceIssues");
    assert_eq!(format!("{}", DefectCategory::GpuKernelBugs), "GpuKernelBugs");
    assert_eq!(format!("{}", DefectCategory::SilentDegradation), "SilentDegradation");
    assert_eq!(format!("{}", DefectCategory::TestDebt), "TestDebt");
    assert_eq!(format!("{}", DefectCategory::HiddenDebt), "HiddenDebt");
    assert_eq!(format!("{}", DefectCategory::ContractGap), "ContractGap");
    assert_eq!(format!("{}", DefectCategory::ModelParityGap), "ModelParityGap");
    assert_eq!(format!("{}", DefectCategory::Unknown), "Unknown");
}

// =========================================================================
// BH-TYP-013: ChannelWeights normalize and combine
// =========================================================================

#[test]
fn test_bh_typ_013_channel_weights_normalize() {
    let mut weights = ChannelWeights {
        spectrum: 2.0,
        mutation: 2.0,
        static_analysis: 2.0,
        semantic: 2.0,
        quality: 2.0,
    };
    weights.normalize();
    // Each should be 0.2 after normalization (2.0 / 10.0)
    assert!((weights.spectrum - 0.2).abs() < 0.001);
    assert!((weights.mutation - 0.2).abs() < 0.001);
    assert!((weights.static_analysis - 0.2).abs() < 0.001);
    assert!((weights.semantic - 0.2).abs() < 0.001);
    assert!((weights.quality - 0.2).abs() < 0.001);
}

#[test]
fn test_bh_typ_013_channel_weights_normalize_zero_sum() {
    let mut weights = ChannelWeights {
        spectrum: 0.0,
        mutation: 0.0,
        static_analysis: 0.0,
        semantic: 0.0,
        quality: 0.0,
    };
    weights.normalize();
    // All zero: no division, should remain zero
    assert!((weights.spectrum).abs() < 0.001);
    assert!((weights.mutation).abs() < 0.001);
}

#[test]
fn test_bh_typ_013_channel_weights_normalize_unequal() {
    let mut weights = ChannelWeights {
        spectrum: 1.0,
        mutation: 0.0,
        static_analysis: 0.0,
        semantic: 0.0,
        quality: 0.0,
    };
    weights.normalize();
    assert!((weights.spectrum - 1.0).abs() < 0.001);
    assert!((weights.mutation).abs() < 0.001);
}

#[test]
fn test_bh_typ_013_channel_weights_combine() {
    let weights = ChannelWeights::default();
    // default: spectrum=0.30, mutation=0.25, static=0.20, semantic=0.15, quality=0.10
    let score = weights.combine(1.0, 1.0, 1.0, 1.0, 1.0);
    // All inputs 1.0: should sum to 1.0
    assert!((score - 1.0).abs() < 0.001);

    let score_zeros = weights.combine(0.0, 0.0, 0.0, 0.0, 0.0);
    assert!((score_zeros).abs() < 0.001);

    let score_partial = weights.combine(1.0, 0.0, 0.0, 0.0, 0.0);
    assert!((score_partial - 0.30).abs() < 0.001);
}

// =========================================================================
// BH-TYP-014: HuntResult summary
// =========================================================================

#[test]
fn test_bh_typ_014_hunt_result_summary_empty() {
    let config = HuntConfig::default();
    let result = HuntResult::new(".", HuntMode::Analyze, config);
    let summary = result.summary();
    assert!(summary.contains("Analyze mode"));
    assert!(summary.contains("0 findings"));
    assert!(summary.contains("0C 0H"));
}

#[test]
fn test_bh_typ_014_hunt_result_summary_with_findings() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new("/project", HuntMode::Hunt, config);
    result.add_finding(
        Finding::new("F-001", "a.rs", 1, "Crit").with_severity(FindingSeverity::Critical),
    );
    result
        .add_finding(Finding::new("F-002", "b.rs", 2, "High").with_severity(FindingSeverity::High));
    result.add_finding(Finding::new("F-003", "a.rs", 3, "Low").with_severity(FindingSeverity::Low));
    result.duration_ms = 1234;
    result.finalize();
    let summary = result.summary();
    assert!(summary.contains("Hunt mode"));
    assert!(summary.contains("3 findings"));
    assert!(summary.contains("2 files"));
    assert!(summary.contains("1C 1H"));
    assert!(summary.contains("1234ms"));
}

#[test]
fn test_bh_typ_014_hunt_result_summary_no_critical_or_high() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new(".", HuntMode::Fuzz, config);
    result.add_finding(
        Finding::new("F-001", "test.rs", 1, "Low finding").with_severity(FindingSeverity::Low),
    );
    result.finalize();
    let summary = result.summary();
    assert!(summary.contains("Fuzz mode"));
    assert!(summary.contains("0C 0H"));
}

// =========================================================================
// BH-TYP-015: HuntResult finalize
// =========================================================================

#[test]
fn test_bh_typ_015_hunt_result_finalize() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new(".", HuntMode::Analyze, config);
    result.add_finding(
        Finding::new("F-001", "a.rs", 1, "Test")
            .with_severity(FindingSeverity::High)
            .with_suspiciousness(0.8),
    );
    result.add_finding(
        Finding::new("F-002", "b.rs", 2, "Test2")
            .with_severity(FindingSeverity::Medium)
            .with_suspiciousness(0.6),
    );
    assert_eq!(result.stats.total_findings, 0); // Before finalize
    result.finalize();
    assert_eq!(result.stats.total_findings, 2);
    assert_eq!(result.stats.files_analyzed, 2);
    assert!((result.stats.avg_suspiciousness - 0.7).abs() < 0.001);
    assert!((result.stats.max_suspiciousness - 0.8).abs() < 0.001);
}

// =========================================================================
// BH-TYP-016: HuntResult with_duration and default
// =========================================================================

#[test]
fn test_bh_typ_016_hunt_result_with_duration() {
    let config = HuntConfig::default();
    let result = HuntResult::new(".", HuntMode::Quick, config).with_duration(5000);
    assert_eq!(result.duration_ms, 5000);
}

#[test]
fn test_bh_typ_016_hunt_result_default() {
    let result = HuntResult::default();
    assert_eq!(result.mode, HuntMode::Quick);
    assert!(result.findings.is_empty());
    assert_eq!(result.duration_ms, 0);
    assert!(result.timestamp.is_empty());
    assert_eq!(result.project_path, PathBuf::new());
}

// =========================================================================
// BH-TYP-017: Finding builder - fix, regression_risk, blame, discovered_by
// =========================================================================

#[test]
fn test_bh_typ_017_finding_with_fix() {
    let finding =
        Finding::new("F-001", "test.rs", 1, "Test").with_fix("Replace unwrap() with expect()");
    assert_eq!(finding.suggested_fix, Some("Replace unwrap() with expect()".to_string()));
}

#[test]
fn test_bh_typ_017_finding_with_regression_risk() {
    let finding = Finding::new("F-001", "test.rs", 1, "Test").with_regression_risk(0.75);
    assert_eq!(finding.regression_risk, Some(0.75));

    // Test clamping above 1.0
    let finding_high = Finding::new("F-002", "test.rs", 1, "Test").with_regression_risk(1.5);
    assert_eq!(finding_high.regression_risk, Some(1.0));

    // Test clamping below 0.0
    let finding_low = Finding::new("F-003", "test.rs", 1, "Test").with_regression_risk(-0.5);
    assert_eq!(finding_low.regression_risk, Some(0.0));
}

#[test]
fn test_bh_typ_017_finding_with_blame() {
    let finding =
        Finding::new("F-001", "test.rs", 1, "Test").with_blame("Alice", "abc123", "2024-01-15");
    assert_eq!(finding.blame_author, Some("Alice".to_string()));
    assert_eq!(finding.blame_commit, Some("abc123".to_string()));
    assert_eq!(finding.blame_date, Some("2024-01-15".to_string()));
}

#[test]
fn test_bh_typ_017_finding_with_discovered_by() {
    let finding = Finding::new("F-001", "test.rs", 1, "Test").with_discovered_by(HuntMode::Fuzz);
    assert_eq!(finding.discovered_by, HuntMode::Fuzz);

    let finding2 =
        Finding::new("F-002", "test.rs", 1, "Test").with_discovered_by(HuntMode::DeepHunt);
    assert_eq!(finding2.discovered_by, HuntMode::DeepHunt);
}

// =========================================================================
// BH-TYP-018: Evidence - fuzzing, concolic, quality_metrics
// =========================================================================

#[test]
fn test_bh_typ_018_evidence_fuzzing() {
    let evidence = FindingEvidence::fuzzing("\\x00\\xff\\x42", "buffer overflow");
    assert_eq!(evidence.evidence_type, EvidenceKind::FuzzingCrash);
    assert_eq!(evidence.description, "buffer overflow");
    assert_eq!(evidence.data, Some("\\x00\\xff\\x42".to_string()));
}

#[test]
fn test_bh_typ_018_evidence_concolic() {
    let evidence = FindingEvidence::concolic("x > 0 && y < 100");
    assert_eq!(evidence.evidence_type, EvidenceKind::ConcolicPath);
    assert_eq!(evidence.description, "Path constraint solved");
    assert_eq!(evidence.data, Some("x > 0 && y < 100".to_string()));
}

#[test]
fn test_bh_typ_018_evidence_quality_metrics() {
    let evidence = FindingEvidence::quality_metrics("A", 92.5, 8);
    assert_eq!(evidence.evidence_type, EvidenceKind::QualityMetrics);
    assert!(evidence.description.contains("PMAT grade A"));
    assert!(evidence.description.contains("TDG: 92.5"));
    assert!(evidence.description.contains("complexity: 8"));
    assert_eq!(evidence.data, Some("92.5".to_string()));
}

#[test]
fn test_bh_typ_018_evidence_mutation_killed() {
    let evidence = FindingEvidence::mutation("mut_002", false);
    assert_eq!(evidence.evidence_type, EvidenceKind::MutationSurvival);
    assert!(evidence.description.contains("KILLED"));
    assert_eq!(evidence.data, Some("KILLED".to_string()));
}

// =========================================================================
// BH-TYP-019: LocalizationStrategy Display
// =========================================================================

#[test]
fn test_bh_typ_019_localization_strategy_display() {
    assert_eq!(format!("{}", LocalizationStrategy::Sbfl), "SBFL");
    assert_eq!(format!("{}", LocalizationStrategy::Mbfl), "MBFL");
    assert_eq!(format!("{}", LocalizationStrategy::Causal), "Causal");
    assert_eq!(format!("{}", LocalizationStrategy::MultiChannel), "MultiChannel");
    assert_eq!(format!("{}", LocalizationStrategy::Hybrid), "Hybrid");
}

// =========================================================================
// BH-TYP-020: Default impls and serialization
// =========================================================================

#[test]
fn test_bh_typ_020_crash_bucketing_mode_default() {
    let mode = CrashBucketingMode::default();
    assert_eq!(mode, CrashBucketingMode::None);
}

#[test]
fn test_bh_typ_020_localization_strategy_default() {
    let strategy = LocalizationStrategy::default();
    assert_eq!(strategy, LocalizationStrategy::Sbfl);
}

#[test]
fn test_bh_typ_020_sbfl_formula_default() {
    let formula = SbflFormula::default();
    assert_eq!(formula, SbflFormula::Ochiai);
}

#[test]
fn test_bh_typ_020_channel_weights_default() {
    let w = ChannelWeights::default();
    assert!((w.spectrum - 0.30).abs() < 0.001);
    assert!((w.mutation - 0.25).abs() < 0.001);
    assert!((w.static_analysis - 0.20).abs() < 0.001);
    assert!((w.semantic - 0.15).abs() < 0.001);
    assert!((w.quality - 0.10).abs() < 0.001);
}

#[test]
fn test_bh_typ_020_phase_timings_default() {
    let timings = PhaseTimings::default();
    assert_eq!(timings.mode_dispatch_ms, 0);
    assert_eq!(timings.pmat_index_ms, 0);
    assert_eq!(timings.pmat_weights_ms, 0);
    assert_eq!(timings.finalize_ms, 0);
}

#[test]
fn test_bh_typ_020_mode_stats_default() {
    let stats = ModeStats::default();
    assert_eq!(stats.mutants_total, 0);
    assert_eq!(stats.mutants_killed, 0);
    assert_eq!(stats.mutants_survived, 0);
    assert_eq!(stats.sbfl_passing_tests, 0);
    assert_eq!(stats.sbfl_failing_tests, 0);
    assert_eq!(stats.fuzz_executions, 0);
    assert_eq!(stats.fuzz_crashes, 0);
    assert!((stats.fuzz_coverage - 0.0).abs() < 0.001);
    assert_eq!(stats.concolic_paths, 0);
    assert_eq!(stats.concolic_constraints_solved, 0);
    assert_eq!(stats.concolic_timeouts, 0);
    assert_eq!(stats.llm_filtered, 0);
    assert_eq!(stats.llm_retained, 0);
}

// =========================================================================
// BH-TYP-021: HuntConfig advanced defaults (BH-11 to BH-25)
// =========================================================================

#[test]
fn test_bh_typ_021_hunt_config_advanced_defaults() {
    let config = HuntConfig::default();
    // BH-11 to BH-15
    assert!(config.spec_path.is_none());
    assert!(config.spec_section.is_none());
    assert!(config.ticket_ref.is_none());
    assert!(!config.update_spec);
    assert!(!config.lib_only);
    assert!(config.bin_target.is_none());
    assert!(config.exclude_tests); // Default true
    assert!(config.suppress_false_positives); // Default true
    assert!(config.coverage_path.is_none());
    assert!((config.coverage_weight - 0.5).abs() < 0.001);
    // BH-16 to BH-20
    assert_eq!(config.localization_strategy, LocalizationStrategy::Sbfl);
    assert!(!config.predictive_mutation);
    assert_eq!(config.crash_bucketing, CrashBucketingMode::None);
    // BH-21 to BH-25
    assert!(!config.use_pmat_quality);
    assert!((config.quality_weight - 0.5).abs() < 0.001);
    assert!(!config.pmat_scope);
    assert!(config.pmat_satd); // Default true
    assert!(config.pmat_query.is_none());
}

// =========================================================================
// BH-TYP-022: Finding with_evidence builder
// =========================================================================

#[test]
fn test_bh_typ_022_finding_with_evidence() {
    let finding = Finding::new("F-001", "test.rs", 1, "Test")
        .with_evidence(FindingEvidence::mutation("mut_001", true))
        .with_evidence(FindingEvidence::sbfl("Ochiai", 0.9));
    assert_eq!(finding.evidence.len(), 2);
    assert_eq!(finding.evidence[0].evidence_type, EvidenceKind::MutationSurvival);
    assert_eq!(finding.evidence[1].evidence_type, EvidenceKind::SbflScore);
}

// =========================================================================
// BH-TYP-023: HuntStats from_findings with unique files tracking
// =========================================================================

#[test]
fn test_bh_typ_023_hunt_stats_unique_files() {
    let findings = vec![
        Finding::new("F-001", "a.rs", 1, "Bug 1")
            .with_severity(FindingSeverity::High)
            .with_suspiciousness(0.8),
        Finding::new("F-002", "a.rs", 10, "Bug 2")
            .with_severity(FindingSeverity::Medium)
            .with_suspiciousness(0.6),
        Finding::new("F-003", "b.rs", 5, "Bug 3")
            .with_severity(FindingSeverity::Low)
            .with_suspiciousness(0.4),
    ];
    let stats = HuntStats::from_findings(&findings);
    // Two unique files: a.rs and b.rs
    assert_eq!(stats.files_analyzed, 2);
    assert_eq!(stats.total_findings, 3);
    assert_eq!(stats.by_category.get(&DefectCategory::Unknown), Some(&3));
}

// =========================================================================
// BH-TYP-024: Serialization round-trips
// =========================================================================

#[test]
fn test_bh_typ_024_hunt_mode_serde_roundtrip() {
    for mode in &[
        HuntMode::Falsify,
        HuntMode::Hunt,
        HuntMode::Analyze,
        HuntMode::Fuzz,
        HuntMode::DeepHunt,
        HuntMode::Quick,
    ] {
        let json = serde_json::to_string(mode).expect("serialize");
        let back: HuntMode = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*mode, back);
    }
}

#[test]
fn test_bh_typ_024_finding_severity_serde_roundtrip() {
    for sev in &[
        FindingSeverity::Info,
        FindingSeverity::Low,
        FindingSeverity::Medium,
        FindingSeverity::High,
        FindingSeverity::Critical,
    ] {
        let json = serde_json::to_string(sev).expect("serialize");
        let back: FindingSeverity = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*sev, back);
    }
}

#[test]
fn test_bh_typ_024_crash_bucketing_mode_serde() {
    for mode in
        &[CrashBucketingMode::None, CrashBucketingMode::StackTrace, CrashBucketingMode::Semantic]
    {
        let json = serde_json::to_string(mode).expect("serialize");
        let back: CrashBucketingMode = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*mode, back);
    }
}

#[test]
fn test_bh_typ_024_finding_serde_roundtrip() {
    let finding = Finding::new("F-001", "src/lib.rs", 42, "Bug found")
        .with_column(10)
        .with_description("A description")
        .with_severity(FindingSeverity::High)
        .with_category(DefectCategory::LogicErrors)
        .with_suspiciousness(0.85)
        .with_discovered_by(HuntMode::Fuzz)
        .with_fix("Fix the bug")
        .with_regression_risk(0.6)
        .with_blame("Author", "abc1234", "2024-06-01")
        .with_evidence(FindingEvidence::fuzzing("input", "crash"));

    let json = serde_json::to_string(&finding).expect("serialize");
    let back: Finding = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.id, "F-001");
    assert_eq!(back.column, Some(10));
    assert_eq!(back.severity, FindingSeverity::High);
    assert_eq!(back.category, DefectCategory::LogicErrors);
    assert_eq!(back.suggested_fix, Some("Fix the bug".to_string()));
    assert_eq!(back.regression_risk, Some(0.6));
    assert_eq!(back.blame_author, Some("Author".to_string()));
    assert_eq!(back.blame_commit, Some("abc1234".to_string()));
    assert_eq!(back.blame_date, Some("2024-06-01".to_string()));
    assert_eq!(back.evidence.len(), 1);
}

// =========================================================================
// BH-TYP-025: HuntResult top_findings edge cases
// =========================================================================

#[test]
fn test_bh_typ_025_top_findings_more_than_available() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new(".", HuntMode::Analyze, config);
    result.add_finding(Finding::new("F-001", "test.rs", 1, "Only one"));
    let top = result.top_findings(10);
    assert_eq!(top.len(), 1);
}

#[test]
fn test_bh_typ_025_top_findings_zero() {
    let config = HuntConfig::default();
    let result = HuntResult::new(".", HuntMode::Analyze, config);
    let top = result.top_findings(0);
    assert!(top.is_empty());
}

// =========================================================================
// BH-TYP-026: ContractGap and ModelParityGap variants (BH-26/BH-27)
// =========================================================================

#[test]
fn test_bh_typ_026_contract_gap_serde() {
    let cat = DefectCategory::ContractGap;
    let json = serde_json::to_string(&cat).expect("serialize");
    let back: DefectCategory = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, DefectCategory::ContractGap);
}

#[test]
fn test_bh_typ_026_model_parity_gap_serde() {
    let cat = DefectCategory::ModelParityGap;
    let json = serde_json::to_string(&cat).expect("serialize");
    let back: DefectCategory = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, DefectCategory::ModelParityGap);
}

#[test]
fn test_bh_typ_026_evidence_contract_binding() {
    let evidence =
        FindingEvidence::contract_binding("matmul-kernel-v1.yaml", "matmul", "not_implemented");
    assert_eq!(evidence.evidence_type, EvidenceKind::ContractBinding);
    assert!(evidence.description.contains("matmul-kernel-v1.yaml"));
    assert!(evidence.description.contains("not_implemented"));
}

#[test]
fn test_bh_typ_026_evidence_model_parity() {
    let evidence = FindingEvidence::model_parity("smollm-135m", "arithmetic", "missing");
    assert_eq!(evidence.evidence_type, EvidenceKind::ModelParity);
    assert!(evidence.description.contains("smollm-135m"));
    assert!(evidence.description.contains("missing"));
}

#[test]
fn test_bh_typ_026_hunt_config_bh26_27_defaults() {
    let config = HuntConfig::default();
    assert!(config.contracts_path.is_none());
    assert!(!config.contracts_auto);
    assert!(config.model_parity_path.is_none());
    assert!(!config.model_parity_auto);
}

#[test]
fn test_bh_typ_026_phase_timings_bh26_27_defaults() {
    let timings = PhaseTimings::default();
    assert_eq!(timings.contract_gap_ms, 0);
    assert_eq!(timings.model_parity_ms, 0);
}

#[test]
fn test_bh_typ_026_evidence_kind_serde() {
    for kind in &[EvidenceKind::ContractBinding, EvidenceKind::ModelParity] {
        let json = serde_json::to_string(kind).expect("serialize");
        let back: EvidenceKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*kind, back);
    }
}
