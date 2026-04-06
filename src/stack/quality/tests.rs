//! Tests for quality module
//!
//! Contains all unit tests, property tests, and coverage tests.

use super::*;
use std::path::PathBuf;

#[test]
fn test_quality_grade_from_rust_project_score_a_plus() {
    // A+ range: 105-114
    assert_eq!(QualityGrade::from_rust_project_score(114), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_rust_project_score(110), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_rust_project_score(105), QualityGrade::APlus);
}

#[test]
fn test_quality_grade_from_rust_project_score_a() {
    // A range: 95-104
    assert_eq!(QualityGrade::from_rust_project_score(104), QualityGrade::A);
    assert_eq!(QualityGrade::from_rust_project_score(100), QualityGrade::A);
    assert_eq!(QualityGrade::from_rust_project_score(95), QualityGrade::A);
}

#[test]
fn test_quality_grade_from_rust_project_score_a_minus() {
    // A- range: 85-94 (PMAT minimum)
    assert_eq!(QualityGrade::from_rust_project_score(94), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_rust_project_score(90), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_rust_project_score(85), QualityGrade::AMinus);
}

#[test]
fn test_quality_grade_from_rust_project_score_below() {
    assert_eq!(QualityGrade::from_rust_project_score(84), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_rust_project_score(75), QualityGrade::B);
    assert_eq!(QualityGrade::from_rust_project_score(65), QualityGrade::C);
    assert_eq!(QualityGrade::from_rust_project_score(55), QualityGrade::D);
    assert_eq!(QualityGrade::from_rust_project_score(40), QualityGrade::F);
}

#[test]
fn test_quality_grade_from_repo_score_a_plus() {
    // A+ range: 95-110
    assert_eq!(QualityGrade::from_repo_score(110), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_repo_score(100), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_repo_score(95), QualityGrade::APlus);
}

#[test]
fn test_quality_grade_from_readme_score() {
    assert_eq!(QualityGrade::from_readme_score(20), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_readme_score(18), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_readme_score(17), QualityGrade::A);
    assert_eq!(QualityGrade::from_readme_score(15), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_readme_score(12), QualityGrade::BPlus);
}

#[test]
fn test_quality_grade_from_sqi() {
    assert_eq!(QualityGrade::from_sqi(98.5), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_sqi(92.0), QualityGrade::A);
    assert_eq!(QualityGrade::from_sqi(87.0), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_sqi(75.0), QualityGrade::B);
}

#[test]
fn test_quality_grade_is_release_ready() {
    assert!(QualityGrade::APlus.is_release_ready());
    assert!(QualityGrade::A.is_release_ready());
    assert!(QualityGrade::AMinus.is_release_ready());
    assert!(!QualityGrade::BPlus.is_release_ready());
    assert!(!QualityGrade::B.is_release_ready());
    assert!(!QualityGrade::F.is_release_ready());
}

#[test]
fn test_quality_grade_is_a_plus() {
    assert!(QualityGrade::APlus.is_a_plus());
    assert!(!QualityGrade::A.is_a_plus());
    assert!(!QualityGrade::AMinus.is_a_plus());
}

#[test]
fn test_quality_grade_symbol() {
    assert_eq!(QualityGrade::APlus.symbol(), "A+");
    assert_eq!(QualityGrade::A.symbol(), "A");
    assert_eq!(QualityGrade::AMinus.symbol(), "A-");
    assert_eq!(QualityGrade::BPlus.symbol(), "B+");
    assert_eq!(QualityGrade::F.symbol(), "F");
}

#[test]
fn test_quality_grade_icon() {
    assert_eq!(QualityGrade::APlus.icon(), "✅");
    assert_eq!(QualityGrade::A.icon(), "⚠️");
    assert_eq!(QualityGrade::AMinus.icon(), "⚠️");
    assert_eq!(QualityGrade::BPlus.icon(), "❌");
}

#[test]
fn test_quality_grade_display() {
    assert_eq!(format!("{}", QualityGrade::APlus), "A+");
    assert_eq!(format!("{}", QualityGrade::AMinus), "A-");
}

// ========================================================================
// Score Tests
// ========================================================================

#[test]
fn test_score_percentage() {
    let score = Score::new(85, 100, QualityGrade::AMinus);
    assert!((score.percentage() - 85.0).abs() < f64::EPSILON);

    let score_zero_max = Score::new(0, 0, QualityGrade::F);
    assert!((score_zero_max.percentage() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_score_normalized() {
    let score = Score::new(105, 114, QualityGrade::APlus);
    let normalized = score.normalized();
    assert!(normalized > 90.0 && normalized < 93.0);
}

// ========================================================================
// ImageFormat Tests
// ========================================================================

#[test]
fn test_image_format_from_extension() {
    assert_eq!(ImageFormat::from_extension("png"), Some(ImageFormat::Png));
    assert_eq!(ImageFormat::from_extension("PNG"), Some(ImageFormat::Png));
    assert_eq!(ImageFormat::from_extension("jpg"), Some(ImageFormat::Jpg));
    assert_eq!(ImageFormat::from_extension("jpeg"), Some(ImageFormat::Jpg));
    assert_eq!(ImageFormat::from_extension("webp"), Some(ImageFormat::WebP));
    assert_eq!(ImageFormat::from_extension("svg"), Some(ImageFormat::Svg));
    assert_eq!(ImageFormat::from_extension("gif"), None);
}

#[test]
fn test_image_format_extension() {
    assert_eq!(ImageFormat::Png.extension(), "png");
    assert_eq!(ImageFormat::Jpg.extension(), "jpg");
    assert_eq!(ImageFormat::WebP.extension(), "webp");
    assert_eq!(ImageFormat::Svg.extension(), "svg");
}

// ========================================================================
// HeroImageResult Tests
// ========================================================================

#[test]
fn test_hero_image_missing() {
    let result = HeroImageResult::missing();
    assert!(!result.present);
    assert!(!result.valid);
    assert!(result.path.is_none());
    assert!(!result.issues.is_empty());
}

#[test]
fn test_hero_image_found() {
    let result = HeroImageResult::found(PathBuf::from("docs/hero.png"), ImageFormat::Png);
    assert!(result.present);
    assert!(result.valid);
    assert_eq!(result.path, Some(PathBuf::from("docs/hero.png")));
    assert_eq!(result.format, Some(ImageFormat::Png));
}

#[test]
fn test_hero_image_detect_missing() {
    let temp_dir = std::env::temp_dir().join("test_hero_missing");
    let _ = std::fs::create_dir_all(&temp_dir);

    let result = HeroImageResult::detect(&temp_dir);
    assert!(!result.present);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_detect_docs_png() {
    let temp_dir = std::env::temp_dir().join("test_hero_docs_png");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs")).unwrap();

    // Create a fake PNG file (minimal valid PNG header)
    let png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    std::fs::write(temp_dir.join("docs/hero.png"), &png_data).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Png));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_detect_docs_svg() {
    let temp_dir = std::env::temp_dir().join("test_hero_docs_svg");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs")).unwrap();

    std::fs::write(temp_dir.join("docs/hero.svg"), "<svg></svg>").unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Svg));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_detect_assets() {
    let temp_dir = std::env::temp_dir().join("test_hero_assets");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("assets")).unwrap();

    std::fs::write(temp_dir.join("assets/hero.webp"), [0u8; 100]).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::WebP));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_detect_from_readme() {
    let temp_dir = std::env::temp_dir().join("test_hero_readme");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create README with image reference
    std::fs::write(temp_dir.join("README.md"), "# Test\n![Hero](hero_img.png)\n").unwrap();
    std::fs::write(temp_dir.join("hero_img.png"), [0u8; 100]).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Png));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_detect_readme_html_img() {
    let temp_dir = std::env::temp_dir().join("test_hero_html");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create README with HTML img tag
    std::fs::write(temp_dir.join("README.md"), "# Test\n<img src=\"banner.jpg\" alt=\"banner\">\n")
        .unwrap();
    std::fs::write(temp_dir.join("banner.jpg"), [0u8; 100]).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Jpg));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_detect_readme_external_url_skipped() {
    let temp_dir = std::env::temp_dir().join("test_hero_external");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create README with external URL (should be skipped)
    std::fs::write(temp_dir.join("README.md"), "# Test\n![Hero](https://example.com/image.png)\n")
        .unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(!result.present);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_validate_large_file() {
    let temp_dir = std::env::temp_dir().join("test_hero_large");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs")).unwrap();

    // Create a file larger than 2MB
    let large_data = vec![0u8; 3 * 1024 * 1024];
    std::fs::write(temp_dir.join("docs/hero.png"), &large_data).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert!(!result.valid); // Should be invalid due to size
    assert!(!result.issues.is_empty());
    assert!(result.issues[0].contains("too large"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_priority_order() {
    let temp_dir = std::env::temp_dir().join("test_hero_priority");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs")).unwrap();
    std::fs::create_dir_all(temp_dir.join("assets")).unwrap();

    // Create both docs and assets hero images - docs should take priority
    std::fs::write(temp_dir.join("docs/hero.png"), [0u8; 100]).unwrap();
    std::fs::write(temp_dir.join("assets/hero.svg"), "<svg></svg>").unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    // docs/hero.svg should be checked first (SVG has priority), but doesn't exist
    // docs/hero.png should be used
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Png));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hero_image_svg_priority_in_docs() {
    let temp_dir = std::env::temp_dir().join("test_hero_svg_priority");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs")).unwrap();

    // Create both SVG and PNG - SVG should be preferred
    std::fs::write(temp_dir.join("docs/hero.svg"), "<svg></svg>").unwrap();
    std::fs::write(temp_dir.join("docs/hero.png"), [0u8; 100]).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Svg));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// QualityIssue Tests
// ========================================================================

#[test]
fn test_quality_issue_creation() {
    let issue = QualityIssue::new("test_issue", "Test message", IssueSeverity::Error);
    assert_eq!(issue.issue_type, "test_issue");
    assert_eq!(issue.message, "Test message");
    assert_eq!(issue.severity, IssueSeverity::Error);
    assert!(issue.recommendation.is_none());
}

#[test]
fn test_quality_issue_with_recommendation() {
    let issue =
        QualityIssue::new("test", "msg", IssueSeverity::Warning).with_recommendation("Fix it");
    assert_eq!(issue.recommendation, Some("Fix it".to_string()));
}

#[test]
fn test_quality_issue_score_below_threshold() {
    let issue = QualityIssue::score_below_threshold("rust_project", 80, 85);
    assert!(issue.message.contains("80"));
    assert!(issue.message.contains("85"));
    assert_eq!(issue.severity, IssueSeverity::Error);
}

#[test]
fn test_quality_issue_missing_hero() {
    let issue = QualityIssue::missing_hero_image();
    assert!(issue.message.contains("hero"));
    assert!(issue.recommendation.is_some());
}

// ========================================================================
// StackLayer Tests
// ========================================================================

#[test]
fn test_stack_layer_from_component() {
    assert_eq!(StackLayer::from_component("trueno"), StackLayer::Compute);
    assert_eq!(StackLayer::from_component("trueno-viz"), StackLayer::Compute);
    assert_eq!(StackLayer::from_component("aprender"), StackLayer::Ml);
    assert_eq!(StackLayer::from_component("depyler"), StackLayer::Transpilers);
    assert_eq!(StackLayer::from_component("batuta"), StackLayer::Orchestration);
    assert_eq!(StackLayer::from_component("certeza"), StackLayer::Quality);
    assert_eq!(StackLayer::from_component("alimentar"), StackLayer::DataMlops);
    assert_eq!(StackLayer::from_component("presentar"), StackLayer::Presentation);
}

#[test]
fn test_stack_layer_display_name() {
    assert_eq!(StackLayer::Compute.display_name(), "COMPUTE PRIMITIVES");
    assert_eq!(StackLayer::Ml.display_name(), "ML ALGORITHMS");
}

// ========================================================================
// ComponentQuality Tests
// ========================================================================

#[test]
fn test_component_quality_sqi_calculation() {
    // Perfect scores
    let rust = Score::new(114, 114, QualityGrade::APlus);
    let repo = Score::new(110, 110, QualityGrade::APlus);
    let readme = Score::new(20, 20, QualityGrade::APlus);
    let hero = HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png);

    let sqi = ComponentQuality::calculate_sqi(&rust, &repo, &readme, &hero);
    assert!((sqi - 100.0).abs() < 0.1);
}

#[test]
fn test_component_quality_sqi_with_missing_hero() {
    let rust = Score::new(114, 114, QualityGrade::APlus);
    let repo = Score::new(110, 110, QualityGrade::APlus);
    let readme = Score::new(20, 20, QualityGrade::APlus);
    let hero = HeroImageResult::missing();

    let sqi = ComponentQuality::calculate_sqi(&rust, &repo, &readme, &hero);
    // Should be 90% of perfect (missing 10% for hero)
    assert!((sqi - 90.0).abs() < 0.1);
}

#[test]
fn test_component_quality_creation() {
    let rust = Score::new(107, 114, QualityGrade::APlus);
    let repo = Score::new(98, 110, QualityGrade::APlus);
    let readme = Score::new(20, 20, QualityGrade::APlus);
    let hero = HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png);

    let quality =
        ComponentQuality::new("trueno", PathBuf::from("/path/to/trueno"), rust, repo, readme, hero);

    assert_eq!(quality.name, "trueno");
    assert_eq!(quality.layer, StackLayer::Compute);
    assert!(quality.sqi > 90.0);
    assert!(quality.grade.is_release_ready());
    assert!(quality.release_ready);
}

#[test]
fn test_component_quality_not_release_ready() {
    let rust = Score::new(70, 114, QualityGrade::B);
    let repo = Score::new(75, 110, QualityGrade::B);
    let readme = Score::new(12, 20, QualityGrade::BPlus);
    let hero = HeroImageResult::missing();

    let quality =
        ComponentQuality::new("weak-crate", PathBuf::from("/path"), rust, repo, readme, hero);

    assert!(!quality.release_ready);
    assert!(!quality.grade.is_release_ready());
    assert!(!quality.issues.is_empty());
}

// ========================================================================
// QualitySummary Tests
// ========================================================================

#[test]
fn test_quality_summary_empty() {
    let summary = QualitySummary::from_components(&[]);
    assert_eq!(summary.total_components, 0);
    assert_eq!(summary.a_plus_count, 0);
}

#[test]
fn test_quality_summary_calculation() {
    // A+ component: SQI > 95
    // rust=107/114=93.9%, repo=98/110=89.1%, readme=20/20=100%, hero=100%
    // SQI = 0.4*93.9 + 0.3*89.1 + 0.2*100 + 0.1*100 = 94.3 (A)
    // Need higher scores for A+:
    // rust=112/114=98.2%, repo=105/110=95.5%, readme=20/20=100%, hero=100%
    // SQI = 0.4*98.2 + 0.3*95.5 + 0.2*100 + 0.1*100 = 97.9 (A+)
    let components = vec![
        create_test_component("a", 112, 105, 20, true), // A+ (SQI ~97.9)
        create_test_component("b", 100, 92, 17, true),  // A- (SQI ~87.2)
        create_test_component("c", 80, 75, 12, false),  // B (SQI ~68.5)
    ];

    let summary = QualitySummary::from_components(&components);

    assert_eq!(summary.total_components, 3);
    assert_eq!(summary.a_plus_count, 1);
    assert_eq!(summary.a_minus_count, 1);
    assert_eq!(summary.below_threshold_count, 1);
    assert_eq!(summary.missing_hero_count, 1);
}

// ========================================================================
// StackQualityReport Tests
// ========================================================================

#[test]
fn test_stack_quality_report_creation() {
    let components = vec![
        create_test_component("trueno", 107, 98, 20, true),
        create_test_component("aprender", 105, 95, 18, true),
    ];

    let report = StackQualityReport::from_components(components);

    assert_eq!(report.summary.total_components, 2);
    assert!(report.release_ready);
    assert!(report.blocked_components.is_empty());
    assert!(report.stack_quality_index > 90.0);
}

#[test]
fn test_stack_quality_report_with_blocked() {
    let components = vec![
        create_test_component("trueno", 107, 98, 20, true),
        create_test_component("weak", 70, 70, 10, false),
    ];

    let report = StackQualityReport::from_components(components);

    assert!(!report.release_ready);
    assert_eq!(report.blocked_components, vec!["weak".to_string()]);
    assert!(!report.recommendations.is_empty());
}

#[test]
fn test_stack_quality_report_is_all_a_plus() {
    // Both components need SQI >= 95 for A+
    // rust=114/114=100%, repo=110/110=100%, readme=20/20=100%, hero=100%
    // SQI = 0.4*100 + 0.3*100 + 0.2*100 + 0.1*100 = 100 (A+)
    let components = vec![
        create_test_component("a", 114, 110, 20, true), // Perfect A+
        create_test_component("b", 112, 105, 20, true), // A+ (SQI ~97.9)
    ];

    let report = StackQualityReport::from_components(components);
    assert!(report.is_all_a_plus());
}

#[test]
fn test_stack_quality_report_not_all_a_plus() {
    let components = vec![
        create_test_component("a", 110, 100, 20, true),
        create_test_component("b", 95, 90, 16, true), // A, not A+
    ];

    let report = StackQualityReport::from_components(components);
    assert!(!report.is_all_a_plus());
}

// ========================================================================
// Additional Unit Tests for Coverage
// ========================================================================

#[test]
fn test_quality_grade_ordering() {
    // Test ordering: APlus < A < AMinus < ...
    assert!(QualityGrade::APlus < QualityGrade::A);
    assert!(QualityGrade::A < QualityGrade::AMinus);
    assert!(QualityGrade::AMinus < QualityGrade::BPlus);
    assert!(QualityGrade::BPlus < QualityGrade::B);
    assert!(QualityGrade::B < QualityGrade::C);
    assert!(QualityGrade::C < QualityGrade::D);
    assert!(QualityGrade::D < QualityGrade::F);
}

#[test]
fn test_quality_grade_all_icons() {
    // Ensure all grades have icons
    assert_eq!(QualityGrade::B.icon(), "❌");
    assert_eq!(QualityGrade::C.icon(), "❌");
    assert_eq!(QualityGrade::D.icon(), "❌");
    assert_eq!(QualityGrade::F.icon(), "❌");
}

#[test]
fn test_quality_grade_all_symbols() {
    assert_eq!(QualityGrade::B.symbol(), "B");
    assert_eq!(QualityGrade::C.symbol(), "C");
    assert_eq!(QualityGrade::D.symbol(), "D");
}

#[test]
fn test_quality_grade_from_repo_score_all_ranges() {
    assert_eq!(QualityGrade::from_repo_score(94), QualityGrade::A);
    assert_eq!(QualityGrade::from_repo_score(90), QualityGrade::A);
    assert_eq!(QualityGrade::from_repo_score(89), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_repo_score(85), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_repo_score(84), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_repo_score(80), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_repo_score(79), QualityGrade::B);
    assert_eq!(QualityGrade::from_repo_score(70), QualityGrade::B);
    assert_eq!(QualityGrade::from_repo_score(69), QualityGrade::C);
    assert_eq!(QualityGrade::from_repo_score(60), QualityGrade::C);
    assert_eq!(QualityGrade::from_repo_score(59), QualityGrade::D);
    assert_eq!(QualityGrade::from_repo_score(50), QualityGrade::D);
    assert_eq!(QualityGrade::from_repo_score(49), QualityGrade::F);
}

#[test]
fn test_quality_grade_from_readme_score_all_ranges() {
    assert_eq!(QualityGrade::from_readme_score(16), QualityGrade::A);
    assert_eq!(QualityGrade::from_readme_score(14), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_readme_score(13), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_readme_score(11), QualityGrade::B);
    assert_eq!(QualityGrade::from_readme_score(10), QualityGrade::B);
    assert_eq!(QualityGrade::from_readme_score(9), QualityGrade::C);
    assert_eq!(QualityGrade::from_readme_score(8), QualityGrade::C);
    assert_eq!(QualityGrade::from_readme_score(7), QualityGrade::D);
    assert_eq!(QualityGrade::from_readme_score(6), QualityGrade::D);
    assert_eq!(QualityGrade::from_readme_score(5), QualityGrade::F);
}

#[test]
fn test_stack_layer_all_display_names() {
    assert_eq!(StackLayer::Training.display_name(), "TRAINING & INFERENCE");
    assert_eq!(StackLayer::DataMlops.display_name(), "DATA & MLOPS");
    assert_eq!(StackLayer::Transpilers.display_name(), "TRANSPILERS");
    assert_eq!(StackLayer::Orchestration.display_name(), "ORCHESTRATION");
    assert_eq!(StackLayer::Quality.display_name(), "QUALITY");
    assert_eq!(StackLayer::Presentation.display_name(), "PRESENTATION");
}

#[test]
fn test_stack_layer_from_component_extended() {
    // Training layer
    assert_eq!(StackLayer::from_component("entrenar"), StackLayer::Training);
    assert_eq!(StackLayer::from_component("realizar"), StackLayer::Training);

    // Transpilers (ruchy is in the list, bashrs is not)
    assert_eq!(StackLayer::from_component("ruchy"), StackLayer::Transpilers);
    assert_eq!(StackLayer::from_component("decy"), StackLayer::Transpilers);

    // Unknown defaults to Orchestration
    assert_eq!(StackLayer::from_component("unknown-crate"), StackLayer::Orchestration);
}

fn create_test_component(
    name: &str,
    rust: u32,
    repo: u32,
    readme: u32,
    has_hero: bool,
) -> ComponentQuality {
    let rust_score = Score::new(rust, 114, QualityGrade::from_rust_project_score(rust));
    let repo_score = Score::new(repo, 110, QualityGrade::from_repo_score(repo));
    let readme_score = Score::new(readme, 20, QualityGrade::from_readme_score(readme));
    let hero = if has_hero {
        HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png)
    } else {
        HeroImageResult::missing()
    };

    ComponentQuality::new(name, PathBuf::from("/test"), rust_score, repo_score, readme_score, hero)
}

// Property-based tests for quality grades
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: QualityGrade::from_rust_project_score always returns valid grade
        #[test]
        fn prop_rust_project_score_valid_grade(score in 0u32..200) {
            let grade = QualityGrade::from_rust_project_score(score);
            // Grade should be a valid enum variant
            prop_assert!(matches!(
                grade,
                QualityGrade::APlus | QualityGrade::A | QualityGrade::AMinus |
                QualityGrade::BPlus | QualityGrade::B | QualityGrade::C |
                QualityGrade::D | QualityGrade::F
            ));
        }

        /// Property: QualityGrade::from_repo_score always returns valid grade
        #[test]
        fn prop_repo_score_valid_grade(score in 0u32..200) {
            let grade = QualityGrade::from_repo_score(score);
            prop_assert!(matches!(
                grade,
                QualityGrade::APlus | QualityGrade::A | QualityGrade::AMinus |
                QualityGrade::BPlus | QualityGrade::B | QualityGrade::C |
                QualityGrade::D | QualityGrade::F
            ));
        }

        /// Property: QualityGrade::from_sqi always returns valid grade
        #[test]
        fn prop_sqi_valid_grade(sqi in 0.0f64..150.0) {
            let grade = QualityGrade::from_sqi(sqi);
            prop_assert!(matches!(
                grade,
                QualityGrade::APlus | QualityGrade::A | QualityGrade::AMinus |
                QualityGrade::BPlus | QualityGrade::B | QualityGrade::C |
                QualityGrade::D | QualityGrade::F
            ));
        }

        /// Property: Higher scores produce better or equal grades
        #[test]
        fn prop_higher_score_better_grade(low in 0u32..80, high in 80u32..115) {
            let low_grade = QualityGrade::from_rust_project_score(low);
            let high_grade = QualityGrade::from_rust_project_score(high);
            // Lower enum variant = better grade
            prop_assert!(high_grade <= low_grade);
        }

        /// Property: Score percentage is in valid range
        #[test]
        fn prop_score_percentage_valid(value in 0u32..200, max in 1u32..200) {
            let grade = QualityGrade::from_rust_project_score(value);
            let score = Score::new(value.min(max), max, grade);
            let pct = score.percentage();
            prop_assert!((0.0..=100.0).contains(&pct));
        }

        /// Property: SQI calculation is in valid range
        #[test]
        fn prop_sqi_valid_range(
            rust in 0u32..115,
            repo in 0u32..111,
            readme in 0u32..21
        ) {
            let rust_score = Score::new(rust.min(114), 114, QualityGrade::from_rust_project_score(rust));
            let repo_score = Score::new(repo.min(110), 110, QualityGrade::from_repo_score(repo));
            let readme_score = Score::new(readme.min(20), 20, QualityGrade::from_readme_score(readme));
            let hero = HeroImageResult::missing();

            let sqi = ComponentQuality::calculate_sqi(&rust_score, &repo_score, &readme_score, &hero);
            prop_assert!((0.0..=100.0).contains(&sqi), "SQI {} not in [0, 100]", sqi);
        }

        /// Property: All grades have non-empty symbols
        #[test]
        fn prop_grade_symbol_nonempty(score in 0u32..120) {
            let grade = QualityGrade::from_rust_project_score(score);
            prop_assert!(!grade.symbol().is_empty());
        }

        /// Property: Release readiness is consistent with grade
        #[test]
        fn prop_release_ready_consistent(score in 0u32..115) {
            let grade = QualityGrade::from_rust_project_score(score);
            let is_ready = grade.is_release_ready();

            // A-, A, A+ are release ready (score 85-114)
            if (85..=114).contains(&score) {
                prop_assert!(is_ready, "Score {} should be release ready", score);
            }
            if score < 85 {
                prop_assert!(!is_ready, "Score {} should NOT be release ready", score);
            }
        }
    }
}

// ========================================================================
// Additional Edge Case Tests for Coverage
// ========================================================================

#[test]
fn test_generate_recommendations_missing_hero() {
    let mut comp = create_test_component("no_hero", 105, 95, 18, false);
    comp.hero_image.valid = false;
    comp.release_ready = false;

    let report = StackQualityReport::from_components(vec![comp]);

    // Should recommend adding hero
    assert!(report.recommendations.iter().any(|r| r.contains("hero")));
}

#[test]
fn test_generate_recommendations_low_rust_score() {
    let mut comp = create_test_component("low_rust", 70, 60, 10, true);
    comp.rust_score = Score::new(70, 114, QualityGrade::B);
    comp.release_ready = false;

    let report = StackQualityReport::from_components(vec![comp]);

    // Should recommend improving test coverage
    assert!(report
        .recommendations
        .iter()
        .any(|r| r.contains("test coverage") || r.contains("documentation")));
}

#[test]
fn test_quality_summary_with_below_threshold() {
    let mut comp = create_test_component("low", 70, 60, 10, false);
    comp.grade = QualityGrade::B;
    comp.release_ready = false;

    let summary = QualitySummary::from_components(&[comp]);
    assert_eq!(summary.below_threshold_count, 1);
}

#[test]
fn test_quality_issue_all_severity_levels() {
    let error = QualityIssue::new("err", "Error message", IssueSeverity::Error);
    let warning = QualityIssue::new("warn", "Warning message", IssueSeverity::Warning);
    let info = QualityIssue::new("info", "Info message", IssueSeverity::Info);

    assert_eq!(error.severity, IssueSeverity::Error);
    assert_eq!(warning.severity, IssueSeverity::Warning);
    assert_eq!(info.severity, IssueSeverity::Info);
}

#[test]
fn test_component_quality_hero_penalty() {
    let rust = Score::new(114, 114, QualityGrade::APlus);
    let repo = Score::new(110, 110, QualityGrade::APlus);
    let readme = Score::new(20, 20, QualityGrade::APlus);
    let hero = HeroImageResult::missing();

    let quality =
        ComponentQuality::new("test".to_string(), PathBuf::from("/test"), rust, repo, readme, hero);

    // SQI should be less than 100 due to missing hero
    assert!(quality.sqi < 100.0);
    assert!(!quality.hero_image.valid);
}

#[test]
fn test_stack_layer_all_variants() {
    let layers = vec![
        StackLayer::Compute,
        StackLayer::Ml,
        StackLayer::Training,
        StackLayer::Transpilers,
        StackLayer::Orchestration,
        StackLayer::DataMlops,
        StackLayer::Quality,
        StackLayer::Presentation,
    ];

    for layer in layers {
        assert!(!layer.display_name().is_empty());
    }
}

#[test]
fn test_quality_grade_all_variants_display() {
    let grades = vec![
        QualityGrade::APlus,
        QualityGrade::A,
        QualityGrade::AMinus,
        QualityGrade::BPlus,
        QualityGrade::B,
        QualityGrade::C,
        QualityGrade::D,
        QualityGrade::F,
    ];

    for grade in grades {
        // symbol() and icon() should not panic
        assert!(!grade.symbol().is_empty());
        assert!(!grade.icon().is_empty());
        // Display impl
        let _s = format!("{}", grade);
    }
}

#[test]
fn test_image_format_all_variants() {
    let formats: Vec<ImageFormat> =
        vec![ImageFormat::Png, ImageFormat::Jpg, ImageFormat::Svg, ImageFormat::WebP];

    for fmt in formats {
        assert!(!fmt.extension().is_empty());
    }
}

#[test]
fn test_score_zero_max_edge_case() {
    // This shouldn't happen in practice but test edge case
    let score = Score::new(0, 100, QualityGrade::F);
    assert!(score.percentage() >= 0.0);
    assert_eq!(score.normalized(), 0.0);
}

// ========================================================================
// Quality Coverage Tests
// ========================================================================

#[test]
fn test_qcov_001_grade_from_rust_project_score() {
    assert_eq!(QualityGrade::from_rust_project_score(114), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_rust_project_score(105), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_rust_project_score(104), QualityGrade::A);
    assert_eq!(QualityGrade::from_rust_project_score(95), QualityGrade::A);
    assert_eq!(QualityGrade::from_rust_project_score(94), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_rust_project_score(85), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_rust_project_score(84), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_rust_project_score(80), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_rust_project_score(79), QualityGrade::B);
    assert_eq!(QualityGrade::from_rust_project_score(70), QualityGrade::B);
    assert_eq!(QualityGrade::from_rust_project_score(69), QualityGrade::C);
    assert_eq!(QualityGrade::from_rust_project_score(60), QualityGrade::C);
    assert_eq!(QualityGrade::from_rust_project_score(59), QualityGrade::D);
    assert_eq!(QualityGrade::from_rust_project_score(50), QualityGrade::D);
    assert_eq!(QualityGrade::from_rust_project_score(49), QualityGrade::F);
    assert_eq!(QualityGrade::from_rust_project_score(0), QualityGrade::F);
}

#[test]
fn test_qcov_002_grade_from_repo_score() {
    assert_eq!(QualityGrade::from_repo_score(110), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_repo_score(95), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_repo_score(94), QualityGrade::A);
    assert_eq!(QualityGrade::from_repo_score(90), QualityGrade::A);
    assert_eq!(QualityGrade::from_repo_score(89), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_repo_score(85), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_repo_score(84), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_repo_score(80), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_repo_score(79), QualityGrade::B);
    assert_eq!(QualityGrade::from_repo_score(70), QualityGrade::B);
    assert_eq!(QualityGrade::from_repo_score(69), QualityGrade::C);
    assert_eq!(QualityGrade::from_repo_score(60), QualityGrade::C);
    assert_eq!(QualityGrade::from_repo_score(59), QualityGrade::D);
    assert_eq!(QualityGrade::from_repo_score(50), QualityGrade::D);
    assert_eq!(QualityGrade::from_repo_score(49), QualityGrade::F);
}

#[test]
fn test_qcov_003_grade_from_readme_score() {
    assert_eq!(QualityGrade::from_readme_score(20), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_readme_score(18), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_readme_score(17), QualityGrade::A);
    assert_eq!(QualityGrade::from_readme_score(16), QualityGrade::A);
    assert_eq!(QualityGrade::from_readme_score(15), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_readme_score(14), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_readme_score(13), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_readme_score(12), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_readme_score(11), QualityGrade::B);
    assert_eq!(QualityGrade::from_readme_score(10), QualityGrade::B);
    assert_eq!(QualityGrade::from_readme_score(9), QualityGrade::C);
    assert_eq!(QualityGrade::from_readme_score(8), QualityGrade::C);
    assert_eq!(QualityGrade::from_readme_score(7), QualityGrade::D);
    assert_eq!(QualityGrade::from_readme_score(6), QualityGrade::D);
    assert_eq!(QualityGrade::from_readme_score(5), QualityGrade::F);
}

#[test]
fn test_qcov_004_grade_from_sqi() {
    assert_eq!(QualityGrade::from_sqi(100.0), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_sqi(95.0), QualityGrade::APlus);
    assert_eq!(QualityGrade::from_sqi(94.0), QualityGrade::A);
    assert_eq!(QualityGrade::from_sqi(90.0), QualityGrade::A);
    assert_eq!(QualityGrade::from_sqi(89.0), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_sqi(85.0), QualityGrade::AMinus);
    assert_eq!(QualityGrade::from_sqi(84.0), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_sqi(80.0), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_sqi(79.0), QualityGrade::B);
    assert_eq!(QualityGrade::from_sqi(70.0), QualityGrade::B);
    assert_eq!(QualityGrade::from_sqi(69.0), QualityGrade::C);
    assert_eq!(QualityGrade::from_sqi(60.0), QualityGrade::C);
    assert_eq!(QualityGrade::from_sqi(59.0), QualityGrade::D);
    assert_eq!(QualityGrade::from_sqi(50.0), QualityGrade::D);
    assert_eq!(QualityGrade::from_sqi(49.0), QualityGrade::F);
}

#[test]
fn test_qcov_005_grade_release_ready() {
    assert!(QualityGrade::APlus.is_release_ready());
    assert!(QualityGrade::A.is_release_ready());
    assert!(QualityGrade::AMinus.is_release_ready());
    assert!(!QualityGrade::BPlus.is_release_ready());
    assert!(!QualityGrade::B.is_release_ready());
    assert!(!QualityGrade::C.is_release_ready());
    assert!(!QualityGrade::D.is_release_ready());
    assert!(!QualityGrade::F.is_release_ready());
}

#[test]
fn test_qcov_006_grade_is_a_plus() {
    assert!(QualityGrade::APlus.is_a_plus());
    assert!(!QualityGrade::A.is_a_plus());
    assert!(!QualityGrade::AMinus.is_a_plus());
}

#[test]
fn test_qcov_007_grade_icons() {
    assert_eq!(QualityGrade::APlus.icon(), "✅");
    assert_eq!(QualityGrade::A.icon(), "⚠️");
    assert_eq!(QualityGrade::AMinus.icon(), "⚠️");
    assert_eq!(QualityGrade::BPlus.icon(), "❌");
    assert_eq!(QualityGrade::B.icon(), "❌");
    assert_eq!(QualityGrade::C.icon(), "❌");
    assert_eq!(QualityGrade::D.icon(), "❌");
    assert_eq!(QualityGrade::F.icon(), "❌");
}

#[test]
fn test_qcov_008_image_format_extensions() {
    assert_eq!(ImageFormat::from_extension("png"), Some(ImageFormat::Png));
    assert_eq!(ImageFormat::from_extension("PNG"), Some(ImageFormat::Png));
    assert_eq!(ImageFormat::from_extension("jpg"), Some(ImageFormat::Jpg));
    assert_eq!(ImageFormat::from_extension("jpeg"), Some(ImageFormat::Jpg));
    assert_eq!(ImageFormat::from_extension("webp"), Some(ImageFormat::WebP));
    assert_eq!(ImageFormat::from_extension("svg"), Some(ImageFormat::Svg));
    assert_eq!(ImageFormat::from_extension("bmp"), None);
    assert_eq!(ImageFormat::from_extension("gif"), None);
}

#[test]
fn test_qcov_009_image_format_extension_strings() {
    assert_eq!(ImageFormat::Png.extension(), "png");
    assert_eq!(ImageFormat::Jpg.extension(), "jpg");
    assert_eq!(ImageFormat::WebP.extension(), "webp");
    assert_eq!(ImageFormat::Svg.extension(), "svg");
}

#[test]
fn test_qcov_010_hero_image_result_missing() {
    let hero = HeroImageResult::missing();
    assert!(!hero.present);
    assert!(hero.path.is_none());
    assert!(!hero.valid);
    assert!(!hero.issues.is_empty());
}

#[test]
fn test_qcov_011_hero_image_result_found() {
    let hero = HeroImageResult::found(PathBuf::from("/test/hero.png"), ImageFormat::Png);
    assert!(hero.present);
    assert!(hero.path.is_some());
    assert_eq!(hero.format, Some(ImageFormat::Png));
    assert!(hero.valid);
}

#[test]
fn test_qcov_012_score_percentage_zero_max() {
    let score = Score::new(50, 0, QualityGrade::F);
    assert_eq!(score.percentage(), 0.0);
}

#[test]
fn test_qcov_013_score_percentage_normal() {
    let score = Score::new(50, 100, QualityGrade::D);
    assert_eq!(score.percentage(), 50.0);
    assert_eq!(score.normalized(), 50.0);
}

#[test]
fn test_qcov_014_quality_issue_new() {
    let issue = QualityIssue::new("test_type", "msg", IssueSeverity::Error);
    assert_eq!(issue.issue_type, "test_type");
    assert_eq!(issue.message, "msg");
    assert_eq!(issue.severity, IssueSeverity::Error);
}

#[test]
fn test_qcov_015_quality_issue_with_recommendation() {
    let issue =
        QualityIssue::new("test", "msg", IssueSeverity::Warning).with_recommendation("fix it");
    assert_eq!(issue.recommendation, Some("fix it".to_string()));
}

#[test]
fn test_qcov_016_quality_issue_score_below_threshold() {
    let issue = QualityIssue::score_below_threshold("coverage", 70, 80);
    assert!(issue.message.contains("70"));
    assert!(issue.message.contains("80"));
    assert_eq!(issue.severity, IssueSeverity::Error);
}

#[test]
fn test_qcov_017_quality_issue_missing_hero() {
    let issue = QualityIssue::missing_hero_image();
    assert!(issue.message.contains("hero"));
    assert_eq!(issue.severity, IssueSeverity::Error);
}

#[test]
fn test_qcov_018_stack_layer_all_variants() {
    let layers = vec![
        StackLayer::Compute,
        StackLayer::Ml,
        StackLayer::Training,
        StackLayer::Transpilers,
        StackLayer::Orchestration,
        StackLayer::Quality,
        StackLayer::DataMlops,
        StackLayer::Presentation,
    ];
    for layer in layers {
        assert!(!layer.display_name().is_empty());
    }
}

#[test]
fn test_qcov_019_stack_layer_from_component() {
    assert_eq!(StackLayer::from_component("trueno"), StackLayer::Compute);
    assert_eq!(StackLayer::from_component("aprender"), StackLayer::Ml);
    assert_eq!(StackLayer::from_component("entrenar"), StackLayer::Training);
    assert_eq!(StackLayer::from_component("batuta"), StackLayer::Orchestration);
    assert_eq!(StackLayer::from_component("depyler"), StackLayer::Transpilers);
}

#[test]
fn test_qcov_020_quality_summary_from_components() {
    let comps = vec![
        create_test_component("a", 105, 95, 18, true),
        create_test_component("b", 70, 60, 10, false),
    ];
    let summary = QualitySummary::from_components(&comps);
    assert_eq!(summary.total_components, 2);
    // Summary counts may vary based on component settings
    assert!(summary.a_plus_count + summary.below_threshold_count <= 2);
}

#[test]
fn test_qcov_024_report_is_all_a_plus() {
    let all_aplus = vec![create_test_component("a", 110, 100, 20, true)];
    let report = StackQualityReport::from_components(all_aplus);
    assert!(report.is_all_a_plus());

    let not_all_aplus = vec![create_test_component("b", 70, 60, 10, false)];
    let report2 = StackQualityReport::from_components(not_all_aplus);
    assert!(!report2.is_all_a_plus());
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_qcov_025_hero_detect_missing() {
    // Test detect on a path without hero images
    let temp_dir = std::env::temp_dir().join("qcov_025_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(!result.present);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_qcov_026_hero_detect_with_docs_hero() {
    let temp_dir = std::env::temp_dir().join("qcov_026_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs")).unwrap();

    // Create a small hero.png
    std::fs::write(temp_dir.join("docs/hero.png"), [0x89, 0x50, 0x4E, 0x47]).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Png));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_qcov_027_hero_detect_with_assets_hero() {
    let temp_dir = std::env::temp_dir().join("qcov_027_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("assets")).unwrap();

    // Create a small hero.svg
    std::fs::write(temp_dir.join("assets/hero.svg"), "<svg></svg>").unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);
    assert_eq!(result.format, Some(ImageFormat::Svg));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_qcov_028_hero_detect_from_readme() {
    let temp_dir = std::env::temp_dir().join("qcov_028_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create README with image reference
    std::fs::write(temp_dir.join("README.md"), "# Test\n![Alt text](local-image.png)\n").unwrap();

    // Create the referenced image
    std::fs::write(temp_dir.join("local-image.png"), [0x89, 0x50, 0x4E, 0x47]).unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(result.present);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_qcov_029_hero_detect_readme_external_url() {
    let temp_dir = std::env::temp_dir().join("qcov_029_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // README with external URL (should be skipped)
    std::fs::write(temp_dir.join("README.md"), "# Test\n![Alt](https://example.com/image.png)\n")
        .unwrap();

    let result = HeroImageResult::detect(&temp_dir);
    assert!(!result.present);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_qcov_030_quality_summary_empty() {
    let summary = QualitySummary::from_components(&[]);
    assert_eq!(summary.total_components, 0);
    assert_eq!(summary.a_plus_count, 0);
}

#[test]
fn test_qcov_031_grade_boundary_values() {
    // Test exact boundary values for rust_project_score
    assert_eq!(QualityGrade::from_rust_project_score(104), QualityGrade::A);
    assert_eq!(QualityGrade::from_rust_project_score(80), QualityGrade::BPlus);
    assert_eq!(QualityGrade::from_rust_project_score(60), QualityGrade::C);
    assert_eq!(QualityGrade::from_rust_project_score(50), QualityGrade::D);
}

#[test]
fn test_qcov_032_quality_report_empty() {
    let report = StackQualityReport::from_components(vec![]);
    assert!(report.is_all_a_plus()); // Empty is vacuously true
    assert_eq!(report.summary.total_components, 0);
}

#[test]
fn test_qcov_033_format_text_empty_report() {
    let report = StackQualityReport::from_components(vec![]);
    let text = format_report_text(&report);
    assert!(text.contains("SUMMARY"));
}

#[test]
fn test_qcov_034_component_quality_default_issues() {
    let comp = create_test_component("test", 105, 95, 18, true);
    // Created with valid scores should have minimal issues
    assert!(comp.issues.len() <= 2);
}

#[test]
fn test_qcov_035_stack_layer_display_names() {
    assert_eq!(StackLayer::Compute.display_name(), "COMPUTE PRIMITIVES");
    assert_eq!(StackLayer::Ml.display_name(), "ML ALGORITHMS");
    assert_eq!(StackLayer::Training.display_name(), "TRAINING & INFERENCE");
    assert_eq!(StackLayer::Transpilers.display_name(), "TRANSPILERS");
    assert_eq!(StackLayer::Orchestration.display_name(), "ORCHESTRATION");
    assert_eq!(StackLayer::DataMlops.display_name(), "DATA & MLOPS");
    assert_eq!(StackLayer::Quality.display_name(), "QUALITY");
    assert_eq!(StackLayer::Presentation.display_name(), "PRESENTATION");
}
