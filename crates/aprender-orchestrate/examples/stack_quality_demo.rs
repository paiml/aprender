//! Stack Quality Matrix Demo
//!
//! Demonstrates the quality enforcement system for PAIML stack components.
//!
//! ## Quality Dimensions
//!
//! - **Rust Project Score** (0-114): Code quality via PMAT
//! - **Repository Score** (0-110): CI/CD, security, community health
//! - **README Score** (0-20): Documentation completeness
//! - **Hero Image**: Visual branding presence (SVG preferred)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example stack_quality_demo --features native
//! ```
//!
//! ## Per Stack Quality Matrix Specification

#[cfg(feature = "native")]
use batuta::{
    ComponentQuality, HeroImageResult, ImageFormat, QualityGrade, QualityStackLayer as StackLayer,
    Score,
};

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    println!("===============================================================");
    println!("     Stack Quality Matrix - Demo");
    println!("===============================================================\n");

    // =========================================================================
    // Phase 1: Quality Grades
    // =========================================================================
    println!("+-------------------------------------------------------------+");
    println!("| Phase 1: Quality Grade System                               |");
    println!("+-------------------------------------------------------------+\n");

    demo_quality_grades();

    // =========================================================================
    // Phase 2: Score Calculations
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 2: Score Calculations                                 |");
    println!("+-------------------------------------------------------------+\n");

    demo_score_calculations();

    // =========================================================================
    // Phase 3: Stack Quality Index (SQI)
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 3: Stack Quality Index (SQI)                          |");
    println!("+-------------------------------------------------------------+\n");

    demo_sqi_calculation();

    // =========================================================================
    // Phase 4: Hero Image Detection
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 4: Hero Image Detection                               |");
    println!("+-------------------------------------------------------------+\n");

    demo_hero_image();

    // =========================================================================
    // Phase 5: Component Quality Report
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 5: Component Quality Report                           |");
    println!("+-------------------------------------------------------------+\n");

    demo_component_quality();

    // =========================================================================
    // Phase 6: Stack Layers
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 6: Stack Layers                                       |");
    println!("+-------------------------------------------------------------+\n");

    demo_stack_layers();

    // =========================================================================
    // Phase 7: Quality Gate Enforcement
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 7: Quality Gate Enforcement                           |");
    println!("+-------------------------------------------------------------+\n");

    demo_quality_gate();

    println!("\n===============================================================");
    println!("     Stack Quality Matrix demo completed!");
    println!("===============================================================\n");

    Ok(())
}

#[cfg(feature = "native")]
fn demo_quality_grades() {
    println!("  Quality Grade Thresholds:");
    println!();
    println!("  Rust Project Score (max 114):");
    println!("    A+ : 105-114  {}", QualityGrade::from_rust_project_score(110).icon());
    println!("    A  : 95-104   {}", QualityGrade::from_rust_project_score(100).icon());
    println!("    A- : 85-94    {}", QualityGrade::from_rust_project_score(90).icon());
    println!("    B+ : 80-84    {}", QualityGrade::from_rust_project_score(82).icon());
    println!("    B  : 70-79    {}", QualityGrade::from_rust_project_score(75).icon());
    println!("    C  : 60-69    {}", QualityGrade::from_rust_project_score(65).icon());
    println!("    D  : 50-59    {}", QualityGrade::from_rust_project_score(55).icon());
    println!("    F  : 0-49     {}", QualityGrade::from_rust_project_score(40).icon());
    println!();
    println!("  Repository Score (max 110):");
    println!("    A+ : 95-110   {}", QualityGrade::from_repo_score(100).icon());
    println!("    A  : 90-94    {}", QualityGrade::from_repo_score(92).icon());
    println!();
    println!("  README Score (max 20):");
    println!("    A+ : 18-20    {}", QualityGrade::from_readme_score(19).icon());
    println!("    A  : 16-17    {}", QualityGrade::from_readme_score(17).icon());
}

#[cfg(feature = "native")]
fn demo_score_calculations() {
    // Create sample scores
    let rust_score = Score::new(108, 114, QualityGrade::from_rust_project_score(108));
    let repo_score = Score::new(96, 110, QualityGrade::from_repo_score(96));
    let readme_score = Score::new(19, 20, QualityGrade::from_readme_score(19));

    println!("  Sample Scores:");
    println!();
    println!(
        "  Rust Project: {}/{} ({:.1}%) - Grade: {}",
        rust_score.value,
        rust_score.max,
        rust_score.percentage(),
        rust_score.grade.symbol()
    );
    println!(
        "  Repository:   {}/{} ({:.1}%) - Grade: {}",
        repo_score.value,
        repo_score.max,
        repo_score.percentage(),
        repo_score.grade.symbol()
    );
    println!(
        "  README:       {}/{} ({:.1}%) - Grade: {}",
        readme_score.value,
        readme_score.max,
        readme_score.percentage(),
        readme_score.grade.symbol()
    );
    println!();
    println!("  Normalized scores (0-100 scale):");
    println!("    Rust:   {:.2}", rust_score.normalized());
    println!("    Repo:   {:.2}", repo_score.normalized());
    println!("    README: {:.2}", readme_score.normalized());
}

#[cfg(feature = "native")]
fn demo_sqi_calculation() {
    println!("  Stack Quality Index Formula:");
    println!();
    println!("  SQI = 0.40 * Rust + 0.30 * Repo + 0.20 * README + 0.10 * Hero");
    println!();

    // Example calculation
    let rust_pct = 94.74; // 108/114
    let repo_pct = 87.27; // 96/110
    let readme_pct = 95.0; // 19/20
    let hero_pct = 100.0; // Present

    let sqi = 0.40 * rust_pct + 0.30 * repo_pct + 0.20 * readme_pct + 0.10 * hero_pct;
    let grade = QualityGrade::from_sqi(sqi);

    println!("  Example Calculation:");
    println!("    Rust Score:   {:.2}% (weight: 0.40)", rust_pct);
    println!("    Repo Score:   {:.2}% (weight: 0.30)", repo_pct);
    println!("    README Score: {:.2}% (weight: 0.20)", readme_pct);
    println!("    Hero Image:   {:.2}% (weight: 0.10)", hero_pct);
    println!();
    println!(
        "  SQI = 0.40*{:.2} + 0.30*{:.2} + 0.20*{:.2} + 0.10*{:.2}",
        rust_pct, repo_pct, readme_pct, hero_pct
    );
    println!("  SQI = {:.2}", sqi);
    println!();
    println!("  Final Grade: {} {}", grade.symbol(), grade.icon());
    println!("  Release Ready: {}", if grade.is_release_ready() { "Yes" } else { "No" });
}

#[cfg(feature = "native")]
fn demo_hero_image() {
    println!("  Hero Image Detection Priority:");
    println!();
    println!("  1. docs/hero.svg (preferred - scalable)");
    println!("  2. docs/hero.png");
    println!("  3. docs/hero.jpg");
    println!("  4. docs/hero.webp");
    println!("  5. assets/hero.* (same order)");
    println!("  6. First image in README.md");
    println!();

    // Detect in current directory
    let result = HeroImageResult::detect(std::path::Path::new("."));
    println!("  Current Directory Scan:");
    println!("    Present: {}", if result.present { "Yes" } else { "No" });
    if let Some(ref path) = result.path {
        println!("    Path: {}", path.display());
    }
    if let Some(format) = result.format {
        println!("    Format: {:?}", format);
    }
    println!("    Valid: {}", if result.valid { "Yes" } else { "No" });
    if !result.issues.is_empty() {
        println!("    Issues:");
        for issue in &result.issues {
            println!("      - {}", issue);
        }
    }
}

#[cfg(feature = "native")]
fn demo_component_quality() {
    // Create a sample component quality report
    let component = ComponentQuality {
        name: "trueno".to_string(),
        path: std::path::PathBuf::from("../trueno"),
        layer: StackLayer::Compute,
        rust_score: Score::new(108, 114, QualityGrade::from_rust_project_score(108)),
        repo_score: Score::new(98, 110, QualityGrade::from_repo_score(98)),
        readme_score: Score::new(20, 20, QualityGrade::from_readme_score(20)),
        hero_image: HeroImageResult {
            present: true,
            path: Some(std::path::PathBuf::from("docs/hero.svg")),
            format: Some(ImageFormat::Svg),
            dimensions: None,
            file_size: Some(15_000),
            valid: true,
            issues: vec![],
        },
        sqi: 96.8,
        grade: QualityGrade::APlus,
        issues: vec![],
        release_ready: true,
    };

    println!("  Sample Component Report: {}", component.name);
    println!();
    println!("  Layer: {}", component.layer.display_name());
    println!(
        "  Rust Project: {}/{} ({})",
        component.rust_score.value,
        component.rust_score.max,
        component.rust_score.grade.symbol()
    );
    println!(
        "  Repository:   {}/{} ({})",
        component.repo_score.value,
        component.repo_score.max,
        component.repo_score.grade.symbol()
    );
    println!(
        "  README:       {}/{} ({})",
        component.readme_score.value,
        component.readme_score.max,
        component.readme_score.grade.symbol()
    );
    println!(
        "  Hero Image:   {} ({})",
        if component.hero_image.present { "Present" } else { "Missing" },
        component
            .hero_image
            .format
            .map(|f| format!("{:?}", f))
            .unwrap_or_else(|| "N/A".to_string())
    );
    println!();
    println!("  Stack Quality Index: {:.1}%", component.sqi);
    println!("  Overall Grade: {} {}", component.grade.symbol(), component.grade.icon());
    println!("  Release Ready: {}", if component.release_ready { "Yes" } else { "No" });
}

#[cfg(feature = "native")]
fn demo_stack_layers() {
    println!("  PAIML Stack Layers:");
    println!();

    let layers: [(StackLayer, &[&str]); 8] = [
        (StackLayer::Compute, &["trueno", "trueno-viz", "trueno-db", "trueno-graph", "trueno-rag"]),
        (StackLayer::Ml, &["aprender", "aprender-shell", "aprender-tsp"]),
        (StackLayer::Training, &["realizar", "renacer"]),
        (StackLayer::Orchestration, &["batuta", "certeza", "presentar", "forjar"]),
        (StackLayer::DataMlops, &["alimentar", "entrenar"]),
        (StackLayer::Quality, &["pmat", "provable-contracts", "tiny-model-ground-truth"]),
        (StackLayer::Transpilers, &["ruchy", "decy", "depyler"]),
        (StackLayer::Presentation, &["sovereign-ai-stack-book"]),
    ];

    for (layer, components) in layers.iter() {
        println!("  {} ({} components)", layer.display_name(), components.len());
        for name in *components {
            let detected = StackLayer::from_component(name);
            let mark = if detected == *layer { "  " } else { "??" };
            println!("    {} {}", mark, name);
        }
        println!();
    }
}

#[cfg(feature = "native")]
fn demo_quality_gate() {
    println!("  Quality Gate Enforcement:");
    println!();
    println!("  The quality gate ensures all stack components meet A- threshold");
    println!("  before commits, releases, or deployments are allowed.");
    println!();
    println!("  Enforcement Points:");
    println!("  ┌─────────────────┬─────────────────────────────────────────────┐");
    println!("  │ Point           │ Action                                      │");
    println!("  ├─────────────────┼─────────────────────────────────────────────┤");
    println!("  │ Pre-commit      │ Blocks git push if any component < A-       │");
    println!("  │ Release         │ Blocks batuta stack release (--no-verify)   │");
    println!("  │ CI Pipeline     │ Blocks PR merge if quality gate fails       │");
    println!("  │ Manual          │ make stack-gate returns exit code 1         │");
    println!("  └─────────────────┴─────────────────────────────────────────────┘");
    println!();
    println!("  CLI Commands:");
    println!();
    println!("    batuta stack gate              # Check all components");
    println!("    batuta stack gate --quiet      # Quiet mode for CI");
    println!("    batuta stack quality           # Detailed quality matrix");
    println!();
    println!("  Makefile Targets:");
    println!();
    println!("    make stack-gate     # Enforce A- threshold");
    println!("    make stack-quality  # Show detailed matrix");
    println!("    make qa-stack       # Full stack QA (includes gate)");
    println!();

    // Demonstrate threshold calculation
    println!("  Threshold Calculation:");
    println!();
    println!("  A component is BLOCKED if SQI < 85 (below A-):");
    println!();

    let examples = [
        ("trueno", 95.9, true),
        ("aprender", 96.2, true),
        ("renacer", 87.0, true),
        ("weak-component", 84.4, false),
    ];

    for (name, sqi, passes) in examples {
        let status = if passes { "✅ PASS" } else { "❌ BLOCKED" };
        let grade = QualityGrade::from_sqi(sqi);
        println!("    {:20} SQI: {:5.1}  Grade: {:3}  {}", name, sqi, grade.symbol(), status);
    }
    println!();
    println!("  Release is BLOCKED if any component fails the gate.");
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example stack_quality_demo --features native");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn test_quality_grade_ordering() {
        assert!(QualityGrade::APlus > QualityGrade::A);
        assert!(QualityGrade::A > QualityGrade::AMinus);
        assert!(QualityGrade::AMinus > QualityGrade::BPlus);
        assert!(QualityGrade::BPlus > QualityGrade::B);
    }

    #[test]
    fn test_release_ready_grades() {
        assert!(QualityGrade::APlus.is_release_ready());
        assert!(QualityGrade::A.is_release_ready());
        assert!(QualityGrade::AMinus.is_release_ready());
        assert!(!QualityGrade::BPlus.is_release_ready());
        assert!(!QualityGrade::B.is_release_ready());
    }

    #[test]
    fn test_sqi_calculation() {
        // A+ requires SQI >= 95
        let sqi = 0.40 * 94.74 + 0.30 * 89.09 + 0.20 * 95.0 + 0.10 * 100.0;
        assert!(sqi > 90.0);

        let grade = QualityGrade::from_sqi(sqi);
        assert!(grade.is_release_ready());
    }

    #[test]
    fn test_layer_detection() {
        assert_eq!(StackLayer::from_component("trueno"), StackLayer::Compute);
        assert_eq!(StackLayer::from_component("aprender"), StackLayer::Ml);
        assert_eq!(StackLayer::from_component("batuta"), StackLayer::Orchestration);
        assert_eq!(StackLayer::from_component("depyler"), StackLayer::Transpilers);
    }

    #[test]
    fn test_score_percentage() {
        let score = Score::new(108, 114, QualityGrade::APlus);
        let pct = score.percentage();
        assert!(pct > 94.0 && pct < 95.0);
    }
}
