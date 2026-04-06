//! Accessibility Demo - WCAG 2.1 AA Compliance Testing
//!
//! Demonstrates Probar's accessibility validation features
//! for WASM games following WCAG 2.1 AA standards.
//!
//! # Running
//!
//! ```bash
//! cargo run --example accessibility_demo -p probar
//! ```
//!
//! # Features
//!
//! - Color contrast analysis (WCAG 2.1 AA)
//! - Flash detection for photosensitivity
//! - Accessibility audits

#![allow(
    clippy::uninlined_format_args,
    clippy::std_instead_of_core,
    clippy::unwrap_used
)]

use jugar_probar::{
    AccessibilityAudit, AccessibilityConfig, AccessibilityValidator, Color, ContrastAnalysis,
    FlashDetector, Severity, MIN_CONTRAST_LARGE, MIN_CONTRAST_NORMAL, MIN_CONTRAST_UI,
};

fn main() {
    println!("=== Probar Accessibility Demo ===\n");

    // Demo 1: Color and Contrast
    demo_color_contrast();

    // Demo 2: Flash Detection
    demo_flash_detection();

    // Demo 3: Full Accessibility Audit
    demo_full_audit();

    println!("\n=== Accessibility Demo Complete ===");
}

fn demo_color_contrast() {
    println!("--- Demo 1: Color Contrast Analysis ---\n");

    // WCAG 2.1 AA minimums
    println!("WCAG 2.1 AA Minimum Contrast Ratios:");
    println!("  Normal text (< 18pt): {:.1}:1", MIN_CONTRAST_NORMAL);
    println!("  Large text (>= 18pt): {:.1}:1", MIN_CONTRAST_LARGE);
    println!("  UI components: {:.1}:1", MIN_CONTRAST_UI);
    println!();

    // Create colors
    let white = Color::new(255, 255, 255);
    let black = Color::new(0, 0, 0);
    let dark_blue = Color::new(0, 0, 139);
    let light_gray = Color::new(200, 200, 200);
    let yellow = Color::new(255, 255, 0);

    // Test contrast pairs
    let pairs = vec![
        ("Black on White", black, white),
        ("Dark Blue on White", dark_blue, Color::new(255, 255, 255)),
        ("Light Gray on White", light_gray, Color::new(255, 255, 255)),
        ("Yellow on White", yellow, Color::new(255, 255, 255)),
    ];

    println!("Contrast Analysis Results:");
    for (name, foreground, background) in &pairs {
        let ratio = foreground.contrast_ratio(background);
        let passes_normal = foreground.meets_wcag_aa_normal(background);
        let passes_large = foreground.meets_wcag_aa_large(background);
        let passes_ui = foreground.meets_wcag_aa_ui(background);

        println!("  {} (ratio: {:.2}:1)", name, ratio);
        println!(
            "    Normal text: {} | Large text: {} | UI: {}",
            pass_fail(passes_normal),
            pass_fail(passes_large),
            pass_fail(passes_ui)
        );
    }

    // Use ContrastAnalysis for batch analysis
    println!("\nBatch Analysis with ContrastAnalysis:");
    let mut analysis = ContrastAnalysis::empty();
    analysis.add_pair(black, white, "Primary text");
    analysis.add_pair(dark_blue, Color::new(255, 255, 255), "Link text");
    analysis.add_pair(light_gray, Color::new(255, 255, 255), "Disabled text");

    println!("  Pairs analyzed: {}", analysis.pairs_analyzed);
    println!("  Min ratio: {:.2}:1", analysis.min_ratio);
    println!("  Max ratio: {:.2}:1", analysis.max_ratio);
    println!("  Avg ratio: {:.2}:1", analysis.avg_ratio);
    println!("  Passes WCAG AA: {}", pass_fail(analysis.passes_wcag_aa));
    if !analysis.failing_pairs.is_empty() {
        println!("  Failing pairs: {}", analysis.failing_pairs.len());
        for pair in &analysis.failing_pairs {
            println!("    - {} (ratio: {:.2}:1)", pair.context, pair.ratio);
        }
    }

    println!();
}

fn demo_flash_detection() {
    println!("--- Demo 2: Flash Detection (Photosensitivity) ---\n");

    // Create flash detector with WCAG 2.3.1 settings
    let detector = FlashDetector::new();

    println!("WCAG 2.3.1 'Three Flashes or Below Threshold':");
    println!("  Max flash rate: {:.1} Hz", detector.max_flash_rate);
    println!("  Max red intensity: {:.1}", detector.max_red_intensity);
    println!("  Max flash area: {:.0}%", detector.max_flash_area * 100.0);
    println!();

    // Simulate safe animation
    println!("Test 1: Slow transition (safe)");
    let result = detector.analyze(
        0.3, // luminance change
        0.2, // red intensity
        0.1, // flash area (10%)
        1.0, // 1 second time delta
    );
    print_flash_result(&result);

    // Simulate dangerous animation
    println!("\nTest 2: Fast strobing (dangerous)");
    let result = detector.analyze(
        0.5,  // high luminance change
        0.3,  // red intensity
        0.2,  // 20% flash area
        0.08, // 12.5 Hz (1/0.08)
    );
    print_flash_result(&result);

    // Simulate red flash
    println!("\nTest 3: High red intensity (dangerous)");
    let result = detector.analyze(
        0.4,  // luminance change
        0.95, // very high red intensity
        0.1,  // 10% area
        0.5,  // 2 Hz
    );
    print_flash_result(&result);

    // Simulate large flash area
    println!("\nTest 4: Large flash area (dangerous)");
    let result = detector.analyze(
        0.3, // luminance change
        0.2, // normal red
        0.5, // 50% of screen
        0.5, // 2 Hz
    );
    print_flash_result(&result);

    println!();
}

fn demo_full_audit() {
    println!("--- Demo 3: Full Accessibility Audit ---\n");

    let config = AccessibilityConfig::default();

    println!("Audit Configuration (AccessibilityConfig::default()):");
    println!("  Check contrast: {}", check_mark(config.check_contrast));
    println!("  Check focus: {}", check_mark(config.check_focus));
    println!(
        "  Check reduced motion: {}",
        check_mark(config.check_reduced_motion)
    );
    println!("  Check keyboard: {}", check_mark(config.check_keyboard));

    let validator = AccessibilityValidator::with_config(config);

    // Run audit with good accessibility
    println!("\n--- Audit 1: Good Accessibility ---");
    let black = Color::new(0, 0, 0);
    let white = Color::new(255, 255, 255);

    let audit = validator.audit(
        &[(black, white, "Main text")],
        true, // has focus indicators
        true, // respects reduced motion
    );

    print_audit_result(&audit);

    // Run audit with poor accessibility
    println!("\n--- Audit 2: Poor Accessibility ---");
    let light_gray = Color::new(180, 180, 180);

    let audit = validator.audit(
        &[(light_gray, white, "Low contrast text")],
        false, // missing focus indicators
        false, // ignores reduced motion
    );

    print_audit_result(&audit);

    println!();
}

// Helper functions
const fn pass_fail(passed: bool) -> &'static str {
    if passed {
        "PASS"
    } else {
        "FAIL"
    }
}

const fn check_mark(enabled: bool) -> &'static str {
    if enabled {
        "[x]"
    } else {
        "[ ]"
    }
}

const fn severity_label(severity: &Severity) -> &'static str {
    match severity {
        Severity::Critical => "CRITICAL",
        Severity::Major => "MAJOR",
        Severity::Minor => "MINOR",
        Severity::Info => "INFO",
    }
}

fn print_flash_result(result: &jugar_probar::FlashResult) {
    println!("  Flash rate: {:.1} Hz", result.flash_rate);
    println!("  Red flash exceeded: {}", result.red_flash_exceeded);
    println!("  Flash area: {:.0}%", result.flash_area * 100.0);
    println!("  Is safe: {}", if result.is_safe { "YES" } else { "NO" });
    if let Some(warning) = &result.warning {
        println!("  Warning: {}", warning);
    }
}

fn print_audit_result(audit: &AccessibilityAudit) {
    println!("Audit Results:");
    println!("  Score: {}/100", audit.score);
    println!("  Has focus indicators: {}", audit.has_focus_indicators);
    println!(
        "  Respects reduced motion: {}",
        audit.respects_reduced_motion
    );
    println!("  Passes: {}", if audit.passes() { "YES" } else { "NO" });

    if !audit.issues.is_empty() {
        println!("\nIssues Found ({}):", audit.issues.len());
        for issue in &audit.issues {
            println!(
                "  [{}] WCAG {}: {}",
                severity_label(&issue.severity),
                issue.wcag_code,
                issue.description
            );
            if let Some(fix) = &issue.fix_suggestion {
                println!("      Fix: {}", fix);
            }
        }
    }
}
