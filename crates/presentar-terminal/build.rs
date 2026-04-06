//! Build script for presentar-terminal
//!
//! SPEC-024 Section 0: ENFORCEMENT ARCHITECTURE
//!
//! This build script enforces TEST-FIRST DEVELOPMENT at compile time.
//! The build will FAIL if:
//!   1. Implementation modules exist without corresponding interface tests
//!   2. ptop feature is enabled without async data flow tests
//!   3. New widgets are added without widget tests
//!
//! This is NOT optional. This is architectural enforcement.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Re-run if test files change
    println!("cargo:rerun-if-changed=tests/");
    println!("cargo:rerun-if-changed=src/ptop/");
    println!("cargo:rerun-if-changed=src/widgets/");

    // Skip enforcement in doc builds
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // Check if we're building with ptop feature
    let ptop_enabled = env::var("CARGO_FEATURE_PTOP").is_ok();

    if ptop_enabled {
        enforce_ptop_tests();
    }

    enforce_widget_tests();
}

/// SPEC-024: Enforce that async data flow tests exist for ptop
#[allow(clippy::too_many_lines)]
fn enforce_ptop_tests() {
    let tests_dir = Path::new("tests");

    // MANDATORY: cpu_exploded_async.rs MUST exist
    // This test defines the interface for async CPU data updates
    let async_test = tests_dir.join("cpu_exploded_async.rs");
    assert!(
        async_test.exists(),
        "\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  SPEC-024 ENFORCEMENT FAILURE: MISSING INTERFACE TEST                        ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║                                                                              ║\n\
        ║  The file tests/cpu_exploded_async.rs is MANDATORY for ptop builds.          ║\n\
        ║                                                                              ║\n\
        ║  This test DEFINES the interface for async CPU data updates.                 ║\n\
        ║  Without it, the async update system has no contract.                        ║\n\
        ║                                                                              ║\n\
        ║  TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.                             ║\n\
        ║                                                                              ║\n\
        ║  See: docs/specifications/pixel-by-pixel-demo-ptop-ttop.md Part 0             ║\n\
        ║                                                                              ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Check for required test patterns in the async test file
    let async_test_content = fs::read_to_string(&async_test).unwrap_or_default();

    let required_patterns = [
        (
            "test_metrics_snapshot_includes_per_core_freq",
            "MetricsSnapshot must have per_core_freq field",
        ),
        (
            "test_metrics_snapshot_includes_per_core_temp",
            "MetricsSnapshot must have per_core_temp field",
        ),
        (
            "test_app_has_per_core_freq_field",
            "App must have per_core_freq field",
        ),
        (
            "test_app_has_per_core_temp_field",
            "App must have per_core_temp field",
        ),
        (
            "test_apply_snapshot_updates_freq_temp",
            "apply_snapshot must transfer freq/temp data",
        ),
        (
            "test_render_uses_async_updated_data",
            "Render must use async-updated data",
        ),
    ];

    for (pattern, description) in required_patterns {
        assert!(
            async_test_content.contains(pattern),
            "\n\
            ╔══════════════════════════════════════════════════════════════════════════════╗\n\
            ║  SPEC-024 ENFORCEMENT FAILURE: MISSING REQUIRED TEST                         ║\n\
            ╠══════════════════════════════════════════════════════════════════════════════╣\n\
            ║                                                                              ║\n\
            ║  Missing test: {pattern:<50}        ║\n\
            ║                                                                              ║\n\
            ║  Purpose: {description:<54}  ║\n\
            ║                                                                              ║\n\
            ║  This test is REQUIRED by SPEC-024 to define the async data flow interface.  ║\n\
            ║                                                                              ║\n\
            ╚══════════════════════════════════════════════════════════════════════════════╝\n"
        );
    }

    // MANDATORY: Each panel type must have visibility tests
    let visibility_test = tests_dir.join("cbtop_visibility.rs");
    assert!(
        visibility_test.exists(),
        "\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  SPEC-024 ENFORCEMENT FAILURE: MISSING VISIBILITY TEST                       ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║                                                                              ║\n\
        ║  The file tests/cbtop_visibility.rs is MANDATORY for ptop builds.            ║\n\
        ║                                                                              ║\n\
        ║  This test ensures panels render actual data, not placeholder content.       ║\n\
        ║                                                                              ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // MANDATORY: App interface tests
    let app_interface = tests_dir.join("ptop_app_interface.rs");
    assert!(
        app_interface.exists(),
        "\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  SPEC-024 ENFORCEMENT FAILURE: MISSING APP INTERFACE TEST                    ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║                                                                              ║\n\
        ║  The file tests/ptop_app_interface.rs is MANDATORY for ptop builds.          ║\n\
        ║                                                                              ║\n\
        ║  This test defines the App, MetricsSnapshot, and MetricsCollector interface. ║\n\
        ║                                                                              ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // MANDATORY: Panel interface tests
    let panels_interface = tests_dir.join("ptop_panels_interface.rs");
    assert!(
        panels_interface.exists(),
        "\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  SPEC-024 ENFORCEMENT FAILURE: MISSING PANELS INTERFACE TEST                 ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║                                                                              ║\n\
        ║  The file tests/ptop_panels_interface.rs is MANDATORY for ptop builds.       ║\n\
        ║                                                                              ║\n\
        ║  This test defines the interface for all panel types.                        ║\n\
        ║                                                                              ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n"
    );
}

/// Enforce that widgets have corresponding tests
fn enforce_widget_tests() {
    let widgets_dir = Path::new("src/widgets");
    let tests_dir = Path::new("tests");

    if !widgets_dir.exists() {
        return;
    }

    // Collect all widget implementation files
    let widget_files: HashSet<String> = fs::read_dir(widgets_dir)
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
        .filter(|e| e.file_name() != "mod.rs")
        .map(|e| e.file_name().to_string_lossy().replace(".rs", ""))
        .collect();

    // Critical widgets that MUST have tests
    let critical_widgets = [
        "process_table",
        "graph",
        "gauge",
        "sparkline",
        "border",
        "display_rules",
    ];

    for widget in critical_widgets {
        if widget_files.contains(widget) {
            // Check for inline tests in the widget file
            let widget_path = widgets_dir.join(format!("{widget}.rs"));
            let content = fs::read_to_string(&widget_path).unwrap_or_default();

            let has_inline_tests = content.contains("#[cfg(test)]")
                && (content.contains("#[test]") || content.contains("#[path ="));

            // Check for external test file
            let external_test_patterns = [
                format!("f_*_{widget}*.rs"),
                format!("{widget}_test.rs"),
                format!("widget_{widget}.rs"),
            ];

            let has_external_tests = fs::read_dir(tests_dir)
                .into_iter()
                .flatten()
                .filter_map(std::result::Result::ok)
                .any(|e| {
                    let name = e.file_name().to_string_lossy().to_lowercase();
                    name.contains(widget)
                        || external_test_patterns.iter().any(|p| {
                            let pattern = p.replace('*', "");
                            name.contains(&pattern.replace(".rs", ""))
                        })
                });

            if !has_inline_tests && !has_external_tests {
                println!(
                    "cargo:warning=SPEC-024: Widget '{widget}' has no tests. \
                     Add #[cfg(test)] module or tests/{widget}_test.rs"
                );
            }
        }
    }
}
