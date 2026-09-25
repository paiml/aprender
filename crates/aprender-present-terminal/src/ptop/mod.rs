//! ptop: Pixel-perfect ttop clone using presentar-terminal widgets.
//!
//! This module provides the application logic for ptop, mirroring ttop's structure:
//! - `app`: Application state and data collectors
//! - `config`: YAML configuration and layout algorithms (SPEC-024 v5.0 Features A, B)
//! - `ui`: Layout and rendering using presentar-terminal widgets
//! - `analyzers`: System analyzers for detailed metrics
//!
//! # SPEC-024 Architectural Enforcement
//!
//! This module CANNOT be compiled without its interface tests.
//! The enforcement below causes a compile error if tests don't exist.
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

// Module-wide clippy allows for style-only warnings (SPEC-024 v5.8.0 quality compliance)
#![allow(clippy::too_many_lines)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::no_effect_underscore_binding)]

// =============================================================================
// SPEC-024 ARCHITECTURAL ENFORCEMENT
// =============================================================================
//
// `enforced_interface_suites` requires each interface test file to exist:
// in tree, a missing suite FAILS `cargo test` (#4129 moved this from compile
// time, because the published tarball ships no tests/).
//
// To add a new feature to ptop:
// 1. Create the interface test FIRST in tests/
// 2. Add its file name to `enforced_interface_suites::SUITES` below
// 3. Implement the feature

/// ENFORCEMENT: the ptop interface suites MUST exist (SPEC-024 Section 30, Appendix E.6).
/// These were compile-time `include_str!("../../tests/...")` consts, but the package excludes
/// `tests/`, so a `--features ptop` build of the published tarball could not compile (#4129).
/// Now a run-time test: in tree (the workspace's `contracts/` beside the crate) a missing suite
/// FAILS; out of tree (the crates.io tarball, which ships no `tests/`) it skips and names itself.
#[cfg(test)]
mod enforced_interface_suites {
    const SUITES: [&str; 7] = [
        "cpu_exploded_async.rs",
        "cbtop_visibility.rs",
        "ptop_app_interface.rs",
        "ptop_panels_interface.rs",
        "widget_interface_tests.rs",
        "exploded_view_interface.rs",
        "design_principles_interface.rs",
    ];

    #[test]
    fn every_enforced_ptop_interface_suite_exists() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !manifest.join("../../contracts").is_dir() {
            eprintln!(
                "SKIP every_enforced_ptop_interface_suite_exists: out of tree (no workspace \
                 contracts/ beside this crate) - a published crate ships no tests/ (#4129)"
            );
            return;
        }
        for name in SUITES {
            let suite = manifest.join("tests").join(name);
            let src = std::fs::read_to_string(&suite)
                .unwrap_or_else(|e| panic!("enforced suite {} missing: {e}", suite.display()));
            // Existence is what the old include_str! enforced; some suites are harnesses
            // (widget_interface_tests.rs is `mod widgets;`) with no #[test] of their own.
            assert!(
                !src.trim().is_empty(),
                "enforced suite {} is empty",
                suite.display()
            );
        }
    }
}

pub mod analyzers;
pub mod app;
pub mod config;
pub mod input;
pub mod ui;
pub mod ui_atoms;

pub use analyzers::{
    AnalyzerRegistry, ConnectionsAnalyzer, ConnectionsData, PsiAnalyzer, PsiData, TcpConnection,
    TcpState,
};
pub use app::{App, MetricsSnapshot};
pub use config::{
    calculate_grid_layout, snap_to_grid, DetailLevel, FocusStyle, LayoutConfig, PanelConfig,
    PanelRect, PanelType, PtopConfig,
};
pub use input::{InputHandler, TimestampedKey};
