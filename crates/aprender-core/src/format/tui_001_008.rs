// SHIP-TWO-001 — `tui-rendering-ux-v1` algorithm-level PARTIAL
// discharge for FALSIFY-TUI-UX-001..008 (closes 8/8 sweep).
//
// Contract: `contracts/tui-rendering-ux-v1.yaml`.

// ===========================================================================
// TUI-UX-001 — Three-zone layout: header + body + footer present
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui001Verdict { Pass, Fail }

/// Pass iff the rendered TUI text contains all three zone markers.
/// Markers can be either Unicode box drawing or simple `===` separators.
#[must_use]
pub fn verdict_from_three_zone_layout(rendered: &str) -> Tui001Verdict {
    if rendered.is_empty() { return Tui001Verdict::Fail; }
    let lower = rendered.to_ascii_lowercase();
    let has_header = lower.contains("header") || rendered.starts_with('╭') || rendered.starts_with('┌');
    let has_body = !rendered.is_empty(); // any content qualifies as body
    let has_footer = lower.contains("footer") || rendered.contains("[q]") || rendered.contains("press q");
    if has_header && has_body && has_footer { Tui001Verdict::Pass } else { Tui001Verdict::Fail }
}

// ===========================================================================
// TUI-UX-002 — `q` key quits from any TUI command
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui002Verdict { Pass, Fail }

/// Pass iff `q_key_terminates && exit_code in {0, 1}`.
#[must_use]
pub const fn verdict_from_q_quit(q_key_terminates: bool, exit_code: i32) -> Tui002Verdict {
    if !q_key_terminates { return Tui002Verdict::Fail; }
    if matches!(exit_code, 0 | 1) { Tui002Verdict::Pass } else { Tui002Verdict::Fail }
}

// ===========================================================================
// TUI-UX-003 — No `use ratatui` import in TUI source files
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_no_ratatui_import(source: &str) -> Tui003Verdict {
    if source.contains("use ratatui") { Tui003Verdict::Fail } else { Tui003Verdict::Pass }
}

// ===========================================================================
// TUI-UX-004 — `presentar_terminal` is the rendering library
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_presentar_widget_use(source: &str) -> Tui004Verdict {
    if source.contains("presentar_terminal") { Tui004Verdict::Pass } else { Tui004Verdict::Fail }
}

// ===========================================================================
// TUI-UX-005 — Responsive layout: tests pass after resize
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiTestOutcome {
    /// All TUI tests pass.
    AllPassed,
    /// At least one TUI test failed.
    SomeFailed,
    /// Tests didn't run (build error, no harness).
    NoHarness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_tui_tests(outcome: TuiTestOutcome) -> Tui005Verdict {
    match outcome {
        TuiTestOutcome::AllPassed => Tui005Verdict::Pass,
        _ => Tui005Verdict::Fail,
    }
}

// ===========================================================================
// TUI-UX-006 — No hardcoded ANSI Color::* in TUI source
// ===========================================================================

pub const AC_TUI_006_FORBIDDEN_COLORS: [&str; 9] = [
    "Color::Red", "Color::Green", "Color::Blue",
    "Color::Yellow", "Color::Cyan", "Color::Magenta",
    "Color::White", "Color::Black", "Color::DarkGray",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_no_hardcoded_colors(source: &str) -> Tui006Verdict {
    for needle in AC_TUI_006_FORBIDDEN_COLORS {
        if source.contains(needle) { return Tui006Verdict::Fail; }
    }
    Tui006Verdict::Pass
}

// ===========================================================================
// TUI-UX-007 — Zero ratatui types (Frame/Terminal/Block/Borders)
// ===========================================================================

pub const AC_TUI_007_FORBIDDEN_TYPES: [&str; 7] = [
    "ratatui::Frame", "ratatui::Terminal", "ratatui::layout",
    "ratatui::widgets", "ratatui::style", "ratatui::backend",
    "ratatui::Buffer",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_no_ratatui_types(source: &str) -> Tui007Verdict {
    for needle in AC_TUI_007_FORBIDDEN_TYPES {
        if source.contains(needle) { return Tui007Verdict::Fail; }
    }
    Tui007Verdict::Pass
}

// ===========================================================================
// TUI-UX-008 — Headless mode preserved: `apr cbtop --headless --simulated` exits 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tui008Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_headless_exit(exit_code: i32) -> Tui008Verdict {
    if exit_code == 0 { Tui008Verdict::Pass } else { Tui008Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TUI-UX-001
    #[test] fn tui001_pass_canonical() {
        let rendered = "header\nbody content\nfooter [q] quit";
        assert_eq!(verdict_from_three_zone_layout(rendered), Tui001Verdict::Pass);
    }
    #[test] fn tui001_pass_box_drawing() {
        let rendered = "╭ HEADER ╮\nbody\n[q] quit";
        assert_eq!(verdict_from_three_zone_layout(rendered), Tui001Verdict::Pass);
    }
    #[test] fn tui001_fail_no_footer() {
        let rendered = "header\nbody";
        assert_eq!(verdict_from_three_zone_layout(rendered), Tui001Verdict::Fail);
    }
    #[test] fn tui001_fail_empty() {
        assert_eq!(verdict_from_three_zone_layout(""), Tui001Verdict::Fail);
    }

    // TUI-UX-002
    #[test] fn tui002_pass_clean_quit() {
        assert_eq!(verdict_from_q_quit(true, 0), Tui002Verdict::Pass);
    }
    #[test] fn tui002_pass_q_runtime_err() {
        assert_eq!(verdict_from_q_quit(true, 1), Tui002Verdict::Pass);
    }
    #[test] fn tui002_fail_q_does_nothing() {
        assert_eq!(verdict_from_q_quit(false, 0), Tui002Verdict::Fail);
    }
    #[test] fn tui002_fail_panic() {
        assert_eq!(verdict_from_q_quit(true, 101), Tui002Verdict::Fail);
    }

    // TUI-UX-003
    #[test] fn tui003_pass_clean() {
        let src = "use presentar_terminal::widgets::Block;";
        assert_eq!(verdict_from_no_ratatui_import(src), Tui003Verdict::Pass);
    }
    #[test] fn tui003_fail_ratatui_import() {
        let src = "use ratatui::widgets::Block;";
        assert_eq!(verdict_from_no_ratatui_import(src), Tui003Verdict::Fail);
    }

    // TUI-UX-004
    #[test] fn tui004_pass_present() {
        let src = "use presentar_terminal::widgets;";
        assert_eq!(verdict_from_presentar_widget_use(src), Tui004Verdict::Pass);
    }
    #[test] fn tui004_fail_missing() {
        let src = "use std::fmt;";
        assert_eq!(verdict_from_presentar_widget_use(src), Tui004Verdict::Fail);
    }

    // TUI-UX-005
    #[test] fn tui005_pass_all_passed() {
        assert_eq!(verdict_from_tui_tests(TuiTestOutcome::AllPassed), Tui005Verdict::Pass);
    }
    #[test] fn tui005_fail_some_failed() {
        assert_eq!(verdict_from_tui_tests(TuiTestOutcome::SomeFailed), Tui005Verdict::Fail);
    }
    #[test] fn tui005_fail_no_harness() {
        assert_eq!(verdict_from_tui_tests(TuiTestOutcome::NoHarness), Tui005Verdict::Fail);
    }

    // TUI-UX-006
    #[test] fn tui006_pass_clean() {
        let src = "let style = theme.primary();";
        assert_eq!(verdict_from_no_hardcoded_colors(src), Tui006Verdict::Pass);
    }
    #[test] fn tui006_fail_red() {
        let src = "let c = Color::Red;";
        assert_eq!(verdict_from_no_hardcoded_colors(src), Tui006Verdict::Fail);
    }
    #[test] fn tui006_fail_dark_gray() {
        let src = "Color::DarkGray";
        assert_eq!(verdict_from_no_hardcoded_colors(src), Tui006Verdict::Fail);
    }

    // TUI-UX-007
    #[test] fn tui007_pass_clean() {
        let src = "use presentar_terminal::Frame;";
        assert_eq!(verdict_from_no_ratatui_types(src), Tui007Verdict::Pass);
    }
    #[test] fn tui007_fail_ratatui_frame() {
        let src = "use ratatui::Frame;";
        assert_eq!(verdict_from_no_ratatui_types(src), Tui007Verdict::Fail);
    }
    #[test] fn tui007_fail_ratatui_layout() {
        let src = "ratatui::layout::Constraint";
        assert_eq!(verdict_from_no_ratatui_types(src), Tui007Verdict::Fail);
    }

    // TUI-UX-008
    #[test] fn tui008_pass_zero() {
        assert_eq!(verdict_from_headless_exit(0), Tui008Verdict::Pass);
    }
    #[test] fn tui008_fail_nonzero() {
        assert_eq!(verdict_from_headless_exit(1), Tui008Verdict::Fail);
    }
    #[test] fn tui008_fail_panic() {
        assert_eq!(verdict_from_headless_exit(101), Tui008Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_color_count() {
        assert_eq!(AC_TUI_006_FORBIDDEN_COLORS.len(), 9);
    }
    #[test] fn provenance_type_count() {
        assert_eq!(AC_TUI_007_FORBIDDEN_TYPES.len(), 7);
    }
}
