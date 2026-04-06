//! Tests for terminal capability detection.
#![allow(clippy::disallowed_methods)]

use super::*;

#[test]
fn test_terminal_mode_default() {
    assert_eq!(TerminalMode::default(), TerminalMode::Unicode);
}

#[test]
fn test_dashboard_layout_default() {
    assert_eq!(DashboardLayout::default(), DashboardLayout::Compact);
}

#[test]
fn test_terminal_capabilities_default() {
    let caps = TerminalCapabilities::default();
    assert_eq!(caps.width, 80);
    assert_eq!(caps.height, 24);
    assert!(caps.unicode);
    assert!(caps.ansi_color);
    assert!(!caps.true_color);
    assert!(caps.is_tty);
}

#[test]
fn test_terminal_capabilities_detect() {
    let caps = TerminalCapabilities::detect();
    // Just check it doesn't panic and returns reasonable values
    assert!(caps.width > 0);
    assert!(caps.height > 0);
}

#[test]
fn test_recommended_mode_no_tty() {
    let caps = TerminalCapabilities { is_tty: false, ..Default::default() };
    assert_eq!(caps.recommended_mode(), TerminalMode::Ascii);
}

#[test]
fn test_recommended_mode_true_color() {
    let caps = TerminalCapabilities { true_color: true, ..Default::default() };
    assert_eq!(caps.recommended_mode(), TerminalMode::Ansi);
}

#[test]
fn test_recommended_mode_unicode() {
    let caps = TerminalCapabilities { unicode: true, true_color: false, ..Default::default() };
    assert_eq!(caps.recommended_mode(), TerminalMode::Unicode);
}

#[test]
fn test_recommended_mode_ascii() {
    let caps = TerminalCapabilities {
        unicode: false,
        true_color: false,
        is_tty: true,
        ..Default::default()
    };
    assert_eq!(caps.recommended_mode(), TerminalMode::Ascii);
}

#[test]
fn test_terminal_mode_variants() {
    assert_eq!(TerminalMode::Ascii, TerminalMode::Ascii);
    assert_eq!(TerminalMode::Unicode, TerminalMode::Unicode);
    assert_eq!(TerminalMode::Ansi, TerminalMode::Ansi);
    assert_ne!(TerminalMode::Ascii, TerminalMode::Unicode);
}

#[test]
fn test_dashboard_layout_variants() {
    assert_eq!(DashboardLayout::Minimal, DashboardLayout::Minimal);
    assert_eq!(DashboardLayout::Compact, DashboardLayout::Compact);
    assert_eq!(DashboardLayout::Full, DashboardLayout::Full);
    assert_ne!(DashboardLayout::Minimal, DashboardLayout::Full);
}

#[test]
fn test_terminal_mode_clone() {
    let mode = TerminalMode::Unicode;
    let cloned = mode;
    assert_eq!(mode, cloned);
}

#[test]
fn test_dashboard_layout_clone() {
    let layout = DashboardLayout::Full;
    let cloned = layout;
    assert_eq!(layout, cloned);
}

#[test]
fn test_terminal_capabilities_clone() {
    let caps = TerminalCapabilities::default();
    let cloned = caps;
    assert_eq!(caps.width, cloned.width);
    assert_eq!(caps.height, cloned.height);
}

#[test]
fn test_terminal_mode_debug() {
    let mode = TerminalMode::Ansi;
    let debug_str = format!("{mode:?}");
    assert!(debug_str.contains("Ansi"));
}

#[test]
fn test_dashboard_layout_debug() {
    let layout = DashboardLayout::Full;
    let debug_str = format!("{layout:?}");
    assert!(debug_str.contains("Full"));
}

#[test]
fn test_terminal_capabilities_debug() {
    let caps = TerminalCapabilities::default();
    let debug_str = format!("{caps:?}");
    assert!(debug_str.contains("width"));
    assert!(debug_str.contains("height"));
}

#[test]
fn test_get_size_fallback() {
    // Clear env vars to test fallback path
    std::env::remove_var("COLUMNS");
    std::env::remove_var("LINES");
    let (width, height) = TerminalCapabilities::get_size();
    // Should get either actual terminal size or fallback 80x24
    assert!(width > 0);
    assert!(height > 0);
}

#[test]
fn test_recommended_mode_all_false() {
    let caps = TerminalCapabilities {
        width: 80,
        height: 24,
        unicode: false,
        ansi_color: false,
        true_color: false,
        is_tty: true,
    };
    assert_eq!(caps.recommended_mode(), TerminalMode::Ascii);
}

#[test]
fn test_terminal_capabilities_eq() {
    let caps1 = TerminalCapabilities::default();
    let caps2 = TerminalCapabilities::default();
    assert_eq!(caps1, caps2);
}

#[test]
fn test_get_size_from_env_vars() {
    // Set COLUMNS and LINES to test env var parsing path
    std::env::set_var("COLUMNS", "120");
    std::env::set_var("LINES", "40");
    let (width, height) = TerminalCapabilities::get_size();
    // Should either use env vars or terminal size
    assert!(width > 0);
    assert!(height > 0);
    // Clean up
    std::env::remove_var("COLUMNS");
    std::env::remove_var("LINES");
}

#[test]
fn test_get_size_invalid_env_vars() {
    // Set invalid COLUMNS/LINES to test fallback
    std::env::set_var("COLUMNS", "not_a_number");
    std::env::set_var("LINES", "also_not_a_number");
    let (width, height) = TerminalCapabilities::get_size();
    // Should fall back to default or terminal size
    assert!(width > 0);
    assert!(height > 0);
    // Clean up
    std::env::remove_var("COLUMNS");
    std::env::remove_var("LINES");
}

#[test]
#[ignore = "Modifies env vars, must run single-threaded: cargo test -- --ignored --test-threads=1"]
fn test_detect_with_utf8_lang() {
    // Save original values
    let orig_lang = std::env::var("LANG").ok();
    let orig_term = std::env::var("TERM").ok();
    let orig_colorterm = std::env::var("COLORTERM").ok();

    // Set UTF-8 LANG
    std::env::set_var("LANG", "en_US.UTF-8");
    std::env::set_var("TERM", "xterm-256color");
    std::env::remove_var("COLORTERM");

    let caps = TerminalCapabilities::detect();
    assert!(caps.unicode, "UTF-8 LANG should enable unicode");
    assert!(caps.ansi_color, "xterm-256color should enable ANSI");

    // Restore
    if let Some(v) = orig_lang {
        std::env::set_var("LANG", v);
    } else {
        std::env::remove_var("LANG");
    }
    if let Some(v) = orig_term {
        std::env::set_var("TERM", v);
    } else {
        std::env::remove_var("TERM");
    }
    if let Some(v) = orig_colorterm {
        std::env::set_var("COLORTERM", v);
    }
}

#[test]
#[ignore = "Modifies env vars, must run single-threaded: cargo test -- --ignored --test-threads=1"]
fn test_detect_with_dumb_term() {
    // Save original values
    let orig_term = std::env::var("TERM").ok();

    // Set dumb terminal
    std::env::set_var("TERM", "dumb");

    let caps = TerminalCapabilities::detect();
    assert!(!caps.ansi_color, "dumb TERM should disable ANSI color");

    // Restore
    if let Some(v) = orig_term {
        std::env::set_var("TERM", v);
    } else {
        std::env::remove_var("TERM");
    }
}

#[test]
#[ignore = "Modifies env vars, must run single-threaded: cargo test -- --ignored --test-threads=1"]
fn test_detect_with_truecolor() {
    // Save original values
    let orig_colorterm = std::env::var("COLORTERM").ok();

    // Set truecolor
    std::env::set_var("COLORTERM", "truecolor");

    let caps = TerminalCapabilities::detect();
    assert!(caps.true_color, "truecolor COLORTERM should enable true_color");

    // Restore
    if let Some(v) = orig_colorterm {
        std::env::set_var("COLORTERM", v);
    } else {
        std::env::remove_var("COLORTERM");
    }
}

#[test]
#[ignore = "Modifies env vars, must run single-threaded: cargo test -- --ignored --test-threads=1"]
fn test_detect_with_24bit() {
    // Save original values
    let orig_colorterm = std::env::var("COLORTERM").ok();

    // Set 24bit
    std::env::set_var("COLORTERM", "24bit");

    let caps = TerminalCapabilities::detect();
    assert!(caps.true_color, "24bit COLORTERM should enable true_color");

    // Restore
    if let Some(v) = orig_colorterm {
        std::env::set_var("COLORTERM", v);
    } else {
        std::env::remove_var("COLORTERM");
    }
}

#[test]
#[ignore = "Modifies env vars, must run single-threaded: cargo test -- --ignored --test-threads=1"]
fn test_detect_with_empty_term() {
    // Save original values
    let orig_term = std::env::var("TERM").ok();

    // Clear TERM
    std::env::remove_var("TERM");

    let caps = TerminalCapabilities::detect();
    assert!(!caps.ansi_color, "empty TERM should disable ANSI color");

    // Restore
    if let Some(v) = orig_term {
        std::env::set_var("TERM", v);
    }
}

#[test]
#[ignore = "Modifies env vars, must run single-threaded: cargo test -- --ignored --test-threads=1"]
fn test_detect_with_lowercase_utf() {
    // Save original values
    let orig_lang = std::env::var("LANG").ok();

    // Set lowercase utf
    std::env::set_var("LANG", "en_US.utf8");

    let caps = TerminalCapabilities::detect();
    assert!(caps.unicode, "lowercase utf should enable unicode");

    // Restore
    if let Some(v) = orig_lang {
        std::env::set_var("LANG", v);
    } else {
        std::env::remove_var("LANG");
    }
}

#[test]
#[ignore = "Modifies env vars, must run single-threaded: cargo test -- --ignored --test-threads=1"]
fn test_detect_without_utf_lang() {
    // Save original values
    let orig_lang = std::env::var("LANG").ok();

    // Set non-UTF LANG
    std::env::set_var("LANG", "C");

    let caps = TerminalCapabilities::detect();
    assert!(!caps.unicode, "C LANG should disable unicode");

    // Restore
    if let Some(v) = orig_lang {
        std::env::set_var("LANG", v);
    } else {
        std::env::remove_var("LANG");
    }
}

#[test]
fn test_terminal_capabilities_all_fields() {
    let caps = TerminalCapabilities {
        width: 160,
        height: 50,
        unicode: true,
        ansi_color: true,
        true_color: true,
        is_tty: true,
    };
    assert_eq!(caps.width, 160);
    assert_eq!(caps.height, 50);
    assert!(caps.unicode);
    assert!(caps.ansi_color);
    assert!(caps.true_color);
    assert!(caps.is_tty);
}
