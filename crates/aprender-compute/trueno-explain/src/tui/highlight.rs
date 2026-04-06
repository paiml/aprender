//! PTX syntax highlighting for the TUI source pane.
//!
//! Classifies PTX source lines by category (comments, directives,
//! labels, memory ops, arithmetic, control flow) and applies
//! appropriate terminal colors via presentar Color values.

use presentar_core::Color;

/// PTX instruction prefix to color category mapping.
const PTX_MEMORY_PREFIXES: &[&str] = &["ld.", "st."];
const PTX_ARITH_PREFIXES: &[&str] = &["add", "sub", "mul", "mad", "fma"];
const PTX_CONTROL_PREFIXES: &[&str] = &["bra", "ret", "setp"];

/// Color constants for PTX syntax categories (f32 RGBA).
const COLOR_COMMENT: Color = Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }; // DarkGray
const COLOR_DIRECTIVE: Color = Color { r: 1.0, g: 0.3, b: 1.0, a: 1.0 }; // Magenta
const COLOR_LABEL: Color = Color { r: 0.3, g: 1.0, b: 1.0, a: 1.0 }; // Cyan
const COLOR_MEMORY: Color = Color { r: 1.0, g: 1.0, b: 0.3, a: 1.0 }; // Yellow
const COLOR_ARITH: Color = Color { r: 0.3, g: 1.0, b: 0.3, a: 1.0 }; // Green
const COLOR_CONTROL: Color = Color { r: 1.0, g: 0.3, b: 0.3, a: 1.0 }; // Red
const COLOR_DEFAULT: Color = Color { r: 0.8, g: 0.8, b: 0.8, a: 1.0 }; // Light gray

/// Classify a PTX instruction's syntax category for color highlighting.
fn ptx_instruction_color(trimmed: &str) -> Option<Color> {
    let categories: &[(&[&str], Color)] = &[
        (PTX_MEMORY_PREFIXES, COLOR_MEMORY),
        (PTX_ARITH_PREFIXES, COLOR_ARITH),
        (PTX_CONTROL_PREFIXES, COLOR_CONTROL),
    ];
    categories
        .iter()
        .find(|(prefixes, _)| prefixes.iter().any(|p| trimmed.starts_with(p)))
        .map(|(_, color)| *color)
}

/// Apply simple syntax highlighting to a single PTX source line.
///
/// Returns `(line_text, color)` for rendering with presentar.
pub(super) fn highlight_ptx_line(line: &str) -> (String, Color) {
    let owned = line.to_string();
    let trimmed = line.trim();

    if trimmed.starts_with("//") {
        return (owned, COLOR_COMMENT);
    }
    if trimmed.starts_with('.') {
        return (owned, COLOR_DIRECTIVE);
    }
    if trimmed.ends_with(':') && !trimmed.contains(' ') {
        return (owned, COLOR_LABEL);
    }
    if line.starts_with('\t') || line.starts_with("    ") {
        if let Some(color) = ptx_instruction_color(trimmed) {
            return (owned, color);
        }
    }

    (owned, COLOR_DEFAULT)
}
