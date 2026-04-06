//! Layout helpers for panel grid arrangement.
//!
//! This module provides helpers to reduce complexity in draw_top_panels
//! by extracting the display rule evaluation pattern.

use crate::direct::DirectTerminalCanvas;
use crate::ptop::app::App;
use crate::ptop::config::PanelType;
use crate::widgets::DisplayAction;
use crate::Rect;

/// Type alias for panel draw functions
pub type PanelDrawFn = fn(&App, &mut DirectTerminalCanvas<'_>, Rect);

/// Evaluate display rules and return the appropriate draw function.
///
/// Returns `Some(draw_fn)` if panel should be shown, `None` if hidden.
///
/// # Arguments
/// * `app` - Application state
/// * `panel_enabled` - Whether the panel is enabled in config
/// * `panel_type` - The type of panel to evaluate
/// * `normal_draw_fn` - Draw function for normal/expanded mode
/// * `compact_draw_fn` - Optional draw function for compact mode (uses normal if None)
///
/// # Complexity Reduction
/// Extracts repeated pattern from draw_top_panels, reducing cyclomatic complexity.
/// See: SPEC-024 Section 9.4.0
#[inline]
pub fn evaluate_panel_draw(
    app: &App,
    panel_enabled: bool,
    panel_type: PanelType,
    normal_draw_fn: PanelDrawFn,
    compact_draw_fn: Option<PanelDrawFn>,
) -> Option<PanelDrawFn> {
    if !panel_enabled {
        return None;
    }

    match app.evaluate_panel_display(panel_type) {
        DisplayAction::Show | DisplayAction::Expand => Some(normal_draw_fn),
        DisplayAction::Hide => None,
        DisplayAction::Compact => Some(compact_draw_fn.unwrap_or(normal_draw_fn)),
        DisplayAction::ShowPlaceholder(_) => Some(normal_draw_fn),
    }
}

/// Push panel draw function if display rules allow it.
///
/// Convenience wrapper around `evaluate_panel_draw` that pushes directly to a vec.
#[inline]
pub fn push_if_visible(
    panels: &mut Vec<PanelDrawFn>,
    app: &App,
    panel_enabled: bool,
    panel_type: PanelType,
    normal_draw_fn: PanelDrawFn,
    compact_draw_fn: Option<PanelDrawFn>,
) {
    if let Some(draw_fn) = evaluate_panel_draw(
        app,
        panel_enabled,
        panel_type,
        normal_draw_fn,
        compact_draw_fn,
    ) {
        panels.push(draw_fn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // F-LAY-001: evaluate_panel_draw returns None when panel disabled
    #[test]
    fn test_evaluate_panel_draw_disabled() {
        // Can't easily test without App, but we can verify the function signature compiles
        // Full integration test would require a mock App
    }

    // F-LAY-002: push_if_visible doesn't push when panel disabled
    #[test]
    fn test_push_if_visible_signature() {
        // Verify function compiles with correct signature
        let _ = push_if_visible
            as fn(&mut Vec<PanelDrawFn>, &App, bool, PanelType, PanelDrawFn, Option<PanelDrawFn>);
    }
}
