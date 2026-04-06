//! `Selection` - Tufte-inspired selection highlighting primitives
//!
//! Framework widgets for making selections VISIBLE following Tufte's principle:
//! "Differences must be immediately perceivable" (Visual Display, 1983)
//!
//! # Components
//! - `RowHighlight` - Full row background + gutter indicator
//! - `CellHighlight` - Single cell emphasis
//! - `FocusRing` - Panel/container focus indicator
//!
//! # Design Principles
//! 1. **Multiple Redundant Cues**: Color + Shape + Position (accessibility)
//! 2. **High Contrast**: Selection must be visible in any terminal
//! 3. **Consistent Language**: Same visual language across all widgets

use presentar_core::{Canvas, Color, Point, Rect, TextStyle};

// =============================================================================
// TTOP-MATCHING SELECTION COLORS
// =============================================================================
// ttop uses SUBTLE selection: barely visible dark bg + ▶ gutter indicator
// The gutter indicator (▶) is the PRIMARY visual cue, not the background

/// Subtle dark selection background (matches ttop's barely-visible highlight)
/// Just slightly brighter than DIMMED_BG to indicate selection
pub const SELECTION_BG: Color = Color {
    r: 0.15,
    g: 0.12,
    b: 0.22,
    a: 1.0,
}; // Subtle dark purple - barely visible, ttop-style

/// Bright cyan for selection indicators (cursors, borders)
/// This is the PRIMARY visual cue for selection (the ▶ indicator)
pub const SELECTION_ACCENT: Color = Color {
    r: 0.4,
    g: 0.9,
    b: 0.4,
    a: 1.0,
}; // Bright green like ttop's ▶ cursor

/// Gutter indicator color (same as accent for consistency with ttop)
pub const SELECTION_GUTTER: Color = Color {
    r: 0.4,
    g: 0.9,
    b: 0.4,
    a: 1.0,
}; // Bright green ▶

/// Dimmed background for non-selected items
pub const DIMMED_BG: Color = Color {
    r: 0.08,
    g: 0.08,
    b: 0.1,
    a: 1.0,
};

// =============================================================================
// ROW HIGHLIGHT
// =============================================================================

/// Tufte-compliant row highlighting with multiple visual cues
///
/// Visual elements:
/// 1. Strong background color (immediate visibility)
/// 2. Left gutter indicator `▐` or `│` (spatial cue)
/// 3. Optional right border (framing)
/// 4. Text color change (contrast)
#[derive(Debug, Clone)]
pub struct RowHighlight {
    /// The row rectangle
    pub bounds: Rect,
    /// Is this row selected?
    pub selected: bool,
    /// Show gutter indicator
    pub show_gutter: bool,
    /// Gutter character (default: ▐)
    pub gutter_char: char,
}

impl RowHighlight {
    pub fn new(bounds: Rect, selected: bool) -> Self {
        Self {
            bounds,
            selected,
            show_gutter: true,
            gutter_char: '▐',
        }
    }

    pub fn with_gutter(mut self, show: bool) -> Self {
        self.show_gutter = show;
        self
    }

    pub fn with_gutter_char(mut self, ch: char) -> Self {
        self.gutter_char = ch;
        self
    }

    /// Paint the row highlight to a canvas
    ///
    /// CRITICAL: Always paints to clear previous frame artifacts.
    /// Terminal buffers retain pixels - non-selected rows need explicit background.
    pub fn paint(&self, canvas: &mut dyn Canvas) {
        if self.selected {
            // 1. Strong background fill for selected row
            canvas.fill_rect(self.bounds, SELECTION_BG);

            // 2. Left gutter indicator
            if self.show_gutter {
                canvas.draw_text(
                    &self.gutter_char.to_string(),
                    Point::new(self.bounds.x - 1.0, self.bounds.y),
                    &TextStyle {
                        color: SELECTION_GUTTER,
                        ..Default::default()
                    },
                );
            }
        } else {
            // Clear any previous selection artifact with dimmed background
            canvas.fill_rect(self.bounds, DIMMED_BG);
        }
    }

    /// Get the text style for content in this row
    pub fn text_style(&self) -> TextStyle {
        if self.selected {
            TextStyle {
                color: Color::WHITE,
                ..Default::default()
            }
        } else {
            TextStyle {
                color: Color::new(0.85, 0.85, 0.85, 1.0),
                ..Default::default()
            }
        }
    }
}

// =============================================================================
// FOCUS RING (Panel Focus)
// =============================================================================

/// Focus indicator for panels/containers
///
/// Uses Tufte's layering principle:
/// 1. Border style change (Double vs Single)
/// 2. Color intensity change (bright vs dim)
/// 3. Optional indicator character (►)
#[derive(Debug, Clone)]
pub struct FocusRing {
    /// Panel bounds
    pub bounds: Rect,
    /// Is focused?
    pub focused: bool,
    /// Base color (panel's theme color)
    pub base_color: Color,
}

impl FocusRing {
    pub fn new(bounds: Rect, focused: bool, base_color: Color) -> Self {
        Self {
            bounds,
            focused,
            base_color,
        }
    }

    /// Get the border color based on focus state
    pub fn border_color(&self) -> Color {
        if self.focused {
            // Blend with cyan accent for visibility
            Color {
                r: (self.base_color.r * 0.4 + SELECTION_ACCENT.r * 0.6).min(1.0),
                g: (self.base_color.g * 0.4 + SELECTION_ACCENT.g * 0.6).min(1.0),
                b: (self.base_color.b * 0.4 + SELECTION_ACCENT.b * 0.6).min(1.0),
                a: 1.0,
            }
        } else {
            // Dim unfocused panels
            Color {
                r: self.base_color.r * 0.4,
                g: self.base_color.g * 0.4,
                b: self.base_color.b * 0.4,
                a: 1.0,
            }
        }
    }

    /// Get title prefix (► for focused)
    pub fn title_prefix(&self) -> &'static str {
        if self.focused {
            "► "
        } else {
            ""
        }
    }
}

// =============================================================================
// COLUMN HIGHLIGHT
// =============================================================================

/// Column header highlight for sortable tables
#[derive(Debug, Clone)]
pub struct ColumnHighlight {
    /// Column bounds
    pub bounds: Rect,
    /// Is this column selected for navigation?
    pub selected: bool,
    /// Is this column the sort column?
    pub sorted: bool,
    /// Sort direction (true = descending)
    pub sort_descending: bool,
}

impl ColumnHighlight {
    pub fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            selected: false,
            sorted: false,
            sort_descending: true,
        }
    }

    pub fn with_selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    pub fn with_sorted(mut self, sorted: bool, descending: bool) -> Self {
        self.sorted = sorted;
        self.sort_descending = descending;
        self
    }

    /// Get background color
    pub fn background(&self) -> Option<Color> {
        if self.selected {
            Some(Color::new(0.15, 0.35, 0.55, 1.0))
        } else {
            None
        }
    }

    /// Get sort indicator character
    pub fn sort_indicator(&self) -> &'static str {
        if self.sorted {
            if self.sort_descending {
                "▼"
            } else {
                "▲"
            }
        } else {
            ""
        }
    }

    /// Get text style
    pub fn text_style(&self) -> TextStyle {
        let color = if self.sorted {
            SELECTION_ACCENT
        } else if self.selected {
            Color::WHITE
        } else {
            Color::new(0.6, 0.6, 0.6, 1.0)
        };

        TextStyle {
            color,
            ..Default::default()
        }
    }
}

// =============================================================================
// CURSOR INDICATOR
// =============================================================================

/// Universal cursor/pointer indicator
pub struct Cursor;

impl Cursor {
    /// Row cursor (appears in gutter)
    pub const ROW: &'static str = "▶";

    /// Column cursor (appears above header)
    pub const COLUMN: &'static str = "▼";

    /// Panel cursor (appears in title)
    pub const PANEL: &'static str = "►";

    /// Get cursor color
    pub fn color() -> Color {
        SELECTION_ACCENT
    }

    /// Paint a row cursor at position
    pub fn paint_row(canvas: &mut dyn Canvas, pos: Point) {
        canvas.draw_text(
            Self::ROW,
            pos,
            &TextStyle {
                color: Self::color(),
                ..Default::default()
            },
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // =========================================================================
    // COLOR CONSTANTS TESTS (ttop-matching subtle style)
    // =========================================================================

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_row_highlight_colors() {
        // Selection background should be subtle dark purple (ttop-style)
        // Just slightly different from DIMMED_BG, barely visible
        assert!(SELECTION_BG.r < 0.25, "Selection bg should be dark");
        assert!(
            SELECTION_BG.b > SELECTION_BG.r,
            "Selection bg should have purple tint"
        );
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_selection_accent_is_green() {
        // ttop uses bright green for the ▶ indicator
        assert!(SELECTION_ACCENT.g > 0.8, "Accent should be bright green");
        assert!(
            SELECTION_ACCENT.r > 0.3,
            "Accent has some red for visibility"
        );
    }

    #[test]
    fn test_selection_gutter_matches_accent() {
        // Gutter should match accent for consistency (ttop-style)
        assert_eq!(SELECTION_GUTTER.r, SELECTION_ACCENT.r);
        assert_eq!(SELECTION_GUTTER.g, SELECTION_ACCENT.g);
        assert_eq!(SELECTION_GUTTER.b, SELECTION_ACCENT.b);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_dimmed_bg_is_dark() {
        assert!(DIMMED_BG.r < 0.15);
        assert!(DIMMED_BG.g < 0.15);
        assert!(DIMMED_BG.b < 0.15);
    }

    // =========================================================================
    // ROW HIGHLIGHT TESTS
    // =========================================================================

    #[test]
    fn test_row_highlight_new() {
        let bounds = Rect::new(0.0, 0.0, 100.0, 1.0);
        let highlight = RowHighlight::new(bounds, true);

        assert_eq!(highlight.bounds, bounds);
        assert!(highlight.selected);
        assert!(highlight.show_gutter);
        assert_eq!(highlight.gutter_char, '▐');
    }

    #[test]
    fn test_row_highlight_not_selected() {
        let bounds = Rect::new(0.0, 0.0, 100.0, 1.0);
        let highlight = RowHighlight::new(bounds, false);

        assert!(!highlight.selected);
    }

    #[test]
    fn test_row_highlight_with_gutter() {
        let highlight = RowHighlight::new(Rect::default(), true).with_gutter(false);
        assert!(!highlight.show_gutter);

        let highlight2 = highlight.with_gutter(true);
        assert!(highlight2.show_gutter);
    }

    #[test]
    fn test_row_highlight_with_gutter_char() {
        let highlight = RowHighlight::new(Rect::default(), true).with_gutter_char('│');
        assert_eq!(highlight.gutter_char, '│');
    }

    #[test]
    fn test_row_highlight_text_style_selected() {
        let highlight = RowHighlight::new(Rect::default(), true);
        let style = highlight.text_style();
        assert_eq!(style.color, Color::WHITE);
    }

    #[test]
    fn test_row_highlight_text_style_not_selected() {
        let highlight = RowHighlight::new(Rect::default(), false);
        let style = highlight.text_style();
        // Should be gray-ish
        assert!(style.color.r > 0.8);
        assert!(style.color.g > 0.8);
        assert!(style.color.b > 0.8);
    }

    #[test]
    fn test_row_highlight_paint_selected() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut buffer = CellBuffer::new(20, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let bounds = Rect::new(2.0, 1.0, 10.0, 1.0);
        let highlight = RowHighlight::new(bounds, true);
        highlight.paint(&mut canvas);

        // The gutter char should be drawn at x-1
        // And background should be filled
    }

    #[test]
    fn test_row_highlight_paint_not_selected() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut buffer = CellBuffer::new(20, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let bounds = Rect::new(2.0, 1.0, 10.0, 1.0);
        let highlight = RowHighlight::new(bounds, false);
        highlight.paint(&mut canvas);

        // Should paint dimmed background
    }

    #[test]
    fn test_row_highlight_paint_selected_no_gutter() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut buffer = CellBuffer::new(20, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let bounds = Rect::new(2.0, 1.0, 10.0, 1.0);
        let highlight = RowHighlight::new(bounds, true).with_gutter(false);
        highlight.paint(&mut canvas);

        // Should only fill background, no gutter indicator
    }

    // =========================================================================
    // FOCUS RING TESTS
    // =========================================================================

    #[test]
    fn test_focus_ring_new() {
        let bounds = Rect::new(0.0, 0.0, 50.0, 20.0);
        let color = Color::new(0.5, 0.5, 1.0, 1.0);
        let ring = FocusRing::new(bounds, true, color);

        assert_eq!(ring.bounds, bounds);
        assert!(ring.focused);
        assert_eq!(ring.base_color, color);
    }

    #[test]
    fn test_focus_ring_color_blend() {
        let base = Color::new(0.5, 0.5, 1.0, 1.0); // Purple-ish
        let ring = FocusRing::new(Rect::default(), true, base);

        let color = ring.border_color();
        // Should be blended toward cyan
        assert!(color.g > base.g);
    }

    #[test]
    fn test_focus_ring_not_focused_is_dimmed() {
        let base = Color::new(1.0, 0.0, 0.0, 1.0); // Red
        let ring = FocusRing::new(Rect::default(), false, base);

        let color = ring.border_color();
        // Should be dimmed to 40%
        assert!((color.r - 0.4).abs() < 0.01);
        assert!(color.g < 0.01);
        assert!(color.b < 0.01);
    }

    #[test]
    fn test_focus_ring_title_prefix_focused() {
        let ring = FocusRing::new(Rect::default(), true, Color::WHITE);
        assert_eq!(ring.title_prefix(), "► ");
    }

    #[test]
    fn test_focus_ring_title_prefix_not_focused() {
        let ring = FocusRing::new(Rect::default(), false, Color::WHITE);
        assert_eq!(ring.title_prefix(), "");
    }

    // =========================================================================
    // COLUMN HIGHLIGHT TESTS
    // =========================================================================

    #[test]
    fn test_column_highlight_new() {
        let bounds = Rect::new(10.0, 0.0, 20.0, 1.0);
        let col = ColumnHighlight::new(bounds);

        assert_eq!(col.bounds, bounds);
        assert!(!col.selected);
        assert!(!col.sorted);
        assert!(col.sort_descending);
    }

    #[test]
    fn test_column_highlight_with_selected() {
        let col = ColumnHighlight::new(Rect::default()).with_selected(true);
        assert!(col.selected);

        let col2 = col.with_selected(false);
        assert!(!col2.selected);
    }

    #[test]
    fn test_column_highlight_with_sorted() {
        let col = ColumnHighlight::new(Rect::default()).with_sorted(true, true);
        assert!(col.sorted);
        assert!(col.sort_descending);

        let col2 = col.with_sorted(true, false);
        assert!(col2.sorted);
        assert!(!col2.sort_descending);
    }

    #[test]
    fn test_column_highlight_sort_indicator_descending() {
        let col = ColumnHighlight::new(Rect::default()).with_sorted(true, true);
        assert_eq!(col.sort_indicator(), "▼");
    }

    #[test]
    fn test_column_highlight_sort_indicator_ascending() {
        let col = ColumnHighlight::new(Rect::default()).with_sorted(true, false);
        assert_eq!(col.sort_indicator(), "▲");
    }

    #[test]
    fn test_column_highlight_sort_indicator_not_sorted() {
        let col = ColumnHighlight::new(Rect::default());
        assert_eq!(col.sort_indicator(), "");
    }

    #[test]
    fn test_column_highlight_background_selected() {
        let col = ColumnHighlight::new(Rect::default()).with_selected(true);
        let bg = col.background();
        assert!(bg.is_some());
        let bg = bg.unwrap();
        assert!(bg.b > bg.r); // Blue-ish
    }

    #[test]
    fn test_column_highlight_background_not_selected() {
        let col = ColumnHighlight::new(Rect::default());
        assert!(col.background().is_none());
    }

    #[test]
    fn test_column_highlight_text_style_sorted() {
        let col = ColumnHighlight::new(Rect::default()).with_sorted(true, true);
        let style = col.text_style();
        assert_eq!(style.color, SELECTION_ACCENT);
    }

    #[test]
    fn test_column_highlight_text_style_selected() {
        let col = ColumnHighlight::new(Rect::default()).with_selected(true);
        let style = col.text_style();
        assert_eq!(style.color, Color::WHITE);
    }

    #[test]
    fn test_column_highlight_text_style_neither() {
        let col = ColumnHighlight::new(Rect::default());
        let style = col.text_style();
        // Should be gray
        assert!(style.color.r > 0.5 && style.color.r < 0.7);
    }

    // =========================================================================
    // CURSOR TESTS
    // =========================================================================

    #[test]
    fn test_cursor_constants() {
        assert_eq!(Cursor::ROW, "▶");
        assert_eq!(Cursor::COLUMN, "▼");
        assert_eq!(Cursor::PANEL, "►");
    }

    #[test]
    fn test_cursor_color() {
        assert_eq!(Cursor::color(), SELECTION_ACCENT);
    }

    #[test]
    fn test_cursor_paint_row() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut buffer = CellBuffer::new(20, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        Cursor::paint_row(&mut canvas, Point::new(0.0, 0.0));
        // Cursor should be painted at position
    }
}
