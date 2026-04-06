//! Differential renderer for optimized terminal I/O.
//!
//! Minimizes terminal escape sequences and syscalls by:
//! - Only rendering dirty cells
//! - Batching output to a buffer
//! - Skipping redundant cursor moves
//! - Caching current style state

use super::cell_buffer::{CellBuffer, Modifiers};
use crate::color::ColorMode;
use crossterm::cursor::MoveTo;
use crossterm::style::{
    Attribute, Color as CrosstermColor, Print, ResetColor, SetAttribute, SetBackgroundColor,
    SetForegroundColor,
};
use crossterm::{queue, QueueableCommand};
use presentar_core::Color;
use std::io::{self, BufWriter, Write};

/// Current terminal style state.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StyleState {
    fg: Color,
    bg: Color,
    modifiers: Modifiers,
}

impl Default for StyleState {
    fn default() -> Self {
        Self {
            fg: Color::WHITE,
            bg: Color::BLACK,
            modifiers: Modifiers::NONE,
        }
    }
}

/// Differential renderer that minimizes terminal I/O.
///
/// Tracks the current cursor position and style state to avoid
/// redundant escape sequences.
#[derive(Debug)]
pub struct DiffRenderer {
    /// Color mode for conversion.
    color_mode: ColorMode,
    /// Last known cursor position (`u16::MAX` = unknown).
    cursor_x: u16,
    cursor_y: u16,
    /// Last known style state.
    last_style: StyleState,
    /// Statistics: number of cells written.
    cells_written: usize,
    /// Statistics: number of cursor moves.
    cursor_moves: usize,
    /// Statistics: number of style changes.
    style_changes: usize,
}

impl Default for DiffRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffRenderer {
    /// Create a new renderer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            color_mode: ColorMode::detect(),
            cursor_x: u16::MAX,
            cursor_y: u16::MAX,
            last_style: StyleState::default(),
            cells_written: 0,
            cursor_moves: 0,
            style_changes: 0,
        }
    }

    /// Create a renderer with specific color mode.
    #[must_use]
    pub fn with_color_mode(color_mode: ColorMode) -> Self {
        Self {
            color_mode,
            cursor_x: u16::MAX,
            cursor_y: u16::MAX,
            last_style: StyleState::default(),
            cells_written: 0,
            cursor_moves: 0,
            style_changes: 0,
        }
    }

    /// Set the color mode.
    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }

    /// Get the color mode.
    #[must_use]
    pub const fn color_mode(&self) -> ColorMode {
        self.color_mode
    }

    /// Reset renderer state (call after terminal resize or clear).
    pub fn reset(&mut self) {
        self.cursor_x = u16::MAX;
        self.cursor_y = u16::MAX;
        self.last_style = StyleState::default();
        self.cells_written = 0;
        self.cursor_moves = 0;
        self.style_changes = 0;
    }

    /// Get cells written in last flush.
    #[must_use]
    pub const fn cells_written(&self) -> usize {
        self.cells_written
    }

    /// Get cursor moves in last flush.
    #[must_use]
    pub const fn cursor_moves(&self) -> usize {
        self.cursor_moves
    }

    /// Get style changes in last flush.
    #[must_use]
    pub const fn style_changes(&self) -> usize {
        self.style_changes
    }

    /// Convert presentar Color to crossterm Color.
    fn to_crossterm_color(&self, color: Color) -> CrosstermColor {
        self.color_mode.to_crossterm(color)
    }

    /// Flush dirty cells to the writer.
    ///
    /// Returns the number of cells written.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the writer fails.
    pub fn flush<W: Write>(
        &mut self,
        buffer: &mut CellBuffer,
        writer: &mut W,
    ) -> io::Result<usize> {
        debug_assert!(buffer.width() > 0, "buffer width must be positive");
        debug_assert!(buffer.height() > 0, "buffer height must be positive");

        // Reset statistics
        self.cells_written = 0;
        self.cursor_moves = 0;
        self.style_changes = 0;

        // Use buffered writer to batch syscalls
        let mut buf_writer = BufWriter::with_capacity(8192, writer);

        // Reset colors at start for clean state
        queue!(buf_writer, ResetColor)?;
        self.last_style = StyleState::default();

        let width = buffer.width();

        for idx in buffer.iter_dirty() {
            let (x, y) = buffer.coords(idx);
            let cell = &buffer.cells()[idx];

            // Skip continuation cells
            if cell.is_continuation() {
                continue;
            }

            // Move cursor if needed
            if self.cursor_x != x || self.cursor_y != y {
                queue!(buf_writer, MoveTo(x, y))?;
                self.cursor_x = x;
                self.cursor_y = y;
                self.cursor_moves += 1;
            }

            // Update style if needed
            let new_style = StyleState {
                fg: cell.fg,
                bg: cell.bg,
                modifiers: cell.modifiers,
            };

            if new_style != self.last_style {
                self.apply_style(&mut buf_writer, new_style)?;
                self.last_style = new_style;
                self.style_changes += 1;
            }

            // Print symbol
            queue!(buf_writer, Print(&cell.symbol))?;

            // Update cursor position
            self.cursor_x = self.cursor_x.saturating_add(cell.width() as u16);
            if self.cursor_x >= width {
                self.cursor_x = u16::MAX; // Unknown after wrap
            }

            self.cells_written += 1;
        }

        // Clear dirty flags
        buffer.clear_dirty();

        // Final flush
        buf_writer.flush()?;

        Ok(self.cells_written)
    }

    /// Apply style changes to the writer.
    fn apply_style<W: Write>(&self, writer: &mut W, style: StyleState) -> io::Result<()> {
        // Reset attributes FIRST (before setting colors!)
        writer.queue(SetAttribute(Attribute::Reset))?;

        // Foreground color
        let fg = self.to_crossterm_color(style.fg);
        writer.queue(SetForegroundColor(fg))?;

        // Background color
        let bg = self.to_crossterm_color(style.bg);
        writer.queue(SetBackgroundColor(bg))?;

        // Apply modifiers
        if style.modifiers.contains(Modifiers::BOLD) {
            writer.queue(SetAttribute(Attribute::Bold))?;
        }
        if style.modifiers.contains(Modifiers::ITALIC) {
            writer.queue(SetAttribute(Attribute::Italic))?;
        }
        if style.modifiers.contains(Modifiers::UNDERLINE) {
            writer.queue(SetAttribute(Attribute::Underlined))?;
        }
        if style.modifiers.contains(Modifiers::STRIKETHROUGH) {
            writer.queue(SetAttribute(Attribute::CrossedOut))?;
        }
        if style.modifiers.contains(Modifiers::DIM) {
            writer.queue(SetAttribute(Attribute::Dim))?;
        }
        if style.modifiers.contains(Modifiers::BLINK) {
            writer.queue(SetAttribute(Attribute::SlowBlink))?;
        }
        if style.modifiers.contains(Modifiers::REVERSE) {
            writer.queue(SetAttribute(Attribute::Reverse))?;
        }
        if style.modifiers.contains(Modifiers::HIDDEN) {
            writer.queue(SetAttribute(Attribute::Hidden))?;
        }

        Ok(())
    }

    /// Render a full frame (marks all dirty then flushes).
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn render_full<W: Write>(
        &mut self,
        buffer: &mut CellBuffer,
        writer: &mut W,
    ) -> io::Result<usize> {
        contract_pre_render!();
        buffer.mark_all_dirty();
        self.flush(buffer, writer)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = DiffRenderer::new();
        assert_eq!(renderer.cursor_x, u16::MAX);
        assert_eq!(renderer.cursor_y, u16::MAX);
    }

    #[test]
    fn test_renderer_with_color_mode() {
        let renderer = DiffRenderer::with_color_mode(ColorMode::Color256);
        assert_eq!(renderer.color_mode(), ColorMode::Color256);
    }

    #[test]
    fn test_renderer_set_color_mode() {
        let mut renderer = DiffRenderer::new();
        renderer.set_color_mode(ColorMode::Color16);
        assert_eq!(renderer.color_mode(), ColorMode::Color16);
    }

    #[test]
    fn test_renderer_reset() {
        let mut renderer = DiffRenderer::new();
        renderer.cursor_x = 10;
        renderer.cursor_y = 5;
        renderer.cells_written = 100;

        renderer.reset();

        assert_eq!(renderer.cursor_x, u16::MAX);
        assert_eq!(renderer.cursor_y, u16::MAX);
        assert_eq!(renderer.cells_written(), 0);
    }

    #[test]
    fn test_renderer_flush_empty() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        let mut output = Vec::new();

        let count = renderer.flush(&mut buffer, &mut output).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    #[allow(clippy::len_zero)]
    fn test_renderer_flush_dirty_cells() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        buffer.update(5, 2, "X", Color::RED, Color::BLACK, Modifiers::NONE);
        let mut output = Vec::new();

        let count = renderer.flush(&mut buffer, &mut output).unwrap();
        assert_eq!(count, 1);
        assert!(output.len() > 0);
    }

    #[test]
    fn test_renderer_flush_multiple_dirty() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        buffer.update(0, 0, "A", Color::WHITE, Color::BLACK, Modifiers::NONE);
        buffer.update(5, 2, "B", Color::WHITE, Color::BLACK, Modifiers::NONE);
        buffer.update(9, 4, "C", Color::WHITE, Color::BLACK, Modifiers::NONE);
        let mut output = Vec::new();

        let count = renderer.flush(&mut buffer, &mut output).unwrap();
        assert_eq!(count, 3);
        assert_eq!(renderer.cursor_moves(), 3);
    }

    #[test]
    fn test_renderer_flush_adjacent_cells() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        // Adjacent cells should minimize cursor moves
        buffer.update(0, 0, "A", Color::WHITE, Color::BLACK, Modifiers::NONE);
        buffer.update(1, 0, "B", Color::WHITE, Color::BLACK, Modifiers::NONE);
        buffer.update(2, 0, "C", Color::WHITE, Color::BLACK, Modifiers::NONE);
        let mut output = Vec::new();

        let count = renderer.flush(&mut buffer, &mut output).unwrap();
        assert_eq!(count, 3);
        // Should only need one cursor move (to start)
        assert_eq!(renderer.cursor_moves(), 1);
    }

    #[test]
    fn test_renderer_style_changes() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        buffer.update(0, 0, "A", Color::RED, Color::BLACK, Modifiers::NONE);
        buffer.update(1, 0, "B", Color::BLUE, Color::BLACK, Modifiers::NONE);
        let mut output = Vec::new();

        renderer.flush(&mut buffer, &mut output).unwrap();
        assert_eq!(renderer.style_changes(), 2);
    }

    #[test]
    fn test_renderer_same_style_no_change() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        // Use non-default colors to force a style change on first cell
        buffer.update(0, 0, "A", Color::RED, Color::BLUE, Modifiers::NONE);
        buffer.update(1, 0, "B", Color::RED, Color::BLUE, Modifiers::NONE);
        let mut output = Vec::new();

        renderer.flush(&mut buffer, &mut output).unwrap();
        // First cell triggers style change, second cell has same style = 1 total
        assert_eq!(renderer.style_changes(), 1);
    }

    #[test]
    fn test_renderer_with_modifiers() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        buffer.update(
            0,
            0,
            "X",
            Color::WHITE,
            Color::BLACK,
            Modifiers::BOLD | Modifiers::ITALIC,
        );
        let mut output = Vec::new();

        let count = renderer.flush(&mut buffer, &mut output).unwrap();
        assert_eq!(count, 1);
        // Output should contain attribute sequences
        assert!(output.len() > 5);
    }

    #[test]
    fn test_renderer_all_modifiers() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        let all_mods = Modifiers::BOLD
            | Modifiers::ITALIC
            | Modifiers::UNDERLINE
            | Modifiers::STRIKETHROUGH
            | Modifiers::DIM
            | Modifiers::BLINK
            | Modifiers::REVERSE
            | Modifiers::HIDDEN;
        buffer.update(0, 0, "X", Color::WHITE, Color::BLACK, all_mods);
        let mut output = Vec::new();

        renderer.flush(&mut buffer, &mut output).unwrap();
        assert!(output.len() > 10);
    }

    #[test]
    fn test_renderer_render_full() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);
        buffer.clear_dirty();

        let mut output = Vec::new();
        let count = renderer.render_full(&mut buffer, &mut output).unwrap();

        // All 50 cells should be rendered
        assert_eq!(count, 50);
    }

    #[test]
    fn test_renderer_skip_continuation() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 5);

        // Set a wide character
        buffer.update(0, 0, "日", Color::WHITE, Color::BLACK, Modifiers::NONE);
        // Mark continuation
        if let Some(cell) = buffer.get_mut(1, 0) {
            cell.make_continuation();
        }
        buffer.mark_dirty(1, 0);

        let mut output = Vec::new();
        let count = renderer.flush(&mut buffer, &mut output).unwrap();

        // Only the main cell should be written
        assert_eq!(count, 1);
    }

    #[test]
    fn test_renderer_cursor_wrap() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(5, 2);
        buffer.update(4, 0, "X", Color::WHITE, Color::BLACK, Modifiers::NONE);
        let mut output = Vec::new();

        renderer.flush(&mut buffer, &mut output).unwrap();
        // After writing at x=4, cursor should wrap/be unknown
        assert_eq!(renderer.cursor_x, u16::MAX);
    }

    #[test]
    fn test_renderer_statistics() {
        let mut renderer = DiffRenderer::new();
        let mut buffer = CellBuffer::new(10, 6);
        buffer.update(0, 0, "A", Color::RED, Color::BLACK, Modifiers::NONE);
        buffer.update(5, 5, "B", Color::BLUE, Color::WHITE, Modifiers::BOLD);
        let mut output = Vec::new();

        renderer.flush(&mut buffer, &mut output).unwrap();

        assert_eq!(renderer.cells_written(), 2);
        assert!(renderer.cursor_moves() >= 2);
        assert!(renderer.style_changes() >= 2);
    }

    #[test]
    fn test_renderer_default() {
        let renderer = DiffRenderer::default();
        assert_eq!(renderer.cursor_x, u16::MAX);
    }

    #[test]
    fn test_style_state_default() {
        let state = StyleState::default();
        assert_eq!(state.fg, Color::WHITE);
        assert_eq!(state.bg, Color::BLACK);
        assert!(state.modifiers.is_empty());
    }

    #[test]
    fn test_style_state_equality() {
        let s1 = StyleState::default();
        let s2 = StyleState::default();
        assert_eq!(s1, s2);

        let s3 = StyleState {
            fg: Color::RED,
            ..Default::default()
        };
        assert_ne!(s1, s3);
    }
}
