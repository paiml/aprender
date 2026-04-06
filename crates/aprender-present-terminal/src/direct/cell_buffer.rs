//! Cell buffer with zero-allocation steady state.
//!
//! Uses `CompactString` to inline small strings (≤24 bytes), avoiding
//! heap allocations for typical terminal content.

use bitvec::prelude::*;
use compact_str::CompactString;
use presentar_core::Color;
use unicode_width::UnicodeWidthStr;

/// Text modifiers for terminal cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers(u8);

impl Modifiers {
    /// No modifiers.
    pub const NONE: Self = Self(0);
    /// Bold text.
    pub const BOLD: Self = Self(1 << 0);
    /// Italic text.
    pub const ITALIC: Self = Self(1 << 1);
    /// Underlined text.
    pub const UNDERLINE: Self = Self(1 << 2);
    /// Strikethrough text.
    pub const STRIKETHROUGH: Self = Self(1 << 3);
    /// Dim/faint text.
    pub const DIM: Self = Self(1 << 4);
    /// Blinking text.
    pub const BLINK: Self = Self(1 << 5);
    /// Reversed colors.
    pub const REVERSE: Self = Self(1 << 6);
    /// Hidden text.
    pub const HIDDEN: Self = Self(1 << 7);

    /// Create empty modifiers.
    #[must_use]
    pub const fn empty() -> Self {
        Self::NONE
    }

    /// Check if modifiers is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if a specific modifier is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Add a modifier.
    #[must_use]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Remove a modifier.
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Get raw bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Create from raw bits.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
}

impl std::ops::BitOr for Modifiers {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Modifiers {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for Modifiers {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

/// A single terminal cell.
///
/// Uses `CompactString` for zero-allocation storage of typical graphemes.
/// Memory layout is optimized for cache efficiency (40 bytes total).
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    /// The symbol displayed in this cell (inlined for ≤24 bytes).
    pub symbol: CompactString,
    /// Foreground color.
    pub fg: Color,
    /// Background color.
    pub bg: Color,
    /// Text modifiers.
    pub modifiers: Modifiers,
    /// Display width of the symbol (1 for normal, 2 for wide chars, 0 for continuation).
    width: u8,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            symbol: CompactString::const_new(" "),
            fg: Color::WHITE,
            // Use transparent background so unpainted areas don't show black
            bg: Color::TRANSPARENT,
            modifiers: Modifiers::NONE,
            width: 1,
        }
    }
}

impl Cell {
    /// Create a new cell with the given content.
    #[must_use]
    pub fn new(symbol: &str, fg: Color, bg: Color, modifiers: Modifiers) -> Self {
        let width = UnicodeWidthStr::width(symbol).min(255) as u8;
        Self {
            symbol: CompactString::new(symbol),
            fg,
            bg,
            modifiers,
            width: width.max(1),
        }
    }

    /// Update the cell content (zero-allocation for small strings).
    pub fn update(&mut self, symbol: &str, fg: Color, bg: Color, modifiers: Modifiers) {
        self.symbol.clear();
        self.symbol.push_str(symbol);
        self.fg = fg;
        self.bg = bg;
        self.modifiers = modifiers;
        self.width = UnicodeWidthStr::width(symbol).clamp(1, 255) as u8;
    }

    /// Mark this cell as a continuation of a wide character.
    pub fn make_continuation(&mut self) {
        self.symbol.clear();
        self.width = 0;
    }

    /// Check if this is a continuation cell.
    #[must_use]
    pub const fn is_continuation(&self) -> bool {
        self.width == 0
    }

    /// Get the display width of this cell.
    #[must_use]
    pub const fn width(&self) -> u8 {
        self.width
    }

    /// Reset to default (space with transparent background).
    pub fn reset(&mut self) {
        self.symbol.clear();
        self.symbol.push(' ');
        self.fg = Color::WHITE;
        self.bg = Color::TRANSPARENT;
        self.modifiers = Modifiers::NONE;
        self.width = 1;
    }
}

/// Buffer of terminal cells with dirty tracking.
///
/// Memory footprint for 80×24 terminal: ~75KB
/// (1920 cells × 40 bytes per cell ≈ 76KB)
#[derive(Debug)]
pub struct CellBuffer {
    /// The cell storage.
    cells: Vec<Cell>,
    /// Terminal width.
    width: u16,
    /// Terminal height.
    height: u16,
    /// Dirty bit per cell (1 bit per cell).
    dirty: BitVec,
}

impl CellBuffer {
    /// Create a new buffer with the given dimensions.
    #[must_use]
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            cells: vec![Cell::default(); size],
            width,
            height,
            dirty: bitvec![0; size],
        }
    }

    /// Get the buffer width.
    #[must_use]
    pub const fn width(&self) -> u16 {
        self.width
    }

    /// Get the buffer height.
    #[must_use]
    pub const fn height(&self) -> u16 {
        self.height
    }

    /// Get total cell count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Convert (x, y) to linear index.
    #[must_use]
    pub fn index(&self, x: u16, y: u16) -> usize {
        debug_assert!(x < self.width, "x coordinate out of bounds");
        debug_assert!(y < self.height, "y coordinate out of bounds");
        (y as usize) * (self.width as usize) + (x as usize)
    }

    /// Convert linear index to (x, y).
    #[must_use]
    pub fn coords(&self, idx: usize) -> (u16, u16) {
        debug_assert!(idx < self.cells.len(), "index out of bounds");
        let x = (idx % (self.width as usize)) as u16;
        let y = (idx / (self.width as usize)) as u16;
        debug_assert!(
            x < self.width && y < self.height,
            "coords must be in bounds"
        );
        (x, y)
    }

    /// Get a cell reference.
    #[must_use]
    pub fn get(&self, x: u16, y: u16) -> Option<&Cell> {
        if x < self.width && y < self.height {
            Some(&self.cells[self.index(x, y)])
        } else {
            None
        }
    }

    /// Get a mutable cell reference.
    pub fn get_mut(&mut self, x: u16, y: u16) -> Option<&mut Cell> {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            Some(&mut self.cells[idx])
        } else {
            None
        }
    }

    /// Set a cell and mark it dirty.
    pub fn set(&mut self, x: u16, y: u16, cell: Cell) {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            self.cells[idx] = cell;
            self.dirty.set(idx, true);
        }
    }

    /// Update a cell's content and mark it dirty.
    pub fn update(
        &mut self,
        x: u16,
        y: u16,
        symbol: &str,
        fg: Color,
        bg: Color,
        modifiers: Modifiers,
    ) {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            self.cells[idx].update(symbol, fg, bg, modifiers);
            self.dirty.set(idx, true);
        }
    }

    /// Mark a cell as dirty.
    pub fn mark_dirty(&mut self, x: u16, y: u16) {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            self.dirty.set(idx, true);
        }
    }

    /// Mark all cells as dirty (for full redraw).
    pub fn mark_all_dirty(&mut self) {
        self.dirty.fill(true);
    }

    /// Clear dirty flags.
    pub fn clear_dirty(&mut self) {
        self.dirty.fill(false);
    }

    /// Count dirty cells.
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.dirty.count_ones()
    }

    /// Iterate over dirty cell indices.
    pub fn iter_dirty(&self) -> impl Iterator<Item = usize> + '_ {
        self.dirty.iter_ones()
    }

    /// Get cells slice.
    #[must_use]
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Get cells mutable slice.
    pub fn cells_mut(&mut self) -> &mut [Cell] {
        &mut self.cells
    }

    /// Resize the buffer (clears all content).
    pub fn resize(&mut self, width: u16, height: u16) {
        let size = (width as usize) * (height as usize);
        self.width = width;
        self.height = height;
        self.cells.clear();
        self.cells.resize(size, Cell::default());
        self.dirty = bitvec![0; size];
        self.mark_all_dirty();
    }

    /// Clear the buffer (reset all cells to default).
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.reset();
        }
        self.mark_all_dirty();
    }

    /// Fill a rectangular region.
    pub fn fill_rect(&mut self, x: u16, y: u16, width: u16, height: u16, fg: Color, bg: Color) {
        let x_end = (x + width).min(self.width);
        let y_end = (y + height).min(self.height);

        for cy in y..y_end {
            for cx in x..x_end {
                self.update(cx, cy, " ", fg, bg, Modifiers::NONE);
            }
        }
    }

    /// Set a single character at the given position (keeps existing colors/modifiers).
    pub fn set_char(&mut self, x: u16, y: u16, ch: char) {
        if let Some(cell) = self.get_mut(x, y) {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            cell.symbol = CompactString::from(&*s);
            self.mark_dirty(x, y);
        }
    }

    /// Write a string starting at the given position (keeps existing colors/modifiers).
    pub fn write_str(&mut self, x: u16, y: u16, s: &str) {
        let mut cx = x;
        for ch in s.chars() {
            self.set_char(cx, y, ch);
            cx = cx.saturating_add(1);
            if cx >= self.width {
                break;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_modifiers_empty() {
        let m = Modifiers::empty();
        assert!(m.is_empty());
        assert_eq!(m.bits(), 0);
    }

    #[test]
    fn test_modifiers_with() {
        let m = Modifiers::NONE.with(Modifiers::BOLD);
        assert!(m.contains(Modifiers::BOLD));
        assert!(!m.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_modifiers_without() {
        let m = Modifiers::BOLD.with(Modifiers::ITALIC);
        let m2 = m.without(Modifiers::BOLD);
        assert!(!m2.contains(Modifiers::BOLD));
        assert!(m2.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_modifiers_bitor() {
        let m = Modifiers::BOLD | Modifiers::ITALIC;
        assert!(m.contains(Modifiers::BOLD));
        assert!(m.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_modifiers_bitor_assign() {
        let mut m = Modifiers::BOLD;
        m |= Modifiers::ITALIC;
        assert!(m.contains(Modifiers::BOLD));
        assert!(m.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_modifiers_bitand() {
        let m1 = Modifiers::BOLD | Modifiers::ITALIC;
        let m2 = Modifiers::BOLD | Modifiers::UNDERLINE;
        let m3 = m1 & m2;
        assert!(m3.contains(Modifiers::BOLD));
        assert!(!m3.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_modifiers_from_bits() {
        let m = Modifiers::from_bits(0b0000_0011);
        assert!(m.contains(Modifiers::BOLD));
        assert!(m.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_cell_default() {
        let cell = Cell::default();
        assert_eq!(cell.symbol.as_str(), " ");
        assert_eq!(cell.fg, Color::WHITE);
        assert_eq!(cell.bg, Color::TRANSPARENT);
        assert_eq!(cell.modifiers, Modifiers::NONE);
        assert_eq!(cell.width(), 1);
    }

    #[test]
    fn test_cell_new() {
        let cell = Cell::new("A", Color::RED, Color::BLUE, Modifiers::BOLD);
        assert_eq!(cell.symbol.as_str(), "A");
        assert_eq!(cell.fg, Color::RED);
        assert_eq!(cell.bg, Color::BLUE);
        assert!(cell.modifiers.contains(Modifiers::BOLD));
        assert_eq!(cell.width(), 1);
    }

    #[test]
    fn test_cell_wide_char() {
        let cell = Cell::new("日", Color::WHITE, Color::BLACK, Modifiers::NONE);
        assert_eq!(cell.width(), 2);
    }

    #[test]
    fn test_cell_update() {
        let mut cell = Cell::default();
        cell.update("X", Color::GREEN, Color::YELLOW, Modifiers::ITALIC);
        assert_eq!(cell.symbol.as_str(), "X");
        assert_eq!(cell.fg, Color::GREEN);
        assert_eq!(cell.bg, Color::YELLOW);
        assert!(cell.modifiers.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_cell_continuation() {
        let mut cell = Cell::new("日", Color::WHITE, Color::BLACK, Modifiers::NONE);
        cell.make_continuation();
        assert!(cell.is_continuation());
        assert_eq!(cell.width(), 0);
    }

    #[test]
    fn test_cell_reset() {
        let mut cell = Cell::new("X", Color::RED, Color::BLUE, Modifiers::BOLD);
        cell.reset();
        assert_eq!(cell.symbol.as_str(), " ");
        assert_eq!(cell.fg, Color::WHITE);
        assert_eq!(cell.bg, Color::TRANSPARENT);
        assert!(cell.modifiers.is_empty());
    }

    #[test]
    fn test_buffer_creation() {
        let buf = CellBuffer::new(80, 24);
        assert_eq!(buf.width(), 80);
        assert_eq!(buf.height(), 24);
        assert_eq!(buf.len(), 1920);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_empty() {
        let buf = CellBuffer::new(0, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_index() {
        let buf = CellBuffer::new(10, 5);
        assert_eq!(buf.index(0, 0), 0);
        assert_eq!(buf.index(5, 0), 5);
        assert_eq!(buf.index(0, 1), 10);
        assert_eq!(buf.index(5, 2), 25);
    }

    #[test]
    fn test_buffer_coords() {
        let buf = CellBuffer::new(10, 5);
        assert_eq!(buf.coords(0), (0, 0));
        assert_eq!(buf.coords(5), (5, 0));
        assert_eq!(buf.coords(10), (0, 1));
        assert_eq!(buf.coords(25), (5, 2));
    }

    #[test]
    fn test_buffer_get() {
        let buf = CellBuffer::new(10, 5);
        assert!(buf.get(0, 0).is_some());
        assert!(buf.get(9, 4).is_some());
        assert!(buf.get(10, 0).is_none());
        assert!(buf.get(0, 5).is_none());
    }

    #[test]
    fn test_buffer_get_mut() {
        let mut buf = CellBuffer::new(10, 5);
        assert!(buf.get_mut(0, 0).is_some());
        assert!(buf.get_mut(10, 0).is_none());
    }

    #[test]
    fn test_buffer_set() {
        let mut buf = CellBuffer::new(10, 5);
        let cell = Cell::new("X", Color::RED, Color::BLUE, Modifiers::NONE);
        buf.set(5, 2, cell);

        let retrieved = buf.get(5, 2).unwrap();
        assert_eq!(retrieved.symbol.as_str(), "X");
        assert!(buf.dirty_count() > 0);
    }

    #[test]
    fn test_buffer_set_out_of_bounds() {
        let mut buf = CellBuffer::new(10, 5);
        let cell = Cell::new("X", Color::RED, Color::BLUE, Modifiers::NONE);
        buf.set(100, 100, cell); // Should not panic
    }

    #[test]
    fn test_buffer_update() {
        let mut buf = CellBuffer::new(10, 5);
        buf.update(3, 3, "Y", Color::GREEN, Color::BLACK, Modifiers::BOLD);

        let cell = buf.get(3, 3).unwrap();
        assert_eq!(cell.symbol.as_str(), "Y");
        assert_eq!(cell.fg, Color::GREEN);
    }

    #[test]
    fn test_buffer_dirty_tracking() {
        let mut buf = CellBuffer::new(10, 5);
        assert_eq!(buf.dirty_count(), 0);

        buf.mark_dirty(0, 0);
        assert_eq!(buf.dirty_count(), 1);

        buf.mark_all_dirty();
        assert_eq!(buf.dirty_count(), 50);

        buf.clear_dirty();
        assert_eq!(buf.dirty_count(), 0);
    }

    #[test]
    fn test_buffer_iter_dirty() {
        let mut buf = CellBuffer::new(10, 5);
        buf.mark_dirty(1, 1);
        buf.mark_dirty(3, 3);

        let dirty: Vec<usize> = buf.iter_dirty().collect();
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&buf.index(1, 1)));
        assert!(dirty.contains(&buf.index(3, 3)));
    }

    #[test]
    fn test_buffer_resize() {
        let mut buf = CellBuffer::new(10, 5);
        buf.update(0, 0, "X", Color::RED, Color::BLACK, Modifiers::NONE);

        buf.resize(20, 10);
        assert_eq!(buf.width(), 20);
        assert_eq!(buf.height(), 10);
        assert_eq!(buf.len(), 200);
        // Content should be cleared
        assert_eq!(buf.get(0, 0).unwrap().symbol.as_str(), " ");
        // All should be dirty after resize
        assert_eq!(buf.dirty_count(), 200);
    }

    #[test]
    fn test_buffer_clear() {
        let mut buf = CellBuffer::new(10, 5);
        buf.update(0, 0, "X", Color::RED, Color::BLACK, Modifiers::BOLD);
        buf.clear_dirty();

        buf.clear();
        let cell = buf.get(0, 0).unwrap();
        assert_eq!(cell.symbol.as_str(), " ");
        assert!(cell.modifiers.is_empty());
        assert_eq!(buf.dirty_count(), 50);
    }

    #[test]
    fn test_buffer_fill_rect() {
        let mut buf = CellBuffer::new(10, 10);
        buf.fill_rect(2, 2, 3, 3, Color::WHITE, Color::RED);

        // Inside rect
        assert_eq!(buf.get(3, 3).unwrap().bg, Color::RED);
        // Outside rect - default is TRANSPARENT
        assert_eq!(buf.get(0, 0).unwrap().bg, Color::TRANSPARENT);
    }

    #[test]
    fn test_buffer_fill_rect_clipped() {
        let mut buf = CellBuffer::new(10, 10);
        buf.fill_rect(8, 8, 5, 5, Color::WHITE, Color::BLUE);

        // Should be clipped to buffer bounds
        assert_eq!(buf.get(9, 9).unwrap().bg, Color::BLUE);
    }

    #[test]
    fn test_buffer_cells_access() {
        let mut buf = CellBuffer::new(10, 5);
        assert_eq!(buf.cells().len(), 50);
        assert_eq!(buf.cells_mut().len(), 50);
    }

    #[test]
    fn test_cell_empty_string() {
        let cell = Cell::new("", Color::WHITE, Color::BLACK, Modifiers::NONE);
        // Width should be at least 1
        assert_eq!(cell.width(), 1);
    }

    #[test]
    fn test_modifiers_all_flags() {
        let all = Modifiers::BOLD
            | Modifiers::ITALIC
            | Modifiers::UNDERLINE
            | Modifiers::STRIKETHROUGH
            | Modifiers::DIM
            | Modifiers::BLINK
            | Modifiers::REVERSE
            | Modifiers::HIDDEN;

        assert!(all.contains(Modifiers::BOLD));
        assert!(all.contains(Modifiers::ITALIC));
        assert!(all.contains(Modifiers::UNDERLINE));
        assert!(all.contains(Modifiers::STRIKETHROUGH));
        assert!(all.contains(Modifiers::DIM));
        assert!(all.contains(Modifiers::BLINK));
        assert!(all.contains(Modifiers::REVERSE));
        assert!(all.contains(Modifiers::HIDDEN));
    }
}
