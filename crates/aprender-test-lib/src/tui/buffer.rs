//! Simple text grid for TUI testing.
//!
//! This module provides a lightweight text buffer that replaces ratatui::Buffer
//! for TUI testing purposes. It stores characters in a grid format and can be
//! converted directly to string lines for frame comparison.

/// Simple text grid for TUI testing (replaces ratatui::Buffer).
///
/// Stores characters in a flat vector with row-major ordering.
/// Designed for testing terminal output without the complexity of full
/// terminal cell attributes.
#[derive(Debug, Clone)]
pub struct TextGrid {
    cells: Vec<char>,
    width: u16,
    height: u16,
}

impl TextGrid {
    /// Create a new text grid filled with spaces.
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            cells: vec![' '; size],
            width,
            height,
        }
    }

    /// Get the width of the grid.
    #[inline]
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the grid.
    #[inline]
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Get the total number of cells.
    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Check if the grid is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Convert (x, y) coordinates to a flat index.
    #[inline]
    fn index(&self, x: u16, y: u16) -> Option<usize> {
        if x < self.width && y < self.height {
            Some((y as usize) * (self.width as usize) + (x as usize))
        } else {
            None
        }
    }

    /// Get the character at (x, y).
    pub fn get(&self, x: u16, y: u16) -> Option<char> {
        self.index(x, y).map(|idx| self.cells[idx])
    }

    /// Set the character at (x, y).
    pub fn set(&mut self, x: u16, y: u16, ch: char) {
        if let Some(idx) = self.index(x, y) {
            self.cells[idx] = ch;
        }
    }

    /// Clear the grid (fill with spaces).
    pub fn clear(&mut self) {
        self.cells.fill(' ');
    }

    /// Alias for clear() to match ratatui::Buffer API.
    pub fn reset(&mut self) {
        self.clear();
    }

    /// Resize the grid. Content is cleared.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        let size = (width as usize) * (height as usize);
        self.cells.clear();
        self.cells.resize(size, ' ');
    }

    /// Write a string starting at (x, y).
    /// Characters that would exceed the grid width are truncated.
    pub fn write_str(&mut self, x: u16, y: u16, s: &str) {
        let mut pos_x = x;
        for ch in s.chars() {
            if pos_x >= self.width {
                break;
            }
            self.set(pos_x, y, ch);
            pos_x += 1;
        }
    }

    /// Convert the grid to a vector of string lines.
    /// Trailing spaces on each line are trimmed.
    pub fn to_lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.height as usize);
        for y in 0..self.height {
            let start = (y as usize) * (self.width as usize);
            let end = start + (self.width as usize);
            let line: String = self.cells[start..end].iter().collect();
            lines.push(line.trim_end().to_string());
        }
        lines
    }

    /// Get a reference to the underlying cells.
    pub fn cells(&self) -> &[char] {
        &self.cells
    }

    /// Get a mutable reference to the underlying cells.
    pub fn cells_mut(&mut self) -> &mut [char] {
        &mut self.cells
    }

    /// Fill a rectangular region with a character.
    pub fn fill_rect(&mut self, x: u16, y: u16, width: u16, height: u16, ch: char) {
        for row in y..y.saturating_add(height).min(self.height) {
            for col in x..x.saturating_add(width).min(self.width) {
                self.set(col, row, ch);
            }
        }
    }
}

impl Default for TextGrid {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let grid = TextGrid::new(10, 5);
        assert_eq!(grid.width(), 10);
        assert_eq!(grid.height(), 5);
        assert_eq!(grid.len(), 50);
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_default() {
        let grid = TextGrid::default();
        assert_eq!(grid.width(), 80);
        assert_eq!(grid.height(), 24);
    }

    #[test]
    fn test_get_set() {
        let mut grid = TextGrid::new(10, 5);
        assert_eq!(grid.get(0, 0), Some(' '));

        grid.set(3, 2, 'X');
        assert_eq!(grid.get(3, 2), Some('X'));

        // Out of bounds
        assert_eq!(grid.get(100, 100), None);
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut grid = TextGrid::new(10, 5);
        grid.set(100, 100, 'X'); // Should not panic
        assert_eq!(grid.get(0, 0), Some(' ')); // Grid unchanged
    }

    #[test]
    fn test_clear() {
        let mut grid = TextGrid::new(10, 5);
        grid.set(0, 0, 'X');
        grid.set(5, 3, 'Y');
        grid.clear();
        assert_eq!(grid.get(0, 0), Some(' '));
        assert_eq!(grid.get(5, 3), Some(' '));
    }

    #[test]
    fn test_reset() {
        let mut grid = TextGrid::new(10, 5);
        grid.set(0, 0, 'X');
        grid.reset();
        assert_eq!(grid.get(0, 0), Some(' '));
    }

    #[test]
    fn test_resize() {
        let mut grid = TextGrid::new(10, 5);
        grid.set(0, 0, 'X');
        grid.resize(20, 10);
        assert_eq!(grid.width(), 20);
        assert_eq!(grid.height(), 10);
        assert_eq!(grid.len(), 200);
        assert_eq!(grid.get(0, 0), Some(' ')); // Content cleared
    }

    #[test]
    fn test_write_str() {
        let mut grid = TextGrid::new(10, 5);
        grid.write_str(2, 1, "Hello");
        assert_eq!(grid.get(2, 1), Some('H'));
        assert_eq!(grid.get(3, 1), Some('e'));
        assert_eq!(grid.get(4, 1), Some('l'));
        assert_eq!(grid.get(5, 1), Some('l'));
        assert_eq!(grid.get(6, 1), Some('o'));
    }

    #[test]
    fn test_write_str_truncation() {
        let mut grid = TextGrid::new(5, 1);
        grid.write_str(2, 0, "Hello World");
        // Only "Hel" fits (positions 2, 3, 4)
        assert_eq!(grid.get(2, 0), Some('H'));
        assert_eq!(grid.get(3, 0), Some('e'));
        assert_eq!(grid.get(4, 0), Some('l'));
    }

    #[test]
    fn test_to_lines() {
        let mut grid = TextGrid::new(10, 3);
        grid.write_str(0, 0, "Line 1");
        grid.write_str(0, 1, "Line 2");
        grid.write_str(0, 2, "Line 3");

        let lines = grid.to_lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "Line 1");
        assert_eq!(lines[1], "Line 2");
        assert_eq!(lines[2], "Line 3");
    }

    #[test]
    fn test_to_lines_trims_trailing_spaces() {
        let mut grid = TextGrid::new(20, 2);
        grid.write_str(0, 0, "Hello");
        grid.write_str(0, 1, "World");

        let lines = grid.to_lines();
        assert_eq!(lines[0], "Hello"); // Not "Hello               "
        assert_eq!(lines[1], "World");
    }

    #[test]
    fn test_fill_rect() {
        let mut grid = TextGrid::new(10, 5);
        grid.fill_rect(2, 1, 3, 2, '#');

        // Check filled area
        assert_eq!(grid.get(2, 1), Some('#'));
        assert_eq!(grid.get(3, 1), Some('#'));
        assert_eq!(grid.get(4, 1), Some('#'));
        assert_eq!(grid.get(2, 2), Some('#'));
        assert_eq!(grid.get(3, 2), Some('#'));
        assert_eq!(grid.get(4, 2), Some('#'));

        // Check outside filled area
        assert_eq!(grid.get(1, 1), Some(' '));
        assert_eq!(grid.get(5, 1), Some(' '));
        assert_eq!(grid.get(2, 0), Some(' '));
        assert_eq!(grid.get(2, 3), Some(' '));
    }

    #[test]
    fn test_fill_rect_clipped() {
        let mut grid = TextGrid::new(5, 5);
        grid.fill_rect(3, 3, 10, 10, 'X'); // Extends beyond grid

        // Only cells within bounds are filled
        assert_eq!(grid.get(3, 3), Some('X'));
        assert_eq!(grid.get(4, 3), Some('X'));
        assert_eq!(grid.get(3, 4), Some('X'));
        assert_eq!(grid.get(4, 4), Some('X'));
    }

    #[test]
    fn test_cells_access() {
        let mut grid = TextGrid::new(3, 2);
        grid.set(0, 0, 'A');
        grid.set(1, 0, 'B');
        grid.set(2, 0, 'C');

        let cells = grid.cells();
        assert_eq!(cells[0], 'A');
        assert_eq!(cells[1], 'B');
        assert_eq!(cells[2], 'C');

        let cells_mut = grid.cells_mut();
        cells_mut[0] = 'X';
        assert_eq!(grid.get(0, 0), Some('X'));
    }

    #[test]
    fn test_empty_grid() {
        let grid = TextGrid::new(0, 0);
        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
        assert_eq!(grid.get(0, 0), None);
    }
}
