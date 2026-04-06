//! Terminal snapshot assertions for presentar testing.
//!
//! Provides cell-based terminal output capture and assertion capabilities.

use std::fmt;

/// RGB color representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Color {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
}

impl Color {
    /// Create a new color from RGB values.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Parse color from hex string (e.g., "#64C8FF").
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix('#').unwrap_or(hex);
        if hex.len() != 6 {
            return None;
        }
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        Some(Self { r, g, b })
    }

    /// Convert to hex string.
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Black color.
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    /// White color.
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    /// Red color.
    pub const RED: Self = Self::rgb(255, 0, 0);
    /// Green color.
    pub const GREEN: Self = Self::rgb(0, 255, 0);
    /// Blue color.
    pub const BLUE: Self = Self::rgb(0, 0, 255);
    /// Cyan color.
    pub const CYAN: Self = Self::rgb(0, 255, 255);
    /// Yellow color.
    pub const YELLOW: Self = Self::rgb(255, 255, 0);
    /// Magenta color.
    pub const MAGENTA: Self = Self::rgb(255, 0, 255);
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// A single terminal cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    /// Character in the cell.
    pub ch: char,
    /// Foreground color.
    pub fg: Color,
    /// Background color.
    pub bg: Color,
    /// Bold attribute.
    pub bold: bool,
    /// Underline attribute.
    pub underline: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: Color::WHITE,
            bg: Color::BLACK,
            bold: false,
            underline: false,
        }
    }
}

impl Cell {
    /// Create a new cell with a character.
    pub fn new(ch: char) -> Self {
        Self {
            ch,
            ..Default::default()
        }
    }

    /// Set foreground color.
    pub fn with_fg(mut self, fg: Color) -> Self {
        self.fg = fg;
        self
    }

    /// Set background color.
    pub fn with_bg(mut self, bg: Color) -> Self {
        self.bg = bg;
        self
    }
}

/// Terminal snapshot for testing.
#[derive(Debug, Clone)]
pub struct TerminalSnapshot {
    cells: Vec<Cell>,
    width: u16,
    height: u16,
}

impl TerminalSnapshot {
    /// Create a new empty snapshot.
    pub fn new(width: u16, height: u16) -> Self {
        let cells = vec![Cell::default(); (width as usize) * (height as usize)];
        Self {
            cells,
            width,
            height,
        }
    }

    /// Create snapshot from a string (for testing).
    pub fn from_string(text: &str, width: u16, height: u16) -> Self {
        let mut snapshot = Self::new(width, height);
        for (y, line) in text.lines().enumerate() {
            if y >= height as usize {
                break;
            }
            for (x, ch) in line.chars().enumerate() {
                if x >= width as usize {
                    break;
                }
                snapshot.set(x as u16, y as u16, Cell::new(ch));
            }
        }
        snapshot
    }

    /// Get cell at position.
    pub fn get(&self, x: u16, y: u16) -> Option<&Cell> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        self.cells.get(idx)
    }

    /// Set cell at position.
    pub fn set(&mut self, x: u16, y: u16, cell: Cell) {
        if x < self.width && y < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            self.cells[idx] = cell;
        }
    }

    /// Get snapshot dimensions.
    pub fn dimensions(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// Convert to string (characters only).
    pub fn to_text(&self) -> String {
        let mut result = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(cell) = self.get(x, y) {
                    result.push(cell.ch);
                }
            }
            result.push('\n');
        }
        result
    }

    /// Check if snapshot contains text.
    pub fn contains(&self, text: &str) -> bool {
        self.to_text().contains(text)
    }

    /// Check if snapshot contains all texts.
    pub fn contains_all(&self, texts: &[&str]) -> bool {
        let content = self.to_text();
        texts.iter().all(|t| content.contains(t))
    }

    /// Check if snapshot contains any of the texts.
    pub fn contains_any(&self, texts: &[&str]) -> bool {
        let content = self.to_text();
        texts.iter().any(|t| content.contains(t))
    }

    /// Get foreground color at position.
    pub fn fg_color_at(&self, x: u16, y: u16) -> Option<Color> {
        self.get(x, y).map(|c| c.fg)
    }

    /// Get background color at position.
    pub fn bg_color_at(&self, x: u16, y: u16) -> Option<Color> {
        self.get(x, y).map(|c| c.bg)
    }

    /// Count occurrences of a character.
    pub fn count_char(&self, ch: char) -> usize {
        self.cells.iter().filter(|c| c.ch == ch).count()
    }

    /// Find first occurrence of text, returns (x, y) position.
    pub fn find(&self, text: &str) -> Option<(u16, u16)> {
        let content = self.to_text();
        let pos = content.find(text)?;
        let line_start = content[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let x = pos - line_start;
        let y = content[..pos].matches('\n').count();
        Some((x as u16, y as u16))
    }

    /// Get a rectangular region as a new snapshot.
    pub fn region(&self, x: u16, y: u16, width: u16, height: u16) -> Self {
        let mut result = Self::new(width, height);
        for dy in 0..height {
            for dx in 0..width {
                if let Some(cell) = self.get(x + dx, y + dy) {
                    result.set(dx, dy, cell.clone());
                }
            }
        }
        result
    }
}

impl fmt::Display for TerminalSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Terminal assertion types.
#[derive(Debug, Clone)]
pub enum TerminalAssertion {
    /// Assert text is present.
    Contains(String),
    /// Assert text is not present.
    NotContains(String),
    /// Assert color at position.
    ColorAt {
        /// X coordinate.
        x: u16,
        /// Y coordinate.
        y: u16,
        /// Expected color.
        expected: Color,
    },
    /// Assert character at position.
    CharAt {
        /// X coordinate.
        x: u16,
        /// Y coordinate.
        y: u16,
        /// Expected character.
        expected: char,
    },
    /// Assert region matches text.
    RegionEquals {
        /// X coordinate of region origin.
        x: u16,
        /// Y coordinate of region origin.
        y: u16,
        /// Region width.
        width: u16,
        /// Region height.
        height: u16,
        /// Expected text content.
        expected: String,
    },
}

impl TerminalAssertion {
    /// Check assertion against snapshot.
    pub fn check(&self, snapshot: &TerminalSnapshot) -> Result<(), String> {
        match self {
            Self::Contains(text) => {
                if snapshot.contains(text) {
                    Ok(())
                } else {
                    Err(format!("Expected to contain: {}", text))
                }
            }
            Self::NotContains(text) => {
                if !snapshot.contains(text) {
                    Ok(())
                } else {
                    Err(format!("Expected not to contain: {}", text))
                }
            }
            Self::ColorAt { x, y, expected } => match snapshot.fg_color_at(*x, *y) {
                Some(actual) if actual == *expected => Ok(()),
                Some(actual) => Err(format!(
                    "Color at ({}, {}): expected {}, got {}",
                    x, y, expected, actual
                )),
                None => Err(format!("Position ({}, {}) out of bounds", x, y)),
            },
            Self::CharAt { x, y, expected } => match snapshot.get(*x, *y) {
                Some(cell) if cell.ch == *expected => Ok(()),
                Some(cell) => Err(format!(
                    "Char at ({}, {}): expected '{}', got '{}'",
                    x, y, expected, cell.ch
                )),
                None => Err(format!("Position ({}, {}) out of bounds", x, y)),
            },
            Self::RegionEquals {
                x,
                y,
                width,
                height,
                expected,
            } => {
                let region = snapshot.region(*x, *y, *width, *height);
                let actual = region.to_text().trim_end().to_string();
                let expected = expected.trim_end();
                if actual == expected {
                    Ok(())
                } else {
                    Err(format!(
                        "Region at ({}, {}) {}x{}: expected\n{}\ngot\n{}",
                        x, y, width, height, expected, actual
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#64C8FF").unwrap();
        assert_eq!(color.r, 100);
        assert_eq!(color.g, 200);
        assert_eq!(color.b, 255);
    }

    #[test]
    fn test_color_to_hex() {
        let color = Color::rgb(100, 200, 255);
        assert_eq!(color.to_hex(), "#64C8FF");
    }

    #[test]
    fn test_color_from_hex_invalid() {
        assert!(Color::from_hex("invalid").is_none());
        assert!(Color::from_hex("#12345").is_none());
        assert!(Color::from_hex("#GGGGGG").is_none());
    }

    #[test]
    fn test_cell_default() {
        let cell = Cell::default();
        assert_eq!(cell.ch, ' ');
        assert_eq!(cell.fg, Color::WHITE);
        assert_eq!(cell.bg, Color::BLACK);
    }

    #[test]
    fn test_cell_builder() {
        let cell = Cell::new('A').with_fg(Color::RED).with_bg(Color::BLUE);
        assert_eq!(cell.ch, 'A');
        assert_eq!(cell.fg, Color::RED);
        assert_eq!(cell.bg, Color::BLUE);
    }

    #[test]
    fn test_snapshot_new() {
        let snapshot = TerminalSnapshot::new(80, 24);
        assert_eq!(snapshot.dimensions(), (80, 24));
    }

    #[test]
    fn test_snapshot_from_string() {
        let snapshot = TerminalSnapshot::from_string("Hello\nWorld", 80, 24);
        assert!(snapshot.contains("Hello"));
        assert!(snapshot.contains("World"));
    }

    #[test]
    fn test_snapshot_get_set() {
        let mut snapshot = TerminalSnapshot::new(10, 10);
        snapshot.set(5, 5, Cell::new('X'));
        let cell = snapshot.get(5, 5).unwrap();
        assert_eq!(cell.ch, 'X');
    }

    #[test]
    fn test_snapshot_get_out_of_bounds() {
        let snapshot = TerminalSnapshot::new(10, 10);
        assert!(snapshot.get(100, 100).is_none());
    }

    #[test]
    fn test_snapshot_contains() {
        let snapshot = TerminalSnapshot::from_string("CPU 45%\nMEM 60%", 80, 24);
        assert!(snapshot.contains("CPU"));
        assert!(snapshot.contains("45%"));
        assert!(!snapshot.contains("GPU"));
    }

    #[test]
    fn test_snapshot_contains_all() {
        let snapshot = TerminalSnapshot::from_string("CPU 45%\nMEM 60%", 80, 24);
        assert!(snapshot.contains_all(&["CPU", "MEM"]));
        assert!(!snapshot.contains_all(&["CPU", "GPU"]));
    }

    #[test]
    fn test_snapshot_contains_any() {
        let snapshot = TerminalSnapshot::from_string("CPU 45%", 80, 24);
        assert!(snapshot.contains_any(&["CPU", "GPU"]));
        assert!(!snapshot.contains_any(&["GPU", "DISK"]));
    }

    #[test]
    fn test_snapshot_find() {
        let snapshot = TerminalSnapshot::from_string("Hello World", 80, 24);
        let pos = snapshot.find("World").unwrap();
        assert_eq!(pos, (6, 0));
    }

    #[test]
    fn test_snapshot_count_char() {
        let snapshot = TerminalSnapshot::from_string("AAABBC", 80, 24);
        assert_eq!(snapshot.count_char('A'), 3);
        assert_eq!(snapshot.count_char('B'), 2);
        assert_eq!(snapshot.count_char('C'), 1);
    }

    #[test]
    fn test_snapshot_region() {
        let snapshot = TerminalSnapshot::from_string("ABCD\nEFGH\nIJKL", 80, 24);
        let region = snapshot.region(1, 1, 2, 2);
        assert!(region.contains("FG"));
    }

    #[test]
    fn test_assertion_contains() {
        let snapshot = TerminalSnapshot::from_string("Hello", 80, 24);
        let assertion = TerminalAssertion::Contains("Hello".into());
        assert!(assertion.check(&snapshot).is_ok());

        let assertion = TerminalAssertion::Contains("World".into());
        assert!(assertion.check(&snapshot).is_err());
    }

    #[test]
    fn test_assertion_not_contains() {
        let snapshot = TerminalSnapshot::from_string("Hello", 80, 24);
        let assertion = TerminalAssertion::NotContains("World".into());
        assert!(assertion.check(&snapshot).is_ok());

        let assertion = TerminalAssertion::NotContains("Hello".into());
        assert!(assertion.check(&snapshot).is_err());
    }

    #[test]
    fn test_assertion_color_at() {
        let mut snapshot = TerminalSnapshot::new(10, 10);
        snapshot.set(5, 5, Cell::new('X').with_fg(Color::RED));

        let assertion = TerminalAssertion::ColorAt {
            x: 5,
            y: 5,
            expected: Color::RED,
        };
        assert!(assertion.check(&snapshot).is_ok());

        let assertion = TerminalAssertion::ColorAt {
            x: 5,
            y: 5,
            expected: Color::BLUE,
        };
        assert!(assertion.check(&snapshot).is_err());
    }

    #[test]
    fn test_assertion_char_at() {
        let snapshot = TerminalSnapshot::from_string("ABC", 80, 24);
        let assertion = TerminalAssertion::CharAt {
            x: 1,
            y: 0,
            expected: 'B',
        };
        assert!(assertion.check(&snapshot).is_ok());
    }

    #[test]
    fn test_color_display() {
        let color = Color::rgb(100, 200, 255);
        assert_eq!(format!("{}", color), "#64C8FF");
    }

    #[test]
    fn test_snapshot_display() {
        let snapshot = TerminalSnapshot::from_string("Test", 10, 1);
        let display = format!("{}", snapshot);
        assert!(display.contains("Test"));
    }
}
