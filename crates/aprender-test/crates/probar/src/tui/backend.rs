//! TUI Test Backend for Frame Capture
//!
//! Provides a test backend that captures frames for assertion.
//! Uses TextGrid instead of ratatui::Buffer for zero external dependencies.
//!
//! ## EXTREME TDD: Tests written FIRST per spec

use super::buffer::TextGrid;
use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A captured TUI frame for testing
#[derive(Clone, Serialize, Deserialize)]
pub struct TuiFrame {
    /// The buffer content
    content: Vec<String>,
    /// Frame width
    width: u16,
    /// Frame height
    height: u16,
    /// Timestamp when captured (milliseconds from test start)
    timestamp_ms: u64,
}

impl TuiFrame {
    /// Create a new TUI frame from a TextGrid
    #[must_use]
    pub fn from_grid(grid: &TextGrid, timestamp_ms: u64) -> Self {
        Self {
            content: grid.to_lines(),
            width: grid.width(),
            height: grid.height(),
            timestamp_ms,
        }
    }

    /// Create a frame from raw text lines
    #[must_use]
    pub fn from_lines(lines: &[&str]) -> Self {
        let height = lines.len() as u16;
        // Count characters, not bytes (important for Unicode TUI content)
        let width = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as u16;
        let content = lines.iter().map(|s| (*s).to_string()).collect();

        Self {
            content,
            width,
            height,
            timestamp_ms: 0,
        }
    }

    /// Get the frame width
    #[must_use]
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the frame height
    #[must_use]
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Get the timestamp
    #[must_use]
    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }

    /// Get the frame content as lines
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.content
    }

    /// Get the full frame content as a single string
    #[must_use]
    pub fn as_text(&self) -> String {
        self.content.join("\n")
    }

    /// Check if the frame contains a substring
    #[must_use]
    pub fn contains(&self, text: &str) -> bool {
        self.content.iter().any(|line| line.contains(text))
    }

    /// Check if the frame matches a regex pattern
    #[must_use]
    pub fn matches(&self, pattern: &str) -> ProbarResult<bool> {
        let re = regex::Regex::new(pattern).map_err(|e| ProbarError::TuiError {
            message: format!("Invalid regex pattern: {e}"),
        })?;
        Ok(self.content.iter().any(|line| re.is_match(line)))
    }

    /// Find all lines matching a pattern
    #[must_use]
    pub fn find_matches(&self, pattern: &str) -> ProbarResult<Vec<&str>> {
        let re = regex::Regex::new(pattern).map_err(|e| ProbarError::TuiError {
            message: format!("Invalid regex pattern: {e}"),
        })?;
        Ok(self
            .content
            .iter()
            .filter(|line| re.is_match(line))
            .map(String::as_str)
            .collect())
    }

    /// Get a specific line by index
    #[must_use]
    pub fn line(&self, index: usize) -> Option<&str> {
        self.content.get(index).map(String::as_str)
    }

    /// Check if two frames are identical
    #[must_use]
    pub fn is_identical(&self, other: &TuiFrame) -> bool {
        self.content == other.content
    }

    /// Get the difference between two frames
    #[must_use]
    pub fn diff(&self, other: &TuiFrame) -> FrameDiff {
        let mut changed_lines = Vec::new();

        let max_lines = self.content.len().max(other.content.len());
        for i in 0..max_lines {
            let self_line = self.content.get(i).map(String::as_str).unwrap_or("");
            let other_line = other.content.get(i).map(String::as_str).unwrap_or("");

            if self_line != other_line {
                changed_lines.push(LineDiff {
                    line_number: i,
                    expected: self_line.to_string(),
                    actual: other_line.to_string(),
                });
            }
        }

        FrameDiff {
            is_identical: changed_lines.is_empty(),
            changed_lines,
        }
    }
}

impl fmt::Debug for TuiFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TuiFrame({}x{}):", self.width, self.height)?;
        for (i, line) in self.content.iter().enumerate() {
            writeln!(f, "  {i:3}: {line}")?;
        }
        Ok(())
    }
}

/// Difference between two frames
#[derive(Debug, Clone)]
pub struct FrameDiff {
    /// Whether frames are identical
    pub is_identical: bool,
    /// Lines that differ
    pub changed_lines: Vec<LineDiff>,
}

/// A single line difference
#[derive(Debug, Clone)]
pub struct LineDiff {
    /// Line number (0-indexed)
    pub line_number: usize,
    /// Expected content
    pub expected: String,
    /// Actual content
    pub actual: String,
}

impl fmt::Display for FrameDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identical {
            write!(f, "Frames are identical")
        } else {
            writeln!(f, "Frame differences:")?;
            for diff in &self.changed_lines {
                writeln!(f, "  Line {}: ", diff.line_number)?;
                writeln!(f, "    Expected: {:?}", diff.expected)?;
                writeln!(f, "    Actual:   {:?}", diff.actual)?;
            }
            Ok(())
        }
    }
}

/// TUI Test Backend for capturing frames
///
/// Provides a text grid and frame capture functionality for testing terminal UIs.
/// Uses TextGrid instead of ratatui::Buffer for zero external dependencies.
#[derive(Debug)]
pub struct TuiTestBackend {
    grid: TextGrid,
    frames: Vec<TuiFrame>,
    start_time: std::time::Instant,
}

impl TuiTestBackend {
    /// Create a new test backend with the given dimensions
    #[must_use]
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            grid: TextGrid::new(width, height),
            frames: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Get the backend dimensions
    #[must_use]
    pub fn size(&self) -> (u16, u16) {
        (self.grid.width(), self.grid.height())
    }

    /// Get a reference to the underlying grid
    #[must_use]
    pub fn grid(&self) -> &TextGrid {
        &self.grid
    }

    /// Get a mutable reference to the underlying grid
    #[must_use]
    pub fn grid_mut(&mut self) -> &mut TextGrid {
        &mut self.grid
    }

    /// Capture the current frame
    pub fn capture_frame(&mut self) -> TuiFrame {
        let timestamp = self.start_time.elapsed().as_millis() as u64;
        let frame = TuiFrame::from_grid(&self.grid, timestamp);
        self.frames.push(frame.clone());
        frame
    }

    /// Get the current frame without storing it
    #[must_use]
    pub fn current_frame(&self) -> TuiFrame {
        let timestamp = self.start_time.elapsed().as_millis() as u64;
        TuiFrame::from_grid(&self.grid, timestamp)
    }

    /// Get all captured frames
    #[must_use]
    pub fn frames(&self) -> &[TuiFrame] {
        &self.frames
    }

    /// Get the number of captured frames
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Clear the grid
    pub fn clear(&mut self) {
        self.grid.clear();
    }

    /// Reset the backend (clear grid and frames)
    pub fn reset(&mut self) {
        self.grid.clear();
        self.frames.clear();
        self.start_time = std::time::Instant::now();
    }

    /// Resize the backend
    pub fn resize(&mut self, width: u16, height: u16) {
        self.grid.resize(width, height);
    }

    /// Write text at a position (for testing)
    pub fn write_text(&mut self, x: u16, y: u16, text: &str) {
        self.grid.write_str(x, y, text);
    }

    /// Write multiple lines starting at a position
    pub fn write_lines(&mut self, x: u16, y: u16, lines: &[&str]) {
        for (i, line) in lines.iter().enumerate() {
            self.grid.write_str(x, y + i as u16, line);
        }
    }
}

impl Default for TuiTestBackend {
    fn default() -> Self {
        Self::new(80, 24) // Standard terminal size
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod tui_frame_tests {
        use super::*;

        #[test]
        fn test_from_lines() {
            let frame = TuiFrame::from_lines(&["Hello", "World"]);
            assert_eq!(frame.width(), 5);
            assert_eq!(frame.height(), 2);
            assert_eq!(frame.lines(), &["Hello", "World"]);
        }

        #[test]
        fn test_from_grid() {
            let mut grid = TextGrid::new(10, 3);
            grid.write_str(0, 0, "Hello");
            grid.write_str(0, 1, "World");

            let frame = TuiFrame::from_grid(&grid, 100);
            assert_eq!(frame.width(), 10);
            assert_eq!(frame.height(), 3);
            assert_eq!(frame.timestamp_ms(), 100);
            assert!(frame.contains("Hello"));
            assert!(frame.contains("World"));
        }

        #[test]
        fn test_as_text() {
            let frame = TuiFrame::from_lines(&["Line 1", "Line 2"]);
            assert_eq!(frame.as_text(), "Line 1\nLine 2");
        }

        #[test]
        fn test_contains() {
            let frame = TuiFrame::from_lines(&["Hello World", "Goodbye"]);
            assert!(frame.contains("World"));
            assert!(frame.contains("Goodbye"));
            assert!(!frame.contains("Missing"));
        }

        #[test]
        fn test_matches_regex() {
            let frame = TuiFrame::from_lines(&["Score: 100", "Lives: 3"]);
            assert!(frame.matches(r"Score: \d+").unwrap());
            assert!(frame.matches(r"Lives: \d").unwrap());
            assert!(!frame.matches(r"Health: \d+").unwrap());
        }

        #[test]
        fn test_find_matches() {
            let frame = TuiFrame::from_lines(&["Error: failed", "Warning: slow", "Info: ok"]);
            let errors = frame.find_matches(r"Error:.*").unwrap();
            assert_eq!(errors.len(), 1);
            assert_eq!(errors[0], "Error: failed");
        }

        #[test]
        fn test_line_access() {
            let frame = TuiFrame::from_lines(&["First", "Second", "Third"]);
            assert_eq!(frame.line(0), Some("First"));
            assert_eq!(frame.line(1), Some("Second"));
            assert_eq!(frame.line(2), Some("Third"));
            assert_eq!(frame.line(3), None);
        }

        #[test]
        fn test_is_identical() {
            let frame1 = TuiFrame::from_lines(&["Same", "Content"]);
            let frame2 = TuiFrame::from_lines(&["Same", "Content"]);
            let frame3 = TuiFrame::from_lines(&["Different", "Content"]);

            assert!(frame1.is_identical(&frame2));
            assert!(!frame1.is_identical(&frame3));
        }

        #[test]
        fn test_diff() {
            let frame1 = TuiFrame::from_lines(&["Same", "Different1"]);
            let frame2 = TuiFrame::from_lines(&["Same", "Different2"]);

            let diff = frame1.diff(&frame2);
            assert!(!diff.is_identical);
            assert_eq!(diff.changed_lines.len(), 1);
            assert_eq!(diff.changed_lines[0].line_number, 1);
            assert_eq!(diff.changed_lines[0].expected, "Different1");
            assert_eq!(diff.changed_lines[0].actual, "Different2");
        }

        #[test]
        fn test_diff_identical() {
            let frame1 = TuiFrame::from_lines(&["Same", "Same"]);
            let frame2 = TuiFrame::from_lines(&["Same", "Same"]);

            let diff = frame1.diff(&frame2);
            assert!(diff.is_identical);
            assert!(diff.changed_lines.is_empty());
        }
    }

    mod tui_test_backend_tests {
        use super::*;

        #[test]
        fn test_new() {
            let backend = TuiTestBackend::new(80, 24);
            assert_eq!(backend.size(), (80, 24));
            assert_eq!(backend.frame_count(), 0);
        }

        #[test]
        fn test_default() {
            let backend = TuiTestBackend::default();
            assert_eq!(backend.size(), (80, 24));
        }

        #[test]
        fn test_write_text() {
            let mut backend = TuiTestBackend::new(20, 5);
            backend.write_text(0, 0, "Hello");

            let frame = backend.current_frame();
            assert!(frame.contains("Hello"));
        }

        #[test]
        fn test_write_lines() {
            let mut backend = TuiTestBackend::new(20, 5);
            backend.write_lines(0, 0, &["Line 1", "Line 2"]);

            let frame = backend.current_frame();
            assert!(frame.contains("Line 1"));
            assert!(frame.contains("Line 2"));
        }

        #[test]
        fn test_capture_frame() {
            let mut backend = TuiTestBackend::new(20, 5);
            backend.write_text(0, 0, "Test");

            let frame = backend.capture_frame();
            assert!(frame.contains("Test"));
            assert_eq!(backend.frame_count(), 1);

            backend.write_text(0, 1, "More");
            let _ = backend.capture_frame();
            assert_eq!(backend.frame_count(), 2);
        }

        #[test]
        fn test_frames() {
            let mut backend = TuiTestBackend::new(20, 5);

            backend.write_text(0, 0, "Frame1");
            let _ = backend.capture_frame();

            backend.write_text(0, 1, "Frame2");
            let _ = backend.capture_frame();

            let frames = backend.frames();
            assert_eq!(frames.len(), 2);
            assert!(frames[0].contains("Frame1"));
            assert!(frames[1].contains("Frame2"));
        }

        #[test]
        fn test_clear() {
            let mut backend = TuiTestBackend::new(20, 5);
            backend.write_text(0, 0, "Hello");
            backend.clear();

            let frame = backend.current_frame();
            assert!(!frame.contains("Hello"));
        }

        #[test]
        fn test_reset() {
            let mut backend = TuiTestBackend::new(20, 5);
            backend.write_text(0, 0, "Hello");
            let _ = backend.capture_frame();

            backend.reset();
            assert_eq!(backend.frame_count(), 0);
            assert!(!backend.current_frame().contains("Hello"));
        }

        #[test]
        fn test_resize() {
            let mut backend = TuiTestBackend::new(20, 5);
            backend.resize(40, 10);
            assert_eq!(backend.size(), (40, 10));
        }

        #[test]
        fn test_grid_access() {
            let mut backend = TuiTestBackend::new(20, 5);

            // Test grid() accessor
            assert_eq!(backend.grid().width(), 20);

            // Test grid_mut() accessor
            backend.grid_mut().set(0, 0, 'X');
            assert_eq!(backend.grid().get(0, 0), Some('X'));
        }
    }
}
