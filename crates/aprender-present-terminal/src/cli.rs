//! CLI utilities for simple terminal output.
//!
//! This module provides lightweight utilities for CLI applications that don't
//! need the full TUI infrastructure. The primary use case is showing loading
//! spinners and progress indicators in command-line tools.
//!
//! # Example
//!
//! ```no_run
//! use presentar_terminal::cli::Spinner;
//!
//! // Start spinner while loading
//! let spinner = Spinner::new().start();
//!
//! // Do some work...
//! std::thread::sleep(std::time::Duration::from_secs(2));
//!
//! // Stop and clear spinner
//! spinner.stop();
//! println!("Done!");
//! ```

use crossterm::{
    cursor::{Hide, MoveToColumn, Show},
    execute,
    style::Print,
    terminal::{Clear, ClearType},
};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Spinner animation frames.
///
/// Each style provides a sequence of characters that animate in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpinnerStyle {
    /// Braille dots animation: ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
    #[default]
    Dots,
    /// Line animation: | / - \
    Line,
    /// Growing dots: .  .. ...
    Growing,
    /// Arc animation: ◐ ◓ ◑ ◒
    Arc,
    /// Bounce animation: ⠁ ⠂ ⠄ ⠂
    Bounce,
}

impl SpinnerStyle {
    /// Get the animation frames for this style.
    #[must_use]
    pub const fn frames(&self) -> &'static [&'static str] {
        match self {
            Self::Dots => &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            Self::Line => &["|", "/", "-", "\\"],
            Self::Growing => &[".  ", ".. ", "..."],
            Self::Arc => &["◐", "◓", "◑", "◒"],
            Self::Bounce => &["⠁", "⠂", "⠄", "⠂"],
        }
    }

    /// Get the recommended interval between frames in milliseconds.
    #[must_use]
    pub const fn interval_ms(&self) -> u64 {
        match self {
            Self::Dots => 80,
            Self::Line => 100,
            Self::Growing => 300,
            Self::Arc => 100,
            Self::Bounce => 120,
        }
    }
}

/// A simple CLI spinner for indicating loading/progress.
///
/// The spinner runs in a background thread and can be stopped at any time.
/// When stopped, it clears its output so the terminal is clean.
///
/// # Example
///
/// ```no_run
/// use presentar_terminal::cli::Spinner;
///
/// let spinner = Spinner::new().start();
/// // ... do work ...
/// spinner.stop();
/// ```
#[derive(Debug)]
pub struct Spinner {
    style: SpinnerStyle,
    message: Option<String>,
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}

impl Spinner {
    /// Create a new spinner with default style (Dots).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            style: SpinnerStyle::Dots,
            message: None,
        }
    }

    /// Set the spinner animation style.
    #[must_use]
    pub const fn style(mut self, style: SpinnerStyle) -> Self {
        self.style = style;
        self
    }

    /// Set an optional message to display after the spinner.
    #[must_use]
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Start the spinner animation in a background thread.
    ///
    /// Returns a `SpinnerHandle` that can be used to stop the spinner.
    pub fn start(self) -> SpinnerHandle {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let frames = self.style.frames();
        let interval = Duration::from_millis(self.style.interval_ms());
        let message = self.message;

        let handle = thread::spawn(move || {
            let mut stdout = io::stdout();
            let mut frame_idx = 0;

            // Hide cursor while spinning
            let _ = execute!(stdout, Hide);

            while running_clone.load(Ordering::Relaxed) {
                let frame = frames[frame_idx % frames.len()];

                // Clear line and print frame
                let _ = execute!(
                    stdout,
                    MoveToColumn(0),
                    Clear(ClearType::CurrentLine),
                    Print(frame)
                );

                if let Some(ref msg) = message {
                    let _ = execute!(stdout, Print(" "), Print(msg));
                }

                let _ = stdout.flush();

                frame_idx = frame_idx.wrapping_add(1);
                thread::sleep(interval);
            }

            // Clean up: clear line and show cursor
            let _ = execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine), Show);
            let _ = stdout.flush();
        });

        SpinnerHandle {
            running,
            handle: Some(handle),
        }
    }
}

/// Handle to a running spinner, used to stop it.
///
/// When dropped, the spinner is automatically stopped.
#[derive(Debug)]
pub struct SpinnerHandle {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl SpinnerHandle {
    /// Stop the spinner and clear its output.
    ///
    /// This blocks until the spinner thread has finished.
    pub fn stop(mut self) {
        self.stop_internal();
    }

    /// Stop the spinner and print a final message in its place.
    pub fn stop_with_message(mut self, msg: &str) {
        self.stop_internal();
        println!("{msg}");
    }

    fn stop_internal(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SpinnerHandle {
    fn drop(&mut self) {
        self.stop_internal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spinner_style_frames() {
        assert!(!SpinnerStyle::Dots.frames().is_empty());
        assert!(!SpinnerStyle::Line.frames().is_empty());
        assert!(!SpinnerStyle::Growing.frames().is_empty());
        assert!(!SpinnerStyle::Arc.frames().is_empty());
        assert!(!SpinnerStyle::Bounce.frames().is_empty());
    }

    #[test]
    fn test_spinner_style_interval() {
        assert!(SpinnerStyle::Dots.interval_ms() > 0);
        assert!(SpinnerStyle::Line.interval_ms() > 0);
    }

    #[test]
    fn test_spinner_builder() {
        let spinner = Spinner::new()
            .style(SpinnerStyle::Line)
            .message("Loading...");

        assert_eq!(spinner.style, SpinnerStyle::Line);
        assert_eq!(spinner.message, Some("Loading...".to_string()));
    }

    #[test]
    fn test_spinner_default() {
        let spinner = Spinner::default();
        assert_eq!(spinner.style, SpinnerStyle::Dots);
        assert!(spinner.message.is_none());
    }

    #[test]
    fn test_spinner_start_stop() {
        // Quick test that spinner can start and stop without panicking
        let handle = Spinner::new().start();
        std::thread::sleep(Duration::from_millis(100));
        handle.stop();
    }

    #[test]
    fn test_spinner_drop_stops() {
        // Test that dropping the handle stops the spinner
        {
            let _handle = Spinner::new().start();
            std::thread::sleep(Duration::from_millis(50));
            // handle dropped here
        }
        // If we get here without hanging, the test passes
    }
}
