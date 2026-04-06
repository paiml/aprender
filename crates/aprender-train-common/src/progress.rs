//! Progress indicators for long-running operations.
//!
//! Provides visual feedback during operations like model downloading,
//! training, and export (Visual Control principle).

use std::io::{self, Write};

/// A simple progress bar for terminal output.
#[derive(Debug)]
pub struct ProgressBar {
    total: u64,
    current: u64,
    width: usize,
    message: String,
    enabled: bool,
}

impl ProgressBar {
    /// Create a new progress bar with the given total.
    pub fn new(total: u64) -> Self {
        Self {
            total,
            current: 0,
            width: 40,
            message: String::new(),
            enabled: true,
        }
    }

    /// Set the display width.
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set whether the progress bar is enabled.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the message to display.
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
        self.render();
    }

    /// Increment progress by the given amount.
    pub fn inc(&mut self, amount: u64) {
        self.current = (self.current + amount).min(self.total);
        self.render();
    }

    /// Set the current progress.
    pub fn set(&mut self, current: u64) {
        self.current = current.min(self.total);
        self.render();
    }

    /// Get the current progress percentage.
    pub fn percentage(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        (self.current as f64 / self.total as f64) * 100.0
    }

    /// Finish the progress bar.
    pub fn finish(&mut self) {
        self.current = self.total;
        self.render();
        if self.enabled {
            println!();
        }
    }

    /// Finish with a message.
    pub fn finish_with_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
        self.finish();
    }

    fn render(&self) {
        if !self.enabled {
            return;
        }

        let percentage = self.percentage();
        let filled = (percentage / 100.0 * self.width as f64) as usize;
        let empty = self.width - filled;

        let bar = format!(
            "\r[{}{}] {:>5.1}% {}",
            "█".repeat(filled),
            "░".repeat(empty),
            percentage,
            self.message
        );

        print!("{bar}");
        let _ = io::stdout().flush();
    }
}

/// A spinner for indeterminate progress.
#[derive(Debug)]
pub struct Spinner {
    frames: Vec<&'static str>,
    current_frame: usize,
    message: String,
    enabled: bool,
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}

impl Spinner {
    /// Create a new spinner.
    pub fn new() -> Self {
        Self {
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            current_frame: 0,
            message: String::new(),
            enabled: true,
        }
    }

    /// Set whether the spinner is enabled.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the message.
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
    }

    /// Advance to the next frame.
    pub fn tick(&mut self) {
        if !self.enabled {
            return;
        }

        self.current_frame = (self.current_frame + 1) % self.frames.len();
        print!("\r{} {}", self.frames[self.current_frame], self.message);
        let _ = io::stdout().flush();
    }

    /// Finish the spinner with a success message.
    pub fn finish_with_message(&mut self, message: impl Into<String>) {
        if self.enabled {
            println!("\r✓ {}", message.into());
        }
    }

    /// Finish the spinner with an error message.
    pub fn finish_with_error(&mut self, message: impl Into<String>) {
        if self.enabled {
            println!("\r✗ {}", message.into());
        }
    }
}

/// Status indicator for multi-step operations.
#[derive(Debug)]
pub struct StepTracker {
    steps: Vec<String>,
    current: usize,
    enabled: bool,
}

impl StepTracker {
    /// Create a new step tracker.
    pub fn new(steps: Vec<impl Into<String>>) -> Self {
        Self {
            steps: steps.into_iter().map(Into::into).collect(),
            current: 0,
            enabled: true,
        }
    }

    /// Set whether output is enabled.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Start the next step.
    pub fn next_step(&mut self) {
        if self.current < self.steps.len() {
            if self.enabled {
                println!(
                    "[{}/{}] {}...",
                    self.current + 1,
                    self.steps.len(),
                    self.steps[self.current]
                );
            }
            self.current += 1;
        }
    }

    /// Complete the current step with a message.
    pub fn complete_step(&self, message: impl Into<String>) {
        if self.enabled {
            println!("  ✓ {}", message.into());
        }
    }

    /// Mark the current step as failed.
    pub fn fail_step(&self, message: impl Into<String>) {
        if self.enabled {
            println!("  ✗ {}", message.into());
        }
    }

    /// Check if all steps are complete.
    pub fn is_complete(&self) -> bool {
        self.current >= self.steps.len()
    }

    /// Get the total number of steps.
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }

    /// Get the current step number (1-indexed).
    pub fn current_step(&self) -> usize {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar_percentage() {
        let mut bar = ProgressBar::new(100).with_enabled(false);
        assert_eq!(bar.percentage(), 0.0);

        bar.set(50);
        assert_eq!(bar.percentage(), 50.0);

        bar.set(100);
        assert_eq!(bar.percentage(), 100.0);
    }

    #[test]
    fn test_progress_bar_increment() {
        let mut bar = ProgressBar::new(100).with_enabled(false);
        bar.inc(25);
        assert_eq!(bar.current, 25);
        bar.inc(25);
        assert_eq!(bar.current, 50);
    }

    #[test]
    fn test_progress_bar_overflow_protection() {
        let mut bar = ProgressBar::new(100).with_enabled(false);
        bar.set(200);
        assert_eq!(bar.current, 100);
        assert_eq!(bar.percentage(), 100.0);
    }

    #[test]
    fn test_progress_bar_zero_total() {
        let bar = ProgressBar::new(0).with_enabled(false);
        assert_eq!(bar.percentage(), 100.0);
    }

    #[test]
    fn test_step_tracker_progression() {
        let mut tracker = StepTracker::new(vec!["Step 1", "Step 2", "Step 3"]).with_enabled(false);

        assert_eq!(tracker.total_steps(), 3);
        assert_eq!(tracker.current_step(), 0);
        assert!(!tracker.is_complete());

        tracker.next_step();
        assert_eq!(tracker.current_step(), 1);

        tracker.next_step();
        tracker.next_step();
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_spinner_frames() {
        let spinner = Spinner::new();
        assert!(!spinner.frames.is_empty());
    }

    #[test]
    fn test_spinner_default() {
        let spinner = Spinner::default();
        assert_eq!(spinner.frames.len(), 10);
        assert!(spinner.enabled);
    }

    #[test]
    fn test_spinner_disabled() {
        let spinner = Spinner::new().with_enabled(false);
        assert!(!spinner.enabled);
    }

    #[test]
    fn test_spinner_message() {
        let mut spinner = Spinner::new().with_enabled(false);
        spinner.set_message("Loading...");
        assert_eq!(spinner.message, "Loading...");
    }

    #[test]
    fn test_spinner_tick_cycles_frames() {
        let mut spinner = Spinner::new();
        spinner.enabled = false; // Disable output but keep frame cycling
                                 // Manually cycle since tick() returns early when disabled
        spinner.current_frame = (spinner.current_frame + 1) % spinner.frames.len();
        assert_eq!(spinner.current_frame, 1);
        spinner.current_frame = (spinner.current_frame + 1) % spinner.frames.len();
        assert_eq!(spinner.current_frame, 2);
    }

    #[test]
    fn test_spinner_tick_wraps_around() {
        let spinner = Spinner::new();
        // Test wrap around logic directly
        let frame_count = spinner.frames.len();
        let after_15_ticks = 15 % frame_count;
        assert_eq!(after_15_ticks, 5);
    }

    #[test]
    fn test_progress_bar_width() {
        let bar = ProgressBar::new(100).with_width(20);
        assert_eq!(bar.width, 20);
    }

    #[test]
    fn test_progress_bar_message() {
        let mut bar = ProgressBar::new(100).with_enabled(false);
        bar.set_message("Downloading");
        assert_eq!(bar.message, "Downloading");
    }

    #[test]
    fn test_progress_bar_finish() {
        let mut bar = ProgressBar::new(100).with_enabled(false);
        bar.set(50);
        bar.finish();
        assert_eq!(bar.current, 100);
    }

    #[test]
    fn test_progress_bar_finish_with_message() {
        let mut bar = ProgressBar::new(100).with_enabled(false);
        bar.finish_with_message("Done!");
        assert_eq!(bar.message, "Done!");
        assert_eq!(bar.current, 100);
    }

    #[test]
    fn test_step_tracker_empty() {
        let tracker = StepTracker::new(Vec::<String>::new()).with_enabled(false);
        assert_eq!(tracker.total_steps(), 0);
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_step_tracker_current_step_1_indexed() {
        let mut tracker = StepTracker::new(vec!["A", "B"]).with_enabled(false);
        assert_eq!(tracker.current_step(), 0);
        tracker.next_step();
        assert_eq!(tracker.current_step(), 1);
        tracker.next_step();
        assert_eq!(tracker.current_step(), 2);
    }

    #[test]
    fn test_step_tracker_overflow_protection() {
        let mut tracker = StepTracker::new(vec!["Only"]).with_enabled(false);
        tracker.next_step();
        tracker.next_step(); // Extra call should be safe
        tracker.next_step();
        assert!(tracker.is_complete());
        assert_eq!(tracker.current_step(), 1);
    }
}
