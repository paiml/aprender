//! TUI Application State
//!
//! Probar: Error prevention - State machine prevents invalid transitions

use crate::core::evaluator::Evaluator;
use crate::core::history::History;
use crate::core::{AnomalyValidator, CalcError};

/// Calculator application state
#[derive(Debug)]
pub struct CalculatorApp {
    /// Current input expression
    input: String,
    /// Current cursor position in input
    cursor: usize,
    /// Last calculation result
    result: Option<Result<f64, CalcError>>,
    /// Calculation history
    history: History,
    /// Expression evaluator
    evaluator: Evaluator,
    /// Whether the app should quit
    should_quit: bool,
}

impl Default for CalculatorApp {
    fn default() -> Self {
        Self::new()
    }
}

impl CalculatorApp {
    /// Creates a new calculator app with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            result: None,
            history: History::new(),
            evaluator: Evaluator::new(),
            should_quit: false,
        }
    }

    /// Creates a calculator app with custom Anomaly validator
    #[must_use]
    pub fn with_validator(validator: AnomalyValidator) -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            result: None,
            history: History::new(),
            evaluator: Evaluator::with_validator(validator),
            should_quit: false,
        }
    }

    /// Returns the current input string
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Returns the cursor position
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Returns the last result
    #[must_use]
    pub fn result(&self) -> Option<&Result<f64, CalcError>> {
        self.result.as_ref()
    }

    /// Returns the calculation history
    #[must_use]
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Returns whether the app should quit
    #[must_use]
    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Sets the quit flag
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Sets the input string directly (for testing)
    pub fn set_input(&mut self, input: &str) {
        self.input = input.to_string();
        self.cursor = self.input.len();
    }

    /// Sets the cursor position directly (for testing)
    pub fn set_cursor(&mut self, pos: usize) {
        self.cursor = pos.min(self.input.len());
    }

    /// Inserts a character at the cursor position
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += 1;
    }

    /// Deletes the character before the cursor (backspace)
    pub fn delete_char(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.input.remove(self.cursor);
        }
    }

    /// Deletes the character at the cursor (delete key)
    pub fn delete_char_forward(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    /// Moves the cursor left
    pub fn move_cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    /// Moves the cursor right
    pub fn move_cursor_right(&mut self) {
        if self.cursor < self.input.len() {
            self.cursor += 1;
        }
    }

    /// Moves cursor to the beginning of input
    pub fn move_cursor_start(&mut self) {
        self.cursor = 0;
    }

    /// Moves cursor to the end of input
    pub fn move_cursor_end(&mut self) {
        self.cursor = self.input.len();
    }

    /// Clears the input
    pub fn clear(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.result = None;
    }

    /// Clears everything including history
    pub fn clear_all(&mut self) {
        self.clear();
        self.history.clear();
    }

    /// Evaluates the current input expression
    pub fn evaluate(&mut self) {
        if self.input.is_empty() {
            return;
        }

        let result = self.evaluator.evaluate_str(&self.input);

        // Record successful calculations in history
        if let Ok(value) = &result {
            self.history.record(&self.input, *value);
        }

        self.result = Some(result);
    }

    /// Returns the result as a display string
    #[must_use]
    pub fn result_display(&self) -> String {
        match &self.result {
            None => String::new(),
            Some(Ok(value)) => format_result(*value),
            Some(Err(e)) => format!("Error: {e}"),
        }
    }

    /// Loads the last history entry into input
    pub fn recall_last(&mut self) {
        if let Some(entry) = self.history.last() {
            self.input = entry.expression.clone();
            self.cursor = self.input.len();
        }
    }

    /// Gets the Anomaly status as a display string
    #[must_use]
    pub fn jidoka_status(&self) -> Vec<String> {
        let mut status = Vec::new();

        match &self.result {
            None => {
                status.push("Ready".into());
            }
            Some(Ok(_)) => {
                status.push("✓ No NaN detected".into());
                status.push("✓ No overflow".into());
                status.push("✓ All invariants satisfied".into());
            }
            Some(Err(CalcError::AnomalyViolation(v))) => {
                status.push(format!("✗ Anomaly violation: {v}"));
            }
            Some(Err(e)) => {
                status.push(format!("✗ Error: {e}"));
            }
        }

        status
    }
}

/// Formats a result value for display
fn format_result(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{:.0}", value)
    } else {
        // Limit decimal places for readability
        let formatted = format!("{:.10}", value);
        // Trim trailing zeros
        let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Constructor tests =====

    #[test]
    fn test_app_new() {
        let app = CalculatorApp::new();
        assert!(app.input().is_empty());
        assert_eq!(app.cursor(), 0);
        assert!(app.result().is_none());
        assert!(app.history().is_empty());
        assert!(!app.should_quit());
    }

    #[test]
    fn test_app_default() {
        let app = CalculatorApp::default();
        assert!(app.input().is_empty());
    }

    #[test]
    fn test_app_with_validator() {
        let validator = AnomalyValidator::with_max_magnitude(100.0);
        let mut app = CalculatorApp::with_validator(validator);
        app.set_input("50 * 3");
        app.evaluate();
        assert!(matches!(
            app.result(),
            Some(Err(CalcError::AnomalyViolation(_)))
        ));
    }

    // ===== Input manipulation tests =====

    #[test]
    fn test_set_input() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 2");
        assert_eq!(app.input(), "2 + 2");
        assert_eq!(app.cursor(), 5);
    }

    #[test]
    fn test_insert_char() {
        let mut app = CalculatorApp::new();
        app.insert_char('1');
        app.insert_char('+');
        app.insert_char('2');
        assert_eq!(app.input(), "1+2");
        assert_eq!(app.cursor(), 3);
    }

    #[test]
    fn test_insert_char_in_middle() {
        let mut app = CalculatorApp::new();
        app.set_input("12");
        app.set_cursor(1); // Between 1 and 2
        app.insert_char('+');
        assert_eq!(app.input(), "1+2");
    }

    #[test]
    fn test_delete_char() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.delete_char();
        assert_eq!(app.input(), "12");
        assert_eq!(app.cursor(), 2);
    }

    #[test]
    fn test_delete_char_at_start() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.set_cursor(0);
        app.delete_char(); // Should do nothing
        assert_eq!(app.input(), "123");
    }

    #[test]
    fn test_delete_char_forward() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.set_cursor(0);
        app.delete_char_forward();
        assert_eq!(app.input(), "23");
    }

    #[test]
    fn test_delete_char_forward_at_end() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.delete_char_forward(); // Cursor at end, nothing to delete
        assert_eq!(app.input(), "123");
    }

    // ===== Cursor movement tests =====

    #[test]
    fn test_move_cursor_left() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.move_cursor_left();
        assert_eq!(app.cursor(), 2);
    }

    #[test]
    fn test_move_cursor_left_at_start() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.set_cursor(0);
        app.move_cursor_left(); // Should stay at 0
        assert_eq!(app.cursor(), 0);
    }

    #[test]
    fn test_move_cursor_right() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.set_cursor(0);
        app.move_cursor_right();
        assert_eq!(app.cursor(), 1);
    }

    #[test]
    fn test_move_cursor_right_at_end() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.move_cursor_right(); // Already at end
        assert_eq!(app.cursor(), 3);
    }

    #[test]
    fn test_move_cursor_start() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.move_cursor_start();
        assert_eq!(app.cursor(), 0);
    }

    #[test]
    fn test_move_cursor_end() {
        let mut app = CalculatorApp::new();
        app.set_input("123");
        app.set_cursor(0);
        app.move_cursor_end();
        assert_eq!(app.cursor(), 3);
    }

    // ===== Clear tests =====

    #[test]
    fn test_clear() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 2");
        app.evaluate();
        app.clear();
        assert!(app.input().is_empty());
        assert_eq!(app.cursor(), 0);
        assert!(app.result().is_none());
        // History should be preserved
        assert!(!app.history().is_empty());
    }

    #[test]
    fn test_clear_all() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 2");
        app.evaluate();
        app.clear_all();
        assert!(app.input().is_empty());
        assert!(app.history().is_empty());
    }

    // ===== Evaluation tests =====

    #[test]
    fn test_evaluate_simple() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 3");
        app.evaluate();
        assert_eq!(app.result(), Some(&Ok(5.0)));
    }

    #[test]
    fn test_evaluate_empty() {
        let mut app = CalculatorApp::new();
        app.evaluate(); // Empty input
        assert!(app.result().is_none());
    }

    #[test]
    fn test_evaluate_error() {
        let mut app = CalculatorApp::new();
        app.set_input("1 / 0");
        app.evaluate();
        assert!(matches!(app.result(), Some(Err(CalcError::DivisionByZero))));
    }

    #[test]
    fn test_evaluate_records_history() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 2");
        app.evaluate();
        assert_eq!(app.history().len(), 1);
        assert_eq!(app.history().last().unwrap().result, 4.0);
    }

    #[test]
    fn test_evaluate_error_not_in_history() {
        let mut app = CalculatorApp::new();
        app.set_input("1 / 0");
        app.evaluate();
        assert!(app.history().is_empty());
    }

    // ===== Result display tests =====

    #[test]
    fn test_result_display_none() {
        let app = CalculatorApp::new();
        assert_eq!(app.result_display(), "");
    }

    #[test]
    fn test_result_display_integer() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 2");
        app.evaluate();
        assert_eq!(app.result_display(), "4");
    }

    #[test]
    fn test_result_display_decimal() {
        let mut app = CalculatorApp::new();
        app.set_input("1 / 3");
        app.evaluate();
        let display = app.result_display();
        assert!(display.starts_with("0.333"));
    }

    #[test]
    fn test_result_display_error() {
        let mut app = CalculatorApp::new();
        app.set_input("1 / 0");
        app.evaluate();
        assert!(app.result_display().contains("Error"));
    }

    // ===== Quit tests =====

    #[test]
    fn test_quit() {
        let mut app = CalculatorApp::new();
        assert!(!app.should_quit());
        app.quit();
        assert!(app.should_quit());
    }

    // ===== History recall tests =====

    #[test]
    fn test_recall_last() {
        let mut app = CalculatorApp::new();
        app.set_input("5 * 5");
        app.evaluate();
        app.clear();
        app.recall_last();
        assert_eq!(app.input(), "5 * 5");
    }

    #[test]
    fn test_recall_last_empty_history() {
        let mut app = CalculatorApp::new();
        app.recall_last(); // Should do nothing
        assert!(app.input().is_empty());
    }

    // ===== Anomaly status tests =====

    #[test]
    fn test_jidoka_status_ready() {
        let app = CalculatorApp::new();
        let status = app.jidoka_status();
        assert!(status.contains(&"Ready".to_string()));
    }

    #[test]
    fn test_jidoka_status_success() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 2");
        app.evaluate();
        let status = app.jidoka_status();
        assert!(status.iter().any(|s| s.contains("✓")));
    }

    #[test]
    fn test_jidoka_status_violation() {
        let validator = AnomalyValidator::with_max_magnitude(10.0);
        let mut app = CalculatorApp::with_validator(validator);
        app.set_input("5 * 5");
        app.evaluate();
        let status = app.jidoka_status();
        assert!(status.iter().any(|s| s.contains("✗")));
    }

    #[test]
    fn test_jidoka_status_error() {
        let mut app = CalculatorApp::new();
        app.set_input("1 / 0");
        app.evaluate();
        let status = app.jidoka_status();
        assert!(status.iter().any(|s| s.contains("Error")));
    }

    // ===== Format result tests =====

    #[test]
    fn test_format_result_integer() {
        assert_eq!(format_result(42.0), "42");
    }

    #[test]
    fn test_format_result_negative_integer() {
        assert_eq!(format_result(-42.0), "-42");
    }

    #[test]
    fn test_format_result_decimal() {
        assert_eq!(format_result(3.14), "3.14");
    }

    #[test]
    fn test_format_result_trailing_zeros() {
        assert_eq!(format_result(1.50), "1.5");
    }

    #[test]
    fn test_format_result_large_integer() {
        assert_eq!(format_result(1e14), "100000000000000");
    }

    #[test]
    fn test_format_result_very_large() {
        // Large number should not be formatted as integer
        let result = format_result(1e16);
        assert!(result.contains('e') || result.len() > 15);
    }
}
