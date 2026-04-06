//! WASM Calculator Bindings
//!
//! This module provides the WASM-specific calculator implementation
//! that wraps the core calculator for browser usage.
//!
//! Probar: Error prevention - Type-safe bindings prevent invalid states

use crate::core::evaluator::Evaluator;
use crate::core::history::{History, HistoryEntry};
use crate::core::{AnomalyValidator, CalcError, CalcResult};

/// WASM Calculator - browser-ready calculator
#[derive(Debug)]
pub struct WasmCalculator {
    /// Expression evaluator
    evaluator: Evaluator,
    /// Expression history
    history: History,
    /// Current input expression
    input: String,
    /// Last result (if any)
    last_result: Option<Result<f64, CalcError>>,
    /// Anomaly status messages
    jidoka_status: Vec<String>,
}

impl Default for WasmCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmCalculator {
    /// Creates a new WASM calculator
    #[must_use]
    pub fn new() -> Self {
        Self {
            evaluator: Evaluator::new(),
            history: History::new(),
            input: String::new(),
            last_result: None,
            jidoka_status: vec!["Ready".to_string()],
        }
    }

    /// Creates a calculator with custom Anomaly validator
    #[must_use]
    pub fn with_validator(validator: AnomalyValidator) -> Self {
        Self {
            evaluator: Evaluator::with_validator(validator),
            history: History::new(),
            input: String::new(),
            last_result: None,
            jidoka_status: vec!["Ready (with validator)".to_string()],
        }
    }

    /// Sets the input expression
    pub fn set_input(&mut self, expr: &str) {
        self.input = expr.to_string();
    }

    /// Gets the current input
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Appends to the input
    pub fn append_input(&mut self, s: &str) {
        self.input.push_str(s);
    }

    /// Removes the last character from input
    pub fn backspace(&mut self) {
        self.input.pop();
    }

    /// Clears all state
    pub fn clear(&mut self) {
        self.input.clear();
        self.last_result = None;
        self.jidoka_status = vec!["Cleared".to_string()];
    }

    /// Clears only the input (keeps result)
    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    /// Evaluates the current input expression
    pub fn evaluate(&mut self) -> CalcResult<f64> {
        if self.input.is_empty() {
            let err = CalcError::EmptyExpression;
            self.last_result = Some(Err(err.clone()));
            self.jidoka_status = vec!["Error: Empty expression".to_string()];
            return Err(err);
        }

        let result = self.evaluator.evaluate_str(&self.input);

        match &result {
            Ok(value) => {
                self.history.record(&self.input, *value);
                self.jidoka_status = vec![
                    format!("✓ Result: {}", value),
                    "✓ Calculation successful".to_string(),
                    "✓ All constraints satisfied".to_string(),
                ];
            }
            Err(e) => {
                self.jidoka_status = vec![format!("✗ Error: {}", e)];
            }
        }

        self.last_result = Some(result.clone());
        result
    }

    /// Gets the last result
    #[must_use]
    pub fn result(&self) -> Option<&Result<f64, CalcError>> {
        self.last_result.as_ref()
    }

    /// Gets a formatted result display string
    #[must_use]
    pub fn result_display(&self) -> String {
        match &self.last_result {
            Some(Ok(value)) => format_number(*value),
            Some(Err(e)) => format!("Error: {}", e),
            None => String::new(),
        }
    }

    /// Gets the calculation history
    #[must_use]
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Gets history as JSON-like string array (for WASM interop)
    #[must_use]
    pub fn history_entries(&self) -> Vec<String> {
        self.history
            .iter()
            .map(|entry| format!("{} = {}", entry.expression, format_number(entry.result)))
            .collect()
    }

    /// Gets history entries in reverse order (newest first)
    #[must_use]
    pub fn history_entries_rev(&self) -> Vec<String> {
        self.history
            .iter_rev()
            .map(|entry| format!("{} = {}", entry.expression, format_number(entry.result)))
            .collect()
    }

    /// Gets the Anomaly status messages
    #[must_use]
    pub fn jidoka_status(&self) -> &[String] {
        &self.jidoka_status
    }

    /// Clears the history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.evaluator.clear_history();
    }

    /// Gets the number of history entries
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Recalls a history entry by index
    pub fn recall_history(&self, index: usize) -> Option<&HistoryEntry> {
        self.history.get(index)
    }

    /// Uses the last result as input for chained calculations
    pub fn use_last_result(&mut self) {
        if let Some(Ok(value)) = &self.last_result {
            self.input = format_number(*value);
        }
    }
}

/// Formats a number for display (removes trailing zeros)
fn format_number(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        // Format with reasonable precision, trim trailing zeros
        let s = format!("{:.10}", n);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Constructor tests =====

    #[test]
    fn test_wasm_calculator_new() {
        let calc = WasmCalculator::new();
        assert!(calc.input().is_empty());
        assert!(calc.result().is_none());
    }

    #[test]
    fn test_wasm_calculator_default() {
        let calc = WasmCalculator::default();
        assert!(calc.input().is_empty());
    }

    #[test]
    fn test_wasm_calculator_with_validator() {
        let validator = AnomalyValidator::with_max_magnitude(100.0);
        let mut calc = WasmCalculator::with_validator(validator);
        // Should fail because 200 > 100
        calc.set_input("100 * 2");
        let _ = calc.evaluate();
        assert!(matches!(
            calc.result(),
            Some(Err(CalcError::AnomalyViolation(_)))
        ));
    }

    #[test]
    fn test_wasm_calculator_debug() {
        let calc = WasmCalculator::new();
        let debug = format!("{:?}", calc);
        assert!(debug.contains("WasmCalculator"));
    }

    // ===== Input tests =====

    #[test]
    fn test_set_input() {
        let mut calc = WasmCalculator::new();
        calc.set_input("2 + 2");
        assert_eq!(calc.input(), "2 + 2");
    }

    #[test]
    fn test_append_input() {
        let mut calc = WasmCalculator::new();
        calc.append_input("2");
        calc.append_input("+");
        calc.append_input("3");
        assert_eq!(calc.input(), "2+3");
    }

    #[test]
    fn test_backspace() {
        let mut calc = WasmCalculator::new();
        calc.set_input("123");
        calc.backspace();
        assert_eq!(calc.input(), "12");
    }

    #[test]
    fn test_backspace_empty() {
        let mut calc = WasmCalculator::new();
        calc.backspace(); // should not panic
        assert!(calc.input().is_empty());
    }

    #[test]
    fn test_clear() {
        let mut calc = WasmCalculator::new();
        calc.set_input("5 * 5");
        calc.evaluate().unwrap();
        calc.clear();
        assert!(calc.input().is_empty());
        assert!(calc.result().is_none());
    }

    #[test]
    fn test_clear_input() {
        let mut calc = WasmCalculator::new();
        calc.set_input("5 * 5");
        calc.evaluate().unwrap();
        calc.clear_input();
        assert!(calc.input().is_empty());
        assert!(calc.result().is_some()); // result preserved
    }

    // ===== Evaluation tests =====

    #[test]
    fn test_evaluate_simple() {
        let mut calc = WasmCalculator::new();
        calc.set_input("2 + 3");
        let result = calc.evaluate();
        assert_eq!(result, Ok(5.0));
    }

    #[test]
    fn test_evaluate_complex() {
        let mut calc = WasmCalculator::new();
        calc.set_input("42 * (3 + 7)");
        let result = calc.evaluate();
        assert_eq!(result, Ok(420.0));
    }

    #[test]
    fn test_evaluate_empty() {
        let mut calc = WasmCalculator::new();
        let result = calc.evaluate();
        assert!(matches!(result, Err(CalcError::EmptyExpression)));
    }

    #[test]
    fn test_evaluate_division_by_zero() {
        let mut calc = WasmCalculator::new();
        calc.set_input("1 / 0");
        let result = calc.evaluate();
        assert!(matches!(result, Err(CalcError::DivisionByZero)));
    }

    #[test]
    fn test_evaluate_parse_error() {
        let mut calc = WasmCalculator::new();
        calc.set_input("2 + +");
        let result = calc.evaluate();
        assert!(matches!(result, Err(CalcError::ParseError(_))));
    }

    // ===== Result display tests =====

    #[test]
    fn test_result_display_integer() {
        let mut calc = WasmCalculator::new();
        calc.set_input("5 * 5");
        calc.evaluate().unwrap();
        assert_eq!(calc.result_display(), "25");
    }

    #[test]
    fn test_result_display_decimal() {
        let mut calc = WasmCalculator::new();
        calc.set_input("7 / 2");
        calc.evaluate().unwrap();
        assert_eq!(calc.result_display(), "3.5");
    }

    #[test]
    fn test_result_display_error() {
        let mut calc = WasmCalculator::new();
        calc.set_input("1 / 0");
        calc.evaluate().ok();
        assert!(calc.result_display().contains("Error"));
    }

    #[test]
    fn test_result_display_none() {
        let calc = WasmCalculator::new();
        assert!(calc.result_display().is_empty());
    }

    // ===== History tests =====

    #[test]
    fn test_history_recording() {
        let mut calc = WasmCalculator::new();
        calc.set_input("1 + 1");
        calc.evaluate().unwrap();
        calc.set_input("2 + 2");
        calc.evaluate().unwrap();
        assert_eq!(calc.history_len(), 2);
    }

    #[test]
    fn test_history_entries() {
        let mut calc = WasmCalculator::new();
        calc.set_input("3 + 3");
        calc.evaluate().unwrap();
        let entries = calc.history_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], "3 + 3 = 6");
    }

    #[test]
    fn test_history_entries_rev() {
        let mut calc = WasmCalculator::new();
        calc.set_input("1 + 1");
        calc.evaluate().unwrap();
        calc.set_input("2 + 2");
        calc.evaluate().unwrap();
        let entries = calc.history_entries_rev();
        assert_eq!(entries[0], "2 + 2 = 4"); // newest first
        assert_eq!(entries[1], "1 + 1 = 2");
    }

    #[test]
    fn test_clear_history() {
        let mut calc = WasmCalculator::new();
        calc.set_input("5 + 5");
        calc.evaluate().unwrap();
        calc.clear_history();
        assert_eq!(calc.history_len(), 0);
    }

    #[test]
    fn test_recall_history() {
        let mut calc = WasmCalculator::new();
        calc.set_input("10 + 10");
        calc.evaluate().unwrap();
        let entry = calc.recall_history(0);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().expression, "10 + 10");
    }

    #[test]
    fn test_recall_history_out_of_bounds() {
        let calc = WasmCalculator::new();
        assert!(calc.recall_history(0).is_none());
    }

    // ===== Anomaly status tests =====

    #[test]
    fn test_jidoka_status_initial() {
        let calc = WasmCalculator::new();
        assert!(!calc.jidoka_status().is_empty());
        assert!(calc.jidoka_status()[0].contains("Ready"));
    }

    #[test]
    fn test_jidoka_status_after_success() {
        let mut calc = WasmCalculator::new();
        calc.set_input("2 + 2");
        calc.evaluate().unwrap();
        let status = calc.jidoka_status();
        assert!(status.iter().any(|s| s.contains('✓')));
    }

    #[test]
    fn test_jidoka_status_after_error() {
        let mut calc = WasmCalculator::new();
        calc.set_input("1 / 0");
        calc.evaluate().ok();
        let status = calc.jidoka_status();
        assert!(status
            .iter()
            .any(|s| s.contains('✗') || s.contains("Error")));
    }

    #[test]
    fn test_jidoka_status_after_clear() {
        let mut calc = WasmCalculator::new();
        calc.set_input("5 + 5");
        calc.evaluate().unwrap();
        calc.clear();
        assert!(calc.jidoka_status()[0].contains("Cleared"));
    }

    // ===== Chained calculation tests =====

    #[test]
    fn test_use_last_result() {
        let mut calc = WasmCalculator::new();
        calc.set_input("5 + 5");
        calc.evaluate().unwrap();
        calc.use_last_result();
        assert_eq!(calc.input(), "10");
    }

    #[test]
    fn test_use_last_result_no_result() {
        let mut calc = WasmCalculator::new();
        calc.use_last_result(); // should not panic
        assert!(calc.input().is_empty());
    }

    #[test]
    fn test_chained_calculations() {
        let mut calc = WasmCalculator::new();
        calc.set_input("10 + 10");
        calc.evaluate().unwrap();
        calc.use_last_result();
        calc.append_input(" * 2");
        calc.evaluate().unwrap();
        assert_eq!(calc.result_display(), "40");
    }

    // ===== format_number tests =====

    #[test]
    fn test_format_number_integer() {
        assert_eq!(format_number(42.0), "42");
    }

    #[test]
    fn test_format_number_decimal() {
        assert_eq!(format_number(3.5), "3.5");
    }

    #[test]
    fn test_format_number_negative() {
        assert_eq!(format_number(-5.0), "-5");
    }

    #[test]
    fn test_format_number_small_decimal() {
        assert_eq!(format_number(0.125), "0.125");
    }

    #[test]
    fn test_format_number_trailing_zeros() {
        assert_eq!(format_number(2.500), "2.5");
    }

    // ===== All operations test =====

    #[test]
    fn test_all_operations() {
        let mut calc = WasmCalculator::new();

        calc.set_input("10 + 5");
        assert_eq!(calc.evaluate(), Ok(15.0));

        calc.set_input("10 - 3");
        assert_eq!(calc.evaluate(), Ok(7.0));

        calc.set_input("6 * 7");
        assert_eq!(calc.evaluate(), Ok(42.0));

        calc.set_input("20 / 4");
        assert_eq!(calc.evaluate(), Ok(5.0));

        calc.set_input("17 % 5");
        assert_eq!(calc.evaluate(), Ok(2.0));

        calc.set_input("2 ^ 10");
        assert_eq!(calc.evaluate(), Ok(1024.0));
    }

    // ===== Integration tests =====

    #[test]
    fn test_full_workflow() {
        let mut calc = WasmCalculator::new();

        // Initial state
        assert!(calc.input().is_empty());
        assert!(calc.result().is_none());

        // Enter expression
        calc.set_input("100 / 4");
        assert_eq!(calc.input(), "100 / 4");

        // Evaluate
        let result = calc.evaluate();
        assert_eq!(result, Ok(25.0));
        assert_eq!(calc.result_display(), "25");

        // Check history
        assert_eq!(calc.history_len(), 1);
        let entries = calc.history_entries();
        assert_eq!(entries[0], "100 / 4 = 25");

        // Chain calculation
        calc.use_last_result();
        calc.append_input(" + 75");
        calc.evaluate().unwrap();
        assert_eq!(calc.result_display(), "100");

        // Clear
        calc.clear();
        assert!(calc.input().is_empty());
        assert!(calc.result().is_none());
    }
}
