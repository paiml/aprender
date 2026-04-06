//! Browser WASM bindings for Calculator
//!
//! This module provides the actual browser integration using wasm-bindgen.
//! Probar: Direct observation - Go and see the real browser behavior

// Note: This module is already conditionally compiled via #[cfg(feature = "wasm")] in mod.rs

use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::core::evaluator::Evaluator;
use crate::core::history::History;
use crate::core::AnomalyValidator;
use crate::wasm::keypad::{KeypadAction, WasmKeypad};

/// Browser Calculator - the main WASM entry point
#[derive(Debug)]
#[wasm_bindgen]
pub struct BrowserCalculator {
    evaluator: Evaluator,
    history: History,
    input: String,
    result: Option<Result<f64, String>>,
    keypad: WasmKeypad,
}

#[wasm_bindgen]
impl BrowserCalculator {
    /// Create a new browser calculator
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Set panic hook for better error messages
        #[cfg(feature = "wasm")]
        console_error_panic_hook::set_once();

        Self {
            evaluator: Evaluator::with_validator(AnomalyValidator::new()),
            history: History::new(),
            input: String::new(),
            result: None,
            keypad: WasmKeypad::new(),
        }
    }

    /// Get the current input
    #[wasm_bindgen(getter)]
    pub fn input(&self) -> String {
        self.input.clone()
    }

    /// Set the input
    #[wasm_bindgen(setter)]
    pub fn set_input(&mut self, value: String) {
        self.input = value;
    }

    /// Get the result as a string
    #[wasm_bindgen(getter)]
    pub fn result(&self) -> String {
        match &self.result {
            Some(Ok(val)) => format_number(*val),
            Some(Err(e)) => format!("Error: {}", e),
            None => String::new(),
        }
    }

    /// Append a character to the input
    pub fn append(&mut self, ch: char) {
        self.input.push(ch);
    }

    /// Append a string to the input
    pub fn append_str(&mut self, s: &str) {
        self.input.push_str(s);
    }

    /// Clear the last character
    pub fn backspace(&mut self) {
        self.input.pop();
    }

    /// Clear the input
    pub fn clear(&mut self) {
        self.input.clear();
        self.result = None;
    }

    /// Evaluate the expression
    pub fn evaluate(&mut self) -> String {
        if self.input.is_empty() {
            return String::new();
        }

        match self.evaluator.evaluate_str(&self.input) {
            Ok(val) => {
                let formatted = format_number(val);
                self.history.record(&self.input, val);
                self.result = Some(Ok(val));
                self.input.clear();
                formatted
            }
            Err(e) => {
                let err_msg: String = e.to_string();
                self.result = Some(Err(err_msg.clone()));
                err_msg
            }
        }
    }

    /// Handle a keypad button click by element ID
    pub fn handle_button(&mut self, button_id: &str) -> Option<String> {
        if let Some(action) = self.keypad.handle_click(button_id) {
            return self.handle_action(action);
        }
        None
    }

    /// Handle a keyboard key press
    pub fn handle_key(&mut self, key: &str) -> Option<String> {
        if let Some(action) = WasmKeypad::key_to_action(key) {
            return self.handle_action(action);
        }
        None
    }

    /// Handle a keypad action
    fn handle_action(&mut self, action: KeypadAction) -> Option<String> {
        match action {
            KeypadAction::Digit(d) => {
                self.input.push(char::from_digit(d as u32, 10)?);
                None
            }
            KeypadAction::Decimal => {
                self.input.push('.');
                None
            }
            KeypadAction::Operator(op) => {
                // Add spaces around operators for readability
                if !self.input.is_empty() && !self.input.ends_with(' ') {
                    self.input.push(' ');
                }
                self.input.push(op);
                self.input.push(' ');
                None
            }
            KeypadAction::OpenParen => {
                self.input.push('(');
                None
            }
            KeypadAction::CloseParen => {
                self.input.push(')');
                None
            }
            KeypadAction::Equals => Some(self.evaluate()),
            KeypadAction::Clear => {
                self.clear();
                None
            }
        }
    }

    /// Get history as JSON
    pub fn history_json(&self) -> String {
        self.history.to_json().unwrap_or_else(|_| "[]".to_string())
    }

    /// Get history count
    pub fn history_count(&self) -> usize {
        self.history.len()
    }

    /// Get a history entry
    pub fn history_entry(&self, index: usize) -> Option<String> {
        self.history.get(index).map(|e| e.display())
    }

    /// Get Anomaly status as a string
    pub fn jidoka_status(&self) -> String {
        let validator = self.evaluator.validator();
        let mut status = Vec::new();

        if let Some(result) = &self.result {
            match result {
                Ok(val) => {
                    if val.is_nan() {
                        status.push("⚠ NaN detected".to_string());
                    } else if val.is_infinite() {
                        status.push("⚠ Overflow detected".to_string());
                    } else {
                        status.push("✓ No anomalies".to_string());
                    }
                }
                Err(_) => {
                    status.push("✗ Error state".to_string());
                }
            }
        } else {
            status.push("Ready".to_string());
        }

        if let Some(v) = validator {
            status.push(format!("Max magnitude: {:.0e}", v.max_magnitude));
        }

        status.join("\n")
    }
}

impl Default for BrowserCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a number for display
fn format_number(val: f64) -> String {
    if val.fract() == 0.0 && val.abs() < 1e15 {
        format!("{:.0}", val)
    } else {
        let s = format!("{:.10}", val);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Initialize the calculator in the browser
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "wasm")]
    {
        console_error_panic_hook::set_once();
        console::log_1(&"Calculator WASM initialized".into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_calculator_new() {
        let calc = BrowserCalculator::new();
        assert!(calc.input.is_empty());
        assert!(calc.result.is_none());
    }

    #[test]
    fn test_append_and_evaluate() {
        let mut calc = BrowserCalculator::new();
        calc.append_str("2 + 2");
        assert_eq!(calc.input, "2 + 2");

        let result = calc.evaluate();
        assert_eq!(result, "4");
    }

    #[test]
    fn test_handle_button() {
        let mut calc = BrowserCalculator::new();
        calc.handle_button("btn-5");
        calc.handle_button("btn-plus");
        calc.handle_button("btn-3");
        assert!(calc.input.contains('5'));
        assert!(calc.input.contains('+'));
        assert!(calc.input.contains('3'));
    }

    #[test]
    fn test_handle_key() {
        let mut calc = BrowserCalculator::new();
        calc.handle_key("7");
        calc.handle_key("*");
        calc.handle_key("6");
        let result = calc.handle_key("Enter");
        assert_eq!(result, Some("42".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut calc = BrowserCalculator::new();
        calc.append_str("123");
        calc.clear();
        assert!(calc.input.is_empty());
    }

    #[test]
    fn test_backspace() {
        let mut calc = BrowserCalculator::new();
        calc.append_str("123");
        calc.backspace();
        assert_eq!(calc.input, "12");
    }

    #[test]
    fn test_history() {
        let mut calc = BrowserCalculator::new();
        calc.append_str("1 + 1");
        calc.evaluate();
        calc.append_str("2 * 3");
        calc.evaluate();
        assert_eq!(calc.history_count(), 2);
    }

    #[test]
    fn test_jidoka_status() {
        let calc = BrowserCalculator::new();
        let status = calc.jidoka_status();
        assert!(status.contains("Ready"));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(42.0), "42");
        assert_eq!(format_number(3.14159), "3.14159");
        assert_eq!(format_number(1.5000), "1.5");
    }
}
