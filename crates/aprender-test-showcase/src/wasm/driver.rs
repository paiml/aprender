//! WASM Driver - Unified Testing Interface
//!
//! This module implements the CalculatorDriver trait for WASM,
//! enabling the same test specifications to run on both TUI and WASM.
//!
//! Probar: Balanced testing - Balanced testing across platforms

use super::calculator::WasmCalculator;
use super::dom::{DomElement, DomEvent, MockDom};
use crate::core::{AnomalyValidator, CalcResult};
use crate::driver::{CalculatorDriver, HistoryItem};

/// WASM Driver wrapping calculator and mock DOM
#[derive(Debug)]
pub struct WasmDriver {
    /// The calculator instance
    calculator: WasmCalculator,
    /// Mock DOM for testing
    dom: MockDom,
}

impl Default for WasmDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmDriver {
    /// Creates a new WASM driver
    #[must_use]
    pub fn new() -> Self {
        Self {
            calculator: WasmCalculator::new(),
            dom: MockDom::calculator(),
        }
    }

    /// Creates a WASM driver with custom Anomaly validator
    #[must_use]
    pub fn with_validator(validator: AnomalyValidator) -> Self {
        Self {
            calculator: WasmCalculator::with_validator(validator),
            dom: MockDom::calculator(),
        }
    }

    /// Creates a WASM driver with existing calculator and DOM
    #[must_use]
    pub fn with_calculator_and_dom(calculator: WasmCalculator, dom: MockDom) -> Self {
        Self { calculator, dom }
    }

    /// Returns a reference to the calculator
    #[must_use]
    pub fn calculator(&self) -> &WasmCalculator {
        &self.calculator
    }

    /// Returns a mutable reference to the calculator
    pub fn calculator_mut(&mut self) -> &mut WasmCalculator {
        &mut self.calculator
    }

    /// Returns a reference to the DOM
    #[must_use]
    pub fn dom(&self) -> &MockDom {
        &self.dom
    }

    /// Returns a mutable reference to the DOM
    pub fn dom_mut(&mut self) -> &mut MockDom {
        &mut self.dom
    }

    /// Simulates typing into the input field
    pub fn type_input(&mut self, text: &str) {
        self.calculator.set_input(text);
        self.dom.dispatch_event(DomEvent::input("calc-input", text));
        self.sync_dom();
    }

    /// Simulates clicking the equals button
    pub fn click_equals(&mut self) {
        self.dom.dispatch_event(DomEvent::click("btn-equals"));
        let _ = self.calculator.evaluate();
        self.sync_dom();
    }

    /// Simulates clicking the clear button
    pub fn click_clear(&mut self) {
        self.dom.dispatch_event(DomEvent::click("btn-clear"));
        self.calculator.clear();
        self.sync_dom();
    }

    /// Simulates pressing Enter key
    pub fn press_enter(&mut self) {
        self.dom.dispatch_event(DomEvent::key_press("Enter"));
        let _ = self.calculator.evaluate();
        self.sync_dom();
    }

    /// Simulates pressing Escape key
    pub fn press_escape(&mut self) {
        self.dom.dispatch_event(DomEvent::key_press("Escape"));
        self.calculator.clear();
        self.sync_dom();
    }

    /// Simulates pressing Backspace key
    pub fn press_backspace(&mut self) {
        self.dom.dispatch_event(DomEvent::key_press("Backspace"));
        self.calculator.backspace();
        self.sync_dom();
    }

    /// Synchronizes DOM state with calculator state
    fn sync_dom(&mut self) {
        // Update input field
        self.dom
            .set_element_text("calc-input", self.calculator.input());

        // Update result display
        self.dom
            .set_element_text("calc-result", &self.calculator.result_display());

        // Update status display
        let status_text = self.calculator.jidoka_status().join("\n");
        self.dom.set_element_text("calc-status", &status_text);

        // Update history
        self.dom.clear_children("calc-history");
        for (i, entry) in self.calculator.history_entries_rev().iter().enumerate() {
            let item = DomElement::new("li")
                .with_id(&format!("history-{}", i))
                .with_text(entry);
            self.dom.append_child("calc-history", item);
        }
    }

    /// Gets the result element's text
    #[must_use]
    pub fn result_element_text(&self) -> Option<&str> {
        self.dom.get_element_text("calc-result")
    }

    /// Gets the input element's text
    #[must_use]
    pub fn input_element_text(&self) -> Option<&str> {
        self.dom.get_element_text("calc-input")
    }

    /// Gets the status element's text
    #[must_use]
    pub fn status_element_text(&self) -> Option<&str> {
        self.dom.get_element_text("calc-status")
    }

    /// Gets history list items
    #[must_use]
    pub fn history_list_items(&self) -> Vec<String> {
        let mut items = Vec::new();
        let mut i = 0;
        while let Some(elem) = self.dom.get_element(&format!("history-{}", i)) {
            items.push(elem.text_content.clone());
            i += 1;
        }
        items
    }
}

impl CalculatorDriver for WasmDriver {
    fn enter_expression(&mut self, expr: &str) -> CalcResult<()> {
        self.type_input(expr);
        self.calculator.evaluate()?;
        self.sync_dom();
        Ok(())
    }

    fn get_result(&self) -> String {
        self.calculator.result_display()
    }

    fn get_input(&self) -> String {
        self.calculator.input().to_string()
    }

    fn clear(&mut self) {
        self.calculator.clear();
        self.sync_dom();
    }

    fn get_history(&self) -> Vec<HistoryItem> {
        self.calculator
            .history()
            .iter_rev()
            .map(|entry| HistoryItem {
                expression: entry.expression.clone(),
                result: format_number(entry.result),
            })
            .collect()
    }

    fn get_jidoka_status(&self) -> Vec<String> {
        self.calculator.jidoka_status().to_vec()
    }
}

/// Formats a number for display (removes trailing zeros)
fn format_number(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        let s = format!("{:.10}", n);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::CalcError;
    use crate::driver::{
        run_full_specification, verify_basic_arithmetic, verify_complex_expressions,
        verify_error_handling, verify_history, verify_jidoka_status, verify_precedence,
    };

    // ===== Constructor tests =====

    #[test]
    fn test_wasm_driver_new() {
        let driver = WasmDriver::new();
        assert!(driver.get_input().is_empty());
        assert!(driver.get_result().is_empty());
    }

    #[test]
    fn test_wasm_driver_default() {
        let driver = WasmDriver::default();
        assert!(driver.get_input().is_empty());
    }

    #[test]
    fn test_wasm_driver_with_validator() {
        let validator = AnomalyValidator::with_max_magnitude(50.0);
        let mut driver = WasmDriver::with_validator(validator);
        let result = driver.enter_expression("10 * 10");
        assert!(matches!(result, Err(CalcError::AnomalyViolation(_))));
    }

    #[test]
    fn test_wasm_driver_with_calculator_and_dom() {
        let calc = WasmCalculator::new();
        let dom = MockDom::calculator();
        let driver = WasmDriver::with_calculator_and_dom(calc, dom);
        assert!(driver.get_input().is_empty());
    }

    #[test]
    fn test_wasm_driver_debug() {
        let driver = WasmDriver::new();
        let debug = format!("{:?}", driver);
        assert!(debug.contains("WasmDriver"));
    }

    // ===== Access tests =====

    #[test]
    fn test_calculator_access() {
        let driver = WasmDriver::new();
        assert!(driver.calculator().input().is_empty());
    }

    #[test]
    fn test_calculator_mut_access() {
        let mut driver = WasmDriver::new();
        driver.calculator_mut().set_input("test");
        assert_eq!(driver.calculator().input(), "test");
    }

    #[test]
    fn test_dom_access() {
        let driver = WasmDriver::new();
        assert!(driver.dom().get_element("calc-input").is_some());
    }

    #[test]
    fn test_dom_mut_access() {
        let mut driver = WasmDriver::new();
        driver.dom_mut().set_element_text("calc-result", "42");
        assert_eq!(driver.dom().get_element_text("calc-result"), Some("42"));
    }

    // ===== Input simulation tests =====

    #[test]
    fn test_type_input() {
        let mut driver = WasmDriver::new();
        driver.type_input("2 + 2");
        assert_eq!(driver.get_input(), "2 + 2");
        assert_eq!(driver.input_element_text(), Some("2 + 2"));
    }

    #[test]
    fn test_click_equals() {
        let mut driver = WasmDriver::new();
        driver.type_input("3 * 3");
        driver.click_equals();
        assert_eq!(driver.get_result(), "9");
        assert_eq!(driver.result_element_text(), Some("9"));
    }

    #[test]
    fn test_click_clear() {
        let mut driver = WasmDriver::new();
        driver.type_input("5 + 5");
        driver.click_equals();
        driver.click_clear();
        assert!(driver.get_result().is_empty());
        assert!(driver.get_input().is_empty());
    }

    #[test]
    fn test_press_enter() {
        let mut driver = WasmDriver::new();
        driver.type_input("4 * 4");
        driver.press_enter();
        assert_eq!(driver.get_result(), "16");
    }

    #[test]
    fn test_press_escape() {
        let mut driver = WasmDriver::new();
        driver.type_input("7 + 7");
        driver.click_equals();
        driver.press_escape();
        assert!(driver.get_input().is_empty());
        assert!(driver.get_result().is_empty());
    }

    #[test]
    fn test_press_backspace() {
        let mut driver = WasmDriver::new();
        driver.type_input("123");
        driver.press_backspace();
        assert_eq!(driver.get_input(), "12");
    }

    // ===== DOM sync tests =====

    #[test]
    fn test_dom_sync_input() {
        let mut driver = WasmDriver::new();
        driver.type_input("test expression");
        assert_eq!(driver.input_element_text(), Some("test expression"));
    }

    #[test]
    fn test_dom_sync_result() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("8 + 8").unwrap();
        assert_eq!(driver.result_element_text(), Some("16"));
    }

    #[test]
    fn test_dom_sync_status() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("2 + 2").unwrap();
        let status = driver.status_element_text();
        assert!(status.is_some());
        assert!(status.unwrap().contains('✓'));
    }

    #[test]
    fn test_dom_sync_history() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("1 + 1").unwrap();
        driver.enter_expression("2 + 2").unwrap();
        let items = driver.history_list_items();
        assert_eq!(items.len(), 2);
        assert!(items[0].contains("2 + 2")); // newest first
    }

    // ===== CalculatorDriver trait tests =====

    #[test]
    fn test_enter_expression() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("6 * 7").unwrap();
        assert_eq!(driver.get_result(), "42");
    }

    #[test]
    fn test_enter_expression_error() {
        let mut driver = WasmDriver::new();
        let result = driver.enter_expression("1 / 0");
        assert!(matches!(result, Err(CalcError::DivisionByZero)));
    }

    #[test]
    fn test_get_result() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("10 - 3").unwrap();
        assert_eq!(driver.get_result(), "7");
    }

    #[test]
    fn test_get_input() {
        let mut driver = WasmDriver::new();
        driver.type_input("expression");
        assert_eq!(driver.get_input(), "expression");
    }

    #[test]
    fn test_clear() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("9 * 9").unwrap();
        driver.clear();
        assert!(driver.get_result().is_empty());
        assert!(driver.get_input().is_empty());
    }

    #[test]
    fn test_get_history() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("1 + 1").unwrap();
        driver.enter_expression("2 + 2").unwrap();
        let history = driver.get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].expression, "2 + 2"); // newest first
        assert_eq!(history[0].result, "4");
    }

    #[test]
    fn test_get_jidoka_status() {
        let mut driver = WasmDriver::new();
        driver.enter_expression("5 + 5").unwrap();
        let status = driver.get_jidoka_status();
        assert!(!status.is_empty());
    }

    // ===== Unified specification tests - THE UNIFICATION PROOF =====

    #[test]
    fn test_unified_basic_arithmetic_wasm() {
        let mut driver = WasmDriver::new();
        verify_basic_arithmetic(&mut driver);
    }

    #[test]
    fn test_unified_precedence_wasm() {
        let mut driver = WasmDriver::new();
        verify_precedence(&mut driver);
    }

    #[test]
    fn test_unified_complex_expressions_wasm() {
        let mut driver = WasmDriver::new();
        verify_complex_expressions(&mut driver);
    }

    #[test]
    fn test_unified_error_handling_wasm() {
        let mut driver = WasmDriver::new();
        verify_error_handling(&mut driver);
    }

    #[test]
    fn test_unified_history_wasm() {
        let mut driver = WasmDriver::new();
        verify_history(&mut driver);
    }

    #[test]
    fn test_unified_jidoka_status_wasm() {
        let mut driver = WasmDriver::new();
        verify_jidoka_status(&mut driver);
    }

    #[test]
    fn test_full_specification_wasm() {
        let mut driver = WasmDriver::new();
        run_full_specification(&mut driver);
    }

    // ===== format_number tests =====

    #[test]
    fn test_format_number_integer() {
        assert_eq!(format_number(100.0), "100");
    }

    #[test]
    fn test_format_number_decimal() {
        assert_eq!(format_number(2.5), "2.5");
    }

    #[test]
    fn test_format_number_negative() {
        assert_eq!(format_number(-42.0), "-42");
    }

    // ===== Event history verification tests =====

    #[test]
    fn test_event_history_input() {
        let mut driver = WasmDriver::new();
        driver.type_input("2 + 2");
        let events = driver.dom().event_history();
        assert!(events.iter().any(|e| matches!(e, DomEvent::Input { .. })));
    }

    #[test]
    fn test_event_history_click() {
        let mut driver = WasmDriver::new();
        driver.type_input("2 + 2");
        driver.click_equals();
        let events = driver.dom().event_history();
        assert!(events.iter().any(|e| matches!(e, DomEvent::Click { .. })));
    }

    #[test]
    fn test_event_history_key_press() {
        let mut driver = WasmDriver::new();
        driver.type_input("2 + 2");
        driver.press_enter();
        let events = driver.dom().event_history();
        assert!(events
            .iter()
            .any(|e| matches!(e, DomEvent::KeyPress { key, .. } if key == "Enter")));
    }
}
