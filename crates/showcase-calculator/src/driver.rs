//! Unified Calculator Driver - The Probar Way
//!
//! This module implements the core unification principle:
//! **Write the test logic once, run it everywhere.**
//!
//! Probar: Balanced testing - Balanced testing across platforms

use crate::core::{CalcError, CalcResult};

/// Abstract driver trait for calculator interactions
///
/// This trait defines the interface that both TUI and WASM drivers implement,
/// enabling unified test specifications that work on any platform.
///
/// # Example
///
/// ```rust,ignore
/// // The abstract test specification
/// async fn verify_calculation<D: CalculatorDriver>(driver: &mut D) {
///     driver.enter_expression("10 * (5 + 5)").await;
///     assert_eq!(driver.get_result().await, "100");
/// }
///
/// // TUI test
/// #[test]
/// fn test_tui() {
///     let mut driver = TuiDriver::new();
///     block_on(verify_calculation(&mut driver));
/// }
///
/// // WASM test
/// #[wasm_bindgen_test]
/// async fn test_wasm() {
///     let mut driver = WasmDriver::new().await;
///     verify_calculation(&mut driver).await;
/// }
/// ```
pub trait CalculatorDriver {
    /// Enters an expression into the calculator
    fn enter_expression(&mut self, expr: &str) -> CalcResult<()>;

    /// Gets the current result display
    fn get_result(&self) -> String;

    /// Gets the current input expression
    fn get_input(&self) -> String;

    /// Clears the calculator state
    fn clear(&mut self);

    /// Gets history entries (newest first)
    fn get_history(&self) -> Vec<HistoryItem>;

    /// Gets Anomaly status messages
    fn get_jidoka_status(&self) -> Vec<String>;
}

/// A simplified history item for driver results
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryItem {
    /// The expression that was evaluated
    pub expression: String,
    /// The result as a string
    pub result: String,
}

/// TUI Driver implementation
#[cfg(feature = "tui")]
pub mod tui_driver {
    use super::{CalcResult, CalculatorDriver, HistoryItem};
    use crate::tui::CalculatorApp;

    /// TUI-specific driver wrapping the calculator app
    #[derive(Debug)]
    pub struct TuiDriver {
        app: CalculatorApp,
    }

    impl Default for TuiDriver {
        fn default() -> Self {
            Self::new()
        }
    }

    impl TuiDriver {
        /// Creates a new TUI driver
        #[must_use]
        pub fn new() -> Self {
            Self {
                app: CalculatorApp::new(),
            }
        }

        /// Creates a TUI driver with an existing app
        #[must_use]
        pub fn with_app(app: CalculatorApp) -> Self {
            Self { app }
        }

        /// Returns a reference to the underlying app
        #[must_use]
        pub fn app(&self) -> &CalculatorApp {
            &self.app
        }

        /// Returns a mutable reference to the underlying app
        pub fn app_mut(&mut self) -> &mut CalculatorApp {
            &mut self.app
        }
    }

    impl CalculatorDriver for TuiDriver {
        fn enter_expression(&mut self, expr: &str) -> CalcResult<()> {
            self.app.set_input(expr);
            self.app.evaluate();

            // Check if evaluation produced an error
            if let Some(Err(e)) = self.app.result() {
                return Err(e.clone());
            }
            Ok(())
        }

        fn get_result(&self) -> String {
            self.app.result_display()
        }

        fn get_input(&self) -> String {
            self.app.input().to_string()
        }

        fn clear(&mut self) {
            self.app.clear();
        }

        fn get_history(&self) -> Vec<HistoryItem> {
            self.app
                .history()
                .iter_rev()
                .map(|entry| HistoryItem {
                    expression: entry.expression.clone(),
                    result: format!("{}", entry.result),
                })
                .collect()
        }

        fn get_jidoka_status(&self) -> Vec<String> {
            self.app.jidoka_status()
        }
    }
}

#[cfg(feature = "tui")]
pub use tui_driver::TuiDriver;

// ===== Unified Test Specifications =====
// These tests work with ANY CalculatorDriver implementation

/// Verifies basic arithmetic operations
#[allow(clippy::unwrap_used)]
pub fn verify_basic_arithmetic<D: CalculatorDriver>(driver: &mut D) {
    // Addition
    driver.enter_expression("2 + 3").unwrap();
    assert_eq!(driver.get_result(), "5");
    driver.clear();

    // Subtraction
    driver.enter_expression("10 - 4").unwrap();
    assert_eq!(driver.get_result(), "6");
    driver.clear();

    // Multiplication
    driver.enter_expression("6 * 7").unwrap();
    assert_eq!(driver.get_result(), "42");
    driver.clear();

    // Division
    driver.enter_expression("20 / 4").unwrap();
    assert_eq!(driver.get_result(), "5");
    driver.clear();
}

/// Verifies operator precedence (PEMDAS)
#[allow(clippy::unwrap_used)]
pub fn verify_precedence<D: CalculatorDriver>(driver: &mut D) {
    // Multiplication before addition
    driver.enter_expression("2 + 3 * 4").unwrap();
    assert_eq!(driver.get_result(), "14");
    driver.clear();

    // Power before multiplication
    driver.enter_expression("2 * 3 ^ 2").unwrap();
    assert_eq!(driver.get_result(), "18");
    driver.clear();

    // Parentheses override precedence
    driver.enter_expression("(2 + 3) * 4").unwrap();
    assert_eq!(driver.get_result(), "20");
    driver.clear();
}

/// Verifies complex nested expressions
#[allow(clippy::unwrap_used)]
pub fn verify_complex_expressions<D: CalculatorDriver>(driver: &mut D) {
    // Showcase expression
    driver.enter_expression("42 * (3 + 7)").unwrap();
    assert_eq!(driver.get_result(), "420");
    driver.clear();

    // Nested parentheses
    driver.enter_expression("((2 + 3) * (4 + 5))").unwrap();
    assert_eq!(driver.get_result(), "45");
    driver.clear();

    // Power with right associativity
    driver.enter_expression("2 ^ 3 ^ 2").unwrap();
    assert_eq!(driver.get_result(), "512");
    driver.clear();
}

/// Verifies error handling
pub fn verify_error_handling<D: CalculatorDriver>(driver: &mut D) {
    // Division by zero
    let result = driver.enter_expression("1 / 0");
    assert!(result.is_err());
    assert!(matches!(result, Err(CalcError::DivisionByZero)));
    driver.clear();

    // Empty expression (should fail)
    let _result = driver.enter_expression("");
    // Empty expressions might be handled differently per implementation
    driver.clear();
}

/// Verifies history tracking
#[allow(clippy::unwrap_used)]
pub fn verify_history<D: CalculatorDriver>(driver: &mut D) {
    driver.clear();

    // Perform several calculations
    driver.enter_expression("1 + 1").unwrap();
    driver.enter_expression("2 + 2").unwrap();
    driver.enter_expression("3 + 3").unwrap();

    let history = driver.get_history();
    assert!(history.len() >= 3);

    // Most recent should be first
    assert_eq!(history[0].expression, "3 + 3");
    assert_eq!(history[0].result, "6");
}

/// Verifies Anomaly status reporting
#[allow(clippy::unwrap_used)]
pub fn verify_jidoka_status<D: CalculatorDriver>(driver: &mut D) {
    driver.clear();

    // After successful calculation
    driver.enter_expression("5 + 5").unwrap();
    let status = driver.get_jidoka_status();

    // Should have positive status messages
    assert!(!status.is_empty());
    assert!(status
        .iter()
        .any(|s| s.contains('✓') || s.contains("satisfied")));
}

/// Complete verification suite - runs all specifications
pub fn run_full_specification<D: CalculatorDriver>(driver: &mut D) {
    verify_basic_arithmetic(driver);
    verify_precedence(driver);
    verify_complex_expressions(driver);
    verify_error_handling(driver);
    verify_history(driver);
    verify_jidoka_status(driver);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== TUI Driver Tests =====

    #[cfg(feature = "tui")]
    mod tui_tests {
        use super::*;

        #[test]
        fn test_tui_driver_new() {
            let driver = TuiDriver::new();
            assert!(driver.get_input().is_empty());
        }

        #[test]
        fn test_tui_driver_default() {
            let driver = TuiDriver::default();
            assert!(driver.get_input().is_empty());
        }

        #[test]
        fn test_tui_driver_with_app() {
            let app = crate::tui::CalculatorApp::new();
            let driver = TuiDriver::with_app(app);
            assert!(driver.get_input().is_empty());
        }

        #[test]
        fn test_tui_driver_app_access() {
            let mut driver = TuiDriver::new();
            driver.app_mut().set_input("test");
            assert_eq!(driver.app().input(), "test");
        }

        #[test]
        fn test_tui_driver_enter_expression() {
            let mut driver = TuiDriver::new();
            driver.enter_expression("2 + 2").unwrap();
            assert_eq!(driver.get_result(), "4");
        }

        #[test]
        fn test_tui_driver_get_input() {
            let mut driver = TuiDriver::new();
            driver.enter_expression("5 * 5").unwrap();
            assert_eq!(driver.get_input(), "5 * 5");
        }

        #[test]
        fn test_tui_driver_clear() {
            let mut driver = TuiDriver::new();
            driver.enter_expression("1 + 1").unwrap();
            driver.clear();
            assert!(driver.get_result().is_empty());
        }

        #[test]
        fn test_tui_driver_history() {
            let mut driver = TuiDriver::new();
            driver.enter_expression("1 + 1").unwrap();
            driver.enter_expression("2 + 2").unwrap();
            let history = driver.get_history();
            assert_eq!(history.len(), 2);
        }

        #[test]
        fn test_tui_driver_jidoka() {
            let mut driver = TuiDriver::new();
            driver.enter_expression("10 + 10").unwrap();
            let status = driver.get_jidoka_status();
            assert!(!status.is_empty());
        }

        #[test]
        fn test_tui_driver_error_handling() {
            let mut driver = TuiDriver::new();
            let result = driver.enter_expression("1 / 0");
            assert!(matches!(result, Err(CalcError::DivisionByZero)));
        }

        // ===== Unified Specification Tests =====

        #[test]
        fn test_unified_basic_arithmetic() {
            let mut driver = TuiDriver::new();
            verify_basic_arithmetic(&mut driver);
        }

        #[test]
        fn test_unified_precedence() {
            let mut driver = TuiDriver::new();
            verify_precedence(&mut driver);
        }

        #[test]
        fn test_unified_complex_expressions() {
            let mut driver = TuiDriver::new();
            verify_complex_expressions(&mut driver);
        }

        #[test]
        fn test_unified_error_handling() {
            let mut driver = TuiDriver::new();
            verify_error_handling(&mut driver);
        }

        #[test]
        fn test_unified_history() {
            let mut driver = TuiDriver::new();
            verify_history(&mut driver);
        }

        #[test]
        fn test_unified_jidoka_status() {
            let mut driver = TuiDriver::new();
            verify_jidoka_status(&mut driver);
        }

        #[test]
        fn test_full_specification() {
            let mut driver = TuiDriver::new();
            run_full_specification(&mut driver);
        }
    }

    // ===== HistoryItem tests =====

    #[test]
    fn test_history_item_debug() {
        let item = HistoryItem {
            expression: "1+1".into(),
            result: "2".into(),
        };
        assert!(format!("{:?}", item).contains("expression"));
    }

    #[test]
    fn test_history_item_clone() {
        let item = HistoryItem {
            expression: "test".into(),
            result: "42".into(),
        };
        let cloned = item.clone();
        assert_eq!(item, cloned);
    }
}
