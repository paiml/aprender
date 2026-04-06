//! AST evaluator with 100% test coverage
//!
//! Probar: Anomaly detection - Automatic anomaly detection during evaluation

use crate::core::parser::AstNode;
use crate::core::{AnomalyValidator, CalcResult, Calculator, Operation};

/// Evaluator for AST expressions
#[derive(Debug)]
pub struct Evaluator {
    calculator: Calculator,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator {
    /// Creates a new evaluator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            calculator: Calculator::new(),
        }
    }

    /// Creates an evaluator with custom Anomaly validator
    #[must_use]
    pub fn with_validator(validator: AnomalyValidator) -> Self {
        Self {
            calculator: Calculator::with_validator(validator),
        }
    }

    /// Evaluates an AST node and returns the result
    pub fn evaluate(&mut self, node: &AstNode) -> CalcResult<f64> {
        match node {
            AstNode::Number(n) => Ok(*n),
            AstNode::Negate(inner) => {
                let value = self.evaluate(inner)?;
                // Negation is multiplication by -1
                self.calculator.calculate(value, -1.0, Operation::Multiply)
            }
            AstNode::BinaryOp { left, op, right } => {
                let left_val = self.evaluate(left)?;
                let right_val = self.evaluate(right)?;
                self.calculator.calculate(left_val, right_val, *op)
            }
        }
    }

    /// Evaluates a string expression
    pub fn evaluate_str(&mut self, input: &str) -> CalcResult<f64> {
        use crate::core::parser::Parser;
        let ast = Parser::parse_str(input)?;
        self.evaluate(&ast)
    }

    /// Returns a reference to the calculator's Anomaly validator history
    #[must_use]
    pub fn history(&self) -> &std::collections::VecDeque<f64> {
        self.calculator.validator.history()
    }

    /// Clears the Anomaly validator history
    pub fn clear_history(&mut self) {
        self.calculator.validator.clear_history();
    }

    /// Returns a reference to the Anomaly validator
    #[must_use]
    pub fn validator(&self) -> Option<&AnomalyValidator> {
        Some(&self.calculator.validator)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core::CalcError;

    // ===== Basic evaluation tests =====

    #[test]
    fn test_evaluate_number() {
        let mut eval = Evaluator::new();
        let ast = AstNode::number(42.0);
        assert_eq!(eval.evaluate(&ast), Ok(42.0));
    }

    #[test]
    fn test_evaluate_negative_number() {
        let mut eval = Evaluator::new();
        let ast = AstNode::negate(AstNode::number(5.0));
        assert_eq!(eval.evaluate(&ast), Ok(-5.0));
    }

    #[test]
    fn test_evaluate_double_negative() {
        let mut eval = Evaluator::new();
        let ast = AstNode::negate(AstNode::negate(AstNode::number(5.0)));
        assert_eq!(eval.evaluate(&ast), Ok(5.0));
    }

    #[test]
    fn test_evaluate_addition() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(AstNode::number(2.0), Operation::Add, AstNode::number(3.0));
        assert_eq!(eval.evaluate(&ast), Ok(5.0));
    }

    #[test]
    fn test_evaluate_subtraction() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(
            AstNode::number(5.0),
            Operation::Subtract,
            AstNode::number(3.0),
        );
        assert_eq!(eval.evaluate(&ast), Ok(2.0));
    }

    #[test]
    fn test_evaluate_multiplication() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(
            AstNode::number(4.0),
            Operation::Multiply,
            AstNode::number(3.0),
        );
        assert_eq!(eval.evaluate(&ast), Ok(12.0));
    }

    #[test]
    fn test_evaluate_division() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(
            AstNode::number(12.0),
            Operation::Divide,
            AstNode::number(4.0),
        );
        assert_eq!(eval.evaluate(&ast), Ok(3.0));
    }

    #[test]
    fn test_evaluate_modulo() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(
            AstNode::number(7.0),
            Operation::Modulo,
            AstNode::number(3.0),
        );
        assert_eq!(eval.evaluate(&ast), Ok(1.0));
    }

    #[test]
    fn test_evaluate_power() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(AstNode::number(2.0), Operation::Power, AstNode::number(3.0));
        assert_eq!(eval.evaluate(&ast), Ok(8.0));
    }

    // ===== Complex expression tests =====

    #[test]
    fn test_evaluate_nested_expression() {
        let mut eval = Evaluator::new();
        // (2 + 3) * 4 = 20
        let ast = AstNode::binary(
            AstNode::binary(AstNode::number(2.0), Operation::Add, AstNode::number(3.0)),
            Operation::Multiply,
            AstNode::number(4.0),
        );
        assert_eq!(eval.evaluate(&ast), Ok(20.0));
    }

    #[test]
    fn test_evaluate_deeply_nested() {
        let mut eval = Evaluator::new();
        // ((1 + 2) * (3 + 4)) = 3 * 7 = 21
        let ast = AstNode::binary(
            AstNode::binary(AstNode::number(1.0), Operation::Add, AstNode::number(2.0)),
            Operation::Multiply,
            AstNode::binary(AstNode::number(3.0), Operation::Add, AstNode::number(4.0)),
        );
        assert_eq!(eval.evaluate(&ast), Ok(21.0));
    }

    #[test]
    fn test_evaluate_with_negative_in_expression() {
        let mut eval = Evaluator::new();
        // 5 + (-3) = 2
        let ast = AstNode::binary(
            AstNode::number(5.0),
            Operation::Add,
            AstNode::negate(AstNode::number(3.0)),
        );
        assert_eq!(eval.evaluate(&ast), Ok(2.0));
    }

    // ===== Error handling tests =====

    #[test]
    fn test_evaluate_division_by_zero() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(
            AstNode::number(10.0),
            Operation::Divide,
            AstNode::number(0.0),
        );
        assert_eq!(eval.evaluate(&ast), Err(CalcError::DivisionByZero));
    }

    #[test]
    fn test_evaluate_modulo_by_zero() {
        let mut eval = Evaluator::new();
        let ast = AstNode::binary(
            AstNode::number(10.0),
            Operation::Modulo,
            AstNode::number(0.0),
        );
        assert_eq!(eval.evaluate(&ast), Err(CalcError::DivisionByZero));
    }

    #[test]
    fn test_evaluate_error_propagates_from_left() {
        let mut eval = Evaluator::new();
        // (10 / 0) + 5 - error in left operand
        let ast = AstNode::binary(
            AstNode::binary(
                AstNode::number(10.0),
                Operation::Divide,
                AstNode::number(0.0),
            ),
            Operation::Add,
            AstNode::number(5.0),
        );
        assert!(matches!(
            eval.evaluate(&ast),
            Err(CalcError::DivisionByZero)
        ));
    }

    #[test]
    fn test_evaluate_error_propagates_from_right() {
        let mut eval = Evaluator::new();
        // 5 + (10 / 0) - error in right operand
        let ast = AstNode::binary(
            AstNode::number(5.0),
            Operation::Add,
            AstNode::binary(
                AstNode::number(10.0),
                Operation::Divide,
                AstNode::number(0.0),
            ),
        );
        assert!(matches!(
            eval.evaluate(&ast),
            Err(CalcError::DivisionByZero)
        ));
    }

    // ===== String evaluation tests =====

    #[test]
    fn test_evaluate_str_simple() {
        let mut eval = Evaluator::new();
        assert_eq!(eval.evaluate_str("2 + 3"), Ok(5.0));
    }

    #[test]
    fn test_evaluate_str_complex() {
        let mut eval = Evaluator::new();
        assert_eq!(eval.evaluate_str("42 * (3 + 7)"), Ok(420.0));
    }

    #[test]
    fn test_evaluate_str_power() {
        let mut eval = Evaluator::new();
        assert_eq!(eval.evaluate_str("2 ^ 3 ^ 2"), Ok(512.0)); // 2^(3^2) = 2^9
    }

    #[test]
    fn test_evaluate_str_precedence() {
        let mut eval = Evaluator::new();
        assert_eq!(eval.evaluate_str("2 + 3 * 4"), Ok(14.0)); // 2 + (3*4)
    }

    #[test]
    fn test_evaluate_str_unary_minus() {
        let mut eval = Evaluator::new();
        assert_eq!(eval.evaluate_str("-5"), Ok(-5.0));
    }

    #[test]
    fn test_evaluate_str_empty() {
        let mut eval = Evaluator::new();
        assert!(matches!(
            eval.evaluate_str(""),
            Err(CalcError::EmptyExpression)
        ));
    }

    #[test]
    fn test_evaluate_str_invalid() {
        let mut eval = Evaluator::new();
        assert!(matches!(
            eval.evaluate_str("2 +"),
            Err(CalcError::ParseError(_))
        ));
    }

    // ===== Anomaly validation tests =====

    #[test]
    fn test_evaluate_with_jidoka_overflow() {
        let validator = AnomalyValidator::with_max_magnitude(100.0);
        let mut eval = Evaluator::with_validator(validator);
        let result = eval.evaluate_str("50 * 3"); // 150 > 100
        assert!(matches!(result, Err(CalcError::AnomalyViolation(_))));
    }

    #[test]
    fn test_evaluator_history() {
        let mut eval = Evaluator::new();
        // Use actual calculations that go through the validator
        eval.evaluate_str("1 + 1").unwrap();
        eval.evaluate_str("2 + 2").unwrap();
        eval.evaluate_str("3 + 3").unwrap();
        assert_eq!(eval.history().len(), 3);
    }

    #[test]
    fn test_evaluator_clear_history() {
        let mut eval = Evaluator::new();
        eval.evaluate_str("1").unwrap();
        eval.evaluate_str("2").unwrap();
        eval.clear_history();
        assert!(eval.history().is_empty());
    }

    // ===== Constructor tests =====

    #[test]
    fn test_evaluator_new() {
        let eval = Evaluator::new();
        assert!(eval.history().is_empty());
    }

    #[test]
    fn test_evaluator_default() {
        let eval = Evaluator::default();
        assert!(eval.history().is_empty());
    }

    #[test]
    fn test_evaluator_with_validator() {
        let validator = AnomalyValidator::with_max_magnitude(50.0);
        let mut eval = Evaluator::with_validator(validator);
        // Should fail because 100 > 50
        let result = eval.evaluate_str("10 * 10");
        assert!(matches!(result, Err(CalcError::AnomalyViolation(_))));
    }

    // ===== Integration tests =====

    #[test]
    fn test_evaluate_all_operations() {
        let mut eval = Evaluator::new();

        // Addition
        assert_eq!(eval.evaluate_str("10 + 5"), Ok(15.0));

        // Subtraction
        assert_eq!(eval.evaluate_str("10 - 3"), Ok(7.0));

        // Multiplication
        assert_eq!(eval.evaluate_str("6 * 7"), Ok(42.0));

        // Division
        assert_eq!(eval.evaluate_str("20 / 4"), Ok(5.0));

        // Modulo
        assert_eq!(eval.evaluate_str("17 % 5"), Ok(2.0));

        // Power
        assert_eq!(eval.evaluate_str("3 ^ 4"), Ok(81.0));
    }

    #[test]
    fn test_evaluate_complex_real_world() {
        let mut eval = Evaluator::new();

        // Quadratic: (b^2 - 4*a*c) for a=1, b=5, c=6
        // discriminant = 25 - 24 = 1
        assert_eq!(eval.evaluate_str("5 ^ 2 - 4 * 1 * 6"), Ok(1.0));
    }
}
