//! Core calculator operations with 100% test coverage
//!
//! Probar: Error prevention - Type-safe operations prevent invalid states

use crate::core::{AnomalyValidator, CalcError, CalcResult};

/// Type-safe operation enum - compile-time guarantee of valid operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Subtract,
    /// Multiplication (*)
    Multiply,
    /// Division (/)
    Divide,
    /// Modulo (%)
    Modulo,
    /// Power (^)
    Power,
}

impl Operation {
    /// Returns the operator symbol for display
    #[must_use]
    pub const fn symbol(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Modulo => "%",
            Self::Power => "^",
        }
    }

    /// Returns the precedence level for operator ordering (higher = evaluated first)
    #[must_use]
    pub const fn precedence(&self) -> u8 {
        match self {
            Self::Add | Self::Subtract => 1,
            Self::Multiply | Self::Divide | Self::Modulo => 2,
            Self::Power => 3,
        }
    }

    /// Returns true if this operation is left-associative
    #[must_use]
    pub const fn is_left_associative(&self) -> bool {
        !matches!(self, Self::Power)
    }
}

/// Core calculator implementing all arithmetic operations
#[derive(Debug, Default)]
pub struct Calculator {
    /// Anomaly validator for anomaly detection
    pub(crate) validator: AnomalyValidator,
}

impl Calculator {
    /// Creates a new calculator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            validator: AnomalyValidator::new(),
        }
    }

    /// Creates a calculator with custom Anomaly validator
    #[must_use]
    pub fn with_validator(validator: AnomalyValidator) -> Self {
        Self { validator }
    }

    /// Performs an operation on two operands
    pub fn calculate(&mut self, a: f64, b: f64, op: Operation) -> CalcResult<f64> {
        let raw_result = match op {
            Operation::Add => Self::add(a, b)?,
            Operation::Subtract => Self::subtract(a, b)?,
            Operation::Multiply => Self::multiply(a, b)?,
            Operation::Divide => Self::divide(a, b)?,
            Operation::Modulo => Self::modulo(a, b)?,
            Operation::Power => Self::power(a, b)?,
        };

        // Anomaly: Validate result
        self.validator
            .validate(raw_result)
            .map_err(|v| CalcError::AnomalyViolation(v))
    }

    /// Addition: a + b
    pub fn add(a: f64, b: f64) -> CalcResult<f64> {
        let result = a + b;
        Self::check_overflow(result)
    }

    /// Subtraction: a - b
    pub fn subtract(a: f64, b: f64) -> CalcResult<f64> {
        let result = a - b;
        Self::check_overflow(result)
    }

    /// Multiplication: a * b
    pub fn multiply(a: f64, b: f64) -> CalcResult<f64> {
        let result = a * b;
        Self::check_overflow(result)
    }

    /// Division: a / b
    pub fn divide(a: f64, b: f64) -> CalcResult<f64> {
        if b == 0.0 {
            return Err(CalcError::DivisionByZero);
        }
        let result = a / b;
        Self::check_overflow(result)
    }

    /// Modulo: a % b
    pub fn modulo(a: f64, b: f64) -> CalcResult<f64> {
        if b == 0.0 {
            return Err(CalcError::DivisionByZero);
        }
        let result = a % b;
        Self::check_overflow(result)
    }

    /// Power: a ^ b
    pub fn power(a: f64, b: f64) -> CalcResult<f64> {
        let result = a.powf(b);
        Self::check_overflow(result)
    }

    /// Checks for overflow (infinity or NaN)
    fn check_overflow(result: f64) -> CalcResult<f64> {
        if result.is_nan() {
            Err(CalcError::InvalidResult("NaN".into()))
        } else if result.is_infinite() {
            Err(CalcError::Overflow)
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AnomalyViolation;
    use proptest::prelude::*;

    // ===== EXTREME TDD: Tests written FIRST =====

    // --- Operation enum tests ---

    #[test]
    fn test_operation_symbol_add() {
        assert_eq!(Operation::Add.symbol(), "+");
    }

    #[test]
    fn test_operation_symbol_subtract() {
        assert_eq!(Operation::Subtract.symbol(), "-");
    }

    #[test]
    fn test_operation_symbol_multiply() {
        assert_eq!(Operation::Multiply.symbol(), "*");
    }

    #[test]
    fn test_operation_symbol_divide() {
        assert_eq!(Operation::Divide.symbol(), "/");
    }

    #[test]
    fn test_operation_symbol_modulo() {
        assert_eq!(Operation::Modulo.symbol(), "%");
    }

    #[test]
    fn test_operation_symbol_power() {
        assert_eq!(Operation::Power.symbol(), "^");
    }

    #[test]
    fn test_operation_precedence_add_subtract() {
        assert_eq!(Operation::Add.precedence(), 1);
        assert_eq!(Operation::Subtract.precedence(), 1);
    }

    #[test]
    fn test_operation_precedence_mul_div_mod() {
        assert_eq!(Operation::Multiply.precedence(), 2);
        assert_eq!(Operation::Divide.precedence(), 2);
        assert_eq!(Operation::Modulo.precedence(), 2);
    }

    #[test]
    fn test_operation_precedence_power() {
        assert_eq!(Operation::Power.precedence(), 3);
    }

    #[test]
    fn test_operation_associativity() {
        assert!(Operation::Add.is_left_associative());
        assert!(Operation::Subtract.is_left_associative());
        assert!(Operation::Multiply.is_left_associative());
        assert!(Operation::Divide.is_left_associative());
        assert!(Operation::Modulo.is_left_associative());
        assert!(!Operation::Power.is_left_associative());
    }

    // --- Calculator creation tests ---

    #[test]
    fn test_calculator_new() {
        let calc = Calculator::new();
        assert!(calc.validator.max_magnitude > 0.0);
    }

    #[test]
    fn test_calculator_with_validator() {
        let validator = AnomalyValidator::with_max_magnitude(100.0);
        let calc = Calculator::with_validator(validator);
        assert_eq!(calc.validator.max_magnitude, 100.0);
    }

    // --- Addition tests ---

    #[test]
    fn test_add_positive_numbers() {
        assert_eq!(Calculator::add(2.0, 3.0), Ok(5.0));
    }

    #[test]
    fn test_add_negative_numbers() {
        assert_eq!(Calculator::add(-2.0, -3.0), Ok(-5.0));
    }

    #[test]
    fn test_add_mixed_numbers() {
        assert_eq!(Calculator::add(-2.0, 5.0), Ok(3.0));
    }

    #[test]
    fn test_add_zero() {
        assert_eq!(Calculator::add(5.0, 0.0), Ok(5.0));
        assert_eq!(Calculator::add(0.0, 5.0), Ok(5.0));
    }

    #[test]
    fn test_add_decimals() {
        let result = Calculator::add(0.1, 0.2).unwrap();
        assert!((result - 0.3).abs() < 1e-10);
    }

    // --- Subtraction tests ---

    #[test]
    fn test_subtract_positive_numbers() {
        assert_eq!(Calculator::subtract(5.0, 3.0), Ok(2.0));
    }

    #[test]
    fn test_subtract_negative_numbers() {
        assert_eq!(Calculator::subtract(-2.0, -3.0), Ok(1.0));
    }

    #[test]
    fn test_subtract_to_negative() {
        assert_eq!(Calculator::subtract(3.0, 5.0), Ok(-2.0));
    }

    #[test]
    fn test_subtract_zero() {
        assert_eq!(Calculator::subtract(5.0, 0.0), Ok(5.0));
    }

    // --- Multiplication tests ---

    #[test]
    fn test_multiply_positive_numbers() {
        assert_eq!(Calculator::multiply(2.0, 3.0), Ok(6.0));
    }

    #[test]
    fn test_multiply_negative_numbers() {
        assert_eq!(Calculator::multiply(-2.0, -3.0), Ok(6.0));
    }

    #[test]
    fn test_multiply_mixed_signs() {
        assert_eq!(Calculator::multiply(-2.0, 3.0), Ok(-6.0));
    }

    #[test]
    fn test_multiply_by_zero() {
        assert_eq!(Calculator::multiply(5.0, 0.0), Ok(0.0));
        assert_eq!(Calculator::multiply(0.0, 5.0), Ok(0.0));
    }

    #[test]
    fn test_multiply_by_one() {
        assert_eq!(Calculator::multiply(5.0, 1.0), Ok(5.0));
        assert_eq!(Calculator::multiply(1.0, 5.0), Ok(5.0));
    }

    // --- Division tests ---

    #[test]
    fn test_divide_positive_numbers() {
        assert_eq!(Calculator::divide(6.0, 2.0), Ok(3.0));
    }

    #[test]
    fn test_divide_by_zero() {
        assert_eq!(
            Calculator::divide(10.0, 0.0),
            Err(CalcError::DivisionByZero)
        );
    }

    #[test]
    fn test_divide_negative_numbers() {
        assert_eq!(Calculator::divide(-6.0, -2.0), Ok(3.0));
    }

    #[test]
    fn test_divide_mixed_signs() {
        assert_eq!(Calculator::divide(-6.0, 2.0), Ok(-3.0));
    }

    #[test]
    fn test_divide_by_one() {
        assert_eq!(Calculator::divide(5.0, 1.0), Ok(5.0));
    }

    #[test]
    fn test_divide_zero_by_number() {
        assert_eq!(Calculator::divide(0.0, 5.0), Ok(0.0));
    }

    // --- Modulo tests ---

    #[test]
    fn test_modulo_positive_numbers() {
        assert_eq!(Calculator::modulo(7.0, 3.0), Ok(1.0));
    }

    #[test]
    fn test_modulo_by_zero() {
        assert_eq!(
            Calculator::modulo(10.0, 0.0),
            Err(CalcError::DivisionByZero)
        );
    }

    #[test]
    fn test_modulo_no_remainder() {
        assert_eq!(Calculator::modulo(6.0, 3.0), Ok(0.0));
    }

    #[test]
    fn test_modulo_negative_dividend() {
        let result = Calculator::modulo(-7.0, 3.0).unwrap();
        assert!((result - -1.0).abs() < 1e-10);
    }

    // --- Power tests ---

    #[test]
    fn test_power_positive_integers() {
        assert_eq!(Calculator::power(2.0, 3.0), Ok(8.0));
    }

    #[test]
    fn test_power_zero_exponent() {
        assert_eq!(Calculator::power(5.0, 0.0), Ok(1.0));
    }

    #[test]
    fn test_power_one_exponent() {
        assert_eq!(Calculator::power(5.0, 1.0), Ok(5.0));
    }

    #[test]
    fn test_power_negative_exponent() {
        assert_eq!(Calculator::power(2.0, -1.0), Ok(0.5));
    }

    #[test]
    fn test_power_fractional_exponent() {
        let result = Calculator::power(4.0, 0.5).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_negative_base_integer_exp() {
        assert_eq!(Calculator::power(-2.0, 2.0), Ok(4.0));
        assert_eq!(Calculator::power(-2.0, 3.0), Ok(-8.0));
    }

    #[test]
    fn test_power_overflow() {
        assert_eq!(Calculator::power(10.0, 1000.0), Err(CalcError::Overflow));
    }

    #[test]
    fn test_power_negative_base_fractional_exp_nan() {
        // (-2)^0.5 is NaN
        let result = Calculator::power(-2.0, 0.5);
        assert!(matches!(result, Err(CalcError::InvalidResult(_))));
    }

    // --- Calculator.calculate() integration tests ---

    #[test]
    fn test_calculate_add() {
        let mut calc = Calculator::new();
        assert_eq!(calc.calculate(2.0, 3.0, Operation::Add), Ok(5.0));
    }

    #[test]
    fn test_calculate_subtract() {
        let mut calc = Calculator::new();
        assert_eq!(calc.calculate(5.0, 3.0, Operation::Subtract), Ok(2.0));
    }

    #[test]
    fn test_calculate_multiply() {
        let mut calc = Calculator::new();
        assert_eq!(calc.calculate(4.0, 3.0, Operation::Multiply), Ok(12.0));
    }

    #[test]
    fn test_calculate_divide() {
        let mut calc = Calculator::new();
        assert_eq!(calc.calculate(12.0, 4.0, Operation::Divide), Ok(3.0));
    }

    #[test]
    fn test_calculate_modulo() {
        let mut calc = Calculator::new();
        assert_eq!(calc.calculate(7.0, 3.0, Operation::Modulo), Ok(1.0));
    }

    #[test]
    fn test_calculate_power() {
        let mut calc = Calculator::new();
        assert_eq!(calc.calculate(2.0, 3.0, Operation::Power), Ok(8.0));
    }

    #[test]
    fn test_calculate_with_jidoka_overflow() {
        let validator = AnomalyValidator::with_max_magnitude(100.0);
        let mut calc = Calculator::with_validator(validator);
        let result = calc.calculate(50.0, 3.0, Operation::Multiply);
        assert!(matches!(
            result,
            Err(CalcError::AnomalyViolation(AnomalyViolation::Overflow(_)))
        ));
    }

    // --- Property-based tests ---

    proptest! {
        #[test]
        fn prop_add_commutative(a in -1e10f64..1e10f64, b in -1e10f64..1e10f64) {
            prop_assume!(!a.is_nan() && !b.is_nan());
            let r1 = Calculator::add(a, b);
            let r2 = Calculator::add(b, a);
            match (r1, r2) {
                (Ok(v1), Ok(v2)) => prop_assert!((v1 - v2).abs() < 1e-10),
                (Err(_), Err(_)) => {}
                _ => prop_assert!(false, "Commutativity violated"),
            }
        }

        #[test]
        fn prop_multiply_commutative(a in -1e5f64..1e5f64, b in -1e5f64..1e5f64) {
            prop_assume!(!a.is_nan() && !b.is_nan());
            let r1 = Calculator::multiply(a, b);
            let r2 = Calculator::multiply(b, a);
            match (r1, r2) {
                (Ok(v1), Ok(v2)) => prop_assert!((v1 - v2).abs() < 1e-10),
                (Err(_), Err(_)) => {}
                _ => prop_assert!(false, "Commutativity violated"),
            }
        }

        #[test]
        fn prop_add_identity(a in -1e10f64..1e10f64) {
            prop_assume!(!a.is_nan());
            let result = Calculator::add(a, 0.0);
            prop_assert_eq!(result, Ok(a));
        }

        #[test]
        fn prop_multiply_identity(a in -1e10f64..1e10f64) {
            prop_assume!(!a.is_nan());
            let result = Calculator::multiply(a, 1.0);
            prop_assert_eq!(result, Ok(a));
        }

        #[test]
        fn prop_multiply_zero(a in -1e10f64..1e10f64) {
            prop_assume!(!a.is_nan());
            let result = Calculator::multiply(a, 0.0);
            prop_assert_eq!(result, Ok(0.0));
        }

        #[test]
        fn prop_divide_by_self(a in -1e10f64..1e10f64) {
            prop_assume!(!a.is_nan() && a != 0.0);
            let result = Calculator::divide(a, a).unwrap();
            prop_assert!((result - 1.0).abs() < 1e-10);
        }

        #[test]
        fn prop_power_zero_exponent(a in 1.0f64..1e5f64) {
            let result = Calculator::power(a, 0.0);
            prop_assert_eq!(result, Ok(1.0));
        }

        #[test]
        fn prop_power_one_exponent(a in -1e5f64..1e5f64) {
            prop_assume!(!a.is_nan());
            let result = Calculator::power(a, 1.0);
            prop_assert_eq!(result, Ok(a));
        }
    }
}
