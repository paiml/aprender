//! Fuzz testing types: FuzzResult, FuzzFailure, FuzzInputValidator, FuzzValidationError

use std::fmt;

/// Result of a fuzz test run
#[derive(Debug, Clone)]
pub struct FuzzResult {
    /// Target name
    pub target: String,
    /// Number of test cases run
    pub test_cases: u64,
    /// Number of failures found
    pub failures: u64,
    /// Edge coverage percentage (0-100)
    pub coverage_percent: f64,
    /// Specific failure details
    pub failure_details: Vec<FuzzFailure>,
    /// Duration of the fuzz run
    pub duration_secs: f64,
}

impl FuzzResult {
    /// Create a new fuzz result
    pub fn new(target: &str) -> Self {
        Self {
            target: target.to_string(),
            test_cases: 0,
            failures: 0,
            coverage_percent: 0.0,
            failure_details: Vec::new(),
            duration_secs: 0.0,
        }
    }

    /// Record a successful test case
    pub fn record_success(&mut self) {
        self.test_cases += 1;
    }

    /// Record a failure
    pub fn record_failure(&mut self, input: String, error: String) {
        self.test_cases += 1;
        self.failures += 1;
        self.failure_details.push(FuzzFailure { input, error });
    }

    /// Check if all tests passed
    pub fn passed(&self) -> bool {
        self.failures == 0
    }

    /// Calculate failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.test_cases == 0 {
            0.0
        } else {
            (self.failures as f64 / self.test_cases as f64) * 100.0
        }
    }
}

/// Details of a fuzz failure
#[derive(Debug, Clone)]
pub struct FuzzFailure {
    /// Input that caused the failure
    pub input: String,
    /// Error message
    pub error: String,
}

/// Input validator for fuzz testing
#[derive(Debug, Clone, Default)]
pub struct FuzzInputValidator {
    /// Maximum string length allowed
    pub max_string_len: usize,
    /// Maximum numeric value allowed
    pub max_numeric: f64,
    /// Minimum numeric value allowed
    pub min_numeric: f64,
    /// Allow NaN values
    pub allow_nan: bool,
    /// Allow infinity values
    pub allow_infinity: bool,
    /// Allow negative values
    pub allow_negative: bool,
    /// Allow zero
    pub allow_zero: bool,
}

impl FuzzInputValidator {
    /// Create a new validator with default settings
    pub fn new() -> Self {
        Self {
            max_string_len: 1024,
            max_numeric: 1e15,
            min_numeric: -1e15,
            allow_nan: false,
            allow_infinity: false,
            allow_negative: true,
            allow_zero: true,
        }
    }

    /// Validate a floating-point value
    pub fn validate_float(&self, value: f64) -> Result<f64, FuzzValidationError> {
        if value.is_nan() && !self.allow_nan {
            return Err(FuzzValidationError::NaN);
        }
        if value.is_infinite() && !self.allow_infinity {
            return Err(FuzzValidationError::Infinity);
        }
        if value < 0.0 && !self.allow_negative {
            return Err(FuzzValidationError::NegativeValue(value));
        }
        if value == 0.0 && !self.allow_zero {
            return Err(FuzzValidationError::ZeroValue);
        }
        if value > self.max_numeric {
            return Err(FuzzValidationError::TooLarge(value));
        }
        if value < self.min_numeric {
            return Err(FuzzValidationError::TooSmall(value));
        }
        Ok(value)
    }

    /// Validate a string
    pub fn validate_string<'a>(&self, value: &'a str) -> Result<&'a str, FuzzValidationError> {
        if value.len() > self.max_string_len {
            return Err(FuzzValidationError::StringTooLong(value.len()));
        }
        // Check for valid UTF-8 (already guaranteed by &str, but check control chars)
        if value
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
        {
            return Err(FuzzValidationError::InvalidControlChars);
        }
        Ok(value)
    }

    /// Validate an integer
    pub fn validate_u64(&self, value: u64) -> Result<u64, FuzzValidationError> {
        let max = self.max_numeric as u64;
        if value > max {
            return Err(FuzzValidationError::TooLarge(value as f64));
        }
        if value == 0 && !self.allow_zero {
            return Err(FuzzValidationError::ZeroValue);
        }
        Ok(value)
    }

    /// Create a validator for positive-only values
    pub fn positive_only() -> Self {
        Self {
            allow_negative: false,
            allow_zero: false,
            ..Self::new()
        }
    }

    /// Create a validator for non-negative values
    pub fn non_negative() -> Self {
        Self {
            allow_negative: false,
            ..Self::new()
        }
    }

    /// Create a strict validator (no special float values)
    pub fn strict() -> Self {
        Self {
            allow_nan: false,
            allow_infinity: false,
            max_numeric: 1e12,
            min_numeric: -1e12,
            ..Self::new()
        }
    }
}

/// Validation errors from fuzz testing
#[derive(Debug, Clone, PartialEq)]
pub enum FuzzValidationError {
    /// NaN value encountered
    NaN,
    /// Infinity value encountered
    Infinity,
    /// Negative value not allowed
    NegativeValue(f64),
    /// Zero value not allowed
    ZeroValue,
    /// Value too large
    TooLarge(f64),
    /// Value too small
    TooSmall(f64),
    /// String too long
    StringTooLong(usize),
    /// Invalid control characters
    InvalidControlChars,
    /// Integer overflow
    IntegerOverflow,
    /// Division by zero
    DivisionByZero,
    /// Empty input
    EmptyInput,
    /// Invalid format
    InvalidFormat(String),
}

impl fmt::Display for FuzzValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FuzzValidationError::NaN => write!(f, "NaN value not allowed"),
            FuzzValidationError::Infinity => write!(f, "Infinity value not allowed"),
            FuzzValidationError::NegativeValue(v) => write!(f, "Negative value not allowed: {}", v),
            FuzzValidationError::ZeroValue => write!(f, "Zero value not allowed"),
            FuzzValidationError::TooLarge(v) => write!(f, "Value too large: {}", v),
            FuzzValidationError::TooSmall(v) => write!(f, "Value too small: {}", v),
            FuzzValidationError::StringTooLong(len) => write!(f, "String too long: {} chars", len),
            FuzzValidationError::InvalidControlChars => write!(f, "Invalid control characters"),
            FuzzValidationError::IntegerOverflow => write!(f, "Integer overflow"),
            FuzzValidationError::DivisionByZero => write!(f, "Division by zero"),
            FuzzValidationError::EmptyInput => write!(f, "Empty input"),
            FuzzValidationError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl std::error::Error for FuzzValidationError {}
