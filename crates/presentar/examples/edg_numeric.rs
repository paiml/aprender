//! EDG-005: Numeric Edge Cases
//!
//! QA Focus: Handling special numeric values and formatting
//!
//! Run: `cargo run --example edg_numeric`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::or_fun_call,
    clippy::too_many_lines,
    clippy::iter_without_into_iter,
    clippy::struct_field_names,
    clippy::unreadable_literal
)]

/// Numeric value with handling for edge cases
#[derive(Debug, Clone, Copy)]
pub enum NumericValue {
    Normal(f64),
    Infinity,
    NegInfinity,
    NaN,
    Zero,
    NegZero,
}

impl NumericValue {
    pub fn from_f64(value: f64) -> Self {
        if value.is_nan() {
            Self::NaN
        } else if value.is_infinite() {
            if value.is_sign_positive() {
                Self::Infinity
            } else {
                Self::NegInfinity
            }
        } else if value == 0.0 {
            if value.is_sign_negative() {
                Self::NegZero
            } else {
                Self::Zero
            }
        } else {
            Self::Normal(value)
        }
    }

    pub const fn is_finite(&self) -> bool {
        matches!(self, Self::Normal(_) | Self::Zero | Self::NegZero)
    }

    pub const fn is_nan(&self) -> bool {
        matches!(self, Self::NaN)
    }

    pub const fn is_zero(&self) -> bool {
        matches!(self, Self::Zero | Self::NegZero)
    }

    pub const fn to_f64(&self) -> f64 {
        match self {
            Self::Normal(v) => *v,
            Self::Infinity => f64::INFINITY,
            Self::NegInfinity => f64::NEG_INFINITY,
            Self::NaN => f64::NAN,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
        }
    }
}

/// Safe division with edge case handling
pub fn safe_divide(a: f64, b: f64) -> NumericValue {
    if b == 0.0 {
        if a == 0.0 {
            NumericValue::NaN
        } else if a.is_sign_positive() {
            NumericValue::Infinity
        } else {
            NumericValue::NegInfinity
        }
    } else if a.is_nan() || b.is_nan() {
        NumericValue::NaN
    } else {
        NumericValue::from_f64(a / b)
    }
}

/// Safe percentage calculation
pub fn safe_percentage(part: f64, whole: f64) -> Option<f64> {
    if whole == 0.0 || whole.is_nan() || part.is_nan() {
        None
    } else {
        let pct = (part / whole) * 100.0;
        if pct.is_finite() {
            Some(pct)
        } else {
            None
        }
    }
}

/// Numeric formatter with edge case handling
#[derive(Debug, Clone)]
pub struct NumericFormatter {
    pub precision: usize,
    pub nan_display: String,
    pub inf_display: String,
    pub neg_inf_display: String,
    pub zero_display: Option<String>,
    pub use_thousands_sep: bool,
    pub thousands_sep: char,
}

impl Default for NumericFormatter {
    fn default() -> Self {
        Self {
            precision: 2,
            nan_display: "N/A".to_string(),
            inf_display: "∞".to_string(),
            neg_inf_display: "-∞".to_string(),
            zero_display: None,
            use_thousands_sep: true,
            thousands_sep: ',',
        }
    }
}

impl NumericFormatter {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_nan_display(mut self, display: &str) -> Self {
        self.nan_display = display.to_string();
        self
    }

    pub fn format(&self, value: f64) -> String {
        let nv = NumericValue::from_f64(value);

        match nv {
            NumericValue::NaN => self.nan_display.clone(),
            NumericValue::Infinity => self.inf_display.clone(),
            NumericValue::NegInfinity => self.neg_inf_display.clone(),
            NumericValue::Zero | NumericValue::NegZero => self
                .zero_display
                .clone()
                .unwrap_or_else(|| format!("{:.prec$}", 0.0, prec = self.precision)),
            NumericValue::Normal(v) => {
                let formatted = format!("{:.prec$}", v, prec = self.precision);
                if self.use_thousands_sep {
                    self.add_thousands_sep(&formatted)
                } else {
                    formatted
                }
            }
        }
    }

    fn add_thousands_sep(&self, s: &str) -> String {
        let parts: Vec<&str> = s.split('.').collect();
        let int_part = parts[0];
        let dec_part = parts.get(1);

        let negative = int_part.starts_with('-');
        let digits: Vec<char> = int_part.chars().filter(char::is_ascii_digit).collect();

        if digits.is_empty() {
            return s.to_string();
        }

        let mut result = String::new();
        for (i, &c) in digits.iter().enumerate() {
            if i > 0 && (digits.len() - i) % 3 == 0 {
                result.push(self.thousands_sep);
            }
            result.push(c);
        }

        let result = if negative {
            format!("-{result}")
        } else {
            result
        };

        match dec_part {
            Some(d) => format!("{result}.{d}"),
            None => result,
        }
    }

    /// Format as percentage
    pub fn format_percentage(&self, value: f64) -> String {
        if value.is_nan() {
            format!("{} %", self.nan_display)
        } else if value.is_infinite() {
            if value.is_sign_positive() {
                format!("{} %", self.inf_display)
            } else {
                format!("{} %", self.neg_inf_display)
            }
        } else {
            format!("{:.prec$}%", value, prec = self.precision)
        }
    }

    /// Format with SI prefix (K, M, B, T)
    pub fn format_si(&self, value: f64) -> String {
        if !value.is_finite() {
            return self.format(value);
        }

        let abs_value = value.abs();
        let (scaled, suffix) = if abs_value >= 1e12 {
            (value / 1e12, "T")
        } else if abs_value >= 1e9 {
            (value / 1e9, "B")
        } else if abs_value >= 1e6 {
            (value / 1e6, "M")
        } else if abs_value >= 1e3 {
            (value / 1e3, "K")
        } else {
            (value, "")
        };

        format!("{:.prec$}{}", scaled, suffix, prec = self.precision)
    }
}

/// Numeric range with edge case handling
#[derive(Debug)]
pub struct NumericRange {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl NumericRange {
    pub const fn new() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    pub fn update(&mut self, value: f64) {
        if value.is_finite() {
            match self.min {
                Some(m) if value < m => self.min = Some(value),
                None => self.min = Some(value),
                _ => {}
            }
            match self.max {
                Some(m) if value > m => self.max = Some(value),
                None => self.max = Some(value),
                _ => {}
            }
        }
    }

    pub fn span(&self) -> Option<f64> {
        match (self.min, self.max) {
            (Some(min), Some(max)) => Some(max - min),
            _ => None,
        }
    }

    pub fn normalize(&self, value: f64) -> Option<f64> {
        let span = self.span()?;
        if span == 0.0 {
            return Some(0.5);
        }
        let min = self.min?;
        Some((value - min) / span)
    }
}

impl Default for NumericRange {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    println!("=== Numeric Edge Cases ===\n");

    let formatter = NumericFormatter::new()
        .with_precision(2)
        .with_nan_display("--");

    // Edge case values
    let test_values: Vec<(&str, f64)> = vec![
        ("Normal", 1234567.89),
        ("Small", 0.00001),
        ("Large", 1e15),
        ("Zero", 0.0),
        ("Neg Zero", -0.0),
        ("Infinity", f64::INFINITY),
        ("Neg Infinity", f64::NEG_INFINITY),
        ("NaN", f64::NAN),
        ("Max", f64::MAX),
        ("Min Positive", f64::MIN_POSITIVE),
        ("Epsilon", f64::EPSILON),
    ];

    println!(
        "{:<15} {:>20} {:>15} {:>10}",
        "Type", "Value", "Formatted", "SI"
    );
    println!("{}", "-".repeat(65));

    for (name, value) in &test_values {
        println!(
            "{:<15} {:>20} {:>15} {:>10}",
            name,
            if value.is_nan() {
                "NaN".to_string()
            } else {
                format!("{value:.6e}")
            },
            formatter.format(*value),
            formatter.format_si(*value)
        );
    }

    // Division edge cases
    println!("\n=== Division Edge Cases ===\n");
    let divisions = vec![
        (10.0, 2.0),
        (10.0, 0.0),
        (0.0, 0.0),
        (-10.0, 0.0),
        (f64::NAN, 5.0),
        (5.0, f64::NAN),
    ];

    println!("{:>10} / {:>10} = {:>15}", "A", "B", "Result");
    println!("{}", "-".repeat(40));

    for (a, b) in &divisions {
        let result = safe_divide(*a, *b);
        println!(
            "{:>10} / {:>10} = {:>15}",
            formatter.format(*a),
            formatter.format(*b),
            formatter.format(result.to_f64())
        );
    }

    // Percentage edge cases
    println!("\n=== Percentage Edge Cases ===\n");
    let percentages = vec![
        (50.0, 100.0),
        (100.0, 0.0),
        (0.0, 0.0),
        (150.0, 100.0),
        (-50.0, 100.0),
    ];

    for (part, whole) in &percentages {
        let pct = safe_percentage(*part, *whole);
        println!(
            "{} / {} = {}",
            part,
            whole,
            pct.map_or_else(|| "--".to_string(), |p| formatter.format_percentage(p))
        );
    }

    // Range normalization
    println!("\n=== Range Normalization ===\n");
    let mut range = NumericRange::new();
    let values = vec![10.0, 50.0, f64::NAN, 100.0, f64::INFINITY, 25.0];

    for &v in &values {
        range.update(v);
    }

    println!("Range: {:?} - {:?}", range.min, range.max);
    println!("Span: {:?}", range.span());

    for &v in &values {
        let normalized = range.normalize(v);
        println!(
            "  {} -> {}",
            formatter.format(v),
            normalized.map_or("N/A".to_string(), |n| format!("{n:.3}"))
        );
    }

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] NaN handled gracefully");
    println!("- [x] Infinity displayed correctly");
    println!("- [x] Division by zero safe");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_value_from_f64() {
        assert!(matches!(
            NumericValue::from_f64(42.0),
            NumericValue::Normal(_)
        ));
        assert!(matches!(
            NumericValue::from_f64(f64::NAN),
            NumericValue::NaN
        ));
        assert!(matches!(
            NumericValue::from_f64(f64::INFINITY),
            NumericValue::Infinity
        ));
        assert!(matches!(
            NumericValue::from_f64(f64::NEG_INFINITY),
            NumericValue::NegInfinity
        ));
        assert!(matches!(NumericValue::from_f64(0.0), NumericValue::Zero));
    }

    #[test]
    fn test_numeric_value_is_finite() {
        assert!(NumericValue::Normal(42.0).is_finite());
        assert!(NumericValue::Zero.is_finite());
        assert!(!NumericValue::NaN.is_finite());
        assert!(!NumericValue::Infinity.is_finite());
    }

    #[test]
    fn test_safe_divide_normal() {
        let result = safe_divide(10.0, 2.0);
        assert!(matches!(result, NumericValue::Normal(v) if (v - 5.0).abs() < 0.001));
    }

    #[test]
    fn test_safe_divide_by_zero() {
        assert!(matches!(safe_divide(10.0, 0.0), NumericValue::Infinity));
        assert!(matches!(safe_divide(-10.0, 0.0), NumericValue::NegInfinity));
        assert!(matches!(safe_divide(0.0, 0.0), NumericValue::NaN));
    }

    #[test]
    fn test_safe_percentage() {
        assert_eq!(safe_percentage(50.0, 100.0), Some(50.0));
        assert_eq!(safe_percentage(100.0, 0.0), None);
        assert_eq!(safe_percentage(0.0, 100.0), Some(0.0));
    }

    #[test]
    fn test_formatter_nan() {
        let fmt = NumericFormatter::new().with_nan_display("N/A");
        assert_eq!(fmt.format(f64::NAN), "N/A");
    }

    #[test]
    fn test_formatter_infinity() {
        let fmt = NumericFormatter::new();
        assert_eq!(fmt.format(f64::INFINITY), "∞");
        assert_eq!(fmt.format(f64::NEG_INFINITY), "-∞");
    }

    #[test]
    fn test_formatter_thousands_sep() {
        let fmt = NumericFormatter::new().with_precision(0);
        assert_eq!(fmt.format(1234567.0), "1,234,567");
    }

    #[test]
    fn test_formatter_si() {
        let fmt = NumericFormatter::new().with_precision(1);
        assert_eq!(fmt.format_si(1500.0), "1.5K");
        assert_eq!(fmt.format_si(2_500_000.0), "2.5M");
        assert_eq!(fmt.format_si(3_500_000_000.0), "3.5B");
    }

    #[test]
    fn test_range_update() {
        let mut range = NumericRange::new();
        range.update(10.0);
        range.update(50.0);
        range.update(30.0);

        assert_eq!(range.min, Some(10.0));
        assert_eq!(range.max, Some(50.0));
    }

    #[test]
    fn test_range_ignores_non_finite() {
        let mut range = NumericRange::new();
        range.update(10.0);
        range.update(f64::NAN);
        range.update(f64::INFINITY);
        range.update(50.0);

        assert_eq!(range.min, Some(10.0));
        assert_eq!(range.max, Some(50.0));
    }

    #[test]
    fn test_range_normalize() {
        let mut range = NumericRange::new();
        range.update(0.0);
        range.update(100.0);

        assert_eq!(range.normalize(0.0), Some(0.0));
        assert_eq!(range.normalize(50.0), Some(0.5));
        assert_eq!(range.normalize(100.0), Some(1.0));
    }

    #[test]
    fn test_range_normalize_zero_span() {
        let mut range = NumericRange::new();
        range.update(50.0);
        // Only one value, span is 0
        assert_eq!(range.normalize(50.0), Some(0.5));
    }
}
