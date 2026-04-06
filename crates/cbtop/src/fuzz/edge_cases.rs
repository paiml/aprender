//! Utility functions and edge-case testers for fuzz testing

use std::fmt;

/// Safe division that returns None on division by zero
pub fn safe_div(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 || b.is_nan() {
        None
    } else {
        let result = a / b;
        if result.is_nan() || result.is_infinite() {
            None
        } else {
            Some(result)
        }
    }
}

/// Checked addition that returns None on overflow
pub fn checked_add_u64(a: u64, b: u64) -> Option<u64> {
    a.checked_add(b)
}

/// Checked multiplication that returns None on overflow
pub fn checked_mul_u64(a: u64, b: u64) -> Option<u64> {
    a.checked_mul(b)
}

/// Bound a value to a range
pub fn bound_value(value: f64, min: f64, max: f64) -> f64 {
    if value.is_nan() {
        (min + max) / 2.0
    } else {
        value.clamp(min, max)
    }
}

/// Sanitize a float value for safe computation
pub fn sanitize_float(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if value.is_infinite() {
        if value > 0.0 {
            f64::MAX
        } else {
            f64::MIN
        }
    } else {
        value
    }
}

/// Test a function with edge case float inputs
pub fn test_float_edge_cases<F, T>(f: F) -> Vec<(f64, Result<T, String>)>
where
    F: Fn(f64) -> T,
    T: fmt::Debug,
{
    let edge_cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e15,
        -1e15,
        1e-15,
        -1e-15,
    ];

    edge_cases
        .iter()
        .map(|&x| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(x)));
            match result {
                Ok(v) => (x, Ok(v)),
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    (x, Err(msg))
                }
            }
        })
        .collect()
}

/// Test a function with edge case u64 inputs
pub fn test_u64_edge_cases<F, T>(f: F) -> Vec<(u64, Result<T, String>)>
where
    F: Fn(u64) -> T,
    T: fmt::Debug,
{
    let edge_cases =
        [0u64, 1, u64::MAX, u64::MAX - 1, u64::MAX / 2, 1000, 1_000_000, 1_000_000_000];

    edge_cases
        .iter()
        .map(|&x| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(x)));
            match result {
                Ok(v) => (x, Ok(v)),
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    (x, Err(msg))
                }
            }
        })
        .collect()
}
