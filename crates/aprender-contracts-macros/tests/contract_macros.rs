use provable_contracts_macros::{ensures, requires};

#[requires(x > 0.0)]
fn sqrt_positive(x: f64) -> f64 {
    x.sqrt()
}

#[ensures(ret > 0)]
fn abs_val(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}

#[requires(n > 0)]
#[ensures(ret >= n)]
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

#[test]
fn test_requires_passes() {
    assert!((sqrt_positive(4.0) - 2.0).abs() < f64::EPSILON);
}

#[test]
fn test_ensures_passes() {
    assert_eq!(abs_val(-5), 5);
    assert_eq!(abs_val(3), 3);
}

#[test]
fn test_stacked_contracts() {
    assert_eq!(factorial(5), 120);
    assert_eq!(factorial(1), 1);
}

#[test]
#[should_panic(expected = "Pre-condition violated")]
fn test_requires_catches_violation() {
    sqrt_positive(-1.0);
}

#[test]
#[should_panic(expected = "Post-condition violated")]
fn test_ensures_catches_violation() {
    #[ensures(ret > 0)]
    fn bad_abs(_x: i32) -> i32 {
        0
    }
    bad_abs(5);
}
