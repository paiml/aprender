use provable_contracts_macros::{ensures, invariant, requires};

#[requires(x > 0.0)]
fn sqrt_positive(x: f64) -> f64 {
    x.sqrt()
}

#[ensures(ret > 0)]
fn abs_val(x: i32) -> i32 {
    if x < 0 {
        -x
    } else {
        x
    }
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

// ====================================================================
// GH-702: Trait impl methods — verify macros work on trait methods
// ====================================================================

trait Validator {
    fn validate(&self, x: i32) -> bool;
    fn transform(&mut self, x: i32) -> i32;
}

struct RangeValidator {
    min: i32,
    max: i32,
    call_count: u32,
}

impl Validator for RangeValidator {
    #[requires(x >= 0)]
    fn validate(&self, x: i32) -> bool {
        x >= self.min && x <= self.max
    }

    #[ensures(ret >= 0)]
    fn transform(&mut self, x: i32) -> i32 {
        self.call_count += 1;
        x.clamp(self.min, self.max)
    }
}

#[test]
fn test_requires_on_trait_impl() {
    let v = RangeValidator {
        min: 0,
        max: 100,
        call_count: 0,
    };
    assert!(v.validate(50));
    assert!(!v.validate(200));
}

#[test]
fn test_ensures_on_trait_impl() {
    let mut v = RangeValidator {
        min: 0,
        max: 100,
        call_count: 0,
    };
    assert_eq!(v.transform(50), 50);
    assert_eq!(v.transform(-10), 0); // clamped to min
    assert_eq!(v.transform(200), 100); // clamped to max
    assert_eq!(v.call_count, 3);
}

#[test]
#[should_panic(expected = "Pre-condition violated")]
fn test_requires_on_trait_impl_catches_violation() {
    let v = RangeValidator {
        min: 0,
        max: 100,
        call_count: 0,
    };
    v.validate(-1); // violates requires(x >= 0)
}

// Invariant on trait impl method
trait Counter {
    fn increment(&mut self);
    fn count(&self) -> u32;
}

struct SafeCounter {
    value: u32,
}

impl Counter for SafeCounter {
    #[invariant(self.value < u32::MAX)]
    fn increment(&mut self) {
        self.value += 1;
    }

    fn count(&self) -> u32 {
        self.value
    }
}

#[test]
fn test_invariant_on_trait_impl() {
    let mut c = SafeCounter { value: 0 };
    c.increment();
    c.increment();
    assert_eq!(c.count(), 2);
}
