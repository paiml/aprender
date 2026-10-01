use provable_contracts_macros::{contract, contract_env_key, ensures, invariant, requires};

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

/// `#[contract]` keeps the function it annotates. The method is inherent and a trait in scope has a default
/// method of the same name, so if the attribute dropped the function the call would silently resolve to the
/// trait's `0` instead of failing to compile (mutant `lib.rs:149` → empty expansion, #4588).
struct ContractProbe;

// Unused exactly when the attribute works. `allow`, not `expect`: under the mutant the trait IS used, an unmet
// `expect` would fail the build, and cargo-mutants would file the mutant as unviable instead of caught.
#[allow(dead_code)]
trait ContractFallback {
    fn value(&self) -> u32 {
        0
    }
}

impl ContractFallback for ContractProbe {}

impl ContractProbe {
    #[contract("macro-probe-v1", equation = "value")]
    fn value(&self) -> u32 {
        7
    }
}

#[test]
fn contract_keeps_the_annotated_function() {
    assert_eq!(ContractProbe.value(), 7);
}

/// `contract_env_key!` expands to the key string. Inside `vec![…]` an empty expansion still compiles (as an
/// empty vec), so the mutant is caught at run time (mutant `lib.rs:335` → empty expansion, #4588).
#[test]
fn contract_env_key_expands_to_the_key() {
    let keys: Vec<&str> = vec![contract_env_key!("rmsnorm-kernel-v1", "rmsnorm")];
    assert_eq!(keys, ["CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"]);
}
