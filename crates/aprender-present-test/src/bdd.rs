//! BDD-style testing helpers for Presentar.
//!
//! Provides `describe`, `it`, and context management for expressive tests.
//!
//! # Example
//!
//! ```rust
//! use presentar_test::bdd::*;
//!
//! #[test]
//! fn button_widget_tests() {
//!     describe("Button", |ctx| {
//!         ctx.before(|| {
//!             // Setup code
//!         });
//!
//!         ctx.it("renders with label", |_| {
//!             // Test code
//!             expect(true).to_be_true();
//!         });
//!
//!         ctx.it("responds to click", |_| {
//!             expect(1 + 1).to_equal(2);
//!         });
//!     });
//! }
//! ```

use std::cell::RefCell;
use std::rc::Rc;

/// Test context for BDD-style tests.
#[derive(Default)]
pub struct TestContext {
    /// Description of current test
    description: String,
    /// Before hooks
    before_hooks: Vec<Box<dyn Fn()>>,
    /// After hooks
    after_hooks: Vec<Box<dyn Fn()>>,
    /// Passed test count
    passed: Rc<RefCell<u32>>,
    /// Failed test count
    failed: Rc<RefCell<u32>>,
    /// Failure messages
    failures: Rc<RefCell<Vec<String>>>,
}

impl TestContext {
    /// Create a new test context.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            before_hooks: Vec::new(),
            after_hooks: Vec::new(),
            passed: Rc::new(RefCell::new(0)),
            failed: Rc::new(RefCell::new(0)),
            failures: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Register a before hook.
    pub fn before<F: Fn() + 'static>(&mut self, f: F) {
        self.before_hooks.push(Box::new(f));
    }

    /// Register an after hook.
    pub fn after<F: Fn() + 'static>(&mut self, f: F) {
        self.after_hooks.push(Box::new(f));
    }

    /// Define a test case.
    pub fn it<F: Fn(&TestContext)>(&self, description: &str, test: F) {
        // Run before hooks
        for hook in &self.before_hooks {
            hook();
        }

        // Run test with panic catching
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            test(self);
        }));

        // Run after hooks
        for hook in &self.after_hooks {
            hook();
        }

        match result {
            Ok(()) => {
                *self.passed.borrow_mut() += 1;
            }
            Err(e) => {
                *self.failed.borrow_mut() += 1;
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    format!("{} - {}: {}", self.description, description, s)
                } else if let Some(s) = e.downcast_ref::<String>() {
                    format!("{} - {}: {}", self.description, description, s)
                } else {
                    format!("{} - {}: test panicked", self.description, description)
                };
                self.failures.borrow_mut().push(msg);
            }
        }
    }

    /// Get passed count.
    pub fn passed(&self) -> u32 {
        *self.passed.borrow()
    }

    /// Get failed count.
    pub fn failed(&self) -> u32 {
        *self.failed.borrow()
    }

    /// Get failures.
    pub fn failures(&self) -> Vec<String> {
        self.failures.borrow().clone()
    }

    /// Check if all tests passed.
    pub fn all_passed(&self) -> bool {
        *self.failed.borrow() == 0
    }
}

/// Describe a test suite.
pub fn describe<F: FnOnce(&mut TestContext)>(description: &str, f: F) -> TestContext {
    let mut ctx = TestContext::new(description);
    f(&mut ctx);
    ctx
}

/// Run a describe block and assert all tests pass.
pub fn describe_and_assert<F: FnOnce(&mut TestContext)>(description: &str, f: F) {
    let ctx = describe(description, f);
    if !ctx.all_passed() {
        panic!(
            "Test suite '{}' failed: {} passed, {} failed\n{}",
            description,
            ctx.passed(),
            ctx.failed(),
            ctx.failures().join("\n")
        );
    }
}

// =============================================================================
// Expectations API
// =============================================================================

/// Wrapper for making assertions.
pub struct Expectation<T> {
    value: T,
    negated: bool,
}

/// Create an expectation from a value.
pub fn expect<T>(value: T) -> Expectation<T> {
    Expectation {
        value,
        negated: false,
    }
}

impl<T> Expectation<T> {
    /// Negate the expectation.
    pub fn not(mut self) -> Self {
        self.negated = !self.negated;
        self
    }
}

impl<T: PartialEq + std::fmt::Debug> Expectation<T> {
    /// Assert equality.
    pub fn to_equal(self, expected: T) {
        let matches = self.value == expected;
        if self.negated {
            if matches {
                panic!("Expected {:?} not to equal {:?}", self.value, expected);
            }
        } else if !matches {
            panic!("Expected {:?} to equal {:?}", self.value, expected);
        }
    }
}

impl<T: PartialOrd + std::fmt::Debug> Expectation<T> {
    /// Assert greater than.
    pub fn to_be_greater_than(self, other: T) {
        let matches = self.value > other;
        if self.negated {
            if matches {
                panic!(
                    "Expected {:?} not to be greater than {:?}",
                    self.value, other
                );
            }
        } else if !matches {
            panic!("Expected {:?} to be greater than {:?}", self.value, other);
        }
    }

    /// Assert less than.
    pub fn to_be_less_than(self, other: T) {
        let matches = self.value < other;
        if self.negated {
            if matches {
                panic!("Expected {:?} not to be less than {:?}", self.value, other);
            }
        } else if !matches {
            panic!("Expected {:?} to be less than {:?}", self.value, other);
        }
    }
}

impl Expectation<bool> {
    /// Assert true.
    pub fn to_be_true(self) {
        if self.negated {
            if self.value {
                panic!("Expected false but got true");
            }
        } else if !self.value {
            panic!("Expected true but got false");
        }
    }

    /// Assert false.
    pub fn to_be_false(self) {
        if self.negated {
            if !self.value {
                panic!("Expected true but got false");
            }
        } else if self.value {
            panic!("Expected false but got true");
        }
    }
}

impl<T> Expectation<Option<T>> {
    /// Assert Some.
    pub fn to_be_some(self) {
        let is_some = self.value.is_some();
        if self.negated {
            if is_some {
                panic!("Expected None but got Some");
            }
        } else if !is_some {
            panic!("Expected Some but got None");
        }
    }

    /// Assert None.
    pub fn to_be_none(self) {
        let is_none = self.value.is_none();
        if self.negated {
            if is_none {
                panic!("Expected Some but got None");
            }
        } else if !is_none {
            panic!("Expected None but got Some");
        }
    }
}

impl<T, E> Expectation<Result<T, E>> {
    /// Assert Ok.
    pub fn to_be_ok(self) {
        let is_ok = self.value.is_ok();
        if self.negated {
            if is_ok {
                panic!("Expected Err but got Ok");
            }
        } else if !is_ok {
            panic!("Expected Ok but got Err");
        }
    }

    /// Assert Err.
    pub fn to_be_err(self) {
        let is_err = self.value.is_err();
        if self.negated {
            if is_err {
                panic!("Expected Ok but got Err");
            }
        } else if !is_err {
            panic!("Expected Err but got Ok");
        }
    }
}

impl<T> Expectation<Vec<T>> {
    /// Assert empty.
    pub fn to_be_empty(self) {
        let is_empty = self.value.is_empty();
        if self.negated {
            if is_empty {
                panic!("Expected non-empty but got empty");
            }
        } else if !is_empty {
            panic!("Expected empty but got {} elements", self.value.len());
        }
    }

    /// Assert length.
    pub fn to_have_length(self, expected: usize) {
        let len = self.value.len();
        if self.negated {
            if len == expected {
                panic!("Expected length not to be {} but it was", expected);
            }
        } else if len != expected {
            panic!("Expected length {} but got {}", expected, len);
        }
    }
}

impl Expectation<&str> {
    /// Assert contains.
    pub fn to_contain(self, needle: &str) {
        let contains = self.value.contains(needle);
        if self.negated {
            if contains {
                panic!("Expected {:?} not to contain {:?}", self.value, needle);
            }
        } else if !contains {
            panic!("Expected {:?} to contain {:?}", self.value, needle);
        }
    }

    /// Assert starts with.
    pub fn to_start_with(self, prefix: &str) {
        let starts = self.value.starts_with(prefix);
        if self.negated {
            if starts {
                panic!("Expected {:?} not to start with {:?}", self.value, prefix);
            }
        } else if !starts {
            panic!("Expected {:?} to start with {:?}", self.value, prefix);
        }
    }

    /// Assert ends with.
    pub fn to_end_with(self, suffix: &str) {
        let ends = self.value.ends_with(suffix);
        if self.negated {
            if ends {
                panic!("Expected {:?} not to end with {:?}", self.value, suffix);
            }
        } else if !ends {
            panic!("Expected {:?} to end with {:?}", self.value, suffix);
        }
    }
}

impl Expectation<String> {
    /// Assert contains.
    pub fn to_contain(self, needle: &str) {
        let contains = self.value.contains(needle);
        if self.negated {
            if contains {
                panic!("Expected {:?} not to contain {:?}", self.value, needle);
            }
        } else if !contains {
            panic!("Expected {:?} to contain {:?}", self.value, needle);
        }
    }

    /// Assert empty.
    pub fn to_be_empty(self) {
        let is_empty = self.value.is_empty();
        if self.negated {
            if is_empty {
                panic!("Expected non-empty string but got empty");
            }
        } else if !is_empty {
            panic!("Expected empty string but got {:?}", self.value);
        }
    }
}

impl Expectation<f32> {
    /// Assert close to (within epsilon).
    pub fn to_be_close_to(self, expected: f32, epsilon: f32) {
        let diff = (self.value - expected).abs();
        let close = diff <= epsilon;
        if self.negated {
            if close {
                panic!(
                    "Expected {} not to be close to {} (within {})",
                    self.value, expected, epsilon
                );
            }
        } else if !close {
            panic!(
                "Expected {} to be close to {} (within {}), diff was {}",
                self.value, expected, epsilon, diff
            );
        }
    }
}

impl Expectation<f64> {
    /// Assert close to (within epsilon).
    pub fn to_be_close_to(self, expected: f64, epsilon: f64) {
        let diff = (self.value - expected).abs();
        let close = diff <= epsilon;
        if self.negated {
            if close {
                panic!(
                    "Expected {} not to be close to {} (within {})",
                    self.value, expected, epsilon
                );
            }
        } else if !close {
            panic!(
                "Expected {} to be close to {} (within {}), diff was {}",
                self.value, expected, epsilon, diff
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_describe_basic() {
        let ctx = describe("Math operations", |ctx| {
            ctx.it("adds numbers", |_| {
                expect(1 + 1).to_equal(2);
            });

            ctx.it("subtracts numbers", |_| {
                expect(5 - 3).to_equal(2);
            });
        });

        assert_eq!(ctx.passed(), 2);
        assert_eq!(ctx.failed(), 0);
        assert!(ctx.all_passed());
    }

    #[test]
    fn test_describe_with_failure() {
        let ctx = describe("Failing tests", |ctx| {
            ctx.it("passes", |_| {
                expect(1).to_equal(1);
            });

            ctx.it("fails", |_| {
                expect(1).to_equal(2);
            });
        });

        assert_eq!(ctx.passed(), 1);
        assert_eq!(ctx.failed(), 1);
        assert!(!ctx.all_passed());
    }

    #[test]
    fn test_expect_equality() {
        expect(42).to_equal(42);
        expect("hello").to_equal("hello");
        expect(vec![1, 2, 3]).to_equal(vec![1, 2, 3]);
    }

    #[test]
    fn test_expect_not() {
        expect(1).not().to_equal(2);
        expect(true).not().to_be_false();
    }

    #[test]
    fn test_expect_bool() {
        expect(true).to_be_true();
        expect(false).to_be_false();
        expect(1 > 0).to_be_true();
    }

    #[test]
    fn test_expect_comparison() {
        expect(10).to_be_greater_than(5);
        expect(3).to_be_less_than(7);
    }

    #[test]
    fn test_expect_option() {
        expect(Some(42)).to_be_some();
        expect(None::<i32>).to_be_none();
    }

    #[test]
    fn test_expect_result() {
        expect(Ok::<i32, &str>(42)).to_be_ok();
        expect(Err::<i32, &str>("error")).to_be_err();
    }

    #[test]
    fn test_expect_vec() {
        expect(Vec::<i32>::new()).to_be_empty();
        expect(vec![1, 2, 3]).to_have_length(3);
        expect(vec![1, 2]).not().to_be_empty();
    }

    #[test]
    fn test_expect_string() {
        expect("hello world").to_contain("world");
        expect("hello").to_start_with("hel");
        expect("hello").to_end_with("llo");
        expect("hello").not().to_contain("xyz");
    }

    #[test]
    fn test_expect_float_close_to() {
        expect(0.1 + 0.2_f32).to_be_close_to(0.3, 0.001);
        expect(1.23456_f64).to_be_close_to(1.23, 0.01);
    }

    #[test]
    fn test_before_after_hooks() {
        use std::cell::Cell;
        use std::rc::Rc;

        let counter = Rc::new(Cell::new(0));
        let counter_clone = counter.clone();
        let counter_clone2 = counter.clone();

        let ctx = describe("Hooks", |ctx| {
            ctx.before(move || {
                counter_clone.set(counter_clone.get() + 1);
            });

            ctx.after(move || {
                counter_clone2.set(counter_clone2.get() + 10);
            });

            ctx.it("first test", |_| {});
            ctx.it("second test", |_| {});
        });

        // 2 tests * (1 before + 10 after) = 22
        assert_eq!(counter.get(), 22);
        assert!(ctx.all_passed());
    }

    #[test]
    fn test_nested_describe() {
        let outer = describe("Outer", |outer_ctx| {
            outer_ctx.it("outer test", |_| {
                expect(true).to_be_true();
            });

            let inner = describe("Inner", |inner_ctx| {
                inner_ctx.it("inner test", |_| {
                    expect(1 + 1).to_equal(2);
                });
            });

            assert!(inner.all_passed());
        });

        assert!(outer.all_passed());
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_expect_string_owned_to_contain() {
        expect("hello world".to_string()).to_contain("world");
    }

    #[test]
    fn test_expect_string_owned_to_be_empty() {
        expect(String::new()).to_be_empty();
    }

    #[test]
    fn test_expect_string_owned_not_empty() {
        expect("hello".to_string()).not().to_be_empty();
    }

    #[test]
    fn test_expect_f32_close_to_negated() {
        expect(10.0_f32).not().to_be_close_to(1.0, 0.1);
    }

    #[test]
    fn test_expect_f64_close_to_negated() {
        expect(10.0_f64).not().to_be_close_to(1.0, 0.1);
    }

    #[test]
    fn test_expect_str_not_start_with() {
        expect("hello").not().to_start_with("xyz");
    }

    #[test]
    fn test_expect_str_not_end_with() {
        expect("hello").not().to_end_with("xyz");
    }

    #[test]
    fn test_expect_option_negated() {
        expect(Some(42)).not().to_be_none();
        expect(None::<i32>).not().to_be_some();
    }

    #[test]
    fn test_expect_result_negated() {
        expect(Ok::<i32, &str>(42)).not().to_be_err();
        expect(Err::<i32, &str>("error")).not().to_be_ok();
    }

    #[test]
    fn test_expect_vec_not_length() {
        expect(vec![1, 2, 3]).not().to_have_length(5);
    }

    #[test]
    fn test_expect_bool_negated() {
        expect(true).not().to_be_false();
        expect(false).not().to_be_true();
    }

    #[test]
    fn test_expect_comparison_negated() {
        expect(3).not().to_be_greater_than(10);
        expect(10).not().to_be_less_than(3);
    }

    #[test]
    fn test_expect_equality_negated() {
        expect(1).not().to_equal(2);
    }

    #[test]
    fn test_context_passed_plus_failed() {
        let ctx = describe("Test", |ctx| {
            ctx.it("pass1", |_| {});
            ctx.it("pass2", |_| {});
            ctx.it("fail", |_| {
                expect(1).to_equal(2);
            });
        });
        assert_eq!(ctx.passed() + ctx.failed(), 3);
    }
}
