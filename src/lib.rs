//! # certeza - Asymptotic Test Effectiveness Framework
//!
//! A scientific experiment into realistic provability with Rust.
//!
//! This crate provides a comprehensive framework for approaching asymptotic test effectiveness
//! in Rust software systems through the pragmatic integration of:
//!
//! - Property-based testing (using proptest)
//! - Mutation testing (using cargo-mutants)
//! - Structural coverage analysis
//! - Selective formal verification (using Kani)
//!
//! ## Tiered TDD-X Workflow
//!
//! The framework implements three-tiered verification:
//!
//! ### Tier 1: ON-SAVE (Sub-second feedback)
//! - Unit tests and focused property tests
//! - Static analysis (`cargo check`, `cargo clippy`)
//! - Enables rapid iteration in flow state
//!
//! ### Tier 2: ON-COMMIT (1-5 minutes)
//! - Full property-based test suite
//! - Coverage analysis (target: 95%+ line coverage)
//! - Integration tests
//!
//! ### Tier 3: ON-MERGE/NIGHTLY (Hours)
//! - Comprehensive mutation testing (target: >85% mutation score)
//! - Formal verification for critical paths
//! - Performance benchmarks
//!
//! ## Example
//!
//! ```rust
//! use certeza::TruenoVec;
//!
//! let mut vec = TruenoVec::new();
//! vec.push(1);
//! vec.push(2);
//! vec.push(3);
//!
//! assert_eq!(vec.len(), 3);
//! assert_eq!(vec.get(1), Some(&2));
//! assert_eq!(vec.pop(), Some(3));
//! ```
//!
//! ## Philosophy
//!
//! While complete verification remains theoretically impossible (Dijkstra: "testing can
//! only prove the presence of bugs, not their absence"), this framework provides a
//! reproducible methodology for achieving practical maximum confidence in critical systems.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

pub mod benchmark;
pub mod chaos;
pub mod vec;

pub use benchmark::*;
pub use vec::{IntoIter, Iter, IterMut, TruenoVec};

/// Adds two numbers together.
///
/// This is a simple example function demonstrating the testing framework.
///
/// # Examples
///
/// ```
/// use certeza::add;
///
/// assert_eq!(add(2, 2), 4);
/// assert_eq!(add(0, 0), 0);
/// assert_eq!(add(1, 1), 2);
/// ```
///
/// # Properties
///
/// This function satisfies several mathematical properties:
/// - Commutativity: `add(a, b) == add(b, a)`
/// - Associativity: `add(add(a, b), c) == add(a, add(b, c))`
/// - Identity: `add(a, 0) == a`
#[must_use]
pub const fn add(left: u64, right: u64) -> u64 {
    left + right
}

/// Subtracts one number from another.
///
/// # Examples
///
/// ```
/// use certeza::subtract;
///
/// assert_eq!(subtract(4, 2), 2);
/// assert_eq!(subtract(10, 5), 5);
/// ```
///
/// # Panics
///
/// This function will panic if `right > left` due to unsigned integer underflow.
/// Consider using checked arithmetic for production code.
#[must_use]
pub const fn subtract(left: u64, right: u64) -> u64 {
    left - right
}

/// Multiplies two numbers together.
///
/// # Examples
///
/// ```
/// use certeza::multiply;
///
/// assert_eq!(multiply(2, 3), 6);
/// assert_eq!(multiply(0, 100), 0);
/// ```
///
/// # Properties
///
/// This function satisfies several mathematical properties:
/// - Commutativity: `multiply(a, b) == multiply(b, a)`
/// - Associativity: `multiply(multiply(a, b), c) == multiply(a, multiply(b, c))`
/// - Identity: `multiply(a, 1) == a`
/// - Zero property: `multiply(a, 0) == 0`
#[must_use]
pub const fn multiply(left: u64, right: u64) -> u64 {
    left * right
}

// ============================================================================
// Unit Tests (Tier 1: Sub-second feedback)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Basic unit tests
    #[test]
    fn test_add_basic() {
        assert_eq!(add(2, 2), 4);
        assert_eq!(add(0, 0), 0);
        assert_eq!(add(1, 1), 2);
    }

    #[test]
    fn test_add_identity() {
        assert_eq!(add(5, 0), 5);
        assert_eq!(add(0, 5), 5);
    }

    #[test]
    fn test_add_commutativity() {
        assert_eq!(add(3, 7), add(7, 3));
        assert_eq!(add(10, 20), add(20, 10));
    }

    #[test]
    fn test_subtract_basic() {
        assert_eq!(subtract(10, 5), 5);
        assert_eq!(subtract(4, 2), 2);
    }

    #[test]
    fn test_subtract_identity() {
        assert_eq!(subtract(5, 0), 5);
        assert_eq!(subtract(100, 0), 100);
    }

    #[test]
    fn test_multiply_basic() {
        assert_eq!(multiply(2, 3), 6);
        assert_eq!(multiply(4, 5), 20);
    }

    #[test]
    fn test_multiply_identity() {
        assert_eq!(multiply(5, 1), 5);
        assert_eq!(multiply(1, 5), 5);
    }

    #[test]
    fn test_multiply_zero() {
        assert_eq!(multiply(0, 5), 0);
        assert_eq!(multiply(5, 0), 0);
        assert_eq!(multiply(0, 0), 0);
    }

    #[test]
    fn test_multiply_commutativity() {
        assert_eq!(multiply(3, 7), multiply(7, 3));
        assert_eq!(multiply(10, 20), multiply(20, 10));
    }
}

// ============================================================================
// Property-Based Tests (Tier 2: 1-5 minutes)
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Property: Addition is commutative
    proptest! {
        #[test]
        fn property_add_commutative(a in 0u64..1_000_000, b in 0u64..1_000_000) {
            prop_assert_eq!(add(a, b), add(b, a));
        }
    }

    // Property: Addition identity element is 0
    proptest! {
        #[test]
        fn property_add_identity(a in 0u64..u64::MAX) {
            prop_assert_eq!(add(a, 0), a);
            prop_assert_eq!(add(0, a), a);
        }
    }

    // Property: Addition is associative
    proptest! {
        #[test]
        fn property_add_associative(
            a in 0u64..1000,
            b in 0u64..1000,
            c in 0u64..1000,
        ) {
            // Limit values to prevent overflow
            let left = add(add(a, b), c);
            let right = add(a, add(b, c));
            prop_assert_eq!(left, right);
        }
    }

    // Property: Subtraction identity
    proptest! {
        #[test]
        fn property_subtract_identity(a in 0u64..u64::MAX) {
            prop_assert_eq!(subtract(a, 0), a);
        }
    }

    // Property: Subtraction inverse of addition
    proptest! {
        #[test]
        fn property_subtract_inverse(a in 0u64..1_000_000, b in 0u64..1_000_000) {
            // Use checked_add to prevent overflow
            if let Some(sum) = a.checked_add(b) {
                prop_assert_eq!(subtract(sum, b), a);
            }
        }
    }

    // Property: Multiplication is commutative
    proptest! {
        #[test]
        fn property_multiply_commutative(a in 0u64..100_000, b in 0u64..100_000) {
            prop_assert_eq!(multiply(a, b), multiply(b, a));
        }
    }

    // Property: Multiplication identity is 1
    proptest! {
        #[test]
        fn property_multiply_identity(a in 0u64..u64::MAX) {
            prop_assert_eq!(multiply(a, 1), a);
            prop_assert_eq!(multiply(1, a), a);
        }
    }

    // Property: Multiplication by zero
    proptest! {
        #[test]
        fn property_multiply_zero(a in 0u64..u64::MAX) {
            prop_assert_eq!(multiply(a, 0), 0);
            prop_assert_eq!(multiply(0, a), 0);
        }
    }

    // Property: Multiplication is associative
    proptest! {
        #[test]
        fn property_multiply_associative(
            a in 1u64..100,
            b in 1u64..100,
            c in 1u64..100,
        ) {
            // Limit values to prevent overflow
            let left = multiply(multiply(a, b), c);
            let right = multiply(a, multiply(b, c));
            prop_assert_eq!(left, right);
        }
    }

    // Property: Distributive property
    proptest! {
        #[test]
        fn property_distributive(
            a in 0u64..1000,
            b in 0u64..1000,
            c in 0u64..1000,
        ) {
            // a * (b + c) == (a * b) + (a * c)
            let left = multiply(a, add(b, c));
            let right = add(multiply(a, b), multiply(a, c));
            prop_assert_eq!(left, right);
        }
    }
}
