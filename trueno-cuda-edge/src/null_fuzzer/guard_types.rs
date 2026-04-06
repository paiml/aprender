//! Non-null device pointer guard and injection strategies.
//!
//! [`NonNullDevicePtr`] enforces at construction time that a GPU device
//! pointer is non-null (non-zero). [`InjectionStrategy`] controls when
//! null injection occurs during fuzzing.

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::error::EdgeError;

/// A GPU device pointer guaranteed to be non-null at construction.
///
/// Wraps a `u64` device address. Construction with address `0` returns
/// [`EdgeError::NullPointer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NonNullDevicePtr<T> {
    addr: u64,
    _marker: PhantomData<T>,
}

impl<T> NonNullDevicePtr<T> {
    /// Create a new non-null device pointer from a raw address.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeError::NullPointer`] if `addr` is 0.
    pub fn new(addr: u64) -> crate::error::Result<Self> {
        if addr == 0 {
            return Err(EdgeError::NullPointer {
                context: "cannot create NonNullDevicePtr from null address".into(),
            });
        }
        Ok(Self { addr, _marker: PhantomData })
    }

    /// Returns the raw device address.
    #[must_use]
    pub fn addr(&self) -> u64 {
        self.addr
    }
}

impl<T> std::fmt::Display for NonNullDevicePtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DevicePtr(0x{:016x})", self.addr)
    }
}

/// Strategy for injecting null pointers during fuzzing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InjectionStrategy {
    /// Inject null at a fixed interval (every N-th call).
    Periodic {
        /// Inject null every `interval` calls.
        interval: u64,
    },
    /// Inject null when an allocation exceeds a size threshold.
    SizeThreshold {
        /// Size threshold in bytes.
        threshold_bytes: u64,
    },
    /// Inject null with a given probability (0.0–1.0).
    Probabilistic {
        /// Probability of null injection per call.
        probability: f64,
    },
    /// Inject null at specific argument positions.
    Targeted {
        /// Zero-based argument indices to inject null into.
        arg_indices: Vec<u32>,
    },
}

impl InjectionStrategy {
    /// Returns whether this strategy should inject null at the given call index.
    ///
    /// For [`Probabilistic`](InjectionStrategy::Probabilistic), this uses a
    /// deterministic threshold check (not random) — inject if
    /// `(call_index % 100) < (probability * 100)`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn should_inject(&self, call_index: u64) -> bool {
        match self {
            Self::Periodic { interval } => *interval > 0 && call_index.is_multiple_of(*interval),
            Self::Probabilistic { probability } => {
                let threshold = (probability * 100.0) as u64;
                (call_index % 100) < threshold
            }
            // SizeThreshold and Targeted require additional context
            Self::SizeThreshold { .. } | Self::Targeted { .. } => false,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn non_null_ptr_rejects_zero() {
        let result = NonNullDevicePtr::<u8>::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn non_null_ptr_accepts_nonzero() {
        let ptr = NonNullDevicePtr::<u8>::new(0x1000).unwrap();
        assert_eq!(ptr.addr(), 0x1000);
    }

    #[test]
    fn non_null_ptr_display() {
        let ptr = NonNullDevicePtr::<f32>::new(0xDEAD_BEEF).unwrap();
        assert!(ptr.to_string().contains("deadbeef"));
    }

    #[test]
    fn non_null_ptr_equality() {
        let a = NonNullDevicePtr::<u8>::new(100).unwrap();
        let b = NonNullDevicePtr::<u8>::new(100).unwrap();
        let c = NonNullDevicePtr::<u8>::new(200).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn periodic_injection_at_interval() {
        let strategy = InjectionStrategy::Periodic { interval: 5 };
        assert!(strategy.should_inject(0));
        assert!(!strategy.should_inject(1));
        assert!(!strategy.should_inject(4));
        assert!(strategy.should_inject(5));
        assert!(strategy.should_inject(10));
    }

    #[test]
    fn periodic_injection_zero_interval_never_injects() {
        let strategy = InjectionStrategy::Periodic { interval: 0 };
        assert!(!strategy.should_inject(0));
        assert!(!strategy.should_inject(1));
    }

    #[test]
    fn size_threshold_does_not_inject_without_context() {
        let strategy = InjectionStrategy::SizeThreshold { threshold_bytes: 1024 };
        assert!(!strategy.should_inject(0));
    }

    #[test]
    fn probabilistic_injection() {
        let strategy = InjectionStrategy::Probabilistic { probability: 0.5 };
        // Deterministic: inject when (call_index % 100) < 50
        assert!(strategy.should_inject(0));
        assert!(strategy.should_inject(49));
        assert!(!strategy.should_inject(50));
        assert!(!strategy.should_inject(99));
    }

    #[test]
    fn targeted_does_not_inject_without_context() {
        let strategy = InjectionStrategy::Targeted { arg_indices: vec![0, 2] };
        assert!(!strategy.should_inject(0));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn nonzero_addr_always_succeeds(addr in 1u64..=u64::MAX) {
            let ptr = NonNullDevicePtr::<u8>::new(addr).unwrap();
            prop_assert_eq!(ptr.addr(), addr);
        }

        #[test]
        fn periodic_injection_deterministic(interval in 1u64..1000, idx in 0u64..10000) {
            let strategy = InjectionStrategy::Periodic { interval };
            let should = strategy.should_inject(idx);
            prop_assert_eq!(should, idx % interval == 0);
        }
    }
}
