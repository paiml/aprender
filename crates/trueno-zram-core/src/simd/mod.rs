//! SIMD detection and dispatch.

mod detect;

pub use detect::{detect_backend, is_available, SimdFeatures};

use crate::SimdBackend;

/// Get the best available SIMD backend for this CPU.
#[must_use]
pub fn best_backend() -> SimdBackend {
    detect_backend()
}

/// Check if a specific backend is available.
#[must_use]
pub fn backend_available(backend: SimdBackend) -> bool {
    is_available(backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_always_available() {
        assert!(backend_available(SimdBackend::Scalar));
    }

    #[test]
    fn test_best_backend_returns_valid() {
        let backend = best_backend();
        assert!(backend_available(backend));
    }

    #[test]
    fn test_best_backend_at_least_scalar() {
        let backend = best_backend();
        // Should always be at least scalar
        assert!(matches!(
            backend,
            SimdBackend::Scalar
                | SimdBackend::Sse42
                | SimdBackend::Avx2
                | SimdBackend::Avx512
                | SimdBackend::Neon
        ));
    }
}
