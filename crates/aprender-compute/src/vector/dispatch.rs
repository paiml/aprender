//! Backend dispatch macros for vector operations
//!
//! These macros handle routing operations to the appropriate SIMD backend
//! based on CPU feature detection and the selected Backend variant.

/// Macro to dispatch binary operations to appropriate backend
///
/// Routes operations like add, sub, mul, div to the best available SIMD backend.
/// Falls back to scalar implementation when a hardware backend is unavailable.
#[macro_export]
macro_rules! dispatch_binary_op {
    ($backend:expr, $op:ident, $a:expr, $b:expr, $result:expr) => {
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            match $backend {
                $crate::Backend::Scalar => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $b, $result)
                }
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::SSE2 | $crate::Backend::AVX => {
                    $crate::backends::sse2::Sse2Backend::$op($a, $b, $result)
                }
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::AVX2 => $crate::backends::avx2::Avx2Backend::$op($a, $b, $result),
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::AVX512 => {
                    $crate::backends::avx512::Avx512Backend::$op($a, $b, $result)
                }
                #[cfg(not(target_arch = "x86_64"))]
                $crate::Backend::SSE2
                | $crate::Backend::AVX
                | $crate::Backend::AVX2
                | $crate::Backend::AVX512 => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $b, $result)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                $crate::Backend::NEON => $crate::backends::neon::NeonBackend::$op($a, $b, $result),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                $crate::Backend::NEON => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $b, $result)
                }
                #[cfg(target_arch = "wasm32")]
                $crate::Backend::WasmSIMD => {
                    $crate::backends::wasm::WasmBackend::$op($a, $b, $result)
                }
                #[cfg(not(target_arch = "wasm32"))]
                $crate::Backend::WasmSIMD => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $b, $result)
                }
                $crate::Backend::GPU | $crate::Backend::Auto => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $b, $result)
                }
            }
        }
    };
}

/// Macro to dispatch reduction operations (return f32)
///
/// Routes operations like sum, max, min to the best available SIMD backend.
/// Falls back to scalar implementation when a hardware backend is unavailable.
#[macro_export]
macro_rules! dispatch_reduction {
    ($backend:expr, $op:ident, $data:expr) => {
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            match $backend {
                $crate::Backend::Scalar => $crate::backends::scalar::ScalarBackend::$op($data),
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::SSE2 | $crate::Backend::AVX => {
                    $crate::backends::sse2::Sse2Backend::$op($data)
                }
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::AVX2 => $crate::backends::avx2::Avx2Backend::$op($data),
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::AVX512 => $crate::backends::avx512::Avx512Backend::$op($data),
                #[cfg(not(target_arch = "x86_64"))]
                $crate::Backend::SSE2
                | $crate::Backend::AVX
                | $crate::Backend::AVX2
                | $crate::Backend::AVX512 => $crate::backends::scalar::ScalarBackend::$op($data),
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                $crate::Backend::NEON => $crate::backends::neon::NeonBackend::$op($data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                $crate::Backend::NEON => $crate::backends::scalar::ScalarBackend::$op($data),
                #[cfg(target_arch = "wasm32")]
                $crate::Backend::WasmSIMD => $crate::backends::wasm::WasmBackend::$op($data),
                #[cfg(not(target_arch = "wasm32"))]
                $crate::Backend::WasmSIMD => $crate::backends::scalar::ScalarBackend::$op($data),
                $crate::Backend::GPU | $crate::Backend::Auto => {
                    $crate::backends::scalar::ScalarBackend::$op($data)
                }
            }
        }
    };
}

/// Macro to dispatch unary operations (a -> result)
///
/// Routes operations like relu, sigmoid to the best available SIMD backend.
/// Falls back to scalar implementation when a hardware backend is unavailable.
#[macro_export]
macro_rules! dispatch_unary_op {
    ($backend:expr, $op:ident, $a:expr, $result:expr) => {
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            match $backend {
                $crate::Backend::Scalar => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $result)
                }
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::SSE2 | $crate::Backend::AVX => {
                    $crate::backends::sse2::Sse2Backend::$op($a, $result)
                }
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::AVX2 => $crate::backends::avx2::Avx2Backend::$op($a, $result),
                #[cfg(target_arch = "x86_64")]
                $crate::Backend::AVX512 => {
                    $crate::backends::avx512::Avx512Backend::$op($a, $result)
                }
                #[cfg(not(target_arch = "x86_64"))]
                $crate::Backend::SSE2
                | $crate::Backend::AVX
                | $crate::Backend::AVX2
                | $crate::Backend::AVX512 => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $result)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                $crate::Backend::NEON => $crate::backends::neon::NeonBackend::$op($a, $result),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                $crate::Backend::NEON => $crate::backends::scalar::ScalarBackend::$op($a, $result),
                #[cfg(target_arch = "wasm32")]
                $crate::Backend::WasmSIMD => $crate::backends::wasm::WasmBackend::$op($a, $result),
                #[cfg(not(target_arch = "wasm32"))]
                $crate::Backend::WasmSIMD => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $result)
                }
                $crate::Backend::GPU | $crate::Backend::Auto => {
                    $crate::backends::scalar::ScalarBackend::$op($a, $result)
                }
            }
        }
    };
}

pub use dispatch_binary_op;
pub use dispatch_reduction;
pub use dispatch_unary_op;
