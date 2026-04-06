//! SIMD capability detection and abstraction.

use serde::{Deserialize, Serialize};

/// SIMD capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SimdCapability {
    /// No SIMD support
    #[default]
    None,
    /// SSE4.1/4.2 (128-bit)
    Sse4,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
}

impl SimdCapability {
    /// Returns the vector width in bits
    pub fn vector_width_bits(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Sse4 => 128,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
            Self::Neon => 128,
        }
    }

    /// Detect SIMD capability of current CPU
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        if is_x86_feature_detected!("avx512f") {
            Self::Avx512
        } else if is_x86_feature_detected!("avx2") {
            Self::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            Self::Sse4
        } else {
            Self::None
        }
    }

    /// Detect SIMD capability of current CPU (ARM)
    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        // NEON is mandatory on aarch64
        Self::Neon
    }

    /// Detect SIMD capability (fallback for other architectures)
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn detect() -> Self {
        Self::None
    }
}

impl std::fmt::Display for SimdCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Sse4 => write!(f, "SSE4"),
            Self::Avx2 => write!(f, "AVX2"),
            Self::Avx512 => write!(f, "AVX-512"),
            Self::Neon => write!(f, "NEON"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capability_default() {
        assert_eq!(SimdCapability::default(), SimdCapability::None);
    }

    #[test]
    fn test_vector_width_bits_none() {
        assert_eq!(SimdCapability::None.vector_width_bits(), 0);
    }

    #[test]
    fn test_vector_width_bits_sse4() {
        assert_eq!(SimdCapability::Sse4.vector_width_bits(), 128);
    }

    #[test]
    fn test_vector_width_bits_avx2() {
        assert_eq!(SimdCapability::Avx2.vector_width_bits(), 256);
    }

    #[test]
    fn test_vector_width_bits_avx512() {
        assert_eq!(SimdCapability::Avx512.vector_width_bits(), 512);
    }

    #[test]
    fn test_vector_width_bits_neon() {
        assert_eq!(SimdCapability::Neon.vector_width_bits(), 128);
    }

    #[test]
    fn test_simd_capability_display_none() {
        assert_eq!(SimdCapability::None.to_string(), "none");
    }

    #[test]
    fn test_simd_capability_display_sse4() {
        assert_eq!(SimdCapability::Sse4.to_string(), "SSE4");
    }

    #[test]
    fn test_simd_capability_display_avx2() {
        assert_eq!(SimdCapability::Avx2.to_string(), "AVX2");
    }

    #[test]
    fn test_simd_capability_display_avx512() {
        assert_eq!(SimdCapability::Avx512.to_string(), "AVX-512");
    }

    #[test]
    fn test_simd_capability_display_neon() {
        assert_eq!(SimdCapability::Neon.to_string(), "NEON");
    }

    #[test]
    fn test_simd_capability_detect() {
        let detected = SimdCapability::detect();
        // Just verify it returns one of the valid variants
        let _ = detected.vector_width_bits(); // Should not panic
    }

    #[test]
    fn test_simd_capability_clone() {
        let cap = SimdCapability::Avx2;
        let cloned = cap;
        assert_eq!(cap, cloned);
    }

    #[test]
    fn test_simd_capability_eq() {
        assert_eq!(SimdCapability::Avx2, SimdCapability::Avx2);
        assert_ne!(SimdCapability::Avx2, SimdCapability::Avx512);
    }

    #[test]
    fn test_simd_capability_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SimdCapability::Avx2);
        set.insert(SimdCapability::Avx2);
        assert_eq!(set.len(), 1);
        set.insert(SimdCapability::Avx512);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_simd_capability_serde() {
        let cap = SimdCapability::Avx512;
        let json = serde_json::to_string(&cap).expect("JSON serialization should succeed");
        let deserialized: SimdCapability =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(cap, deserialized);
    }

    #[test]
    fn test_simd_capability_debug() {
        assert_eq!(format!("{:?}", SimdCapability::None), "None");
        assert_eq!(format!("{:?}", SimdCapability::Sse4), "Sse4");
        assert_eq!(format!("{:?}", SimdCapability::Avx2), "Avx2");
        assert_eq!(format!("{:?}", SimdCapability::Avx512), "Avx512");
        assert_eq!(format!("{:?}", SimdCapability::Neon), "Neon");
    }
}
