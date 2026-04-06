//! Quantization boundary value generation.
//!
//! [`BoundaryValueGenerator`] produces critical test values for each
//! quantization format, including universal boundaries (0, NaN, Inf, −0)
//! and format-specific quantization levels.

use serde::{Deserialize, Serialize};

/// Supported quantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantFormat {
    /// 4-bit K-quant (block size 32, super-blocks of 8).
    Q4K,
    /// 5-bit K-quant.
    Q5K,
    /// 6-bit K-quant.
    Q6K,
    /// 8-bit symmetric quantization.
    Q8_0,
    /// 32-bit floating point (no quantization).
    F32,
    /// 16-bit floating point.
    F16,
}

impl QuantFormat {
    /// Returns the acceptable tolerance (epsilon) for parity checks.
    #[must_use]
    pub fn tolerance(&self) -> f64 {
        match self {
            Self::Q4K => 0.05,
            Self::Q5K => 0.02,
            Self::Q6K => 0.01,
            Self::Q8_0 => 0.005,
            Self::F16 => 0.001,
            Self::F32 => f64::EPSILON,
        }
    }

    /// Returns the number of quantization levels for this format.
    #[must_use]
    pub fn levels(&self) -> u64 {
        match self {
            Self::Q4K => 16,
            Self::Q5K => 32,
            Self::Q6K => 64,
            Self::Q8_0 => 256,
            Self::F16 => 65536,
            Self::F32 => u64::from(u32::MAX),
        }
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4K => write!(f, "Q4_K"),
            Self::Q5K => write!(f, "Q5_K"),
            Self::Q6K => write!(f, "Q6_K"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
        }
    }
}

/// Generates boundary test values for a specific quantization format.
#[derive(Debug, Clone)]
pub struct BoundaryValueGenerator {
    format: QuantFormat,
}

impl BoundaryValueGenerator {
    /// Create a generator for the given format.
    #[must_use]
    pub fn new(format: QuantFormat) -> Self {
        Self { format }
    }

    /// Returns the quantization format.
    #[must_use]
    pub fn format(&self) -> QuantFormat {
        self.format
    }

    /// Generate universal boundary values common to all formats.
    #[must_use]
    pub fn universal_boundaries(&self) -> Vec<f64> {
        vec![
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
        ]
    }

    /// Generate format-specific boundary values at quantization level edges.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn format_boundaries(&self) -> Vec<f64> {
        let levels = self.format.levels();
        let mut values = Vec::new();

        // Generate boundaries at quantization level transitions
        match self.format {
            // K-quants: enumerate all levels
            QuantFormat::Q4K | QuantFormat::Q5K | QuantFormat::Q6K => {
                for i in 0..levels {
                    let normalized = i as f64 / (levels - 1) as f64;
                    values.push(normalized);
                    values.push(-normalized);
                }
            }
            QuantFormat::Q8_0 => {
                // 8-bit: sample a subset of the 256 levels
                for i in (0..levels).step_by(16) {
                    let normalized = i as f64 / (levels - 1) as f64;
                    values.push(normalized);
                    values.push(-normalized);
                }
            }
            QuantFormat::F16 | QuantFormat::F32 => {
                // Float formats: test denormals, smallest/largest
                values.extend_from_slice(&[
                    f64::from(f32::MIN_POSITIVE),
                    f64::from(-f32::MIN_POSITIVE),
                    f64::from(f32::MAX),
                    f64::from(f32::MIN),
                    1.0e-38,
                    -1.0e-38,
                ]);
            }
        }

        values
    }

    /// Generate all boundary values (universal + format-specific).
    #[must_use]
    pub fn all_boundaries(&self) -> Vec<f64> {
        let mut values = self.universal_boundaries();
        values.extend(self.format_boundaries());
        values
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn q4k_tolerance() {
        assert!((QuantFormat::Q4K.tolerance() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn f32_tolerance_is_epsilon() {
        assert!((QuantFormat::F32.tolerance() - f64::EPSILON).abs() < f64::EPSILON);
    }

    #[test]
    fn q4k_has_16_levels() {
        assert_eq!(QuantFormat::Q4K.levels(), 16);
    }

    #[test]
    fn q6k_has_64_levels() {
        assert_eq!(QuantFormat::Q6K.levels(), 64);
    }

    #[test]
    fn universal_boundaries_contain_nan() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
        let bounds = gen.universal_boundaries();
        assert!(bounds.iter().any(|v| v.is_nan()));
    }

    #[test]
    fn universal_boundaries_contain_infinity() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
        let bounds = gen.universal_boundaries();
        assert!(bounds
            .iter()
            .any(|v| v.is_infinite() && v.is_sign_positive()));
        assert!(bounds
            .iter()
            .any(|v| v.is_infinite() && v.is_sign_negative()));
    }

    #[test]
    fn universal_boundaries_contain_zero() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
        let bounds = gen.universal_boundaries();
        assert!(bounds.contains(&0.0));
    }

    #[test]
    fn format_boundaries_q4k_has_correct_count() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
        let bounds = gen.format_boundaries();
        // 16 levels × 2 (positive + negative) = 32
        assert_eq!(bounds.len(), 32);
    }

    #[test]
    fn all_boundaries_includes_both() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q5K);
        let universal = gen.universal_boundaries();
        let format_specific = gen.format_boundaries();
        let all = gen.all_boundaries();
        assert_eq!(all.len(), universal.len() + format_specific.len());
    }

    #[test]
    fn quant_format_display() {
        assert_eq!(QuantFormat::Q4K.to_string(), "Q4_K");
        assert_eq!(QuantFormat::Q5K.to_string(), "Q5_K");
        assert_eq!(QuantFormat::Q6K.to_string(), "Q6_K");
        assert_eq!(QuantFormat::Q8_0.to_string(), "Q8_0");
        assert_eq!(QuantFormat::F32.to_string(), "F32");
        assert_eq!(QuantFormat::F16.to_string(), "F16");
    }

    #[test]
    fn format_boundaries_q8_0_has_correct_count() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q8_0);
        let bounds = gen.format_boundaries();
        // 256 levels, step_by(16) = 16 samples × 2 (positive + negative) = 32
        assert_eq!(bounds.len(), 32);
    }

    #[test]
    fn format_boundaries_f16_has_correct_count() {
        let gen = BoundaryValueGenerator::new(QuantFormat::F16);
        let bounds = gen.format_boundaries();
        // Float formats: 6 fixed values
        assert_eq!(bounds.len(), 6);
    }

    #[test]
    fn format_boundaries_f32_has_correct_count() {
        let gen = BoundaryValueGenerator::new(QuantFormat::F32);
        let bounds = gen.format_boundaries();
        // Float formats: 6 fixed values
        assert_eq!(bounds.len(), 6);
    }

    #[test]
    fn generator_format_accessor() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q6K);
        assert_eq!(gen.format(), QuantFormat::Q6K);
    }

    #[test]
    fn f16_levels() {
        assert_eq!(QuantFormat::F16.levels(), 65536);
    }

    #[test]
    fn f32_levels() {
        assert_eq!(QuantFormat::F32.levels(), u64::from(u32::MAX));
    }

    #[test]
    fn q5k_boundaries_count() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q5K);
        let bounds = gen.format_boundaries();
        // 32 levels × 2 = 64
        assert_eq!(bounds.len(), 64);
    }

    #[test]
    fn q6k_boundaries_count() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q6K);
        let bounds = gen.format_boundaries();
        // 64 levels × 2 = 128
        assert_eq!(bounds.len(), 128);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_quant_format() -> impl Strategy<Value = QuantFormat> {
        prop_oneof![
            Just(QuantFormat::Q4K),
            Just(QuantFormat::Q5K),
            Just(QuantFormat::Q6K),
            Just(QuantFormat::Q8_0),
            Just(QuantFormat::F16),
            Just(QuantFormat::F32),
        ]
    }

    proptest! {
        #[test]
        fn tolerance_is_positive(fmt in arb_quant_format()) {
            prop_assert!(fmt.tolerance() > 0.0);
        }
    }
}
