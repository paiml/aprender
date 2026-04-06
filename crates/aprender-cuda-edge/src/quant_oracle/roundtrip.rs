//! Quantization roundtrip idempotence testing.
//!
//! Verifies that quantize→dequantize→quantize produces stable results.

use serde::{Deserialize, Serialize};

use super::boundary::QuantFormat;

/// Trait for types that can quantize and dequantize values.
pub trait Quantizer {
    /// Quantize a float value to the internal representation.
    fn quantize(&self, value: f64) -> i64;

    /// Dequantize the internal representation back to float.
    fn dequantize(&self, quantized: i64) -> f64;
}

/// Result of a roundtrip idempotence test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundtripResult {
    /// Original input value.
    pub original: f64,
    /// Value after quantize → dequantize.
    pub after_first: f64,
    /// Value after quantize → dequantize → quantize → dequantize.
    pub after_second: f64,
    /// Whether the roundtrip was idempotent (first == second).
    pub idempotent: bool,
}

/// Test that roundtrip is idempotent: q(d(q(x))) == q(x).
///
/// Returns true if the quantized representation is stable after one roundtrip.
pub fn roundtrip_idempotence<Q: Quantizer>(quantizer: &Q, value: f64) -> RoundtripResult {
    let q1 = quantizer.quantize(value);
    let d1 = quantizer.dequantize(q1);

    let q2 = quantizer.quantize(d1);
    let d2 = quantizer.dequantize(q2);

    RoundtripResult {
        original: value,
        after_first: d1,
        after_second: d2,
        idempotent: q1 == q2,
    }
}

/// A mock quantizer for testing purposes.
#[derive(Debug, Clone)]
pub struct MockQuantizer {
    /// Number of quantization levels.
    pub levels: i64,
    /// Scale factor.
    pub scale: f64,
}

impl MockQuantizer {
    /// Create a mock quantizer for the given format.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn for_format(format: QuantFormat) -> Self {
        Self {
            levels: format.levels() as i64,
            scale: 1.0,
        }
    }
}

impl Quantizer for MockQuantizer {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn quantize(&self, value: f64) -> i64 {
        let scaled = value * self.scale;
        let clamped = scaled.clamp(-(self.levels / 2) as f64, (self.levels / 2 - 1) as f64);
        clamped.round() as i64
    }

    #[allow(clippy::cast_precision_loss)]
    fn dequantize(&self, quantized: i64) -> f64 {
        quantized as f64 / self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_quantizer_roundtrip() {
        let q = MockQuantizer::for_format(QuantFormat::Q8_0);
        let result = roundtrip_idempotence(&q, 0.5);
        assert!(result.idempotent);
    }

    #[test]
    fn zero_is_idempotent() {
        let q = MockQuantizer::for_format(QuantFormat::Q4K);
        let result = roundtrip_idempotence(&q, 0.0);
        assert!(result.idempotent);
        assert!((result.after_first - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mock_quantizer_clamps() {
        let q = MockQuantizer {
            levels: 16,
            scale: 1.0,
        };
        // Value outside range gets clamped
        let quantized = q.quantize(100.0);
        assert_eq!(quantized, 7); // 16/2 - 1 = 7
    }
}
