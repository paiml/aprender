//! Digital clipping detection.
//!
//! Detects samples at or exceeding +/- 1.0 (digital full scale),
//! which indicates signal distortion.

use super::types::ClippingReport;

/// Threshold for clipping detection (samples at or above this are clipped).
const CLIPPING_THRESHOLD: f32 = 1.0;

/// Detect digital clipping in PCM samples.
///
/// Counts samples whose absolute value reaches or exceeds 1.0.
pub fn detect_clipping(samples: &[f32]) -> ClippingReport {
    if samples.is_empty() {
        return ClippingReport {
            clipped_samples: 0,
            clipped_pct: 0.0,
            passed: true,
        };
    }

    let clipped = samples
        .iter()
        .filter(|&&s| s.abs() >= CLIPPING_THRESHOLD)
        .count();

    #[allow(clippy::cast_precision_loss)]
    let pct = (clipped as f64 / samples.len() as f64) * 100.0;

    ClippingReport {
        clipped_samples: clipped,
        clipped_pct: pct,
        passed: clipped == 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_clipping() {
        let samples = vec![0.0f32, 0.5, -0.5, 0.99, -0.99];
        let report = detect_clipping(&samples);
        assert!(report.passed);
        assert_eq!(report.clipped_samples, 0);
        assert!(report.clipped_pct < f64::EPSILON);
    }

    #[test]
    fn test_clipping_at_positive_one() {
        let samples = vec![0.0f32, 0.5, 1.0, 0.8];
        let report = detect_clipping(&samples);
        assert!(!report.passed);
        assert_eq!(report.clipped_samples, 1);
    }

    #[test]
    fn test_clipping_at_negative_one() {
        let samples = vec![0.0f32, -1.0, 0.5];
        let report = detect_clipping(&samples);
        assert!(!report.passed);
        assert_eq!(report.clipped_samples, 1);
    }

    #[test]
    fn test_clipping_above_one() {
        let samples = vec![1.1f32, -1.2, 0.5];
        let report = detect_clipping(&samples);
        assert!(!report.passed);
        assert_eq!(report.clipped_samples, 2);
    }

    #[test]
    fn test_all_clipped() {
        let samples = vec![1.0f32; 100];
        let report = detect_clipping(&samples);
        assert!(!report.passed);
        assert_eq!(report.clipped_samples, 100);
        assert!((report.clipped_pct - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_samples() {
        let report = detect_clipping(&[]);
        assert!(report.passed);
        assert_eq!(report.clipped_samples, 0);
    }

    #[test]
    fn test_just_below_threshold() {
        let samples = vec![0.9999f32; 100];
        let report = detect_clipping(&samples);
        assert!(report.passed);
    }

    #[test]
    fn test_clipping_percentage() {
        let mut samples = vec![0.5f32; 90];
        samples.extend(vec![1.0f32; 10]);
        let report = detect_clipping(&samples);
        assert!(!report.passed);
        assert!((report.clipped_pct - 10.0).abs() < f64::EPSILON);
    }
}
