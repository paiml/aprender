//! Reference Curve Overlay (ENT-067)
//!
//! Compare current training with a "golden" reference run.

use super::sparkline::sparkline_range;

/// Reference curve for comparison with current training run.
#[derive(Debug, Clone)]
pub struct ReferenceCurve {
    /// Reference values (from a "golden" run)
    values: Vec<f32>,
    /// Tolerance for deviation detection
    tolerance: f32,
}

impl ReferenceCurve {
    /// Create from a vector of reference values.
    pub fn new(values: Vec<f32>, tolerance: f32) -> Self {
        Self { values, tolerance }
    }

    /// Load from JSON file.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let values: Vec<f32> = serde_json::from_str(json)?;
        Ok(Self::new(values, 0.1))
    }

    /// Get reference value at epoch.
    pub fn get(&self, epoch: usize) -> Option<f32> {
        self.values.get(epoch).copied()
    }

    /// Check if current value deviates from reference.
    pub fn check_deviation(&self, epoch: usize, current: f32) -> Option<f32> {
        if let Some(reference) = self.get(epoch) {
            let deviation = (current - reference).abs() / reference.abs().max(f32::EPSILON);
            if deviation > self.tolerance {
                return Some(deviation);
            }
        }
        None
    }

    /// Generate comparison sparkline.
    pub fn comparison_sparkline(&self, current: &[f32], width: usize) -> String {
        let len = current.len().min(self.values.len());
        if len == 0 {
            return String::new();
        }

        // Show deviation from reference
        let deviations: Vec<f32> = current
            .iter()
            .zip(self.values.iter())
            .map(|(c, r)| (c - r) / r.abs().max(f32::EPSILON))
            .collect();

        // Use signed sparkline (negative = better, positive = worse for loss)
        sparkline_range(&deviations, width, -0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_curve_new() {
        let curve = ReferenceCurve::new(vec![1.0, 0.8, 0.6], 0.1);
        assert_eq!(curve.get(0), Some(1.0));
        assert_eq!(curve.get(1), Some(0.8));
        assert_eq!(curve.get(2), Some(0.6));
        assert_eq!(curve.get(3), None);
    }

    #[test]
    fn test_reference_curve_from_json() {
        let json = "[1.0, 0.8, 0.6, 0.4]";
        let curve = ReferenceCurve::from_json(json).expect("operation should succeed");
        assert_eq!(curve.get(0), Some(1.0));
        assert_eq!(curve.get(3), Some(0.4));
    }

    #[test]
    fn test_reference_curve_from_json_invalid() {
        let json = "not valid json";
        assert!(ReferenceCurve::from_json(json).is_err());
    }

    #[test]
    fn test_reference_curve_check_deviation_within_tolerance() {
        let curve = ReferenceCurve::new(vec![1.0], 0.1);
        // 1.05 is 5% off from 1.0, within 10% tolerance
        assert!(curve.check_deviation(0, 1.05).is_none());
    }

    #[test]
    fn test_reference_curve_check_deviation_exceeds_tolerance() {
        let curve = ReferenceCurve::new(vec![1.0], 0.1);
        // 1.15 is 15% off from 1.0, exceeds 10% tolerance
        let deviation = curve.check_deviation(0, 1.15);
        assert!(deviation.is_some());
        assert!((deviation.expect("operation should succeed") - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_reference_curve_check_deviation_no_reference() {
        let curve = ReferenceCurve::new(vec![1.0], 0.1);
        assert!(curve.check_deviation(5, 1.0).is_none());
    }

    #[test]
    fn test_reference_curve_comparison_sparkline() {
        let curve = ReferenceCurve::new(vec![1.0, 0.8, 0.6, 0.4], 0.1);
        let current = vec![1.0, 0.9, 0.5, 0.4];
        let sparkline = curve.comparison_sparkline(&current, 4);
        assert_eq!(sparkline.chars().count(), 4);
    }

    #[test]
    fn test_reference_curve_comparison_sparkline_empty() {
        let curve = ReferenceCurve::new(vec![], 0.1);
        let current = vec![1.0];
        assert_eq!(curve.comparison_sparkline(&current, 4), "");
    }
}
