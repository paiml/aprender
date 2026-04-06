//! Easing curve verification.
//!
//! Verifies that rendered animation curves match expected easing functions
//! by comparing sampled keyframe values against the mathematical model.

use super::types::EasingFunction;

/// A sampled keyframe from a rendered animation.
#[derive(Clone, Debug)]
pub struct Keyframe {
    /// Normalized time (0.0-1.0)
    pub t: f64,
    /// Observed value (0.0-1.0)
    pub value: f64,
}

/// Easing curve verification result.
#[derive(Clone, Debug)]
pub struct EasingVerification {
    /// Expected easing function
    pub expected: EasingFunction,
    /// Maximum deviation from expected curve
    pub max_deviation: f64,
    /// Mean deviation
    pub mean_deviation: f64,
    /// Whether verification passed
    pub passed: bool,
    /// Per-keyframe deviations
    pub deviations: Vec<f64>,
}

/// Verify sampled keyframes against an expected easing function.
///
/// For each keyframe, evaluates the expected easing at time t
/// and compares against the observed value.
#[must_use]
pub fn verify_easing(
    keyframes: &[Keyframe],
    expected: &EasingFunction,
    tolerance: f64,
) -> EasingVerification {
    if keyframes.is_empty() {
        return EasingVerification {
            expected: expected.clone(),
            max_deviation: 0.0,
            mean_deviation: 0.0,
            passed: true,
            deviations: Vec::new(),
        };
    }

    let mut max_dev: f64 = 0.0;
    let mut sum_dev: f64 = 0.0;
    let mut deviations = Vec::with_capacity(keyframes.len());

    for kf in keyframes {
        let expected_value = expected.evaluate(kf.t);
        let deviation = (kf.value - expected_value).abs();
        deviations.push(deviation);
        sum_dev += deviation;
        if deviation > max_dev {
            max_dev = deviation;
        }
    }

    let mean_dev = sum_dev / keyframes.len() as f64;
    let passed = max_dev <= tolerance;

    EasingVerification {
        expected: expected.clone(),
        max_deviation: max_dev,
        mean_deviation: mean_dev,
        passed,
        deviations,
    }
}

/// Sample an easing function at N equally spaced points.
///
/// Useful for generating reference curves.
#[must_use]
pub fn sample_easing(easing: &EasingFunction, num_samples: usize) -> Vec<Keyframe> {
    if num_samples == 0 {
        return Vec::new();
    }
    if num_samples == 1 {
        return vec![Keyframe {
            t: 0.0,
            value: easing.evaluate(0.0),
        }];
    }

    (0..num_samples)
        .map(|i| {
            let t = i as f64 / (num_samples - 1) as f64;
            Keyframe {
                t,
                value: easing.evaluate(t),
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_easing_perfect_match() {
        let easing = EasingFunction::Linear;
        let keyframes: Vec<Keyframe> = (0..=10)
            .map(|i| {
                let t = i as f64 / 10.0;
                Keyframe { t, value: t }
            })
            .collect();
        let result = verify_easing(&keyframes, &easing, 0.01);
        assert!(result.passed);
        assert!(result.max_deviation < 0.001);
    }

    #[test]
    fn test_verify_easing_mismatch() {
        let easing = EasingFunction::EaseIn; // quadratic
                                             // Provide linear values instead
        let keyframes: Vec<Keyframe> = (0..=10)
            .map(|i| {
                let t = i as f64 / 10.0;
                Keyframe { t, value: t } // linear, not quadratic
            })
            .collect();
        let result = verify_easing(&keyframes, &easing, 0.01);
        assert!(!result.passed);
        assert!(result.max_deviation > 0.01);
    }

    #[test]
    fn test_verify_easing_empty() {
        let easing = EasingFunction::Linear;
        let result = verify_easing(&[], &easing, 0.01);
        assert!(result.passed);
        assert!(result.deviations.is_empty());
    }

    #[test]
    fn test_verify_easing_within_tolerance() {
        let easing = EasingFunction::Linear;
        let keyframes = vec![
            Keyframe {
                t: 0.5,
                value: 0.505,
            }, // 0.5% off
        ];
        let result = verify_easing(&keyframes, &easing, 0.01);
        assert!(result.passed);
    }

    #[test]
    fn test_sample_easing_linear() {
        let samples = sample_easing(&EasingFunction::Linear, 11);
        assert_eq!(samples.len(), 11);
        assert!((samples[0].t).abs() < f64::EPSILON);
        assert!((samples[0].value).abs() < f64::EPSILON);
        assert!((samples[10].t - 1.0).abs() < f64::EPSILON);
        assert!((samples[10].value - 1.0).abs() < f64::EPSILON);
        // Check midpoint
        assert!((samples[5].t - 0.5).abs() < f64::EPSILON);
        assert!((samples[5].value - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sample_easing_ease_in() {
        let samples = sample_easing(&EasingFunction::EaseIn, 11);
        assert_eq!(samples.len(), 11);
        // Ease-in: midpoint should be below linear
        assert!(samples[5].value < 0.5);
    }

    #[test]
    fn test_sample_easing_empty() {
        let samples = sample_easing(&EasingFunction::Linear, 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_sample_easing_single() {
        let samples = sample_easing(&EasingFunction::Linear, 1);
        assert_eq!(samples.len(), 1);
        assert!((samples[0].t).abs() < f64::EPSILON);
    }

    #[test]
    fn test_deviations_count() {
        let easing = EasingFunction::Linear;
        let keyframes = vec![
            Keyframe { t: 0.0, value: 0.0 },
            Keyframe { t: 0.5, value: 0.5 },
            Keyframe { t: 1.0, value: 1.0 },
        ];
        let result = verify_easing(&keyframes, &easing, 0.01);
        assert_eq!(result.deviations.len(), 3);
    }

    #[test]
    fn test_mean_deviation() {
        let easing = EasingFunction::Linear;
        let keyframes = vec![
            Keyframe {
                t: 0.5,
                value: 0.52,
            }, // 0.02 off
            Keyframe {
                t: 0.8,
                value: 0.84,
            }, // 0.04 off
        ];
        let result = verify_easing(&keyframes, &easing, 0.05);
        assert!((result.mean_deviation - 0.03).abs() < 0.001);
    }
}
