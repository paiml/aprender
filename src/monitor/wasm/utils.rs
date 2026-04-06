//! Utility functions for WASM dashboard rendering.

/// Normalize values to 0.0-1.0 range.
pub fn normalize_values(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-10 {
        // All values are the same
        return vec![0.5; values.len()];
    }

    values.iter().map(|v| (v - min) / (max - min)).collect()
}

/// Generate a sparkline string from values.
pub fn generate_sparkline(values: &[f64], max_len: usize) -> String {
    const CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    if values.is_empty() {
        return String::new();
    }

    // Subsample if needed
    let subsampled: Vec<f64> = if values.len() > max_len {
        let step = values.len() as f64 / max_len as f64;
        (0..max_len).map(|i| values[(i as f64 * step) as usize]).collect()
    } else {
        values.to_vec()
    };

    let normalized = normalize_values(&subsampled);
    normalized
        .iter()
        .map(|&v| {
            let idx = ((v * 7.0).round() as usize).min(7);
            CHARS[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_values_empty() {
        let result = normalize_values(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_normalize_values_single() {
        let result = normalize_values(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_values_range() {
        let result = normalize_values(&[0.0, 5.0, 10.0]);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_values_constant() {
        let result = normalize_values(&[5.0, 5.0, 5.0]);
        assert_eq!(result.len(), 3);
        // All same value should return 0.5
        for v in &result {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_generate_sparkline_empty() {
        let result = generate_sparkline(&[], 20);
        assert!(result.is_empty());
    }

    #[test]
    fn test_generate_sparkline_basic() {
        let result = generate_sparkline(&[0.0, 0.5, 1.0], 20);
        assert_eq!(result.chars().count(), 3);
    }

    #[test]
    fn test_generate_sparkline_subsample() {
        let values: Vec<f64> = (0..100).map(|i| f64::from(i)).collect();
        let result = generate_sparkline(&values, 20);
        assert_eq!(result.chars().count(), 20);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Normalization always produces values in [0, 1]
        #[test]
        fn prop_normalize_bounded(values in prop::collection::vec(-1000.0f64..1000.0, 2..100)) {
            let normalized = normalize_values(&values);

            for v in &normalized {
                prop_assert!(*v >= 0.0 - 1e-10);
                prop_assert!(*v <= 1.0 + 1e-10);
            }
        }

        /// Property: Normalization preserves length
        #[test]
        fn prop_normalize_preserves_length(values in prop::collection::vec(-1000.0f64..1000.0, 1..100)) {
            let normalized = normalize_values(&values);
            prop_assert_eq!(normalized.len(), values.len());
        }

        /// Property: Sparkline length is bounded
        #[test]
        fn prop_sparkline_bounded(values in prop::collection::vec(0.0f64..100.0, 1..200)) {
            let sparkline = generate_sparkline(&values, 20);
            let char_count = sparkline.chars().count();
            prop_assert!(char_count <= 20);
        }

        /// Property: Sparkline chars are valid Unicode blocks
        #[test]
        fn prop_sparkline_valid_chars(values in prop::collection::vec(0.0f64..100.0, 1..50)) {
            let sparkline = generate_sparkline(&values, 20);
            let valid_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

            for c in sparkline.chars() {
                prop_assert!(valid_chars.contains(&c));
            }
        }
    }
}
