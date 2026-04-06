//! Sparkline - Unicode Visualization (ENT-057)
//!
//! Unicode sparklines for inline metric visualization.
//! Reference: Tufte, E. R. (2006). *Beautiful Evidence*. Graphics Press.

/// Unicode sparkline characters for inline metric visualization.
pub const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Generate a sparkline string from a slice of values.
///
/// Uses Unicode block elements to create a compact inline chart.
///
/// # Arguments
///
/// * `values` - The values to visualize
/// * `width` - Maximum width (values will be subsampled if needed)
///
/// # Returns
///
/// A string of Unicode block characters representing the values.
pub fn sparkline(values: &[f32], width: usize) -> String {
    if values.is_empty() || width == 0 {
        return String::new();
    }

    // Subsample if needed
    let values: Vec<f32> = if values.len() > width {
        let step = values.len() as f32 / width as f32;
        (0..width)
            .map(|i| {
                let idx = (i as f32 * step) as usize;
                values[idx.min(values.len() - 1)]
            })
            .collect()
    } else {
        values.to_vec()
    };

    // Find extent
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    // Handle constant values
    if range < f32::EPSILON {
        return SPARK_CHARS[4].to_string().repeat(values.len());
    }

    // Map to sparkline characters
    values
        .iter()
        .map(|v| {
            let normalized = (v - min) / range;
            let idx = (normalized * 7.0).round() as usize;
            SPARK_CHARS[idx.min(7)]
        })
        .collect()
}

/// Generate a sparkline with custom range.
pub fn sparkline_range(values: &[f32], width: usize, min: f32, max: f32) -> String {
    if values.is_empty() || width == 0 {
        return String::new();
    }

    let range = max - min;
    if range < f32::EPSILON {
        return SPARK_CHARS[4].to_string().repeat(values.len().min(width));
    }

    let values: Vec<f32> = if values.len() > width {
        let step = values.len() as f32 / width as f32;
        (0..width).map(|i| values[(i as f32 * step) as usize]).collect()
    } else {
        values.to_vec()
    };

    values
        .iter()
        .map(|v| {
            let clamped = v.clamp(min, max);
            let normalized = (clamped - min) / range;
            let idx = (normalized * 7.0).round() as usize;
            SPARK_CHARS[idx.min(7)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparkline_empty() {
        assert_eq!(sparkline(&[], 10), "");
    }

    #[test]
    fn test_sparkline_zero_width() {
        assert_eq!(sparkline(&[1.0, 2.0, 3.0], 0), "");
    }

    #[test]
    fn test_sparkline_constant() {
        let result = sparkline(&[5.0, 5.0, 5.0, 5.0], 10);
        assert!(result.chars().all(|c| c == SPARK_CHARS[4]));
    }

    #[test]
    fn test_sparkline_ascending() {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let result = sparkline(&values, 8);
        // First should be low, last should be high
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], SPARK_CHARS[0]);
        assert_eq!(chars[7], SPARK_CHARS[7]);
    }

    #[test]
    fn test_sparkline_descending() {
        let values: Vec<f32> = (0..8).rev().map(|i| i as f32).collect();
        let result = sparkline(&values, 8);
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], SPARK_CHARS[7]);
        assert_eq!(chars[7], SPARK_CHARS[0]);
    }

    #[test]
    fn test_sparkline_subsampling() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = sparkline(&values, 10);
        assert_eq!(result.chars().count(), 10);
    }

    #[test]
    fn test_sparkline_range_empty() {
        assert_eq!(sparkline_range(&[], 10, 0.0, 1.0), "");
    }

    #[test]
    fn test_sparkline_range_clamping() {
        let values = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
        let result = sparkline_range(&values, 5, 0.0, 1.0);
        let chars: Vec<char> = result.chars().collect();
        // -1.0 clamped to 0.0 -> SPARK_CHARS[0]
        assert_eq!(chars[0], SPARK_CHARS[0]);
        // 2.0 clamped to 1.0 -> SPARK_CHARS[7]
        assert_eq!(chars[4], SPARK_CHARS[7]);
    }

    #[test]
    fn test_sparkline_range_zero_width() {
        assert_eq!(sparkline_range(&[1.0, 2.0], 0, 0.0, 1.0), "");
    }

    #[test]
    fn test_sparkline_range_constant_range() {
        let values = vec![5.0, 5.0, 5.0];
        let result = sparkline_range(&values, 5, 5.0, 5.0);
        // Constant range should produce middle sparkline char
        assert!(result.chars().all(|c| c == SPARK_CHARS[4]));
    }

    #[test]
    fn test_sparkline_range_subsampling() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = sparkline_range(&values, 10, 0.0, 99.0);
        assert_eq!(result.chars().count(), 10);
    }

    #[test]
    fn test_spark_chars_length() {
        assert_eq!(SPARK_CHARS.len(), 8);
    }

    #[test]
    fn test_sparkline_single_value() {
        let result = sparkline(&[5.0], 10);
        // Single value with constant range
        assert_eq!(result.chars().count(), 1);
    }

    #[test]
    fn test_sparkline_range_middle_value() {
        let result = sparkline_range(&[0.5], 1, 0.0, 1.0);
        let chars: Vec<char> = result.chars().collect();
        // 0.5 normalized = 0.5 * 7 = 3.5, rounds to 4
        assert_eq!(chars[0], SPARK_CHARS[4]);
    }
}
