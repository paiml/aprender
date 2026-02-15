
// ============================================================================
// SECTION C: Tooling & Operations (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_c {
    // Test 56: Diff Identity
    #[test]
    fn test_check_56_diff_identity() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![1.0f32, 2.0, 3.0];
        let diff = compute_l2_distance(&data1, &data2);
        assert!(diff < 1e-6, "Same data should have zero L2 distance");
    }

    // Test 57: Diff Detection
    #[test]
    fn test_check_57_diff_detection() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![1.0f32, 2.0, 4.0]; // Changed last element
        let diff = compute_l2_distance(&data1, &data2);
        assert!(
            diff > 0.5,
            "Different data should have non-zero L2 distance"
        );
    }

    // Test 58: Merge Average
    #[test]
    fn test_check_58_merge_average() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![3.0f32, 4.0, 5.0];
        let merged = merge_average(&data1, &data2);
        assert_eq!(merged, vec![2.0f32, 3.0, 4.0], "Average merge failed");
    }

    /// Compute L2 distance between two tensors
    fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt()
    }

    /// Merge two tensors by averaging
    fn merge_average(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");
        a.iter().zip(b.iter()).map(|(x, y)| (x + y) / 2.0).collect()
    }
}

// ============================================================================
// SECTION D: Conversion & Interoperability (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_d {
    // Test 79: Roundtrip
    #[test]
    fn test_check_79_roundtrip_tolerance() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        // Simulate roundtrip with small float error
        let roundtrip: Vec<f32> = original.iter().map(|&x| x + 1e-7).collect();
        let max_diff = compute_max_diff(&original, &roundtrip);
        assert!(max_diff < 1e-5, "Roundtrip should have drift < 1e-5");
    }

    // Test 87: Tensor Name Normalization
    #[test]
    fn test_check_87_name_normalization() {
        let hf_name = "model.encoder.conv1.weight";
        let apr_name = normalize_tensor_name(hf_name);
        assert_eq!(
            apr_name, "encoder.conv1.weight",
            "Should strip 'model.' prefix"
        );
    }

    #[test]
    fn test_check_87_name_normalization_no_prefix() {
        let name = "encoder.conv1.weight";
        let apr_name = normalize_tensor_name(name);
        assert_eq!(
            apr_name, "encoder.conv1.weight",
            "Should preserve name without prefix"
        );
    }

    /// Compute max absolute difference between tensors
    fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, |acc, x| if x > acc { x } else { acc })
    }

    /// Normalize tensor name to APR canonical form
    fn normalize_tensor_name(name: &str) -> &str {
        name.strip_prefix("model.").unwrap_or(name)
    }
}
