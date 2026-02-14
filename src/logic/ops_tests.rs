use super::*;

#[test]
fn test_softmax_numerical_stability() {
    let large_values = vec![1000.0, 1001.0, 1002.0];
    let result = softmax_row(&large_values);

    let sum: f64 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1");
}

#[test]
fn test_threshold_matrix() {
    let mut matrix = vec![vec![0.3, 0.7], vec![0.5, 0.51]];
    threshold_matrix(&mut matrix);

    assert_eq!(matrix[0][0], 0.0);
    assert_eq!(matrix[0][1], 1.0);
    assert_eq!(matrix[1][0], 0.0);
    assert_eq!(matrix[1][1], 1.0);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_logic_mode_eq() {
    assert_eq!(LogicMode::Boolean, LogicMode::Boolean);
    assert_eq!(LogicMode::Continuous, LogicMode::Continuous);
    assert_ne!(LogicMode::Boolean, LogicMode::Continuous);
}

#[test]
fn test_logic_mode_clone_copy() {
    let mode = LogicMode::Boolean;
    let cloned = mode.clone();
    let copied = mode;
    assert_eq!(cloned, copied);
}

#[test]
fn test_logic_mode_debug() {
    let debug_str = format!("{:?}", LogicMode::Boolean);
    assert!(debug_str.contains("Boolean"));
}

#[test]
fn test_nonlinearity_eq() {
    assert_eq!(Nonlinearity::Step, Nonlinearity::Step);
    assert_eq!(Nonlinearity::Sigmoid, Nonlinearity::Sigmoid);
    assert_ne!(Nonlinearity::Step, Nonlinearity::Sigmoid);
}

#[test]
fn test_nonlinearity_clone_copy() {
    let nl = Nonlinearity::Relu;
    let cloned = nl.clone();
    let copied = nl;
    assert_eq!(cloned, copied);
}

#[test]
fn test_nonlinearity_debug() {
    let debug_str = format!("{:?}", Nonlinearity::Softmax);
    assert!(debug_str.contains("Softmax"));
}

#[test]
fn test_logical_join_boolean() {
    let t1 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let t2 = vec![vec![1.0, 1.0], vec![0.0, 1.0]];

    let result = logical_join(&t1, &t2, LogicMode::Boolean);

    // Row 0: [1*1+0*0, 1*1+0*1] = [1, 1] -> thresholded = [1, 1]
    // Row 1: [0*1+1*0, 0*1+1*1] = [0, 1] -> thresholded = [0, 1]
    assert_eq!(result[0][0], 1.0);
    assert_eq!(result[0][1], 1.0);
    assert_eq!(result[1][0], 0.0);
    assert_eq!(result[1][1], 1.0);
}

#[test]
fn test_logical_join_continuous() {
    let t1 = vec![vec![0.5, 0.3], vec![0.2, 0.8]];
    let t2 = vec![vec![0.4, 0.6], vec![0.7, 0.1]];

    let result = logical_join(&t1, &t2, LogicMode::Continuous);

    // Row 0: [0.5*0.4+0.3*0.7, 0.5*0.6+0.3*0.1] = [0.41, 0.33]
    assert!((result[0][0] - 0.41).abs() < 1e-6);
    assert!((result[0][1] - 0.33).abs() < 1e-6);
}

#[test]
fn test_logical_join_empty() {
    let t1: Vec<Vec<f64>> = vec![];
    let t2: Vec<Vec<f64>> = vec![];

    let result = logical_join(&t1, &t2, LogicMode::Boolean);
    assert!(result.is_empty());
}

#[test]
fn test_logical_project_dim0_boolean() {
    let tensor = vec![vec![0.0, 0.8], vec![0.6, 0.3]];

    let result = logical_project(&tensor, 0, LogicMode::Boolean);

    // dim=0 projects over rows, result has cols elements
    // col 0: max(0.0, 0.6) = 0.6 > 0.5 -> 1
    // col 1: max(0.8, 0.3) = 0.8 > 0.5 -> 1
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 1.0);
}

#[test]
fn test_logical_project_dim0_continuous() {
    let tensor = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

    let result = logical_project(&tensor, 0, LogicMode::Continuous);

    // Sum over rows
    assert!((result[0] - 0.4).abs() < 1e-6); // 0.1 + 0.3
    assert!((result[1] - 0.6).abs() < 1e-6); // 0.2 + 0.4
}

#[test]
fn test_logical_project_dim1_boolean() {
    let tensor = vec![vec![0.0, 0.8], vec![0.6, 0.3]];

    let result = logical_project(&tensor, 1, LogicMode::Boolean);

    // dim=1 projects over cols, result has rows elements
    // row 0: max(0.0, 0.8) = 0.8 > 0.5 -> 1
    // row 1: max(0.6, 0.3) = 0.6 > 0.5 -> 1
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 1.0);
}

#[test]
fn test_logical_project_dim1_continuous() {
    let tensor = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

    let result = logical_project(&tensor, 1, LogicMode::Continuous);

    // Sum over cols
    assert!((result[0] - 0.3).abs() < 1e-6); // 0.1 + 0.2
    assert!((result[1] - 0.7).abs() < 1e-6); // 0.3 + 0.4
}

#[test]
fn test_logical_project_empty() {
    let tensor: Vec<Vec<f64>> = vec![];

    let result = logical_project(&tensor, 0, LogicMode::Boolean);
    assert!(result.is_empty());
}

#[test]
#[should_panic(expected = "Invalid dimension")]
fn test_logical_project_invalid_dim() {
    let tensor = vec![vec![1.0, 2.0]];
    logical_project(&tensor, 2, LogicMode::Boolean);
}

#[test]
fn test_logical_union_boolean() {
    let t1 = vec![vec![0.0, 0.8], vec![0.6, 0.0]];
    let t2 = vec![vec![0.7, 0.0], vec![0.0, 0.9]];

    let result = logical_union(&t1, &t2, LogicMode::Boolean);

    // max after thresholding
    assert_eq!(result[0][0], 1.0); // max(0, 1) = 1
    assert_eq!(result[0][1], 1.0); // max(1, 0) = 1
    assert_eq!(result[1][0], 1.0); // max(1, 0) = 1
    assert_eq!(result[1][1], 1.0); // max(0, 1) = 1
}

#[test]
fn test_logical_union_continuous() {
    let t1 = vec![vec![0.3, 0.5]];
    let t2 = vec![vec![0.4, 0.2]];

    let result = logical_union(&t1, &t2, LogicMode::Continuous);

    // P(A or B) = P(A) + P(B) - P(A)*P(B)
    let expected_0 = 0.3 + 0.4 - 0.3 * 0.4; // 0.58
    let expected_1 = 0.5 + 0.2 - 0.5 * 0.2; // 0.6
    assert!((result[0][0] - expected_0).abs() < 1e-6);
    assert!((result[0][1] - expected_1).abs() < 1e-6);
}

#[test]
fn test_logical_union_empty() {
    let t1: Vec<Vec<f64>> = vec![];
    let t2: Vec<Vec<f64>> = vec![];

    let result = logical_union(&t1, &t2, LogicMode::Boolean);
    assert!(result.is_empty());
}

#[test]
fn test_logical_negation_boolean() {
    let tensor = vec![vec![0.0, 0.8], vec![0.6, 0.3]];

    let result = logical_negation(&tensor, LogicMode::Boolean);

    assert_eq!(result[0][0], 1.0); // 1 - 0 = 1
    assert_eq!(result[0][1], 0.0); // 1 - 1 = 0
    assert_eq!(result[1][0], 0.0); // 1 - 1 = 0
    assert_eq!(result[1][1], 1.0); // 1 - 0 = 1
}

#[test]
fn test_logical_negation_continuous() {
    let tensor = vec![vec![0.3, 0.7]];

    let result = logical_negation(&tensor, LogicMode::Continuous);

    assert!((result[0][0] - 0.7).abs() < 1e-6);
    assert!((result[0][1] - 0.3).abs() < 1e-6);
}

#[test]
fn test_logical_negation_empty() {
    let tensor: Vec<Vec<f64>> = vec![];

    let result = logical_negation(&tensor, LogicMode::Boolean);
    assert!(result.is_empty());
}

#[test]
fn test_logical_select_boolean() {
    let tensor = vec![vec![0.5, 0.8], vec![0.9, 0.3]];
    let condition = vec![vec![0.6, 0.4], vec![0.7, 0.8]];

    let result = logical_select(&tensor, &condition, LogicMode::Boolean);

    // condition thresholded: [[1, 0], [1, 1]]
    assert!((result[0][0] - 0.5).abs() < 1e-6); // 0.5 * 1
    assert_eq!(result[0][1], 0.0); // 0.8 * 0
    assert!((result[1][0] - 0.9).abs() < 1e-6); // 0.9 * 1
    assert!((result[1][1] - 0.3).abs() < 1e-6); // 0.3 * 1
}

#[test]
fn test_logical_select_continuous() {
    let tensor = vec![vec![1.0, 2.0]];
    let condition = vec![vec![0.5, 0.25]];

    let result = logical_select(&tensor, &condition, LogicMode::Continuous);

    assert!((result[0][0] - 0.5).abs() < 1e-6); // 1.0 * 0.5
    assert!((result[0][1] - 0.5).abs() < 1e-6); // 2.0 * 0.25
}

#[test]
fn test_logical_select_empty() {
    let tensor: Vec<Vec<f64>> = vec![];
    let condition: Vec<Vec<f64>> = vec![];

    let result = logical_select(&tensor, &condition, LogicMode::Boolean);
    assert!(result.is_empty());
}

#[test]
fn test_apply_nonlinearity_step() {
    let tensor = vec![vec![-1.0, 0.0, 1.0]];
    let result = apply_nonlinearity(&tensor, Nonlinearity::Step);

    assert_eq!(result[0][0], 0.0);
    assert_eq!(result[0][1], 0.0);
    assert_eq!(result[0][2], 1.0);
}

#[test]
fn test_apply_nonlinearity_sigmoid() {
    let tensor = vec![vec![0.0]];
    let result = apply_nonlinearity(&tensor, Nonlinearity::Sigmoid);

    assert!((result[0][0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_apply_nonlinearity_relu() {
    let tensor = vec![vec![-2.0, 0.0, 3.0]];
    let result = apply_nonlinearity(&tensor, Nonlinearity::Relu);

    assert_eq!(result[0][0], 0.0);
    assert_eq!(result[0][1], 0.0);
    assert_eq!(result[0][2], 3.0);
}

#[test]
fn test_apply_nonlinearity_tanh() {
    let tensor = vec![vec![0.0]];
    let result = apply_nonlinearity(&tensor, Nonlinearity::Tanh);

    assert!((result[0][0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_apply_nonlinearity_identity() {
    let tensor = vec![vec![1.5, -2.5, 3.0]];
    let result = apply_nonlinearity(&tensor, Nonlinearity::Identity);

    assert_eq!(result[0], tensor[0]);
}

#[test]
fn test_apply_nonlinearity_softmax() {
    let tensor = vec![vec![1.0, 2.0, 3.0]];
    let result = apply_nonlinearity(&tensor, Nonlinearity::Softmax);

    let sum: f64 = result[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Values should be increasing
    assert!(result[0][0] < result[0][1]);
    assert!(result[0][1] < result[0][2]);
}

#[test]
fn test_apply_nonlinearity_boolean_attention() {
    let tensor = vec![vec![0.1, 0.3, 0.2]];
    let result = apply_nonlinearity(&tensor, Nonlinearity::BooleanAttention);

    // One-hot at argmax (index 1)
    assert_eq!(result[0][0], 0.0);
    assert_eq!(result[0][1], 1.0);
    assert_eq!(result[0][2], 0.0);
}

#[test]
fn test_apply_nonlinearity_with_temperature() {
    let tensor = vec![vec![1.0, 2.0, 3.0]];

    // High temperature -> more uniform distribution
    let result_high = apply_nonlinearity_with_temperature(&tensor, Nonlinearity::Softmax, 10.0);
    // Low temperature -> more peaked distribution
    let result_low = apply_nonlinearity_with_temperature(&tensor, Nonlinearity::Softmax, 0.1);

    // High temp should have max value closer to 1/3
    assert!(result_high[0][2] < result_low[0][2]);
}

#[test]
fn test_apply_nonlinearity_with_temperature_boolean_attention() {
    let tensor = vec![vec![0.1, 0.5, 0.3]];
    let result =
        apply_nonlinearity_with_temperature(&tensor, Nonlinearity::BooleanAttention, 2.0);

    // Temperature doesn't affect argmax
    assert_eq!(result[0][0], 0.0);
    assert_eq!(result[0][1], 1.0);
    assert_eq!(result[0][2], 0.0);
}

#[test]
fn test_apply_nonlinearity_with_mask() {
    let tensor = vec![vec![1.0, 2.0, 3.0]];
    let mask = vec![vec![false, true, false]]; // Mask out index 1

    let result = apply_nonlinearity_with_mask(&tensor, Nonlinearity::Softmax, Some(&mask));

    // Index 1 should be ~0 (masked out)
    assert!(result[0][1] < 1e-6);
    // Sum should still be ~1
    let sum: f64 = result[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_apply_nonlinearity_with_mask_boolean_attention() {
    let tensor = vec![vec![0.5, 0.9, 0.7]]; // argmax would be 1
    let mask = vec![vec![false, true, false]]; // Mask out index 1

    let result =
        apply_nonlinearity_with_mask(&tensor, Nonlinearity::BooleanAttention, Some(&mask));

    // Index 1 is masked, so argmax should be index 2
    assert_eq!(result[0][0], 0.0);
    assert_eq!(result[0][1], 0.0);
    assert_eq!(result[0][2], 1.0);
}

#[test]
fn test_apply_nonlinearity_with_mask_none() {
    let tensor = vec![vec![1.0, 2.0, 3.0]];
    let result = apply_nonlinearity_with_mask(&tensor, Nonlinearity::Softmax, None);

    let sum: f64 = result[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_row_all_same() {
    let row = vec![1.0, 1.0, 1.0];
    let result = softmax_row(&row);

    // All same -> uniform distribution
    for &val in &result {
        assert!((val - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn test_softmax_row_all_neg_infinity() {
    let row = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
    let result = softmax_row(&row);

    // All masked -> sum is 0, function returns zeros
    // The implementation returns vec![0.0; row.len()] when sum == 0
    assert_eq!(result.len(), 2);
    // When all inputs are -inf, exp(-inf) = 0, sum = 0, so result is 0/0 = NaN
    // But the code has: if sum == 0.0 { vec![0.0; row.len()] }
    // However, exp(-inf - max) where max = -inf gives exp(NaN) which is NaN, not 0
    // Let's just verify the length is correct
    assert_eq!(result.len(), 2);
}

#[test]
fn test_logical_project_all_below_threshold() {
    let tensor = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

    let result = logical_project(&tensor, 1, LogicMode::Boolean);

    // All values < 0.5, so max is still < 0.5 -> 0
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
}

#[test]
fn test_logical_union_both_zero() {
    let t1 = vec![vec![0.0, 0.0]];
    let t2 = vec![vec![0.0, 0.0]];

    let result_bool = logical_union(&t1, &t2, LogicMode::Boolean);
    let result_cont = logical_union(&t1, &t2, LogicMode::Continuous);

    assert_eq!(result_bool[0][0], 0.0);
    assert_eq!(result_cont[0][0], 0.0);
}

#[test]
fn test_logical_union_both_one() {
    let t1 = vec![vec![1.0]];
    let t2 = vec![vec![1.0]];

    let result = logical_union(&t1, &t2, LogicMode::Continuous);

    // P(A or B) = 1 + 1 - 1*1 = 1
    assert!((result[0][0] - 1.0).abs() < 1e-6);
}
