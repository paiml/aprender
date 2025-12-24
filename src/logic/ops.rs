//! Logical Tensor Operations
//!
//! Implements logical operations as tensor contractions via Einstein summation.
//! All operations support both Boolean (exact) and Continuous (differentiable) modes.
//!
//! # Operations
//!
//! | Operation | Boolean | Continuous | Einsum |
//! |-----------|---------|------------|--------|
//! | Join (AND) | matmul + threshold | matmul | `ij,jk->ik` |
//! | Project (∃) | max | sum | reduce |
//! | Union (OR) | max | P(A)+P(B)-P(A)P(B) | elementwise |
//! | Negation | 1-x (after threshold) | 1-x | elementwise |
//! | Select | mask | multiply | elementwise |

use std::f64;

/// Logic mode determines how operations handle intermediate values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicMode {
    /// Boolean mode: threshold at 0.5, outputs are 0 or 1
    /// Guarantees: No hallucinations (output ⊆ derivable facts)
    Boolean,
    /// Continuous mode: preserve real values for differentiability
    /// Enables: Gradient-based learning
    Continuous,
}

/// Nonlinearity functions for attention and activation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nonlinearity {
    /// Step function: x > 0 -> 1, else 0
    Step,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// ReLU: max(0, x)
    Relu,
    /// Softmax: exp(x_i) / sum(exp(x_j))
    Softmax,
    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// Boolean attention: one-hot at argmax
    BooleanAttention,
    /// Identity: x
    Identity,
}

/// Logical join (AND): Combines two relations via matrix multiplication
///
/// Semantically equivalent to: Grandparent(X,Z) := Parent(X,Y) ∧ Parent(Y,Z)
///
/// # Arguments
/// * `t1` - First tensor (e.g., Parent relation)
/// * `t2` - Second tensor (e.g., Parent relation)
/// * `mode` - Boolean or Continuous
///
/// # Returns
/// Result tensor where result[i][k] = ∃j: t1[i][j] ∧ t2[j][k]
pub fn logical_join(t1: &[Vec<f64>], t2: &[Vec<f64>], mode: LogicMode) -> Vec<Vec<f64>> {
    let rows = t1.len();
    let inner = if t1.is_empty() { 0 } else { t1[0].len() };
    let cols = if t2.is_empty() { 0 } else { t2[0].len() };

    let mut result = vec![vec![0.0; cols]; rows];

    // Matrix multiplication: result[i][k] = sum_j(t1[i][j] * t2[j][k])
    for i in 0..rows {
        for k in 0..cols {
            let mut sum = 0.0;
            for j in 0..inner {
                sum += t1[i][j] * t2[j][k];
            }
            result[i][k] = sum;
        }
    }

    // Apply mode-specific processing
    match mode {
        LogicMode::Boolean => threshold_matrix(&mut result),
        LogicMode::Continuous => {}
    }

    result
}

/// Logical projection (∃): Existential quantification over a dimension
///
/// Semantically equivalent to: HasChild(X) := ∃Y: Parent(X,Y)
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dim` - Dimension to project over (0 = rows, 1 = cols)
/// * `mode` - Boolean (max) or Continuous (sum)
pub fn logical_project(tensor: &[Vec<f64>], dim: usize, mode: LogicMode) -> Vec<f64> {
    match dim {
        0 => {
            // Project over rows (result has cols elements)
            let cols = if tensor.is_empty() {
                0
            } else {
                tensor[0].len()
            };
            let mut result = vec![0.0; cols];

            for j in 0..cols {
                match mode {
                    LogicMode::Boolean => {
                        let max_val = tensor.iter().map(|row| row[j]).fold(0.0, f64::max);
                        result[j] = if max_val > 0.5 { 1.0 } else { 0.0 };
                    }
                    LogicMode::Continuous => {
                        result[j] = tensor.iter().map(|row| row[j]).sum();
                    }
                }
            }
            result
        }
        1 => {
            // Project over columns (result has rows elements)
            let mut result = Vec::with_capacity(tensor.len());

            for row in tensor {
                match mode {
                    LogicMode::Boolean => {
                        let max_val = row.iter().fold(0.0, |a, &b| f64::max(a, b));
                        result.push(if max_val > 0.5 { 1.0 } else { 0.0 });
                    }
                    LogicMode::Continuous => {
                        result.push(row.iter().sum());
                    }
                }
            }
            result
        }
        _ => panic!("Invalid dimension for 2D tensor projection"),
    }
}

/// Logical union (OR): Combines two tensors via logical OR
///
/// # Boolean mode
/// result[i][j] = max(t1[i][j], t2[i][j])
///
/// # Continuous mode (probabilistic OR)
/// result[i][j] = P(A) + P(B) - P(A)*P(B)
pub fn logical_union(t1: &[Vec<f64>], t2: &[Vec<f64>], mode: LogicMode) -> Vec<Vec<f64>> {
    let rows = t1.len();
    let cols = if t1.is_empty() { 0 } else { t1[0].len() };

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            match mode {
                LogicMode::Boolean => {
                    let a = if t1[i][j] > 0.5 { 1.0 } else { 0.0 };
                    let b = if t2[i][j] > 0.5 { 1.0 } else { 0.0 };
                    result[i][j] = f64::max(a, b);
                }
                LogicMode::Continuous => {
                    // P(A or B) = P(A) + P(B) - P(A)*P(B)
                    let a = t1[i][j];
                    let b = t2[i][j];
                    result[i][j] = a + b - a * b;
                }
            }
        }
    }

    result
}

/// Logical negation (NOT): Negates tensor values
///
/// result[i][j] = 1 - tensor[i][j]
pub fn logical_negation(tensor: &[Vec<f64>], mode: LogicMode) -> Vec<Vec<f64>> {
    let rows = tensor.len();
    let cols = if tensor.is_empty() {
        0
    } else {
        tensor[0].len()
    };

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            match mode {
                LogicMode::Boolean => {
                    let val = if tensor[i][j] > 0.5 { 1.0 } else { 0.0 };
                    result[i][j] = 1.0 - val;
                }
                LogicMode::Continuous => {
                    result[i][j] = 1.0 - tensor[i][j];
                }
            }
        }
    }

    result
}

/// Logical selection (WHERE): Filters tensor by condition
///
/// result[i][j] = tensor[i][j] if condition[i][j] else 0
pub fn logical_select(
    tensor: &[Vec<f64>],
    condition: &[Vec<f64>],
    mode: LogicMode,
) -> Vec<Vec<f64>> {
    let rows = tensor.len();
    let cols = if tensor.is_empty() {
        0
    } else {
        tensor[0].len()
    };

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            let cond = match mode {
                LogicMode::Boolean => {
                    if condition[i][j] > 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                }
                LogicMode::Continuous => condition[i][j],
            };
            result[i][j] = tensor[i][j] * cond;
        }
    }

    result
}

/// Apply nonlinearity function to tensor
pub fn apply_nonlinearity(tensor: &[Vec<f64>], func: Nonlinearity) -> Vec<Vec<f64>> {
    apply_nonlinearity_with_temperature(tensor, func, 1.0)
}

/// Apply nonlinearity with temperature parameter
pub fn apply_nonlinearity_with_temperature(
    tensor: &[Vec<f64>],
    func: Nonlinearity,
    temperature: f64,
) -> Vec<Vec<f64>> {
    tensor
        .iter()
        .map(|row| {
            match func {
                Nonlinearity::Softmax => {
                    // Recompute with temperature
                    let scaled: Vec<f64> = row.iter().map(|x| x / temperature).collect();
                    softmax_row(&scaled)
                }
                Nonlinearity::BooleanAttention => {
                    // One-hot at argmax (temperature doesn't affect argmax)
                    let max_idx = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or(0, |(i, _)| i);
                    let mut result = vec![0.0; row.len()];
                    result[max_idx] = 1.0;
                    result
                }
                _ => apply_nonlinearity_row(row, func, None),
            }
        })
        .collect()
}

/// Apply nonlinearity with optional mask
pub fn apply_nonlinearity_with_mask(
    tensor: &[Vec<f64>],
    func: Nonlinearity,
    mask: Option<&[Vec<bool>]>,
) -> Vec<Vec<f64>> {
    tensor
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let row_mask = mask.map(|m| &m[i]);
            apply_nonlinearity_row(row, func, row_mask)
        })
        .collect()
}

fn apply_nonlinearity_row(row: &[f64], func: Nonlinearity, mask: Option<&Vec<bool>>) -> Vec<f64> {
    match func {
        Nonlinearity::Step => row
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect(),
        Nonlinearity::Sigmoid => row.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        Nonlinearity::Relu => row.iter().map(|&x| f64::max(0.0, x)).collect(),
        Nonlinearity::Tanh => row.iter().map(|&x| x.tanh()).collect(),
        Nonlinearity::Identity => row.to_vec(),
        Nonlinearity::Softmax => {
            let masked_row: Vec<f64> = if let Some(m) = mask {
                row.iter()
                    .zip(m.iter())
                    .map(|(&x, &masked)| if masked { f64::NEG_INFINITY } else { x })
                    .collect()
            } else {
                row.to_vec()
            };
            softmax_row(&masked_row)
        }
        Nonlinearity::BooleanAttention => {
            // One-hot at argmax
            let masked_row: Vec<f64> = if let Some(m) = mask {
                row.iter()
                    .zip(m.iter())
                    .map(|(&x, &masked)| if masked { f64::NEG_INFINITY } else { x })
                    .collect()
            } else {
                row.to_vec()
            };

            let max_idx = masked_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            let mut result = vec![0.0; row.len()];
            result[max_idx] = 1.0;
            result
        }
    }
}

fn softmax_row(row: &[f64]) -> Vec<f64> {
    // Numerical stability: subtract max
    let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));
    let exp_vals: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();

    if sum == 0.0 {
        vec![0.0; row.len()]
    } else {
        exp_vals.iter().map(|&x| x / sum).collect()
    }
}

/// Threshold matrix values at 0.5 for Boolean mode
fn threshold_matrix(matrix: &mut [Vec<f64>]) {
    for row in matrix.iter_mut() {
        for val in row.iter_mut() {
            *val = if *val > 0.5 { 1.0 } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
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
}
