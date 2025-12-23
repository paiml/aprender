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
}
