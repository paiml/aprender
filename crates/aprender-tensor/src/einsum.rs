//! Einstein summation via TTGT (Transpose-Transpose-GEMM-Transpose).
//!
//! Parses subscript notation like `"ijk,jkl->il"` and reduces tensor
//! contractions to matrix multiplication.

use crate::error::TensorError;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Parsed einsum subscript describing one contraction.
#[derive(Debug)]
struct EinsumPlan {
    /// Index labels for each input tensor.
    input_labels: Vec<Vec<char>>,
    /// Index labels for the output tensor.
    output_labels: Vec<char>,
}

/// Parse an einsum subscript string like `"ijk,jkl->il"`.
fn parse_subscripts(subscripts: &str) -> Result<EinsumPlan, TensorError> {
    let parts: Vec<&str> = subscripts.split("->").collect();
    if parts.len() != 2 {
        return Err(TensorError::InvalidSubscript("expected exactly one '->' separator".into()));
    }

    let input_part = parts[0];
    let output_part = parts[1];

    let input_labels: Vec<Vec<char>> = input_part.split(',').map(|s| s.chars().collect()).collect();

    if input_labels.is_empty() {
        return Err(TensorError::InvalidSubscript("no input tensors specified".into()));
    }

    for labels in &input_labels {
        for &c in labels {
            if !c.is_ascii_lowercase() {
                return Err(TensorError::InvalidSubscript(format!(
                    "index label must be lowercase ascii, got '{c}'"
                )));
            }
        }
    }

    let output_labels: Vec<char> = output_part.chars().collect();
    for &c in &output_labels {
        if !c.is_ascii_lowercase() {
            return Err(TensorError::InvalidSubscript(format!(
                "output label must be lowercase ascii, got '{c}'"
            )));
        }
    }

    Ok(EinsumPlan { input_labels, output_labels })
}

/// Build a map from index label to dimension size, validating consistency.
fn build_index_sizes(
    plan: &EinsumPlan,
    inputs: &[&Tensor],
) -> Result<HashMap<char, usize>, TensorError> {
    let mut sizes: HashMap<char, usize> = HashMap::new();

    for (t, labels) in plan.input_labels.iter().enumerate() {
        let shape = inputs[t].shape();
        if labels.len() != shape.len() {
            return Err(TensorError::InvalidSubscript(format!(
                "input {} has {} labels but shape has {} dims",
                t,
                labels.len(),
                shape.len()
            )));
        }
        for (&label, &dim) in labels.iter().zip(shape.iter()) {
            if let Some(&existing) = sizes.get(&label) {
                if existing != dim {
                    return Err(TensorError::ContractionDimensionMismatch {
                        index: label,
                        size_a: existing,
                        size_b: dim,
                    });
                }
            } else {
                sizes.insert(label, dim);
            }
        }
    }

    Ok(sizes)
}

/// Perform Einstein summation on two input tensors.
///
/// Supports subscript notation like `"ijk,jkl->il"`.
/// Reduces tensor contractions to explicit loops (TTGT strategy).
///
/// # Errors
///
/// Returns error if subscripts are invalid or dimensions don't match.
pub fn einsum(subscripts: &str, a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    einsum_binary(subscripts, a, b)
}

/// Perform Einstein summation on an arbitrary number of input tensors.
///
/// Supports subscript notation like `"ij,jk,kl->il"` with pairwise reduction.
/// For 2 inputs, delegates directly. For N>2 inputs, reduces left-to-right.
///
/// # Errors
///
/// Returns error if subscripts are invalid or dimensions don't match.
pub fn einsum_nary(subscripts: &str, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    let plan = parse_subscripts(subscripts)?;

    if plan.input_labels.len() != inputs.len() {
        return Err(TensorError::InvalidSubscript(format!(
            "subscript has {} inputs but {} tensors provided",
            plan.input_labels.len(),
            inputs.len()
        )));
    }

    if inputs.is_empty() {
        return Err(TensorError::InvalidSubscript("no input tensors specified".into()));
    }

    if inputs.len() == 1 {
        // Single tensor: just select/sum as needed
        return einsum_single(&plan, inputs[0]);
    }

    if inputs.len() == 2 {
        return einsum_binary(subscripts, inputs[0], inputs[1]);
    }

    // N-ary: pairwise left-to-right reduction
    // First contraction: inputs[0] and inputs[1]
    let mut result = reduce_pair(
        inputs[0],
        inputs[1],
        &plan.input_labels[0],
        &plan.input_labels[1],
        &plan.output_labels,
        &plan.input_labels[2..],
    )?;

    // Track which labels the current result carries
    let mut result_labels = intermediate_labels(
        &plan.input_labels[0],
        &plan.input_labels[1],
        &plan.output_labels,
        &plan.input_labels[2..],
    );

    // Contract remaining inputs one at a time
    for i in 2..inputs.len() {
        let next = reduce_pair(
            &result,
            inputs[i],
            &result_labels,
            &plan.input_labels[i],
            &plan.output_labels,
            &plan.input_labels[i + 1..],
        )?;
        result_labels = intermediate_labels(
            &result_labels,
            &plan.input_labels[i],
            &plan.output_labels,
            &plan.input_labels[i + 1..],
        );
        result = next;
    }

    Ok(result)
}

/// Compute intermediate labels for a pair contraction.
/// Keep all labels that appear in the final output or in remaining inputs.
fn intermediate_labels(
    a_labels: &[char],
    b_labels: &[char],
    output_labels: &[char],
    remaining: &[Vec<char>],
) -> Vec<char> {
    let mut labels = Vec::new();
    let mut seen = Vec::new();

    for &l in a_labels.iter().chain(b_labels.iter()) {
        if seen.contains(&l) {
            continue;
        }
        seen.push(l);
        let needed_later = remaining.iter().any(|r| r.contains(&l)) || output_labels.contains(&l);
        if needed_later {
            labels.push(l);
        }
    }
    labels
}

/// Contract two tensors, keeping indices needed for later contractions.
fn reduce_pair(
    a: &Tensor,
    b: &Tensor,
    a_labels: &[char],
    b_labels: &[char],
    final_output: &[char],
    remaining: &[Vec<char>],
) -> Result<Tensor, TensorError> {
    let out_labels = intermediate_labels(a_labels, b_labels, final_output, remaining);

    // Build subscript string for this pair
    let a_str: String = a_labels.iter().collect();
    let b_str: String = b_labels.iter().collect();
    let o_str: String = out_labels.iter().collect();
    let sub = format!("{a_str},{b_str}->{o_str}");

    einsum_binary(&sub, a, b)
}

/// Single-tensor einsum (trace, transpose, etc.)
fn einsum_single(plan: &EinsumPlan, a: &Tensor) -> Result<Tensor, TensorError> {
    let inputs = [a];
    let index_sizes = build_index_sizes(plan, &inputs)?;
    let a_labels = &plan.input_labels[0];
    let out_labels = &plan.output_labels;

    let out_shape: Vec<usize> = out_labels.iter().map(|l| index_sizes[l]).collect();
    let mut output = Tensor::zeros(out_shape);

    // Contracted indices: in input but not in output
    let contracted: Vec<char> =
        a_labels.iter().filter(|l| !out_labels.contains(l)).copied().collect();

    let mut all_labels: Vec<char> = out_labels.clone();
    all_labels.extend_from_slice(&contracted);
    let all_sizes: Vec<usize> = all_labels.iter().map(|l| index_sizes[l]).collect();
    let total: usize = all_sizes.iter().product();

    if total == 0 {
        return Ok(output);
    }

    let ndim = all_labels.len();
    let mut indices = vec![0usize; ndim];

    for _ in 0..total {
        let label_vals: HashMap<char, usize> =
            all_labels.iter().zip(indices.iter()).map(|(&l, &v)| (l, v)).collect();

        let a_idx: Vec<usize> = a_labels.iter().map(|l| label_vals[l]).collect();
        let out_idx: Vec<usize> = out_labels.iter().map(|l| label_vals[l]).collect();

        let val = a.get(&a_idx);
        let cur = output.get(&out_idx);
        output.set(&out_idx, cur + val);

        increment_indices(&mut indices, &all_sizes);
    }

    Ok(output)
}

/// Increment a multi-index (odometer style) given dimension sizes.
fn increment_indices(indices: &mut [usize], sizes: &[usize]) {
    let mut d = indices.len();
    loop {
        if d == 0 {
            break;
        }
        d -= 1;
        indices[d] += 1;
        if indices[d] < sizes[d] {
            break;
        }
        indices[d] = 0;
    }
}

/// Binary einsum (the core 2-input implementation).
fn einsum_binary(subscripts: &str, a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let plan = parse_subscripts(subscripts)?;

    if plan.input_labels.len() != 2 {
        return Err(TensorError::InvalidSubscript(
            "binary einsum requires exactly 2 input operands in subscript".into(),
        ));
    }

    let inputs = [a, b];
    let index_sizes = build_index_sizes(&plan, &inputs)?;

    let a_labels = &plan.input_labels[0];
    let b_labels = &plan.input_labels[1];
    let out_labels = &plan.output_labels;

    // Identify contracted indices (in inputs but not in output)
    let contracted: Vec<char> = {
        let mut c = Vec::new();
        for &label in a_labels {
            if !out_labels.contains(&label) && !c.contains(&label) {
                c.push(label);
            }
        }
        for &label in b_labels {
            if !out_labels.contains(&label) && !c.contains(&label) {
                c.push(label);
            }
        }
        c
    };

    // Validate contracted indices appear in both tensors
    for &c in &contracted {
        let in_a = a_labels.contains(&c);
        let in_b = b_labels.contains(&c);
        if !in_a || !in_b {
            return Err(TensorError::InvalidSubscript(format!(
                "contracted index '{c}' must appear in both inputs"
            )));
        }
    }

    // Compute output shape
    let out_shape: Vec<usize> = out_labels.iter().map(|l| index_sizes[l]).collect();

    let mut output = Tensor::zeros(out_shape);

    // Compute contraction via explicit nested iteration
    contract_tensors(a, b, &mut output, a_labels, b_labels, out_labels, &contracted, &index_sizes);

    Ok(output)
}

/// Core contraction loop: iterate over all output and contracted indices.
#[allow(clippy::too_many_arguments)]
fn contract_tensors(
    a: &Tensor,
    b: &Tensor,
    output: &mut Tensor,
    a_labels: &[char],
    b_labels: &[char],
    out_labels: &[char],
    contracted: &[char],
    index_sizes: &HashMap<char, usize>,
) {
    // All unique indices = output indices + contracted indices
    let mut all_labels: Vec<char> = out_labels.to_vec();
    all_labels.extend_from_slice(contracted);

    let all_sizes: Vec<usize> = all_labels.iter().map(|l| index_sizes[l]).collect();

    let total: usize = all_sizes.iter().product();
    if total == 0 {
        return;
    }

    let ndim = all_labels.len();
    let mut indices = vec![0usize; ndim];

    for _ in 0..total {
        // Build label->value map
        let label_vals: HashMap<char, usize> =
            all_labels.iter().zip(indices.iter()).map(|(&l, &v)| (l, v)).collect();

        // Index into A
        let a_idx: Vec<usize> = a_labels.iter().map(|l| label_vals[l]).collect();
        // Index into B
        let b_idx: Vec<usize> = b_labels.iter().map(|l| label_vals[l]).collect();
        // Index into output
        let out_idx: Vec<usize> = out_labels.iter().map(|l| label_vals[l]).collect();

        let val = a.get(&a_idx) * b.get(&b_idx);
        let cur = output.get(&out_idx);
        output.set(&out_idx, cur + val);

        increment_indices(&mut indices, &all_sizes);
    }
}

/// Matrix multiply: `"ij,jk->ik"` convenience wrapper.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    einsum("ij,jk->ik", a, b)
}

/// Batch matrix multiply: `"bij,bjk->bik"` convenience wrapper.
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    einsum("bij,bjk->bik", a, b)
}

/// Outer product: `"i,j->ij"` convenience wrapper.
pub fn outer(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    einsum("i,j->ij", a, b)
}

/// Trace via einsum: `"ii->"` but implemented as sum of diagonal.
pub fn trace(a: &Tensor) -> Result<f32, TensorError> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(TensorError::InvalidSubscript("trace requires a square 2D tensor".into()));
    }
    let n = a.shape()[0];
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a.get(&[i, i]);
    }
    Ok(sum)
}
