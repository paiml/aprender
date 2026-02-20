
/// Generate an N:M structured sparsity mask.
///
/// # Arguments
/// * `scores` - Importance scores tensor
/// * `n` - Number of non-zeros per group
/// * `m` - Group size
///
/// # Returns
/// Mask with exactly N non-zeros per M consecutive elements.
///
/// # References
/// Zhou, A., et al. (2021). Learning N:M fine-grained structured sparse neural networks. ICLR.
pub fn generate_nm_mask(scores: &Tensor, n: usize, m: usize) -> Result<SparsityMask, PruningError> {
    if n > m {
        return Err(PruningError::InvalidPattern {
            message: format!("N ({n}) must be <= M ({m})"),
        });
    }
    if m == 0 {
        return Err(PruningError::InvalidPattern {
            message: "M must be > 0".to_string(),
        });
    }

    let data = scores.data();
    if data.len() % m != 0 {
        return Err(PruningError::InvalidPattern {
            message: format!("Tensor length {} not divisible by M={}", data.len(), m),
        });
    }

    let mut mask_data = vec![0.0f32; data.len()];

    // Process each group of M elements
    for (group_idx, chunk) in data.chunks(m).enumerate() {
        // Find indices of top N elements in this group
        let mut indexed: Vec<(usize, f32)> =
            chunk.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Set top N to 1
        for (local_idx, _) in indexed.into_iter().take(n) {
            mask_data[group_idx * m + local_idx] = 1.0;
        }
    }

    SparsityMask::new(
        Tensor::new(&mask_data, scores.shape()),
        SparsityPattern::NM { n, m },
    )
}

/// Generate a block sparsity mask.
///
/// # Arguments
/// * `scores` - 2D importance scores tensor
/// * `block_height` - Height of each block
/// * `block_width` - Width of each block
/// * `target_sparsity` - Fraction of blocks to prune
///
/// # Returns
/// Mask where entire blocks are either kept (1s) or pruned (0s).
///
/// # References
/// Block sparsity enables efficient inference on standard hardware.
pub fn generate_block_mask(
    scores: &Tensor,
    block_height: usize,
    block_width: usize,
    target_sparsity: f32,
) -> Result<SparsityMask, PruningError> {
    let shape = scores.shape();
    let (rows, cols) =
        validate_block_mask_inputs(shape, block_height, block_width, target_sparsity)?;

    let data = scores.data();
    let block_scores = compute_sorted_block_scores(data, rows, cols, block_height, block_width);

    let num_prune = (block_scores.len() as f32 * target_sparsity) as usize;
    let mut mask_data = vec![1.0f32; rows * cols];

    zero_out_blocks(
        &mut mask_data,
        &block_scores[..num_prune],
        cols,
        block_height,
        block_width,
    );

    SparsityMask::new(
        Tensor::new(&mask_data, shape),
        SparsityPattern::Block {
            height: block_height,
            width: block_width,
        },
    )
}

/// Validate inputs for block mask generation. Returns (rows, cols).
fn validate_block_mask_inputs(
    shape: &[usize],
    block_height: usize,
    block_width: usize,
    target_sparsity: f32,
) -> Result<(usize, usize), PruningError> {
    if shape.len() != 2 {
        return Err(PruningError::ShapeMismatch {
            expected: vec![0, 0],
            got: shape.to_vec(),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    if rows % block_height != 0 || cols % block_width != 0 {
        return Err(PruningError::InvalidPattern {
            message: format!(
                "Shape [{rows}, {cols}] not divisible by block size [{block_height}, {block_width}]"
            ),
        });
    }
    if !(0.0..=1.0).contains(&target_sparsity) {
        return Err(PruningError::InvalidSparsity {
            value: target_sparsity,
            constraint: "must be between 0.0 and 1.0".to_string(),
        });
    }
    Ok((rows, cols))
}

/// Compute block importance scores, sorted ascending by sum.
fn compute_sorted_block_scores(
    data: &[f32],
    rows: usize,
    cols: usize,
    block_height: usize,
    block_width: usize,
) -> Vec<(usize, usize, f32)> {
    let num_block_rows = rows / block_height;
    let num_block_cols = cols / block_width;
    let mut block_scores: Vec<(usize, usize, f32)> =
        Vec::with_capacity(num_block_rows * num_block_cols);
    for br in 0..num_block_rows {
        for bc in 0..num_block_cols {
            let mut sum = 0.0f32;
            for r in 0..block_height {
                for c in 0..block_width {
                    sum += data[(br * block_height + r) * cols + bc * block_width + c];
                }
            }
            block_scores.push((br, bc, sum));
        }
    }
    block_scores.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    block_scores
}

/// Zero out blocks in the mask for the given block positions.
fn zero_out_blocks(
    mask_data: &mut [f32],
    blocks_to_prune: &[(usize, usize, f32)],
    cols: usize,
    block_height: usize,
    block_width: usize,
) {
    for &(br, bc, _) in blocks_to_prune {
        for r in 0..block_height {
            for c in 0..block_width {
                mask_data[(br * block_height + r) * cols + bc * block_width + c] = 0.0;
            }
        }
    }
}

/// Generate a row sparsity mask.
///
/// # Arguments
/// * `scores` - 2D importance scores tensor
/// * `target_sparsity` - Fraction of rows to prune
///
/// # Returns
/// Mask where entire rows are either kept (1s) or pruned (0s).
///
/// Row sparsity is equivalent to pruning output neurons.
pub fn generate_row_mask(
    scores: &Tensor,
    target_sparsity: f32,
) -> Result<SparsityMask, PruningError> {
    let shape = scores.shape();
    if shape.len() != 2 {
        return Err(PruningError::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: shape.to_vec(),
        });
    }

    let rows = shape[0];
    let cols = shape[1];

    if !(0.0..=1.0).contains(&target_sparsity) {
        return Err(PruningError::InvalidSparsity {
            value: target_sparsity,
            constraint: "must be between 0.0 and 1.0".to_string(),
        });
    }

    let num_prune = (rows as f32 * target_sparsity) as usize;
    let data = scores.data();

    // Compute row importance (sum of element importance in row)
    let mut row_scores: Vec<(usize, f32)> = Vec::with_capacity(rows);
    for r in 0..rows {
        let sum: f32 = (0..cols).map(|c| data[r * cols + c]).sum();
        row_scores.push((r, sum));
    }

    // Sort by importance (ascending - lowest first)
    row_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Create mask (all ones initially)
    let mut mask_data = vec![1.0f32; rows * cols];

    // Zero out the lowest-importance rows
    for &(row, _) in row_scores.iter().take(num_prune) {
        for c in 0..cols {
            mask_data[row * cols + c] = 0.0;
        }
    }

    SparsityMask::new(Tensor::new(&mask_data, shape), SparsityPattern::Row)
}

/// Generate a column sparsity mask.
///
/// # Arguments
/// * `scores` - 2D importance scores tensor
/// * `target_sparsity` - Fraction of columns to prune
///
/// # Returns
/// Mask where entire columns are either kept (1s) or pruned (0s).
///
/// Column sparsity is equivalent to pruning input features.
pub fn generate_column_mask(
    scores: &Tensor,
    target_sparsity: f32,
) -> Result<SparsityMask, PruningError> {
    let shape = scores.shape();
    if shape.len() != 2 {
        return Err(PruningError::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: shape.to_vec(),
        });
    }

    let rows = shape[0];
    let cols = shape[1];

    if !(0.0..=1.0).contains(&target_sparsity) {
        return Err(PruningError::InvalidSparsity {
            value: target_sparsity,
            constraint: "must be between 0.0 and 1.0".to_string(),
        });
    }

    let num_prune = (cols as f32 * target_sparsity) as usize;
    let data = scores.data();

    // Compute column importance (sum of element importance in column)
    let mut col_scores: Vec<(usize, f32)> = Vec::with_capacity(cols);
    for c in 0..cols {
        let sum: f32 = (0..rows).map(|r| data[r * cols + c]).sum();
        col_scores.push((c, sum));
    }

    // Sort by importance (ascending - lowest first)
    col_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Create mask (all ones initially)
    let mut mask_data = vec![1.0f32; rows * cols];

    // Zero out the lowest-importance columns
    for &(col, _) in col_scores.iter().take(num_prune) {
        for r in 0..rows {
            mask_data[r * cols + col] = 0.0;
        }
    }

    SparsityMask::new(Tensor::new(&mask_data, shape), SparsityPattern::Column)
}

#[cfg(test)]
mod tests;
