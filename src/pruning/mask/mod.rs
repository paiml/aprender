//! Sparsity mask representation and pattern validation.
//!
//! # Toyota Way: Poka-Yoke
//! Masks validate shape compatibility and binary values at construction,
//! preventing invalid operations downstream.
//!
//! # References
//! - Zhou, A., et al. (2021). Learning N:M fine-grained structured sparse networks. ICLR.
//! - Mishra, A., et al. (2021). Accelerating sparse deep neural networks.

use super::error::PruningError;
use crate::autograd::Tensor;

/// Sparsity pattern constraints.
///
/// Defines the structural constraints on which weights can be pruned.
/// Different patterns offer different trade-offs between flexibility
/// and hardware acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsityPattern {
    /// No structural constraint - any element can be pruned.
    ///
    /// Maximum flexibility but requires sparse hardware for speedup.
    Unstructured,

    /// N:M sparsity - in every M consecutive elements, exactly N are non-zero.
    ///
    /// # Hardware Support
    /// - 2:4 sparsity: NVIDIA Ampere (A100, RTX 30xx) - 2x speedup
    /// - 4:8 sparsity: Future hardware
    NM {
        /// Number of non-zero elements per group
        n: usize,
        /// Group size
        m: usize,
    },

    /// Block sparsity - entire blocks of size (height, width) are pruned together.
    ///
    /// Useful for structured pruning where entire neurons or filters are removed.
    Block {
        /// Block height
        height: usize,
        /// Block width
        width: usize,
    },

    /// Row sparsity - entire rows (output channels) pruned.
    ///
    /// Equivalent to pruning entire output neurons.
    Row,

    /// Column sparsity - entire columns (input channels) pruned.
    ///
    /// Equivalent to removing input features.
    Column,
}

impl SparsityPattern {
    /// Check if this pattern configuration is valid.
    ///
    /// # Returns
    /// `true` if the pattern parameters are valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        match self {
            SparsityPattern::NM { n, m } => *n <= *m && *m > 0,
            SparsityPattern::Block { height, width } => *height > 0 && *width > 0,
            _ => true,
        }
    }

    /// Get the theoretical sparsity for this pattern.
    ///
    /// # Returns
    /// Sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    #[must_use]
    pub fn theoretical_sparsity(&self) -> Option<f32> {
        match self {
            SparsityPattern::NM { n, m } => Some(1.0 - (*n as f32 / *m as f32)),
            _ => None, // Variable sparsity for other patterns
        }
    }

    /// Validate a mask tensor against this pattern's constraints.
    ///
    /// # Arguments
    /// * `mask` - Binary mask tensor to validate
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(PruningError::InvalidPattern)` if not.
    pub fn validate(&self, mask: &Tensor) -> Result<(), PruningError> {
        match self {
            SparsityPattern::Unstructured => Ok(()),

            SparsityPattern::NM { n, m } => {
                let data = mask.data();
                if data.len() % m != 0 {
                    return Err(PruningError::InvalidPattern {
                        message: format!("Mask length {} not divisible by M={}", data.len(), m),
                    });
                }

                // Check each group of M elements
                for (i, chunk) in data.chunks(*m).enumerate() {
                    let count: usize = chunk.iter().filter(|&&v| (v - 1.0).abs() < 1e-6).count();
                    if count != *n {
                        return Err(PruningError::InvalidPattern {
                            message: format!(
                                "Group {i} has {count} non-zeros, expected {n} for {n}:{m} pattern"
                            ),
                        });
                    }
                }
                Ok(())
            }

            SparsityPattern::Block { height, width } => {
                let shape = mask.shape();
                if shape.len() < 2 {
                    return Err(PruningError::InvalidPattern {
                        message: "Block sparsity requires 2D mask".to_string(),
                    });
                }
                if shape[0] % height != 0 || shape[1] % width != 0 {
                    return Err(PruningError::InvalidPattern {
                        message: format!(
                            "Mask shape {shape:?} not divisible by block size ({height}, {width})"
                        ),
                    });
                }

                // Check each block is uniform (all 0s or all 1s)
                let data = mask.data();
                let rows = shape[0];
                let cols = shape[1];

                for br in 0..(rows / height) {
                    for bc in 0..(cols / width) {
                        let first_val = data[br * height * cols + bc * width];
                        for r in 0..*height {
                            for c in 0..*width {
                                let idx = (br * height + r) * cols + (bc * width + c);
                                let val = data[idx];
                                if (val - first_val).abs() > 1e-6 {
                                    return Err(PruningError::InvalidPattern {
                                        message: format!("Block ({br}, {bc}) is not uniform"),
                                    });
                                }
                            }
                        }
                    }
                }
                Ok(())
            }

            SparsityPattern::Row => {
                let shape = mask.shape();
                if shape.len() < 2 {
                    return Err(PruningError::InvalidPattern {
                        message: "Row sparsity requires 2D mask".to_string(),
                    });
                }

                let data = mask.data();
                let cols = shape[1];

                // Check each row is uniform
                for (row_idx, chunk) in data.chunks(cols).enumerate() {
                    let first = chunk[0];
                    if !chunk.iter().all(|&v| (v - first).abs() < 1e-6) {
                        return Err(PruningError::InvalidPattern {
                            message: format!("Row {row_idx} is not uniform"),
                        });
                    }
                }
                Ok(())
            }

            SparsityPattern::Column => {
                let shape = mask.shape();
                if shape.len() < 2 {
                    return Err(PruningError::InvalidPattern {
                        message: "Column sparsity requires 2D mask".to_string(),
                    });
                }

                let data = mask.data();
                let rows = shape[0];
                let cols = shape[1];

                // Check each column is uniform
                for col in 0..cols {
                    let first = data[col];
                    for row in 1..rows {
                        let val = data[row * cols + col];
                        if (val - first).abs() > 1e-6 {
                            return Err(PruningError::InvalidPattern {
                                message: format!("Column {col} is not uniform"),
                            });
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

impl Default for SparsityPattern {
    fn default() -> Self {
        SparsityPattern::Unstructured
    }
}

/// Sparsity mask with validation.
///
/// # Toyota Way: Poka-Yoke
/// The mask validates binary values and pattern constraints at construction,
/// preventing invalid masks from being created.
///
/// # Invariants
/// - All values are exactly 0.0 or 1.0
/// - Pattern constraints are satisfied
/// - Sparsity is precomputed and cached
#[derive(Debug, Clone)]
pub struct SparsityMask {
    /// Binary mask tensor (1 = keep, 0 = prune)
    mask: Tensor,
    /// Pattern used to generate this mask
    pattern: SparsityPattern,
    /// Cached sparsity ratio
    sparsity: f32,
}

impl SparsityMask {
    /// Create a new mask with validation.
    ///
    /// # Arguments
    /// * `mask` - Binary tensor with values in {0.0, 1.0}
    /// * `pattern` - Sparsity pattern constraint
    ///
    /// # Returns
    /// * `Ok(SparsityMask)` - Valid mask
    /// * `Err(PruningError::InvalidMask)` - If values are not binary
    /// * `Err(PruningError::InvalidPattern)` - If pattern constraints violated
    pub fn new(mask: Tensor, pattern: SparsityPattern) -> Result<Self, PruningError> {
        // Validate binary values
        for &v in mask.data() {
            if (v - 0.0).abs() > 1e-6 && (v - 1.0).abs() > 1e-6 {
                return Err(PruningError::InvalidMask {
                    reason: format!("Mask contains non-binary value: {v}"),
                });
            }
        }

        // Validate pattern constraints
        pattern.validate(&mask)?;

        // Compute sparsity (fraction of zeros)
        let data = mask.data();
        let sparsity = if data.is_empty() {
            0.0
        } else {
            let zeros = data.iter().filter(|&&v| v < 0.5).count();
            zeros as f32 / data.len() as f32
        };

        Ok(Self {
            mask,
            pattern,
            sparsity,
        })
    }

    /// Create an all-ones (dense) mask.
    ///
    /// # Arguments
    /// * `shape` - Shape of the mask
    #[must_use]
    pub fn dense(shape: &[usize]) -> Self {
        let mask = Tensor::ones(shape);
        Self {
            mask,
            pattern: SparsityPattern::Unstructured,
            sparsity: 0.0,
        }
    }

    /// Get the sparsity ratio (0.0 = dense, 1.0 = all zeros).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Get the pattern used for this mask.
    #[must_use]
    pub fn pattern(&self) -> SparsityPattern {
        self.pattern
    }

    /// Get the underlying mask tensor.
    #[must_use]
    pub fn tensor(&self) -> &Tensor {
        &self.mask
    }

    /// Get the shape of the mask.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.mask.shape()
    }

    /// Apply mask to weights in-place.
    ///
    /// # Arguments
    /// * `weights` - Tensor to apply mask to (modified in-place)
    ///
    /// # Returns
    /// * `Ok(())` - Mask applied successfully
    /// * `Err(PruningError::ShapeMismatch)` - If shapes don't match
    ///
    /// # Toyota Way: Poka-Yoke
    /// Shape validation prevents applying mask to wrong tensor.
    pub fn apply(&self, weights: &mut Tensor) -> Result<(), PruningError> {
        if weights.shape() != self.mask.shape() {
            return Err(PruningError::ShapeMismatch {
                expected: self.mask.shape().to_vec(),
                got: weights.shape().to_vec(),
            });
        }

        // Element-wise multiplication
        let mask_data = self.mask.data();
        let weight_data = weights.data_mut();
        for (w, &m) in weight_data.iter_mut().zip(mask_data.iter()) {
            *w *= m;
        }

        Ok(())
    }

    /// Count the number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.mask.data().iter().filter(|&&v| v > 0.5).count()
    }

    /// Count the number of zero elements.
    #[must_use]
    pub fn num_zeros(&self) -> usize {
        self.mask.data().iter().filter(|&&v| v < 0.5).count()
    }
}

/// Generate an unstructured sparsity mask based on importance scores.
///
/// # Arguments
/// * `scores` - Importance scores tensor
/// * `target_sparsity` - Fraction of weights to prune (0.0 to 1.0)
///
/// # Returns
/// Mask where lowest-importance weights are set to 0.
pub fn generate_unstructured_mask(
    scores: &Tensor,
    target_sparsity: f32,
) -> Result<SparsityMask, PruningError> {
    if !(0.0..=1.0).contains(&target_sparsity) {
        return Err(PruningError::InvalidSparsity {
            value: target_sparsity,
            constraint: "must be between 0.0 and 1.0".to_string(),
        });
    }

    let data = scores.data();
    if data.is_empty() {
        return SparsityMask::new(Tensor::new(&[], &[0]), SparsityPattern::Unstructured);
    }

    // Find threshold for target sparsity
    let num_prune = (data.len() as f32 * target_sparsity) as usize;

    // Sort scores to find threshold
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = if num_prune == 0 {
        f32::NEG_INFINITY
    } else if num_prune >= sorted.len() {
        f32::INFINITY
    } else {
        sorted[num_prune - 1]
    };

    // Generate mask (1 = keep, 0 = prune)
    let mask_data: Vec<f32> = data
        .iter()
        .map(|&v| if v > threshold { 1.0 } else { 0.0 })
        .collect();

    SparsityMask::new(
        Tensor::new(&mask_data, scores.shape()),
        SparsityPattern::Unstructured,
    )
}

include!("mod_part_02.rs");
