//! Multi-GPU and Distributed Inference
//!
//! Per spec §10: Implements parallelism strategies for 70B+ model inference.
//! Reference: [11] Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter LMs"
//!
//! ## Parallelism Strategies
//!
//! | Strategy | Description | Use Case | Scaling |
//! |----------|-------------|----------|---------|
//! | Tensor Parallel (TP) | Split tensors across GPUs | Within node | 2-8 GPUs |
//! | Pipeline Parallel (PP) | Split layers across GPUs | Across nodes | 2-64 GPUs |
//! | Data Parallel (DP) | Replicate model, split batches | High throughput | Any |
//!
//! ## Performance Target
//!
//! Per spec §1.3: >85% scaling efficiency for 2-8 GPUs (Amdahl's law measurement)

// Module-level clippy allows
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error type for parallelism operations
#[derive(Debug, Error)]
pub enum ParallelError {
    /// Invalid rank
    #[error("Invalid rank {rank} for world size {world_size}")]
    InvalidRank {
        /// The invalid rank value
        rank: usize,
        /// The total world size
        world_size: usize,
    },

    /// Invalid world size
    #[error("Invalid world size: {0}")]
    InvalidWorldSize(usize),

    /// Communication error
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// Tensor shape mismatch
    #[error("Tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        got: Vec<usize>,
    },

    /// Pipeline stage error
    #[error("Pipeline stage error: {0}")]
    PipelineError(String),

    /// Not initialized
    #[error("Parallel context not initialized")]
    NotInitialized,
}

/// Reduce operation for collective communications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum all values
    Sum,
    /// Take maximum
    Max,
    /// Take minimum
    Min,
    /// Average (Sum / world_size)
    Avg,
}

/// Parallel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Tensor parallel size (within node)
    pub tp_size: usize,
    /// Pipeline parallel size (across nodes)
    pub pp_size: usize,
    /// Data parallel size (batch distribution)
    pub dp_size: usize,
    /// Current global rank
    pub rank: usize,
    /// Total world size
    pub world_size: usize,
}

impl ParallelConfig {
    /// Create a new parallel configuration
    ///
    /// # Arguments
    ///
    /// * `tp_size` - Tensor parallel size (typically 2, 4, or 8)
    /// * `pp_size` - Pipeline parallel size (number of stages)
    /// * `dp_size` - Data parallel size (batch replication)
    /// * `rank` - Current process rank
    pub fn new(
        tp_size: usize,
        pp_size: usize,
        dp_size: usize,
        rank: usize,
    ) -> Result<Self, ParallelError> {
        let world_size = tp_size * pp_size * dp_size;

        if world_size == 0 {
            return Err(ParallelError::InvalidWorldSize(0));
        }

        if rank >= world_size {
            return Err(ParallelError::InvalidRank { rank, world_size });
        }

        Ok(Self {
            tp_size,
            pp_size,
            dp_size,
            rank,
            world_size,
        })
    }

    /// Create single-GPU configuration (no parallelism)
    pub fn single() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            dp_size: 1,
            rank: 0,
            world_size: 1,
        }
    }

    /// Get tensor parallel rank within TP group
    pub fn tp_rank(&self) -> usize {
        self.rank % self.tp_size
    }

    /// Get pipeline parallel stage
    pub fn pp_stage(&self) -> usize {
        (self.rank / self.tp_size) % self.pp_size
    }

    /// Get data parallel rank
    pub fn dp_rank(&self) -> usize {
        self.rank / (self.tp_size * self.pp_size)
    }

    /// Check if this is the first TP rank
    pub fn is_tp_first(&self) -> bool {
        self.tp_rank() == 0
    }

    /// Check if this is the last TP rank
    pub fn is_tp_last(&self) -> bool {
        self.tp_rank() == self.tp_size - 1
    }

    /// Check if this is the first PP stage
    pub fn is_pp_first(&self) -> bool {
        self.pp_stage() == 0
    }

    /// Check if this is the last PP stage
    pub fn is_pp_last(&self) -> bool {
        self.pp_stage() == self.pp_size - 1
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::single()
    }
}

/// Mock tensor for parallelism testing
/// In production, this would be replaced with trueno::Tensor
#[derive(Debug, Clone)]
pub struct ParallelTensor {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data (f32 for simplicity)
    pub data: Vec<f32>,
}

impl ParallelTensor {
    /// Create a new tensor
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, ParallelError> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![expected_size],
                got: vec![data.len()],
            });
        }
        Ok(Self { shape, data })
    }

    /// Create a zeros tensor
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Get a narrow slice along a dimension
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, ParallelError> {
        if dim >= self.shape.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![dim],
                got: self.shape.clone(),
            });
        }

        // For 2D tensors (matrices), implement proper narrowing
        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];

            if dim == 0 {
                // Narrow rows
                let mut new_data = Vec::with_capacity(length * cols);
                for row in start..(start + length) {
                    let row_start = row * cols;
                    new_data.extend_from_slice(&self.data[row_start..row_start + cols]);
                }
                let new_shape = vec![length, cols];
                return Ok(Self {
                    shape: new_shape,
                    data: new_data,
                });
            }
            // Narrow columns
            let mut new_data = Vec::with_capacity(rows * length);
            for row in 0..rows {
                let row_start = row * cols;
                new_data
                    .extend_from_slice(&self.data[row_start + start..row_start + start + length]);
            }
            let new_shape = vec![rows, length];
            return Ok(Self {
                shape: new_shape,
                data: new_data,
            });
        }

        // For 1D tensors
        if self.shape.len() == 1 {
            let new_data = self.data[start..start + length].to_vec();
            return Ok(Self {
                shape: vec![length],
                data: new_data,
            });
        }

        // Fallback: simplified implementation
        let new_data = self.data[start..start + length].to_vec();
        let mut new_shape = self.shape.clone();
        new_shape[dim] = length;
        Ok(Self {
            shape: new_shape,
            data: new_data,
        })
    }

    /// Transpose for 2D tensors
    pub fn transpose(&self) -> Result<Self, ParallelError> {
        contract_pre_transpose!();
        if self.shape.len() != 2 {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.shape.len()],
            });
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut new_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        contract_post_transpose!(&new_data);
        Ok(Self {
            shape: vec![cols, rows],
            data: new_data,
        })
    }

    /// Matrix multiplication (simplified)
    pub fn matmul(&self, other: &Self) -> Result<Self, ParallelError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![2, 2],
                got: vec![self.shape.len(), other.shape.len()],
            });
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        if k != other.shape[0] {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![k],
                got: vec![other.shape[0]],
            });
        }

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Self {
            shape: vec![m, n],
            data: result,
        })
    }

    /// Add another tensor element-wise
    pub fn add(&self, other: &Self) -> Result<Self, ParallelError> {
        if self.shape != other.shape {
            return Err(ParallelError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Mock communicator for collective operations
/// In production, this would use NCCL or MPI
#[derive(Debug, Clone)]
pub struct Communicator {
    /// World size
    world_size: usize,
    /// Current rank
    rank: usize,
}

include!("mod_all_reduce_communicator.rs");
include!("distributed_context_impl.rs");
