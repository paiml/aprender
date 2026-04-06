
impl Communicator {
    /// Create a new communicator
    pub fn new(world_size: usize, rank: usize) -> Result<Self, ParallelError> {
        if rank >= world_size {
            return Err(ParallelError::InvalidRank { rank, world_size });
        }
        Ok(Self {
            world_size,
            rank,
        })
    }

    /// All-reduce operation
    pub fn all_reduce(
        &self,
        tensor: &ParallelTensor,
        op: ReduceOp,
    ) -> Result<ParallelTensor, ParallelError> {
        // In a real implementation, this would use NCCL
        // For testing, we simulate single-process behavior
        match op {
            ReduceOp::Sum => {
                // Single process: multiply by world_size to simulate sum from all ranks
                let data: Vec<f32> = tensor
                    .data
                    .iter()
                    .map(|x| x * self.world_size as f32)
                    .collect();
                Ok(ParallelTensor {
                    shape: tensor.shape.clone(),
                    data,
                })
            },
            ReduceOp::Avg => {
                // Average: no change in single process (sum / world_size = value)
                Ok(tensor.clone())
            },
            ReduceOp::Max | ReduceOp::Min => {
                // Single process: return as-is
                Ok(tensor.clone())
            },
        }
    }

    /// All-gather operation
    pub fn all_gather(&self, tensor: &ParallelTensor) -> Result<ParallelTensor, ParallelError> {
        // Simulate all-gather by replicating data world_size times
        let mut data = Vec::with_capacity(tensor.data.len() * self.world_size);
        for _ in 0..self.world_size {
            data.extend_from_slice(&tensor.data);
        }

        let mut new_shape = tensor.shape.clone();
        if !new_shape.is_empty() {
            new_shape[0] *= self.world_size;
        }

        Ok(ParallelTensor {
            shape: new_shape,
            data,
        })
    }

    /// Reduce-scatter operation
    pub fn reduce_scatter(
        &self,
        tensor: &ParallelTensor,
        op: ReduceOp,
    ) -> Result<ParallelTensor, ParallelError> {
        // Reduce then scatter: each rank gets 1/world_size of the result
        let chunk_size = tensor.data.len() / self.world_size;
        let start = self.rank * chunk_size;
        let end = start + chunk_size;

        let chunk_data: Vec<f32> = match op {
            ReduceOp::Sum => tensor.data[start..end]
                .iter()
                .map(|x| x * self.world_size as f32)
                .collect(),
            ReduceOp::Avg | ReduceOp::Max | ReduceOp::Min => tensor.data[start..end].to_vec(),
        };

        let mut new_shape = tensor.shape.clone();
        if !new_shape.is_empty() {
            new_shape[0] /= self.world_size;
        }

        Ok(ParallelTensor {
            shape: new_shape,
            data: chunk_data,
        })
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> Result<(), ParallelError> {
        // In real implementation, this would synchronize all processes
        Ok(())
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Tensor Parallelism for multi-GPU inference
/// Reference: [11] Megatron-LM tensor parallelism
#[derive(Debug)]
pub struct TensorParallel {
    /// Number of tensor parallel ranks
    tp_size: usize,
    /// Current rank within TP group
    rank: usize,
    /// Communication group
    comm: Communicator,
}

impl TensorParallel {
    /// Create a new tensor parallel context
    pub fn new(tp_size: usize, rank: usize) -> Result<Self, ParallelError> {
        if tp_size == 0 {
            return Err(ParallelError::InvalidWorldSize(0));
        }
        if rank >= tp_size {
            return Err(ParallelError::InvalidRank {
                rank,
                world_size: tp_size,
            });
        }

        let comm = Communicator::new(tp_size, rank)?;

        Ok(Self {
            tp_size,
            rank,
            comm,
        })
    }

    /// Get chunk size for weight sharding
    pub fn chunk_size(&self, total_size: usize) -> usize {
        total_size / self.tp_size
    }

    /// Column-parallel linear (for MLP first layer, attention QKV)
    ///
    /// Each rank holds weight[:, rank*chunk:(rank+1)*chunk]
    /// No communication needed as outputs are independent
    pub fn column_parallel_linear(
        &self,
        input: &ParallelTensor,
        weight: &ParallelTensor,
        bias: Option<&ParallelTensor>,
    ) -> Result<ParallelTensor, ParallelError> {
        // Get local weight slice
        let output_dim = weight.shape[0];
        let chunk = self.chunk_size(output_dim);
        let local_weight = weight.narrow(0, self.rank * chunk, chunk)?;

        // Transpose weight for matmul: (out_chunk, in) -> (in, out_chunk)
        let weight_t = local_weight.transpose()?;

        // Local matmul: (batch, in) @ (in, out_chunk) -> (batch, out_chunk)
        let mut local_output = input.matmul(&weight_t)?;

        // Add local bias if present
        if let Some(b) = bias {
            let local_bias = b.narrow(0, self.rank * chunk, chunk)?;
            // Broadcast bias addition
            let bias_expanded = ParallelTensor {
                shape: local_output.shape.clone(),
                data: local_output
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v + local_bias.data[i % local_bias.data.len()])
                    .collect(),
            };
            local_output = bias_expanded;
        }

        Ok(local_output)
    }

    /// Row-parallel linear (for MLP second layer, attention output)
    ///
    /// Each rank holds weight[rank*chunk:(rank+1)*chunk, :]
    /// Requires all-reduce to sum partial results
    pub fn row_parallel_linear(
        &self,
        input: &ParallelTensor,
        weight: &ParallelTensor,
        bias: Option<&ParallelTensor>,
    ) -> Result<ParallelTensor, ParallelError> {
        // Get local weight slice (rows)
        let input_dim = weight.shape[0];
        let chunk = self.chunk_size(input_dim);
        let local_weight = weight.narrow(0, self.rank * chunk, chunk)?;

        // Transpose for matmul
        let weight_t = local_weight.transpose()?;

        // Local matmul
        let local_output = input.matmul(&weight_t)?;

        // All-reduce to sum partial results
        let mut output = self.comm.all_reduce(&local_output, ReduceOp::Sum)?;

        // Add bias only on rank 0 to avoid double counting
        if self.rank == 0 {
            if let Some(b) = bias {
                output = output.add(b)?;
            }
        }

        Ok(output)
    }

    /// Get TP rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get TP size
    pub fn tp_size(&self) -> usize {
        self.tp_size
    }
}

/// Pipeline stage info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage index
    pub index: usize,
    /// Start layer index
    pub start_layer: usize,
    /// End layer index (exclusive)
    pub end_layer: usize,
    /// Number of layers in this stage
    pub num_layers: usize,
}

/// Pipeline Parallelism for multi-node inference
/// Reference: [11] GPipe-style pipeline parallelism
#[derive(Debug)]
pub struct PipelineParallel {
    /// Number of pipeline stages
    pp_size: usize,
    /// Current stage
    stage: usize,
    /// Stage info
    stage_info: PipelineStage,
    /// Micro-batch size for pipelining
    micro_batch_size: usize,
    /// Stats
    stats: PipelineStats,
}

/// Pipeline execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total micro-batches processed
    pub micro_batches_processed: u64,
    /// Total pipeline bubbles (idle time)
    pub bubble_time_ms: f64,
    /// Average stage latency
    pub avg_stage_latency_ms: f64,
    /// Total forward passes
    pub forward_passes: u64,
}

impl PipelineParallel {
    /// Create a new pipeline parallel context
    ///
    /// # Arguments
    ///
    /// * `pp_size` - Number of pipeline stages
    /// * `stage` - Current stage index (0 to pp_size-1)
    /// * `total_layers` - Total number of layers to distribute
    /// * `micro_batch_size` - Size of micro-batches for pipelining
    pub fn new(
        pp_size: usize,
        stage: usize,
        total_layers: usize,
        micro_batch_size: usize,
    ) -> Result<Self, ParallelError> {
        if pp_size == 0 {
            return Err(ParallelError::InvalidWorldSize(0));
        }
        if stage >= pp_size {
            return Err(ParallelError::InvalidRank {
                rank: stage,
                world_size: pp_size,
            });
        }

        // Distribute layers evenly across stages
        let layers_per_stage = total_layers / pp_size;
        let extra_layers = total_layers % pp_size;

        // Earlier stages get extra layers if uneven
        let start_layer = stage * layers_per_stage + stage.min(extra_layers);
        let num_layers = layers_per_stage + usize::from(stage < extra_layers);
        let end_layer = start_layer + num_layers;

        let stage_info = PipelineStage {
            index: stage,
            start_layer,
            end_layer,
            num_layers,
        };

        Ok(Self {
            pp_size,
            stage,
            stage_info,
            micro_batch_size,
            stats: PipelineStats::default(),
        })
    }

    /// Get stage info
    pub fn stage_info(&self) -> &PipelineStage {
        &self.stage_info
    }

    /// Get micro-batch size
    pub fn micro_batch_size(&self) -> usize {
        self.micro_batch_size
    }

    /// Check if this is the first stage
    pub fn is_first_stage(&self) -> bool {
        self.stage == 0
    }

    /// Check if this is the last stage
    pub fn is_last_stage(&self) -> bool {
        self.stage == self.pp_size - 1
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.pp_size
    }

    /// Get current stage index
    pub fn stage(&self) -> usize {
        self.stage
    }

    /// Calculate theoretical bubble ratio (idle time fraction)
    /// Bubble ratio = (pp_size - 1) / (pp_size + num_microbatches - 1)
    pub fn bubble_ratio(&self, num_microbatches: usize) -> f32 {
        if num_microbatches == 0 {
            return 1.0;
        }
        (self.pp_size - 1) as f32 / (self.pp_size + num_microbatches - 1) as f32
    }

    /// Get statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Record a micro-batch processed
    pub fn record_micro_batch(&mut self, stage_latency_ms: f64) {
        self.stats.micro_batches_processed += 1;
        self.stats.forward_passes += 1;

        // Update running average
        let n = self.stats.micro_batches_processed as f64;
        self.stats.avg_stage_latency_ms =
            (self.stats.avg_stage_latency_ms * (n - 1.0) + stage_latency_ms) / n;
    }
}

/// ZeRO-Inference memory offload
/// Reference: [10] Microsoft DeepSpeed ZeRO-Inference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // Config struct - bools are appropriate
pub struct ZeroOffload {
    /// Offload optimizer states to CPU
    pub offload_optimizer: bool,
    /// Offload parameters to CPU
    pub offload_params: bool,
    /// Offload activations to CPU
    pub offload_activations: bool,
    /// Pin memory for faster CPU-GPU transfer
    pub pin_memory: bool,
    /// Overlap compute and communication
    pub overlap_comm: bool,
}

impl Default for ZeroOffload {
    fn default() -> Self {
        Self {
            offload_optimizer: true,
            offload_params: false,
            offload_activations: false,
            pin_memory: true,
            overlap_comm: true,
        }
    }
}

impl ZeroOffload {
    /// Create inference-optimized config (offload everything)
    pub fn inference() -> Self {
        Self {
            offload_optimizer: false, // No optimizer in inference
            offload_params: true,
            offload_activations: true,
            pin_memory: true,
            overlap_comm: true,
        }
    }

    /// Estimate memory savings ratio
    pub fn memory_savings_ratio(&self) -> f32 {
        let mut ratio = 1.0;
        if self.offload_params {
            ratio *= 0.5; // Params on CPU
        }
        if self.offload_activations {
            ratio *= 0.7; // Activations on CPU
        }
        1.0 - ratio
    }
}
