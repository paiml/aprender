
/// Distributed inference context combining all parallelism strategies
#[derive(Debug)]
pub struct DistributedContext {
    /// Parallel configuration
    config: ParallelConfig,
    /// Tensor parallelism (if enabled)
    tensor_parallel: Option<TensorParallel>,
    /// Pipeline parallelism (if enabled)
    pipeline_parallel: Option<PipelineParallel>,
    /// ZeRO offload settings
    zero_offload: ZeroOffload,
    /// Initialized flag
    initialized: bool,
}

impl DistributedContext {
    /// Create a new distributed context
    pub fn new(config: ParallelConfig) -> Result<Self, ParallelError> {
        let tensor_parallel = if config.tp_size > 1 {
            Some(TensorParallel::new(config.tp_size, config.tp_rank())?)
        } else {
            None
        };

        // Note: Pipeline parallel requires layer count, initialized separately
        let pipeline_parallel = None;

        Ok(Self {
            config,
            tensor_parallel,
            pipeline_parallel,
            zero_offload: ZeroOffload::default(),
            initialized: true,
        })
    }

    /// Initialize pipeline parallelism
    pub fn init_pipeline(
        &mut self,
        total_layers: usize,
        micro_batch_size: usize,
    ) -> Result<(), ParallelError> {
        if self.config.pp_size > 1 {
            self.pipeline_parallel = Some(PipelineParallel::new(
                self.config.pp_size,
                self.config.pp_stage(),
                total_layers,
                micro_batch_size,
            )?);
        }
        Ok(())
    }

    /// Set ZeRO offload configuration
    pub fn set_zero_offload(&mut self, zero: ZeroOffload) {
        self.zero_offload = zero;
    }

    /// Get parallel configuration
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    /// Get tensor parallel context
    pub fn tensor_parallel(&self) -> Option<&TensorParallel> {
        self.tensor_parallel.as_ref()
    }

    /// Get pipeline parallel context
    pub fn pipeline_parallel(&self) -> Option<&PipelineParallel> {
        self.pipeline_parallel.as_ref()
    }

    /// Get mutable pipeline parallel context
    pub fn pipeline_parallel_mut(&mut self) -> Option<&mut PipelineParallel> {
        self.pipeline_parallel.as_mut()
    }

    /// Get ZeRO offload config
    pub fn zero_offload(&self) -> &ZeroOffload {
        &self.zero_offload
    }

    /// Check if distributed execution is enabled
    pub fn is_distributed(&self) -> bool {
        self.config.world_size > 1
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests;
