//! Transformer-specific training utilities
//!
//! Provides specialized training components for transformer language models,
//! including tokenized batch creation and language modeling training loops.

mod batch;
mod config;
mod cuda_trainer;
pub mod distributed_checkpoint;
mod distributed_trainer;
pub mod elastic;
pub mod gpu_grad_accumulator;
pub mod grad_accumulator;
pub mod pipeline;
pub mod sequence_parallel;
pub mod step_profiler;
pub mod tensor_parallel;
mod trainer;
mod utils;
#[cfg(feature = "gpu")]
pub mod wgpu_attention;
#[cfg(feature = "gpu")]
pub mod wgpu_backward;
#[cfg(feature = "gpu")]
pub mod wgpu_checkpoint;
#[cfg(feature = "gpu")]
pub mod wgpu_nf4;
#[cfg(feature = "gpu")]
pub mod wgpu_runner;
#[cfg(feature = "gpu")]
pub mod wgpu_trainer;
pub mod zero;

#[cfg(test)]
mod falsify_lora_tests;
#[cfg(test)]
mod tests;

// Re-export all public types
pub use batch::LMBatch;
pub use config::{
    DistributedBackend, DistributedRole, DistributedTrainConfig, TransformerTrainConfig,
};
pub use cuda_trainer::CudaTransformerTrainer;
pub use distributed_checkpoint::DistributedCheckpointCoordinator;
pub use distributed_trainer::shard_batches;
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub use distributed_trainer::{DistributedComm, DistributedCudaTrainer, GradientMessage};
pub use elastic::ElasticCoordinator;
pub use grad_accumulator::{BlockGradientSet, PerBlockGradientAccumulator};
pub use pipeline::{PipelineAction, PipelineActivationBuffer, PipelineStage};
pub use sequence_parallel::{
    CausalMaskType, RingAttentionSchedule, SequenceParallelConfig, SpCommCost,
};
pub use tensor_parallel::{
    ColumnParallelShard, RowParallelShard, TensorParallelConfig, TpCommCost,
};
pub use trainer::TransformerTrainer;
pub use utils::{perplexity, tokens_per_second};
pub use zero::{OptimizerShard, ZeroShardMap};
