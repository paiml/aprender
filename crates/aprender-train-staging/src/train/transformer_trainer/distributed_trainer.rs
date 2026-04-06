//! Distributed CUDA trainer for data-parallel pretraining.
//!
//! Wraps `CudaTransformerTrainer` with per-block gradient accumulation
//! and AllReduce communication for multi-GPU / multi-node DDP.
//!
//! # Architecture
//!
//! ```text
//! DistributedCudaTrainer
//! ├── trainer: CudaTransformerTrainer     (local GPU training)
//! ├── comm: DistributedComm               (local channels or TCP)
//! └── dist_config: DistributedTrainConfig
//! ```
//!
//! # Training Step (DDP)
//!
//! 1. Each worker runs forward+backward on its data shard (accumulate_only=true)
//! 2. Pre-average local gradients (divide by accumulated_count)
//! 3. Per-block AllReduce (reverse order):
//!    - Send block gradients to coordinator
//!    - Receive averaged gradients
//!    - Overwrite local accum
//! 4. AllReduce non-block: LM head, final norm, embedding
//! 5. Upload averaged gradients to GPU, run optimizer step
//!
//! # Contract
//!
//! C-DDP-001: After AllReduce + optimizer step, all workers hold identical weights.

#[cfg(feature = "cuda")]
use std::sync::mpsc;

#[cfg(feature = "cuda")]
use super::config::DistributedTrainConfig;
#[cfg(feature = "cuda")]
use super::grad_accumulator::BlockGradientSet;

/// Communication backend for distributed training.
#[cfg(feature = "cuda")]
pub enum DistributedComm {
    /// Single-machine multi-GPU via crossbeam channels.
    ///
    /// Each worker has a send/recv pair for each other worker.
    /// AllReduce is done by sending gradients to rank 0,
    /// averaging, and broadcasting back.
    Local {
        /// Send gradient to coordinator
        tx: mpsc::Sender<GradientMessage>,
        /// Receive averaged gradient from coordinator
        rx: mpsc::Receiver<GradientMessage>,
    },
    /// Multi-node via TCP using the existing WorkerClient/GradientServer.
    Remote {
        /// TCP client for gradient exchange
        client: crate::finetune::WorkerClient,
    },
}

/// Message types for local (channel-based) gradient exchange.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub enum GradientMessage {
    /// Per-block gradient from a worker
    BlockGradient { block_idx: usize, gradients: Vec<f32>, component_sizes: Vec<u32> },
    /// Averaged per-block gradient from coordinator
    AveragedBlockGradient { block_idx: usize, gradients: Vec<f32>, component_sizes: Vec<u32> },
    /// Non-block gradient (LM head, final norm, embedding)
    NonBlockGradient { component: u8, gradients: Vec<f32> },
    /// Averaged non-block gradient
    AveragedNonBlockGradient { component: u8, gradients: Vec<f32> },
    /// Synchronization barrier
    Barrier,
}

/// Distributed CUDA trainer for data-parallel pretraining.
///
/// Wraps a single-GPU `CudaTransformerTrainer` with communication
/// and gradient averaging logic. The actual CUDA operations remain
/// in the underlying trainer — this layer only handles:
///
/// 1. Running forward+backward in accumulate-only mode
/// 2. Pre-averaging local gradients
/// 3. AllReducing gradients across workers
/// 4. Applying averaged gradients via the underlying trainer
///
/// # Safety
///
/// C-STREAMSYNC-001 applies: stream.synchronize() before all D2H transfers
/// is handled by the underlying CudaTransformerTrainer.
#[cfg(feature = "cuda")]
pub struct DistributedCudaTrainer {
    /// Underlying single-GPU trainer
    trainer: super::cuda_trainer::CudaTransformerTrainer,
    /// Communication backend
    comm: DistributedComm,
    /// Distributed configuration
    dist_config: DistributedTrainConfig,
    /// Current training step
    step: usize,
}

#[cfg(feature = "cuda")]
impl DistributedCudaTrainer {
    /// Create a new distributed trainer.
    ///
    /// Ensures the underlying trainer has gradient accumulation buffers
    /// (required for DDP even with accumulation_steps=1, since gradients
    /// must be downloaded to CPU for AllReduce).
    ///
    /// # Arguments
    /// * `trainer` - Pre-initialized single-GPU trainer (with ensure_grad_accum called)
    /// * `comm` - Communication backend (local channels or TCP)
    /// * `dist_config` - Distributed training configuration
    pub fn new(
        mut trainer: super::cuda_trainer::CudaTransformerTrainer,
        comm: DistributedComm,
        dist_config: DistributedTrainConfig,
    ) -> Self {
        // DDP always needs grad accum buffers for CPU-side AllReduce
        trainer.ensure_grad_accum();

        Self { trainer, comm, dist_config, step: 0 }
    }

    /// DDP training step: forward+backward → AllReduce → optimizer.
    ///
    /// 1. Local forward+backward (accumulate_only=true → grads to CPU accum)
    /// 2. Pre-average local gradients
    /// 3. AllReduce all gradient components across workers
    /// 4. Apply averaged gradients (upload to GPU + optimizer step)
    ///
    /// Returns average loss for this worker's batch.
    pub fn train_batch(&mut self, batch: &super::batch::LMBatch) -> f32 {
        // 1. Local forward+backward (accumulate only)
        let loss = self.trainer.forward_backward_batch(batch);

        // 2-3. Pre-average and AllReduce
        let step = self.step as u64;
        Self::allreduce_impl(step, &self.comm, &mut self.trainer);

        // 4. Apply averaged gradients
        self.trainer.apply_ddp_gradients();

        self.step += 1;
        loss
    }

    /// Pre-average local gradients, then AllReduce across workers.
    ///
    /// Separated as a static method to satisfy the borrow checker:
    /// `comm` and `trainer` are disjoint fields borrowed independently.
    fn allreduce_impl(
        step: u64,
        comm: &DistributedComm,
        trainer: &mut super::cuda_trainer::CudaTransformerTrainer,
    ) {
        // Phase 0: Pre-average local gradients before AllReduce.
        // Each worker divides by its local accumulated_count so the coordinator
        // averages per-sample means (not raw sums). This ensures C-DDP-001 even
        // if workers process different numbers of valid sequences.
        let local_count = {
            let accum = trainer.grad_accum_mut().unwrap();
            let count = accum.accumulated_count;
            accum.average(); // divides block + non-block grads by count
            count
        };
        // Average embedding grad separately (lives in CPU model, not in accum)
        if local_count > 1 {
            if let Some(mut eg) = trainer.embed_grad_vec() {
                let inv = 1.0 / local_count as f32;
                for g in &mut eg {
                    *g *= inv;
                }
                trainer.set_embed_grad(eg);
            }
        }

        // Phase 1-3: AllReduce via configured transport
        match comm {
            DistributedComm::Remote { client } => {
                Self::allreduce_remote(step, client, trainer);
            }
            DistributedComm::Local { tx, rx } => {
                Self::allreduce_local(step, tx, rx, trainer);
            }
        }
    }

    /// AllReduce via TCP (multi-process DDP).
    fn allreduce_remote(
        step: u64,
        client: &crate::finetune::WorkerClient,
        trainer: &mut super::cuda_trainer::CudaTransformerTrainer,
    ) {
        // Phase 1: Per-block AllReduce (reverse order matches backward pass)
        {
            let accum = trainer.grad_accum_mut().unwrap();
            let num_blocks = accum.num_blocks();
            for block_idx in (0..num_blocks).rev() {
                let flat = accum.block_grads[block_idx].flatten();
                let sizes = accum.block_grads[block_idx].component_sizes_u32();
                client
                    .send_block_gradient(step, block_idx as u32, num_blocks as u32, flat, sizes)
                    .expect("block gradient send failed");
                let avg = client.receive_averaged_block().expect("block gradient receive failed");
                accum.block_grads[block_idx] =
                    BlockGradientSet::from_flat(&avg.gradients, &avg.component_sizes);
            }
        }

        // Phase 2: Non-block AllReduce (LM head + final norm)
        {
            let accum = trainer.grad_accum_mut().unwrap();

            // LM head (component=0)
            let lm_grad = accum.lm_head_grad.clone();
            client.send_non_block_gradient(step, 0, lm_grad).expect("lm_head gradient send failed");
            let avg = client.receive_averaged_non_block().expect("lm_head gradient receive failed");
            accum.lm_head_grad = avg.gradients;

            // Final norm (component=1)
            let norm_grad = accum.final_norm_grad.clone();
            client
                .send_non_block_gradient(step, 1, norm_grad)
                .expect("final_norm gradient send failed");
            let avg =
                client.receive_averaged_non_block().expect("final_norm gradient receive failed");
            accum.final_norm_grad = avg.gradients;

            // Prevent re-averaging in gpu_optimizer_from_accum
            accum.accumulated_count = 1;
        }

        // Phase 3: Embedding AllReduce (CPU gradient, separate from accum)
        {
            let embed_grad = trainer.embed_grad_vec().unwrap_or_default();
            client
                .send_non_block_gradient(step, 2, embed_grad)
                .expect("embedding gradient send failed");
            let avg =
                client.receive_averaged_non_block().expect("embedding gradient receive failed");
            trainer.set_embed_grad(avg.gradients);
        }
    }

    /// AllReduce via mpsc channels (single-machine multi-GPU).
    fn allreduce_local(
        step: u64,
        tx: &mpsc::Sender<GradientMessage>,
        rx: &mpsc::Receiver<GradientMessage>,
        trainer: &mut super::cuda_trainer::CudaTransformerTrainer,
    ) {
        let _ = step; // used for logging in future

        // Phase 1: Per-block AllReduce via channels
        {
            let accum = trainer.grad_accum_mut().unwrap();
            let num_blocks = accum.num_blocks();
            for block_idx in (0..num_blocks).rev() {
                let flat = accum.block_grads[block_idx].flatten();
                let sizes = accum.block_grads[block_idx].component_sizes_u32();
                tx.send(GradientMessage::BlockGradient {
                    block_idx,
                    gradients: flat,
                    component_sizes: sizes,
                })
                .expect("channel send failed");

                match rx.recv().expect("channel recv failed") {
                    GradientMessage::AveragedBlockGradient {
                        gradients, component_sizes, ..
                    } => {
                        accum.block_grads[block_idx] =
                            BlockGradientSet::from_flat(&gradients, &component_sizes);
                    }
                    other => panic!("expected AveragedBlockGradient, got {other:?}"),
                }
            }
        }

        // Phase 2: Non-block AllReduce via channels
        {
            let accum = trainer.grad_accum_mut().unwrap();

            // LM head
            let lm_grad = accum.lm_head_grad.clone();
            tx.send(GradientMessage::NonBlockGradient { component: 0, gradients: lm_grad })
                .expect("channel send failed");
            match rx.recv().expect("channel recv failed") {
                GradientMessage::AveragedNonBlockGradient { gradients, .. } => {
                    accum.lm_head_grad = gradients;
                }
                other => panic!("expected AveragedNonBlockGradient, got {other:?}"),
            }

            // Final norm
            let norm_grad = accum.final_norm_grad.clone();
            tx.send(GradientMessage::NonBlockGradient { component: 1, gradients: norm_grad })
                .expect("channel send failed");
            match rx.recv().expect("channel recv failed") {
                GradientMessage::AveragedNonBlockGradient { gradients, .. } => {
                    accum.final_norm_grad = gradients;
                }
                other => panic!("expected AveragedNonBlockGradient, got {other:?}"),
            }

            accum.accumulated_count = 1;
        }

        // Phase 3: Embedding AllReduce
        {
            let embed_grad = trainer.embed_grad_vec().unwrap_or_default();
            tx.send(GradientMessage::NonBlockGradient { component: 2, gradients: embed_grad })
                .expect("channel send failed");
            match rx.recv().expect("channel recv failed") {
                GradientMessage::AveragedNonBlockGradient { gradients, .. } => {
                    trainer.set_embed_grad(gradients);
                }
                other => panic!("expected AveragedNonBlockGradient, got {other:?}"),
            }
        }
    }

    /// Get the distributed configuration.
    pub fn dist_config(&self) -> &DistributedTrainConfig {
        &self.dist_config
    }

    /// Get the current step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get a reference to the underlying trainer.
    pub fn trainer(&self) -> &super::cuda_trainer::CudaTransformerTrainer {
        &self.trainer
    }

    /// Get a mutable reference to the underlying trainer.
    pub fn trainer_mut(&mut self) -> &mut super::cuda_trainer::CudaTransformerTrainer {
        &mut self.trainer
    }

    /// Check if this worker is the coordinator (rank 0).
    pub fn is_coordinator(&self) -> bool {
        self.dist_config.rank == 0
    }

    /// Get world size.
    pub fn world_size(&self) -> usize {
        self.dist_config.world_size
    }

    /// Get rank.
    pub fn rank(&self) -> usize {
        self.dist_config.rank
    }

    /// Check if max_steps has been reached.
    pub fn reached_max_steps(&self) -> bool {
        self.trainer.reached_max_steps()
    }
}

/// Create a local communication pair for single-machine multi-GPU training.
///
/// Returns (coordinator_comm, worker_comms) where worker_comms[i] is for worker i.
/// The coordinator aggregates gradients and broadcasts averages.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn create_local_comm_pair() -> (
    (mpsc::Sender<GradientMessage>, mpsc::Receiver<GradientMessage>),
    (mpsc::Sender<GradientMessage>, mpsc::Receiver<GradientMessage>),
) {
    let (tx_to_coord, rx_at_coord) = mpsc::channel();
    let (tx_to_worker, rx_at_worker) = mpsc::channel();
    ((tx_to_worker, rx_at_coord), (tx_to_coord, rx_at_worker))
}

/// Shard batch indices across workers by interleaving.
///
/// Worker N gets batches N, N+world_size, N+2*world_size, ...
/// This ensures disjoint+complete coverage of the dataset.
pub fn shard_batches(num_batches: usize, rank: usize, world_size: usize) -> Vec<usize> {
    (rank..num_batches).step_by(world_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_compiles() {
        assert!(true);
    }

    #[test]
    fn test_data_sharding_by_rank() {
        // 10 batches, 2 workers
        let shard0 = shard_batches(10, 0, 2);
        let shard1 = shard_batches(10, 1, 2);

        // Worker 0 gets even indices
        assert_eq!(shard0, vec![0, 2, 4, 6, 8]);
        // Worker 1 gets odd indices
        assert_eq!(shard1, vec![1, 3, 5, 7, 9]);

        // Disjoint
        for idx in &shard0 {
            assert!(!shard1.contains(idx));
        }
        // Complete
        let mut all: Vec<usize> = shard0.iter().chain(shard1.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_data_sharding_uneven() {
        // 7 batches, 3 workers
        let shard0 = shard_batches(7, 0, 3);
        let shard1 = shard_batches(7, 1, 3);
        let shard2 = shard_batches(7, 2, 3);

        assert_eq!(shard0, vec![0, 3, 6]);
        assert_eq!(shard1, vec![1, 4]);
        assert_eq!(shard2, vec![2, 5]);

        let mut all: Vec<usize> =
            shard0.iter().chain(shard1.iter()).chain(shard2.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn test_data_sharding_single_worker() {
        let shard = shard_batches(5, 0, 1);
        assert_eq!(shard, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_data_sharding_more_workers_than_batches() {
        let shard0 = shard_batches(2, 0, 4);
        let shard1 = shard_batches(2, 1, 4);
        let shard2 = shard_batches(2, 2, 4);
        let shard3 = shard_batches(2, 3, 4);

        assert_eq!(shard0, vec![0]);
        assert_eq!(shard1, vec![1]);
        assert!(shard2.is_empty());
        assert!(shard3.is_empty());
    }
}
