//! TCP worker client for distributed training (worker side)
//!
//! The `WorkerClient` runs on each worker node and:
//! 1. Connects to the coordinator
//! 2. Receives shard assignments per training step
//! 3. Computes forward/backward locally
//! 4. Sends gradients to coordinator
//! 5. Receives averaged gradients and applies optimizer step
//!
//! # Contract: F-DP-004 (Backend Fallback)
//!
//! The worker uses `forward_hidden_dispatch()` which falls back:
//! CUDA → wgpu → CPU. Training always proceeds regardless of GPU availability.

use super::distributed::{DistributedConfig, WireMessage};
use super::gradient_server::{read_wire_message, send_wire_message};
use std::net::TcpStream;

/// Worker client that connects to the coordinator.
pub struct WorkerClient {
    config: DistributedConfig,
    stream: TcpStream,
    worker_id: u32,
    total_workers: u32,
}

/// Shard assignment received from coordinator.
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    pub step: u64,
    pub shard_start: usize,
    pub shard_end: usize,
}

/// Averaged gradient received from coordinator after AllReduce.
#[derive(Debug, Clone)]
pub struct AveragedResult {
    pub step: u64,
    pub gradients: Vec<f32>,
    pub global_loss: f32,
}

/// Averaged block gradient received from coordinator (v2 per-block DDP).
#[derive(Debug, Clone)]
pub struct AveragedBlockResult {
    pub step: u64,
    pub block_idx: u32,
    pub gradients: Vec<f32>,
    pub component_sizes: Vec<u32>,
}

/// Averaged non-block gradient received from coordinator (v2 DDP).
#[derive(Debug, Clone)]
pub struct AveragedNonBlockResult {
    pub step: u64,
    pub component: u8,
    pub gradients: Vec<f32>,
}

impl WorkerClient {
    /// Connect to the coordinator and complete the join handshake.
    ///
    /// # Arguments
    /// * `config` - Worker configuration with coordinator address
    /// * `gpu_count` - Number of GPUs this worker has
    /// * `backend` - Backend name (e.g., "wgpu", "cuda", "cpu")
    ///
    /// # Errors
    /// Returns error if connection or handshake fails.
    pub fn connect(
        config: DistributedConfig,
        gpu_count: u32,
        backend: &str,
    ) -> Result<Self, String> {
        let coord_addr = config
            .coordinator_addr
            .ok_or_else(|| "worker config must have coordinator_addr".to_string())?;

        eprintln!("[worker {}] Connecting to coordinator at {coord_addr}...", config.node_id);

        let stream = TcpStream::connect(coord_addr)
            .map_err(|e| format!("failed to connect to {coord_addr}: {e}"))?;

        // Send JoinRequest
        let join = WireMessage::JoinRequest {
            node_id: config.node_id.clone(),
            gpu_count,
            backend: backend.to_string(),
        };
        send_wire_message(&stream, &join)?;

        // Read JoinAccepted
        let response = read_wire_message(&stream)?;
        match response {
            WireMessage::JoinAccepted { worker_id, total_workers } => {
                eprintln!(
                    "[worker {}] Joined as worker {worker_id}/{total_workers}",
                    config.node_id
                );
                Ok(Self { config, stream, worker_id, total_workers })
            }
            other => Err(format!("expected JoinAccepted, got {other:?}")),
        }
    }

    /// Receive shard assignment for the next training step.
    ///
    /// Returns `None` if the coordinator sends a Shutdown message.
    ///
    /// # Errors
    /// Returns error on communication failure.
    pub fn receive_shard(&self) -> Result<Option<ShardAssignment>, String> {
        let msg = read_wire_message(&self.stream)?;
        match msg {
            WireMessage::ShardAssignment { step, shard_start, shard_end } => {
                Ok(Some(ShardAssignment { step, shard_start, shard_end }))
            }
            WireMessage::Shutdown => {
                eprintln!("[worker {}] Received shutdown from coordinator", self.config.node_id);
                Ok(None)
            }
            other => Err(format!("expected ShardAssignment or Shutdown, got {other:?}")),
        }
    }

    /// Send computed gradients to the coordinator.
    ///
    /// # Arguments
    /// * `step` - Training step number
    /// * `gradients` - Gradient vector (flattened LoRA params + classifier head)
    /// * `loss` - Average loss for this shard
    /// * `correct` - Number of correct predictions
    /// * `total` - Total samples in shard
    ///
    /// # Errors
    /// Returns error on send failure.
    pub fn send_gradients(
        &self,
        step: u64,
        gradients: Vec<f32>,
        loss: f32,
        correct: usize,
        total: usize,
    ) -> Result<(), String> {
        let msg = WireMessage::GradientPayload {
            step,
            worker_id: self.worker_id,
            gradients,
            loss,
            correct,
            total,
        };
        send_wire_message(&self.stream, &msg)
    }

    /// Receive averaged gradients from coordinator after AllReduce.
    ///
    /// # Errors
    /// Returns error on communication failure.
    pub fn receive_averaged(&self) -> Result<AveragedResult, String> {
        let msg = read_wire_message(&self.stream)?;
        match msg {
            WireMessage::AveragedGradient { step, gradients, global_loss } => {
                Ok(AveragedResult { step, gradients, global_loss })
            }
            WireMessage::Shutdown => Err("shutdown during AllReduce".to_string()),
            other => Err(format!("expected AveragedGradient, got {other:?}")),
        }
    }

    // --- v2 per-block DDP methods ---

    /// Send per-block gradient to coordinator for AllReduce (v2 DDP).
    ///
    /// # Arguments
    /// * `step` - Training step number
    /// * `block_idx` - Transformer block index (0-based)
    /// * `num_blocks` - Total number of transformer blocks
    /// * `gradients` - Flattened gradient vector (9 components concatenated)
    /// * `component_sizes` - Element count for each of the 9 components
    pub fn send_block_gradient(
        &self,
        step: u64,
        block_idx: u32,
        num_blocks: u32,
        gradients: Vec<f32>,
        component_sizes: Vec<u32>,
    ) -> Result<(), String> {
        let msg = WireMessage::BlockGradientPayload {
            step,
            worker_id: self.worker_id,
            block_idx,
            num_blocks,
            gradients,
            component_sizes,
        };
        send_wire_message(&self.stream, &msg)
    }

    /// Receive averaged block gradient from coordinator after AllReduce (v2 DDP).
    pub fn receive_averaged_block(&self) -> Result<AveragedBlockResult, String> {
        let msg = read_wire_message(&self.stream)?;
        match msg {
            WireMessage::AveragedBlockGradient { step, block_idx, gradients, component_sizes } => {
                Ok(AveragedBlockResult { step, block_idx, gradients, component_sizes })
            }
            WireMessage::Shutdown => Err("shutdown during block AllReduce".to_string()),
            other => Err(format!("expected AveragedBlockGradient, got {other:?}")),
        }
    }

    /// Send non-block gradient to coordinator for AllReduce (v2 DDP).
    ///
    /// # Arguments
    /// * `step` - Training step number
    /// * `component` - 0=lm_head, 1=final_norm, 2=embedding
    /// * `gradients` - Gradient vector for this component
    pub fn send_non_block_gradient(
        &self,
        step: u64,
        component: u8,
        gradients: Vec<f32>,
    ) -> Result<(), String> {
        let msg = WireMessage::NonBlockGradientPayload {
            step,
            worker_id: self.worker_id,
            component,
            gradients,
        };
        send_wire_message(&self.stream, &msg)
    }

    /// Receive averaged non-block gradient from coordinator after AllReduce (v2 DDP).
    pub fn receive_averaged_non_block(&self) -> Result<AveragedNonBlockResult, String> {
        let msg = read_wire_message(&self.stream)?;
        match msg {
            WireMessage::AveragedNonBlockGradient { step, component, gradients } => {
                Ok(AveragedNonBlockResult { step, component, gradients })
            }
            WireMessage::Shutdown => Err("shutdown during non-block AllReduce".to_string()),
            other => Err(format!("expected AveragedNonBlockGradient, got {other:?}")),
        }
    }

    /// This worker's assigned ID
    #[must_use]
    pub fn worker_id(&self) -> u32 {
        self.worker_id
    }

    /// Total number of workers in the cluster
    #[must_use]
    pub fn total_workers(&self) -> u32 {
        self.total_workers
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::super::distributed::DistributedConfig;
    use super::super::gradient_server::GradientServer;
    use super::*;
    use std::thread;

    #[test]
    fn test_worker_connect_and_join() {
        let server_config =
            DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let mut server = GradientServer::bind(server_config).expect("valid");
        let addr = server.local_addr();

        let handle = thread::spawn(move || {
            let worker_config = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(worker_config, 1, "cpu").expect("valid");
            assert_eq!(client.worker_id(), 0);
            assert_eq!(client.total_workers(), 1);
            client
        });

        server.wait_for_workers().expect("valid");
        let _client = handle.join().expect("valid");
    }

    #[test]
    fn test_worker_block_gradient_roundtrip() {
        let server_config =
            DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let mut server = GradientServer::bind(server_config).expect("valid");
        let addr = server.local_addr();

        let component_sizes = vec![4, 2, 2, 4, 8, 8, 8, 1, 1];
        let total: u32 = component_sizes.iter().sum();
        let grads: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();

        let grads_clone = grads.clone();
        let sizes_clone = component_sizes.clone();
        let handle = thread::spawn(move || {
            let worker_config = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(worker_config, 1, "cuda").expect("valid");

            // Send block gradient
            client.send_block_gradient(0, 5, 24, grads_clone, sizes_clone).expect("valid");

            // Receive averaged block gradient
            let avg = client.receive_averaged_block().expect("valid");
            assert_eq!(avg.step, 0);
            assert_eq!(avg.block_idx, 5);
            // Single worker: averaged == original
            assert_eq!(avg.gradients.len(), total as usize);
            avg
        });

        server.wait_for_workers().expect("valid");
        let result = server.collect_and_reduce_block(0, 5).expect("valid");
        assert_eq!(result.block_idx, 5);
        assert_eq!(result.avg_gradients.len(), total as usize);
        server.broadcast_averaged_block(0, &result).expect("valid");

        let avg = handle.join().expect("valid");
        // Single worker: averaged gradients should equal original
        for (a, b) in avg.gradients.iter().zip(grads.iter()) {
            assert!((a - b).abs() < 1e-6, "gradient mismatch: {a} != {b}");
        }
    }

    #[test]
    fn test_worker_non_block_gradient_roundtrip() {
        let server_config =
            DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let mut server = GradientServer::bind(server_config).expect("valid");
        let addr = server.local_addr();

        let grads = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let grads_clone = grads.clone();
        let handle = thread::spawn(move || {
            let worker_config = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(worker_config, 1, "cuda").expect("valid");

            // Send non-block gradient (component=0 = lm_head)
            client.send_non_block_gradient(0, 0, grads_clone).expect("valid");

            // Receive averaged
            let avg = client.receive_averaged_non_block().expect("valid");
            assert_eq!(avg.step, 0);
            assert_eq!(avg.component, 0);
            avg
        });

        server.wait_for_workers().expect("valid");
        let result = server.collect_and_reduce_non_block(0, 0).expect("valid");
        assert_eq!(result.component, 0);
        server.broadcast_averaged_non_block(0, &result).expect("valid");

        let avg = handle.join().expect("valid");
        for (a, b) in avg.gradients.iter().zip(grads.iter()) {
            assert!((a - b).abs() < 1e-6, "gradient mismatch: {a} != {b}");
        }
    }

    #[test]
    fn test_two_worker_block_allreduce() {
        let server_config =
            DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 2);
        let mut server = GradientServer::bind(server_config).expect("valid");
        let addr = server.local_addr();

        let component_sizes = vec![2, 1, 1, 2, 2, 2, 2, 1, 1];
        let total: u32 = component_sizes.iter().sum();

        // Worker 0: gradients = [1.0, 1.0, ...]
        let sizes0 = component_sizes.clone();
        let h0 = thread::spawn(move || {
            let cfg = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(cfg, 1, "cuda").expect("valid");
            let grads = vec![1.0f32; total as usize];
            client.send_block_gradient(0, 0, 1, grads, sizes0).expect("valid");
            client.receive_averaged_block().expect("valid")
        });

        // Worker 1: gradients = [3.0, 3.0, ...]
        let sizes1 = component_sizes.clone();
        let h1 = thread::spawn(move || {
            let cfg = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(cfg, 1, "cuda").expect("valid");
            let grads = vec![3.0f32; total as usize];
            client.send_block_gradient(0, 0, 1, grads, sizes1).expect("valid");
            client.receive_averaged_block().expect("valid")
        });

        server.wait_for_workers().expect("valid");
        let result = server.collect_and_reduce_block(0, 0).expect("valid");
        server.broadcast_averaged_block(0, &result).expect("valid");

        let avg0 = h0.join().expect("valid");
        let avg1 = h1.join().expect("valid");

        // Average of [1.0, 1.0, ...] and [3.0, 3.0, ...] = [2.0, 2.0, ...]
        for g in &avg0.gradients {
            assert!((g - 2.0).abs() < 1e-6, "expected 2.0, got {g}");
        }
        for g in &avg1.gradients {
            assert!((g - 2.0).abs() < 1e-6, "expected 2.0, got {g}");
        }
    }

    #[test]
    fn test_worker_full_training_step() {
        let server_config =
            DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let mut server = GradientServer::bind(server_config).expect("valid");
        let addr = server.local_addr();

        let handle = thread::spawn(move || {
            let worker_config = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(worker_config, 1, "cpu").expect("valid");

            // Receive shard
            let shard = client.receive_shard().expect("valid").expect("should get shard");
            assert_eq!(shard.step, 0);
            assert_eq!(shard.shard_start, 0);
            assert_eq!(shard.shard_end, 50);

            // Send gradients
            client.send_gradients(0, vec![1.0, 2.0, 3.0], 0.5, 48, 50).expect("valid");

            // Receive averaged
            let avg = client.receive_averaged().expect("valid");
            assert_eq!(avg.step, 0);
            assert_eq!(avg.gradients, vec![1.0, 2.0, 3.0]); // Single worker, no averaging
            assert!((avg.global_loss - 0.5).abs() < 1e-5);

            client
        });

        server.wait_for_workers().expect("valid");
        server.set_total_samples(50);
        server.send_shard_assignments(0).expect("valid");
        let result = server.collect_and_reduce(0).expect("valid");
        server.broadcast_averaged(0, &result).expect("valid");

        let _client = handle.join().expect("valid");
    }
}
