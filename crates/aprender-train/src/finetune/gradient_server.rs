//! TCP gradient server for distributed training (coordinator side)
//!
//! The `GradientServer` runs on the coordinator node and:
//! 1. Accepts worker connections
//! 2. Assigns shard ranges per training step
//! 3. Collects gradients from all workers
//! 4. Computes AllReduce (average) and broadcasts result
//!
//! # Contract: F-DP-001 (Weight Consistency)
//!
//! After broadcasting averaged gradients, all workers apply the same optimizer
//! step, maintaining weight consistency.
//!
//! # Contract: F-DP-003 (Gradient Stability)
//!
//! If any worker sends NaN/Inf gradients, the server halts training (Jidoka).

use super::data_parallel::{average_gradients, has_non_finite};
use super::distributed::{DistributedConfig, WireMessage};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

/// Connected worker info tracked by the server.
#[derive(Debug)]
struct WorkerConnection {
    worker_id: u32,
    #[allow(dead_code)]
    node_id: String,
    #[allow(dead_code)]
    gpu_count: u32,
    #[allow(dead_code)]
    backend: String,
    stream: TcpStream,
}

/// Gradient server running on the coordinator node.
pub struct GradientServer {
    config: DistributedConfig,
    listener: TcpListener,
    workers: Vec<WorkerConnection>,
    total_samples: usize,
}

/// Result of one AllReduce step across all workers.
#[derive(Debug, Clone)]
pub struct AllReduceResult {
    /// Averaged gradient vector
    pub avg_gradients: Vec<f32>,
    /// Sample-weighted average loss
    pub global_loss: f32,
    /// Total correct predictions
    pub total_correct: usize,
    /// Total samples processed
    pub total_samples: usize,
    /// AllReduce wall time in milliseconds
    pub allreduce_ms: f64,
}

/// Result of per-block AllReduce for DDP pretraining.
#[derive(Debug, Clone)]
pub struct BlockAllReduceResult {
    /// Block index
    pub block_idx: u32,
    /// Averaged gradient vector (flattened, same layout as BlockGradientPayload)
    pub avg_gradients: Vec<f32>,
    /// Component sizes (for reconstructing block gradient structure)
    pub component_sizes: Vec<u32>,
    /// AllReduce wall time in milliseconds
    pub allreduce_ms: f64,
}

/// Result of non-block AllReduce for DDP pretraining.
#[derive(Debug, Clone)]
pub struct NonBlockAllReduceResult {
    /// Component ID (0=lm_head, 1=final_norm, 2=embedding)
    pub component: u8,
    /// Averaged gradient vector
    pub avg_gradients: Vec<f32>,
    /// AllReduce wall time in milliseconds
    pub allreduce_ms: f64,
}

impl GradientServer {
    /// Create and bind the gradient server.
    ///
    /// # Errors
    /// Returns error if binding fails.
    pub fn bind(config: DistributedConfig) -> Result<Self, String> {
        let listener = TcpListener::bind(config.bind_addr)
            .map_err(|e| format!("failed to bind {}: {e}", config.bind_addr))?;
        eprintln!(
            "[coordinator] Listening on {} (expecting {} workers)",
            config.bind_addr, config.expect_workers
        );
        Ok(Self { config, listener, workers: Vec::new(), total_samples: 0 })
    }

    /// Wait for all expected workers to connect.
    ///
    /// Blocks until `expect_workers` workers have sent JoinRequest messages.
    ///
    /// # Errors
    /// Returns error if any connection fails or timeout is exceeded.
    pub fn wait_for_workers(&mut self) -> Result<(), String> {
        let expected = self.config.expect_workers;
        eprintln!("[coordinator] Waiting for {expected} workers to connect...");

        while self.workers.len() < expected {
            let (stream, addr) =
                self.listener.accept().map_err(|e| format!("accept failed: {e}"))?;
            eprintln!("[coordinator] Connection from {addr}");

            // Read JoinRequest
            let msg = read_wire_message(&stream)?;
            match msg {
                WireMessage::JoinRequest { node_id, gpu_count, backend } => {
                    let worker_id = self.workers.len() as u32;
                    eprintln!(
                        "[coordinator] Worker {worker_id} joined: {node_id} ({gpu_count} GPUs, {backend})"
                    );

                    // Send JoinAccepted
                    let response =
                        WireMessage::JoinAccepted { worker_id, total_workers: expected as u32 };
                    send_wire_message(&stream, &response)?;

                    self.workers.push(WorkerConnection {
                        worker_id,
                        node_id,
                        gpu_count,
                        backend,
                        stream,
                    });
                }
                other => {
                    return Err(format!("expected JoinRequest, got {other:?}"));
                }
            }
        }

        eprintln!("[coordinator] All {expected} workers connected");
        Ok(())
    }

    /// Set total sample count for sharding
    pub fn set_total_samples(&mut self, n: usize) {
        self.total_samples = n;
    }

    /// Send shard assignments to all workers for a given step.
    ///
    /// # Errors
    /// Returns error if any send fails.
    pub fn send_shard_assignments(&mut self, step: u64) -> Result<(), String> {
        let n = self.workers.len();
        let shard_size = self.total_samples / n;

        for (i, worker) in self.workers.iter().enumerate() {
            let start = i * shard_size;
            let end = if i == n - 1 { self.total_samples } else { start + shard_size };
            let msg = WireMessage::ShardAssignment { step, shard_start: start, shard_end: end };
            send_wire_message(&worker.stream, &msg)?;
        }
        Ok(())
    }

    /// Collect gradients from all workers and compute AllReduce.
    ///
    /// # Contract: F-DP-003
    ///
    /// If any gradient contains NaN/Inf, returns an error (Jidoka halt).
    ///
    /// # Errors
    /// Returns error on communication failure or non-finite gradient.
    pub fn collect_and_reduce(&mut self, step: u64) -> Result<AllReduceResult, String> {
        let start = Instant::now();
        let n = self.workers.len();
        let mut all_grads: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut total_loss = 0.0f32;
        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        for worker in &self.workers {
            let msg = read_wire_message(&worker.stream)?;
            match msg {
                WireMessage::GradientPayload {
                    step: recv_step,
                    gradients,
                    loss,
                    correct,
                    total,
                    ..
                } => {
                    if recv_step != step {
                        return Err(format!("step mismatch: expected {step}, got {recv_step}"));
                    }

                    // Jidoka: halt on NaN/Inf (F-DP-003)
                    if has_non_finite(&gradients) {
                        return Err(format!(
                            "JIDOKA HALT: worker {} sent non-finite gradient at step {step}",
                            worker.worker_id
                        ));
                    }

                    total_loss += loss * total as f32;
                    total_correct += correct;
                    total_samples += total;
                    all_grads.push(gradients);
                }
                other => {
                    return Err(format!(
                        "expected GradientPayload from worker {}, got {other:?}",
                        worker.worker_id
                    ));
                }
            }
        }

        // AllReduce: average gradients (F-DP-001)
        let avg_gradients = average_gradients(&all_grads);
        let global_loss = if total_samples > 0 { total_loss / total_samples as f32 } else { 0.0 };

        let allreduce_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(AllReduceResult {
            avg_gradients,
            global_loss,
            total_correct,
            total_samples,
            allreduce_ms,
        })
    }

    /// Broadcast averaged gradients to all workers.
    ///
    /// # Errors
    /// Returns error if any send fails.
    pub fn broadcast_averaged(
        &mut self,
        step: u64,
        result: &AllReduceResult,
    ) -> Result<(), String> {
        let msg = WireMessage::AveragedGradient {
            step,
            gradients: result.avg_gradients.clone(),
            global_loss: result.global_loss,
        };
        for worker in &self.workers {
            send_wire_message(&worker.stream, &msg)?;
        }
        Ok(())
    }

    /// Send shutdown message to all workers.
    pub fn shutdown_workers(&mut self) {
        for worker in &self.workers {
            let _ = send_wire_message(&worker.stream, &WireMessage::Shutdown);
        }
    }

    /// Number of connected workers
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Collect and reduce per-block gradients from all workers.
    ///
    /// Waits for `BlockGradientPayload` from each worker for the specified
    /// block index, averages them, and returns the result.
    ///
    /// # Contract: C-DDP-001
    ///
    /// Output equals arithmetic mean of all workers' block gradients.
    /// Jidoka halt on NaN/Inf gradients (F-DP-003).
    ///
    /// # Errors
    ///
    /// Returns error on communication failure, step mismatch, or NaN gradient.
    pub fn collect_and_reduce_block(
        &mut self,
        step: u64,
        block_idx: u32,
    ) -> Result<BlockAllReduceResult, String> {
        let start = Instant::now();
        let n = self.workers.len();
        let mut all_grads: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut component_sizes = Vec::new();

        for worker in &self.workers {
            let msg = read_wire_message(&worker.stream)?;
            match msg {
                WireMessage::BlockGradientPayload {
                    step: recv_step,
                    block_idx: recv_block_idx,
                    gradients,
                    component_sizes: cs,
                    ..
                } => {
                    if recv_step != step {
                        return Err(format!("step mismatch: expected {step}, got {recv_step}"));
                    }
                    if recv_block_idx != block_idx {
                        return Err(format!(
                            "block_idx mismatch: expected {block_idx}, got {recv_block_idx}"
                        ));
                    }
                    if has_non_finite(&gradients) {
                        return Err(format!(
                            "JIDOKA HALT: worker {} sent non-finite block {block_idx} gradient at step {step}",
                            worker.worker_id
                        ));
                    }
                    if component_sizes.is_empty() {
                        component_sizes = cs;
                    }
                    all_grads.push(gradients);
                }
                other => {
                    return Err(format!(
                        "expected BlockGradientPayload from worker {}, got {other:?}",
                        worker.worker_id
                    ));
                }
            }
        }

        let avg_gradients = average_gradients(&all_grads);
        let allreduce_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BlockAllReduceResult { block_idx, avg_gradients, component_sizes, allreduce_ms })
    }

    /// Broadcast averaged block gradient to all workers.
    ///
    /// # Errors
    /// Returns error if any send fails.
    pub fn broadcast_averaged_block(
        &mut self,
        step: u64,
        result: &BlockAllReduceResult,
    ) -> Result<(), String> {
        let msg = WireMessage::AveragedBlockGradient {
            step,
            block_idx: result.block_idx,
            gradients: result.avg_gradients.clone(),
            component_sizes: result.component_sizes.clone(),
        };
        for worker in &self.workers {
            send_wire_message(&worker.stream, &msg)?;
        }
        Ok(())
    }

    /// Collect and reduce non-block gradient from all workers.
    ///
    /// Used for LM head, final norm, and embedding gradients.
    ///
    /// # Errors
    /// Returns error on communication failure or NaN gradient.
    pub fn collect_and_reduce_non_block(
        &mut self,
        step: u64,
        expected_component: u8,
    ) -> Result<NonBlockAllReduceResult, String> {
        let start = Instant::now();
        let n = self.workers.len();
        let mut all_grads: Vec<Vec<f32>> = Vec::with_capacity(n);

        for worker in &self.workers {
            let msg = read_wire_message(&worker.stream)?;
            match msg {
                WireMessage::NonBlockGradientPayload {
                    step: recv_step,
                    component,
                    gradients,
                    ..
                } => {
                    if recv_step != step {
                        return Err(format!("step mismatch: expected {step}, got {recv_step}"));
                    }
                    if component != expected_component {
                        return Err(format!(
                            "component mismatch: expected {expected_component}, got {component}"
                        ));
                    }
                    if has_non_finite(&gradients) {
                        return Err(format!(
                            "JIDOKA HALT: worker {} sent non-finite component {component} gradient at step {step}",
                            worker.worker_id
                        ));
                    }
                    all_grads.push(gradients);
                }
                other => {
                    return Err(format!(
                        "expected NonBlockGradientPayload from worker {}, got {other:?}",
                        worker.worker_id
                    ));
                }
            }
        }

        let avg_gradients = average_gradients(&all_grads);
        let allreduce_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(NonBlockAllReduceResult { component: expected_component, avg_gradients, allreduce_ms })
    }

    /// Broadcast averaged non-block gradient to all workers.
    pub fn broadcast_averaged_non_block(
        &mut self,
        step: u64,
        result: &NonBlockAllReduceResult,
    ) -> Result<(), String> {
        let msg = WireMessage::AveragedNonBlockGradient {
            step,
            component: result.component,
            gradients: result.avg_gradients.clone(),
        };
        for worker in &self.workers {
            send_wire_message(&worker.stream, &msg)?;
        }
        Ok(())
    }
}

// ─── TCP IO helpers ──────────────────────────────────────────────────────────

/// Read a length-prefixed wire message from a TCP stream.
pub(crate) fn read_wire_message(stream: &TcpStream) -> Result<WireMessage, String> {
    let mut len_buf = [0u8; 4];
    (&*stream).read_exact(&mut len_buf).map_err(|e| format!("read length failed: {e}"))?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if len > 100_000_000 {
        return Err(format!("message too large: {len} bytes"));
    }

    let mut payload = vec![0u8; len];
    (&*stream).read_exact(&mut payload).map_err(|e| format!("read payload failed: {e}"))?;

    WireMessage::from_payload(&payload)
}

/// Send a wire message to a TCP stream.
pub(crate) fn send_wire_message(stream: &TcpStream, msg: &WireMessage) -> Result<(), String> {
    let bytes = msg.to_bytes();
    (&*stream).write_all(&bytes).map_err(|e| format!("send failed: {e}"))?;
    (&*stream).flush().map_err(|e| format!("flush failed: {e}"))?;
    Ok(())
}

impl GradientServer {
    /// Get the local address this server is listening on.
    ///
    /// Useful when binding to port 0 (OS-assigned) in tests.
    #[must_use]
    pub fn local_addr(&self) -> std::net::SocketAddr {
        self.listener.local_addr().expect("listener has local addr")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::net::TcpStream;
    use std::thread;

    #[test]
    fn test_server_bind() {
        // Bind to random port
        let config = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let server = GradientServer::bind(config);
        assert!(server.is_ok());
    }

    #[test]
    fn test_server_worker_count_initially_zero() {
        let config = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let server = GradientServer::bind(config).expect("valid");
        assert_eq!(server.worker_count(), 0);
    }

    #[test]
    fn test_server_accept_worker() {
        let config = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let mut server = GradientServer::bind(config).expect("valid");
        let addr = server.local_addr();

        // Spawn a worker that sends JoinRequest
        let handle = thread::spawn(move || {
            let stream = TcpStream::connect(addr).expect("valid");
            let join = WireMessage::JoinRequest {
                node_id: "test-worker".to_string(),
                gpu_count: 1,
                backend: "cpu".to_string(),
            };
            send_wire_message(&stream, &join).expect("valid");

            // Read JoinAccepted
            let response = read_wire_message(&stream).expect("valid");
            match response {
                WireMessage::JoinAccepted { worker_id, total_workers } => {
                    assert_eq!(worker_id, 0);
                    assert_eq!(total_workers, 1);
                }
                other => panic!("expected JoinAccepted, got {other:?}"),
            }
            stream
        });

        server.wait_for_workers().expect("valid");
        assert_eq!(server.worker_count(), 1);

        let _stream = handle.join().expect("valid");
    }

    #[test]
    fn test_server_shard_and_reduce() {
        let config = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 2);
        let mut server = GradientServer::bind(config).expect("valid");
        let addr = server.local_addr();

        // Spawn 2 workers
        let handles: Vec<_> = (0..2)
            .map(|i| {
                thread::spawn(move || {
                    let stream = TcpStream::connect(addr).expect("valid");
                    let join = WireMessage::JoinRequest {
                        node_id: format!("worker-{i}"),
                        gpu_count: 1,
                        backend: "cpu".to_string(),
                    };
                    send_wire_message(&stream, &join).expect("valid");
                    let _ = read_wire_message(&stream).expect("valid"); // JoinAccepted

                    // Read shard assignment
                    let shard_msg = read_wire_message(&stream).expect("valid");
                    let (shard_start, shard_end) = match shard_msg {
                        WireMessage::ShardAssignment { shard_start, shard_end, .. } => {
                            (shard_start, shard_end)
                        }
                        other => panic!("expected ShardAssignment, got {other:?}"),
                    };

                    // Send gradient
                    let grad = WireMessage::GradientPayload {
                        step: 0,
                        worker_id: i,
                        gradients: vec![1.0 + i as f32, 2.0 + i as f32],
                        loss: 0.5 + i as f32 * 0.1,
                        correct: shard_end - shard_start,
                        total: shard_end - shard_start,
                    };
                    send_wire_message(&stream, &grad).expect("valid");

                    // Read averaged gradient
                    let avg_msg = read_wire_message(&stream).expect("valid");
                    match avg_msg {
                        WireMessage::AveragedGradient { gradients, .. } => {
                            // Average of [1,2] and [2,3] should be [1.5, 2.5]
                            assert!((gradients[0] - 1.5).abs() < 1e-5);
                            assert!((gradients[1] - 2.5).abs() < 1e-5);
                        }
                        other => panic!("expected AveragedGradient, got {other:?}"),
                    }

                    stream
                })
            })
            .collect();

        // Server flow
        server.wait_for_workers().expect("valid");
        server.set_total_samples(100);
        server.send_shard_assignments(0).expect("valid");
        let result = server.collect_and_reduce(0).expect("valid");

        assert!((result.avg_gradients[0] - 1.5).abs() < 1e-5);
        assert!((result.avg_gradients[1] - 2.5).abs() < 1e-5);
        assert_eq!(result.total_samples, 100);
        assert!(result.allreduce_ms >= 0.0);

        server.broadcast_averaged(0, &result).expect("valid");

        for h in handles {
            let _stream = h.join().expect("valid");
        }
    }

    #[test]
    fn test_server_jidoka_halt_on_nan() {
        let config = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
        let mut server = GradientServer::bind(config).expect("valid");
        let addr = server.local_addr();

        let handle = thread::spawn(move || {
            let stream = TcpStream::connect(addr).expect("valid");
            let join = WireMessage::JoinRequest {
                node_id: "bad-worker".to_string(),
                gpu_count: 1,
                backend: "cpu".to_string(),
            };
            send_wire_message(&stream, &join).expect("valid");
            let _ = read_wire_message(&stream).expect("valid");

            // Read shard
            let _ = read_wire_message(&stream).expect("valid");

            // Send NaN gradient
            let grad = WireMessage::GradientPayload {
                step: 0,
                worker_id: 0,
                gradients: vec![1.0, f32::NAN, 3.0],
                loss: 0.5,
                correct: 5,
                total: 10,
            };
            send_wire_message(&stream, &grad).expect("valid");
            stream
        });

        server.wait_for_workers().expect("valid");
        server.set_total_samples(10);
        server.send_shard_assignments(0).expect("valid");
        let result = server.collect_and_reduce(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JIDOKA HALT"));

        let _stream = handle.join().expect("valid");
    }
}
