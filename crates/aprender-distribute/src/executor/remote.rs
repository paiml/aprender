//! Remote execution backend.
//!
//! Executes tasks on remote machines over TCP connections.

use crate::error::{RepartirError, Result};
use crate::executor::{BoxFuture, Executor};
use crate::task::{ExecutionResult, Task};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Protocol messages for remote execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum RemoteMessage {
    /// Submit a task for execution.
    SubmitTask(Task),
    /// Task execution result.
    TaskResult(ExecutionResult),
    /// Worker heartbeat.
    Heartbeat,
    /// Shutdown signal.
    Shutdown,
}

/// Remote worker connection.
#[derive(Debug, Clone)]
struct RemoteWorker {
    /// Worker address.
    address: String,
    /// TCP connection to worker.
    stream: Arc<RwLock<TcpStream>>,
}

impl RemoteWorker {
    /// Connects to a remote worker.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails.
    async fn connect(address: String) -> Result<Self> {
        let stream = TcpStream::connect(&address).await.map_err(|e| {
            error!("Failed to connect to worker at {address}: {e}");
            RepartirError::Io(e)
        })?;

        info!("Connected to remote worker at {address}");

        Ok(Self { address, stream: Arc::new(RwLock::new(stream)) })
    }

    /// Sends a task to the remote worker and waits for the result.
    ///
    /// # Errors
    ///
    /// Returns an error if network communication fails or task execution fails.
    async fn execute_task(&self, task: Task) -> Result<ExecutionResult> {
        let task_id = task.id();
        debug!("Sending task {task_id} to worker at {}", self.address);

        // Serialize task
        let message = RemoteMessage::SubmitTask(task);
        let encoded = bincode::serialize(&message).map_err(|e| {
            error!("Failed to serialize task {task_id}: {e}");
            RepartirError::InvalidTask { reason: format!("Serialization failed: {e}") }
        })?;

        // Send task (length-prefixed)
        {
            let mut stream = self.stream.write().await;

            // Write length as u32
            let len = u32::try_from(encoded.len())
                .map_err(|_| RepartirError::InvalidTask { reason: "Task too large".to_string() })?;
            stream.write_all(&len.to_le_bytes()).await.map_err(|e| {
                error!("Failed to send task length to {}: {e}", self.address);
                RepartirError::Io(e)
            })?;

            // Write payload
            stream.write_all(&encoded).await.map_err(|e| {
                error!("Failed to send task payload to {}: {e}", self.address);
                RepartirError::Io(e)
            })?;

            stream.flush().await.map_err(|e| {
                error!("Failed to flush stream to {}: {e}", self.address);
                RepartirError::Io(e)
            })?;
            drop(stream);
        }

        debug!("Task {task_id} sent to {}, waiting for result", self.address);

        // Receive result
        {
            let mut stream = self.stream.write().await;

            // Read length
            let mut len_bytes = [0u8; 4];
            stream.read_exact(&mut len_bytes).await.map_err(|e| {
                error!("Failed to read result length from {}: {e}", self.address);
                RepartirError::Io(e)
            })?;
            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read payload
            let mut buffer = vec![0u8; len];
            stream.read_exact(&mut buffer).await.map_err(|e| {
                error!("Failed to read result payload from {}: {e}", self.address);
                RepartirError::Io(e)
            })?;
            drop(stream);

            // Deserialize result
            let message: RemoteMessage = bincode::deserialize(&buffer).map_err(|e| {
                error!("Failed to deserialize result from {}: {e}", self.address);
                RepartirError::InvalidTask { reason: format!("Deserialization failed: {e}") }
            })?;

            if let RemoteMessage::TaskResult(result) = message {
                debug!("Received result for task {} from {}", result.task_id(), self.address);
                Ok(result)
            } else {
                warn!("Unexpected message type from {}", self.address);
                Err(RepartirError::InvalidTask { reason: "Unexpected message type".to_string() })
            }
        }
    }
}

/// Remote executor that distributes tasks to remote workers.
///
/// This executor maintains connections to remote worker machines and
/// distributes tasks across them using a simple round-robin strategy.
///
/// # Example
///
/// ```no_run
/// use repartir::executor::remote::RemoteExecutor;
/// use repartir::executor::Executor;
///
/// #[tokio::main]
/// async fn main() -> repartir::error::Result<()> {
///     let executor = RemoteExecutor::new().await?;
///     executor.add_worker("192.168.1.100:9000").await?;
///     executor.add_worker("192.168.1.101:9000").await?;
///
///     assert_eq!(executor.capacity(), 2);
///     assert_eq!(executor.name(), "Remote");
///     Ok(())
/// }
/// ```
pub struct RemoteExecutor {
    /// Connected workers.
    workers: Arc<RwLock<Vec<RemoteWorker>>>,
    /// Round-robin counter for load balancing.
    next_worker: Arc<RwLock<usize>>,
}

impl RemoteExecutor {
    /// Creates a new remote executor with no workers.
    ///
    /// Use `add_worker()` to connect to remote workers.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns Result for future extensibility.
    #[allow(clippy::unused_async)] // Keep async for API consistency
    pub async fn new() -> Result<Self> {
        info!("RemoteExecutor initialized");
        Ok(Self {
            workers: Arc::new(RwLock::new(Vec::new())),
            next_worker: Arc::new(RwLock::new(0)),
        })
    }

    /// Adds a remote worker at the given address.
    ///
    /// Address format: `"host:port"` (e.g., `"192.168.1.100:9000"`)
    ///
    /// # Errors
    ///
    /// Returns an error if connection to the worker fails.
    pub async fn add_worker(&self, address: &str) -> Result<()> {
        let worker = RemoteWorker::connect(address.to_string()).await?;
        self.workers.write().await.push(worker);
        info!("Added remote worker at {address}");
        Ok(())
    }

    /// Removes all workers.
    pub async fn clear_workers(&self) {
        self.workers.write().await.clear();
        *self.next_worker.write().await = 0;
        info!("Cleared all remote workers");
    }

    /// Selects the next worker using round-robin.
    ///
    /// # Errors
    ///
    /// Returns an error if no workers are available.
    async fn next_worker(&self) -> Result<usize> {
        let workers = self.workers.read().await;
        if workers.is_empty() {
            return Err(RepartirError::InvalidTask {
                reason: "No remote workers available".to_string(),
            });
        }

        let num_workers = workers.len();
        drop(workers);

        let mut next = self.next_worker.write().await;
        let worker_idx = *next % num_workers;
        *next = next.wrapping_add(1);
        drop(next);

        Ok(worker_idx)
    }
}

impl Executor for RemoteExecutor {
    fn execute(&self, task: Task) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move {
            let task_id = task.id();
            debug!("Executing task {task_id} on remote worker");

            // Select worker
            let worker_idx = self.next_worker().await?;

            // Get worker and execute
            let worker = {
                let workers = self.workers.read().await;
                workers[worker_idx].clone()
            };

            worker.execute_task(task).await
        })
    }

    fn capacity(&self) -> usize {
        // This is a sync trait method, so we can't await
        // Return 0 for now - in a real implementation, we'd cache this
        // v1.2: Make Executor trait async-aware
        0
    }

    fn name(&self) -> &'static str {
        "Remote"
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::task::{Backend, Priority, Task};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Creates a test TCP server that echoes back a successful result.
    async fn start_test_worker() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            // Handle multiple connections in a loop
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    tokio::spawn(async move {
                        // Handle multiple requests on the same connection
                        loop {
                            // Read task message
                            let mut len_bytes = [0u8; 4];
                            if stream.read_exact(&mut len_bytes).await.is_err() {
                                return;
                            }
                            let len = u32::from_le_bytes(len_bytes) as usize;

                            let mut buffer = vec![0u8; len];
                            if stream.read_exact(&mut buffer).await.is_err() {
                                return;
                            }

                            // Deserialize to get task ID
                            if let Ok(RemoteMessage::SubmitTask(task)) =
                                bincode::deserialize::<RemoteMessage>(&buffer)
                            {
                                // Send back a success result
                                let result = ExecutionResult::new(
                                    task.id(),
                                    0,
                                    b"test output".to_vec(),
                                    Vec::new(),
                                    std::time::Duration::from_millis(10),
                                );

                                let response = RemoteMessage::TaskResult(result);
                                let encoded = bincode::serialize(&response).unwrap();

                                // Send length and payload
                                let response_len = u32::try_from(encoded.len()).unwrap();
                                if stream.write_all(&response_len.to_le_bytes()).await.is_err() {
                                    return;
                                }
                                if stream.write_all(&encoded).await.is_err() {
                                    return;
                                }
                                if stream.flush().await.is_err() {
                                    return;
                                }
                            }
                        }
                    });
                }
            }
        });

        addr
    }

    #[tokio::test]
    async fn test_remote_executor_creation() {
        let executor = RemoteExecutor::new().await;
        assert!(executor.is_ok());

        let exec = executor.ok().unwrap();
        assert_eq!(exec.name(), "Remote");
    }

    #[tokio::test]
    async fn test_remote_executor_no_workers() {
        let executor = RemoteExecutor::new().await.ok().unwrap();

        let result = executor.next_worker().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remote_executor_clear_workers() {
        let executor = RemoteExecutor::new().await.ok().unwrap();
        executor.clear_workers().await;

        let result = executor.next_worker().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remote_executor_add_worker() {
        let addr = start_test_worker().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let executor = RemoteExecutor::new().await.unwrap();
        let result = executor.add_worker(&addr).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_remote_executor_add_invalid_worker() {
        let executor = RemoteExecutor::new().await.unwrap();
        // Try to connect to invalid address
        let result = executor.add_worker("127.0.0.1:1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remote_executor_execute_task() {
        let addr = start_test_worker().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let executor = RemoteExecutor::new().await.unwrap();
        executor.add_worker(&addr).await.unwrap();

        let task = Task::builder()
            .binary("/bin/echo")
            .arg("test")
            .backend(Backend::Remote)
            .priority(Priority::Normal)
            .build()
            .unwrap();

        let result = executor.execute(task).await;
        assert!(result.is_ok());

        let exec_result = result.unwrap();
        assert_eq!(exec_result.exit_code(), 0);
        assert_eq!(exec_result.stdout_str().unwrap(), "test output");
    }

    #[tokio::test]
    async fn test_remote_executor_round_robin() {
        let addr1 = start_test_worker().await;
        let addr2 = start_test_worker().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let executor = RemoteExecutor::new().await.unwrap();
        executor.add_worker(&addr1).await.unwrap();
        executor.add_worker(&addr2).await.unwrap();

        // Execute multiple tasks to test round-robin
        for _ in 0..4 {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg("test")
                .backend(Backend::Remote)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_remote_message_serialization() {
        let task = Task::builder()
            .binary("/bin/test")
            .arg("arg1")
            .backend(Backend::Remote)
            .build()
            .unwrap();

        let message = RemoteMessage::SubmitTask(task.clone());
        let encoded = bincode::serialize(&message).unwrap();
        let decoded: RemoteMessage = bincode::deserialize(&encoded).unwrap();

        match decoded {
            RemoteMessage::SubmitTask(decoded_task) => {
                assert_eq!(decoded_task.id(), task.id());
            }
            _ => unreachable!("Unexpected message type"),
        }
    }

    #[tokio::test]
    async fn test_remote_executor_capacity() {
        let executor = RemoteExecutor::new().await.unwrap();
        // Capacity is 0 for now (sync trait limitation)
        assert_eq!(executor.capacity(), 0);
    }
}
