//! Remote worker daemon for repartir.
//!
//! This binary listens for tasks on a TCP port and executes them using
//! the CPU executor.

use repartir::error::Result;
use repartir::executor::cpu::CpuExecutor;
use repartir::executor::Executor;
use repartir::task::{ExecutionResult, Task};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
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

/// Handles a single client connection.
async fn handle_client(mut stream: TcpStream, executor: Arc<CpuExecutor>) -> Result<()> {
    let peer_addr = stream.peer_addr().map_err(|e| {
        error!("Failed to get peer address: {e}");
        repartir::error::RepartirError::Io(e)
    })?;

    info!("New connection from {peer_addr}");

    loop {
        // Read message length
        let mut len_bytes = [0u8; 4];
        match stream.read_exact(&mut len_bytes).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                info!("Client {peer_addr} disconnected");
                break;
            }
            Err(e) => {
                error!("Failed to read message length from {peer_addr}: {e}");
                return Err(repartir::error::RepartirError::Io(e));
            }
        }

        let len = u32::from_le_bytes(len_bytes) as usize;

        // Sanity check on message size (max 10MB)
        if len > 10 * 1024 * 1024 {
            warn!("Message too large from {peer_addr}: {len} bytes");
            return Err(repartir::error::RepartirError::InvalidTask {
                reason: format!("Message too large: {len} bytes"),
            });
        }

        // Read message payload
        let mut buffer = vec![0u8; len];
        stream.read_exact(&mut buffer).await.map_err(|e| {
            error!("Failed to read message payload from {peer_addr}: {e}");
            repartir::error::RepartirError::Io(e)
        })?;

        // Deserialize message
        let message: RemoteMessage = bincode::deserialize(&buffer).map_err(|e| {
            error!("Failed to deserialize message from {peer_addr}: {e}");
            repartir::error::RepartirError::InvalidTask {
                reason: format!("Deserialization failed: {e}"),
            }
        })?;

        match message {
            RemoteMessage::SubmitTask(task) => {
                let task_id = task.id();
                debug!("Received task {task_id} from {peer_addr}");

                // Execute task
                let result = executor.execute(task).await?;

                debug!("Task {task_id} completed, sending result to {peer_addr}");

                // Send result back
                let response = RemoteMessage::TaskResult(result);
                let encoded = bincode::serialize(&response).map_err(|e| {
                    error!("Failed to serialize result: {e}");
                    repartir::error::RepartirError::InvalidTask {
                        reason: format!("Serialization failed: {e}"),
                    }
                })?;

                // Write length + payload
                let len = u32::try_from(encoded.len()).map_err(|_| {
                    repartir::error::RepartirError::InvalidTask {
                        reason: "Result too large".to_string(),
                    }
                })?;

                stream.write_all(&len.to_le_bytes()).await.map_err(|e| {
                    error!("Failed to write result length to {peer_addr}: {e}");
                    repartir::error::RepartirError::Io(e)
                })?;

                stream.write_all(&encoded).await.map_err(|e| {
                    error!("Failed to write result payload to {peer_addr}: {e}");
                    repartir::error::RepartirError::Io(e)
                })?;

                stream.flush().await.map_err(|e| {
                    error!("Failed to flush stream to {peer_addr}: {e}");
                    repartir::error::RepartirError::Io(e)
                })?;

                debug!("Result sent to {peer_addr}");
            }
            RemoteMessage::Heartbeat => {
                debug!("Heartbeat from {peer_addr}");
                // Respond with heartbeat
                let response = RemoteMessage::Heartbeat;
                let encoded = bincode::serialize(&response).map_err(|e| {
                    repartir::error::RepartirError::InvalidTask {
                        reason: format!("Serialization failed: {e}"),
                    }
                })?;

                let len = u32::try_from(encoded.len()).map_err(|_| {
                    repartir::error::RepartirError::InvalidTask {
                        reason: "Heartbeat message too large".to_string(),
                    }
                })?;

                stream.write_all(&len.to_le_bytes()).await?;
                stream.write_all(&encoded).await?;
                stream.flush().await?;
            }
            RemoteMessage::Shutdown => {
                info!("Shutdown signal from {peer_addr}");
                break;
            }
            RemoteMessage::TaskResult(_) => {
                warn!("Unexpected message type from {peer_addr}");
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "--version" | "-V" => {
                println!("repartir-worker {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" => {
                println!("Repartir Remote Worker — distributed task execution daemon");
                println!();
                println!("Usage: repartir-worker [BIND_ADDR]");
                println!();
                println!("Arguments:");
                println!("  [BIND_ADDR]  Address to listen on [default: 0.0.0.0:9000]");
                println!();
                println!("Options:");
                println!("  -h, --help     Print help");
                println!("  -V, --version  Print version");
                return Ok(());
            }
            _ => {}
        }
    }
    let bind_addr = if args.len() > 1 { &args[1] } else { "0.0.0.0:9000" };

    info!("Repartir Remote Worker v1.1");
    info!("Binding to {bind_addr}");

    // Create CPU executor
    let executor = Arc::new(CpuExecutor::new());
    info!("CPU executor initialized with {} cores", executor.capacity());

    // Listen for connections
    let listener = TcpListener::bind(bind_addr).await.map_err(|e| {
        error!("Failed to bind to {bind_addr}: {e}");
        repartir::error::RepartirError::Io(e)
    })?;

    info!("Worker ready and listening on {bind_addr}");

    // Accept connections
    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                info!("Accepted connection from {addr}");
                let executor = Arc::clone(&executor);

                // Spawn task to handle this client
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, executor).await {
                        error!("Error handling client {addr}: {e}");
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept connection: {e}");
            }
        }
    }
}
