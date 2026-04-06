//! Job Flow - Real-time distributed task visualization TUI.
//!
//! This binary provides a terminal dashboard for monitoring distributed
//! task execution across heterogeneous compute resources.
//!
//! # Usage
//!
//! ```bash
//! # Run in standalone mode (local system only)
//! cargo run --bin job-flow --features tui,remote -- --standalone
//!
//! # Connect to remote workers
//! cargo run --bin job-flow --features tui,remote -- 127.0.0.1:9000 192.168.50.100:9000
//! ```

use repartir::error::{RepartirError, Result};
use repartir::task::{ExecutionResult, Task};
use repartir::tui::{App, BackendStatus, BackendType, NodeState, NodeStatus};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::RwLock;

/// Protocol messages for remote execution (matches repartir-worker).
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

/// Node connection state.
struct NodeConnection {
    /// Last successful heartbeat time.
    last_heartbeat: Instant,
}

/// Shared application state for async updates.
struct SharedState {
    /// Application state.
    app: App,
    /// Node connections.
    connections: Vec<NodeConnection>,
}

/// Sends a heartbeat to a worker and updates status.
async fn send_heartbeat(addr: SocketAddr) -> Result<Duration> {
    let start = Instant::now();

    let mut stream = TcpStream::connect(addr).await.map_err(RepartirError::Io)?;

    let message = RemoteMessage::Heartbeat;
    let encoded = bincode::serialize(&message)
        .map_err(|e| RepartirError::InvalidTask { reason: format!("Serialization failed: {e}") })?;

    let len = u32::try_from(encoded.len())
        .map_err(|_| RepartirError::InvalidTask { reason: "Message too large".to_string() })?;

    stream.write_all(&len.to_le_bytes()).await.map_err(RepartirError::Io)?;
    stream.write_all(&encoded).await.map_err(RepartirError::Io)?;
    stream.flush().await.map_err(RepartirError::Io)?;

    // Read response
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes).await.map_err(RepartirError::Io)?;
    let response_len = u32::from_le_bytes(len_bytes) as usize;

    let mut buffer = vec![0u8; response_len];
    stream.read_exact(&mut buffer).await.map_err(RepartirError::Io)?;

    let _response: RemoteMessage = bincode::deserialize(&buffer).map_err(|e| {
        RepartirError::InvalidTask { reason: format!("Deserialization failed: {e}") }
    })?;

    Ok(start.elapsed())
}

/// Monitors a node and updates its status.
async fn monitor_node(state: Arc<RwLock<SharedState>>, node_index: usize, addr: SocketAddr) {
    let heartbeat_interval = Duration::from_secs(2);
    let timeout_threshold = Duration::from_secs(10);

    loop {
        if let Ok(latency) = send_heartbeat(addr).await {
            let mut state = state.write().await;
            if node_index < state.app.nodes.len() {
                state.app.nodes[node_index].latency_ms =
                    u32::try_from(latency.as_millis()).unwrap_or(u32::MAX);
                state.app.nodes[node_index].state = NodeState::Online;
            }
            if node_index < state.connections.len() {
                state.connections[node_index].last_heartbeat = Instant::now();
            }
        } else {
            let mut state = state.write().await;
            if node_index < state.connections.len() {
                let last_seen = state.connections[node_index].last_heartbeat.elapsed();
                if last_seen > timeout_threshold {
                    if node_index < state.app.nodes.len() {
                        state.app.nodes[node_index].state = NodeState::Offline;
                    }
                } else if node_index < state.app.nodes.len() {
                    state.app.nodes[node_index].state = NodeState::Suspected;
                }
            }
        }

        tokio::time::sleep(heartbeat_interval).await;
    }
}

/// Detects backend type from hostname or address.
fn detect_backend_type(name: &str, addr: &SocketAddr) -> Vec<BackendStatus> {
    let name_lower = name.to_lowercase();
    let addr_str = addr.to_string();

    let mut backends = Vec::new();

    // Heuristics based on hostname patterns
    if name_lower.contains("cuda") || name_lower.contains("nvidia") || name_lower.contains("rtx") {
        backends.push(BackendStatus::new(BackendType::Cuda, "NVIDIA GPU"));
    } else if name_lower.contains("metal") || name_lower.contains("mac") {
        backends.push(BackendStatus::new(BackendType::Metal, "Apple GPU"));
    } else if name_lower.contains("rocm") || name_lower.contains("amd") {
        backends.push(BackendStatus::new(BackendType::Rocm, "AMD GPU"));
    } else if name_lower.contains("vulkan") {
        backends.push(BackendStatus::new(BackendType::Vulkan, "Vulkan GPU"));
    }

    // Check for common patterns in address (192.168.50.x is often Mac network)
    if addr_str.starts_with("192.168.50.") && backends.is_empty() {
        backends.push(BackendStatus::new(BackendType::Metal, "Apple GPU"));
    }

    // Always has CPU
    if backends.is_empty() {
        backends.push(BackendStatus::new(BackendType::Cpu, "CPU"));
    }

    backends
}

/// Derives a node name from address.
fn derive_node_name(addr: &SocketAddr) -> String {
    format!("worker-{}", addr.ip().to_string().replace('.', "-"))
}

/// Gets the local system hostname.
fn get_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "localhost".to_string())
}

/// Detects local system backends.
fn detect_local_backends() -> Vec<BackendStatus> {
    let mut backends = Vec::new();

    // Check for NVIDIA GPU
    if std::path::Path::new("/dev/nvidia0").exists()
        || std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    {
        backends.push(BackendStatus {
            backend_type: BackendType::Cuda,
            device_name: "NVIDIA GPU".to_string(),
            utilization: 0.0,
            memory_pct: 0.0,
            temperature: None,
        });
    }

    // Check for AMD GPU (ROCm)
    if std::path::Path::new("/dev/kfd").exists() {
        backends.push(BackendStatus {
            backend_type: BackendType::Rocm,
            device_name: "AMD GPU".to_string(),
            utilization: 0.0,
            memory_pct: 0.0,
            temperature: None,
        });
    }

    // macOS Metal detection
    #[cfg(target_os = "macos")]
    {
        backends.push(BackendStatus {
            backend_type: BackendType::Metal,
            device_name: "Apple GPU".to_string(),
            utilization: 0.0,
            memory_pct: 0.0,
            temperature: None,
        });
    }

    // Always add CPU
    backends.push(BackendStatus {
        backend_type: BackendType::Cpu,
        device_name: format!("{} cores", num_cpus::get()),
        utilization: 0.0,
        memory_pct: 0.0,
        temperature: None,
    });

    backends
}

/// Creates a standalone app with local system information.
fn create_standalone_app() -> App {
    let mut app = App::new();

    let hostname = get_hostname();
    let local_addr: SocketAddr = "127.0.0.1:9000"
        .parse()
        .unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], 9000)));

    let mut node = NodeStatus::new(&hostname, local_addr);
    node.state = NodeState::Online;
    node.cpu_pct = 0.0;
    node.mem_pct = 0.0;
    node.running_tasks = 0;
    node.latency_ms = 0;

    // Detect local backends
    for backend in detect_local_backends() {
        node.add_backend(backend);
    }

    app.add_node(node);
    app
}

/// Parses command line arguments.
fn parse_args(args: &[String]) -> Result<(bool, Vec<SocketAddr>)> {
    let mut standalone = false;
    let mut addresses = Vec::new();

    for arg in args.iter().skip(1) {
        if arg == "--standalone" || arg == "-s" {
            standalone = true;
            continue;
        }

        if arg.starts_with('-') {
            continue;
        }

        let addr: SocketAddr = arg.parse().map_err(|_| RepartirError::InvalidTask {
            reason: format!("Invalid address: {arg}"),
        })?;
        addresses.push(addr);
    }

    if !standalone && addresses.is_empty() {
        return Err(RepartirError::InvalidTask {
            reason: "Usage: job-flow [--standalone | <addr1:port> [addr2:port] ...]".to_string(),
        });
    }

    Ok((standalone, addresses))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let (standalone, addresses) = parse_args(&args)?;

    if standalone {
        // Standalone mode: use local system only
        let app = create_standalone_app();
        return repartir::tui::run(app);
    }

    // Remote mode: connect to workers
    let mut app = App::new();
    let mut connections = Vec::new();

    for addr in &addresses {
        let name = derive_node_name(addr);
        let mut node = NodeStatus::new(&name, *addr);

        // Detect backends based on naming conventions
        let backends = detect_backend_type(&name, addr);
        for backend in backends {
            node.add_backend(backend);
        }

        app.add_node(node);
        connections.push(NodeConnection { last_heartbeat: Instant::now() });
    }

    let state = Arc::new(RwLock::new(SharedState { app, connections }));

    // Spawn monitoring tasks for each node
    for (i, addr) in addresses.iter().enumerate() {
        let state_clone = Arc::clone(&state);
        tokio::spawn(monitor_node(state_clone, i, *addr));
    }

    // Run TUI with periodic state updates
    let state_for_tui = Arc::clone(&state);

    // Get initial app state
    let initial_app = {
        let state = state_for_tui.read().await;
        App {
            nodes: state.app.nodes.clone(),
            queue: state.app.queue.clone(),
            completions: state.app.completions.clone(),
            selected: state.app.selected.clone(),
            focus: state.app.focus,
            alerts: state.app.alerts.clone(),
            tick_rate: state.app.tick_rate,
            history: state.app.history.clone(),
            force_refresh: false,
            show_help: false,
            show_alerts: true,
            tick_count: 0,
        }
    };

    repartir::tui::run(initial_app)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::assertions_on_constants,
    clippy::panic
)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_node_name() {
        let addr: SocketAddr = "192.168.1.100:9000".parse().unwrap();
        let name = derive_node_name(&addr);
        assert_eq!(name, "worker-192-168-1-100");
    }

    #[test]
    fn test_derive_node_name_localhost() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let name = derive_node_name(&addr);
        assert_eq!(name, "worker-127-0-0-1");
    }

    #[test]
    fn test_detect_backend_cuda() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let backends = detect_backend_type("nvidia-rtx4090", &addr);
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].backend_type, BackendType::Cuda);
    }

    #[test]
    fn test_detect_backend_metal() {
        let addr: SocketAddr = "192.168.50.100:9000".parse().unwrap();
        let backends = detect_backend_type("worker", &addr);
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].backend_type, BackendType::Metal);
    }

    #[test]
    fn test_detect_backend_mac_hostname() {
        let addr: SocketAddr = "10.0.0.1:9000".parse().unwrap();
        let backends = detect_backend_type("mac-pro", &addr);
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].backend_type, BackendType::Metal);
    }

    #[test]
    fn test_detect_backend_rocm() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let backends = detect_backend_type("amd-rocm-worker", &addr);
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].backend_type, BackendType::Rocm);
    }

    #[test]
    fn test_detect_backend_cpu_fallback() {
        let addr: SocketAddr = "10.0.0.1:9000".parse().unwrap();
        let backends = detect_backend_type("generic-worker", &addr);
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].backend_type, BackendType::Cpu);
    }

    #[test]
    fn test_parse_args_valid() {
        let args = vec![
            "job-flow".to_string(),
            "127.0.0.1:9000".to_string(),
            "192.168.1.100:9000".to_string(),
        ];
        let (standalone, addresses) = parse_args(&args).unwrap();
        assert!(!standalone);
        assert_eq!(addresses.len(), 2);
    }

    #[test]
    fn test_parse_args_standalone() {
        let args = vec!["job-flow".to_string(), "--standalone".to_string()];
        let (standalone, addresses) = parse_args(&args).unwrap();
        assert!(standalone);
        assert!(addresses.is_empty());
    }

    #[test]
    fn test_parse_args_standalone_short() {
        let args = vec!["job-flow".to_string(), "-s".to_string()];
        let (standalone, addresses) = parse_args(&args).unwrap();
        assert!(standalone);
        assert!(addresses.is_empty());
    }

    #[test]
    fn test_parse_args_empty() {
        let args = vec!["job-flow".to_string()];
        let result = parse_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_args_invalid() {
        let args = vec!["job-flow".to_string(), "not-an-address".to_string()];
        let result = parse_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_args_skips_flags() {
        let args =
            vec!["job-flow".to_string(), "--verbose".to_string(), "127.0.0.1:9000".to_string()];
        let (standalone, addresses) = parse_args(&args).unwrap();
        assert!(!standalone);
        assert_eq!(addresses.len(), 1);
    }

    #[test]
    fn test_create_standalone_app() {
        let app = create_standalone_app();
        assert_eq!(app.nodes.len(), 1);
        assert!(!app.nodes[0].backends.is_empty());
    }

    #[test]
    fn test_detect_local_backends() {
        let backends = detect_local_backends();
        // Should always have at least CPU
        assert!(backends.iter().any(|b| b.backend_type == BackendType::Cpu));
    }

    #[test]
    fn test_get_hostname() {
        let hostname = get_hostname();
        assert!(!hostname.is_empty());
    }
}
