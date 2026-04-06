//! Containers Analyzer
//!
//! Queries Docker and Podman for container statistics.
//! Uses Unix domain sockets to communicate with the container runtime API.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use super::{Analyzer, AnalyzerError};

/// Container runtime type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerRuntime {
    Docker,
    Podman,
}

impl ContainerRuntime {
    /// Get the socket path for this runtime
    pub fn socket_path(&self) -> &'static str {
        match self {
            Self::Docker => "/var/run/docker.sock",
            Self::Podman => "/run/podman/podman.sock",
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Docker => "Docker",
            Self::Podman => "Podman",
        }
    }
}

/// Container state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerState {
    Running,
    Paused,
    Exited,
    Created,
    Restarting,
    Removing,
    Dead,
    Unknown,
}

impl ContainerState {
    /// Parse from API string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "running" => Self::Running,
            "paused" => Self::Paused,
            "exited" => Self::Exited,
            "created" => Self::Created,
            "restarting" => Self::Restarting,
            "removing" => Self::Removing,
            "dead" => Self::Dead,
            _ => Self::Unknown,
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "Running",
            Self::Paused => "Paused",
            Self::Exited => "Exited",
            Self::Created => "Created",
            Self::Restarting => "Restarting",
            Self::Removing => "Removing",
            Self::Dead => "Dead",
            Self::Unknown => "Unknown",
        }
    }

    /// Short form for display
    pub fn short(&self) -> &'static str {
        match self {
            Self::Running => "UP",
            Self::Paused => "PAUSE",
            Self::Exited => "EXIT",
            Self::Created => "NEW",
            Self::Restarting => "RSTR",
            Self::Removing => "DEL",
            Self::Dead => "DEAD",
            Self::Unknown => "?",
        }
    }
}

/// Container resource usage statistics
#[derive(Debug, Clone, Default)]
pub struct ContainerStats {
    /// CPU usage percentage (0-100)
    pub cpu_percent: f32,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Memory limit in bytes
    pub memory_limit: u64,
    /// Memory usage percentage
    pub memory_percent: f32,
    /// Network RX bytes
    pub net_rx_bytes: u64,
    /// Network TX bytes
    pub net_tx_bytes: u64,
    /// Block I/O read bytes
    pub block_read_bytes: u64,
    /// Block I/O write bytes
    pub block_write_bytes: u64,
    /// Number of PIDs
    pub pids: u32,
}

/// A single container
#[derive(Debug, Clone)]
pub struct Container {
    /// Container ID (short form)
    pub id: String,
    /// Container name
    pub name: String,
    /// Image name
    pub image: String,
    /// Container state
    pub state: ContainerState,
    /// Status string from API
    pub status: String,
    /// Container runtime
    pub runtime: ContainerRuntime,
    /// Resource usage stats
    pub stats: ContainerStats,
    /// Container creation time (Unix timestamp)
    pub created: i64,
    /// Port mappings (host:container)
    pub ports: Vec<(u16, u16)>,
}

impl Container {
    /// Format name for display (truncate if needed)
    pub fn display_name(&self, max_len: usize) -> String {
        if self.name.len() <= max_len {
            self.name.clone()
        } else {
            format!("{}…", &self.name[..max_len - 1])
        }
    }

    /// Format image for display (remove registry prefix)
    pub fn display_image(&self) -> &str {
        self.image.rsplit('/').next().unwrap_or(&self.image)
    }

    /// Format memory for display
    pub fn display_memory(&self) -> String {
        format_bytes(self.stats.memory_bytes)
    }

    /// Format memory limit for display
    pub fn display_memory_limit(&self) -> String {
        format_bytes(self.stats.memory_limit)
    }
}

/// Containers data
#[derive(Debug, Clone, Default)]
pub struct ContainersData {
    /// All containers
    pub containers: Vec<Container>,
    /// Active runtime
    pub runtime: Option<ContainerRuntime>,
    /// Count by state
    pub state_counts: HashMap<ContainerState, usize>,
    /// Total CPU usage across all containers
    pub total_cpu: f32,
    /// Total memory usage across all containers
    pub total_memory: u64,
}

impl ContainersData {
    /// Get running containers only
    pub fn running(&self) -> impl Iterator<Item = &Container> {
        self.containers
            .iter()
            .filter(|c| c.state == ContainerState::Running)
    }

    /// Total container count
    pub fn total(&self) -> usize {
        self.containers.len()
    }

    /// Running container count
    pub fn running_count(&self) -> usize {
        *self
            .state_counts
            .get(&ContainerState::Running)
            .unwrap_or(&0)
    }
}

/// Analyzer for container stats
pub struct ContainersAnalyzer {
    data: ContainersData,
    interval: Duration,
    runtime: Option<ContainerRuntime>,
}

impl Default for ContainersAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ContainersAnalyzer {
    /// Create a new containers analyzer
    pub fn new() -> Self {
        // Detect available runtime
        let runtime = if Path::new(ContainerRuntime::Docker.socket_path()).exists() {
            Some(ContainerRuntime::Docker)
        } else if Path::new(ContainerRuntime::Podman.socket_path()).exists() {
            Some(ContainerRuntime::Podman)
        } else {
            None
        };

        Self {
            data: ContainersData::default(),
            interval: Duration::from_secs(2),
            runtime,
        }
    }

    /// Get the current containers data
    pub fn data(&self) -> &ContainersData {
        &self.data
    }

    /// Send HTTP request over Unix socket
    fn http_get(&self, path: &str) -> Result<String, AnalyzerError> {
        let Some(runtime) = self.runtime else {
            return Err(AnalyzerError::NotAvailable(
                "No container runtime available".to_string(),
            ));
        };

        let socket_path = runtime.socket_path();
        let mut stream = UnixStream::connect(socket_path)
            .map_err(|e| AnalyzerError::IoError(format!("Socket connect failed: {}", e)))?;

        // Set timeout
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(5))).ok();

        // Send HTTP request
        let request = format!(
            "GET {} HTTP/1.0\r\nHost: localhost\r\nAccept: application/json\r\n\r\n",
            path
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|e| AnalyzerError::IoError(format!("Write failed: {}", e)))?;

        // Read response
        let mut reader = BufReader::new(stream);
        let mut response = String::new();

        // Skip HTTP headers
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if line == "\r\n" {
                        break; // End of headers
                    }
                }
                Err(e) => return Err(AnalyzerError::IoError(format!("Read failed: {}", e))),
            }
        }

        // Read body
        reader
            .read_to_string(&mut response)
            .map_err(|e| AnalyzerError::IoError(format!("Read body failed: {}", e)))?;

        Ok(response)
    }

    /// List containers from API
    fn list_containers(&self) -> Result<Vec<Container>, AnalyzerError> {
        let response = self.http_get("/containers/json?all=true")?;
        self.parse_container_list(&response)
    }

    /// Parse container list JSON (minimal JSON parsing without dependencies)
    fn parse_container_list(&self, json: &str) -> Result<Vec<Container>, AnalyzerError> {
        let runtime = self
            .runtime
            .ok_or_else(|| AnalyzerError::NotAvailable("No runtime".to_string()))?;

        let mut containers = Vec::new();

        // Simple JSON array parsing (containers are in a JSON array)
        // This is a minimal parser - for production, use serde_json
        let json = json.trim();
        if !json.starts_with('[') || !json.ends_with(']') {
            return Ok(containers); // Empty or invalid
        }

        // Split by container objects (look for "Id": pattern)
        for chunk in json.split(r#""Id":"#).skip(1) {
            let container = self.parse_container_object(chunk, runtime);
            if let Some(c) = container {
                containers.push(c);
            }
        }

        Ok(containers)
    }

    /// Parse a single container object from JSON chunk
    fn parse_container_object(&self, chunk: &str, runtime: ContainerRuntime) -> Option<Container> {
        // Extract ID (first quoted string)
        let id = extract_string_after(chunk, "")?;
        let id = if id.len() > 12 {
            id[..12].to_string()
        } else {
            id
        };

        // Extract other fields
        let image = extract_json_string(chunk, "Image")?;
        let state_str = extract_json_string(chunk, "State").unwrap_or_default();
        let status = extract_json_string(chunk, "Status").unwrap_or_default();
        let created = extract_json_number(chunk, "Created").unwrap_or(0);

        // Extract name (from Names array)
        let name = extract_name_from_names(chunk).unwrap_or_else(|| id.clone());

        let state = ContainerState::from_str(&state_str);

        Some(Container {
            id,
            name,
            image,
            state,
            status,
            runtime,
            stats: ContainerStats::default(),
            created,
            ports: Vec::new(),
        })
    }

    /// Get stats for a container
    fn get_container_stats(&self, container_id: &str) -> Option<ContainerStats> {
        // Stats endpoint: /containers/{id}/stats?stream=false
        let path = format!("/containers/{}/stats?stream=false", container_id);
        let response = self.http_get(&path).ok()?;

        self.parse_container_stats(&response)
    }

    /// Parse container stats JSON
    fn parse_container_stats(&self, json: &str) -> Option<ContainerStats> {
        // Extract memory stats
        let memory_bytes = extract_json_number(json, "usage")
            .or_else(|| extract_json_number(json, "rss"))
            .unwrap_or(0) as u64;

        let memory_limit = extract_json_number(json, "limit").unwrap_or(0) as u64;

        // Extract CPU stats
        let cpu_total = extract_json_number(json, "total_usage").unwrap_or(0) as u64;
        let system_cpu = extract_json_number(json, "system_cpu_usage").unwrap_or(1) as u64;
        let percpu_len = json.matches("percpu_usage").count().max(1);

        let cpu_percent = if system_cpu > 0 {
            (cpu_total as f64 / system_cpu as f64 * 100.0 * percpu_len as f64) as f32
        } else {
            0.0
        };

        let memory_percent = if memory_limit > 0 {
            (memory_bytes as f64 / memory_limit as f64 * 100.0) as f32
        } else {
            0.0
        };

        // Extract network I/O
        let net_rx = extract_json_number(json, "rx_bytes").unwrap_or(0) as u64;
        let net_tx = extract_json_number(json, "tx_bytes").unwrap_or(0) as u64;

        // Extract block I/O
        let block_read = extract_json_number(json, "read").unwrap_or(0) as u64;
        let block_write = extract_json_number(json, "write").unwrap_or(0) as u64;

        // Extract PIDs
        let pids = extract_json_number(json, "pids_stats")
            .or_else(|| extract_json_number(json, "current"))
            .unwrap_or(0) as u32;

        Some(ContainerStats {
            cpu_percent,
            memory_bytes,
            memory_limit,
            memory_percent,
            net_rx_bytes: net_rx,
            net_tx_bytes: net_tx,
            block_read_bytes: block_read,
            block_write_bytes: block_write,
            pids,
        })
    }
}

impl Analyzer for ContainersAnalyzer {
    fn name(&self) -> &'static str {
        "containers"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let mut containers = self.list_containers()?;

        // Collect stats for running containers
        for container in &mut containers {
            if container.state == ContainerState::Running {
                if let Some(stats) = self.get_container_stats(&container.id) {
                    container.stats = stats;
                }
            }
        }

        // Calculate aggregates
        let mut state_counts: HashMap<ContainerState, usize> = HashMap::new();
        let mut total_cpu = 0.0_f32;
        let mut total_memory = 0_u64;

        for container in &containers {
            *state_counts.entry(container.state).or_insert(0) += 1;
            if container.state == ContainerState::Running {
                total_cpu += container.stats.cpu_percent;
                total_memory += container.stats.memory_bytes;
            }
        }

        self.data = ContainersData {
            containers,
            runtime: self.runtime,
            state_counts,
            total_cpu,
            total_memory,
        };

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        self.runtime.is_some()
    }
}

// Helper functions for minimal JSON parsing

/// Extract a string value after a given position (first quoted string)
fn extract_string_after(s: &str, _marker: &str) -> Option<String> {
    // Find first quote
    let start = s.find('"')? + 1;
    let rest = &s[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract a JSON string value by key
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":\"", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract a JSON number value by key
fn extract_json_number(json: &str, key: &str) -> Option<i64> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..].trim_start();

    // Read digits
    let mut num_str = String::new();
    for ch in rest.chars() {
        if ch.is_ascii_digit() || ch == '-' {
            num_str.push(ch);
        } else {
            break;
        }
    }

    num_str.parse().ok()
}

/// Extract container name from Names array
fn extract_name_from_names(json: &str) -> Option<String> {
    // Names is typically ["\/name"]
    let pattern = "\"Names\":[";
    let start = json.find(pattern)? + pattern.len();
    let rest = &json[start..];

    // Find first name
    let name_start = rest.find('"')? + 1;
    let name_rest = &rest[name_start..];
    let name_end = name_rest.find('"')?;

    let name = &name_rest[..name_end];
    // Remove leading slash if present
    Some(
        name.trim_start_matches("\\/")
            .trim_start_matches('/')
            .to_string(),
    )
}

/// Format bytes for human-readable display
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_state_parsing() {
        assert_eq!(ContainerState::from_str("running"), ContainerState::Running);
        assert_eq!(ContainerState::from_str("RUNNING"), ContainerState::Running);
        assert_eq!(ContainerState::from_str("exited"), ContainerState::Exited);
        assert_eq!(ContainerState::from_str("xyz"), ContainerState::Unknown);
    }

    #[test]
    fn test_container_state_display() {
        assert_eq!(ContainerState::Running.as_str(), "Running");
        assert_eq!(ContainerState::Running.short(), "UP");
        assert_eq!(ContainerState::Exited.short(), "EXIT");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(1024), "1.0K");
        assert_eq!(format_bytes(1536), "1.5K");
        assert_eq!(format_bytes(1048576), "1.0M");
        assert_eq!(format_bytes(1073741824), "1.0G");
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"Name":"test","Image":"nginx"}"#;
        assert_eq!(extract_json_string(json, "Name"), Some("test".to_string()));
        assert_eq!(
            extract_json_string(json, "Image"),
            Some("nginx".to_string())
        );
        assert_eq!(extract_json_string(json, "Missing"), None);
    }

    #[test]
    fn test_extract_json_number() {
        let json = r#"{"Created":1234567890,"Size":1024}"#;
        assert_eq!(extract_json_number(json, "Created"), Some(1234567890));
        assert_eq!(extract_json_number(json, "Size"), Some(1024));
        assert_eq!(extract_json_number(json, "Missing"), None);
    }

    #[test]
    fn test_container_display_name() {
        let container = Container {
            id: "abc123".to_string(),
            name: "my-very-long-container-name".to_string(),
            image: "registry.example.com/org/nginx:latest".to_string(),
            state: ContainerState::Running,
            status: "Up 5 minutes".to_string(),
            runtime: ContainerRuntime::Docker,
            stats: ContainerStats::default(),
            created: 0,
            ports: vec![],
        };

        assert_eq!(container.display_name(10), "my-very-l…");
        assert_eq!(container.display_image(), "nginx:latest");
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ContainersAnalyzer::new();
        // Just verify it doesn't panic
        let _ = analyzer.available();
    }

    #[test]
    fn test_runtime_socket_path() {
        assert_eq!(
            ContainerRuntime::Docker.socket_path(),
            "/var/run/docker.sock"
        );
        assert_eq!(
            ContainerRuntime::Podman.socket_path(),
            "/run/podman/podman.sock"
        );
    }

    // Additional ContainerRuntime tests
    #[test]
    fn test_container_runtime_as_str() {
        assert_eq!(ContainerRuntime::Docker.as_str(), "Docker");
        assert_eq!(ContainerRuntime::Podman.as_str(), "Podman");
    }

    #[test]
    fn test_container_runtime_debug() {
        let rt = ContainerRuntime::Docker;
        let debug = format!("{:?}", rt);
        assert!(debug.contains("Docker"));
    }

    #[test]
    fn test_container_runtime_clone() {
        let rt = ContainerRuntime::Podman;
        let cloned = rt.clone();
        assert_eq!(rt, cloned);
    }

    #[test]
    fn test_container_runtime_copy() {
        let rt = ContainerRuntime::Docker;
        let copied: ContainerRuntime = rt;
        assert_eq!(copied, ContainerRuntime::Docker);
    }

    // ContainerState tests
    #[test]
    fn test_container_state_paused() {
        assert_eq!(ContainerState::from_str("paused"), ContainerState::Paused);
        assert_eq!(ContainerState::Paused.as_str(), "Paused");
        assert_eq!(ContainerState::Paused.short(), "PAUSE");
    }

    #[test]
    fn test_container_state_created() {
        assert_eq!(ContainerState::from_str("created"), ContainerState::Created);
        assert_eq!(ContainerState::Created.as_str(), "Created");
        assert_eq!(ContainerState::Created.short(), "NEW");
    }

    #[test]
    fn test_container_state_restarting() {
        assert_eq!(
            ContainerState::from_str("restarting"),
            ContainerState::Restarting
        );
        assert_eq!(ContainerState::Restarting.as_str(), "Restarting");
        assert_eq!(ContainerState::Restarting.short(), "RSTR");
    }

    #[test]
    fn test_container_state_removing() {
        assert_eq!(
            ContainerState::from_str("removing"),
            ContainerState::Removing
        );
        assert_eq!(ContainerState::Removing.as_str(), "Removing");
        assert_eq!(ContainerState::Removing.short(), "DEL");
    }

    #[test]
    fn test_container_state_dead() {
        assert_eq!(ContainerState::from_str("dead"), ContainerState::Dead);
        assert_eq!(ContainerState::Dead.as_str(), "Dead");
        assert_eq!(ContainerState::Dead.short(), "DEAD");
    }

    #[test]
    fn test_container_state_unknown() {
        assert_eq!(ContainerState::Unknown.as_str(), "Unknown");
        assert_eq!(ContainerState::Unknown.short(), "?");
    }

    #[test]
    fn test_container_state_debug() {
        let state = ContainerState::Running;
        let debug = format!("{:?}", state);
        assert!(debug.contains("Running"));
    }

    #[test]
    fn test_container_state_clone() {
        let state = ContainerState::Exited;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_container_state_hash() {
        let mut map: HashMap<ContainerState, usize> = HashMap::new();
        map.insert(ContainerState::Running, 5);
        map.insert(ContainerState::Exited, 3);
        assert_eq!(map.get(&ContainerState::Running), Some(&5));
    }

    // ContainerStats tests
    #[test]
    fn test_container_stats_default() {
        let stats = ContainerStats::default();
        assert!((stats.cpu_percent - 0.0).abs() < f32::EPSILON);
        assert_eq!(stats.memory_bytes, 0);
        assert_eq!(stats.memory_limit, 0);
        assert_eq!(stats.net_rx_bytes, 0);
        assert_eq!(stats.net_tx_bytes, 0);
        assert_eq!(stats.block_read_bytes, 0);
        assert_eq!(stats.block_write_bytes, 0);
        assert_eq!(stats.pids, 0);
    }

    #[test]
    fn test_container_stats_debug() {
        let stats = ContainerStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ContainerStats"));
    }

    #[test]
    fn test_container_stats_clone() {
        let stats = ContainerStats {
            cpu_percent: 50.0,
            memory_bytes: 1024,
            memory_limit: 2048,
            memory_percent: 50.0,
            net_rx_bytes: 100,
            net_tx_bytes: 200,
            block_read_bytes: 300,
            block_write_bytes: 400,
            pids: 5,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.cpu_percent, 50.0);
        assert_eq!(cloned.memory_bytes, 1024);
    }

    // Container tests
    #[test]
    fn test_container_display_name_short() {
        let container = Container {
            id: "abc".to_string(),
            name: "short".to_string(),
            image: "nginx".to_string(),
            state: ContainerState::Running,
            status: "Up".to_string(),
            runtime: ContainerRuntime::Docker,
            stats: ContainerStats::default(),
            created: 0,
            ports: vec![],
        };
        assert_eq!(container.display_name(10), "short");
    }

    #[test]
    fn test_container_display_image_no_registry() {
        let container = Container {
            id: "abc".to_string(),
            name: "test".to_string(),
            image: "nginx:latest".to_string(),
            state: ContainerState::Running,
            status: "Up".to_string(),
            runtime: ContainerRuntime::Docker,
            stats: ContainerStats::default(),
            created: 0,
            ports: vec![],
        };
        assert_eq!(container.display_image(), "nginx:latest");
    }

    #[test]
    fn test_container_display_memory() {
        let container = Container {
            id: "abc".to_string(),
            name: "test".to_string(),
            image: "nginx".to_string(),
            state: ContainerState::Running,
            status: "Up".to_string(),
            runtime: ContainerRuntime::Docker,
            stats: ContainerStats {
                memory_bytes: 1024 * 1024, // 1MB
                memory_limit: 2 * 1024 * 1024,
                ..Default::default()
            },
            created: 0,
            ports: vec![],
        };
        assert_eq!(container.display_memory(), "1.0M");
        assert_eq!(container.display_memory_limit(), "2.0M");
    }

    #[test]
    fn test_container_debug() {
        let container = Container {
            id: "abc".to_string(),
            name: "test".to_string(),
            image: "nginx".to_string(),
            state: ContainerState::Running,
            status: "Up".to_string(),
            runtime: ContainerRuntime::Docker,
            stats: ContainerStats::default(),
            created: 0,
            ports: vec![],
        };
        let debug = format!("{:?}", container);
        assert!(debug.contains("Container"));
    }

    #[test]
    fn test_container_clone() {
        let container = Container {
            id: "abc".to_string(),
            name: "test".to_string(),
            image: "nginx".to_string(),
            state: ContainerState::Running,
            status: "Up".to_string(),
            runtime: ContainerRuntime::Docker,
            stats: ContainerStats::default(),
            created: 12345,
            ports: vec![(8080, 80)],
        };
        let cloned = container.clone();
        assert_eq!(cloned.id, "abc");
        assert_eq!(cloned.created, 12345);
        assert_eq!(cloned.ports.len(), 1);
    }

    // ContainersData tests
    #[test]
    fn test_containers_data_default() {
        let data = ContainersData::default();
        assert!(data.containers.is_empty());
        assert!(data.runtime.is_none());
        assert!(data.state_counts.is_empty());
        assert!((data.total_cpu - 0.0).abs() < f32::EPSILON);
        assert_eq!(data.total_memory, 0);
    }

    #[test]
    fn test_containers_data_running() {
        let data = ContainersData {
            containers: vec![
                Container {
                    id: "1".to_string(),
                    name: "running".to_string(),
                    image: "nginx".to_string(),
                    state: ContainerState::Running,
                    status: "Up".to_string(),
                    runtime: ContainerRuntime::Docker,
                    stats: ContainerStats::default(),
                    created: 0,
                    ports: vec![],
                },
                Container {
                    id: "2".to_string(),
                    name: "stopped".to_string(),
                    image: "redis".to_string(),
                    state: ContainerState::Exited,
                    status: "Exited".to_string(),
                    runtime: ContainerRuntime::Docker,
                    stats: ContainerStats::default(),
                    created: 0,
                    ports: vec![],
                },
            ],
            runtime: Some(ContainerRuntime::Docker),
            state_counts: HashMap::new(),
            total_cpu: 0.0,
            total_memory: 0,
        };
        let running: Vec<_> = data.running().collect();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].name, "running");
    }

    #[test]
    fn test_containers_data_total() {
        let data = ContainersData {
            containers: vec![
                Container {
                    id: "1".to_string(),
                    name: "a".to_string(),
                    image: "nginx".to_string(),
                    state: ContainerState::Running,
                    status: "Up".to_string(),
                    runtime: ContainerRuntime::Docker,
                    stats: ContainerStats::default(),
                    created: 0,
                    ports: vec![],
                },
                Container {
                    id: "2".to_string(),
                    name: "b".to_string(),
                    image: "redis".to_string(),
                    state: ContainerState::Exited,
                    status: "Exited".to_string(),
                    runtime: ContainerRuntime::Docker,
                    stats: ContainerStats::default(),
                    created: 0,
                    ports: vec![],
                },
            ],
            runtime: Some(ContainerRuntime::Docker),
            state_counts: HashMap::new(),
            total_cpu: 0.0,
            total_memory: 0,
        };
        assert_eq!(data.total(), 2);
    }

    #[test]
    fn test_containers_data_running_count() {
        let mut state_counts = HashMap::new();
        state_counts.insert(ContainerState::Running, 3);
        state_counts.insert(ContainerState::Exited, 2);
        let data = ContainersData {
            containers: vec![],
            runtime: None,
            state_counts,
            total_cpu: 0.0,
            total_memory: 0,
        };
        assert_eq!(data.running_count(), 3);
    }

    #[test]
    fn test_containers_data_running_count_zero() {
        let data = ContainersData::default();
        assert_eq!(data.running_count(), 0);
    }

    #[test]
    fn test_containers_data_debug() {
        let data = ContainersData::default();
        let debug = format!("{:?}", data);
        assert!(debug.contains("ContainersData"));
    }

    #[test]
    fn test_containers_data_clone() {
        let data = ContainersData {
            containers: vec![],
            runtime: Some(ContainerRuntime::Podman),
            state_counts: HashMap::new(),
            total_cpu: 10.0,
            total_memory: 1024,
        };
        let cloned = data.clone();
        assert_eq!(cloned.runtime, Some(ContainerRuntime::Podman));
        assert_eq!(cloned.total_cpu, 10.0);
    }

    // ContainersAnalyzer tests
    #[test]
    fn test_containers_analyzer_default() {
        let analyzer = ContainersAnalyzer::default();
        let _ = analyzer.name();
    }

    #[test]
    fn test_containers_analyzer_name() {
        let analyzer = ContainersAnalyzer::new();
        assert_eq!(analyzer.name(), "containers");
    }

    #[test]
    fn test_containers_analyzer_data() {
        let analyzer = ContainersAnalyzer::new();
        let data = analyzer.data();
        assert!(data.containers.is_empty());
    }

    #[test]
    fn test_containers_analyzer_interval() {
        let analyzer = ContainersAnalyzer::new();
        let interval = analyzer.interval();
        assert_eq!(interval.as_secs(), 2);
    }

    #[test]
    fn test_containers_analyzer_parse_container_list_empty() {
        let analyzer = ContainersAnalyzer::new();
        let result = analyzer.parse_container_list("[]");
        // May fail if no runtime available
        let _ = result;
    }

    #[test]
    fn test_containers_analyzer_parse_container_list_invalid() {
        let analyzer = ContainersAnalyzer::new();
        let result = analyzer.parse_container_list("not json");
        // Should return empty or error
        let _ = result;
    }

    #[test]
    fn test_containers_analyzer_parse_container_stats_empty() {
        let analyzer = ContainersAnalyzer::new();
        let result = analyzer.parse_container_stats("{}");
        assert!(result.is_some());
    }

    // Helper function tests
    #[test]
    fn test_extract_string_after() {
        let result = extract_string_after(r#""test""#, "");
        assert_eq!(result, Some("test".to_string()));
    }

    #[test]
    fn test_extract_string_after_no_quote() {
        let result = extract_string_after("no quotes", "");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_json_string_nested() {
        let json = r#"{"outer":{"Name":"inner"}}"#;
        let result = extract_json_string(json, "Name");
        assert_eq!(result, Some("inner".to_string()));
    }

    #[test]
    fn test_extract_json_number_negative() {
        let json = r#"{"value":-123}"#;
        let result = extract_json_number(json, "value");
        assert_eq!(result, Some(-123));
    }

    #[test]
    fn test_extract_json_number_with_spaces() {
        let json = r#"{"value": 456}"#;
        let result = extract_json_number(json, "value");
        assert_eq!(result, Some(456));
    }

    #[test]
    fn test_extract_name_from_names_with_slash() {
        let json = r#""Names":["\/my-container"]"#;
        let result = extract_name_from_names(json);
        assert_eq!(result, Some("my-container".to_string()));
    }

    #[test]
    fn test_extract_name_from_names_missing() {
        let json = r#"{"Id":"123"}"#;
        let result = extract_name_from_names(json);
        assert!(result.is_none());
    }

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0B");
    }

    #[test]
    fn test_format_bytes_exact_kb() {
        assert_eq!(format_bytes(1024), "1.0K");
    }

    #[test]
    fn test_format_bytes_exact_mb() {
        assert_eq!(format_bytes(1024 * 1024), "1.0M");
    }

    #[test]
    fn test_format_bytes_exact_gb() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0G");
    }
}
