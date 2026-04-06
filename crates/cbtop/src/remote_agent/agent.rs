//! Remote agent for distributed benchmark collection.

use std::collections::HashMap;
use std::time::Instant;

use super::json;
use super::metrics;
use super::types::{
    AggregatedResult, CommandResult, HostBenchmark, HostConfig, HostHealth, HostState,
    RemoteAgentConfig, RemoteError, RemoteResult,
};

/// Remote agent for distributed benchmark collection
#[derive(Debug)]
pub struct RemoteAgent {
    /// Agent configuration
    config: RemoteAgentConfig,
    /// Registered hosts
    hosts: HashMap<String, HostState>,
    /// Command execution history
    history: Vec<CommandResult>,
    /// Maximum history size
    pub(crate) max_history: usize,
}

impl RemoteAgent {
    /// Create a new remote agent
    pub fn new(config: RemoteAgentConfig) -> Self {
        Self { config, hosts: HashMap::new(), history: Vec::new(), max_history: 1000 }
    }

    /// Add a host to the agent pool
    pub fn add_host(&mut self, config: HostConfig) {
        let key = format!("{}:{}", config.host, config.port);
        self.hosts.insert(key, HostState::new(config));
    }

    /// Remove a host from the pool
    pub fn remove_host(&mut self, host: &str) -> Option<HostState> {
        // Try with default port if not specified
        self.hosts.remove(host).or_else(|| self.hosts.remove(&format!("{}:22", host)))
    }

    /// Get all registered hosts
    pub fn hosts(&self) -> impl Iterator<Item = &HostState> {
        self.hosts.values()
    }

    /// Get available (healthy) hosts
    pub fn available_hosts(&self) -> impl Iterator<Item = &HostState> {
        self.hosts.values().filter(|h| h.is_available())
    }

    /// Get host count
    pub fn host_count(&self) -> usize {
        self.hosts.len()
    }

    /// Get available host count
    pub fn available_count(&self) -> usize {
        self.available_hosts().count()
    }

    /// Execute a command on a specific host (simulated for now)
    pub fn execute_on_host(
        &mut self,
        host_key: &str,
        command: &str,
    ) -> RemoteResult<CommandResult> {
        // Clone config to avoid borrow conflict
        let config = {
            let state = self
                .hosts
                .get(host_key)
                .ok_or_else(|| RemoteError::HostNotFound { host: host_key.to_string() })?;
            state.config.clone()
        };

        // Simulate command execution
        let start = Instant::now();

        // In a real implementation, this would use SSH
        // For now, we simulate based on command content
        let (exit_code, stdout, stderr) = self.simulate_command(command, &config);

        let duration_ms = start.elapsed().as_millis() as u64;

        let result =
            CommandResult { host: host_key.to_string(), exit_code, stdout, stderr, duration_ms };

        // Update host state
        if let Some(state) = self.hosts.get_mut(host_key) {
            if result.success() {
                state.record_success(duration_ms);
            } else {
                state.record_failure(&result.stderr);
            }
        }

        // Add to history
        self.add_to_history(result.clone());

        if result.success() {
            Ok(result)
        } else {
            Err(RemoteError::CommandFailed {
                host: host_key.to_string(),
                exit_code: result.exit_code,
                stderr: result.stderr,
            })
        }
    }

    /// Simulate command execution (placeholder for real SSH)
    fn simulate_command(&self, command: &str, config: &HostConfig) -> (i32, String, String) {
        // Check for health check command
        if command.contains("echo") && command.contains("health") {
            return (0, "OK".to_string(), String::new());
        }

        // Check for cbtop benchmark command
        if command.contains("cbtop") && command.contains("--json") {
            let arch = config.architecture.as_deref().unwrap_or("x86_64");
            let json = format!(
                r#"{{"host":"{}","arch":"{}","throughput":1000000,"latency_p50":50,"latency_p99":200,"memory":1073741824}}"#,
                config.host, arch
            );
            return (0, json, String::new());
        }

        // Unknown command - simulate failure
        (1, String::new(), "Unknown command".to_string())
    }

    /// Execute a command on all available hosts
    pub fn execute_on_all(&mut self, command: &str) -> Vec<Result<CommandResult, RemoteError>> {
        let host_keys: Vec<String> = self
            .available_hosts()
            .map(|h| format!("{}:{}", h.config.host, h.config.port))
            .collect();

        let mut results = Vec::new();
        for key in host_keys {
            results.push(self.execute_on_host(&key, command));
        }
        results
    }

    /// Run a health check on all hosts
    pub fn health_check(&mut self) -> HashMap<String, HostHealth> {
        let host_keys: Vec<String> = self.hosts.keys().cloned().collect();
        let mut results = HashMap::new();

        for key in host_keys {
            let _ = self.execute_on_host(&key, "echo health");
            if let Some(state) = self.hosts.get(&key) {
                results.insert(key, state.health.clone());
            }
        }

        results
    }

    /// Collect benchmarks from all available hosts
    pub fn collect_benchmarks(&mut self) -> RemoteResult<AggregatedResult> {
        let start = Instant::now();
        let command = format!("{} benchmark --json", self.config.remote_binary_path);

        let results = self.execute_on_all(&command);

        let mut host_benchmarks = Vec::new();
        let mut failures = Vec::new();

        for result in results {
            match result {
                Ok(cmd_result) => {
                    if let Some(benchmark) = self.parse_benchmark_json(&cmd_result) {
                        host_benchmarks.push(benchmark);
                    }
                }
                Err(e) => {
                    failures.push(e.to_string());
                }
            }
        }

        if host_benchmarks.is_empty() {
            return Err(RemoteError::AllHostsFailed { failures });
        }

        // Aggregate results
        let aggregated = self.aggregate_results(&host_benchmarks, failures.len());

        Ok(AggregatedResult {
            host_results: host_benchmarks,
            throughput_geomean: aggregated.0,
            latency_p50_mean_us: aggregated.1,
            latency_p99_max_us: aggregated.2,
            hosts_succeeded: aggregated.3,
            hosts_failed: aggregated.4,
            collection_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Parse benchmark JSON output
    fn parse_benchmark_json(&self, result: &CommandResult) -> Option<HostBenchmark> {
        let stdout = &result.stdout;

        let host = json::extract_json_string(stdout, "host")?;
        let arch =
            json::extract_json_string(stdout, "arch").unwrap_or_else(|| "unknown".to_string());
        let throughput = json::extract_json_number(stdout, "throughput")?;
        let latency_p50 = json::extract_json_number(stdout, "latency_p50")?;
        let latency_p99 = json::extract_json_number(stdout, "latency_p99")?;
        let memory = json::extract_json_number(stdout, "memory")? as u64;

        Some(HostBenchmark {
            host,
            architecture: arch,
            throughput_ops: throughput,
            latency_p50_us: latency_p50,
            latency_p99_us: latency_p99,
            memory_bytes: memory,
            gpu_utilization: None,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        })
    }

    /// Aggregate benchmark results based on strategy
    pub(crate) fn aggregate_results(
        &self,
        benchmarks: &[HostBenchmark],
        failure_count: usize,
    ) -> (f64, f64, f64, usize, usize) {
        if benchmarks.is_empty() {
            return (0.0, 0.0, 0.0, 0, failure_count);
        }

        let (throughput, latency_p50, latency_p99) =
            metrics::compute_metrics(benchmarks, self.config.aggregation);

        (throughput, latency_p50, latency_p99, benchmarks.len(), failure_count)
    }

    /// Add result to history with size limit
    fn add_to_history(&mut self, result: CommandResult) {
        self.history.push(result);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get command history
    pub fn history(&self) -> &[CommandResult] {
        &self.history
    }

    /// Get configuration
    pub fn config(&self) -> &RemoteAgentConfig {
        &self.config
    }

    /// Filter hosts by a predicate on their config.
    fn filter_hosts(&self, pred: impl Fn(&HostConfig) -> bool) -> Vec<&HostState> {
        self.hosts.values().filter(|h| pred(&h.config)).collect()
    }

    /// Filter hosts by label
    pub fn hosts_with_label(&self, key: &str, value: &str) -> Vec<&HostState> {
        self.filter_hosts(|c| c.labels.get(key).map(|v| v == value).unwrap_or(false))
    }

    /// Filter hosts by architecture
    pub fn hosts_with_arch(&self, arch: &str) -> Vec<&HostState> {
        self.filter_hosts(|c| c.architecture.as_deref() == Some(arch))
    }
}
