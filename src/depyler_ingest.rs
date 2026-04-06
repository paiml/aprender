//! Depyler Decision Trace Ingestion (Sprint 49 - Ticket #18)
//!
//! Configures renacer to ingest depyler decision traces from msgpack mmap files.
//!
//! # Overview
//!
//! This module provides:
//! - Configuration for watching decision trace files
//! - File watcher with incremental polling
//! - Sampling and circuit breaker for rate limiting
//!
//! # Example
//!
//! ```no_run
//! use renacer::depyler_ingest::{DepylerIngestConfig, DepylerWatcher};
//!
//! let config = DepylerIngestConfig::default();
//! let mut watcher = DepylerWatcher::new(config).expect("test");
//!
//! // Poll for new decisions
//! let decisions = watcher.poll().expect("test");
//! println!("Got {} new decisions", decisions.len());
//! ```
//!
//! # Reference
//!
//! paiml/depyler docs/specifications/decision-traces-signal-spec.md Section 5.1

use crate::decision_trace::{read_decisions_from_msgpack, sampling, DecisionTrace};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

/// Configuration for depyler decision trace ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DepylerIngestConfig {
    /// Paths to watch for decision trace files
    #[serde(default = "default_watch_paths")]
    pub watch_paths: Vec<PathBuf>,

    /// Poll interval in milliseconds
    #[serde(default = "default_poll_interval")]
    pub poll_interval_ms: u64,

    /// Sampling rate for remote export (0.0 - 1.0)
    #[serde(default = "default_sample_rate")]
    pub remote_sample_rate: f64,

    /// Maximum decisions per second for remote export (circuit breaker)
    #[serde(default = "default_max_rate")]
    pub max_remote_rate: u64,
}

fn default_watch_paths() -> Vec<PathBuf> {
    vec![PathBuf::from("/tmp/depyler_decisions.msgpack")]
}

fn default_poll_interval() -> u64 {
    100
}

fn default_sample_rate() -> f64 {
    0.1
}

fn default_max_rate() -> u64 {
    1000
}

impl Default for DepylerIngestConfig {
    fn default() -> Self {
        Self {
            watch_paths: default_watch_paths(),
            poll_interval_ms: default_poll_interval(),
            remote_sample_rate: default_sample_rate(),
            max_remote_rate: default_max_rate(),
        }
    }
}

/// Statistics for ingestion monitoring
#[derive(Debug, Clone, Default)]
pub struct IngestStats {
    /// Total decisions seen across all polls
    pub total_decisions_seen: u64,
    /// Total decisions that passed sampling
    pub total_decisions_sampled: u64,
    /// Total decisions exported (passed circuit breaker)
    pub total_decisions_exported: u64,
    /// Number of times circuit breaker tripped
    pub circuit_breaker_trips: u64,
}

/// File state tracker for incremental polling
#[derive(Debug, Default)]
struct FileState {
    /// Last modification time
    last_mtime: Option<SystemTime>,
    /// Number of decisions seen in last read
    last_count: usize,
}

/// Watcher for depyler decision trace files
pub struct DepylerWatcher {
    config: DepylerIngestConfig,
    file_states: HashMap<PathBuf, FileState>,
    stats: IngestStats,
    last_poll: Option<Instant>,
    /// Decisions exported this second (for circuit breaker)
    decisions_this_second: u64,
    /// Start of current second window
    second_window_start: Instant,
}

impl DepylerWatcher {
    /// Create a new watcher from configuration
    pub fn new(config: DepylerIngestConfig) -> Result<Self, String> {
        Ok(Self {
            config,
            file_states: HashMap::new(),
            stats: IngestStats::default(),
            last_poll: None,
            decisions_this_second: 0,
            second_window_start: Instant::now(),
        })
    }

    /// Poll for new decisions from all watched files
    ///
    /// Returns only NEW decisions since the last poll.
    pub fn poll(&mut self) -> Result<Vec<DecisionTrace>, String> {
        let mut all_new_decisions = Vec::new();

        for path in &self.config.watch_paths.clone() {
            let new_decisions = self.poll_file(path)?;
            all_new_decisions.extend(new_decisions);
        }

        self.stats.total_decisions_seen += all_new_decisions.len() as u64;
        self.last_poll = Some(Instant::now());

        Ok(all_new_decisions)
    }

    /// Poll a single file for new decisions
    fn poll_file(&mut self, path: &PathBuf) -> Result<Vec<DecisionTrace>, String> {
        // Check if file exists
        if !path.exists() {
            return Ok(Vec::new());
        }

        // Get current mtime
        let metadata = std::fs::metadata(path)
            .map_err(|e| format!("Failed to get metadata for {path:?}: {e}"))?;
        let mtime = metadata.modified().ok();

        // Get or create file state
        let state = self.file_states.entry(path.clone()).or_default();

        // Check if file has changed
        if state.last_mtime == mtime {
            return Ok(Vec::new());
        }

        // Read all decisions
        let all_decisions = read_decisions_from_msgpack(path)?;

        // Calculate new decisions (everything after last_count)
        let new_decisions = if all_decisions.len() > state.last_count {
            all_decisions[state.last_count..].to_vec()
        } else {
            // File was rewritten with fewer decisions - return all
            all_decisions.clone()
        };

        // Update state
        state.last_mtime = mtime;
        state.last_count = all_decisions.len();

        Ok(new_decisions)
    }

    /// Poll for new decisions with sampling applied
    ///
    /// Returns only decisions that pass the sampling rate.
    pub fn poll_sampled(&mut self) -> Result<Vec<DecisionTrace>, String> {
        let all_decisions = self.poll()?;
        let sample_rate = self.config.remote_sample_rate;

        let sampled: Vec<DecisionTrace> = all_decisions
            .into_iter()
            .filter(|_| sampling::should_sample_trace(sample_rate))
            .collect();

        self.stats.total_decisions_sampled += sampled.len() as u64;

        Ok(sampled)
    }

    /// Poll for new decisions with circuit breaker applied
    ///
    /// Returns decisions up to the `max_remote_rate` limit.
    pub fn poll_with_circuit_breaker(&mut self) -> Result<Vec<DecisionTrace>, String> {
        let all_decisions = self.poll()?;

        // Reset window if more than 1 second has passed
        if self.second_window_start.elapsed() >= Duration::from_secs(1) {
            self.decisions_this_second = 0;
            self.second_window_start = Instant::now();
        }

        // Calculate how many we can export
        let remaining_quota =
            self.config.max_remote_rate.saturating_sub(self.decisions_this_second);

        let exported: Vec<DecisionTrace> =
            all_decisions.into_iter().take(remaining_quota as usize).collect();

        // Check if circuit breaker tripped
        if exported.len() < remaining_quota as usize {
            // We exported fewer than quota - no trip
        } else if remaining_quota == 0 {
            self.stats.circuit_breaker_trips += 1;
        }

        self.decisions_this_second += exported.len() as u64;
        self.stats.total_decisions_exported += exported.len() as u64;

        Ok(exported)
    }

    /// Get current statistics
    pub fn stats(&self) -> &IngestStats {
        &self.stats
    }

    /// Get poll interval as Duration
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.config.poll_interval_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DepylerIngestConfig::default();
        assert_eq!(config.poll_interval_ms, 100);
        assert!((config.remote_sample_rate - 0.1).abs() < f64::EPSILON);
        assert_eq!(config.max_remote_rate, 1000);
        assert_eq!(config.watch_paths.len(), 1);
    }

    #[test]
    fn test_config_default_functions() {
        assert_eq!(default_poll_interval(), 100);
        assert!((default_sample_rate() - 0.1).abs() < f64::EPSILON);
        assert_eq!(default_max_rate(), 1000);
        let paths = default_watch_paths();
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_watcher_creation() {
        let config = DepylerIngestConfig::default();
        let watcher = DepylerWatcher::new(config);
        assert!(watcher.is_ok());
    }

    #[test]
    fn test_watcher_poll_nonexistent_file() {
        let config = DepylerIngestConfig {
            watch_paths: vec![PathBuf::from("/nonexistent/path.msgpack")],
            ..Default::default()
        };
        let mut watcher = DepylerWatcher::new(config).expect("test");
        let result = watcher.poll();
        assert!(result.is_ok());
        assert!(result.expect("test").is_empty());
    }

    #[test]
    fn test_watcher_stats() {
        let config = DepylerIngestConfig::default();
        let watcher = DepylerWatcher::new(config).expect("test");
        let stats = watcher.stats();
        assert_eq!(stats.total_decisions_seen, 0);
        assert_eq!(stats.total_decisions_sampled, 0);
        assert_eq!(stats.total_decisions_exported, 0);
        assert_eq!(stats.circuit_breaker_trips, 0);
    }

    #[test]
    fn test_watcher_poll_interval() {
        let config = DepylerIngestConfig { poll_interval_ms: 500, ..Default::default() };
        let watcher = DepylerWatcher::new(config).expect("test");
        assert_eq!(watcher.poll_interval(), Duration::from_millis(500));
    }

    #[test]
    fn test_ingest_stats_default() {
        let stats = IngestStats::default();
        assert_eq!(stats.total_decisions_seen, 0);
        assert_eq!(stats.total_decisions_sampled, 0);
        assert_eq!(stats.total_decisions_exported, 0);
        assert_eq!(stats.circuit_breaker_trips, 0);
    }

    #[test]
    fn test_ingest_stats_clone() {
        let stats = IngestStats {
            total_decisions_seen: 100,
            total_decisions_sampled: 50,
            total_decisions_exported: 25,
            circuit_breaker_trips: 2,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_decisions_seen, 100);
        assert_eq!(cloned.total_decisions_sampled, 50);
    }

    #[test]
    fn test_config_clone() {
        let config = DepylerIngestConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.poll_interval_ms, config.poll_interval_ms);
    }

    #[test]
    fn test_config_debug() {
        let config = DepylerIngestConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("DepylerIngestConfig"));
    }

    #[test]
    fn test_ingest_stats_debug() {
        let stats = IngestStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("IngestStats"));
    }

    #[test]
    fn test_file_state_default() {
        let state = FileState::default();
        assert!(state.last_mtime.is_none());
        assert_eq!(state.last_count, 0);
    }

    #[test]
    fn test_watcher_poll_sampled_nonexistent() {
        let config = DepylerIngestConfig {
            watch_paths: vec![PathBuf::from("/nonexistent/path.msgpack")],
            remote_sample_rate: 1.0, // 100% sampling
            ..Default::default()
        };
        let mut watcher = DepylerWatcher::new(config).expect("test");
        let result = watcher.poll_sampled();
        assert!(result.is_ok());
        assert!(result.expect("test").is_empty());
    }

    #[test]
    fn test_watcher_poll_with_circuit_breaker_nonexistent() {
        let config = DepylerIngestConfig {
            watch_paths: vec![PathBuf::from("/nonexistent/path.msgpack")],
            max_remote_rate: 100,
            ..Default::default()
        };
        let mut watcher = DepylerWatcher::new(config).expect("test");
        let result = watcher.poll_with_circuit_breaker();
        assert!(result.is_ok());
        assert!(result.expect("test").is_empty());
    }

    #[test]
    fn test_watcher_empty_watch_paths() {
        let config = DepylerIngestConfig { watch_paths: vec![], ..Default::default() };
        let mut watcher = DepylerWatcher::new(config).expect("test");
        let result = watcher.poll();
        assert!(result.is_ok());
        assert!(result.expect("test").is_empty());
    }

    #[test]
    fn test_watcher_multiple_polls() {
        let config = DepylerIngestConfig {
            watch_paths: vec![PathBuf::from("/nonexistent/path.msgpack")],
            ..Default::default()
        };
        let mut watcher = DepylerWatcher::new(config).expect("test");

        // Multiple polls should be fine
        for _ in 0..5 {
            let result = watcher.poll();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_watcher_circuit_breaker_stats() {
        let config = DepylerIngestConfig {
            watch_paths: vec![PathBuf::from("/nonexistent/path.msgpack")],
            max_remote_rate: 0, // Zero quota
            ..Default::default()
        };
        let mut watcher = DepylerWatcher::new(config).expect("test");

        // Poll with zero quota - should record stats
        let _ = watcher.poll_with_circuit_breaker();
        let stats = watcher.stats();
        assert_eq!(stats.total_decisions_exported, 0);
    }
}
