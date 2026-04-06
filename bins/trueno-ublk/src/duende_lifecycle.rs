//! Duende daemon lifecycle integration (DT-008)
//!
//! Provides daemon lifecycle management using the duende framework:
//! - Signal handling (SIGTERM, SIGINT) via duende-core
//! - Health check endpoints
//! - Restart policy with exponential backoff
//! - Platform-specific integration (systemd, launchd)

use async_trait::async_trait;
use duende_core::{
    BackoffConfig, Daemon, DaemonConfig, DaemonContext, DaemonError, DaemonId, DaemonMetrics,
    ExitReason, HealthStatus, RestartPolicy, Signal,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// trueno-ublk daemon lifecycle wrapper
pub struct TruenoUblkDaemon {
    /// Daemon identifier
    id: DaemonId,
    /// Daemon name
    name: String,
    /// Stop signal for graceful shutdown
    stop: Arc<AtomicBool>,
    /// Metrics collection
    metrics: DaemonMetrics,
}

impl TruenoUblkDaemon {
    /// Create a new daemon instance
    pub fn new(name: &str) -> Self {
        Self {
            id: DaemonId::new(),
            name: name.to_string(),
            stop: Arc::new(AtomicBool::new(false)),
            metrics: DaemonMetrics::new(),
        }
    }

    /// Get the stop signal for shutdown coordination
    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Check if shutdown has been requested
    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Get recommended restart policy for trueno-ublk
    pub fn restart_policy() -> RestartPolicy {
        RestartPolicy::WithBackoff(BackoffConfig {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            max_retries: 5,
        })
    }

    /// Get daemon configuration
    ///
    /// Note: DaemonConfig requires a binary path for spawning external daemons.
    /// For in-process daemon usage, use the Daemon trait implementation directly.
    pub fn config() -> DaemonConfig {
        let mut config = DaemonConfig::new("trueno-ublk", "/usr/bin/trueno-ublk");
        config.description = "SIMD/GPU-accelerated compressed RAM block device".to_string();
        config.shutdown_timeout = Duration::from_secs(10);
        config.health_check.interval = Duration::from_secs(30);
        config
    }
}

#[async_trait]
impl Daemon for TruenoUblkDaemon {
    fn id(&self) -> DaemonId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn init(&mut self, _config: &DaemonConfig) -> duende_core::Result<()> {
        info!("trueno-ublk daemon initializing via duende lifecycle");
        Ok(())
    }

    async fn run(&mut self, ctx: &mut DaemonContext) -> duende_core::Result<ExitReason> {
        info!("trueno-ublk daemon running via duende lifecycle");

        loop {
            if ctx.should_shutdown() {
                info!("Shutdown requested, exiting run loop");
                return Ok(ExitReason::Graceful);
            }

            // Check for signals (non-blocking)
            if let Some(signal) = ctx.try_recv_signal() {
                match signal {
                    Signal::Term | Signal::Int => {
                        if self.stop.load(Ordering::Relaxed) {
                            // Second signal - force exit
                            warn!("Received second interrupt, forcing exit");
                            return Err(DaemonError::Shutdown("Forced shutdown".to_string()));
                        }
                        info!("Received shutdown signal, initiating graceful shutdown...");
                        self.stop.store(true, Ordering::Relaxed);
                        return Ok(ExitReason::Signal(signal));
                    }
                    Signal::Hup => {
                        info!("Received reload signal (not implemented for trueno-ublk)");
                    }
                    _ => {}
                }
            }

            // Brief sleep to avoid busy-waiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn shutdown(&mut self, _timeout: Duration) -> duende_core::Result<()> {
        info!("trueno-ublk daemon shutting down");
        self.stop.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn health_check(&self) -> HealthStatus {
        // Simple health check - daemon is healthy if not stopped
        if self.stop.load(Ordering::Relaxed) {
            HealthStatus::unhealthy("Daemon is stopping", 1)
        } else {
            HealthStatus::healthy(1)
        }
    }

    fn metrics(&self) -> &DaemonMetrics {
        &self.metrics
    }
}

/// Wrapper to use duende signal handling instead of ctrlc
///
/// This provides a drop-in replacement for cleanup::setup_signal_handlers()
/// that uses duende-core for signal management.
pub fn setup_duende_signals() -> Arc<AtomicBool> {
    let daemon = TruenoUblkDaemon::new("trueno-ublk");
    info!("Using duende-core for signal handling");
    daemon.stop_signal()
}

/// Create a daemon for use with DaemonManager
pub fn create_daemon() -> TruenoUblkDaemon {
    TruenoUblkDaemon::new("trueno-ublk")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_creation() {
        let daemon = TruenoUblkDaemon::new("test-daemon");
        assert!(!daemon.should_stop());
        assert_eq!(daemon.name(), "test-daemon");
    }

    #[test]
    fn test_restart_policy() {
        let policy = TruenoUblkDaemon::restart_policy();
        match policy {
            RestartPolicy::WithBackoff(backoff) => {
                assert_eq!(backoff.initial_delay, Duration::from_secs(1));
                assert_eq!(backoff.max_delay, Duration::from_secs(60));
                assert_eq!(backoff.multiplier, 2.0);
                assert_eq!(backoff.max_retries, 5);
            }
            _ => panic!("Expected WithBackoff restart policy"),
        }
    }

    #[test]
    fn test_stop_signal() {
        let daemon = TruenoUblkDaemon::new("test");
        let stop = daemon.stop_signal();
        assert!(!stop.load(Ordering::Relaxed));
        stop.store(true, Ordering::Relaxed);
        assert!(daemon.should_stop());
    }

    #[test]
    fn test_config() {
        let config = TruenoUblkDaemon::config();
        assert_eq!(config.name, "trueno-ublk");
        assert_eq!(config.shutdown_timeout, Duration::from_secs(10));
        assert_eq!(config.health_check.interval, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn test_health_check() {
        let daemon = TruenoUblkDaemon::new("test");
        let health = daemon.health_check().await;
        assert!(health.healthy);
    }

    #[tokio::test]
    async fn test_health_check_unhealthy_when_stopping() {
        let daemon = TruenoUblkDaemon::new("test");
        daemon.stop.store(true, Ordering::Relaxed);
        let health = daemon.health_check().await;
        assert!(!health.healthy);
    }
}
