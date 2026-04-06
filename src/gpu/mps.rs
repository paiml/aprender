//! Experimental CUDA MPS (Multi-Process Service) support (GPU-SHARE §1.5).
//!
//! MPS is **not** auto-started. This module provides opt-in setup for users
//! who understand the risks:
//!
//! - A GPU fault in any MPS client kills ALL clients on that GPU
//! - Thread percentage is static once set (no rebalancing)
//! - Jetson MPS is experimental below 30% thread allocation
//!
//! # Usage
//!
//! ```bash
//! apr finetune model.apr --vram 8 --experimental-mps --gpu-share 50
//! ```

use std::env;

/// MPS configuration for experimental GPU sharing.
#[derive(Debug, Clone)]
pub struct MpsConfig {
    /// Percentage of GPU SMs allocated to this process (1-100).
    pub thread_percentage: u32,
    /// Pinned device memory limit per client (MB). Prevents OOM cascades.
    pub pinned_mem_limit_mb: Option<u64>,
    /// Override checkpoint frequency to every N steps (limits blast radius).
    pub checkpoint_every_steps: usize,
}

impl Default for MpsConfig {
    fn default() -> Self {
        Self { thread_percentage: 50, pinned_mem_limit_mb: None, checkpoint_every_steps: 100 }
    }
}

impl MpsConfig {
    /// Create MPS config with the given thread share percentage.
    ///
    /// # Panics
    /// Panics if `thread_pct` is 0 or > 100.
    #[must_use]
    pub fn with_share(thread_pct: u32) -> Self {
        assert!(thread_pct > 0 && thread_pct <= 100, "thread_pct must be 1-100");
        Self { thread_percentage: thread_pct, ..Default::default() }
    }

    /// Set pinned device memory limit (MB) to prevent OOM cascades.
    #[must_use]
    pub fn with_mem_limit(mut self, limit_mb: u64) -> Self {
        self.pinned_mem_limit_mb = Some(limit_mb);
        self
    }
}

/// Set MPS environment variables before CUDA context creation.
///
/// **MUST be called before any CUDA API call** (cuCtxCreate, cuInit, etc.).
/// Environment variables set after context creation have no effect.
///
/// # Returns
/// List of environment variables that were set.
pub fn setup_mps_env(config: &MpsConfig) -> Vec<(String, String)> {
    let mut vars = Vec::new();

    // Thread percentage: controls SM allocation for this MPS client
    let thread_pct = config.thread_percentage.to_string();
    #[allow(clippy::disallowed_methods)]
    env::set_var("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", &thread_pct);
    vars.push(("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE".to_string(), thread_pct));

    // Pinned memory limit: prevents OOM cascades across MPS clients
    if let Some(limit_mb) = config.pinned_mem_limit_mb {
        let limit_str = format!("0={limit_mb}MB");
        #[allow(clippy::disallowed_methods)]
        env::set_var("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", &limit_str);
        vars.push(("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT".to_string(), limit_str));
    }

    vars
}

/// Print MPS safety warning to stderr.
///
/// This warning is mandatory whenever `--experimental-mps` is used.
pub fn print_mps_warning(config: &MpsConfig) {
    eprintln!("WARNING: MPS enabled — a GPU fault in any job will crash ALL jobs on this GPU.");
    eprintln!("  Thread allocation: {}%", config.thread_percentage);
    if let Some(limit) = config.pinned_mem_limit_mb {
        eprintln!("  Pinned memory limit: {limit} MB");
    }
    eprintln!(
        "  Checkpoint frequency: every {} steps (blast radius limit)",
        config.checkpoint_every_steps
    );
    eprintln!("  Use --experimental-mps only if you understand the risks.");
    eprintln!();
}

/// Check if MPS daemon appears to be running.
///
/// Checks for the existence of the MPS control pipe. This is a best-effort
/// check — the daemon may still be unhealthy even if the pipe exists.
#[must_use]
pub fn is_mps_daemon_running() -> bool {
    // MPS control pipe is at /tmp/nvidia-mps/control by default
    std::path::Path::new("/tmp/nvidia-mps/control").exists()
}

/// Validate MPS configuration for known issues.
///
/// Returns a list of warnings (non-fatal) and errors (fatal).
pub fn validate_mps_config(config: &MpsConfig) -> MpsValidation {
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    if config.thread_percentage < 30 {
        warnings.push(
            "Thread percentage below 30% is unreliable on Jetson (NVIDIA Forum).".to_string(),
        );
    }

    if config.thread_percentage < 10 {
        errors
            .push("Thread percentage below 10% causes severe performance degradation.".to_string());
    }

    if config.pinned_mem_limit_mb.is_none() {
        warnings.push(
            "No pinned memory limit set. OOM in one job may crash all MPS clients.".to_string(),
        );
    }

    MpsValidation { warnings, errors }
}

/// Result of MPS configuration validation.
#[derive(Debug, Clone)]
pub struct MpsValidation {
    /// Non-fatal warnings.
    pub warnings: Vec<String>,
    /// Fatal errors (should abort).
    pub errors: Vec<String>,
}

impl MpsValidation {
    /// Whether there are fatal errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MpsConfig::default();
        assert_eq!(config.thread_percentage, 50);
        assert!(config.pinned_mem_limit_mb.is_none());
        assert_eq!(config.checkpoint_every_steps, 100);
    }

    #[test]
    fn test_with_share() {
        let config = MpsConfig::with_share(33);
        assert_eq!(config.thread_percentage, 33);
    }

    #[test]
    fn test_with_mem_limit() {
        let config = MpsConfig::with_share(50).with_mem_limit(8000);
        assert_eq!(config.pinned_mem_limit_mb, Some(8000));
    }

    #[test]
    #[should_panic(expected = "thread_pct must be 1-100")]
    fn test_zero_thread_pct_panics() {
        let _ = MpsConfig::with_share(0);
    }

    #[test]
    #[should_panic(expected = "thread_pct must be 1-100")]
    fn test_over_100_thread_pct_panics() {
        let _ = MpsConfig::with_share(101);
    }

    #[test]
    fn test_setup_mps_env_sets_thread_pct() {
        let config = MpsConfig::with_share(33);
        let vars = setup_mps_env(&config);
        assert!(vars.iter().any(|(k, v)| k == "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" && v == "33"));
    }

    #[test]
    fn test_setup_mps_env_sets_mem_limit() {
        let config = MpsConfig::with_share(50).with_mem_limit(8000);
        let vars = setup_mps_env(&config);
        assert!(vars
            .iter()
            .any(|(k, v)| k == "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" && v == "0=8000MB"));
    }

    #[test]
    fn test_validate_ok() {
        let config = MpsConfig::with_share(50).with_mem_limit(8000);
        let result = validate_mps_config(&config);
        assert!(!result.has_errors());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_validate_low_thread_warning() {
        let config = MpsConfig::with_share(25);
        let result = validate_mps_config(&config);
        assert!(!result.has_errors());
        assert!(result.warnings.iter().any(|w| w.contains("below 30%")));
        // No mem limit → also warns
        assert!(result.warnings.iter().any(|w| w.contains("pinned memory")));
    }

    #[test]
    fn test_validate_very_low_thread_error() {
        let config = MpsConfig::with_share(5);
        let result = validate_mps_config(&config);
        assert!(result.has_errors());
        assert!(result.errors.iter().any(|e| e.contains("below 10%")));
    }

    #[test]
    fn test_mps_daemon_check() {
        // Just verify it returns a bool without crashing
        let _running = is_mps_daemon_running();
    }
}
