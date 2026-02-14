//! Server type definitions and data models for APR serve command

// Allow dead code and unused during development - these are planned features
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(clippy::format_push_string)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::if_not_else)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::inefficient_to_string)]

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
    /// Enable CORS (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub cors: bool,
    /// Request timeout in seconds (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub timeout_secs: u64,
    /// Maximum concurrent requests (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub max_concurrent: usize,
    /// Enable Prometheus metrics endpoint
    pub metrics: bool,
    /// Disable GPU acceleration (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub no_gpu: bool,
    /// Force GPU acceleration (requires CUDA feature)
    pub gpu: bool,
    /// Enable batched GPU inference for 2X+ throughput
    pub batch: bool,
    /// Enable inference tracing (PMAT-SHOWCASE-METHODOLOGY-001)
    pub trace: bool,
    /// Trace detail level (none, basic, layer)
    pub trace_level: String,
    /// Enable inline Roofline profiling (adds X-Profile headers)
    pub profile: bool,
    /// GH-152: Enable verbose request/response logging
    pub verbose: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            cors: true,
            timeout_secs: 30,
            max_concurrent: 10,
            metrics: true,
            no_gpu: false,
            gpu: false,
            batch: false,
            trace: false,
            trace_level: "basic".to_string(),
            profile: false,
            verbose: false,
        }
    }
}

impl ServerConfig {
    /// Create config with custom port (builder pattern, used in tests)
    #[cfg(test)]
    pub(crate) fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Create config with custom host (builder pattern, used in tests)

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
