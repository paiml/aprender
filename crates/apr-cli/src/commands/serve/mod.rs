//! APR model serving command (PMAT-200: split from monolithic serve.rs)
//!
//! Serves ML models via HTTP API with support for APR, GGUF, and SafeTensors formats.
//! Implements OpenAI-compatible endpoints for generation, prediction, and transcription.

// Submodules (PMAT-200: split from 4351-line serve.rs)
#[cfg(feature = "inference")]
pub mod handlers;
pub mod routes;
#[cfg(feature = "inference")]
pub mod safetensors;
pub mod types;

// Re-exports for backward compatibility
pub use types::*;

// Test modules
#[cfg(test)]
mod tests;

use std::path::Path;

use colored::Colorize;

use crate::error::{CliError, Result};

/// Serve command entry point (blocking)
pub(crate) fn run(model_path: &Path, config: &ServerConfig) -> Result<()> {
    println!("{}", "=== APR Serve ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Binding: {}", config.bind_addr());
    println!();

    // Validate model
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    let state = ServerState::new(model_path.to_path_buf(), config.clone())?;

    println!(
        "{}",
        format!(
            "Model loading: {}",
            if state.uses_mmap { "mmap" } else { "full" }
        )
        .dimmed()
    );

    println!();
    println!("{}", "Endpoints:".green().bold());
    println!("  POST /predict        - Model prediction (APR)");
    println!("  POST /generate       - Text generation (GGUF)");
    println!("  GET  /health         - Health check");
    if config.metrics {
        println!("  GET  /metrics        - Prometheus metrics");
    }

    // GH-153: "Server ready" message now printed AFTER TcpListener::bind succeeds
    // in start_*_server functions, not here (was misleading since bind happens later)
    println!();
    println!("{}", "Press Ctrl+C to stop".dimmed());

    // Try to start real server with realizar
    #[cfg(feature = "inference")]
    {
        handlers::start_realizar_server(model_path, config)
    }

    // Fallback: stub mode
    #[cfg(not(feature = "inference"))]
    {
        println!();
        println!("{}", "[Server requires --features inference]".yellow());
        Ok(())
    }
}
