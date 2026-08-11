//! APR model serving command (PMAT-200: split from monolithic serve.rs)
//!
//! Serves ML models via HTTP API with support for APR, GGUF, and SafeTensors formats.
//! Implements OpenAI-compatible endpoints for generation, prediction, and transcription.

// Submodules (PMAT-200: split from 4351-line serve.rs)
pub mod auth;
#[cfg(feature = "inference")]
pub mod handlers;
#[cfg(feature = "inference")]
pub mod ollama;
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
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "long_running_graceful")]
pub(crate) fn run(model_path: &Path, config: &ServerConfig) -> Result<()> {
    // Record which file we are serving so the metadata endpoints can MEASURE
    // it instead of reporting constants. Everything downstream takes
    // `&ServerConfig`, so stamping it once here reaches every serve path.
    let config = &ServerConfig {
        model_path: Some(model_path.to_path_buf()),
        ..config.clone()
    };
    contract_pre_graceful_shutdown!();
    contract_pre_resource_cleanup!();
    contract_pre_concurrent_isolation!();
    contract_pre_request_routing!();
    contract_pre_cors_negotiation!();
    contract_pre_concurrent_model_access!();
    contract_pre_server_lifecycle!();
    // PMAT-297: Configure rayon thread pool to physical core count.
    // Default (all threads incl. HT) causes 44% regression from contention.
    #[cfg(feature = "inference")]
    if let Err(e) = realizar::inference::configure_optimal_thread_pool() {
        eprintln!("[PMAT-297] Thread pool config: {e} (may already be initialized)");
    }

    // GH-286: Set env vars for realizr's KV cache and FP8 control
    std::env::set_var("REALIZR_CONTEXT_LENGTH", config.context_length.to_string());
    if config.no_fp8_cache {
        std::env::set_var("REALIZR_NO_FP8_CACHE", "1");
    }

    println!("{}", "=== APR Serve ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Binding: {}", config.bind_addr());
    if config.context_length != 4096 {
        println!(
            "Context length: {} (--context-length)",
            config.context_length
        );
    }
    if config.no_fp8_cache {
        println!("FP8 cache: DISABLED (--no-fp8-cache, saves ~1.5 GB)");
    }
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
    println!("  POST /v1/predict     - Model prediction (APR)");
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
    let result = { handlers::start_realizar_server(model_path, config) };

    // Fallback: stub mode
    #[cfg(not(feature = "inference"))]
    let result = {
        println!();
        println!("{}", "[Server requires --features inference]".yellow());
        Ok(())
    };

    contract_post_graceful_shutdown!(&());
    contract_post_resource_cleanup!(&());
    contract_post_concurrent_isolation!(&());
    contract_post_request_routing!(&());
    contract_post_cors_negotiation!(&());
    contract_post_concurrent_model_access!(&());
    contract_post_server_lifecycle!(&());
    result
}
