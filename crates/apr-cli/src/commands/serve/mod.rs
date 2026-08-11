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

/// The endpoint lines the startup banner may print, for the model actually being served.
///
/// aprender#2376(8): this banner printed `POST /v1/predict - Model prediction (APR)`
/// unconditionally. `/v1/predict` is the APR predictor; on the GGUF path it is
/// mounted but can only ever answer 503 "No APR model loaded", so an operator
/// reading the banner was told to call an endpoint that could not work for the
/// file they had just passed. It also printed `POST /generate - Text generation
/// (GGUF)` for APR models, where that route is not mounted at all.
///
/// A banner may under-advertise (the bound server prints its own, fuller list);
/// it must never advertise a route the served model cannot use. Kept pure and
/// separate from the printing so the invariant is unit-testable.
#[cfg_attr(not(feature = "inference"), allow(unused_variables))]
fn banner_endpoints(model_path: &Path, metrics: bool) -> Vec<String> {
    let mut lines = Vec::new();

    // Magic-byte detection, not the extension — the same call the serve path makes
    // to pick a server. `None` means "not yet known": print only what every router
    // mounts rather than guess.
    #[cfg(feature = "inference")]
    let format = realizar::format::detect_format_from_path(model_path).ok();
    #[cfg(not(feature = "inference"))]
    let format: Option<()> = None;

    #[cfg(feature = "inference")]
    {
        use realizar::format::ModelFormat;
        match format {
            Some(ModelFormat::Apr) => {
                lines.push("  POST /v1/predict     - Model prediction (APR)".to_string());
            }
            Some(ModelFormat::Gguf | ModelFormat::SafeTensors) => {
                lines.push("  POST /generate       - Text generation".to_string());
            }
            None => {}
        }
    }

    lines.push("  POST /v1/completions - OpenAI-compatible completions".to_string());
    lines.push("  GET  /health         - Health check".to_string());
    if metrics {
        lines.push("  GET  /metrics        - Prometheus metrics".to_string());
    }
    lines
}

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
    for line in banner_endpoints(model_path, config.metrics) {
        println!("{line}");
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
