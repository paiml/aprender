//! Shared pieces of the Chronos-Bolt Lambda transport: the streamable-HTTP config
//! and model resolution.
//!
//! A lib so the `bootstrap` binary and any transport test exercise the SAME
//! configuration — no drift between what ships and what is tested (mirroring
//! the proven `aprender-mcp-setfit-lambda` pattern).

use std::sync::Arc;

pub use aprender_mcp_chronos::{Model, ModelLoadError};
use pmcp::server::streamable_http_server::StreamableHttpServerConfig;

/// The Chronos-Bolt weights staged by `aprender-mcp-chronos`.
pub static EMBEDDED_WEIGHTS: &[u8] = aprender_mcp_chronos::EMBEDDED_WEIGHTS;

/// The checkpoint `config.json` staged by `aprender-mcp-chronos`.
pub static EMBEDDED_CONFIG: &[u8] = aprender_mcp_chronos::EMBEDDED_CONFIG;

/// The streamable-HTTP config for the deployed Lambda: **`stateless()`**.
///
/// Stateless/JSON is the only mode that survives serverless: sessions and SSE
/// do not carry across independent Lambda invocations/containers, and `stateless()`'s
/// any-origin setting defers CORS trust to the pmcp.run API Gateway in front.
#[must_use]
pub fn server_config() -> StreamableHttpServerConfig {
    StreamableHttpServerConfig::stateless()
}

/// Load the model this deployment serves: embedded bytes if staged, else
/// `CHRONOS_MODEL_DIR` read as a runtime path.
pub fn resolve_model() -> Result<Model, ModelLoadError> {
    aprender_mcp_chronos::resolve_model()
}

/// Build the MCP server instance over the resolved model.
pub fn build_server(model: Arc<Model>, name: &str, version: &str) -> pmcp::Result<pmcp::Server> {
    aprender_mcp_chronos::build_server(model, name, version)
}
