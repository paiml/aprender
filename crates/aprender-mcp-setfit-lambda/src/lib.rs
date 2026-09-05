//! Shared pieces of the SetFit Lambda transport: the streamable-HTTP config
//! and the embedded-or-env model resolution.
//!
//! A lib so the `bootstrap` binary and any transport test exercise the SAME
//! configuration — no drift between what ships and what is tested (the
//! pattern quote-pricing-lambda proved out).

use std::sync::Arc;

use aprender_mcp_setfit::ModelLoadError;
use pmcp::server::streamable_http_server::StreamableHttpServerConfig;

/// The model bytes staged by `build.rs`.
///
/// Non-empty when the deploy build set `APRENDER_SETFIT_MODEL`; empty in a
/// plain CI build, where [`resolve_model`] falls back to reading the same
/// variable as a runtime path.
pub static EMBEDDED_MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.apr"));

/// The streamable-HTTP config for the deployed Lambda: **`stateless()`**.
///
/// Stateless/JSON is the only mode that works under the loopback-proxy Lambda
/// pattern: sessions and SSE do not survive serverless (API Gateway routes
/// successive requests to different containers, each with a fresh in-memory
/// loopback server), and `stateless()`'s any-origin setting defers CORS trust
/// to the pmcp.run API Gateway in front. Measured in the reference
/// deployments: a stateful/SSE config returns `503 "Server is in error
/// state"` once deployed.
///
/// `max_request_bytes` is narrowed from pmcp's 4 MB default to the contract's
/// `MAX_REQUEST_BODY_BYTES` (~1 MiB). pmcp rejects an oversized body with HTTP
/// 413 BEFORE any JSON parsing — the bound core's own doc demands of a reading
/// surface, since "a bound checked after tokenization is not a bound on the work
/// an attacker can request" (`contracts/setfit-apr-v1.yaml`). Left at the
/// default, the deployed server accepted four times what it advertises, and the
/// only enforcement was `precheck`'s post-parse estimate. `precheck` still runs:
/// it is the sole bound on the stdio transport, which has no HTTP body at all.
#[must_use]
pub fn server_config() -> StreamableHttpServerConfig {
    StreamableHttpServerConfig {
        max_request_bytes: aprender_mcp_setfit::MAX_REQUEST_BODY_BYTES as usize,
        ..StreamableHttpServerConfig::stateless()
    }
}

/// Load the model this deployment serves: embedded bytes if the build staged
/// them, else the `APRENDER_SETFIT_MODEL` env var read as a runtime path.
///
/// Both doors run the same artifact verification ladder; neither trusts the
/// artifact because of where it came from.
///
/// # Errors
///
/// [`ModelLoadError`] when no model is available either way, or when the
/// verification ladder refuses the artifact.
pub fn resolve_model() -> Result<Arc<aprender_mcp_setfit::Model>, ModelLoadError> {
    if !EMBEDDED_MODEL.is_empty() {
        return aprender_mcp_setfit::load_model_from_bytes(EMBEDDED_MODEL).map(Arc::new);
    }
    let path = std::env::var_os(aprender_mcp_setfit::ENV_MODEL).ok_or(ModelLoadError::NoModel)?;
    aprender_mcp_setfit::load_model_from_path(std::path::Path::new(&path)).map(Arc::new)
}
