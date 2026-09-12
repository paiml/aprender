//! `bootstrap` — the AWS Lambda entry for the thin SetFit MCP server.
//!
//! The loopback-proxy pattern the quote-pricing deployment proved: the pmcp
//! streamable-HTTP server runs as an in-process background task bound to
//! 127.0.0.1, and each Lambda invocation is proxied to it over loopback.
//! The model resolves ONCE per container (embedded bytes or env path — see
//! `aprender_mcp_setfit_lambda::resolve_model`) and stays warm across
//! invocations.

use std::net::SocketAddr;
use std::sync::Arc;

use aprender_mcp_setfit::SERVER_NAME;
use lambda_http::http::header::{CONTENT_LENGTH, HOST, TRANSFER_ENCODING};
use lambda_http::http::Method;
use lambda_http::{run, service_fn, Body, Error, Request, Response};
use reqwest::Client;
use tokio::sync::OnceCell;
use tracing_subscriber::EnvFilter;

/// The loopback base URL and the client that talks to it.
///
/// ONE cell, not two: they are set together and are meaningless apart, so a
/// pair makes "initialized" a single fact the compiler enforces rather than a
/// convention two `set()` calls have to keep. `get_or_try_init` also removes
/// the lost-race arm and the "client must exist" error that could never fire.
static UPSTREAM: OnceCell<(String, Client)> = OnceCell::const_new();

/// Loopback port when `PORT` is unset. Lambda's runtime does not set it; the
/// value only has to be free inside the container.
const DEFAULT_PORT: u16 = 8080;

/// Start the streamable-HTTP server in the background; return the bound addr.
async fn start_http_in_background() -> pmcp::Result<SocketAddr> {
    let model = aprender_mcp_setfit_lambda::resolve_model()
        .map_err(|e| pmcp::Error::internal(e.to_string()))?;
    let server = aprender_mcp_setfit::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION"))?;
    let server = Arc::new(tokio::sync::Mutex::new(server));

    let port = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(DEFAULT_PORT);
    let host = std::env::var("MCP_HTTP_HOST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::net::IpAddr::from([127, 0, 0, 1]));
    let addr = SocketAddr::new(host, port);

    let http_server = pmcp::server::streamable_http_server::StreamableHttpServer::with_config(
        addr,
        server,
        aprender_mcp_setfit_lambda::server_config(),
    );
    let (bound, handle) = http_server.start().await?;
    tracing::info!("{SERVER_NAME}: MCP server started on {bound}");

    tokio::spawn(async move {
        if let Err(e) = handle.await {
            tracing::error!("HTTP server error: {e}");
        }
    });

    Ok(bound)
}

/// The loopback endpoint, started exactly once per container.
async fn upstream() -> Result<&'static (String, Client), Error> {
    UPSTREAM
        .get_or_try_init(|| async {
            let bound = start_http_in_background().await?;
            Ok::<_, Error>((format!("http://{bound}"), Client::builder().build()?))
        })
        .await
}

fn plain_response(status: u16, body: String) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("access-control-allow-origin", "*")
        .body(Body::Text(body))?)
}

/// Proxy one Lambda invocation to the loopback MCP server.
// serde_json::json! expands to .unwrap() internally (health-check body). Scoped
// to this function, not the crate: at crate scope it would lift the repo-wide
// `unwrap()` ban off the proxy and the startup path too.
#[allow(clippy::disallowed_methods)]
async fn handler(event: Request) -> Result<Response<Body>, Error> {
    let (parts, body) = event.into_parts();
    let path_q = parts
        .uri
        .path_and_query()
        .map_or_else(|| String::from("/"), |pq| pq.as_str().to_string());

    // Health check — also what pmcp.run's landing page probes.
    if parts.method == Method::GET {
        let body = serde_json::json!({
            "ok": true,
            "server": SERVER_NAME,
            "message": "SetFit classification MCP server. POST JSON-RPC to '/' for MCP requests."
        })
        .to_string();
        return plain_response(200, body);
    }

    // CORS preflight.
    if parts.method == Method::OPTIONS {
        return Ok(Response::builder()
            .status(200)
            .header("access-control-allow-origin", "*")
            .header("access-control-allow-methods", "POST, OPTIONS, GET")
            .header(
                "access-control-allow-headers",
                "content-type, authorization",
            )
            .body(Body::Empty)?);
    }

    let (base, client) = upstream().await?;

    // `parts.method` already IS a `reqwest::Method`: reqwest re-exports the
    // `http` crate's type, and both resolve http 1.x here. The former
    // `Method::from_bytes(...)` round-trip re-parsed a value it already had.
    let mut req = client.request(parts.method, format!("{base}{path_q}"));
    for (name, value) in &parts.headers {
        if name == HOST {
            continue;
        }
        let Ok(val) = value.to_str() else { continue };
        req = req.header(name, val);
    }
    // Moved, not copied: `event` was consumed by `into_parts`, so the body is
    // ours to hand to reqwest — up to MAX_REQUEST_BODY_BYTES per invocation
    // that no longer round-trips through an intermediate Vec.
    req = match body {
        Body::Empty => req.body(Vec::new()),
        Body::Text(s) => req.body(s),
        Body::Binary(b) => req.body(b),
    };

    let resp = req.send().await?;
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.bytes().await?;

    let mut builder = Response::builder()
        .status(status)
        .header("access-control-allow-origin", "*");
    for (name, value) in &headers {
        if name == TRANSFER_ENCODING || name == CONTENT_LENGTH {
            continue;
        }
        let Ok(val) = value.to_str() else { continue };
        builder = builder.header(name, val);
    }
    // `Bytes -> Vec<u8>` reclaims the allocation instead of copying it, since
    // the `Bytes` uniquely owns the buffer straight out of `Response::bytes()`.
    Ok(builder.body(Body::Binary(bytes.into()))?)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_ansi(false)
        .try_init();

    run(service_fn(handler)).await
}
