//! `bootstrap` — the AWS Lambda entry for the thin SetFit MCP server.
//!
//! The loopback-proxy pattern the quote-pricing deployment proved: the pmcp
//! streamable-HTTP server runs as an in-process background task bound to
//! 127.0.0.1, and each Lambda invocation is proxied to it over loopback.
//! The model resolves ONCE per container (embedded bytes or env path — see
//! `aprender_mcp_setfit_lambda::resolve_model`) and stays warm across
//! invocations.

// serde_json::json! expands to .unwrap() internally (health-check body).
#![allow(clippy::disallowed_methods)]

use std::net::SocketAddr;
use std::sync::Arc;

use lambda_http::{run, service_fn, Body, Error, Request, Response};
use once_cell::sync::OnceCell;
use reqwest::Client;
use tracing_subscriber::EnvFilter;

static BASE_URL: OnceCell<String> = OnceCell::new();
static HTTP: OnceCell<Client> = OnceCell::new();

const SERVER_NAME: &str = "aprender-setfit-predict";

/// Build the MCP server over the resolved model.
fn build_server() -> pmcp::Result<pmcp::Server> {
    let model = aprender_mcp_setfit_lambda::resolve_model()
        .map_err(|e| pmcp::Error::internal(e.to_string()))?;
    aprender_mcp_setfit::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION"))
}

/// Start the streamable-HTTP server in the background; return the bound addr.
async fn start_http_in_background(default_port: u16) -> pmcp::Result<SocketAddr> {
    let server = build_server()?;
    let server = Arc::new(tokio::sync::Mutex::new(server));

    let port = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(default_port);
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

/// Start the background server exactly once per container.
async fn ensure_server_started() -> Result<String, Error> {
    if let Some(url) = BASE_URL.get() {
        return Ok(url.clone());
    }
    let bound = start_http_in_background(8080)
        .await
        .map_err(|e| Error::from(e.to_string()))?;
    let base = format!("http://{bound}");
    let _ = BASE_URL.set(base.clone());
    let client = Client::builder()
        .build()
        .map_err(|e| Error::from(e.to_string()))?;
    let _ = HTTP.set(client);
    Ok(base)
}

fn plain_response(status: u16, body: String) -> Result<Response<Body>, Error> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("access-control-allow-origin", "*")
        .body(Body::Text(body))
        .map_err(|e| Error::from(e.to_string()))
}

/// Proxy one Lambda invocation to the loopback MCP server.
async fn handler(event: Request) -> Result<Response<Body>, Error> {
    let method = event.method().clone();
    let path_q = event
        .uri()
        .path_and_query()
        .map_or_else(|| String::from("/"), |pq| pq.as_str().to_string());

    // Health check — also what pmcp.run's landing page probes.
    if method.as_str() == "GET" {
        let body = serde_json::json!({
            "ok": true,
            "server": SERVER_NAME,
            "message": "SetFit classification MCP server. POST JSON-RPC to '/' for MCP requests."
        })
        .to_string();
        return plain_response(200, body);
    }

    // CORS preflight.
    if method.as_str() == "OPTIONS" {
        return Response::builder()
            .status(200)
            .header("access-control-allow-origin", "*")
            .header("access-control-allow-methods", "POST, OPTIONS, GET")
            .header(
                "access-control-allow-headers",
                "content-type, authorization",
            )
            .body(Body::Empty)
            .map_err(|e| Error::from(e.to_string()));
    }

    let base = ensure_server_started().await?;
    let client = HTTP
        .get()
        .ok_or_else(|| Error::from("http client must exist after ensure_server_started"))?;

    let url = format!("{base}{path_q}");
    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| Error::from(e.to_string()))?;

    let mut req = client.request(reqwest_method, &url);
    for (name, value) in event.headers() {
        if let Ok(val) = value.to_str() {
            if name.as_str().eq_ignore_ascii_case("host") {
                continue;
            }
            req = req.header(name.as_str(), val);
        }
    }
    let body_bytes = match event.body() {
        Body::Empty => Vec::new(),
        Body::Text(s) => s.as_bytes().to_vec(),
        Body::Binary(b) => b.clone(),
    };
    req = req.body(body_bytes);

    let resp = req.send().await.map_err(|e| Error::from(e.to_string()))?;
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.bytes().await.map_err(|e| Error::from(e.to_string()))?;

    let mut builder = Response::builder().status(status.as_u16());
    builder = builder.header("access-control-allow-origin", "*");
    for (name, value) in &headers {
        if let Ok(val) = value.to_str() {
            if name.as_str().eq_ignore_ascii_case("transfer-encoding")
                || name.as_str().eq_ignore_ascii_case("content-length")
            {
                continue;
            }
            builder = builder.header(name.as_str(), val);
        }
    }
    builder
        .body(Body::Binary(bytes.to_vec()))
        .map_err(|e| Error::from(e.to_string()))
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
