//! Lambda entry: the loopback-proxy pattern of `aprender-mcp-chronos-lambda` (stateless pmcp server on
//! 127.0.0.1, each invocation proxied to it). `EAGER_LOAD=1` loads the model in the INIT phase (before the
//! runtime loop); the default loads it inside the first invocation.
use std::net::SocketAddr;
use std::sync::Arc;

use lambda_http::{run, service_fn, Body, Error, Request, Response};
use once_cell::sync::OnceCell;
use pmcp::server::streamable_http_server::{StreamableHttpServer, StreamableHttpServerConfig};

static BASE: OnceCell<(String, reqwest::Client)> = OnceCell::new();

async fn ensure_started() -> Result<&'static (String, reqwest::Client), Error> {
    if let Some(b) = BASE.get() {
        return Ok(b);
    }
    let server = laya_mcp_cloud::build_server("laya-decide").map_err(|e| Error::from(e.to_string()))?;
    let http = StreamableHttpServer::with_config(
        SocketAddr::from(([127, 0, 0, 1], 8080)),
        Arc::new(tokio::sync::Mutex::new(server)),
        StreamableHttpServerConfig::stateless(),
    );
    let (bound, handle) = http.start().await.map_err(|e| Error::from(e.to_string()))?;
    tokio::spawn(async move {
        let _ = handle.await;
    });
    let _ = BASE.set((format!("http://{bound}"), reqwest::Client::new()));
    BASE.get().ok_or_else(|| Error::from("base"))
}

async fn handler(event: Request) -> Result<Response<Body>, Error> {
    let (base, client) = ensure_started().await?;
    let path = event.uri().path_and_query().map_or("/".to_string(), |p| p.as_str().to_string());
    let method = reqwest::Method::from_bytes(event.method().as_str().as_bytes()).map_err(|e| Error::from(e.to_string()))?;
    let mut req = client.request(method, format!("{base}{path}"));
    for (k, v) in event.headers() {
        if !k.as_str().eq_ignore_ascii_case("host") {
            if let Ok(v) = v.to_str() {
                req = req.header(k.as_str(), v);
            }
        }
    }
    let body = match event.body() {
        Body::Empty => Vec::new(),
        Body::Text(s) => s.as_bytes().to_vec(),
        Body::Binary(b) => b.clone(),
    };
    let resp = req.body(body).send().await.map_err(|e| Error::from(e.to_string()))?;
    let mut out = Response::builder().status(resp.status().as_u16());
    for (k, v) in resp.headers() {
        if k != "transfer-encoding" && k != "content-length" {
            out = out.header(k, v);
        }
    }
    let bytes = resp.bytes().await.map_err(|e| Error::from(e.to_string()))?;
    out.body(Body::Binary(bytes.to_vec())).map_err(|e| Error::from(e.to_string()))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    laya_mcp_cloud::mark_process_start();
    laya_mcp_cloud::init_tracing();
    if std::env::var("EAGER_LOAD").as_deref() == Ok("1") {
        laya_mcp_cloud::load_engine().await.map_err(Error::from)?;
    }
    run(service_fn(handler)).await
}
