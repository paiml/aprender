//! Container entry (Fargate): load the model, THEN bind, so "port open" means "ready to decide".
//! `EAGER_LOAD=0` binds first and loads on the first call instead.
use std::net::SocketAddr;
use std::sync::Arc;

use pmcp::server::streamable_http_server::{StreamableHttpServer, StreamableHttpServerConfig};

#[tokio::main]
async fn main() -> Result<(), String> {
    laya_mcp_cloud::mark_process_start();
    laya_mcp_cloud::init_tracing();
    if std::env::var("EAGER_LOAD").as_deref() != Ok("0") {
        laya_mcp_cloud::load_engine().await?;
    }
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let server = laya_mcp_cloud::build_server("laya-decide").map_err(|e| e.to_string())?;
    let http = StreamableHttpServer::with_config(addr, Arc::new(tokio::sync::Mutex::new(server)), StreamableHttpServerConfig::stateless());
    let (bound, handle) = http.start().await.map_err(|e| e.to_string())?;
    tracing::info!("laya-decide MCP server listening on {bound}");
    handle.await.map_err(|e| e.to_string())
}
