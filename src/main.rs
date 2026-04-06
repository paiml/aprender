//! trueno-db server binary.
//!
//! Analytics database server with HTTP API for SQL queries.
//!
//! Usage:
//!   trueno-db --config /path/to/config.yaml
//!   trueno-db --version

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::net::TcpListener;
use tracing::{error, info};
use trueno_db::query::{QueryEngine, QueryExecutor};
use trueno_db::storage::StorageEngine;

/// trueno-db: GPU-first embedded analytics database server.
#[derive(Parser)]
#[command(name = "trueno-db", version, about)]
struct Cli {
    /// Path to YAML configuration file.
    #[arg(long)]
    config: PathBuf,
}

/// Server configuration loaded from YAML.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ServerConfig {
    /// Listen address (e.g., "0.0.0.0:5433")
    listen: String,

    /// Data directory for Parquet files
    #[serde(default = "default_data_dir")]
    data_dir: String,

    /// Maximum memory in MB
    #[serde(default = "default_max_memory")]
    max_memory_mb: u64,

    /// Maximum concurrent connections
    #[serde(default = "default_max_connections")]
    max_connections: u32,

    /// Enable write-ahead logging
    #[serde(default = "default_true")]
    wal_enabled: bool,

    /// Sync mode: normal, aggressive, none
    #[serde(default = "default_sync_mode")]
    sync_mode: String,

    /// Compaction interval in seconds (0 = disabled)
    #[serde(default)]
    compaction_interval_secs: u64,
}

fn default_data_dir() -> String {
    "/opt/trueno-db/data".to_string()
}
fn default_max_memory() -> u64 {
    2048
}
fn default_max_connections() -> u32 {
    128
}
fn default_true() -> bool {
    true
}
fn default_sync_mode() -> String {
    "normal".to_string()
}

/// Shared application state.
struct AppState {
    storage: RwLock<StorageEngine>,
    query_engine: QueryEngine,
    executor: QueryExecutor,
    config: ServerConfig,
}

/// Query request body.
#[derive(Deserialize)]
struct QueryRequest {
    sql: String,
}

/// Query response.
#[derive(Serialize)]
struct QueryResponse {
    columns: Vec<String>,
    rows: Vec<Vec<serde_json::Value>>,
    row_count: usize,
}

/// Error response.
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    let config_str = std::fs::read_to_string(&cli.config)
        .map_err(|e| anyhow::anyhow!("cannot read config {}: {}", cli.config.display(), e))?;
    let config: ServerConfig = serde_yaml_ng::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("invalid config: {}", e))?;

    info!(
        listen = %config.listen,
        data_dir = %config.data_dir,
        max_memory_mb = config.max_memory_mb,
        "starting trueno-db server"
    );

    // Create data directory if it doesn't exist
    std::fs::create_dir_all(&config.data_dir)?;

    // Load any existing Parquet files from data_dir
    let storage = load_data_dir(&config.data_dir)?;

    let state = Arc::new(AppState {
        storage: RwLock::new(storage),
        query_engine: QueryEngine::new(),
        executor: QueryExecutor::new(),
        config,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/query", post(query))
        .route("/status", get(status))
        .with_state(state.clone());

    let addr: SocketAddr =
        state.config.listen.parse().map_err(|e| {
            anyhow::anyhow!("invalid listen address '{}': {}", state.config.listen, e)
        })?;

    info!(%addr, "trueno-db listening");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;

    info!("trueno-db shutdown complete");
    Ok(())
}

/// Load all Parquet files from a directory into a single StorageEngine.
fn load_data_dir(dir: &str) -> anyhow::Result<StorageEngine> {
    let path = std::path::Path::new(dir);
    if !path.exists() {
        return Ok(StorageEngine::new(vec![]));
    }

    let mut batches = vec![];
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("parquet") {
            info!(file = %p.display(), "loading parquet file");
            match StorageEngine::load_parquet(&p) {
                Ok(engine) => batches.extend(engine.batches().to_vec()),
                Err(e) => error!(file = %p.display(), error = %e, "failed to load parquet"),
            }
        }
    }

    info!(files = batches.len(), "data loaded");
    Ok(StorageEngine::new(batches))
}

/// GET /health — returns 200 OK.
async fn health() -> &'static str {
    "OK"
}

/// GET /status — returns server status as JSON.
async fn status(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let row_count = state
        .storage
        .read()
        .map(|s| s.batches().iter().map(|b| b.num_rows()).sum::<usize>())
        .unwrap_or(0);

    axum::Json(serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "data_dir": state.config.data_dir,
        "max_memory_mb": state.config.max_memory_mb,
        "row_count": row_count,
    }))
}

/// POST /query — execute SQL query.
async fn query(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<QueryRequest>,
) -> Result<axum::Json<QueryResponse>, (StatusCode, axum::Json<ErrorResponse>)> {
    let plan = state.query_engine.parse(&req.sql).map_err(|e| {
        (StatusCode::BAD_REQUEST, axum::Json(ErrorResponse { error: format!("parse error: {e}") }))
    })?;

    let storage = state.storage.read().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(ErrorResponse { error: format!("storage lock: {e}") }),
        )
    })?;

    let result = state.executor.execute(&plan, &storage).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(ErrorResponse { error: format!("query error: {e}") }),
        )
    })?;

    // Convert RecordBatch to JSON rows
    let columns: Vec<String> = result.schema().fields().iter().map(|f| f.name().clone()).collect();

    let mut rows = Vec::with_capacity(result.num_rows());
    for row_idx in 0..result.num_rows() {
        let mut row = Vec::with_capacity(columns.len());
        for col_idx in 0..result.num_columns() {
            let col = result.column(col_idx);
            let value = arrow_value_to_json(col, row_idx);
            row.push(value);
        }
        rows.push(row);
    }

    let row_count = rows.len();
    Ok(axum::Json(QueryResponse { columns, rows, row_count }))
}

/// Convert an Arrow array value at a given index to a JSON value.
fn arrow_value_to_json(array: &dyn arrow::array::Array, index: usize) -> serde_json::Value {
    #[allow(clippy::wildcard_imports)]
    use arrow::array::*;
    use arrow::datatypes::DataType;

    if array.is_null(index) {
        return serde_json::Value::Null;
    }

    match array.data_type() {
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>().unwrap();
            serde_json::Value::Number(a.value(index).into())
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>().unwrap();
            serde_json::Value::Number(a.value(index).into())
        }
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>().unwrap();
            serde_json::json!(a.value(index))
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
            serde_json::json!(a.value(index))
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            serde_json::Value::String(a.value(index).to_string())
        }
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            serde_json::Value::Bool(a.value(index))
        }
        _ => serde_json::Value::String(format!("<unsupported: {:?}>", array.data_type())),
    }
}

/// Wait for SIGTERM or Ctrl+C for graceful shutdown.
async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c().await.expect("ctrl+c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => info!("received ctrl+c"),
        () = terminate => info!("received SIGTERM"),
    }
}
