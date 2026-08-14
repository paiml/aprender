//! Analytics database server: HTTP API for SQL queries over Parquet data.
//!
//! This module used to be `src/main.rs` behind the standalone `aprender-db`
//! (`trueno-db`) binary. That binary is gone (APR-MONO: one installed binary,
//! `apr`); the server is reached as `apr db serve --config <FILE>`.
//!
//! The entry point for non-async callers is [`run_blocking`], which owns the
//! tokio runtime so `apr`'s synchronous dispatch does not have to.

use crate::query::{QueryEngine, QueryExecutor};
use crate::storage::StorageEngine;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use tokio::net::TcpListener;
use tracing::{error, info};

/// Server configuration loaded from YAML.
///
/// Several fields are parsed but not yet acted on (`max_connections`,
/// `wal_enabled`, `sync_mode`, `compaction_interval_secs`). That is
/// pre-existing behaviour of this server, unchanged by the move into the
/// library; it is documented here rather than silently accepted.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ServerConfig {
    /// Listen address (e.g., "0.0.0.0:5433")
    pub listen: String,

    /// Data directory for Parquet files
    #[serde(default = "default_data_dir")]
    pub data_dir: String,

    /// Maximum memory in MB
    #[serde(default = "default_max_memory")]
    pub max_memory_mb: u64,

    /// Maximum concurrent connections
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,

    /// Enable write-ahead logging
    #[serde(default = "default_true")]
    pub wal_enabled: bool,

    /// Sync mode: normal, aggressive, none
    #[serde(default = "default_sync_mode")]
    pub sync_mode: String,

    /// Compaction interval in seconds (0 = disabled)
    #[serde(default)]
    pub compaction_interval_secs: u64,
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
pub struct AppState {
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

/// Parse a YAML server config from disk.
///
/// # Errors
///
/// Returns an error when the file cannot be read or is not a valid config.
pub fn load_config(config_path: &std::path::Path) -> anyhow::Result<ServerConfig> {
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow::anyhow!("cannot read config {}: {}", config_path.display(), e))?;
    serde_yaml_ng::from_str(&config_str).map_err(|e| anyhow::anyhow!("invalid config: {e}"))
}

/// Run the server from a synchronous caller, owning the tokio runtime.
///
/// This is what `apr db serve --config <FILE>` calls: `apr`'s dispatch is
/// synchronous, so the runtime belongs to whoever needs it rather than to a
/// `#[tokio::main]` attribute on a binary that no longer exists.
///
/// # Errors
///
/// Returns an error when the runtime cannot be built, the config cannot be
/// loaded, or the server fails to bind or serve.
pub fn run_blocking(config_path: &std::path::Path) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("cannot build tokio runtime: {e}"))?;
    runtime.block_on(serve(config_path))
}

/// Serve the HTTP API described by the config at `config_path` until SIGTERM
/// or Ctrl+C.
///
/// # Errors
///
/// Returns an error when the config is unreadable or invalid, the data
/// directory cannot be created, the listen address is malformed, or the bind
/// fails.
pub async fn serve(config_path: &std::path::Path) -> anyhow::Result<()> {
    // `try_init`, not `init`: `apr` may already have installed a tracing
    // subscriber, and a second `init` panics. A server that aborts because
    // logging was already configured is worse than one sharing the logger.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();

    let config = load_config(config_path)?;

    info!(
        listen = %config.listen,
        data_dir = %config.data_dir,
        max_memory_mb = config.max_memory_mb,
        "starting apr db server"
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

    info!(%addr, "apr db listening");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;

    info!("apr db shutdown complete");
    Ok(())
}

/// Load all Parquet files from a directory into a single `StorageEngine`.
///
/// # Errors
///
/// Returns an error when the directory cannot be read.
pub fn load_data_dir(dir: &str) -> anyhow::Result<StorageEngine> {
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
        .map(|s| s.batches().iter().map(arrow::array::RecordBatch::num_rows).sum::<usize>())
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

    // Scoped so the read guard is released before the (potentially long)
    // RecordBatch-to-JSON conversion below; holding it across that work
    // blocks every writer for no reason.
    let result = {
        let storage = state.storage.read().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(ErrorResponse { error: format!("storage lock: {e}") }),
            )
        })?;

        state.executor.execute(&plan, &storage).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(ErrorResponse { error: format!("query error: {e}") }),
            )
        })?
    };

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
            let a = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Arrow array downcast to Int32Array");
            serde_json::Value::Number(a.value(index).into())
        }
        DataType::Int64 => {
            let a = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64Array downcast failed for Int64 DataType column");
            serde_json::Value::Number(a.value(index).into())
        }
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>().expect("Arrow array downcast");
            serde_json::json!(a.value(index))
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>().expect("Arrow array downcast");
            serde_json::json!(a.value(index))
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().expect("Arrow array downcast");
            serde_json::Value::String(a.value(index).to_string())
        }
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>().expect("Arrow array downcast");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("apr-db-server-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create fixture");
        f.write_all(body.as_bytes()).expect("write fixture");
        path
    }

    /// A minimal config must load, and the documented defaults must apply.
    #[test]
    fn minimal_config_loads_with_documented_defaults() {
        let path = write_temp("minimal.yaml", "listen: \"127.0.0.1:5433\"\n");
        let config = load_config(&path).expect("a config with `listen` is valid");
        let _ = std::fs::remove_file(&path);

        assert_eq!(config.listen, "127.0.0.1:5433");
        assert_eq!(config.data_dir, "/opt/trueno-db/data");
        assert_eq!(config.max_memory_mb, 2048);
        assert_eq!(config.max_connections, 128);
        assert!(config.wal_enabled);
        assert_eq!(config.sync_mode, "normal");
        assert_eq!(config.compaction_interval_secs, 0);
    }

    /// Every overridable key must actually override. A key parsed into a field
    /// nobody reads back is how a config silently does nothing.
    #[test]
    fn every_config_key_overrides_its_default() {
        let path = write_temp(
            "full.yaml",
            "listen: \"0.0.0.0:9999\"\n\
             data_dir: \"/tmp/apr-db\"\n\
             max_memory_mb: 4096\n\
             max_connections: 7\n\
             wal_enabled: false\n\
             sync_mode: \"aggressive\"\n\
             compaction_interval_secs: 30\n",
        );
        let config = load_config(&path).expect("a fully specified config is valid");
        let _ = std::fs::remove_file(&path);

        assert_eq!(config.listen, "0.0.0.0:9999");
        assert_eq!(config.data_dir, "/tmp/apr-db");
        assert_eq!(config.max_memory_mb, 4096);
        assert_eq!(config.max_connections, 7);
        assert!(!config.wal_enabled);
        assert_eq!(config.sync_mode, "aggressive");
        assert_eq!(config.compaction_interval_secs, 30);
    }

    /// `listen` has no default; a config without it must be REFUSED, not
    /// defaulted to some address the operator never chose.
    #[test]
    fn config_without_listen_is_refused() {
        let path = write_temp("no-listen.yaml", "data_dir: \"/tmp/x\"\n");
        let outcome = load_config(&path);
        let _ = std::fs::remove_file(&path);

        let err = outcome.expect_err("a config with no `listen` must be refused");
        assert!(
            err.to_string().contains("invalid config"),
            "error must name the config as invalid, got: {err}"
        );
    }

    /// A missing config file must be refused with a message naming the path.
    #[test]
    fn missing_config_file_is_refused_and_names_the_path() {
        let missing = std::path::Path::new("/nonexistent/apr-db-not-here.yaml");
        let err = load_config(missing).expect_err("a missing config must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("cannot read config") && msg.contains("apr-db-not-here.yaml"),
            "error must say it could not read, and name the path; got: {msg}"
        );
    }

    /// Malformed YAML is refused rather than silently yielding defaults.
    #[test]
    fn malformed_yaml_is_refused() {
        let path = write_temp("bad.yaml", "listen: [this: is, not: a string\n");
        let outcome = load_config(&path);
        let _ = std::fs::remove_file(&path);
        assert!(outcome.is_err(), "malformed YAML must be refused, not parsed into defaults");
    }
}
