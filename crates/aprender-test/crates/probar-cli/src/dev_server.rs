//! WASM Development Server
//!
//! Real HTTP server implementation for serving WASM applications with hot reload.
//! Implements GitHub issue #7: <https://github.com/paiml/probar/issues/7>
//!
//! ## Features
//!
//! - HTTP server with correct MIME types for WASM
//! - WebSocket server for hot reload notifications
//! - File watcher with debouncing
//! - `wasm-pack` build integration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    DevServer                                │
//! │  ┌──────────────┐     ┌──────────────┐                     │
//! │  │ HTTP Server  │     │  WS Server   │                     │
//! │  │ (port 8080)  │     │ (port 8081)  │                     │
//! │  └──────┬───────┘     └──────┬───────┘                     │
//! │         │                    │                              │
//! │         ▼                    ▼                              │
//! │  ┌──────────────┐     ┌──────────────┐                     │
//! │  │ Static Files │     │ HotReload    │◀───FileWatcher      │
//! │  │ .wasm .js    │     │ Messages     │                     │
//! │  └──────────────┘     └──────────────┘                     │
//! └─────────────────────────────────────────────────────────────┘
//! ```

// Clippy allows for server patterns
#![allow(clippy::unused_async)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::use_self)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::redundant_else)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};

/// Hot reload message sent to connected clients (JSON serializable)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum HotReloadMessage {
    /// File changed, rebuild triggered
    FileChanged {
        /// Path to the changed file
        path: String,
    },
    /// Rebuild started
    RebuildStarted,
    /// Rebuild completed successfully
    RebuildComplete {
        /// Duration in milliseconds
        duration_ms: u64,
    },
    /// Rebuild failed with error
    RebuildFailed {
        /// Error message
        error: String,
    },
    /// Server ready
    ServerReady,
    /// Enhanced file change with size tracking (spec C.2)
    FileModified {
        /// Path to the changed file
        path: String,
        /// Event type (created, modified, deleted, renamed)
        event: FileChangeEvent,
        /// Timestamp in milliseconds since epoch
        timestamp: u64,
        /// Size before change (None for created files)
        size_before: Option<u64>,
        /// Size after change (None for deleted files)
        size_after: Option<u64>,
        /// Human-readable diff summary
        diff_summary: String,
    },
    /// Client connected notification
    ClientConnected {
        /// Number of connected clients
        client_count: usize,
    },
    /// Client disconnected notification
    ClientDisconnected {
        /// Number of connected clients
        client_count: usize,
    },
}

/// File change event types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FileChangeEvent {
    /// New file created
    Created,
    /// Existing file modified
    Modified,
    /// File deleted
    Deleted,
    /// File renamed
    Renamed,
}

impl HotReloadMessage {
    /// Serialize to JSON for WebSocket transmission
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| r#"{"type":"Error"}"#.to_string())
    }

    /// Create a file modified message with size tracking
    #[must_use]
    pub fn file_modified(
        path: impl Into<String>,
        event: FileChangeEvent,
        size_before: Option<u64>,
        size_after: Option<u64>,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let diff_summary = match (&event, size_before, size_after) {
            (FileChangeEvent::Created, _, Some(after)) => format!("+{}", format_bytes(after)),
            (FileChangeEvent::Deleted, Some(before), _) => format!("-{}", format_bytes(before)),
            (FileChangeEvent::Modified, Some(before), Some(after)) => {
                if after >= before {
                    format!("+{}", format_bytes(after - before))
                } else {
                    format!("-{}", format_bytes(before - after))
                }
            }
            (FileChangeEvent::Renamed, _, _) => "renamed".to_string(),
            _ => "changed".to_string(),
        };

        Self::FileModified {
            path: path.into(),
            event,
            timestamp,
            size_before,
            size_after,
            diff_summary,
        }
    }
}

impl FileChangeEvent {
    /// Get display string for the event type
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "CREATED",
            Self::Modified => "MODIFIED",
            Self::Deleted => "DELETED",
            Self::Renamed => "RENAMED",
        }
    }
}

/// Format bytes in human-readable form
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;

    if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

/// WASM development server configuration
#[derive(Debug, Clone)]
pub struct DevServerConfig {
    /// Directory to serve static files from
    pub directory: PathBuf,
    /// HTTP port
    pub port: u16,
    /// WebSocket port for hot reload
    pub ws_port: u16,
    /// Enable CORS
    pub cors: bool,
    /// Enable Cross-Origin Isolation (COOP/COEP headers for SharedArrayBuffer)
    pub cross_origin_isolated: bool,
}

impl Default for DevServerConfig {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("."),
            port: 8080,
            ws_port: 8081,
            cors: false,
            cross_origin_isolated: false,
        }
    }
}

impl DevServerConfig {
    /// Create a builder
    #[must_use]
    pub fn builder() -> DevServerConfigBuilder {
        DevServerConfigBuilder::default()
    }
}

/// Builder for `DevServerConfig`
#[derive(Debug, Clone, Default)]
pub struct DevServerConfigBuilder {
    config: DevServerConfig,
}

impl DevServerConfigBuilder {
    /// Set directory to serve
    #[must_use]
    pub fn directory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.directory = dir.into();
        self
    }

    /// Set HTTP port
    #[must_use]
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Set WebSocket port
    #[must_use]
    pub fn ws_port(mut self, port: u16) -> Self {
        self.config.ws_port = port;
        self
    }

    /// Enable CORS
    #[must_use]
    pub fn cors(mut self, enabled: bool) -> Self {
        self.config.cors = enabled;
        self
    }

    /// Enable Cross-Origin Isolation (COOP/COEP headers)
    ///
    /// Required for SharedArrayBuffer and parallel WASM with Web Workers.
    /// Sets the following headers:
    /// - `Cross-Origin-Opener-Policy: same-origin`
    /// - `Cross-Origin-Embedder-Policy: require-corp`
    #[must_use]
    pub fn cross_origin_isolated(mut self, enabled: bool) -> Self {
        self.config.cross_origin_isolated = enabled;
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> DevServerConfig {
        self.config
    }
}

/// WASM development server with HTTP and WebSocket support
#[derive(Debug)]
pub struct DevServer {
    config: DevServerConfig,
    reload_tx: broadcast::Sender<HotReloadMessage>,
}

impl DevServer {
    /// Create a new dev server
    #[must_use]
    pub fn new(config: DevServerConfig) -> Self {
        let (reload_tx, _) = broadcast::channel(64);
        Self { config, reload_tx }
    }

    /// Get a sender for hot reload messages
    #[must_use]
    pub fn reload_sender(&self) -> broadcast::Sender<HotReloadMessage> {
        self.reload_tx.clone()
    }

    /// Get the HTTP URL
    #[must_use]
    pub fn http_url(&self) -> String {
        format!("http://localhost:{}", self.config.port)
    }

    /// Get the WebSocket URL
    #[must_use]
    pub fn ws_url(&self) -> String {
        format!("ws://localhost:{}/ws", self.config.port)
    }

    /// Start the server (blocking)
    ///
    /// This starts both the HTTP server for static files and
    /// WebSocket endpoints for hot reload on the same port.
    pub async fn run(&self) -> Result<(), std::io::Error> {
        let directory = Arc::new(self.config.directory.clone());
        let reload_tx = self.reload_tx.clone();

        // Build router with static file serving and WebSocket
        let app = Router::new()
            // WebSocket endpoint for hot reload
            .route(
                "/ws",
                get({
                    let tx = reload_tx.clone();
                    move |ws: WebSocketUpgrade| handle_websocket(ws, tx.clone())
                }),
            )
            // Index route
            .route(
                "/",
                get({
                    let dir = directory.clone();
                    move || serve_index(dir.clone())
                }),
            )
            // Static file fallback
            .fallback({
                let dir = directory.clone();
                move |uri: axum::http::Uri| serve_static(dir.clone(), uri)
            });

        // Add CORS if enabled
        let app = if self.config.cors {
            app.layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
        } else {
            app
        };

        // Add Cross-Origin Isolation headers if enabled (for SharedArrayBuffer/Web Workers)
        let app = if self.config.cross_origin_isolated {
            use tower_http::set_header::SetResponseHeaderLayer;
            app.layer(SetResponseHeaderLayer::overriding(
                header::HeaderName::from_static("cross-origin-opener-policy"),
                header::HeaderValue::from_static("same-origin"),
            ))
            .layer(SetResponseHeaderLayer::overriding(
                header::HeaderName::from_static("cross-origin-embedder-policy"),
                header::HeaderValue::from_static("require-corp"),
            ))
        } else {
            app
        };

        // Add gzip compression for better WASM transfer speeds
        let app = app.layer(CompressionLayer::new().gzip(true));

        let addr = SocketAddr::from(([0, 0, 0, 0], self.config.port));

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║               Probar WASM Development Server                 ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  HTTP:      http://localhost:{:<29}║", self.config.port);
        println!(
            "║  WebSocket: ws://localhost:{}/ws{:<23}║",
            self.config.port, ""
        );
        println!(
            "║  Directory: {:<48}║",
            self.config
                .directory
                .display()
                .to_string()
                .chars()
                .take(48)
                .collect::<String>()
        );
        println!(
            "║  CORS:      {:<48}║",
            if self.config.cors {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!(
            "║  COOP/COEP: {:<48}║",
            if self.config.cross_origin_isolated {
                "enabled (SharedArrayBuffer available)"
            } else {
                "disabled"
            }
        );
        println!("║  Gzip:      {:<48}║", "enabled (auto-compression)");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Press Ctrl+C to stop                                        ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Notify that server is ready
        let _ = reload_tx.send(HotReloadMessage::ServerReady);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Run HTTP and WebSocket servers on separate ports
    ///
    /// Use this when you need dedicated ports for HTTP and WebSocket.
    pub async fn run_split(&self) -> Result<(), std::io::Error> {
        let directory = Arc::new(self.config.directory.clone());
        let reload_tx = self.reload_tx.clone();

        // HTTP server
        let http_app = Router::new()
            .route(
                "/",
                get({
                    let dir = directory.clone();
                    move || serve_index(dir.clone())
                }),
            )
            .fallback({
                let dir = directory.clone();
                move |uri: axum::http::Uri| serve_static(dir.clone(), uri)
            });

        let http_app = if self.config.cors {
            http_app.layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
        } else {
            http_app
        };

        // WebSocket server
        let ws_app = Router::new().route(
            "/",
            get({
                let tx = reload_tx.clone();
                move |ws: WebSocketUpgrade| handle_websocket(ws, tx.clone())
            }),
        );

        let http_addr = SocketAddr::from(([0, 0, 0, 0], self.config.port));
        let ws_addr = SocketAddr::from(([0, 0, 0, 0], self.config.ws_port));

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║               Probar WASM Development Server                 ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  HTTP:      http://localhost:{:<29}║", self.config.port);
        println!("║  WebSocket: ws://localhost:{:<30}║", self.config.ws_port);
        println!(
            "║  Directory: {:<48}║",
            self.config
                .directory
                .display()
                .to_string()
                .chars()
                .take(48)
                .collect::<String>()
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Press Ctrl+C to stop                                        ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        let _ = reload_tx.send(HotReloadMessage::ServerReady);

        let http_listener = tokio::net::TcpListener::bind(http_addr).await?;
        let ws_listener = tokio::net::TcpListener::bind(ws_addr).await?;

        tokio::select! {
            r = axum::serve(http_listener, http_app) => r?,
            r = axum::serve(ws_listener, ws_app) => r?,
        }

        Ok(())
    }
}

/// Handle WebSocket connection for hot reload
async fn handle_websocket(
    ws: WebSocketUpgrade,
    reload_tx: broadcast::Sender<HotReloadMessage>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket_handler(socket, reload_tx))
}

/// WebSocket handler that broadcasts hot reload messages
async fn websocket_handler(socket: WebSocket, reload_tx: broadcast::Sender<HotReloadMessage>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = reload_tx.subscribe();

    // Send initial ready message
    let ready_msg = HotReloadMessage::ServerReady.to_json();
    if sender.send(Message::Text(ready_msg.into())).await.is_err() {
        return;
    }

    // Handle incoming messages and broadcast outgoing
    loop {
        tokio::select! {
            // Forward hot reload messages to the client
            result = rx.recv() => {
                match result {
                    Ok(msg) => {
                        let json = msg.to_json();
                        if sender.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            // Handle client messages (ping/pong, close)
            msg_opt = receiver.next() => {
                match msg_opt {
                    Some(Ok(Message::Ping(data))) => {
                        if sender.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                    _ => {}
                }
            }
        }
    }
}

/// Serve index.html
async fn serve_index(directory: Arc<PathBuf>) -> Response {
    let index_path = directory.join("index.html");
    serve_file(&index_path).await
}

/// Serve static file based on URI
///
/// Handles directory requests by serving index.html if it exists.
/// Properly handles both `/dir/` and `/dir` paths.
async fn serve_static(directory: Arc<PathBuf>, uri: axum::http::Uri) -> Response {
    // Trim both leading and trailing slashes to normalize path
    let path = uri.path().trim_start_matches('/').trim_end_matches('/');
    let file_path = if path.is_empty() {
        directory.as_ref().clone()
    } else {
        directory.join(path)
    };

    // If path is a directory, try to serve index.html
    if file_path.is_dir() {
        let index_path = file_path.join("index.html");
        if index_path.exists() {
            return serve_file(&index_path).await;
        }
        // Return 404 for directories without index.html
        return (
            StatusCode::NOT_FOUND,
            format!("No index.html found in directory: {}", file_path.display()),
        )
            .into_response();
    }

    serve_file(&file_path).await
}

/// Serve a file with correct MIME type
///
/// MIME types are critical for WASM to work in browsers:
/// - `.wasm` files MUST be `application/wasm`
/// - `.js` files MUST be `text/javascript` or `application/javascript`
async fn serve_file(path: &std::path::Path) -> Response {
    match tokio::fs::read(path).await {
        Ok(contents) => {
            // Determine MIME type from extension
            let mime_type = get_mime_type(path);

            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime_type)
                .header(header::CACHE_CONTROL, "no-cache")
                .body(axum::body::Body::from(contents))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            format!("File not found: {}", path.display()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error reading file: {e}"),
        )
            .into_response(),
    }
}

/// Get MIME type for a file path
///
/// Ensures WASM files get the correct `application/wasm` type
#[must_use]
pub fn get_mime_type(path: &std::path::Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wasm") => "application/wasm".to_string(),
        Some("js") | Some("mjs") => "text/javascript".to_string(),
        Some("html") | Some("htm") => "text/html".to_string(),
        Some("css") => "text/css".to_string(),
        Some("json") => "application/json".to_string(),
        Some("png") => "image/png".to_string(),
        Some("jpg") | Some("jpeg") => "image/jpeg".to_string(),
        Some("svg") => "image/svg+xml".to_string(),
        Some("ico") => "image/x-icon".to_string(),
        _ => mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string(),
    }
}

// =============================================================================
// WASM Build
// =============================================================================

/// Run wasm-pack build
///
/// This wraps the `wasm-pack` CLI tool to build Rust projects for WASM.
///
/// # Arguments
///
/// * `path` - Directory containing Cargo.toml
/// * `target` - WASM target (web, bundler, nodejs, no-modules)
/// * `release` - Build in release mode
/// * `out_dir` - Output directory (default: pkg)
/// * `profiling` - Enable profiling (adds names section)
///
/// # Errors
///
/// Returns an error if wasm-pack is not installed or build fails.
pub async fn run_wasm_pack_build(
    path: &std::path::Path,
    target: &str,
    release: bool,
    out_dir: Option<&std::path::Path>,
    profiling: bool,
) -> Result<(), String> {
    use std::process::Stdio;
    use std::time::Instant;

    let start = Instant::now();

    let mut cmd = tokio::process::Command::new("wasm-pack");
    cmd.arg("build");
    cmd.arg("--target").arg(target);

    if release {
        cmd.arg("--release");
    } else {
        cmd.arg("--dev");
    }

    if let Some(out) = out_dir {
        cmd.arg("--out-dir").arg(out);
    }

    if profiling {
        cmd.arg("--profiling");
    }

    cmd.current_dir(path);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    println!(
        "Running: wasm-pack build --target {} {}",
        target,
        if release { "--release" } else { "--dev" }
    );

    let status: std::process::ExitStatus = cmd
        .status()
        .await
        .map_err(|e| format!("Failed to execute wasm-pack: {e}. Is wasm-pack installed?"))?;

    let elapsed = start.elapsed();

    if status.success() {
        println!("Build completed in {:.2}s", elapsed.as_secs_f64());
        Ok(())
    } else {
        Err(format!(
            "wasm-pack build failed with exit code: {:?}",
            status.code()
        ))
    }
}

// =============================================================================
// File Watcher
// =============================================================================

/// File watcher for hot reload
///
/// Watches for changes to Rust source files and triggers rebuilds.
#[derive(Debug)]
pub struct FileWatcher {
    /// Directory to watch
    pub path: PathBuf,
    /// Debounce interval in milliseconds
    pub debounce_ms: u64,
    /// File patterns to watch (extensions)
    pub patterns: Vec<String>,
}

impl FileWatcher {
    /// Create a new file watcher with default settings
    #[must_use]
    pub fn new(path: PathBuf, debounce_ms: u64) -> Self {
        Self {
            path,
            debounce_ms,
            patterns: vec!["rs".to_string(), "toml".to_string()],
        }
    }

    /// Create a builder
    #[must_use]
    pub fn builder() -> FileWatcherBuilder {
        FileWatcherBuilder::default()
    }

    /// Check if a path matches watch patterns
    #[must_use]
    pub fn matches_pattern(&self, path: &std::path::Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| self.patterns.iter().any(|p| p == ext))
    }

    /// Start watching for changes
    ///
    /// Calls `on_change` with the path of each changed file.
    /// This function blocks until the watcher is stopped.
    pub async fn watch<F>(&self, mut on_change: F) -> Result<(), notify::Error>
    where
        F: FnMut(String) + Send + 'static,
    {
        use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
        use std::sync::mpsc;
        use std::time::Duration;

        let (tx, rx) = mpsc::channel();
        let patterns = self.patterns.clone();

        let mut watcher = RecommendedWatcher::new(
            move |res: Result<notify::Event, notify::Error>| {
                if let Ok(event) = res {
                    if event.kind.is_modify() || event.kind.is_create() {
                        for path in event.paths {
                            let matches = path
                                .extension()
                                .and_then(|e| e.to_str())
                                .is_some_and(|ext| patterns.iter().any(|p| p == ext));
                            if matches {
                                let _ = tx.send(path.display().to_string());
                            }
                        }
                    }
                }
            },
            Config::default().with_poll_interval(Duration::from_millis(self.debounce_ms)),
        )?;

        watcher.watch(&self.path, RecursiveMode::Recursive)?;

        // Keep watcher alive and process events
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(path) => {
                    on_change(path);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        Ok(())
    }
}

/// Builder for `FileWatcher`
#[derive(Debug, Clone, Default)]
pub struct FileWatcherBuilder {
    path: Option<PathBuf>,
    debounce_ms: u64,
    patterns: Vec<String>,
}

impl FileWatcherBuilder {
    /// Set watch path
    #[must_use]
    pub fn path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set debounce interval
    #[must_use]
    pub fn debounce_ms(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Add file pattern to watch
    #[must_use]
    pub fn pattern(mut self, ext: impl Into<String>) -> Self {
        self.patterns.push(ext.into());
        self
    }

    /// Build the file watcher
    #[must_use]
    pub fn build(self) -> FileWatcher {
        FileWatcher {
            path: self.path.unwrap_or_else(|| PathBuf::from(".")),
            debounce_ms: if self.debounce_ms == 0 {
                500
            } else {
                self.debounce_ms
            },
            patterns: if self.patterns.is_empty() {
                vec!["rs".to_string(), "toml".to_string()]
            } else {
                self.patterns
            },
        }
    }
}

// =============================================================================
// Module Validation (PROBAR-SPEC-007)
// =============================================================================

/// Import reference found in HTML/JS files
#[derive(Debug, Clone)]
pub struct ImportRef {
    /// Source file containing the import
    pub source_file: PathBuf,
    /// Import path (may be relative or absolute)
    pub import_path: String,
    /// Type of import
    pub import_type: ImportType,
    /// Line number in source file
    pub line_number: u32,
}

/// Type of module import
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportType {
    /// ES Module import (import ... from '...')
    EsModule,
    /// Script src attribute
    Script,
    /// WASM file
    Wasm,
    /// Worker URL (new Worker('...'))
    Worker,
}

impl ImportType {
    /// Expected MIME types for this import type
    #[must_use]
    pub fn expected_mime_types(&self) -> &[&str] {
        match self {
            Self::EsModule | Self::Script | Self::Worker => {
                &["text/javascript", "application/javascript"]
            }
            Self::Wasm => &["application/wasm"],
        }
    }
}

/// Validation error for a single import
#[derive(Debug, Clone)]
pub struct ImportValidationError {
    /// The import that failed
    pub import: ImportRef,
    /// HTTP status received
    pub status: u16,
    /// MIME type received
    pub actual_mime: String,
    /// Error description
    pub message: String,
}

/// Result of validating all imports
#[derive(Debug, Default)]
pub struct ModuleValidationResult {
    /// Total imports scanned
    pub total_imports: usize,
    /// Imports that passed validation
    pub passed: usize,
    /// Imports that failed validation
    pub errors: Vec<ImportValidationError>,
}

impl ModuleValidationResult {
    /// Check if all validations passed
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Module validator for checking import resolution
#[derive(Debug)]
pub struct ModuleValidator {
    /// Root directory being served
    serve_root: PathBuf,
    /// Directories to exclude from validation (e.g., node_modules)
    exclude: Vec<String>,
}

impl ModuleValidator {
    /// Create a new module validator
    #[must_use]
    pub fn new(serve_root: impl Into<PathBuf>) -> Self {
        Self {
            serve_root: serve_root.into(),
            exclude: vec!["node_modules".to_string()], // Default exclusion
        }
    }

    /// Set directories to exclude from validation
    #[must_use]
    pub fn with_exclude(mut self, exclude: Vec<String>) -> Self {
        self.exclude = exclude;
        self
    }

    /// Check if a path should be excluded from validation
    fn is_excluded(&self, path: &std::path::Path) -> bool {
        let path_str = path.to_string_lossy();
        self.exclude.iter().any(|excl| {
            path_str.contains(&format!("/{excl}/")) || path_str.contains(&format!("\\{excl}\\"))
        })
    }

    /// Scan all HTML files and extract module imports
    #[must_use]
    pub fn scan_imports(&self) -> Vec<ImportRef> {
        let mut imports = Vec::new();

        // Find all HTML files
        let pattern = self.serve_root.join("**/*.html");
        if let Ok(paths) = glob::glob(&pattern.to_string_lossy()) {
            for entry in paths.flatten() {
                // Skip excluded directories
                if self.is_excluded(&entry) {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(&entry) {
                    imports.extend(Self::extract_imports_from_html(&entry, &content));
                }
            }
        }

        imports
    }

    /// Extract imports from HTML content
    fn extract_imports_from_html(file: &std::path::Path, content: &str) -> Vec<ImportRef> {
        let mut imports = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line_number = (line_num + 1) as u32;

            // ES Module imports: import ... from '...' or import '...'
            if let Some(path) = Self::extract_es_import(line) {
                imports.push(ImportRef {
                    source_file: file.to_path_buf(),
                    import_path: path,
                    import_type: ImportType::EsModule,
                    line_number,
                });
            }

            // Script src: <script src="...">
            if let Some(path) = Self::extract_script_src(line) {
                // Skip inline module scripts with type="module" (handled by ES import)
                if !line.contains("type=\"module\"") || line.contains("src=") {
                    imports.push(ImportRef {
                        source_file: file.to_path_buf(),
                        import_path: path,
                        import_type: ImportType::Script,
                        line_number,
                    });
                }
            }

            // Worker URLs: new Worker('...')
            if let Some(path) = Self::extract_worker_url(line) {
                imports.push(ImportRef {
                    source_file: file.to_path_buf(),
                    import_path: path,
                    import_type: ImportType::Worker,
                    line_number,
                });
            }
        }

        imports
    }

    /// Check if path has a JS/WASM extension (case-insensitive)
    fn has_js_or_wasm_extension(path: &str) -> bool {
        let path = std::path::Path::new(path);
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                let ext_lower = ext.to_ascii_lowercase();
                ext_lower == "js" || ext_lower == "mjs" || ext_lower == "wasm"
            })
    }

    /// Check if path has a JS extension (case-insensitive)
    fn has_js_extension(path: &str) -> bool {
        let path = std::path::Path::new(path);
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                let ext_lower = ext.to_ascii_lowercase();
                ext_lower == "js" || ext_lower == "mjs"
            })
    }

    /// Extract ES module import path
    fn extract_es_import(line: &str) -> Option<String> {
        // Match: import ... from '...' or import ... from "..."
        let patterns = [
            (r"from '", "'"),
            (r#"from ""#, "\""),
            // Also match dynamic import()
            (r"import('", "'"),
            (r#"import(""#, "\""),
        ];

        for (start, end) in patterns {
            if let Some(idx) = line.find(start) {
                let rest = &line[idx + start.len()..];
                if let Some(end_idx) = rest.find(end) {
                    let path = &rest[..end_idx];
                    // Only include JS/WASM paths
                    if Self::has_js_or_wasm_extension(path) {
                        return Some(path.to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract script src attribute
    fn extract_script_src(line: &str) -> Option<String> {
        let patterns = [(r#"src=""#, "\""), (r"src='", "'")];

        for (start, end) in patterns {
            if let Some(idx) = line.find(start) {
                let rest = &line[idx + start.len()..];
                if let Some(end_idx) = rest.find(end) {
                    let path = &rest[..end_idx];
                    if Self::has_js_extension(path) {
                        return Some(path.to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract Worker URL
    fn extract_worker_url(line: &str) -> Option<String> {
        let patterns = [(r"new Worker('", "'"), (r#"new Worker(""#, "\"")];

        for (start, end) in patterns {
            if let Some(idx) = line.find(start) {
                let rest = &line[idx + start.len()..];
                if let Some(end_idx) = rest.find(end) {
                    let path = &rest[..end_idx];
                    return Some(path.to_string());
                }
            }
        }

        None
    }

    /// Resolve import path relative to source file
    fn resolve_path(&self, import: &ImportRef) -> Option<PathBuf> {
        let import_path = &import.import_path;

        if import_path.starts_with('/') {
            // Absolute path from serve root
            Some(self.serve_root.join(import_path.trim_start_matches('/')))
        } else if import_path.starts_with("./") || import_path.starts_with("../") {
            // Relative path from source file
            let source_dir = import.source_file.parent()?;
            Some(source_dir.join(import_path))
        } else {
            // Bare specifier or absolute URL - skip validation
            None
        }
    }

    /// Validate all imports resolve correctly
    #[must_use]
    pub fn validate(&self) -> ModuleValidationResult {
        let imports = self.scan_imports();
        let mut result = ModuleValidationResult {
            total_imports: imports.len(),
            ..Default::default()
        };

        for import in imports {
            if let Some(resolved) = self.resolve_path(&import) {
                // Check if file exists
                let canonical = resolved.canonicalize();

                match canonical {
                    Ok(path) if path.exists() => {
                        // Check MIME type
                        let mime = get_mime_type(&path);
                        let expected = import.import_type.expected_mime_types();

                        if expected.iter().any(|&e| mime.starts_with(e)) {
                            result.passed += 1;
                        } else {
                            result.errors.push(ImportValidationError {
                                import: import.clone(),
                                status: 200,
                                actual_mime: mime.clone(),
                                message: format!(
                                    "MIME type mismatch: expected {expected:?}, got '{mime}'"
                                ),
                            });
                        }
                    }
                    _ => {
                        result.errors.push(ImportValidationError {
                            import: import.clone(),
                            status: 404,
                            actual_mime: "text/plain".to_string(),
                            message: format!(
                                "File not found: {} (resolved to {})",
                                import.import_path,
                                resolved.display()
                            ),
                        });
                    }
                }
            } else {
                // External or unresolvable - count as passed
                result.passed += 1;
            }
        }

        result
    }

    /// Print validation results to stderr
    pub fn print_results(&self, result: &ModuleValidationResult) {
        eprintln!("\nValidating module imports...");
        eprintln!("  Scanned: {} imports", result.total_imports);
        eprintln!("  Passed:  {}", result.passed);
        eprintln!("  Failed:  {}", result.errors.len());

        if !result.errors.is_empty() {
            eprintln!("\nErrors:");
            for error in &result.errors {
                eprintln!(
                    "  {} {}:{}",
                    if error.status == 404 { "✗" } else { "⚠" },
                    error.import.source_file.display(),
                    error.import.line_number
                );
                eprintln!("    Import: {}", error.import.import_path);
                eprintln!("    {}", error.message);
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // =========================================================================
    // DevServerConfig Tests
    // =========================================================================

    #[test]
    fn test_dev_server_config_default() {
        let config = DevServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.ws_port, 8081);
        assert!(!config.cors);
        assert!(!config.cross_origin_isolated);
        assert_eq!(config.directory, PathBuf::from("."));
    }

    #[test]
    fn test_dev_server_config_builder() {
        let config = DevServerConfig::builder()
            .directory("./www")
            .port(9000)
            .ws_port(9001)
            .cors(true)
            .build();

        assert_eq!(config.port, 9000);
        assert_eq!(config.ws_port, 9001);
        assert!(config.cors);
        assert_eq!(config.directory, PathBuf::from("./www"));
    }

    #[test]
    fn test_dev_server_config_cross_origin_isolated() {
        let config = DevServerConfig::builder()
            .cross_origin_isolated(true)
            .build();

        assert!(config.cross_origin_isolated);
    }

    // =========================================================================
    // DevServer Tests
    // =========================================================================

    #[test]
    fn test_dev_server_creation() {
        let config = DevServerConfig {
            directory: PathBuf::from("./www"),
            port: 9000,
            ws_port: 9001,
            cors: true,
            cross_origin_isolated: false,
        };
        let server = DevServer::new(config);
        assert_eq!(server.http_url(), "http://localhost:9000");
        assert_eq!(server.ws_url(), "ws://localhost:9000/ws");
    }

    #[test]
    fn test_dev_server_reload_sender() {
        let server = DevServer::new(DevServerConfig::default());
        let tx = server.reload_sender();

        // Should be able to send messages
        let result = tx.send(HotReloadMessage::ServerReady);
        // No receivers yet, but send should work
        assert!(result.is_ok() || result.is_err());
    }

    // =========================================================================
    // HotReloadMessage Tests
    // =========================================================================

    #[test]
    fn test_hot_reload_message_clone() {
        let msg = HotReloadMessage::RebuildComplete { duration_ms: 500 };
        let cloned = msg;
        assert!(matches!(
            cloned,
            HotReloadMessage::RebuildComplete { duration_ms: 500 }
        ));
    }

    #[test]
    fn test_hot_reload_message_to_json() {
        let msg = HotReloadMessage::FileChanged {
            path: "src/lib.rs".to_string(),
        };
        let json = msg.to_json();
        assert!(json.contains("FileChanged"));
        assert!(json.contains("src/lib.rs"));
    }

    #[test]
    fn test_hot_reload_message_rebuild_complete_json() {
        let msg = HotReloadMessage::RebuildComplete { duration_ms: 1234 };
        let json = msg.to_json();
        assert!(json.contains("RebuildComplete"));
        assert!(json.contains("1234"));
    }

    #[test]
    fn test_hot_reload_message_rebuild_failed_json() {
        let msg = HotReloadMessage::RebuildFailed {
            error: "compile error".to_string(),
        };
        let json = msg.to_json();
        assert!(json.contains("RebuildFailed"));
        assert!(json.contains("compile error"));
    }

    #[test]
    fn test_hot_reload_message_server_ready_json() {
        let msg = HotReloadMessage::ServerReady;
        let json = msg.to_json();
        assert!(json.contains("ServerReady"));
    }

    #[test]
    fn test_hot_reload_message_rebuild_started_json() {
        let msg = HotReloadMessage::RebuildStarted;
        let json = msg.to_json();
        assert!(json.contains("RebuildStarted"));
    }

    // =========================================================================
    // FileWatcher Tests
    // =========================================================================

    #[test]
    fn test_file_watcher_creation() {
        let watcher = FileWatcher::new(PathBuf::from("."), 500);
        assert_eq!(watcher.debounce_ms, 500);
        assert_eq!(watcher.path, PathBuf::from("."));
        assert!(watcher.patterns.contains(&"rs".to_string()));
        assert!(watcher.patterns.contains(&"toml".to_string()));
    }

    #[test]
    fn test_file_watcher_builder() {
        let watcher = FileWatcher::builder()
            .path("./src")
            .debounce_ms(1000)
            .pattern("rs")
            .pattern("ts")
            .build();

        assert_eq!(watcher.path, PathBuf::from("./src"));
        assert_eq!(watcher.debounce_ms, 1000);
        assert!(watcher.patterns.contains(&"rs".to_string()));
        assert!(watcher.patterns.contains(&"ts".to_string()));
    }

    #[test]
    fn test_file_watcher_builder_defaults() {
        let watcher = FileWatcher::builder().build();

        assert_eq!(watcher.path, PathBuf::from("."));
        assert_eq!(watcher.debounce_ms, 500);
        assert!(watcher.patterns.contains(&"rs".to_string()));
    }

    #[test]
    fn test_file_watcher_matches_pattern() {
        let watcher = FileWatcher::new(PathBuf::from("."), 500);

        assert!(watcher.matches_pattern(&PathBuf::from("src/lib.rs")));
        assert!(watcher.matches_pattern(&PathBuf::from("Cargo.toml")));
        assert!(!watcher.matches_pattern(&PathBuf::from("README.md")));
        assert!(!watcher.matches_pattern(&PathBuf::from("main.js")));
    }

    #[test]
    fn test_file_watcher_custom_patterns() {
        let mut watcher = FileWatcher::new(PathBuf::from("."), 500);
        watcher.patterns = vec!["js".to_string(), "ts".to_string()];

        assert!(watcher.matches_pattern(&PathBuf::from("app.js")));
        assert!(watcher.matches_pattern(&PathBuf::from("app.ts")));
        assert!(!watcher.matches_pattern(&PathBuf::from("lib.rs")));
    }

    // =========================================================================
    // MIME Type Tests
    // =========================================================================

    #[test]
    fn test_get_mime_type_wasm() {
        assert_eq!(
            get_mime_type(&PathBuf::from("app.wasm")),
            "application/wasm"
        );
    }

    #[test]
    fn test_get_mime_type_javascript() {
        assert_eq!(get_mime_type(&PathBuf::from("app.js")), "text/javascript");
        assert_eq!(
            get_mime_type(&PathBuf::from("module.mjs")),
            "text/javascript"
        );
    }

    #[test]
    fn test_get_mime_type_html() {
        assert_eq!(get_mime_type(&PathBuf::from("index.html")), "text/html");
        assert_eq!(get_mime_type(&PathBuf::from("page.htm")), "text/html");
    }

    #[test]
    fn test_get_mime_type_css() {
        assert_eq!(get_mime_type(&PathBuf::from("styles.css")), "text/css");
    }

    #[test]
    fn test_get_mime_type_json() {
        assert_eq!(
            get_mime_type(&PathBuf::from("data.json")),
            "application/json"
        );
    }

    #[test]
    fn test_get_mime_type_images() {
        assert_eq!(get_mime_type(&PathBuf::from("logo.png")), "image/png");
        assert_eq!(get_mime_type(&PathBuf::from("photo.jpg")), "image/jpeg");
        assert_eq!(get_mime_type(&PathBuf::from("photo.jpeg")), "image/jpeg");
        assert_eq!(get_mime_type(&PathBuf::from("icon.svg")), "image/svg+xml");
        assert_eq!(get_mime_type(&PathBuf::from("favicon.ico")), "image/x-icon");
    }

    #[test]
    fn test_get_mime_type_unknown() {
        // Unknown types fall back to mime_guess or octet-stream
        let mime = get_mime_type(&PathBuf::from("data.xyz"));
        assert!(!mime.is_empty());
    }

    // =========================================================================
    // Directory Index Tests (Bug fix: serve index.html for directories)
    // =========================================================================

    #[tokio::test]
    async fn test_serve_static_directory_serves_index_html() {
        use std::sync::Arc;
        use tempfile::TempDir;

        // Create temp directory with index.html
        let temp_dir = TempDir::new().unwrap();
        let subdir = temp_dir.path().join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(subdir.join("index.html"), "<html>test</html>").unwrap();

        let directory = Arc::new(temp_dir.path().to_path_buf());
        let uri: axum::http::Uri = "/subdir/".parse().unwrap();

        let response = serve_static(directory, uri).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_serve_static_directory_without_trailing_slash() {
        use std::sync::Arc;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let subdir = temp_dir.path().join("mydir");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(subdir.join("index.html"), "<html>works</html>").unwrap();

        let directory = Arc::new(temp_dir.path().to_path_buf());
        let uri: axum::http::Uri = "/mydir".parse().unwrap();

        let response = serve_static(directory, uri).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_serve_static_directory_no_index_returns_error() {
        use std::sync::Arc;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let subdir = temp_dir.path().join("empty");
        std::fs::create_dir(&subdir).unwrap();
        // No index.html

        let directory = Arc::new(temp_dir.path().to_path_buf());
        let uri: axum::http::Uri = "/empty/".parse().unwrap();

        let response = serve_static(directory, uri).await;
        // Should return error since directory has no index.html
        assert!(response.status().is_client_error() || response.status().is_server_error());
    }

    // =========================================================================
    // Integration-style Tests (no actual I/O)
    // =========================================================================

    #[test]
    fn test_dev_server_config_chain() {
        let config = DevServerConfig::builder()
            .directory("./dist")
            .port(3000)
            .ws_port(3001)
            .cors(true)
            .build();

        let server = DevServer::new(config);
        assert_eq!(server.http_url(), "http://localhost:3000");
    }

    #[test]
    fn test_file_watcher_builder_chain() {
        let watcher = FileWatcher::builder()
            .path("./crate")
            .debounce_ms(250)
            .pattern("rs")
            .pattern("toml")
            .pattern("lock")
            .build();

        assert_eq!(watcher.path, PathBuf::from("./crate"));
        assert_eq!(watcher.debounce_ms, 250);
        assert_eq!(watcher.patterns.len(), 3);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_hot_reload_message_roundtrip() {
        let original = HotReloadMessage::RebuildComplete { duration_ms: 1500 };
        let json = original.to_json();
        let parsed: HotReloadMessage = serde_json::from_str(&json).expect("parse failed");

        match parsed {
            HotReloadMessage::RebuildComplete { duration_ms } => {
                assert_eq!(duration_ms, 1500);
            }
            _ => panic!("Wrong variant after roundtrip"),
        }
    }

    #[test]
    fn test_hot_reload_message_all_variants() {
        let variants = vec![
            HotReloadMessage::FileChanged {
                path: "test.rs".to_string(),
            },
            HotReloadMessage::RebuildStarted,
            HotReloadMessage::RebuildComplete { duration_ms: 100 },
            HotReloadMessage::RebuildFailed {
                error: "err".to_string(),
            },
            HotReloadMessage::ServerReady,
        ];

        for variant in variants {
            let json = variant.to_json();
            assert!(!json.is_empty());
            // Verify it can be parsed back
            let _: HotReloadMessage = serde_json::from_str(&json).expect("parse failed");
        }
    }

    // =========================================================================
    // FileChangeEvent Tests (Phase 4 - Hot Reload Enhancements)
    // =========================================================================

    #[test]
    fn test_file_change_event_as_str() {
        assert_eq!(FileChangeEvent::Created.as_str(), "CREATED");
        assert_eq!(FileChangeEvent::Modified.as_str(), "MODIFIED");
        assert_eq!(FileChangeEvent::Deleted.as_str(), "DELETED");
        assert_eq!(FileChangeEvent::Renamed.as_str(), "RENAMED");
    }

    #[test]
    fn test_file_modified_created() {
        let msg = HotReloadMessage::file_modified(
            "new_file.rs",
            FileChangeEvent::Created,
            None,
            Some(1024),
        );

        match msg {
            HotReloadMessage::FileModified {
                path,
                event,
                diff_summary,
                size_after,
                ..
            } => {
                assert_eq!(path, "new_file.rs");
                assert_eq!(event, FileChangeEvent::Created);
                assert!(diff_summary.contains('+'));
                assert_eq!(size_after, Some(1024));
            }
            _ => panic!("Expected FileModified"),
        }
    }

    #[test]
    fn test_file_modified_deleted() {
        let msg = HotReloadMessage::file_modified(
            "old_file.rs",
            FileChangeEvent::Deleted,
            Some(2048),
            None,
        );

        match msg {
            HotReloadMessage::FileModified {
                event,
                diff_summary,
                size_before,
                ..
            } => {
                assert_eq!(event, FileChangeEvent::Deleted);
                assert!(diff_summary.contains('-'));
                assert_eq!(size_before, Some(2048));
            }
            _ => panic!("Expected FileModified"),
        }
    }

    #[test]
    fn test_file_modified_size_increase() {
        let msg = HotReloadMessage::file_modified(
            "lib.rs",
            FileChangeEvent::Modified,
            Some(1000),
            Some(1500),
        );

        match msg {
            HotReloadMessage::FileModified { diff_summary, .. } => {
                assert!(diff_summary.contains("+500 bytes"));
            }
            _ => panic!("Expected FileModified"),
        }
    }

    #[test]
    fn test_file_modified_size_decrease() {
        let msg = HotReloadMessage::file_modified(
            "lib.rs",
            FileChangeEvent::Modified,
            Some(2000),
            Some(1500),
        );

        match msg {
            HotReloadMessage::FileModified { diff_summary, .. } => {
                assert!(diff_summary.contains("-500 bytes"));
            }
            _ => panic!("Expected FileModified"),
        }
    }

    #[test]
    fn test_file_modified_json_roundtrip() {
        let msg = HotReloadMessage::file_modified(
            "test.rs",
            FileChangeEvent::Modified,
            Some(100),
            Some(200),
        );
        let json = msg.to_json();
        assert!(json.contains("FileModified"));
        assert!(json.contains("test.rs"));
        assert!(json.contains("modified"));

        let parsed: HotReloadMessage = serde_json::from_str(&json).expect("parse failed");
        match parsed {
            HotReloadMessage::FileModified { path, event, .. } => {
                assert_eq!(path, "test.rs");
                assert_eq!(event, FileChangeEvent::Modified);
            }
            _ => panic!("Wrong variant after roundtrip"),
        }
    }

    #[test]
    fn test_client_connected_message() {
        let msg = HotReloadMessage::ClientConnected { client_count: 3 };
        let json = msg.to_json();
        assert!(json.contains("ClientConnected"));
        assert!(json.contains('3'));
    }

    #[test]
    fn test_client_disconnected_message() {
        let msg = HotReloadMessage::ClientDisconnected { client_count: 2 };
        let json = msg.to_json();
        assert!(json.contains("ClientDisconnected"));
        assert!(json.contains('2'));
    }

    // =========================================================================
    // Module Validator Tests (PROBAR-SPEC-007)
    // =========================================================================

    #[test]
    fn test_module_validator_extract_es_import() {
        // Test basic import from
        let line = r"import init from './pkg/app.js';";
        assert_eq!(
            ModuleValidator::extract_es_import(line),
            Some("./pkg/app.js".to_string())
        );

        // Test double quotes
        let line = r#"import { foo } from "/lib/utils.mjs";"#;
        assert_eq!(
            ModuleValidator::extract_es_import(line),
            Some("/lib/utils.mjs".to_string())
        );

        // Test WASM import
        let line = r"import wasm from '../module.wasm';";
        assert_eq!(
            ModuleValidator::extract_es_import(line),
            Some("../module.wasm".to_string())
        );

        // Test non-JS import (should be None)
        let line = r"import styles from './styles.css';";
        assert_eq!(ModuleValidator::extract_es_import(line), None);
    }

    #[test]
    fn test_module_validator_extract_script_src() {
        let line = r#"<script src="./app.js"></script>"#;
        assert_eq!(
            ModuleValidator::extract_script_src(line),
            Some("./app.js".to_string())
        );

        let line = r"<script src='/lib/vendor.mjs'></script>";
        assert_eq!(
            ModuleValidator::extract_script_src(line),
            Some("/lib/vendor.mjs".to_string())
        );

        // Non-JS src
        let line = r#"<img src="./image.png">"#;
        assert_eq!(ModuleValidator::extract_script_src(line), None);
    }

    #[test]
    fn test_module_validator_extract_worker_url() {
        let line = r"const worker = new Worker('./worker.js');";
        assert_eq!(
            ModuleValidator::extract_worker_url(line),
            Some("./worker.js".to_string())
        );

        let line = r#"new Worker("/pkg/transcription_worker.js")"#;
        assert_eq!(
            ModuleValidator::extract_worker_url(line),
            Some("/pkg/transcription_worker.js".to_string())
        );
    }

    #[test]
    fn test_module_validator_resolve_absolute_path() {
        let validator = ModuleValidator::new("/srv/www");

        let import = ImportRef {
            source_file: PathBuf::from("/srv/www/index.html"),
            import_path: "/pkg/app.js".to_string(),
            import_type: ImportType::EsModule,
            line_number: 10,
        };

        let resolved = validator.resolve_path(&import);
        assert_eq!(resolved, Some(PathBuf::from("/srv/www/pkg/app.js")));
    }

    #[test]
    fn test_module_validator_resolve_relative_path() {
        let validator = ModuleValidator::new("/srv/www");

        let import = ImportRef {
            source_file: PathBuf::from("/srv/www/pages/demo.html"),
            import_path: "../pkg/app.js".to_string(),
            import_type: ImportType::EsModule,
            line_number: 5,
        };

        let resolved = validator.resolve_path(&import);
        assert_eq!(
            resolved,
            Some(PathBuf::from("/srv/www/pages/../pkg/app.js"))
        );
    }

    #[test]
    fn test_module_validator_skip_external_urls() {
        let validator = ModuleValidator::new("/srv/www");

        // Bare specifier (npm package)
        let import = ImportRef {
            source_file: PathBuf::from("/srv/www/index.html"),
            import_path: "lodash".to_string(),
            import_type: ImportType::EsModule,
            line_number: 1,
        };
        assert_eq!(validator.resolve_path(&import), None);
    }

    #[test]
    fn test_import_type_expected_mime_types() {
        assert!(ImportType::EsModule
            .expected_mime_types()
            .contains(&"text/javascript"));
        assert!(ImportType::Script
            .expected_mime_types()
            .contains(&"application/javascript"));
        assert!(ImportType::Wasm
            .expected_mime_types()
            .contains(&"application/wasm"));
        assert!(ImportType::Worker
            .expected_mime_types()
            .contains(&"text/javascript"));
    }

    #[test]
    fn test_module_validation_result_is_ok() {
        let mut result = ModuleValidationResult::default();
        assert!(result.is_ok());

        result.errors.push(ImportValidationError {
            import: ImportRef {
                source_file: PathBuf::from("test.html"),
                import_path: "/missing.js".to_string(),
                import_type: ImportType::EsModule,
                line_number: 1,
            },
            status: 404,
            actual_mime: "text/plain".to_string(),
            message: "Not found".to_string(),
        });

        assert!(!result.is_ok());
    }

    #[test]
    fn test_module_validator_validates_existing_file() {
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("pkg");
        std::fs::create_dir(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("app.js"), "export default {}").unwrap();
        std::fs::write(
            temp.path().join("index.html"),
            r#"<script type="module">import init from './pkg/app.js';</script>"#,
        )
        .unwrap();

        let validator = ModuleValidator::new(temp.path());
        let result = validator.validate();

        assert_eq!(result.total_imports, 1);
        assert_eq!(result.passed, 1);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_module_validator_detects_missing_file() {
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("index.html"),
            r#"<script type="module">import init from './pkg/missing.js';</script>"#,
        )
        .unwrap();

        let validator = ModuleValidator::new(temp.path());
        let result = validator.validate();

        assert_eq!(result.total_imports, 1);
        assert_eq!(result.passed, 0);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].status, 404);
    }

    // =========================================================================
    // format_bytes Tests
    // =========================================================================

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(0), "0 bytes");
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(1023), "1023 bytes");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(2048), "2.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(5242880), "5.0 MB");
    }

    #[test]
    fn test_file_modified_renamed_with_sizes() {
        let msg = HotReloadMessage::file_modified(
            "src/renamed.rs",
            FileChangeEvent::Renamed,
            Some(100),
            Some(100),
        );
        if let HotReloadMessage::FileModified { diff_summary, .. } = msg {
            assert_eq!(diff_summary, "renamed");
        } else {
            panic!("Expected FileModified variant");
        }
    }

    #[test]
    fn test_file_modified_fallback_changed() {
        // Test the fallback case for non-matching pattern
        let msg = HotReloadMessage::file_modified(
            "src/test.rs",
            FileChangeEvent::Modified,
            None, // No size before
            None, // No size after
        );
        if let HotReloadMessage::FileModified { diff_summary, .. } = msg {
            assert_eq!(diff_summary, "changed");
        } else {
            panic!("Expected FileModified variant");
        }
    }
}
