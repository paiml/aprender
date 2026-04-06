//! Request audit logging for Banco.
//!
//! Logs every API request as a JSON line to an in-memory buffer.
//! Phase 1: in-memory ring buffer. Phase 2+: append to `~/.banco/audit.jsonl`.

use axum::{
    body::Body,
    http::{Method, Request, Response, StatusCode},
    middleware::Next,
};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// A single audit log entry.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    pub ts: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub latency_ms: u64,
}

/// In-memory audit log (ring buffer, max 10,000 entries) with optional disk persistence.
#[derive(Debug, Clone)]
pub struct AuditLog {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
    /// Path to append audit entries as JSONL (None = in-memory only).
    log_path: Option<std::path::PathBuf>,
}

const MAX_ENTRIES: usize = 10_000;

impl AuditLog {
    #[must_use]
    pub fn new() -> Self {
        Self { entries: Arc::new(Mutex::new(Vec::with_capacity(256))), log_path: None }
    }

    /// Create an audit log that also appends to a JSONL file.
    #[must_use]
    pub fn with_file(path: std::path::PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Self { entries: Arc::new(Mutex::new(Vec::with_capacity(256))), log_path: Some(path) }
    }

    pub fn push(&self, entry: AuditEntry) {
        // Append to disk if configured
        if let Some(ref path) = self.log_path {
            if let Ok(json) = serde_json::to_string(&entry) {
                let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(
                    |mut f| {
                        use std::io::Write;
                        writeln!(f, "{json}")
                    },
                );
            }
        }

        if let Ok(mut entries) = self.entries.lock() {
            if entries.len() >= MAX_ENTRIES {
                entries.remove(0);
            }
            entries.push(entry);
        }
    }

    #[must_use]
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        self.entries
            .lock()
            .map(|e| e.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the log file path (if configured).
    #[must_use]
    pub fn log_path(&self) -> Option<&std::path::Path> {
        self.log_path.as_deref()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Axum middleware that logs every request to the audit log.
pub async fn audit_layer(
    audit_log: AuditLog,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    let response = next.run(request).await;

    let entry = AuditEntry {
        ts: iso_now(),
        method: method.to_string(),
        path,
        status: response.status().as_u16(),
        latency_ms: start.elapsed().as_millis() as u64,
    };
    audit_log.push(entry);

    response
}

fn iso_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    // Simple ISO-ish format without chrono dependency
    format!("{secs}")
}
