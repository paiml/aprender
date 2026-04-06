//! Debug Mode Implementation
//!
//! Provides verbose request/response logging and step-by-step playback
//! for debugging WASM applications.
//!
//! ## Features
//!
//! - Request/response tracing with full headers
//! - File resolution debugging (shows which rules matched)
//! - CORS and COOP/COEP header visibility
//! - Suggestions for common issues (404s, MIME types)

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Debug verbosity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugVerbosity {
    /// Errors only
    Minimal,
    /// Errors + warnings
    Normal,
    /// All requests/responses
    #[default]
    Verbose,
    /// Everything including internal state
    Trace,
}

/// Debug event category
#[derive(Debug, Clone, Copy)]
pub enum DebugCategory {
    /// Server lifecycle events
    Server,
    /// Incoming requests
    Request,
    /// File resolution
    Resolve,
    /// Outgoing responses
    Response,
    /// Errors
    Error,
    /// WebSocket events
    WebSocket,
    /// File watcher events
    Watcher,
}

impl DebugCategory {
    /// Get the display string for this category
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Server => "SERVER",
            Self::Request => "REQUEST",
            Self::Resolve => "RESOLVE",
            Self::Response => "RESPONSE",
            Self::Error => "ERROR",
            Self::WebSocket => "WS",
            Self::Watcher => "WATCHER",
        }
    }

    /// Get ANSI color code for this category
    #[must_use]
    pub const fn color(&self) -> &'static str {
        match self {
            Self::Server => "\x1b[36m",    // Cyan
            Self::Request => "\x1b[34m",   // Blue
            Self::Resolve => "\x1b[35m",   // Magenta
            Self::Response => "\x1b[32m",  // Green
            Self::Error => "\x1b[31m",     // Red
            Self::WebSocket => "\x1b[33m", // Yellow
            Self::Watcher => "\x1b[90m",   // Gray
        }
    }
}

/// Resolution rule that matched a request
#[derive(Debug, Clone, Copy)]
pub enum ResolutionRule {
    /// Served index.html for directory
    DirectoryIndex,
    /// Served static file directly
    StaticFile,
    /// Fallback to default
    Fallback,
    /// File not found
    NotFound,
}

impl ResolutionRule {
    /// Get display string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::DirectoryIndex => "Directory index (index.html)",
            Self::StaticFile => "Static file",
            Self::Fallback => "Fallback",
            Self::NotFound => "Not found",
        }
    }
}

/// Debug tracer for the development server
#[derive(Debug)]
pub struct DebugTracer {
    /// Whether debug mode is enabled
    enabled: bool,
    /// Verbosity level
    verbosity: DebugVerbosity,
    /// Start time for relative timestamps
    start_time: Instant,
    /// Request counter
    request_count: AtomicU64,
    /// Whether to use colors
    use_colors: bool,
}

impl Default for DebugTracer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl DebugTracer {
    /// Create a new debug tracer
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            verbosity: DebugVerbosity::Verbose,
            start_time: Instant::now(),
            request_count: AtomicU64::new(0),
            use_colors: atty::is(atty::Stream::Stdout),
        }
    }

    /// Create enabled tracer
    #[must_use]
    pub fn enabled() -> Self {
        Self::new(true)
    }

    /// Set verbosity level
    #[must_use]
    pub const fn with_verbosity(mut self, verbosity: DebugVerbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Check if debug mode is enabled
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get elapsed time since start
    fn elapsed_str(&self) -> String {
        let elapsed = self.start_time.elapsed();
        let secs = elapsed.as_secs();
        let millis = elapsed.subsec_millis();
        format!("{secs:02}:{millis:03}")
    }

    /// Format a log line
    fn format_line(&self, category: DebugCategory, message: &str) -> String {
        let timestamp = self.elapsed_str();
        let cat_str = category.as_str();

        if self.use_colors {
            let color = category.color();
            let reset = "\x1b[0m";
            format!("[{timestamp}] {color}{cat_str:8}{reset} │ {message}")
        } else {
            format!("[{timestamp}] {cat_str:8} │ {message}")
        }
    }

    /// Log a debug event
    pub fn log(&self, category: DebugCategory, message: &str) {
        if !self.enabled {
            return;
        }
        println!("{}", self.format_line(category, message));
    }

    /// Log a multi-line debug event
    pub fn log_multi(&self, category: DebugCategory, lines: &[&str]) {
        if !self.enabled || lines.is_empty() {
            return;
        }

        // First line with category
        println!("{}", self.format_line(category, lines[0]));

        // Continuation lines with padding
        let padding = "                      │ ";

        for line in &lines[1..] {
            println!("{padding}{line}");
        }
    }

    /// Log server startup
    pub fn log_server_start(&self, port: u16, directory: &Path, cors: bool, coop_coep: bool) {
        if !self.enabled {
            return;
        }

        println!();
        self.log(DebugCategory::Server, "DEBUG MODE ACTIVE");
        println!("━━━━━━━━━━━━━━━━━");
        println!();

        self.log(
            DebugCategory::Server,
            &format!("Binding to 127.0.0.1:{port}"),
        );
        self.log(DebugCategory::Server, "Registered routes:");
        self.log(
            DebugCategory::Server,
            &format!("  GET / -> {}/index.html", directory.display()),
        );
        self.log(
            DebugCategory::Server,
            &format!("  GET /* -> {} (static)", directory.display()),
        );
        self.log(DebugCategory::Server, "  GET /ws -> WebSocket");

        self.log(
            DebugCategory::Server,
            &format!(
                "CORS headers: {}",
                if cors {
                    "enabled (Access-Control-Allow-Origin: *)"
                } else {
                    "disabled"
                }
            ),
        );

        self.log(
            DebugCategory::Server,
            &format!(
                "COOP/COEP headers: {}",
                if coop_coep {
                    "enabled (SharedArrayBuffer available)"
                } else {
                    "disabled"
                }
            ),
        );

        println!();
    }

    /// Log an incoming request
    pub fn log_request(
        &self,
        method: &str,
        path: &str,
        client_addr: Option<&str>,
        user_agent: Option<&str>,
    ) {
        if !self.enabled {
            return;
        }

        let req_num = self.request_count.fetch_add(1, Ordering::SeqCst) + 1;

        let mut lines = vec![format!("#{req_num} {method} {path}")];

        if let Some(addr) = client_addr {
            lines.push(format!("Client: {addr}"));
        }

        if let Some(ua) = user_agent {
            // Truncate long user agents
            let ua_short = if ua.len() > 50 {
                format!("{}...", &ua[..47])
            } else {
                ua.to_string()
            };
            lines.push(format!("User-Agent: {ua_short}"));
        }

        let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        self.log_multi(DebugCategory::Request, &line_refs);
    }

    /// Log file resolution
    pub fn log_resolve(&self, request_path: &str, resolved_path: &Path, rule: ResolutionRule) {
        if !self.enabled {
            return;
        }

        self.log_multi(
            DebugCategory::Resolve,
            &[
                &format!("Path: {request_path}"),
                &format!("Resolved: {}", resolved_path.display()),
                &format!("Rule: {}", rule.as_str()),
            ],
        );
    }

    /// Log a response
    pub fn log_response(
        &self,
        status: u16,
        content_type: &str,
        content_length: usize,
        latency_ms: u64,
    ) {
        if !self.enabled {
            return;
        }

        let status_str = match status {
            200 => "200 OK",
            304 => "304 Not Modified",
            404 => "404 Not Found",
            500 => "500 Internal Server Error",
            _ => "Unknown",
        };

        self.log_multi(
            DebugCategory::Response,
            &[
                &format!("Status: {status_str}"),
                &format!("Content-Type: {content_type}"),
                &format!("Content-Length: {content_length}"),
                &format!("Latency: {latency_ms}ms"),
            ],
        );
    }

    /// Log a 404 error with suggestions
    pub fn log_not_found(
        &self,
        request_path: &str,
        searched_paths: &[PathBuf],
        suggestions: &[String],
    ) {
        if !self.enabled {
            return;
        }

        let mut lines = vec![
            format!("GET {request_path}"),
            "Error: File not found".to_string(),
        ];

        lines.push("Searched paths:".to_string());
        for (i, path) in searched_paths.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, path.display()));
        }

        if !suggestions.is_empty() {
            lines.push("Suggestions:".to_string());
            for suggestion in suggestions {
                lines.push(format!("  - {suggestion}"));
            }
        }

        let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        self.log_multi(DebugCategory::Error, &line_refs);
    }

    /// Log MIME type information (especially for WASM)
    pub fn log_mime_check(&self, path: &Path, mime_type: &str, is_correct: bool) {
        if !self.enabled {
            return;
        }

        let status = if is_correct {
            "✓ CORRECT"
        } else {
            "✗ INCORRECT"
        };

        self.log(
            DebugCategory::Response,
            &format!(
                "Content-Type: {} {} ({})",
                mime_type,
                status,
                path.display()
            ),
        );
    }

    /// Log WebSocket connection
    pub fn log_ws_connect(&self, client_addr: &str) {
        if !self.enabled {
            return;
        }
        self.log(
            DebugCategory::WebSocket,
            &format!("Client connected: {client_addr}"),
        );
    }

    /// Log WebSocket disconnection
    pub fn log_ws_disconnect(&self, client_addr: &str) {
        if !self.enabled {
            return;
        }
        self.log(
            DebugCategory::WebSocket,
            &format!("Client disconnected: {client_addr}"),
        );
    }

    /// Log file change event
    pub fn log_file_change(&self, path: &str, event_type: &str) {
        if !self.enabled {
            return;
        }
        self.log(DebugCategory::Watcher, &format!("{event_type}: {path}"));
    }
}

/// Create a shared debug tracer
#[must_use]
pub fn create_tracer(enabled: bool) -> Arc<DebugTracer> {
    Arc::new(DebugTracer::new(enabled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_debug_tracer_creation() {
        let tracer = DebugTracer::new(false);
        assert!(!tracer.is_enabled());

        let tracer = DebugTracer::enabled();
        assert!(tracer.is_enabled());
    }

    #[test]
    fn test_debug_verbosity_default() {
        let verbosity = DebugVerbosity::default();
        assert_eq!(verbosity, DebugVerbosity::Verbose);
    }

    #[test]
    fn test_debug_category_str() {
        assert_eq!(DebugCategory::Server.as_str(), "SERVER");
        assert_eq!(DebugCategory::Request.as_str(), "REQUEST");
        assert_eq!(DebugCategory::Error.as_str(), "ERROR");
    }

    #[test]
    fn test_resolution_rule_str() {
        assert_eq!(
            ResolutionRule::DirectoryIndex.as_str(),
            "Directory index (index.html)"
        );
        assert_eq!(ResolutionRule::StaticFile.as_str(), "Static file");
    }

    #[test]
    fn test_tracer_disabled_no_output() {
        let tracer = DebugTracer::new(false);
        // Should not panic even when disabled
        tracer.log(DebugCategory::Server, "test");
        tracer.log_multi(DebugCategory::Request, &["line1", "line2"]);
    }

    #[test]
    fn test_tracer_with_verbosity() {
        let tracer = DebugTracer::new(true).with_verbosity(DebugVerbosity::Minimal);
        assert!(tracer.is_enabled());
    }

    #[test]
    fn test_create_tracer() {
        let tracer = create_tracer(true);
        assert!(tracer.is_enabled());
    }

    #[test]
    fn test_format_line() {
        let tracer = DebugTracer::new(true);
        let line = tracer.format_line(DebugCategory::Server, "test message");
        assert!(line.contains("SERVER"));
        assert!(line.contains("test message"));
    }

    // =========================================================================
    // Additional coverage tests for debug.rs
    // =========================================================================

    #[test]
    fn test_debug_category_all_variants_str() {
        // Test all category string representations
        assert_eq!(DebugCategory::Server.as_str(), "SERVER");
        assert_eq!(DebugCategory::Request.as_str(), "REQUEST");
        assert_eq!(DebugCategory::Resolve.as_str(), "RESOLVE");
        assert_eq!(DebugCategory::Response.as_str(), "RESPONSE");
        assert_eq!(DebugCategory::Error.as_str(), "ERROR");
        assert_eq!(DebugCategory::WebSocket.as_str(), "WS");
        assert_eq!(DebugCategory::Watcher.as_str(), "WATCHER");
    }

    #[test]
    fn test_debug_category_all_variants_color() {
        // Test all category color codes
        assert!(DebugCategory::Server.color().contains("\x1b["));
        assert!(DebugCategory::Request.color().contains("\x1b["));
        assert!(DebugCategory::Resolve.color().contains("\x1b["));
        assert!(DebugCategory::Response.color().contains("\x1b["));
        assert!(DebugCategory::Error.color().contains("\x1b["));
        assert!(DebugCategory::WebSocket.color().contains("\x1b["));
        assert!(DebugCategory::Watcher.color().contains("\x1b["));
    }

    #[test]
    fn test_resolution_rule_all_variants_str() {
        // Test all resolution rule string representations
        assert_eq!(
            ResolutionRule::DirectoryIndex.as_str(),
            "Directory index (index.html)"
        );
        assert_eq!(ResolutionRule::StaticFile.as_str(), "Static file");
        assert_eq!(ResolutionRule::Fallback.as_str(), "Fallback");
        assert_eq!(ResolutionRule::NotFound.as_str(), "Not found");
    }

    #[test]
    fn test_debug_verbosity_all_variants() {
        // Ensure all variants are distinct
        assert_ne!(DebugVerbosity::Minimal, DebugVerbosity::Normal);
        assert_ne!(DebugVerbosity::Normal, DebugVerbosity::Verbose);
        assert_ne!(DebugVerbosity::Verbose, DebugVerbosity::Trace);
    }

    #[test]
    fn test_tracer_default() {
        let tracer = DebugTracer::default();
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_create_tracer_disabled() {
        let tracer = create_tracer(false);
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_log_multi_empty() {
        let tracer = DebugTracer::new(false);
        // Empty lines should not panic
        tracer.log_multi(DebugCategory::Server, &[]);
    }

    #[test]
    fn test_all_log_methods_disabled() {
        // All log methods should not panic when tracer is disabled
        let tracer = DebugTracer::new(false);
        let path = PathBuf::from("/test/path");
        let searched: Vec<PathBuf> = vec![PathBuf::from("/search1"), PathBuf::from("/search2")];
        let suggestions: Vec<String> = vec!["suggestion1".to_string()];

        tracer.log(DebugCategory::Server, "test");
        tracer.log_multi(DebugCategory::Request, &["line1", "line2"]);
        tracer.log_server_start(8080, &path, true, true);
        tracer.log_request("GET", "/", Some("127.0.0.1"), Some("Mozilla/5.0"));
        tracer.log_resolve("/", &path, ResolutionRule::DirectoryIndex);
        tracer.log_response(200, "text/html", 1024, 50);
        tracer.log_not_found("/missing.txt", &searched, &suggestions);
        tracer.log_mime_check(&path, "text/html", true);
        tracer.log_ws_connect("127.0.0.1");
        tracer.log_ws_disconnect("127.0.0.1");
        tracer.log_file_change("/test/file.rs", "modified");
    }

    #[test]
    fn test_format_line_no_colors() {
        // Force no colors by creating a tracer with use_colors = false
        let mut tracer = DebugTracer::new(true);
        tracer.use_colors = false;

        let line = tracer.format_line(DebugCategory::Server, "test");
        assert!(line.contains("SERVER"));
        assert!(!line.contains("\x1b[")); // No ANSI codes
    }

    #[test]
    fn test_format_line_with_colors() {
        // Force colors
        let mut tracer = DebugTracer::new(true);
        tracer.use_colors = true;

        let line = tracer.format_line(DebugCategory::Server, "test");
        assert!(line.contains("SERVER"));
        assert!(line.contains("\x1b[")); // Has ANSI codes
    }

    #[test]
    fn test_elapsed_str_format() {
        let tracer = DebugTracer::new(true);
        let elapsed = tracer.elapsed_str();
        // Should be in format "SS:MMM"
        assert!(elapsed.contains(':'));
        assert!(elapsed.len() >= 5); // At least "00:000"
    }

    #[test]
    fn test_debug_category_debug_impl() {
        // Test Debug trait implementation
        let cat = DebugCategory::Server;
        let debug_str = format!("{cat:?}");
        assert!(debug_str.contains("Server"));
    }

    #[test]
    fn test_debug_verbosity_clone() {
        let verbosity = DebugVerbosity::Verbose;
        let cloned = verbosity;
        assert_eq!(verbosity, cloned);
    }

    #[test]
    fn test_resolution_rule_debug_impl() {
        let rule = ResolutionRule::StaticFile;
        let debug_str = format!("{rule:?}");
        assert!(debug_str.contains("StaticFile"));
    }

    #[test]
    fn test_all_log_methods_enabled() {
        // Test all log methods when tracer IS enabled (exercises the printing code)
        let tracer = DebugTracer::enabled();
        let path = PathBuf::from("/test/path");
        let searched: Vec<PathBuf> = vec![PathBuf::from("/search1")];
        let suggestions: Vec<String> = vec!["Try checking the path".to_string()];

        // These should all execute without panicking
        tracer.log(DebugCategory::Server, "Server message");
        tracer.log(DebugCategory::Request, "Request message");
        tracer.log(DebugCategory::Resolve, "Resolve message");
        tracer.log(DebugCategory::Response, "Response message");
        tracer.log(DebugCategory::Error, "Error message");
        tracer.log(DebugCategory::WebSocket, "WebSocket message");
        tracer.log(DebugCategory::Watcher, "Watcher message");

        tracer.log_multi(DebugCategory::Server, &["line1", "line2", "line3"]);
        tracer.log_server_start(3000, &path, true, false);
        tracer.log_server_start(3001, &path, false, true);
        tracer.log_request("POST", "/api/test", Some("192.168.1.1"), None);
        tracer.log_request("GET", "/", None, Some("curl/7.0"));
        tracer.log_resolve("/index.html", &path, ResolutionRule::StaticFile);
        tracer.log_resolve("/", &path, ResolutionRule::Fallback);
        tracer.log_resolve("/missing", &path, ResolutionRule::NotFound);
        tracer.log_response(404, "text/plain", 0, 1);
        tracer.log_response(500, "application/json", 100, 5);
        tracer.log_not_found("/missing.css", &searched, &suggestions);
        tracer.log_not_found("/missing.js", &[], &[]);
        tracer.log_mime_check(&path, "application/wasm", false);
        tracer.log_ws_connect("10.0.0.1");
        tracer.log_ws_disconnect("10.0.0.1");
        tracer.log_file_change("/src/main.rs", "created");
        tracer.log_file_change("/src/lib.rs", "deleted");
    }

    #[test]
    fn test_tracer_with_all_verbosity_levels() {
        let tracer_minimal = DebugTracer::enabled().with_verbosity(DebugVerbosity::Minimal);
        let tracer_normal = DebugTracer::enabled().with_verbosity(DebugVerbosity::Normal);
        let tracer_verbose = DebugTracer::enabled().with_verbosity(DebugVerbosity::Verbose);
        let tracer_trace = DebugTracer::enabled().with_verbosity(DebugVerbosity::Trace);

        // All should be enabled
        assert!(tracer_minimal.is_enabled());
        assert!(tracer_normal.is_enabled());
        assert!(tracer_verbose.is_enabled());
        assert!(tracer_trace.is_enabled());
    }

    #[test]
    fn test_debug_category_copy_semantics() {
        let cat = DebugCategory::Error;
        let copied = cat;
        assert_eq!(cat.as_str(), copied.as_str());
        assert_eq!(cat.color(), copied.color());
    }

    #[test]
    fn test_resolution_rule_copy_semantics() {
        let rule = ResolutionRule::Fallback;
        let copied = rule;
        assert_eq!(rule.as_str(), copied.as_str());
    }

    #[test]
    fn test_debug_verbosity_eq_all_pairs() {
        let variants = [
            DebugVerbosity::Minimal,
            DebugVerbosity::Normal,
            DebugVerbosity::Verbose,
            DebugVerbosity::Trace,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
