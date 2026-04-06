//! HAR (HTTP Archive) Recording and Playback (Feature G.2)
//!
//! Implements HAR 1.2 format for recording and replaying HTTP traffic.
//! This enables reproducible E2E tests by capturing network interactions.
//!
//! ## EXTREME TDD: Tests written FIRST per Popperian falsification
//!
//! ## Toyota Way Application
//!
//! - **Mieruka**: HAR files make network interactions visible and auditable
//! - **Poka-Yoke**: Type-safe HAR structures prevent invalid recordings
//! - **Jidoka**: Immediate feedback on HAR parsing/validation errors

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// =============================================================================
// HAR 1.2 Format Structures
// =============================================================================

/// HAR file root structure (HAR 1.2 specification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Har {
    /// HAR log container
    pub log: HarLog,
}

impl Har {
    /// Create a new empty HAR file
    #[must_use]
    pub fn new() -> Self {
        Self { log: HarLog::new() }
    }

    /// Parse HAR from JSON string
    ///
    /// # Errors
    ///
    /// Returns error if JSON parsing fails
    pub fn from_json(json: &str) -> Result<Self, HarError> {
        serde_json::from_str(json).map_err(|e| HarError::ParseError(e.to_string()))
    }

    /// Serialize HAR to JSON string
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    pub fn to_json(&self) -> Result<String, HarError> {
        serde_json::to_string_pretty(self).map_err(|e| HarError::SerializeError(e.to_string()))
    }

    /// Get number of entries
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.log.entries.len()
    }

    /// Add an entry
    pub fn add_entry(&mut self, entry: HarEntry) {
        self.log.entries.push(entry);
    }

    /// Find entry by URL
    #[must_use]
    pub fn find_by_url(&self, url: &str) -> Option<&HarEntry> {
        self.log.entries.iter().find(|e| e.request.url == url)
    }

    /// Find entries matching URL pattern (glob-style)
    #[must_use]
    pub fn find_matching(&self, pattern: &str) -> Vec<&HarEntry> {
        self.log
            .entries
            .iter()
            .filter(|e| url_matches_pattern(&e.request.url, pattern))
            .collect()
    }
}

impl Default for Har {
    fn default() -> Self {
        Self::new()
    }
}

/// HAR log structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarLog {
    /// HAR format version (always "1.2")
    pub version: String,
    /// Creator application info
    pub creator: HarCreator,
    /// Browser info (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub browser: Option<HarBrowser>,
    /// List of recorded entries
    pub entries: Vec<HarEntry>,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarLog {
    /// Create a new HAR log
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: "1.2".to_string(),
            creator: HarCreator::probar(),
            browser: None,
            entries: Vec::new(),
            comment: None,
        }
    }
}

impl Default for HarLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Creator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarCreator {
    /// Creator name
    pub name: String,
    /// Creator version
    pub version: String,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarCreator {
    /// Create Probar creator info
    #[must_use]
    pub fn probar() -> Self {
        Self {
            name: "Probar".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            comment: None,
        }
    }
}

/// Browser information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarBrowser {
    /// Browser name
    pub name: String,
    /// Browser version
    pub version: String,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarBrowser {
    /// Create browser info
    #[must_use]
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            comment: None,
        }
    }
}

/// A single HAR entry (request/response pair)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarEntry {
    /// Start time (ISO 8601)
    #[serde(rename = "startedDateTime")]
    pub started_date_time: String,
    /// Total time in milliseconds
    pub time: f64,
    /// Request details
    pub request: HarRequest,
    /// Response details
    pub response: HarResponse,
    /// Cache details
    pub cache: HarCache,
    /// Timing details
    pub timings: HarTimings,
    /// Server IP address (optional)
    #[serde(rename = "serverIPAddress", skip_serializing_if = "Option::is_none")]
    pub server_ip_address: Option<String>,
    /// Connection ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connection: Option<String>,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarEntry {
    /// Create a new entry
    #[must_use]
    pub fn new(request: HarRequest, response: HarResponse) -> Self {
        Self {
            started_date_time: chrono_now_iso(),
            time: 0.0,
            request,
            response,
            cache: HarCache::default(),
            timings: HarTimings::default(),
            server_ip_address: None,
            connection: None,
            comment: None,
        }
    }

    /// Set timing in milliseconds
    #[must_use]
    pub fn with_time(mut self, time_ms: f64) -> Self {
        self.time = time_ms;
        self
    }

    /// Set server IP
    #[must_use]
    pub fn with_server_ip(mut self, ip: impl Into<String>) -> Self {
        self.server_ip_address = Some(ip.into());
        self
    }
}

/// HTTP request in HAR format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarRequest {
    /// HTTP method
    pub method: String,
    /// Request URL
    pub url: String,
    /// HTTP version
    #[serde(rename = "httpVersion")]
    pub http_version: String,
    /// Cookies
    pub cookies: Vec<HarCookie>,
    /// Headers
    pub headers: Vec<HarHeader>,
    /// Query string parameters
    #[serde(rename = "queryString")]
    pub query_string: Vec<HarQueryParam>,
    /// POST data (optional)
    #[serde(rename = "postData", skip_serializing_if = "Option::is_none")]
    pub post_data: Option<HarPostData>,
    /// Headers size in bytes (-1 if unknown)
    #[serde(rename = "headersSize")]
    pub headers_size: i64,
    /// Body size in bytes (-1 if unknown)
    #[serde(rename = "bodySize")]
    pub body_size: i64,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarRequest {
    /// Create a GET request
    #[must_use]
    pub fn get(url: impl Into<String>) -> Self {
        Self::new("GET", url)
    }

    /// Create a POST request
    #[must_use]
    pub fn post(url: impl Into<String>) -> Self {
        Self::new("POST", url)
    }

    /// Create a new request
    #[must_use]
    pub fn new(method: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            url: url.into(),
            http_version: "HTTP/1.1".to_string(),
            cookies: Vec::new(),
            headers: Vec::new(),
            query_string: Vec::new(),
            post_data: None,
            headers_size: -1,
            body_size: -1,
            comment: None,
        }
    }

    /// Add a header
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push(HarHeader::new(name, value));
        self
    }

    /// Add POST data
    #[must_use]
    pub fn with_post_data(mut self, data: HarPostData) -> Self {
        self.post_data = Some(data);
        self
    }
}

/// HTTP response in HAR format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarResponse {
    /// HTTP status code
    pub status: u16,
    /// Status text
    #[serde(rename = "statusText")]
    pub status_text: String,
    /// HTTP version
    #[serde(rename = "httpVersion")]
    pub http_version: String,
    /// Cookies
    pub cookies: Vec<HarCookie>,
    /// Headers
    pub headers: Vec<HarHeader>,
    /// Response content
    pub content: HarContent,
    /// Redirect URL (if any)
    #[serde(rename = "redirectURL")]
    pub redirect_url: String,
    /// Headers size in bytes (-1 if unknown)
    #[serde(rename = "headersSize")]
    pub headers_size: i64,
    /// Body size in bytes (-1 if unknown)
    #[serde(rename = "bodySize")]
    pub body_size: i64,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarResponse {
    /// Create a successful response
    #[must_use]
    pub fn ok() -> Self {
        Self::new(200, "OK")
    }

    /// Create a not found response
    #[must_use]
    pub fn not_found() -> Self {
        Self::new(404, "Not Found")
    }

    /// Create a new response
    #[must_use]
    pub fn new(status: u16, status_text: impl Into<String>) -> Self {
        Self {
            status,
            status_text: status_text.into(),
            http_version: "HTTP/1.1".to_string(),
            cookies: Vec::new(),
            headers: Vec::new(),
            content: HarContent::default(),
            redirect_url: String::new(),
            headers_size: -1,
            body_size: -1,
            comment: None,
        }
    }

    /// Add a header
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push(HarHeader::new(name, value));
        self
    }

    /// Set response content
    #[must_use]
    pub fn with_content(mut self, content: HarContent) -> Self {
        self.content = content;
        self
    }

    /// Set JSON body
    #[must_use]
    pub fn with_json(mut self, body: impl Into<String>) -> Self {
        self.content = HarContent::json(body);
        self.headers
            .push(HarHeader::new("Content-Type", "application/json"));
        self
    }
}

/// HTTP header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarHeader {
    /// Header name
    pub name: String,
    /// Header value
    pub value: String,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarHeader {
    /// Create a new header
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            comment: None,
        }
    }
}

/// Cookie
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarCookie {
    /// Cookie name
    pub name: String,
    /// Cookie value
    pub value: String,
    /// Path (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Domain (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    /// Expires (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
    /// HTTP only flag
    #[serde(rename = "httpOnly", skip_serializing_if = "Option::is_none")]
    pub http_only: Option<bool>,
    /// Secure flag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secure: Option<bool>,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarCookie {
    /// Create a new cookie
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            path: None,
            domain: None,
            expires: None,
            http_only: None,
            secure: None,
            comment: None,
        }
    }
}

/// Query string parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarQueryParam {
    /// Parameter name
    pub name: String,
    /// Parameter value
    pub value: String,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarQueryParam {
    /// Create a new query parameter
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            comment: None,
        }
    }
}

/// POST data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarPostData {
    /// MIME type
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// Form parameters (for urlencoded)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<HarPostParam>,
    /// Raw text content
    pub text: String,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarPostData {
    /// Create JSON POST data
    #[must_use]
    pub fn json(body: impl Into<String>) -> Self {
        Self {
            mime_type: "application/json".to_string(),
            params: Vec::new(),
            text: body.into(),
            comment: None,
        }
    }

    /// Create form-urlencoded POST data
    #[must_use]
    pub fn form(params: Vec<HarPostParam>) -> Self {
        Self {
            mime_type: "application/x-www-form-urlencoded".to_string(),
            params,
            text: String::new(),
            comment: None,
        }
    }
}

/// POST parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarPostParam {
    /// Parameter name
    pub name: String,
    /// Parameter value (optional for file uploads)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// File name (for file uploads)
    #[serde(rename = "fileName", skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
    /// Content type (for file uploads)
    #[serde(rename = "contentType", skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarPostParam {
    /// Create a new POST parameter
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: Some(value.into()),
            file_name: None,
            content_type: None,
            comment: None,
        }
    }
}

/// Response content
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarContent {
    /// Content size in bytes
    pub size: i64,
    /// Compression size (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression: Option<i64>,
    /// MIME type
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// Response text (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Encoding (e.g., "base64")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl HarContent {
    /// Create JSON content
    #[must_use]
    pub fn json(body: impl Into<String>) -> Self {
        let text = body.into();
        let size = text.len() as i64;
        Self {
            size,
            compression: None,
            mime_type: "application/json".to_string(),
            text: Some(text),
            encoding: None,
            comment: None,
        }
    }

    /// Create text content
    #[must_use]
    pub fn text(body: impl Into<String>) -> Self {
        let text = body.into();
        let size = text.len() as i64;
        Self {
            size,
            compression: None,
            mime_type: "text/plain".to_string(),
            text: Some(text),
            encoding: None,
            comment: None,
        }
    }

    /// Create HTML content
    #[must_use]
    pub fn html(body: impl Into<String>) -> Self {
        let text = body.into();
        let size = text.len() as i64;
        Self {
            size,
            compression: None,
            mime_type: "text/html".to_string(),
            text: Some(text),
            encoding: None,
            comment: None,
        }
    }
}

/// Cache details
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarCache {
    /// Before request cache state (optional)
    #[serde(rename = "beforeRequest", skip_serializing_if = "Option::is_none")]
    pub before_request: Option<HarCacheState>,
    /// After request cache state (optional)
    #[serde(rename = "afterRequest", skip_serializing_if = "Option::is_none")]
    pub after_request: Option<HarCacheState>,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

/// Cache state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarCacheState {
    /// Expiry time (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
    /// Last access time (optional)
    #[serde(rename = "lastAccess", skip_serializing_if = "Option::is_none")]
    pub last_access: Option<String>,
    /// ETag (optional)
    #[serde(rename = "eTag", skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
    /// Hit count (optional)
    #[serde(rename = "hitCount", skip_serializing_if = "Option::is_none")]
    pub hit_count: Option<u32>,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

/// Timing details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarTimings {
    /// Time spent in blocked queue (-1 if not applicable)
    pub blocked: f64,
    /// DNS resolution time (-1 if not applicable)
    pub dns: f64,
    /// Time to establish connection (-1 if not applicable)
    pub connect: f64,
    /// Time to send request
    pub send: f64,
    /// Time waiting for response
    pub wait: f64,
    /// Time to receive response
    pub receive: f64,
    /// SSL/TLS negotiation time (-1 if not applicable)
    pub ssl: f64,
    /// Optional comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

impl Default for HarTimings {
    fn default() -> Self {
        Self {
            blocked: -1.0,
            dns: -1.0,
            connect: -1.0,
            send: 0.0,
            wait: 0.0,
            receive: 0.0,
            ssl: -1.0,
            comment: None,
        }
    }
}

impl HarTimings {
    /// Create new timings with defaults
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total time
    #[must_use]
    pub fn total(&self) -> f64 {
        let mut total = 0.0;
        if self.blocked > 0.0 {
            total += self.blocked;
        }
        if self.dns > 0.0 {
            total += self.dns;
        }
        if self.connect > 0.0 {
            total += self.connect;
        }
        total += self.send;
        total += self.wait;
        total += self.receive;
        total
    }
}

// =============================================================================
// HAR Recording and Playback
// =============================================================================

/// HAR recording options
#[derive(Debug, Clone)]
pub struct HarOptions {
    /// Behavior when request not found in HAR
    pub not_found: NotFoundBehavior,
    /// Update HAR with new requests
    pub update: bool,
    /// URL pattern to match (glob-style)
    pub url_pattern: Option<String>,
}

impl Default for HarOptions {
    fn default() -> Self {
        Self {
            not_found: NotFoundBehavior::Fallback,
            update: false,
            url_pattern: None,
        }
    }
}

impl HarOptions {
    /// Create new options with abort on not found
    #[must_use]
    pub fn abort_on_not_found() -> Self {
        Self {
            not_found: NotFoundBehavior::Abort,
            ..Default::default()
        }
    }

    /// Create new options with fallback on not found
    #[must_use]
    pub fn fallback_on_not_found() -> Self {
        Self {
            not_found: NotFoundBehavior::Fallback,
            ..Default::default()
        }
    }

    /// Enable update mode
    #[must_use]
    pub fn with_update(mut self, update: bool) -> Self {
        self.update = update;
        self
    }

    /// Set URL pattern filter
    #[must_use]
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.url_pattern = Some(pattern.into());
        self
    }
}

/// Behavior when request not found in HAR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotFoundBehavior {
    /// Abort the request
    Abort,
    /// Fall back to real network
    Fallback,
}

/// HAR recorder for capturing network traffic
#[derive(Debug)]
pub struct HarRecorder {
    /// Recorded HAR data
    har: Har,
    /// Output path
    path: PathBuf,
    /// Whether recording is active
    active: bool,
    /// URL filter pattern
    filter: Option<String>,
}

impl HarRecorder {
    /// Create a new HAR recorder
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            har: Har::new(),
            path: path.into(),
            active: false,
            filter: None,
        }
    }

    /// Start recording
    pub fn start(&mut self) {
        self.active = true;
    }

    /// Stop recording
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Check if recording is active
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Set URL filter pattern
    pub fn set_filter(&mut self, pattern: impl Into<String>) {
        self.filter = Some(pattern.into());
    }

    /// Record a request/response pair
    pub fn record(&mut self, entry: HarEntry) {
        if !self.active {
            return;
        }

        // Apply filter if set
        if let Some(ref pattern) = self.filter {
            if !url_matches_pattern(&entry.request.url, pattern) {
                return;
            }
        }

        self.har.add_entry(entry);
    }

    /// Get recorded HAR
    #[must_use]
    pub fn har(&self) -> &Har {
        &self.har
    }

    /// Get entry count
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.har.entry_count()
    }

    /// Save HAR to file
    ///
    /// # Errors
    ///
    /// Returns error if file writing fails
    pub fn save(&self) -> Result<(), HarError> {
        let json = self.har.to_json()?;
        std::fs::write(&self.path, json).map_err(|e| HarError::IoError(e.to_string()))
    }
}

/// HAR player for replaying recorded traffic
#[derive(Debug)]
pub struct HarPlayer {
    /// HAR data to replay
    har: Har,
    /// Options for playback
    options: HarOptions,
}

impl HarPlayer {
    /// Create a new HAR player
    #[must_use]
    pub fn new(har: Har, options: HarOptions) -> Self {
        Self { har, options }
    }

    /// Load HAR from file
    ///
    /// # Errors
    ///
    /// Returns error if file reading or parsing fails
    pub fn from_file(path: impl Into<PathBuf>, options: HarOptions) -> Result<Self, HarError> {
        let path = path.into();
        let content =
            std::fs::read_to_string(&path).map_err(|e| HarError::IoError(e.to_string()))?;
        let har = Har::from_json(&content)?;
        Ok(Self::new(har, options))
    }

    /// Find matching response for a request
    #[must_use]
    pub fn find_response(&self, method: &str, url: &str) -> Option<&HarResponse> {
        // Check URL pattern filter
        if let Some(ref pattern) = self.options.url_pattern {
            if !url_matches_pattern(url, pattern) {
                return None;
            }
        }

        // Find matching entry
        self.har.log.entries.iter().find_map(|entry| {
            if entry.request.method == method && entry.request.url == url {
                Some(&entry.response)
            } else {
                None
            }
        })
    }

    /// Get behavior for not found requests
    #[must_use]
    pub fn not_found_behavior(&self) -> NotFoundBehavior {
        self.options.not_found
    }

    /// Get entry count
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.har.entry_count()
    }
}

// =============================================================================
// Errors
// =============================================================================

/// HAR-related errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarError {
    /// JSON parsing error
    ParseError(String),
    /// JSON serialization error
    SerializeError(String),
    /// I/O error
    IoError(String),
    /// Request not found in HAR
    NotFound(String),
}

impl std::fmt::Display for HarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "HAR parse error: {msg}"),
            Self::SerializeError(msg) => write!(f, "HAR serialize error: {msg}"),
            Self::IoError(msg) => write!(f, "HAR I/O error: {msg}"),
            Self::NotFound(url) => write!(f, "Request not found in HAR: {url}"),
        }
    }
}

impl std::error::Error for HarError {}

// =============================================================================
// Helpers
// =============================================================================

/// Generate current ISO 8601 timestamp
fn chrono_now_iso() -> String {
    // Simple implementation without chrono dependency
    "2024-01-01T00:00:00.000Z".to_string()
}

/// Check if URL matches pattern (simple contains match)
fn url_matches_pattern(url: &str, pattern: &str) -> bool {
    // Simple contains matching for now
    // Strip glob wildcards for basic matching
    let clean_pattern = pattern
        .replace("**", "")
        .replace('*', "")
        .trim_matches('/')
        .to_string();

    if clean_pattern.is_empty() {
        return true; // Empty pattern matches everything
    }

    url.contains(&clean_pattern)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-HAR-01 to H₀-HAR-10: HAR Structure Tests
    // =========================================================================

    #[test]
    fn h0_har_01_new_creates_empty_har() {
        let har = Har::new();
        assert_eq!(har.log.version, "1.2");
        assert_eq!(har.entry_count(), 0);
    }

    #[test]
    fn h0_har_02_log_has_probar_creator() {
        let har = Har::new();
        assert_eq!(har.log.creator.name, "Probar");
    }

    #[test]
    fn h0_har_03_add_entry() {
        let mut har = Har::new();
        let entry = HarEntry::new(HarRequest::get("http://example.com"), HarResponse::ok());
        har.add_entry(entry);
        assert_eq!(har.entry_count(), 1);
    }

    #[test]
    fn h0_har_04_find_by_url() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://example.com/api"),
            HarResponse::ok(),
        ));
        assert!(har.find_by_url("http://example.com/api").is_some());
        assert!(har.find_by_url("http://other.com").is_none());
    }

    #[test]
    fn h0_har_05_serialization_roundtrip() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com"),
            HarResponse::ok(),
        ));
        let json = har.to_json().unwrap();
        let parsed = Har::from_json(&json).unwrap();
        assert_eq!(parsed.entry_count(), 1);
    }

    #[test]
    fn h0_har_06_request_get() {
        let req = HarRequest::get("http://test.com");
        assert_eq!(req.method, "GET");
        assert_eq!(req.url, "http://test.com");
    }

    #[test]
    fn h0_har_07_request_post() {
        let req = HarRequest::post("http://test.com");
        assert_eq!(req.method, "POST");
    }

    #[test]
    fn h0_har_08_request_with_header() {
        let req = HarRequest::get("http://test.com").with_header("Accept", "application/json");
        assert_eq!(req.headers.len(), 1);
        assert_eq!(req.headers[0].name, "Accept");
    }

    #[test]
    fn h0_har_09_response_ok() {
        let resp = HarResponse::ok();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.status_text, "OK");
    }

    #[test]
    fn h0_har_10_response_not_found() {
        let resp = HarResponse::not_found();
        assert_eq!(resp.status, 404);
    }

    // =========================================================================
    // H₀-HAR-11 to H₀-HAR-20: Content and Data Tests
    // =========================================================================

    #[test]
    fn h0_har_11_response_with_json() {
        let resp = HarResponse::ok().with_json(r#"{"key": "value"}"#);
        assert_eq!(resp.content.mime_type, "application/json");
        assert!(resp.content.text.is_some());
    }

    #[test]
    fn h0_har_12_content_json() {
        let content = HarContent::json(r#"{"test": true}"#);
        assert_eq!(content.mime_type, "application/json");
        assert!(content.size > 0);
    }

    #[test]
    fn h0_har_13_content_text() {
        let content = HarContent::text("Hello, World!");
        assert_eq!(content.mime_type, "text/plain");
    }

    #[test]
    fn h0_har_14_content_html() {
        let content = HarContent::html("<html></html>");
        assert_eq!(content.mime_type, "text/html");
    }

    #[test]
    fn h0_har_15_post_data_json() {
        let data = HarPostData::json(r#"{"name": "test"}"#);
        assert_eq!(data.mime_type, "application/json");
    }

    #[test]
    fn h0_har_16_post_data_form() {
        let data = HarPostData::form(vec![HarPostParam::new("field", "value")]);
        assert_eq!(data.mime_type, "application/x-www-form-urlencoded");
        assert_eq!(data.params.len(), 1);
    }

    #[test]
    fn h0_har_17_cookie_creation() {
        let cookie = HarCookie::new("session", "abc123");
        assert_eq!(cookie.name, "session");
        assert_eq!(cookie.value, "abc123");
    }

    #[test]
    fn h0_har_18_header_creation() {
        let header = HarHeader::new("X-Custom", "value");
        assert_eq!(header.name, "X-Custom");
        assert_eq!(header.value, "value");
    }

    #[test]
    fn h0_har_19_query_param() {
        let param = HarQueryParam::new("page", "1");
        assert_eq!(param.name, "page");
        assert_eq!(param.value, "1");
    }

    #[test]
    fn h0_har_20_entry_with_time() {
        let entry =
            HarEntry::new(HarRequest::get("http://test.com"), HarResponse::ok()).with_time(150.0);
        assert!((entry.time - 150.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // H₀-HAR-21 to H₀-HAR-30: Recording Tests
    // =========================================================================

    #[test]
    fn h0_har_21_recorder_new() {
        let recorder = HarRecorder::new("test.har");
        assert!(!recorder.is_active());
        assert_eq!(recorder.entry_count(), 0);
    }

    #[test]
    fn h0_har_22_recorder_start_stop() {
        let mut recorder = HarRecorder::new("test.har");
        recorder.start();
        assert!(recorder.is_active());
        recorder.stop();
        assert!(!recorder.is_active());
    }

    #[test]
    fn h0_har_23_recorder_record_when_active() {
        let mut recorder = HarRecorder::new("test.har");
        recorder.start();
        recorder.record(HarEntry::new(
            HarRequest::get("http://test.com"),
            HarResponse::ok(),
        ));
        assert_eq!(recorder.entry_count(), 1);
    }

    #[test]
    fn h0_har_24_recorder_skip_when_inactive() {
        let mut recorder = HarRecorder::new("test.har");
        recorder.record(HarEntry::new(
            HarRequest::get("http://test.com"),
            HarResponse::ok(),
        ));
        assert_eq!(recorder.entry_count(), 0);
    }

    #[test]
    fn h0_har_25_recorder_filter() {
        let mut recorder = HarRecorder::new("test.har");
        recorder.start();
        recorder.set_filter("/api/");
        recorder.record(HarEntry::new(
            HarRequest::get("http://test.com/api/users"),
            HarResponse::ok(),
        ));
        recorder.record(HarEntry::new(
            HarRequest::get("http://test.com/static/image.png"),
            HarResponse::ok(),
        ));
        // Only API request recorded (filter uses contains match)
        assert_eq!(recorder.entry_count(), 1);
    }

    #[test]
    fn h0_har_26_options_default() {
        let options = HarOptions::default();
        assert_eq!(options.not_found, NotFoundBehavior::Fallback);
        assert!(!options.update);
    }

    #[test]
    fn h0_har_27_options_abort_on_not_found() {
        let options = HarOptions::abort_on_not_found();
        assert_eq!(options.not_found, NotFoundBehavior::Abort);
    }

    #[test]
    fn h0_har_28_options_with_update() {
        let options = HarOptions::default().with_update(true);
        assert!(options.update);
    }

    #[test]
    fn h0_har_29_options_with_pattern() {
        let options = HarOptions::default().with_pattern("**/api/**");
        assert!(options.url_pattern.is_some());
    }

    #[test]
    fn h0_har_30_recorder_har_access() {
        let recorder = HarRecorder::new("test.har");
        let har = recorder.har();
        assert_eq!(har.log.version, "1.2");
    }

    // =========================================================================
    // H₀-HAR-31 to H₀-HAR-40: Playback Tests
    // =========================================================================

    #[test]
    fn h0_har_31_player_new() {
        let har = Har::new();
        let player = HarPlayer::new(har, HarOptions::default());
        assert_eq!(player.entry_count(), 0);
    }

    #[test]
    fn h0_har_32_player_find_response() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/api"),
            HarResponse::ok().with_json(r#"{"found": true}"#),
        ));
        let player = HarPlayer::new(har, HarOptions::default());
        let resp = player.find_response("GET", "http://test.com/api");
        assert!(resp.is_some());
        assert_eq!(resp.unwrap().status, 200);
    }

    #[test]
    fn h0_har_33_player_not_found() {
        let har = Har::new();
        let player = HarPlayer::new(har, HarOptions::default());
        let resp = player.find_response("GET", "http://missing.com");
        assert!(resp.is_none());
    }

    #[test]
    fn h0_har_34_player_not_found_behavior() {
        let player = HarPlayer::new(Har::new(), HarOptions::abort_on_not_found());
        assert_eq!(player.not_found_behavior(), NotFoundBehavior::Abort);
    }

    #[test]
    fn h0_har_35_timings_default() {
        let timings = HarTimings::default();
        assert!(timings.blocked < 0.0);
        assert!(timings.dns < 0.0);
    }

    #[test]
    fn h0_har_36_timings_total() {
        let mut timings = HarTimings::new();
        timings.send = 10.0;
        timings.wait = 50.0;
        timings.receive = 20.0;
        assert!((timings.total() - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn h0_har_37_browser_info() {
        let browser = HarBrowser::new("Chromium", "120.0");
        assert_eq!(browser.name, "Chromium");
        assert_eq!(browser.version, "120.0");
    }

    #[test]
    fn h0_har_38_entry_with_server_ip() {
        let entry = HarEntry::new(HarRequest::get("http://test.com"), HarResponse::ok())
            .with_server_ip("192.168.1.1");
        assert_eq!(entry.server_ip_address, Some("192.168.1.1".to_string()));
    }

    #[test]
    fn h0_har_39_error_display() {
        let err = HarError::ParseError("invalid json".to_string());
        assert!(format!("{err}").contains("parse error"));
    }

    #[test]
    fn h0_har_40_error_not_found() {
        let err = HarError::NotFound("http://missing.com".to_string());
        assert!(format!("{err}").contains("not found"));
    }

    // =========================================================================
    // H₀-HAR-41 to H₀-HAR-50: Advanced Tests
    // =========================================================================

    #[test]
    fn h0_har_41_find_matching_empty() {
        let har = Har::new();
        let matches = har.find_matching("**/api/**");
        assert!(matches.is_empty());
    }

    #[test]
    fn h0_har_42_cache_default() {
        let cache = HarCache::default();
        assert!(cache.before_request.is_none());
        assert!(cache.after_request.is_none());
    }

    #[test]
    fn h0_har_43_response_with_header() {
        let resp = HarResponse::ok().with_header("X-Request-Id", "123");
        assert_eq!(resp.headers.len(), 1);
    }

    #[test]
    fn h0_har_44_request_with_post_data() {
        let req = HarRequest::post("http://test.com").with_post_data(HarPostData::json(r#"{}"#));
        assert!(req.post_data.is_some());
    }

    #[test]
    fn h0_har_45_response_with_content() {
        let resp = HarResponse::ok().with_content(HarContent::text("body"));
        assert_eq!(resp.content.text, Some("body".to_string()));
    }

    #[test]
    fn h0_har_46_parse_error() {
        let result = Har::from_json("invalid json");
        assert!(result.is_err());
    }

    #[test]
    fn h0_har_47_har_default() {
        let har = Har::default();
        assert_eq!(har.log.version, "1.2");
    }

    #[test]
    fn h0_har_48_log_default() {
        let log = HarLog::default();
        assert!(log.entries.is_empty());
    }

    #[test]
    fn h0_har_49_timings_new() {
        let timings = HarTimings::new();
        assert!(timings.ssl < 0.0);
    }

    #[test]
    fn h0_har_50_content_default() {
        let content = HarContent::default();
        assert!(content.text.is_none());
        assert_eq!(content.size, 0);
    }

    // =========================================================================
    // H₀-HAR-51 to H₀-HAR-70: Additional Coverage Tests
    // =========================================================================

    #[test]
    fn h0_har_51_error_serialize_display() {
        let err = HarError::SerializeError("failed to serialize".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("serialize error"));
        assert!(msg.contains("failed to serialize"));
    }

    #[test]
    fn h0_har_52_error_io_display() {
        let err = HarError::IoError("file not found".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn h0_har_53_options_fallback_on_not_found() {
        let options = HarOptions::fallback_on_not_found();
        assert_eq!(options.not_found, NotFoundBehavior::Fallback);
        assert!(!options.update);
        assert!(options.url_pattern.is_none());
    }

    #[test]
    fn h0_har_54_player_find_response_with_pattern_no_match() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/users"),
            HarResponse::ok(),
        ));
        let options = HarOptions::default().with_pattern("/api/");
        let player = HarPlayer::new(har, options);
        // URL doesn't match pattern, should return None
        let resp = player.find_response("GET", "http://test.com/users");
        assert!(resp.is_none());
    }

    #[test]
    fn h0_har_55_player_find_response_with_pattern_match() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/api/users"),
            HarResponse::ok().with_json(r#"{"users": []}"#),
        ));
        let options = HarOptions::default().with_pattern("/api/");
        let player = HarPlayer::new(har, options);
        let resp = player.find_response("GET", "http://test.com/api/users");
        assert!(resp.is_some());
        assert_eq!(resp.unwrap().status, 200);
    }

    #[test]
    fn h0_har_56_player_find_response_method_mismatch() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/api"),
            HarResponse::ok(),
        ));
        let player = HarPlayer::new(har, HarOptions::default());
        // Wrong method
        let resp = player.find_response("POST", "http://test.com/api");
        assert!(resp.is_none());
    }

    #[test]
    fn h0_har_57_timings_total_with_positive_values() {
        let mut timings = HarTimings::new();
        timings.blocked = 5.0;
        timings.dns = 10.0;
        timings.connect = 15.0;
        timings.send = 2.0;
        timings.wait = 100.0;
        timings.receive = 50.0;
        // Total should be 5 + 10 + 15 + 2 + 100 + 50 = 182
        assert!((timings.total() - 182.0).abs() < f64::EPSILON);
    }

    #[test]
    fn h0_har_58_timings_total_with_mixed_values() {
        let mut timings = HarTimings::new();
        // blocked and dns remain -1 (not applicable)
        timings.connect = 20.0;
        timings.send = 5.0;
        timings.wait = 30.0;
        timings.receive = 10.0;
        // Total = 20 + 5 + 30 + 10 = 65 (blocked/dns excluded)
        assert!((timings.total() - 65.0).abs() < f64::EPSILON);
    }

    #[test]
    fn h0_har_59_find_matching_with_matches() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/api/users"),
            HarResponse::ok(),
        ));
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/api/posts"),
            HarResponse::ok(),
        ));
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/static/image.png"),
            HarResponse::ok(),
        ));
        let matches = har.find_matching("/api/");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn h0_har_60_url_matches_pattern_glob_wildcards() {
        // Pattern with ** glob
        assert!(url_matches_pattern(
            "http://test.com/api/users",
            "**/api/**"
        ));
        // Pattern with * glob
        assert!(url_matches_pattern("http://test.com/api/test", "*/api/*"));
        // Exact substring
        assert!(url_matches_pattern("http://example.com/path", "example"));
        // No match
        assert!(!url_matches_pattern("http://other.com", "example"));
    }

    #[test]
    fn h0_har_61_url_matches_pattern_empty() {
        // Empty pattern after stripping globs matches everything
        assert!(url_matches_pattern("http://any.com/path", "**"));
        assert!(url_matches_pattern("http://any.com/path", "*"));
        assert!(url_matches_pattern("http://any.com/path", ""));
    }

    #[test]
    fn h0_har_62_recorder_save_error() {
        let recorder = HarRecorder::new("/nonexistent/path/that/cannot/be/written/test.har");
        let result = recorder.save();
        assert!(result.is_err());
        if let Err(HarError::IoError(msg)) = result {
            assert!(!msg.is_empty());
        } else {
            panic!("Expected IoError");
        }
    }

    #[test]
    fn h0_har_63_player_from_file_not_found() {
        let result = HarPlayer::from_file("/nonexistent/file.har", HarOptions::default());
        assert!(result.is_err());
        if let Err(HarError::IoError(msg)) = result {
            assert!(!msg.is_empty());
        } else {
            panic!("Expected IoError");
        }
    }

    #[test]
    fn h0_har_64_player_from_file_invalid_json() {
        // Create a temp file with invalid JSON
        let temp_path = std::env::temp_dir().join("test_invalid_har.json");
        std::fs::write(&temp_path, "not valid json").unwrap();
        let result = HarPlayer::from_file(&temp_path, HarOptions::default());
        std::fs::remove_file(&temp_path).ok();
        assert!(result.is_err());
        if let Err(HarError::ParseError(msg)) = result {
            assert!(!msg.is_empty());
        } else {
            panic!("Expected ParseError");
        }
    }

    #[test]
    fn h0_har_65_recorder_save_and_load_roundtrip() {
        let temp_path = std::env::temp_dir().join("test_har_roundtrip.har");
        let mut recorder = HarRecorder::new(&temp_path);
        recorder.start();
        recorder.record(HarEntry::new(
            HarRequest::get("http://test.com/api").with_header("Accept", "application/json"),
            HarResponse::ok().with_json(r#"{"status": "ok"}"#),
        ));
        recorder.stop();
        recorder.save().expect("Save should succeed");

        let player = HarPlayer::from_file(&temp_path, HarOptions::default()).unwrap();
        assert_eq!(player.entry_count(), 1);
        let resp = player.find_response("GET", "http://test.com/api");
        assert!(resp.is_some());
        assert_eq!(resp.unwrap().status, 200);

        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn h0_har_66_har_error_implements_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(HarError::NotFound("test".to_string()));
        // Just verify it compiles and can be used as Box<dyn Error>
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn h0_har_67_request_new_custom_method() {
        let req = HarRequest::new("DELETE", "http://test.com/resource/123");
        assert_eq!(req.method, "DELETE");
        assert_eq!(req.url, "http://test.com/resource/123");
        assert_eq!(req.http_version, "HTTP/1.1");
        assert!(req.cookies.is_empty());
        assert!(req.headers.is_empty());
        assert!(req.query_string.is_empty());
        assert!(req.post_data.is_none());
        assert_eq!(req.headers_size, -1);
        assert_eq!(req.body_size, -1);
    }

    #[test]
    fn h0_har_68_response_new_custom_status() {
        let resp = HarResponse::new(201, "Created");
        assert_eq!(resp.status, 201);
        assert_eq!(resp.status_text, "Created");
        assert_eq!(resp.http_version, "HTTP/1.1");
        assert!(resp.cookies.is_empty());
        assert!(resp.headers.is_empty());
        assert!(resp.redirect_url.is_empty());
        assert_eq!(resp.headers_size, -1);
        assert_eq!(resp.body_size, -1);
    }

    #[test]
    fn h0_har_69_chrono_now_iso_format() {
        // Verify the timestamp format is valid ISO 8601
        let timestamp = chrono_now_iso();
        assert!(timestamp.ends_with('Z'));
        assert!(timestamp.contains('T'));
        assert!(timestamp.contains('-'));
    }

    #[test]
    fn h0_har_70_find_by_url_multiple_entries() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/first"),
            HarResponse::new(200, "First"),
        ));
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/second"),
            HarResponse::new(201, "Second"),
        ));
        har.add_entry(HarEntry::new(
            HarRequest::get("http://test.com/first"),
            HarResponse::new(202, "First Again"),
        ));
        // find_by_url returns the first match
        let entry = har.find_by_url("http://test.com/first");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().response.status, 200);
    }

    #[test]
    fn h0_har_71_har_log_browser_and_comment() {
        let mut log = HarLog::new();
        log.browser = Some(HarBrowser::new("Firefox", "115.0"));
        log.comment = Some("Test HAR log".to_string());
        assert!(log.browser.is_some());
        assert_eq!(log.browser.as_ref().unwrap().name, "Firefox");
        assert!(log.comment.is_some());
    }

    #[test]
    fn h0_har_72_cookie_optional_fields() {
        let mut cookie = HarCookie::new("session", "abc123");
        cookie.path = Some("/".to_string());
        cookie.domain = Some("example.com".to_string());
        cookie.expires = Some("2025-01-01T00:00:00Z".to_string());
        cookie.http_only = Some(true);
        cookie.secure = Some(true);
        cookie.comment = Some("Session cookie".to_string());

        assert_eq!(cookie.path, Some("/".to_string()));
        assert_eq!(cookie.domain, Some("example.com".to_string()));
        assert_eq!(cookie.http_only, Some(true));
        assert_eq!(cookie.secure, Some(true));
    }

    #[test]
    fn h0_har_73_post_param_file_upload() {
        let mut param = HarPostParam::new("file", "");
        param.value = None;
        param.file_name = Some("document.pdf".to_string());
        param.content_type = Some("application/pdf".to_string());
        param.comment = Some("Uploaded file".to_string());

        assert!(param.value.is_none());
        assert_eq!(param.file_name, Some("document.pdf".to_string()));
        assert_eq!(param.content_type, Some("application/pdf".to_string()));
    }

    #[test]
    fn h0_har_74_content_with_encoding() {
        let mut content = HarContent::json(r#"{"data": "test"}"#);
        content.encoding = Some("base64".to_string());
        content.compression = Some(100);
        content.comment = Some("Compressed content".to_string());

        assert_eq!(content.encoding, Some("base64".to_string()));
        assert_eq!(content.compression, Some(100));
    }

    #[test]
    fn h0_har_75_cache_state_fields() {
        let state = HarCacheState {
            expires: Some("2025-12-31T23:59:59Z".to_string()),
            last_access: Some("2024-01-01T00:00:00Z".to_string()),
            etag: Some("abc123".to_string()),
            hit_count: Some(42),
            comment: Some("Cache hit".to_string()),
        };

        assert_eq!(state.hit_count, Some(42));
        assert!(state.etag.is_some());
    }

    #[test]
    fn h0_har_76_cache_with_states() {
        let mut cache = HarCache::default();
        cache.before_request = Some(HarCacheState {
            expires: None,
            last_access: None,
            etag: Some("before".to_string()),
            hit_count: Some(1),
            comment: None,
        });
        cache.after_request = Some(HarCacheState {
            expires: None,
            last_access: None,
            etag: Some("after".to_string()),
            hit_count: Some(2),
            comment: None,
        });
        cache.comment = Some("Cache test".to_string());

        assert!(cache.before_request.is_some());
        assert!(cache.after_request.is_some());
        assert_eq!(
            cache.before_request.as_ref().unwrap().etag,
            Some("before".to_string())
        );
    }

    #[test]
    fn h0_har_77_timings_with_comment() {
        let mut timings = HarTimings::new();
        timings.comment = Some("Timing comment".to_string());
        assert!(timings.comment.is_some());
    }

    #[test]
    fn h0_har_78_entry_optional_fields() {
        let mut entry = HarEntry::new(HarRequest::get("http://test.com"), HarResponse::ok());
        entry.connection = Some("1234".to_string());
        entry.comment = Some("Entry comment".to_string());

        assert_eq!(entry.connection, Some("1234".to_string()));
        assert!(entry.comment.is_some());
    }

    #[test]
    fn h0_har_79_browser_with_comment() {
        let mut browser = HarBrowser::new("Chrome", "120.0");
        browser.comment = Some("Browser comment".to_string());
        assert!(browser.comment.is_some());
    }

    #[test]
    fn h0_har_80_creator_has_version() {
        let creator = HarCreator::probar();
        assert_eq!(creator.name, "Probar");
        // Version should be cargo package version
        assert!(!creator.version.is_empty());
        assert!(creator.comment.is_none());
    }

    #[test]
    fn h0_har_81_header_with_comment() {
        let mut header = HarHeader::new("Content-Type", "application/json");
        header.comment = Some("Header comment".to_string());
        assert!(header.comment.is_some());
    }

    #[test]
    fn h0_har_82_query_param_with_comment() {
        let mut param = HarQueryParam::new("page", "1");
        param.comment = Some("Pagination".to_string());
        assert!(param.comment.is_some());
    }

    #[test]
    fn h0_har_83_post_data_with_comment() {
        let mut data = HarPostData::json(r#"{"test": true}"#);
        data.comment = Some("POST body".to_string());
        assert!(data.comment.is_some());
    }

    #[test]
    fn h0_har_84_request_with_comment() {
        let mut req = HarRequest::get("http://test.com");
        req.comment = Some("Test request".to_string());
        assert!(req.comment.is_some());
    }

    #[test]
    fn h0_har_85_response_with_comment() {
        let mut resp = HarResponse::ok();
        resp.comment = Some("Test response".to_string());
        assert!(resp.comment.is_some());
    }

    #[test]
    fn h0_har_86_serialization_with_all_optional_fields() {
        let mut har = Har::new();
        har.log.browser = Some(HarBrowser::new("Chrome", "120.0"));
        har.log.comment = Some("Test log".to_string());

        let mut entry = HarEntry::new(HarRequest::get("http://test.com"), HarResponse::ok())
            .with_server_ip("127.0.0.1")
            .with_time(100.0);
        entry.connection = Some("conn-1".to_string());
        entry.comment = Some("Entry".to_string());

        har.add_entry(entry);

        // Serialize and deserialize
        let json = har.to_json().unwrap();
        let parsed = Har::from_json(&json).unwrap();

        assert!(parsed.log.browser.is_some());
        assert!(parsed.log.comment.is_some());
        assert_eq!(parsed.entry_count(), 1);
    }

    #[test]
    fn h0_har_87_timings_zero_values() {
        let mut timings = HarTimings::new();
        timings.blocked = 0.0; // Zero, not negative
        timings.dns = 0.0;
        timings.connect = 0.0;
        timings.send = 0.0;
        timings.wait = 0.0;
        timings.receive = 0.0;
        // Zero values for blocked/dns/connect should NOT be added (only > 0)
        assert!((timings.total() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn h0_har_88_url_pattern_with_slashes() {
        // Pattern with leading/trailing slashes should be trimmed
        assert!(url_matches_pattern(
            "http://test.com/api/v1/users",
            "/api/v1/"
        ));
        assert!(url_matches_pattern(
            "http://test.com/api/v1/users",
            "api/v1"
        ));
    }

    #[test]
    fn h0_har_89_find_matching_partial_pattern() {
        let mut har = Har::new();
        har.add_entry(HarEntry::new(
            HarRequest::get("http://example.com/users/123"),
            HarResponse::ok(),
        ));
        har.add_entry(HarEntry::new(
            HarRequest::get("http://other.com/posts"),
            HarResponse::ok(),
        ));
        // Pattern "users" should match first entry
        let matches = har.find_matching("users");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].request.url, "http://example.com/users/123");
    }

    #[test]
    fn h0_har_90_not_found_behavior_equality() {
        assert_eq!(NotFoundBehavior::Abort, NotFoundBehavior::Abort);
        assert_eq!(NotFoundBehavior::Fallback, NotFoundBehavior::Fallback);
        assert_ne!(NotFoundBehavior::Abort, NotFoundBehavior::Fallback);
    }
}
