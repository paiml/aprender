//! Pacha protocol loader for data sources and models.
//!
//! Handles `pacha://` URIs for loading data and models from the Sovereign AI Stack.
//!
//! # URI Format
//!
//! ```text
//! pacha://[host]/path/to/resource[?query]
//!
//! Examples:
//! - pacha://data/metrics           - Local data file
//! - pacha://models/classifier      - Local model file
//! - pacha://localhost:8080/api/v1  - Local Pacha server
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Pacha resource types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// Data resource (.ald, .csv, .json)
    Data,
    /// Model resource (.apr)
    Model,
    /// API endpoint
    Api,
}

/// Parsed Pacha URI.
#[derive(Debug, Clone, PartialEq)]
pub struct PachaUri {
    /// Resource type
    pub resource_type: ResourceType,
    /// Host (if remote)
    pub host: Option<String>,
    /// Port (if specified)
    pub port: Option<u16>,
    /// Resource path
    pub path: String,
    /// Query parameters
    pub query: HashMap<String, String>,
}

impl PachaUri {
    /// Parse a pacha:// URI string.
    ///
    /// # Errors
    ///
    /// Returns error if the URI is malformed.
    pub fn parse(uri: &str) -> Result<Self, PachaError> {
        if !uri.starts_with("pacha://") {
            return Err(PachaError::InvalidProtocol(uri.to_string()));
        }

        let rest = &uri[8..]; // Skip "pacha://"

        // Split query string
        let (path_part, query) = if let Some(idx) = rest.find('?') {
            let query_str = &rest[idx + 1..];
            let query = parse_query(query_str);
            (&rest[..idx], query)
        } else {
            (rest, HashMap::new())
        };

        // Check for host:port
        let (host, port, path) = if path_part.contains(':') && !path_part.starts_with('/') {
            // Has host:port
            let parts: Vec<&str> = path_part.splitn(2, '/').collect();
            let host_port = parts[0];
            let path = if parts.len() > 1 {
                format!("/{}", parts[1])
            } else {
                "/".to_string()
            };

            let hp: Vec<&str> = host_port.split(':').collect();
            let host = hp[0].to_string();
            let port = hp.get(1).and_then(|p| p.parse().ok());

            (Some(host), port, path)
        } else if path_part.starts_with('/') {
            (None, None, path_part.to_string())
        } else {
            // No host, just path like "data/metrics"
            (None, None, format!("/{}", path_part))
        };

        // Determine resource type from path
        let resource_type = if path.starts_with("/models") || path.starts_with("/model") {
            ResourceType::Model
        } else if path.starts_with("/api") {
            ResourceType::Api
        } else {
            ResourceType::Data
        };

        Ok(Self {
            resource_type,
            host,
            port,
            path,
            query,
        })
    }

    /// Check if this is a local resource.
    #[must_use]
    pub fn is_local(&self) -> bool {
        self.host.is_none() || self.host.as_deref() == Some("localhost")
    }

    /// Check if this is a remote resource.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        !self.is_local()
    }

    /// Get the local file path for this resource.
    ///
    /// # Arguments
    ///
    /// * `base_dir` - Base directory for resolving relative paths
    #[must_use]
    pub fn to_local_path(&self, base_dir: &Path) -> PathBuf {
        let path = self.path.trim_start_matches('/');
        base_dir.join(path)
    }
}

fn parse_query(query: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in query.split('&') {
        if let Some(idx) = pair.find('=') {
            let key = &pair[..idx];
            let value = &pair[idx + 1..];
            map.insert(key.to_string(), value.to_string());
        }
    }
    map
}

/// Pacha protocol error.
#[derive(Debug, Clone, PartialEq)]
pub enum PachaError {
    /// Invalid protocol (not pacha://)
    InvalidProtocol(String),
    /// Resource not found
    NotFound(String),
    /// Connection error
    ConnectionError(String),
    /// Parse error
    ParseError(String),
    /// IO error
    IoError(String),
    /// Unsupported format
    UnsupportedFormat(String),
}

impl std::fmt::Display for PachaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtocol(uri) => write!(f, "Invalid protocol: {uri}"),
            Self::NotFound(path) => write!(f, "Resource not found: {path}"),
            Self::ConnectionError(msg) => write!(f, "Connection error: {msg}"),
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
        }
    }
}

impl std::error::Error for PachaError {}

/// Pacha resource loader.
pub struct PachaLoader {
    /// Base directory for local resources
    base_dir: PathBuf,
    /// Cache of loaded resources
    cache: HashMap<String, LoadedResource>,
}

/// A loaded resource.
#[derive(Debug, Clone)]
pub struct LoadedResource {
    /// Resource URI
    pub uri: String,
    /// Raw data bytes
    pub data: Vec<u8>,
    /// Content type
    pub content_type: ContentType,
    /// Last modified timestamp
    pub last_modified: Option<u64>,
}

/// Content type of loaded resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Alimentar dataset
    Ald,
    /// Aprender model
    Apr,
    /// JSON data
    Json,
    /// CSV data
    Csv,
    /// Unknown/binary
    Binary,
}

impl ContentType {
    /// Detect content type from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "ald" => Self::Ald,
            "apr" => Self::Apr,
            "json" => Self::Json,
            "csv" => Self::Csv,
            _ => Self::Binary,
        }
    }

    /// Get file extension for content type.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Ald => "ald",
            Self::Apr => "apr",
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Binary => "bin",
        }
    }
}

impl PachaLoader {
    /// Create a new loader with the given base directory.
    #[must_use]
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            cache: HashMap::new(),
        }
    }

    /// Create a loader using current directory.
    #[must_use]
    pub fn current_dir() -> Self {
        Self::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    /// Load a resource from a pacha:// URI.
    ///
    /// # Errors
    ///
    /// Returns error if the resource cannot be loaded.
    pub fn load(&mut self, uri: &str) -> Result<&LoadedResource, PachaError> {
        // Check cache first
        if self.cache.contains_key(uri) {
            return Ok(self.cache.get(uri).unwrap());
        }

        let parsed = PachaUri::parse(uri)?;
        let resource = self.load_uri(&parsed, uri)?;
        self.cache.insert(uri.to_string(), resource);
        Ok(self.cache.get(uri).unwrap())
    }

    /// Load without caching.
    ///
    /// # Errors
    ///
    /// Returns error if the resource cannot be loaded.
    pub fn load_fresh(&self, uri: &str) -> Result<LoadedResource, PachaError> {
        let parsed = PachaUri::parse(uri)?;
        self.load_uri(&parsed, uri)
    }

    fn load_uri(&self, parsed: &PachaUri, uri: &str) -> Result<LoadedResource, PachaError> {
        if parsed.is_remote() && parsed.host.as_deref() != Some("localhost") {
            return Err(PachaError::ConnectionError(
                "Remote Pacha servers not yet supported".to_string(),
            ));
        }

        // Load from local filesystem
        let path = parsed.to_local_path(&self.base_dir);

        // Try with various extensions if no extension specified
        let paths_to_try = if path.extension().is_none() {
            vec![
                path.with_extension("ald"),
                path.with_extension("apr"),
                path.with_extension("json"),
                path.with_extension("csv"),
                path.clone(),
            ]
        } else {
            vec![path.clone()]
        };

        for try_path in paths_to_try {
            if try_path.exists() {
                let data =
                    std::fs::read(&try_path).map_err(|e| PachaError::IoError(e.to_string()))?;

                let content_type = try_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(ContentType::from_extension)
                    .unwrap_or(ContentType::Binary);

                let last_modified = try_path
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());

                return Ok(LoadedResource {
                    uri: uri.to_string(),
                    data,
                    content_type,
                    last_modified,
                });
            }
        }

        Err(PachaError::NotFound(path.display().to_string()))
    }

    /// Clear the resource cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get a cached resource if available.
    #[must_use]
    pub fn get_cached(&self, uri: &str) -> Option<&LoadedResource> {
        self.cache.get(uri)
    }

    /// Check if a resource is cached.
    #[must_use]
    pub fn is_cached(&self, uri: &str) -> bool {
        self.cache.contains_key(uri)
    }
}

// =============================================================================
// HTTP Client Abstraction
// =============================================================================

/// HTTP request method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    /// GET request
    Get,
    /// POST request
    Post,
    /// PUT request
    Put,
    /// DELETE request
    Delete,
}

impl HttpMethod {
    /// Get method as string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
        }
    }
}

/// HTTP request configuration.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// Request URL
    pub url: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request body (for POST/PUT)
    pub body: Option<Vec<u8>>,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

impl HttpRequest {
    /// Create a new GET request.
    #[must_use]
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            method: HttpMethod::Get,
            headers: HashMap::new(),
            body: None,
            timeout_ms: Some(30_000),
        }
    }

    /// Create a new POST request.
    #[must_use]
    pub fn post(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            method: HttpMethod::Post,
            headers: HashMap::new(),
            body: None,
            timeout_ms: Some(30_000),
        }
    }

    /// Set a header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Set the request body.
    #[must_use]
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Set timeout in milliseconds.
    #[must_use]
    pub const fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }
}

/// HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code
    pub status: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Check if the response is successful (2xx).
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// Get a header value.
    #[must_use]
    pub fn get_header(&self, name: &str) -> Option<&str> {
        // Case-insensitive header lookup
        let lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }

    /// Get content type from headers.
    #[must_use]
    pub fn content_type(&self) -> Option<&str> {
        self.get_header("content-type")
    }

    /// Detect ContentType from response.
    #[must_use]
    pub fn detect_content_type(&self) -> ContentType {
        if let Some(ct) = self.content_type() {
            if ct.contains("json") {
                return ContentType::Json;
            }
            if ct.contains("csv") {
                return ContentType::Csv;
            }
        }
        ContentType::Binary
    }
}

/// HTTP client trait for platform-specific implementations.
pub trait HttpClient: Send + Sync {
    /// Perform an HTTP request.
    ///
    /// # Errors
    ///
    /// Returns error if the request fails.
    fn request(&self, req: HttpRequest) -> Result<HttpResponse, PachaError>;
}

/// A no-op HTTP client that always returns an error.
/// Used as fallback when no real client is available.
#[derive(Debug, Default)]
pub struct NoopHttpClient;

impl HttpClient for NoopHttpClient {
    fn request(&self, _req: HttpRequest) -> Result<HttpResponse, PachaError> {
        Err(PachaError::ConnectionError(
            "HTTP client not available - use WASM WebFetch or configure a client".to_string(),
        ))
    }
}

/// Retry configuration for HTTP requests.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries (milliseconds)
    pub initial_delay_ms: u64,
    /// Maximum delay between retries (milliseconds)
    pub max_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 500,
            max_delay_ms: 10_000,
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Calculate delay for a given attempt number.
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        if attempt == 0 {
            return 0;
        }
        let delay = self.initial_delay_ms as f32 * self.backoff_multiplier.powi(attempt as i32 - 1);
        (delay as u64).min(self.max_delay_ms)
    }

    /// Check if we should retry.
    #[must_use]
    pub const fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
}

/// Remote Pacha loader with HTTP support.
pub struct RemotePachaLoader<C: HttpClient = NoopHttpClient> {
    /// HTTP client
    client: C,
    /// Retry configuration
    retry_config: RetryConfig,
    /// Cache of loaded resources
    cache: HashMap<String, LoadedResource>,
    /// Cache TTL in milliseconds (None = no expiry)
    cache_ttl_ms: Option<u64>,
}

impl<C: HttpClient> RemotePachaLoader<C> {
    /// Create a new remote loader with the given client.
    #[must_use]
    pub fn new(client: C) -> Self {
        Self {
            client,
            retry_config: RetryConfig::default(),
            cache: HashMap::new(),
            cache_ttl_ms: None,
        }
    }

    /// Set retry configuration.
    #[must_use]
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Set cache TTL in milliseconds.
    #[must_use]
    pub const fn with_cache_ttl(mut self, ms: u64) -> Self {
        self.cache_ttl_ms = Some(ms);
        self
    }

    /// Load a resource from a remote pacha:// URI.
    ///
    /// # Errors
    ///
    /// Returns error if the resource cannot be loaded.
    pub fn load(&mut self, uri: &str) -> Result<&LoadedResource, PachaError> {
        // Check cache first
        if self.cache.contains_key(uri) {
            return Ok(self.cache.get(uri).expect("just checked"));
        }

        let resource = self.load_fresh(uri)?;
        self.cache.insert(uri.to_string(), resource);
        Ok(self.cache.get(uri).expect("just inserted"))
    }

    /// Load a resource without caching.
    ///
    /// # Errors
    ///
    /// Returns error if the resource cannot be loaded.
    pub fn load_fresh(&self, uri: &str) -> Result<LoadedResource, PachaError> {
        let parsed = PachaUri::parse(uri)?;

        if !parsed.is_remote() {
            return Err(PachaError::ConnectionError(
                "Use PachaLoader for local resources".to_string(),
            ));
        }

        let http_url = self.build_http_url(&parsed);

        // Build request with retry support
        let mut last_error = PachaError::ConnectionError("No attempts made".to_string());

        for attempt in 0..self.retry_config.max_attempts {
            if attempt > 0 {
                // Would sleep here in real impl, but we're sync
                // In WASM, the caller would handle async retry
            }

            let req = HttpRequest::get(&http_url)
                .with_header("Accept", "application/json, application/octet-stream, */*")
                .with_header("User-Agent", "Presentar/0.1");

            match self.client.request(req) {
                Ok(response) if response.is_success() => {
                    let content_type = response.detect_content_type();
                    return Ok(LoadedResource {
                        uri: uri.to_string(),
                        data: response.body,
                        content_type,
                        last_modified: None, // Could parse from headers
                    });
                }
                Ok(response) => {
                    last_error =
                        PachaError::ConnectionError(format!("HTTP {} error", response.status));
                    // Don't retry 4xx errors
                    if response.status >= 400 && response.status < 500 {
                        break;
                    }
                }
                Err(e) => {
                    last_error = e;
                }
            }
        }

        Err(last_error)
    }

    /// Build HTTP URL from parsed Pacha URI.
    fn build_http_url(&self, parsed: &PachaUri) -> String {
        let scheme = "https"; // Default to HTTPS for remote
        let host = parsed.host.as_deref().unwrap_or("localhost");
        let port = parsed.port.map_or(String::new(), |p| format!(":{p}"));

        let mut url = format!("{scheme}://{host}{port}{}", parsed.path);

        if !parsed.query.is_empty() {
            let query: Vec<String> = parsed
                .query
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            url.push('?');
            url.push_str(&query.join("&"));
        }

        url
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Parse refresh interval string to milliseconds.
///
/// Supports formats: "1s", "5m", "1h", "30s", "100ms"
///
/// # Errors
///
/// Returns None if the format is invalid.
#[must_use]
pub fn parse_refresh_interval(interval: &str) -> Option<u64> {
    let interval = interval.trim();
    if interval.is_empty() {
        return None;
    }

    // Find where the number ends and unit begins
    let mut num_end = interval.len();
    for (i, c) in interval.char_indices() {
        if !c.is_ascii_digit() && c != '.' {
            num_end = i;
            break;
        }
    }

    if num_end == 0 {
        return None;
    }

    let num: f64 = interval[..num_end].parse().ok()?;
    let unit = &interval[num_end..];

    let ms = match unit {
        "ms" => num,
        "s" => num * 1000.0,
        "m" => num * 60_000.0,
        "h" => num * 3_600_000.0,
        "" => num * 1000.0, // Default to seconds
        _ => return None,
    };

    Some(ms as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PachaUri parsing tests
    // =========================================================================

    #[test]
    fn test_parse_simple_data_uri() {
        let uri = PachaUri::parse("pacha://data/metrics").unwrap();
        assert_eq!(uri.resource_type, ResourceType::Data);
        assert!(uri.host.is_none());
        assert!(uri.port.is_none());
        assert_eq!(uri.path, "/data/metrics");
        assert!(uri.is_local());
    }

    #[test]
    fn test_parse_model_uri() {
        let uri = PachaUri::parse("pacha://models/classifier").unwrap();
        assert_eq!(uri.resource_type, ResourceType::Model);
        assert_eq!(uri.path, "/models/classifier");
    }

    #[test]
    fn test_parse_api_uri() {
        let uri = PachaUri::parse("pacha://api/v1/data").unwrap();
        assert_eq!(uri.resource_type, ResourceType::Api);
        assert_eq!(uri.path, "/api/v1/data");
    }

    #[test]
    fn test_parse_uri_with_host() {
        let uri = PachaUri::parse("pacha://localhost:8080/data/metrics").unwrap();
        assert_eq!(uri.host, Some("localhost".to_string()));
        assert_eq!(uri.port, Some(8080));
        assert_eq!(uri.path, "/data/metrics");
        assert!(uri.is_local()); // localhost is still local
    }

    #[test]
    fn test_parse_uri_with_remote_host() {
        let uri = PachaUri::parse("pacha://server.example.com:9000/api/v1").unwrap();
        assert_eq!(uri.host, Some("server.example.com".to_string()));
        assert_eq!(uri.port, Some(9000));
        assert!(uri.is_remote());
    }

    #[test]
    fn test_parse_uri_with_query() {
        let uri = PachaUri::parse("pacha://data/metrics?limit=100&format=json").unwrap();
        assert_eq!(uri.query.get("limit"), Some(&"100".to_string()));
        assert_eq!(uri.query.get("format"), Some(&"json".to_string()));
    }

    #[test]
    fn test_parse_invalid_protocol() {
        let result = PachaUri::parse("http://example.com");
        assert!(matches!(result, Err(PachaError::InvalidProtocol(_))));
    }

    #[test]
    fn test_parse_empty_path() {
        let uri = PachaUri::parse("pacha://localhost:8080").unwrap();
        assert_eq!(uri.path, "/");
    }

    // =========================================================================
    // ResourceType tests
    // =========================================================================

    #[test]
    fn test_resource_type_detection() {
        assert_eq!(
            PachaUri::parse("pacha://data/foo").unwrap().resource_type,
            ResourceType::Data
        );
        assert_eq!(
            PachaUri::parse("pacha://models/bar").unwrap().resource_type,
            ResourceType::Model
        );
        assert_eq!(
            PachaUri::parse("pacha://model/baz").unwrap().resource_type,
            ResourceType::Model
        );
        assert_eq!(
            PachaUri::parse("pacha://api/v1").unwrap().resource_type,
            ResourceType::Api
        );
    }

    // =========================================================================
    // ContentType tests
    // =========================================================================

    #[test]
    fn test_content_type_from_extension() {
        assert_eq!(ContentType::from_extension("ald"), ContentType::Ald);
        assert_eq!(ContentType::from_extension("apr"), ContentType::Apr);
        assert_eq!(ContentType::from_extension("json"), ContentType::Json);
        assert_eq!(ContentType::from_extension("csv"), ContentType::Csv);
        assert_eq!(ContentType::from_extension("unknown"), ContentType::Binary);
    }

    #[test]
    fn test_content_type_extension() {
        assert_eq!(ContentType::Ald.extension(), "ald");
        assert_eq!(ContentType::Apr.extension(), "apr");
        assert_eq!(ContentType::Json.extension(), "json");
        assert_eq!(ContentType::Csv.extension(), "csv");
        assert_eq!(ContentType::Binary.extension(), "bin");
    }

    // =========================================================================
    // Local path tests
    // =========================================================================

    #[test]
    fn test_to_local_path() {
        let uri = PachaUri::parse("pacha://data/metrics").unwrap();
        let path = uri.to_local_path(Path::new("/app"));
        assert_eq!(path, PathBuf::from("/app/data/metrics"));
    }

    #[test]
    fn test_to_local_path_nested() {
        let uri = PachaUri::parse("pacha://data/nested/deep/file").unwrap();
        let path = uri.to_local_path(Path::new("/base"));
        assert_eq!(path, PathBuf::from("/base/data/nested/deep/file"));
    }

    // =========================================================================
    // PachaLoader tests
    // =========================================================================

    #[test]
    fn test_loader_new() {
        let loader = PachaLoader::new(PathBuf::from("/test"));
        assert!(!loader.is_cached("pacha://data/test"));
    }

    #[test]
    fn test_loader_not_found() {
        let loader = PachaLoader::new(PathBuf::from("/nonexistent"));
        let result = loader.load_fresh("pacha://data/missing");
        assert!(matches!(result, Err(PachaError::NotFound(_))));
    }

    #[test]
    fn test_loader_remote_not_supported() {
        let loader = PachaLoader::new(PathBuf::from("."));
        let result = loader.load_fresh("pacha://remote.server.com:8080/data");
        assert!(matches!(result, Err(PachaError::ConnectionError(_))));
    }

    // =========================================================================
    // Refresh interval parsing tests
    // =========================================================================

    #[test]
    fn test_parse_refresh_seconds() {
        assert_eq!(parse_refresh_interval("1s"), Some(1000));
        assert_eq!(parse_refresh_interval("30s"), Some(30000));
        assert_eq!(parse_refresh_interval("5"), Some(5000)); // Default to seconds
    }

    #[test]
    fn test_parse_refresh_milliseconds() {
        assert_eq!(parse_refresh_interval("100ms"), Some(100));
        assert_eq!(parse_refresh_interval("500ms"), Some(500));
    }

    #[test]
    fn test_parse_refresh_minutes() {
        assert_eq!(parse_refresh_interval("1m"), Some(60000));
        assert_eq!(parse_refresh_interval("5m"), Some(300_000));
    }

    #[test]
    fn test_parse_refresh_hours() {
        assert_eq!(parse_refresh_interval("1h"), Some(3_600_000));
        assert_eq!(parse_refresh_interval("2h"), Some(7_200_000));
    }

    #[test]
    fn test_parse_refresh_fractional() {
        assert_eq!(parse_refresh_interval("1.5s"), Some(1500));
        assert_eq!(parse_refresh_interval("0.5m"), Some(30000));
    }

    #[test]
    fn test_parse_refresh_invalid() {
        assert_eq!(parse_refresh_interval(""), None);
        assert_eq!(parse_refresh_interval("abc"), None);
        assert_eq!(parse_refresh_interval("1x"), None);
    }

    // =========================================================================
    // PachaError display tests
    // =========================================================================

    #[test]
    fn test_error_display() {
        let err = PachaError::NotFound("test/path".to_string());
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("test/path"));
    }

    #[test]
    fn test_error_types() {
        assert!(matches!(
            PachaError::InvalidProtocol("x".to_string()),
            PachaError::InvalidProtocol(_)
        ));
        assert!(matches!(
            PachaError::ConnectionError("x".to_string()),
            PachaError::ConnectionError(_)
        ));
        assert!(matches!(
            PachaError::ParseError("x".to_string()),
            PachaError::ParseError(_)
        ));
        assert!(matches!(
            PachaError::IoError("x".to_string()),
            PachaError::IoError(_)
        ));
        assert!(matches!(
            PachaError::UnsupportedFormat("x".to_string()),
            PachaError::UnsupportedFormat(_)
        ));
    }

    // =========================================================================
    // HTTP Client tests
    // =========================================================================

    #[test]
    fn test_http_method_as_str() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert_eq!(HttpMethod::Post.as_str(), "POST");
        assert_eq!(HttpMethod::Put.as_str(), "PUT");
        assert_eq!(HttpMethod::Delete.as_str(), "DELETE");
    }

    #[test]
    fn test_http_request_get() {
        let req = HttpRequest::get("https://example.com/data");
        assert_eq!(req.url, "https://example.com/data");
        assert_eq!(req.method, HttpMethod::Get);
        assert!(req.body.is_none());
        assert_eq!(req.timeout_ms, Some(30_000));
    }

    #[test]
    fn test_http_request_post() {
        let req = HttpRequest::post("https://example.com/api");
        assert_eq!(req.method, HttpMethod::Post);
    }

    #[test]
    fn test_http_request_with_header() {
        let req = HttpRequest::get("http://test.com")
            .with_header("Authorization", "Bearer token123")
            .with_header("Content-Type", "application/json");

        assert_eq!(
            req.headers.get("Authorization"),
            Some(&"Bearer token123".to_string())
        );
        assert_eq!(
            req.headers.get("Content-Type"),
            Some(&"application/json".to_string())
        );
    }

    #[test]
    fn test_http_request_with_body() {
        let body = b"test body".to_vec();
        let req = HttpRequest::post("http://test.com").with_body(body.clone());
        assert_eq!(req.body, Some(body));
    }

    #[test]
    fn test_http_request_with_timeout() {
        let req = HttpRequest::get("http://test.com").with_timeout(5000);
        assert_eq!(req.timeout_ms, Some(5000));
    }

    #[test]
    fn test_http_response_is_success() {
        let make_response = |status| HttpResponse {
            status,
            headers: HashMap::new(),
            body: vec![],
        };

        assert!(make_response(200).is_success());
        assert!(make_response(201).is_success());
        assert!(make_response(299).is_success());
        assert!(!make_response(300).is_success());
        assert!(!make_response(400).is_success());
        assert!(!make_response(500).is_success());
    }

    #[test]
    fn test_http_response_get_header() {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Custom".to_string(), "value".to_string());

        let response = HttpResponse {
            status: 200,
            headers,
            body: vec![],
        };

        // Case-insensitive lookup
        assert_eq!(
            response.get_header("content-type"),
            Some("application/json")
        );
        assert_eq!(
            response.get_header("Content-Type"),
            Some("application/json")
        );
        assert_eq!(
            response.get_header("CONTENT-TYPE"),
            Some("application/json")
        );
        assert_eq!(response.get_header("x-custom"), Some("value"));
        assert!(response.get_header("nonexistent").is_none());
    }

    #[test]
    fn test_http_response_detect_content_type() {
        let make_response = |ct: &str| {
            let mut headers = HashMap::new();
            headers.insert("Content-Type".to_string(), ct.to_string());
            HttpResponse {
                status: 200,
                headers,
                body: vec![],
            }
        };

        assert_eq!(
            make_response("application/json").detect_content_type(),
            ContentType::Json
        );
        assert_eq!(
            make_response("application/json; charset=utf-8").detect_content_type(),
            ContentType::Json
        );
        assert_eq!(
            make_response("text/csv").detect_content_type(),
            ContentType::Csv
        );
        assert_eq!(
            make_response("application/octet-stream").detect_content_type(),
            ContentType::Binary
        );
    }

    #[test]
    fn test_noop_http_client() {
        let client = NoopHttpClient;
        let req = HttpRequest::get("http://test.com");
        let result = client.request(req);
        assert!(matches!(result, Err(PachaError::ConnectionError(_))));
    }

    // =========================================================================
    // RetryConfig tests
    // =========================================================================

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay_ms, 500);
        assert_eq!(config.max_delay_ms, 10_000);
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_retry_config_delay_for_attempt() {
        let config = RetryConfig::default();

        assert_eq!(config.delay_for_attempt(0), 0);
        assert_eq!(config.delay_for_attempt(1), 500); // initial
        assert_eq!(config.delay_for_attempt(2), 1000); // 500 * 2
        assert_eq!(config.delay_for_attempt(3), 2000); // 500 * 4
    }

    #[test]
    fn test_retry_config_delay_capped() {
        let config = RetryConfig {
            max_delay_ms: 1000,
            ..RetryConfig::default()
        };

        // Should be capped at max_delay_ms
        assert_eq!(config.delay_for_attempt(5), 1000);
    }

    #[test]
    fn test_retry_config_should_retry() {
        let config = RetryConfig {
            max_attempts: 3,
            ..RetryConfig::default()
        };

        assert!(config.should_retry(0));
        assert!(config.should_retry(1));
        assert!(config.should_retry(2));
        assert!(!config.should_retry(3));
        assert!(!config.should_retry(4));
    }

    // =========================================================================
    // RemotePachaLoader tests
    // =========================================================================

    struct MockHttpClient {
        response: HttpResponse,
    }

    impl HttpClient for MockHttpClient {
        fn request(&self, _req: HttpRequest) -> Result<HttpResponse, PachaError> {
            Ok(self.response.clone())
        }
    }

    #[test]
    fn test_remote_loader_new() {
        let loader = RemotePachaLoader::new(NoopHttpClient);
        assert!(loader.cache.is_empty());
    }

    #[test]
    fn test_remote_loader_with_retry() {
        let config = RetryConfig {
            max_attempts: 5,
            ..RetryConfig::default()
        };
        let loader = RemotePachaLoader::new(NoopHttpClient).with_retry(config);
        assert_eq!(loader.retry_config.max_attempts, 5);
    }

    #[test]
    fn test_remote_loader_with_cache_ttl() {
        let loader = RemotePachaLoader::new(NoopHttpClient).with_cache_ttl(60_000);
        assert_eq!(loader.cache_ttl_ms, Some(60_000));
    }

    #[test]
    fn test_remote_loader_load_fresh_local_fails() {
        let loader = RemotePachaLoader::new(NoopHttpClient);
        let result = loader.load_fresh("pacha://data/local");
        assert!(matches!(result, Err(PachaError::ConnectionError(_))));
    }

    #[test]
    fn test_remote_loader_load_success() {
        let client = MockHttpClient {
            response: HttpResponse {
                status: 200,
                headers: {
                    let mut h = HashMap::new();
                    h.insert("Content-Type".to_string(), "application/json".to_string());
                    h
                },
                body: b"{}".to_vec(),
            },
        };

        let loader = RemotePachaLoader::new(client);
        let result = loader.load_fresh("pacha://remote.server.com:8080/api/data");
        assert!(result.is_ok());

        let resource = result.unwrap();
        assert_eq!(resource.data, b"{}");
        assert_eq!(resource.content_type, ContentType::Json);
    }

    #[test]
    fn test_remote_loader_caching() {
        let client = MockHttpClient {
            response: HttpResponse {
                status: 200,
                headers: HashMap::new(),
                body: b"test".to_vec(),
            },
        };

        let mut loader = RemotePachaLoader::new(client);

        // First load - use host:port format for remote
        let result = loader.load("pacha://server.com:8080/data");
        assert!(result.is_ok());
        assert!(!loader.cache.is_empty());

        // Clear cache
        loader.clear_cache();
        assert!(loader.cache.is_empty());
    }

    #[test]
    fn test_build_http_url() {
        let loader = RemotePachaLoader::new(NoopHttpClient);

        let uri1 = PachaUri::parse("pacha://server.com:8080/api/data").unwrap();
        assert_eq!(
            loader.build_http_url(&uri1),
            "https://server.com:8080/api/data"
        );

        let uri2 = PachaUri::parse("pacha://server.com/data?limit=10").unwrap();
        assert!(loader.build_http_url(&uri2).contains("limit=10"));
    }

    // ===== Additional Coverage Tests =====

    #[test]
    fn test_pacha_error_display_all_variants() {
        assert!(PachaError::InvalidProtocol("http://x".to_string())
            .to_string()
            .contains("Invalid protocol"));
        assert!(PachaError::NotFound("path".to_string())
            .to_string()
            .contains("not found"));
        assert!(PachaError::ConnectionError("timeout".to_string())
            .to_string()
            .contains("Connection error"));
        assert!(PachaError::ParseError("bad json".to_string())
            .to_string()
            .contains("Parse error"));
        assert!(PachaError::IoError("disk full".to_string())
            .to_string()
            .contains("IO error"));
        assert!(PachaError::UnsupportedFormat("xyz".to_string())
            .to_string()
            .contains("Unsupported format"));
    }

    #[test]
    fn test_pacha_error_is_error_trait() {
        let err = PachaError::NotFound("test".to_string());
        // Just verify it implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_loader_current_dir() {
        let loader = PachaLoader::current_dir();
        // Just verify it doesn't panic
        assert!(!loader.is_cached("pacha://data/nonexistent"));
    }

    #[test]
    fn test_loader_cache_operations() {
        let loader = PachaLoader::new(PathBuf::from("/tmp"));
        assert!(!loader.is_cached("pacha://data/test"));
        assert!(loader.get_cached("pacha://data/test").is_none());
    }

    #[test]
    fn test_resource_type_clone() {
        let rt = ResourceType::Model;
        let cloned = rt;
        assert_eq!(cloned, ResourceType::Model);
    }

    #[test]
    fn test_resource_type_debug() {
        let rt = ResourceType::Api;
        let debug = format!("{:?}", rt);
        assert!(debug.contains("Api"));
    }

    #[test]
    fn test_pacha_uri_clone() {
        let uri = PachaUri::parse("pacha://data/test").unwrap();
        let cloned = uri;
        assert_eq!(cloned.path, "/data/test");
    }

    #[test]
    fn test_loaded_resource_clone() {
        let resource = LoadedResource {
            uri: "pacha://data/test".to_string(),
            data: vec![1, 2, 3],
            content_type: ContentType::Json,
            last_modified: Some(12345),
        };
        let cloned = resource;
        assert_eq!(cloned.uri, "pacha://data/test");
        assert_eq!(cloned.data, vec![1, 2, 3]);
        assert_eq!(cloned.content_type, ContentType::Json);
        assert_eq!(cloned.last_modified, Some(12345));
    }

    #[test]
    fn test_content_type_clone() {
        let ct = ContentType::Csv;
        let cloned = ct;
        assert_eq!(cloned, ContentType::Csv);
    }

    #[test]
    fn test_content_type_from_extension_uppercase() {
        assert_eq!(ContentType::from_extension("JSON"), ContentType::Json);
        assert_eq!(ContentType::from_extension("CSV"), ContentType::Csv);
        assert_eq!(ContentType::from_extension("ALD"), ContentType::Ald);
        assert_eq!(ContentType::from_extension("APR"), ContentType::Apr);
    }

    #[test]
    fn test_http_response_content_type() {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "text/plain".to_string());
        let response = HttpResponse {
            status: 200,
            headers,
            body: vec![],
        };
        assert_eq!(response.content_type(), Some("text/plain"));
    }

    #[test]
    fn test_http_response_no_content_type() {
        let response = HttpResponse {
            status: 200,
            headers: HashMap::new(),
            body: vec![],
        };
        assert!(response.content_type().is_none());
        assert_eq!(response.detect_content_type(), ContentType::Binary);
    }

    #[test]
    fn test_noop_http_client_default() {
        let client = NoopHttpClient;
        let req = HttpRequest::get("http://test.com");
        assert!(client.request(req).is_err());
    }

    #[test]
    fn test_http_method_clone() {
        let method = HttpMethod::Put;
        let cloned = method;
        assert_eq!(cloned, HttpMethod::Put);
    }

    #[test]
    fn test_http_method_debug() {
        let method = HttpMethod::Delete;
        let debug = format!("{:?}", method);
        assert!(debug.contains("Delete"));
    }

    #[test]
    fn test_http_request_clone() {
        let req = HttpRequest::get("http://test.com")
            .with_header("X-Test", "value")
            .with_timeout(5000);
        let cloned = req;
        assert_eq!(cloned.url, "http://test.com");
        assert_eq!(cloned.timeout_ms, Some(5000));
    }

    #[test]
    fn test_retry_config_clone() {
        let config = RetryConfig {
            max_attempts: 5,
            ..RetryConfig::default()
        };
        let cloned = config;
        assert_eq!(cloned.max_attempts, 5);
    }

    #[test]
    fn test_parse_absolute_path() {
        let uri = PachaUri::parse("pacha:///absolute/path").unwrap();
        assert_eq!(uri.path, "/absolute/path");
        assert!(uri.is_local());
    }

    #[test]
    fn test_remote_loader_4xx_no_retry() {
        struct Mock4xx;
        impl HttpClient for Mock4xx {
            fn request(&self, _req: HttpRequest) -> Result<HttpResponse, PachaError> {
                Ok(HttpResponse {
                    status: 404,
                    headers: HashMap::new(),
                    body: vec![],
                })
            }
        }

        let loader = RemotePachaLoader::new(Mock4xx).with_retry(RetryConfig {
            max_attempts: 3,
            ..RetryConfig::default()
        });

        let result = loader.load_fresh("pacha://remote.server.com:8080/api/data");
        assert!(matches!(result, Err(PachaError::ConnectionError(_))));
    }

    #[test]
    fn test_remote_loader_5xx_retries() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        struct Mock5xx {
            attempts: Arc<AtomicU32>,
        }

        impl HttpClient for Mock5xx {
            fn request(&self, _req: HttpRequest) -> Result<HttpResponse, PachaError> {
                self.attempts.fetch_add(1, Ordering::SeqCst);
                Ok(HttpResponse {
                    status: 500,
                    headers: HashMap::new(),
                    body: vec![],
                })
            }
        }

        let attempts = Arc::new(AtomicU32::new(0));
        let loader = RemotePachaLoader::new(Mock5xx {
            attempts: attempts.clone(),
        })
        .with_retry(RetryConfig {
            max_attempts: 3,
            initial_delay_ms: 0,
            max_delay_ms: 0,
            backoff_multiplier: 1.0,
        });

        let _ = loader.load_fresh("pacha://remote.server.com:8080/api/data");
        // Should attempt max_attempts times
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
