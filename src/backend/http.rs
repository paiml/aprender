//! HTTP/HTTPS storage backend (read-only).
//!
//! Provides read-only access to datasets hosted on HTTP/HTTPS servers.
//! Useful for accessing public datasets without requiring cloud credentials.

use bytes::Bytes;
use reqwest::{
    blocking::Client,
    header::{CONTENT_LENGTH, RANGE},
};

use super::StorageBackend;
use crate::error::{Error, Result};

// ============================================================================
// HTTP Client Trait for Dependency Injection and Testing
// ============================================================================

/// Response from an HTTP operation.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Whether the status indicates success (2xx).
    pub is_success: bool,
    /// Response body bytes.
    pub body: Bytes,
    /// Content-Length header value, if present.
    pub content_length: Option<u64>,
    /// Accept-Ranges header value, if present.
    pub accept_ranges: Option<String>,
}

impl HttpResponse {
    /// Creates a successful response with the given body.
    #[cfg(test)]
    pub fn ok(body: impl Into<Bytes>) -> Self {
        Self {
            status: 200,
            is_success: true,
            body: body.into(),
            content_length: None,
            accept_ranges: None,
        }
    }

    /// Creates a 404 Not Found response.
    #[cfg(test)]
    pub fn not_found() -> Self {
        Self {
            status: 404,
            is_success: false,
            body: Bytes::new(),
            content_length: None,
            accept_ranges: None,
        }
    }

    /// Creates a 206 Partial Content response.
    #[cfg(test)]
    pub fn partial_content(body: impl Into<Bytes>) -> Self {
        Self {
            status: 206,
            is_success: true,
            body: body.into(),
            content_length: None,
            accept_ranges: Some("bytes".to_string()),
        }
    }

    /// Sets the content length.
    #[cfg(test)]
    pub fn with_content_length(mut self, length: u64) -> Self {
        self.content_length = Some(length);
        self
    }

    /// Sets accept ranges.
    #[cfg(test)]
    pub fn with_accept_ranges(mut self, value: impl Into<String>) -> Self {
        self.accept_ranges = Some(value.into());
        self
    }
}

/// Trait for HTTP client operations.
///
/// This trait abstracts HTTP operations to allow for testing with mock
/// implementations. The real implementation uses reqwest, while tests can use
/// `MockHttpClient`.
pub trait HttpClient: Send + Sync {
    /// Performs an HTTP GET request.
    fn get(&self, url: &str) -> Result<HttpResponse>;

    /// Performs an HTTP HEAD request.
    fn head(&self, url: &str) -> Result<HttpResponse>;

    /// Performs an HTTP GET request with a Range header.
    fn get_range(&self, url: &str, start: u64, end: u64) -> Result<HttpResponse>;
}

/// Real HTTP client implementation using reqwest.
#[derive(Debug)]
pub struct ReqwestHttpClient {
    client: Client,
}

impl ReqwestHttpClient {
    /// Creates a new reqwest-based HTTP client.
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .user_agent("alimentar/0.1.0")
            .build()
            .map_err(|e| Error::storage(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client })
    }

    /// Creates a new reqwest-based HTTP client with a timeout.
    pub fn with_timeout(timeout_secs: u64) -> Result<Self> {
        let client = Client::builder()
            .user_agent("alimentar/0.1.0")
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| Error::storage(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client })
    }
}

impl HttpClient for ReqwestHttpClient {
    fn get(&self, url: &str) -> Result<HttpResponse> {
        let response = self
            .client
            .get(url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP GET error for '{}': {}", url, e)))?;

        let status = response.status().as_u16();
        let is_success = response.status().is_success();
        let content_length = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok());
        let accept_ranges = response
            .headers()
            .get("accept-ranges")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        let body = response
            .bytes()
            .map_err(|e| Error::storage(format!("Failed to read HTTP response body: {e}")))?;

        Ok(HttpResponse {
            status,
            is_success,
            body,
            content_length,
            accept_ranges,
        })
    }

    fn head(&self, url: &str) -> Result<HttpResponse> {
        let response = self
            .client
            .head(url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP HEAD error for '{}': {}", url, e)))?;

        let status = response.status().as_u16();
        let is_success = response.status().is_success();
        let content_length = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok());
        let accept_ranges = response
            .headers()
            .get("accept-ranges")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        Ok(HttpResponse {
            status,
            is_success,
            body: Bytes::new(),
            content_length,
            accept_ranges,
        })
    }

    fn get_range(&self, url: &str, start: u64, end: u64) -> Result<HttpResponse> {
        let range_header = format!("bytes={}-{}", start, end);

        let response = self
            .client
            .get(url)
            .header(RANGE, range_header)
            .send()
            .map_err(|e| Error::storage(format!("HTTP GET range error for '{}': {}", url, e)))?;

        let status = response.status().as_u16();
        let is_success = status == 206 || response.status().is_success();
        let content_length = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok());
        let accept_ranges = response
            .headers()
            .get("accept-ranges")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        let body = response
            .bytes()
            .map_err(|e| Error::storage(format!("Failed to read HTTP response body: {e}")))?;

        Ok(HttpResponse {
            status,
            is_success,
            body,
            content_length,
            accept_ranges,
        })
    }
}

/// Mock HTTP client for testing.
///
/// This client allows tests to configure expected responses for specific URLs
/// without making actual HTTP requests.
#[cfg(test)]
#[derive(Debug, Default, Clone)]
pub struct MockHttpClient {
    /// Responses to return for GET requests, keyed by URL.
    get_responses: std::collections::HashMap<String, HttpResponse>,
    /// Responses to return for HEAD requests, keyed by URL.
    head_responses: std::collections::HashMap<String, HttpResponse>,
    /// Default response for URLs not in the map.
    default_response: Option<HttpResponse>,
}

#[cfg(test)]
impl MockHttpClient {
    /// Creates a new mock HTTP client.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a GET response for a specific URL.
    pub fn with_get_response(mut self, url: impl Into<String>, response: HttpResponse) -> Self {
        self.get_responses.insert(url.into(), response);
        self
    }

    /// Adds a HEAD response for a specific URL.
    pub fn with_head_response(mut self, url: impl Into<String>, response: HttpResponse) -> Self {
        self.head_responses.insert(url.into(), response);
        self
    }

    /// Sets the default response for URLs not in the map.
    pub fn with_default_response(mut self, response: HttpResponse) -> Self {
        self.default_response = Some(response);
        self
    }
}

#[cfg(test)]
impl HttpClient for MockHttpClient {
    fn get(&self, url: &str) -> Result<HttpResponse> {
        if let Some(response) = self.get_responses.get(url) {
            return Ok(response.clone());
        }
        if let Some(ref default) = self.default_response {
            return Ok(default.clone());
        }
        Err(Error::storage(format!("No mock response for GET {}", url)))
    }

    fn head(&self, url: &str) -> Result<HttpResponse> {
        if let Some(response) = self.head_responses.get(url) {
            return Ok(response.clone());
        }
        if let Some(ref default) = self.default_response {
            return Ok(default.clone());
        }
        Err(Error::storage(format!("No mock response for HEAD {}", url)))
    }

    fn get_range(&self, url: &str, _start: u64, _end: u64) -> Result<HttpResponse> {
        // Use same responses as GET for simplicity
        self.get(url)
    }
}

/// A read-only storage backend using HTTP/HTTPS.
///
/// This backend is designed for accessing publicly hosted datasets
/// over HTTP/HTTPS. It supports range requests for efficient partial
/// reads when the server supports them.
///
/// # Limitations
///
/// - Read-only: `put` and `delete` operations will return errors
/// - `list` is not supported (HTTP doesn't have directory listings)
///
/// # Example
///
/// ```no_run
/// use alimentar::backend::{HttpBackend, StorageBackend};
///
/// let backend = HttpBackend::new("https://huggingface.co/datasets").unwrap();
/// let data = backend.get("squad/train.parquet").unwrap();
/// ```
#[derive(Debug)]
pub struct HttpBackend {
    client: Client,
    base_url: String,
}

impl HttpBackend {
    /// Creates a new HTTP backend with the given base URL.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL for all requests. Keys will be appended to this.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let client = Client::builder()
            .user_agent("alimentar/0.1.0")
            .build()
            .map_err(|e| Error::storage(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client, base_url })
    }

    /// Creates a new HTTP backend with custom client configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL for all requests
    /// * `timeout_secs` - Request timeout in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn with_timeout(base_url: impl Into<String>, timeout_secs: u64) -> Result<Self> {
        let base_url = base_url.into();
        let client = Client::builder()
            .user_agent("alimentar/0.1.0")
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| Error::storage(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client, base_url })
    }

    /// Returns the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Constructs the full URL for a key.
    fn url_for(&self, key: &str) -> String {
        if self.base_url.ends_with('/') {
            format!("{}{}", self.base_url, key)
        } else {
            format!("{}/{}", self.base_url, key)
        }
    }
}

impl StorageBackend for HttpBackend {
    fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        // HTTP doesn't support directory listings
        Err(Error::storage(
            "HTTP backend does not support listing (use a specific key instead)",
        ))
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        let url = self.url_for(key);

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP GET error for '{}': {}", url, e)))?;

        if !response.status().is_success() {
            return Err(Error::storage(format!(
                "HTTP GET failed for '{}': status {}",
                url,
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| Error::storage(format!("Failed to read HTTP response body: {e}")))?;

        Ok(bytes)
    }

    fn put(&self, key: &str, _data: Bytes) -> Result<()> {
        Err(Error::storage(format!(
            "HTTP backend is read-only, cannot write to '{}'",
            key
        )))
    }

    fn delete(&self, key: &str) -> Result<()> {
        Err(Error::storage(format!(
            "HTTP backend is read-only, cannot delete '{}'",
            key
        )))
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let url = self.url_for(key);

        let response = self
            .client
            .head(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP HEAD error for '{}': {}", url, e)))?;

        Ok(response.status().is_success())
    }

    fn size(&self, key: &str) -> Result<u64> {
        let url = self.url_for(key);

        let response = self
            .client
            .head(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP HEAD error for '{}': {}", url, e)))?;

        if !response.status().is_success() {
            return Err(Error::storage(format!(
                "HTTP HEAD failed for '{}': status {}",
                url,
                response.status()
            )));
        }

        // Try to get Content-Length header
        if let Some(content_length) = response.headers().get(CONTENT_LENGTH) {
            if let Ok(len_str) = content_length.to_str() {
                if let Ok(len) = len_str.parse::<u64>() {
                    return Ok(len);
                }
            }
        }

        Err(Error::storage(format!(
            "Server did not provide Content-Length for '{}'",
            url
        )))
    }
}

/// HTTP backend with support for partial/range requests.
///
/// This variant supports reading specific byte ranges, which is useful
/// for large files when only a portion is needed.
#[derive(Debug)]
pub struct RangeHttpBackend {
    inner: HttpBackend,
}

impl RangeHttpBackend {
    /// Creates a new range-capable HTTP backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the base URL is invalid.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        Ok(Self {
            inner: HttpBackend::new(base_url)?,
        })
    }

    /// Reads a specific byte range from a key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to read from
    /// * `start` - Starting byte offset (inclusive)
    /// * `end` - Ending byte offset (inclusive)
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the server doesn't support
    /// ranges.
    pub fn get_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let url = self.inner.url_for(key);
        let range_header = format!("bytes={}-{}", start, end);

        let response = self
            .inner
            .client
            .get(&url)
            .header(RANGE, range_header)
            .send()
            .map_err(|e| Error::storage(format!("HTTP GET range error for '{}': {}", url, e)))?;

        // 206 Partial Content is expected for range requests
        if response.status().as_u16() != 206 && !response.status().is_success() {
            return Err(Error::storage(format!(
                "HTTP GET range failed for '{}': status {}",
                url,
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| Error::storage(format!("Failed to read HTTP response body: {e}")))?;

        Ok(bytes)
    }

    /// Checks if the server supports range requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP HEAD request fails.
    pub fn supports_range(&self, key: &str) -> Result<bool> {
        let url = self.inner.url_for(key);

        let response = self
            .inner
            .client
            .head(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP HEAD error for '{}': {}", url, e)))?;

        if !response.status().is_success() {
            return Ok(false);
        }

        // Check for Accept-Ranges header
        if let Some(accept_ranges) = response.headers().get("accept-ranges") {
            if let Ok(value) = accept_ranges.to_str() {
                return Ok(value != "none");
            }
        }

        Ok(false)
    }
}

impl StorageBackend for RangeHttpBackend {
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        self.inner.get(key)
    }

    fn put(&self, key: &str, data: Bytes) -> Result<()> {
        self.inner.put(key, data)
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key)
    }

    fn size(&self, key: &str) -> Result<u64> {
        self.inner.size(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_construction() {
        let backend = HttpBackend::new("https://example.com/data")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("file.txt"),
            "https://example.com/data/file.txt"
        );

        let backend_slash = HttpBackend::new("https://example.com/data/")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend_slash.url_for("file.txt"),
            "https://example.com/data/file.txt"
        );
    }

    #[test]
    fn test_base_url() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(backend.base_url(), "https://example.com");
    }

    #[test]
    fn test_put_is_read_only() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.put("test.txt", Bytes::from("data"));
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_is_read_only() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.delete("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_not_supported() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.list("");
        assert!(result.is_err());
    }

    #[test]
    fn test_with_timeout() {
        let backend = HttpBackend::with_timeout("https://example.com", 30);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_range_http_backend_new() {
        let backend = RangeHttpBackend::new("https://example.com");
        assert!(backend.is_ok());
    }

    #[test]
    fn test_range_http_backend_list_not_supported() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.list("");
        assert!(result.is_err());
    }

    #[test]
    fn test_range_http_backend_put_is_read_only() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.put("test.txt", Bytes::from("data"));
        assert!(result.is_err());
    }

    #[test]
    fn test_range_http_backend_delete_is_read_only() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.delete("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_url_construction_nested_path() {
        let backend = HttpBackend::new("https://example.com/api/v1/data")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("datasets/train.parquet"),
            "https://example.com/api/v1/data/datasets/train.parquet"
        );
    }

    #[test]
    fn test_http_backend_debug() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("HttpBackend"));
        assert!(debug_str.contains("example.com"));
    }

    #[test]
    fn test_range_http_backend_debug() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("RangeHttpBackend"));
    }

    // === Additional coverage tests for HTTP operations ===

    #[test]
    fn test_url_construction_empty_key() {
        let backend = HttpBackend::new("https://example.com/data")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(backend.url_for(""), "https://example.com/data/");
    }

    #[test]
    fn test_url_construction_with_query_params() {
        let backend = HttpBackend::new("https://example.com/data")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("file.txt?version=1"),
            "https://example.com/data/file.txt?version=1"
        );
    }

    #[test]
    fn test_put_error_message_contains_key() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.put("my_file.txt", Bytes::from("data"));
        let err = result.err().expect("Should be error");
        let msg = format!("{:?}", err);
        assert!(msg.contains("my_file.txt"));
    }

    #[test]
    fn test_delete_error_message_contains_key() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.delete("my_file.txt");
        let err = result.err().expect("Should be error");
        let msg = format!("{:?}", err);
        assert!(msg.contains("my_file.txt"));
    }

    #[test]
    fn test_list_error_message() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.list("prefix/");
        let err = result.err().expect("Should be error");
        let msg = format!("{:?}", err);
        assert!(msg.contains("directory") || msg.contains("listing"));
    }

    #[test]
    fn test_with_timeout_zero() {
        // Zero timeout should still create a valid backend
        let backend = HttpBackend::with_timeout("https://example.com", 0);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_with_timeout_large() {
        let backend = HttpBackend::with_timeout("https://example.com", 3600);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_base_url_with_trailing_slash() {
        let backend = HttpBackend::new("https://example.com/path/")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(backend.base_url(), "https://example.com/path/");
    }

    #[test]
    fn test_range_http_backend_get_delegates() {
        // Test that RangeHttpBackend.get delegates to inner
        let backend = RangeHttpBackend::new("https://httpbin.org")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        // We can't test actual HTTP calls without a server, but we can verify
        // the delegation path is exercised
        let result = backend.get("nonexistent-file.txt");
        // This will fail because no server, but exercises the code path
        assert!(result.is_err());
    }

    #[test]
    fn test_range_http_backend_exists_delegates() {
        let backend = RangeHttpBackend::new("https://httpbin.org")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.exists("nonexistent-file.txt");
        // Either error (network) or false (not found)
        match result {
            Ok(exists) => assert!(!exists),
            Err(_) => {} // Network error is acceptable
        }
    }

    #[test]
    fn test_range_http_backend_size_delegates() {
        let backend = RangeHttpBackend::new("https://httpbin.org")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.size("nonexistent-file.txt");
        // Will fail - exercises code path
        assert!(result.is_err());
    }

    #[test]
    fn test_url_construction_special_chars() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        // URL with spaces (should be encoded by caller)
        assert_eq!(
            backend.url_for("file%20name.txt"),
            "https://example.com/file%20name.txt"
        );
    }

    #[test]
    fn test_url_construction_unicode() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("données.txt"),
            "https://example.com/données.txt"
        );
    }

    #[test]
    fn test_multiple_backends_independent() {
        let backend1 = HttpBackend::new("https://example1.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let backend2 = HttpBackend::new("https://example2.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        assert_eq!(backend1.base_url(), "https://example1.com");
        assert_eq!(backend2.base_url(), "https://example2.com");
    }

    #[test]
    fn test_range_backend_delegation_put() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        // Should delegate to inner.put which returns read-only error
        let result = backend.put("test.txt", Bytes::from("data"));
        assert!(result.is_err());
    }

    #[test]
    fn test_range_backend_delegation_delete() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        // Should delegate to inner.delete which returns read-only error
        let result = backend.delete("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_range_backend_delegation_list() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        // Should delegate to inner.list which returns not-supported error
        let result = backend.list("");
        assert!(result.is_err());
    }

    // === Additional HTTP backend tests ===

    #[test]
    fn test_http_backend_url_with_port() {
        let backend = HttpBackend::new("https://example.com:8080/api")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("data.json"),
            "https://example.com:8080/api/data.json"
        );
    }

    #[test]
    fn test_http_backend_url_with_path_segments() {
        let backend = HttpBackend::new("https://cdn.example.com/v1/datasets")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("train/data.parquet"),
            "https://cdn.example.com/v1/datasets/train/data.parquet"
        );
    }

    #[test]
    fn test_http_backend_list_error_contains_context() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.list("some/prefix");
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("listing") || msg.contains("directory"));
        }
    }

    #[test]
    fn test_http_backend_put_error_includes_key() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.put("path/to/file.txt", Bytes::from("content"));
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("path/to/file.txt") || msg.contains("read-only"));
        }
    }

    #[test]
    fn test_http_backend_delete_error_includes_key() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.delete("path/to/file.txt");
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("path/to/file.txt") || msg.contains("read-only"));
        }
    }

    #[test]
    fn test_range_http_backend_creation_variations() {
        // Test with various URL formats
        assert!(RangeHttpBackend::new("https://example.com").is_ok());
        assert!(RangeHttpBackend::new("https://example.com/").is_ok());
        assert!(RangeHttpBackend::new("https://example.com/path").is_ok());
        assert!(RangeHttpBackend::new("http://localhost:3000").is_ok());
    }

    #[test]
    fn test_http_backend_with_timeout_variations() {
        // Very short timeout
        assert!(HttpBackend::with_timeout("https://example.com", 1).is_ok());
        // Medium timeout
        assert!(HttpBackend::with_timeout("https://example.com", 30).is_ok());
        // Long timeout
        assert!(HttpBackend::with_timeout("https://example.com", 600).is_ok());
    }

    #[test]
    fn test_http_backend_url_construction_edge_cases() {
        // Double slash prevention
        let backend = HttpBackend::new("https://example.com/")
            .ok()
            .unwrap_or_else(|| panic!("backend"));
        // Should not have double slashes
        let url = backend.url_for("file.txt");
        assert!(!url.contains("//file"));

        // Leading slash in key
        let url2 = backend.url_for("/file.txt");
        // The URL is simply concatenated, so this is expected behavior
        assert!(url2.contains("file.txt"));
    }

    #[test]
    fn test_http_backend_base_url_preserved() {
        let urls = vec![
            "https://example.com",
            "https://example.com/",
            "https://example.com/api/v1",
            "https://example.com/api/v1/",
            "http://localhost:8080",
        ];

        for url in urls {
            let backend = HttpBackend::new(url)
                .ok()
                .unwrap_or_else(|| panic!("Should create backend for {}", url));
            assert_eq!(backend.base_url(), url);
        }
    }

    #[test]
    fn test_range_http_backend_put_delegates_error() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("backend"));

        let result = backend.put("any/path.txt", Bytes::from("data"));
        assert!(result.is_err());

        // Error should indicate read-only
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("read-only") || msg.contains("any/path.txt"));
        }
    }

    #[test]
    fn test_range_http_backend_delete_delegates_error() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("backend"));

        let result = backend.delete("some/file.parquet");
        assert!(result.is_err());

        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("read-only") || msg.contains("some/file.parquet"));
        }
    }

    #[test]
    fn test_range_http_backend_list_delegates_error() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("backend"));

        let result = backend.list("prefix/");
        assert!(result.is_err());

        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("listing") || msg.contains("directory"));
        }
    }

    #[test]
    fn test_http_backend_url_for_with_fragment() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("backend"));

        // URL with fragment (though unusual for data files)
        let url = backend.url_for("file.txt#section");
        assert_eq!(url, "https://example.com/file.txt#section");
    }

    // ========================================================================
    // Mock HTTP Client Tests
    // ========================================================================

    #[test]
    fn test_mock_http_client_get_response() {
        let mock = MockHttpClient::new()
            .with_get_response("https://example.com/data.txt", HttpResponse::ok("hello"));

        let response = mock.get("https://example.com/data.txt").unwrap();
        assert!(response.is_success);
        assert_eq!(response.status, 200);
        assert_eq!(response.body, Bytes::from("hello"));
    }

    #[test]
    fn test_mock_http_client_head_response() {
        let mock = MockHttpClient::new().with_head_response(
            "https://example.com/file.txt",
            HttpResponse::ok(Bytes::new()).with_content_length(1024),
        );

        let response = mock.head("https://example.com/file.txt").unwrap();
        assert!(response.is_success);
        assert_eq!(response.content_length, Some(1024));
    }

    #[test]
    fn test_mock_http_client_default_response() {
        let mock = MockHttpClient::new().with_default_response(HttpResponse::not_found());

        let response = mock.get("https://any-url.com/anything").unwrap();
        assert!(!response.is_success);
        assert_eq!(response.status, 404);
    }

    #[test]
    fn test_mock_http_client_no_response_error() {
        let mock = MockHttpClient::new();

        let result = mock.get("https://example.com/missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_http_client_get_range() {
        let mock = MockHttpClient::new().with_get_response(
            "https://example.com/large.bin",
            HttpResponse::partial_content("partial content"),
        );

        let response = mock
            .get_range("https://example.com/large.bin", 0, 100)
            .unwrap();
        assert!(response.is_success);
        assert_eq!(response.status, 206);
    }

    #[test]
    fn test_http_response_builder_methods() {
        let response = HttpResponse::ok("test")
            .with_content_length(100)
            .with_accept_ranges("bytes");

        assert_eq!(response.content_length, Some(100));
        assert_eq!(response.accept_ranges, Some("bytes".to_string()));
    }

    #[test]
    fn test_mock_http_client_clone() {
        let mock =
            MockHttpClient::new().with_get_response("https://example.com/a", HttpResponse::ok("a"));

        let cloned = mock.clone();
        let response = cloned.get("https://example.com/a").unwrap();
        assert!(response.is_success);
    }

    #[test]
    fn test_http_response_debug() {
        let response = HttpResponse::ok("test");
        let debug = format!("{:?}", response);
        assert!(debug.contains("HttpResponse"));
    }

    // ========================================================================
    // ReqwestHttpClient Tests
    // ========================================================================

    #[test]
    fn test_reqwest_http_client_new() {
        let client = ReqwestHttpClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_reqwest_http_client_with_timeout() {
        let client = ReqwestHttpClient::with_timeout(30);
        assert!(client.is_ok());
        // Verify the client was created successfully (timeout is applied internally)
        let _client = client.unwrap();
    }

    #[test]
    fn test_reqwest_http_client_debug() {
        let client = ReqwestHttpClient::new().unwrap();
        let debug = format!("{:?}", client);
        assert!(debug.contains("ReqwestHttpClient"));
    }

    // ========================================================================
    // HttpResponse Tests
    // ========================================================================

    #[test]
    fn test_http_response_not_found() {
        let response = HttpResponse::not_found();
        assert_eq!(response.status, 404);
        assert!(!response.is_success);
        assert!(response.body.is_empty());
    }

    #[test]
    fn test_http_response_partial_content() {
        let response = HttpResponse::partial_content("partial");
        assert_eq!(response.status, 206);
        assert!(response.is_success);
        assert_eq!(response.body, Bytes::from("partial"));
        assert_eq!(response.accept_ranges, Some("bytes".to_string()));
    }

    #[test]
    fn test_http_response_clone() {
        let response = HttpResponse::ok("test").with_content_length(100);
        let cloned = response.clone();
        assert_eq!(cloned.status, response.status);
        assert_eq!(cloned.body, response.body);
        assert_eq!(cloned.content_length, response.content_length);
    }

    // ========================================================================
    // Mock Client Integration Tests
    // ========================================================================

    #[test]
    fn test_mock_client_multiple_urls() {
        let mock = MockHttpClient::new()
            .with_get_response("https://a.com/1", HttpResponse::ok("first"))
            .with_get_response("https://b.com/2", HttpResponse::ok("second"));

        let r1 = mock.get("https://a.com/1").unwrap();
        let r2 = mock.get("https://b.com/2").unwrap();

        assert_eq!(r1.body, Bytes::from("first"));
        assert_eq!(r2.body, Bytes::from("second"));
    }

    #[test]
    fn test_mock_client_head_uses_own_map() {
        let mock = MockHttpClient::new()
            .with_get_response("https://example.com/file", HttpResponse::ok("content"))
            .with_head_response(
                "https://example.com/file",
                HttpResponse::ok(Bytes::new()).with_content_length(7),
            );

        let get_resp = mock.get("https://example.com/file").unwrap();
        let head_resp = mock.head("https://example.com/file").unwrap();

        assert_eq!(get_resp.body, Bytes::from("content"));
        assert_eq!(head_resp.content_length, Some(7));
    }

    #[test]
    fn test_mock_client_default_fallback_for_head() {
        let mock = MockHttpClient::new()
            .with_default_response(HttpResponse::ok(Bytes::new()).with_content_length(999));

        let response = mock.head("https://any.com/file").unwrap();
        assert_eq!(response.content_length, Some(999));
    }
}
