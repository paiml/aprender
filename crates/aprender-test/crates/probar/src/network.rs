//! Network Request Interception (Feature 7)
//!
//! Mock API responses and intercept network requests for testing.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Poka-Yoke**: Type-safe route matching prevents invalid patterns
//! - **Jidoka**: Immediate feedback on unexpected requests
//! - **Muda**: Only intercept relevant requests

use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// =============================================================================
// PMAT-006: Network Features (Playwright Parity)
// =============================================================================

/// Reasons for aborting a network request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AbortReason {
    /// Request failed
    Failed,
    /// Request was aborted
    Aborted,
    /// Request timed out
    TimedOut,
    /// Access was denied
    AccessDenied,
    /// Connection was closed
    ConnectionClosed,
    /// Connection failed to establish
    ConnectionFailed,
    /// Connection was refused
    ConnectionRefused,
    /// Connection was reset
    ConnectionReset,
    /// Internet is disconnected
    InternetDisconnected,
    /// DNS name could not be resolved
    NameNotResolved,
    /// Request was blocked by client
    BlockedByClient,
}

impl AbortReason {
    /// Get the error message for this abort reason
    #[must_use]
    pub const fn message(&self) -> &'static str {
        match self {
            Self::Failed => "net::ERR_FAILED",
            Self::Aborted => "net::ERR_ABORTED",
            Self::TimedOut => "net::ERR_TIMED_OUT",
            Self::AccessDenied => "net::ERR_ACCESS_DENIED",
            Self::ConnectionClosed => "net::ERR_CONNECTION_CLOSED",
            Self::ConnectionFailed => "net::ERR_CONNECTION_FAILED",
            Self::ConnectionRefused => "net::ERR_CONNECTION_REFUSED",
            Self::ConnectionReset => "net::ERR_CONNECTION_RESET",
            Self::InternetDisconnected => "net::ERR_INTERNET_DISCONNECTED",
            Self::NameNotResolved => "net::ERR_NAME_NOT_RESOLVED",
            Self::BlockedByClient => "net::ERR_BLOCKED_BY_CLIENT",
        }
    }
}

/// Action to take when a route matches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouteAction {
    /// Respond with a mock response
    Respond(MockResponse),
    /// Abort the request
    Abort(AbortReason),
    /// Continue the request (let it pass through)
    Continue,
}

impl Default for RouteAction {
    fn default() -> Self {
        Self::Continue
    }
}

/// HTTP methods for request matching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    /// GET request
    Get,
    /// POST request
    Post,
    /// PUT request
    Put,
    /// DELETE request
    Delete,
    /// PATCH request
    Patch,
    /// HEAD request
    Head,
    /// OPTIONS request
    Options,
    /// Any method
    Any,
}

impl HttpMethod {
    /// Parse from string
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "GET" => Self::Get,
            "POST" => Self::Post,
            "PUT" => Self::Put,
            "DELETE" => Self::Delete,
            "PATCH" => Self::Patch,
            "HEAD" => Self::Head,
            "OPTIONS" => Self::Options,
            _ => Self::Any,
        }
    }

    /// Convert to string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
            Self::Any => "*",
        }
    }

    /// Check if this method matches another
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        *self == Self::Any || *other == Self::Any || *self == *other
    }
}

/// A mocked HTTP response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockResponse {
    /// HTTP status code
    pub status: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body
    pub body: Vec<u8>,
    /// Content type
    pub content_type: String,
    /// Artificial delay in milliseconds
    pub delay_ms: u64,
}

impl Default for MockResponse {
    fn default() -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: Vec::new(),
            content_type: "application/json".to_string(),
            delay_ms: 0,
        }
    }
}

impl MockResponse {
    /// Create a new mock response
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a JSON response
    #[must_use]
    pub fn json<T: Serialize>(data: &T) -> ProbarResult<Self> {
        let body = serde_json::to_vec(data)?;
        Ok(Self {
            status: 200,
            headers: HashMap::new(),
            body,
            content_type: "application/json".to_string(),
            delay_ms: 0,
        })
    }

    /// Create a text response
    #[must_use]
    pub fn text(content: &str) -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: content.as_bytes().to_vec(),
            content_type: "text/plain".to_string(),
            delay_ms: 0,
        }
    }

    /// Create an error response
    #[must_use]
    pub fn error(status: u16, message: &str) -> Self {
        let body = serde_json::json!({ "error": message }).to_string();
        Self {
            status,
            headers: HashMap::new(),
            body: body.into_bytes(),
            content_type: "application/json".to_string(),
            delay_ms: 0,
        }
    }

    /// Set status code
    #[must_use]
    pub const fn with_status(mut self, status: u16) -> Self {
        self.status = status;
        self
    }

    /// Set body
    #[must_use]
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = body;
        self
    }

    /// Set JSON body
    pub fn with_json<T: Serialize>(mut self, data: &T) -> ProbarResult<Self> {
        self.body = serde_json::to_vec(data)?;
        self.content_type = "application/json".to_string();
        Ok(self)
    }

    /// Add a header
    #[must_use]
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Set content type
    #[must_use]
    pub fn with_content_type(mut self, content_type: &str) -> Self {
        self.content_type = content_type.to_string();
        self
    }

    /// Set delay
    #[must_use]
    pub const fn with_delay(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Get body as string
    #[must_use]
    pub fn body_string(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }
}

/// Pattern for matching request URLs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrlPattern {
    /// Exact URL match
    Exact(String),
    /// Prefix match
    Prefix(String),
    /// Contains substring
    Contains(String),
    /// Regex match
    Regex(String),
    /// Glob pattern (e.g., "**/api/users/*")
    Glob(String),
    /// Match any URL
    Any,
}

impl UrlPattern {
    /// Check if a URL matches this pattern
    #[must_use]
    pub fn matches(&self, url: &str) -> bool {
        match self {
            Self::Exact(pattern) => url == pattern,
            Self::Prefix(pattern) => url.starts_with(pattern),
            Self::Contains(pattern) => url.contains(pattern),
            Self::Regex(pattern) => regex::Regex::new(pattern)
                .map(|re| re.is_match(url))
                .unwrap_or(false),
            Self::Glob(pattern) => Self::glob_matches(pattern, url),
            Self::Any => true,
        }
    }

    /// Simple glob matching for URLs
    fn glob_matches(pattern: &str, url: &str) -> bool {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.is_empty() {
            return url.is_empty();
        }

        let mut pos = 0;
        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }
            if let Some(found) = url[pos..].find(part) {
                if i == 0 && found != 0 {
                    return false;
                }
                pos += found + part.len();
            } else {
                return false;
            }
        }

        // If pattern ends with *, any remaining URL is fine
        // Otherwise, must have consumed all of URL
        pattern.ends_with('*') || pos == url.len()
    }
}

/// A captured network request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedRequest {
    /// Request URL
    pub url: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request body
    pub body: Option<Vec<u8>>,
    /// Timestamp (milliseconds since interception start)
    pub timestamp_ms: u64,
}

impl CapturedRequest {
    /// Create a new captured request
    #[must_use]
    pub fn new(url: &str, method: HttpMethod, timestamp_ms: u64) -> Self {
        Self {
            url: url.to_string(),
            method,
            headers: HashMap::new(),
            body: None,
            timestamp_ms,
        }
    }

    /// Get body as string
    #[must_use]
    pub fn body_string(&self) -> Option<String> {
        self.body
            .as_ref()
            .map(|b| String::from_utf8_lossy(b).to_string())
    }

    /// Parse body as JSON
    pub fn body_json<T: for<'de> Deserialize<'de>>(&self) -> ProbarResult<T> {
        let body = self
            .body
            .as_ref()
            .ok_or_else(|| ProbarError::AssertionFailed {
                message: "No request body".to_string(),
            })?;
        let data = serde_json::from_slice(body)?;
        Ok(data)
    }
}

/// A route definition for interception
#[derive(Debug, Clone)]
pub struct Route {
    /// URL pattern to match
    pub pattern: UrlPattern,
    /// HTTP method to match
    pub method: HttpMethod,
    /// Response to return
    pub response: MockResponse,
    /// Number of times this route should be used (None = unlimited)
    pub times: Option<usize>,
    /// Number of times this route has been matched
    pub match_count: usize,
}

impl Route {
    /// Create a new route
    #[must_use]
    pub fn new(pattern: UrlPattern, method: HttpMethod, response: MockResponse) -> Self {
        Self {
            pattern,
            method,
            response,
            times: None,
            match_count: 0,
        }
    }

    /// Set how many times this route should match
    #[must_use]
    pub const fn times(mut self, n: usize) -> Self {
        self.times = Some(n);
        self
    }

    /// Check if this route matches a request
    #[must_use]
    pub fn matches(&self, url: &str, method: &HttpMethod) -> bool {
        // Check if we've exceeded our match limit
        if let Some(max) = self.times {
            if self.match_count >= max {
                return false;
            }
        }
        self.pattern.matches(url) && self.method.matches(method)
    }

    /// Record a match
    pub fn record_match(&mut self) {
        self.match_count += 1;
    }

    /// Check if route is exhausted
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.times.is_some_and(|max| self.match_count >= max)
    }
}

/// Network interception handler
#[derive(Debug)]
pub struct NetworkInterception {
    /// Registered routes
    routes: Vec<Route>,
    /// Captured requests
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
    /// Whether to capture all requests (not just intercepted)
    capture_all: bool,
    /// Whether interception is active
    active: bool,
    /// Start timestamp
    start_time: std::time::Instant,
    /// Block unmatched requests
    block_unmatched: bool,
}

impl Default for NetworkInterception {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkInterception {
    /// Create a new network interception handler
    #[must_use]
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            captured: Arc::new(Mutex::new(Vec::new())),
            capture_all: false,
            active: false,
            start_time: std::time::Instant::now(),
            block_unmatched: false,
        }
    }

    /// Enable capturing all requests
    #[must_use]
    pub const fn capture_all(mut self) -> Self {
        self.capture_all = true;
        self
    }

    /// Block unmatched requests
    #[must_use]
    pub const fn block_unmatched(mut self) -> Self {
        self.block_unmatched = true;
        self
    }

    /// Start interception
    pub fn start(&mut self) {
        self.active = true;
        self.start_time = std::time::Instant::now();
    }

    /// Stop interception
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Check if interception is active
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Add a route
    pub fn route(&mut self, route: Route) {
        self.routes.push(route);
    }

    /// Add a GET route
    pub fn get(&mut self, pattern: &str, response: MockResponse) {
        self.routes.push(Route::new(
            UrlPattern::Contains(pattern.to_string()),
            HttpMethod::Get,
            response,
        ));
    }

    /// Add a POST route
    pub fn post(&mut self, pattern: &str, response: MockResponse) {
        self.routes.push(Route::new(
            UrlPattern::Contains(pattern.to_string()),
            HttpMethod::Post,
            response,
        ));
    }

    /// Add a PUT route
    pub fn put(&mut self, pattern: &str, response: MockResponse) {
        self.routes.push(Route::new(
            UrlPattern::Contains(pattern.to_string()),
            HttpMethod::Put,
            response,
        ));
    }

    /// Add a DELETE route
    pub fn delete(&mut self, pattern: &str, response: MockResponse) {
        self.routes.push(Route::new(
            UrlPattern::Contains(pattern.to_string()),
            HttpMethod::Delete,
            response,
        ));
    }

    /// Handle an incoming request
    pub fn handle_request(
        &mut self,
        url: &str,
        method: HttpMethod,
        headers: HashMap<String, String>,
        body: Option<Vec<u8>>,
    ) -> Option<MockResponse> {
        if !self.active {
            return None;
        }

        let timestamp_ms = self.start_time.elapsed().as_millis() as u64;

        // Capture the request
        if self.capture_all {
            let mut request = CapturedRequest::new(url, method, timestamp_ms);
            request.headers = headers.clone();
            request.body = body.clone();
            if let Ok(mut captured) = self.captured.lock() {
                captured.push(request);
            }
        }

        // Find matching route
        for route in &mut self.routes {
            if route.matches(url, &method) {
                route.record_match();

                // Capture matched request
                if !self.capture_all {
                    let mut request = CapturedRequest::new(url, method, timestamp_ms);
                    request.headers = headers;
                    request.body = body;
                    if let Ok(mut captured) = self.captured.lock() {
                        captured.push(request);
                    }
                }

                return Some(route.response.clone());
            }
        }

        // Return 404 if blocking unmatched requests
        if self.block_unmatched {
            Some(MockResponse::error(404, "No route matched"))
        } else {
            None
        }
    }

    /// Get all captured requests
    #[must_use]
    pub fn captured_requests(&self) -> Vec<CapturedRequest> {
        self.captured.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Get captured requests matching a URL pattern
    #[must_use]
    pub fn requests_matching(&self, pattern: &UrlPattern) -> Vec<CapturedRequest> {
        self.captured_requests()
            .into_iter()
            .filter(|r| pattern.matches(&r.url))
            .collect()
    }

    /// Get captured requests by method
    #[must_use]
    pub fn requests_by_method(&self, method: HttpMethod) -> Vec<CapturedRequest> {
        self.captured_requests()
            .into_iter()
            .filter(|r| r.method == method)
            .collect()
    }

    /// Assert a request was made
    pub fn assert_requested(&self, pattern: &UrlPattern) -> ProbarResult<()> {
        let requests = self.requests_matching(pattern);
        if requests.is_empty() {
            return Err(ProbarError::AssertionFailed {
                message: format!("Expected request matching {:?}, but none found", pattern),
            });
        }
        Ok(())
    }

    /// Assert a request was made N times
    pub fn assert_requested_times(&self, pattern: &UrlPattern, times: usize) -> ProbarResult<()> {
        let requests = self.requests_matching(pattern);
        if requests.len() != times {
            return Err(ProbarError::AssertionFailed {
                message: format!(
                    "Expected {} requests matching {:?}, but found {}",
                    times,
                    pattern,
                    requests.len()
                ),
            });
        }
        Ok(())
    }

    /// Assert no requests were made matching a pattern
    pub fn assert_not_requested(&self, pattern: &UrlPattern) -> ProbarResult<()> {
        let requests = self.requests_matching(pattern);
        if !requests.is_empty() {
            return Err(ProbarError::AssertionFailed {
                message: format!(
                    "Expected no requests matching {:?}, but found {}",
                    pattern,
                    requests.len()
                ),
            });
        }
        Ok(())
    }

    /// Clear captured requests
    pub fn clear_captured(&self) {
        if let Ok(mut captured) = self.captured.lock() {
            captured.clear();
        }
    }

    /// Get route count
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Clear all routes
    pub fn clear_routes(&mut self) {
        self.routes.clear();
    }

    // =========================================================================
    // PMAT-006: Network Abort & Wait Features (Playwright Parity)
    // =========================================================================

    /// Abort requests matching a pattern with a specific reason
    ///
    /// Per Playwright: `page.route('**/api/*', route => route.abort())`
    pub fn abort(&mut self, pattern: &str, reason: AbortReason) {
        let abort_response = MockResponse::new()
            .with_status(0)
            .with_body(reason.message().as_bytes().to_vec());
        self.routes.push(Route::new(
            UrlPattern::Contains(pattern.to_string()),
            HttpMethod::Any,
            abort_response,
        ));
    }

    /// Abort requests matching a pattern (default reason: Aborted)
    pub fn abort_pattern(&mut self, pattern: UrlPattern, reason: AbortReason) {
        let abort_response = MockResponse::new()
            .with_status(0)
            .with_body(reason.message().as_bytes().to_vec());
        self.routes
            .push(Route::new(pattern, HttpMethod::Any, abort_response));
    }

    /// Wait for a request matching a pattern (synchronous check)
    ///
    /// Per Playwright: `page.waitForRequest(url)`
    #[must_use]
    pub fn find_request(&self, pattern: &UrlPattern) -> Option<CapturedRequest> {
        self.requests_matching(pattern).into_iter().next()
    }

    /// Wait for a response matching a pattern (returns the response from route)
    ///
    /// Per Playwright: `page.waitForResponse(url)`
    #[must_use]
    pub fn find_response_for(&self, pattern: &UrlPattern) -> Option<MockResponse> {
        for route in &self.routes {
            if route.pattern.matches(&pattern.to_string()) || route.match_count > 0 {
                return Some(route.response.clone());
            }
        }
        None
    }

    /// Check if a request was aborted
    #[must_use]
    pub fn was_aborted(&self, pattern: &UrlPattern) -> bool {
        for route in &self.routes {
            if route.match_count > 0 && route.response.status == 0 {
                if let UrlPattern::Contains(p) = &route.pattern {
                    if pattern.matches(p) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Get captured responses (mock responses that were returned)
    #[must_use]
    pub fn captured_responses(&self) -> Vec<MockResponse> {
        self.routes
            .iter()
            .filter(|r| r.match_count > 0)
            .map(|r| r.response.clone())
            .collect()
    }
}

impl std::fmt::Display for UrlPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact(s)
            | Self::Prefix(s)
            | Self::Contains(s)
            | Self::Regex(s)
            | Self::Glob(s) => write!(f, "{}", s),
            Self::Any => write!(f, "*"),
        }
    }
}

/// Builder for creating network interception
#[derive(Debug, Default)]
pub struct NetworkInterceptionBuilder {
    interception: NetworkInterception,
}

impl NetworkInterceptionBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable capturing all requests
    #[must_use]
    pub fn capture_all(mut self) -> Self {
        self.interception.capture_all = true;
        self
    }

    /// Block unmatched requests
    #[must_use]
    pub fn block_unmatched(mut self) -> Self {
        self.interception.block_unmatched = true;
        self
    }

    /// Add a GET route
    #[must_use]
    pub fn get(mut self, pattern: &str, response: MockResponse) -> Self {
        self.interception.get(pattern, response);
        self
    }

    /// Add a POST route
    #[must_use]
    pub fn post(mut self, pattern: &str, response: MockResponse) -> Self {
        self.interception.post(pattern, response);
        self
    }

    /// Add a custom route
    #[must_use]
    pub fn route(mut self, route: Route) -> Self {
        self.interception.route(route);
        self
    }

    /// Build the interception handler
    #[must_use]
    pub fn build(self) -> NetworkInterception {
        self.interception
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::default_trait_access
)]
mod tests {
    use super::*;

    mod http_method_tests {
        use super::*;

        #[test]
        fn test_from_str() {
            assert_eq!(HttpMethod::from_str("GET"), HttpMethod::Get);
            assert_eq!(HttpMethod::from_str("post"), HttpMethod::Post);
            assert_eq!(HttpMethod::from_str("PUT"), HttpMethod::Put);
            assert_eq!(HttpMethod::from_str("DELETE"), HttpMethod::Delete);
            assert_eq!(HttpMethod::from_str("unknown"), HttpMethod::Any);
        }

        #[test]
        fn test_as_str() {
            assert_eq!(HttpMethod::Get.as_str(), "GET");
            assert_eq!(HttpMethod::Post.as_str(), "POST");
            assert_eq!(HttpMethod::Any.as_str(), "*");
        }

        #[test]
        fn test_matches() {
            assert!(HttpMethod::Get.matches(&HttpMethod::Get));
            assert!(HttpMethod::Any.matches(&HttpMethod::Get));
            assert!(HttpMethod::Get.matches(&HttpMethod::Any));
            assert!(!HttpMethod::Get.matches(&HttpMethod::Post));
        }
    }

    mod mock_response_tests {
        use super::*;

        #[test]
        fn test_default() {
            let response = MockResponse::default();
            assert_eq!(response.status, 200);
            assert_eq!(response.content_type, "application/json");
        }

        #[test]
        fn test_json() {
            let data = serde_json::json!({"name": "test"});
            let response = MockResponse::json(&data).unwrap();
            assert_eq!(response.status, 200);
            assert!(response.body_string().contains("test"));
        }

        #[test]
        fn test_text() {
            let response = MockResponse::text("Hello World");
            assert_eq!(response.body_string(), "Hello World");
            assert_eq!(response.content_type, "text/plain");
        }

        #[test]
        fn test_error() {
            let response = MockResponse::error(404, "Not Found");
            assert_eq!(response.status, 404);
            assert!(response.body_string().contains("Not Found"));
        }

        #[test]
        fn test_with_status() {
            let response = MockResponse::new().with_status(201);
            assert_eq!(response.status, 201);
        }

        #[test]
        fn test_with_header() {
            let response = MockResponse::new().with_header("X-Custom", "value");
            assert_eq!(response.headers.get("X-Custom"), Some(&"value".to_string()));
        }

        #[test]
        fn test_with_delay() {
            let response = MockResponse::new().with_delay(100);
            assert_eq!(response.delay_ms, 100);
        }
    }

    mod url_pattern_tests {
        use super::*;

        #[test]
        fn test_exact() {
            let pattern = UrlPattern::Exact("https://api.example.com/users".to_string());
            assert!(pattern.matches("https://api.example.com/users"));
            assert!(!pattern.matches("https://api.example.com/users/1"));
        }

        #[test]
        fn test_prefix() {
            let pattern = UrlPattern::Prefix("https://api.example.com".to_string());
            assert!(pattern.matches("https://api.example.com/users"));
            assert!(pattern.matches("https://api.example.com/posts"));
            assert!(!pattern.matches("https://other.com"));
        }

        #[test]
        fn test_contains() {
            let pattern = UrlPattern::Contains("/api/".to_string());
            assert!(pattern.matches("https://example.com/api/users"));
            assert!(!pattern.matches("https://example.com/users"));
        }

        #[test]
        fn test_regex() {
            let pattern = UrlPattern::Regex(r"/users/\d+".to_string());
            assert!(pattern.matches("https://api.example.com/users/123"));
            assert!(!pattern.matches("https://api.example.com/users/abc"));
        }

        #[test]
        fn test_glob() {
            let pattern = UrlPattern::Glob("*/api/users/*".to_string());
            assert!(pattern.matches("https://example.com/api/users/123"));
            assert!(!pattern.matches("https://example.com/api/posts/123"));
        }

        #[test]
        fn test_any() {
            let pattern = UrlPattern::Any;
            assert!(pattern.matches("anything"));
            assert!(pattern.matches(""));
        }
    }

    mod captured_request_tests {
        use super::*;

        #[test]
        fn test_new() {
            let request = CapturedRequest::new("https://api.example.com", HttpMethod::Get, 1000);
            assert_eq!(request.url, "https://api.example.com");
            assert_eq!(request.method, HttpMethod::Get);
            assert_eq!(request.timestamp_ms, 1000);
        }

        #[test]
        fn test_body_string() {
            let mut request = CapturedRequest::new("url", HttpMethod::Post, 0);
            request.body = Some(b"test body".to_vec());
            assert_eq!(request.body_string(), Some("test body".to_string()));
        }

        #[test]
        fn test_body_json() {
            let mut request = CapturedRequest::new("url", HttpMethod::Post, 0);
            request.body = Some(b"{\"name\":\"test\"}".to_vec());
            let data: serde_json::Value = request.body_json().unwrap();
            assert_eq!(data["name"], "test");
        }
    }

    mod route_tests {
        use super::*;

        #[test]
        fn test_new() {
            let route = Route::new(
                UrlPattern::Contains("/api".to_string()),
                HttpMethod::Get,
                MockResponse::new(),
            );
            assert_eq!(route.match_count, 0);
            assert!(route.times.is_none());
        }

        #[test]
        fn test_times() {
            let route = Route::new(UrlPattern::Any, HttpMethod::Get, MockResponse::new()).times(3);
            assert_eq!(route.times, Some(3));
        }

        #[test]
        fn test_matches() {
            let route = Route::new(
                UrlPattern::Contains("/users".to_string()),
                HttpMethod::Get,
                MockResponse::new(),
            );
            assert!(route.matches("https://api.example.com/users", &HttpMethod::Get));
            assert!(!route.matches("https://api.example.com/users", &HttpMethod::Post));
            assert!(!route.matches("https://api.example.com/posts", &HttpMethod::Get));
        }

        #[test]
        fn test_record_match() {
            let mut route = Route::new(UrlPattern::Any, HttpMethod::Any, MockResponse::new());
            route.record_match();
            assert_eq!(route.match_count, 1);
        }

        #[test]
        fn test_is_exhausted() {
            let mut route =
                Route::new(UrlPattern::Any, HttpMethod::Any, MockResponse::new()).times(2);

            assert!(!route.is_exhausted());
            route.record_match();
            assert!(!route.is_exhausted());
            route.record_match();
            assert!(route.is_exhausted());
        }

        #[test]
        fn test_exhausted_route_no_longer_matches() {
            let mut route =
                Route::new(UrlPattern::Any, HttpMethod::Any, MockResponse::new()).times(1);

            assert!(route.matches("url", &HttpMethod::Get));
            route.record_match();
            assert!(!route.matches("url", &HttpMethod::Get));
        }
    }

    mod network_interception_tests {
        use super::*;

        #[test]
        fn test_new() {
            let interception = NetworkInterception::new();
            assert!(!interception.is_active());
            assert_eq!(interception.route_count(), 0);
        }

        #[test]
        fn test_start_stop() {
            let mut interception = NetworkInterception::new();
            interception.start();
            assert!(interception.is_active());
            interception.stop();
            assert!(!interception.is_active());
        }

        #[test]
        fn test_add_routes() {
            let mut interception = NetworkInterception::new();
            interception.get("/api/users", MockResponse::text("users"));
            interception.post("/api/users", MockResponse::new().with_status(201));
            interception.put("/api/users/1", MockResponse::new());
            interception.delete("/api/users/1", MockResponse::new().with_status(204));

            assert_eq!(interception.route_count(), 4);
        }

        #[test]
        fn test_handle_request() {
            let mut interception = NetworkInterception::new();
            interception.get("/api/users", MockResponse::text("users list"));
            interception.start();

            let response = interception.handle_request(
                "https://api.example.com/api/users",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            assert!(response.is_some());
            let response = response.unwrap();
            assert_eq!(response.body_string(), "users list");
        }

        #[test]
        fn test_handle_request_no_match() {
            let mut interception = NetworkInterception::new();
            interception.get("/api/users", MockResponse::text("users"));
            interception.start();

            let response = interception.handle_request(
                "https://api.example.com/api/posts",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            assert!(response.is_none());
        }

        #[test]
        fn test_block_unmatched() {
            let mut interception = NetworkInterception::new().block_unmatched();
            interception.start();

            let response = interception.handle_request(
                "https://api.example.com/unknown",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            assert!(response.is_some());
            assert_eq!(response.unwrap().status, 404);
        }

        #[test]
        fn test_capture_requests() {
            let mut interception = NetworkInterception::new().capture_all();
            interception.get("/api/users", MockResponse::new());
            interception.start();

            interception.handle_request(
                "https://api.example.com/api/users",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            let captured = interception.captured_requests();
            assert_eq!(captured.len(), 1);
            assert_eq!(captured[0].url, "https://api.example.com/api/users");
        }

        #[test]
        fn test_requests_matching() {
            let mut interception = NetworkInterception::new().capture_all();
            interception.start();

            interception.handle_request(
                "https://api.example.com/api/users",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );
            interception.handle_request(
                "https://api.example.com/api/posts",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            let users = interception.requests_matching(&UrlPattern::Contains("/users".to_string()));
            assert_eq!(users.len(), 1);
        }

        #[test]
        fn test_requests_by_method() {
            let mut interception = NetworkInterception::new().capture_all();
            interception.start();

            interception.handle_request("url1", HttpMethod::Get, HashMap::new(), None);
            interception.handle_request("url2", HttpMethod::Post, HashMap::new(), None);

            let gets = interception.requests_by_method(HttpMethod::Get);
            assert_eq!(gets.len(), 1);
        }

        #[test]
        fn test_assert_requested() {
            let mut interception = NetworkInterception::new().capture_all();
            interception.start();

            interception.handle_request(
                "https://api.example.com/api/users",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            assert!(interception
                .assert_requested(&UrlPattern::Contains("/users".to_string()))
                .is_ok());

            assert!(interception
                .assert_requested(&UrlPattern::Contains("/posts".to_string()))
                .is_err());
        }

        #[test]
        fn test_assert_requested_times() {
            let mut interception = NetworkInterception::new().capture_all();
            interception.start();

            interception.handle_request("url", HttpMethod::Get, HashMap::new(), None);
            interception.handle_request("url", HttpMethod::Get, HashMap::new(), None);

            assert!(interception
                .assert_requested_times(&UrlPattern::Any, 2)
                .is_ok());

            assert!(interception
                .assert_requested_times(&UrlPattern::Any, 3)
                .is_err());
        }

        #[test]
        fn test_assert_not_requested() {
            let interception = NetworkInterception::new().capture_all();

            assert!(interception.assert_not_requested(&UrlPattern::Any).is_ok());
        }

        #[test]
        fn test_clear_captured() {
            let mut interception = NetworkInterception::new().capture_all();
            interception.start();

            interception.handle_request("url", HttpMethod::Get, HashMap::new(), None);
            assert_eq!(interception.captured_requests().len(), 1);

            interception.clear_captured();
            assert_eq!(interception.captured_requests().len(), 0);
        }

        #[test]
        fn test_clear_routes() {
            let mut interception = NetworkInterception::new();
            interception.get("/api", MockResponse::new());
            assert_eq!(interception.route_count(), 1);

            interception.clear_routes();
            assert_eq!(interception.route_count(), 0);
        }
    }

    mod network_interception_builder_tests {
        use super::*;

        #[test]
        fn test_builder() {
            let interception = NetworkInterceptionBuilder::new()
                .capture_all()
                .block_unmatched()
                .get("/api/users", MockResponse::text("users"))
                .post("/api/users", MockResponse::new().with_status(201))
                .build();

            assert!(interception.capture_all);
            assert!(interception.block_unmatched);
            assert_eq!(interception.route_count(), 2);
        }

        #[test]
        fn test_builder_with_route() {
            let route = Route::new(
                UrlPattern::Regex(r"/users/\d+".to_string()),
                HttpMethod::Get,
                MockResponse::new(),
            );

            let interception = NetworkInterceptionBuilder::new().route(route).build();

            assert_eq!(interception.route_count(), 1);
        }
    }

    // =========================================================================
    // PMAT-006: Network Abort & Wait Tests
    // =========================================================================

    mod abort_tests {
        use super::*;

        #[test]
        fn test_abort_reason_messages() {
            assert_eq!(AbortReason::Failed.message(), "net::ERR_FAILED");
            assert_eq!(AbortReason::Aborted.message(), "net::ERR_ABORTED");
            assert_eq!(AbortReason::TimedOut.message(), "net::ERR_TIMED_OUT");
            assert_eq!(
                AbortReason::AccessDenied.message(),
                "net::ERR_ACCESS_DENIED"
            );
            assert_eq!(
                AbortReason::ConnectionClosed.message(),
                "net::ERR_CONNECTION_CLOSED"
            );
            assert_eq!(
                AbortReason::ConnectionFailed.message(),
                "net::ERR_CONNECTION_FAILED"
            );
            assert_eq!(
                AbortReason::ConnectionRefused.message(),
                "net::ERR_CONNECTION_REFUSED"
            );
            assert_eq!(
                AbortReason::ConnectionReset.message(),
                "net::ERR_CONNECTION_RESET"
            );
            assert_eq!(
                AbortReason::InternetDisconnected.message(),
                "net::ERR_INTERNET_DISCONNECTED"
            );
            assert_eq!(
                AbortReason::NameNotResolved.message(),
                "net::ERR_NAME_NOT_RESOLVED"
            );
            assert_eq!(
                AbortReason::BlockedByClient.message(),
                "net::ERR_BLOCKED_BY_CLIENT"
            );
        }

        #[test]
        fn test_abort_request() {
            let mut interception = NetworkInterception::new();
            interception.abort("/api/blocked", AbortReason::BlockedByClient);
            interception.start();

            let response = interception.handle_request(
                "https://example.com/api/blocked/resource",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            assert!(response.is_some());
            let resp = response.unwrap();
            assert_eq!(resp.status, 0); // Aborted requests have status 0
            assert!(String::from_utf8_lossy(&resp.body).contains("ERR_BLOCKED_BY_CLIENT"));
        }

        #[test]
        fn test_abort_pattern() {
            let mut interception = NetworkInterception::new();
            interception.abort_pattern(
                UrlPattern::Prefix("https://blocked.com".to_string()),
                AbortReason::AccessDenied,
            );
            interception.start();

            let response = interception.handle_request(
                "https://blocked.com/any/path",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            assert!(response.is_some());
            assert_eq!(response.unwrap().status, 0);
        }

        #[test]
        fn test_route_action_default() {
            let action: RouteAction = Default::default();
            assert!(matches!(action, RouteAction::Continue));
        }

        #[test]
        fn test_route_action_respond() {
            let action = RouteAction::Respond(MockResponse::text("test"));
            if let RouteAction::Respond(resp) = action {
                assert_eq!(resp.body_string(), "test");
            } else {
                panic!("Expected Respond action");
            }
        }

        #[test]
        fn test_route_action_abort() {
            let action = RouteAction::Abort(AbortReason::TimedOut);
            if let RouteAction::Abort(reason) = action {
                assert_eq!(reason, AbortReason::TimedOut);
            } else {
                panic!("Expected Abort action");
            }
        }
    }

    mod wait_tests {
        use super::*;

        #[test]
        fn test_find_request() {
            let mut interception = NetworkInterception::new().capture_all();
            interception.start();

            interception.handle_request(
                "https://api.example.com/users/123",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            let request = interception.find_request(&UrlPattern::Contains("users".to_string()));
            assert!(request.is_some());
            assert!(request.unwrap().url.contains("users"));

            let not_found = interception.find_request(&UrlPattern::Contains("posts".to_string()));
            assert!(not_found.is_none());
        }

        #[test]
        fn test_find_response_for() {
            let mut interception = NetworkInterception::new();
            interception.get("/api/users", MockResponse::text("user data"));
            interception.start();

            // Trigger the route
            interception.handle_request(
                "https://example.com/api/users",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            // Response should be found for matched routes
            let resp = interception.find_response_for(&UrlPattern::Contains("users".to_string()));
            assert!(resp.is_some());
        }

        #[test]
        fn test_captured_responses() {
            let mut interception = NetworkInterception::new();
            interception.get("/api/users", MockResponse::text("users"));
            interception.post("/api/posts", MockResponse::text("posts"));
            interception.start();

            // Only trigger one route
            interception.handle_request(
                "https://example.com/api/users",
                HttpMethod::Get,
                HashMap::new(),
                None,
            );

            let responses = interception.captured_responses();
            assert_eq!(responses.len(), 1);
            assert_eq!(responses[0].body_string(), "users");
        }

        #[test]
        fn test_url_pattern_to_string() {
            let exact = UrlPattern::Exact("https://example.com".to_string());
            let prefix = UrlPattern::Prefix("https://".to_string());
            let contains = UrlPattern::Contains("api".to_string());
            let regex = UrlPattern::Regex(r"\d+".to_string());
            let glob = UrlPattern::Glob("**/api/*".to_string());
            let any = UrlPattern::Any;

            assert_eq!(exact.to_string(), "https://example.com");
            assert_eq!(prefix.to_string(), "https://");
            assert_eq!(contains.to_string(), "api");
            assert_eq!(regex.to_string(), r"\d+");
            assert_eq!(glob.to_string(), "**/api/*");
            assert_eq!(any.to_string(), "*");
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Network Interception Tests (Spec G.2 P0)
    // =========================================================================

    mod h0_network_tests {
        use super::*;

        #[test]
        fn h0_network_01_abort_reason_failed_message() {
            assert_eq!(AbortReason::Failed.message(), "net::ERR_FAILED");
        }

        #[test]
        fn h0_network_02_abort_reason_timed_out() {
            assert_eq!(AbortReason::TimedOut.message(), "net::ERR_TIMED_OUT");
        }

        #[test]
        fn h0_network_03_abort_reason_access_denied() {
            assert_eq!(
                AbortReason::AccessDenied.message(),
                "net::ERR_ACCESS_DENIED"
            );
        }

        #[test]
        fn h0_network_04_abort_reason_connection_refused() {
            assert_eq!(
                AbortReason::ConnectionRefused.message(),
                "net::ERR_CONNECTION_REFUSED"
            );
        }

        #[test]
        fn h0_network_05_abort_reason_internet_disconnected() {
            assert_eq!(
                AbortReason::InternetDisconnected.message(),
                "net::ERR_INTERNET_DISCONNECTED"
            );
        }

        #[test]
        fn h0_network_06_route_action_default_continue() {
            let action: RouteAction = Default::default();
            assert!(matches!(action, RouteAction::Continue));
        }

        #[test]
        fn h0_network_07_http_method_from_str_get() {
            let method = HttpMethod::from_str("GET");
            assert_eq!(method, HttpMethod::Get);
        }

        #[test]
        fn h0_network_08_http_method_from_str_post() {
            let method = HttpMethod::from_str("POST");
            assert_eq!(method, HttpMethod::Post);
        }

        #[test]
        fn h0_network_09_http_method_from_str_put() {
            let method = HttpMethod::from_str("PUT");
            assert_eq!(method, HttpMethod::Put);
        }

        #[test]
        fn h0_network_10_http_method_from_str_delete() {
            let method = HttpMethod::from_str("DELETE");
            assert_eq!(method, HttpMethod::Delete);
        }
    }

    mod h0_mock_response_tests {
        use super::*;

        #[test]
        fn h0_network_11_mock_response_text() {
            let resp = MockResponse::text("hello");
            assert_eq!(resp.body_string(), "hello");
        }

        #[test]
        fn h0_network_12_mock_response_json() {
            let resp = MockResponse::json(&serde_json::json!({"key": "value"})).unwrap();
            assert_eq!(resp.content_type, "application/json");
        }

        #[test]
        fn h0_network_13_mock_response_error() {
            let resp = MockResponse::error(404, "Not Found");
            assert_eq!(resp.status, 404);
        }

        #[test]
        fn h0_network_14_mock_response_status_200() {
            let resp = MockResponse::text("ok");
            assert_eq!(resp.status, 200);
        }

        #[test]
        fn h0_network_15_mock_response_with_body() {
            let resp = MockResponse::new().with_body(vec![1, 2, 3]);
            assert_eq!(resp.body, vec![1, 2, 3]);
        }

        #[test]
        fn h0_network_16_mock_response_with_header() {
            let resp = MockResponse::text("body").with_header("X-Custom", "value");
            assert_eq!(resp.headers.get("X-Custom"), Some(&"value".to_string()));
        }

        #[test]
        fn h0_network_17_mock_response_with_content_type() {
            let resp = MockResponse::new().with_content_type("text/plain");
            assert_eq!(resp.content_type, "text/plain");
        }

        #[test]
        fn h0_network_18_mock_response_default() {
            let resp = MockResponse::default();
            assert_eq!(resp.status, 200);
        }

        #[test]
        fn h0_network_19_mock_response_clone() {
            let resp1 = MockResponse::text("cloned");
            let resp2 = resp1.clone();
            assert_eq!(resp1.body, resp2.body);
        }

        #[test]
        fn h0_network_20_mock_response_debug() {
            let resp = MockResponse::text("test");
            let debug = format!("{:?}", resp);
            assert!(debug.contains("MockResponse"));
        }
    }

    mod h0_url_pattern_tests {
        use super::*;

        #[test]
        fn h0_network_21_url_pattern_exact_match() {
            let pattern = UrlPattern::Exact("https://example.com".to_string());
            assert!(pattern.matches("https://example.com"));
        }

        #[test]
        fn h0_network_22_url_pattern_exact_no_match() {
            let pattern = UrlPattern::Exact("https://example.com".to_string());
            assert!(!pattern.matches("https://example.com/path"));
        }

        #[test]
        fn h0_network_23_url_pattern_prefix_match() {
            let pattern = UrlPattern::Prefix("https://api.".to_string());
            assert!(pattern.matches("https://api.example.com/v1"));
        }

        #[test]
        fn h0_network_24_url_pattern_contains_match() {
            let pattern = UrlPattern::Contains("/api/".to_string());
            assert!(pattern.matches("https://example.com/api/users"));
        }

        #[test]
        fn h0_network_25_url_pattern_any_matches_all() {
            let pattern = UrlPattern::Any;
            assert!(pattern.matches("https://any-url.com/any/path"));
        }

        #[test]
        fn h0_network_26_url_pattern_glob_single_star() {
            let pattern = UrlPattern::Glob("**/api/*".to_string());
            assert!(pattern.matches("https://example.com/api/users"));
        }

        #[test]
        fn h0_network_27_url_pattern_regex() {
            let pattern = UrlPattern::Regex(r"/users/\d+".to_string());
            assert!(pattern.matches("https://api.com/users/123"));
        }

        #[test]
        fn h0_network_28_url_pattern_to_string_exact() {
            let pattern = UrlPattern::Exact("test".to_string());
            assert_eq!(pattern.to_string(), "test");
        }

        #[test]
        fn h0_network_29_url_pattern_to_string_any() {
            let pattern = UrlPattern::Any;
            assert_eq!(pattern.to_string(), "*");
        }

        #[test]
        fn h0_network_30_url_pattern_clone() {
            let pattern1 = UrlPattern::Contains("api".to_string());
            let pattern2 = pattern1;
            assert!(pattern2.matches("https://api.com"));
        }
    }

    mod h0_network_interception_tests {
        use super::*;

        #[test]
        fn h0_network_31_interception_new() {
            let interception = NetworkInterception::new();
            assert!(!interception.is_active());
        }

        #[test]
        fn h0_network_32_interception_start_stop() {
            let mut interception = NetworkInterception::new();
            interception.start();
            assert!(interception.is_active());
            interception.stop();
            assert!(!interception.is_active());
        }

        #[test]
        fn h0_network_33_interception_get_route() {
            let mut interception = NetworkInterception::new();
            interception.get("/api/users", MockResponse::text("users"));
            assert_eq!(interception.route_count(), 1);
        }

        #[test]
        fn h0_network_34_interception_post_route() {
            let mut interception = NetworkInterception::new();
            interception.post("/api/create", MockResponse::text("data"));
            assert_eq!(interception.route_count(), 1);
        }

        #[test]
        fn h0_network_35_interception_put_route() {
            let mut interception = NetworkInterception::new();
            interception.put("/api/update", MockResponse::text("updated"));
            assert_eq!(interception.route_count(), 1);
        }

        #[test]
        fn h0_network_36_interception_delete_route() {
            let mut interception = NetworkInterception::new();
            interception.delete("/api/remove", MockResponse::text("deleted"));
            assert_eq!(interception.route_count(), 1);
        }

        #[test]
        fn h0_network_37_interception_abort() {
            let mut interception = NetworkInterception::new();
            interception.abort("/blocked", AbortReason::BlockedByClient);
            assert_eq!(interception.route_count(), 1);
        }

        #[test]
        fn h0_network_38_interception_clear() {
            let mut interception = NetworkInterception::new();
            interception.get("/api", MockResponse::text("data"));
            interception.clear_routes();
            assert_eq!(interception.route_count(), 0);
        }

        #[test]
        fn h0_network_39_interception_captured_requests() {
            let interception = NetworkInterception::new();
            assert!(interception.captured_requests().is_empty());
        }

        #[test]
        fn h0_network_40_interception_captured_responses() {
            let interception = NetworkInterception::new();
            assert!(interception.captured_responses().is_empty());
        }
    }

    mod h0_route_tests {
        use super::*;

        #[test]
        fn h0_network_41_route_new() {
            let route = Route::new(
                UrlPattern::Contains("/api".to_string()),
                HttpMethod::Get,
                MockResponse::text("response"),
            );
            assert!(route.pattern.matches("https://example.com/api"));
        }

        #[test]
        fn h0_network_42_route_matches_url_and_method() {
            let route = Route::new(
                UrlPattern::Contains("/api".to_string()),
                HttpMethod::Post,
                MockResponse::text("data"),
            );
            assert!(route.matches("https://example.com/api", &HttpMethod::Post));
        }

        #[test]
        fn h0_network_43_route_no_match_wrong_method() {
            let route = Route::new(
                UrlPattern::Contains("/api".to_string()),
                HttpMethod::Get,
                MockResponse::text("data"),
            );
            assert!(!route.matches("https://example.com/api", &HttpMethod::Post));
        }

        #[test]
        fn h0_network_44_route_record_match() {
            let mut route =
                Route::new(UrlPattern::Any, HttpMethod::Get, MockResponse::text("data"));
            route.record_match();
            assert_eq!(route.match_count, 1);
        }

        #[test]
        fn h0_network_45_captured_request_new() {
            let request = CapturedRequest::new("https://api.com/users", HttpMethod::Get, 1000);
            assert_eq!(request.url, "https://api.com/users");
            assert_eq!(request.method, HttpMethod::Get);
        }

        #[test]
        fn h0_network_46_captured_request_body_string() {
            let mut request = CapturedRequest::new("https://api.com", HttpMethod::Post, 0);
            request.body = Some(b"hello".to_vec());
            assert_eq!(request.body_string(), Some("hello".to_string()));
        }

        #[test]
        fn h0_network_47_http_method_matches_same() {
            assert!(HttpMethod::Get.matches(&HttpMethod::Get));
        }

        #[test]
        fn h0_network_48_http_method_matches_any() {
            assert!(HttpMethod::Any.matches(&HttpMethod::Get));
            assert!(HttpMethod::Any.matches(&HttpMethod::Post));
        }

        #[test]
        fn h0_network_49_http_method_no_match_different() {
            assert!(!HttpMethod::Get.matches(&HttpMethod::Post));
        }

        #[test]
        fn h0_network_50_abort_reason_all_variants() {
            let reasons = vec![
                AbortReason::Failed,
                AbortReason::Aborted,
                AbortReason::TimedOut,
                AbortReason::AccessDenied,
                AbortReason::ConnectionClosed,
                AbortReason::ConnectionFailed,
                AbortReason::ConnectionRefused,
                AbortReason::ConnectionReset,
                AbortReason::InternetDisconnected,
                AbortReason::NameNotResolved,
                AbortReason::BlockedByClient,
            ];
            for reason in reasons {
                assert!(!reason.message().is_empty());
            }
        }
    }
}
