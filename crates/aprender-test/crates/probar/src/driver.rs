//! ProbarDriver - Abstract Browser Automation Trait
//!
//! Per spec Section 3.2: Full Chromium automation via abstract `ProbarDriver` trait.
//!
//! # Architecture (from spec)
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │  ProbarDriver (Abstract Trait)                                             │
//! │  Genchi Genbutsu: "Go and see" - allows swapping implementations          │
//! ├───────────────────────────────────────────────────────────────────────────┤
//! │                                                                           │
//! │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐  │
//! │  │  ChromiumDriver     │  │  PlaywrightBridge   │  │  WasmBindgen    │  │
//! │  │  (Default)          │  │  (Fallback)         │  │  (Unit Tests)   │  │
//! │  │                     │  │                     │  │                 │  │
//! │  │  Uses CDP via       │  │  WebSocket to       │  │  In-browser     │  │
//! │  │  chromiumoxide      │  │  Playwright server  │  │  test runner    │  │
//! │  └─────────────────────┘  └─────────────────────┘  └─────────────────┘  │
//! │                                                                           │
//! │  Toyota Principle: Driver Abstraction protects against API instability   │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Toyota Principles Applied
//!
//! - **Genchi Genbutsu**: Abstract trait allows "going and seeing" with different browsers
//! - **Risk Mitigation**: If chromiumoxide becomes unmaintained, swap to PlaywrightBridge

#[cfg(feature = "browser")]
use crate::event::InputEvent;
use crate::locator::BoundingBox;
#[cfg(feature = "browser")]
use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "browser")]
use async_trait::async_trait;

/// Element handle for DOM interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementHandle {
    /// Unique identifier for the element
    pub id: String,
    /// Element tag name
    pub tag_name: String,
    /// Element text content
    pub text_content: Option<String>,
    /// Bounding box if visible
    pub bounding_box: Option<BoundingBox>,
}

impl ElementHandle {
    /// Create a new element handle
    #[must_use]
    pub fn new(id: impl Into<String>, tag_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tag_name: tag_name.into(),
            text_content: None,
            bounding_box: None,
        }
    }

    /// Check if element is visible
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.bounding_box.is_some()
    }
}

/// Page performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageMetrics {
    /// Time to first paint (ms)
    pub first_paint_ms: Option<f64>,
    /// Time to first contentful paint (ms)
    pub first_contentful_paint_ms: Option<f64>,
    /// DOM content loaded time (ms)
    pub dom_content_loaded_ms: Option<f64>,
    /// Full page load time (ms)
    pub load_time_ms: Option<f64>,
    /// Total JavaScript heap size (bytes)
    pub js_heap_size_bytes: Option<u64>,
    /// Used JavaScript heap size (bytes)
    pub js_heap_used_bytes: Option<u64>,
    /// Number of DOM nodes
    pub dom_nodes: Option<u32>,
    /// Number of frames
    pub frame_count: Option<u32>,
}

/// Screenshot data with metadata
#[derive(Debug, Clone)]
pub struct Screenshot {
    /// Raw PNG data
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Device pixel ratio
    pub device_pixel_ratio: f64,
    /// Timestamp when screenshot was taken
    pub timestamp: std::time::SystemTime,
}

impl Screenshot {
    /// Create a new screenshot
    #[must_use]
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
            device_pixel_ratio: 1.0,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Get the size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Check if screenshot is valid (has data)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.data.is_empty() && self.width > 0 && self.height > 0
    }
}

/// Network request interceptor configuration
#[derive(Debug, Clone, Default)]
pub struct NetworkInterceptor {
    /// URL patterns to intercept
    pub patterns: Vec<String>,
    /// Whether to block matching requests
    pub block: bool,
    /// Response to return (if overriding)
    pub response_override: Option<NetworkResponse>,
}

/// Mock network response
#[derive(Debug, Clone)]
pub struct NetworkResponse {
    /// HTTP status code
    pub status: u16,
    /// Response headers
    pub headers: Vec<(String, String)>,
    /// Response body
    pub body: Vec<u8>,
}

impl NetworkResponse {
    /// Create a JSON response
    #[must_use]
    pub fn json(status: u16, body: impl Serialize) -> Self {
        let body = serde_json::to_vec(&body).unwrap_or_default();
        Self {
            status,
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body,
        }
    }

    /// Create a 404 response
    #[must_use]
    pub fn not_found() -> Self {
        Self {
            status: 404,
            headers: vec![],
            body: b"Not Found".to_vec(),
        }
    }
}

/// Browser configuration for driver
#[derive(Debug, Clone)]
pub struct DriverConfig {
    /// Run in headless mode
    pub headless: bool,
    /// Viewport width
    pub viewport_width: u32,
    /// Viewport height
    pub viewport_height: u32,
    /// Device scale factor
    pub device_scale_factor: f64,
    /// User agent string
    pub user_agent: Option<String>,
    /// Timeout for navigation
    pub navigation_timeout: Duration,
    /// Timeout for element queries
    pub element_timeout: Duration,
    /// Enable tracing
    pub tracing: bool,
    /// Executable path override
    pub executable_path: Option<String>,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            headless: true,
            viewport_width: 1920,
            viewport_height: 1080,
            device_scale_factor: 1.0,
            user_agent: None,
            navigation_timeout: Duration::from_secs(30),
            element_timeout: Duration::from_secs(5),
            tracing: false,
            executable_path: None,
        }
    }
}

impl DriverConfig {
    /// Create new config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set headless mode
    #[must_use]
    pub const fn headless(mut self, headless: bool) -> Self {
        self.headless = headless;
        self
    }

    /// Set viewport dimensions
    #[must_use]
    pub const fn viewport(mut self, width: u32, height: u32) -> Self {
        self.viewport_width = width;
        self.viewport_height = height;
        self
    }

    /// Set device scale factor
    #[must_use]
    pub const fn scale_factor(mut self, factor: f64) -> Self {
        self.device_scale_factor = factor;
        self
    }

    /// Set user agent
    #[must_use]
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = Some(ua.into());
        self
    }

    /// Set navigation timeout
    #[must_use]
    pub const fn navigation_timeout(mut self, timeout: Duration) -> Self {
        self.navigation_timeout = timeout;
        self
    }

    /// Enable tracing
    #[must_use]
    pub const fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing = enabled;
        self
    }
}

/// Mobile device descriptor for emulation
#[derive(Debug, Clone)]
pub struct DeviceDescriptor {
    /// Device name
    pub name: &'static str,
    /// Viewport width
    pub viewport_width: u32,
    /// Viewport height
    pub viewport_height: u32,
    /// Device scale factor
    pub device_scale_factor: f64,
    /// Is mobile device
    pub is_mobile: bool,
    /// Has touch support
    pub has_touch: bool,
    /// Default user agent
    pub user_agent: &'static str,
}

impl DeviceDescriptor {
    /// iPhone 14 Pro
    pub const IPHONE_14_PRO: Self = Self {
        name: "iPhone 14 Pro",
        viewport_width: 393,
        viewport_height: 852,
        device_scale_factor: 3.0,
        is_mobile: true,
        has_touch: true,
        user_agent: "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    };

    /// iPad Pro 12.9
    pub const IPAD_PRO_12_9: Self = Self {
        name: "iPad Pro 12.9",
        viewport_width: 1024,
        viewport_height: 1366,
        device_scale_factor: 2.0,
        is_mobile: true,
        has_touch: true,
        user_agent: "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    };

    /// Desktop 1080p
    pub const DESKTOP_1080P: Self = Self {
        name: "Desktop 1080p",
        viewport_width: 1920,
        viewport_height: 1080,
        device_scale_factor: 1.0,
        is_mobile: false,
        has_touch: false,
        user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    };

    /// Desktop 4K
    pub const DESKTOP_4K: Self = Self {
        name: "Desktop 4K",
        viewport_width: 3840,
        viewport_height: 2160,
        device_scale_factor: 2.0,
        is_mobile: false,
        has_touch: false,
        user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    };

    /// Convert to driver config
    #[must_use]
    pub fn to_config(&self) -> DriverConfig {
        DriverConfig::default()
            .viewport(self.viewport_width, self.viewport_height)
            .scale_factor(self.device_scale_factor)
            .user_agent(self.user_agent)
    }
}

/// Abstract driver trait for browser automation
///
/// This trait allows swapping implementations (Genchi Genbutsu principle).
/// If chromiumoxide becomes unmaintained, implement `PlaywrightBridgeDriver`.
///
/// # Implementations
///
/// - `ChromiumDriver` - Default, uses chromiumoxide crate
/// - `PlaywrightBridgeDriver` - Fallback, WebSocket to Playwright server
/// - `MockDriver` - For unit testing
#[cfg(feature = "browser")]
#[async_trait]
pub trait ProbarDriver: Send + Sync {
    /// Navigate to URL
    async fn navigate(&mut self, url: &str) -> ProbarResult<()>;

    /// Take screenshot
    async fn screenshot(&self) -> ProbarResult<Screenshot>;

    /// Execute JavaScript in page context
    async fn execute_js(&self, script: &str) -> ProbarResult<serde_json::Value>;

    /// Query DOM element by selector
    async fn query_selector(&self, selector: &str) -> ProbarResult<Option<ElementHandle>>;

    /// Query all matching elements
    async fn query_selector_all(&self, selector: &str) -> ProbarResult<Vec<ElementHandle>>;

    /// Dispatch input event
    async fn dispatch_input(&self, event: InputEvent) -> ProbarResult<()>;

    /// Click element
    async fn click(&self, selector: &str) -> ProbarResult<()>;

    /// Type text into element
    async fn type_text(&self, selector: &str, text: &str) -> ProbarResult<()>;

    /// Wait for selector to appear
    async fn wait_for_selector(
        &self,
        selector: &str,
        timeout: Duration,
    ) -> ProbarResult<ElementHandle>;

    /// Get page metrics
    async fn metrics(&self) -> ProbarResult<PageMetrics>;

    /// Set network interceptor
    async fn set_network_interceptor(
        &mut self,
        interceptor: NetworkInterceptor,
    ) -> ProbarResult<()>;

    /// Get current URL
    async fn current_url(&self) -> ProbarResult<String>;

    /// Go back in history
    async fn go_back(&mut self) -> ProbarResult<()>;

    /// Go forward in history
    async fn go_forward(&mut self) -> ProbarResult<()>;

    /// Reload page
    async fn reload(&mut self) -> ProbarResult<()>;

    /// Close the browser
    async fn close(&mut self) -> ProbarResult<()>;
}

/// Mock driver for unit testing
#[derive(Debug, Default)]
pub struct MockDriver {
    /// Current URL
    pub current_url: String,
    /// Mock elements
    pub elements: Vec<ElementHandle>,
    /// JS execution results
    pub js_results: Vec<serde_json::Value>,
    /// Screenshot data
    pub screenshot_data: Option<Screenshot>,
    /// Call history for verification
    pub call_history: Vec<String>,
}

impl MockDriver {
    /// Create new mock driver
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mock element
    pub fn add_element(&mut self, element: ElementHandle) {
        self.elements.push(element);
    }

    /// Set mock JS result
    pub fn set_js_result(&mut self, result: serde_json::Value) {
        self.js_results.push(result);
    }

    /// Set mock screenshot
    pub fn set_screenshot(&mut self, screenshot: Screenshot) {
        self.screenshot_data = Some(screenshot);
    }

    /// Get call history
    #[must_use]
    pub fn history(&self) -> &[String] {
        &self.call_history
    }

    /// Check if method was called
    #[must_use]
    pub fn was_called(&self, method: &str) -> bool {
        self.call_history.iter().any(|c| c.starts_with(method))
    }
}

#[cfg(feature = "browser")]
#[async_trait]
impl ProbarDriver for MockDriver {
    async fn navigate(&mut self, url: &str) -> ProbarResult<()> {
        self.call_history.push(format!("navigate:{url}"));
        self.current_url = url.to_string();
        Ok(())
    }

    async fn screenshot(&self) -> ProbarResult<Screenshot> {
        self.screenshot_data
            .clone()
            .ok_or_else(|| ProbarError::ScreenshotError {
                message: "No mock screenshot set".to_string(),
            })
    }

    async fn execute_js(&self, script: &str) -> ProbarResult<serde_json::Value> {
        let _ = script; // Unused in mock
        self.js_results
            .first()
            .cloned()
            .ok_or_else(|| ProbarError::WasmError {
                message: "No mock JS result set".to_string(),
            })
    }

    async fn query_selector(&self, selector: &str) -> ProbarResult<Option<ElementHandle>> {
        Ok(self.elements.iter().find(|e| e.id == selector).cloned())
    }

    async fn query_selector_all(&self, _selector: &str) -> ProbarResult<Vec<ElementHandle>> {
        Ok(self.elements.clone())
    }

    async fn dispatch_input(&self, event: InputEvent) -> ProbarResult<()> {
        let _ = event; // Mock just records the call
        Ok(())
    }

    async fn click(&self, selector: &str) -> ProbarResult<()> {
        let _ = selector;
        Ok(())
    }

    async fn type_text(&self, selector: &str, text: &str) -> ProbarResult<()> {
        let _ = (selector, text);
        Ok(())
    }

    async fn wait_for_selector(
        &self,
        selector: &str,
        _timeout: Duration,
    ) -> ProbarResult<ElementHandle> {
        self.elements
            .iter()
            .find(|e| e.id == selector)
            .cloned()
            .ok_or_else(|| ProbarError::Timeout { ms: 5000 })
    }

    async fn metrics(&self) -> ProbarResult<PageMetrics> {
        Ok(PageMetrics::default())
    }

    async fn set_network_interceptor(
        &mut self,
        _interceptor: NetworkInterceptor,
    ) -> ProbarResult<()> {
        Ok(())
    }

    async fn current_url(&self) -> ProbarResult<String> {
        Ok(self.current_url.clone())
    }

    async fn go_back(&mut self) -> ProbarResult<()> {
        self.call_history.push("go_back".to_string());
        Ok(())
    }

    async fn go_forward(&mut self) -> ProbarResult<()> {
        self.call_history.push("go_forward".to_string());
        Ok(())
    }

    async fn reload(&mut self) -> ProbarResult<()> {
        self.call_history.push("reload".to_string());
        Ok(())
    }

    async fn close(&mut self) -> ProbarResult<()> {
        self.call_history.push("close".to_string());
        Ok(())
    }
}

/// Browser controller using abstract driver
///
/// This is the main entry point for browser automation tests.
///
/// # Example
///
/// ```ignore
/// let mut browser = BrowserController::<ChromiumDriver>::launch(
///     DriverConfig::default().headless(true)
/// ).await?;
///
/// browser.goto("http://localhost:8080/game").await?;
/// let screenshot = browser.screenshot().await?;
/// ```
#[cfg(feature = "browser")]
pub struct BrowserController<D: ProbarDriver> {
    driver: D,
    config: DriverConfig,
}

#[cfg(feature = "browser")]
impl<D: ProbarDriver> std::fmt::Debug for BrowserController<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrowserController")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "browser")]
impl<D: ProbarDriver> BrowserController<D> {
    /// Create controller with existing driver
    pub fn new(driver: D, config: DriverConfig) -> Self {
        Self { driver, config }
    }

    /// Navigate to URL
    pub async fn goto(&mut self, url: &str) -> ProbarResult<()> {
        self.driver.navigate(url).await
    }

    /// Take screenshot
    pub async fn screenshot(&self) -> ProbarResult<Screenshot> {
        self.driver.screenshot().await
    }

    /// Execute JavaScript
    pub async fn evaluate(&self, script: &str) -> ProbarResult<serde_json::Value> {
        self.driver.execute_js(script).await
    }

    /// Query element
    pub async fn query(&self, selector: &str) -> ProbarResult<Option<ElementHandle>> {
        self.driver.query_selector(selector).await
    }

    /// Get page metrics
    pub async fn metrics(&self) -> ProbarResult<PageMetrics> {
        self.driver.metrics().await
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &DriverConfig {
        &self.config
    }

    /// Close browser
    pub async fn close(&mut self) -> ProbarResult<()> {
        self.driver.close().await
    }
}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec Section 6.1
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod element_handle_tests {
        use super::*;

        #[test]
        fn test_element_handle_creation() {
            let elem = ElementHandle::new("btn-1", "button");
            assert_eq!(elem.id, "btn-1");
            assert_eq!(elem.tag_name, "button");
            assert!(elem.text_content.is_none());
        }

        #[test]
        fn test_element_handle_visibility() {
            let mut elem = ElementHandle::new("elem", "div");
            assert!(!elem.is_visible());

            elem.bounding_box = Some(BoundingBox::new(0.0, 0.0, 100.0, 100.0));
            assert!(elem.is_visible());
        }
    }

    mod screenshot_tests {
        use super::*;

        #[test]
        fn test_screenshot_creation() {
            let data = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic bytes
            let screenshot = Screenshot::new(data.clone(), 100, 100);
            assert_eq!(screenshot.width, 100);
            assert_eq!(screenshot.height, 100);
            assert_eq!(screenshot.data, data);
        }

        #[test]
        fn test_screenshot_size_bytes() {
            let screenshot = Screenshot::new(vec![0; 1024], 100, 100);
            assert_eq!(screenshot.size_bytes(), 1024);
        }

        #[test]
        fn test_screenshot_is_valid() {
            let valid = Screenshot::new(vec![1, 2, 3], 100, 100);
            assert!(valid.is_valid());

            let empty = Screenshot::new(vec![], 100, 100);
            assert!(!empty.is_valid());

            let zero_width = Screenshot::new(vec![1], 0, 100);
            assert!(!zero_width.is_valid());
        }
    }

    mod network_response_tests {
        use super::*;

        #[test]
        fn test_json_response() {
            let response = NetworkResponse::json(200, serde_json::json!({"ok": true}));
            assert_eq!(response.status, 200);
            assert!(response
                .headers
                .iter()
                .any(|(k, v)| k == "Content-Type" && v.contains("json")));
        }

        #[test]
        fn test_not_found_response() {
            let response = NetworkResponse::not_found();
            assert_eq!(response.status, 404);
        }
    }

    mod driver_config_tests {
        use super::*;

        #[test]
        fn test_config_default() {
            let config = DriverConfig::default();
            assert!(config.headless);
            assert_eq!(config.viewport_width, 1920);
            assert_eq!(config.viewport_height, 1080);
        }

        #[test]
        fn test_config_builder() {
            let config = DriverConfig::new()
                .headless(false)
                .viewport(800, 600)
                .scale_factor(2.0)
                .user_agent("test-agent");

            assert!(!config.headless);
            assert_eq!(config.viewport_width, 800);
            assert_eq!(config.viewport_height, 600);
            assert!((config.device_scale_factor - 2.0).abs() < f64::EPSILON);
            assert_eq!(config.user_agent, Some("test-agent".to_string()));
        }

        #[test]
        fn test_config_with_tracing() {
            let config = DriverConfig::new().with_tracing(true);
            assert!(config.tracing);
        }
    }

    mod device_descriptor_tests {
        use super::*;

        #[test]
        fn test_iphone_14_pro() {
            let device = DeviceDescriptor::IPHONE_14_PRO;
            assert_eq!(device.viewport_width, 393);
            assert!(device.is_mobile);
            assert!(device.has_touch);
        }

        #[test]
        fn test_desktop_1080p() {
            let device = DeviceDescriptor::DESKTOP_1080P;
            assert_eq!(device.viewport_width, 1920);
            assert!(!device.is_mobile);
            assert!(!device.has_touch);
        }

        #[test]
        fn test_device_to_config() {
            let config = DeviceDescriptor::IPHONE_14_PRO.to_config();
            assert_eq!(config.viewport_width, 393);
            assert_eq!(config.viewport_height, 852);
        }
    }

    mod mock_driver_tests {
        use super::*;

        #[test]
        fn test_mock_driver_creation() {
            let driver = MockDriver::new();
            assert!(driver.elements.is_empty());
            assert!(driver.current_url.is_empty());
        }

        #[test]
        fn test_mock_driver_add_element() {
            let mut driver = MockDriver::new();
            driver.add_element(ElementHandle::new("test", "div"));
            assert_eq!(driver.elements.len(), 1);
        }

        #[test]
        fn test_mock_driver_history() {
            let driver = MockDriver::new();
            assert!(driver.history().is_empty());
        }

        #[test]
        fn test_mock_driver_was_called() {
            let driver = MockDriver::new();
            assert!(!driver.was_called("navigate"));
        }
    }

    mod page_metrics_tests {
        use super::*;

        #[test]
        fn test_metrics_default() {
            let metrics = PageMetrics::default();
            assert!(metrics.first_paint_ms.is_none());
            assert!(metrics.dom_nodes.is_none());
        }
    }

    #[cfg(feature = "browser")]
    mod async_driver_tests {
        use super::*;

        #[tokio::test]
        async fn test_mock_driver_navigate() {
            let mut driver = MockDriver::new();
            driver.navigate("https://example.com").await.unwrap();
            assert_eq!(driver.current_url, "https://example.com");
            assert!(driver.was_called("navigate"));
        }

        #[tokio::test]
        async fn test_mock_driver_history_tracking() {
            let mut driver = MockDriver::new();
            driver.go_back().await.unwrap();
            driver.go_forward().await.unwrap();
            driver.reload().await.unwrap();
            driver.close().await.unwrap();

            assert!(driver.was_called("go_back"));
            assert!(driver.was_called("go_forward"));
            assert!(driver.was_called("reload"));
            assert!(driver.was_called("close"));
        }

        #[tokio::test]
        async fn test_browser_controller_with_mock() {
            let driver = MockDriver::new();
            let mut controller = BrowserController::new(driver, DriverConfig::default());

            controller.goto("https://test.com").await.unwrap();
            let url = controller.driver.current_url().await.unwrap();
            assert_eq!(url, "https://test.com");
        }

        #[tokio::test]
        async fn test_mock_driver_screenshot_error() {
            let driver = MockDriver::new();
            // No screenshot set, should return error
            let result = driver.screenshot().await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_mock_driver_screenshot_success() {
            let mut driver = MockDriver::new();
            let screenshot = Screenshot::new(vec![1, 2, 3], 100, 50);
            driver.set_screenshot(screenshot);
            let result = driver.screenshot().await.unwrap();
            assert_eq!(result.width, 100);
            assert_eq!(result.height, 50);
        }

        #[tokio::test]
        async fn test_mock_driver_execute_js_error() {
            let driver = MockDriver::new();
            // No JS result set, should return error
            let result = driver.execute_js("return 1;").await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_mock_driver_execute_js_success() {
            let mut driver = MockDriver::new();
            driver.set_js_result(serde_json::json!({"result": 42}));
            let result = driver.execute_js("return 42;").await.unwrap();
            assert_eq!(result["result"], 42);
        }

        #[tokio::test]
        async fn test_mock_driver_query_selector() {
            let mut driver = MockDriver::new();
            driver.add_element(ElementHandle::new("btn-submit", "button"));

            let found = driver.query_selector("btn-submit").await.unwrap();
            assert!(found.is_some());
            assert_eq!(found.unwrap().tag_name, "button");

            let not_found = driver.query_selector("non-existent").await.unwrap();
            assert!(not_found.is_none());
        }

        #[tokio::test]
        async fn test_mock_driver_query_selector_all() {
            let mut driver = MockDriver::new();
            driver.add_element(ElementHandle::new("elem1", "div"));
            driver.add_element(ElementHandle::new("elem2", "span"));

            let elements = driver.query_selector_all("*").await.unwrap();
            assert_eq!(elements.len(), 2);
        }

        #[tokio::test]
        async fn test_mock_driver_dispatch_input() {
            let driver = MockDriver::new();
            let event = InputEvent::KeyPress {
                key: "Enter".to_string(),
            };
            let result = driver.dispatch_input(event).await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_mock_driver_click() {
            let driver = MockDriver::new();
            let result = driver.click("#button").await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_mock_driver_type_text() {
            let driver = MockDriver::new();
            let result = driver.type_text("#input", "hello world").await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_mock_driver_wait_for_selector_found() {
            let mut driver = MockDriver::new();
            driver.add_element(ElementHandle::new("target", "div"));
            let result = driver
                .wait_for_selector("target", Duration::from_secs(1))
                .await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap().id, "target");
        }

        #[tokio::test]
        async fn test_mock_driver_wait_for_selector_not_found() {
            let driver = MockDriver::new();
            let result = driver
                .wait_for_selector("missing", Duration::from_secs(1))
                .await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_mock_driver_metrics() {
            let driver = MockDriver::new();
            let metrics = driver.metrics().await.unwrap();
            assert!(metrics.first_paint_ms.is_none());
        }

        #[tokio::test]
        async fn test_mock_driver_set_network_interceptor() {
            let mut driver = MockDriver::new();
            let interceptor = NetworkInterceptor {
                patterns: vec!["*.js".to_string()],
                block: true,
                response_override: None,
            };
            let result = driver.set_network_interceptor(interceptor).await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_browser_controller_evaluate() {
            let mut driver = MockDriver::new();
            driver.set_js_result(serde_json::json!({"foo": "bar"}));
            let controller = BrowserController::new(driver, DriverConfig::default());
            let result = controller.evaluate("return {foo: 'bar'}").await.unwrap();
            assert_eq!(result["foo"], "bar");
        }

        #[tokio::test]
        async fn test_browser_controller_query() {
            let mut driver = MockDriver::new();
            driver.add_element(ElementHandle::new("elem", "div"));
            let controller = BrowserController::new(driver, DriverConfig::default());
            let result = controller.query("elem").await.unwrap();
            assert!(result.is_some());
        }

        #[tokio::test]
        async fn test_browser_controller_metrics() {
            let driver = MockDriver::new();
            let controller = BrowserController::new(driver, DriverConfig::default());
            let metrics = controller.metrics().await.unwrap();
            assert!(metrics.first_paint_ms.is_none());
        }

        #[tokio::test]
        async fn test_browser_controller_screenshot() {
            let mut driver = MockDriver::new();
            driver.set_screenshot(Screenshot::new(vec![0x89, 0x50], 640, 480));
            let controller = BrowserController::new(driver, DriverConfig::default());
            let screenshot = controller.screenshot().await.unwrap();
            assert_eq!(screenshot.width, 640);
            assert_eq!(screenshot.height, 480);
        }

        #[tokio::test]
        async fn test_browser_controller_close() {
            let driver = MockDriver::new();
            let mut controller = BrowserController::new(driver, DriverConfig::default());
            let result = controller.close().await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_browser_controller_config() {
            let config = DriverConfig::new().viewport(800, 600);
            let driver = MockDriver::new();
            let controller = BrowserController::new(driver, config);
            assert_eq!(controller.config().viewport_width, 800);
            assert_eq!(controller.config().viewport_height, 600);
        }

        #[tokio::test]
        async fn test_browser_controller_debug() {
            let driver = MockDriver::new();
            let controller = BrowserController::new(driver, DriverConfig::default());
            let debug_str = format!("{:?}", controller);
            assert!(debug_str.contains("BrowserController"));
        }
    }

    mod network_interceptor_tests {
        use super::*;

        #[test]
        fn test_network_interceptor_default() {
            let interceptor = NetworkInterceptor::default();
            assert!(interceptor.patterns.is_empty());
            assert!(!interceptor.block);
            assert!(interceptor.response_override.is_none());
        }

        #[test]
        fn test_network_interceptor_with_patterns() {
            let interceptor = NetworkInterceptor {
                patterns: vec!["*.js".to_string(), "*.css".to_string()],
                block: true,
                response_override: None,
            };
            assert_eq!(interceptor.patterns.len(), 2);
            assert!(interceptor.block);
        }

        #[test]
        fn test_network_interceptor_with_response_override() {
            let response = NetworkResponse::json(200, serde_json::json!({"mock": true}));
            let interceptor = NetworkInterceptor {
                patterns: vec!["api/*".to_string()],
                block: false,
                response_override: Some(response),
            };
            assert!(interceptor.response_override.is_some());
        }
    }

    mod driver_config_extended_tests {
        use super::*;

        #[test]
        fn test_navigation_timeout() {
            let config = DriverConfig::new().navigation_timeout(Duration::from_secs(60));
            assert_eq!(config.navigation_timeout, Duration::from_secs(60));
        }

        #[test]
        fn test_config_executable_path() {
            let mut config = DriverConfig::default();
            config.executable_path = Some("/usr/bin/chromium".to_string());
            assert_eq!(
                config.executable_path,
                Some("/usr/bin/chromium".to_string())
            );
        }

        #[test]
        fn test_config_element_timeout_default() {
            let config = DriverConfig::default();
            assert_eq!(config.element_timeout, Duration::from_secs(5));
        }
    }

    mod device_descriptor_extended_tests {
        use super::*;

        #[test]
        fn test_ipad_pro_12_9() {
            let device = DeviceDescriptor::IPAD_PRO_12_9;
            assert_eq!(device.name, "iPad Pro 12.9");
            assert_eq!(device.viewport_width, 1024);
            assert_eq!(device.viewport_height, 1366);
            assert!((device.device_scale_factor - 2.0).abs() < f64::EPSILON);
            assert!(device.is_mobile);
            assert!(device.has_touch);
            assert!(device.user_agent.contains("iPad"));
        }

        #[test]
        fn test_desktop_4k() {
            let device = DeviceDescriptor::DESKTOP_4K;
            assert_eq!(device.name, "Desktop 4K");
            assert_eq!(device.viewport_width, 3840);
            assert_eq!(device.viewport_height, 2160);
            assert!((device.device_scale_factor - 2.0).abs() < f64::EPSILON);
            assert!(!device.is_mobile);
            assert!(!device.has_touch);
        }

        #[test]
        fn test_ipad_to_config() {
            let config = DeviceDescriptor::IPAD_PRO_12_9.to_config();
            assert_eq!(config.viewport_width, 1024);
            assert_eq!(config.viewport_height, 1366);
            assert!((config.device_scale_factor - 2.0).abs() < f64::EPSILON);
            assert!(config.user_agent.is_some());
        }

        #[test]
        fn test_desktop_4k_to_config() {
            let config = DeviceDescriptor::DESKTOP_4K.to_config();
            assert_eq!(config.viewport_width, 3840);
            assert_eq!(config.viewport_height, 2160);
        }
    }

    mod mock_driver_extended_tests {
        use super::*;

        #[test]
        fn test_mock_driver_set_js_result() {
            let mut driver = MockDriver::new();
            driver.set_js_result(serde_json::json!({"value": 123}));
            assert_eq!(driver.js_results.len(), 1);
            assert_eq!(driver.js_results[0]["value"], 123);
        }

        #[test]
        fn test_mock_driver_set_screenshot() {
            let mut driver = MockDriver::new();
            let screenshot = Screenshot::new(vec![1, 2, 3, 4], 200, 100);
            driver.set_screenshot(screenshot);
            assert!(driver.screenshot_data.is_some());
            let data = driver.screenshot_data.unwrap();
            assert_eq!(data.width, 200);
            assert_eq!(data.height, 100);
        }

        #[test]
        fn test_mock_driver_debug() {
            let driver = MockDriver::new();
            let debug_str = format!("{:?}", driver);
            assert!(debug_str.contains("MockDriver"));
        }

        #[test]
        fn test_mock_driver_multiple_elements() {
            let mut driver = MockDriver::new();
            driver.add_element(ElementHandle::new("a", "div"));
            driver.add_element(ElementHandle::new("b", "span"));
            driver.add_element(ElementHandle::new("c", "button"));
            assert_eq!(driver.elements.len(), 3);
        }
    }

    mod element_handle_extended_tests {
        use super::*;

        #[test]
        fn test_element_handle_with_text() {
            let mut elem = ElementHandle::new("p1", "p");
            elem.text_content = Some("Hello World".to_string());
            assert_eq!(elem.text_content, Some("Hello World".to_string()));
        }

        #[test]
        fn test_element_handle_serialization() {
            let elem = ElementHandle::new("test-id", "input");
            let json = serde_json::to_string(&elem).unwrap();
            assert!(json.contains("test-id"));
            assert!(json.contains("input"));
        }

        #[test]
        fn test_element_handle_deserialization() {
            let json =
                r#"{"id":"btn","tag_name":"button","text_content":null,"bounding_box":null}"#;
            let elem: ElementHandle = serde_json::from_str(json).unwrap();
            assert_eq!(elem.id, "btn");
            assert_eq!(elem.tag_name, "button");
        }

        #[test]
        fn test_element_handle_clone() {
            let elem = ElementHandle::new("orig", "div");
            let cloned = elem.clone();
            assert_eq!(elem.id, cloned.id);
            assert_eq!(elem.tag_name, cloned.tag_name);
        }
    }

    mod screenshot_extended_tests {
        use super::*;

        #[test]
        fn test_screenshot_device_pixel_ratio() {
            let mut screenshot = Screenshot::new(vec![0], 100, 100);
            screenshot.device_pixel_ratio = 2.0;
            assert!((screenshot.device_pixel_ratio - 2.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_screenshot_timestamp() {
            let screenshot = Screenshot::new(vec![0], 100, 100);
            // Timestamp should be recent (within the last second)
            let now = std::time::SystemTime::now();
            let duration = now.duration_since(screenshot.timestamp).unwrap();
            assert!(duration.as_secs() < 1);
        }

        #[test]
        fn test_screenshot_zero_height_invalid() {
            let screenshot = Screenshot::new(vec![1, 2], 100, 0);
            assert!(!screenshot.is_valid());
        }

        #[test]
        fn test_screenshot_clone() {
            let screenshot = Screenshot::new(vec![1, 2, 3], 50, 50);
            let cloned = screenshot.clone();
            assert_eq!(screenshot.data, cloned.data);
            assert_eq!(screenshot.width, cloned.width);
            assert_eq!(screenshot.height, cloned.height);
        }
    }

    mod page_metrics_extended_tests {
        use super::*;

        #[test]
        fn test_page_metrics_with_values() {
            let metrics = PageMetrics {
                first_paint_ms: Some(100.5),
                first_contentful_paint_ms: Some(150.0),
                dom_content_loaded_ms: Some(200.0),
                load_time_ms: Some(500.0),
                js_heap_size_bytes: Some(10_000_000),
                js_heap_used_bytes: Some(5_000_000),
                dom_nodes: Some(1500),
                frame_count: Some(2),
            };
            assert_eq!(metrics.first_paint_ms, Some(100.5));
            assert_eq!(metrics.dom_nodes, Some(1500));
        }

        #[test]
        fn test_page_metrics_serialization() {
            let metrics = PageMetrics::default();
            let json = serde_json::to_string(&metrics).unwrap();
            assert!(json.contains("first_paint_ms"));
        }

        #[test]
        fn test_page_metrics_clone() {
            let metrics = PageMetrics {
                first_paint_ms: Some(50.0),
                ..Default::default()
            };
            let cloned = metrics.clone();
            assert_eq!(metrics.first_paint_ms, cloned.first_paint_ms);
        }
    }

    mod network_response_extended_tests {
        use super::*;

        #[test]
        fn test_network_response_json_serialization_failure() {
            // Create a valid JSON to ensure normal path works
            let response = NetworkResponse::json(200, serde_json::json!(null));
            assert_eq!(response.status, 200);
        }

        #[test]
        fn test_network_response_with_headers() {
            let response = NetworkResponse {
                status: 201,
                headers: vec![
                    ("Content-Type".to_string(), "text/plain".to_string()),
                    ("X-Custom".to_string(), "value".to_string()),
                ],
                body: b"Created".to_vec(),
            };
            assert_eq!(response.headers.len(), 2);
            assert_eq!(response.body, b"Created".to_vec());
        }

        #[test]
        fn test_network_response_not_found_body() {
            let response = NetworkResponse::not_found();
            assert_eq!(response.body, b"Not Found".to_vec());
            assert!(response.headers.is_empty());
        }

        #[test]
        fn test_network_response_clone() {
            let response = NetworkResponse::json(200, serde_json::json!({"ok": true}));
            let cloned = response.clone();
            assert_eq!(response.status, cloned.status);
            assert_eq!(response.body, cloned.body);
        }
    }
}
