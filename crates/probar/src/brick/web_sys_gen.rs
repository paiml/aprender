//! WebSys Code Generation (PROBAR-SPEC-009-P7: PROBAR-WEBSYS-001)
//!
//! Generates web_sys binding code from brick definitions.
//! This module provides abstractions that replace hand-written web_sys calls.
//!
//! # Design Philosophy
//!
//! Instead of hand-writing web_sys calls like:
//! ```rust,ignore
//! let start = web_sys::window().unwrap().performance().unwrap().now();
//! ```
//!
//! Use generated abstractions:
//! ```rust,ignore
//! use probar::brick::web_sys_gen::Performance;
//! let start = Performance::now();
//! ```
//!
//! The generated code is still web_sys underneath, but:
//! 1. It's derived from brick specifications (traceable)
//! 2. Error handling is consistent
//! 3. No hand-written web_sys in application code

use std::fmt;

// ============================================================================
// Performance Timing (replaces web_sys::window().performance())
// ============================================================================

/// Generated performance timing utilities
///
/// Replaces hand-written:
/// ```rust,ignore
/// web_sys::window().unwrap().performance().unwrap().now()
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PerformanceTiming;

impl PerformanceTiming {
    /// Get current timestamp in milliseconds (high resolution)
    ///
    /// Generated binding for `performance.now()`
    #[cfg(target_arch = "wasm32")]
    #[must_use]
    pub fn now() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0)
    }

    /// Get current timestamp (native fallback)
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn now() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    /// Measure duration of an operation
    #[must_use]
    pub fn measure<F, T>(f: F) -> (T, f64)
    where
        F: FnOnce() -> T,
    {
        let start = Self::now();
        let result = f();
        let duration = Self::now() - start;
        (result, duration)
    }
}

// ============================================================================
// Custom Events (replaces web_sys::CustomEvent)
// ============================================================================

/// Event detail that can be serialized to JS
#[derive(Debug, Clone)]
pub enum EventDetail {
    /// No detail
    None,
    /// String detail
    String(String),
    /// Number detail
    Number(f64),
    /// Boolean detail
    Bool(bool),
    /// JSON object detail
    Json(String),
}

impl EventDetail {
    /// Create from a string
    #[must_use]
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Create from a number
    #[must_use]
    pub fn number(n: f64) -> Self {
        Self::Number(n)
    }

    /// Create JSON detail from serializable value
    #[must_use]
    pub fn json<T: serde::Serialize>(value: &T) -> Self {
        match serde_json::to_string(value) {
            Ok(s) => Self::Json(s),
            Err(_) => Self::None,
        }
    }
}

/// Generated custom event dispatcher
///
/// Replaces hand-written:
/// ```rust,ignore
/// let init = web_sys::CustomEventInit::new();
/// init.set_detail(&detail.into());
/// let event = web_sys::CustomEvent::new_with_event_init_dict("my-event", &init)?;
/// window.dispatch_event(&event)?;
/// ```
#[derive(Debug, Clone)]
pub struct CustomEventDispatcher {
    #[allow(dead_code)] // Used only in wasm32 target
    event_name: String,
}

impl CustomEventDispatcher {
    /// Create a new event dispatcher for a specific event type
    #[must_use]
    pub fn new(event_name: impl Into<String>) -> Self {
        Self {
            event_name: event_name.into(),
        }
    }

    /// Dispatch event with no detail
    #[cfg(target_arch = "wasm32")]
    pub fn dispatch(&self) -> Result<bool, WebSysError> {
        use wasm_bindgen::JsCast;

        let window = web_sys::window().ok_or(WebSysError::NoWindow)?;

        let event = web_sys::CustomEvent::new(&self.event_name)
            .map_err(|_| WebSysError::EventCreationFailed)?;

        window
            .dispatch_event(&event)
            .map_err(|_| WebSysError::DispatchFailed)
    }

    /// Dispatch event with detail
    #[cfg(target_arch = "wasm32")]
    pub fn dispatch_with_detail(&self, detail: EventDetail) -> Result<bool, WebSysError> {
        use wasm_bindgen::JsValue;

        let window = web_sys::window().ok_or(WebSysError::NoWindow)?;

        let init = web_sys::CustomEventInit::new();

        let js_detail: JsValue = match detail {
            EventDetail::None => JsValue::NULL,
            EventDetail::String(s) => JsValue::from_str(&s),
            EventDetail::Number(n) => JsValue::from_f64(n),
            EventDetail::Bool(b) => JsValue::from_bool(b),
            EventDetail::Json(json) => js_sys::JSON::parse(&json).unwrap_or(JsValue::NULL),
        };

        init.set_detail(&js_detail);

        let event = web_sys::CustomEvent::new_with_event_init_dict(&self.event_name, &init)
            .map_err(|_| WebSysError::EventCreationFailed)?;

        window
            .dispatch_event(&event)
            .map_err(|_| WebSysError::DispatchFailed)
    }

    /// Native fallback - no-op
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch(&self) -> Result<bool, WebSysError> {
        Ok(true)
    }

    /// Native fallback - no-op
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_with_detail(&self, _detail: EventDetail) -> Result<bool, WebSysError> {
        Ok(true)
    }
}

// ============================================================================
// Fetch API (replaces web_sys::window().fetch_with_str())
// ============================================================================

/// Generated fetch result
#[derive(Debug)]
pub struct FetchResult {
    /// Response status code
    pub status: u16,
    /// Response body as bytes
    pub body: Vec<u8>,
}

/// Generated fetch client
///
/// Replaces hand-written fetch calls
#[derive(Debug, Clone, Default)]
pub struct FetchClient;

impl FetchClient {
    /// Create a new fetch client
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Fetch bytes from a URL (WASM)
    /// Works in both main thread (window) and Web Worker (self) contexts
    #[cfg(target_arch = "wasm32")]
    pub async fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>, WebSysError> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        // Use global fetch which works in both Window and Worker contexts
        let global = js_sys::global();
        let fetch_fn = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("fetch"))
            .map_err(|_| WebSysError::NoWindow)?;
        let fetch_fn: js_sys::Function = fetch_fn.dyn_into().map_err(|_| WebSysError::NoWindow)?;

        let promise = fetch_fn
            .call1(
                &wasm_bindgen::JsValue::UNDEFINED,
                &wasm_bindgen::JsValue::from_str(url),
            )
            .map_err(|_| WebSysError::FetchFailed)?;

        let response = JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|_| WebSysError::FetchFailed)?;

        let response: web_sys::Response =
            response.dyn_into().map_err(|_| WebSysError::FetchFailed)?;

        let array_buffer = JsFuture::from(
            response
                .array_buffer()
                .map_err(|_| WebSysError::FetchFailed)?,
        )
        .await
        .map_err(|_| WebSysError::FetchFailed)?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        Ok(uint8_array.to_vec())
    }

    /// Fetch bytes from a URL (native fallback - returns error)
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::unused_async)] // Must be async for API compatibility with WASM target
    pub async fn fetch_bytes(&self, _url: &str) -> Result<Vec<u8>, WebSysError> {
        Err(WebSysError::NotInBrowser)
    }
}

// ============================================================================
// Blob URL Generation (replaces web_sys::Blob, web_sys::Url)
// ============================================================================

/// Generated blob URL creator
///
/// Replaces hand-written:
/// ```rust,ignore
/// let options = web_sys::BlobPropertyBag::new();
/// options.set_type("application/javascript");
/// let blob = web_sys::Blob::new_with_blob_sequence_and_options(&parts, &options)?;
/// web_sys::Url::create_object_url_with_blob(&blob)?
/// ```
#[derive(Debug, Clone)]
pub struct BlobUrl;

impl BlobUrl {
    /// Create a blob URL from JavaScript code
    #[cfg(target_arch = "wasm32")]
    pub fn from_js_code(code: &str) -> Result<String, WebSysError> {
        use wasm_bindgen::JsValue;

        let options = web_sys::BlobPropertyBag::new();
        options.set_type("application/javascript");

        let js_string = JsValue::from_str(code);
        let blob_parts = js_sys::Array::new();
        blob_parts.push(&js_string);

        let blob = web_sys::Blob::new_with_blob_sequence_and_options(&blob_parts, &options)
            .map_err(|_| WebSysError::BlobCreationFailed)?;

        web_sys::Url::create_object_url_with_blob(&blob).map_err(|_| WebSysError::UrlCreationFailed)
    }

    /// Revoke a blob URL
    #[cfg(target_arch = "wasm32")]
    pub fn revoke(url: &str) -> Result<(), WebSysError> {
        web_sys::Url::revoke_object_url(url).map_err(|_| WebSysError::UrlRevokeFailed)
    }

    /// Native fallback
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_js_code(_code: &str) -> Result<String, WebSysError> {
        Err(WebSysError::NotInBrowser)
    }

    /// Native fallback
    #[cfg(not(target_arch = "wasm32"))]
    pub fn revoke(_url: &str) -> Result<(), WebSysError> {
        Ok(())
    }
}

// ============================================================================
// Base URL (replaces web_sys::window().location().href())
// ============================================================================

/// Get the base URL of the current page
#[cfg(target_arch = "wasm32")]
#[must_use]
pub fn get_base_url() -> Option<String> {
    web_sys::window()
        .and_then(|w| w.location().href().ok())
        .and_then(|href| {
            // Strip filename to get directory
            href.rsplit_once('/').map(|(base, _)| format!("{}/", base))
        })
}

/// Native fallback
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn get_base_url() -> Option<String> {
    Some("http://localhost/".to_string())
}

// ============================================================================
// Web Worker Creation (replaces web_sys::Worker)
// ============================================================================

/// Generated web worker handle
#[cfg(target_arch = "wasm32")]
pub struct GeneratedWorker {
    inner: web_sys::Worker,
    _on_message: wasm_bindgen::closure::Closure<dyn Fn(web_sys::MessageEvent)>,
}

#[cfg(target_arch = "wasm32")]
impl GeneratedWorker {
    /// Create a new worker from JavaScript code
    pub fn from_code<F>(code: &str, on_message: F) -> Result<Self, WebSysError>
    where
        F: Fn(web_sys::MessageEvent) + 'static,
    {
        use wasm_bindgen::closure::Closure;
        use wasm_bindgen::JsCast;

        let worker_url = BlobUrl::from_js_code(code)?;

        let worker_options = web_sys::WorkerOptions::new();
        worker_options.set_type(web_sys::WorkerType::Module);

        let worker = web_sys::Worker::new_with_options(&worker_url, &worker_options)
            .map_err(|_| WebSysError::WorkerCreationFailed)?;

        // Revoke the blob URL after worker is created
        let _ = BlobUrl::revoke(&worker_url);

        // Set up message handler
        let on_message_closure =
            Closure::wrap(Box::new(on_message) as Box<dyn Fn(web_sys::MessageEvent)>);

        worker.set_onmessage(Some(on_message_closure.as_ref().unchecked_ref()));

        Ok(Self {
            inner: worker,
            _on_message: on_message_closure,
        })
    }

    /// Post a message to the worker
    pub fn post_message(&self, message: &wasm_bindgen::JsValue) -> Result<(), WebSysError> {
        self.inner
            .post_message(message)
            .map_err(|_| WebSysError::PostMessageFailed)
    }

    /// Terminate the worker
    pub fn terminate(&self) {
        self.inner.terminate();
    }
}

#[cfg(target_arch = "wasm32")]
impl fmt::Debug for GeneratedWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeneratedWorker").finish()
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors from web_sys operations
#[derive(Debug, Clone)]
pub enum WebSysError {
    /// No window object available
    NoWindow,
    /// Event creation failed
    EventCreationFailed,
    /// Event dispatch failed
    DispatchFailed,
    /// Fetch operation failed
    FetchFailed,
    /// Not running in browser
    NotInBrowser,
    /// Blob creation failed
    BlobCreationFailed,
    /// URL creation failed
    UrlCreationFailed,
    /// URL revoke failed
    UrlRevokeFailed,
    /// Worker creation failed
    WorkerCreationFailed,
    /// Post message failed
    PostMessageFailed,
}

impl fmt::Display for WebSysError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoWindow => write!(f, "No window object available"),
            Self::EventCreationFailed => write!(f, "Failed to create custom event"),
            Self::DispatchFailed => write!(f, "Failed to dispatch event"),
            Self::FetchFailed => write!(f, "Fetch operation failed"),
            Self::NotInBrowser => write!(f, "Not running in browser environment"),
            Self::BlobCreationFailed => write!(f, "Failed to create blob"),
            Self::UrlCreationFailed => write!(f, "Failed to create object URL"),
            Self::UrlRevokeFailed => write!(f, "Failed to revoke object URL"),
            Self::WorkerCreationFailed => write!(f, "Failed to create web worker"),
            Self::PostMessageFailed => write!(f, "Failed to post message to worker"),
        }
    }
}

impl std::error::Error for WebSysError {}

// ============================================================================
// Code Generation Metadata
// ============================================================================

/// Marker trait for generated web_sys code
///
/// All generated web_sys code implements this trait for traceability
pub trait GeneratedWebSys {
    /// Source brick that generated this code
    fn source_brick() -> &'static str;

    /// Generation timestamp
    fn generated_at() -> &'static str;
}

/// Metadata about generated code
#[derive(Debug, Clone)]
pub struct GenerationMetadata {
    /// Source specification
    pub spec: &'static str,
    /// Ticket reference
    pub ticket: &'static str,
    /// Generation method
    pub method: &'static str,
}

/// Standard generation metadata for this module
pub const GENERATION_METADATA: GenerationMetadata = GenerationMetadata {
    spec: "PROBAR-SPEC-009-P7",
    ticket: "PROBAR-WEBSYS-001",
    method: "probar::brick::web_sys_gen",
};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_timing_native() {
        let t1 = PerformanceTiming::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = PerformanceTiming::now();

        assert!(t2 > t1);
    }

    #[test]
    fn test_performance_measure() {
        let (result, duration) = PerformanceTiming::measure(|| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= 4.0); // Allow some slack
    }

    #[test]
    fn test_custom_event_dispatcher() {
        let dispatcher = CustomEventDispatcher::new("test-event");
        // Native fallback returns Ok
        assert!(dispatcher.dispatch().is_ok());
    }

    #[test]
    fn test_event_detail_variants() {
        let string = EventDetail::string("hello");
        assert!(matches!(string, EventDetail::String(_)));

        let number = EventDetail::number(42.0);
        assert!(matches!(number, EventDetail::Number(_)));

        let json = EventDetail::json(&vec![1, 2, 3]);
        assert!(matches!(json, EventDetail::Json(_)));
    }

    #[test]
    fn test_fetch_client_native_fallback() {
        let client = FetchClient::new();
        // Can't actually test async in sync test, but verify it compiles
        let _ = client;
    }

    #[test]
    fn test_blob_url_native_fallback() {
        let result = BlobUrl::from_js_code("console.log('test')");
        assert!(matches!(result, Err(WebSysError::NotInBrowser)));

        // Revoke should succeed (no-op)
        assert!(BlobUrl::revoke("blob:test").is_ok());
    }

    #[test]
    fn test_get_base_url_native() {
        let url = get_base_url();
        assert!(url.is_some());
        assert!(url.unwrap().starts_with("http"));
    }

    #[test]
    fn test_web_sys_error_display() {
        let err = WebSysError::NoWindow;
        assert_eq!(format!("{}", err), "No window object available");
    }

    #[test]
    fn test_generation_metadata() {
        assert_eq!(GENERATION_METADATA.spec, "PROBAR-SPEC-009-P7");
        assert_eq!(GENERATION_METADATA.ticket, "PROBAR-WEBSYS-001");
    }

    // ========================================================================
    // Additional tests for 95%+ coverage
    // ========================================================================

    #[test]
    fn test_web_sys_error_all_variants_display() {
        // Test all WebSysError variants for Display implementation
        let errors = [
            (WebSysError::NoWindow, "No window object available"),
            (
                WebSysError::EventCreationFailed,
                "Failed to create custom event",
            ),
            (WebSysError::DispatchFailed, "Failed to dispatch event"),
            (WebSysError::FetchFailed, "Fetch operation failed"),
            (
                WebSysError::NotInBrowser,
                "Not running in browser environment",
            ),
            (WebSysError::BlobCreationFailed, "Failed to create blob"),
            (
                WebSysError::UrlCreationFailed,
                "Failed to create object URL",
            ),
            (WebSysError::UrlRevokeFailed, "Failed to revoke object URL"),
            (
                WebSysError::WorkerCreationFailed,
                "Failed to create web worker",
            ),
            (
                WebSysError::PostMessageFailed,
                "Failed to post message to worker",
            ),
        ];

        for (error, expected_msg) in errors {
            assert_eq!(format!("{}", error), expected_msg);
        }
    }

    #[test]
    fn test_web_sys_error_debug() {
        let err = WebSysError::NoWindow;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("NoWindow"));
    }

    #[test]
    fn test_web_sys_error_clone() {
        let err = WebSysError::FetchFailed;
        let cloned = err;
        assert!(matches!(cloned, WebSysError::FetchFailed));
    }

    #[test]
    fn test_web_sys_error_std_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(WebSysError::NoWindow);
        // Verify it can be used as a trait object
        let _ = err.to_string();
    }

    #[test]
    fn test_dispatch_with_detail_native_fallback() {
        let dispatcher = CustomEventDispatcher::new("test-event");

        // Test all EventDetail variants through dispatch_with_detail
        assert!(dispatcher.dispatch_with_detail(EventDetail::None).is_ok());
        assert!(dispatcher
            .dispatch_with_detail(EventDetail::String("hello".to_string()))
            .is_ok());
        assert!(dispatcher
            .dispatch_with_detail(EventDetail::Number(42.0))
            .is_ok());
        assert!(dispatcher
            .dispatch_with_detail(EventDetail::Bool(true))
            .is_ok());
        assert!(dispatcher
            .dispatch_with_detail(EventDetail::Json(r#"{"key":"value"}"#.to_string()))
            .is_ok());
    }

    #[test]
    fn test_event_detail_none() {
        let detail = EventDetail::None;
        assert!(matches!(detail, EventDetail::None));
    }

    #[test]
    fn test_event_detail_bool() {
        let detail_true = EventDetail::Bool(true);
        let detail_false = EventDetail::Bool(false);
        assert!(matches!(detail_true, EventDetail::Bool(true)));
        assert!(matches!(detail_false, EventDetail::Bool(false)));
    }

    #[test]
    fn test_event_detail_debug() {
        let detail = EventDetail::String("test".to_string());
        let debug_str = format!("{:?}", detail);
        assert!(debug_str.contains("String"));
    }

    #[test]
    fn test_event_detail_clone() {
        let detail = EventDetail::Number(42.0);
        let cloned = detail;
        assert!(matches!(cloned, EventDetail::Number(n) if (n - 42.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_event_detail_json_serialization_failure() {
        // Test the error path when serialization fails
        // Create a type that always fails to serialize
        struct FailsToSerialize;

        impl serde::Serialize for FailsToSerialize {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(serde::ser::Error::custom("intentional failure"))
            }
        }

        let json = EventDetail::json(&FailsToSerialize);
        // Should return None when serialization fails
        assert!(matches!(json, EventDetail::None));

        // Also verify the happy path still works
        let json_ok = EventDetail::json(&"simple string");
        assert!(matches!(json_ok, EventDetail::Json(_)));
    }

    #[tokio::test]
    async fn test_fetch_bytes_native_fallback() {
        // Test the async fetch_bytes method in native mode
        let client = FetchClient::new();

        let result = client.fetch_bytes("http://example.com").await;

        // Native fallback should return NotInBrowser error
        assert!(matches!(result, Err(WebSysError::NotInBrowser)));
    }

    #[test]
    fn test_fetch_client_default() {
        let client = FetchClient;
        let debug_str = format!("{:?}", client);
        assert!(debug_str.contains("FetchClient"));
    }

    #[test]
    fn test_fetch_client_clone() {
        let client = FetchClient::new();
        let cloned = client;
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_fetch_result_struct() {
        let result = FetchResult {
            status: 200,
            body: vec![1, 2, 3, 4],
        };
        assert_eq!(result.status, 200);
        assert_eq!(result.body.len(), 4);

        // Test Debug
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("FetchResult"));
        assert!(debug_str.contains("200"));
    }

    #[test]
    fn test_custom_event_dispatcher_debug() {
        let dispatcher = CustomEventDispatcher::new("my-event");
        let debug_str = format!("{:?}", dispatcher);
        assert!(debug_str.contains("CustomEventDispatcher"));
    }

    #[test]
    fn test_custom_event_dispatcher_clone() {
        let dispatcher = CustomEventDispatcher::new("clone-test");
        let cloned = dispatcher;
        assert!(cloned.dispatch().is_ok());
    }

    #[test]
    fn test_performance_timing_debug() {
        let timing = PerformanceTiming;
        let debug_str = format!("{:?}", timing);
        assert!(debug_str.contains("PerformanceTiming"));
    }

    #[test]
    fn test_performance_timing_clone_copy() {
        let timing1 = PerformanceTiming;
        let timing2 = timing1; // Copy
        let timing3 = timing1; // Clone
        let _ = format!("{:?}", timing2);
        let _ = format!("{:?}", timing3);
    }

    #[test]
    fn test_blob_url_debug() {
        let blob = BlobUrl;
        let debug_str = format!("{:?}", blob);
        assert!(debug_str.contains("BlobUrl"));
    }

    #[test]
    fn test_blob_url_clone() {
        let blob1 = BlobUrl;
        let blob2 = blob1;
        let _ = format!("{:?}", blob2);
    }

    #[test]
    fn test_generation_metadata_debug() {
        let debug_str = format!("{:?}", GENERATION_METADATA);
        assert!(debug_str.contains("GenerationMetadata"));
        assert!(debug_str.contains("PROBAR-SPEC-009-P7"));
    }

    #[test]
    fn test_generation_metadata_clone() {
        let cloned = GENERATION_METADATA.clone();
        assert_eq!(cloned.spec, GENERATION_METADATA.spec);
        assert_eq!(cloned.ticket, GENERATION_METADATA.ticket);
        assert_eq!(cloned.method, GENERATION_METADATA.method);
    }

    #[test]
    fn test_generation_metadata_method_field() {
        assert_eq!(GENERATION_METADATA.method, "probar::brick::web_sys_gen");
    }

    #[test]
    fn test_get_base_url_native_format() {
        let url = get_base_url().expect("should return Some in native mode");
        assert_eq!(url, "http://localhost/");
    }

    #[test]
    fn test_performance_timing_now_returns_positive() {
        let now = PerformanceTiming::now();
        assert!(now > 0.0, "Timestamp should be positive");
    }

    #[test]
    fn test_performance_timing_monotonic() {
        let t1 = PerformanceTiming::now();
        let t2 = PerformanceTiming::now();
        let t3 = PerformanceTiming::now();
        assert!(t2 >= t1);
        assert!(t3 >= t2);
    }

    #[test]
    fn test_event_detail_string_with_into() {
        // Test that Into<String> works with various types
        let s1 = EventDetail::string("literal str");
        let s2 = EventDetail::string(String::from("String type"));

        match s1 {
            EventDetail::String(s) => assert_eq!(s, "literal str"),
            _ => panic!("Expected String variant"),
        }

        match s2 {
            EventDetail::String(s) => assert_eq!(s, "String type"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_event_detail_number_special_values() {
        // Test special floating point values
        let nan = EventDetail::number(f64::NAN);
        let inf = EventDetail::number(f64::INFINITY);
        let neg_inf = EventDetail::number(f64::NEG_INFINITY);
        let zero = EventDetail::number(0.0);
        let neg_zero = EventDetail::number(-0.0);

        assert!(matches!(nan, EventDetail::Number(_)));
        assert!(matches!(inf, EventDetail::Number(_)));
        assert!(matches!(neg_inf, EventDetail::Number(_)));
        assert!(matches!(zero, EventDetail::Number(_)));
        assert!(matches!(neg_zero, EventDetail::Number(_)));
    }

    #[test]
    fn test_event_detail_json_complex_structures() {
        use std::collections::HashMap;

        // Test with nested HashMap
        let mut map: HashMap<&str, Vec<i32>> = HashMap::new();
        map.insert("numbers", vec![1, 2, 3]);

        let json = EventDetail::json(&map);
        match json {
            EventDetail::Json(s) => {
                assert!(s.contains("numbers"));
                assert!(s.contains("[1,2,3]"));
            }
            _ => panic!("Expected Json variant"),
        }
    }

    #[test]
    fn test_custom_event_dispatcher_new_with_string() {
        let dispatcher1 = CustomEventDispatcher::new("event-name");
        let dispatcher2 = CustomEventDispatcher::new(String::from("event-name-string"));
        assert!(dispatcher1.dispatch().is_ok());
        assert!(dispatcher2.dispatch().is_ok());
    }

    #[test]
    fn test_blob_url_revoke_empty_string() {
        // Revoke with empty string should still succeed in native fallback
        assert!(BlobUrl::revoke("").is_ok());
    }

    #[test]
    fn test_blob_url_from_js_code_empty() {
        // Empty JS code should still return NotInBrowser in native
        let result = BlobUrl::from_js_code("");
        assert!(matches!(result, Err(WebSysError::NotInBrowser)));
    }

    #[test]
    fn test_performance_measure_with_panic_safe() {
        // Test measure with a quick operation
        let (result, duration) = PerformanceTiming::measure(|| {
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            sum
        });

        assert!(result > 0);
        assert!(duration >= 0.0);
    }

    #[test]
    fn test_fetch_result_empty_body() {
        let result = FetchResult {
            status: 204,
            body: vec![],
        };
        assert_eq!(result.status, 204);
        assert!(result.body.is_empty());
    }

    #[test]
    fn test_fetch_result_large_body() {
        let result = FetchResult {
            status: 200,
            body: vec![0u8; 10000],
        };
        assert_eq!(result.body.len(), 10000);
    }

    #[test]
    fn test_web_sys_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<WebSysError>();
        assert_sync::<WebSysError>();
    }

    /// Test implementing GeneratedWebSys trait
    struct TestGeneratedCode;

    impl GeneratedWebSys for TestGeneratedCode {
        fn source_brick() -> &'static str {
            "test-brick"
        }

        fn generated_at() -> &'static str {
            "2024-01-01T00:00:00Z"
        }
    }

    #[test]
    fn test_generated_websys_trait() {
        assert_eq!(TestGeneratedCode::source_brick(), "test-brick");
        assert_eq!(TestGeneratedCode::generated_at(), "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_all_event_detail_variants_in_match() {
        let variants = vec![
            EventDetail::None,
            EventDetail::String("test".to_string()),
            EventDetail::Number(1.5),
            EventDetail::Bool(false),
            EventDetail::Json("{}".to_string()),
        ];

        for variant in variants {
            let _ = match &variant {
                EventDetail::None => "none",
                EventDetail::String(_) => "string",
                EventDetail::Number(_) => "number",
                EventDetail::Bool(_) => "bool",
                EventDetail::Json(_) => "json",
            };
        }
    }
}
