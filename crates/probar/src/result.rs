//! Result and error types for Probar.

use thiserror::Error;

/// Result type for Probar operations
pub type ProbarResult<T> = Result<T, ProbarError>;

/// Errors that can occur in Probar
#[derive(Debug, Error)]
pub enum ProbarError {
    /// Browser executable not found
    #[error("Browser not found. Install Chromium or set CHROMIUM_PATH")]
    BrowserNotFound,

    /// Browser launch error
    #[error("Failed to launch browser: {message}")]
    BrowserLaunchError {
        /// Error message
        message: String,
    },

    /// Connection to browser failed
    #[error("Failed to connect to browser: {message}")]
    ConnectionFailed {
        /// Error message
        message: String,
    },

    /// Page error
    #[error("Page error: {message}")]
    PageError {
        /// Error message
        message: String,
    },

    /// Navigation error
    #[error("Navigation to {url} failed: {message}")]
    NavigationError {
        /// URL that failed
        url: String,
        /// Error message
        message: String,
    },

    /// Operation timed out
    #[error("Operation timed out after {ms}ms")]
    Timeout {
        /// Timeout in milliseconds
        ms: u64,
    },

    /// Timeout with message
    #[error("Timeout: {message}")]
    TimeoutError {
        /// Error message
        message: String,
    },

    /// WASM evaluation error
    #[error("WASM evaluation failed: {message}")]
    WasmError {
        /// Error message
        message: String,
    },

    /// Input simulation error
    #[error("Input simulation failed: {message}")]
    InputError {
        /// Error message
        message: String,
    },

    /// Screenshot error
    #[error("Screenshot failed: {message}")]
    ScreenshotError {
        /// Error message
        message: String,
    },

    /// Element not found error
    #[error("Element not found: {selector} - {message}")]
    ElementNotFound {
        /// CSS selector
        selector: String,
        /// Error message
        message: String,
    },

    /// Assertion failed
    #[error("Assertion failed: {message}")]
    AssertionFailed {
        /// Error message
        message: String,
    },

    /// Assertion error (from `expect()`)
    #[error("Assertion error: {message}")]
    AssertionError {
        /// Error message
        message: String,
    },

    /// Snapshot mismatch
    #[error("Snapshot mismatch: {name} differs by {difference:.2}%")]
    SnapshotMismatch {
        /// Snapshot name
        name: String,
        /// Difference percentage
        difference: f64,
    },

    /// Page navigation error (legacy)
    #[error("Navigation failed: {url}")]
    NavigationFailed {
        /// URL that failed
        url: String,
    },

    /// Image comparison error
    #[error("Image comparison failed: {message}")]
    ImageComparisonError {
        /// Error message
        message: String,
    },

    /// Image processing error (resizing, encoding, etc.)
    #[error("Image processing failed: {message}")]
    ImageProcessing {
        /// Error message
        message: String,
    },

    /// Invalid state error (operation called in wrong state)
    #[error("Invalid state: {message}")]
    InvalidState {
        /// Error message
        message: String,
    },

    /// Video recording error
    #[error("Video recording failed: {message}")]
    VideoRecording {
        /// Error message
        message: String,
    },

    /// Fixture error (setup/teardown failed)
    #[error("Fixture error: {message}")]
    FixtureError {
        /// Error message
        message: String,
    },

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// TUI testing error (Feature 21 - EDD Compliance)
    #[error("TUI error: {message}")]
    TuiError {
        /// Error message
        message: String,
    },

    /// Snapshot serialization error
    #[error("Snapshot serialization error: {message}")]
    SnapshotSerializationError {
        /// Error message
        message: String,
    },

    /// HTML generation error (Feature E - Zero-JavaScript Policy)
    #[error("HTML generation error: {0}")]
    HtmlGeneration(String),

    /// CSS generation error (Feature E - Zero-JavaScript Policy)
    #[error("CSS generation error: {0}")]
    CssGeneration(String),

    /// JavaScript generation error (Feature E - Zero-JavaScript Policy)
    #[error("JavaScript generation error: {0}")]
    JsGeneration(String),

    /// Web validation error (Feature E - Zero-JavaScript Policy)
    #[error("Web validation error: {0}")]
    WebValidation(String),

    /// Pixel coverage error (Feature A - Pixel-Level GUI Coverage)
    #[error("Pixel coverage error: {0}")]
    PixelCoverage(String),

    /// Performance tracing error (Feature C - Renacer Integration)
    #[error("Performance tracing error: {0}")]
    PerformanceTracing(String),

    /// WASM runner error (Feature D - WASM Runner)
    #[error("WASM runner error: {0}")]
    WasmRunner(String),

    /// Serialization error (Issue #9 - Renacer Integration)
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message
        message: String,
    },

    /// AV sync verification error
    #[error("AV sync error: {message}")]
    AvSyncError {
        /// Error message
        message: String,
    },

    /// ffmpeg execution error
    #[error("ffmpeg error: {message}")]
    FfmpegError {
        /// Error message
        message: String,
    },
}
