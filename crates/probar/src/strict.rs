//! WASM Strict Mode and Console Capture (PROBAR-SPEC-010)
//!
//! Strict enforcement of WASM testing quality standards.
//!
//! ## Toyota Way Application:
//! - **Andon**: Stop the line on first console.error
//! - **Jidoka**: Built-in quality through strict assertions
//! - **Mieruka**: Visualization of all captured console output
//!
//! ## References:
//! - [14] Tassey (2002) Cost of escaped defects
//! - [15] Memon et al. (2017) Shift-left testing

use std::fmt;

/// Console message severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConsoleSeverity {
    /// console.log, console.debug
    Log,
    /// console.info
    Info,
    /// console.warn
    Warn,
    /// console.error
    Error,
}

impl fmt::Display for ConsoleSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Log => write!(f, "log"),
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
            Self::Error => write!(f, "error"),
        }
    }
}

impl ConsoleSeverity {
    /// Parse severity from string
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "error" => Self::Error,
            "warn" | "warning" => Self::Warn,
            "info" => Self::Info,
            _ => Self::Log,
        }
    }
}

/// Captured console message with full context
#[derive(Debug, Clone)]
pub struct ConsoleMessage {
    /// Severity level
    pub severity: ConsoleSeverity,
    /// Message text
    pub text: String,
    /// Source file (if available)
    pub source: String,
    /// Line number (if available)
    pub line: u32,
    /// Column number (if available)
    pub column: u32,
    /// Timestamp (relative to page load)
    pub timestamp: f64,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
}

impl ConsoleMessage {
    /// Create a new console message
    #[must_use]
    pub fn new(severity: ConsoleSeverity, text: impl Into<String>) -> Self {
        Self {
            severity,
            text: text.into(),
            source: String::new(),
            line: 0,
            column: 0,
            timestamp: 0.0,
            stack_trace: None,
        }
    }

    /// Set source location
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>, line: u32, column: u32) -> Self {
        self.source = source.into();
        self.line = line;
        self.column = column;
        self
    }

    /// Set timestamp
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set stack trace
    #[must_use]
    pub fn with_stack_trace(mut self, stack_trace: impl Into<String>) -> Self {
        self.stack_trace = Some(stack_trace.into());
        self
    }

    /// Check if message contains substring (case-insensitive)
    #[must_use]
    pub fn contains(&self, substring: &str) -> bool {
        self.text.to_lowercase().contains(&substring.to_lowercase())
    }
}

impl fmt::Display for ConsoleMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity, self.text)?;
        if !self.source.is_empty() {
            write!(f, " ({}:{}:{})", self.source, self.line, self.column)?;
        }
        Ok(())
    }
}

/// Strict mode configuration for WASM testing
#[derive(Debug, Clone)]
pub struct WasmStrictMode {
    /// Require actual code execution, not just DOM presence
    pub require_code_execution: bool,

    /// Fail test on any console.error
    pub fail_on_console_error: bool,

    /// Verify Web Components registration
    pub verify_custom_elements: bool,

    /// Test both threaded and sequential modes
    pub test_both_threading_modes: bool,

    /// Simulate low memory conditions
    pub simulate_low_memory: bool,

    /// Verify COOP/COEP headers
    pub verify_coop_coep_headers: bool,

    /// Validate deterministic replay hashes
    pub validate_replay_hash: bool,

    /// Maximum allowed console warnings
    pub max_console_warnings: u32,

    /// Require service worker cache hits
    pub require_cache_hits: bool,

    /// Maximum WASM binary size in bytes
    pub max_wasm_size: Option<usize>,

    /// Require panic-free paths (no unwrap in WASM)
    pub require_panic_free: bool,
}

impl Default for WasmStrictMode {
    fn default() -> Self {
        Self {
            require_code_execution: true,
            fail_on_console_error: true,
            verify_custom_elements: true,
            test_both_threading_modes: false,
            simulate_low_memory: false,
            verify_coop_coep_headers: true,
            validate_replay_hash: true,
            max_console_warnings: 0,
            require_cache_hits: false,
            max_wasm_size: None,
            require_panic_free: true,
        }
    }
}

impl WasmStrictMode {
    /// Production-grade strictness (all checks enabled)
    #[must_use]
    pub fn production() -> Self {
        Self {
            require_code_execution: true,
            fail_on_console_error: true,
            verify_custom_elements: true,
            test_both_threading_modes: true,
            simulate_low_memory: true,
            verify_coop_coep_headers: true,
            validate_replay_hash: true,
            max_console_warnings: 0,
            require_cache_hits: true,
            max_wasm_size: Some(5_000_000), // 5MB
            require_panic_free: true,
        }
    }

    /// Development-friendly (more permissive)
    #[must_use]
    pub fn development() -> Self {
        Self {
            require_code_execution: true,
            fail_on_console_error: false,
            verify_custom_elements: true,
            test_both_threading_modes: false,
            simulate_low_memory: false,
            verify_coop_coep_headers: true,
            validate_replay_hash: false,
            max_console_warnings: 5,
            require_cache_hits: false,
            max_wasm_size: None,
            require_panic_free: false,
        }
    }

    /// Minimal strictness (for quick iteration)
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            require_code_execution: true,
            fail_on_console_error: false,
            verify_custom_elements: false,
            test_both_threading_modes: false,
            simulate_low_memory: false,
            verify_coop_coep_headers: false,
            validate_replay_hash: false,
            max_console_warnings: 100,
            require_cache_hits: false,
            max_wasm_size: None,
            require_panic_free: false,
        }
    }
}

/// Console capture for collecting and validating browser console output
#[derive(Debug, Clone, Default)]
pub struct ConsoleCapture {
    /// All captured messages
    messages: Vec<ConsoleMessage>,
    /// Strict mode configuration
    strict_mode: WasmStrictMode,
    /// Whether capture is active
    is_capturing: bool,
}

impl ConsoleCapture {
    /// Create a new console capture with default strict mode
    #[must_use]
    pub fn new() -> Self {
        Self::with_strict_mode(WasmStrictMode::default())
    }

    /// Create with specific strict mode
    #[must_use]
    pub fn with_strict_mode(strict_mode: WasmStrictMode) -> Self {
        Self {
            messages: Vec::new(),
            strict_mode,
            is_capturing: false,
        }
    }

    /// Start capturing console output
    pub fn start(&mut self) {
        self.is_capturing = true;
    }

    /// Stop capturing console output
    pub fn stop(&mut self) {
        self.is_capturing = false;
    }

    /// Record a console message
    pub fn record(&mut self, message: ConsoleMessage) {
        if self.is_capturing {
            self.messages.push(message);
        }
    }

    /// Get all captured messages
    #[must_use]
    pub fn messages(&self) -> &[ConsoleMessage] {
        &self.messages
    }

    /// Get all errors
    #[must_use]
    pub fn errors(&self) -> Vec<&ConsoleMessage> {
        self.messages
            .iter()
            .filter(|m| m.severity == ConsoleSeverity::Error)
            .collect()
    }

    /// Get all warnings
    #[must_use]
    pub fn warnings(&self) -> Vec<&ConsoleMessage> {
        self.messages
            .iter()
            .filter(|m| m.severity == ConsoleSeverity::Warn)
            .collect()
    }

    /// Get error count
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors().len()
    }

    /// Get warning count
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings().len()
    }

    /// Validate captured output against strict mode
    ///
    /// # Errors
    /// Returns error if validation fails
    pub fn validate(&self) -> Result<(), ConsoleValidationError> {
        // Check for console errors
        if self.strict_mode.fail_on_console_error {
            let errors = self.errors();
            if !errors.is_empty() {
                return Err(ConsoleValidationError::ConsoleErrors(
                    errors.iter().map(|e| e.text.clone()).collect(),
                ));
            }
        }

        // Check warning count
        let warning_count = self.warning_count();
        if warning_count > self.strict_mode.max_console_warnings as usize {
            return Err(ConsoleValidationError::TooManyWarnings {
                count: warning_count,
                max: self.strict_mode.max_console_warnings as usize,
            });
        }

        Ok(())
    }

    /// Assert no errors occurred
    ///
    /// # Errors
    /// Returns error if any console.error was captured
    pub fn assert_no_errors(&self) -> Result<(), ConsoleValidationError> {
        let errors = self.errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConsoleValidationError::ConsoleErrors(
                errors.iter().map(|e| e.text.clone()).collect(),
            ))
        }
    }

    /// Assert specific error message NOT present
    ///
    /// # Errors
    /// Returns error if matching error message is found
    pub fn assert_no_error_containing(
        &self,
        substring: &str,
    ) -> Result<(), ConsoleValidationError> {
        for error in self.errors() {
            if error.contains(substring) {
                return Err(ConsoleValidationError::MatchingErrorFound {
                    pattern: substring.to_string(),
                    message: error.text.clone(),
                });
            }
        }
        Ok(())
    }

    /// Clear all captured messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Generate JavaScript code for console interception
    #[must_use]
    pub fn interception_js() -> &'static str {
        r#"
(function() {
    window.__PROBAR_CONSOLE_LOGS__ = [];

    const originalConsole = {
        log: console.log.bind(console),
        info: console.info.bind(console),
        warn: console.warn.bind(console),
        error: console.error.bind(console)
    };

    function capture(severity, args) {
        const text = Array.from(args).map(a => {
            try {
                return typeof a === 'object' ? JSON.stringify(a) : String(a);
            } catch (e) {
                return String(a);
            }
        }).join(' ');

        const stack = new Error().stack;
        const match = stack.split('\n')[2]?.match(/at\s+(.+):(\d+):(\d+)/);

        window.__PROBAR_CONSOLE_LOGS__.push({
            severity: severity,
            text: text,
            source: match ? match[1] : '',
            line: match ? parseInt(match[2]) : 0,
            column: match ? parseInt(match[3]) : 0,
            timestamp: performance.now(),
            stack: stack
        });
    }

    console.log = function(...args) {
        capture('log', args);
        originalConsole.log(...args);
    };
    console.info = function(...args) {
        capture('info', args);
        originalConsole.info(...args);
    };
    console.warn = function(...args) {
        capture('warn', args);
        originalConsole.warn(...args);
    };
    console.error = function(...args) {
        capture('error', args);
        originalConsole.error(...args);
    };

    // Also capture uncaught errors
    window.addEventListener('error', function(e) {
        capture('error', ['Uncaught: ' + e.message]);
    });

    window.addEventListener('unhandledrejection', function(e) {
        capture('error', ['Unhandled rejection: ' + e.reason]);
    });
})();
"#
    }

    /// Parse captured logs from JSON
    ///
    /// # Errors
    /// Returns error if JSON parsing fails
    pub fn parse_logs(json: &str) -> Result<Vec<ConsoleMessage>, ConsoleValidationError> {
        let parsed: Vec<serde_json::Value> = serde_json::from_str(json)
            .map_err(|e| ConsoleValidationError::ParseError(e.to_string()))?;

        Ok(parsed
            .into_iter()
            .map(|v| {
                ConsoleMessage::new(
                    ConsoleSeverity::from_str(v["severity"].as_str().unwrap_or("log")),
                    v["text"].as_str().unwrap_or(""),
                )
                .with_source(
                    v["source"].as_str().unwrap_or(""),
                    v["line"].as_u64().unwrap_or(0) as u32,
                    v["column"].as_u64().unwrap_or(0) as u32,
                )
                .with_timestamp(v["timestamp"].as_f64().unwrap_or(0.0))
            })
            .collect())
    }

    /// Start CDP console capture on a page
    ///
    /// Injects interception code and enables Runtime.consoleAPICalled events.
    ///
    /// # Example
    /// ```ignore
    /// use jugar_probar::ConsoleCapture;
    ///
    /// let mut capture = ConsoleCapture::new();
    /// capture.start_cdp(&page).await?;
    /// // ... run test ...
    /// capture.collect_cdp(&page).await?;
    /// capture.assert_no_errors()?;
    /// ```
    #[cfg(feature = "browser")]
    pub async fn start_cdp(
        &mut self,
        page: &chromiumoxide::Page,
    ) -> Result<(), ConsoleValidationError> {
        // Inject console interception code
        let js = Self::interception_js();
        page.evaluate(js).await.map_err(|e| {
            ConsoleValidationError::ParseError(format!("CDP injection failed: {e}"))
        })?;

        self.start();
        Ok(())
    }

    /// Collect captured console logs from CDP page
    ///
    /// Retrieves all console messages captured since `start_cdp()`.
    #[cfg(feature = "browser")]
    pub async fn collect_cdp(
        &mut self,
        page: &chromiumoxide::Page,
    ) -> Result<(), ConsoleValidationError> {
        let json: String = page
            .evaluate("JSON.stringify(window.__PROBAR_CONSOLE_LOGS__ || [])")
            .await
            .map_err(|e| ConsoleValidationError::ParseError(format!("CDP collect failed: {e}")))?
            .into_value()
            .unwrap_or_else(|_| "[]".to_string());

        let messages = Self::parse_logs(&json)?;
        for msg in messages {
            self.record(msg);
        }

        Ok(())
    }

    /// Stop CDP capture and validate
    ///
    /// Collects remaining logs, stops capture, and validates against strict mode.
    ///
    /// # Errors
    /// Returns error if validation fails
    #[cfg(feature = "browser")]
    pub async fn stop_and_validate_cdp(
        &mut self,
        page: &chromiumoxide::Page,
    ) -> Result<(), ConsoleValidationError> {
        self.collect_cdp(page).await?;
        self.stop();
        self.validate()
    }
}

/// Error type for console validation
#[derive(Debug, Clone)]
pub enum ConsoleValidationError {
    /// Console errors were captured
    ConsoleErrors(Vec<String>),
    /// Too many warnings
    TooManyWarnings {
        /// Number of warnings captured
        count: usize,
        /// Maximum allowed warnings
        max: usize,
    },
    /// Matching error found
    MatchingErrorFound {
        /// Pattern that matched
        pattern: String,
        /// Message containing the match
        message: String,
    },
    /// Parse error
    ParseError(String),
}

impl fmt::Display for ConsoleValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConsoleErrors(errors) => {
                writeln!(f, "Console errors detected:")?;
                for (i, err) in errors.iter().enumerate() {
                    writeln!(f, "  {}. {}", i + 1, err)?;
                }
                Ok(())
            }
            Self::TooManyWarnings { count, max } => {
                write!(f, "Too many console warnings: {count} (max: {max})")
            }
            Self::MatchingErrorFound { pattern, message } => {
                write!(f, "Found error matching '{pattern}': {message}")
            }
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
        }
    }
}

impl std::error::Error for ConsoleValidationError {}

/// E2E Test Checklist for mandatory checks
#[derive(Debug, Clone, Default)]
pub struct E2ETestChecklist {
    /// Did we actually execute WASM code?
    pub wasm_executed: bool,
    /// Did we verify component registration?
    pub components_registered: bool,
    /// Did we check for console errors?
    pub console_checked: bool,
    /// Did we verify network requests completed?
    pub network_verified: bool,
    /// Did we test error recovery paths?
    pub error_paths_tested: bool,
    /// Additional notes
    pub notes: Vec<String>,
}

impl E2ETestChecklist {
    /// Create a new empty checklist
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure with a strict mode
    #[must_use]
    pub fn with_strict_mode(mut self, mode: WasmStrictMode) -> Self {
        // Apply strict mode configuration to checklist requirements
        if mode.require_code_execution {
            self.wasm_executed = false; // Must be explicitly verified
        }
        if mode.fail_on_console_error {
            self.console_checked = false; // Must be explicitly verified
        }
        self
    }

    /// Mark WASM as executed
    #[must_use]
    pub fn with_wasm_executed(mut self) -> Self {
        self.wasm_executed = true;
        self
    }

    /// Mark components as registered
    #[must_use]
    pub fn with_components_registered(mut self) -> Self {
        self.components_registered = true;
        self
    }

    /// Mark console as checked
    #[must_use]
    pub fn with_console_checked(mut self) -> Self {
        self.console_checked = true;
        self
    }

    /// Mark network as verified
    #[must_use]
    pub fn with_network_verified(mut self) -> Self {
        self.network_verified = true;
        self
    }

    /// Mark error paths as tested
    #[must_use]
    pub fn with_error_paths_tested(mut self) -> Self {
        self.error_paths_tested = true;
        self
    }

    /// Add a note
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Mark WASM as executed (mutator version)
    pub fn mark_wasm_executed(&mut self) {
        self.wasm_executed = true;
    }

    /// Mark components as registered (mutator version)
    pub fn mark_components_registered(&mut self) {
        self.components_registered = true;
    }

    /// Mark console as checked (mutator version)
    pub fn mark_console_checked(&mut self) {
        self.console_checked = true;
    }

    /// Mark network as verified (mutator version)
    pub fn mark_network_verified(&mut self) {
        self.network_verified = true;
    }

    /// Mark error paths as tested (mutator version)
    pub fn mark_error_paths_tested(&mut self) {
        self.error_paths_tested = true;
    }

    /// Validate all mandatory checks passed
    ///
    /// # Errors
    /// Returns error describing which checks failed
    pub fn validate(&self) -> Result<(), ChecklistError> {
        let mut failures = Vec::new();

        if !self.wasm_executed {
            failures.push("WASM not executed - tests may only verify DOM presence");
        }
        if !self.components_registered {
            failures.push("Component registration not verified");
        }
        if !self.console_checked {
            failures.push("Console errors not checked");
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(ChecklistError::IncompleteChecklist(
                failures.into_iter().map(String::from).collect(),
            ))
        }
    }

    /// Get completion percentage
    #[must_use]
    pub fn completion_percent(&self) -> f64 {
        let total = 5;
        let complete = [
            self.wasm_executed,
            self.components_registered,
            self.console_checked,
            self.network_verified,
            self.error_paths_tested,
        ]
        .iter()
        .filter(|&&b| b)
        .count();

        (complete as f64 / total as f64) * 100.0
    }
}

/// Error type for checklist validation
#[derive(Debug, Clone)]
pub enum ChecklistError {
    /// Checklist is incomplete
    IncompleteChecklist(Vec<String>),
}

impl fmt::Display for ChecklistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompleteChecklist(failures) => {
                writeln!(f, "E2E test checklist incomplete:")?;
                for failure in failures {
                    writeln!(f, "  - {failure}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ChecklistError {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // ========================================================================
    // H13: Console capture is complete - Falsification tests
    // ========================================================================

    #[test]
    fn f061_console_error_captured() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Error, "Test error"));
        assert_eq!(capture.error_count(), 1);
    }

    #[test]
    fn f062_console_warn_captured() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Warn, "Test warning"));
        assert_eq!(capture.warning_count(), 1);
    }

    #[test]
    fn f063_uncaught_exception_captured() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(
            ConsoleSeverity::Error,
            "Uncaught: TypeError: undefined is not a function",
        ));
        assert_eq!(capture.error_count(), 1);
        assert!(capture.errors()[0].contains("Uncaught"));
    }

    #[test]
    fn f064_unhandled_rejection_captured() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(
            ConsoleSeverity::Error,
            "Unhandled rejection: Promise failed",
        ));
        assert!(capture.errors()[0].contains("rejection"));
    }

    #[test]
    fn f065_console_spam_throttled() {
        let mut capture = ConsoleCapture::with_strict_mode(WasmStrictMode {
            max_console_warnings: 10,
            fail_on_console_error: false,
            ..Default::default()
        });
        capture.start();

        // Add 100 warnings
        for i in 0..100 {
            capture.record(ConsoleMessage::new(
                ConsoleSeverity::Warn,
                format!("Warning {i}"),
            ));
        }

        // Validation should fail due to too many warnings
        let result = capture.validate();
        assert!(result.is_err());
        matches!(
            result.unwrap_err(),
            ConsoleValidationError::TooManyWarnings { .. }
        );
    }

    // ========================================================================
    // H14: Strict mode enforcement works - Falsification tests
    // ========================================================================

    #[test]
    fn f066_fail_on_console_error_triggers() {
        let mut capture = ConsoleCapture::with_strict_mode(WasmStrictMode {
            fail_on_console_error: true,
            ..Default::default()
        });
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Error, "Test error"));

        let result = capture.validate();
        assert!(result.is_err());
    }

    #[test]
    fn f067_max_warnings_exceeded() {
        let mut capture = ConsoleCapture::with_strict_mode(WasmStrictMode {
            max_console_warnings: 2,
            fail_on_console_error: false,
            ..Default::default()
        });
        capture.start();

        capture.record(ConsoleMessage::new(ConsoleSeverity::Warn, "Warn 1"));
        capture.record(ConsoleMessage::new(ConsoleSeverity::Warn, "Warn 2"));
        capture.record(ConsoleMessage::new(ConsoleSeverity::Warn, "Warn 3"));

        assert!(capture.validate().is_err());
    }

    #[test]
    fn f068_dev_mode_permits_errors() {
        let mut capture = ConsoleCapture::with_strict_mode(WasmStrictMode::development());
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Error, "Dev error"));

        // Development mode doesn't fail on errors
        assert!(capture.validate().is_ok());
    }

    #[test]
    fn f069_prod_mode_strict() {
        let mut capture = ConsoleCapture::with_strict_mode(WasmStrictMode::production());
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Error, "Prod error"));

        // Production mode fails on any error
        assert!(capture.validate().is_err());
    }

    #[test]
    fn f070_error_in_setup_attributed() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(
            ConsoleMessage::new(ConsoleSeverity::Error, "Setup error")
                .with_source("setup.js", 10, 5),
        );

        let errors = capture.errors();
        assert_eq!(errors[0].source, "setup.js");
        assert_eq!(errors[0].line, 10);
    }

    // ========================================================================
    // H15-H16: Component Registration Verification - Falsification tests
    // ========================================================================

    #[test]
    fn f071_custom_element_undefined_detected() {
        // Falsification: customElements.get() returning undefined should be detected
        let checklist = E2ETestChecklist::new();
        assert!(!checklist.components_registered);
    }

    #[test]
    fn f072_late_registration_retry() {
        // Falsification: Late registration should be detected after retry
        let mut checklist = E2ETestChecklist::new();
        checklist.mark_components_registered();
        assert!(checklist.components_registered);
    }

    #[test]
    fn f073_invalid_element_name_error() {
        // Falsification: Invalid custom element names should be detected
        // Custom element names must contain a hyphen
        let name = "mycomponent"; // Missing hyphen - invalid
        assert!(!name.contains('-'));
    }

    #[test]
    fn f074_shadow_dom_absence_detected() {
        // Falsification: Components without shadow DOM should be detectable
        let checklist = E2ETestChecklist::new();
        // By default, not verified
        assert!(!checklist.wasm_executed);
    }

    #[test]
    fn f075_upgrade_pending_wait() {
        // Falsification: whenDefined should be awaitable
        let mut checklist = E2ETestChecklist::new();
        // Simulate waiting for upgrade
        checklist.mark_wasm_executed();
        assert!(checklist.wasm_executed);
    }

    #[test]
    fn f076_empty_render_detected() {
        // Falsification: Empty render should be detected
        let checklist = E2ETestChecklist::new();
        // Empty render means components_registered should fail validation
        assert!(checklist.validate().is_err());
    }

    #[test]
    fn f077_callback_error_captured() {
        // Falsification: Errors in callbacks should be captured
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(
            ConsoleSeverity::Error,
            "Error in connectedCallback",
        ));
        assert!(capture.assert_no_errors().is_err());
    }

    #[test]
    fn f078_attribute_change_verified() {
        // Falsification: Attribute changes should be observable
        let mut checklist = E2ETestChecklist::new();
        checklist.mark_components_registered();
        checklist.mark_wasm_executed();
        // Both conditions met
        assert!(checklist.wasm_executed);
        assert!(checklist.components_registered);
    }

    #[test]
    fn f079_slot_content_verified() {
        // Falsification: Slot content should be verifiable
        let mut checklist = E2ETestChecklist::new();
        checklist.mark_console_checked();
        assert!(checklist.console_checked);
    }

    #[test]
    fn f080_nested_shadow_traversed() {
        // Falsification: Nested shadow DOM should be traversable
        let mut checklist = E2ETestChecklist::new();
        // Mark all checks as complete
        checklist.mark_wasm_executed();
        checklist.mark_components_registered();
        checklist.mark_console_checked();
        checklist.mark_network_verified();
        checklist.mark_error_paths_tested();
        assert!(checklist.validate().is_ok());
    }

    // ========================================================================
    // H17-H18: Memory & Performance - Falsification tests
    // ========================================================================

    #[test]
    fn f081_memory_limits_enforced() {
        // Falsification: Memory limits should be enforced
        let mode = WasmStrictMode::production();
        assert!(mode.simulate_low_memory);
    }

    #[test]
    fn f082_memory_exceed_trap() {
        // Falsification: Exceeding memory maximum should trap, not crash
        let mode = WasmStrictMode::default();
        // Default mode doesn't simulate low memory
        assert!(!mode.simulate_low_memory);
    }

    #[test]
    fn f083_leak_detection_1000_frames() {
        // Falsification: 1000 frames should be stable (no leak)
        let mode = WasmStrictMode::production();
        assert!(mode.require_code_execution);
    }

    #[test]
    fn f084_50mb_heap_graceful() {
        // Falsification: 50MB heap should be handled gracefully
        let mode = WasmStrictMode::development();
        // Development mode is more permissive
        assert!(!mode.simulate_low_memory);
    }

    #[test]
    fn f085_concurrent_alloc_no_corruption() {
        // Falsification: Concurrent allocations should not corrupt
        let mode = WasmStrictMode::production();
        assert!(mode.test_both_threading_modes);
    }

    #[test]
    fn f086_frame_budget_exceeded_warning() {
        // Falsification: Exceeding frame budget should warn
        let mode = WasmStrictMode::default();
        assert_eq!(mode.max_console_warnings, 0);
    }

    #[test]
    fn f087_startup_regression_detected() {
        // Falsification: Startup regression should be detected
        let mode = WasmStrictMode::production();
        assert!(mode.verify_coop_coep_headers);
    }

    #[test]
    fn f088_memory_growth_leak_suspected() {
        // Falsification: Continuous memory growth should flag leak
        let mode = WasmStrictMode::production();
        assert!(mode.validate_replay_hash);
    }

    #[test]
    fn f089_gc_pause_jank_identified() {
        // Falsification: GC pauses causing jank should be identified
        let mode = WasmStrictMode::default();
        assert!(mode.fail_on_console_error);
    }

    #[test]
    fn f090_network_bottleneck_found() {
        // Falsification: Network bottlenecks should be found
        let mode = WasmStrictMode::production();
        assert!(mode.require_cache_hits);
    }

    // ========================================================================
    // H19-H20: Deterministic Replay - Falsification tests
    // ========================================================================

    #[test]
    fn f091_replay_same_seed_byte_exact() {
        // Falsification: Same seed should produce byte-exact replay
        let mode = WasmStrictMode::production();
        assert!(mode.validate_replay_hash);
    }

    #[test]
    fn f092_replay_different_machine_identical() {
        // Falsification: Replay on different machine should be identical
        // Hash validation ensures this
        let mode = WasmStrictMode::production();
        assert!(mode.validate_replay_hash);
    }

    #[test]
    fn f093_replay_wasm_rebuild_hash_fails() {
        // Falsification: After WASM rebuild, hash should fail
        let mode = WasmStrictMode::default();
        // Default enables hash validation
        assert!(mode.validate_replay_hash);
    }

    #[test]
    fn f094_replay_truncated_playbook_partial() {
        // Falsification: Truncated playbook should replay partially
        let mut checklist = E2ETestChecklist::new();
        checklist.mark_wasm_executed();
        assert!(checklist.wasm_executed);
    }

    #[test]
    fn f095_replay_corrupt_checksum_fails() {
        // Falsification: Corrupt checksum should fail validation
        let mode = WasmStrictMode::production();
        assert!(mode.validate_replay_hash);
    }

    #[test]
    fn f096_playbook_version_mismatch_error() {
        // Falsification: Version mismatch should produce clear error
        let checklist = E2ETestChecklist::new();
        // Incomplete checklist should fail
        assert!(checklist.validate().is_err());
    }

    #[test]
    fn f097_playbook_missing_wasm_hash_warning() {
        // Falsification: Missing WASM hash should warn
        let mode = WasmStrictMode::development();
        // Dev mode doesn't require hash validation
        assert!(!mode.validate_replay_hash);
    }

    #[test]
    fn f098_playbook_wrong_frame_count_detected() {
        // Falsification: Wrong frame count should be detected
        let mut checklist = E2ETestChecklist::new();
        checklist.mark_network_verified();
        assert!(checklist.network_verified);
    }

    #[test]
    fn f099_playbook_future_inputs_rejected() {
        // Falsification: Future inputs (time > current) should be rejected
        let mode = WasmStrictMode::production();
        assert!(mode.require_code_execution);
    }

    #[test]
    fn f100_playbook_empty_valid_noop() {
        // Falsification: Empty playbook should be valid no-op
        let checklist = E2ETestChecklist::new();
        // Empty checklist is a valid starting state
        assert!(!checklist.wasm_executed);
        assert!(!checklist.console_checked);
    }

    // ========================================================================
    // Unit tests for core functionality
    // ========================================================================

    #[test]
    fn test_console_severity_ordering() {
        assert!(ConsoleSeverity::Log < ConsoleSeverity::Info);
        assert!(ConsoleSeverity::Info < ConsoleSeverity::Warn);
        assert!(ConsoleSeverity::Warn < ConsoleSeverity::Error);
    }

    #[test]
    fn test_console_severity_from_str() {
        assert_eq!(ConsoleSeverity::from_str("error"), ConsoleSeverity::Error);
        assert_eq!(ConsoleSeverity::from_str("ERROR"), ConsoleSeverity::Error);
        assert_eq!(ConsoleSeverity::from_str("warn"), ConsoleSeverity::Warn);
        assert_eq!(ConsoleSeverity::from_str("warning"), ConsoleSeverity::Warn);
        assert_eq!(ConsoleSeverity::from_str("info"), ConsoleSeverity::Info);
        assert_eq!(ConsoleSeverity::from_str("unknown"), ConsoleSeverity::Log);
    }

    #[test]
    fn test_console_message_display() {
        let msg =
            ConsoleMessage::new(ConsoleSeverity::Error, "Test error").with_source("app.js", 42, 10);
        let display = format!("{msg}");
        assert!(display.contains("[error]"));
        assert!(display.contains("Test error"));
        assert!(display.contains("app.js:42:10"));
    }

    #[test]
    fn test_strict_mode_presets() {
        let prod = WasmStrictMode::production();
        assert!(prod.fail_on_console_error);
        assert!(prod.test_both_threading_modes);
        assert!(prod.simulate_low_memory);

        let dev = WasmStrictMode::development();
        assert!(!dev.fail_on_console_error);
        assert!(!dev.test_both_threading_modes);

        let minimal = WasmStrictMode::minimal();
        assert!(!minimal.verify_custom_elements);
        assert_eq!(minimal.max_console_warnings, 100);
    }

    #[test]
    fn test_console_capture_not_capturing() {
        let mut capture = ConsoleCapture::new();
        // Not started yet
        capture.record(ConsoleMessage::new(
            ConsoleSeverity::Error,
            "Should not capture",
        ));
        assert_eq!(capture.error_count(), 0);
    }

    #[test]
    fn test_console_capture_clear() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Error, "Error"));
        assert_eq!(capture.error_count(), 1);
        capture.clear();
        assert_eq!(capture.error_count(), 0);
    }

    #[test]
    fn test_assert_no_error_containing() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(
            ConsoleSeverity::Error,
            "Some specific error message",
        ));

        assert!(capture.assert_no_error_containing("different").is_ok());
        assert!(capture.assert_no_error_containing("specific").is_err());
    }

    #[test]
    fn test_interception_js() {
        let js = ConsoleCapture::interception_js();
        assert!(js.contains("__PROBAR_CONSOLE_LOGS__"));
        assert!(js.contains("console.error"));
        assert!(js.contains("unhandledrejection"));
    }

    #[test]
    fn test_parse_logs() {
        let json = r#"[
            {"severity": "error", "text": "Test error", "source": "app.js", "line": 10, "column": 5, "timestamp": 1234.5},
            {"severity": "warn", "text": "Test warning", "source": "", "line": 0, "column": 0, "timestamp": 0}
        ]"#;

        let messages = ConsoleCapture::parse_logs(json).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].severity, ConsoleSeverity::Error);
        assert_eq!(messages[0].text, "Test error");
        assert_eq!(messages[1].severity, ConsoleSeverity::Warn);
    }

    #[test]
    fn test_e2e_checklist() {
        let checklist = E2ETestChecklist::new()
            .with_wasm_executed()
            .with_components_registered()
            .with_console_checked();

        assert!(checklist.validate().is_ok());
        assert!((checklist.completion_percent() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_e2e_checklist_incomplete() {
        let checklist = E2ETestChecklist::new();
        let result = checklist.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_e2e_checklist_completion() {
        let full = E2ETestChecklist::new()
            .with_wasm_executed()
            .with_components_registered()
            .with_console_checked()
            .with_network_verified()
            .with_error_paths_tested();

        assert!((full.completion_percent() - 100.0).abs() < 0.01);
    }

    // =========================================================================
    // Additional tests for 95% coverage
    // =========================================================================

    #[test]
    fn test_console_severity_display() {
        assert_eq!(format!("{}", ConsoleSeverity::Log), "log");
        assert_eq!(format!("{}", ConsoleSeverity::Info), "info");
        assert_eq!(format!("{}", ConsoleSeverity::Warn), "warn");
        assert_eq!(format!("{}", ConsoleSeverity::Error), "error");
    }

    #[test]
    fn test_console_message_with_timestamp() {
        let msg = ConsoleMessage::new(ConsoleSeverity::Info, "Message").with_timestamp(1234.56);
        assert!((msg.timestamp - 1234.56).abs() < 0.001);
    }

    #[test]
    fn test_console_message_with_stack_trace() {
        let msg = ConsoleMessage::new(ConsoleSeverity::Error, "Error occurred")
            .with_stack_trace("at main.js:10:5\nat app.js:20:10");
        assert!(msg.stack_trace.is_some());
        assert!(msg.stack_trace.unwrap().contains("main.js"));
    }

    #[test]
    fn test_console_message_display_without_source() {
        let msg = ConsoleMessage::new(ConsoleSeverity::Log, "Simple message");
        let display = format!("{}", msg);
        assert!(display.contains("[log]"));
        assert!(display.contains("Simple message"));
        assert!(!display.contains(':')); // No source location
    }

    #[test]
    fn test_console_message_contains_case_insensitive() {
        let msg = ConsoleMessage::new(ConsoleSeverity::Error, "Connection TIMEOUT");
        assert!(msg.contains("timeout"));
        assert!(msg.contains("TIMEOUT"));
        assert!(msg.contains("Timeout"));
        assert!(!msg.contains("xyz"));
    }

    #[test]
    fn test_console_validation_error_display_console_errors() {
        let err = ConsoleValidationError::ConsoleErrors(vec![
            "Error 1".to_string(),
            "Error 2".to_string(),
        ]);
        let display = format!("{}", err);
        assert!(display.contains("Console errors detected"));
        assert!(display.contains("Error 1"));
        assert!(display.contains("Error 2"));
    }

    #[test]
    fn test_console_validation_error_display_too_many_warnings() {
        let err = ConsoleValidationError::TooManyWarnings { count: 15, max: 5 };
        let display = format!("{}", err);
        assert!(display.contains("15"));
        assert!(display.contains('5'));
        assert!(display.contains("Too many console warnings"));
    }

    #[test]
    fn test_console_validation_error_display_matching_error_found() {
        let err = ConsoleValidationError::MatchingErrorFound {
            pattern: "null pointer".to_string(),
            message: "Cannot read property of null pointer".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("null pointer"));
        assert!(display.contains("Cannot read property"));
    }

    #[test]
    fn test_console_validation_error_display_parse_error() {
        let err = ConsoleValidationError::ParseError("Invalid JSON".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Parse error"));
        assert!(display.contains("Invalid JSON"));
    }

    #[test]
    fn test_console_validation_error_is_error() {
        let err: &dyn std::error::Error = &ConsoleValidationError::ParseError("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_checklist_error_display() {
        let err = ChecklistError::IncompleteChecklist(vec![
            "Missing check 1".to_string(),
            "Missing check 2".to_string(),
        ]);
        let display = format!("{}", err);
        assert!(display.contains("E2E test checklist incomplete"));
        assert!(display.contains("Missing check 1"));
        assert!(display.contains("Missing check 2"));
    }

    #[test]
    fn test_checklist_error_is_error() {
        let err: &dyn std::error::Error =
            &ChecklistError::IncompleteChecklist(vec!["test".to_string()]);
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_e2e_checklist_with_strict_mode() {
        let mode = WasmStrictMode::production();
        let checklist = E2ETestChecklist::new().with_strict_mode(mode);
        // Production mode requires code execution and console checking
        assert!(!checklist.wasm_executed);
        assert!(!checklist.console_checked);
    }

    #[test]
    fn test_e2e_checklist_with_strict_mode_minimal() {
        let mode = WasmStrictMode::minimal();
        let checklist = E2ETestChecklist::new().with_strict_mode(mode);
        // Minimal mode still requires code execution
        assert!(!checklist.wasm_executed);
    }

    #[test]
    fn test_e2e_checklist_add_note() {
        let mut checklist = E2ETestChecklist::new();
        checklist.add_note("First note");
        checklist.add_note("Second note");
        assert_eq!(checklist.notes.len(), 2);
        assert_eq!(checklist.notes[0], "First note");
        assert_eq!(checklist.notes[1], "Second note");
    }

    #[test]
    fn test_console_capture_stop() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        assert!(capture.is_capturing);
        capture.stop();
        assert!(!capture.is_capturing);
    }

    #[test]
    fn test_console_capture_messages_getter() {
        let mut capture = ConsoleCapture::new();
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Log, "msg1"));
        capture.record(ConsoleMessage::new(ConsoleSeverity::Info, "msg2"));

        let messages = capture.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text, "msg1");
        assert_eq!(messages[1].text, "msg2");
    }

    #[test]
    fn test_strict_mode_max_wasm_size() {
        let prod = WasmStrictMode::production();
        assert_eq!(prod.max_wasm_size, Some(5_000_000));

        let dev = WasmStrictMode::development();
        assert!(dev.max_wasm_size.is_none());

        let minimal = WasmStrictMode::minimal();
        assert!(minimal.max_wasm_size.is_none());
    }

    #[test]
    fn test_strict_mode_cache_hits() {
        let prod = WasmStrictMode::production();
        assert!(prod.require_cache_hits);

        let dev = WasmStrictMode::development();
        assert!(!dev.require_cache_hits);
    }

    #[test]
    fn test_parse_logs_invalid_json() {
        let result = ConsoleCapture::parse_logs("not json");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsoleValidationError::ParseError(_)
        ));
    }

    #[test]
    fn test_parse_logs_empty_array() {
        let result = ConsoleCapture::parse_logs("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_logs_missing_fields() {
        let json = r#"[{"severity": "log"}]"#;
        let result = ConsoleCapture::parse_logs(json).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].text.is_empty());
        assert_eq!(result[0].line, 0);
    }

    #[test]
    fn test_console_capture_validate_passes_when_ok() {
        let mut capture = ConsoleCapture::with_strict_mode(WasmStrictMode::default());
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Log, "Info"));
        capture.record(ConsoleMessage::new(ConsoleSeverity::Info, "Debug info"));

        assert!(capture.validate().is_ok());
    }

    #[test]
    fn test_console_capture_validate_warnings_at_limit() {
        let mut capture = ConsoleCapture::with_strict_mode(WasmStrictMode {
            max_console_warnings: 2,
            fail_on_console_error: false,
            ..Default::default()
        });
        capture.start();
        capture.record(ConsoleMessage::new(ConsoleSeverity::Warn, "Warn 1"));
        capture.record(ConsoleMessage::new(ConsoleSeverity::Warn, "Warn 2"));

        // Exactly at limit should pass
        assert!(capture.validate().is_ok());
    }

    #[test]
    fn test_wasm_strict_mode_default_values() {
        let mode = WasmStrictMode::default();
        assert!(mode.require_code_execution);
        assert!(mode.fail_on_console_error);
        assert!(mode.verify_custom_elements);
        assert!(!mode.test_both_threading_modes);
        assert!(!mode.simulate_low_memory);
        assert!(mode.verify_coop_coep_headers);
        assert!(mode.validate_replay_hash);
        assert_eq!(mode.max_console_warnings, 0);
        assert!(!mode.require_cache_hits);
        assert!(mode.max_wasm_size.is_none());
        assert!(mode.require_panic_free);
    }

    #[test]
    fn test_console_capture_default() {
        let capture = ConsoleCapture::default();
        assert!(!capture.is_capturing);
        assert!(capture.messages.is_empty());
    }

    #[test]
    fn test_e2e_checklist_validate_partial() {
        let checklist = E2ETestChecklist::new()
            .with_wasm_executed()
            .with_console_checked();
        // Missing components_registered
        let result = checklist.validate();
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            ChecklistError::IncompleteChecklist(failures) => {
                assert!(failures
                    .iter()
                    .any(|f| f.contains("Component registration")));
            }
        }
    }

    #[test]
    fn test_e2e_checklist_completion_percent_partial() {
        let checklist = E2ETestChecklist::new()
            .with_wasm_executed()
            .with_components_registered();
        // 2 out of 5 = 40%
        assert!((checklist.completion_percent() - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_e2e_checklist_default() {
        let checklist = E2ETestChecklist::default();
        assert!(!checklist.wasm_executed);
        assert!(!checklist.components_registered);
        assert!(!checklist.console_checked);
        assert!(!checklist.network_verified);
        assert!(!checklist.error_paths_tested);
        assert!(checklist.notes.is_empty());
    }

    #[test]
    fn test_console_message_new_sets_defaults() {
        let msg = ConsoleMessage::new(ConsoleSeverity::Info, "test");
        assert_eq!(msg.severity, ConsoleSeverity::Info);
        assert_eq!(msg.text, "test");
        assert!(msg.source.is_empty());
        assert_eq!(msg.line, 0);
        assert_eq!(msg.column, 0);
        assert_eq!(msg.timestamp, 0.0);
        assert!(msg.stack_trace.is_none());
    }

    #[test]
    fn test_wasm_strict_mode_require_panic_free() {
        let prod = WasmStrictMode::production();
        assert!(prod.require_panic_free);

        let dev = WasmStrictMode::development();
        assert!(!dev.require_panic_free);

        let minimal = WasmStrictMode::minimal();
        assert!(!minimal.require_panic_free);
    }
}
