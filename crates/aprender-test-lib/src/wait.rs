//! Wait Mechanisms (PMAT-005)
//!
//! Playwright-compatible wait mechanisms for synchronization.
//!
//! ## EXTREME TDD: Tests written FIRST per spec Section 6.1
//!
//! ## Toyota Way Application
//!
//! - **Jidoka**: Automatic detection of ready state
//! - **Poka-Yoke**: Type-safe wait conditions prevent invalid waits
//! - **Muda**: Efficient polling reduces wasted CPU cycles

use crate::network::UrlPattern;
use crate::result::{ProbarError, ProbarResult};
use std::time::{Duration, Instant};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Default timeout for wait operations (30 seconds)
pub const DEFAULT_WAIT_TIMEOUT_MS: u64 = 30_000;

/// Default polling interval (50ms)
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 50;

/// Network idle threshold (500ms without requests)
pub const NETWORK_IDLE_THRESHOLD_MS: u64 = 500;

// =============================================================================
// LOAD STATE
// =============================================================================

/// Page load states (Playwright parity)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadState {
    /// Wait for the `load` event to fire
    Load,
    /// Wait for `DOMContentLoaded` event
    DomContentLoaded,
    /// Wait for network to be idle (no requests for 500ms)
    NetworkIdle,
}

impl LoadState {
    /// Get the JavaScript event name for this load state
    #[must_use]
    pub const fn event_name(&self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::DomContentLoaded => "DOMContentLoaded",
            Self::NetworkIdle => "networkidle",
        }
    }

    /// Get default timeout for this load state
    #[must_use]
    pub const fn default_timeout_ms(&self) -> u64 {
        match self {
            Self::Load => 30_000,
            Self::DomContentLoaded => 30_000,
            Self::NetworkIdle => 60_000, // Network idle can take longer
        }
    }
}

impl Default for LoadState {
    fn default() -> Self {
        Self::Load
    }
}

impl std::fmt::Display for LoadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.event_name())
    }
}

// =============================================================================
// WAIT OPTIONS
// =============================================================================

/// Options for wait operations
#[derive(Debug, Clone)]
pub struct WaitOptions {
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Polling interval in milliseconds
    pub poll_interval_ms: u64,
    /// State to wait for (for navigation)
    pub wait_until: LoadState,
}

impl Default for WaitOptions {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            wait_until: LoadState::Load,
        }
    }
}

impl WaitOptions {
    /// Create new wait options with defaults
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timeout in milliseconds
    #[must_use]
    pub const fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set polling interval in milliseconds
    #[must_use]
    pub const fn with_poll_interval(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = poll_interval_ms;
        self
    }

    /// Set load state to wait for
    #[must_use]
    pub const fn with_wait_until(mut self, state: LoadState) -> Self {
        self.wait_until = state;
        self
    }

    /// Get timeout as Duration
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    /// Get poll interval as Duration
    #[must_use]
    pub const fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_interval_ms)
    }
}

// =============================================================================
// NAVIGATION OPTIONS
// =============================================================================

/// Options for navigation wait
#[derive(Debug, Clone)]
pub struct NavigationOptions {
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Load state to wait for
    pub wait_until: LoadState,
    /// URL pattern to match (optional)
    pub url_pattern: Option<UrlPattern>,
}

impl Default for NavigationOptions {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
            wait_until: LoadState::Load,
            url_pattern: None,
        }
    }
}

impl NavigationOptions {
    /// Create new navigation options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set load state
    #[must_use]
    pub const fn with_wait_until(mut self, state: LoadState) -> Self {
        self.wait_until = state;
        self
    }

    /// Set URL pattern
    #[must_use]
    pub fn with_url(mut self, pattern: UrlPattern) -> Self {
        self.url_pattern = Some(pattern);
        self
    }
}

// =============================================================================
// PAGE EVENTS
// =============================================================================

/// Page event types (Playwright parity)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PageEvent {
    /// Page closed
    Close,
    /// Console message
    Console,
    /// Page crashed
    Crash,
    /// Dialog opened (alert, confirm, prompt)
    Dialog,
    /// Download started
    Download,
    /// File chooser opened
    FileChooser,
    /// Frame attached
    FrameAttached,
    /// Frame detached
    FrameDetached,
    /// Frame navigated
    FrameNavigated,
    /// Page loaded
    Load,
    /// DOM content loaded
    DomContentLoaded,
    /// Page error
    PageError,
    /// Popup opened
    Popup,
    /// Request made
    Request,
    /// Request failed
    RequestFailed,
    /// Request finished
    RequestFinished,
    /// Response received
    Response,
    /// WebSocket created
    WebSocket,
    /// Worker created
    Worker,
}

impl PageEvent {
    /// Get the event name string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Close => "close",
            Self::Console => "console",
            Self::Crash => "crash",
            Self::Dialog => "dialog",
            Self::Download => "download",
            Self::FileChooser => "filechooser",
            Self::FrameAttached => "frameattached",
            Self::FrameDetached => "framedetached",
            Self::FrameNavigated => "framenavigated",
            Self::Load => "load",
            Self::DomContentLoaded => "domcontentloaded",
            Self::PageError => "pageerror",
            Self::Popup => "popup",
            Self::Request => "request",
            Self::RequestFailed => "requestfailed",
            Self::RequestFinished => "requestfinished",
            Self::Response => "response",
            Self::WebSocket => "websocket",
            Self::Worker => "worker",
        }
    }
}

impl std::fmt::Display for PageEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// WAIT CONDITION TRAIT
// =============================================================================

/// Trait for custom wait conditions
pub trait WaitCondition: Send + Sync {
    /// Check if the condition is satisfied
    fn check(&self) -> bool;

    /// Get description for error messages
    fn description(&self) -> String;
}

/// A function-based wait condition
pub struct FnCondition<F: Fn() -> bool + Send + Sync> {
    func: F,
    description: String,
}

impl<F: Fn() -> bool + Send + Sync> std::fmt::Debug for FnCondition<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnCondition")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl<F: Fn() -> bool + Send + Sync> FnCondition<F> {
    /// Create a new function condition
    pub fn new(func: F, description: impl Into<String>) -> Self {
        Self {
            func,
            description: description.into(),
        }
    }
}

impl<F: Fn() -> bool + Send + Sync> WaitCondition for FnCondition<F> {
    fn check(&self) -> bool {
        (self.func)()
    }

    fn description(&self) -> String {
        self.description.clone()
    }
}

// =============================================================================
// WAIT RESULT
// =============================================================================

/// Result of a wait operation
#[derive(Debug, Clone)]
pub struct WaitResult {
    /// Whether the wait was successful
    pub success: bool,
    /// Time spent waiting
    pub elapsed: Duration,
    /// Description of what was waited for
    pub waited_for: String,
}

impl WaitResult {
    /// Create a successful wait result
    #[must_use]
    pub fn success(elapsed: Duration, waited_for: impl Into<String>) -> Self {
        Self {
            success: true,
            elapsed,
            waited_for: waited_for.into(),
        }
    }

    /// Create a timeout wait result
    #[must_use]
    pub fn timeout(elapsed: Duration, waited_for: impl Into<String>) -> Self {
        Self {
            success: false,
            elapsed,
            waited_for: waited_for.into(),
        }
    }
}

// =============================================================================
// WAITER IMPLEMENTATION
// =============================================================================

/// Waiter for synchronization operations
#[derive(Debug, Clone, Default)]
pub struct Waiter {
    /// Default options (reserved for future use)
    #[allow(dead_code)]
    options: WaitOptions,
    /// Current URL (for URL matching)
    current_url: Option<String>,
    /// Current load state
    load_state: LoadState,
    /// Pending network requests count
    pending_requests: usize,
    /// Time of last network activity
    last_network_activity: Option<Instant>,
    /// Events that have occurred
    events: Vec<PageEvent>,
}

impl Waiter {
    /// Create a new waiter with default options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom options
    #[must_use]
    pub fn with_options(options: WaitOptions) -> Self {
        Self {
            options,
            ..Default::default()
        }
    }

    /// Set current URL (for testing)
    pub fn set_url(&mut self, url: impl Into<String>) {
        self.current_url = Some(url.into());
    }

    /// Set load state (for testing)
    pub fn set_load_state(&mut self, state: LoadState) {
        self.load_state = state;
    }

    /// Update pending request count
    pub fn set_pending_requests(&mut self, count: usize) {
        self.pending_requests = count;
        if count > 0 {
            self.last_network_activity = Some(Instant::now());
        }
    }

    /// Record an event
    pub fn record_event(&mut self, event: PageEvent) {
        self.events.push(event);
    }

    /// Clear recorded events
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Wait for a custom condition
    pub fn wait_for<C: WaitCondition>(
        &self,
        condition: &C,
        options: &WaitOptions,
    ) -> ProbarResult<WaitResult> {
        let start = Instant::now();
        let timeout = Duration::from_millis(options.timeout_ms);
        let poll_interval = Duration::from_millis(options.poll_interval_ms);

        while start.elapsed() < timeout {
            if condition.check() {
                return Ok(WaitResult::success(
                    start.elapsed(),
                    condition.description(),
                ));
            }
            std::thread::sleep(poll_interval);
        }

        Err(ProbarError::Timeout {
            ms: options.timeout_ms,
        })
    }

    /// Wait for URL to match pattern
    pub fn wait_for_url(
        &self,
        pattern: &UrlPattern,
        options: &WaitOptions,
    ) -> ProbarResult<WaitResult> {
        let start = Instant::now();
        let timeout = Duration::from_millis(options.timeout_ms);
        let poll_interval = Duration::from_millis(options.poll_interval_ms);

        while start.elapsed() < timeout {
            if let Some(ref url) = self.current_url {
                if pattern.matches(url) {
                    return Ok(WaitResult::success(
                        start.elapsed(),
                        format!("URL matching {:?}", pattern),
                    ));
                }
            }
            std::thread::sleep(poll_interval);
        }

        Err(ProbarError::Timeout {
            ms: options.timeout_ms,
        })
    }

    /// Wait for load state
    pub fn wait_for_load_state(
        &self,
        state: LoadState,
        options: &WaitOptions,
    ) -> ProbarResult<WaitResult> {
        let start = Instant::now();
        let timeout = Duration::from_millis(options.timeout_ms);
        let poll_interval = Duration::from_millis(options.poll_interval_ms);

        while start.elapsed() < timeout {
            let state_reached = match state {
                LoadState::Load => self.load_state == LoadState::Load,
                LoadState::DomContentLoaded => {
                    self.load_state == LoadState::DomContentLoaded
                        || self.load_state == LoadState::Load
                }
                LoadState::NetworkIdle => self.is_network_idle(),
            };

            if state_reached {
                return Ok(WaitResult::success(
                    start.elapsed(),
                    format!("Load state: {}", state),
                ));
            }
            std::thread::sleep(poll_interval);
        }

        Err(ProbarError::Timeout {
            ms: options.timeout_ms,
        })
    }

    /// Check if network is idle
    fn is_network_idle(&self) -> bool {
        if self.pending_requests > 0 {
            return false;
        }

        match self.last_network_activity {
            Some(last) => last.elapsed() >= Duration::from_millis(NETWORK_IDLE_THRESHOLD_MS),
            None => true, // No network activity = idle
        }
    }

    /// Wait for navigation to complete
    pub fn wait_for_navigation(&self, options: &NavigationOptions) -> ProbarResult<WaitResult> {
        let wait_options = WaitOptions::new()
            .with_timeout(options.timeout_ms)
            .with_wait_until(options.wait_until);

        // If URL pattern specified, wait for URL first
        if let Some(ref pattern) = options.url_pattern {
            self.wait_for_url(pattern, &wait_options)?;
        }

        // Then wait for load state
        self.wait_for_load_state(options.wait_until, &wait_options)
    }

    /// Wait for a page event to occur
    pub fn wait_for_event(
        &self,
        event: &PageEvent,
        options: &WaitOptions,
    ) -> ProbarResult<WaitResult> {
        let start = Instant::now();
        let timeout = Duration::from_millis(options.timeout_ms);
        let poll_interval = Duration::from_millis(options.poll_interval_ms);

        while start.elapsed() < timeout {
            if self.events.contains(event) {
                return Ok(WaitResult::success(
                    start.elapsed(),
                    format!("Event: {}", event),
                ));
            }
            std::thread::sleep(poll_interval);
        }

        Err(ProbarError::Timeout {
            ms: options.timeout_ms,
        })
    }

    /// Wait for function/predicate to return true
    pub fn wait_for_function<F>(
        &self,
        predicate: F,
        options: &WaitOptions,
    ) -> ProbarResult<WaitResult>
    where
        F: Fn() -> bool,
    {
        let start = Instant::now();
        let timeout = Duration::from_millis(options.timeout_ms);
        let poll_interval = Duration::from_millis(options.poll_interval_ms);

        while start.elapsed() < timeout {
            if predicate() {
                return Ok(WaitResult::success(start.elapsed(), "custom function"));
            }
            std::thread::sleep(poll_interval);
        }

        Err(ProbarError::Timeout {
            ms: options.timeout_ms,
        })
    }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Wait for a condition with default options
pub fn wait_until<F>(predicate: F, timeout_ms: u64) -> ProbarResult<()>
where
    F: Fn() -> bool,
{
    let waiter = Waiter::new();
    let options = WaitOptions::new().with_timeout(timeout_ms);
    waiter.wait_for_function(predicate, &options)?;
    Ok(())
}

/// Wait for a fixed duration (discouraged - use wait conditions instead)
pub fn wait_timeout(duration_ms: u64) {
    std::thread::sleep(Duration::from_millis(duration_ms));
}

// =============================================================================
// TESTS - EXTREME TDD: Tests written FIRST
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // =========================================================================
    // LoadState Tests
    // =========================================================================

    mod load_state_tests {
        use super::*;

        #[test]
        fn test_load_state_event_names() {
            assert_eq!(LoadState::Load.event_name(), "load");
            assert_eq!(LoadState::DomContentLoaded.event_name(), "DOMContentLoaded");
            assert_eq!(LoadState::NetworkIdle.event_name(), "networkidle");
        }

        #[test]
        fn test_load_state_default_timeouts() {
            assert_eq!(LoadState::Load.default_timeout_ms(), 30_000);
            assert_eq!(LoadState::DomContentLoaded.default_timeout_ms(), 30_000);
            assert_eq!(LoadState::NetworkIdle.default_timeout_ms(), 60_000);
        }

        #[test]
        fn test_load_state_default() {
            assert_eq!(LoadState::default(), LoadState::Load);
        }

        #[test]
        fn test_load_state_display() {
            assert_eq!(format!("{}", LoadState::Load), "load");
            assert_eq!(
                format!("{}", LoadState::DomContentLoaded),
                "DOMContentLoaded"
            );
            assert_eq!(format!("{}", LoadState::NetworkIdle), "networkidle");
        }

        #[test]
        fn test_load_state_equality() {
            assert_eq!(LoadState::Load, LoadState::Load);
            assert_ne!(LoadState::Load, LoadState::DomContentLoaded);
        }

        #[test]
        fn test_load_state_clone() {
            let state = LoadState::NetworkIdle;
            let cloned = state;
            assert_eq!(state, cloned);
        }
    }

    // =========================================================================
    // WaitOptions Tests
    // =========================================================================

    mod wait_options_tests {
        use super::*;

        #[test]
        fn test_wait_options_default() {
            let opts = WaitOptions::default();
            assert_eq!(opts.timeout_ms, DEFAULT_WAIT_TIMEOUT_MS);
            assert_eq!(opts.poll_interval_ms, DEFAULT_POLL_INTERVAL_MS);
            assert_eq!(opts.wait_until, LoadState::Load);
        }

        #[test]
        fn test_wait_options_new() {
            let opts = WaitOptions::new();
            assert_eq!(opts.timeout_ms, DEFAULT_WAIT_TIMEOUT_MS);
        }

        #[test]
        fn test_wait_options_with_timeout() {
            let opts = WaitOptions::new().with_timeout(5000);
            assert_eq!(opts.timeout_ms, 5000);
        }

        #[test]
        fn test_wait_options_with_poll_interval() {
            let opts = WaitOptions::new().with_poll_interval(100);
            assert_eq!(opts.poll_interval_ms, 100);
        }

        #[test]
        fn test_wait_options_with_wait_until() {
            let opts = WaitOptions::new().with_wait_until(LoadState::NetworkIdle);
            assert_eq!(opts.wait_until, LoadState::NetworkIdle);
        }

        #[test]
        fn test_wait_options_chained() {
            let opts = WaitOptions::new()
                .with_timeout(10_000)
                .with_poll_interval(200)
                .with_wait_until(LoadState::DomContentLoaded);
            assert_eq!(opts.timeout_ms, 10_000);
            assert_eq!(opts.poll_interval_ms, 200);
            assert_eq!(opts.wait_until, LoadState::DomContentLoaded);
        }

        #[test]
        fn test_wait_options_timeout_duration() {
            let opts = WaitOptions::new().with_timeout(5000);
            assert_eq!(opts.timeout(), Duration::from_millis(5000));
        }

        #[test]
        fn test_wait_options_poll_interval_duration() {
            let opts = WaitOptions::new().with_poll_interval(100);
            assert_eq!(opts.poll_interval(), Duration::from_millis(100));
        }
    }

    // =========================================================================
    // NavigationOptions Tests
    // =========================================================================

    mod navigation_options_tests {
        use super::*;

        #[test]
        fn test_navigation_options_default() {
            let opts = NavigationOptions::default();
            assert_eq!(opts.timeout_ms, DEFAULT_WAIT_TIMEOUT_MS);
            assert_eq!(opts.wait_until, LoadState::Load);
            assert!(opts.url_pattern.is_none());
        }

        #[test]
        fn test_navigation_options_with_timeout() {
            let opts = NavigationOptions::new().with_timeout(10_000);
            assert_eq!(opts.timeout_ms, 10_000);
        }

        #[test]
        fn test_navigation_options_with_wait_until() {
            let opts = NavigationOptions::new().with_wait_until(LoadState::NetworkIdle);
            assert_eq!(opts.wait_until, LoadState::NetworkIdle);
        }

        #[test]
        fn test_navigation_options_with_url() {
            let opts =
                NavigationOptions::new().with_url(UrlPattern::Contains("example.com".into()));
            assert!(opts.url_pattern.is_some());
        }

        #[test]
        fn test_navigation_options_chained() {
            let opts = NavigationOptions::new()
                .with_timeout(5000)
                .with_wait_until(LoadState::DomContentLoaded)
                .with_url(UrlPattern::Exact("https://example.com".into()));
            assert_eq!(opts.timeout_ms, 5000);
            assert_eq!(opts.wait_until, LoadState::DomContentLoaded);
            assert!(opts.url_pattern.is_some());
        }
    }

    // =========================================================================
    // PageEvent Tests
    // =========================================================================

    mod page_event_tests {
        use super::*;

        #[test]
        fn test_page_event_as_str() {
            assert_eq!(PageEvent::Load.as_str(), "load");
            assert_eq!(PageEvent::DomContentLoaded.as_str(), "domcontentloaded");
            assert_eq!(PageEvent::Close.as_str(), "close");
            assert_eq!(PageEvent::Console.as_str(), "console");
            assert_eq!(PageEvent::Dialog.as_str(), "dialog");
            assert_eq!(PageEvent::Popup.as_str(), "popup");
            assert_eq!(PageEvent::Request.as_str(), "request");
            assert_eq!(PageEvent::Response.as_str(), "response");
        }

        #[test]
        fn test_page_event_display() {
            assert_eq!(format!("{}", PageEvent::Load), "load");
            assert_eq!(format!("{}", PageEvent::Popup), "popup");
        }

        #[test]
        fn test_page_event_equality() {
            assert_eq!(PageEvent::Load, PageEvent::Load);
            assert_ne!(PageEvent::Load, PageEvent::Close);
        }

        #[test]
        fn test_all_page_events() {
            let events = vec![
                PageEvent::Close,
                PageEvent::Console,
                PageEvent::Crash,
                PageEvent::Dialog,
                PageEvent::Download,
                PageEvent::FileChooser,
                PageEvent::FrameAttached,
                PageEvent::FrameDetached,
                PageEvent::FrameNavigated,
                PageEvent::Load,
                PageEvent::DomContentLoaded,
                PageEvent::PageError,
                PageEvent::Popup,
                PageEvent::Request,
                PageEvent::RequestFailed,
                PageEvent::RequestFinished,
                PageEvent::Response,
                PageEvent::WebSocket,
                PageEvent::Worker,
            ];
            assert_eq!(events.len(), 19);
            for event in events {
                assert!(!event.as_str().is_empty());
            }
        }
    }

    // =========================================================================
    // WaitResult Tests
    // =========================================================================

    mod wait_result_tests {
        use super::*;

        #[test]
        fn test_wait_result_success() {
            let result = WaitResult::success(Duration::from_millis(100), "test");
            assert!(result.success);
            assert_eq!(result.elapsed, Duration::from_millis(100));
            assert_eq!(result.waited_for, "test");
        }

        #[test]
        fn test_wait_result_timeout() {
            let result = WaitResult::timeout(Duration::from_secs(30), "test condition");
            assert!(!result.success);
            assert_eq!(result.elapsed, Duration::from_secs(30));
            assert_eq!(result.waited_for, "test condition");
        }
    }

    // =========================================================================
    // Waiter Tests
    // =========================================================================

    mod waiter_tests {
        use super::*;

        #[test]
        fn test_waiter_new() {
            let waiter = Waiter::new();
            assert!(waiter.current_url.is_none());
            assert_eq!(waiter.load_state, LoadState::default());
        }

        #[test]
        fn test_waiter_set_url() {
            let mut waiter = Waiter::new();
            waiter.set_url("https://example.com");
            assert_eq!(waiter.current_url, Some("https://example.com".to_string()));
        }

        #[test]
        fn test_waiter_set_load_state() {
            let mut waiter = Waiter::new();
            waiter.set_load_state(LoadState::NetworkIdle);
            assert_eq!(waiter.load_state, LoadState::NetworkIdle);
        }

        #[test]
        fn test_waiter_record_event() {
            let mut waiter = Waiter::new();
            waiter.record_event(PageEvent::Load);
            waiter.record_event(PageEvent::DomContentLoaded);
            assert_eq!(waiter.events.len(), 2);
            assert!(waiter.events.contains(&PageEvent::Load));
        }

        #[test]
        fn test_waiter_clear_events() {
            let mut waiter = Waiter::new();
            waiter.record_event(PageEvent::Load);
            waiter.clear_events();
            assert!(waiter.events.is_empty());
        }

        #[test]
        fn test_waiter_wait_for_function_immediate_success() {
            let waiter = Waiter::new();
            let options = WaitOptions::new().with_timeout(100);
            let result = waiter.wait_for_function(|| true, &options);
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result.success);
        }

        #[test]
        fn test_waiter_wait_for_function_timeout() {
            let waiter = Waiter::new();
            let options = WaitOptions::new().with_timeout(100).with_poll_interval(10);
            let result = waiter.wait_for_function(|| false, &options);
            assert!(result.is_err());
            match result {
                Err(ProbarError::Timeout { ms }) => assert_eq!(ms, 100),
                _ => panic!("Expected Timeout error"),
            }
        }

        #[test]
        fn test_waiter_wait_for_url_success() {
            let mut waiter = Waiter::new();
            waiter.set_url("https://example.com/test");
            let options = WaitOptions::new().with_timeout(100);
            let pattern = UrlPattern::Contains("example.com".into());
            let result = waiter.wait_for_url(&pattern, &options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_waiter_wait_for_url_timeout() {
            let mut waiter = Waiter::new();
            waiter.set_url("https://other.com");
            let options = WaitOptions::new().with_timeout(100).with_poll_interval(10);
            let pattern = UrlPattern::Contains("example.com".into());
            let result = waiter.wait_for_url(&pattern, &options);
            assert!(result.is_err());
        }

        #[test]
        fn test_waiter_wait_for_load_state_load() {
            let mut waiter = Waiter::new();
            waiter.set_load_state(LoadState::Load);
            let options = WaitOptions::new().with_timeout(100);
            let result = waiter.wait_for_load_state(LoadState::Load, &options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_waiter_wait_for_load_state_dom_content_loaded() {
            let mut waiter = Waiter::new();
            waiter.set_load_state(LoadState::DomContentLoaded);
            let options = WaitOptions::new().with_timeout(100);
            let result = waiter.wait_for_load_state(LoadState::DomContentLoaded, &options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_waiter_wait_for_load_state_dom_satisfied_by_load() {
            let mut waiter = Waiter::new();
            waiter.set_load_state(LoadState::Load);
            let options = WaitOptions::new().with_timeout(100);
            // Load state satisfies DomContentLoaded
            let result = waiter.wait_for_load_state(LoadState::DomContentLoaded, &options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_waiter_wait_for_event_success() {
            let mut waiter = Waiter::new();
            waiter.record_event(PageEvent::Popup);
            let options = WaitOptions::new().with_timeout(100);
            let result = waiter.wait_for_event(&PageEvent::Popup, &options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_waiter_wait_for_event_timeout() {
            let waiter = Waiter::new();
            let options = WaitOptions::new().with_timeout(100).with_poll_interval(10);
            let result = waiter.wait_for_event(&PageEvent::Popup, &options);
            assert!(result.is_err());
        }

        #[test]
        fn test_waiter_wait_for_navigation() {
            let mut waiter = Waiter::new();
            waiter.set_url("https://example.com");
            waiter.set_load_state(LoadState::Load);
            let options = NavigationOptions::new()
                .with_timeout(100)
                .with_url(UrlPattern::Contains("example.com".into()));
            let result = waiter.wait_for_navigation(&options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_waiter_network_idle_no_requests() {
            let waiter = Waiter::new();
            assert!(waiter.is_network_idle());
        }

        #[test]
        fn test_waiter_network_idle_with_pending() {
            let mut waiter = Waiter::new();
            waiter.set_pending_requests(1);
            assert!(!waiter.is_network_idle());
        }

        #[test]
        fn test_waiter_with_options() {
            let options = WaitOptions::new().with_timeout(5000);
            let waiter = Waiter::with_options(options);
            assert_eq!(waiter.options.timeout_ms, 5000);
        }
    }

    // =========================================================================
    // Convenience Function Tests
    // =========================================================================

    mod convenience_tests {
        use super::*;

        #[test]
        fn test_wait_until_success() {
            let result = wait_until(|| true, 100);
            assert!(result.is_ok());
        }

        #[test]
        fn test_wait_until_timeout() {
            let result = wait_until(|| false, 100);
            assert!(result.is_err());
        }

        #[test]
        fn test_wait_timeout() {
            let start = Instant::now();
            wait_timeout(50);
            assert!(start.elapsed() >= Duration::from_millis(50));
        }
    }

    // =========================================================================
    // WaitCondition Trait Tests
    // =========================================================================

    mod wait_condition_tests {
        use super::*;

        #[test]
        fn test_fn_condition_check_true() {
            let condition = FnCondition::new(|| true, "always true");
            assert!(condition.check());
        }

        #[test]
        fn test_fn_condition_check_false() {
            let condition = FnCondition::new(|| false, "always false");
            assert!(!condition.check());
        }

        #[test]
        fn test_fn_condition_description() {
            let condition = FnCondition::new(|| true, "my condition");
            assert_eq!(condition.description(), "my condition");
        }

        #[test]
        fn test_waiter_wait_for_condition() {
            let waiter = Waiter::new();
            let options = WaitOptions::new().with_timeout(100);
            let condition = FnCondition::new(|| true, "test condition");
            let result = waiter.wait_for(&condition, &options);
            assert!(result.is_ok());
        }
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    mod integration_tests {
        use super::*;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        #[test]
        fn test_wait_for_condition_becomes_true() {
            let flag = Arc::new(AtomicBool::new(false));
            let flag_clone = flag.clone();

            // Set flag after 50ms
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                flag_clone.store(true, Ordering::SeqCst);
            });

            let waiter = Waiter::new();
            let options = WaitOptions::new().with_timeout(200).with_poll_interval(10);
            let result = waiter.wait_for_function(|| flag.load(Ordering::SeqCst), &options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_multiple_wait_operations() {
            let mut waiter = Waiter::new();

            // First wait - URL
            waiter.set_url("https://example.com");
            let options = WaitOptions::new().with_timeout(100);
            let result = waiter.wait_for_url(&UrlPattern::Contains("example".into()), &options);
            assert!(result.is_ok());

            // Second wait - load state
            waiter.set_load_state(LoadState::Load);
            let result = waiter.wait_for_load_state(LoadState::Load, &options);
            assert!(result.is_ok());

            // Third wait - event
            waiter.record_event(PageEvent::Load);
            let result = waiter.wait_for_event(&PageEvent::Load, &options);
            assert!(result.is_ok());
        }

        #[test]
        fn test_url_pattern_types() {
            let mut waiter = Waiter::new();
            waiter.set_url("https://example.com/path/to/page");
            let options = WaitOptions::new().with_timeout(100);

            // Exact match
            let result = waiter.wait_for_url(
                &UrlPattern::Exact("https://example.com/path/to/page".into()),
                &options,
            );
            assert!(result.is_ok());

            // Contains match
            let result = waiter.wait_for_url(&UrlPattern::Contains("/path/".into()), &options);
            assert!(result.is_ok());

            // Prefix match
            let result =
                waiter.wait_for_url(&UrlPattern::Prefix("https://example".into()), &options);
            assert!(result.is_ok());
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Wait Mechanism Tests (G.1 Auto-waiting P0)
    // =========================================================================

    mod h0_constants_tests {
        use super::*;

        #[test]
        fn h0_wait_01_default_timeout_ms() {
            assert_eq!(DEFAULT_WAIT_TIMEOUT_MS, 30_000);
        }

        #[test]
        fn h0_wait_02_default_poll_interval_ms() {
            assert_eq!(DEFAULT_POLL_INTERVAL_MS, 50);
        }

        #[test]
        fn h0_wait_03_network_idle_threshold_ms() {
            assert_eq!(NETWORK_IDLE_THRESHOLD_MS, 500);
        }
    }

    mod h0_load_state_tests {
        use super::*;

        #[test]
        fn h0_wait_04_load_state_load_event() {
            assert_eq!(LoadState::Load.event_name(), "load");
        }

        #[test]
        fn h0_wait_05_load_state_dom_event() {
            assert_eq!(LoadState::DomContentLoaded.event_name(), "DOMContentLoaded");
        }

        #[test]
        fn h0_wait_06_load_state_network_event() {
            assert_eq!(LoadState::NetworkIdle.event_name(), "networkidle");
        }

        #[test]
        fn h0_wait_07_load_state_default() {
            assert_eq!(LoadState::default(), LoadState::Load);
        }

        #[test]
        fn h0_wait_08_load_state_timeout_load() {
            assert_eq!(LoadState::Load.default_timeout_ms(), 30_000);
        }

        #[test]
        fn h0_wait_09_load_state_timeout_network() {
            assert_eq!(LoadState::NetworkIdle.default_timeout_ms(), 60_000);
        }

        #[test]
        fn h0_wait_10_load_state_display() {
            assert_eq!(format!("{}", LoadState::Load), "load");
        }
    }

    mod h0_wait_options_tests {
        use super::*;

        #[test]
        fn h0_wait_11_options_default_timeout() {
            let opts = WaitOptions::default();
            assert_eq!(opts.timeout_ms, 30_000);
        }

        #[test]
        fn h0_wait_12_options_default_poll() {
            let opts = WaitOptions::default();
            assert_eq!(opts.poll_interval_ms, 50);
        }

        #[test]
        fn h0_wait_13_options_default_state() {
            let opts = WaitOptions::default();
            assert_eq!(opts.wait_until, LoadState::Load);
        }

        #[test]
        fn h0_wait_14_options_with_timeout() {
            let opts = WaitOptions::new().with_timeout(5000);
            assert_eq!(opts.timeout_ms, 5000);
        }

        #[test]
        fn h0_wait_15_options_with_poll() {
            let opts = WaitOptions::new().with_poll_interval(100);
            assert_eq!(opts.poll_interval_ms, 100);
        }

        #[test]
        fn h0_wait_16_options_with_state() {
            let opts = WaitOptions::new().with_wait_until(LoadState::NetworkIdle);
            assert_eq!(opts.wait_until, LoadState::NetworkIdle);
        }

        #[test]
        fn h0_wait_17_options_timeout_duration() {
            let opts = WaitOptions::new().with_timeout(1000);
            assert_eq!(opts.timeout(), Duration::from_secs(1));
        }

        #[test]
        fn h0_wait_18_options_poll_duration() {
            let opts = WaitOptions::new().with_poll_interval(100);
            assert_eq!(opts.poll_interval(), Duration::from_millis(100));
        }

        #[test]
        fn h0_wait_19_options_clone() {
            let opts = WaitOptions::new().with_timeout(1000);
            let cloned = opts;
            assert_eq!(cloned.timeout_ms, 1000);
        }

        #[test]
        fn h0_wait_20_options_debug() {
            let opts = WaitOptions::default();
            let debug = format!("{:?}", opts);
            assert!(debug.contains("WaitOptions"));
        }
    }

    mod h0_navigation_options_tests {
        use super::*;

        #[test]
        fn h0_wait_21_nav_default_timeout() {
            let opts = NavigationOptions::default();
            assert_eq!(opts.timeout_ms, 30_000);
        }

        #[test]
        fn h0_wait_22_nav_default_state() {
            let opts = NavigationOptions::default();
            assert_eq!(opts.wait_until, LoadState::Load);
        }

        #[test]
        fn h0_wait_23_nav_default_no_pattern() {
            let opts = NavigationOptions::default();
            assert!(opts.url_pattern.is_none());
        }

        #[test]
        fn h0_wait_24_nav_with_timeout() {
            let opts = NavigationOptions::new().with_timeout(5000);
            assert_eq!(opts.timeout_ms, 5000);
        }

        #[test]
        fn h0_wait_25_nav_with_state() {
            let opts = NavigationOptions::new().with_wait_until(LoadState::DomContentLoaded);
            assert_eq!(opts.wait_until, LoadState::DomContentLoaded);
        }

        #[test]
        fn h0_wait_26_nav_with_url() {
            let opts = NavigationOptions::new().with_url(UrlPattern::Contains("test".into()));
            assert!(opts.url_pattern.is_some());
        }

        #[test]
        fn h0_wait_27_nav_clone() {
            let opts = NavigationOptions::new().with_timeout(1000);
            let cloned = opts;
            assert_eq!(cloned.timeout_ms, 1000);
        }

        #[test]
        fn h0_wait_28_nav_debug() {
            let opts = NavigationOptions::default();
            let debug = format!("{:?}", opts);
            assert!(debug.contains("NavigationOptions"));
        }
    }

    mod h0_page_event_tests {
        use super::*;

        #[test]
        fn h0_wait_29_event_close() {
            assert_eq!(PageEvent::Close.as_str(), "close");
        }

        #[test]
        fn h0_wait_30_event_console() {
            assert_eq!(PageEvent::Console.as_str(), "console");
        }

        #[test]
        fn h0_wait_31_event_crash() {
            assert_eq!(PageEvent::Crash.as_str(), "crash");
        }

        #[test]
        fn h0_wait_32_event_dialog() {
            assert_eq!(PageEvent::Dialog.as_str(), "dialog");
        }

        #[test]
        fn h0_wait_33_event_download() {
            assert_eq!(PageEvent::Download.as_str(), "download");
        }

        #[test]
        fn h0_wait_34_event_display() {
            assert_eq!(format!("{}", PageEvent::Request), "request");
        }

        #[test]
        fn h0_wait_35_event_equality() {
            assert_eq!(PageEvent::Load, PageEvent::Load);
            assert_ne!(PageEvent::Load, PageEvent::Close);
        }

        #[test]
        fn h0_wait_36_event_clone() {
            let event = PageEvent::Response;
            let cloned = event;
            assert_eq!(cloned, PageEvent::Response);
        }
    }

    mod h0_wait_result_tests {
        use super::*;

        #[test]
        fn h0_wait_37_result_success_flag() {
            let result = WaitResult::success(Duration::from_millis(100), "test");
            assert!(result.success);
        }

        #[test]
        fn h0_wait_38_result_success_elapsed() {
            let result = WaitResult::success(Duration::from_millis(500), "test");
            assert_eq!(result.elapsed, Duration::from_millis(500));
        }

        #[test]
        fn h0_wait_39_result_timeout_flag() {
            let result = WaitResult::timeout(Duration::from_secs(30), "test");
            assert!(!result.success);
        }

        #[test]
        fn h0_wait_40_result_waited_for() {
            let result = WaitResult::success(Duration::ZERO, "my condition");
            assert_eq!(result.waited_for, "my condition");
        }

        #[test]
        fn h0_wait_41_result_clone() {
            let result = WaitResult::success(Duration::from_millis(100), "test");
            let cloned = result;
            assert!(cloned.success);
        }

        #[test]
        fn h0_wait_42_result_debug() {
            let result = WaitResult::success(Duration::ZERO, "test");
            let debug = format!("{:?}", result);
            assert!(debug.contains("WaitResult"));
        }
    }

    mod h0_waiter_tests {
        use super::*;

        #[test]
        fn h0_wait_43_waiter_new() {
            let waiter = Waiter::new();
            assert!(waiter.current_url.is_none());
        }

        #[test]
        fn h0_wait_44_waiter_set_url() {
            let mut waiter = Waiter::new();
            waiter.set_url("https://test.com");
            assert_eq!(waiter.current_url, Some("https://test.com".to_string()));
        }

        #[test]
        fn h0_wait_45_waiter_set_load_state() {
            let mut waiter = Waiter::new();
            waiter.set_load_state(LoadState::NetworkIdle);
            assert_eq!(waiter.load_state, LoadState::NetworkIdle);
        }

        #[test]
        fn h0_wait_46_waiter_record_event() {
            let mut waiter = Waiter::new();
            waiter.record_event(PageEvent::Load);
            assert_eq!(waiter.events.len(), 1);
        }

        #[test]
        fn h0_wait_47_waiter_clear_events() {
            let mut waiter = Waiter::new();
            waiter.record_event(PageEvent::Load);
            waiter.clear_events();
            assert!(waiter.events.is_empty());
        }

        #[test]
        fn h0_wait_48_waiter_pending_requests() {
            let mut waiter = Waiter::new();
            waiter.set_pending_requests(5);
            assert_eq!(waiter.pending_requests, 5);
        }

        #[test]
        fn h0_wait_49_waiter_default() {
            let waiter = Waiter::default();
            assert_eq!(waiter.pending_requests, 0);
        }

        #[test]
        fn h0_wait_50_waiter_with_options() {
            let options = WaitOptions::new().with_timeout(5000);
            let waiter = Waiter::with_options(options);
            assert_eq!(waiter.options.timeout_ms, 5000);
        }
    }
}
