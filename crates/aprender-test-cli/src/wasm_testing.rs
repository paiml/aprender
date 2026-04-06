//! WASM Testing Features Module
//!
//! Implements the top 5 WASM/TUI testing features from the spec (Section E):
//!
//! 1. **Deterministic Replay** - Record and replay test sessions
//! 2. **Memory Profiling** - Track WASM linear memory usage
//! 3. **State Machine Validation** - Playbook integration
//! 4. **Cross-Browser Testing** - Multi-browser matrix
//! 5. **Performance Regression** - Baseline tracking

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::io_other_error)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// =============================================================================
// E.1 Deterministic Replay
// =============================================================================

/// A recorded event for deterministic replay
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RecordedEvent {
    /// Mouse click event
    Click {
        /// X coordinate
        x: i32,
        /// Y coordinate
        y: i32,
        /// CSS selector of target element
        selector: Option<String>,
        /// Timestamp in milliseconds since recording start
        timestamp_ms: u64,
    },
    /// Keyboard input event
    KeyPress {
        /// Key code
        key: String,
        /// Modifier keys
        modifiers: KeyModifiers,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
    /// Text input event
    TextInput {
        /// Input text
        text: String,
        /// Target selector
        selector: Option<String>,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
    /// Network request completed
    NetworkComplete {
        /// Request URL
        url: String,
        /// Response status
        status: u16,
        /// Duration in milliseconds
        duration_ms: u64,
        /// Timestamp
        timestamp_ms: u64,
    },
    /// WASM module loaded
    WasmLoaded {
        /// Module URL
        url: String,
        /// Module size in bytes
        size: u64,
        /// Timestamp
        timestamp_ms: u64,
    },
    /// State transition
    StateChange {
        /// Previous state
        from: String,
        /// New state
        to: String,
        /// Event that triggered the transition
        event: String,
        /// Timestamp
        timestamp_ms: u64,
    },
    /// Assertion check
    Assertion {
        /// Assertion name
        name: String,
        /// Whether assertion passed
        passed: bool,
        /// Actual value
        actual: String,
        /// Expected value
        expected: String,
        /// Timestamp
        timestamp_ms: u64,
    },
}

/// Keyboard modifier keys
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyModifiers {
    /// Ctrl/Command key
    pub ctrl: bool,
    /// Alt key
    pub alt: bool,
    /// Shift key
    pub shift: bool,
    /// Meta key (Windows/Command)
    pub meta: bool,
}

/// A recorded test session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    /// Recording version
    pub version: String,
    /// Recording name
    pub name: String,
    /// URL where recording was made
    pub url: String,
    /// Browser user agent
    pub user_agent: String,
    /// Viewport dimensions
    pub viewport: Viewport,
    /// Start timestamp (Unix milliseconds)
    pub start_time: u64,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Recorded events
    pub events: Vec<RecordedEvent>,
    /// Metadata
    pub metadata: RecordingMetadata,
}

/// Viewport dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Viewport {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Device pixel ratio
    pub device_pixel_ratio: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            device_pixel_ratio: 1.0,
        }
    }
}

/// Recording metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecordingMetadata {
    /// Git commit hash
    pub commit: Option<String>,
    /// Test name
    pub test_name: Option<String>,
    /// Description
    pub description: Option<String>,
}

impl Recording {
    /// Create a new recording
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            version: "1.0.0".to_string(),
            name: name.into(),
            url: url.into(),
            user_agent: String::new(),
            viewport: Viewport::default(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            duration_ms: 0,
            events: Vec::new(),
            metadata: RecordingMetadata::default(),
        }
    }

    /// Add an event to the recording
    pub fn add_event(&mut self, event: RecordedEvent) {
        self.events.push(event);
    }

    /// Get event count
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Save recording to file
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load recording from file
    pub fn load(path: &PathBuf) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// E.2 Memory Profiling
// =============================================================================

/// Memory profile snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Heap size in bytes
    pub heap_bytes: u64,
    /// Timestamp since start (milliseconds)
    pub timestamp_ms: u64,
    /// Label for this snapshot
    pub label: Option<String>,
}

/// Memory profile for a WASM module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Module name
    pub module_name: String,
    /// Initial heap size
    pub initial_heap: u64,
    /// Peak heap size
    pub peak_heap: u64,
    /// Current heap size
    pub current_heap: u64,
    /// Memory snapshots over time
    pub snapshots: Vec<MemorySnapshot>,
    /// Growth events
    pub growth_events: Vec<MemoryGrowthEvent>,
}

/// Memory growth event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGrowthEvent {
    /// Size before growth
    pub from_bytes: u64,
    /// Size after growth
    pub to_bytes: u64,
    /// Timestamp (ms)
    pub timestamp_ms: u64,
    /// Reason for growth (if known)
    pub reason: Option<String>,
}

impl MemoryProfile {
    /// Create a new memory profile
    pub fn new(module_name: impl Into<String>, initial_heap: u64) -> Self {
        Self {
            module_name: module_name.into(),
            initial_heap,
            peak_heap: initial_heap,
            current_heap: initial_heap,
            snapshots: vec![MemorySnapshot {
                heap_bytes: initial_heap,
                timestamp_ms: 0,
                label: Some("initial".to_string()),
            }],
            growth_events: Vec::new(),
        }
    }

    /// Record a memory snapshot
    pub fn snapshot(&mut self, heap_bytes: u64, timestamp_ms: u64, label: Option<String>) {
        if heap_bytes > self.current_heap {
            self.growth_events.push(MemoryGrowthEvent {
                from_bytes: self.current_heap,
                to_bytes: heap_bytes,
                timestamp_ms,
                reason: label.clone(),
            });
        }

        self.current_heap = heap_bytes;
        if heap_bytes > self.peak_heap {
            self.peak_heap = heap_bytes;
        }

        self.snapshots.push(MemorySnapshot {
            heap_bytes,
            timestamp_ms,
            label,
        });
    }

    /// Check if memory exceeds threshold
    pub fn exceeds_threshold(&self, threshold_bytes: u64) -> bool {
        self.peak_heap > threshold_bytes
    }

    /// Get memory growth percentage
    pub fn growth_percentage(&self) -> f64 {
        if self.initial_heap == 0 {
            return 0.0;
        }
        ((self.peak_heap - self.initial_heap) as f64 / self.initial_heap as f64) * 100.0
    }
}

// =============================================================================
// E.4 Cross-Browser Testing
// =============================================================================

/// Supported browser engines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Browser {
    /// Chromium-based (Chrome, Edge, etc.)
    Chrome,
    /// Gecko-based (Firefox)
    Firefox,
    /// WebKit-based (Safari)
    Safari,
    /// iOS Safari
    IosSafari,
    /// Chrome Android
    ChromeAndroid,
}

impl Browser {
    /// Get browser display name
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Chrome => "Chrome",
            Self::Firefox => "Firefox",
            Self::Safari => "Safari",
            Self::IosSafari => "iOS Safari",
            Self::ChromeAndroid => "Chrome Android",
        }
    }

    /// Get browser engine
    pub const fn engine(&self) -> &'static str {
        match self {
            Self::Chrome | Self::ChromeAndroid => "Chromium",
            Self::Firefox => "Gecko",
            Self::Safari | Self::IosSafari => "WebKit",
        }
    }

    /// Get all desktop browsers
    pub fn desktop_browsers() -> Vec<Self> {
        vec![Self::Chrome, Self::Firefox, Self::Safari]
    }

    /// Get all mobile browsers
    pub fn mobile_browsers() -> Vec<Self> {
        vec![Self::IosSafari, Self::ChromeAndroid]
    }
}

/// Cross-browser test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserMatrix {
    /// Browsers to test
    pub browsers: Vec<Browser>,
    /// Viewports to test
    pub viewports: Vec<Viewport>,
    /// Run in parallel
    pub parallel: bool,
}

impl Default for BrowserMatrix {
    fn default() -> Self {
        Self {
            browsers: Browser::desktop_browsers(),
            viewports: vec![
                Viewport {
                    width: 1920,
                    height: 1080,
                    device_pixel_ratio: 1.0,
                },
                Viewport {
                    width: 1280,
                    height: 720,
                    device_pixel_ratio: 1.0,
                },
                Viewport {
                    width: 375,
                    height: 667,
                    device_pixel_ratio: 2.0,
                }, // Mobile
            ],
            parallel: true,
        }
    }
}

/// Cross-browser test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserTestResult {
    /// Browser used
    pub browser: Browser,
    /// Viewport used
    pub viewport: Viewport,
    /// Whether test passed
    pub passed: bool,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Screenshots taken
    pub screenshots: Vec<String>,
}

// =============================================================================
// E.5 Performance Regression Detection
// =============================================================================

/// Performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Unit (e.g., "ms", "MB", "x")
    pub unit: String,
}

/// Performance baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline version
    pub version: String,
    /// Git commit hash
    pub commit: String,
    /// Timestamp
    pub timestamp: u64,
    /// Metrics
    pub metrics: Vec<PerformanceMetric>,
}

impl PerformanceBaseline {
    /// Create a new baseline
    pub fn new(commit: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            version: "1.0.0".to_string(),
            commit: commit.into(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            metrics: Vec::new(),
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64, unit: impl Into<String>) {
        self.metrics.push(PerformanceMetric {
            name: name.into(),
            value,
            unit: unit.into(),
        });
    }

    /// Save baseline to file
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load baseline from file
    pub fn load(path: &PathBuf) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Performance comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// Metric name
    pub name: String,
    /// Baseline value
    pub baseline: f64,
    /// Current value
    pub current: f64,
    /// Change percentage
    pub change_percent: f64,
    /// Status (ok, warn, fail)
    pub status: ComparisonStatus,
}

/// Comparison status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComparisonStatus {
    /// Within acceptable range
    Ok,
    /// Approaching threshold
    Warn,
    /// Exceeds threshold
    Fail,
}

impl ComparisonStatus {
    /// Get display symbol
    pub const fn symbol(&self) -> &'static str {
        match self {
            Self::Ok => "✓",
            Self::Warn => "⚠",
            Self::Fail => "✗",
        }
    }
}

/// Compare current metrics against baseline
pub fn compare_performance(
    baseline: &PerformanceBaseline,
    current: &[PerformanceMetric],
    threshold_percent: f64,
) -> Vec<PerformanceComparison> {
    let mut results = Vec::new();

    for current_metric in current {
        if let Some(baseline_metric) = baseline
            .metrics
            .iter()
            .find(|m| m.name == current_metric.name)
        {
            let change = if baseline_metric.value != 0.0 {
                ((current_metric.value - baseline_metric.value) / baseline_metric.value) * 100.0
            } else {
                0.0
            };

            let status = if change.abs() > threshold_percent {
                ComparisonStatus::Fail
            } else if change.abs() > threshold_percent * 0.8 {
                ComparisonStatus::Warn
            } else {
                ComparisonStatus::Ok
            };

            results.push(PerformanceComparison {
                name: current_metric.name.clone(),
                baseline: baseline_metric.value,
                current: current_metric.value,
                change_percent: change,
                status,
            });
        }
    }

    results
}

/// Render performance comparison as text
pub fn render_performance_report(
    baseline: &PerformanceBaseline,
    comparisons: &[PerformanceComparison],
) -> String {
    let mut output = String::new();

    output.push_str("PERFORMANCE REGRESSION CHECK\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    output.push_str(&format!(
        "Baseline: {} (commit {})\n\n",
        baseline.version,
        &baseline.commit[..8.min(baseline.commit.len())]
    ));

    output.push_str("┌────────────────────┬──────────┬──────────┬──────────┬────────┐\n");
    output.push_str("│ Metric             │ Baseline │ Current  │ Delta    │ Status │\n");
    output.push_str("├────────────────────┼──────────┼──────────┼──────────┼────────┤\n");

    for comp in comparisons {
        let delta = if comp.change_percent >= 0.0 {
            format!("+{:.1}%", comp.change_percent)
        } else {
            format!("{:.1}%", comp.change_percent)
        };

        output.push_str(&format!(
            "│ {:<18} │ {:>8.1} │ {:>8.1} │ {:>8} │ {} {:>4} │\n",
            comp.name,
            comp.baseline,
            comp.current,
            delta,
            comp.status.symbol(),
            match comp.status {
                ComparisonStatus::Ok => "OK",
                ComparisonStatus::Warn => "WARN",
                ComparisonStatus::Fail => "FAIL",
            }
        ));
    }

    output.push_str("└────────────────────┴──────────┴──────────┴──────────┴────────┘\n");

    let warnings = comparisons
        .iter()
        .filter(|c| c.status == ComparisonStatus::Warn)
        .count();
    let failures = comparisons
        .iter()
        .filter(|c| c.status == ComparisonStatus::Fail)
        .count();

    output.push_str(&format!(
        "\nResult: {} warnings, {} failures\n",
        warnings, failures
    ));

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // Recording tests
    #[test]
    fn test_recording_new() {
        let recording = Recording::new("test", "http://localhost:8080");
        assert_eq!(recording.name, "test");
        assert_eq!(recording.url, "http://localhost:8080");
        assert_eq!(recording.event_count(), 0);
    }

    #[test]
    fn test_recording_add_event() {
        let mut recording = Recording::new("test", "http://localhost");
        recording.add_event(RecordedEvent::Click {
            x: 100,
            y: 200,
            selector: Some("#button".to_string()),
            timestamp_ms: 1000,
        });
        assert_eq!(recording.event_count(), 1);
    }

    #[test]
    fn test_key_modifiers_default() {
        let mods = KeyModifiers::default();
        assert!(!mods.ctrl);
        assert!(!mods.alt);
        assert!(!mods.shift);
        assert!(!mods.meta);
    }

    // Memory profiling tests
    #[test]
    fn test_memory_profile_new() {
        let profile = MemoryProfile::new("test_module", 1024 * 1024);
        assert_eq!(profile.initial_heap, 1024 * 1024);
        assert_eq!(profile.peak_heap, 1024 * 1024);
    }

    #[test]
    fn test_memory_profile_snapshot() {
        let mut profile = MemoryProfile::new("test", 1000);
        profile.snapshot(2000, 100, Some("allocation".to_string()));

        assert_eq!(profile.current_heap, 2000);
        assert_eq!(profile.peak_heap, 2000);
        assert_eq!(profile.snapshots.len(), 2);
        assert_eq!(profile.growth_events.len(), 1);
    }

    #[test]
    fn test_memory_profile_threshold() {
        let mut profile = MemoryProfile::new("test", 100);
        profile.snapshot(500, 100, None);

        assert!(profile.exceeds_threshold(400));
        assert!(!profile.exceeds_threshold(600));
    }

    #[test]
    fn test_memory_profile_growth_percentage() {
        let mut profile = MemoryProfile::new("test", 100);
        profile.snapshot(200, 100, None);

        assert!((profile.growth_percentage() - 100.0).abs() < 0.1);
    }

    // Browser tests
    #[test]
    fn test_browser_name() {
        assert_eq!(Browser::Chrome.name(), "Chrome");
        assert_eq!(Browser::Firefox.name(), "Firefox");
        assert_eq!(Browser::Safari.name(), "Safari");
    }

    #[test]
    fn test_browser_engine() {
        assert_eq!(Browser::Chrome.engine(), "Chromium");
        assert_eq!(Browser::Firefox.engine(), "Gecko");
        assert_eq!(Browser::Safari.engine(), "WebKit");
    }

    #[test]
    fn test_browser_matrix_default() {
        let matrix = BrowserMatrix::default();
        assert_eq!(matrix.browsers.len(), 3);
        assert_eq!(matrix.viewports.len(), 3);
        assert!(matrix.parallel);
    }

    // Performance tests
    #[test]
    fn test_performance_baseline_new() {
        let baseline = PerformanceBaseline::new("abc123");
        assert_eq!(baseline.commit, "abc123");
        assert!(baseline.metrics.is_empty());
    }

    #[test]
    fn test_performance_baseline_add_metric() {
        let mut baseline = PerformanceBaseline::new("abc123");
        baseline.add_metric("rtf", 1.5, "x");
        baseline.add_metric("latency_p95", 45.0, "ms");

        assert_eq!(baseline.metrics.len(), 2);
    }

    #[test]
    fn test_compare_performance_ok() {
        let mut baseline = PerformanceBaseline::new("old");
        baseline.add_metric("latency", 100.0, "ms");

        let current = vec![PerformanceMetric {
            name: "latency".to_string(),
            value: 105.0,
            unit: "ms".to_string(),
        }];

        let results = compare_performance(&baseline, &current, 10.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, ComparisonStatus::Ok);
    }

    #[test]
    fn test_compare_performance_warn() {
        let mut baseline = PerformanceBaseline::new("old");
        baseline.add_metric("latency", 100.0, "ms");

        let current = vec![PerformanceMetric {
            name: "latency".to_string(),
            value: 109.0,
            unit: "ms".to_string(),
        }];

        let results = compare_performance(&baseline, &current, 10.0);
        assert_eq!(results[0].status, ComparisonStatus::Warn);
    }

    #[test]
    fn test_compare_performance_fail() {
        let mut baseline = PerformanceBaseline::new("old");
        baseline.add_metric("latency", 100.0, "ms");

        let current = vec![PerformanceMetric {
            name: "latency".to_string(),
            value: 115.0,
            unit: "ms".to_string(),
        }];

        let results = compare_performance(&baseline, &current, 10.0);
        assert_eq!(results[0].status, ComparisonStatus::Fail);
    }

    #[test]
    fn test_comparison_status_symbol() {
        assert_eq!(ComparisonStatus::Ok.symbol(), "✓");
        assert_eq!(ComparisonStatus::Warn.symbol(), "⚠");
        assert_eq!(ComparisonStatus::Fail.symbol(), "✗");
    }

    #[test]
    fn test_render_performance_report() {
        let mut baseline = PerformanceBaseline::new("abc12345");
        baseline.add_metric("latency", 100.0, "ms");

        let comparisons = vec![PerformanceComparison {
            name: "latency".to_string(),
            baseline: 100.0,
            current: 105.0,
            change_percent: 5.0,
            status: ComparisonStatus::Ok,
        }];

        let output = render_performance_report(&baseline, &comparisons);
        assert!(output.contains("PERFORMANCE REGRESSION"));
        assert!(output.contains("latency"));
        assert!(output.contains("+5.0%"));
    }

    #[test]
    fn test_viewport_default() {
        let vp = Viewport::default();
        assert_eq!(vp.width, 1920);
        assert_eq!(vp.height, 1080);
    }

    #[test]
    fn test_recording_save_load() {
        let mut recording = Recording::new("test-session", "http://localhost/test");
        recording.add_event(RecordedEvent::Click {
            x: 100,
            y: 200,
            selector: Some("#button".to_string()),
            timestamp_ms: 1000,
        });
        recording.add_event(RecordedEvent::KeyPress {
            key: "Enter".to_string(),
            modifiers: KeyModifiers::default(),
            timestamp_ms: 2000,
        });

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_recording.json");

        // Save
        recording.save(&path).expect("Failed to save recording");

        // Load
        let loaded = Recording::load(&path).expect("Failed to load recording");
        assert_eq!(loaded.name, "test-session");
        assert_eq!(loaded.event_count(), 2);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_recording_load_nonexistent() {
        let result = Recording::load(&PathBuf::from("/nonexistent/path.json"));
        assert!(result.is_err());
    }

    // Additional tests for full coverage

    #[test]
    fn test_recorded_event_all_variants() {
        // TextInput
        let text_input = RecordedEvent::TextInput {
            text: "hello".to_string(),
            selector: Some("#input".to_string()),
            timestamp_ms: 100,
        };
        assert!(matches!(text_input, RecordedEvent::TextInput { .. }));

        // NetworkComplete
        let network = RecordedEvent::NetworkComplete {
            url: "https://api.example.com".to_string(),
            status: 200,
            duration_ms: 50,
            timestamp_ms: 200,
        };
        assert!(matches!(network, RecordedEvent::NetworkComplete { .. }));

        // WasmLoaded
        let wasm = RecordedEvent::WasmLoaded {
            url: "/game.wasm".to_string(),
            size: 1024000,
            timestamp_ms: 300,
        };
        assert!(matches!(wasm, RecordedEvent::WasmLoaded { .. }));

        // StateChange
        let state_change = RecordedEvent::StateChange {
            from: "menu".to_string(),
            to: "game".to_string(),
            event: "start_clicked".to_string(),
            timestamp_ms: 400,
        };
        assert!(matches!(state_change, RecordedEvent::StateChange { .. }));

        // Assertion
        let assertion = RecordedEvent::Assertion {
            name: "score_check".to_string(),
            passed: true,
            actual: "100".to_string(),
            expected: "100".to_string(),
            timestamp_ms: 500,
        };
        assert!(matches!(assertion, RecordedEvent::Assertion { .. }));
    }

    #[test]
    fn test_browser_ios_safari() {
        assert_eq!(Browser::IosSafari.name(), "iOS Safari");
        assert_eq!(Browser::IosSafari.engine(), "WebKit");
    }

    #[test]
    fn test_browser_chrome_android() {
        assert_eq!(Browser::ChromeAndroid.name(), "Chrome Android");
        assert_eq!(Browser::ChromeAndroid.engine(), "Chromium");
    }

    #[test]
    fn test_browser_desktop_browsers() {
        let browsers = Browser::desktop_browsers();
        assert_eq!(browsers.len(), 3);
        assert!(browsers.contains(&Browser::Chrome));
        assert!(browsers.contains(&Browser::Firefox));
        assert!(browsers.contains(&Browser::Safari));
    }

    #[test]
    fn test_browser_mobile_browsers() {
        let browsers = Browser::mobile_browsers();
        assert_eq!(browsers.len(), 2);
        assert!(browsers.contains(&Browser::IosSafari));
        assert!(browsers.contains(&Browser::ChromeAndroid));
    }

    #[test]
    fn test_browser_test_result() {
        let result = BrowserTestResult {
            browser: Browser::Chrome,
            viewport: Viewport::default(),
            passed: true,
            duration_ms: 1500,
            error: None,
            screenshots: vec!["screenshot1.png".to_string()],
        };
        assert!(result.passed);
        assert!(result.error.is_none());
        assert_eq!(result.screenshots.len(), 1);
    }

    #[test]
    fn test_browser_test_result_failed() {
        let result = BrowserTestResult {
            browser: Browser::Firefox,
            viewport: Viewport {
                width: 1280,
                height: 720,
                device_pixel_ratio: 1.0,
            },
            passed: false,
            duration_ms: 500,
            error: Some("Element not found".to_string()),
            screenshots: vec![],
        };
        assert!(!result.passed);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_memory_profile_growth_zero_initial() {
        let profile = MemoryProfile::new("test", 0);
        assert_eq!(profile.growth_percentage(), 0.0);
    }

    #[test]
    fn test_memory_profile_no_growth() {
        let mut profile = MemoryProfile::new("test", 1000);
        profile.snapshot(800, 100, Some("shrink".to_string()));

        assert_eq!(profile.current_heap, 800);
        assert_eq!(profile.peak_heap, 1000);
        assert!(profile.growth_events.is_empty());
    }

    #[test]
    fn test_compare_performance_zero_baseline() {
        let mut baseline = PerformanceBaseline::new("old");
        baseline.add_metric("count", 0.0, "n");

        let current = vec![PerformanceMetric {
            name: "count".to_string(),
            value: 10.0,
            unit: "n".to_string(),
        }];

        let results = compare_performance(&baseline, &current, 10.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].change_percent, 0.0);
    }

    #[test]
    fn test_compare_performance_no_match() {
        let mut baseline = PerformanceBaseline::new("old");
        baseline.add_metric("latency", 100.0, "ms");

        let current = vec![PerformanceMetric {
            name: "throughput".to_string(),
            value: 500.0,
            unit: "req/s".to_string(),
        }];

        let results = compare_performance(&baseline, &current, 10.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_render_performance_report_negative_change() {
        let mut baseline = PerformanceBaseline::new("abc12345");
        baseline.add_metric("latency", 100.0, "ms");

        let comparisons = vec![PerformanceComparison {
            name: "latency".to_string(),
            baseline: 100.0,
            current: 90.0,
            change_percent: -10.0,
            status: ComparisonStatus::Ok,
        }];

        let output = render_performance_report(&baseline, &comparisons);
        assert!(output.contains("-10.0%"));
    }

    #[test]
    fn test_render_performance_report_warnings_and_failures() {
        let baseline = PerformanceBaseline::new("abc12345");

        let comparisons = vec![
            PerformanceComparison {
                name: "metric1".to_string(),
                baseline: 100.0,
                current: 109.0,
                change_percent: 9.0,
                status: ComparisonStatus::Warn,
            },
            PerformanceComparison {
                name: "metric2".to_string(),
                baseline: 100.0,
                current: 120.0,
                change_percent: 20.0,
                status: ComparisonStatus::Fail,
            },
        ];

        let output = render_performance_report(&baseline, &comparisons);
        assert!(output.contains("1 warnings"));
        assert!(output.contains("1 failures"));
    }

    #[test]
    fn test_recording_metadata() {
        let mut recording = Recording::new("test", "http://localhost");
        recording.metadata = RecordingMetadata {
            commit: Some("abc123".to_string()),
            test_name: Some("login_test".to_string()),
            description: Some("Tests the login flow".to_string()),
        };

        assert_eq!(recording.metadata.commit, Some("abc123".to_string()));
        assert_eq!(recording.metadata.test_name, Some("login_test".to_string()));
    }

    #[test]
    fn test_recording_serde() {
        let mut recording = Recording::new("serde_test", "http://localhost");
        recording.add_event(RecordedEvent::Click {
            x: 10,
            y: 20,
            selector: None,
            timestamp_ms: 0,
        });

        let json = serde_json::to_string(&recording).unwrap();
        let parsed: Recording = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "serde_test");
        assert_eq!(parsed.event_count(), 1);
    }

    #[test]
    fn test_performance_baseline_save_load() {
        let mut baseline = PerformanceBaseline::new("commit123");
        baseline.add_metric("rtf", 1.5, "x");
        baseline.add_metric("fps", 60.0, "fps");

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_baseline.json");

        baseline.save(&path).expect("Failed to save");

        let loaded = PerformanceBaseline::load(&path).expect("Failed to load");
        assert_eq!(loaded.commit, "commit123");
        assert_eq!(loaded.metrics.len(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_performance_baseline_load_nonexistent() {
        let result = PerformanceBaseline::load(&PathBuf::from("/nonexistent/baseline.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_key_modifiers_all_true() {
        let mods = KeyModifiers {
            ctrl: true,
            alt: true,
            shift: true,
            meta: true,
        };
        assert!(mods.ctrl);
        assert!(mods.alt);
        assert!(mods.shift);
        assert!(mods.meta);
    }

    #[test]
    fn test_browser_eq() {
        assert_eq!(Browser::Chrome, Browser::Chrome);
        assert_ne!(Browser::Chrome, Browser::Firefox);
    }

    #[test]
    fn test_comparison_status_eq() {
        assert_eq!(ComparisonStatus::Ok, ComparisonStatus::Ok);
        assert_ne!(ComparisonStatus::Ok, ComparisonStatus::Fail);
    }
}
