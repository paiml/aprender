# Probar: Missing Features Analysis for Pure Rust WASM Implementation

**Document Version:** 1.0.0
**Status:** Draft Specification
**PMAT Work Reference:** PROBAR-SPEC-001
**Quality Standard:** EXTREME TDD + Toyota Way

---

## Executive Summary

This specification analyzes the top 20 features from Puppeteer and Playwright that are missing from Probar and can be implemented in **pure Rust**. The architecture is divided into two distinct contexts:

1.  **Host CLI** (Running on Linux/macOS/Windows): Orchestrates tests, manages browsers via CDP, and handles file I/O (reporting, media saving).
2.  **WASM Runtime** (Running in Browser): The core test driver injected into the browser, which must be **ABSOLUTE ZERO JavaScript**.

### Constraints

1.  **Pure Rust Only**: No JavaScript, TypeScript, or any JS runtime dependencies.
2.  **WASM-Compatible (Runtime)**: In-browser components must compile to `wasm32-unknown-unknown`.
3.  **CDP-Based (Host)**: Leverage Chrome DevTools Protocol via `chromiumoxide` crate.
4.  **Deterministic**: All operations must support deterministic replay.
5.  **Toyota Way**: Built-in quality (Jidoka), mistake-proofing (Poka-Yoke), continuous improvement (Kaizen).

---

## Toyota Way Quality Principles Applied

| Principle | Application |
|-----------|-------------|
| **Jidoka** (Autonomation) | Fail-fast on quality issues; Andon cord pattern for test failures |
| **Poka-Yoke** (Mistake-Proofing) | Type-safe selectors prevent runtime errors at compile time |
| **Muda** (Waste Elimination) | Zero-copy memory views; lazy evaluation where possible |
| **Heijunka** (Level Loading) | Superblock scheduling for consistent test execution timing |
| **Genchi Genbutsu** (Go and See) | Abstract drivers allow swapping browser/runtime implementations |
| **Kaizen** (Continuous Improvement) | Mutation testing integration; coverage-guided test generation |

---

## Feature Analysis: Top 20 Missing Features

### Category 1: Media Generation & Recording

#### Feature 1: Animated GIF Recording

**Priority:** P0 (Critical)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** Record test execution as animated GIF for documentation, bug reports, and visual verification.

**Rust Crates Required:**
- `gif` (v0.13) - GIF encoding [1]
- `image` (v0.25) - Image processing

**API Design:**
```rust
pub struct GifRecorder {
    frames: Vec<Frame>,
    config: GifConfig,
}

pub struct GifConfig {
    pub fps: u8,           // 10-30 typical
    pub width: u32,
    pub height: u32,
    pub quality: u8,       // 1-100, palette quantization
    pub loop_count: u16,   // 0 = infinite
}

impl GifRecorder {
    pub fn start(&mut self) -> ProbarResult<()>;
    pub fn capture_frame(&mut self, screenshot: &Screenshot) -> ProbarResult<()>;
    pub fn stop(&mut self) -> ProbarResult<Vec<u8>>;
    pub fn save(&self, path: &Path) -> ProbarResult<()>;
}
```

**Toyota Way Application:**
- **Poka-Yoke**: Type-safe frame capture prevents format mismatches
- **Muda**: Lazy frame encoding reduces memory pressure

**Test Coverage Requirements:** ≥95% line coverage, mutation score ≥80%

---

#### Feature 2: PNG Screenshot Export

**Priority:** P0 (Critical)
**Complexity:** Low
**Pure Rust Feasibility:** ✅ Yes

**Description:** High-quality PNG screenshots with configurable compression and metadata.

**Rust Crates Required:**
- `png` (v0.17) - PNG encoding
- `image` (v0.25) - Image processing

**API Design:**
```rust
pub struct PngExporter {
    compression: CompressionLevel,
    metadata: PngMetadata,
}

pub struct PngMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub timestamp: Option<SystemTime>,
    pub test_name: Option<String>,
}

impl PngExporter {
    pub fn export(&self, screenshot: &Screenshot) -> ProbarResult<Vec<u8>>;
    pub fn export_with_annotations(&self, screenshot: &Screenshot, annotations: &[Annotation]) -> ProbarResult<Vec<u8>>;
    pub fn save(&self, screenshot: &Screenshot, path: &Path) -> ProbarResult<()>;
}
```

---

#### Feature 3: SVG Screenshot Generation

**Priority:** P1 (High)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** Vector-based screenshots for resolution-independent documentation.

**Rust Crates Required:**
- `svg` (v0.17) - SVG generation
- `resvg` (v0.44) - SVG rendering (optional, for validation)

**API Design:**
```rust
pub struct SvgExporter {
    config: SvgConfig,
}

pub struct SvgConfig {
    pub viewbox: (u32, u32),
    pub preserve_aspect_ratio: bool,
    pub embed_fonts: bool,
    pub compression: SvgCompression,
}

impl SvgExporter {
    pub fn from_screenshot(&self, screenshot: &Screenshot) -> ProbarResult<String>;
    pub fn from_dom_tree(&self, dom: &DomSnapshot) -> ProbarResult<String>;
    pub fn add_annotations(&mut self, annotations: &[Annotation]) -> &mut Self;
}
```

---

#### Feature 4: MP4 Video Recording

**Priority:** P1 (High)
**Complexity:** High
**Pure Rust Feasibility:** ✅ Yes (with constraints)

**Description:** Record test execution as MP4 video for comprehensive test documentation.

**Rust Crates Required:**
- `openh264` (v0.6) - H.264 encoding (pure Rust)
- `mp4` (v0.14) - MP4 container format

**Note:** Full H.264/H.265 encoding in pure Rust has performance implications. Consider:
1. Frame-limited recording (30fps max)
2. Resolution constraints (1080p max)
3. Optional hardware acceleration detection

**API Design:**
```rust
pub struct VideoRecorder {
    encoder: H264Encoder,
    config: VideoConfig,
    frames: Vec<EncodedFrame>,
}

pub struct VideoConfig {
    pub fps: u8,
    pub width: u32,
    pub height: u32,
    pub bitrate: u32,
    pub codec: VideoCodec,
    pub max_duration_secs: u32,
}

impl VideoRecorder {
    pub fn start(&mut self) -> ProbarResult<()>;
    pub fn capture_frame(&mut self, screenshot: &Screenshot) -> ProbarResult<()>;
    pub fn stop(&mut self) -> ProbarResult<Vec<u8>>;
    pub fn save(&self, path: &Path) -> ProbarResult<()>;
}
```

---

### Category 2: CLI & Automation

#### Feature 5: Command-Line Interface

**Priority:** P0 (Critical)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** Full-featured CLI for running tests, generating reports, and project management.

**Rust Crates Required:**
- `clap` (v4.5) - CLI argument parsing
- `indicatif` (v0.17) - Progress bars
- `console` (v0.15) - Terminal formatting

**CLI Commands:**
```bash
probar test                     # Run all tests
probar test --filter "game::*"  # Filter tests
probar test --parallel 4        # Parallel execution
probar test --coverage          # With coverage
probar test --mutants            # With mutation testing

probar record <test> --gif      # Record as GIF
probar record <test> --mp4      # Record as MP4
probar record <test> --png      # Screenshots only

probar report --html            # Generate HTML report
probar report --junit           # Generate JUnit XML
probar report --lcov            # Generate LCOV coverage

probar init                     # Initialize new project
probar config                   # Manage configuration
```

**API Design:**
```rust
pub struct ProbarCli {
    config: CliConfig,
    runner: TestRunner,
    reporter: Reporter,
}

pub struct CliConfig {
    pub verbosity: Verbosity,
    pub color: ColorChoice,
    pub output_format: OutputFormat,
    pub parallel_jobs: usize,
}

impl ProbarCli {
    pub fn run(args: &[String]) -> ProbarResult<ExitCode>;
    pub fn run_tests(&self, filter: Option<&str>) -> ProbarResult<TestResults>;
    pub fn generate_report(&self, format: ReportFormat) -> ProbarResult<()>;
}
```

---

#### Feature 6: Watch Mode with Hot Reload

**Priority:** P1 (High)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** Automatic test re-execution on file changes.

**Rust Crates Required:**
- `notify` (v7.0) - File system watching
- `tokio` (v1.42) - Async runtime

**API Design:**
```rust
pub struct WatchMode {
    watcher: FileWatcher,
    debounce_ms: u64,
    patterns: Vec<GlobPattern>,
}

impl WatchMode {
    pub async fn start(&mut self, callback: impl Fn(&[PathBuf])) -> ProbarResult<()>;
    pub fn add_pattern(&mut self, pattern: &str) -> &mut Self;
    pub fn set_debounce(&mut self, ms: u64) -> &mut Self;
}
```

---

### Category 3: Network & Protocol

#### Feature 7: Network Request Interception

**Priority:** P0 (Critical)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes (via CDP)

**Description:** Intercept and modify network requests/responses for testing.

**Implementation:** Uses CDP's `Fetch` domain.

**API Design:**
```rust
pub struct NetworkInterceptor {
    patterns: Vec<UrlPattern>,
    handlers: HashMap<String, RequestHandler>,
}

pub trait RequestHandler: Send + Sync {
    fn handle(&self, request: &InterceptedRequest) -> InterceptAction;
}

pub enum InterceptAction {
    Continue,
    Modify(ModifiedRequest),
    Fulfill(MockResponse),
    Abort(AbortReason),
}

impl NetworkInterceptor {
    pub fn on_request(&mut self, pattern: &str, handler: impl RequestHandler);
    pub fn mock_response(&mut self, pattern: &str, response: MockResponse);
    pub fn block(&mut self, pattern: &str);
    pub fn throttle(&mut self, pattern: &str, kbps: u32);
}
```

---

#### Feature 8: WebSocket Monitoring

**Priority:** P1 (High)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes (via CDP)

**Description:** Monitor and mock WebSocket connections.

**API Design:**
```rust
pub struct WebSocketMonitor {
    connections: Vec<WebSocketConnection>,
    handlers: Vec<Box<dyn WebSocketHandler>>,
}

pub trait WebSocketHandler: Send + Sync {
    fn on_message(&self, msg: &WebSocketMessage) -> Option<WebSocketMessage>;
    fn on_close(&self, code: u16, reason: &str);
}

impl WebSocketMonitor {
    pub fn intercept(&mut self, url_pattern: &str, handler: impl WebSocketHandler);
    pub fn send_mock(&mut self, connection_id: u64, message: &[u8]) -> ProbarResult<()>;
    pub fn close(&mut self, connection_id: u64, code: u16) -> ProbarResult<()>;
}
```

---

### Category 4: Tracing & Debugging

#### Feature 9: Execution Tracing

**Priority:** P0 (Critical)
**Complexity:** High
**Pure Rust Feasibility:** ✅ Yes

**Description:** Comprehensive tracing of test execution for debugging.

**Rust Crates Required:**
- `tracing` (v0.1) - Structured logging
- `tracing-subscriber` (v0.3) - Trace collection

**API Design:**
```rust
pub struct ExecutionTracer {
    config: TracingConfig,
    spans: Vec<TracedSpan>,
    events: Vec<TracedEvent>,
}

pub struct TracingConfig {
    pub capture_screenshots: bool,
    pub capture_network: bool,
    pub capture_console: bool,
    pub capture_performance: bool,
}

pub struct TraceArchive {
    pub metadata: TraceMetadata,
    pub spans: Vec<TracedSpan>,
    pub screenshots: Vec<(u64, Screenshot)>,
    pub network_events: Vec<NetworkEvent>,
    pub console_logs: Vec<ConsoleMessage>,
}

impl ExecutionTracer {
    pub fn start(&mut self) -> ProbarResult<()>;
    pub fn stop(&mut self) -> ProbarResult<TraceArchive>;
    pub fn save(&self, path: &Path) -> ProbarResult<()>;
    pub fn export_html(&self) -> ProbarResult<String>;
}
```

---

#### Feature 10: Performance Profiling

**Priority:** P1 (High)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes (via CDP)

**Description:** Capture performance metrics during test execution.

**API Design:**
```rust
pub struct PerformanceProfiler {
    config: ProfilerConfig,
    metrics: PerformanceMetrics,
}

pub struct PerformanceMetrics {
    pub frame_times: Vec<Duration>,
    pub memory_usage: Vec<MemorySample>,
    pub cpu_usage: Vec<f64>,
    pub network_timing: Vec<NetworkTiming>,
    pub paint_events: Vec<PaintEvent>,
}

impl PerformanceProfiler {
    pub fn start(&mut self) -> ProbarResult<()>;
    pub fn stop(&mut self) -> ProbarResult<PerformanceReport>;
    pub fn get_fps(&self) -> f64;
    pub fn get_frame_budget_violations(&self, target_ms: f64) -> Vec<FrameViolation>;
}
```

---

### Category 5: Code Coverage & Analysis

#### Feature 11: LCOV Report Generation

**Priority:** P0 (Critical)
**Complexity:** Low
**Pure Rust Feasibility:** ✅ Yes

**Description:** Generate LCOV-format coverage reports for CI integration.

**API Design:**
```rust
pub struct LcovReporter {
    coverage_data: CoverageData,
}

impl LcovReporter {
    pub fn from_coverage(data: &CoverageData) -> Self;
    pub fn generate(&self) -> String;
    pub fn save(&self, path: &Path) -> ProbarResult<()>;
}

// LCOV format output:
// TN:<test name>
// SF:<source file>
// FN:<line>,<function name>
// FNDA:<execution count>,<function name>
// FNF:<functions found>
// FNH:<functions hit>
// DA:<line>,<execution count>
// LF:<lines found>
// LH:<lines hit>
// end_of_record
```

---

#### Feature 12: HTML Coverage Report

**Priority:** P0 (Critical)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** Interactive HTML coverage reports with source code highlighting.

**API Design:**
```rust
pub struct HtmlCoverageReporter {
    coverage_data: CoverageData,
    config: HtmlReportConfig,
}

pub struct HtmlReportConfig {
    pub title: String,
    pub highlight_uncovered: bool,
    pub include_branch_coverage: bool,
    pub theme: Theme,
}

impl HtmlCoverageReporter {
    pub fn generate(&self) -> ProbarResult<String>;
    pub fn save(&self, output_dir: &Path) -> ProbarResult<()>;
}
```

---

#### Feature 13: Cobertura XML Report

**Priority:** P1 (High)
**Complexity:** Low
**Pure Rust Feasibility:** ✅ Yes

**Description:** Cobertura-format coverage reports for CI tools (Jenkins, GitLab CI).

**API Design:**
```rust
pub struct CoberturaReporter {
    coverage_data: CoverageData,
}

impl CoberturaReporter {
    pub fn generate(&self) -> String;
    pub fn save(&self, path: &Path) -> ProbarResult<()>;
}
```

---

### Category 6: Browser Automation Enhancements

#### Feature 14: Multi-Browser Context Management

**Priority:** P1 (High)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes (via CDP)

**Description:** Manage multiple isolated browser contexts for parallel testing.

**API Design:**
```rust
pub struct BrowserContextManager {
    browser: Browser,
    contexts: HashMap<String, BrowserContext>,
}

pub struct BrowserContext {
    pub id: String,
    pub pages: Vec<Page>,
    pub storage: StorageState,
}

impl BrowserContextManager {
    pub async fn create_context(&mut self, name: &str) -> ProbarResult<&BrowserContext>;
    pub async fn close_context(&mut self, name: &str) -> ProbarResult<()>;
    pub async fn save_storage(&self, context: &str, path: &Path) -> ProbarResult<()>;
    pub async fn load_storage(&mut self, context: &str, path: &Path) -> ProbarResult<()>;
}
```

---

#### Feature 15: Device Emulation

**Priority:** P1 (High)
**Complexity:** Low
**Pure Rust Feasibility:** ✅ Yes (via CDP)

**Description:** Emulate mobile devices, screen sizes, and device capabilities.

**API Design:**
```rust
pub struct DeviceEmulator {
    presets: HashMap<String, DeviceDescriptor>,
}

pub struct DeviceDescriptor {
    pub name: String,
    pub viewport: Viewport,
    pub user_agent: String,
    pub device_scale_factor: f64,
    pub is_mobile: bool,
    pub has_touch: bool,
}

impl DeviceEmulator {
    pub fn iphone_14() -> DeviceDescriptor;
    pub fn ipad_pro() -> DeviceDescriptor;
    pub fn pixel_7() -> DeviceDescriptor;
    pub fn custom(viewport: Viewport, user_agent: &str) -> DeviceDescriptor;
}
```

---

#### Feature 16: Geolocation Mocking

**Priority:** P2 (Medium)
**Complexity:** Low
**Pure Rust Feasibility:** ✅ Yes (via CDP)

**Description:** Mock geolocation for location-based game testing.

**API Design:**
```rust
pub struct GeolocationMock {
    latitude: f64,
    longitude: f64,
    accuracy: f64,
}

impl GeolocationMock {
    pub fn set_position(lat: f64, lon: f64, accuracy: f64) -> Self;
    pub fn new_york() -> Self;
    pub fn tokyo() -> Self;
    pub fn london() -> Self;
    pub async fn apply(&self, page: &Page) -> ProbarResult<()>;
}
```

---

### Category 7: Assertions & Expectations

#### Feature 17: Soft Assertions

**Priority:** P1 (High)
**Complexity:** Low
**Pure Rust Feasibility:** ✅ Yes

**Description:** Collect multiple assertion failures without stopping test execution.

**API Design:**
```rust
pub struct SoftAssertions {
    failures: Vec<AssertionFailure>,
    mode: AssertionMode,
}

impl SoftAssertions {
    pub fn new() -> Self;
    pub fn assert_eq<T: PartialEq + Debug>(&mut self, actual: &T, expected: &T);
    pub fn assert_true(&mut self, condition: bool, message: &str);
    pub fn assert_visible(&mut self, locator: &Locator);
    pub fn verify(&self) -> ProbarResult<()>;
    pub fn failures(&self) -> &[AssertionFailure];
}
```

---

#### Feature 18: Retry Assertions with Polling

**Priority:** P0 (Critical)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** Auto-retrying assertions for eventually-consistent UI states.

**API Design:**
```rust
pub struct RetryAssertion<T> {
    assertion: Box<dyn Fn() -> AssertionResult>,
    timeout: Duration,
    poll_interval: Duration,
}

impl<T> RetryAssertion<T> {
    pub fn new(assertion: impl Fn() -> AssertionResult + 'static) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub fn with_poll_interval(self, interval: Duration) -> Self;
    pub async fn verify(&self) -> ProbarResult<()>;
}

// Usage:
// expect!(locator).to_be_visible().with_timeout(5000).verify().await?;
```

---

### Category 8: Advanced Testing Patterns

#### Feature 19: Page Object Model Support

**Priority:** P1 (High)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** First-class support for Page Object Model pattern.

**API Design:**
```rust
pub trait PageObject {
    fn url_pattern(&self) -> &str;
    async fn navigate(&self, page: &Page) -> ProbarResult<()>;
    async fn is_loaded(&self, page: &Page) -> bool;
}

#[derive(PageObject)]
pub struct LoginPage {
    #[locator("input[name='username']")]
    username_input: Locator,
    #[locator("input[name='password']")]
    password_input: Locator,
    #[locator("button[type='submit']")]
    submit_button: Locator,
}

impl LoginPage {
    pub async fn login(&self, username: &str, password: &str) -> ProbarResult<()> {
        self.username_input.fill(username).await?;
        self.password_input.fill(password).await?;
        self.submit_button.click().await?;
        Ok(())
    }
}
```

---

#### Feature 20: Fixture Management

**Priority:** P1 (High)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes

**Description:** Test fixture setup and teardown with dependency injection.

**API Design:**
```rust
pub trait Fixture: Send + Sync {
    async fn setup(&mut self) -> ProbarResult<()>;
    async fn teardown(&mut self) -> ProbarResult<()>;
}

pub struct FixtureManager {
    fixtures: HashMap<TypeId, Box<dyn Fixture>>,
    setup_order: Vec<TypeId>,
}

impl FixtureManager {
    pub fn register<F: Fixture + 'static>(&mut self, fixture: F);
    pub async fn setup_all(&mut self) -> ProbarResult<()>;
    pub async fn teardown_all(&mut self) -> ProbarResult<()>;
    pub fn get<F: Fixture + 'static>(&self) -> Option<&F>;
}

// Attribute macro for automatic fixture injection:
#[probar_test]
#[fixture(BrowserFixture)]
#[fixture(DatabaseFixture)]
async fn test_user_login(browser: &Browser, db: &Database) -> ProbarResult<()> {
    // Test implementation
}
```

---

## Category 7: EDD (Equation-Driven Development) Support

> **PRIORITY: P0 (CRITICAL)** - Required for simular EDD compliance
>
> These features enable "100% provable UX" for simulation interfaces as required
> by the EDD specification. See GitHub Issues #1-4 for detailed requirements.

#### Feature 21: TUI Testing for Ratatui Applications

**Priority:** P0 (Critical)
**Complexity:** High
**Pure Rust Feasibility:** ✅ Yes
**GitHub Issue:** #1

**Description:** Add TUI (Terminal User Interface) testing support for ratatui-based applications, enabling 100% provable UX for terminal simulations.

**Rust Crates Required:**
- `ratatui` (v0.28) - Terminal UI framework
- `crossterm` (v0.28) - Terminal manipulation

**API Design:**
```rust
/// TUI test backend for capturing frames
pub struct TuiTestBackend {
    buffer: Buffer,
    width: u16,
    height: u16,
}

/// TUI test assertions
#[macro_export]
macro_rules! assert_contains {
    ($frame:expr, $text:expr) => { ... };
}

#[macro_export]
macro_rules! assert_matches {
    ($frame:expr, $pattern:expr) => { ... };
}

#[macro_export]
macro_rules! send_key {
    ($tui:expr, $key:expr) => { ... };
}

/// TUI snapshot testing
pub struct TuiSnapshot {
    frames: Vec<Buffer>,
}

impl TuiSnapshot {
    pub fn capture(&mut self, backend: &TuiTestBackend);
    pub fn assert_matches(&self, golden: &Path) -> ProbarResult<()>;
    pub fn save_golden(&self, path: &Path) -> ProbarResult<()>;
}
```

**Toyota Way Application:**
- **Poka-Yoke**: Type-safe frame capture with Buffer type
- **Muda**: Direct buffer comparison without rendering overhead
- **Genchi Genbutsu**: TestBackend reflects actual terminal behavior

---

#### Feature 22: Equation Verification Assertions

**Priority:** P0 (Critical)
**Complexity:** Medium
**Pure Rust Feasibility:** ✅ Yes
**GitHub Issue:** #2

**Description:** Specialized assertions for verifying simulation UIs correctly display mathematical equations, EMC references, and live computed values.

**API Design:**
```rust
/// Assert equation text is displayed (Unicode math support)
#[macro_export]
macro_rules! assert_equation {
    ($frame:expr, $equation:expr) => { ... };
}

/// Assert EMC (Equation Model Card) reference is visible
#[macro_export]
macro_rules! assert_emc_ref {
    ($frame:expr, $ref:expr) => { ... };
}

/// Track value changes across frames
pub struct ValueTracker {
    name: String,
    history: Vec<f64>,
}

impl ValueTracker {
    pub fn new(name: &str) -> Self;
    pub fn record(&mut self, value: f64);
    pub fn assert_changed(&self) -> ProbarResult<()>;
    pub fn assert_decreasing(&self) -> ProbarResult<()>;
    pub fn assert_increasing(&self) -> ProbarResult<()>;
}

/// Assert units are displayed correctly
#[macro_export]
macro_rules! assert_unit {
    ($frame:expr, $quantity:expr, $unit:expr) => { ... };
}

/// Assert falsification criterion status
#[macro_export]
macro_rules! assert_criterion_status {
    ($frame:expr, $criterion:expr, passed: $passed:expr) => { ... };
}
```

**Toyota Way Application:**
- **Jidoka**: Fail-fast on equation mismatch
- **Poka-Yoke**: Unicode normalization prevents false negatives
- **Kaizen**: ValueTracker enables trend analysis

---

#### Feature 23: Deterministic Replay for Simulation UX

**Priority:** P0 (Critical)
**Complexity:** High
**Pure Rust Feasibility:** ✅ Yes
**GitHub Issue:** #3

**Description:** Deterministic replay support ensuring same seed produces identical UI states.

**Rust Crates Required:**
- `serde_yaml` (v0.9) - YAML config loading
- `sha2` (v0.10) - State hashing

**API Design:**
```rust
/// Load simulation config from YAML (EDD requirement)
#[macro_export]
macro_rules! load_yaml {
    ($path:expr) => { ... };
}

/// Record user interactions for replay
pub struct SessionRecorder {
    events: Vec<RecordedEvent>,
    start_time: Instant,
}

#[derive(Serialize, Deserialize)]
pub struct RecordedEvent {
    timestamp_ms: u64,
    event: InputEvent,
}

impl SessionRecorder {
    pub fn new() -> Self;
    pub fn record_key(&mut self, key: KeyCode);
    pub fn record_mouse(&mut self, x: u16, y: u16, button: MouseButton);
    pub fn save(&self, path: &Path) -> ProbarResult<()>;
    pub fn load(path: &Path) -> ProbarResult<Self>;
}

/// Replay recorded session
pub fn replay_session<T: TuiApp>(
    app: &mut T,
    session: &SessionRecorder,
) -> ProbarResult<()>;

/// Frame sequence for golden file comparison
pub struct FrameSequence {
    frames: Vec<Buffer>,
    timestamps: Vec<u64>,
}

impl FrameSequence {
    pub fn capture(&mut self, backend: &TuiTestBackend);
    pub fn assert_matches(&self, golden: &Path) -> ProbarResult<()>;
}

/// Hash state for cross-platform verification
pub fn hash_state<T: Serialize>(state: &T) -> String;

/// Assert bitwise identical frames
#[macro_export]
macro_rules! assert_identical {
    ($frame1:expr, $frame2:expr) => { ... };
}
```

**Toyota Way Application:**
- **Heijunka**: Consistent frame timing in replay
- **Genchi Genbutsu**: YAML configs reflect actual experiment setup
- **Jidoka**: Hash verification catches non-determinism

---

#### Feature 24: 100% UX Coverage Metrics

**Priority:** P0 (Critical)
**Complexity:** High
**Pure Rust Feasibility:** ✅ Yes
**GitHub Issue:** #4

**Description:** UX coverage metrics enabling verification that 100% of simulation UI elements are tested.

**API Design:**
```rust
/// Track UX element coverage
pub struct UxCoverage {
    elements: HashMap<String, bool>,
    required_elements: Vec<String>,
}

impl UxCoverage {
    pub fn track(elements: &[&str]) -> Self;
    pub fn mark_tested(&mut self, element: &str);
    pub fn assert_complete(&self) -> ProbarResult<()>;
    pub fn coverage_percent(&self) -> f64;
}

/// Track interaction coverage
pub struct InteractionCoverage {
    interactions: HashMap<String, bool>,
}

impl InteractionCoverage {
    pub fn track(interactions: &[(&str, &str)]) -> Self;  // (key, action)
    pub fn test_key(&mut self, key: KeyCode, action: &str) -> ProbarResult<()>;
    pub fn assert_complete(&self) -> ProbarResult<()>;
}

/// Track equation coverage from EMC
pub struct EquationCoverage {
    equations: HashMap<String, bool>,
    criteria: HashMap<String, bool>,
}

impl EquationCoverage {
    pub fn from_emc(emc_path: &Path) -> ProbarResult<Self>;
    pub fn verify_equation(&mut self, name: &str, expected: &str) -> ProbarResult<()>;
    pub fn verify_criterion(&mut self, name: &str) -> ProbarResult<()>;
    pub fn assert_complete(&self) -> ProbarResult<()>;
}

/// Coverage report generation
pub struct UxCoverageReport {
    pub element_coverage: f64,
    pub interaction_coverage: f64,
    pub equation_coverage: f64,
    pub total_coverage: f64,
}

impl UxCoverageReport {
    pub fn generate(
        ux: &UxCoverage,
        interaction: &InteractionCoverage,
        equation: &EquationCoverage,
    ) -> Self;

    pub fn save_html(&self, path: &Path) -> ProbarResult<()>;
    pub fn save_json(&self, path: &Path) -> ProbarResult<()>;
    pub fn generate_badge(&self, path: &Path) -> ProbarResult<()>;
}

/// Attribute for enforcing coverage threshold
#[proc_macro_attribute]
pub fn coverage(threshold: u8) { ... }
```

**Toyota Way Application:**
- **Jidoka**: Automatic enforcement of 100% coverage
- **Mieruka**: Visual coverage reports
- **Kaizen**: Coverage trends over time

---

## Implementation Roadmap

### Phase 0: EDD Compliance (PRIORITY - NOW)
> **BLOCKING**: Required for simular EDD integration

- Feature 21: TUI Testing for Ratatui (Issue #1)
- Feature 22: Equation Verification Assertions (Issue #2)
- Feature 23: Deterministic Replay (Issue #3)
- Feature 24: 100% UX Coverage Metrics (Issue #4)

### Phase 1: Core Media Generation ✅ COMPLETED
- Feature 1: GIF Recording ✅
- Feature 2: PNG Export ✅
- Feature 5: CLI (basic) ✅

### Phase 2: Coverage & Reporting ✅ COMPLETED
- Feature 11: LCOV Reports ✅
- Feature 12: HTML Coverage ✅
- Feature 13: Cobertura XML ✅

### Phase 3: Advanced Media ✅ COMPLETED
- Feature 3: SVG Export ✅
- Feature 4: MP4 Recording ✅
- Feature 9: Execution Tracing (pending)

### Phase 4: Browser Enhancements (PARTIAL)
- Feature 7: Network Interception (pending)
- Feature 14: Context Management (pending)
- Feature 15: Device Emulation ✅
- Feature 16: Geolocation Mocking ✅

### Phase 5: Testing Patterns ✅ COMPLETED
- Feature 17: Soft Assertions ✅
- Feature 18: Retry Assertions ✅
- Feature 19: Page Object Model ✅
- Feature 20: Fixture Management ✅

### Phase 6: Advanced Features (pending)
- Feature 6: Watch Mode
- Feature 8: WebSocket Monitoring
- Feature 10: Performance Profiling

---

## Peer-Reviewed Academic Citations

### Foundational Testing Research

1. Claessen, K., & Hughes, J. (2000). **QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs**. In *Proceedings of the Fifth ACM SIGPLAN International Conference on Functional Programming (ICFP '00)*, pp. 268-279. ACM. https://dl.acm.org/doi/10.1145/357766.351266

2. Goldstein, H., et al. (2024). **Property-Based Testing in Practice**. In *Proceedings of the IEEE/ACM 46th International Conference on Software Engineering*. https://dl.acm.org/doi/10.1145/3597503.3639581

3. Löscher, A., & Sagonas, K. (2017). **Targeted Property-Based Testing**. In *Proceedings of the 26th ACM SIGSOFT International Symposium on Software Testing and Analysis*, pp. 46-56. ACM. https://dl.acm.org/doi/10.1145/3092703.3092711

### Mutation Testing

4. Jia, Y., & Harman, M. (2011). **An Analysis and Survey of the Development of Mutation Testing**. *IEEE Transactions on Software Engineering*, 37(5), 649-678. https://dl.acm.org/doi/10.1109/TSE.2010.62

5. Coles, H., Laurent, T., Henard, C., Papadakis, M., & Ventresque, A. (2016). **PIT: A Practical Mutation Testing Tool for Java**. In *Proceedings of the 25th International Symposium on Software Testing and Analysis (ISSTA)*. https://dl.acm.org/doi/10.1145/2931037.2948707

6. Gopinath, R., et al. (2022). **Mutation Testing in Evolving Systems: Studying the Relevance of Mutants to Code Evolution**. *ACM Transactions on Software Engineering and Methodology*. https://dl.acm.org/doi/10.1145/3530786

### Browser Automation & Web Testing

7. Leotta, M., et al. (2024). **Exploring Browser Automation: A Comparative Study of Selenium, Cypress, Puppeteer, and Playwright**. In *QUATIC 2024*, Springer. https://link.springer.com/chapter/10.1007/978-3-031-70245-7_10

8. Bajammal, M., & Mesbah, A. (2021). **Semantic Web Accessibility Testing via Hierarchical Visual Analysis**. In *2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)*, pp. 1610-1621.

9. Wang, W., et al. (2024). **Enhancing the Resiliency of Automated Web Tests with Natural Language**. In *ACM International Conference on AI, Automation and Algorithms*. https://dl.acm.org/doi/10.1145/3700523.3700536

### Visual GUI Testing

10. Alegroth, E., & Feldt, R. (2017). **Visual GUI Testing in Continuous Integration Environment**. *IEEE Conference Publication*. https://ieeexplore.ieee.org/document/7910301/

11. Yusop, N. S. M., et al. (2018). **Continuous Integration and Visual GUI Testing: Benefits and Drawbacks in Industrial Practice**. *IEEE Conference Publication*. https://ieeexplore.ieee.org/document/8367046/

12. Chang, T. H., et al. (2010). **GUI Testing Using Computer Vision**. In *CHI '10: Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*, pp. 1535-1544.

### WebAssembly Testing

13. Haas, A., et al. (2017). **Bringing the Web up to Speed with WebAssembly**. In *Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation*, pp. 185-200. https://dl.acm.org/doi/10.1145/3062341.3062363

14. Hilbig, A., Lehmann, D., & Pradel, M. (2024). **Wasm-R3: Record-Reduce-Replay for Realistic and Standalone WebAssembly Benchmarks**. In *Proceedings of the ACM on Programming Languages (OOPSLA)*. https://dl.acm.org/doi/10.1145/3689787

15. Romano, A., et al. (2024). **Research on WebAssembly Runtimes: A Survey**. *ACM Transactions on Software Engineering and Methodology*. https://dl.acm.org/doi/10.1145/3714465

### Fuzzing & Security Testing

16. Manes, V., et al. (2021). **The Art, Science, and Engineering of Fuzzing: A Survey**. *IEEE Transactions on Software Engineering*, 47(11), 2312-2331.

17. Fioraldi, A., et al. (2020). **AFL++: Combining Incremental Steps of Fuzzing Research**. In *14th USENIX Workshop on Offensive Technologies (WOOT'20)*.

18. Fioraldi, A., et al. (2022). **LibAFL: A Framework to Build Modular and Reusable Fuzzers**. In *ACM SIGSAC Conference on Computer and Communications Security*. https://dl.acm.org/doi/10.1145/3548606.3560602

### Code Coverage

19. Inozemtseva, L., & Holmes, R. (2014). **Coverage is Not Strongly Correlated with Test Suite Effectiveness**. In *Proceedings of the 36th International Conference on Software Engineering*, pp. 435-445.

20. Gopinath, R., Jensen, C., & Groce, A. (2015). **Code Coverage and Test Suite Effectiveness: Empirical Study with Real Bugs in Large Systems**. *IEEE Conference Publication*. https://ieeexplore.ieee.org/document/7081877/

### Accessibility Testing

21. Bajammal, M., & Mesbah, A. (2024). **Enhancing Web Accessibility: Automated Detection of Issues with Generative AI**. *ACM Proceedings on Software Engineering*. https://dl.acm.org/doi/10.1145/3729371

22. Vigo, M., Brown, J., & Conway, V. (2013). **Benchmarking Web Accessibility Evaluation Tools**. In *Proceedings of the 10th International Cross-Disciplinary Conference on Web Accessibility (W4A)*. https://dl.acm.org/doi/10.1145/2461121.2461124

### Toyota Production System in Software

23. Poppendieck, M., & Poppendieck, T. (2003). **Lean Software Development: An Agile Toolkit**. Addison-Wesley Professional.

24. Liker, J. K. (2004). **The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer**. McGraw-Hill.

25. Hibbs, C., Jewett, S., & Sullivan, M. (2009). **The Art of Lean Software Development: A Practical and Incremental Approach**. O'Reilly Media.

### Rust Safety & Formal Methods (Poka-Yoke)

26. Jung, R., Jourdan, J. H., Krebbers, R., & Dreyer, D. (2017). **RustBelt: Securing the Foundations of the Rust Programming Language**. In *Proceedings of the ACM on Programming Languages (POPL)*, 2(POPL), 66. https://dl.acm.org/doi/10.1145/3158154 (The ultimate Poka-Yoke: formal proof of safety).

27. Matsakis, N. D., & Klock, F. S. (2014). **The Rust Language**. In *ACM SIGAda Ada Letters*, 34(3), 103-104. https://dl.acm.org/doi/10.1145/2692956.2663188

28. Balasubramanian, A., Baranowski, M. S., Burtsev, A., Panda, A., Rakamarić, Z., & Ryzhyk, L. (2017). **System Programming in Rust: Beyond Safety**. In *Proceedings of the 16th Workshop on Hot Topics in Operating Systems (HotOS)*, pp. 156-161. https://dl.acm.org/doi/10.1145/3102980.3103006

29. Lamport, L. (2002). **Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers**. Addison-Wesley. (Formal specification as mistake-proofing).

### Lean DevOps & Continuous Quality (Jidoka/Kaizen)

30. Forsgren, N., Humble, J., & Kim, G. (2018). **Accelerate: The Science of Lean Software and DevOps: Building and Scaling High Performing Technology Organizations**. IT Revolution Press. (Scientific link between Lean principles and software performance).

31. Humble, J., & Farley, D. (2010). **Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation**. Addison-Wesley. (Foundational text on the "Andon Cord" in software pipelines).

32. Fowler, M. (2006). **Continuous Integration**. *martinfowler.com*. https://martinfowler.com/articles/continuousIntegration.html (Early definition of automated quality gates).

33. Saff, D., & Ernst, M. D. (2004). **An Experimental Evaluation of Continuous Testing During Development**. In *Proceedings of the 2004 International Symposium on Software Testing and Analysis (ISSTA)*, pp. 76-85. https://dl.acm.org/doi/10.1145/1007512.1007523

34. Memon, A., et al. (2017). **Taming Google-Scale Continuous Testing**. In *Proceedings of the 39th International Conference on Software Engineering (ICSE)*, pp. 233-243. https://dl.acm.org/doi/10.1109/ICSE-SEIP.2017.26

35. Bell, J., Legunsen, O., Hilton, M., Eloussi, L., Yung, T., & Marinov, D. (2018). **DeFlaker: Automatically Detecting Flaky Tests**. In *Proceedings of the 40th International Conference on Software Engineering (ICSE)*, pp. 433-444. https://dl.acm.org/doi/10.1145/3180155.3180164 (Crucial for maintaining trust in the Andon cord).

---

## Quality Requirements (EXTREME TDD)

### Test Coverage Requirements

| Module | Line Coverage | Branch Coverage | Mutation Score |
|--------|--------------|-----------------|----------------|
| gif_recorder | ≥95% | ≥90% | ≥80% |
| png_exporter | ≥95% | ≥90% | ≥80% |
| svg_exporter | ≥95% | ≥90% | ≥80% |
| video_recorder | ≥95% | ≥90% | ≥80% |
| cli | ≥95% | ≥90% | ≥80% |
| network_interceptor | ≥95% | ≥90% | ≥80% |
| execution_tracer | ≥95% | ≥90% | ≥80% |
| lcov_reporter | ≥95% | ≥90% | ≥80% |
| All modules | ≥95% | ≥90% | ≥80% |

### Property-Based Testing Requirements

Each feature must include property-based tests using `proptest`:

```rust
proptest! {
    #[test]
    fn prop_gif_frames_preserve_dimensions(width in 1u32..4096, height in 1u32..4096) {
        let config = GifConfig { width, height, ..Default::default() };
        let recorder = GifRecorder::new(config);
        // Property: output dimensions match input
        prop_assert_eq!(recorder.config().width, width);
        prop_assert_eq!(recorder.config().height, height);
    }
}
```

### Mutation Testing Requirements

All features must pass mutation testing with cargo-mutants:

```bash
cargo mutants --package probar --minimum-score 0.80
```

---

## Appendix A: Rust Crate Dependencies

| Feature | Crate | Version | Purpose |
|---------|-------|---------|---------|
| GIF | `gif` | 0.13 | GIF encoding |
| GIF | `image` | 0.25 | Image processing |
| PNG | `png` | 0.17 | PNG encoding |
| SVG | `svg` | 0.17 | SVG generation |
| Video | `openh264` | 0.6 | H.264 encoding |
| Video | `mp4` | 0.14 | MP4 container |
| CLI | `clap` | 4.5 | Argument parsing |
| CLI | `indicatif` | 0.17 | Progress bars |
| CLI | `console` | 0.15 | Terminal formatting |
| Watch | `notify` | 7.0 | File watching |
| Tracing | `tracing` | 0.1 | Structured logging |
| Async | `tokio` | 1.42 | Async runtime |

---

## Appendix B: Non-Goals (Features Requiring JavaScript)

The following features from Puppeteer/Playwright are **explicitly out of scope** as they require JavaScript:

1. **JavaScript Execution** - `page.evaluate()` with JS strings
2. **JavaScript Coverage** - V8's built-in JS coverage
3. **Service Worker Interception** - Requires JS context
4. **Web Workers** - JS-based worker threads
5. **IndexedDB Manipulation** - Browser JS API
6. **LocalStorage Direct Access** - Requires JS context

These features would violate the **ABSOLUTE ZERO JavaScript** constraint.

---

## Appendix C: Runnable Examples (`cargo run --example`)

All features must include runnable examples demonstrating real-world usage patterns.

### Required Examples

| Example | Feature | Description | Command |
|---------|---------|-------------|---------|
| `basic_test` | Core | Simple test execution | `cargo run --example basic_test` |
| `gif_recording` | Feature 1 | Record test as GIF | `cargo run --example gif_recording` |
| `png_screenshot` | Feature 2 | Capture PNG screenshots | `cargo run --example png_screenshot` |
| `svg_export` | Feature 3 | Generate SVG screenshots | `cargo run --example svg_export` |
| `video_recording` | Feature 4 | Record MP4 video | `cargo run --example video_recording` |
| `watch_mode` | Feature 6 | Hot reload on file changes | `cargo run --example watch_mode` |
| `network_intercept` | Feature 7 | Mock network requests | `cargo run --example network_intercept` |
| `websocket_monitor` | Feature 8 | Monitor WebSocket traffic | `cargo run --example websocket_monitor` |
| `execution_trace` | Feature 9 | Generate execution trace | `cargo run --example execution_trace` |
| `performance_profile` | Feature 10 | Profile performance metrics | `cargo run --example performance_profile` |
| `lcov_report` | Feature 11 | Generate LCOV coverage | `cargo run --example lcov_report` |
| `html_coverage` | Feature 12 | Generate HTML coverage | `cargo run --example html_coverage` |
| `cobertura_report` | Feature 13 | Generate Cobertura XML | `cargo run --example cobertura_report` |
| `multi_context` | Feature 14 | Parallel browser contexts | `cargo run --example multi_context` |
| `device_emulation` | Feature 15 | Mobile device emulation | `cargo run --example device_emulation` |
| `geolocation_mock` | Feature 16 | Mock GPS location | `cargo run --example geolocation_mock` |
| `soft_assertions` | Feature 17 | Collect multiple failures | `cargo run --example soft_assertions` |
| `retry_assertions` | Feature 18 | Auto-retry assertions | `cargo run --example retry_assertions` |
| `page_object` | Feature 19 | Page Object Model | `cargo run --example page_object` |
| `fixtures` | Feature 20 | Test fixture management | `cargo run --example fixtures` |
| `tui_testing` | Feature 21 | TUI/Ratatui testing | `cargo run --example tui_testing` |
| `equation_verify` | Feature 22 | Equation verification | `cargo run --example equation_verify` |
| `deterministic_replay` | Feature 23 | Replay recorded sessions | `cargo run --example deterministic_replay` |
| `ux_coverage` | Feature 24 | 100% UX coverage | `cargo run --example ux_coverage` |

### Example Template

Each example must follow this structure:

```rust
//! Example: {Feature Name}
//!
//! Demonstrates: {Brief description}
//!
//! Run with: `cargo run --example {name}`
//!
//! Toyota Way: {Principle applied}

use probar::prelude::*;

fn main() -> ProbarResult<()> {
    // 1. Setup
    println!("=== {Feature Name} Example ===\n");

    // 2. Demonstrate feature
    // ... implementation ...

    // 3. Show results
    println!("\n✅ Example completed successfully!");
    Ok(())
}
```

### Example Quality Requirements

- **Compilable**: All examples must compile with `cargo build --examples`
- **Runnable**: All examples must execute without external dependencies
- **Self-contained**: No network or file system requirements beyond temp files
- **Documented**: Each example must have doc comments explaining usage
- **Tested**: Examples are tested in CI via `cargo test --examples`

---

## Appendix D: Book Documentation Updates

The mdBook documentation must be updated to cover all implemented features.

### New Book Chapters Required

| Chapter | Location | Content |
|---------|----------|---------|
| Media Recording | `src/probar/media-recording.md` | GIF, PNG, SVG, MP4 recording |
| Watch Mode | `src/probar/watch-mode.md` | Hot reload development |
| Network Interception | `src/probar/network-interception.md` | Request mocking |
| WebSocket Testing | `src/probar/websocket-testing.md` | WebSocket monitoring |
| Execution Tracing | `src/probar/execution-tracing.md` | Debug trace generation |
| Performance Profiling | `src/probar/performance-profiling.md` | FPS, memory, CPU metrics |
| Coverage Reports | `src/probar/coverage-reports.md` | LCOV, HTML, Cobertura |
| Browser Contexts | `src/probar/browser-contexts.md` | Parallel isolated contexts |
| Device Emulation | `src/probar/device-emulation.md` | Mobile device testing |
| Assertions Guide | `src/probar/assertions-guide.md` | Soft, retry, equation assertions |
| Page Objects | `src/probar/page-objects.md` | POM pattern guide |
| Fixtures | `src/probar/fixtures.md` | Test fixture management |
| TUI Testing | `src/probar/tui-testing.md` | Terminal UI testing |
| UX Coverage | `src/probar/ux-coverage.md` | 100% UX coverage metrics |

### Updated SUMMARY.md Structure

```markdown
# Summary

[Introduction](./introduction.md)

# User Guide

- [Why Probar?](./probar/why-probar.md)
- [Quick Start](./probar/quick-start.md)
- [Overview](./probar/overview.md)

# Core Concepts

- [Locators](./probar/locators.md)
- [Assertions](./probar/assertions.md)
  - [Soft Assertions](./probar/assertions-soft.md)
  - [Retry Assertions](./probar/assertions-retry.md)
  - [Equation Verification](./probar/assertions-equation.md)
- [Simulation](./probar/simulation.md)
- [Deterministic Replay](./probar/deterministic-replay.md)

# Media & Recording

- [Media Recording](./probar/media-recording.md)
  - [GIF Recording](./probar/media-gif.md)
  - [PNG Screenshots](./probar/media-png.md)
  - [SVG Export](./probar/media-svg.md)
  - [MP4 Video](./probar/media-mp4.md)

# Network & Protocol

- [Network Interception](./probar/network-interception.md)
- [WebSocket Testing](./probar/websocket-testing.md)

# Browser Automation

- [Browser Contexts](./probar/browser-contexts.md)
- [Device Emulation](./probar/device-emulation.md)
- [Geolocation Mocking](./probar/geolocation-mocking.md)

# Testing Patterns

- [Page Objects](./probar/page-objects.md)
- [Fixtures](./probar/fixtures.md)
- [TUI Testing](./probar/tui-testing.md)

# Analysis & Debugging

- [Execution Tracing](./probar/execution-tracing.md)
- [Performance Profiling](./probar/performance-profiling.md)
- [Watch Mode](./probar/watch-mode.md)

# Coverage & Reports

- [Coverage Tooling](./probar/coverage-tooling.md)
- [LCOV Reports](./probar/coverage-lcov.md)
- [HTML Reports](./probar/coverage-html.md)
- [Cobertura XML](./probar/coverage-cobertura.md)
- [UX Coverage](./probar/ux-coverage.md)

# Advanced Topics

- [Fuzzing](./probar/fuzzing.md)
- [Accessibility Testing](./probar/accessibility.md)

# Reference

- [API Reference](./probar/api-reference.md)
- [CLI Reference](./probar/cli-reference.md)
- [Configuration](./probar/configuration.md)
```

### Chapter Template

Each new chapter must follow this template:

```markdown
# {Feature Name}

> **Toyota Way**: {Principle} - {Brief explanation}

## Overview

{2-3 sentence description of the feature}

## Quick Start

\`\`\`rust
use probar::prelude::*;

// Minimal working example
\`\`\`

## API Reference

### {MainStruct}

\`\`\`rust
pub struct {MainStruct} {
    // fields...
}

impl {MainStruct} {
    pub fn new() -> Self;
    // key methods...
}
\`\`\`

## Examples

### Basic Usage

\`\`\`rust
// Example code
\`\`\`

### Advanced Usage

\`\`\`rust
// More complex example
\`\`\`

## Best Practices

1. {Practice 1}
2. {Practice 2}
3. {Practice 3}

## See Also

- [Related Feature](./related.md)
- [API Docs](https://docs.rs/probar)
```

---

## Appendix E: 100-Point QA Checklist

### Quality Gate: All items must pass before release

#### Section 1: Build & Compilation (10 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 1 | Library compiles | `cargo build --lib` | x |
| 2 | All crates compile | `cargo build --workspace` | x |
| 3 | Release build succeeds | `cargo build --release` | x |
| 4 | WASM target compiles | `cargo build --target wasm32-unknown-unknown` | ☐ |
| 5 | No compiler warnings | `cargo build 2>&1 \| grep -c warning` = 0 | ☐ |
| 6 | Examples compile | `cargo build --examples` | x |
| 7 | Benchmarks compile | `cargo build --benches` | x |
| 8 | Documentation compiles | `cargo doc --no-deps` | x |
| 9 | CLI binary builds | `cargo build -p probar-cli` | x |
| 10 | All features compile | `cargo build --all-features` | x |

#### Section 2: Test Suite (20 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 11 | Unit tests pass | `cargo test --lib` | x |
| 12 | Integration tests pass | `cargo test --test '*'` | x |
| 13 | Doc tests pass | `cargo test --doc` | x |
| 14 | All tests pass | `cargo test --workspace` | x |
| 15 | No ignored tests | `cargo test -- --ignored` (all pass) | ☐ |
| 16 | Tests run in parallel | `cargo test -- --test-threads=4` | x |
| 17 | No flaky tests | Run tests 3x, all pass each time | x |
| 18 | Example tests pass | `cargo test --examples` | x |
| 19 | Property tests pass | `cargo test proptest` | x |
| 20 | Test count ≥1000 | `cargo test 2>&1 \| grep passed` | x |
| 21 | All features tested | Each feature has ≥10 tests | x |
| 22 | Edge cases covered | Boundary conditions tested | x |
| 23 | Error paths tested | All error variants tested | x |
| 24 | No test warnings | `RUSTFLAGS=-Dwarnings cargo test` | x |
| 25 | Tests complete <60s | `time cargo test` | x |
| 26 | No test output noise | Tests use `#[should_panic]` correctly | x |
| 27 | Async tests work | `tokio::test` tests pass | x |
| 28 | Mock tests work | All mocks validate correctly | x |
| 29 | Snapshot tests work | Golden files match | x |
| 30 | Regression tests exist | Bug fixes have tests | x |

#### Section 3: Code Quality (15 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 31 | Clippy passes | `cargo clippy -- -D warnings` | ☐ |
| 32 | Format check passes | `cargo fmt -- --check` | ☐ |
| 33 | No unsafe code | `grep -r "unsafe" --include="*.rs"` (allowed list only) | ☐ |
| 34 | No unwrap in lib | `grep -r "\.unwrap()" src/` (0 in non-test) | ☐ |
| 35 | No panic in lib | `grep -r "panic!" src/` (0 in non-test) | ☐ |
| 36 | No todo comments | `grep -r "TODO\|FIXME" src/` = 0 | x |
| 37 | No dead code | `cargo build 2>&1 \| grep "dead_code"` = 0 | ☐ |
| 38 | No unused imports | `cargo build 2>&1 \| grep "unused_import"` = 0 | ☐ |
| 39 | Consistent naming | snake_case functions, CamelCase types | ☐ |
| 40 | Module structure clean | Max 3 levels of nesting | ☐ |
| 41 | Error handling proper | All errors use ProbarError | ☐ |
| 42 | No magic numbers | Constants are named | ☐ |
| 43 | Functions ≤50 lines | No oversized functions | ☐ |
| 44 | Cyclomatic complexity ≤10 | No overly complex functions | ☐ |
| 45 | No code duplication | DRY principle followed | ☐ |

#### Section 4: Coverage & Mutation (15 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 46 | Line coverage ≥95% | `cargo llvm-cov --lcov` | ☐ |
| 47 | Branch coverage ≥90% | `cargo llvm-cov --branch` | x |
| 48 | Function coverage 100% | All public functions tested | ☐ |
| 49 | Mutation score ≥80% | `cargo mutants --minimum-score 0.80` | ☐ |
| 50 | No trivial mutants | All mutations are meaningful | ☐ |
| 51 | Coverage report gen | `cargo llvm-cov --html` succeeds | x |
| 52 | LCOV report valid | LCOV file parses correctly | x |
| 53 | Cobertura report valid | XML validates against schema | x |
| 54 | Coverage trending up | Compare to previous release | x |
| 55 | New code ≥95% covered | Changed files meet threshold | x |
| 56 | Critical paths 100% | Core logic fully covered | x |
| 57 | Error paths covered | All error branches tested | x |
| 58 | Edge cases covered | Boundary conditions tested | x |
| 59 | Integration coverage | Cross-module paths covered | x |
| 60 | No coverage gaps | Continuous coverage map | x |

#### Section 5: Documentation (10 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 61 | All public items documented | `cargo doc 2>&1 \| grep "missing"` = 0 | ☐ |
| 62 | README complete | All features documented | x |
| 63 | CHANGELOG updated | Version changes documented | x |
| 64 | Examples in docs | All APIs have examples | x |
| 65 | Book builds | `mdbook build` succeeds | x |
| 66 | Book has no broken links | `mdbook test` passes | x |
| 67 | API docs complete | All modules documented | x |
| 68 | CLI help complete | `probar --help` shows all commands | x |
| 69 | Error messages clear | All errors have actionable messages | x |
| 70 | Version numbers correct | Cargo.toml matches release | x |

#### Section 6: Examples (10 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 71 | All examples compile | `cargo build --examples` | x |
| 72 | All examples run | Each example executes | x |
| 73 | Examples are documented | Each has header comments | x |
| 74 | Examples are idiomatic | Follow Rust best practices | x |
| 75 | Examples cover all features | 24 feature examples exist | x |
| 76 | Examples are self-contained | No external dependencies | x |
| 77 | Examples show best practices | Toyota Way applied | x |
| 78 | Examples include error handling | Proper Result usage | x |
| 79 | Examples are testable | Can run in CI | x |
| 80 | Examples match docs | Book examples match code | x |

#### Section 7: CI/CD & Release (10 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 81 | CI pipeline passes | GitHub Actions green | x |
| 82 | All platforms build | Linux, macOS, Windows | x |
| 83 | WASM CI passes | wasm32 target in CI | ☐ |
| 84 | Cargo.toml valid | `cargo package --list` succeeds | x |
| 85 | No unpublished deps | All deps on crates.io | x |
| 86 | Version bumped | Semantic versioning followed | x |
| 87 | License correct | MIT/Apache-2.0 | x |
| 88 | Keywords set | Cargo.toml has keywords | x |
| 89 | Categories set | Cargo.toml has categories | x |
| 90 | Publish dry-run passes | `cargo publish --dry-run` | ☐ |

#### Section 8: Performance & Security (10 points)

| # | Check | Command | Pass |
|---|-------|---------|------|
| 91 | No memory leaks | `cargo valgrind test` passes | x |
| 92 | No stack overflow | Deep recursion tested | x |
| 93 | Benchmarks pass | `cargo bench` succeeds | x |
| 94 | No perf regressions | Compare to baseline | x |
| 95 | Dependencies audited | `cargo audit` clean | ☐ |
| 96 | No vulnerable deps | `cargo deny check` passes | ☐ |
| 97 | WASM size <2MB | Binary size check | x |
| 98 | Startup time <500ms | Mobile target check | x |
| 99 | 60 FPS maintained | Frame budget check | x |
| 100 | No GC pauses | Zero allocation hot paths | x |

### QA Gate Summary

| Section | Points | Required | Status |
|---------|--------|----------|--------|
| Build & Compilation | 8 | 10/10 | ☐ |
| Test Suite | 19 | 18/20 | x |
| Code Quality | 1 | 14/15 | ☐ |
| Coverage & Mutation | 10 | 13/15 | ☐ |
| Documentation | 9 | 9/10 | x |
| Examples | 10 | 9/10 | x |
| CI/CD & Release | 8 | 10/10 | ☐ |
| Performance & Security | 8 | 9/10 | ☐ |
| **TOTAL** | **73** | **92/100** | ☐ |

**Release Criteria**: Must score ≥92/100 to release

### Critical Actions Required (Dec 12, 2025)

1.  **Security**: Upgrade `wasmtime` to fix 2 reported vulnerabilities (RUSTSEC-2025-0118, RUSTSEC-2025-0046).
2.  **Quality**: Fix 288 Clippy warnings and remove 212 `unwrap()` calls.
3.  **WASM**: Fix `crossterm` compilation on `wasm32-unknown-unknown` target.
4.  **Release**: Add version requirement to `probar-derive` dependency in `crates/probar/Cargo.toml`.
5.  **Coverage**: Increase line coverage from 92.3% to 95%.


---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-01-XX | Probar Team | Initial specification |
| 1.1.0 | 2025-12-12 | Probar Team | Added examples, book updates, 100-point QA checklist |

---

*This specification follows PMAT Work standards and EXTREME TDD methodology. All implementations must achieve ≥95% test coverage and ≥80% mutation score before merge.*
