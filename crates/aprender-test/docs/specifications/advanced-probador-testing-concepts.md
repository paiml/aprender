# Advanced Probador Testing Concepts

**Version**: 1.1.0
**Status**: SPECIFICATION
**Ticket**: PROBAR-SPEC-010
**Toyota Principle**: Jidoka (Built-in Quality) + Poka-Yoke (Error Prevention)
**Architecture**: Zero JavaScript — Pure Rust WASM Testing

---

## Executive Summary

This specification defines advanced testing capabilities for probar/probador based on empirical analysis of five production WASM codebases:

| Codebase | Key Insights |
|----------|--------------|
| **interactive.paiml.com** | Catastrophic mock testing failure RCA, Service Worker caching gaps |
| **whisper.apr** | SharedArrayBuffer/threading patterns, streaming state machines |
| **trueno** | SIMD backend verification, WASM memory bounds |
| **paiml-mcp-agent-toolkit** | pmat comply integration model |
| **trueno-zram** | Compression statistics, falsification test patterns |

**Critical Finding**: "You cannot mock your way to quality. You must test the real thing."

---

## 1. WASM Testing Gaps: The Top 10

### 1.1 Critical Priority Gaps

| # | Gap | Evidence | GitHub Issue |
|---|-----|----------|--------------|
| **1** | SharedArrayBuffer availability detection | whisper.apr `is_threaded_available()` checks `crossOriginIsolated` but no test validates fallback paths | [#13](https://github.com/paiml/probar/issues/13) |
| **2** | Worker thread message ordering/loss | whisper.apr `worker.rs` defines 8 message types with no property tests for ordering | [#15](https://github.com/paiml/probar/issues/15) |
| **3** | Lazy WASM initialization | interactive.paiml.com fixed Firefox "slow script" warning via lazy init — no regression tests | — |
| **4** | Service worker WASM caching | interactive.paiml.com regex missed 'ruchy' book — offline testing is manual | — |
| **5** | Real execution vs DOM presence | 26 mock tests passed, 0% bug detection — tests verified element exists, not that code runs | — |

### 1.2 High Priority Gaps

| # | Gap | Evidence | GitHub Issue |
|---|-----|----------|--------------|
| **6** | Memory limits/OOM handling | whisper.apr `recommendedWasmPages()` untested for graceful degradation | — |
| **7** | Streaming state machine edge cases | whisper.apr 5-state `ProcessorState` + overlap buffers have untested boundaries | [#16](https://github.com/paiml/probar/issues/16) |
| **8** | SIMD correctness vs scalar | trueno tests scalar≈simd but probar doesn't verify WASM SIMD128 determinism | — |
| **9** | Console error capture | interactive.paiml.com RCA: tests never captured `console.error` | — |
| **10** | Component registration verification | Tests should call `customElements.get()`, not just check DOM | — |

---

## 2. New Emulator APIs

### 2.1 AudioEmulator (#12)

**Purpose**: Mock `getUserMedia` with controlled audio for streaming ASR testing.

```rust
use jugar_probar::emulators::AudioEmulator;

/// Audio source types for injection
pub enum AudioSource {
    /// Sine wave at specified frequency (Hz)
    SineWave { frequency: f32, amplitude: f32 },

    /// Speech-like audio (fundamental + harmonics)
    SpeechPattern {
        fundamental_hz: f32,    // 100-300 Hz typical
        harmonics: Vec<f32>,    // [0.5, 0.3, 0.2, 0.1]
        variation_hz: f32,      // Pitch variation
    },

    /// Silence with optional background noise
    Silence { noise_floor_db: f32 },

    /// Pre-recorded audio file
    File { path: PathBuf, loop_: bool },

    /// Real-time programmatic generation
    Callback(Box<dyn Fn(f32) -> f32 + Send>),
}

impl AudioEmulator {
    /// Inject audio source into page's getUserMedia
    /// Implementation: Overrides navigator.mediaDevices.getUserMedia
    pub async fn inject(page: &Page, source: AudioSource) -> ProbarResult<Self>;

    /// Generate audio samples for specified duration
    pub async fn generate(&self, duration_ms: u64) -> ProbarResult<()>;

    /// Stop audio injection
    pub async fn stop(&self) -> ProbarResult<()>;

    /// Get injected sample count
    pub fn samples_injected(&self) -> u64;
}
```

**Usage Example**:

```rust
#[probar::test]
async fn test_whisper_transcription(ctx: &mut TestContext) -> ProbarResult<()> {
    let page = ctx.page();

    // Inject speech-like audio (not pure sine wave!)
    let audio = AudioEmulator::inject(&page, AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2, 0.1],
        variation_hz: 20.0,
    }).await?;

    // Generate 3 seconds of audio
    audio.generate(3000).await?;

    // Wait for transcription
    page.locator("#transcription")
        .wait_for_text_change(Duration::from_secs(5))
        .await?;

    Ok(())
}
```

### 2.2 WasmThreadCapabilities (#13)

**Purpose**: Verify SharedArrayBuffer/COOP-COEP headers and threading availability.

```rust
use jugar_probar::capabilities::WasmThreadCapabilities;

pub struct WasmThreadCapabilities {
    /// Whether crossOriginIsolated is true
    pub cross_origin_isolated: bool,

    /// Whether SharedArrayBuffer is available
    pub shared_array_buffer: bool,

    /// Whether Atomics is available
    pub atomics: bool,

    /// navigator.hardwareConcurrency value
    pub hardware_concurrency: u32,

    /// COOP header value (if present)
    pub coop_header: Option<String>,

    /// COEP header value (if present)
    pub coep_header: Option<String>,
}

impl WasmThreadCapabilities {
    /// Detect capabilities from page context
    pub async fn detect(page: &Page) -> ProbarResult<Self>;

    /// Assert all requirements for multi-threaded WASM
    pub fn assert_threading_ready(&self) -> ProbarResult<()>;

    /// Assert requirements for streaming audio processing
    pub fn assert_streaming_ready(&self) -> ProbarResult<()>;

    /// Get recommended thread count
    pub fn optimal_threads(&self) -> u32 {
        self.hardware_concurrency.saturating_sub(1).max(1).min(8)
    }
}

/// Header requirements
pub struct RequiredHeaders {
    pub coop: &'static str,  // "same-origin"
    pub coep: &'static str,  // "require-corp"
}

impl WasmThreadCapabilities {
    pub const REQUIRED: RequiredHeaders = RequiredHeaders {
        coop: "same-origin",
        coep: "require-corp",
    };
}
```

**Usage Example**:

```rust
#[probar::test]
async fn test_threading_availability(ctx: &mut TestContext) -> ProbarResult<()> {
    let page = ctx.page();
    let caps = WasmThreadCapabilities::detect(&page).await?;

    // Verify headers are correct
    assert_eq!(caps.coop_header.as_deref(), Some("same-origin"));
    assert_eq!(caps.coep_header.as_deref(), Some("require-corp"));

    // Verify threading is available
    caps.assert_threading_ready()?;

    println!("Optimal threads: {}", caps.optimal_threads());

    Ok(())
}
```

### 2.3 WorkerEmulator (#15)

**Purpose**: Test Web Worker lifecycle, message ordering, and error handling.

```rust
use jugar_probar::emulators::WorkerEmulator;

/// Worker message for injection/interception
#[derive(Debug, Clone)]
pub struct WorkerMessage {
    pub type_: String,
    pub data: serde_json::Value,
    pub timestamp: f64,
}

/// Worker state for state machine testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Uninitialized,
    Loading,
    Ready,
    Processing,
    Error,
    Terminated,
}

pub struct WorkerEmulator {
    worker_id: String,
    message_log: Vec<WorkerMessage>,
    state_transitions: Vec<(WorkerState, f64)>`,
}

impl WorkerEmulator {
    /// Attach to existing worker by selector
    /// Implementation: Hooks window.Worker constructor
    pub async fn attach(page: &Page, worker_selector: &str) -> ProbarResult<Self>;

    /// Intercept all messages to/from worker
    pub async fn intercept_messages(&mut self) -> ProbarResult<()>;

    /// Inject message to worker
    pub async fn post_message(&self, msg: WorkerMessage) -> ProbarResult<()>;

    /// Wait for specific message type
    pub async fn wait_for_message(&self, type_: &str, timeout: Duration) -> ProbarResult<WorkerMessage>;

    /// Assert message ordering (for property testing)
    pub fn assert_message_order(&self, expected: &[&str]) -> ProbarResult<()>;

    /// Get all state transitions
    pub fn state_transitions(&self) -> &[(WorkerState, f64)];

    /// Assert state machine sequence
    pub fn assert_state_sequence(&self, expected: &[WorkerState]) -> ProbarResult<()>;

    /// Simulate worker termination
    pub async fn terminate(&self) -> ProbarResult<()>;

    /// Simulate worker error
    pub async fn inject_error(&self, error: &str) -> ProbarResult<()>;
}
```

**Usage Example**:

```rust
#[probar::test]
async fn test_worker_state_machine(ctx: &mut TestContext) -> ProbarResult<()> {
    let page = ctx.page();
    let mut worker = WorkerEmulator::attach(&page, "#whisper-worker").await?;

    worker.intercept_messages().await?;

    // Send init message
    worker.post_message(WorkerMessage {
        type_: "Init".into(),
        data: json!({ "model": "tiny" }),
        timestamp: 0.0,
    }).await?;

    // Wait for ready
    let ready = worker.wait_for_message("Ready", Duration::from_secs(30)).await?;

    // Verify state transitions
    worker.assert_state_sequence(&[
        WorkerState::Uninitialized,
        WorkerState::Loading,
        WorkerState::Ready,
    ])?;

    Ok(())
}
```

### 2.4 StreamingUxValidator (#16)

**Purpose**: Validate real-time UX elements (VU meters, progress bars, state labels).

```rust
use jugar_probar::validators::StreamingUxValidator;

/// VU meter validation parameters
pub struct VuMeterConfig {
    /// Minimum expected level (0.0-1.0)
    pub min_level: f32,
    /// Maximum expected level (0.0-1.0)
    pub max_level: f32,
    /// Update frequency (Hz)
    pub update_rate_hz: f32,
    /// Smoothing factor validation
    pub smoothing_tolerance: f32,
}

/// State transition event
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: String,
    pub to: String,
    pub timestamp: f64,
    pub duration_ms: f64,
}

pub struct StreamingUxValidator {
    state_history: Vec<StateTransition>,
    vu_samples: Vec<(f64, f32)>`,
    partial_results: Vec<(f64, String)>`,
}

impl StreamingUxValidator {
    /// Track state changes on element
    pub async fn track_state(page: &Page, selector: &str) -> ProbarResult<Self>;

    /// Track VU meter levels
    pub async fn track_vu_meter(&mut self, selector: &str) -> ProbarResult<()>;

    /// Track partial transcription results
    pub async fn track_partials(&mut self, selector: &str) -> ProbarResult<()>;

    /// Assert state sequence occurred
    pub fn assert_state_sequence(&self, expected: &[&str]) -> ProbarResult<()>;

    /// Assert VU meter was active during period
    pub async fn assert_vu_meter_active(
        &self,
        page: &Page,
        selector: &str,
        min_level: f32,
        duration_ms: u64,
    ) -> ProbarResult<()>;

    /// Assert no UI jank (state updates within threshold)
    pub fn assert_no_jank(&self, max_gap_ms: f64) -> ProbarResult<()>;

    /// Assert partial results appeared before final
    pub fn assert_partials_before_final(&self) -> ProbarResult<()>;

    /// Get average state transition time
    pub fn avg_transition_time_ms(&self) -> f64;
}
```

**Usage Example**:

```rust
#[probar::test]
async fn test_streaming_ux(ctx: &mut TestContext) -> ProbarResult<()> {
    let page = ctx.page();

    // Verify threading capabilities first
    let caps = WasmThreadCapabilities::detect(&page).await?;
    caps.assert_streaming_ready()?;

    // Inject speech-like audio
    let audio = AudioEmulator::inject(&page, AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2, 0.1],
        variation_hz: 20.0,
    }).await?;

    // Track UX state
    let mut validator = StreamingUxValidator::track_state(&page, "#state_label").await?;
    validator.track_vu_meter("#vu_meter").await?;
    validator.track_partials("#transcription").await?;

    // Generate audio
    audio.generate(5000).await?;

    // Validate UX
    validator.assert_state_sequence(&["Listening", "Recording", "Transcribing", "Done"])?;
    validator.assert_vu_meter_active(&page, "#vu_meter", 0.1, 3000).await?;
    validator.assert_no_jank(100.0)?;  // Max 100ms gap between updates
    validator.assert_partials_before_final()?;

    Ok(())
}
```

---

## 3. Dual Compliance: PMAT Comply + Probador Comply

Probar implements **two compliance systems** that work together:

| System | Scope | Purpose |
|--------|-------|---------|
| **pmat comply** | Project-wide | General code quality, TDG, coverage, SATD |
| **probador comply** | WASM-specific | Threading, browser testing, GUI coverage |

### 3.1 Probador Comply Command

A native `probador comply` command for WASM-specific compliance:

```bash
# Check WASM testing compliance
probador comply check --path . --strict

# Migrate to latest probador standards
probador comply migrate --path . --dry-run

# Show WASM testing changelog
probador comply diff --from 0.2.0 --to 0.3.0

# Install WASM quality hooks
probador comply enforce --path .

# Generate WASM compliance report
probador comply report --format markdown --output wasm-compliance.md
```

#### Probador Comply Subcommands

```rust
/// Supported output formats for reports
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Markdown,
    Html,
}

/// Probador compliance command structure
#[derive(Debug, clap::Subcommand)]
pub enum ProbadorComplyCommands {
    /// Check WASM testing compliance
    Check {
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Exit with error if non-compliant
        #[arg(long)]
        strict: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Migrate to latest probador standards
    Migrate {
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Target version
        #[arg(long)]
        version: Option<String>,

        /// Preview changes without applying
        #[arg(long)]
        dry_run: bool,
    },

    /// Show changelog between versions
    Diff {
        #[arg(long)]
        from: Option<String>,

        #[arg(long)]
        to: Option<String>,

        /// Show only breaking changes
        #[arg(long)]
        breaking_only: bool,
    },

    /// Install WASM quality hooks
    Enforce {
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Skip confirmation
        #[arg(long)]
        yes: bool,

        /// Remove hooks
        #[arg(long)]
        disable: bool,
    },

    /// Generate compliance report
    Report {
        #[arg(long, default_value = ".")]
        path: PathBuf,

        #[arg(long, default_value = "text")]
        format: OutputFormat,

        #[arg(long)]
        output: Option<PathBuf>,
    },
}
```

#### Probador Compliance Checks (10 checks)

```rust
/// WASM-specific compliance checks
pub fn probador_compliance_checks(project_path: &Path) -> Vec<ComplianceCheck> {
    vec![
        // Threading & Concurrency
        Check::sharedarraybuffer_fallback(project_path),   // C001
        Check::worker_lifecycle_tests(project_path),       // C002
        Check::threading_mode_coverage(project_path),      // C003

        // Browser Testing
        Check::gui_coverage_threshold(project_path),       // C004 (>= 95%)
        Check::console_error_assertions(project_path),     // C005
        Check::component_registration_tests(project_path), // C006

        // WASM Quality
        Check::panic_free_paths(project_path),             // C007
        Check::wasm_binary_size(project_path),             // C008
        Check::deterministic_replay(project_path),         // C009

        // Headers & Security
        Check::coop_coep_headers(project_path),            // C010
    ]
}

/// Check result structure
pub struct ComplianceCheck {
    pub id: String,           // "C001"
    pub name: String,         // "SharedArrayBuffer Fallback"
    pub status: CheckStatus,  // Pass | Warn | Fail | Skip
    pub message: String,
    pub severity: Severity,   // Info | Warning | Error | Critical
    pub fix_command: Option<String>,
}

/// Example check implementation
fn check_sharedarraybuffer_fallback(path: &Path) -> ComplianceCheck {
    // Look for tests that verify both threaded and non-threaded paths
    let test_files = glob::glob(&format!("{}/**/tests/**/*.rs", path.display()))
        .unwrap()
        .filter_map(|p| p.ok());

    let has_fallback_tests = test_files.any(|f| {
        let content = std::fs::read_to_string(&f).unwrap_or_default();
        content.contains("test_both_threading_modes")
            || content.contains("is_threaded_available")
            || content.contains("sequential_fallback")
    });

    ComplianceCheck {
        id: "C001".into(),
        name: "SharedArrayBuffer Fallback".into(),
        status: if has_fallback_tests { CheckStatus::Pass } else { CheckStatus::Warn },
        message: if has_fallback_tests {
            "Both threaded and sequential paths are tested".into()
        } else {
            "No tests verify behavior when SharedArrayBuffer is unavailable".into()
        },
        severity: Severity::Warning,
        fix_command: Some("probador generate --template threading-fallback-test".into()),
    }
}
```

### 3.2 PMAT Comply Integration: `pmat comply wasm-test`

In addition to `probador comply`, integrate with pmat comply:

```rust
/// WASM-specific compliance checks
pub struct WasmComplianceCheck {
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
    pub severity: Severity,
}

/// Check categories for WASM projects
pub fn wasm_compliance_checks(project_path: &Path) -> Vec<WasmComplianceCheck> {
    vec![
        check_gui_coverage_threshold(project_path),      // >= 95%
        check_wasm_binary_size(project_path),            // No bloat regression
        check_panic_free_paths(project_path),            // No unwrap in WASM
        check_sharedarraybuffer_fallback(project_path),  // Both paths tested
        check_deterministic_replay(project_path),        // Playbook hash matches
        check_console_error_assertions(project_path),    // No silent failures
    ]
}

fn check_gui_coverage_threshold(path: &Path) -> WasmComplianceCheck {
    // Parse probar coverage report
    let coverage = parse_probar_coverage(path);

    WasmComplianceCheck {
        name: "GUI Coverage Threshold".into(),
        status: if coverage >= 95.0 { CheckStatus::Pass } else { CheckStatus::Fail },
        message: format!("GUI coverage: {:.1}% (minimum: 95%)", coverage),
        severity: Severity::Error,
    }
}

fn check_panic_free_paths(path: &Path) -> WasmComplianceCheck {
    // Scan for unwrap() in WASM code
    let unwraps = count_unwraps_in_wasm(path);

    WasmComplianceCheck {
        name: "Panic-Free Paths".into(),
        status: if unwraps == 0 { CheckStatus::Pass } else { CheckStatus::Fail },
        message: format!("Found {} unwrap() calls in WASM code", unwraps),
        severity: Severity::Error,
    }
}
```

### 3.3 Unified Compliance: Running Both Together

Both systems should run as part of CI/CD:

```bash
# Run both compliance systems
pmat comply check --strict && probador comply check --strict

# Or use the combined command (planned)
pmat comply check --include-wasm --strict
```

#### Example Output: `probador comply check`

```
============================================================
Probador WASM Compliance Report
============================================================

Project: whisper.apr
Probador Version: 0.3.0
Scan Time: 2026-01-05T10:30:00Z

Checks:
  ✓ C001 SharedArrayBuffer Fallback: Both paths tested
  ✓ C002 Worker Lifecycle Tests: 8 message types covered
  ⚠ C003 Threading Mode Coverage: Only threaded path tested
  ✓ C004 GUI Coverage Threshold: 96.2% (minimum: 95%)
  ✓ C005 Console Error Assertions: All tests capture errors
  ✓ C006 Component Registration: customElements.get() verified
  ✓ C007 Panic-Free Paths: 0 unwrap() in WASM code
  ✓ C008 WASM Binary Size: 2.3MB (limit: 5MB)
  ⚠ C009 Deterministic Replay: No playbook hash validation
  ✓ C010 COOP/COEP Headers: Headers verified in tests

Summary: 8/10 passed, 2 warnings, 0 failures

Recommendations:
  • C003: Add test with WasmThreadCapabilities::mock_unavailable()
  • C009: Add probador replay --validate-hash to CI

============================================================
```

#### Combined Report: `pmat comply check --include-wasm`

```
============================================================
PMAT + Probador Combined Compliance Report
============================================================

=== PMAT Checks ===
  ✓ Version Currency: On latest v2.5.1
  ✓ Config Files: All required files present
  ✓ Git Hooks: Pre-commit installed
  ✓ Quality Thresholds: Configured
  ✓ Deprecated Features: None detected
  ✓ TDG Grade: A (92 points)
  ✓ SATD Comments: 0 TODO/FIXME/HACK
  ✓ Test Coverage: 96.2% (minimum: 95%)

=== Probador WASM Checks ===
  ✓ C001-C002: Threading tests complete
  ⚠ C003: Sequential fallback not tested
  ✓ C004-C008: Browser + WASM quality passed
  ⚠ C009: Replay hash validation missing
  ✓ C010: Security headers verified

Combined Status: COMPLIANT (with warnings)
PMAT: 8/8 passed | Probador: 8/10 passed

============================================================
```

### 3.4 Pre-Commit Hook Integration

```bash
#!/bin/bash
# .git/hooks/pre-commit - WASM quality gates

set -e

# 1. WASM binary size regression check
MAX_WASM_SIZE=5000000  # 5MB
wasm_size=$(stat -c%s target/wasm32-unknown-unknown/release/*.wasm 2>/dev/null || echo 0)
if [ "$wasm_size" -gt "$MAX_WASM_SIZE" ]; then
    echo "ERROR: WASM binary size regression: ${wasm_size} > ${MAX_WASM_SIZE}"
    exit 1
fi

# 2. No panic paths in WASM
if grep -rn "unwrap()" --include="*.rs" src/wasm/ 2>/dev/null | grep -v "// SAFETY:"; then
    echo "ERROR: unwrap() found in WASM code without safety comment"
    exit 1
fi

# 3. Probar GUI coverage check
coverage=$(cargo test -p jugar-probar --format json 2>/dev/null |
    jq -r '.gui_coverage // 0')
if [ "$(echo "$coverage < 95" | bc -l)" -eq 1 ]; then
    echo "ERROR: GUI coverage ${coverage}% below 95% threshold"
    exit 1
fi

# 4. Console error capture in tests
if ! grep -r "fail_on_console_error" tests/probar/*.rs 2>/dev/null | grep -q .; then
    echo "WARNING: No tests check for console errors"
fi

echo "WASM quality gates passed"
```

---

## 4. Strictness Improvements

### 4.1 WasmStrictMode Configuration

```rust
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
}

impl Default for WasmStrictMode {
    fn default() -> Self {
        Self {
            require_code_execution: true,
            fail_on_console_error: true,
            verify_custom_elements: true,
            test_both_threading_modes: false,  // Opt-in
            simulate_low_memory: false,        // Opt-in
            verify_coop_coep_headers: true,
            validate_replay_hash: true,
            max_console_warnings: 0,
            require_cache_hits: false,         // Opt-in
        }
    }
}

impl WasmStrictMode {
    /// Production-grade strictness
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
        }
    }

    /// Development-friendly (more permissive)
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
        }
    }
}
```

### 4.2 Console Capture API

```rust
use jugar_probar::console::ConsoleCapture;

/// Console message severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsoleSeverity {
    Log,
    Info,
    Warn,
    Error,
}

/// Captured console message
#[derive(Debug, Clone)]
pub struct ConsoleMessage {
    pub severity: ConsoleSeverity,
    pub text: String,
    pub source: String,
    pub line: u32,
    pub timestamp: f64,
}

pub struct ConsoleCapture {
    messages: Vec<ConsoleMessage>,
    strict_mode: WasmStrictMode,
}

impl ConsoleCapture {
    /// Start capturing console output
    pub async fn start(page: &Page, strict_mode: WasmStrictMode) -> ProbarResult<Self>;

    /// Stop capturing and validate
    pub async fn stop_and_validate(&self) -> ProbarResult<()>;

    /// Get all errors
    pub fn errors(&self) -> Vec<&ConsoleMessage>;

    /// Get all warnings
    pub fn warnings(&self) -> Vec<&ConsoleMessage>;

    /// Assert no errors occurred
    pub fn assert_no_errors(&self) -> ProbarResult<()>;

    /// Assert specific error message NOT present
    pub fn assert_no_error_containing(&self, substring: &str) -> ProbarResult<()>;
}
```

**Usage Example**:

```rust
#[probar::test]
async fn test_no_console_errors(ctx: &mut TestContext) -> ProbarResult<()> {
    let page = ctx.page();
    let console = ConsoleCapture::start(&page, WasmStrictMode::production()).await?;

    page.goto("http://localhost:8080/app").await?;

    // Interact with the app
    page.click("#start-button").await?;
    page.wait_for_selector("#result").await?;

    // Validate no errors
    console.stop_and_validate()?;
    console.assert_no_errors()?;

    Ok(())
}
```

---

## 5. Trueno-ZRAM Integration Features

### 5.1 Compression Statistics for Test Monitoring

From trueno-zram, we can adopt compression statistics tracking:

```rust
/// Test execution statistics with compression metrics
pub struct TestExecutionStats {
    /// Total game states captured
    pub states_captured: u64,

    /// Bytes before compression
    pub bytes_raw: u64,

    /// Bytes after compression
    pub bytes_compressed: u64,

    /// Compression ratio (raw / compressed)
    pub compression_ratio: f64,

    /// Compression throughput (bytes/sec)
    pub compress_throughput: f64,

    /// Same-fill pages detected (high compression)
    pub same_fill_pages: u64,
}

impl TestExecutionStats {
    /// Calculate compression efficiency
    pub fn efficiency(&self) -> f64 {
        1.0 - (self.bytes_compressed as f64 / self.bytes_raw as f64)
    }

    /// Estimate storage savings
    pub fn storage_savings_mb(&self) -> f64 {
        (self.bytes_raw - self.bytes_compressed) as f64 / 1_000_000.0
    }
}
```

### 5.2 Entropy-Based Screenshot Classification

```rust
/// Classify screenshot content for optimal compression
pub enum ScreenshotContent {
    /// UI-heavy (text, buttons) - high compressibility
    UiDominated { entropy: f32 },

    /// Physics/game world - medium compressibility
    GameWorld { entropy: f32 },

    /// Random/noise - low compressibility
    HighEntropy { entropy: f32 },

    /// Mostly uniform - very high compressibility (same-fill)
    Uniform { fill_value: u8 },
}

impl ScreenshotContent {
    /// Classify screenshot by entropy analysis
    pub fn classify(pixels: &[u8]) -> Self;

    /// Recommended compression algorithm
    pub fn recommended_algorithm(&self) -> CompressionAlgorithm;
}
```

### 5.3 Falsification Test Pattern

From trueno-zram's F076-F085 safety tests:

```rust
/// Falsification test categories (Popperian methodology)
pub mod falsification {
    /// F076: No buffer overflows
    pub fn test_boundary_conditions();

    /// F077: No integer overflow in sizes
    pub fn test_size_calculations();

    /// F078: No use-after-free (Rust borrow checker)
    pub fn test_lifetime_safety();

    /// F079: No data races
    pub fn test_concurrent_access();

    /// F080: No undefined behavior
    pub fn test_edge_case_patterns();

    /// F081: No panics on valid input
    pub fn test_panic_freedom();

    /// F082: Error types implement std::error::Error
    pub fn test_error_trait_impl();

    /// F083: Secure memory management
    pub fn test_memory_zeroization();

    /// F084: Timing consistency
    pub fn test_timing_side_channels();

    /// F085: No external FFI
    pub fn test_pure_rust();
}
```

---

## 6. Catastrophic Testing Failure Prevention

### 6.1 Lessons from interactive.paiml.com RCA

The interactive.paiml.com codebase experienced a "catastrophic testing failure":
- 26 mock tests: 100% pass rate
- 0% actual bug detection rate
- 1 Playwright E2E test caught production bug immediately

**Root Cause**: Mock Pyodide only checked user code indentation, not wrapper script:

```typescript
// BROKEN: Mock only checked user's code, not wrapper script
if (code.match(/^\s+\S/)) {
    return Promise.reject(new Error("IndentationError"));
}

// THE BUG: Wrapper had 8 spaces of leading whitespace
const result = await this.pyodide.runPythonAsync(`
        import sys  // <-- 8 spaces!
`);
```

### 6.2 Anti-Patterns to Avoid

| Anti-Pattern | Description | Probar Prevention |
|--------------|-------------|-------------------|
| **Mock Everything** | Mocks don't catch integration bugs | `require_code_execution: true` |
| **DOM Presence Only** | Element exists ≠ element works | `verify_custom_elements: true` |
| **Ignore Console** | Silent failures go unnoticed | `fail_on_console_error: true` |
| **Single Threading Path** | Threading bugs escape | `test_both_threading_modes: true` |
| **Happy Path Only** | OOM, network errors untested | `simulate_low_memory: true` |

### 6.3 Mandatory Test Checklist

```rust
/// Mandatory checks for WASM E2E tests
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
}

impl E2ETestChecklist {
    /// Validate all mandatory checks passed
    pub fn validate(&self) -> ProbarResult<()> {
        if !self.wasm_executed {
            return Err(ProbarError::ChecklistFailed("WASM not executed"));
        }
        if !self.components_registered {
            return Err(ProbarError::ChecklistFailed("Components not verified"));
        }
        if !self.console_checked {
            return Err(ProbarError::ChecklistFailed("Console not checked"));
        }
        // ...
        Ok(())
    }
}
```

---

## 7. Implementation Roadmap

### Phase 1: Core Emulators (4 weeks)

| Component | Status | GitHub Issue |
|-----------|--------|--------------|
| AudioEmulator | Specified | [#12](https://github.com/paiml/probar/issues/12) |
| WasmThreadCapabilities | Specified | [#13](https://github.com/paiml/probar/issues/13) |
| WorkerEmulator | Specified | [#15](https://github.com/paiml/probar/issues/15) |
| StreamingUxValidator | Specified | [#16](https://github.com/paiml/probar/issues/16) |

### Phase 2: Strictness Framework (2 weeks)

| Component | Status |
|-----------|--------|
| WasmStrictMode | Specified |
| ConsoleCapture | Specified |
| E2ETestChecklist | Specified |

### Phase 3: PMAT Integration (2 weeks)

| Component | Status |
|-----------|--------|
| `pmat comply wasm-test` | Proposed |
| Pre-commit hooks | Proposed |
| Coverage reporting | Proposed |

### Phase 4: Compression & Statistics (2 weeks)

| Component | Status |
|-----------|--------|
| TestExecutionStats | Specified |
| ScreenshotContent classifier | Specified |
| Falsification test suite | Proposed |

---

## 8. Quality Gates

| Metric | Target | Enforcement |
|--------|--------|-------------|
| GUI Coverage | >= 95% | `pmat comply wasm-test` |
| Console Errors | 0 | `fail_on_console_error: true` |
| WASM Binary Size | < 5MB | Pre-commit hook |
| Panic Paths | 0 unwrap() | Clippy + hook |
| Threading Coverage | Both paths | `test_both_threading_modes` |
| Component Registration | Verified | `verify_custom_elements` |

---

## 9. References

### Internal Codebases Analyzed

1. **interactive.paiml.com** - Production WASM + Pyodide site with catastrophic testing failure RCA
2. **whisper.apr** - Streaming ASR with SharedArrayBuffer threading and 2,397 unit tests
3. **trueno** - SIMD tensor library with 48,621 LOC and multi-backend architecture
4. **paiml-mcp-agent-toolkit** - pmat comply implementation with Toyota Way principles
5. **trueno-zram** - Compression library with 478 falsification tests

### GitHub Issues Created

- [#12](https://github.com/paiml/probar/issues/12) - AudioEmulator
- [#13](https://github.com/paiml/probar/issues/13) - SharedArrayBuffer/COOP-COEP
- [#14](https://github.com/paiml/probar/issues/14) - EPIC: Multi-threaded WASM UX Testing
- [#15](https://github.com/paiml/probar/issues/15) - WorkerEmulator
- [#16](https://github.com/paiml/probar/issues/16) - StreamingUxValidator

---

## 10. Technical Implementation Strategy

### 10.1 AudioEmulator Injection
To achieve deterministic audio testing without external hardware:
1. **Interception**: Override `navigator.mediaDevices.getUserMedia` using `Object.defineProperty` to be read-only and unconfigurable during test execution.
2. **Generation**: Use Web Audio API `AudioContext` with `OscillatorNode` (for sine/harmonics) or `AudioBufferSourceNode` (for files).
3. **Stream Creation**: Connect nodes to a `MediaStreamAudioDestinationNode`.
4. **Delivery**: Return the `stream` property of the destination node as the result of the mocked `getUserMedia` promise.

### 10.2 WorkerEmulator Proxying
To intercept worker communication without modifying application code:
1. **Constructor Hook**: Overwrite `window.Worker` with a `Proxy` object.
2. **Message Channel**: Create a `MessageChannel` for the test harness to snoop/inject messages between the main thread and the real worker.
3. **Event Listeners**: Wrap `addEventListener` on the worker instance to capture `message` and `error` events before they reach application code.
4. **Deterministic Delays**: Use a virtual clock (via `sinon.useFakeTimers` or similar) to control message delivery timing if running in a simulated time environment.

### 10.3 ConsoleCapture Mechanism
To capture console output reliably and map it back to Rust structs:
1. **Proxy Method**: Replace `console.log`, `console.error`, `console.warn` with wrapper functions.
2. **Serialization**: Convert arguments to JSON strings immediately to avoid "live object" reference issues where the value changes after logging.
3. **StackTrace**: Capture `new Error().stack` to pinpoint the source of the log message.
4. **Transport**: Push captured logs to a dedicated generic `Array` in global scope (e.g., `window.__PROBAR_CONSOLE_LOGS__`) that `probar` polls via `page.evaluate()`.

---

## Appendix A: Mock vs Real Verification Matrix

| Feature | Mock/Emulator | Real Execution | Reason |
|---------|---------------|----------------|--------|
| **DOM Elements** | ❌ | ✅ | Elements must be interactive and rendered, not just present in the DOM tree |
| **Audio Input** | ✅ (AudioEmulator) | ❌ | Real microphone input is flaky and CI-incompatible; Audio physics is deterministic math |
| **Time/Timers** | ✅ (Clock) | ❌ | Tests must be fast and deterministic; real time is too slow and flaky |
| **Network API** | ❌ | ✅ | Mocks hide serialization/CORS bugs; Browser `fetch` implementation is complex |
| **Network Responses** | ✅ (MockServer) | ⚠️ (Integration) | Control data permutations and error states; verify API contract separately |
| **Web Workers** | ⚠️ (Proxy) | ✅ | Threading bugs require real threads; Proxy is only for observability/injection |
| **WASM Memory** | ❌ | ✅ | Must test actual OOM/growth behavior and allocation limits |

---

## 11. Peer-Reviewed Citations (Toyota Way Spirit)

### 11.1 Foundational Testing Theory

| # | Citation | Application |
|---|----------|-------------|
| **[1]** | **Popper, K. R.** (1959). *The Logic of Scientific Discovery*. Hutchinson. | Falsificationism methodology |
| **[2]** | **Liker, J. K.** (2004). *The Toyota Way*. McGraw-Hill. | Jidoka, Poka-Yoke, Genchi Genbutsu |
| **[3]** | **Shingo, S.** (1986). *Zero Quality Control*. Productivity Press. | Error-proofing via type-safe APIs |
| **[4]** | **Myers, G. J.** (2011). *The Art of Software Testing* (3rd ed.). Wiley. | Mutation testing foundations |
| **[5]** | **Beizer, B.** (1990). *Software Testing Techniques*. Van Nostrand. | State machine testing |

### 11.2 WebAssembly & Concurrency

| # | Citation | Application |
|---|----------|-------------|
| **[6]** | **Haas, A., et al.** (2017). *Bringing the Web up to Speed with WebAssembly*. PLDI. | WASM memory model |
| **[7]** | **Herlihy, M. & Shavit, N.** (2012). *The Art of Multiprocessor Programming*. Morgan Kaufmann. | SharedArrayBuffer testing |
| **[8]** | **Lamport, L.** (1978). *Time, Clocks, and the Ordering of Events*. CACM. | Worker message ordering |
| **[9]** | **Luo, Q., et al.** (2014). *An Empirical Analysis of Flaky Tests*. FSE. | Auto-waiting justification |
| **[10]** | **Lehmann, D. & Pradel, M.** (2020). *Wasabi: Dynamic Analysis for WASM*. ASPLOS. | Binary instrumentation |

### 11.3 Audio & Streaming

| # | Citation | Application |
|---|----------|-------------|
| **[11]** | **Radford, A., et al.** (2023). *Robust Speech Recognition via Weak Supervision*. OpenAI. | Whisper streaming patterns |
| **[12]** | **Sohn, K., et al.** (2015). *Learning Structured Output for VAD*. IEEE ASRU. | VAD state machine testing |
| **[13]** | **Graves, A., et al.** (2006). *Connectionist Temporal Classification*. ICML. | Partial results validation |

### 11.4 Testing Economics

| # | Citation | Application |
|---|----------|-------------|
| **[14]** | **Tassey, G.** (2002). *Economic Impacts of Inadequate Software Testing*. NIST. | Cost justification ($59.5B/yr) |
| **[15]** | **Memon, A., et al.** (2017). *Taming Google-Scale Testing*. ICSE-SEIP. | Shift-left (100x cost reduction) |
| **[16]** | **Papadakis, M., et al.** (2019). *Mutation Testing Advances*. Advances in Computers. | 85% mutation threshold |

---

## 12. 100-Point Popperian Falsification QA Checklist

> "The criterion of the scientific status of a theory is its falsifiability." — Karl Popper [1]

### Scoring: PASS=1, FAIL=0, BLOCKED=-1 | Minimum: 90/100

---

### Section 1: Threading & SharedArrayBuffer (15 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 1 | H1: Threading detection reliable | crossOriginIsolated=false | Graceful fallback | |
| 2 | H1 | SharedArrayBuffer undefined | Clear error | |
| 3 | H1 | COOP header missing | Fix hint shown | |
| 4 | H1 | COEP header wrong | Validation fail | |
| 5 | H1 | Atomics.wait blocked | Alt sync used | |
| 6 | H2: Thread pool safe | Zero hardware concurrency | Fallback to 1 | |
| 7 | H2 | 256 cores reported | Capped at 8 | |
| 8 | H2 | Init race condition | Single init | |
| 9 | H2 | Worker creation fails | Error propagates | |
| 10 | H2 | Pool exhaustion | Queue, no crash | |
| 11 | H3: Worker protocol robust | Msg to uninitialized | Queue or error | |
| 12 | H3 | Unknown message type | Logged, ignored | |
| 13 | H3 | Malformed payload | Validation error | |
| 14 | H3 | Out-of-order messages | Handled | |
| 15 | H3 | Worker terminated mid-op | Graceful cleanup | |

### Section 2: Audio Emulator (15 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 16 | H4: Audio injection reliable | Permission denied | Error handled | |
| 17 | H4 | Context suspended (mobile) | Resume on tap | |
| 18 | H4 | Sample rate mismatch | Auto resample | |
| 19 | H4 | Zero-length audio | Warning, no crash | |
| 20 | H4 | Ultrasonic (>22kHz) | Filtered | |
| 21 | H5: Pattern generation accurate | 0 Hz sine | DC or error | |
| 22 | H5 | Negative amplitude | Absolute or error | |
| 23 | H5 | Amplitude > 1.0 | Clamped | |
| 24 | H5 | No harmonics | Pure fundamental | |
| 25 | H5 | Callback throws | Graceful stop | |
| 26 | H6: VAD works | Pure silence | No speech | |
| 27 | H6 | White noise | Not speech | |
| 28 | H6 | Threshold boundary | Consistent | |
| 29 | H6 | 10ms burst | Min duration enforced | |
| 30 | H6 | Long pauses | Proper segments | |

### Section 3: Streaming UX (15 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 31 | H7: State tracking accurate | Element not found | Clear error | |
| 32 | H7 | Rapid state changes | All captured | |
| 33 | H7 | State reverts | Recorded | |
| 34 | H7 | Element removed mid-track | Graceful | |
| 35 | H7 | Multiple elements | Selector error | |
| 36 | H8: VU meter validation | Not animated | Inactivity detected | |
| 37 | H8 | Constant value | Staleness detected | |
| 38 | H8 | Slow updates | Rate warning | |
| 39 | H8 | Exceeds 100% | Clipping detected | |
| 40 | H8 | Negative values | Error or abs | |
| 41 | H9: Partials tracked | None emitted | Failure | |
| 42 | H9 | After final | Ordering error | |
| 43 | H9 | Duplicate content | Deduplicated | |
| 44 | H9 | Empty partial | Filtered | |
| 45 | H9 | Special chars | Encoded | |

### Section 4: Compliance System (15 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 46 | H10: Comply thorough | No tests project | All fail | |
| 47 | H10 | Mock-only tests | C005 warning | |
| 48 | H10 | 10MB WASM | C008 fail | |
| 49 | H10 | unwrap() in tests | C007 fail | |
| 50 | H10 | No header tests | C010 warning | |
| 51 | H11: Migration safe | Uncommitted changes | Require --force | |
| 52 | H11 | Non-existent version | Not found | |
| 53 | H11 | Breaking changes | Confirm required | |
| 54 | H11 | Dry-run | Zero changes | |
| 55 | H11 | Backup fails | Abort | |
| 56 | H12: Enforcement works | Failing checks | Exit 1 | |
| 57 | H12 | All pass | Exit 0 | |
| 58 | H12 | --disable | Hooks removed | |
| 59 | H12 | Re-enable | Hooks restored | |
| 60 | H12 | CI environment | Correct behavior | |

### Section 5: Console Capture (10 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 61 | H13: Capture complete | console.error | Logged | |
| 62 | H13 | console.warn | Logged | |
| 63 | H13 | Uncaught exception | Captured | |
| 64 | H13 | Unhandled rejection | Captured | |
| 65 | H13 | 1000 msgs/sec | Throttled | |
| 66 | H14: Strict mode works | fail_on_console_error | Test fails | |
| 67 | H14 | max_warnings exceeded | Test fails | |
| 68 | H14 | Dev mode errors | Passes with warn | |
| 69 | H14 | Prod mode errors | Fails | |
| 70 | H14 | Error in setup | Attributed | |

### Section 6: Component Registration (10 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 71 | H15: Verification accurate | get() undefined | Detected | |
| 72 | H15 | Late registration | Retry works | |
| 73 | H15 | Invalid name | Error | |
| 74 | H15 | No shadow DOM | Detected | |
| 75 | H15 | Upgrade pending | Wait works | |
| 76 | H16: Execution verified | Empty render | Detected | |
| 77 | H16 | Callback error | Captured | |
| 78 | H16 | Attr change | Verified | |
| 79 | H16 | Slot content | Verified | |
| 80 | H16 | Nested shadow | Traversed | |

### Section 7: Memory & Performance (10 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 81 | H17: Memory limits enforced | Exceed initial | Grow or trap | |
| 82 | H17 | Exceed maximum | Trap, no crash | |
| 83 | H17 | 1000 frame leak | Stable | |
| 84 | H17 | 50MB heap | Graceful | |
| 85 | H17 | Concurrent alloc | No corruption | |
| 86 | H18: Perf regression detected | Frame budget exceeded | Warning | |
| 87 | H18 | Startup regression | Detected | |
| 88 | H18 | Memory growth | Leak suspected | |
| 89 | H18 | GC pause | Jank identified | |
| 90 | H18 | Network bottleneck | Found | |

### Section 8: Deterministic Replay (10 pts)

| # | Hypothesis | Falsification Attempt | Expected | Score |
|---|------------|----------------------|----------|-------|
| 91 | H19: Replay deterministic | Same seed | Byte-exact | |
| 92 | H19 | Different machine | Identical | |
| 93 | H19 | After WASM rebuild | Hash fails | |
| 94 | H19 | Truncated playbook | Partial replay | |
| 95 | H19 | Corrupt checksum | Validation fails | |
| 96 | H20: Playbook validation | Version mismatch | Clear error | |
| 97 | H20 | Missing WASM hash | Warning | |
| 98 | H20 | Wrong frame count | Detected | |
| 99 | H20 | Future inputs | Rejected | |
| 100 | H20 | Empty playbook | Valid no-op | |

---

### Score Interpretation

| Score | Grade | Action |
|-------|-------|--------|
| 95-100 | A+ | Ship |
| 90-94 | A | Minor fixes |
| 80-89 | B | Fix before release |
| <80 | F | Block release |

---

**Document Version**: 1.2.0
**Last Updated**: 2026-01-05
**Authors**: PAIML Team
**Methodology**: Popperian Falsificationism + Toyota Production System
**Citations**: 16 peer-reviewed references
**Falsification Tests**: 100 points across 20 hypotheses
**Toyota Principles**: Jidoka, Poka-Yoke, Genchi Genbutsu, Muda, Andon, Mieruka