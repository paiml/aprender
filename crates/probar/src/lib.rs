//! Probar: Rust-Native Testing Framework for WASM Games
//!
//! Per spec Section 6.1: Probar (Spanish: "to test/prove") is a pure Rust
//! alternative to Playwright/Puppeteer, designed for WASM game testing.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    PROBAR Architecture                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │   ┌────────────┐    ┌────────────┐    ┌────────────┐            │
//! │   │ Test Spec  │    │ WASM       │    │ Headless   │            │
//! │   │ (Rust)     │───►│ Test       │───►│ Browser    │            │
//! │   │            │    │ Harness    │    │ (chromium) │            │
//! │   └────────────┘    └────────────┘    └────────────┘            │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

#![allow(missing_docs)]
// Lints are configured in workspace Cargo.toml [workspace.lints.clippy]
// Allow large stack arrays/frames in tests (e.g., test data generation)
#![cfg_attr(test, allow(clippy::large_stack_arrays, clippy::large_stack_frames))]

// Contract assertions from YAML (pv codegen)
#[macro_use]
#[allow(unused_macros, clippy::duplicated_attributes)]
mod generated_contracts;

/// Brick Architecture: Tests ARE the Interface (PROBAR-SPEC-009)
///
/// Core abstraction where UI components are defined by test assertions.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod brick;

/// BrickHouse: Budgeted Composition of Bricks (PROBAR-SPEC-009)
///
/// Compose multiple bricks with a total performance budget.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::expect_used
)]
pub mod brick_house;

#[allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::struct_excessive_bools,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::unnecessary_wraps,
    clippy::doc_markdown
)]
mod accessibility;
mod assertion;
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
mod bridge;
mod browser;
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    dead_code
)]
mod driver;
mod event;
mod fuzzer;
mod harness;
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::unnecessary_wraps,
    clippy::doc_markdown
)]
mod locator;
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::format_push_string,
    clippy::needless_raw_string_hashes
)]
mod reporter;
mod result;
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::unnecessary_wraps,
    clippy::doc_markdown,
    clippy::if_not_else,
    clippy::ptr_as_ptr,
    clippy::expect_used,
    unsafe_code
)]
mod runtime;
mod simulation;
mod snapshot;
#[cfg(feature = "media")]
mod visual_regression;

/// State Synchronization Linting (PROBAR-SPEC-WASM-001)
///
/// Static analysis for detecting WASM closure state sync anti-patterns.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod lint;

/// Mock Runtime for WASM Callback Testing (PROBAR-SPEC-WASM-001)
///
/// Test WASM callback patterns without browser APIs.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod mock;

/// Compliance Checking for WASM Threading (PROBAR-SPEC-WASM-001)
///
/// Verify projects follow WASM threading best practices.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod comply;

/// Page Object Model Support (Feature 19)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
mod page_object;

/// Fixture Management (Feature 20)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
mod fixture;

/// TUI Testing Support (Feature 21 - EDD Compliance)
#[cfg(feature = "tui")]
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod tui;

/// TUI Load Testing (Framework-Agnostic Performance Testing)
///
/// Test TUI performance with large datasets, hang detection, and frame timing.
/// Works with any TUI framework (presentar, ratatui, crossterm, etc.).
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod tui_load;

/// Deterministic Replay System (Feature 23 - EDD Compliance)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod replay;

/// UX Coverage Metrics (Feature 24 - EDD Compliance)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod ux_coverage;

/// Device Emulation and Geolocation Mocking (Features 15-16)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod emulation;

/// Media Generation Module (Spec: missing-features-in-pure-rust.md)
#[cfg(feature = "media")]
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_possible_truncation
)]
pub mod media;

/// Watch Mode with Hot Reload (Feature 6)
/// Note: Not available on WASM targets (requires filesystem access)
#[cfg(all(not(target_arch = "wasm32"), feature = "watch"))]
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod watch;

/// Execution Tracing (Feature 9)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_possible_truncation
)]
pub mod tracing_support;

/// Network Request Interception (Feature 7)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod network;

/// Wait Mechanisms (PMAT-005)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod wait;

/// WebSocket Monitoring (Feature 8)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod websocket;

/// Performance Profiling (Feature 10)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_possible_truncation
)]
pub mod performance;

/// Multi-Browser Context Management (Feature 14)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod context;

/// WASM Coverage Tooling (spec: probar-wasm-coverage-tooling.md)
#[allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::use_self,
    clippy::inline_always,
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::suboptimal_flops,
    clippy::uninlined_format_args,
    clippy::redundant_closure_for_method_calls
)]
pub mod coverage;

/// Zero-JavaScript Web Asset Generation (Advanced Feature E)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod web;

/// Pixel-Level GUI Coverage Visualization (Advanced Feature A)
///
/// Requires the `media` feature for `image` crate support.
#[cfg(feature = "media")]
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
pub mod pixel_coverage;

/// GPU Pixel Testing: Atomic verification of CUDA kernel correctness
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod gpu_pixels;

/// WASM Runner with Hot Reload (Advanced Feature D)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod runner;

/// Performance Benchmarking with Renacer Integration (Advanced Feature C)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
pub mod perf;

/// Renacer Integration for Deep WASM Tracing (Issue #9)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod renacer_integration;

/// CDP Profiler-based Code Coverage (Issue #10)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod cdp_coverage;

/// Test Sharding for Distributed Execution (Feature G.5)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod shard;

/// Clock Manipulation for Deterministic Tests (Feature G.6)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod clock;

/// WASM Thread Capabilities Detection (Advanced Testing Concepts)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod capabilities;

/// WASM Strict Mode Enforcement (Advanced Testing Concepts)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod strict;

/// Streaming UX Validators (Advanced Testing Concepts)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod validators;

/// Zero-JavaScript Validation for WASM-First Applications (PROBAR-SPEC-012).
///
/// Validates that WASM applications contain NO user-generated JavaScript, CSS, or HTML.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph
)]
pub mod zero_js;

/// WASM Worker Test Harness (PROBAR-SPEC-013).
///
/// Comprehensive testing framework for Web Workers in WASM applications.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph
)]
pub mod worker_harness;

/// Docker-based Cross-Browser WASM Testing (PROBAR-SPEC-014).
///
/// Enables cross-browser testing via Docker containers with COOP/COEP support.
#[cfg(feature = "docker")]
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph
)]
pub mod docker;

/// Dialog Handling for E2E Testing (Feature G.8)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod dialog;

/// File Upload/Download Operations (Feature G.8)
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod file_ops;

/// HAR Recording Module (Spec: G.2 Network Interception)
#[allow(
    clippy::missing_docs_in_private_items,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod har;

/// Playbook Testing: State Machine Verification (PROBAR-004)
/// YAML-driven state machine testing with M1-M5 mutation classes.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::suspicious_operation_groupings,
    missing_docs,
    missing_debug_implementations
)]
pub mod playbook;

/// AV Sync Testing: Verify rendered audio-visual synchronization against EDL ground truth.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod av_sync;

/// Audio Quality Verification: levels, clipping, silence analysis.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
pub mod audio_quality;

/// Video Quality Verification: codec, resolution, FPS, duration validation.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
pub mod video_quality;

/// Animation Verification: timing, easing curves, physics events.
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod animation;

/// Presentar YAML Support (PROBAR-SPEC-015)
///
/// Native support for testing presentar TUI configurations with
/// 100-point falsification protocol (F001-F100).
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod presentar;

/// LLM Testing: Correctness assertions and load testing for OpenAI-compatible APIs.
///
/// Feature-gated behind `llm`. Provides HTTP client, assertion builders,
/// concurrent load testing, and Markdown/JSON reporting.
#[cfg(any(feature = "llm-types", feature = "llm"))]
#[allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::doc_markdown
)]
pub mod llm;

pub use accessibility::{
    AccessibilityAudit, AccessibilityConfig, AccessibilityIssue, AccessibilityValidator, Color,
    ContrastAnalysis, ContrastPair, FlashDetector, FlashResult, FocusConfig, KeyboardIssue,
    Severity, MIN_CONTRAST_LARGE, MIN_CONTRAST_NORMAL, MIN_CONTRAST_UI,
};
pub use animation::{
    sample_easing, verify_easing, verify_events, verify_timeline, AnimationEvent,
    AnimationEventType, AnimationReport, AnimationTimeline, AnimationVerdict, EasingFunction,
    EasingVerification, EventResult, Keyframe, ObservedEvent,
};
pub use assertion::{
    retry_contains, retry_eq, retry_none, retry_some, retry_true, Assertion, AssertionCheckResult,
    AssertionFailure, AssertionMode, AssertionResult, AssertionSummary, EnergyVerifier,
    EquationContext, EquationResult, EquationVerifier, InvariantVerifier, KinematicVerifier,
    MomentumVerifier, RetryAssertion, RetryConfig, RetryError, RetryResult, SoftAssertionError,
    SoftAssertions, Variable,
};
pub use audio_quality::{
    analyze_audio, analyze_samples, detect_clipping, detect_silence, AudioLevels,
    AudioQualityConfig, AudioQualityReport, AudioVerdict, ClippingReport, SilenceRegion,
    SilenceReport,
};
pub use av_sync::{
    compare_edl_to_onsets, default_edl_path, detect_onsets, extract_audio, AudioOnset,
    AudioTickPlacement, AvSyncReport, DetectionConfig, EditDecision, EditDecisionList,
    SegmentSyncResult, SyncVerdict, TickDelta, DEFAULT_SAMPLE_RATE,
};
pub use bridge::{
    BridgeConnection, DiffRegion, EntitySnapshot, GameStateData, GameStateSnapshot, SnapshotCache,
    StateBridge, VisualDiff,
};
pub use browser::{Browser, BrowserConfig, BrowserConsoleLevel, BrowserConsoleMessage, Page};
pub use capabilities::{
    CapabilityError, CapabilityStatus, RequiredHeaders, WasmThreadCapabilities, WorkerEmulator,
    WorkerMessage, WorkerState,
};
pub use cdp_coverage::{
    CoverageConfig, CoverageRange, CoverageReport, CoveredFunction, FunctionCoverage, JsCoverage,
    LineCoverage, ScriptCoverage, SourceMapEntry, WasmCoverage, WasmSourceMap,
};
pub use clock::{
    create_clock, Clock, ClockController, ClockError, ClockOptions, ClockState, FakeClock,
};
pub use context::{
    BrowserContext, ContextConfig, ContextManager, ContextPool, ContextPoolStats, ContextState,
    Cookie, Geolocation, SameSite, StorageState,
};
pub use dialog::{
    AutoDialogBehavior, Dialog, DialogAction, DialogExpectation, DialogHandler,
    DialogHandlerBuilder, DialogType,
};
#[cfg(feature = "browser")]
pub use driver::{BrowserController, ProbarDriver};
pub use driver::{
    DeviceDescriptor, DriverConfig, ElementHandle, MockDriver, NetworkInterceptor, NetworkResponse,
    PageMetrics, Screenshot,
};
pub use event::{InputEvent, Touch, TouchAction};
pub use file_ops::{
    guess_mime_type, Download, DownloadManager, DownloadState, FileChooser, FileInput,
};
pub use fixture::{
    Fixture, FixtureBuilder, FixtureManager, FixtureScope, FixtureState, SimpleFixture,
};
pub use fuzzer::{
    FuzzerConfig, InputFuzzer, InvariantCheck, InvariantChecker, InvariantViolation, Seed,
};
pub use har::{
    Har, HarBrowser, HarCache, HarContent, HarCookie, HarCreator, HarEntry, HarError, HarHeader,
    HarLog, HarOptions, HarPlayer, HarPostData, HarPostParam, HarQueryParam, HarRecorder,
    HarRequest, HarResponse, HarTimings, NotFoundBehavior,
};
pub use harness::{TestCase, TestHarness, TestResult, TestSuite};
pub use locator::{
    expect, BoundingBox, DragBuilder, DragOperation, Expect, ExpectAssertion, Locator,
    LocatorAction, LocatorOptions, LocatorQuery, Point, Selector, DEFAULT_POLL_INTERVAL_MS,
    DEFAULT_TIMEOUT_MS,
};
pub use network::{
    CapturedRequest, HttpMethod, MockResponse, NetworkInterception, NetworkInterceptionBuilder,
    Route, UrlPattern,
};
pub use page_object::{
    PageObject, PageObjectBuilder, PageObjectInfo, PageRegistry, SimplePageObject, UrlMatcher,
};
pub use performance::{
    Measurement, MetricStats, MetricType, PerformanceMonitor, PerformanceProfile,
    PerformanceProfiler, PerformanceProfilerBuilder, PerformanceSummary, PerformanceThreshold,
};
pub use playbook::{
    calculate_mutation_score, check_complexity_violation, to_dot, Action as PlaybookAction,
    ActionExecutor, Assertion as PlaybookAssertion, AssertionFailure as PlaybookAssertionFailure,
    ComplexityAnalyzer, ComplexityClass, ComplexityResult, DeterminismInfo,
    ExecutionResult as PlaybookExecutionResult, ExecutorError, Invariant, IssueSeverity,
    MutantResult, MutationClass, MutationGenerator, MutationScore, PerformanceBudget, Playbook,
    PlaybookError, PlaybookExecutor, ReachabilityInfo, State as PlaybookState, StateMachine,
    StateMachineValidator, Transition as PlaybookTransition, ValidationIssue, ValidationResult,
    WaitCondition as PlaybookWaitCondition,
};
pub use presentar::{
    generate_falsification_playbook, parse_and_validate as parse_and_validate_presentar,
    validate_config as validate_presentar_config, Cell as PresentarCell, Color as PresentarColor,
    FalsificationCheck, FalsificationResult, KeybindingConfig, LayoutConfig, PanelConfig,
    PanelConfigs, PanelType, PresentarConfig, PresentarError, TerminalAssertion, TerminalSnapshot,
    ThemeConfig, ValidationResult as PresentarValidationResult, FALSIFICATION_COUNT,
    SCHEMA_VERSION,
};
pub use renacer_integration::{
    ChromeTrace, ChromeTraceEvent, TraceCollector, TraceContext, TraceSpan,
    TracingConfig as RenacerTracingConfig,
};
pub use replay::{
    Replay, ReplayHeader, ReplayPlayer, ReplayRecorder, StateCheckpoint, TimedInput,
    VerificationResult, REPLAY_FORMAT_VERSION,
};
pub use reporter::{
    AndonCordPulled, FailureMode, Reporter, TestResultEntry, TestStatus, TraceData,
};
pub use result::{ProbarError, ProbarResult};
pub use runtime::{
    ComponentId, EntityId, FrameResult, GameHostState, MemoryView, ProbarComponent, ProbarEntity,
    RuntimeConfig, StateDelta, WasmRuntime,
};
pub use shard::{ShardConfig, ShardParseError, ShardReport, ShardedRunner};
pub use simulation::{
    run_replay, run_simulation, RandomWalkAgent, RecordedFrame, ReplayResult, SimulatedGameState,
    SimulationConfig, SimulationRecording,
};
pub use snapshot::{Snapshot, SnapshotConfig, SnapshotDiff};
pub use strict::{
    ChecklistError, ConsoleCapture, ConsoleSeverity, ConsoleValidationError, E2ETestChecklist,
    WasmStrictMode,
};
pub use tracing_support::{
    ConsoleLevel, ConsoleMessage, EventCategory, EventLevel, ExecutionTracer, NetworkEvent,
    SpanStatus, TraceArchive, TraceMetadata, TracedEvent, TracedSpan, TracingConfig,
};
#[cfg(feature = "tui")]
pub use tui::{
    expect_frame, FrameAssertion, FrameSequence, MultiValueTracker, SnapshotManager, TuiFrame,
    TuiSnapshot, TuiTestBackend, ValueTracker,
};
pub use tui_load::{
    ComponentTimings, DataGenerator, IntegrationLoadTest, SyntheticItem, TuiFrameMetrics,
    TuiLoadAssertion, TuiLoadConfig, TuiLoadError, TuiLoadResult, TuiLoadTest,
};
pub use ux_coverage::{
    calculator_coverage, game_coverage, ElementCoverage, ElementId, InteractionType, StateId,
    TrackedInteraction, UxCoverageBuilder, UxCoverageReport, UxCoverageTracker,
};
pub use validators::{
    CompressionAlgorithm, PartialResult, ScreenshotContent, StateTransition, StreamingMetric,
    StreamingMetricRecord, StreamingState, StreamingUxValidator, StreamingValidationError,
    StreamingValidationResult, TestExecutionStats, VuMeterConfig, VuMeterError, VuMeterSample,
};
pub use video_quality::{
    build_ffprobe_args, parse_ffprobe_json, probe_video, validate_video, VideoCheck,
    VideoExpectations, VideoProbe, VideoQualityReport, VideoVerdict,
};
#[cfg(feature = "media")]
pub use visual_regression::{
    perceptual_diff, ImageDiffResult, MaskRegion, ScreenshotComparison, VisualRegressionConfig,
    VisualRegressionTester,
};
pub use wait::{
    wait_timeout, wait_until, FnCondition, LoadState, NavigationOptions, PageEvent, WaitCondition,
    WaitOptions, WaitResult, Waiter, DEFAULT_WAIT_TIMEOUT_MS, NETWORK_IDLE_THRESHOLD_MS,
};
#[cfg(all(not(target_arch = "wasm32"), feature = "watch"))]
pub use watch::{
    FileChange, FileChangeKind, FileWatcher, FnWatchHandler, WatchBuilder, WatchConfig,
    WatchHandler, WatchStats,
};
// Brick Architecture (PROBAR-SPEC-009)
pub use brick::{
    Brick, BrickAssertion, BrickBudget, BrickError, BrickPhase, BrickResult, BrickVerification,
    BudgetViolation,
};
// Zero-Artifact Architecture (PROBAR-SPEC-009-P7)
pub use brick::{
    AudioBrick, AudioParam, BrickWorkerMessage, BrickWorkerMessageDirection, EventBinding,
    EventBrick, EventHandler, EventType, FieldType, RingBufferConfig, WorkerBrick,
    WorkerTransition,
};
pub use brick_house::{BrickHouse, BrickHouseBuilder, BrickTiming, BudgetReport, JidokaAlert};
pub use websocket::{
    MessageDirection, MessageType, MockWebSocketResponse, WebSocketConnection, WebSocketMessage,
    WebSocketMock, WebSocketMonitor, WebSocketMonitorBuilder, WebSocketState,
};

/// Prelude for convenient imports
pub mod prelude {
    pub use super::accessibility::*;
    pub use super::animation::{
        sample_easing, verify_easing, verify_events, verify_timeline, AnimationEvent,
        AnimationEventType, AnimationReport, AnimationTimeline, AnimationVerdict, EasingFunction,
        EasingVerification, EventResult, Keyframe, ObservedEvent,
    };
    pub use super::assertion::*;
    pub use super::audio_quality::{
        analyze_audio, analyze_samples, detect_clipping, detect_silence, AudioLevels,
        AudioQualityConfig, AudioQualityReport, AudioVerdict, ClippingReport, SilenceRegion,
        SilenceReport,
    };
    pub use super::av_sync::{
        compare_edl_to_onsets, default_edl_path, detect_onsets, extract_audio, AudioOnset,
        AudioTickPlacement, AvSyncReport, DetectionConfig, EditDecision, EditDecisionList,
        SegmentSyncResult, SyncVerdict, TickDelta,
    };
    pub use super::video_quality::{
        build_ffprobe_args, parse_ffprobe_json, probe_video, validate_video, VideoCheck,
        VideoExpectations, VideoProbe, VideoQualityReport, VideoVerdict,
    };
    // Brick Architecture (PROBAR-SPEC-009)
    pub use super::brick::*;
    pub use super::brick_house::*;
    pub use super::bridge::*;
    pub use super::browser::*;
    pub use super::capabilities::*;
    pub use super::clock::*;
    pub use super::context::*;
    pub use super::dialog::*;
    pub use super::driver::*;
    pub use super::event::*;
    pub use super::file_ops::*;
    pub use super::fixture::*;
    pub use super::fuzzer::*;
    pub use super::gpu_pixels::*;
    pub use super::har::*;
    pub use super::harness::*;
    pub use super::locator::*;
    pub use super::network::*;
    pub use super::page_object::*;
    pub use super::perf::*;
    pub use super::performance::*;
    #[cfg(feature = "media")]
    pub use super::pixel_coverage::*;
    pub use super::replay::*;
    pub use super::reporter::*;
    pub use super::result::*;
    pub use super::runner::*;
    pub use super::runtime::*;
    pub use super::shard::*;
    pub use super::simulation::*;
    pub use super::snapshot::*;
    // Note: strict::ConsoleMessage conflicts with tracing_support::ConsoleMessage
    // Use explicit imports instead of glob
    pub use super::strict::{
        ChecklistError, ConsoleCapture, ConsoleSeverity, ConsoleValidationError, E2ETestChecklist,
        WasmStrictMode,
    };
    pub use super::tracing_support::*;
    #[cfg(feature = "tui")]
    pub use super::tui::*;
    pub use super::tui_load::{
        ComponentTimings, DataGenerator, IntegrationLoadTest, SyntheticItem, TuiFrameMetrics,
        TuiLoadAssertion, TuiLoadConfig, TuiLoadError, TuiLoadResult, TuiLoadTest,
    };
    pub use super::ux_coverage::*;
    pub use super::validators::*;
    #[cfg(feature = "media")]
    pub use super::visual_regression::*;
    pub use super::worker_harness::*;
    pub use super::zero_js::*;
    // WASM Threading Testing (PROBAR-SPEC-WASM-001)
    pub use super::comply::*;
    pub use super::lint::*;
    pub use super::mock::*;
    // Docker module types are exported with Docker prefix to avoid conflicts
    #[cfg(feature = "docker")]
    pub use super::docker::{
        check_shared_array_buffer_support, validate_coop_coep_headers, Browser as DockerBrowser,
        ContainerConfig, ContainerState, CoopCoepConfig, DockerConfig, DockerError, DockerResult,
        DockerTestRunner, DockerTestRunnerBuilder, ParallelRunner, ParallelRunnerBuilder,
        TestResult as DockerTestResult, TestResults as DockerTestResults,
    };
    #[cfg(feature = "llm")]
    pub use super::llm::*;
    pub use super::wait::{
        wait_timeout, wait_until, FnCondition, LoadState, NavigationOptions, PageEvent,
        WaitCondition, WaitOptions, WaitResult, Waiter, DEFAULT_WAIT_TIMEOUT_MS,
        NETWORK_IDLE_THRESHOLD_MS,
    };
    #[cfg(all(not(target_arch = "wasm32"), feature = "watch"))]
    pub use super::watch::*;
    pub use super::web::*;
    pub use super::websocket::*;
    // Note: renacer_integration types are available as RenacerTracingConfig, etc.
    // to avoid conflicts with tracing_support::TracingConfig
    pub use super::renacer_integration::{
        ChromeTrace as RenacerChromeTrace, ChromeTraceEvent, TraceCollector, TraceContext,
        TraceSpan, TracingConfig as RenacerTracingConfig,
    };
}

/// Standard invariants for game testing
pub mod standard_invariants {
    pub use super::fuzzer::standard_invariants::*;
}

// Re-export derive macros when the `derive` feature is enabled (Phase 4: Poka-Yoke)
#[cfg(feature = "derive")]
pub use jugar_probar_derive::{probar_test, ProbarComponent, ProbarEntity, ProbarSelector};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // ========================================================================
    // EXTREME TDD: Tests written FIRST per spec Section 6.1
    // ========================================================================

    mod browser_tests {
        use super::*;

        #[test]
        fn test_browser_config_defaults() {
            let config = BrowserConfig::default();
            assert!(config.headless);
            assert_eq!(config.viewport_width, 800);
            assert_eq!(config.viewport_height, 600);
        }

        #[test]
        fn test_browser_config_builder() {
            let config = BrowserConfig::default()
                .with_viewport(1024, 768)
                .with_headless(false);
            assert!(!config.headless);
            assert_eq!(config.viewport_width, 1024);
            assert_eq!(config.viewport_height, 768);
        }
    }

    mod touch_tests {
        use super::*;

        #[test]
        fn test_touch_tap() {
            let touch = Touch::tap(100.0, 200.0);
            assert!((touch.x - 100.0).abs() < f32::EPSILON);
            assert!((touch.y - 200.0).abs() < f32::EPSILON);
            assert!(matches!(touch.action, TouchAction::Tap));
        }

        #[test]
        fn test_touch_swipe() {
            let touch = Touch::swipe(0.0, 0.0, 100.0, 0.0, 300);
            assert!(matches!(touch.action, TouchAction::Swipe { .. }));
        }

        #[test]
        fn test_touch_hold() {
            let touch = Touch::hold(50.0, 50.0, 500);
            assert!(matches!(touch.action, TouchAction::Hold { .. }));
        }
    }

    mod assertion_tests {
        use super::*;

        #[test]
        fn test_assertion_result_pass() {
            let result = AssertionResult::pass();
            assert!(result.passed);
            assert!(result.message.is_empty());
        }

        #[test]
        fn test_assertion_result_fail() {
            let result = AssertionResult::fail("test error message");
            assert!(!result.passed);
            assert_eq!(result.message, "test error message");
        }

        #[test]
        fn test_assertion_equals_pass() {
            let result = Assertion::equals(&42, &42);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_equals_fail() {
            let result = Assertion::equals(&42, &43);
            assert!(!result.passed);
            assert!(result.message.contains("expected"));
        }

        #[test]
        fn test_assertion_contains_pass() {
            let result = Assertion::contains("hello world", "world");
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_contains_fail() {
            let result = Assertion::contains("hello world", "foo");
            assert!(!result.passed);
            assert!(result.message.contains("contain"));
        }

        #[test]
        fn test_assertion_in_range_pass() {
            let result = Assertion::in_range(5.0, 0.0, 10.0);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_in_range_fail() {
            let result = Assertion::in_range(15.0, 0.0, 10.0);
            assert!(!result.passed);
            assert!(result.message.contains("range"));
        }

        #[test]
        fn test_assertion_in_range_at_boundaries() {
            // At min boundary
            let result = Assertion::in_range(0.0, 0.0, 10.0);
            assert!(result.passed);
            // At max boundary
            let result = Assertion::in_range(10.0, 0.0, 10.0);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_is_true_pass() {
            let result = Assertion::is_true(true, "should be true");
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_is_true_fail() {
            let result = Assertion::is_true(false, "expected true");
            assert!(!result.passed);
            assert_eq!(result.message, "expected true");
        }

        #[test]
        fn test_assertion_is_false_pass() {
            let result = Assertion::is_false(false, "should be false");
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_is_false_fail() {
            let result = Assertion::is_false(true, "expected false");
            assert!(!result.passed);
            assert_eq!(result.message, "expected false");
        }

        #[test]
        fn test_assertion_is_some_pass() {
            let opt = Some(42);
            let result = Assertion::is_some(&opt);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_is_some_fail() {
            let opt: Option<i32> = None;
            let result = Assertion::is_some(&opt);
            assert!(!result.passed);
            assert!(result.message.contains("None"));
        }

        #[test]
        fn test_assertion_is_none_pass() {
            let opt: Option<i32> = None;
            let result = Assertion::is_none(&opt);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_is_none_fail() {
            let opt = Some(42);
            let result = Assertion::is_none(&opt);
            assert!(!result.passed);
            assert!(result.message.contains("Some"));
        }

        #[test]
        fn test_assertion_is_ok_pass() {
            let res: Result<i32, &str> = Ok(42);
            let result = Assertion::is_ok(&res);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_is_ok_fail() {
            let res: Result<i32, &str> = Err("error");
            let result = Assertion::is_ok(&res);
            assert!(!result.passed);
            assert!(result.message.contains("Err"));
        }

        #[test]
        fn test_assertion_is_err_pass() {
            let res: Result<i32, &str> = Err("error");
            let result = Assertion::is_err(&res);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_is_err_fail() {
            let res: Result<i32, &str> = Ok(42);
            let result = Assertion::is_err(&res);
            assert!(!result.passed);
            assert!(result.message.contains("Ok"));
        }

        #[test]
        fn test_assertion_approx_eq_pass() {
            let result = Assertion::approx_eq(1.0, 1.0001, 0.01);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_approx_eq_fail() {
            let result = Assertion::approx_eq(1.0, 2.0, 0.01);
            assert!(!result.passed);
            assert!(result.message.contains("≈"));
        }

        #[test]
        fn test_assertion_has_length_pass() {
            let data = vec![1, 2, 3, 4, 5];
            let result = Assertion::has_length(&data, 5);
            assert!(result.passed);
        }

        #[test]
        fn test_assertion_has_length_fail() {
            let data = vec![1, 2, 3];
            let result = Assertion::has_length(&data, 5);
            assert!(!result.passed);
            assert!(result.message.contains("length"));
        }

        #[test]
        fn test_assertion_has_length_empty() {
            let data: Vec<i32> = vec![];
            let result = Assertion::has_length(&data, 0);
            assert!(result.passed);
        }
    }

    mod snapshot_tests {
        use super::*;

        #[test]
        fn test_snapshot_creation() {
            let snapshot = Snapshot::new("test-snapshot", vec![0, 1, 2, 3]);
            assert_eq!(snapshot.name, "test-snapshot");
            assert_eq!(snapshot.data.len(), 4);
            assert_eq!(snapshot.width, 0);
            assert_eq!(snapshot.height, 0);
        }

        #[test]
        fn test_snapshot_with_dimensions() {
            let snapshot = Snapshot::new("test", vec![1, 2, 3, 4]).with_dimensions(800, 600);
            assert_eq!(snapshot.width, 800);
            assert_eq!(snapshot.height, 600);
        }

        #[test]
        fn test_snapshot_size() {
            let snapshot = Snapshot::new("test", vec![1, 2, 3, 4, 5]);
            assert_eq!(snapshot.size(), 5);
        }

        #[test]
        fn test_snapshot_diff_identical() {
            let snap1 = Snapshot::new("test", vec![1, 2, 3]);
            let snap2 = Snapshot::new("test", vec![1, 2, 3]);
            let diff = snap1.diff(&snap2);
            assert!(diff.is_identical());
            assert_eq!(diff.difference_count, 0);
            assert!((diff.difference_percent - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_snapshot_diff_different() {
            let snap1 = Snapshot::new("test", vec![1, 2, 3]);
            let snap2 = Snapshot::new("test", vec![1, 2, 4]);
            let diff = snap1.diff(&snap2);
            assert!(!diff.is_identical());
            assert_eq!(diff.difference_count, 1);
        }

        #[test]
        fn test_snapshot_diff_empty() {
            let snap1 = Snapshot::new("test", vec![]);
            let snap2 = Snapshot::new("test", vec![]);
            let diff = snap1.diff(&snap2);
            assert!(diff.is_identical());
            assert!((diff.difference_percent - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_snapshot_diff_different_lengths() {
            let snap1 = Snapshot::new("test", vec![1, 2, 3]);
            let snap2 = Snapshot::new("test", vec![1, 2, 3, 4, 5]);
            let diff = snap1.diff(&snap2);
            assert!(!diff.is_identical());
            // Missing bytes count as differences
            assert_eq!(diff.difference_count, 2);
        }

        #[test]
        fn test_snapshot_diff_within_threshold() {
            let snap1 = Snapshot::new("test", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            let snap2 = Snapshot::new("test", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11]);
            let diff = snap1.diff(&snap2);
            // 1 difference out of 10 = 10%
            assert!(diff.within_threshold(0.1)); // 10% threshold
            assert!(!diff.within_threshold(0.05)); // 5% threshold
        }

        #[test]
        fn test_snapshot_config_default() {
            let config = SnapshotConfig::default();
            assert!(!config.update_snapshots);
            assert!((config.threshold - 0.01).abs() < f64::EPSILON);
            assert_eq!(config.snapshot_dir, "__snapshots__");
        }

        #[test]
        fn test_snapshot_config_with_update() {
            let config = SnapshotConfig::default().with_update(true);
            assert!(config.update_snapshots);
        }

        #[test]
        fn test_snapshot_config_with_threshold() {
            let config = SnapshotConfig::default().with_threshold(0.05);
            assert!((config.threshold - 0.05).abs() < f64::EPSILON);
        }

        #[test]
        fn test_snapshot_config_with_dir() {
            let config = SnapshotConfig::default().with_dir("custom_snapshots");
            assert_eq!(config.snapshot_dir, "custom_snapshots");
        }

        #[test]
        fn test_snapshot_config_chained_builders() {
            let config = SnapshotConfig::default()
                .with_update(true)
                .with_threshold(0.02)
                .with_dir("my_snaps");
            assert!(config.update_snapshots);
            assert!((config.threshold - 0.02).abs() < f64::EPSILON);
            assert_eq!(config.snapshot_dir, "my_snaps");
        }
    }

    mod harness_tests {
        use super::*;
        use harness::{SuiteResults, TestCase};
        use std::time::Duration;

        #[test]
        fn test_test_suite_creation() {
            let suite = TestSuite::new("Game Tests");
            assert_eq!(suite.name, "Game Tests");
            assert!(suite.tests.is_empty());
        }

        #[test]
        fn test_test_suite_add_test() {
            let mut suite = TestSuite::new("Suite");
            suite.add_test(TestCase::new("test1"));
            suite.add_test(TestCase::new("test2"));
            assert_eq!(suite.test_count(), 2);
        }

        #[test]
        fn test_test_case_creation() {
            let case = TestCase::new("my_test");
            assert_eq!(case.name, "my_test");
            assert_eq!(case.timeout_ms, 30000); // default timeout
        }

        #[test]
        fn test_test_case_with_timeout() {
            let case = TestCase::new("my_test").with_timeout(5000);
            assert_eq!(case.timeout_ms, 5000);
        }

        #[test]
        fn test_test_result_pass() {
            let result = TestResult::pass("test_example");
            assert!(result.passed);
            assert_eq!(result.name, "test_example");
            assert!(result.error.is_none());
            assert_eq!(result.duration, Duration::ZERO);
        }

        #[test]
        fn test_test_result_fail() {
            let result = TestResult::fail("test_example", "assertion failed");
            assert!(!result.passed);
            assert!(result.error.is_some());
            assert_eq!(result.error.unwrap(), "assertion failed");
        }

        #[test]
        fn test_test_result_with_duration() {
            let result = TestResult::pass("test").with_duration(Duration::from_millis(100));
            assert_eq!(result.duration, Duration::from_millis(100));
        }

        #[test]
        fn test_suite_results_all_passed() {
            let results = SuiteResults {
                suite_name: "test".to_string(),
                results: vec![TestResult::pass("test1"), TestResult::pass("test2")],
                duration: Duration::ZERO,
            };
            assert!(results.all_passed());
        }

        #[test]
        fn test_suite_results_not_all_passed() {
            let results = SuiteResults {
                suite_name: "test".to_string(),
                results: vec![
                    TestResult::pass("test1"),
                    TestResult::fail("test2", "error"),
                ],
                duration: Duration::ZERO,
            };
            assert!(!results.all_passed());
        }

        #[test]
        fn test_suite_results_counts() {
            let results = SuiteResults {
                suite_name: "test".to_string(),
                results: vec![
                    TestResult::pass("test1"),
                    TestResult::fail("test2", "error"),
                    TestResult::pass("test3"),
                ],
                duration: Duration::ZERO,
            };
            assert_eq!(results.passed_count(), 2);
            assert_eq!(results.failed_count(), 1);
            assert_eq!(results.total(), 3);
        }

        #[test]
        fn test_suite_results_failures() {
            let results = SuiteResults {
                suite_name: "test".to_string(),
                results: vec![
                    TestResult::pass("test1"),
                    TestResult::fail("test2", "error2"),
                    TestResult::fail("test3", "error3"),
                ],
                duration: Duration::ZERO,
            };
            let failures = results.failures();
            assert_eq!(failures.len(), 2);
            assert_eq!(failures[0].name, "test2");
            assert_eq!(failures[1].name, "test3");
        }

        #[test]
        fn test_harness_run_empty_suite() {
            let harness = TestHarness::new();
            let suite = TestSuite::new("Empty");
            let results = harness.run(&suite);
            assert!(results.all_passed());
            assert_eq!(results.total(), 0);
        }

        #[test]
        fn test_harness_with_fail_fast() {
            let harness = TestHarness::new().with_fail_fast();
            assert!(harness.fail_fast);
        }

        #[test]
        fn test_harness_with_parallel() {
            let harness = TestHarness::new().with_parallel();
            assert!(harness.parallel);
        }

        #[test]
        fn test_harness_default() {
            let harness = TestHarness::default();
            assert!(!harness.fail_fast);
            assert!(!harness.parallel);
        }
    }

    mod input_event_tests {
        use super::*;

        #[test]
        fn test_input_event_touch() {
            let event = InputEvent::touch(100.0, 200.0);
            assert!(
                matches!(event, InputEvent::Touch { x, y } if (x - 100.0).abs() < f32::EPSILON && (y - 200.0).abs() < f32::EPSILON)
            );
        }

        #[test]
        fn test_input_event_key_press() {
            let event = InputEvent::key_press("ArrowUp");
            assert!(matches!(event, InputEvent::KeyPress { key } if key == "ArrowUp"));
        }

        #[test]
        fn test_input_event_key_release() {
            let event = InputEvent::key_release("Space");
            assert!(matches!(event, InputEvent::KeyRelease { key } if key == "Space"));
        }

        #[test]
        fn test_input_event_mouse_click() {
            let event = InputEvent::mouse_click(50.0, 75.0);
            assert!(
                matches!(event, InputEvent::MouseClick { x, y } if (x - 50.0).abs() < f32::EPSILON && (y - 75.0).abs() < f32::EPSILON)
            );
        }

        #[test]
        fn test_input_event_mouse_move() {
            let event = InputEvent::mouse_move(150.0, 250.0);
            assert!(
                matches!(event, InputEvent::MouseMove { x, y } if (x - 150.0).abs() < f32::EPSILON && (y - 250.0).abs() < f32::EPSILON)
            );
        }

        #[test]
        fn test_input_event_gamepad_button_pressed() {
            let event = InputEvent::gamepad_button(0, true);
            assert!(matches!(
                event,
                InputEvent::GamepadButton {
                    button: 0,
                    pressed: true
                }
            ));
        }

        #[test]
        fn test_input_event_gamepad_button_released() {
            let event = InputEvent::gamepad_button(1, false);
            assert!(matches!(
                event,
                InputEvent::GamepadButton {
                    button: 1,
                    pressed: false
                }
            ));
        }

        #[test]
        fn test_touch_tap_coordinates() {
            let touch = Touch::tap(100.0, 200.0);
            assert!((touch.x - 100.0).abs() < f32::EPSILON);
            assert!((touch.y - 200.0).abs() < f32::EPSILON);
            assert!(matches!(touch.action, TouchAction::Tap));
        }

        #[test]
        fn test_touch_swipe_full_properties() {
            let touch = Touch::swipe(10.0, 20.0, 100.0, 200.0, 300);
            assert!((touch.x - 10.0).abs() < f32::EPSILON);
            assert!((touch.y - 20.0).abs() < f32::EPSILON);
            match touch.action {
                TouchAction::Swipe {
                    end_x,
                    end_y,
                    duration_ms,
                } => {
                    assert!((end_x - 100.0).abs() < f32::EPSILON);
                    assert!((end_y - 200.0).abs() < f32::EPSILON);
                    assert_eq!(duration_ms, 300);
                }
                _ => panic!("expected Swipe action"),
            }
        }

        #[test]
        fn test_touch_hold_full_properties() {
            let touch = Touch::hold(50.0, 60.0, 500);
            assert!((touch.x - 50.0).abs() < f32::EPSILON);
            assert!((touch.y - 60.0).abs() < f32::EPSILON);
            match touch.action {
                TouchAction::Hold { duration_ms } => {
                    assert_eq!(duration_ms, 500);
                }
                _ => panic!("expected Hold action"),
            }
        }

        #[test]
        fn test_touch_action_equality() {
            assert_eq!(TouchAction::Tap, TouchAction::Tap);
            let swipe1 = TouchAction::Swipe {
                end_x: 1.0,
                end_y: 2.0,
                duration_ms: 100,
            };
            let swipe2 = TouchAction::Swipe {
                end_x: 1.0,
                end_y: 2.0,
                duration_ms: 100,
            };
            assert_eq!(swipe1, swipe2);
            let hold1 = TouchAction::Hold { duration_ms: 500 };
            let hold2 = TouchAction::Hold { duration_ms: 500 };
            assert_eq!(hold1, hold2);
        }

        #[test]
        fn test_touch_equality() {
            let t1 = Touch::tap(100.0, 200.0);
            let t2 = Touch::tap(100.0, 200.0);
            assert_eq!(t1, t2);
        }

        #[test]
        fn test_input_event_equality() {
            let e1 = InputEvent::touch(10.0, 20.0);
            let e2 = InputEvent::touch(10.0, 20.0);
            assert_eq!(e1, e2);
        }
    }

    mod error_tests {
        use super::*;

        #[test]
        fn test_probar_error_display() {
            let err = ProbarError::BrowserNotFound;
            let msg = err.to_string();
            assert!(msg.contains("browser") || msg.contains("Browser"));
        }

        #[test]
        fn test_probar_error_timeout() {
            let err = ProbarError::Timeout { ms: 5000 };
            let msg = err.to_string();
            assert!(msg.contains("5000"));
        }
    }
}
