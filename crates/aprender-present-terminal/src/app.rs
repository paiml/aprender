//! TUI application runner with Jidoka verification gates.
//!
//! ## Non-Blocking UI Pattern (CB-INPUT-006)
//!
//! For applications with heavy data collection (system monitors, dashboards),
//! use the [`AsyncCollector`] pattern to ensure the main thread never blocks.
//!
//! ```ignore
//! // Background thread owns collectors, sends snapshots through channel
//! let (tx, rx) = mpsc::channel::<MySnapshot>();
//!
//! std::thread::spawn(move || {
//!     let mut collector = MyCollector::new();
//!     loop {
//!         let snapshot = collector.collect();  // Can take seconds
//!         tx.send(snapshot).ok();
//!         std::thread::sleep(Duration::from_secs(1));
//!     }
//! });
//!
//! // Main thread: input + render only (always <16ms)
//! loop {
//!     while let Ok(snapshot) = rx.try_recv() {
//!         app.apply_snapshot(snapshot);  // O(1) operation
//!     }
//!     app.handle_input();  // Non-blocking
//!     app.render();        // <16ms budget
//! }
//! ```

#![allow(dead_code, unreachable_pub)]

use crate::color::ColorMode;
use crate::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use crate::error::{TuiError, VerificationError};
use crate::input::InputHandler;
use crossterm::{
    cursor,
    event::{self, Event as CrosstermEvent, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use presentar_core::{Constraints, Rect, Widget};
use std::io::{self, Stdout, Write};
use std::time::{Duration, Instant};

// =============================================================================
// Non-Blocking UI Pattern (CB-INPUT-006)
// =============================================================================

/// Snapshot of collected metrics, transportable via channel.
///
/// Implement this trait for data structures that are sent from a background
/// collector thread to the main UI thread.
///
/// # Requirements
/// - Must be `Clone` (for potential buffering)
/// - Must be `Send` (for channel transport)
/// - Must be `'static` (for thread safety)
pub trait Snapshot: Clone + Send + 'static {
    /// Create an empty snapshot for initial state before first collection.
    fn empty() -> Self;
}

/// Background collector that produces snapshots.
///
/// Implement this trait for objects that collect metrics in a background thread.
/// The collector owns all heavy I/O objects (System, Disks, Networks, etc.)
/// and produces lightweight snapshots that can be sent through a channel.
///
/// # Example
/// ```ignore
/// struct SystemCollector {
///     system: System,
///     disks: Disks,
/// }
///
/// impl AsyncCollector for SystemCollector {
///     type Snapshot = MetricsSnapshot;
///
///     fn collect(&mut self) -> MetricsSnapshot {
///         self.system.refresh_all();  // Heavy I/O
///         MetricsSnapshot {
///             cpu_usage: self.system.global_cpu_usage(),
///             // ... extract other data
///         }
///     }
/// }
/// ```
pub trait AsyncCollector: Send + 'static {
    /// The snapshot type produced by this collector.
    type Snapshot: Snapshot;

    /// Collect metrics and return a snapshot.
    ///
    /// This method may take seconds to complete (heavy I/O).
    /// It runs in a background thread, never blocking the UI.
    fn collect(&mut self) -> Self::Snapshot;
}

/// Application that can apply snapshots to update its state.
///
/// Implement this trait for your application state. The `apply_snapshot`
/// method is called on the main thread and MUST complete in <1ms.
///
/// # Example
/// ```ignore
/// impl SnapshotReceiver for MyApp {
///     type Snapshot = MetricsSnapshot;
///
///     fn apply_snapshot(&mut self, snapshot: MetricsSnapshot) {
///         // O(1) operations only - just copy/swap data
///         self.cpu_usage = snapshot.cpu_usage;
///         self.processes = snapshot.processes;
///     }
/// }
/// ```
pub trait SnapshotReceiver {
    /// The snapshot type this receiver accepts.
    type Snapshot: Snapshot;

    /// Apply a snapshot to update the application state.
    ///
    /// **MUST be O(1) and complete in <1ms.**
    /// Only perform simple assignments, no I/O or heavy computation.
    fn apply_snapshot(&mut self, snapshot: Self::Snapshot);
}

/// QA timing diagnostics for non-blocking UI verification.
///
/// Use this struct to collect timing data for `--qa-timing` output.
#[derive(Debug, Clone, Default)]
pub struct QaTimings {
    /// Input event processing times in microseconds.
    pub input_times_us: Vec<u64>,
    /// Lock acquisition times in microseconds (should be 0 with channel pattern).
    pub lock_times_us: Vec<u64>,
    /// Render times in microseconds.
    pub render_times_us: Vec<u64>,
    /// Last collect duration in microseconds (from background thread).
    pub last_collect_us: u64,
}

impl QaTimings {
    /// Create new QA timing collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an input event processing time.
    pub fn record_input(&mut self, duration: Duration) {
        self.input_times_us.push(duration.as_micros() as u64);
    }

    /// Record a lock acquisition time.
    pub fn record_lock(&mut self, duration: Duration) {
        self.lock_times_us.push(duration.as_micros() as u64);
    }

    /// Record a render time.
    pub fn record_render(&mut self, duration: Duration) {
        contract_pre_render!();
        self.render_times_us.push(duration.as_micros() as u64);
    }

    /// Format timing report for stderr output.
    #[must_use]
    pub fn format_report(&self) -> String {
        let avg = |v: &[u64]| {
            if v.is_empty() {
                0
            } else {
                v.iter().sum::<u64>() / v.len() as u64
            }
        };
        let max = |v: &[u64]| v.iter().max().copied().unwrap_or(0);

        format!(
            "[QA] input: avg={}us max={}us | lock: avg={}us max={}us | render: avg={}us max={}us | collect: {}us",
            avg(&self.input_times_us), max(&self.input_times_us),
            avg(&self.lock_times_us), max(&self.lock_times_us),
            avg(&self.render_times_us), max(&self.render_times_us),
            self.last_collect_us
        )
    }

    /// Clear accumulated timing data.
    pub fn clear(&mut self) {
        self.input_times_us.clear();
        self.lock_times_us.clear();
        self.render_times_us.clear();
    }
}

// =============================================================================
// Terminal Abstraction
// =============================================================================

/// Terminal abstraction for testability.
pub trait Terminal {
    /// Enter raw mode and alternate screen.
    fn enter(&mut self) -> Result<(), TuiError>;
    /// Leave alternate screen and raw mode.
    fn leave(&mut self) -> Result<(), TuiError>;
    /// Get terminal size (width, height).
    fn size(&self) -> Result<(u16, u16), TuiError>;
    /// Poll for events with timeout.
    fn poll(&self, timeout: Duration) -> Result<bool, TuiError>;
    /// Read the next event.
    fn read_event(&self) -> Result<CrosstermEvent, TuiError>;
    /// Flush output.
    fn flush(
        &mut self,
        buffer: &mut CellBuffer,
        renderer: &mut DiffRenderer,
    ) -> Result<(), TuiError>;
    /// Enable mouse capture.
    fn enable_mouse(&mut self) -> Result<(), TuiError>;
    /// Disable mouse capture.
    fn disable_mouse(&mut self) -> Result<(), TuiError>;
}

/// Backend trait for raw terminal operations (crossterm calls).
/// This layer exists purely for testability.
pub trait TerminalBackend {
    fn enable_raw_mode(&mut self) -> Result<(), TuiError>;
    fn disable_raw_mode(&mut self) -> Result<(), TuiError>;
    fn enter_alternate_screen(&mut self) -> Result<(), TuiError>;
    fn leave_alternate_screen(&mut self) -> Result<(), TuiError>;
    fn hide_cursor(&mut self) -> Result<(), TuiError>;
    fn show_cursor(&mut self) -> Result<(), TuiError>;
    fn size(&self) -> Result<(u16, u16), TuiError>;
    fn poll(&self, timeout: Duration) -> Result<bool, TuiError>;
    fn read_event(&self) -> Result<CrosstermEvent, TuiError>;
    fn write_flush(
        &mut self,
        buffer: &mut CellBuffer,
        renderer: &mut DiffRenderer,
    ) -> Result<(), TuiError>;
    fn enable_mouse_capture(&mut self) -> Result<(), TuiError>;
    fn disable_mouse_capture(&mut self) -> Result<(), TuiError>;
}

/// Real crossterm backend.
pub struct CrosstermBackend {
    stdout: Stdout,
}

impl CrosstermBackend {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
        }
    }
}

impl Default for CrosstermBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalBackend for CrosstermBackend {
    fn enable_raw_mode(&mut self) -> Result<(), TuiError> {
        enable_raw_mode()?;
        Ok(())
    }
    fn disable_raw_mode(&mut self) -> Result<(), TuiError> {
        let _ = disable_raw_mode();
        Ok(())
    }
    fn enter_alternate_screen(&mut self) -> Result<(), TuiError> {
        execute!(self.stdout, EnterAlternateScreen)?;
        Ok(())
    }
    fn leave_alternate_screen(&mut self) -> Result<(), TuiError> {
        let _ = execute!(self.stdout, LeaveAlternateScreen);
        Ok(())
    }
    fn hide_cursor(&mut self) -> Result<(), TuiError> {
        execute!(self.stdout, cursor::Hide)?;
        Ok(())
    }
    fn show_cursor(&mut self) -> Result<(), TuiError> {
        let _ = execute!(self.stdout, cursor::Show);
        Ok(())
    }
    fn size(&self) -> Result<(u16, u16), TuiError> {
        Ok(crossterm::terminal::size()?)
    }
    fn poll(&self, timeout: Duration) -> Result<bool, TuiError> {
        Ok(event::poll(timeout)?)
    }
    fn read_event(&self) -> Result<CrosstermEvent, TuiError> {
        Ok(event::read()?)
    }
    fn write_flush(
        &mut self,
        buffer: &mut CellBuffer,
        renderer: &mut DiffRenderer,
    ) -> Result<(), TuiError> {
        renderer.flush(buffer, &mut self.stdout)?;
        self.stdout.flush()?;
        Ok(())
    }
    fn enable_mouse_capture(&mut self) -> Result<(), TuiError> {
        execute!(self.stdout, crossterm::event::EnableMouseCapture)?;
        Ok(())
    }
    fn disable_mouse_capture(&mut self) -> Result<(), TuiError> {
        let _ = execute!(self.stdout, crossterm::event::DisableMouseCapture);
        Ok(())
    }
}

/// Testable backend with generic writer for capturing escape sequences.
/// This backend allows testing terminal output without a real TTY.
#[allow(clippy::struct_excessive_bools)]
pub struct TestableBackend<W: Write> {
    writer: W,
    size: (u16, u16),
    raw_mode: bool,
    alternate_screen: bool,
    cursor_hidden: bool,
    mouse_captured: bool,
    events: std::cell::RefCell<std::collections::VecDeque<CrosstermEvent>>,
    poll_results: std::cell::RefCell<std::collections::VecDeque<bool>>,
}

impl<W: Write> TestableBackend<W> {
    /// Create a new testable backend with the given writer and size.
    pub fn new(writer: W, width: u16, height: u16) -> Self {
        Self {
            writer,
            size: (width, height),
            raw_mode: false,
            alternate_screen: false,
            cursor_hidden: false,
            mouse_captured: false,
            events: std::cell::RefCell::new(std::collections::VecDeque::new()),
            poll_results: std::cell::RefCell::new(std::collections::VecDeque::new()),
        }
    }

    /// Queue events to be returned by `read_event`.
    pub fn with_events(self, events: Vec<CrosstermEvent>) -> Self {
        *self.events.borrow_mut() = events.into_iter().collect();
        self
    }

    /// Queue poll results.
    pub fn with_polls(self, polls: Vec<bool>) -> Self {
        *self.poll_results.borrow_mut() = polls.into_iter().collect();
        self
    }

    /// Check if raw mode was enabled.
    pub fn is_raw_mode(&self) -> bool {
        self.raw_mode
    }

    /// Check if alternate screen was entered.
    pub fn is_alternate_screen(&self) -> bool {
        self.alternate_screen
    }

    /// Check if cursor is hidden.
    pub fn is_cursor_hidden(&self) -> bool {
        self.cursor_hidden
    }

    /// Check if mouse is captured.
    pub fn is_mouse_captured(&self) -> bool {
        self.mouse_captured
    }

    /// Get the underlying writer (consumes self).
    pub fn into_writer(self) -> W {
        self.writer
    }
}

impl<W: Write> TerminalBackend for TestableBackend<W> {
    fn enable_raw_mode(&mut self) -> Result<(), TuiError> {
        self.raw_mode = true;
        Ok(())
    }

    fn disable_raw_mode(&mut self) -> Result<(), TuiError> {
        self.raw_mode = false;
        Ok(())
    }

    fn enter_alternate_screen(&mut self) -> Result<(), TuiError> {
        self.alternate_screen = true;
        // Write the actual escape sequence for testing
        execute!(self.writer, EnterAlternateScreen)?;
        Ok(())
    }

    fn leave_alternate_screen(&mut self) -> Result<(), TuiError> {
        self.alternate_screen = false;
        let _ = execute!(self.writer, LeaveAlternateScreen);
        Ok(())
    }

    fn hide_cursor(&mut self) -> Result<(), TuiError> {
        self.cursor_hidden = true;
        execute!(self.writer, cursor::Hide)?;
        Ok(())
    }

    fn show_cursor(&mut self) -> Result<(), TuiError> {
        self.cursor_hidden = false;
        let _ = execute!(self.writer, cursor::Show);
        Ok(())
    }

    fn size(&self) -> Result<(u16, u16), TuiError> {
        Ok(self.size)
    }

    fn poll(&self, _timeout: Duration) -> Result<bool, TuiError> {
        Ok(self.poll_results.borrow_mut().pop_front().unwrap_or(false))
    }

    fn read_event(&self) -> Result<CrosstermEvent, TuiError> {
        self.events
            .borrow_mut()
            .pop_front()
            .ok_or_else(|| TuiError::Io(io::Error::new(io::ErrorKind::WouldBlock, "no events")))
    }

    fn write_flush(
        &mut self,
        buffer: &mut CellBuffer,
        renderer: &mut DiffRenderer,
    ) -> Result<(), TuiError> {
        renderer.flush(buffer, &mut self.writer)?;
        self.writer.flush()?;
        Ok(())
    }

    fn enable_mouse_capture(&mut self) -> Result<(), TuiError> {
        self.mouse_captured = true;
        execute!(self.writer, crossterm::event::EnableMouseCapture)?;
        Ok(())
    }

    fn disable_mouse_capture(&mut self) -> Result<(), TuiError> {
        self.mouse_captured = false;
        let _ = execute!(self.writer, crossterm::event::DisableMouseCapture);
        Ok(())
    }
}

/// Generic terminal implementation using a backend.
pub struct GenericTerminal<B: TerminalBackend> {
    backend: B,
}

impl<B: TerminalBackend> GenericTerminal<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: TerminalBackend> Terminal for GenericTerminal<B> {
    fn enter(&mut self) -> Result<(), TuiError> {
        self.backend.enable_raw_mode()?;
        self.backend.enter_alternate_screen()?;
        self.backend.hide_cursor()?;
        Ok(())
    }

    fn leave(&mut self) -> Result<(), TuiError> {
        self.backend.show_cursor()?;
        self.backend.leave_alternate_screen()?;
        self.backend.disable_raw_mode()?;
        Ok(())
    }

    fn size(&self) -> Result<(u16, u16), TuiError> {
        self.backend.size()
    }

    fn poll(&self, timeout: Duration) -> Result<bool, TuiError> {
        self.backend.poll(timeout)
    }

    fn read_event(&self) -> Result<CrosstermEvent, TuiError> {
        self.backend.read_event()
    }

    fn flush(
        &mut self,
        buffer: &mut CellBuffer,
        renderer: &mut DiffRenderer,
    ) -> Result<(), TuiError> {
        self.backend.write_flush(buffer, renderer)
    }

    fn enable_mouse(&mut self) -> Result<(), TuiError> {
        self.backend.enable_mouse_capture()
    }

    fn disable_mouse(&mut self) -> Result<(), TuiError> {
        self.backend.disable_mouse_capture()
    }
}

/// Convenience alias for crossterm-backed terminal.
pub type CrosstermTerminal = GenericTerminal<CrosstermBackend>;

/// Configuration for the TUI application.
#[derive(Debug, Clone)]
pub struct TuiConfig {
    /// Tick rate in milliseconds for input polling.
    pub tick_rate_ms: u64,
    /// Enable mouse support.
    pub enable_mouse: bool,
    /// Color mode (auto-detected if not specified).
    pub color_mode: Option<ColorMode>,
    /// Skip Brick verification (DANGEROUS - for debugging only).
    pub skip_verification: bool,
    /// Target frame rate (used for budget calculation).
    pub target_fps: u32,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            tick_rate_ms: 250,
            enable_mouse: false,
            color_mode: None,
            target_fps: 60,
            skip_verification: false,
        }
    }
}

impl TuiConfig {
    /// Create a high-performance config (60fps, fast tick).
    #[must_use]
    pub fn high_performance() -> Self {
        Self {
            tick_rate_ms: 16,
            target_fps: 60,
            ..Default::default()
        }
    }

    /// Create a power-saving config (30fps, slow tick).
    #[must_use]
    pub fn power_saving() -> Self {
        Self {
            tick_rate_ms: 100,
            target_fps: 30,
            ..Default::default()
        }
    }
}

/// Frame timing metrics.
#[derive(Debug, Clone, Default)]
pub struct FrameMetrics {
    /// Time spent in verification phase.
    pub verify_time: Duration,
    /// Time spent in measure phase.
    pub measure_time: Duration,
    /// Time spent in layout phase.
    pub layout_time: Duration,
    /// Time spent in paint phase.
    pub paint_time: Duration,
    /// Total frame time.
    pub total_time: Duration,
    /// Frame number.
    pub frame_count: u64,
}

/// Main TUI application runner.
pub struct TuiApp<W: Widget> {
    root: W,
    config: TuiConfig,
    input_handler: InputHandler,
    metrics: FrameMetrics,
    should_quit: bool,
    color_mode: ColorMode,
}

impl<W: Widget> std::fmt::Debug for TuiApp<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiApp")
            .field("config", &self.config)
            .field("input_handler", &self.input_handler)
            .field("metrics", &self.metrics)
            .field("should_quit", &self.should_quit)
            .field("color_mode", &self.color_mode)
            .finish_non_exhaustive()
    }
}

/// Internal app runner that accepts a Terminal implementation.
struct AppRunner<'a, W: Widget, T: Terminal> {
    app: &'a mut TuiApp<W>,
    terminal: T,
    buffer: CellBuffer,
    renderer: DiffRenderer,
}

impl<W: Widget, T: Terminal> AppRunner<'_, W, T> {
    fn run_loop(&mut self) -> Result<(), TuiError> {
        let tick_duration = Duration::from_millis(self.app.config.tick_rate_ms);

        loop {
            let frame_start = Instant::now();

            // Check for terminal resize
            let (width, height) = self.terminal.size()?;
            if width != self.buffer.width() || height != self.buffer.height() {
                self.buffer.resize(width, height);
                self.renderer.reset();
            }

            // Phase 1: Verify (Jidoka gate)
            let verify_start = Instant::now();
            if !self.app.config.skip_verification {
                let verification = self.app.root.verify();
                if !verification.is_valid() {
                    return Err(TuiError::VerificationFailed(VerificationError::from(
                        verification,
                    )));
                }
            }
            self.app.metrics.verify_time = verify_start.elapsed();

            // Phase 2: Render frame
            self.app.render_frame(&mut self.buffer);

            // Phase 3: Flush to terminal
            self.terminal.flush(&mut self.buffer, &mut self.renderer)?;

            self.app.metrics.total_time = frame_start.elapsed();
            self.app.metrics.frame_count += 1;

            // Phase 4: Handle input
            if self.terminal.poll(tick_duration)? {
                if let CrosstermEvent::Key(key) = self.terminal.read_event()? {
                    if key.code == KeyCode::Char('q')
                        || key.code == KeyCode::Char('c')
                            && key
                                .modifiers
                                .contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        self.app.should_quit = true;
                    }

                    if let Some(event) = self.app.input_handler.convert(CrosstermEvent::Key(key)) {
                        let _ = self.app.root.event(&event);
                    }
                }
            }

            if self.app.should_quit {
                break;
            }
        }

        Ok(())
    }
}

impl<W: Widget> TuiApp<W> {
    /// Create a new TUI application with the given root widget.
    pub fn new(root: W) -> Result<Self, TuiError> {
        // Jidoka: reject Bricks with no assertions
        if root.assertions().is_empty() {
            return Err(TuiError::InvalidBrick(
                "Root widget has no assertions - every Brick must have at least one falsifiable assertion".to_string(),
            ));
        }

        Ok(Self {
            root,
            config: TuiConfig::default(),
            input_handler: InputHandler::new(),
            metrics: FrameMetrics::default(),
            should_quit: false,
            color_mode: ColorMode::detect(),
        })
    }

    /// Set the configuration.
    #[must_use]
    pub fn with_config(mut self, config: TuiConfig) -> Self {
        if let Some(mode) = config.color_mode {
            self.color_mode = mode;
        }
        self.config = config;
        self
    }

    /// Set the input handler.
    #[must_use]
    pub fn with_input_handler(mut self, handler: InputHandler) -> Self {
        self.input_handler = handler;
        self
    }

    /// Get a reference to the root widget.
    #[must_use]
    pub fn root(&self) -> &W {
        &self.root
    }

    /// Get a mutable reference to the root widget.
    pub fn root_mut(&mut self) -> &mut W {
        &mut self.root
    }

    /// Get the current frame metrics.
    #[must_use]
    pub fn metrics(&self) -> &FrameMetrics {
        &self.metrics
    }

    /// Request the application to quit.
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Run the application (blocking).
    pub fn run(&mut self) -> Result<(), TuiError> {
        let backend = CrosstermBackend::new();
        let terminal = GenericTerminal::new(backend);
        self.run_with_terminal(terminal)
    }

    /// Run the application with a custom terminal implementation.
    /// This is the testable entry point.
    pub fn run_with_terminal<T: Terminal>(&mut self, mut terminal: T) -> Result<(), TuiError> {
        terminal.enter()?;

        if self.config.enable_mouse {
            terminal.enable_mouse()?;
        }

        // Get initial terminal size
        let (width, height) = terminal.size()?;
        let buffer = CellBuffer::new(width, height);
        let renderer = DiffRenderer::with_color_mode(self.color_mode);

        let mut runner = AppRunner {
            app: self,
            terminal,
            buffer,
            renderer,
        };

        let result = runner.run_loop();

        if runner.app.config.enable_mouse {
            runner.terminal.disable_mouse()?;
        }
        runner.terminal.leave()?;

        result
    }

    fn render_frame(&mut self, buffer: &mut CellBuffer) {
        let width = buffer.width();
        let height = buffer.height();

        // Phase 2a: Measure
        let measure_start = Instant::now();
        let constraints = Constraints::new(0.0, f32::from(width), 0.0, f32::from(height));
        let _size = self.root.measure(constraints);
        self.metrics.measure_time = measure_start.elapsed();

        // Phase 2b: Layout
        let layout_start = Instant::now();
        let bounds = Rect::new(0.0, 0.0, f32::from(width), f32::from(height));
        let _ = self.root.layout(bounds);
        self.metrics.layout_time = layout_start.elapsed();

        // Phase 2c: Paint
        let paint_start = Instant::now();
        {
            let mut canvas = DirectTerminalCanvas::new(buffer);
            self.root.paint(&mut canvas);
        }
        self.metrics.paint_time = paint_start.elapsed();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
#[path = "app_tests.rs"]
mod tests;
