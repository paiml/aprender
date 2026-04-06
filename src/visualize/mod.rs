//! Real-Time Tracing Visualization Module (Sprint 52-55)
//!
//! Provides terminal-based visualization of syscall traces, anomalies, ML clustering,
//! and OpenTelemetry spans using ttop-identical architecture and performance characteristics.
//!
//! # Architecture
//!
//! ```text
//! ptrace syscall intercept → Tracer Thread → mpsc channel → VisualizeApp → ratatui → Terminal
//!                                   ↓
//!                           stdout → /dev/null (suppressed)
//! ```
//!
//! # Bridge Design (QA Falsification Protocol Findings)
//!
//! The tracer runs in a background thread and communicates via unbounded mpsc channel:
//!
//! - **Stdout Suppression**: Tracer thread redirects stdout to /dev/null before tracing.
//!   Without this, tracer output corrupts the TUI (print_syscall_entry, stats summaries).
//! - **Channel Draining**: UI thread drains all pending events every 50ms tick via try_recv().
//! - **Lifecycle**: When traced process exits, tracer thread finishes, UI shows "TRACE COMPLETE"
//!   banner and waits for 'q' to exit.
//! - **Panic Safety**: TerminalGuard ensures raw mode cleanup even on panic.
//!
//! # Performance Targets (ttop-identical)
//!
//! - Event tick rate: 50ms
//! - Average frame time: 8ms
//! - P99 frame time: <16ms
//! - Memory: <10MB RSS
//!
//! # Toyota Way Principles
//!
//! - **Genchi Genbutsu**: Direct syscall observation, no intermediaries
//! - **Jidoka**: Automatic anomaly detection with visual alerts (Andon)
//! - **Muda**: Zero-copy ring buffers, no allocation in hot path
//! - **Poka-Yoke**: Type-safe panel IDs prevent configuration errors

pub mod app;
pub mod collectors;
pub mod panels;
pub mod ring_buffer;
pub mod theme;
pub mod ui;
pub mod widgets;

// Re-exports for convenience
pub use app::VisualizeApp;
pub use collectors::{AnomalyCollector, Collector, SpanReceiver, SyscallCollector};
pub use ring_buffer::HistoryBuffer;
pub use theme::{percent_color, severity_color};

use crate::tracer::{self, TracerConfig, VisualizerEvent};
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for the visualize subcommand
#[derive(Debug, Clone)]
pub struct VisualizeConfig {
    /// Tick rate in milliseconds (default: 50, matches ttop)
    pub tick_rate_ms: u64,

    /// Enable anomaly detection panel
    pub enable_anomaly: bool,

    /// Enable ML clustering panel
    pub enable_ml: bool,

    /// Number of ML clusters (default: 3)
    pub ml_clusters: usize,

    /// Anomaly Z-score threshold (default: 3.0)
    pub anomaly_threshold: f32,

    /// History buffer size (default: 300, matches ttop)
    pub history_size: usize,

    /// Enable deterministic mode for testing
    pub deterministic: bool,

    /// Show FPS overlay
    pub show_fps: bool,

    /// Process ID to attach to (None = trace command)
    pub pid: Option<i32>,

    /// Enable source correlation
    pub enable_source: bool,

    /// Filter expression
    pub filter: Option<String>,

    /// OTLP endpoint for span export
    pub otlp_endpoint: Option<String>,

    // Sprint 56: Metrics and Alerting Configuration
    /// Enable metrics collection panel (counters, gauges, histograms)
    pub enable_metrics: bool,

    /// Enable alerting panel (threshold/rate/absence alerts)
    pub enable_alerts: bool,

    /// Alert threshold for syscall latency anomaly (microseconds)
    pub alert_latency_threshold_us: u64,

    /// Alert threshold for error rate percentage
    pub alert_error_rate_percent: f32,
}

impl Default for VisualizeConfig {
    fn default() -> Self {
        Self {
            tick_rate_ms: 50,
            enable_anomaly: true,
            enable_ml: false,
            ml_clusters: 3,
            anomaly_threshold: 3.0,
            history_size: 300,
            deterministic: false,
            show_fps: false,
            pid: None,
            enable_source: false,
            filter: None,
            otlp_endpoint: None,
            enable_metrics: false,
            enable_alerts: false,
            alert_latency_threshold_us: 10_000,
            alert_error_rate_percent: 5.0,
        }
    }
}

/// Inject synthetic syscall data for demo mode
fn inject_demo_data(app: &mut app::VisualizeApp, tick: u64) {
    // Simple LCG for deterministic pseudo-random numbers
    let mut seed = tick.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next_rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed
    };

    // Syscall names by category for realistic distribution
    const FILE_SYSCALLS: &[&str] = &["read", "write", "open", "close", "stat", "fstat", "lseek"];
    const NET_SYSCALLS: &[&str] = &["socket", "connect", "sendto", "recvfrom", "bind", "listen"];
    const MEM_SYSCALLS: &[&str] = &["mmap", "munmap", "brk", "mprotect"];
    const PROC_SYSCALLS: &[&str] = &["fork", "clone", "execve", "wait4", "getpid"];

    // Inject 5-20 syscalls per tick for realistic rate
    let count = 5 + (next_rand() % 16) as usize;

    for _ in 0..count {
        let r = next_rand();
        let (syscalls, base_duration) = match r % 100 {
            0..=49 => (FILE_SYSCALLS, 50),  // 50% file I/O
            50..=74 => (NET_SYSCALLS, 200), // 25% network
            75..=89 => (MEM_SYSCALLS, 20),  // 15% memory
            _ => (PROC_SYSCALLS, 500),      // 10% process
        };

        let name = syscalls[(next_rand() as usize) % syscalls.len()];

        // Duration with variance (some outliers for anomaly detection)
        let duration = if next_rand() % 100 < 3 {
            // 3% chance of anomaly (10-100x normal)
            base_duration * (10 + (next_rand() % 90))
        } else {
            base_duration + (next_rand() % (base_duration * 2))
        };

        // 2% error rate
        let result = if next_rand() % 100 < 2 { -1 } else { 0 };

        app.record_syscall(name, duration, result);

        // Occasionally record anomalies for high z-scores
        if duration > base_duration * 5 {
            let z_score = (duration as f32 / base_duration as f32).min(10.0);
            app.record_anomaly(
                name.to_string(),
                duration,
                z_score,
                Some("demo.rs".to_string()),
                Some((tick % 500) as u32 + 1),
            );
        }
    }
}

/// Redirect stdout and stderr to /dev/null to prevent tracer/child output from corrupting TUI
///
/// This is called in the tracer thread before running trace_command/attach_to_pid.
/// Without this, print_syscall_entry(), stats summaries, and child process output
/// would write to the same terminal that ratatui is using for TUI rendering.
#[allow(unsafe_code)]
fn suppress_stdio() {
    if let Ok(devnull) = File::open("/dev/null") {
        let devnull_fd = devnull.as_raw_fd();
        // SAFETY: dup2 is safe here because:
        // 1. devnull_fd is a valid file descriptor from File::open
        // 2. STDOUT_FILENO/STDERR_FILENO are valid standard fd constants
        // 3. dup2 atomically closes the target fd if open, preventing fd leaks
        unsafe {
            libc::dup2(devnull_fd, libc::STDOUT_FILENO);
            libc::dup2(devnull_fd, libc::STDERR_FILENO);
        }
        // devnull File is dropped here, but the dup'd fds stay open
    }
}

/// Result from tracer handle creation
struct TracerSetup {
    handle: Option<thread::JoinHandle<Result<i32>>>,
    demo_mode: bool,
}

/// Create tracer handle based on command/pid configuration
fn create_tracer_handle(
    command: Option<&[String]>,
    pid: Option<i32>,
    tx: mpsc::Sender<VisualizerEvent>,
) -> TracerSetup {
    if let Some(cmd) = command {
        let cmd_owned: Vec<String> = cmd.to_vec();
        let tracer_config = TracerConfig { visualizer_sink: Some(tx), ..Default::default() };
        let handle = thread::spawn(move || {
            suppress_stdio();
            tracer::trace_command(&cmd_owned, tracer_config)
        });
        return TracerSetup { handle: Some(handle), demo_mode: false };
    }

    if let Some(pid) = pid {
        let tracer_config = TracerConfig { visualizer_sink: Some(tx), ..Default::default() };
        let handle = thread::spawn(move || {
            suppress_stdio();
            tracer::attach_to_pid(pid, tracer_config)
        });
        return TracerSetup { handle: Some(handle), demo_mode: false };
    }

    // Demo mode - no tracer, drop tx so rx doesn't block
    drop(tx);
    TracerSetup { handle: None, demo_mode: true }
}

/// Open terminal for TUI rendering
fn open_tty() -> File {
    OpenOptions::new().read(true).write(true).open("/dev/tty").unwrap_or_else(|_| {
        // Fallback to stdout if /dev/tty not available (e.g., in CI)
        OpenOptions::new().write(true).open("/proc/self/fd/1").expect("Failed to open terminal")
    })
}

/// Terminal cleanup guard - ensures terminal is restored on panic
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        if let Ok(mut tty) = OpenOptions::new().write(true).open("/dev/tty") {
            let _ = execute!(tty, LeaveAlternateScreen, DisableMouseCapture);
        }
    }
}

/// Update frame timing statistics
fn update_frame_stats(app: &mut app::VisualizeApp, frame_times: &mut Vec<u64>, frame_time: u64) {
    frame_times.push(frame_time);
    if frame_times.len() > 60 {
        frame_times.remove(0);
    }
    app.avg_frame_time_us = frame_times.iter().sum::<u64>() / frame_times.len().max(1) as u64;
    app.max_frame_time_us = frame_times.iter().copied().max().unwrap_or(0);
    app.frame_id += 1;
}

/// Handle keyboard input, returns true if should quit
fn handle_key_input(app: &mut app::VisualizeApp, key: event::KeyEvent) -> bool {
    if key.code == KeyCode::Char('q')
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
    {
        return true;
    }
    app.handle_key(key.code, key.modifiers);
    false
}

/// Handle tracer thread completion
fn handle_tracer_finish(
    app: &mut app::VisualizeApp,
    tracer_handle: &mut Option<thread::JoinHandle<Result<i32>>>,
) {
    if tracer_handle.as_ref().is_some_and(std::thread::JoinHandle::is_finished) {
        app.collect_metrics();
        app.trace_complete = true;
        app.event_receiver = None;

        if let Some(handle) = tracer_handle.take() {
            let _ = handle.join();
        }
        let _ = enable_raw_mode();
    }
}

/// Process a tick interval: demo data injection and metric collection
fn process_tick(
    app: &mut app::VisualizeApp,
    demo_mode: bool,
    demo_tick: &mut u64,
    last_tick: &mut Instant,
) {
    if demo_mode {
        inject_demo_data(app, *demo_tick);
        *demo_tick = demo_tick.wrapping_add(1);
    }
    app.collect_metrics();
    *last_tick = Instant::now();
}

/// Poll for keyboard events, returns Some(true) to quit, Some(false) to continue, None on no event
fn poll_keyboard(app: &mut app::VisualizeApp, timeout: Duration) -> Result<Option<bool>> {
    if event::poll(timeout)? {
        if let Event::Key(key) = event::read()? {
            return Ok(Some(handle_key_input(app, key)));
        }
    }
    Ok(None)
}

/// Initialize the terminal for TUI rendering
fn init_terminal() -> Result<(Terminal<CrosstermBackend<File>>, TerminalGuard)> {
    enable_raw_mode()?;
    let mut tty = open_tty();
    let guard = TerminalGuard;
    execute!(tty, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(tty);
    let terminal = Terminal::new(backend)?;
    Ok((terminal, guard))
}

/// Restore terminal state after TUI shutdown
fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<File>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}

/// Run the visualization TUI
///
/// # Arguments
///
/// * `command` - Command to trace (with arguments)
/// * `config` - Visualization configuration
///
/// # Returns
///
/// Exit code from the traced process (0 if successful)
/// Result of a single event loop iteration.
enum LoopAction {
    /// Continue to the next iteration
    Continue,
    /// Exit the loop with the given exit code
    Exit(i32),
}

/// Process one iteration of the event loop: draw, handle input, tick.
fn run_event_loop_iteration(
    terminal: &mut Terminal<CrosstermBackend<File>>,
    app: &mut app::VisualizeApp,
    frame_times: &mut Vec<u64>,
    tick_rate: Duration,
    demo_mode: bool,
    demo_tick: &mut u64,
    last_tick: &mut Instant,
    tracer_handle: &mut Option<std::thread::JoinHandle<Result<i32>>>,
) -> Result<LoopAction> {
    let frame_start = Instant::now();
    terminal.draw(|f| ui::draw(f, app))?;
    update_frame_stats(app, frame_times, frame_start.elapsed().as_micros() as u64);

    let timeout = tick_rate.saturating_sub(last_tick.elapsed());
    if let Some(true) = poll_keyboard(app, timeout)? {
        return Ok(LoopAction::Exit(0));
    }

    if last_tick.elapsed() >= tick_rate {
        process_tick(app, demo_mode, demo_tick, last_tick);
    }

    if app.should_exit {
        return Ok(LoopAction::Exit(app.exit_code));
    }

    handle_tracer_finish(app, tracer_handle);
    Ok(LoopAction::Continue)
}

pub fn run_visualize(command: Option<&[String]>, config: VisualizeConfig) -> Result<i32> {
    let (tx, rx) = mpsc::channel::<VisualizerEvent>();
    let setup = create_tracer_handle(command, config.pid, tx);
    let mut tracer_handle = setup.handle;
    let demo_mode = setup.demo_mode;

    let (mut terminal, _guard) = init_terminal()?;

    let receiver = if tracer_handle.is_some() { Some(rx) } else { None };
    let mut app = app::VisualizeApp::with_receiver(config.clone(), receiver);

    let mut demo_tick: u64 = 0;
    let tick_rate = Duration::from_millis(config.tick_rate_ms);
    let mut last_tick = Instant::now();
    let mut frame_times: Vec<u64> = Vec::with_capacity(60);

    let result = loop {
        match run_event_loop_iteration(
            &mut terminal,
            &mut app,
            &mut frame_times,
            tick_rate,
            demo_mode,
            &mut demo_tick,
            &mut last_tick,
            &mut tracer_handle,
        )? {
            LoopAction::Continue => {}
            LoopAction::Exit(code) => break Ok(code),
        }
    };

    if let Some(handle) = tracer_handle {
        let _ = handle.join();
    }

    restore_terminal(&mut terminal)?;
    result
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
