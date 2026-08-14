//! The ptop application entry point: option struct + run loop.
//!
//! This is the whole of what used to be `src/bin/ptop.rs::main` and its
//! helpers. It moved into the library so `apr top` can call it directly
//! instead of shelling out to, or duplicating, a second binary named `ptop`
//! — a name owned by several unrelated system monitors, which
//! `cargo install` would have dropped into `~/.cargo/bin` unqualified.
//!
//! Nothing about the behaviour changed in the move: the option names, their
//! defaults, the render-once path, the background collector and the QA
//! timing report are the code that used to live in the binary.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crossterm::{
    cursor,
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{self, ClearType},
};

use crate::direct::{CellBuffer, DiffRenderer};
use crate::ptop::{config::PtopConfig, ui, App, PanelType};
use crate::ColorMode;

/// Panel names `--explode` accepts, including every alias the original
/// `parse_panel_type` matched.
///
/// The binary took a free-form `String` here and, on an unrecognised value,
/// printed a warning to stderr, left `exploded_panel` at `None` and exited 0
/// — so `--explode cpuu` rendered the ordinary dashboard and reported
/// success. That is the #2418 silently-dropped-argument shape. Every value
/// that used to work still works; a typo is now refused by the parser.
pub const PANEL_VALUES: [&str; 23] = [
    "cpu",
    "memory",
    "mem",
    "disk",
    "network",
    "net",
    "process",
    "proc",
    "processes",
    "gpu",
    "sensors",
    "sensor",
    "connections",
    "conn",
    "psi",
    "pressure",
    "files",
    "file",
    "battery",
    "bat",
    "containers",
    "container",
    "docker",
];

/// Every option the `ptop` binary accepted, with its original default.
#[derive(Debug, Clone)]
pub struct PtopOptions {
    /// Refresh interval in milliseconds (`--refresh`, default 1000).
    pub refresh: u64,
    /// Deterministic mode for testing (`--deterministic`).
    pub deterministic: bool,
    /// Plain-text output with no colour (`--no-color`).
    pub no_color: bool,
    /// Render one frame to stdout and exit (`--render-once`).
    pub render_once: bool,
    /// Terminal width used by render-once mode (`--width`, default 120).
    pub width: u16,
    /// Terminal height used by render-once mode (`--height`, default 40).
    pub height: u16,
    /// Custom YAML config file (`--config`).
    pub config: Option<PathBuf>,
    /// Print the default configuration and exit (`--dump-config`).
    pub dump_config: bool,
    /// Emit timing diagnostics to stderr (`--qa-timing`).
    pub qa_timing: bool,
    /// Panel to explode (`--explode`); one of [`PANEL_VALUES`].
    pub explode: Option<String>,
}

impl Default for PtopOptions {
    fn default() -> Self {
        Self {
            refresh: 1000,
            deterministic: false,
            no_color: false,
            render_once: false,
            width: 120,
            height: 40,
            config: None,
            dump_config: false,
            qa_timing: false,
            explode: None,
        }
    }
}

/// Load configuration from file or default location.
fn load_config(config_path: Option<&PathBuf>) -> PtopConfig {
    if let Some(path) = config_path {
        PtopConfig::load_from_file(path).unwrap_or_else(|| {
            eprintln!("[ptop] Warning: Could not load config from {path:?}, using defaults");
            PtopConfig::default()
        })
    } else {
        PtopConfig::load()
    }
}

/// Handle render-once mode for testing/comparison.
fn handle_render_once(opts: &PtopOptions, config: PtopConfig) -> io::Result<()> {
    let mut app = App::with_config_lightweight(opts.deterministic, config);
    if !opts.deterministic {
        app.collect_metrics();
        std::thread::sleep(Duration::from_millis(100));
        app.collect_metrics();
    }
    if let Some(ref panel_name) = opts.explode {
        app.exploded_panel = parse_panel_type(panel_name);
    }
    render_once(&app, opts.width, opts.height)
}

/// Setup terminal for interactive mode.
fn setup_terminal(stdout: &mut io::Stdout) -> io::Result<()> {
    terminal::enable_raw_mode()?;
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All)
    )
}

/// Cleanup terminal after interactive mode.
fn cleanup_terminal(stdout: &mut io::Stdout) -> io::Result<()> {
    execute!(stdout, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()
}

/// Run ptop with `opts`. This is verbatim the old `ptop::main` body.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] if the terminal cannot be put into
/// raw mode, or if writing a frame to stdout fails.
pub fn run(opts: &PtopOptions) -> io::Result<()> {
    if opts.dump_config {
        println!("{}", PtopConfig::default_yaml());
        return Ok(());
    }

    let config = load_config(opts.config.as_ref());

    if opts.render_once {
        return handle_render_once(opts, config);
    }

    let app = App::with_config(opts.deterministic, config);
    let mut stdout = io::stdout();

    setup_terminal(&mut stdout)?;

    let color_mode = if opts.no_color {
        ColorMode::Mono
    } else {
        ColorMode::TrueColor
    };
    let result = run_app(&mut stdout, app, opts.refresh, color_mode, opts.qa_timing);

    cleanup_terminal(&mut stdout)?;
    result
}

/// Render a single frame to stdout (for comparison/testing)
fn render_once(app: &App, width: u16, height: u16) -> io::Result<()> {
    let mut buffer = CellBuffer::new(width, height);
    ui::draw(app, &mut buffer);

    let mut stdout = io::stdout();

    // Output each row as plain text (no ANSI sequences)
    for y in 0..height {
        for x in 0..width {
            if let Some(cell) = buffer.get(x, y) {
                // Get first char of symbol (handles multi-byte)
                let ch = cell.symbol.chars().next().unwrap_or(' ');
                write!(stdout, "{ch}")?;
            } else {
                write!(stdout, " ")?;
            }
        }
        writeln!(stdout)?;
    }

    stdout.flush()?;
    Ok(())
}

/// Spawn background metrics collector thread.
/// Returns (receiver, `running_flag`, `collect_time_atomic`).
fn spawn_metrics_collector(
    refresh_ms: u64,
    deterministic: bool,
) -> (
    std::sync::mpsc::Receiver<crate::ptop::app::MetricsSnapshot>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::sync::Arc<std::sync::atomic::AtomicU64>,
) {
    use crate::ptop::app::MetricsCollector;
    use crate::AsyncCollector;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{mpsc, Arc};

    let collect_interval = Duration::from_millis(refresh_ms);
    let collect_time_us = Arc::new(AtomicU64::new(0));
    let collect_time_bg = Arc::clone(&collect_time_us);
    let bg_running = Arc::new(AtomicBool::new(true));
    let bg_running_thread = Arc::clone(&bg_running);

    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let mut collector = MetricsCollector::new(deterministic);
        while bg_running_thread.load(Ordering::Relaxed) {
            let collect_start = Instant::now();
            let snapshot = collector.collect();
            collect_time_bg.store(
                collect_start.elapsed().as_micros() as u64,
                Ordering::Relaxed,
            );
            if tx.send(snapshot).is_err() {
                break;
            }
            std::thread::sleep(collect_interval);
        }
    });

    (rx, bg_running, collect_time_us)
}

/// Process all pending input events. Returns true if app should quit.
fn process_input(app: &mut App) -> io::Result<bool> {
    while event::poll(Duration::from_millis(1))? {
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press && app.handle_key(key.code, key.modifiers) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Render frame and flush to terminal.
fn render_frame(
    stdout: &mut io::Stdout,
    app: &App,
    renderer: &mut DiffRenderer,
    mode_changed: bool,
) -> io::Result<()> {
    let (width, height) = terminal::size()?;
    let mut buffer = CellBuffer::new(width, height);
    ui::draw(app, &mut buffer);

    execute!(stdout, cursor::MoveTo(0, 0))?;
    let mut output = Vec::with_capacity(32768);

    if mode_changed {
        renderer.render_full(&mut buffer, &mut output)?;
    } else {
        renderer.flush(&mut buffer, &mut output)?;
    }

    stdout.write_all(&output)?;
    stdout.flush()
}

/// Report QA timing stats to stderr.
fn report_qa_stats(input_times: &[u64], render_times: &[u64], collect_time_us: u64) {
    let avg = |v: &[u64]| {
        if v.is_empty() {
            0
        } else {
            v.iter().sum::<u64>() / v.len() as u64
        }
    };
    let max = |v: &[u64]| v.iter().max().copied().unwrap_or(0);
    eprintln!(
        "[QA] input: avg={}us max={}us | render: avg={}us max={}us | collect: {}us (NO LOCK)",
        avg(input_times),
        max(input_times),
        avg(render_times),
        max(render_times),
        collect_time_us
    );
}

/// Track frame time, keeping only the last 60 samples.
fn track_frame_time(frame_times: &mut Vec<Duration>, elapsed: Duration) {
    frame_times.push(elapsed);
    if frame_times.len() > 60 {
        frame_times.remove(0);
    }
}

/// QA timing state for performance reporting.
struct QaTimingState {
    input_times: Vec<u64>,
    render_times: Vec<u64>,
    report_interval: Instant,
}

impl QaTimingState {
    fn new() -> Self {
        Self {
            input_times: Vec::with_capacity(100),
            render_times: Vec::with_capacity(100),
            report_interval: Instant::now(),
        }
    }

    fn record_input(&mut self, elapsed: Duration) {
        self.input_times.push(elapsed.as_micros() as u64);
    }

    fn record_render(&mut self, elapsed: Duration) {
        self.render_times.push(elapsed.as_micros() as u64);
    }

    fn maybe_report(&mut self, collect_time_us: u64) {
        if self.report_interval.elapsed() >= Duration::from_secs(2) {
            report_qa_stats(&self.input_times, &self.render_times, collect_time_us);
            self.input_times.clear();
            self.render_times.clear();
            self.report_interval = Instant::now();
        }
    }
}

/// Apply all pending snapshots from the metrics collector.
fn apply_pending_snapshots(
    rx: &std::sync::mpsc::Receiver<crate::ptop::MetricsSnapshot>,
    app: &mut App,
) {
    while let Ok(snapshot) = rx.try_recv() {
        app.apply_snapshot(snapshot);
    }
}

/// Check if exploded view mode changed.
fn check_mode_change(app: &App, was_exploded: &mut bool) -> bool {
    let is_exploded = app.exploded_panel.is_some();
    let changed = is_exploded != *was_exploded;
    *was_exploded = is_exploded;
    changed
}

/// Record input timing if QA mode enabled.
#[inline]
fn record_qa_input(qa_timing: bool, qa_state: &mut QaTimingState, elapsed: Duration) {
    if qa_timing {
        qa_state.record_input(elapsed);
    }
}

/// Record render timing and maybe report if QA mode enabled.
#[inline]
fn record_qa_render(
    qa_timing: bool,
    qa_state: &mut QaTimingState,
    render_elapsed: Duration,
    collect_time_us: u64,
) {
    if qa_timing {
        qa_state.record_render(render_elapsed);
        qa_state.maybe_report(collect_time_us);
    }
}

fn run_app(
    stdout: &mut io::Stdout,
    mut app: App,
    refresh_ms: u64,
    color_mode: ColorMode,
    qa_timing: bool,
) -> io::Result<()> {
    use std::sync::atomic::Ordering;

    let mut renderer = DiffRenderer::with_color_mode(color_mode);
    let (rx, bg_running, collect_time_us) = spawn_metrics_collector(refresh_ms, app.deterministic);

    let render_interval = Duration::from_millis(16);
    let mut last_render = Instant::now();
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);
    let mut was_exploded = false;
    let mut qa_state = QaTimingState::new();

    loop {
        let input_start = Instant::now();
        if process_input(&mut app)? {
            bg_running.store(false, Ordering::Relaxed);
            return Ok(());
        }
        record_qa_input(qa_timing, &mut qa_state, input_start.elapsed());

        apply_pending_snapshots(&rx, &mut app);

        if last_render.elapsed() < render_interval {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }

        let render_start = Instant::now();
        let mode_changed = check_mode_change(&app, &mut was_exploded);
        render_frame(stdout, &app, &mut renderer, mode_changed)?;

        if !app.running {
            bg_running.store(false, Ordering::Relaxed);
            break;
        }

        last_render = Instant::now();
        track_frame_time(&mut frame_times, render_start.elapsed());
        app.update_frame_stats(&frame_times);

        record_qa_render(
            qa_timing,
            &mut qa_state,
            render_start.elapsed(),
            collect_time_us.load(Ordering::Relaxed),
        );
    }

    Ok(())
}

/// Parse panel type from string for `--explode`.
///
/// Returns `None` only for a name outside [`PANEL_VALUES`]; callers that parse
/// through clap's `value_parser` never reach that arm.
#[must_use]
pub fn parse_panel_type(name: &str) -> Option<PanelType> {
    match name.to_lowercase().as_str() {
        "cpu" => Some(PanelType::Cpu),
        "memory" | "mem" => Some(PanelType::Memory),
        "disk" => Some(PanelType::Disk),
        "network" | "net" => Some(PanelType::Network),
        "process" | "proc" | "processes" => Some(PanelType::Process),
        "gpu" => Some(PanelType::Gpu),
        "sensors" | "sensor" => Some(PanelType::Sensors),
        "connections" | "conn" => Some(PanelType::Connections),
        "psi" | "pressure" => Some(PanelType::Psi),
        "files" | "file" => Some(PanelType::Files),
        "battery" | "bat" => Some(PanelType::Battery),
        "containers" | "container" | "docker" => Some(PanelType::Containers),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name `PANEL_VALUES` advertises must actually resolve to a panel.
    ///
    /// This is the guard against the rehome dropping an alias: the binary's
    /// `--explode` accepted twelve panels under twenty-three spellings, and a
    /// dropped alias would have degraded to "unknown panel, render the normal
    /// dashboard, exit 0".
    #[test]
    fn every_advertised_panel_value_resolves() {
        let unresolved: Vec<&str> = PANEL_VALUES
            .iter()
            .copied()
            .filter(|name| parse_panel_type(name).is_none())
            .collect();
        assert_eq!(
            unresolved,
            Vec::<&str>::new(),
            "PANEL_VALUES advertises names --explode cannot resolve"
        );
    }

    /// The twelve distinct panels are all reachable through `PANEL_VALUES`.
    #[test]
    fn all_twelve_panels_are_reachable() {
        let distinct: std::collections::BTreeSet<String> = PANEL_VALUES
            .iter()
            .filter_map(|name| parse_panel_type(name))
            .map(|p| format!("{p:?}"))
            .collect();
        assert_eq!(
            distinct.len(),
            12,
            "expected 12 distinct panels reachable from PANEL_VALUES, got {distinct:?}"
        );
    }

    /// A name outside the table is refused, not silently ignored.
    #[test]
    fn unknown_panel_name_is_refused() {
        assert_eq!(parse_panel_type("cpuu"), None);
    }

    /// The defaults are the binary's documented defaults.
    #[test]
    fn defaults_match_the_original_binary() {
        let d = PtopOptions::default();
        assert_eq!(d.refresh, 1000);
        assert_eq!(d.width, 120);
        assert_eq!(d.height, 40);
    }
}
