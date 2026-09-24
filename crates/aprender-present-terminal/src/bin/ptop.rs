//! ptop: System monitor using presentar-terminal widget composition
//!
//! Run: cargo run -p presentar-terminal --features ptop --bin ptop

#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::unnecessary_debug_formatting)]

use std::io::{self, Write};
use std::time::{Duration, Instant};

use clap::Parser;
use crossterm::{
    cursor,
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{self, ClearType},
};

use presentar_terminal::direct::{CellBuffer, DiffRenderer};
use presentar_terminal::ptop::{config::PtopConfig, ui, App, PanelType};
use presentar_terminal::ColorMode;

/// Presentar System Monitor - widget composition demo
#[derive(Parser)]
#[command(name = "ptop", version, about, long_about = None)]
struct Cli {
    /// Refresh interval in milliseconds
    #[arg(short, long, default_value = "1000")]
    refresh: u64,

    /// Enable deterministic mode for testing (disables timestamps/dynamic data)
    #[arg(long)]
    deterministic: bool,

    /// Disable colors (use plain text)
    #[arg(long)]
    no_color: bool,

    /// Render once to stdout and exit (for comparison/testing)
    #[arg(long)]
    render_once: bool,

    /// Terminal width for render-once mode
    #[arg(long, default_value = "120")]
    width: u16,

    /// Terminal height for render-once mode
    #[arg(long, default_value = "40")]
    height: u16,

    /// Path to custom config file (YAML)
    #[arg(short, long, value_name = "PATH")]
    config: Option<std::path::PathBuf>,

    /// Dump default configuration to stdout and exit
    #[arg(long)]
    dump_config: bool,

    /// QA timing mode: output timing diagnostics to stderr
    #[arg(long)]
    qa_timing: bool,

    /// Explode a specific panel for QA (cpu, memory, disk, network, process, gpu, sensors, connections, psi, files, battery, containers)
    #[arg(long, value_name = "PANEL")]
    explode: Option<String>,
}

/// Load configuration from file or default location.
fn load_config(config_path: Option<&std::path::PathBuf>) -> PtopConfig {
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
fn handle_render_once(cli: &Cli, config: PtopConfig) -> io::Result<()> {
    let mut app = App::with_config_lightweight(cli.deterministic, config);
    if !cli.deterministic {
        app.collect_metrics();
        std::thread::sleep(Duration::from_millis(100));
        app.collect_metrics();
    }
    if let Some(ref panel_name) = cli.explode {
        app.exploded_panel = parse_panel_type(panel_name);
    }
    render_once(&app, cli.width, cli.height)
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

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    if cli.dump_config {
        println!("{}", PtopConfig::default_yaml());
        return Ok(());
    }

    let config = load_config(cli.config.as_ref());

    if cli.render_once {
        return handle_render_once(&cli, config);
    }

    let app = App::with_config(cli.deterministic, config);
    let mut stdout = io::stdout();

    setup_terminal(&mut stdout)?;

    let color_mode = if cli.no_color {
        ColorMode::Mono
    } else {
        ColorMode::TrueColor
    };
    let result = run_app(&mut stdout, app, cli.refresh, color_mode, cli.qa_timing);

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
    std::sync::mpsc::Receiver<presentar_terminal::ptop::app::MetricsSnapshot>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::sync::Arc<std::sync::atomic::AtomicU64>,
) {
    use presentar_terminal::ptop::app::MetricsCollector;
    use presentar_terminal::AsyncCollector;
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
    rx: &std::sync::mpsc::Receiver<presentar_terminal::ptop::MetricsSnapshot>,
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

/// Parse panel type from string for --explode flag
fn parse_panel_type(name: &str) -> Option<PanelType> {
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
        _ => {
            eprintln!("[ptop] Unknown panel: {name}. Valid: cpu, memory, disk, network, process, gpu, sensors, connections, psi, files, battery, containers");
            None
        }
    }
}
