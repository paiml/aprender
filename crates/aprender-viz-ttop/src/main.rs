//! ttop v2: Terminal Top - Sovereign AI Stack System Monitor
//!
//! Pure Rust system monitor built on presentar-terminal (no ratatui).
//! Zero-allocation steady-state rendering via CellBuffer + DiffRenderer.
//!
//! Install: `cargo install ttop`
//! Run: `ttop`

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
use presentar_terminal::ptop::app::MetricsCollector;
use presentar_terminal::ptop::{config::PtopConfig, ui, App, PanelType};
use presentar_terminal::{AsyncCollector, ColorMode};

/// ttop: Terminal Top - Sovereign AI Stack System Monitor
#[derive(Parser)]
#[command(name = "ttop", version, about, long_about = None)]
struct Cli {
    /// Refresh interval in milliseconds
    #[arg(short, long, default_value = "1000")]
    refresh: u64,

    /// Enable deterministic mode for testing
    #[arg(long)]
    deterministic: bool,

    /// Disable colors
    #[arg(long)]
    no_color: bool,

    /// Render once to stdout and exit (for testing/comparison)
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

    /// Explode a specific panel (cpu, memory, disk, network, process, gpu, sensors, etc.)
    #[arg(long, value_name = "PANEL")]
    explode: Option<String>,
}

fn load_config(config_path: Option<&std::path::PathBuf>) -> PtopConfig {
    if let Some(path) = config_path {
        PtopConfig::load_from_file(path).unwrap_or_else(|| {
            eprintln!("[ttop] Warning: Could not load config from {path:?}, using defaults");
            PtopConfig::default()
        })
    } else {
        PtopConfig::load()
    }
}

fn render_once(app: &App, width: u16, height: u16) -> io::Result<()> {
    let mut buffer = CellBuffer::new(width, height);
    ui::draw(app, &mut buffer);

    let mut stdout = io::stdout();
    for y in 0..height {
        for x in 0..width {
            if let Some(cell) = buffer.get(x, y) {
                let ch = cell.symbol.chars().next().unwrap_or(' ');
                write!(stdout, "{ch}")?;
            } else {
                write!(stdout, " ")?;
            }
        }
        writeln!(stdout)?;
    }
    stdout.flush()
}

fn setup_terminal(stdout: &mut io::Stdout) -> io::Result<()> {
    terminal::enable_raw_mode()?;
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All)
    )
}

fn cleanup_terminal(stdout: &mut io::Stdout) -> io::Result<()> {
    execute!(stdout, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()
}

fn spawn_metrics_collector(
    refresh_ms: u64,
    deterministic: bool,
) -> (
    std::sync::mpsc::Receiver<presentar_terminal::ptop::MetricsSnapshot>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc};

    let collect_interval = Duration::from_millis(refresh_ms);
    let bg_running = Arc::new(AtomicBool::new(true));
    let bg_running_thread = Arc::clone(&bg_running);

    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let mut collector = MetricsCollector::new(deterministic);
        while bg_running_thread.load(Ordering::Relaxed) {
            let snapshot = collector.collect();
            if tx.send(snapshot).is_err() {
                break;
            }
            std::thread::sleep(collect_interval);
        }
    });

    (rx, bg_running)
}

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

fn run_app(
    stdout: &mut io::Stdout,
    mut app: App,
    refresh_ms: u64,
    color_mode: ColorMode,
) -> io::Result<()> {
    use std::sync::atomic::Ordering;

    let mut renderer = DiffRenderer::with_color_mode(color_mode);
    let (rx, bg_running) = spawn_metrics_collector(refresh_ms, app.deterministic);

    let render_interval = Duration::from_millis(16);
    let mut last_render = Instant::now()
        .checked_sub(render_interval)
        .unwrap_or_else(Instant::now);
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);
    let mut was_exploded = false;
    let mut first_frame = true;

    loop {
        if process_input(&mut app)? {
            bg_running.store(false, Ordering::Relaxed);
            return Ok(());
        }

        // Apply pending metrics snapshots
        while let Ok(snapshot) = rx.try_recv() {
            app.apply_snapshot(snapshot);
        }

        if last_render.elapsed() < render_interval {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }

        let render_start = Instant::now();
        let is_exploded = app.exploded_panel.is_some();
        let mode_changed = first_frame || is_exploded != was_exploded;
        was_exploded = is_exploded;
        first_frame = false;

        render_frame(stdout, &app, &mut renderer, mode_changed)?;

        if !app.running {
            bg_running.store(false, Ordering::Relaxed);
            break;
        }

        last_render = Instant::now();
        let elapsed = render_start.elapsed();
        frame_times.push(elapsed);
        if frame_times.len() > 60 {
            frame_times.remove(0);
        }
        app.update_frame_stats(&frame_times);
    }

    Ok(())
}

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
            eprintln!("[ttop] Unknown panel: {name}. Valid: cpu, memory, disk, network, process, gpu, sensors, connections, psi, files, battery, containers");
            None
        }
    }
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    if cli.dump_config {
        println!("{}", PtopConfig::default_yaml());
        return Ok(());
    }

    let config = load_config(cli.config.as_ref());

    if cli.render_once {
        let mut app = App::with_config_lightweight(cli.deterministic, config);
        if !cli.deterministic {
            app.collect_metrics();
            std::thread::sleep(Duration::from_millis(100));
            app.collect_metrics();
        }
        if let Some(ref panel_name) = cli.explode {
            app.exploded_panel = parse_panel_type(panel_name);
        }
        return render_once(&app, cli.width, cli.height);
    }

    let mut app = App::with_config(cli.deterministic, config);
    if let Some(ref panel_name) = cli.explode {
        app.exploded_panel = parse_panel_type(panel_name);
    }

    let mut stdout = io::stdout();
    setup_terminal(&mut stdout)?;

    let color_mode = if cli.no_color {
        ColorMode::Mono
    } else {
        ColorMode::TrueColor
    };

    let result = run_app(&mut stdout, app, cli.refresh, color_mode);
    cleanup_terminal(&mut stdout)?;
    result
}
