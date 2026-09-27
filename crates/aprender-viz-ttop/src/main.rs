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
use presentar_terminal::ptop::{config::PtopConfig, ui, App, MetricsSnapshot, PanelType};
use presentar_terminal::{AsyncCollector, ColorMode};

use aprender_viz_ttop::runtime::{
    needs_full_repaint, parse_panel_type, rss_kib, spawn_bounded_collector,
};
use aprender_viz_ttop::timings::{time, Timings};

/// ttop: Terminal Top - Sovereign AI Stack System Monitor
#[derive(Parser)]
#[command(
    name = "ttop",
    version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("APR_GIT_SHA"), ")"),
    about,
    long_about = None
)]
struct Cli {
    /// Refresh interval in milliseconds (>= 1; 0 would spin the collector)
    #[arg(short, long, default_value = "1000", value_parser = clap::value_parser!(u64).range(1..))]
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
    #[arg(long, value_name = "PANEL", value_parser = parse_panel_type)]
    explode: Option<PanelType>,

    /// Headless soak: run N collect+draw+diff frames with no terminal, printing
    /// one JSON line per 10 frames ({"frame":N,"rss_kib":K}), then exit.
    /// Used by the #4511 RSS leak gate (scripts/ttop_soak_gate.sh).
    #[arg(long, value_name = "FRAMES")]
    soak_frames: Option<u64>,

    /// Time every frame phase (input, apply, layout_render, diff, write) and each
    /// collector (cpu, mem, process, disk, net, gpu, analyzers); print p50/p99 per
    /// phase as JSON lines on exit (stderr; stdout under --soak-frames). Off by default.
    #[arg(long)]
    timings: bool,
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
    timings: bool,
) -> (
    std::sync::mpsc::Receiver<MetricsSnapshot>,
    aprender_viz_ttop::runtime::StopOnDrop,
) {
    let mut collector = MetricsCollector::new(deterministic);
    collector.set_timings(timings);
    spawn_bounded_collector(Duration::from_millis(refresh_ms), move || {
        collector.collect()
    })
}

/// Apply one snapshot; under --timings also fold in the collector's own phase times.
fn apply(app: &mut App, mut snapshot: MetricsSnapshot, t: Option<&mut Timings>) {
    match t {
        None => app.apply_snapshot(snapshot),
        Some(t) => {
            for (phase, us) in std::mem::take(&mut snapshot.collect_phase_us) {
                t.record_us(phase, us);
            }
            time(Some(t), "apply", || app.apply_snapshot(snapshot));
        }
    }
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
    t: Option<&mut Timings>,
) -> io::Result<()> {
    let (width, height) = terminal::size()?;
    draw_diff(
        app,
        renderer,
        width,
        height,
        mode_changed,
        &mut Vec::with_capacity(32768),
        t,
        |out| {
            execute!(stdout, cursor::MoveTo(0, 0))?;
            stdout.write_all(out)?;
            stdout.flush()
        },
    )
}

/// Draw one frame into a fresh buffer and diff it into `output`, then hand the bytes
/// to `sink`. Shared by the terminal loop and the headless soak.
#[allow(clippy::too_many_arguments)]
fn draw_diff(
    app: &App,
    renderer: &mut DiffRenderer,
    width: u16,
    height: u16,
    full: bool,
    output: &mut Vec<u8>,
    mut t: Option<&mut Timings>,
    sink: impl FnOnce(&[u8]) -> io::Result<()>,
) -> io::Result<()> {
    let mut buffer = CellBuffer::new(width, height);
    time(t.as_deref_mut(), "layout_render", || {
        ui::draw(app, &mut buffer)
    });
    output.clear();
    time(t.as_deref_mut(), "diff", || {
        if full {
            renderer.render_full(&mut buffer, output)
        } else {
            renderer.flush(&mut buffer, output)
        }
    })?;
    time(t, "write", || sink(output))
}

fn run_app(
    stdout: &mut io::Stdout,
    mut app: App,
    refresh_ms: u64,
    color_mode: ColorMode,
    mut t: Option<&mut Timings>,
) -> io::Result<()> {
    let mut renderer = DiffRenderer::with_color_mode(color_mode);
    // `_stop` ends the collector on every exit path, `?` errors included (#4511 D6)
    let (rx, _stop) = spawn_metrics_collector(refresh_ms, app.deterministic, t.is_some());

    let render_interval = Duration::from_millis(16);
    let mut last_render = Instant::now()
        .checked_sub(render_interval)
        .unwrap_or_else(Instant::now);
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);
    let mut was_exploded = false;
    let mut first_frame = true;
    let mut last_size = (0u16, 0u16);

    loop {
        if time(t.as_deref_mut(), "input", || process_input(&mut app))? {
            return Ok(());
        }

        // Apply pending metrics snapshots
        while let Ok(snapshot) = rx.try_recv() {
            apply(&mut app, snapshot, t.as_deref_mut());
        }

        if last_render.elapsed() < render_interval {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }

        let render_start = Instant::now();
        let is_exploded = app.exploded_panel.is_some();
        // a resize must repaint everything: the diff only rewrites cells drawn this
        // frame, so cells outside the new layout kept stale glyphs (#4511 D4)
        let size = terminal::size()?;
        let mode_changed =
            needs_full_repaint(first_frame, is_exploded != was_exploded, size, last_size);
        last_size = size;
        was_exploded = is_exploded;
        first_frame = false;

        render_frame(stdout, &app, &mut renderer, mode_changed, t.as_deref_mut())?;

        if !app.running {
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

/// Headless soak for the #4511 leak gate: the real collector thread, snapshot
/// apply, draw and diff, with the bytes discarded instead of written to a tty.
fn soak(
    mut app: App,
    frames: u64,
    refresh_ms: u64,
    width: u16,
    height: u16,
    mut t: Option<&mut Timings>,
) -> io::Result<()> {
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);
    let (rx, _stop) = spawn_metrics_collector(refresh_ms, app.deterministic, t.is_some());
    let mut output = Vec::with_capacity(32768);
    let mut stdout = io::stdout();
    for frame in 1..=frames {
        while let Ok(snapshot) = rx.try_recv() {
            apply(&mut app, snapshot, t.as_deref_mut());
        }
        draw_diff(
            &app,
            &mut renderer,
            width,
            height,
            frame == 1,
            &mut output,
            t.as_deref_mut(),
            |_| Ok(()),
        )?;
        if frame % 10 == 0 || frame == frames {
            let rss = rss_kib().map_or_else(|| "null".to_string(), |k| k.to_string());
            writeln!(stdout, "{{\"frame\":{frame},\"rss_kib\":{rss}}}")?;
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    stdout.flush()
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
        app.exploded_panel = cli.explode;
        return render_once(&app, cli.width, cli.height);
    }

    if let Some(frames) = cli.soak_frames {
        let mut app = App::with_config_lightweight(cli.deterministic, config);
        app.exploded_panel = cli.explode;
        let mut timings = cli.timings.then(Timings::new);
        soak(
            app,
            frames,
            cli.refresh,
            cli.width,
            cli.height,
            timings.as_mut(),
        )?;
        if let Some(t) = timings {
            print!("{}", t.to_json_lines());
        }
        return Ok(());
    }

    let mut app = App::with_config(cli.deterministic, config);
    app.exploded_panel = cli.explode;

    // a panic must not leave the user's terminal raw, hidden-cursor and alt-screen (#4511 D7)
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = cleanup_terminal(&mut io::stdout());
        default_hook(info);
    }));

    let mut stdout = io::stdout();
    setup_terminal(&mut stdout)?;

    let color_mode = if cli.no_color {
        ColorMode::Mono
    } else {
        ColorMode::TrueColor
    };

    let mut timings = cli.timings.then(Timings::new);
    let result = run_app(&mut stdout, app, cli.refresh, color_mode, timings.as_mut());
    cleanup_terminal(&mut stdout)?;
    if let Some(t) = timings {
        eprint!("{}", t.to_json_lines());
    }
    result
}
