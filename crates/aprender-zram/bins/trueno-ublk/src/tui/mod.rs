//! TUI module - interactive dashboard for trueno-ublk
//!
//! Provides real-time monitoring similar to `htop` for compression performance.

mod widgets;

use crate::cli::TopArgs;
use crate::device::UblkDevice;
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Table},
    Frame, Terminal,
};
use std::io;
use std::time::{Duration, Instant};

/// Application state
struct App {
    devices: Vec<UblkDevice>,
    selected_device: usize,
    demo_mode: bool,
    demo_counter: u64,
    should_quit: bool,
    last_update: Instant,
}

impl App {
    fn new(args: &TopArgs) -> Result<Self> {
        let devices = if let Some(ref path) = args.device {
            vec![UblkDevice::open(path)?]
        } else {
            UblkDevice::list_all()?
        };

        Ok(Self {
            devices,
            selected_device: 0,
            demo_mode: args.demo,
            demo_counter: 0,
            should_quit: false,
            last_update: Instant::now(),
        })
    }

    fn on_tick(&mut self) {
        if self.demo_mode {
            self.demo_counter += 1;
        }
        self.last_update = Instant::now();
    }

    fn on_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected_device > 0 {
                    self.selected_device -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected_device + 1 < self.devices.len() {
                    self.selected_device += 1;
                }
            }
            _ => {}
        }
    }
}

/// Initialize terminal for TUI rendering
fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).map_err(Into::into)
}

/// Restore terminal to normal state
fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}

/// Handle a single key event
fn handle_key_event(app: &mut App) -> Result<()> {
    if let Event::Key(key) = event::read()? {
        if key.kind == KeyEventKind::Press {
            app.on_key(key.code);
        }
    }
    Ok(())
}

/// Run the TUI
pub fn run(args: TopArgs) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new(&args)?;

    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout =
            tick_rate.checked_sub(last_tick.elapsed()).unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            handle_key_event(&mut app)?;
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }

        if app.should_quit {
            break;
        }
    }

    restore_terminal(&mut terminal)
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(8),  // Overview
            Constraint::Min(10),    // Device list
            Constraint::Length(12), // Selected device details
            Constraint::Length(3),  // Footer
        ])
        .split(f.area());

    render_header(f, chunks[0]);
    render_overview(f, chunks[1], app);
    render_device_list(f, chunks[2], app);
    render_device_details(f, chunks[3], app);
    render_footer(f, chunks[4]);
}

fn render_header(f: &mut Frame, area: Rect) {
    let header = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            " trueno-ublk ",
            Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" GPU-accelerated ZRAM replacement"),
    ])])
    .block(Block::default().borders(Borders::BOTTOM));

    f.render_widget(header, area);
}

fn render_overview(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // Calculate totals
    let mut total_orig = 0u64;
    let mut total_compr = 0u64;
    let mut total_throughput = 0.0f64;
    let mut gpu_count = 0;

    for device in &app.devices {
        if app.demo_mode {
            total_orig += 100_000_000_000 + app.demo_counter * 1_000_000;
            total_compr += 30_000_000_000 + app.demo_counter * 300_000;
            total_throughput += 3.5 + (app.demo_counter % 10) as f64 * 0.1;
            gpu_count += 1;
        } else {
            let stats = device.stats();
            total_orig += stats.orig_data_size;
            total_compr += stats.compr_data_size;
            total_throughput += stats.throughput_gbps;
            if device.config().gpu_enabled {
                gpu_count += 1;
            }
        }
    }

    let ratio = if total_compr > 0 { total_orig as f64 / total_compr as f64 } else { 1.0 };

    // Data stored
    let data_block = Block::default().title(" Data ").borders(Borders::ALL);
    let data_text = format!("{}\nRatio: {:.2}:1", crate::cli::format_size(total_orig), ratio);
    let data = Paragraph::new(data_text).block(data_block).style(Style::default().fg(Color::Green));
    f.render_widget(data, chunks[0]);

    // Compression ratio gauge
    let savings = if ratio > 1.0 { ((1.0 - 1.0 / ratio) * 100.0) as u16 } else { 0 };
    let ratio_gauge = Gauge::default()
        .block(Block::default().title(" Savings ").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(savings.min(100))
        .label(format!("{}%", savings));
    f.render_widget(ratio_gauge, chunks[1]);

    // Throughput
    let throughput_block = Block::default().title(" Throughput ").borders(Borders::ALL);
    let throughput_text = format!("{:.2} GB/s", total_throughput);
    let throughput = Paragraph::new(throughput_text)
        .block(throughput_block)
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(throughput, chunks[2]);

    // GPU status
    let gpu_block = Block::default().title(" GPU ").borders(Borders::ALL);
    let gpu_text =
        if gpu_count > 0 { format!("{} active", gpu_count) } else { "disabled".to_string() };
    let gpu = Paragraph::new(gpu_text)
        .block(gpu_block)
        .style(Style::default().fg(if gpu_count > 0 { Color::Magenta } else { Color::DarkGray }));
    f.render_widget(gpu, chunks[3]);
}

fn render_device_list(f: &mut Frame, area: Rect, app: &App) {
    let header = Row::new(vec!["NAME", "SIZE", "DATA", "COMPR", "RATIO", "ALGO", "GPU", "BACKEND"])
        .style(Style::default().add_modifier(Modifier::BOLD))
        .height(1);

    let rows: Vec<Row> = if app.demo_mode {
        vec![
            Row::new(vec![
                "ublkb0".to_string(),
                "1.0T".to_string(),
                crate::cli::format_size(100_000_000_000 + app.demo_counter * 1_000_000),
                crate::cli::format_size(30_000_000_000 + app.demo_counter * 300_000),
                "3.33:1".to_string(),
                "lz4".to_string(),
                "yes".to_string(),
                "avx512".to_string(),
            ]),
            Row::new(vec![
                "ublkb1".to_string(),
                "256.0G".to_string(),
                crate::cli::format_size(50_000_000_000),
                crate::cli::format_size(20_000_000_000),
                "2.50:1".to_string(),
                "zstd3".to_string(),
                "no".to_string(),
                "avx2".to_string(),
            ]),
        ]
    } else {
        app.devices
            .iter()
            .map(|d| {
                let stats = d.stats();
                let config = d.config();
                let ratio = if stats.compr_data_size > 0 {
                    stats.orig_data_size as f64 / stats.compr_data_size as f64
                } else {
                    1.0
                };
                Row::new(vec![
                    d.name(),
                    crate::cli::format_size(config.size),
                    crate::cli::format_size(stats.orig_data_size),
                    crate::cli::format_size(stats.compr_data_size),
                    format!("{:.2}:1", ratio),
                    format!("{:?}", config.algorithm).to_lowercase(),
                    if config.gpu_enabled { "yes" } else { "no" }.to_string(),
                    stats.simd_backend,
                ])
            })
            .collect()
    };

    let widths = [
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(5),
        Constraint::Length(8),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().title(" Devices ").borders(Borders::ALL))
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(table, area);
}

fn render_device_details(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Entropy distribution
    let entropy_block = Block::default().title(" Entropy Distribution ").borders(Borders::ALL);

    let (gpu_pct, simd_pct, scalar_pct) = if app.demo_mode {
        let offset = app.demo_counter % 20;
        (40 + offset, 45 - offset / 2, 15 + offset / 2)
    } else if let Some(device) = app.devices.get(app.selected_device) {
        let stats = device.stats();
        let total = stats.gpu_pages + stats.simd_pages + stats.scalar_pages;
        if total > 0 {
            (
                stats.gpu_pages * 100 / total,
                stats.simd_pages * 100 / total,
                stats.scalar_pages * 100 / total,
            )
        } else {
            (0, 0, 0)
        }
    } else {
        (0, 0, 0)
    };

    let entropy_text = vec![
        Line::from(vec![
            Span::styled("GPU batch:   ", Style::default().fg(Color::Magenta)),
            Span::raw(format!("{:>3}%", gpu_pct)),
            Span::raw(" "),
            Span::styled(
                "▓".repeat((gpu_pct as usize / 5).min(20)),
                Style::default().fg(Color::Magenta),
            ),
        ]),
        Line::from(vec![
            Span::styled("SIMD:        ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{:>3}%", simd_pct)),
            Span::raw(" "),
            Span::styled(
                "▓".repeat((simd_pct as usize / 5).min(20)),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("Scalar/Skip: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{:>3}%", scalar_pct)),
            Span::raw(" "),
            Span::styled(
                "▓".repeat((scalar_pct as usize / 5).min(20)),
                Style::default().fg(Color::Yellow),
            ),
        ]),
    ];

    let entropy = Paragraph::new(entropy_text).block(entropy_block);
    f.render_widget(entropy, chunks[0]);

    // I/O stats
    let io_block = Block::default().title(" I/O Statistics ").borders(Borders::ALL);

    let (failed_r, failed_w, invalid, zero_pages) = if app.demo_mode {
        (0, 0, 0, 50000 + app.demo_counter)
    } else if let Some(device) = app.devices.get(app.selected_device) {
        let stats = device.stats();
        (stats.failed_reads, stats.failed_writes, stats.invalid_io, stats.same_pages)
    } else {
        (0, 0, 0, 0)
    };

    let io_text = vec![
        Line::from(format!("Failed reads:  {}", failed_r)),
        Line::from(format!("Failed writes: {}", failed_w)),
        Line::from(format!("Invalid I/O:   {}", invalid)),
        Line::from(format!("Zero pages:    {}", zero_pages)),
    ];

    let io = Paragraph::new(io_text).block(io_block);
    f.render_widget(io, chunks[1]);
}

fn render_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" q ", Style::default().bg(Color::DarkGray)),
        Span::raw(" Quit  "),
        Span::styled(" ↑↓ ", Style::default().bg(Color::DarkGray)),
        Span::raw(" Navigate  "),
        Span::styled(" r ", Style::default().bg(Color::DarkGray)),
        Span::raw(" Refresh  "),
    ]));
    f.render_widget(footer, area);
}
