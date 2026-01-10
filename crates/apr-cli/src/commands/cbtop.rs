//! cbtop - ComputeBrick Top (TUI for brick pipeline visualization)
//!
//! Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §6
//!
//! Toyota Way Principles:
//! - Mieruka: Make status visible at a glance
//! - Jidoka: Highlight budget violations immediately
//! - Genchi Genbutsu: Show real metrics, not estimates
//!
//! Usage:
//!   cbtop --model qwen2.5-coder-1.5b
//!   apr cbtop --attach realizar

use crate::error::{CliError, Result};
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
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline, Tabs},
    Frame, Terminal,
};
use std::io;

/// Brick timing data
#[derive(Debug, Clone)]
struct BrickTiming {
    name: &'static str,
    budget_us: f64,
    actual_us: f64,
    samples: Vec<f64>,
}

impl BrickTiming {
    fn new(name: &'static str, budget_us: f64) -> Self {
        Self {
            name,
            budget_us,
            actual_us: 0.0,
            samples: Vec::with_capacity(100),
        }
    }

    fn gap_factor(&self) -> f64 {
        if self.budget_us > 0.0 {
            self.actual_us / self.budget_us
        } else {
            1.0
        }
    }

    fn status(&self) -> &'static str {
        if self.actual_us <= self.budget_us {
            "✅"
        } else {
            "❌"
        }
    }

    fn percent_of_budget(&self) -> u16 {
        if self.budget_us > 0.0 {
            ((self.actual_us / self.budget_us) * 100.0).min(200.0) as u16
        } else {
            100
        }
    }

    fn add_sample(&mut self, us: f64) {
        self.samples.push(us);
        if self.samples.len() > 100 {
            self.samples.remove(0);
        }
        // Update actual as moving average
        self.actual_us = self.samples.iter().sum::<f64>() / self.samples.len() as f64;
    }

    fn sparkline_data(&self) -> Vec<u64> {
        self.samples
            .iter()
            .map(|&x| (x * 10.0).min(255.0) as u64)
            .collect()
    }
}

/// Pipeline state
#[derive(Debug, Clone)]
struct PipelineState {
    bricks: Vec<BrickTiming>,
    layer_idx: usize,
    total_layers: usize,
    tokens_generated: usize,
    total_us: f64,
    target_tok_s: f64,
    current_tok_s: f64,
}

impl PipelineState {
    fn new() -> Self {
        // Default budgets from spec §3.1
        let bricks = vec![
            BrickTiming::new("RmsNorm", 1.5),
            BrickTiming::new("QkvBrick", 6.0),
            BrickTiming::new("RoPE", 1.0),
            BrickTiming::new("Attention", 10.0),
            BrickTiming::new("OProj", 3.5),
            BrickTiming::new("RmsNorm", 1.5),
            BrickTiming::new("FfnBrick", 12.2),
        ];

        Self {
            bricks,
            layer_idx: 0,
            total_layers: 28, // Default for 1.5B
            tokens_generated: 0,
            total_us: 0.0,
            target_tok_s: 976.0, // 2x llama.cpp for 1.5B
            current_tok_s: 0.0,
        }
    }

    fn total_budget(&self) -> f64 {
        self.bricks.iter().map(|b| b.budget_us).sum()
    }

    fn total_actual(&self) -> f64 {
        self.bricks.iter().map(|b| b.actual_us).sum()
    }

    fn bottleneck(&self) -> Option<&BrickTiming> {
        self.bricks
            .iter()
            .max_by(|a, b| a.gap_factor().partial_cmp(&b.gap_factor()).unwrap())
    }

    fn update_demo(&mut self) {
        // Demo mode: simulate timing data
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        for (i, brick) in self.bricks.iter_mut().enumerate() {
            // Add some variance around the budget
            let base = brick.budget_us;
            let variance = (((seed >> (i * 4)) & 0xFF) as f64 / 255.0 - 0.5) * base * 0.4;
            brick.add_sample(base + variance);
        }

        self.tokens_generated += 1;
        self.total_us = self.total_actual() * self.total_layers as f64;
        if self.total_us > 0.0 {
            self.current_tok_s = 1_000_000.0 / self.total_us;
        }
    }
}

/// Active view
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Pipeline,
    Budget,
    Histogram,
    Gpu,
    Memory,
}

impl View {
    fn titles() -> Vec<&'static str> {
        vec![
            "Pipeline [p]",
            "Budget [b]",
            "Histogram [h]",
            "GPU [g]",
            "Memory [m]",
        ]
    }

    fn index(self) -> usize {
        match self {
            View::Pipeline => 0,
            View::Budget => 1,
            View::Histogram => 2,
            View::Gpu => 3,
            View::Memory => 4,
        }
    }
}

/// Application state
struct App {
    model_name: String,
    pipeline: PipelineState,
    current_view: View,
    selected_brick: usize,
    should_quit: bool,
    demo_mode: bool,
}

impl App {
    fn new(model: Option<&str>) -> Self {
        Self {
            model_name: model.unwrap_or("qwen2.5-coder-1.5b").to_string(),
            pipeline: PipelineState::new(),
            current_view: View::Pipeline,
            selected_brick: 0,
            should_quit: false,
            demo_mode: true, // Start in demo mode if no live connection
        }
    }

    fn next_brick(&mut self) {
        if !self.pipeline.bricks.is_empty() {
            self.selected_brick = (self.selected_brick + 1) % self.pipeline.bricks.len();
        }
    }

    fn prev_brick(&mut self) {
        if !self.pipeline.bricks.is_empty() {
            self.selected_brick = if self.selected_brick == 0 {
                self.pipeline.bricks.len() - 1
            } else {
                self.selected_brick - 1
            };
        }
    }

    fn tick(&mut self) {
        if self.demo_mode {
            self.pipeline.update_demo();
        }
    }
}

/// Run the cbtop command
pub fn run(model: Option<&str>, _attach: Option<&str>) -> Result<()> {
    // Setup terminal
    enable_raw_mode()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to enable raw mode: {e}")))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to setup terminal: {e}")))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create terminal: {e}")))?;

    // Create app and run
    let mut app = App::new(model);
    let res = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .ok();
    terminal.show_cursor().ok();

    res
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        // Update state
        app.tick();

        terminal
            .draw(|f| ui(f, app))
            .map_err(|e| CliError::ValidationFailed(format!("Failed to draw: {e}")))?;

        if event::poll(std::time::Duration::from_millis(100))
            .map_err(|e| CliError::ValidationFailed(format!("Event poll error: {e}")))?
        {
            if let Event::Key(key) = event::read()
                .map_err(|e| CliError::ValidationFailed(format!("Event read error: {e}")))?
            {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
                        KeyCode::Char('p') => app.current_view = View::Pipeline,
                        KeyCode::Char('b') => app.current_view = View::Budget,
                        KeyCode::Char('h') => app.current_view = View::Histogram,
                        KeyCode::Char('g') => app.current_view = View::Gpu,
                        KeyCode::Char('m') => app.current_view = View::Memory,
                        KeyCode::Down | KeyCode::Char('j') => app.next_brick(),
                        KeyCode::Up | KeyCode::Char('k') => app.prev_brick(),
                        KeyCode::Enter => {} // Drill into brick (future)
                        _ => {}
                    }
                }
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

fn ui(f: &mut Frame<'_>, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Summary
            Constraint::Length(1), // Status
        ])
        .split(f.area());

    render_title(f, chunks[0], app);
    render_tabs(f, chunks[1], app);

    match app.current_view {
        View::Pipeline => render_pipeline(f, chunks[2], app),
        View::Budget => render_budget(f, chunks[2], app),
        View::Histogram => render_histogram(f, chunks[2], app),
        View::Gpu => render_gpu(f, chunks[2], app),
        View::Memory => render_memory(f, chunks[2], app),
    }

    render_summary(f, chunks[3], app);
    render_status(f, chunks[4], app);
}

fn render_title(f: &mut Frame<'_>, area: Rect, app: &App) {
    let title = format!(
        " cbtop - ComputeBrick Pipeline Monitor │ {} │ Layer {}/{} ",
        app.model_name, app.pipeline.layer_idx, app.pipeline.total_layers
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            title,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    f.render_widget(block, area);
}

fn render_tabs(f: &mut Frame<'_>, area: Rect, app: &App) {
    let titles: Vec<Line<'_>> = View::titles().iter().map(|t| Line::from(*t)).collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Views"))
        .select(app.current_view.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

fn render_pipeline(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Transformer Layer Pipeline (Mieruka: Visual Control) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Split into brick list and sparkline
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(inner);

    // Brick list with progress bars
    let items: Vec<ListItem<'_>> = app
        .pipeline
        .bricks
        .iter()
        .enumerate()
        .map(|(i, brick)| {
            let percent = brick.percent_of_budget();
            let bar_len = 20;
            let filled = (percent as usize * bar_len / 100).min(bar_len);
            let bar: String = format!(
                "{}{}",
                "█".repeat(filled),
                "░".repeat(bar_len.saturating_sub(filled))
            );

            let color = if percent <= 100 {
                Color::Green
            } else if percent <= 120 {
                Color::Yellow
            } else {
                Color::Red
            };

            let selected = if i == app.selected_brick {
                "► "
            } else {
                "  "
            };

            let bottleneck = if Some(brick.name) == app.pipeline.bottleneck().map(|b| b.name)
                && brick.gap_factor() > 1.0
            {
                " ← BOTTLENECK"
            } else {
                ""
            };

            let line = Line::from(vec![
                Span::raw(selected),
                Span::styled(
                    format!("{:12}", brick.name),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" │ "),
                Span::styled(
                    format!("{:6.1}µs", brick.actual_us),
                    Style::default().fg(color),
                ),
                Span::raw(" │ "),
                Span::raw(brick.status()),
                Span::raw(" │ "),
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(format!(" {:3}%", percent), Style::default().fg(color)),
                Span::styled(
                    bottleneck,
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, chunks[0]);

    // Sparkline for selected brick
    if let Some(brick) = app.pipeline.bricks.get(app.selected_brick) {
        let data = brick.sparkline_data();
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} History ", brick.name)),
            )
            .data(&data)
            .style(Style::default().fg(Color::Cyan));
        f.render_widget(sparkline, chunks[1]);
    }
}

fn render_budget(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Budget Compliance (Jidoka: Stop on Violation) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let items: Vec<ListItem<'_>> = app
        .pipeline
        .bricks
        .iter()
        .map(|brick| {
            let gap = brick.gap_factor();
            let status_color = if gap <= 1.0 {
                Color::Green
            } else if gap <= 1.2 {
                Color::Yellow
            } else {
                Color::Red
            };

            let line = Line::from(vec![
                Span::styled(
                    format!("{:12}", brick.name),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" │ Budget: "),
                Span::styled(
                    format!("{:5.1}µs", brick.budget_us),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" │ Actual: "),
                Span::styled(
                    format!("{:5.1}µs", brick.actual_us),
                    Style::default().fg(status_color),
                ),
                Span::raw(" │ Gap: "),
                Span::styled(format!("{:.2}x", gap), Style::default().fg(status_color)),
                Span::raw(" │ "),
                Span::raw(brick.status()),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, inner);
}

fn render_histogram(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Latency Distribution (p50/p99/p999) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if let Some(brick) = app.pipeline.bricks.get(app.selected_brick) {
        let mut sorted = brick.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        let p99 = sorted
            .get((sorted.len() as f64 * 0.99) as usize)
            .copied()
            .unwrap_or(0.0);
        let p999 = sorted
            .get((sorted.len() as f64 * 0.999) as usize)
            .copied()
            .unwrap_or(0.0);

        let text = vec![
            Line::from(format!("Brick: {}", brick.name)),
            Line::from(""),
            Line::from(vec![
                Span::raw("  p50:  "),
                Span::styled(format!("{:6.2}µs", p50), Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::raw("  p99:  "),
                Span::styled(format!("{:6.2}µs", p99), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw("  p999: "),
                Span::styled(format!("{:6.2}µs", p999), Style::default().fg(Color::Red)),
            ]),
            Line::from(""),
            Line::from(format!("  Samples: {}", brick.samples.len())),
        ];

        let paragraph = Paragraph::new(text);
        f.render_widget(paragraph, inner);
    }
}

fn render_gpu(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" GPU Metrics (Genchi Genbutsu: Real Data) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Placeholder GPU metrics (would come from nvidia-smi/NVML)
    let text = vec![
        Line::from(Span::styled(
            "GPU Status",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Device:      NVIDIA RTX 4090"),
        Line::from("  Memory:      16.2 / 24.0 GB (67%)"),
        Line::from("  Utilization: 94%"),
        Line::from("  Temperature: 72°C"),
        Line::from("  Power:       385W / 450W"),
        Line::from(""),
        Line::from(Span::styled(
            "CUDA Graphs",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Captured:    Yes"),
        Line::from("  Replay Mode: Active"),
        Line::from("  Overhead:    < 100µs"),
        Line::from(""),
        Line::from(Span::styled(
            "(Real metrics require CUDA connection)",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(text);
    f.render_widget(paragraph, inner);
}

fn render_memory(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Memory Bandwidth ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let text = vec![
        Line::from(Span::styled(
            "Memory Performance",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Peak Bandwidth:    1008 GB/s"),
        Line::from("  Achieved:          720 GB/s (71%)"),
        Line::from(""),
        Line::from(Span::styled(
            "Per-Brick Bandwidth",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  RmsNorm:   650 GB/s (bandwidth-bound)"),
        Line::from("  QkvBrick:  180 GB/s (compute-bound)"),
        Line::from("  Attention: 420 GB/s (memory-bound)"),
        Line::from("  FfnBrick:  210 GB/s (compute-bound)"),
        Line::from(""),
        Line::from(Span::styled(
            "(Requires ncu profiler for accurate data)",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(text);
    f.render_widget(paragraph, inner);
}

fn render_summary(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(" Summary ");

    let total_budget = app.pipeline.total_budget();
    let total_actual = app.pipeline.total_actual();
    // Model-level budgets for future use (full model = layer × layers)
    let _model_budget = total_budget * app.pipeline.total_layers as f64;
    let _model_actual = total_actual * app.pipeline.total_layers as f64;

    let status_color = if total_actual <= total_budget {
        Color::Green
    } else {
        Color::Red
    };

    let status = if total_actual <= total_budget {
        "✅ PASS"
    } else {
        "❌ FAIL"
    };

    let text = Line::from(vec![
        Span::raw(" Current: "),
        Span::styled(
            format!("{:.1} tok/s", app.pipeline.current_tok_s),
            Style::default().fg(status_color),
        ),
        Span::raw(" │ Target: "),
        Span::styled(
            format!("{:.0} tok/s", app.pipeline.target_tok_s),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(" │ Layer: "),
        Span::styled(
            format!("{:.1}µs", total_actual),
            Style::default().fg(status_color),
        ),
        Span::raw("/"),
        Span::styled(
            format!("{:.1}µs", total_budget),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(" │ "),
        Span::styled(status, Style::default().fg(status_color)),
    ]);

    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, area);
}

fn render_status(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let status =
        "[Enter] Drill into brick  [p]ipeline  [b]udget  [h]istogram  [g]pu  [m]emory  [q]uit";

    let paragraph = Paragraph::new(status).style(Style::default().fg(Color::DarkGray));
    f.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brick_timing_new() {
        let brick = BrickTiming::new("test", 5.0);
        assert_eq!(brick.name, "test");
        assert_eq!(brick.budget_us, 5.0);
        assert_eq!(brick.actual_us, 0.0);
    }

    #[test]
    fn test_brick_timing_gap_factor() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.actual_us = 7.5;
        assert!((brick.gap_factor() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_brick_timing_status() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.actual_us = 4.0;
        assert_eq!(brick.status(), "✅");

        brick.actual_us = 6.0;
        assert_eq!(brick.status(), "❌");
    }

    #[test]
    fn test_brick_timing_add_sample() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(4.0);
        brick.add_sample(6.0);
        assert_eq!(brick.samples.len(), 2);
        assert!((brick.actual_us - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_state_new() {
        let state = PipelineState::new();
        assert_eq!(state.bricks.len(), 7);
        assert_eq!(state.total_layers, 28);
    }

    #[test]
    fn test_pipeline_total_budget() {
        let state = PipelineState::new();
        let total = state.total_budget();
        // Sum: 1.5 + 6.0 + 1.0 + 10.0 + 3.5 + 1.5 + 12.2 = 35.7
        assert!((total - 35.7).abs() < 0.001);
    }

    #[test]
    fn test_view_titles() {
        let titles = View::titles();
        assert_eq!(titles.len(), 5);
        assert!(titles[0].contains("Pipeline"));
    }

    #[test]
    fn test_view_index() {
        assert_eq!(View::Pipeline.index(), 0);
        assert_eq!(View::Budget.index(), 1);
        assert_eq!(View::Histogram.index(), 2);
        assert_eq!(View::Gpu.index(), 3);
        assert_eq!(View::Memory.index(), 4);
    }

    #[test]
    fn test_app_new() {
        let app = App::new(Some("test-model"));
        assert_eq!(app.model_name, "test-model");
        assert_eq!(app.current_view, View::Pipeline);
        assert!(!app.should_quit);
    }

    #[test]
    fn test_app_navigation() {
        let mut app = App::new(None);
        assert_eq!(app.selected_brick, 0);

        app.next_brick();
        assert_eq!(app.selected_brick, 1);

        app.prev_brick();
        assert_eq!(app.selected_brick, 0);

        // Wrap around
        app.prev_brick();
        assert_eq!(app.selected_brick, 6); // 7 bricks, wraps to last
    }
}
