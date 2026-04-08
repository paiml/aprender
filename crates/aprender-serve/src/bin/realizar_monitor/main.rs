//! realizar-monitor: Real-time btop-style TUI for inference server monitoring
//!
//! A rich terminal dashboard inspired by simular's TUI patterns.
//! Uses presentar-terminal CellBuffer + DiffRenderer for rendering.
//!
//! # Usage
//!
//! ```bash
//! # Start server in one terminal
//! realizar serve --model model.gguf --batch
//!
//! # Monitor in another terminal
//! realizar-monitor --url http://127.0.0.1:8080
//! ```
//!
//! # Layout
//!
//! ```text
//! +-------------------------------------+----------------------------+
//! | Throughput                     60%  | Metrics                40% |
//! |                                     |                            |
//! | Current: 192.4 tok/s                | Throughput: 192.4 tok/s    |
//! | Peak:    256.1 tok/s                | Latency P50: 5.2ms        |
//! | Trend:   Rising                     | Latency P95: 7.8ms        |
//! +-------------------------------------| Latency P99: 12.1ms       |
//! | GPU Memory                          |                            |
//! | ==================== 67%            | Requests: 1,234           |
//! | 16.1 GB / 24.0 GB                   | Tokens:   50,000          |
//! |                                     +----------------------------+
//! | CUDA: Active                        | System                     |
//! | Batch Size: 32                      | Model: phi-2-q4_k_m        |
//! | Queue: 8 pending                    | Batch: 32 optimal          |
//! +-------------------------------------+----------------------------+
//! | [q] Quit  [r] Reset  [p] Pause                                  |
//! +------------------------------------------------------------------+
//! ```

use std::collections::VecDeque;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use clap::Parser;
use crossterm::{
    cursor,
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use presentar_terminal::{CellBuffer, Color, DiffRenderer, Modifiers};
use serde::Deserialize;

/// CYAN color constant
const CYAN: Color = Color {
    r: 0.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

/// Monitor state machine (PARITY-108a)
/// Explicit state enum for QA compliance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorState {
    /// Not connected to server
    Disconnected,
    /// Connected and receiving metrics
    Connected,
    /// Connected but updates paused
    Paused,
}

impl MonitorState {
    /// Check if the monitor is receiving updates
    fn is_active(&self) -> bool {
        matches!(self, MonitorState::Connected)
    }
}

/// GPU utilization color coding (PARITY-108c)
/// Returns appropriate color based on GPU utilization percentage
///
/// - Green: <=70% (healthy)
/// - Yellow: 71-90% (warning)
/// - Red: >90% (critical)
fn gpu_color(percent: u16) -> Color {
    if percent > 90 {
        Color::RED
    } else if percent > 70 {
        Color::YELLOW
    } else {
        Color::GREEN
    }
}

/// Command line arguments
#[derive(Parser, Debug)]
#[command(name = "realizar-monitor")]
#[command(about = "Real-time btop-style monitoring TUI for realizar inference server")]
#[command(version)]
struct Args {
    /// Server URL to monitor
    #[arg(short, long, default_value = "http://127.0.0.1:8080")]
    url: String,

    /// Refresh rate in milliseconds
    #[arg(short, long, default_value = "100")]
    refresh_ms: u64,
}

/// Metrics from server /v1/metrics endpoint
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerMetrics {
    #[serde(default)]
    pub throughput_tok_per_sec: f64,
    #[serde(default)]
    pub latency_p50_ms: f64,
    #[serde(default)]
    pub latency_p95_ms: f64,
    #[serde(default)]
    pub latency_p99_ms: f64,
    #[serde(default)]
    pub gpu_memory_used_bytes: u64,
    #[serde(default)]
    pub gpu_memory_total_bytes: u64,
    #[serde(default)]
    pub gpu_utilization_percent: u32,
    #[serde(default)]
    pub cuda_path_active: bool,
    #[serde(default)]
    pub batch_size: usize,
    #[serde(default)]
    pub queue_depth: usize,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub total_requests: u64,
    #[serde(default)]
    pub uptime_secs: u64,
    #[serde(default)]
    pub model_name: String,
}

/// Time series data with circular buffer (inspired by simular)
#[derive(Debug, Clone)]
struct TimeSeries {
    data: VecDeque<f64>,
    capacity: usize,
}

impl TimeSeries {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, value: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    fn as_u64_vec(&self) -> Vec<u64> {
        self.data.iter().map(|&v| v as u64).collect()
    }

    fn min(&self) -> Option<f64> {
        self.data.iter().cloned().reduce(f64::min)
    }

    fn max(&self) -> Option<f64> {
        self.data.iter().cloned().reduce(f64::max)
    }
    fn last(&self) -> Option<f64> {
        self.data.back().copied()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    /// Compute trend direction (inspired by trueno-viz sparkline)
    fn trend(&self) -> &'static str {
        if self.data.len() < 5 {
            return "->"; // Not enough data
        }
        let recent: Vec<f64> = self.data.iter().rev().take(5).cloned().collect();
        let first_avg = (recent[3] + recent[4]) / 2.0;
        let last_avg = (recent[0] + recent[1]) / 2.0;

        let range = self.max().unwrap_or(1.0) - self.min().unwrap_or(0.0);
        let threshold = range * 0.05;

        if last_avg > first_avg + threshold {
            "UP"
        } else if last_avg < first_avg - threshold {
            "DN"
        } else {
            "--"
        }
    }

    /// Render sparkline as Unicode string (PARITY-109 QA-E08)
    /// Uses block characters to visualize time series data.
    fn sparkline(&self, width: usize) -> String {
        const CHARS: [char; 8] = ['\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];

        let min = self.min().unwrap_or(0.0);
        let max = self.max().unwrap_or(1.0);
        let range = (max - min).max(0.001);

        self.data
            .iter()
            .rev()
            .take(width)
            .rev()
            .map(|&v| {
                let idx = ((v - min) / range * 7.0) as usize;
                CHARS[idx.min(7)]
            })
            .collect()
    }
}

/// Monitor application state (inspired by simular's DashboardState)
struct MonitorApp {
    /// Server URL
    url: String,
    /// Current metrics
    metrics: ServerMetrics,
    /// Throughput time series
    throughput_series: TimeSeries,
    /// Latency time series
    latency_series: TimeSeries,
    /// Peak throughput
    peak_throughput: f64,
    /// Monitor state machine (PARITY-108a)
    state: MonitorState,
    /// Last error message
    last_error: Option<String>,
    /// Should quit
    should_quit: bool,
    /// Start time
    start_time: Instant,
    /// Cell buffer for rendering
    buffer: CellBuffer,
    /// Diff renderer for efficient updates
    renderer: DiffRenderer,
    /// Terminal width
    width: u16,
    /// Terminal height
    height: u16,
}

impl MonitorApp {
    fn new(url: String) -> Self {
        let (width, height) = crossterm::terminal::size().unwrap_or((100, 30));
        Self {
            url,
            metrics: ServerMetrics::default(),
            throughput_series: TimeSeries::new(60),
            latency_series: TimeSeries::new(60),
            peak_throughput: 0.0,
            state: MonitorState::Disconnected,
            last_error: None,
            should_quit: false,
            start_time: Instant::now(),
            buffer: CellBuffer::new(width, height),
            renderer: DiffRenderer::new(),
            width,
            height,
        }
    }

    /// Fetch metrics from server
    fn fetch_metrics(&mut self) {
        // Only fetch if in Connected state (not Paused or Disconnected)
        if self.state == MonitorState::Paused {
            return;
        }

        let metrics_url = format!("{}/v1/metrics", self.url);

        match ureq::get(&metrics_url)
            .timeout(Duration::from_millis(500))
            .call()
        {
            Ok(response) => {
                match response.into_json::<ServerMetrics>() {
                    Ok(metrics) => {
                        // Update time series
                        self.throughput_series.push(metrics.throughput_tok_per_sec);
                        self.latency_series.push(metrics.latency_p50_ms);

                        // Track peak
                        if metrics.throughput_tok_per_sec > self.peak_throughput {
                            self.peak_throughput = metrics.throughput_tok_per_sec;
                        }

                        self.metrics = metrics;
                        self.state = MonitorState::Connected;
                        self.last_error = None;
                    },
                    Err(e) => {
                        self.state = MonitorState::Disconnected;
                        self.last_error = Some(format!("JSON parse error: {}", e));
                    },
                }
            },
            Err(e) => {
                self.state = MonitorState::Disconnected;
                self.last_error = Some(format!("Connection error: {}", e));
            },
        }
    }

    /// Format uptime as human-readable string
    fn format_uptime(&self) -> String {
        let secs = self.metrics.uptime_secs;
        if secs >= 3600 {
            format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
        } else if secs >= 60 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}s", secs)
        }
    }

    /// Format bytes as GB
    fn format_gb(bytes: u64) -> String {
        format!("{:.1} GB", bytes as f64 / 1e9)
    }

    /// Reset statistics
    fn reset(&mut self) {
        self.throughput_series = TimeSeries::new(60);
        self.latency_series = TimeSeries::new(60);
        self.peak_throughput = 0.0;
    }

    /// Write a string at (x, y) with color
    fn write_str(&mut self, x: u16, y: u16, s: &str, fg: Color) {
        let mut cx = x;
        for ch in s.chars() {
            if cx >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            self.buffer
                .update(cx, y, encoded, fg, Color::TRANSPARENT, Modifiers::NONE);
            cx = cx.saturating_add(1);
        }
    }

    /// Set a single character with color
    fn set_char(&mut self, x: u16, y: u16, ch: char, fg: Color) {
        if x < self.width && y < self.height {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer
                .update(x, y, s, fg, Color::TRANSPARENT, Modifiers::NONE);
        }
    }

    /// Draw a box with border and title
    fn draw_box(&mut self, x: u16, y: u16, w: u16, h: u16, title: &str) {
        if w < 2 || h < 2 {
            return;
        }

        // Top border
        self.set_char(x, y, '\u{250C}', Color::WHITE);
        for i in 1..w - 1 {
            self.set_char(x + i, y, '\u{2500}', Color::WHITE);
        }
        self.set_char(x + w - 1, y, '\u{2510}', Color::WHITE);

        // Title
        if !title.is_empty() && w > title.len() as u16 + 2 {
            let title_x = x + 2;
            self.write_str(title_x, y, title, CYAN);
        }

        // Sides
        for i in 1..h - 1 {
            self.set_char(x, y + i, '\u{2502}', Color::WHITE);
            self.set_char(x + w - 1, y + i, '\u{2502}', Color::WHITE);
        }

        // Bottom border
        self.set_char(x, y + h - 1, '\u{2514}', Color::WHITE);
        for i in 1..w - 1 {
            self.set_char(x + i, y + h - 1, '\u{2500}', Color::WHITE);
        }
        self.set_char(x + w - 1, y + h - 1, '\u{2518}', Color::WHITE);
    }

    /// Render a gauge/progress bar
    fn render_gauge(&mut self, x: u16, y: u16, w: u16, percent: u16, color: Color) {
        let inner_w = w.saturating_sub(2) as usize;
        let filled = ((percent as usize) * inner_w / 100).min(inner_w);
        let empty = inner_w.saturating_sub(filled);
        let bar = format!(
            "{}{}",
            "\u{2588}".repeat(filled),
            "\u{2591}".repeat(empty)
        );
        self.write_str(x, y, &bar[..bar.len().min(inner_w)], color);
        let label = format!(" {}%", percent);
        self.write_str(x + inner_w as u16, y, &label, Color::WHITE);
    }

    /// Render the full dashboard
    fn render(&mut self) {
        let w = self.width;
        let h = self.height;

        // Layout: left 60%, right 40%
        let left_w = (w * 60) / 100;
        let right_w = w.saturating_sub(left_w);

        // Left panels: throughput (55%), GPU (30%), controls (15%)
        let left_tp_h = (h * 55) / 100;
        let left_gpu_h = (h * 30) / 100;
        let left_ctrl_h = h.saturating_sub(left_tp_h + left_gpu_h);

        // Right panels: metrics (60%), system (40%)
        let right_met_h = (h * 60) / 100;
        let right_sys_h = h.saturating_sub(right_met_h);

        // === LEFT: Throughput ===
        self.draw_box(0, 0, left_w, left_tp_h, " Throughput ");

        // Sparkline
        let spark_w = left_w.saturating_sub(4) as usize;
        let spark = self.throughput_series.sparkline(spark_w);
        self.write_str(2, 2, &spark, CYAN);

        // Stats
        let current = self.metrics.throughput_tok_per_sec;
        let peak = self.peak_throughput;
        let trend = self.throughput_series.trend();
        let trend_color = match trend {
            "UP" => Color::GREEN,
            "DN" => Color::RED,
            _ => Color::YELLOW,
        };

        let stat1 = format!("Current: {:.1} tok/s", current);
        self.write_str(2, left_tp_h.saturating_sub(4), &stat1, Color::YELLOW);
        self.write_str(
            2 + stat1.len() as u16 + 1,
            left_tp_h.saturating_sub(4),
            trend,
            trend_color,
        );

        let stat2 = format!("Peak: {:.1} tok/s", peak);
        self.write_str(2, left_tp_h.saturating_sub(3), &stat2, Color::GREEN);

        let stat3 = format!(
            "Samples: {}  Min: {:.1}  Max: {:.1}",
            self.throughput_series.len(),
            self.throughput_series.min().unwrap_or(0.0),
            self.throughput_series.max().unwrap_or(0.0),
        );
        self.write_str(2, left_tp_h.saturating_sub(2), &stat3, Color::WHITE);

        // === LEFT: GPU Memory ===
        let gpu_y = left_tp_h;
        self.draw_box(0, gpu_y, left_w, left_gpu_h, " GPU Memory ");

        let gpu_percent = if self.metrics.gpu_memory_total_bytes > 0 {
            ((self.metrics.gpu_memory_used_bytes as f64
                / self.metrics.gpu_memory_total_bytes as f64)
                * 100.0) as u16
        } else {
            0
        };

        let gauge_color = gpu_color(gpu_percent);
        self.render_gauge(2, gpu_y + 1, left_w.saturating_sub(4), gpu_percent, gauge_color);

        let used = Self::format_gb(self.metrics.gpu_memory_used_bytes);
        let total = Self::format_gb(self.metrics.gpu_memory_total_bytes);
        let mem_text = format!("{} / {}", used, total);
        self.write_str(2, gpu_y + 2, &mem_text, Color::WHITE);

        let cuda_label = if self.metrics.cuda_path_active {
            "CUDA: Active"
        } else {
            "CUDA: Inactive"
        };
        let cuda_color = if self.metrics.cuda_path_active {
            Color::GREEN
        } else {
            Color::RED
        };
        self.write_str(2, gpu_y + 4, cuda_label, cuda_color);

        let batch_text = format!(
            "Batch: {}   Queue: {}",
            self.metrics.batch_size, self.metrics.queue_depth
        );
        self.write_str(2, gpu_y + 5, &batch_text, CYAN);

        // === LEFT: Controls ===
        let ctrl_y = gpu_y + left_gpu_h;
        self.draw_box(0, ctrl_y, left_w, left_ctrl_h, " Controls ");

        let (status_color, status_text) = match self.state {
            MonitorState::Paused => (Color::YELLOW, "PAUSED"),
            MonitorState::Connected => (Color::GREEN, "CONNECTED"),
            MonitorState::Disconnected => (Color::RED, "DISCONNECTED"),
        };
        self.write_str(2, ctrl_y + 1, status_text, status_color);

        let controls = "[q] Quit  [r] Reset  [p] Pause";
        self.write_str(2 + status_text.len() as u16 + 3, ctrl_y + 1, controls, CYAN);

        // === RIGHT: Metrics ===
        self.draw_box(left_w, 0, right_w, right_met_h, " Metrics ");

        let mut my = 2u16;
        let mx = left_w + 2;

        let tp_line = format!("Throughput: {:.1} tok/s", self.metrics.throughput_tok_per_sec);
        self.write_str(mx, my, &tp_line, Color::YELLOW);
        my += 2;

        let latency_trend = self.latency_series.trend();
        let latency_color = match latency_trend {
            "UP" => Color::RED,
            "DN" => Color::GREEN,
            _ => Color::YELLOW,
        };

        let p50_line = format!("Latency P50: {:.1} ms {}", self.metrics.latency_p50_ms, latency_trend);
        self.write_str(mx, my, &p50_line, latency_color);
        my += 1;

        let p95_line = format!("Latency P95: {:.1} ms", self.metrics.latency_p95_ms);
        self.write_str(mx, my, &p95_line, Color::YELLOW);
        my += 1;

        let p99_line = format!("Latency P99: {:.1} ms", self.metrics.latency_p99_ms);
        self.write_str(mx, my, &p99_line, Color::RED);
        my += 2;

        let req_line = format!("Requests: {:>10}", format_number(self.metrics.total_requests));
        self.write_str(mx, my, &req_line, CYAN);
        my += 1;

        let tok_line = format!("Tokens:   {:>10}", format_number(self.metrics.total_tokens));
        self.write_str(mx, my, &tok_line, CYAN);
        my += 1;

        let up_line = format!("Uptime:   {:>10}", self.format_uptime());
        self.write_str(mx, my, &up_line, Color::WHITE);

        // === RIGHT: System ===
        self.draw_box(left_w, right_met_h, right_w, right_sys_h, " System ");

        let sys_y = right_met_h + 1;

        let model_name = if self.metrics.model_name.is_empty() {
            "N/A".to_string()
        } else {
            self.metrics.model_name.clone()
        };

        let model_line = format!("Model: {}", model_name);
        self.write_str(mx, sys_y, &model_line, CYAN);

        let server_line = format!("Server: {}", self.url);
        let gray = Color::new(0.5, 0.5, 0.5, 1.0);
        self.write_str(mx, sys_y + 1, &server_line, gray);

        if let Some(ref error) = self.last_error {
            let err_line = format!("Error: {}", error);
            self.write_str(mx, sys_y + 3, &err_line, Color::RED);
        } else {
            self.write_str(mx, sys_y + 3, "Status: OK", Color::GREEN);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture, cursor::Hide)?;

    // Create app
    let mut app = MonitorApp::new(args.url);
    let refresh_duration = Duration::from_millis(args.refresh_ms);

    // Main loop
    let result = run_app(&mut stdout, &mut app, refresh_duration);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        stdout,
        LeaveAlternateScreen,
        DisableMouseCapture,
        cursor::Show
    )?;

    if let Err(err) = result {
        eprintln!("Error: {err}");
    }

    Ok(())
}

fn run_app(
    stdout: &mut io::Stdout,
    app: &mut MonitorApp,
    refresh: Duration,
) -> io::Result<()> {
    let mut last_fetch = Instant::now();

    loop {
        // Fetch metrics periodically
        if last_fetch.elapsed() >= refresh {
            app.fetch_metrics();
            last_fetch = Instant::now();
        }

        // Update terminal size
        let (w, h) = crossterm::terminal::size().unwrap_or((100, 30));
        if w != app.width || h != app.height {
            app.width = w;
            app.height = h;
            app.buffer.resize(w, h);
            app.renderer.reset();
        }

        // Clear and render
        app.buffer.clear();
        app.render();

        // Flush to terminal
        app.renderer.flush(&mut app.buffer, stdout)?;
        stdout.flush()?;

        // Handle input (non-blocking)
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            app.should_quit = true;
                        },
                        KeyCode::Char('r') => {
                            app.reset();
                        },
                        KeyCode::Char('p') => {
                            // Toggle between Connected and Paused states
                            app.state = match app.state {
                                MonitorState::Connected => MonitorState::Paused,
                                MonitorState::Paused => MonitorState::Connected,
                                MonitorState::Disconnected => MonitorState::Disconnected,
                            };
                        },
                        _ => {},
                    }
                }
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

/// Format number with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
