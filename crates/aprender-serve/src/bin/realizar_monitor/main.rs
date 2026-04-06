//! realizar-monitor: Real-time btop-style TUI for inference server monitoring
//!
//! A rich terminal dashboard inspired by simular's TUI patterns.
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
//! ┌─────────────────────────────────────┬────────────────────────────┐
//! │ Throughput                     60%  │ Metrics                40% │
//! │ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁    │                            │
//! │                                     │ Throughput: 192.4 tok/s ↑ │
//! │ Current: 192.4 tok/s                │ Latency P50: 5.2ms        │
//! │ Peak:    256.1 tok/s                │ Latency P95: 7.8ms        │
//! │ Trend:   ↑ Rising                   │ Latency P99: 12.1ms       │
//! ├─────────────────────────────────────┤                            │
//! │ GPU Memory                          │ Requests: 1,234           │
//! │ ████████████████████░░░░░░░░ 67%    │ Tokens:   50,000          │
//! │ 16.1 GB / 24.0 GB                   │ Uptime:   1h 23m 45s      │
//! │                                     ├────────────────────────────┤
//! │ CUDA: ● Active                      │ System                     │
//! │ Batch Size: 32                      │ Model: phi-2-q4_k_m        │
//! │ Queue: 8 pending                    │ Batch: 32 optimal          │
//! └─────────────────────────────────────┴────────────────────────────┘
//! │ [q] Quit  [r] Reset  [p] Pause                               │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use clap::Parser;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline as RatatuiSparkline},
};
use serde::Deserialize;

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
/// - Green: ≤70% (healthy)
/// - Yellow: 71-90% (warning)
/// - Red: >90% (critical)
fn gpu_color(percent: u16) -> Color {
    if percent > 90 {
        Color::Red
    } else if percent > 70 {
        Color::Yellow
    } else {
        Color::Green
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
            return "→"; // Not enough data
        }
        let recent: Vec<f64> = self.data.iter().rev().take(5).cloned().collect();
        let first_avg = (recent[3] + recent[4]) / 2.0;
        let last_avg = (recent[0] + recent[1]) / 2.0;

        let range = self.max().unwrap_or(1.0) - self.min().unwrap_or(0.0);
        let threshold = range * 0.05;

        if last_avg > first_avg + threshold {
            "↑"
        } else if last_avg < first_avg - threshold {
            "↓"
        } else {
            "→"
        }
    }

    /// Render sparkline as Unicode string (PARITY-109 QA-E08)
    /// Uses block characters to visualize time series data.
    fn sparkline(&self, width: usize) -> String {
        const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

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
}

impl MonitorApp {
    fn new(url: String) -> Self {
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = MonitorApp::new(args.url);
    let refresh_duration = Duration::from_millis(args.refresh_ms);

    // Main loop
    let result = run_app(&mut terminal, &mut app, refresh_duration);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {err}");
    }

    Ok(())
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
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

        // Draw UI
        terminal.draw(|f| ui(f, app))?;

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

include!("main_02.rs");
