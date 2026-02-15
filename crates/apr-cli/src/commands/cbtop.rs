//! cbtop - ComputeBrick Top (TUI for brick pipeline visualization)
//!
//! Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §6 + §12.11
//!
//! Toyota Way Principles:
//! - Mieruka: Make status visible at a glance
//! - Jidoka: Highlight budget violations immediately
//! - Genchi Genbutsu: Show real metrics, not estimates
//!
//! Usage:
//!   cbtop --model qwen2.5-coder-1.5b
//!   apr cbtop --attach realizar
//!   apr cbtop --model-path /path/to/model.gguf --headless --json  # GGUF profiling
//!   apr cbtop --model-path /path/to/model.safetensors --headless --json  # SafeTensors
//!   apr cbtop --model-path /path/to/model.apr --headless --json  # APR profiling
//!
//! Headless mode for CI:
//!   apr cbtop --headless --json --output results.json
//!   apr cbtop --headless --ci --throughput 400 --brick-score 90
//!
//! Real profiling mode (§12.11 Unified BrickProfiler):
//!   apr cbtop --model-path model.{gguf,safetensors,apr} --headless --json
//!   - Uses realizar for actual inference (CUDA or CPU)
//!   - Unified BrickProfiler timing for ALL formats
//!   - Reports real hardware info from CUDA context

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
use std::path::PathBuf;
use std::time::Instant;

/// Supported model formats for unified BrickProfiler (§12.11)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// GGUF format (.gguf) - llama.cpp compatible quantized models
    Gguf,
    /// SafeTensors format (.safetensors) - HuggingFace f16/bf16 models
    SafeTensors,
    /// APR format (.apr) - Our native optimized format
    Apr,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // Idiomatic &self for enum methods
impl ModelFormat {
    /// Detect format from file extension
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "gguf" => Some(Self::Gguf),
            "safetensors" => Some(Self::SafeTensors),
            "apr" => Some(Self::Apr),
            _ => None,
        }
    }

    /// Format-specific brick name prefix (per §12.11.1)
    pub fn brick_prefix(&self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::SafeTensors => "st",
            Self::Apr => "apr",
        }
    }
}

/// Configuration for cbtop command
#[derive(Debug, Clone)]
pub struct CbtopConfig {
    pub model: Option<String>,
    pub attach: Option<String>,
    /// Path to model file for real profiling (§12.11 Unified BrickProfiler)
    /// Supports: .gguf, .safetensors, .apr
    pub model_path: Option<PathBuf>,
    pub headless: bool,
    pub json: bool,
    pub output: Option<PathBuf>,
    pub ci: bool,
    pub throughput_threshold: Option<f64>,
    pub brick_score_threshold: Option<u32>,
    pub warmup: usize,
    pub iterations: usize,
    /// PAR-100: Enable speculative decoding benchmark
    pub speculative: bool,
    /// PAR-100: Number of tokens to draft speculatively (default: 4)
    pub speculation_k: usize,
    /// PAR-099: Path to draft model for speculative decoding
    pub draft_model_path: Option<PathBuf>,
    /// PAR-102: Number of concurrent requests for aggregate throughput measurement
    pub concurrent: usize,
    /// Use simulated data (for CI testing only - explicitly opts out of real profiling)
    pub simulated: bool,
}

impl Default for CbtopConfig {
    fn default() -> Self {
        Self {
            model: None,
            attach: None,
            model_path: None,
            headless: false,
            json: false,
            output: None,
            ci: false,
            throughput_threshold: None,
            brick_score_threshold: None,
            warmup: 10,
            iterations: 100,
            speculative: false,
            speculation_k: 4,
            draft_model_path: None,
            concurrent: 1, // PAR-102: Default to single request
            simulated: false,
        }
    }
}

/// Headless report output per spec section 7.0.1
#[derive(Debug, Clone)]
pub struct HeadlessReport {
    pub model: String,
    pub timestamp: String,
    pub hardware: HardwareInfo,
    pub throughput: ThroughputMetrics,
    pub brick_scores: Vec<BrickScore>,
    pub pmat_scores: PmatScores,
    pub falsification: FalsificationSummary,
    pub status: String,
    pub ci_result: String,
}

/// PMAT quality scores per spec section 7.0.1
#[derive(Debug, Clone)]
pub struct PmatScores {
    pub rust_project_score: f64,
    pub tdg_score: f64,
    pub cuda_tdg_score: f64,
    pub brick_score: u32,
    pub grade: String,
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub gpu: String,
    pub cpu: String,
    pub memory_gb: u32,
}

#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    pub tokens_per_sec: f64,
    pub ttft_ms: f64,
    pub cv_percent: f64,
    pub p50_us: f64,
    pub p99_us: f64,
}

#[derive(Debug, Clone)]
pub struct BrickScore {
    pub name: String,
    pub score: u32,
    pub grade: String,
    pub budget_us: f64,
    pub actual_us: f64,
    pub gap_factor: f64,
}

#[derive(Debug, Clone)]
pub struct FalsificationSummary {
    pub total_points: u32,
    pub passed: u32,
    pub failed: u32,
    pub blocked: u32,
}

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
        self.bricks.iter().max_by(|a, b| {
            a.gap_factor()
                .partial_cmp(&b.gap_factor())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn update_demo(&mut self) {
        // Demo mode: simulate timing data
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before Unix epoch")
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
pub fn run(config: CbtopConfig) -> Result<()> {
    if config.headless {
        run_headless(config)
    } else {
        run_tui(config.model.as_deref(), config.attach.as_deref())
    }
}

/// Run headless mode for CI/automation
fn run_headless(config: CbtopConfig) -> Result<()> {
    // Toyota Way: Genchi Genbutsu - Use real data by default.
    // Simulation is only allowed when explicitly requested with --simulated.

    // If --simulated is set, use simulated data (for CI testing)
    if config.simulated {
        eprintln!("cbtop: WARNING - Using simulated data (--simulated flag set)");
        eprintln!("       For real profiling: apr cbtop --model-path <FILE> --headless");
        return run_headless_simulated(config);
    }

    #[cfg(feature = "inference")]
    {
        if config.model_path.is_some() {
            run_headless_real(config)
        } else {
            Err(CliError::ValidationFailed(
                "Headless mode requires --model-path for real profiling.\n\
                 For CI testing with simulated data, use: apr cbtop --headless --simulated\n\
                 For real profiling, use: apr cbtop --model-path <FILE> --headless"
                    .to_string(),
            ))
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        return Err(CliError::ValidationFailed(
            "Headless mode requires --model-path and the 'inference' feature.\n\
             For CI testing with simulated data, use: apr cbtop --headless --simulated\n\
             Rebuild with: cargo build -p apr-cli --features inference"
                .to_string(),
        ));
    }
}

include!("cbtop_part_02.rs");
include!("cbtop_part_03.rs");
include!("cbtop_part_04.rs");
include!("cbtop_part_05.rs");
include!("cbtop_part_06.rs");
