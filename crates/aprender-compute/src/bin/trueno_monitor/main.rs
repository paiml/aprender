//! Trueno Compute Monitor TUI (TRUENO-SPEC-020)
//!
//! Real-time terminal UI for monitoring compute flow, memory utilization,
//! and data movement across heterogeneous hardware.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin trueno-monitor --features tui-monitor
//! cargo run --bin trueno-monitor --features tui-monitor,cuda -- --stress-test
//! RUST_LOG=debug trueno-monitor  # Enable verbose logging
//! ```
//!
//! # Logging
//!
//! Logs are written to `~/.trueno/monitor.log` by default.
//! Set `RUST_LOG=debug` for verbose output.

mod render;
mod render_stress;
mod stress;
mod types;

use types::*;

use std::io::stdout;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};

use presentar_terminal::direct::{CellBuffer, DiffRenderer};
use presentar_terminal::{ColorMode, Theme};

use tracing::{debug, info, warn};

/// Matrix dimension used for CPU stress test workloads.
pub(crate) const STRESS_TEST_MATRIX_SIZE: usize = 512;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use trueno_gpu::monitor::{cuda_monitoring_available, ComputeDevice, CpuDevice, MemoryMetrics};
#[cfg(feature = "cuda")]
use trueno_gpu::monitor::{CudaDeviceInfo, CudaMemoryInfo};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "cuda")]
fn vram_stats(ctx: &CudaContext, info: &CudaDeviceInfo) -> (f64, f64, f64) {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    if let Ok(mem) = CudaMemoryInfo::query(ctx) {
        (mem.used() as f64 / GIB, mem.total as f64 / GIB, mem.usage_percent())
    } else {
        (0.0, info.total_memory_gb(), 0.0)
    }
}

impl App {
    fn new() -> Self {
        let mut cpu = CpuDevice::new();
        if let Err(e) = cpu.refresh() {
            warn!("CPU refresh failed during init: {e}");
        }

        info!(cpu_name = cpu.device_name(), cores = num_cpus::get(), "CPU detected");

        // Try to enumerate real CUDA GPUs
        let cuda_available = cuda_monitoring_available();
        #[allow(unused_mut)] // mutated inside #[cfg(feature = "cuda")] block
        let mut gpus = Vec::new();
        #[allow(unused_mut)]
        let mut gpu_vram_history = Vec::new();

        debug!(cuda_available, "CUDA monitoring check");

        if cuda_available {
            #[cfg(feature = "cuda")]
            {
                if let Ok(devices) = CudaDeviceInfo::enumerate() {
                    info!(gpu_count = devices.len(), "CUDA GPUs enumerated");
                    for info in devices {
                        if let Ok(ctx) = CudaContext::new(info.index as i32) {
                            let (vram_used_gb, vram_total_gb, vram_percent) =
                                vram_stats(&ctx, &info);

                            info!(
                                gpu_index = info.index,
                                gpu_name = %info.name,
                                vram_total_gb,
                                vram_used_gb,
                                "GPU initialized"
                            );

                            gpus.push(GpuState {
                                info,
                                ctx,
                                vram_used_gb,
                                vram_total_gb,
                                vram_percent,
                            });
                            gpu_vram_history.push(vec![0; 60]);
                        }
                    }
                }
            }
        } else {
            warn!("CUDA monitoring not available");
        }

        Self {
            cpu,
            memory: MemoryMetrics::default(),
            cpu_history: vec![0; 60],
            mem_history: vec![0; 60],
            selected_tab: 0,
            stress_running: false,
            stress_config: None,
            show_help: false,
            tick: 0,
            gpus,
            gpu_vram_history,
            cpu_workers: Vec::new(),
            mem_worker: None,
            gpu_workers: Vec::new(),
            cpu_ops_per_sec: 0,
            cpu_ops_history: vec![0; 60],
            mem_ops_per_sec: 0,
            mem_ops_history: vec![0; 60],
            gpu_ops_per_sec: 0,
            gpu_ops_history: vec![0; 60],
            stress_start: None,
            peak_cpu_ops: 0,
            peak_mem_ops: 0,
            peak_gpu_ops: 0,
            peak_cpu_util: 0.0,
            peak_ram_util: 0.0,
            peak_vram_util: 0.0,
            stress_report: None,
        }
    }

    fn on_tick(&mut self) {
        self.tick += 1;

        // Refresh CPU metrics — GH-194: log failures instead of discarding
        if let Err(e) = self.cpu.refresh() {
            warn!("CPU refresh failed: {e}");
        }

        // Update CPU history
        let cpu_pct = self.cpu.compute_utilization().unwrap_or(0.0) as u64;
        self.cpu_history.remove(0);
        self.cpu_history.push(cpu_pct);

        // Update memory metrics
        self.memory.refresh();

        // Update memory history
        let mem_pct = self.memory.ram_usage_percent() as u64;
        self.mem_history.remove(0);
        self.mem_history.push(mem_pct);

        // Update GPU metrics from real hardware
        #[cfg(feature = "cuda")]
        for (i, gpu) in self.gpus.iter_mut().enumerate() {
            if let Ok(mem) = CudaMemoryInfo::query(&gpu.ctx) {
                gpu.vram_used_gb = mem.used() as f64 / (1024.0 * 1024.0 * 1024.0);
                gpu.vram_total_gb = mem.total as f64 / (1024.0 * 1024.0 * 1024.0);
                gpu.vram_percent = mem.usage_percent();

                if i < self.gpu_vram_history.len() {
                    self.gpu_vram_history[i].remove(0);
                    self.gpu_vram_history[i].push(gpu.vram_percent as u64);
                }
            }
        }

        // Update stress test metrics
        if self.stress_running {
            // Sum CPU ops from all workers and reset counters
            let mut total_cpu_ops: u64 = 0;
            for worker in &self.cpu_workers {
                let ops = worker.ops_count.swap(0, Ordering::Relaxed);
                total_cpu_ops += ops;
            }
            // Convert to ops/sec (tick is 100ms, so multiply by 10)
            self.cpu_ops_per_sec = total_cpu_ops * 10;
            if self.cpu_ops_per_sec > self.peak_cpu_ops {
                self.peak_cpu_ops = self.cpu_ops_per_sec;
            }

            // Memory ops
            if let Some(ref worker) = self.mem_worker {
                let ops = worker.ops_count.swap(0, Ordering::Relaxed);
                self.mem_ops_per_sec = ops * 10;
                if self.mem_ops_per_sec > self.peak_mem_ops {
                    self.peak_mem_ops = self.mem_ops_per_sec;
                }
            }

            // GPU ops
            let mut total_gpu_ops: u64 = 0;
            for worker in &self.gpu_workers {
                let ops = worker.ops_count.swap(0, Ordering::Relaxed);
                total_gpu_ops += ops;
            }
            self.gpu_ops_per_sec = total_gpu_ops * 10;
            if self.gpu_ops_per_sec > self.peak_gpu_ops {
                self.peak_gpu_ops = self.gpu_ops_per_sec;
            }

            // Update histories
            self.cpu_ops_history.remove(0);
            self.cpu_ops_history.push(self.cpu_ops_per_sec / 1_000_000); // M ops/sec
            self.mem_ops_history.remove(0);
            self.mem_ops_history.push(self.mem_ops_per_sec / 1_000_000);
            self.gpu_ops_history.remove(0);
            self.gpu_ops_history.push(self.gpu_ops_per_sec / 1_000_000_000); // G FLOPS

            // Track peak utilization for stress report
            let cpu_util = self.cpu.compute_utilization().unwrap_or(0.0);
            if cpu_util > self.peak_cpu_util {
                self.peak_cpu_util = cpu_util;
            }
            let ram_util = self.memory.ram_usage_percent();
            if ram_util > self.peak_ram_util {
                self.peak_ram_util = ram_util;
            }
            let vram_util = self.gpus.iter().map(|g| g.vram_percent).fold(0.0_f64, f64::max);
            if vram_util > self.peak_vram_util {
                self.peak_vram_util = vram_util;
            }
        }
    }

    fn next_tab(&mut self) {
        self.selected_tab = (self.selected_tab + 1) % 4;
    }

    fn prev_tab(&mut self) {
        if self.selected_tab > 0 {
            self.selected_tab -= 1;
        } else {
            self.selected_tab = 3;
        }
    }

    fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }
}

/// Initialize logging to file
fn init_logging() -> tracing_appender::non_blocking::WorkerGuard {
    // Create log directory
    let log_dir = dirs_log_path();
    std::fs::create_dir_all(&log_dir).ok();

    // Setup file appender with daily rotation
    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "monitor.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // Build subscriber with env filter (default: info)
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_err| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        )
        .init();

    guard
}

/// Get log directory path (~/.trueno)
fn dirs_log_path() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".trueno")
}

/// Returns true if the app should quit
fn handle_key_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    if key.kind != KeyEventKind::Press {
        return false;
    }
    match key.code {
        KeyCode::Char('q') => return true,
        KeyCode::Char('?') => app.toggle_help(),
        KeyCode::Char('s') => app.toggle_stress(),
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Left => app.prev_tab(),
        KeyCode::Right => app.next_tab(),
        _ => {} // Ignore unmapped keys
    }
    false
}

/// Poll for a key event and handle it. Returns true if the app should quit.
fn poll_and_handle_key(
    app: &mut App,
    timeout: Duration,
) -> Result<bool, Box<dyn std::error::Error>> {
    if !event::poll(timeout)? {
        return Ok(false);
    }
    match event::read()? {
        Event::Key(key) => Ok(handle_key_event(app, key)),
        _ => Ok(false),
    }
}

fn run_event_loop(
    buffer: &mut CellBuffer,
    renderer: &mut DiffRenderer,
    theme: &Theme,
    app: &mut App,
) -> Result<(), Box<dyn std::error::Error>> {
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        // Resize buffer if terminal size changed
        let (width, height) = crossterm::terminal::size()?;
        if buffer.width() != width || buffer.height() != height {
            *buffer = CellBuffer::new(width, height);
        }

        render::ui(buffer, renderer, theme, app)?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if poll_and_handle_key(app, timeout)? {
            break;
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let _log_guard = init_logging();

    let args: Vec<String> = std::env::args().collect();
    let start_stress = args.iter().any(|a| a == "--stress-test");

    info!(
        version = env!("CARGO_PKG_VERSION"),
        stress_mode = start_stress,
        "Trueno Monitor starting"
    );

    let (width, height) = crossterm::terminal::size()?;
    let mut buffer = CellBuffer::new(width, height);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);
    let theme = Theme::tokyo_night();

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    stdout().execute(crossterm::cursor::Hide)?;

    let mut app = App::new();
    if start_stress {
        app.toggle_stress();
    }

    let result = run_event_loop(&mut buffer, &mut renderer, &theme, &mut app);

    stdout().execute(crossterm::cursor::Show)?;
    stdout().execute(LeaveAlternateScreen)?;
    disable_raw_mode()?;

    info!("Trueno Monitor shutdown complete");

    result
}
