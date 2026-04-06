//! Input handling and load iteration for CbtopApp.

use std::time::Instant;

use crossterm::event::KeyCode;

use crate::bricks::generators::SimdLoadBrick;
use crate::config::{ComputeBackend, WorkloadType};

use super::hardware::LoadMetrics;
use super::panels::ActivePanel;
use super::CbtopApp;

impl CbtopApp {
    /// Run one iteration of real compute load
    pub(super) fn run_load_iteration(&mut self) {
        let start = Instant::now();

        // Execute real compute work via SIMD generator
        let duration = self.simd_generator.run_iteration();

        // Update metrics
        self.load_metrics.total_bricks += 1;

        let latency_us = duration.as_secs_f64() * 1_000_000.0;
        if latency_us > 0.0 {
            // Exponential moving average for latency
            self.load_metrics.avg_latency_us =
                0.9 * self.load_metrics.avg_latency_us + 0.1 * latency_us;
        }

        // Calculate bricks/second
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let current_bps = 1.0 / elapsed;
            self.load_metrics.bricks_per_second =
                0.9 * self.load_metrics.bricks_per_second + 0.1 * current_bps;
        }

        // Calculate ops/second (2 * N ops for FMA: mul + add)
        let work_size = ((self.problem_size as f64) * self.intensity) as usize;
        self.load_metrics.ops_per_second =
            self.load_metrics.bricks_per_second * (work_size * 2) as f64;

        // Calculate bytes/second (3 arrays * N * 4 bytes)
        self.load_metrics.bytes_per_second =
            self.load_metrics.bricks_per_second * (work_size * 3 * 4) as f64;

        self.bricks_history.push(self.load_metrics.bricks_per_second);
    }

    /// Collect real system metrics
    pub(super) fn collect_real_metrics(&mut self) {
        // Read real CPU usage from /proc/stat
        let cpu_usage = self.read_cpu_usage();
        self.cpu_history.push(cpu_usage);
        self.load_metrics.cpu_usage = cpu_usage;

        // Read memory breakdown from /proc/meminfo (PMAT-012 UI-04)
        self.load_metrics.memory = Self::read_memory();

        // Read network metrics from /proc/net/dev (PMAT-012 UI-07 P2)
        self.load_metrics.network = self.read_network();

        // Read disk metrics from statvfs (PMAT-012 UI-08 P2)
        self.load_metrics.disks = Self::read_disks();

        // Track frame time
        let now = Instant::now();
        let frame_ms = now.duration_since(self.last_frame).as_secs_f64() * 1000.0;
        self.frame_times.push(frame_ms);
        self.last_frame = now;
        self.frame_count += 1;
    }

    /// Handle key press
    #[allow(clippy::wildcard_enum_match_arm)]
    pub(super) fn handle_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char(' ') => {
                self.is_running = !self.is_running;
                if self.is_running {
                    self.simd_generator.start();
                } else {
                    self.simd_generator.stop();
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.intensity = (self.intensity + 0.1).min(1.0);
                self.simd_generator.set_intensity(self.intensity);
            }
            KeyCode::Char('-') => {
                self.intensity = (self.intensity - 0.1).max(0.0);
                self.simd_generator.set_intensity(self.intensity);
            }
            KeyCode::Char('b') => {
                self.backend = match self.backend {
                    ComputeBackend::Simd => ComputeBackend::Wgpu,
                    ComputeBackend::Wgpu => ComputeBackend::Cuda,
                    ComputeBackend::Cuda => ComputeBackend::All,
                    ComputeBackend::All => ComputeBackend::Simd,
                };
            }
            KeyCode::Char('w') => {
                self.workload = match self.workload {
                    WorkloadType::Gemm => WorkloadType::Conv2d,
                    WorkloadType::Conv2d => WorkloadType::Attention,
                    WorkloadType::Attention => WorkloadType::Bandwidth,
                    WorkloadType::Bandwidth => WorkloadType::Elementwise,
                    WorkloadType::Elementwise => WorkloadType::Reduction,
                    WorkloadType::Reduction => WorkloadType::All,
                    WorkloadType::All => WorkloadType::Gemm,
                };
            }
            KeyCode::Char('[') => {
                self.problem_size = (self.problem_size / 2).max(1024);
                self.simd_generator = SimdLoadBrick::new(self.problem_size);
                self.simd_generator.set_intensity(self.intensity);
                if self.is_running {
                    self.simd_generator.start();
                }
            }
            KeyCode::Char(']') => {
                self.problem_size = (self.problem_size * 2).min(1_073_741_824);
                self.simd_generator = SimdLoadBrick::new(self.problem_size);
                self.simd_generator.set_intensity(self.intensity);
                if self.is_running {
                    self.simd_generator.start();
                }
            }
            KeyCode::Char('r') => {
                self.cpu_history.clear();
                self.bricks_history.clear();
                self.frame_times.clear();
                self.frame_count = 0;
                self.load_metrics = LoadMetrics::default();
            }
            KeyCode::Char(c) if c.is_ascii_digit() => {
                if let Some(panel) = ActivePanel::from_key(c) {
                    self.active_panel = panel;
                }
            }
            _ => {}
        }
    }
}
