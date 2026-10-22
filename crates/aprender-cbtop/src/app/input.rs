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

        self.bricks_history
            .push(self.load_metrics.bricks_per_second);
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

#[cfg(test)]
mod handle_key_tests {
    use super::*;
    use crate::config::Config;

    fn app() -> CbtopApp {
        CbtopApp::with_size(Config::default(), 80, 24)
    }

    fn press(app: &mut CbtopApp, keys: &str) {
        for c in keys.chars() {
            app.handle_key(KeyCode::Char(c));
        }
    }

    #[test]
    fn quit_keys_set_should_quit() {
        let mut a = app();
        a.handle_key(KeyCode::Char('x'));
        assert!(!a.should_quit);
        a.handle_key(KeyCode::Char('q'));
        assert!(a.should_quit);
        let mut b = app();
        b.handle_key(KeyCode::Esc);
        assert!(b.should_quit);
    }

    #[test]
    fn space_toggles_running_and_generator() {
        let mut a = app();
        assert!(!a.is_running);
        press(&mut a, " ");
        assert!(a.is_running);
        assert!(a.simd_generator.is_running());
        press(&mut a, " ");
        assert!(!a.is_running);
        assert!(!a.simd_generator.is_running());
    }

    #[test]
    fn intensity_clamps_to_unit_interval_and_reaches_generator() {
        let mut a = app();
        press(&mut a, "++++++++++++");
        assert!((a.intensity - 1.0).abs() < f64::EPSILON);
        assert!((a.simd_generator.intensity() - 1.0).abs() < f64::EPSILON);
        press(&mut a, "=");
        assert!((a.intensity - 1.0).abs() < f64::EPSILON);
        press(&mut a, "------------");
        assert!(a.intensity.abs() < f64::EPSILON);
        assert!(a.simd_generator.intensity().abs() < f64::EPSILON);
    }

    #[test]
    fn backend_cycles_through_all_four_and_wraps() {
        let mut a = app();
        a.backend = ComputeBackend::Simd;
        let mut seen = Vec::new();
        for _ in 0..4 {
            press(&mut a, "b");
            seen.push(a.backend);
        }
        assert_eq!(
            seen,
            [
                ComputeBackend::Wgpu,
                ComputeBackend::Cuda,
                ComputeBackend::All,
                ComputeBackend::Simd
            ]
        );
    }

    #[test]
    fn workload_cycles_through_all_seven_and_wraps() {
        let mut a = app();
        a.workload = WorkloadType::Gemm;
        let mut seen = Vec::new();
        for _ in 0..7 {
            press(&mut a, "w");
            seen.push(a.workload);
        }
        assert_eq!(
            seen,
            [
                WorkloadType::Conv2d,
                WorkloadType::Attention,
                WorkloadType::Bandwidth,
                WorkloadType::Elementwise,
                WorkloadType::Reduction,
                WorkloadType::All,
                WorkloadType::Gemm
            ]
        );
    }

    #[test]
    fn problem_size_halves_doubles_and_floors_at_1024() {
        let mut a = app();
        a.problem_size = 4096;
        press(&mut a, "[");
        assert_eq!(a.problem_size, 2048);
        press(&mut a, "[[[");
        assert_eq!(a.problem_size, 1024);
        press(&mut a, "]");
        assert_eq!(a.problem_size, 2048);
    }

    #[test]
    fn resize_rebuilds_generator_and_keeps_running_state() {
        let mut a = app();
        a.problem_size = 4096;
        a.intensity = 0.5;
        press(&mut a, "[");
        assert!(!a.simd_generator.is_running());
        assert!((a.simd_generator.intensity() - 0.5).abs() < f64::EPSILON);
        press(&mut a, " ");
        press(&mut a, "]");
        assert!(a.simd_generator.is_running());
    }

    #[test]
    fn reset_clears_histories_and_metrics() {
        let mut a = app();
        a.cpu_history.push(1.0);
        a.bricks_history.push(2.0);
        a.frame_times.push(3.0);
        a.frame_count = 9;
        a.load_metrics.total_bricks = 7;
        press(&mut a, "r");
        assert_eq!(a.frame_count, 0);
        assert_eq!(a.load_metrics.total_bricks, 0);
        assert_eq!(a.cpu_history.len(), 0);
        assert_eq!(a.bricks_history.len(), 0);
        assert_eq!(a.frame_times.len(), 0);
    }

    #[test]
    fn digits_select_panels_and_zero_or_unknown_keys_do_nothing() {
        let mut a = app();
        for (key, panel) in [
            ('1', ActivePanel::Overview),
            ('2', ActivePanel::Cpu),
            ('3', ActivePanel::Gpu),
            ('4', ActivePanel::Pcie),
            ('5', ActivePanel::Memory),
            ('6', ActivePanel::Thermal),
            ('7', ActivePanel::Load),
            ('8', ActivePanel::Config),
            ('9', ActivePanel::Help),
        ] {
            a.handle_key(KeyCode::Char(key));
            assert_eq!(a.active_panel, panel, "key {key}");
        }
        a.handle_key(KeyCode::Char('0'));
        a.handle_key(KeyCode::Char('z'));
        a.handle_key(KeyCode::Tab);
        assert_eq!(a.active_panel, ActivePanel::Help);
        assert!(!a.should_quit);
    }
}
