//! Stress test orchestration: start/stop workers, report generation.
//!
//! Extracted from main.rs for file health (TRUENO-SPEC-020).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use trueno::Matrix;
use trueno_gpu::monitor::{ChaosPreset, StressTarget, StressTestConfig};

use super::types::*;
use super::STRESS_TEST_MATRIX_SIZE;

/// Matrix dimension used for CPU stress test workloads.
const STRESS_TEST_FLOPS_PER_MATMUL: u64 = 2
    * (STRESS_TEST_MATRIX_SIZE as u64)
    * (STRESS_TEST_MATRIX_SIZE as u64)
    * (STRESS_TEST_MATRIX_SIZE as u64);

impl App {
    pub(crate) fn toggle_stress(&mut self) {
        if self.stress_running {
            // Stop stress test
            self.stop_stress();
        } else {
            // Start stress test
            self.start_stress();
        }
    }

    fn start_stress(&mut self) {
        let num_workers = (num_cpus::get() / 4).max(1);
        info!(
            cpu_workers = num_workers,
            matrix_size = "512x512",
            flops_per_matmul = STRESS_TEST_FLOPS_PER_MATMUL,
            gpu_count = self.gpus.len(),
            gpu_buffer_size = "256MB x 4",
            "Starting stress test with trueno SIMD/CUDA"
        );

        self.stress_running = true;
        self.stress_start = Some(Instant::now());
        self.stress_config = Some(StressTestConfig {
            target: StressTarget::All,
            duration: Duration::from_secs(60),
            intensity: 0.8,
            ramp_up: Duration::from_secs(5),
            chaos_preset: Some(ChaosPreset::Gentle),
            collect_metrics: true,
            export_report: false,
        });

        self.reset_stress_peaks();
        self.spawn_cpu_stress_workers();
        self.spawn_mem_stress_worker();
        #[cfg(feature = "cuda")]
        self.spawn_gpu_stress_workers();
    }

    fn reset_stress_peaks(&mut self) {
        self.peak_cpu_ops = 0;
        self.peak_mem_ops = 0;
        self.peak_gpu_ops = 0;
        self.peak_cpu_util = 0.0;
        self.peak_ram_util = 0.0;
        self.peak_vram_util = 0.0;
        self.cpu_ops_history = vec![0; 60];
        self.mem_ops_history = vec![0; 60];
        self.gpu_ops_history = vec![0; 60];
        self.stress_report = None;
    }

    fn spawn_cpu_stress_workers(&mut self) {
        let num_workers = (num_cpus::get() / 4).max(1);
        let matrix_size = STRESS_TEST_MATRIX_SIZE;

        for worker_id in 0..num_workers {
            let running = Arc::new(AtomicBool::new(true));
            let ops_count = Arc::new(AtomicU64::new(0));

            let r = running.clone();
            let o = ops_count.clone();

            let thread = thread::spawn(move || {
                let n = matrix_size;
                let data_a: Vec<f32> =
                    (0..n * n).map(|i| ((i + worker_id) % 1000) as f32 * 0.001).collect();
                let data_b: Vec<f32> =
                    (0..n * n).map(|i| ((i * 7 + worker_id) % 1000) as f32 * 0.001).collect();

                let a = Matrix::from_vec(n, n, data_a).expect("stress test matrix A creation");
                let b = Matrix::from_vec(n, n, data_b).expect("stress test matrix B creation");

                while r.load(Ordering::Relaxed) {
                    // GH-194: Only count FLOPs on successful matmul
                    if a.matmul(&b).is_ok() {
                        o.fetch_add((n * n * n * 2) as u64, Ordering::Relaxed);
                    }
                }
            });

            self.cpu_workers.push(StressWorker { running, ops_count, thread: Some(thread) });
        }
    }

    fn spawn_mem_stress_worker(&mut self) {
        let running = Arc::new(AtomicBool::new(true));
        let ops_count = Arc::new(AtomicU64::new(0));

        let r = running.clone();
        let o = ops_count.clone();

        let thread = thread::spawn(move || {
            let mut buffers: Vec<Vec<u8>> = Vec::new();
            let chunk_size = 64 * 1024 * 1024;

            while r.load(Ordering::Relaxed) {
                if buffers.len() < 8 {
                    let mut buf = vec![0u8; chunk_size];
                    for i in (0..buf.len()).step_by(4096) {
                        buf[i] = (i & 0xFF) as u8;
                    }
                    buffers.push(buf);
                    o.fetch_add(chunk_size as u64 / 4096, Ordering::Relaxed);
                } else {
                    for buf in &mut buffers {
                        for i in (0..buf.len()).step_by(4096) {
                            buf[i] = buf[i].wrapping_add(1);
                        }
                        o.fetch_add(buf.len() as u64 / 4096, Ordering::Relaxed);
                    }
                    if buffers.len() > 4 {
                        buffers.pop();
                    }
                }
                thread::sleep(Duration::from_millis(10));
            }
        });

        self.mem_worker = Some(StressWorker { running, ops_count, thread: Some(thread) });
    }

    #[cfg(feature = "cuda")]
    fn spawn_gpu_stress_workers(&mut self) {
        use trueno_gpu::driver::{CudaContext, GpuBuffer};

        let num_gpus = self.gpus.len();
        for gpu_idx in 0..num_gpus {
            let running = Arc::new(AtomicBool::new(true));
            let ops_count = Arc::new(AtomicU64::new(0));

            let r = running.clone();
            let o = ops_count.clone();

            let thread = thread::spawn(move || {
                if let Ok(ctx) = CudaContext::new(gpu_idx as i32) {
                    let n: usize = 64 * 1024 * 1024;
                    let data: Vec<f32> = (0..n).map(|i| (i % 10000) as f32 * 0.0001).collect();

                    let mut buffers: Vec<GpuBuffer<f32>> = Vec::new();
                    for _ in 0..4 {
                        if let Ok(buf) = GpuBuffer::<f32>::new(&ctx, n) {
                            buffers.push(buf);
                        }
                    }

                    if buffers.len() >= 2 {
                        let mut result = vec![0.0f32; n];

                        while r.load(Ordering::Relaxed) {
                            // GH-194: Only count bytes for successful transfers
                            let mut successful_copies = 0usize;
                            for buf in &mut buffers {
                                if buf.copy_from_host(&data).is_ok() {
                                    successful_copies += 1;
                                }
                            }
                            if buffers[0].copy_to_host(&mut result).is_ok() {
                                successful_copies += 1;
                            }
                            let bytes_transferred = successful_copies * n * 4;
                            o.fetch_add(bytes_transferred as u64, Ordering::Relaxed);
                        }
                    }
                }
            });

            self.gpu_workers.push(StressWorker { running, ops_count, thread: Some(thread) });
        }
    }

    fn stop_stress(&mut self) {
        self.stress_running = false;

        let duration_secs = self.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        let cpu_worker_count = self.cpu_workers.len();
        let gpu_worker_count = self.gpu_workers.len();

        info!(
            duration_secs,
            cpu_workers = cpu_worker_count,
            gpu_workers = gpu_worker_count,
            "Stopping stress test"
        );

        self.signal_all_workers_stop();
        self.join_all_workers();
        self.generate_stress_report(duration_secs, cpu_worker_count, gpu_worker_count);

        self.cpu_workers.clear();
        self.mem_worker = None;
        self.gpu_workers.clear();
        self.stress_config = None;
    }

    fn signal_all_workers_stop(&self) {
        for worker in &self.cpu_workers {
            worker.running.store(false, Ordering::Relaxed);
        }
        if let Some(ref worker) = self.mem_worker {
            worker.running.store(false, Ordering::Relaxed);
        }
        for worker in &self.gpu_workers {
            worker.running.store(false, Ordering::Relaxed);
        }
    }

    fn join_all_workers(&mut self) {
        // GH-194: Log worker thread panics instead of silently discarding
        for (i, worker) in self.cpu_workers.iter_mut().enumerate() {
            if let Some(thread) = worker.thread.take() {
                if let Err(e) = thread.join() {
                    eprintln!("Warning: CPU stress worker {} panicked: {:?}", i, e);
                }
            }
        }
        if let Some(ref mut worker) = self.mem_worker {
            if let Some(thread) = worker.thread.take() {
                if let Err(e) = thread.join() {
                    eprintln!("Warning: memory stress worker panicked: {:?}", e);
                }
            }
        }
        for (i, worker) in self.gpu_workers.iter_mut().enumerate() {
            if let Some(thread) = worker.thread.take() {
                if let Err(e) = thread.join() {
                    eprintln!("Warning: GPU stress worker {} panicked: {:?}", i, e);
                }
            }
        }
    }

    fn generate_stress_report(
        &mut self,
        duration_secs: u64,
        cpu_worker_count: usize,
        gpu_worker_count: usize,
    ) {
        let verdict =
            if self.peak_cpu_util > 95.0 && (gpu_worker_count == 0 || self.peak_vram_util > 10.0) {
                StressTestVerdict::Pass
            } else if self.peak_cpu_util > 70.0 {
                StressTestVerdict::PassWithNotes
            } else {
                StressTestVerdict::Fail
            };

        let mut recommendations = Vec::new();
        if self.peak_cpu_util < 90.0 {
            recommendations.push(
                "CPU saturation below 90% - consider more compute-intensive workloads".to_string(),
            );
        }
        if gpu_worker_count > 0 && self.peak_vram_util < 20.0 {
            recommendations.push("GPU VRAM usage low - consider larger buffer sizes".to_string());
        }
        if self.peak_ram_util > 90.0 {
            recommendations
                .push("High RAM pressure detected - monitor for OOM conditions".to_string());
        }

        self.stress_report = Some(StressTestReport {
            duration_secs,
            peak_cpu_ops: self.peak_cpu_ops,
            peak_mem_ops: self.peak_mem_ops,
            peak_gpu_ops: self.peak_gpu_ops,
            peak_cpu_util: self.peak_cpu_util,
            peak_ram_util: self.peak_ram_util,
            peak_vram_util: self.peak_vram_util,
            cpu_workers: cpu_worker_count,
            gpu_workers: gpu_worker_count,
            verdict,
            recommendations: recommendations.clone(),
        });

        info!(
            duration_secs,
            verdict = %verdict,
            peak_cpu_ops = self.peak_cpu_ops,
            peak_mem_ops = self.peak_mem_ops,
            peak_gpu_ops = self.peak_gpu_ops,
            peak_cpu_util = format!("{:.1}%", self.peak_cpu_util),
            peak_ram_util = format!("{:.1}%", self.peak_ram_util),
            peak_vram_util = format!("{:.1}%", self.peak_vram_util),
            "Stress test completed"
        );
        for rec in &recommendations {
            warn!(recommendation = %rec, "Stress test recommendation");
        }
    }
}
