//! Panel Verification System (probar-compliant)
//!
//! Follows probar's Brick pattern for TUI panel verification:
//! - can_render() - Jidoka gate
//! - verify() - Data validation
//! - budget_ms - Performance budget
//!
//! ## Toyota Way Application
//!
//! - **Jidoka**: Fail-fast when data is invalid
//! - **Poka-Yoke**: Type-safe verification prevents rendering garbage
//! - **Genchi Genbutsu**: Test actual panel output, not mocked data

use super::state::{GpuTelemetry, SamplePeek, TrainingSnapshot};
use std::time::Duration;

/// Panel verification result
#[derive(Debug, Clone)]
pub struct PanelVerification {
    /// Panel name
    pub name: &'static str,
    /// Whether the panel can render
    pub can_render: bool,
    /// Passed assertions
    pub passed: Vec<&'static str>,
    /// Failed assertions with reasons
    pub failed: Vec<(&'static str, String)>,
    /// Verification duration
    pub duration: Duration,
}

impl PanelVerification {
    /// Create a new verification result
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            can_render: true,
            passed: Vec::new(),
            failed: Vec::new(),
            duration: Duration::ZERO,
        }
    }

    /// Check if all assertions passed
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.failed.is_empty() && self.can_render
    }

    /// Add a passed assertion
    pub fn pass(&mut self, assertion: &'static str) {
        self.passed.push(assertion);
    }

    /// Add a failed assertion
    pub fn fail(&mut self, assertion: &'static str, reason: impl Into<String>) {
        self.failed.push((assertion, reason.into()));
        self.can_render = false;
    }

    /// Score as percentage (0.0 - 1.0)
    #[must_use]
    pub fn score(&self) -> f32 {
        let total = self.passed.len() + self.failed.len();
        if total == 0 {
            1.0
        } else {
            self.passed.len() as f32 / total as f32
        }
    }
}

/// Panel trait following probar's Brick pattern
pub trait Panel {
    /// Panel name
    fn name(&self) -> &'static str;

    /// Check if panel can render (Jidoka gate)
    fn can_render(&self) -> bool;

    /// Verify panel data
    fn verify(&self) -> PanelVerification;

    /// Performance budget in milliseconds
    fn budget_ms(&self) -> u32 {
        16 // 60fps default
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOSS CURVE PANEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Loss curve panel data wrapper
pub struct LossCurvePanel<'a> {
    pub snapshot: &'a TrainingSnapshot,
}

impl Panel for LossCurvePanel<'_> {
    fn name(&self) -> &'static str {
        "LossCurve"
    }

    fn can_render(&self) -> bool {
        // Can always render, even with empty history (shows placeholder)
        true
    }

    fn verify(&self) -> PanelVerification {
        let start = std::time::Instant::now();
        let mut v = PanelVerification::new(self.name());

        // Loss must be finite
        if self.snapshot.loss.is_finite() {
            v.pass("loss_finite");
        } else {
            v.fail("loss_finite", format!("loss is {}", self.snapshot.loss));
        }

        // Loss history not too long (memory)
        if self.snapshot.loss_history.len() <= 1000 {
            v.pass("history_bounded");
        } else {
            v.fail(
                "history_bounded",
                format!("history len {} > 1000", self.snapshot.loss_history.len()),
            );
        }

        // Gradient norm finite
        if self.snapshot.gradient_norm.is_finite() {
            v.pass("grad_norm_finite");
        } else {
            v.fail("grad_norm_finite", format!("gradient_norm is {}", self.snapshot.gradient_norm));
        }

        v.duration = start.elapsed();
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU PANEL
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU panel data wrapper
pub struct GpuPanel<'a> {
    pub gpu: Option<&'a GpuTelemetry>,
}

impl Panel for GpuPanel<'_> {
    fn name(&self) -> &'static str {
        "Gpu"
    }

    fn can_render(&self) -> bool {
        self.gpu.is_some()
    }

    fn verify(&self) -> PanelVerification {
        let start = std::time::Instant::now();
        let mut v = PanelVerification::new(self.name());

        if let Some(gpu) = self.gpu {
            // Utilization in valid range
            if (0.0..=100.0).contains(&gpu.utilization_percent) {
                v.pass("util_range");
            } else {
                v.fail(
                    "util_range",
                    format!("utilization {}% out of range", gpu.utilization_percent),
                );
            }

            // VRAM used <= total
            if gpu.vram_used_gb <= gpu.vram_total_gb {
                v.pass("vram_valid");
            } else {
                v.fail(
                    "vram_valid",
                    format!("vram_used {}G > vram_total {}G", gpu.vram_used_gb, gpu.vram_total_gb),
                );
            }

            // Temperature reasonable
            if (0.0..=120.0).contains(&gpu.temperature_celsius) {
                v.pass("temp_range");
            } else {
                v.fail(
                    "temp_range",
                    format!("temperature {}°C out of range", gpu.temperature_celsius),
                );
            }

            // Device name not empty
            if gpu.device_name.is_empty() {
                v.fail("device_name", "device_name is empty");
            } else {
                v.pass("device_name");
            }
        } else {
            v.fail("gpu_present", "GPU telemetry is None");
        }

        v.duration = start.elapsed();
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROCESS PANEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Process panel data wrapper
pub struct ProcessPanel<'a> {
    pub gpu: Option<&'a GpuTelemetry>,
}

impl ProcessPanel<'_> {
    /// Find the training process from GPU processes
    pub fn training_process(&self) -> Option<&super::state::GpuProcessInfo> {
        self.gpu.as_ref().and_then(|g| {
            g.processes
                .iter()
                .find(|p| p.exe_path.contains("finetune") || p.exe_path.contains("entrenar"))
        })
    }
}

impl Panel for ProcessPanel<'_> {
    fn name(&self) -> &'static str {
        "Process"
    }

    fn can_render(&self) -> bool {
        self.training_process().is_some()
    }

    fn verify(&self) -> PanelVerification {
        let start = std::time::Instant::now();
        let mut v = PanelVerification::new(self.name());

        if let Some(gpu) = self.gpu {
            // Processes list present
            if gpu.processes.is_empty() {
                v.fail("processes_present", "GPU process list is empty");
            } else {
                v.pass("processes_present");
            }

            // Training process found
            if let Some(proc) = self.training_process() {
                v.pass("training_process_found");

                // Valid PID
                if proc.pid > 0 {
                    v.pass("valid_pid");
                } else {
                    v.fail("valid_pid", format!("invalid PID: {}", proc.pid));
                }

                // Exe path not empty
                if proc.exe_path.is_empty() {
                    v.fail("exe_path_present", "exe_path is empty");
                } else {
                    v.pass("exe_path_present");
                }
            } else {
                v.fail(
                    "training_process_found",
                    format!(
                        "no process matching 'finetune' or 'entrenar' in {} processes",
                        gpu.processes.len()
                    ),
                );
            }
        } else {
            v.fail("gpu_present", "GPU telemetry is None");
        }

        v.duration = start.elapsed();
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLE PANEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Sample preview panel data wrapper
pub struct SamplePanel<'a> {
    pub sample: Option<&'a SamplePeek>,
}

impl Panel for SamplePanel<'_> {
    fn name(&self) -> &'static str {
        "Sample"
    }

    fn can_render(&self) -> bool {
        self.sample.is_some()
    }

    fn verify(&self) -> PanelVerification {
        let start = std::time::Instant::now();
        let mut v = PanelVerification::new(self.name());

        if let Some(sample) = self.sample {
            // Input preview present
            if sample.input_preview.is_empty() {
                v.fail("input_present", "input_preview is empty");
            } else {
                v.pass("input_present");
            }

            // Target preview present
            if sample.target_preview.is_empty() {
                v.fail("target_present", "target_preview is empty");
            } else {
                v.pass("target_present");
            }

            // Token match in valid range
            if (0.0..=100.0).contains(&sample.token_match_percent) {
                v.pass("match_range");
            } else {
                v.fail(
                    "match_range",
                    format!("token_match {}% out of range", sample.token_match_percent),
                );
            }
        } else {
            v.fail("sample_present", "sample is None");
        }

        v.duration = start.elapsed();
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRAINING METRICS PANEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Training metrics panel data wrapper
pub struct MetricsPanel<'a> {
    pub snapshot: &'a TrainingSnapshot,
}

impl Panel for MetricsPanel<'_> {
    fn name(&self) -> &'static str {
        "Metrics"
    }

    fn can_render(&self) -> bool {
        true
    }

    fn verify(&self) -> PanelVerification {
        let start = std::time::Instant::now();
        let mut v = PanelVerification::new(self.name());

        // Epoch valid
        if self.snapshot.epoch <= self.snapshot.total_epochs || self.snapshot.total_epochs == 0 {
            v.pass("epoch_valid");
        } else {
            v.fail(
                "epoch_valid",
                format!(
                    "epoch {} > total_epochs {}",
                    self.snapshot.epoch, self.snapshot.total_epochs
                ),
            );
        }

        // Step valid
        if self.snapshot.step <= self.snapshot.steps_per_epoch || self.snapshot.steps_per_epoch == 0
        {
            v.pass("step_valid");
        } else {
            v.fail(
                "step_valid",
                format!(
                    "step {} > steps_per_epoch {}",
                    self.snapshot.step, self.snapshot.steps_per_epoch
                ),
            );
        }

        // Learning rate finite and positive
        if self.snapshot.learning_rate.is_finite() && self.snapshot.learning_rate >= 0.0 {
            v.pass("lr_valid");
        } else {
            v.fail("lr_valid", format!("learning_rate {} invalid", self.snapshot.learning_rate));
        }

        // Tokens per second non-negative
        if self.snapshot.tokens_per_second >= 0.0 {
            v.pass("throughput_valid");
        } else {
            v.fail(
                "throughput_valid",
                format!("tokens_per_second {} < 0", self.snapshot.tokens_per_second),
            );
        }

        v.duration = start.elapsed();
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL LAYOUT VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Verify all panels in a layout
pub fn verify_layout(snapshot: &TrainingSnapshot) -> Vec<PanelVerification> {
    vec![
        LossCurvePanel { snapshot }.verify(),
        GpuPanel { gpu: snapshot.gpu.as_ref() }.verify(),
        ProcessPanel { gpu: snapshot.gpu.as_ref() }.verify(),
        SamplePanel { sample: snapshot.sample.as_ref() }.verify(),
        MetricsPanel { snapshot }.verify(),
    ]
}

/// Check if layout can render (all critical panels pass Jidoka gate)
pub fn layout_can_render(snapshot: &TrainingSnapshot) -> bool {
    // LossCurve and Metrics are always renderable
    // GPU, Process, Sample are optional
    LossCurvePanel { snapshot }.can_render() && MetricsPanel { snapshot }.can_render()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::tui::state::{GpuProcessInfo, TrainingStatus};

    fn make_snapshot() -> TrainingSnapshot {
        TrainingSnapshot {
            timestamp_ms: 1000,
            epoch: 5,
            total_epochs: 10,
            step: 8,
            steps_per_epoch: 16,
            loss: 2.5,
            loss_history: vec![3.0, 2.8, 2.6, 2.5],
            learning_rate: 0.0001,
            gradient_norm: 1.5,
            tokens_per_second: 100.0,
            start_timestamp_ms: 0,
            gpu: Some(GpuTelemetry {
                device_name: "RTX 4090".into(),
                utilization_percent: 80.0,
                vram_used_gb: 4.0,
                vram_total_gb: 24.0,
                temperature_celsius: 65.0,
                power_watts: 300.0,
                power_limit_watts: 450.0,
                processes: vec![GpuProcessInfo {
                    pid: 1234,
                    exe_path: "/path/to/finetune_real".into(),
                    gpu_memory_mb: 2048,
                    cpu_percent: 50.0,
                    rss_mb: 1024,
                }],
            }),
            sample: Some(SamplePeek {
                input_preview: "fn is_prime(n: u64)".into(),
                target_preview: "#[test] fn test()".into(),
                generated_preview: "#[test] fn test()".into(),
                token_match_percent: 95.0,
            }),
            status: TrainingStatus::Running,
            experiment_id: "test".into(),
            model_name: "test-model".into(),
            lr_history: vec![0.0001; 4],
            model_path: "/models/test.safetensors".into(),
            optimizer_name: "AdamW".into(),
            batch_size: 4,
            checkpoint_path: "./checkpoints".into(),
            executable_path: "/path/to/finetune_real".into(),
            accuracy: 0.0,
            samples_per_second: 0.0,
        }
    }

    #[test]
    fn test_loss_curve_panel_valid() {
        let snapshot = make_snapshot();
        let panel = LossCurvePanel { snapshot: &snapshot };
        assert!(panel.can_render());
        let v = panel.verify();
        assert!(v.is_valid());
        assert_eq!(v.score(), 1.0);
    }

    #[test]
    fn test_loss_curve_panel_nan_loss() {
        let mut snapshot = make_snapshot();
        snapshot.loss = f32::NAN;
        let panel = LossCurvePanel { snapshot: &snapshot };
        let v = panel.verify();
        assert!(!v.is_valid());
        assert!(v.failed.iter().any(|(a, _)| *a == "loss_finite"));
    }

    #[test]
    fn test_gpu_panel_valid() {
        let snapshot = make_snapshot();
        let panel = GpuPanel { gpu: snapshot.gpu.as_ref() };
        assert!(panel.can_render());
        let v = panel.verify();
        assert!(v.is_valid());
    }

    #[test]
    fn test_gpu_panel_missing() {
        let panel = GpuPanel { gpu: None };
        assert!(!panel.can_render());
        let v = panel.verify();
        assert!(!v.is_valid());
    }

    #[test]
    fn test_process_panel_valid() {
        let snapshot = make_snapshot();
        let panel = ProcessPanel { gpu: snapshot.gpu.as_ref() };
        assert!(panel.can_render());
        assert!(panel.training_process().is_some());
        let v = panel.verify();
        assert!(v.is_valid());
    }

    #[test]
    fn test_process_panel_no_training_process() {
        let mut snapshot = make_snapshot();
        if let Some(ref mut gpu) = snapshot.gpu {
            gpu.processes[0].exe_path = "/usr/bin/other".into();
        }
        let panel = ProcessPanel { gpu: snapshot.gpu.as_ref() };
        assert!(!panel.can_render());
        let v = panel.verify();
        assert!(!v.is_valid());
        assert!(v.failed.iter().any(|(a, _)| *a == "training_process_found"));
    }

    #[test]
    fn test_sample_panel_valid() {
        let snapshot = make_snapshot();
        let panel = SamplePanel { sample: snapshot.sample.as_ref() };
        assert!(panel.can_render());
        let v = panel.verify();
        assert!(v.is_valid());
    }

    #[test]
    fn test_sample_panel_missing() {
        let panel = SamplePanel { sample: None };
        assert!(!panel.can_render());
        let v = panel.verify();
        assert!(!v.is_valid());
    }

    #[test]
    fn test_metrics_panel_valid() {
        let snapshot = make_snapshot();
        let panel = MetricsPanel { snapshot: &snapshot };
        assert!(panel.can_render());
        let v = panel.verify();
        assert!(v.is_valid());
    }

    #[test]
    fn test_metrics_panel_overflow_step() {
        let mut snapshot = make_snapshot();
        snapshot.step = 100;
        snapshot.steps_per_epoch = 16;
        let panel = MetricsPanel { snapshot: &snapshot };
        let v = panel.verify();
        assert!(!v.is_valid());
        assert!(v.failed.iter().any(|(a, _)| *a == "step_valid"));
    }

    #[test]
    fn test_verify_layout_complete() {
        let snapshot = make_snapshot();
        let verifications = verify_layout(&snapshot);
        assert_eq!(verifications.len(), 5);
        assert!(verifications.iter().all(super::PanelVerification::is_valid));
    }

    #[test]
    fn test_layout_can_render() {
        let snapshot = make_snapshot();
        assert!(layout_can_render(&snapshot));
    }
}
