//! Training State for IPC (SPEC-FT-001 Section 10.1)
//!
//! Atomic state updates written by the trainer, read by the TUI monitor.
//! Uses JSON file as the IPC mechanism for simplicity and portability.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Process using GPU resources
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuProcessInfo {
    /// Process ID
    pub pid: u32,
    /// Full path to executable
    pub exe_path: String,
    /// GPU memory used by this process in MB
    pub gpu_memory_mb: u64,
    /// CPU usage percentage (0-100)
    pub cpu_percent: f32,
    /// Resident set size (RSS) in MB
    pub rss_mb: u64,
}

/// GPU telemetry snapshot (NVML-inspired)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuTelemetry {
    /// GPU device name (e.g., "RTX 4090")
    pub device_name: String,
    /// GPU utilization percentage (0-100)
    pub utilization_percent: f32,
    /// VRAM used in GB
    pub vram_used_gb: f32,
    /// VRAM total in GB
    pub vram_total_gb: f32,
    /// Temperature in Celsius
    pub temperature_celsius: f32,
    /// Power draw in watts
    pub power_watts: f32,
    /// Power limit in watts
    pub power_limit_watts: f32,
    /// Processes using GPU
    #[serde(default)]
    pub processes: Vec<GpuProcessInfo>,
}

impl GpuTelemetry {
    /// VRAM utilization as percentage
    pub fn vram_percent(&self) -> f32 {
        if self.vram_total_gb > 0.0 {
            (self.vram_used_gb / self.vram_total_gb) * 100.0
        } else {
            0.0
        }
    }

    /// Check if thermal throttling is likely
    pub fn is_thermal_throttling(&self) -> bool {
        self.temperature_celsius > 83.0
    }

    /// Check if power limited
    pub fn is_power_limited(&self) -> bool {
        self.power_watts >= self.power_limit_watts * 0.95
    }
}

/// Sample peek for live decoding visualization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SamplePeek {
    /// Input function code (truncated for display)
    pub input_preview: String,
    /// Target test code (truncated for display)
    pub target_preview: String,
    /// Generated test code (truncated for display)
    pub generated_preview: String,
    /// Token match percentage (0-100)
    pub token_match_percent: f32,
}

/// Complete training state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSnapshot {
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Current epoch (1-indexed)
    pub epoch: usize,
    /// Total epochs
    pub total_epochs: usize,
    /// Current step within epoch
    pub step: usize,
    /// Total steps per epoch
    pub steps_per_epoch: usize,
    /// Current loss value
    pub loss: f32,
    /// Loss history (last N values for sparkline)
    pub loss_history: Vec<f32>,
    /// Current learning rate
    pub learning_rate: f32,
    /// Learning rate history (per-step, for epoch summaries)
    #[serde(default)]
    pub lr_history: Vec<f32>,
    /// Gradient norm
    pub gradient_norm: f32,
    /// Training accuracy (0.0 to 1.0)
    #[serde(default)]
    pub accuracy: f32,
    /// Throughput in tokens per second
    pub tokens_per_second: f32,
    /// Throughput in samples per second
    #[serde(default)]
    pub samples_per_second: f32,
    /// Training start timestamp (ms)
    pub start_timestamp_ms: u64,
    /// GPU telemetry (optional)
    pub gpu: Option<GpuTelemetry>,
    /// Sample peek (optional)
    pub sample: Option<SamplePeek>,
    /// Training status
    pub status: TrainingStatus,
    /// Experiment name/ID
    pub experiment_id: String,
    /// Model name
    pub model_name: String,
    /// Model path (e.g., path to .safetensors or .gguf)
    #[serde(default)]
    pub model_path: String,
    /// Optimizer name (e.g., "AdamW", "SGD")
    #[serde(default)]
    pub optimizer_name: String,
    /// Batch size
    #[serde(default)]
    pub batch_size: usize,
    /// Checkpoint path (where checkpoints are saved)
    #[serde(default)]
    pub checkpoint_path: String,
    /// Executable path (path to training binary)
    #[serde(default)]
    pub executable_path: String,
}

/// Training status enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingStatus {
    /// Training is initializing
    Initializing,
    /// Training is running
    Running,
    /// Training is paused
    Paused,
    /// Training completed successfully
    Completed,
    /// Training failed with error
    Failed(String),
}

impl Default for TrainingSnapshot {
    fn default() -> Self {
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);

        Self {
            timestamp_ms: now,
            epoch: 0,
            total_epochs: 0,
            step: 0,
            steps_per_epoch: 0,
            loss: 0.0,
            loss_history: Vec::new(),
            learning_rate: 0.0,
            lr_history: Vec::new(),
            gradient_norm: 0.0,
            accuracy: 0.0,
            tokens_per_second: 0.0,
            samples_per_second: 0.0,
            start_timestamp_ms: now,
            gpu: None,
            sample: None,
            status: TrainingStatus::Initializing,
            experiment_id: String::new(),
            model_name: String::new(),
            model_path: String::new(),
            optimizer_name: String::new(),
            batch_size: 0,
            checkpoint_path: String::new(),
            executable_path: String::new(),
        }
    }
}

impl TrainingSnapshot {
    /// Calculate elapsed time since training start
    /// Uses the snapshot's timestamp_ms for deterministic/reproducible output (ENT-140)
    pub fn elapsed(&self) -> Duration {
        Duration::from_millis(self.timestamp_ms.saturating_sub(self.start_timestamp_ms))
    }

    /// Calculate estimated remaining time
    pub fn estimated_remaining(&self) -> Option<Duration> {
        if self.tokens_per_second <= 0.0 {
            return None;
        }

        let total_steps = self.total_epochs * self.steps_per_epoch;
        let completed_steps = (self.epoch.saturating_sub(1)) * self.steps_per_epoch + self.step;

        if completed_steps == 0 || total_steps == 0 {
            return None;
        }

        let progress = completed_steps as f64 / total_steps as f64;
        if progress >= 1.0 {
            return Some(Duration::ZERO);
        }

        let elapsed_ms = self.timestamp_ms.saturating_sub(self.start_timestamp_ms);
        let total_estimated_ms = (elapsed_ms as f64 / progress) as u64;
        let remaining_ms = total_estimated_ms.saturating_sub(elapsed_ms);

        Some(Duration::from_millis(remaining_ms))
    }

    /// Global step (epoch * steps_per_epoch + step)
    pub fn global_step(&self) -> usize {
        (self.epoch.saturating_sub(1)) * self.steps_per_epoch + self.step
    }

    /// Progress percentage (0-100)
    pub fn progress_percent(&self) -> f32 {
        let total = self.total_epochs * self.steps_per_epoch;
        if total == 0 {
            return 0.0;
        }
        (self.global_step() as f32 / total as f32) * 100.0
    }

    /// Compute loss trend from recent history
    ///
    /// Returns:
    /// - `LossTrend::Decreasing` if loss is going down (good)
    /// - `LossTrend::Stable` if loss is plateauing
    /// - `LossTrend::Increasing` if loss is going up (bad)
    /// - `LossTrend::Unknown` if not enough data
    pub fn loss_trend(&self) -> LossTrend {
        // Need at least 5 samples to compute trend
        if self.loss_history.len() < 5 {
            return LossTrend::Unknown;
        }

        // Use last 10 samples (or all if less)
        let window = self.loss_history.len().min(10);
        let recent = &self.loss_history[self.loss_history.len() - window..];

        // Compare first half vs second half
        let mid = window / 2;
        let first_half: f32 = recent[..mid].iter().sum::<f32>() / mid as f32;
        let second_half: f32 = recent[mid..].iter().sum::<f32>() / (window - mid) as f32;

        // Calculate relative change
        let change = (second_half - first_half) / first_half.abs().max(1e-6);

        // Threshold: 2% change considered significant
        const THRESHOLD: f32 = 0.02;

        if change < -THRESHOLD {
            LossTrend::Decreasing
        } else if change > THRESHOLD {
            LossTrend::Increasing
        } else {
            LossTrend::Stable
        }
    }
}

/// Loss trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossTrend {
    /// Loss is decreasing (good)
    Decreasing,
    /// Loss is stable/plateauing
    Stable,
    /// Loss is increasing (bad)
    Increasing,
    /// Not enough data to determine
    Unknown,
}

impl LossTrend {
    /// Get Unicode arrow for display
    pub fn arrow(&self) -> &'static str {
        match self {
            LossTrend::Decreasing => "↓",
            LossTrend::Stable => "→",
            LossTrend::Increasing => "↑",
            LossTrend::Unknown => "?",
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            LossTrend::Decreasing => "decreasing",
            LossTrend::Stable => "stable",
            LossTrend::Increasing => "increasing",
            LossTrend::Unknown => "unknown",
        }
    }
}

/// Training state manager for IPC
///
/// Handles atomic read/write of training snapshots via JSON file.
pub struct TrainingState {
    /// Path to the state file
    state_path: std::path::PathBuf,
    /// Last read snapshot (cached)
    last_snapshot: Option<TrainingSnapshot>,
    /// Last modification time
    last_modified: Option<std::time::SystemTime>,
}

impl TrainingState {
    /// Create a new training state manager
    ///
    /// # Arguments
    ///
    /// * `experiment_dir` - Path to the experiment directory
    pub fn new<P: AsRef<Path>>(experiment_dir: P) -> Self {
        let state_path = experiment_dir.as_ref().join("training_state.json");
        Self { state_path, last_snapshot: None, last_modified: None }
    }

    /// Write a training snapshot atomically
    ///
    /// Uses staged write + rename for atomicity.
    pub fn write(&self, snapshot: &TrainingSnapshot) -> std::io::Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.state_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Stage writes via intermediate file for atomic persistence
        let temp_path = self.state_path.with_extension("json.tmp");
        let file = File::create(&temp_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Atomic rename
        fs::rename(&temp_path, &self.state_path)?;

        Ok(())
    }

    /// Read the current training snapshot
    ///
    /// Returns `None` if the state file doesn't exist yet.
    pub fn read(&mut self) -> std::io::Result<Option<TrainingSnapshot>> {
        if !self.state_path.exists() {
            return Ok(None);
        }

        // Check if file was modified since last read
        let metadata = fs::metadata(&self.state_path)?;
        let modified = metadata.modified()?;

        if self.last_modified == Some(modified) {
            // Return cached snapshot if file hasn't changed
            return Ok(self.last_snapshot.clone());
        }

        // Read and parse
        let file = File::open(&self.state_path)?;
        let reader = BufReader::new(file);
        let snapshot: TrainingSnapshot = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Update cache
        self.last_snapshot = Some(snapshot.clone());
        self.last_modified = Some(modified);

        Ok(Some(snapshot))
    }

    /// Check if training state file exists
    pub fn exists(&self) -> bool {
        self.state_path.exists()
    }

    /// Get the state file path
    pub fn path(&self) -> &Path {
        &self.state_path
    }

    /// Wait for the state file to appear (with timeout)
    pub fn wait_for_state(&mut self, timeout: Duration) -> std::io::Result<bool> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if self.exists() {
                return Ok(true);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::TempDir;

    #[test]
    fn test_training_snapshot_default() {
        let snapshot = TrainingSnapshot::default();
        assert_eq!(snapshot.epoch, 0);
        assert_eq!(snapshot.status, TrainingStatus::Initializing);
    }

    #[test]
    fn test_training_snapshot_progress() {
        let mut snapshot = TrainingSnapshot::default();
        snapshot.epoch = 2;
        snapshot.total_epochs = 10;
        snapshot.step = 50;
        snapshot.steps_per_epoch = 100;

        // Global step = (2-1) * 100 + 50 = 150
        assert_eq!(snapshot.global_step(), 150);

        // Progress = 150 / 1000 = 15%
        assert!((snapshot.progress_percent() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_telemetry_vram_percent() {
        let gpu = GpuTelemetry { vram_used_gb: 4.0, vram_total_gb: 24.0, ..Default::default() };
        assert!((gpu.vram_percent() - 16.67).abs() < 0.1);
    }

    #[test]
    fn test_training_state_write_read() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut state = TrainingState::new(temp_dir.path());

        let snapshot = TrainingSnapshot {
            epoch: 5,
            total_epochs: 10,
            loss: 0.42,
            status: TrainingStatus::Running,
            ..Default::default()
        };

        state.write(&snapshot).expect("file write should succeed");
        assert!(state.exists());

        let read_snapshot =
            state.read().expect("file read should succeed").expect("file read should succeed");
        assert_eq!(read_snapshot.epoch, 5);
        assert!((read_snapshot.loss - 0.42).abs() < 0.001);
    }

    #[test]
    fn test_training_state_caching() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut state = TrainingState::new(temp_dir.path());

        let snapshot = TrainingSnapshot { epoch: 1, ..Default::default() };

        state.write(&snapshot).expect("file write should succeed");

        // First read
        let _ = state.read().expect("file read should succeed");

        // Second read should return cached
        let cached =
            state.read().expect("file read should succeed").expect("file read should succeed");
        assert_eq!(cached.epoch, 1);
    }

    // Property-based tests for monitoring integration (ENT-121)

    proptest! {
        /// TrainingSnapshot JSON serialization round-trip
        #[test]
        fn prop_snapshot_json_roundtrip(
            epoch in 1usize..1000,
            total_epochs in 1usize..100,
            step in 0usize..10000,
            steps_per_epoch in 1usize..10000,
            loss in 0.0f32..100.0,
            learning_rate in 1e-10f32..1.0,
            gradient_norm in 0.0f32..1000.0,
            tokens_per_second in 0.0f32..10000.0,
        ) {
            let snapshot = TrainingSnapshot {
                timestamp_ms: 12345678,
                epoch,
                total_epochs,
                step,
                steps_per_epoch,
                loss,
                loss_history: vec![loss * 1.1, loss * 1.05, loss],
                learning_rate,
                lr_history: vec![learning_rate; 3],
                gradient_norm,
                accuracy: 0.0,
                tokens_per_second,
                samples_per_second: 0.0,
                start_timestamp_ms: 12345000,
                gpu: None,
                sample: None,
                status: TrainingStatus::Running,
                experiment_id: "test".to_string(),
                model_name: "model".to_string(),
                model_path: String::new(),
                optimizer_name: "AdamW".to_string(),
                batch_size: 4,
                checkpoint_path: String::new(),
                executable_path: String::new(),
            };

            // Serialize
            let json = serde_json::to_string(&snapshot).expect("JSON serialization should succeed");

            // Deserialize
            let restored: TrainingSnapshot = serde_json::from_str(&json).expect("JSON deserialization should succeed");

            // Verify all fields preserved
            prop_assert_eq!(restored.epoch, epoch);
            prop_assert_eq!(restored.total_epochs, total_epochs);
            prop_assert_eq!(restored.step, step);
            prop_assert_eq!(restored.steps_per_epoch, steps_per_epoch);
            prop_assert!((restored.loss - loss).abs() < 1e-5);
            prop_assert!((restored.learning_rate - learning_rate).abs() < 1e-10);
            prop_assert!((restored.gradient_norm - gradient_norm).abs() < 1e-5);
        }

        /// Loss trend detection is consistent with loss history direction
        #[test]
        fn prop_loss_trend_consistent(
            base_loss in 1.0f32..10.0,
            trend_factor in -0.1f32..0.1,
        ) {
            // Generate 10 loss values with consistent trend
            // Positive factor = loss going up, Negative factor = loss going down
            let history: Vec<f32> = (0..10)
                .map(|i| base_loss + (i as f32 * trend_factor))
                .collect();

            let snapshot = TrainingSnapshot {
                loss_history: history,
                ..Default::default()
            };

            let trend = snapshot.loss_trend();

            // With 10 samples and consistent trend, we should detect it
            // Positive factor = loss increasing over time = LossTrend::Increasing
            // Negative factor = loss decreasing over time = LossTrend::Decreasing
            if trend_factor > 0.05 {
                prop_assert_eq!(trend, LossTrend::Increasing);
            } else if trend_factor < -0.05 {
                prop_assert_eq!(trend, LossTrend::Decreasing);
            }
            // Stable if |trend_factor| <= 0.05 (within threshold)
        }

        /// GPU telemetry VRAM percentage is always 0-100
        #[test]
        fn prop_gpu_vram_percent_bounded(
            vram_used in 0.0f32..100.0,
            vram_total in 1.0f32..100.0,
        ) {
            let gpu = GpuTelemetry {
                vram_used_gb: vram_used.min(vram_total),
                vram_total_gb: vram_total,
                ..Default::default()
            };

            let percent = gpu.vram_percent();
            prop_assert!(percent >= 0.0);
            prop_assert!(percent <= 100.0);
        }

        /// Progress percent is always 0-100
        #[test]
        fn prop_progress_percent_bounded(
            epoch in 1usize..100,
            total_epochs in 1usize..100,
            step in 0usize..1000,
            steps_per_epoch in 1usize..1000,
        ) {
            let epoch = epoch.min(total_epochs);
            let step = step.min(steps_per_epoch);

            let snapshot = TrainingSnapshot {
                epoch,
                total_epochs,
                step,
                steps_per_epoch,
                ..Default::default()
            };

            let progress = snapshot.progress_percent();
            prop_assert!(progress >= 0.0);
            prop_assert!(progress <= 100.0);
        }

        /// State file write/read preserves all data
        #[test]
        fn prop_state_file_roundtrip(
            epoch in 1usize..100,
            loss in 0.0f32..100.0,
            lr in 1e-6f32..0.1,
        ) {
            let temp_dir = TempDir::new().expect("temp file creation should succeed");
            let mut state = TrainingState::new(temp_dir.path());

            let snapshot = TrainingSnapshot {
                epoch,
                total_epochs: 10,
                loss,
                learning_rate: lr,
                status: TrainingStatus::Running,
                ..Default::default()
            };

            state.write(&snapshot).expect("file write should succeed");

            // Clear cache and re-read
            state.last_modified = None;
            let restored = state.read().expect("file read should succeed").expect("file read should succeed");

            prop_assert_eq!(restored.epoch, epoch);
            prop_assert!((restored.loss - loss).abs() < 1e-5);
            prop_assert!((restored.learning_rate - lr).abs() < 1e-10);
        }
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_gpu_telemetry_vram_percent_zero_total() {
        let gpu = GpuTelemetry { vram_used_gb: 4.0, vram_total_gb: 0.0, ..Default::default() };
        assert!((gpu.vram_percent() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gpu_telemetry_thermal_throttling() {
        let gpu = GpuTelemetry { temperature_celsius: 84.0, ..Default::default() };
        assert!(gpu.is_thermal_throttling());

        let gpu2 = GpuTelemetry { temperature_celsius: 83.0, ..Default::default() };
        assert!(!gpu2.is_thermal_throttling());

        let gpu3 = GpuTelemetry { temperature_celsius: 70.0, ..Default::default() };
        assert!(!gpu3.is_thermal_throttling());
    }

    #[test]
    fn test_gpu_telemetry_power_limited() {
        let gpu =
            GpuTelemetry { power_watts: 380.0, power_limit_watts: 400.0, ..Default::default() };
        assert!(gpu.is_power_limited()); // 380 >= 400*0.95 = 380

        let gpu2 =
            GpuTelemetry { power_watts: 300.0, power_limit_watts: 400.0, ..Default::default() };
        assert!(!gpu2.is_power_limited());
    }

    #[test]
    fn test_training_snapshot_elapsed() {
        let snapshot =
            TrainingSnapshot { start_timestamp_ms: 1000, timestamp_ms: 6000, ..Default::default() };
        assert_eq!(snapshot.elapsed(), Duration::from_millis(5000));
    }

    #[test]
    fn test_training_snapshot_elapsed_same_time() {
        let snapshot =
            TrainingSnapshot { start_timestamp_ms: 5000, timestamp_ms: 5000, ..Default::default() };
        assert_eq!(snapshot.elapsed(), Duration::ZERO);
    }

    #[test]
    fn test_training_snapshot_estimated_remaining_none_zero_tps() {
        let snapshot = TrainingSnapshot { tokens_per_second: 0.0, ..Default::default() };
        assert!(snapshot.estimated_remaining().is_none());
    }

    #[test]
    fn test_training_snapshot_estimated_remaining_none_zero_steps() {
        let snapshot = TrainingSnapshot {
            tokens_per_second: 100.0,
            total_epochs: 0,
            steps_per_epoch: 0,
            ..Default::default()
        };
        assert!(snapshot.estimated_remaining().is_none());
    }

    #[test]
    fn test_training_snapshot_estimated_remaining_completed() {
        let snapshot = TrainingSnapshot {
            tokens_per_second: 100.0,
            epoch: 10,
            total_epochs: 10,
            step: 100,
            steps_per_epoch: 100,
            start_timestamp_ms: 1000,
            timestamp_ms: 11000,
            ..Default::default()
        };
        let remaining = snapshot.estimated_remaining();
        assert!(remaining.is_some());
        assert_eq!(remaining.unwrap(), Duration::ZERO);
    }

    #[test]
    fn test_training_snapshot_estimated_remaining_halfway() {
        let snapshot = TrainingSnapshot {
            tokens_per_second: 100.0,
            epoch: 5,
            total_epochs: 10,
            step: 50,
            steps_per_epoch: 100,
            start_timestamp_ms: 0,
            timestamp_ms: 10000,
            ..Default::default()
        };
        let remaining = snapshot.estimated_remaining();
        assert!(remaining.is_some());
        // 50% complete at 10s elapsed -> ~10s remaining
        let rem_ms = remaining.unwrap().as_millis();
        assert!(rem_ms > 5000 && rem_ms < 30000);
    }

    #[test]
    fn test_training_snapshot_global_step() {
        let snapshot =
            TrainingSnapshot { epoch: 3, steps_per_epoch: 100, step: 42, ..Default::default() };
        // global_step = (3-1)*100 + 42 = 242
        assert_eq!(snapshot.global_step(), 242);
    }

    #[test]
    fn test_training_snapshot_global_step_first_epoch() {
        let snapshot =
            TrainingSnapshot { epoch: 1, steps_per_epoch: 50, step: 10, ..Default::default() };
        // global_step = (1-1)*50 + 10 = 10
        assert_eq!(snapshot.global_step(), 10);
    }

    #[test]
    fn test_training_snapshot_progress_zero() {
        let snapshot =
            TrainingSnapshot { total_epochs: 0, steps_per_epoch: 0, ..Default::default() };
        assert!((snapshot.progress_percent() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_loss_trend_unknown_few_samples() {
        let snapshot = TrainingSnapshot { loss_history: vec![1.0, 2.0, 3.0], ..Default::default() };
        assert_eq!(snapshot.loss_trend(), LossTrend::Unknown);
    }

    #[test]
    fn test_loss_trend_decreasing() {
        let snapshot = TrainingSnapshot {
            loss_history: vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ..Default::default()
        };
        assert_eq!(snapshot.loss_trend(), LossTrend::Decreasing);
    }

    #[test]
    fn test_loss_trend_increasing() {
        let snapshot = TrainingSnapshot {
            loss_history: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ..Default::default()
        };
        assert_eq!(snapshot.loss_trend(), LossTrend::Increasing);
    }

    #[test]
    fn test_loss_trend_stable() {
        let snapshot = TrainingSnapshot {
            loss_history: vec![5.0, 5.01, 4.99, 5.0, 5.01, 4.99, 5.0, 5.01, 4.99, 5.0],
            ..Default::default()
        };
        assert_eq!(snapshot.loss_trend(), LossTrend::Stable);
    }

    #[test]
    fn test_loss_trend_arrow() {
        assert_eq!(LossTrend::Decreasing.arrow(), "\u{2193}");
        assert_eq!(LossTrend::Stable.arrow(), "\u{2192}");
        assert_eq!(LossTrend::Increasing.arrow(), "\u{2191}");
        assert_eq!(LossTrend::Unknown.arrow(), "?");
    }

    #[test]
    fn test_loss_trend_description() {
        assert_eq!(LossTrend::Decreasing.description(), "decreasing");
        assert_eq!(LossTrend::Stable.description(), "stable");
        assert_eq!(LossTrend::Increasing.description(), "increasing");
        assert_eq!(LossTrend::Unknown.description(), "unknown");
    }

    #[test]
    fn test_training_state_new_path() {
        let state = TrainingState::new("/tmp/test-exp");
        assert_eq!(state.path(), std::path::Path::new("/tmp/test-exp/training_state.json"));
    }

    #[test]
    fn test_training_state_exists_missing() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let state = TrainingState::new(temp_dir.path().join("nonexistent"));
        assert!(!state.exists());
    }

    #[test]
    fn test_training_state_read_missing_file() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut state = TrainingState::new(temp_dir.path().join("nonexistent"));
        let result = state.read().expect("should not error for missing file");
        assert!(result.is_none());
    }

    #[test]
    fn test_training_state_wait_for_state_already_exists() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut state = TrainingState::new(temp_dir.path());
        let snapshot = TrainingSnapshot::default();
        state.write(&snapshot).expect("write should succeed");

        let found = state.wait_for_state(Duration::from_millis(100)).expect("ok");
        assert!(found);
    }

    #[test]
    fn test_training_state_wait_for_state_timeout() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut state = TrainingState::new(temp_dir.path().join("never-exists"));
        let found = state.wait_for_state(Duration::from_millis(200)).expect("ok");
        assert!(!found);
    }

    #[test]
    fn test_gpu_process_info_default() {
        let info = GpuProcessInfo::default();
        assert_eq!(info.pid, 0);
        assert!(info.exe_path.is_empty());
        assert_eq!(info.gpu_memory_mb, 0);
    }

    #[test]
    fn test_sample_peek_default() {
        let sample = SamplePeek::default();
        assert!(sample.input_preview.is_empty());
        assert!(sample.target_preview.is_empty());
        assert!(sample.generated_preview.is_empty());
        assert!((sample.token_match_percent - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_training_status_equality() {
        assert_eq!(TrainingStatus::Running, TrainingStatus::Running);
        assert_eq!(TrainingStatus::Completed, TrainingStatus::Completed);
        assert_ne!(TrainingStatus::Running, TrainingStatus::Completed);
        assert_eq!(
            TrainingStatus::Failed("a".to_string()),
            TrainingStatus::Failed("a".to_string())
        );
        assert_ne!(
            TrainingStatus::Failed("a".to_string()),
            TrainingStatus::Failed("b".to_string())
        );
    }

    #[test]
    fn test_gpu_telemetry_serde_roundtrip() {
        let gpu = GpuTelemetry {
            device_name: "RTX 4090".to_string(),
            utilization_percent: 95.0,
            vram_used_gb: 20.0,
            vram_total_gb: 24.0,
            temperature_celsius: 72.0,
            power_watts: 350.0,
            power_limit_watts: 400.0,
            processes: vec![GpuProcessInfo {
                pid: 1234,
                exe_path: "/usr/bin/python3".to_string(),
                gpu_memory_mb: 19000,
                cpu_percent: 50.0,
                rss_mb: 4096,
            }],
        };
        let json = serde_json::to_string(&gpu).expect("serialize");
        let restored: GpuTelemetry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.device_name, "RTX 4090");
        assert_eq!(restored.processes.len(), 1);
        assert_eq!(restored.processes[0].pid, 1234);
    }
}
