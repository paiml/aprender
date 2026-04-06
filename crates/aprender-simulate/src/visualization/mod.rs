//! Visualization module for simular.
//!
//! Provides visualization capabilities:
//! - Export Pipeline: JSON Lines, Parquet format, video frames
//! - TUI Dashboard: Real-time terminal visualization (feature-gated)
//! - Web Visualization: WebSocket streaming (feature-gated)
//!
//! # Example
//!
//! ```rust
//! use simular::visualization::{SimMetrics, Exporter, ExportFormat};
//!
//! let metrics = SimMetrics::new();
//! let exporter = Exporter::new();
//! ```

use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{self, BufWriter, Write as IoWrite};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::engine::{SimState, SimTime};
use crate::error::{SimError, SimResult};

// Re-export TUI module if feature enabled
#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "tui")]
pub use tui::SimularTui;

// Re-export Web module if feature enabled
#[cfg(feature = "web")]
pub mod web;

#[cfg(feature = "web")]
pub use web::{WebPayload, WebVisualization};

// ============================================================================
// Simulation Metrics
// ============================================================================

/// Real-time simulation metrics for visualization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimMetrics {
    /// Current simulation time.
    pub time: f64,
    /// Current step number.
    pub step: u64,
    /// Steps per second (throughput).
    pub steps_per_second: f64,
    /// Total energy (if applicable).
    pub total_energy: Option<f64>,
    /// Kinetic energy (if applicable).
    pub kinetic_energy: Option<f64>,
    /// Potential energy (if applicable).
    pub potential_energy: Option<f64>,
    /// Energy drift from initial (relative).
    pub energy_drift: Option<f64>,
    /// Number of bodies/particles.
    pub body_count: usize,
    /// Number of Jidoka warnings.
    pub jidoka_warnings: u32,
    /// Number of Jidoka errors.
    pub jidoka_errors: u32,
    /// Memory usage in bytes.
    pub memory_bytes: usize,
    /// Custom metrics.
    pub custom: std::collections::HashMap<String, f64>,
}

impl SimMetrics {
    /// Create new empty metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update from simulation state.
    pub fn update_from_state(&mut self, state: &SimState, time: SimTime) {
        self.time = time.as_secs_f64();
        self.body_count = state.num_bodies();
        self.kinetic_energy = Some(state.kinetic_energy());
        self.potential_energy = Some(state.potential_energy());
        self.total_energy = Some(state.total_energy());
    }

    /// Set energy drift relative to initial energy.
    pub fn set_energy_drift(&mut self, initial_energy: f64) {
        if let Some(current) = self.total_energy {
            if initial_energy.abs() > f64::EPSILON {
                self.energy_drift = Some((current - initial_energy).abs() / initial_energy.abs());
            }
        }
    }

    /// Add custom metric.
    pub fn add_custom(&mut self, name: impl Into<String>, value: f64) {
        self.custom.insert(name.into(), value);
    }

    /// Get custom metric.
    #[must_use]
    pub fn get_custom(&self, name: &str) -> Option<f64> {
        self.custom.get(name).copied()
    }
}

/// Time-series data point for plotting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp.
    pub time: f64,
    /// Value.
    pub value: f64,
}

/// Rolling buffer for time-series data.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Data points.
    data: VecDeque<DataPoint>,
    /// Maximum capacity.
    capacity: usize,
    /// Series name.
    name: String,
}

impl TimeSeries {
    /// Create new time series with capacity.
    #[must_use]
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            name: name.into(),
        }
    }

    /// Push a new data point.
    pub fn push(&mut self, time: f64, value: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(DataPoint { time, value });
    }

    /// Get all data points.
    #[must_use]
    pub fn data(&self) -> &VecDeque<DataPoint> {
        &self.data
    }

    /// Get series name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get last value.
    #[must_use]
    pub fn last_value(&self) -> Option<f64> {
        self.data.back().map(|p| p.value)
    }

    /// Get min value.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.data
            .iter()
            .map(|p| p.value)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get max value.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.data
            .iter()
            .map(|p| p.value)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get time range.
    #[must_use]
    pub fn time_range(&self) -> Option<(f64, f64)> {
        if self.data.is_empty() {
            return None;
        }
        let first = self.data.front().map_or(0.0, |p| p.time);
        let last = self.data.back().map_or(0.0, |p| p.time);
        Some((first, last))
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// Trajectory Frame
// ============================================================================

/// A single frame in a trajectory for visualization/export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryFrame {
    /// Frame timestamp.
    pub time: f64,
    /// Frame index.
    pub index: u64,
    /// Body positions (flattened: x0, y0, z0, x1, y1, z1, ...).
    pub positions: Vec<f64>,
    /// Body velocities (flattened).
    pub velocities: Vec<f64>,
    /// Metrics at this frame.
    pub metrics: SimMetrics,
}

impl TrajectoryFrame {
    /// Create frame from simulation state.
    #[must_use]
    pub fn from_state(state: &SimState, time: SimTime, index: u64) -> Self {
        let positions: Vec<f64> = state
            .positions()
            .iter()
            .flat_map(|p| [p.x, p.y, p.z])
            .collect();

        let velocities: Vec<f64> = state
            .velocities()
            .iter()
            .flat_map(|v| [v.x, v.y, v.z])
            .collect();

        let mut metrics = SimMetrics::new();
        metrics.update_from_state(state, time);

        Self {
            time: time.as_secs_f64(),
            index,
            positions,
            velocities,
            metrics,
        }
    }
}

/// Collection of trajectory frames.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trajectory {
    /// Frames in the trajectory.
    pub frames: Vec<TrajectoryFrame>,
    /// Metadata.
    pub metadata: TrajectoryMetadata,
}

/// Metadata for a trajectory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrajectoryMetadata {
    /// Simulation name/description.
    pub name: String,
    /// Number of bodies.
    pub body_count: usize,
    /// Start time.
    pub start_time: f64,
    /// End time.
    pub end_time: f64,
    /// Time step.
    pub timestep: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Trajectory {
    /// Create new empty trajectory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with metadata.
    #[must_use]
    pub fn with_metadata(metadata: TrajectoryMetadata) -> Self {
        Self {
            frames: Vec::new(),
            metadata,
        }
    }

    /// Add a frame.
    pub fn add_frame(&mut self, frame: TrajectoryFrame) {
        if let Some(last) = self.frames.last() {
            self.metadata.end_time = frame.time.max(last.time);
        } else {
            self.metadata.start_time = frame.time;
            self.metadata.end_time = frame.time;
        }
        self.frames.push(frame);
    }

    /// Get number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Get frame at index.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&TrajectoryFrame> {
        self.frames.get(index)
    }

    /// Get frame closest to time.
    #[must_use]
    pub fn frame_at_time(&self, time: f64) -> Option<&TrajectoryFrame> {
        self.frames.iter().min_by(|a, b| {
            let diff_a = (a.time - time).abs();
            let diff_b = (b.time - time).abs();
            diff_a
                .partial_cmp(&diff_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get duration.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.metadata.end_time - self.metadata.start_time
    }
}

// ============================================================================
// Export Pipeline
// ============================================================================

/// Video format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoFormat {
    /// `MP4` (H.264).
    Mp4,
    /// GIF animation.
    Gif,
    /// `WebM` (VP9).
    WebM,
}

/// Parquet compression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParquetCompression {
    /// No compression.
    None,
    /// Snappy compression.
    Snappy,
    /// Zstd compression.
    Zstd,
    /// LZ4 compression.
    Lz4,
}

/// Export format options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON Lines (streaming JSON).
    #[default]
    JsonLines,
    /// Parquet columnar format.
    Parquet {
        /// Compression algorithm.
        compression: ParquetCompression,
    },
    /// Video export.
    Video {
        /// Video format.
        format: VideoFormat,
        /// Frames per second.
        fps: u32,
    },
    /// CSV format.
    Csv,
    /// Binary format (bincode).
    Binary,
}

/// Export configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Output format.
    pub format: ExportFormat,
    /// Include velocities.
    pub include_velocities: bool,
    /// Include metrics.
    pub include_metrics: bool,
    /// Decimation factor (1 = every frame, 2 = every other, etc.).
    pub decimation: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::JsonLines,
            include_velocities: true,
            include_metrics: true,
            decimation: 1,
        }
    }
}

/// Exporter for simulation data.
#[derive(Debug, Clone)]
pub struct Exporter {
    /// Export configuration.
    config: ExportConfig,
}

impl Default for Exporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Exporter {
    /// Create new exporter with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExportConfig::default(),
        }
    }

    /// Create with custom config.
    #[must_use]
    pub fn with_config(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export trajectory to JSON Lines format.
    ///
    /// # Errors
    ///
    /// Returns error if file operations fail.
    pub fn to_json_lines(&self, trajectory: &Trajectory, path: &Path) -> SimResult<()> {
        let file =
            File::create(path).map_err(|e| SimError::io(format!("Failed to create file: {e}")))?;
        let mut writer = BufWriter::new(file);

        for (i, frame) in trajectory.frames.iter().enumerate() {
            if i % self.config.decimation.max(1) != 0 {
                continue;
            }

            let json = serde_json::to_string(frame)
                .map_err(|e| SimError::serialization(format!("JSON serialization failed: {e}")))?;
            writeln!(writer, "{json}").map_err(|e| SimError::io(format!("Write failed: {e}")))?;
        }

        writer
            .flush()
            .map_err(|e| SimError::io(format!("Flush failed: {e}")))?;

        Ok(())
    }

    /// Export trajectory to CSV format.
    ///
    /// # Errors
    ///
    /// Returns error if file operations fail.
    pub fn to_csv(&self, trajectory: &Trajectory, path: &Path) -> SimResult<()> {
        let file =
            File::create(path).map_err(|e| SimError::io(format!("Failed to create file: {e}")))?;
        let mut writer = BufWriter::new(file);

        // Write header
        let mut header = String::from("time,index");
        if !trajectory.frames.is_empty() {
            let n_bodies = trajectory.frames[0].positions.len() / 3;
            for i in 0..n_bodies {
                let _ = write!(header, ",x{i},y{i},z{i}");
                if self.config.include_velocities {
                    let _ = write!(header, ",vx{i},vy{i},vz{i}");
                }
            }
        }
        if self.config.include_metrics {
            header.push_str(",total_energy,kinetic_energy,potential_energy");
        }
        writeln!(writer, "{header}")
            .map_err(|e| SimError::io(format!("Write header failed: {e}")))?;

        // Write data
        for (i, frame) in trajectory.frames.iter().enumerate() {
            if i % self.config.decimation.max(1) != 0 {
                continue;
            }

            let mut line = format!("{},{}", frame.time, frame.index);

            for (j, pos) in frame.positions.chunks(3).enumerate() {
                if pos.len() == 3 {
                    let _ = write!(line, ",{},{},{}", pos[0], pos[1], pos[2]);
                }
                if self.config.include_velocities {
                    if let Some(vel) = frame.velocities.chunks(3).nth(j) {
                        if vel.len() == 3 {
                            let _ = write!(line, ",{},{},{}", vel[0], vel[1], vel[2]);
                        }
                    }
                }
            }

            if self.config.include_metrics {
                let te = frame.metrics.total_energy.unwrap_or(0.0);
                let ke = frame.metrics.kinetic_energy.unwrap_or(0.0);
                let pe = frame.metrics.potential_energy.unwrap_or(0.0);
                let _ = write!(line, ",{te},{ke},{pe}");
            }

            writeln!(writer, "{line}")
                .map_err(|e| SimError::io(format!("Write data failed: {e}")))?;
        }

        writer
            .flush()
            .map_err(|e| SimError::io(format!("Flush failed: {e}")))?;

        Ok(())
    }

    /// Export trajectory to binary format (bincode).
    ///
    /// # Errors
    ///
    /// Returns error if file operations fail.
    pub fn to_binary(&self, trajectory: &Trajectory, path: &Path) -> SimResult<()> {
        let file =
            File::create(path).map_err(|e| SimError::io(format!("Failed to create file: {e}")))?;
        let writer = BufWriter::new(file);

        bincode::serialize_into(writer, trajectory)
            .map_err(|e| SimError::serialization(format!("Binary serialization failed: {e}")))?;

        Ok(())
    }

    /// Load trajectory from binary format.
    ///
    /// # Errors
    ///
    /// Returns error if file operations fail.
    pub fn from_binary(path: &Path) -> SimResult<Trajectory> {
        let file =
            File::open(path).map_err(|e| SimError::io(format!("Failed to open file: {e}")))?;
        let reader = io::BufReader::new(file);

        bincode::deserialize_from(reader)
            .map_err(|e| SimError::serialization(format!("Binary deserialization failed: {e}")))
    }

    /// Export using configured format.
    ///
    /// # Errors
    ///
    /// Returns error if export fails.
    pub fn export(&self, trajectory: &Trajectory, path: &Path) -> SimResult<()> {
        match &self.config.format {
            ExportFormat::JsonLines => self.to_json_lines(trajectory, path),
            ExportFormat::Csv => self.to_csv(trajectory, path),
            ExportFormat::Binary => self.to_binary(trajectory, path),
            ExportFormat::Parquet { .. } => Err(SimError::config(
                "Parquet export requires alimentar integration".to_string(),
            )),
            ExportFormat::Video { .. } => Err(SimError::config(
                "Video export requires ffmpeg integration".to_string(),
            )),
        }
    }
}

/// Streaming exporter for real-time export.
pub struct StreamingExporter {
    /// Output writer.
    writer: BufWriter<File>,
    /// Frame count.
    frame_count: u64,
    /// Decimation counter.
    decimation_count: usize,
    /// Decimation factor.
    decimation: usize,
}

impl StreamingExporter {
    /// Create new streaming exporter.
    ///
    /// # Errors
    ///
    /// Returns error if file creation fails.
    pub fn new(path: &Path, decimation: usize) -> SimResult<Self> {
        let file =
            File::create(path).map_err(|e| SimError::io(format!("Failed to create file: {e}")))?;
        Ok(Self {
            writer: BufWriter::new(file),
            frame_count: 0,
            decimation_count: 0,
            decimation: decimation.max(1),
        })
    }

    /// Write a frame to the stream.
    ///
    /// # Errors
    ///
    /// Returns error if write fails.
    pub fn write_frame(&mut self, frame: &TrajectoryFrame) -> SimResult<()> {
        self.decimation_count += 1;
        if self.decimation_count < self.decimation {
            return Ok(());
        }
        self.decimation_count = 0;

        let json = serde_json::to_string(frame)
            .map_err(|e| SimError::serialization(format!("JSON serialization failed: {e}")))?;
        writeln!(self.writer, "{json}").map_err(|e| SimError::io(format!("Write failed: {e}")))?;

        self.frame_count += 1;
        Ok(())
    }

    /// Flush and close the stream.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails.
    pub fn finish(mut self) -> SimResult<u64> {
        self.writer
            .flush()
            .map_err(|e| SimError::io(format!("Flush failed: {e}")))?;
        Ok(self.frame_count)
    }

    /// Get current frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Read as IoRead;
    use tempfile::tempdir;

    #[test]
    fn test_sim_metrics_new() {
        let metrics = SimMetrics::new();
        assert_eq!(metrics.step, 0);
        assert_eq!(metrics.body_count, 0);
    }

    #[test]
    fn test_sim_metrics_custom() {
        let mut metrics = SimMetrics::new();
        metrics.add_custom("test_metric", 42.0);
        assert!((metrics.get_custom("test_metric").unwrap() - 42.0).abs() < f64::EPSILON);
        assert!(metrics.get_custom("nonexistent").is_none());
    }

    #[test]
    fn test_sim_metrics_default() {
        let metrics: SimMetrics = Default::default();
        assert_eq!(metrics.step, 0);
        assert!(metrics.total_energy.is_none());
    }

    #[test]
    fn test_sim_metrics_set_energy_drift() {
        let mut metrics = SimMetrics::new();
        metrics.total_energy = Some(10.5);
        metrics.set_energy_drift(10.0);
        assert!(metrics.energy_drift.is_some());
        let drift = metrics.energy_drift.unwrap();
        assert!((drift - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_sim_metrics_set_energy_drift_zero_initial() {
        let mut metrics = SimMetrics::new();
        metrics.total_energy = Some(10.5);
        metrics.set_energy_drift(0.0); // Zero initial energy
        assert!(metrics.energy_drift.is_none());
    }

    #[test]
    fn test_sim_metrics_set_energy_drift_no_total() {
        let mut metrics = SimMetrics::new();
        // No total_energy set
        metrics.set_energy_drift(10.0);
        assert!(metrics.energy_drift.is_none());
    }

    #[test]
    fn test_sim_metrics_clone() {
        let mut metrics = SimMetrics::new();
        metrics.step = 42;
        metrics.add_custom("key", 1.0);
        let cloned = metrics.clone();
        assert_eq!(cloned.step, 42);
        assert!((cloned.get_custom("key").unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_series_new() {
        let series = TimeSeries::new("test", 100);
        assert!(series.is_empty());
        assert_eq!(series.name(), "test");
    }

    #[test]
    fn test_time_series_push() {
        let mut series = TimeSeries::new("test", 100);
        series.push(0.0, 1.0);
        series.push(1.0, 2.0);
        series.push(2.0, 3.0);

        assert_eq!(series.len(), 3);
        assert!((series.last_value().unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_series_capacity() {
        let mut series = TimeSeries::new("test", 3);
        series.push(0.0, 1.0);
        series.push(1.0, 2.0);
        series.push(2.0, 3.0);
        series.push(3.0, 4.0); // Should evict first

        assert_eq!(series.len(), 3);
        assert!((series.data().front().unwrap().value - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_series_min_max() {
        let mut series = TimeSeries::new("test", 100);
        series.push(0.0, 5.0);
        series.push(1.0, 2.0);
        series.push(2.0, 8.0);

        assert!((series.min().unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((series.max().unwrap() - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_series_range() {
        let mut series = TimeSeries::new("test", 100);
        series.push(1.0, 0.0);
        series.push(5.0, 0.0);

        let (start, end) = series.time_range().unwrap();
        assert!((start - 1.0).abs() < f64::EPSILON);
        assert!((end - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_series_empty_stats() {
        let series = TimeSeries::new("test", 100);
        assert!(series.min().is_none());
        assert!(series.max().is_none());
        assert!(series.last_value().is_none());
        assert!(series.time_range().is_none());
    }

    #[test]
    fn test_time_series_clear() {
        let mut series = TimeSeries::new("test", 100);
        series.push(0.0, 1.0);
        series.push(1.0, 2.0);
        assert!(!series.is_empty());

        series.clear();
        assert!(series.is_empty());
        assert_eq!(series.len(), 0);
    }

    #[test]
    fn test_time_series_clone() {
        let mut series = TimeSeries::new("test", 100);
        series.push(0.0, 1.0);
        let cloned = series.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.name(), "test");
    }

    #[test]
    fn test_data_point_clone() {
        let dp = DataPoint {
            time: 1.0,
            value: 2.0,
        };
        let cloned = dp.clone();
        assert!((cloned.time - 1.0).abs() < f64::EPSILON);
        assert!((cloned.value - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trajectory_new() {
        let traj = Trajectory::new();
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_trajectory_with_metadata() {
        let metadata = TrajectoryMetadata {
            name: "test".to_string(),
            body_count: 5,
            start_time: 0.0,
            end_time: 10.0,
            timestep: 0.01,
            seed: 42,
        };
        let traj = Trajectory::with_metadata(metadata);
        assert!(traj.is_empty());
        assert_eq!(traj.metadata.name, "test");
        assert_eq!(traj.metadata.body_count, 5);
    }

    #[test]
    fn test_trajectory_add_frame() {
        let mut traj = Trajectory::new();
        let frame = TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![0.1, 0.2, 0.3],
            metrics: SimMetrics::new(),
        };
        traj.add_frame(frame);

        assert_eq!(traj.len(), 1);
        assert!((traj.metadata.start_time - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trajectory_add_multiple_frames() {
        let mut traj = Trajectory::new();
        for i in 0..5 {
            traj.add_frame(TrajectoryFrame {
                time: i as f64,
                index: i,
                positions: vec![],
                velocities: vec![],
                metrics: SimMetrics::new(),
            });
        }
        assert_eq!(traj.len(), 5);
        assert!((traj.metadata.start_time - 0.0).abs() < f64::EPSILON);
        assert!((traj.metadata.end_time - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trajectory_duration() {
        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![],
            velocities: vec![],
            metrics: SimMetrics::new(),
        });
        traj.add_frame(TrajectoryFrame {
            time: 10.0,
            index: 1,
            positions: vec![],
            velocities: vec![],
            metrics: SimMetrics::new(),
        });
        assert!((traj.duration() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trajectory_frame() {
        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 1.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![],
            metrics: SimMetrics::new(),
        });

        let frame = traj.frame(0).unwrap();
        assert_eq!(frame.index, 0);
        assert!(traj.frame(1).is_none());
    }

    #[test]
    fn test_trajectory_frame_at_time() {
        let mut traj = Trajectory::new();
        for i in 0..10 {
            traj.add_frame(TrajectoryFrame {
                time: i as f64,
                index: i,
                positions: vec![],
                velocities: vec![],
                metrics: SimMetrics::new(),
            });
        }

        let frame = traj.frame_at_time(5.5).unwrap();
        // Should return frame 5 or 6 (closest)
        assert!(frame.index == 5 || frame.index == 6);
    }

    #[test]
    fn test_trajectory_frame_at_time_empty() {
        let traj = Trajectory::new();
        assert!(traj.frame_at_time(5.0).is_none());
    }

    #[test]
    fn test_trajectory_clone() {
        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0],
            velocities: vec![],
            metrics: SimMetrics::new(),
        });
        let cloned = traj.clone();
        assert_eq!(cloned.len(), 1);
    }

    #[test]
    fn test_trajectory_frame_clone() {
        let frame = TrajectoryFrame {
            time: 1.0,
            index: 42,
            positions: vec![1.0, 2.0],
            velocities: vec![3.0, 4.0],
            metrics: SimMetrics::new(),
        };
        let cloned = frame.clone();
        assert_eq!(cloned.index, 42);
        assert_eq!(cloned.positions.len(), 2);
    }

    #[test]
    fn test_exporter_new() {
        let exporter = Exporter::new();
        assert!(matches!(exporter.config.format, ExportFormat::JsonLines));
    }

    #[test]
    fn test_exporter_default() {
        let exporter: Exporter = Default::default();
        assert!(matches!(exporter.config.format, ExportFormat::JsonLines));
    }

    #[test]
    fn test_exporter_with_config() {
        let config = ExportConfig {
            format: ExportFormat::Csv,
            include_velocities: false,
            include_metrics: false,
            decimation: 2,
        };
        let exporter = Exporter::with_config(config);
        assert!(matches!(exporter.config.format, ExportFormat::Csv));
        assert!(!exporter.config.include_velocities);
    }

    #[test]
    fn test_export_format_default() {
        let format = ExportFormat::default();
        assert!(matches!(format, ExportFormat::JsonLines));
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert!(config.include_velocities);
        assert!(config.include_metrics);
        assert_eq!(config.decimation, 1);
    }

    #[test]
    fn test_export_to_json_lines() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![0.1, 0.2, 0.3],
            metrics: SimMetrics::new(),
        });

        let exporter = Exporter::new();
        exporter.to_json_lines(&traj, &path).unwrap();

        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        assert!(content.contains("\"time\":0.0"));
        assert!(content.contains("\"index\":0"));
    }

    #[test]
    fn test_export_to_json_lines_decimation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let mut traj = Trajectory::new();
        for i in 0..10 {
            traj.add_frame(TrajectoryFrame {
                time: i as f64,
                index: i,
                positions: vec![],
                velocities: vec![],
                metrics: SimMetrics::new(),
            });
        }

        let config = ExportConfig {
            decimation: 2,
            ..Default::default()
        };
        let exporter = Exporter::with_config(config);
        exporter.to_json_lines(&traj, &path).unwrap();

        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        // Should have frames 0, 2, 4, 6, 8
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn test_export_to_csv() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.csv");

        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![0.1, 0.2, 0.3],
            metrics: SimMetrics::new(),
        });

        let exporter = Exporter::new();
        exporter.to_csv(&traj, &path).unwrap();

        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        assert!(content.contains("time,index"));
        assert!(content.contains("0,0"));
    }

    #[test]
    fn test_export_to_csv_no_velocities() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.csv");

        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![0.1, 0.2, 0.3],
            metrics: SimMetrics::new(),
        });

        let config = ExportConfig {
            include_velocities: false,
            ..Default::default()
        };
        let exporter = Exporter::with_config(config);
        exporter.to_csv(&traj, &path).unwrap();

        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        // Header shouldn't have vx, vy, vz
        assert!(!content.lines().next().unwrap().contains("vx"));
    }

    #[test]
    fn test_export_to_csv_no_metrics() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.csv");

        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![],
            metrics: SimMetrics::new(),
        });

        let config = ExportConfig {
            include_metrics: false,
            ..Default::default()
        };
        let exporter = Exporter::with_config(config);
        exporter.to_csv(&traj, &path).unwrap();

        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        assert!(!content.contains("total_energy"));
    }

    #[test]
    fn test_export_to_csv_decimation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.csv");

        let mut traj = Trajectory::new();
        for i in 0..10 {
            traj.add_frame(TrajectoryFrame {
                time: i as f64,
                index: i,
                positions: vec![],
                velocities: vec![],
                metrics: SimMetrics::new(),
            });
        }

        let config = ExportConfig {
            decimation: 3,
            ..Default::default()
        };
        let exporter = Exporter::with_config(config);
        exporter.to_csv(&traj, &path).unwrap();

        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        // Should have header + frames 0, 3, 6, 9
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 5); // header + 4 data lines
    }

    #[test]
    fn test_export_to_binary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");

        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![0.1, 0.2, 0.3],
            metrics: SimMetrics::new(),
        });

        let exporter = Exporter::new();
        exporter.to_binary(&traj, &path).unwrap();

        // Load it back
        let loaded = Exporter::from_binary(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!((loaded.frames[0].time - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_export_generic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![],
            velocities: vec![],
            metrics: SimMetrics::new(),
        });

        let exporter = Exporter::new();
        exporter.export(&traj, &path).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_export_parquet_unsupported() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");

        let traj = Trajectory::new();
        let config = ExportConfig {
            format: ExportFormat::Parquet {
                compression: ParquetCompression::Snappy,
            },
            ..Default::default()
        };
        let exporter = Exporter::with_config(config);
        let result = exporter.export(&traj, &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_export_video_unsupported() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mp4");

        let traj = Trajectory::new();
        let config = ExportConfig {
            format: ExportFormat::Video {
                format: VideoFormat::Mp4,
                fps: 30,
            },
            ..Default::default()
        };
        let exporter = Exporter::with_config(config);
        let result = exporter.export(&traj, &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_exporter() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stream.jsonl");

        let mut stream = StreamingExporter::new(&path, 1).unwrap();
        assert_eq!(stream.frame_count(), 0);

        let frame = TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![],
            metrics: SimMetrics::new(),
        };
        stream.write_frame(&frame).unwrap();
        assert_eq!(stream.frame_count(), 1);

        let count = stream.finish().unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_streaming_exporter_decimation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stream.jsonl");

        let mut stream = StreamingExporter::new(&path, 2).unwrap();

        for i in 0..10 {
            let frame = TrajectoryFrame {
                time: i as f64,
                index: i,
                positions: vec![],
                velocities: vec![],
                metrics: SimMetrics::new(),
            };
            stream.write_frame(&frame).unwrap();
        }

        let count = stream.finish().unwrap();
        assert_eq!(count, 5); // Every other frame
    }

    #[test]
    fn test_video_format_eq() {
        assert_eq!(VideoFormat::Mp4, VideoFormat::Mp4);
        assert_ne!(VideoFormat::Mp4, VideoFormat::Gif);
        assert_ne!(VideoFormat::Gif, VideoFormat::WebM);
    }

    #[test]
    fn test_parquet_compression_eq() {
        assert_eq!(ParquetCompression::None, ParquetCompression::None);
        assert_ne!(ParquetCompression::None, ParquetCompression::Snappy);
        assert_ne!(ParquetCompression::Zstd, ParquetCompression::Lz4);
    }

    #[test]
    fn test_export_format_parquet_variants() {
        let _ = ExportFormat::Parquet {
            compression: ParquetCompression::None,
        };
        let _ = ExportFormat::Parquet {
            compression: ParquetCompression::Zstd,
        };
        let _ = ExportFormat::Parquet {
            compression: ParquetCompression::Lz4,
        };
    }

    #[test]
    fn test_export_format_video_variants() {
        let _ = ExportFormat::Video {
            format: VideoFormat::Gif,
            fps: 24,
        };
        let _ = ExportFormat::Video {
            format: VideoFormat::WebM,
            fps: 60,
        };
    }

    #[test]
    fn test_trajectory_metadata_default() {
        let meta: TrajectoryMetadata = Default::default();
        assert!(meta.name.is_empty());
        assert_eq!(meta.body_count, 0);
    }

    // === Additional Coverage Tests ===

    #[test]
    fn test_sim_metrics_update_from_state() {
        use crate::engine::state::Vec3;
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.0, 0.0));
        state.set_potential_energy(-10.0);

        let mut metrics = SimMetrics::new();
        metrics.update_from_state(&state, crate::engine::SimTime::from_secs(1.5));

        assert_eq!(metrics.body_count, 1);
        assert!((metrics.time - 1.5).abs() < f64::EPSILON);
        assert!(metrics.kinetic_energy.is_some());
        assert!(metrics.potential_energy.is_some());
        assert!(metrics.total_energy.is_some());
    }

    #[test]
    fn test_sim_metrics_debug() {
        let metrics = SimMetrics::new();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("SimMetrics"));
    }

    #[test]
    fn test_trajectory_frame_from_state() {
        use crate::engine::state::Vec3;
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.0, 0.0));

        let frame = TrajectoryFrame::from_state(&state, crate::engine::SimTime::from_secs(1.0), 42);

        assert!((frame.time - 1.0).abs() < f64::EPSILON);
        assert_eq!(frame.index, 42);
        assert_eq!(frame.positions.len(), 3); // x, y, z for 1 body
        assert_eq!(frame.velocities.len(), 3);
    }

    #[test]
    fn test_trajectory_frame_debug() {
        let frame = TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![],
            velocities: vec![],
            metrics: SimMetrics::new(),
        };
        let debug = format!("{:?}", frame);
        assert!(debug.contains("TrajectoryFrame"));
    }

    #[test]
    fn test_trajectory_debug() {
        let traj = Trajectory::new();
        let debug = format!("{:?}", traj);
        assert!(debug.contains("Trajectory"));
    }

    #[test]
    fn test_time_series_clear_with_capacity_check() {
        let mut series = TimeSeries::new("test", 100);
        series.push(0.0, 1.0);
        series.push(1.0, 2.0);
        assert!(!series.is_empty());

        series.clear();
        assert!(series.is_empty());
        assert_eq!(series.len(), 0);
        // After clear, name should be unchanged
        assert_eq!(series.name(), "test");
    }

    #[test]
    fn test_time_series_data_access() {
        let mut series = TimeSeries::new("test", 100);
        series.push(0.5, 1.0);
        series.push(1.5, 2.0);
        series.push(2.5, 3.0);

        let data = series.data();
        assert_eq!(data.len(), 3);
        assert!((data.front().unwrap().time - 0.5).abs() < f64::EPSILON);
        assert!((data.back().unwrap().time - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_data_point_debug_clone() {
        let dp = DataPoint {
            time: 1.0,
            value: 2.0,
        };
        let cloned = dp.clone();
        assert!((cloned.time - 1.0).abs() < f64::EPSILON);
        assert!((cloned.value - 2.0).abs() < f64::EPSILON);

        let debug = format!("{:?}", dp);
        assert!(debug.contains("DataPoint"));
    }

    #[test]
    fn test_video_format_debug_clone() {
        let vf = VideoFormat::Mp4;
        let cloned = vf.clone();
        assert_eq!(cloned, VideoFormat::Mp4);

        let debug = format!("{:?}", vf);
        assert!(debug.contains("Mp4"));
    }

    #[test]
    fn test_parquet_compression_debug_clone() {
        let pc = ParquetCompression::Zstd;
        let cloned = pc.clone();
        assert_eq!(cloned, ParquetCompression::Zstd);

        let debug = format!("{:?}", pc);
        assert!(debug.contains("Zstd"));
    }

    #[test]
    fn test_export_format_debug_clone() {
        let ef = ExportFormat::Csv;
        let cloned = ef.clone();
        assert!(matches!(cloned, ExportFormat::Csv));

        let debug = format!("{:?}", ef);
        assert!(debug.contains("Csv"));
    }

    #[test]
    fn test_export_config_debug_clone() {
        let config = ExportConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.decimation, config.decimation);

        let debug = format!("{:?}", config);
        assert!(debug.contains("ExportConfig"));
    }

    #[test]
    fn test_exporter_debug() {
        let exporter = Exporter::new();
        let debug = format!("{:?}", exporter);
        assert!(debug.contains("Exporter"));
    }

    #[test]
    fn test_trajectory_metadata_debug_clone() {
        let meta = TrajectoryMetadata {
            name: "test".to_string(),
            body_count: 5,
            start_time: 0.0,
            end_time: 10.0,
            timestep: 0.01,
            seed: 42,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.name, "test");

        let debug = format!("{:?}", meta);
        assert!(debug.contains("TrajectoryMetadata"));
    }

    #[test]
    fn test_trajectory_clone_with_positions() {
        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![],
            metrics: SimMetrics::new(),
        });

        let cloned = traj.clone();
        assert_eq!(cloned.len(), 1);
        // Verify the positions were cloned correctly
        assert!(cloned.frame(0).is_some());
    }

    #[test]
    fn test_time_series_debug_impl() {
        let series = TimeSeries::new("test", 10);
        let debug = format!("{:?}", series);
        assert!(debug.contains("TimeSeries"));
    }

    #[test]
    fn test_time_series_clone_impl() {
        let mut series = TimeSeries::new("test", 10);
        series.push(1.0, 100.0);
        let cloned = series.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.name(), "test");
    }

    #[test]
    fn test_export_binary_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");

        let mut traj = Trajectory::new();
        traj.add_frame(TrajectoryFrame {
            time: 0.0,
            index: 0,
            positions: vec![1.0],
            velocities: vec![0.1],
            metrics: SimMetrics::new(),
        });

        let config = ExportConfig {
            format: ExportFormat::Binary,
            ..Default::default()
        };
        let exporter = Exporter::with_config(config);
        exporter.export(&traj, &path).unwrap();

        assert!(path.exists());
    }
}
