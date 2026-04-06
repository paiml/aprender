//! Progress Bar with Kalman-filtered ETA (ENT-058)
//!
//! Reference: Welch, G., & Bishop, G. (1995). "An Introduction to the Kalman Filter."

use std::time::Instant;

/// Kalman filter for ETA estimation.
#[derive(Debug, Clone)]
pub struct KalmanEta {
    /// Estimated step duration (seconds)
    estimate: f64,
    /// Error covariance
    error_cov: f64,
    /// Process noise
    process_noise: f64,
    /// Measurement noise
    measurement_noise: f64,
}

impl Default for KalmanEta {
    fn default() -> Self {
        Self { estimate: 1.0, error_cov: 1.0, process_noise: 0.01, measurement_noise: 0.1 }
    }
}

impl KalmanEta {
    /// Create a new Kalman filter for ETA estimation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with a new step duration measurement.
    pub fn update(&mut self, measured_duration: f64) {
        // Prediction step
        let predicted_estimate = self.estimate;
        let predicted_error = self.error_cov + self.process_noise;

        // Update step
        let kalman_gain = predicted_error / (predicted_error + self.measurement_noise);
        self.estimate = predicted_estimate + kalman_gain * (measured_duration - predicted_estimate);
        self.error_cov = (1.0 - kalman_gain) * predicted_error;
    }

    /// Get estimated time remaining for N steps.
    pub fn eta_seconds(&self, remaining_steps: usize) -> f64 {
        self.estimate * remaining_steps as f64
    }

    /// Format ETA as human-readable string.
    pub fn eta_string(&self, remaining_steps: usize) -> String {
        let secs = self.eta_seconds(remaining_steps);
        format_duration(secs)
    }
}

/// Format duration in seconds to human-readable string.
pub fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.0}s")
    } else if secs < 3600.0 {
        let mins = (secs / 60.0).floor();
        let s = (secs % 60.0).floor();
        format!("{mins}m {s:02.0}s")
    } else {
        let hours = (secs / 3600.0).floor();
        let mins = ((secs % 3600.0) / 60.0).floor();
        format!("{hours}h {mins:02.0}m")
    }
}

/// Progress bar renderer.
#[derive(Debug, Clone)]
pub struct ProgressBar {
    /// Total steps
    total: usize,
    /// Current step
    current: usize,
    /// Bar width in characters
    width: usize,
    /// Fill character
    fill_char: char,
    /// Empty character
    empty_char: char,
    /// Kalman filter for ETA
    kalman: KalmanEta,
    /// Last step time
    last_step_time: Option<Instant>,
}

impl ProgressBar {
    /// Create a new progress bar.
    pub fn new(total: usize, width: usize) -> Self {
        Self {
            total,
            current: 0,
            width,
            fill_char: '█',
            empty_char: '░',
            kalman: KalmanEta::new(),
            last_step_time: None,
        }
    }

    /// Update progress.
    pub fn update(&mut self, current: usize) {
        let now = Instant::now();
        if let Some(last_time) = self.last_step_time {
            let elapsed = now.duration_since(last_time).as_secs_f64();
            let steps = current.saturating_sub(self.current);
            if steps > 0 {
                let per_step = elapsed / steps as f64;
                self.kalman.update(per_step);
            }
        }
        self.current = current;
        self.last_step_time = Some(now);
    }

    /// Get progress percentage.
    pub fn percent(&self) -> f32 {
        if self.total == 0 {
            return 100.0;
        }
        (self.current as f32 / self.total as f32) * 100.0
    }

    /// Render progress bar to string.
    pub fn render(&self) -> String {
        let percent = self.percent();
        let filled = ((percent / 100.0) * self.width as f32).round() as usize;
        let empty = self.width.saturating_sub(filled);

        let bar: String = std::iter::repeat_n(self.fill_char, filled)
            .chain(std::iter::repeat_n(self.empty_char, empty))
            .collect();

        let remaining = self.total.saturating_sub(self.current);
        let eta = self.kalman.eta_string(remaining);

        format!("[{bar}] {percent:>5.1}% │ ETA: {eta}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(59.9), "60s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60.0), "1m 00s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3599.0), "59m 59s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3600.0), "1h 00m");
        assert_eq!(format_duration(5400.0), "1h 30m");
        assert_eq!(format_duration(7200.0), "2h 00m");
    }

    #[test]
    fn test_kalman_eta_new() {
        let kalman = KalmanEta::new();
        assert_eq!(kalman.estimate, 1.0);
    }

    #[test]
    fn test_kalman_eta_update() {
        let mut kalman = KalmanEta::new();
        kalman.update(0.5);
        assert!(kalman.estimate < 1.0);
        assert!(kalman.estimate > 0.5);
    }

    #[test]
    fn test_kalman_eta_seconds() {
        let kalman = KalmanEta::new();
        assert_eq!(kalman.eta_seconds(10), 10.0);
    }

    #[test]
    fn test_progress_bar_new() {
        let bar = ProgressBar::new(100, 20);
        assert_eq!(bar.percent(), 0.0);
    }

    #[test]
    fn test_progress_bar_percent() {
        let mut bar = ProgressBar::new(100, 20);
        bar.current = 50;
        assert_eq!(bar.percent(), 50.0);
    }

    #[test]
    fn test_progress_bar_percent_zero_total() {
        let bar = ProgressBar::new(0, 20);
        assert_eq!(bar.percent(), 100.0);
    }

    #[test]
    fn test_progress_bar_render() {
        let bar = ProgressBar::new(100, 10);
        let rendered = bar.render();
        assert!(rendered.contains('['));
        assert!(rendered.contains(']'));
        assert!(rendered.contains("ETA:"));
    }
}
