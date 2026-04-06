//! Anomaly detector with Z-score, IQR, and change point detection.

use super::{
    Anomaly, AnomalyReport, AnomalyType, ChangePoint, DEFAULT_IQR_MULTIPLIER,
    DEFAULT_ZSCORE_THRESHOLD, MIN_SAMPLES_FOR_DETECTION,
};

/// Anomaly detector
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Data buffer
    data: Vec<f64>,
    /// Z-score threshold
    zscore_threshold: f64,
    /// IQR multiplier
    iqr_multiplier: f64,
    /// Sliding window size for real-time detection
    window_size: usize,
    /// Detected anomalies
    anomalies: Vec<Anomaly>,
    /// Detected change points
    change_points: Vec<ChangePoint>,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            zscore_threshold: DEFAULT_ZSCORE_THRESHOLD,
            iqr_multiplier: DEFAULT_IQR_MULTIPLIER,
            window_size: 50,
            anomalies: Vec::new(),
            change_points: Vec::new(),
        }
    }
}

impl AnomalyDetector {
    /// Create new detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Z-score threshold
    pub fn with_zscore_threshold(mut self, threshold: f64) -> Self {
        self.zscore_threshold = threshold.max(1.0);
        self
    }

    /// Set IQR multiplier
    pub fn with_iqr_multiplier(mut self, multiplier: f64) -> Self {
        self.iqr_multiplier = multiplier.max(0.5);
        self
    }

    /// Set sliding window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(10);
        self
    }

    /// Add data point
    pub fn add(&mut self, value: f64) {
        self.data.push(value);
    }

    /// Add multiple data points
    pub fn add_all(&mut self, values: &[f64]) {
        self.data.extend_from_slice(values);
    }

    /// Get data count
    pub fn data_count(&self) -> usize {
        self.data.len()
    }

    /// Check if sufficient data for analysis
    pub fn has_sufficient_data(&self) -> bool {
        self.data.len() >= MIN_SAMPLES_FOR_DETECTION
    }

    /// Calculate mean of slice
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation
    fn std_dev(data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate percentile
    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }
        let idx = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[idx.min(sorted_data.len() - 1)]
    }

    /// Detect outliers using Z-score method
    pub fn detect_zscore_outliers(&mut self) -> Vec<Anomaly> {
        if !self.has_sufficient_data() {
            return Vec::new();
        }

        let mean = Self::mean(&self.data);
        let std_dev = Self::std_dev(&self.data, mean);

        if std_dev < 1e-10 {
            return Vec::new(); // No variation
        }

        let mut outliers = Vec::new();
        for (i, &value) in self.data.iter().enumerate() {
            let z_score = (value - mean) / std_dev;
            if z_score.abs() > self.zscore_threshold {
                let anomaly_type = if z_score > 0.0 {
                    AnomalyType::Spike
                } else {
                    AnomalyType::Drop
                };
                outliers.push(Anomaly::new(i, value, mean, z_score, anomaly_type));
            }
        }

        self.anomalies.extend(outliers.clone());
        outliers
    }

    /// Detect outliers using IQR method (robust to heavy tails)
    pub fn detect_iqr_outliers(&mut self) -> Vec<Anomaly> {
        if !self.has_sufficient_data() {
            return Vec::new();
        }

        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = Self::percentile(&sorted, 25.0);
        let q3 = Self::percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        if iqr < 1e-10 {
            return Vec::new(); // No variation
        }

        let lower_bound = q1 - self.iqr_multiplier * iqr;
        let upper_bound = q3 + self.iqr_multiplier * iqr;
        let median = Self::percentile(&sorted, 50.0);

        let mut outliers = Vec::new();
        for (i, &value) in self.data.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                let deviation = if value < lower_bound {
                    (lower_bound - value) / iqr
                } else {
                    (value - upper_bound) / iqr
                };
                let anomaly_type = if value > median {
                    AnomalyType::Spike
                } else {
                    AnomalyType::Drop
                };
                outliers.push(Anomaly::new(i, value, median, deviation, anomaly_type));
            }
        }

        self.anomalies.extend(outliers.clone());
        outliers
    }

    /// Detect change points using CUSUM-like algorithm
    pub fn detect_change_points(&mut self) -> Vec<ChangePoint> {
        if self.data.len() < 20 {
            return Vec::new();
        }

        let overall_mean = Self::mean(&self.data);
        let overall_std = Self::std_dev(&self.data, overall_mean);

        if overall_std < 1e-10 {
            return Vec::new();
        }

        let mut change_points = Vec::new();
        let min_segment = 10;

        // Simple change point detection: find points where mean shifts significantly
        for i in min_segment..(self.data.len() - min_segment) {
            let before = &self.data[..i];
            let after = &self.data[i..];

            let mean_before = Self::mean(before);
            let mean_after = Self::mean(after);

            let change = ChangePoint::new(i, mean_before, mean_after);

            // Check if change is significant (>1 std dev shift)
            if change.magnitude > overall_std {
                change_points.push(change);
            }
        }

        // Filter to keep only most significant change points
        if change_points.len() > 1 {
            change_points.sort_by(|a, b| {
                b.magnitude
                    .partial_cmp(&a.magnitude)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Keep top change points
            change_points.truncate(3);
        }

        self.change_points.extend(change_points.clone());
        change_points
    }

    /// Classify anomaly based on context
    pub fn classify_anomaly(&self, anomaly: &Anomaly) -> AnomalyType {
        // Check if it's part of a change point
        for cp in &self.change_points {
            if (anomaly.index as i64 - cp.index as i64).abs() < 5 {
                return AnomalyType::ChangePoint;
            }
        }

        // Check if correlated with other anomalies (cluster)
        let nearby_count = self
            .anomalies
            .iter()
            .filter(|a| {
                (a.index as i64 - anomaly.index as i64).abs() < 3 && a.index != anomaly.index
            })
            .count();

        if nearby_count >= 2 {
            return AnomalyType::Correlated;
        }

        anomaly.anomaly_type
    }

    /// Run all detection methods and generate report
    pub fn analyze(&mut self) -> AnomalyReport {
        self.anomalies.clear();
        self.change_points.clear();

        let zscore_outliers = self.detect_zscore_outliers();
        let _iqr_outliers = self.detect_iqr_outliers();
        let change_points = self.detect_change_points();

        // Deduplicate anomalies by index
        let mut seen = std::collections::HashSet::new();
        self.anomalies.retain(|a| seen.insert(a.index));

        // Sort by severity (critical first)
        self.anomalies.sort_by(|a, b| b.severity.cmp(&a.severity));

        let mean = Self::mean(&self.data);
        let std_dev = Self::std_dev(&self.data, mean);

        AnomalyReport {
            total_points: self.data.len(),
            anomalies: self.anomalies.clone(),
            change_points,
            mean,
            std_dev,
            method: if zscore_outliers.is_empty() {
                "iqr"
            } else {
                "zscore"
            },
        }
    }

    /// Real-time anomaly detection on sliding window
    pub fn detect_realtime(&mut self, new_value: f64) -> Option<Anomaly> {
        self.data.push(new_value);

        if self.data.len() < self.window_size {
            return None;
        }

        // Use sliding window
        let start = self.data.len().saturating_sub(self.window_size);
        let window = &self.data[start..self.data.len() - 1]; // Exclude new value

        let mean = Self::mean(window);
        let std_dev = Self::std_dev(window, mean);

        if std_dev < 1e-10 {
            return None;
        }

        let z_score = (new_value - mean) / std_dev;
        if z_score.abs() > self.zscore_threshold {
            let anomaly_type = if z_score > 0.0 {
                AnomalyType::Spike
            } else {
                AnomalyType::Drop
            };
            let anomaly = Anomaly::new(self.data.len() - 1, new_value, mean, z_score, anomaly_type);
            self.anomalies.push(anomaly.clone());
            return Some(anomaly);
        }

        None
    }

    /// Get all detected anomalies
    pub fn get_anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// Get all detected change points
    pub fn get_change_points(&self) -> &[ChangePoint] {
        &self.change_points
    }

    /// Clear all data and detections
    pub fn clear(&mut self) {
        self.data.clear();
        self.anomalies.clear();
        self.change_points.clear();
    }
}
